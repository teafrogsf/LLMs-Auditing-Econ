import json
import numpy as np
import math
import os
from typing import Dict, List
from src.simulator.provider import ProviderManager
from src.utils import Logger
import random



class GameManager:
    game_config: Dict
    providers: List[ProviderManager]
    def __init__(self, game_config):
        self.game_config = game_config
        self.B = game_config['B']
        self.T = game_config['num_tasks']
        self.M = game_config['M']
        self.K = game_config['K']
        self.reward_coff = game_config["reward_param"]
        self.output_dir = game_config['output_dir']
        self.task_ids = json.load(open(game_config["task_ids_path"]))[:self.T]
        self.t = 0
        self.delta_1 = None
        self.delta_2 = None
        self.best_utility = None
        self.best_provider_idx = None
        self.second_utility = None
        self.second_provider_idx = None
        self.delegation_history = []
        self._init_providers()
        self._init_output_dir_logger()
        


        if self.B*len(self.providers) > 0.5 * self.T:
            raise ValueError(f"B*len(providers) > 0.5 * T: {self.B*len(self.providers)} > {0.5 * self.T}")

    
    def _init_output_dir_logger(self):
        os.makedirs(self.output_dir, exist_ok=True)
        log_path = os.path.join(self.output_dir, 'log.txt')
        self.logger = Logger(self.output_dir, log_path)
        


        


    def _init_providers(self):
        self.providers = []
        for provider_config in self.game_config['providers']:
            provider_config['reward_param'] = self.reward_coff
            self.providers.append(ProviderManager(provider_config))
        
        self.provider_avg_info ={
            "avg_rewards": None,
            "avg_utilities": None
        }
        self.num_providers = len(self.providers)

    def _compute_provider_avg_info(self):
        self.provider_avg_info['avg_rewards'] = [provider.get_avg_reward() for provider in self.providers]
        self.provider_avg_info['avg_utilities'] = [provider.get_avg_utility() for provider in self.providers]

    def _delegate_task(self, provider_idx, phase, R=None):
        if self.t >= self.T:
            raise IndexError(f'self.t > self.T')
        task_id = self.task_ids[self.t]
        result = self.providers[provider_idx].run_task(task_id, phase, self.second_utility, R)
        self.delegation_history.append(result)
        self.t += 1

    def phase1_exploration(self):
        """
        stage1: delegate each provider B Times
        """
        self.logger.log('='*20+'Stage1: Delegate each provider B Times'+ '='*20)
    
        for i in range(len(self.providers)):
            self.logger.log(f'Delegate provider{i} {self.B} Times')
            for _ in range(self.B):
                self._delegate_task(provider_idx=i, phase=1)
        
        self._compute_provider_avg_info()
        sorted_provider_utilities = sorted(self.provider_avg_info['avg_utilities'])
        self.logger.log(self.provider_avg_info)
        max_utility, min_utility, second_utility = sorted_provider_utilities[-1], sorted_provider_utilities[0], sorted_provider_utilities[-2]
        min_utility = min(self.provider_avg_info['avg_utilities'])
        if min_utility <= 0:
            min_utility = 1e-6
            self.logger.log(f"warning")
        
        self.delta_1 = -math.log(min_utility) + 2 + self.M
        self.delta_2 = math.log(max_utility)
        self.logger.log(f"δ1={self.delta_1:.4f}, δ2={self.delta_2:.4f}")
        if len(set(self.provider_avg_info['avg_utilities'])) != len(self.provider_avg_info['avg_utilities']):
            raise ValueError(f'1')

        self.best_provider_idx = self.provider_avg_info['avg_utilities'].index(max_utility)
        # second_provider_idx = self.provider_avg_info['avg_utilities'].index(second_utility)
        self.second_utility = second_utility

        self.logger.log(f"  阶段1完成，最佳服务商：{self.best_provider_idx}，平均效用：{max_utility:.4f}")
        self.logger.log(f"  第二好效用：{second_utility:.4f}")
            
    
    def phase2_exploration(self):
        early_stop_flag = False
        threshold = self.second_utility - self.M
        R = int(max(0, self.T - (max(self.delta_1, self.delta_2) + 3)*self.B*self.K))
        nums_remain_tasks = min(R, self.T - self.t)
        self.logger.log(f"  计划委托{nums_remain_tasks}次，阈值：{threshold:.4f}")
        delegation_count = 0
        for _ in range(nums_remain_tasks):
            if self.t >= self.T:
                raise ValueError(f'badvalue')
            self._delegate_task(self.best_provider_idx, phase=2, R=R)
            delegation_count += 1
            if delegation_count >= self.B:
                phase2_avg_utility = self.providers[self.best_provider_idx].get_recent_avg_utility(delegation_count)
                if phase2_avg_utility < threshold:
                    self.logger.log(f"  在{delegation_count}次委托后停止，最近平均utility：{phase2_avg_utility:.4f} < {threshold:.4f}")
                    early_stop_flag = True
                    break
        
        if not early_stop_flag and self.t < self.T:
            self.logger.log('bonust B times for best')
            for _ in range(min(self.B, self.T-self.t)):
                self._delegate_task(provider_idx=self.best_provider_idx, phase=3)
            

    def phase2_incentive(self):
        for i in range(self.num_providers):
            if i == self.best_provider_idx:
                continue
            for _ in range(self.B):
                self._delegate_task(i, phase=3)
    
    def phase3_utility_based(self):
        for i in range(self.K):  
            if self.t >= self.T:
                return
            
            avg_reward = self.providers[i].get_avg_reward()
            avg_bar_u = avg_reward - self.M
            if avg_bar_u <= 0:
                self.logger.log(f"provider {i}的回报差值{avg_bar_u:.4f} <= 0，跳过")
                continue

            utility = self.delta_1 + math.log(avg_bar_u)
            if utility >= 0:
                num_delegations = int(utility) * self.B
                for _ in range(num_delegations):
                    self._delegate_task(i, phase=4)

                
                frac_part = utility - int(utility)
                num_delegations = min(self.B, self.T - self.t)
                if random.random() < frac_part:
                    for _ in range(num_delegations):
                        self._delegate_task(i, phase=4)
    

    def _get_result(self):
        general_results = {
            'total_time': self.T,
            'total_delegations': self.t,
            'best_provider_idx': self.best_provider_idx,
            'providers': []
        }
        for provider in self.providers:
            general_results['providers'].append(provider.get_results())

        json.dump(general_results, open(os.path.join(self.output_dir, 'result.json'), 'w'), indent=2)
        return general_results

    def run_game(self):
        self.phase1_exploration()
        self.phase2_exploration()
        self.phase2_incentive()
        self.phase3_utility_based()

        return self._get_result()
        
