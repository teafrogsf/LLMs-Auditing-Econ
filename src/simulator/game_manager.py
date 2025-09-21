import json
from tkinter import NO
import numpy as np
import math
import os
from typing import Dict, List
from src.model import MODEL_PRICING
from src.simulator.provider import ProviderManager
from src.utils import Logger
import random



class ToyModel:
    def __init__(self, config) -> None:
        self.score_mu = float(config['score_mu'])
        self.score_sigma = float(config['score_sigma'])
        self.token_mu = float(config['token_mu'])
        self.token_sigma = float(config['token_sigma'])
        self.price = float(config['price'])
        self.reward_param = float(config['reward_param'])
        self.utility_mu = self.reward_param * self.score_mu - self.token_mu * self.price

    def generate(self, sample):
        """
        sample:
        {
            "id": int,
            "input_tokens": int
        }
        """
        score = random.gauss(self.score_mu, self.score_sigma)
        output_tokens_float = abs(random.gauss(self.token_mu, self.token_sigma))
        output_tokens = max(50, int(round(output_tokens_float)))  

        return {
            "id": sample["id"],
            "input_tokens": sample["input_tokens"],
            "score": score,
            "output_tokens": output_tokens
        }



class GameManager:
    game_config: Dict
    providers: List[ProviderManager]
    def __init__(self, game_config):
        self.game_config = game_config
        self.B = int(game_config['B'])
        self.T = int(game_config['num_tasks'])
        self.M = float(game_config['M'])
        self.K = int(game_config['K'])
        self.gamma = float(game_config['gamma'])
        self.reward_param = float(game_config["reward_param"])
        self.output_dir = game_config['output_dir']
        self.task_ids = json.load(open(game_config["task_ids_path"]))[:self.T]
        self.t = 0
        self.L = None
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
            provider_config['reward_param'] = self.reward_param
            self.providers.append(ProviderManager(provider_config))
        self.L = max([provider.get_priori_max_tokens() for provider in self.providers])
        miu_rs = []
        miu_ls = []

        for provider in self.providers:
            priori_model_info = provider.get_priori_model_info()
            miu_rs.extend([priori_model_info[model]['avg_reward'] for model in priori_model_info])
            miu_ls.extend([priori_model_info[model]['avg_tokens'] for model in priori_model_info])

        self.miu_r = min(miu_rs)
        self.miu_l = max(miu_ls)

        self.delta_1 = -math.log(self.miu_r) + math.log(self.miu_l) + self.miu_l / self.L + 1
        self.delta_2 = math.log(self.reward_param)

        
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
            return False
        task_id = self.task_ids[self.t]
        result = self.providers[provider_idx].run_task(task_id, phase, self.second_utility, R)
        self.delegation_history.append(result)
        self.t += 1
        return True

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
        # Todo: 下面这和 p_i 是哪个 i
        pi_max = self.providers[self.best_provider_idx].get_max_pi()
        threshold = self.second_utility - self.M * (self.reward_param + self.L * pi_max) / self.gamma
        self.logger.log(
            f"threshold calc -> second_utility={self.second_utility:.6f}, M={self.M:.6f}, "
            f"reward_param={self.reward_param:.6f}, L={self.L:.6f}, best_provider_idx={self.best_provider_idx}, "
            f"max_pi={pi_max:.6f}, threshold={threshold:.6f}"
        )
        R = int(max(0, self.T - (max(self.delta_1, self.delta_2) + 3)*self.B*self.K))
        self.logger.log(R)
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
                self.logger.log('不奖励了')
                continue
            self.logger.log(f'delegate {self.B} times for provider {i}')
            for _ in range(self.B):
                self._delegate_task(i, phase=3)
    
    def phase3_utility_based(self):
        for i in range(self.K):  
            if self.t >= self.T:
                return
            
            avg_tau = self.providers[i].get_avg_output_tokens()

            delta = self.delta_1 + math.log(self.providers[i].get_avg_reward()) - math.log(avg_tau) - avg_tau / self.L
            if delta > 0:
                self.logger.log(f'delegate {self.B} times for provider {i}')
                num_delegations = int(delta) * self.B
                for _ in range(num_delegations):
                    self._delegate_task(i, phase=4)
                
                frac_part = delta - int(delta)
                if random.random() < frac_part:
                    
                    num_delegations = min(self.B, self.T - self.t)
                    self.logger.log(f'delegate {num_delegations} times for provider {i}')
                    for _ in range(num_delegations):
                        self._delegate_task(i, phase=4)



    def _get_result(self):
        general_results = {
            'total_time': self.T,
            'total_delegations': self.t,
            'best_provider_idx': self.best_provider_idx,
            'providers': []
        }
        for i in range(self.K):
            provider_result = self.provider_hi
            
            general_results['providers'].append()

        json.dump(general_results, open(os.path.join(self.output_dir, 'result.json'), 'w'), indent=2)
        return general_results

    def run_game(self):
        self.phase1_exploration()
        self.phase2_exploration()
        self.phase2_incentive()
        self.phase3_utility_based()

        return self._get_result()
        
