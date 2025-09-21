import json
import math
import os
import random
from sys import path_hooks
from typing import Dict, List
import numpy as np
from src.utils import Logger
from pathlib import Path

ITEM_RECORDS = ['tokens','input_tokens','output_tokens','price','cost',
                    'reward', 'user_utility', 'provider_utility',]

class ToyModel:
    def __init__(self, config) -> None:
        self.score_mu = float(config['score_mu'])
        self.score_sigma = float(config['score_sigma'])
        self.token_mu = float(config['token_mu'])
        self.token_sigma = float(config['token_sigma'])
        self.token_price = float(config['token_price'])
        self.reward_param = float(config['reward_param'])
        self.eta = float(config['eta'])
        self.utility_mu = self.reward_param * self.score_mu - self.token_mu * self.token_price
        self.model_cost_mu = self.token_mu * self.token_price

    def generate(self, input_tokens, L):

        score = max(0, min(1, random.gauss(self.score_mu, self.score_sigma)))
        output_tokens_float = abs(random.gauss(self.token_mu, self.token_sigma))
        output_tokens = min(max(50, int(round(output_tokens_float))), L)
        tokens = input_tokens + output_tokens
        cost = (input_tokens + output_tokens) * self.token_price * self.eta


        return {
            "tokens": tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reward": score * self.reward_param,
            "cost": cost
        }


class ToyProvider:
    models: List[ToyModel]
    def __init__(self, config) -> None:
        self.num_models = len(config['models'])
        self.eta = float(config['eta'])
        self.strategy = config['strategy']
        models = []
        for model_config in config['models']:
            model_config['reward_param'] = config['reward_param']
            model_config['eta'] = self.eta
            models.append(ToyModel(model_config))
        # model_index_list = list(range(len(models)))

        models = sorted(models, key=lambda x: x.utility_mu, reverse=True)
        models_cost = [model.model_cost_mu for model in models]
        min_model_cost_mu = min(models_cost)
        self.lie_model_idx = models_cost.index(min_model_cost_mu)
        self.models = models
    
    
    def honset_run(self, model_index, input_tokens, L):
        model_run_result = self.models[model_index].generate(input_tokens, L)
        price = model_run_result['tokens'] * self.models[0].token_price
        user_utility = model_run_result['reward'] - price
        model_run_result['price'] = price
        model_run_result['user_utility'] = user_utility
        model_run_result['provider_utility'] = price  - model_run_result['cost']
        return model_run_result
    
    def lie_run(self, input_tokens, L):
        model_run_result = self.models[self.lie_model_idx].generate(input_tokens, L)
        price = self.models[0].token_price * L
        user_utility = model_run_result['reward'] - price
        provider_utility = price  - model_run_result['cost']
        model_run_result['price'] = price
        model_run_result['user_utility'] = user_utility
        model_run_result['provider_utility'] = provider_utility
        return model_run_result
         

    def run_task(self, input_tokens, phase, L, second_utility=None):
        if self.strategy == 'honest':
            result = self.honset_run(0, input_tokens, L)
        elif self.strategy == 'worst':
            result = self.lie_run(input_tokens, L)
        
        elif self.strategy == 'ours':
            if phase == 1:
                result = self.honset_run(0, input_tokens, L)
            
            elif phase == 2:
                for idx, model in enumerate(self.models[::-1]):
                    if model.utility_mu > second_utility:
                        result = self.honset_run(self.num_models-idx-1, input_tokens, L)
                        break
                else:
                    raise ValueError('111')
            else:
                result = self.lie_run(input_tokens, L)
        
        elif self.strategy == 'h1w2':
            if phase == 1:
                result = self.honset_run(0, input_tokens, L)
            else:
                result = self.lie_run(input_tokens, L)
        
        elif self.strategy == 'w1h2':
            if phase == 1:
                result = self.lie_run(input_tokens, L)
            else:
                result =self.honset_run(0, input_tokens, L)
        
        elif self.strategy == 'random':
            if random.random() < 0.5:
                result =self.honset_run(0, input_tokens, L)
            else:
                result = self.lie_run(input_tokens, L)
        
        else:
            raise ValueError(f'{self.strategy} is not supported!')
        
        return result


    
class ToyGameManager:
    game_config: Dict
    providers: List[ToyProvider]

    def __init__(self, game_config: Dict):
        self.game_config = game_config
        self.B = int(game_config['B'])
        self.T = int(game_config['num_tasks'])
        self.M = float(game_config['M'])
        self.K = int(game_config['K'])
        self.output_dir = Path(game_config['output_dir'])
        self._init_output_dir_logger()
        self.logger.log(game_config)
        self.gamma = float(game_config.get('gamma', 1.0))
        self.reward_param = float(game_config['reward_param'])
        self.input_tokens_mu = float(game_config['input_tokens_mu'])
        self.input_tokens_sigma = float(game_config['input_tokens_gamma'])
        self.task_tokens = np.random.normal(
            float(game_config['input_tokens_mu']), 
            float(game_config['input_tokens_gamma']),
            self.T
        
        )
        self.task_tokens = np.maximum(self.task_tokens, 50).astype(int)


        self.t = 0
        self.L = int(game_config["L"])
        self.delta_1 = None
        self.delta_2 = None
        self.best_provider_idx = None
        self.second_user_utility = None
        self.providers = []
        self.providers_his = []
       

        self._init_providers()
        

    def _init_providers(self) -> None:
        self.providers = []
        mu_r = []
        mu_l = []
        for provider_config in self.game_config['providers']:
            provider_config['reward_param'] = self.reward_param
            provider = ToyProvider(provider_config)
            self.providers.append(provider)
            for model in provider.models:
                mu_r.append(model.score_mu * self.reward_param)
                mu_l.append(model.token_mu)
            
            self.providers_his.append(
                {
                    'tokens':[],
                    'input_tokens': [],
                    'output_tokens':[],
                    'price': [],
                    'cost': [],
                    'reward': [],
                    'user_utility': [],
                    'provider_utility': [],
                    'delegation_results': []
                }
            )
                
        min_mu_r = min(mu_r)
        max_mu_l = max(mu_l)
        self.logger.log(f"min_mu_r={min_mu_r:.4f}, max_mu_l={max_mu_l:.4f}")
        self.delta_1 = -math.log(min_mu_r) + math.log(max_mu_l) + max_mu_l / self.L + 1
        self.delta_2 = math.log(self.reward_param)
        self.logger.log(f"delta_1={self.delta_1:.4f}, delta_2={self.delta_2:.4f}")
     
    def _init_output_dir_logger(self):
        os.makedirs(self.output_dir, exist_ok=True)
        log_path = self.output_dir / 'log.txt'
        self.logger = Logger(self.output_dir.suffix, log_path)
        


    def _delegate_task(self, provider_idx: int, phase: int):
        if self.t >= self.T:
            return None
        input_tokens = self.task_tokens[self.t]
        result = self.providers[provider_idx].run_task(input_tokens, phase, self.L, self.second_user_utility)
        for key in result:
            self.providers_his[provider_idx][key].append(result[key])
        self.providers_his[provider_idx]['delegation_results'].append(result)
        self.providers_his[provider_idx]['avg_user_utility'] = np.mean(self.providers_his[provider_idx]['user_utility'])
        self.t += 1
        return True


        

    def phase1_exploration(self):

        self.logger.log('='*20+f'Stage1: Delegate each provider {self.B} Times'+ '='*20)
        for i in range(len(self.providers)):
            for _ in range(self.B):
                self._delegate_task(provider_idx=i, phase=1)


        sorted_provider_idx = sorted(list(range(self.K)), key=lambda idx: self.providers_his[idx]['avg_user_utility'])
        self.best_provider_idx = sorted_provider_idx[-1]
        second_provider_idx = sorted_provider_idx[-2]
       
        best_user_utility = self.providers_his[self.best_provider_idx]['avg_user_utility']
        self.second_user_utility = self.providers_his[second_provider_idx]['avg_user_utility']

        self.logger.log(f"  阶段1完成，最佳服务商：{self.best_provider_idx}，平均效用：{best_user_utility:.4f}")
        self.logger.log(f"  第二好效用：{self.second_user_utility:.4f}")

    def phase2_exploitation(self):
        pi_max = max([max([m.token_price for m in p.models]) for p in self.providers])
        threshold = self.second_user_utility - self.M * (self.reward_param + self.L * pi_max) / self.gamma
        R = int(max(0, self.T - (max(self.delta_1, self.delta_2) + 3) * self.B * self.K))
        nums_remain_tasks = min(R, self.T - self.t)
        self.logger.log(f"  计划委托{nums_remain_tasks}次，阈值：{threshold:.4f}")
        delegation_count = 0
        early_stop_flag = False
        for _ in range(nums_remain_tasks):
            if self.t >= self.T:
                break
            self._delegate_task(self.best_provider_idx, phase=2)
            delegation_count += 1
            if delegation_count >= self.B:

                phase2_avg_user_utility = np.mean(self.providers_his[self.best_provider_idx]['user_utility'][-delegation_count:])
                if phase2_avg_user_utility < threshold:
                    self.logger.log(f"  在{delegation_count}次委托后停止，最近平均utility：{phase2_avg_user_utility:.4f} < {threshold:.4f}")

                    early_stop_flag = True
                    break
        
        if not early_stop_flag:
            if self.t < self.T:
                for _ in range(min(self.B, self.T - self.t)):
                    self._delegate_task(provider_idx=self.best_provider_idx, phase=3)

    def phase2_incentive(self):
        for i in range(self.K):
            if i == self.best_provider_idx:
                continue
            for _ in range(self.B):
                self._delegate_task(i, phase=3)

    def phase3_utility_based(self):
        for i in range(self.K):
            if self.t >= self.T:
                return
            avg_reward = np.mean(self.providers_his[i]['reward'])
            avg_output_tokens = np.mean(self.providers_his[i]['output_tokens'])
            delta = self.delta_1 + math.log(avg_reward) - math.log(avg_output_tokens) - avg_output_tokens / self.L
            self.logger.log(f"delta: {delta}")
            if delta > 0:
                
                num_delegations = int(delta) * self.B
                for _ in range(num_delegations):
                    self._delegate_task(i, phase=4)
                frac_part = delta - int(delta)
                if random.random() < frac_part:
                    num_delegations = min(self.B, self.T - self.t)
                    for _ in range(num_delegations):
                        self._delegate_task(i, phase=4)
    
    def get_result(self):
        game_result = {
            'total_time': self.T,
            'total_delegations': self.t,
            'best_provider_idx': self.best_provider_idx,
            'providers': []
        }
        for i in range(self.K):
            provider_result = self.providers_his[i]
            save_result = {
                'delegations': len(provider_result['delegation_results']),
            }
            for key in ITEM_RECORDS:
                save_result['total_'+key] = float(np.sum(provider_result[key]))
            
            game_result['providers'].append(save_result)
        print(game_result)
        json.dump(game_result, open(self.output_dir / 'result.json', 'w'), indent=2)



    def run_game(self):
        self.phase1_exploration()
        self.phase2_exploitation()
        self.phase2_incentive()
        self.phase3_utility_based()
        self.get_result()
      

