import json
import math
import os
import random
from sys import path_hooks
from typing import Dict, List
import numpy as np
import yaml

from dev_temp.get_model_info import input_token_price

from src.utils import Logger
from pathlib import Path

ITEM_RECORDS = ['tokens','input_tokens','output_tokens','price','cost',
                    'reward', 'user_utility', 'provider_utility',]

STRAGETY = [
    ['honest', 'honest', 'honest', 'honest'],
    ['honest', 'honest', 'lie_all', 'lie_all'],
    ['honest', 'lie_ours', 'lie_all', 'lie_all'],
    ['honest', 'lie_model', 'lie_all', 'lie_all'],
    ['honest', 'lie_token', 'lie_all', 'lie_all'],
    ['honest', 'lie_all', 'lie_all', 'lie_all'],
    ['lie_ours', 'lie_ours', 'lie_all', 'lie_all'],
    ['lie_ours', 'lie_model', 'lie_all', 'lie_all'],
    ['lie_ours', 'lie_token', 'lie_all', 'lie_all'],
    ['lie_ours', 'lie_all', 'lie_all', 'lie_all']
]

MODEL_CONFIG = yaml.safe_load(open('config/nl_graph/model_config.yaml'))
print(MODEL_CONFIG)
def load_records(path):
    return [json.loads(item) for item in open(path).readlines()]

class RealModel:
    def __init__(self, config) -> None:
        self.model_name = config['model_name']
        self.score_mu = float(config['score_mu'])
        self.output_tokens_mu = float(config['output_tokens_mu'])
        self.input_token_price = float(config['input_token_price'])
        self.output_token_price = float(config['output_token_price'])
        self.reward_param = float(config['reward_param'])
        self.max_output_tokens = int(config['max_output_tokens'])
        self.eta = float(config['eta'])
        self.utility_mu = self.reward_param * self.score_mu - self.output_tokens_mu * self.output_token_price
        self.model_cost_mu = self.output_tokens_mu * self.output_token_price
        self.task_records = load_records(f'data/local_records/nlgraph_new/{self.model_name}_test_result.jsonl')

    def generate(self, task_id, L):
        task_result = self.task_records[task_id]


        score = task_result['score']
        input_tokens = task_result['input_tokens']
        output_tokens = task_result['output_tokens']
        if output_tokens > L:
            score = output_tokens / L * score
            output_tokens = L
        real_price = input_tokens * self.input_token_price + output_tokens * self.output_token_price
        cost = real_price * self.eta
        return {
            "tokens": input_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reward": score * self.reward_param,
            "cost": cost,
            "real_price": real_price
        }

        



class Provider:
    models: List[RealModel]
    def __init__(self, config, logger) -> None:
        self.num_models = len(config['models'])
        self.eta = float(config['eta'])
        self.strategy = int(config['strategy'])
        # self.token_factor = config['token_factor']
        models = []
        for model in config['models']:
            model_config = MODEL_CONFIG[model]
            model_config['model_name'] = model
            model_config['reward_param'] = config['reward_param']
            model_config['eta'] = self.eta
            models.append(RealModel(model_config))
        # model_index_list = list(range(len(models)))

        models = sorted(models, key=lambda x: x.utility_mu, reverse=True)
        self.models = models

        # models_cost = [model.model_cost_mu for model in models]
        # min_model_cost_mu = min(models_cost)
        # self.lie_model_idx = self.num_models - 1
        # self.max_input_token_price = max([model.input_token_price for model in models])
        # max_output_token_price_model = max([model for model in models], key=lambda x: x.output_token_price)
        self.max_output_token_price = self.models[0].output_token_price
        self.max_input_token_price = self.models[0].input_token_price
        
 

        # self.models = models
        self.max_tokens = max([item.max_output_tokens for item in models])
        self.logger = logger
        
        self.logger.log([model.model_name for model in models])
        # self.logger.log(f'lie model: {models[self.lie_model_idx].model_name}')        
        self.logger.log('Model prices:')
        self.logger.log([model.output_token_price for model in models])

        self.logger.log('Model cost_mu:')
        self.logger.log([model.model_cost_mu for model in models])

        self.logger.log('Model utility_mu:')
        self.logger.log([model.utility_mu for model in models])
    
    
    def honset_run(self, task_id, L):
        model_run_result = self.models[0].generate(task_id, L)
        price = model_run_result['input_tokens'] * self.models[0].input_token_price + \
            model_run_result['output_tokens'] * self.models[0].output_token_price
        real_price = model_run_result['real_price']
        
        user_utility = model_run_result['reward'] - price
        model_run_result['price'] = price
        model_run_result['user_utility'] = user_utility
        model_run_result['provider_utility'] = self.eta * (price  -  real_price)
        return model_run_result
    
    def lie_run_model(self, real_model_idx, task_id, L):
        # the worst model's utility > second best utility
        if(real_model_idx == 2):
            result1 = self.models[-2].generate(task_id, L)
            result2 = self.models[-1].generate(task_id, L)
            price1 = result1['input_tokens'] * self.max_input_token_price + \
                result1['output_tokens'] * self.max_output_token_price
            price2 = result2['input_tokens'] * self.max_input_token_price + \
                result2['output_tokens'] * self.max_output_token_price
            utility1 = self.eta * (price1 - result1['real_price'])
            utility2 = self.eta * (price2 - result2['real_price'])
            model_run_result = result1 if utility1 > utility2 else result2
        else:
            model_run_result = self.models[real_model_idx].generate(task_id, L)

        price = model_run_result['input_tokens'] * self.max_input_token_price+ \
            model_run_result['output_tokens'] * self.max_output_token_price
        user_utility = model_run_result['reward'] - price
        model_run_result['price'] = price
        model_run_result['user_utility'] = user_utility
        model_run_result['provider_utility'] = self.eta * (price  -  model_run_result['real_price'])
        return model_run_result
    
    def lie_run_tokens(self, task_id, L):
        model_run_result = self.models[0].generate(task_id, L)
        reported_tokens = L
        model_run_result['output_tokens'] = reported_tokens
        model_run_result['tokens'] = reported_tokens + model_run_result['input_tokens']


        price = model_run_result['input_tokens'] * self.models[0].input_token_price + \
            reported_tokens * self.models[0].output_token_price
        

        user_utility = model_run_result['reward'] - price
        model_run_result['price'] = price
        model_run_result['user_utility'] = user_utility
        model_run_result['provider_utility'] = self.eta * (price  -  model_run_result['real_price'])
        return model_run_result

    
    def lie_run_all(self, task_id, L):
        model_run_result = self.models[-1].generate(task_id, L)
        reproted_tokens = L
        model_run_result['output_tokens'] = reproted_tokens
        model_run_result['tokens'] = model_run_result['input_tokens'] + reproted_tokens

        price = model_run_result['input_tokens'] * self.max_input_token_price+ \
             self.max_output_token_price *  reproted_tokens

        user_utility = model_run_result['reward'] - price
        provider_utility = self.eta * (price  - model_run_result['real_price'])
        model_run_result['price'] = price
        model_run_result['user_utility'] = user_utility
        model_run_result['provider_utility'] = provider_utility
        return model_run_result
         

    def run_task(self, task_id, phase, L, second_utility=None):
        strategy = STRAGETY[self.strategy][phase-1]
        if strategy == 'honest':
            result = self.honset_run(task_id, L)
            
        elif strategy == 'lie_all':
            result = self.lie_run_all(task_id, L)
        
        elif strategy == 'lie_ours':
            if second_utility is None:
                    result = self.lie_run_model(1, task_id, L)
            else:
                for idx, model in enumerate(self.models[::-1]):
                
                    if model.utility_mu > second_utility:

                        result = self.lie_run_model(self.num_models-idx-1, task_id, L)
                        break
                else:
                    result = self.honset_run(task_id, L)

        elif strategy == 'lie_model':
            result = self.lie_run_model(-1, task_id, L)
        
        elif strategy == 'lie_token':
            result = self.lie_run_tokens(task_id, L)

        
        else:
            raise ValueError(f'{self.strategy} is not supported!')
        
        return result


    
class GameManager:
    game_config: Dict
    providers: List[Provider]

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
        self.task_ids = json.load(open('data/task_ids_shuffled.json'))
        self.task_ids = self.task_ids * 10
        # random.shuffle(self.task_ids)

        self.t = 0
        self.L = None
        self.delta_1 = None
        self.delta_2 = None
        self.delta_3 = None
        self.best_provider_idx = None
        self.second_user_utility = None
        self.providers = []
        self.providers_his = []
       

        self._init_providers()
        

    def _init_providers(self) -> None:
        self.providers = []
        mu_r = []
        mu_l = []
        mu_r_l = []
        for provider_config in self.game_config['providers']:
            provider_config['reward_param'] = self.reward_param
            self.logger.log(f'init provider {provider_config["id"]}')
            provider = Provider(provider_config, self.logger)
            self.providers.append(provider)
            for model in provider.models:
                mu_r_i = model.score_mu * self.reward_param
                mu_l_i = model.output_tokens_mu
                mu_r.append(mu_r_i)
                mu_l.append(mu_l_i)
                mu_r_l.append(mu_r_i / mu_l_i)

            self.providers_his.append(
                {
                    'tokens':[],
                    'input_tokens': [],
                    'output_tokens':[],
                    'price': [],
                    'cost': [],
                    'reward': [],
                    'real_price': [],
                    'user_utility': [],
                    'provider_utility': [],
                    'delegation_results': []
                }
            )
        # exit()
        self.L = max([provider.max_tokens for provider in self.providers]) 
        self.logger.log(f'L: {self.L}')
                
        min_mu_r = min(mu_r)
        max_mu_l = max(mu_l)
        self.logger.log(f"min_mu_r={min_mu_r:.4f}, max_mu_l={max_mu_l:.4f}, L: {self.L}")
        # self.delta_1 = -math.log(min_mu_r) + math.log(max_mu_l) + max_mu_l / self.L + 1
        self.delta_1 = -math.log(min(mu_r_l)) + max(mu_l) / self.L + 1
        self.delta_2 = math.log(self.reward_param)
        
        
        self.logger.log(f"delta_1={self.delta_1:.4f}, delta_2={self.delta_2:.4f}")
     
    def _init_output_dir_logger(self):
        os.makedirs(self.output_dir, exist_ok=True)
        log_path = self.output_dir / 'log.txt'
        self.logger = Logger(self.output_dir.name, log_path)
        self.logger.log(f'log to file {log_path}, logger name is {self.output_dir.name}')
        

    def _delegate_task(self, provider_idx: int, phase: int):
        if self.t >= self.T:
            return None
        task_id = self.task_ids[self.t]
        result = self.providers[provider_idx].run_task(task_id, phase, self.L, self.second_user_utility)
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

        self.logger.log(f"Providers sorted by avg_user_utility: {sorted_provider_idx}")
        for i in range(self.K):
            his = self.providers_his[i]
            avg_provider_utility = np.mean(his['provider_utility']) if len(his['provider_utility']) > 0 else 0.0
            self.logger.log(f"Provider {i} avg_user_utility: {his['avg_user_utility']:.4f}")
            self.logger.log(f"Provider {i} avg_provider_utility: {avg_provider_utility:.4f}")
        
        self.logger.log(f"  阶段1完成，最佳服务商：{self.best_provider_idx}，平均效用：{best_user_utility:.4f}")
        self.logger.log(f"  第二好效用：{self.second_user_utility:.4f}")
        
        # 输出Provider历史信息
        for i in range(self.K):
            self.logger.log(f'\n\n====================Provider-{i} History Info:==================')
            provider_his = self.providers_his[i]
            for key in ITEM_RECORDS:
                if key in provider_his and len(provider_his[key]) > 0:
                    self.logger.log(f'{key}: {sum(provider_his[key])}\t avg_{key}: {np.mean(provider_his[key])}')
            # 添加额外的统计信息
            if 'true_output_tokens' in provider_his and len(provider_his['true_output_tokens']) > 0:
                self.logger.log(f'true_output_tokens: {sum(provider_his["true_output_tokens"])}\t avg_true_output_tokens: {np.mean(provider_his["true_output_tokens"])}')
            if 'reported_output_token_price' in provider_his and len(provider_his['reported_output_token_price']) > 0:
                self.logger.log(f'reported_output_token_price: {sum(provider_his["reported_output_token_price"])}\t avg_reported_output_token_price: {np.mean(provider_his["reported_output_token_price"])}')
            if 'reported_input_token_price' in provider_his and len(provider_his['reported_input_token_price']) > 0:
                self.logger.log(f'reported_input_token_price: {sum(provider_his["reported_input_token_price"])}\t avg_reported_input_token_price: {np.mean(provider_his["reported_input_token_price"])}')
        
        # 输出Provider策略信息
        for i in range(self.K):
            provider = self.providers[i]
            strategy_name = STRAGETY[provider.strategy] if hasattr(provider, 'strategy') else ['unknown']
            best_model = provider.models[0].model_name if len(provider.models) > 0 else 'unknown'
            self.logger.log(f'Provider {i} phase1_exploration: 使用模型={best_model}, 诚实时应使用={best_model}, 策略={strategy_name[0]}')
        
        self.logger.log('phase1_exploration phase result calculated and stored')
        
        self.logger.log("\n\n\n"+"="*20+"phase1_exploration end"+"="*20)

    def phase2_exploitation(self):
        self.logger.log("\n\n\n"+"="*20+"phase2_exploitation start"+"="*20)
        pi_max = self.providers[self.best_provider_idx].models[0].output_token_price
        threshold = self.second_user_utility - self.M * (self.reward_param + self.L * pi_max) / self.gamma
        delta_3 = 0
        for i in range(self.K):
            if i == self.best_provider_idx:
                continue
            avg_v = np.mean(self.providers_his[i]['reward'])
            avg_tau = np.mean(self.providers_his[i]['output_tokens'])
            self.logger.log(f"Phase2 delta_3 components | provider {i}: avg_reward={avg_v:.4f}, avg_output_tokens={avg_tau:.2f}")
            delta_3 += math.log(avg_v) - math.log(avg_tau) - avg_tau / self.L

        # R = int(max(0, self.T - (max(self.delta_1, self.delta_2) + 3) * self.B * self.K))
        self.logger.log(f'delta_1 = {self.delta_1}')
        self.logger.log(f'delta_2 = {self.delta_2}')
        self.logger.log(f'delta_3 = {delta_3}')
        R = self.T - ((self.delta_1 +3) * self.K + self.delta_2 + delta_3)*self.B
        R = int(R)
        nums_remain_tasks = min(R, self.T - self.t)
        self.logger.log(f"  计划委托{(max(self.delta_1, self.delta_2) + 3) * self.B * self.K}次，阈值：{threshold:.4f}")
        self.logger.log(f"  计划委托{nums_remain_tasks}次，阈值：{threshold:.4f}")
        delegation_count = 0
        early_stop_flag = False
        phase2_sum_provider_utility = 0
        for _ in range(nums_remain_tasks):
            if self.t >= self.T:
                break
            self._delegate_task(self.best_provider_idx, phase=2)
            delegation_count += 1
            if delegation_count >= self.B:
                # 计算最近B次委托的平均用户效用
                phase2_avg_user_utility = np.mean(self.providers_his[self.best_provider_idx]['user_utility'][-delegation_count:])
                # phase2_sum_user_utility = np.sum(self.providers_his[self.best_provider_idx]['user_utility'][-delegation_count:])
                # phase2_avg_provider_utility = np.mean(self.providers_his[self.best_provider_idx]['provider_utility'][-delegation_count:])
                phase2_sum_provider_utility = np.sum(self.providers_his[self.best_provider_idx]['provider_utility'][-delegation_count:])
                
                
                if phase2_avg_user_utility < threshold:
                    early_stop_flag = True
                    break
        
        if early_stop_flag:
            self.logger.log(f"  早停原因：最近{delegation_count}次平均用户效用 {phase2_avg_user_utility:.4f} < 阈值 {threshold:.4f}")
            self.logger.log(f"  早停时状态：总任务数t={self.t}，剩余任务数={self.T-self.t}，总provider_utility：{phase2_sum_provider_utility:.4f}")
        else:
            if delegation_count > 0:
                final_avg_user_utility = np.mean(self.providers_his[self.best_provider_idx]['user_utility'][-delegation_count:])
                self.logger.log(f"  正常结束时状态：最近{delegation_count}次平均用户效用 {final_avg_user_utility:.4f} >= 阈值 {threshold:.4f}")
            self.logger.log(f"  正常结束时状态：总任务数t={self.t}，剩余任务数={self.T-self.t}，总provider_utility：{phase2_sum_provider_utility:.4f}")
        
        if not early_stop_flag:

            if self.t < self.T:
                delegation_count = 0 
                for _ in range(min(self.B, self.T - self.t)):
                    self._delegate_task(provider_idx=self.best_provider_idx, phase=3)
                    delegation_count += 1
                phase2_sum_provider_utility = np.sum(self.providers_his[self.best_provider_idx]['provider_utility'][-delegation_count:])
                self.logger.log(f" 奖励在{delegation_count}次委托后停止，总provider utility：{phase2_sum_provider_utility:.4f}")
        
        self.logger.log("\n\n\n"+"="*20+"phase2_exploitation end"+"="*20)
        
        # 输出Provider历史信息
        for i in range(self.K):
            self.logger.log(f'\n\n====================Provider-{i} History Info:==================')
            provider_his = self.providers_his[i]
            for key in ITEM_RECORDS:
                if key in provider_his and len(provider_his[key]) > 0:
                    self.logger.log(f'{key}: {sum(provider_his[key])}\t avg_{key}: {np.mean(provider_his[key])}')
            # 添加额外的统计信息
            if 'true_output_tokens' in provider_his and len(provider_his['true_output_tokens']) > 0:
                self.logger.log(f'true_output_tokens: {sum(provider_his["true_output_tokens"])}\t avg_true_output_tokens: {np.mean(provider_his["true_output_tokens"])}')
            if 'reported_output_token_price' in provider_his and len(provider_his['reported_output_token_price']) > 0:
                self.logger.log(f'reported_output_token_price: {sum(provider_his["reported_output_token_price"])}\t avg_reported_output_token_price: {np.mean(provider_his["reported_output_token_price"])}')
            if 'reported_input_token_price' in provider_his and len(provider_his['reported_input_token_price']) > 0:
                self.logger.log(f'reported_input_token_price: {sum(provider_his["reported_input_token_price"])}\t avg_reported_input_token_price: {np.mean(provider_his["reported_input_token_price"])}')
        
        self.logger.log('phase2_exploitation phase result calculated and stored')


    def phase2_incentive(self):
        self.logger.log("\n\n\n"+"="*20+"phase2_incentive start"+"="*20)
        self.logger.log('='*20+f'Stage2 Incentive: Delegate non-best providers {self.B} Times each'+ '='*20)
        for i in range(self.K):
            if i == self.best_provider_idx:
                continue

            for _ in range(self.B):
                self._delegate_task(i, phase=3)
        
        self.logger.log("\n\n\n"+"="*20+"phase2_incentive end"+"="*20)
        
        # 输出Provider历史信息
        for i in range(self.K):
            self.logger.log(f'\n\n====================Provider-{i} History Info:==================')
            provider_his = self.providers_his[i]
            for key in ITEM_RECORDS:
                if key in provider_his and len(provider_his[key]) > 0:
                    self.logger.log(f'{key}: {sum(provider_his[key])}\t avg_{key}: {np.mean(provider_his[key])}')
            # 添加额外的统计信息
            if 'true_output_tokens' in provider_his and len(provider_his['true_output_tokens']) > 0:
                self.logger.log(f'true_output_tokens: {sum(provider_his["true_output_tokens"])}\t avg_true_output_tokens: {np.mean(provider_his["true_output_tokens"])}')
            if 'reported_output_token_price' in provider_his and len(provider_his['reported_output_token_price']) > 0:
                self.logger.log(f'reported_output_token_price: {sum(provider_his["reported_output_token_price"])}\t avg_reported_output_token_price: {np.mean(provider_his["reported_output_token_price"])}')
            if 'reported_input_token_price' in provider_his and len(provider_his['reported_input_token_price']) > 0:
                self.logger.log(f'reported_input_token_price: {sum(provider_his["reported_input_token_price"])}\t avg_reported_input_token_price: {np.mean(provider_his["reported_input_token_price"])}')
        
        self.logger.log('phase2_incentive phase result calculated and stored')

    def phase3_utility_based(self):
        self.logger.log("\n\n\n"+"="*20+"phase3_utility_based start"+"="*20)
        for i in range(self.K):
            if self.t >= self.T:
                return
            avg_reward = np.mean(self.providers_his[i]['reward'])
            avg_output_tokens = np.mean(self.providers_his[i]['output_tokens'])
            
            delta = self.delta_1 + math.log(avg_reward) - math.log(avg_output_tokens) - avg_output_tokens / self.L
            self.logger.log(f'for provider {i}: avg_reward:{avg_reward}, avg_output_tokens: {avg_output_tokens}, delta: {delta}')
            self.logger.log(f"delta: {delta}")
            if delta > 0:
                
                num_delegations = int(delta) * self.B
                num_delegations = min(self.T- self.t, num_delegations)
                self.logger.log(f'provider {i}: 计划委托次数 {num_delegations} (整数部分)')

                delegation_count = 0
                for _ in range(num_delegations):
                    self._delegate_task(i, phase=4)
                    delegation_count += 1
                
                phase3_sum_provider_utility = np.sum(self.providers_his[i]['provider_utility'][-delegation_count:])
                self.logger.log(f'provider {i}: 被委托了{delegation_count}次: 收了provider_utility:{phase3_sum_provider_utility if delegation_count>0 else 0}')

                frac_part = delta - int(delta)
            
                direct_delegations = int(self.B * frac_part)
                direct_delegations = min(direct_delegations, self.T - self.t)

                delegation_count = 0
                for _ in range(direct_delegations):
                    self._delegate_task(i, phase=4)
                    delegation_count += 1
                if delegation_count > 0:
                    phase3_sum_provider_utility = np.sum(self.providers_his[i]['provider_utility'][-delegation_count:])
                    self.logger.log(f'provider {i} 被委托了{delegation_count}次: 收了provider_utility:{phase3_sum_provider_utility}')
                

                remaining_frac = self.B * frac_part - int(self.B * frac_part)
                if random.random() < remaining_frac and self.t < self.T:
                    self._delegate_task(i, phase=4)
                    utility = self.providers_his[i]['provider_utility'][-1]
                    self.logger.log(f'狗运：provider {i} 被委托了1次: 收了provider_utility:{utility}')
            else:
                self.logger.log(f'provider {i}: delta<=0，不进行委托')
        
        self.logger.log("\n\n\n"+"="*20+"phase3_utility_based end"+"="*20)
        
        # 输出Provider历史信息
        for i in range(self.K):
            self.logger.log(f'\n\n====================Provider-{i} History Info:==================')
            provider_his = self.providers_his[i]
            for key in ITEM_RECORDS:
                if key in provider_his and len(provider_his[key]) > 0:
                    self.logger.log(f'{key}: {sum(provider_his[key])}\t avg_{key}: {np.mean(provider_his[key])}')
            # 添加额外的统计信息
            if 'true_output_tokens' in provider_his and len(provider_his['true_output_tokens']) > 0:
                self.logger.log(f'true_output_tokens: {sum(provider_his["true_output_tokens"])}\t avg_true_output_tokens: {np.mean(provider_his["true_output_tokens"])}')
            if 'reported_output_token_price' in provider_his and len(provider_his['reported_output_token_price']) > 0:
                self.logger.log(f'reported_output_token_price: {sum(provider_his["reported_output_token_price"])}\t avg_reported_output_token_price: {np.mean(provider_his["reported_output_token_price"])}')
            if 'reported_input_token_price' in provider_his and len(provider_his['reported_input_token_price']) > 0:
                self.logger.log(f'reported_input_token_price: {sum(provider_his["reported_input_token_price"])}\t avg_reported_input_token_price: {np.mean(provider_his["reported_input_token_price"])}')
        
        self.logger.log('phase3_utility_based phase result calculated and stored')
    
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
        
        self.logger.log('Final game result summary with phase stats generated')
        print(game_result)
        json.dump(game_result, open(self.output_dir / 'result.json', 'w'), indent=2)



    def run_game(self):
        self.phase1_exploration()
        self.phase2_exploitation()
        self.phase2_incentive()
        self.phase3_utility_based()
        self.get_result()
      

