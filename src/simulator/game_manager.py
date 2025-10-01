import json
import math
import os
import random
from typing import Dict, List
import numpy as np
import yaml

from src.simulator.utils import ProviderHisotryManager
from src.utils import Logger
from pathlib import Path

ITEM_RECORDS = ['tokens','input_tokens','output_tokens','price','cost',
                    'reward', 'user_utility', 'provider_utility',]

STRAGETY = [
   ['honest', 'lie_ours', 'lie_all', 'lie_all'],
    ['honest', 'honest', 'honest', 'honest'],
    ['honest', 'lie_model', 'lie_all', 'lie_all'],
    ['honest', 'lie_token', 'lie_all', 'lie_all'],
    ['lie_all', 'lie_all', 'lie_all', 'lie_all'],
    ['honest', 'lie_second_best', 'lie_all', 'lie_all'],
    
    
]

MODEL_CONFIG = yaml.safe_load(open('config/nl_graph/model_config.yaml'))
print(MODEL_CONFIG)
def load_records(path):
    return [json.loads(item) for item in open(path).readlines()]

class RealModel:
    def __init__(self, config) -> None:
        self.id = None
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
        self.data = np.load(f'data/local_records/nlgraph_new/{self.model_name}_test_result.npz')
        self.task_len = len(self.data['scores'])
    
    def generate(self, task_ids, input_tokens, _, batch_size):
        assert len(task_ids) == batch_size
   
        score = self.data['scores'][task_ids]
        output_tokens = self.data['output_tokens'][task_ids]
        real_price = input_tokens * self.input_token_price + output_tokens * self.output_token_price
        cost = real_price * self.eta
        return {
            "model_id": self.id,
            "tokens": input_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "rewards": score * self.reward_param,
            "costs": cost,
        }

        

class Provider:
    models: List[RealModel]
    def __init__(self, config, logger) -> None:
        self.num_models = len(config['models'])
        self.eta = float(config['eta'])
        self.strategy = int(config['strategy'])
        # self.reward_scale = config['reward_scale']
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
        for idx, model in enumerate(models):
            model.id = idx
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

        self.logger.log('Model utility mu:')
        self.logger.log([model.utility_mu for model in models])
        self.logger.log(f'Provider strategy schedule: {STRAGETY[self.strategy]}')
    
    
    def honset_run(self, task_ids, input_tokens, L, batch_size):
        model_run_result = self.models[0].generate(task_ids, input_tokens, L, batch_size)
        model_run_result['reported_output_tokens'] = model_run_result['output_tokens']
        model_run_result['reported_model_id'] = model_run_result['model_id']
        return model_run_result

    
    def lie_run(self, task_ids, real_model_idx, input_tokens, L, batch_size, reported_model_id, token_lie_add):
        model_run_result = self.models[real_model_idx].generate(task_ids, input_tokens, L, batch_size)
        model_run_result['reported_output_tokens'] = np.clip(model_run_result['output_tokens'] + token_lie_add, None, L)
        model_run_result['reported_model_id'] = reported_model_id
        return model_run_result
        

    def run_task(self, task_ids, input_tokens, phase, L, second_utility=None, batch_size=1):
        strategy = STRAGETY[self.strategy][phase-1]
        # print(strategy)
        if strategy == 'honest':
            result = self.honset_run(task_ids, input_tokens, L, batch_size)
            
        elif strategy == 'lie_all':
            real_model_id = self.num_models - 1
            reported_model_id = 0
            token_lie_add = L
            result = self.lie_run(task_ids, real_model_id, input_tokens, L, batch_size, reported_model_id, token_lie_add)
        
        # todo first phase
        elif strategy == 'lie_ours':
            if second_utility is None:
                raise ValueError(f'second utility should not be None')
            exp_model_idx = 0
            exp_model_pro_utility = 0
            exp_lie_add = 0

            for idx, model in enumerate(self.models):
                if model.utility_mu < second_utility:
                    continue

                lie_addition = (model.score_mu * model.reward_param - second_utility - np.mean(input_tokens)*\
                    self.max_input_token_price) / self.max_output_token_price - model.output_tokens_mu
                lie_addition = int(lie_addition) 
                if lie_addition < 0:
                    lie_addition = 0
                model_pro_u =self.max_input_token_price*np.mean(input_tokens) + \
                    self.max_output_token_price* (min(L, model.output_tokens_mu+ lie_addition)) -\
                     (model.output_tokens_mu * model.output_token_price + np.mean(input_tokens)* model.input_token_price)



                if exp_model_pro_utility < model_pro_u:
                    exp_model_pro_utility = model_pro_u
                    exp_model_idx = idx
                    exp_lie_add = lie_addition
                    
                
            # self.logger.log(f'lie model is {exp_model_idx}')
            # self.logger.log(f'lie_addition is {exp_lie_add}')
            result = self.lie_run(task_ids, exp_model_idx, input_tokens, L, batch_size, 0, exp_lie_add)


        elif strategy == 'lie_model':
            result = self.lie_run(task_ids, -1, input_tokens, L, batch_size, 0, 0)
        
        elif strategy == 'lie_second_best':
            if second_utility is None:
                raise ValueError(f'second utility should not be None')
            exp_model_idx = 0
            exp_model_pro_utility = 0


            for idx, model in enumerate(self.models):
                if model.utility_mu < second_utility:
                    continue

                
                model_pro_u =self.max_input_token_price*np.mean(input_tokens) + self.max_output_token_price* model.output_tokens_mu -\
                     (model.output_tokens_mu * model.output_token_price + np.mean(input_tokens)* model.input_token_price)


                if exp_model_pro_utility < model_pro_u:
                    exp_model_pro_utility = model_pro_u
                    exp_model_idx = idx

                    
                
      
            result = self.lie_run(task_ids, exp_model_idx, input_tokens, L, batch_size, 0, 0)

        
        
        elif strategy == 'lie_token':
            result = self.lie_run(task_ids, 0, input_tokens, L, batch_size, 0, L)

        else:
            raise ValueError(f'{self.strategy} is not supported!')

        # print([key for key in result])
        result['batch_size'] = batch_size
        reported_model_id = result['reported_model_id']
        reported_output_tokens = result['reported_output_tokens']
        input_tokens = result['input_tokens']
        rewards = result['rewards']
        costs = result['costs']
        reported_input_token_price = self.models[reported_model_id].input_token_price
        reported_output_token_price = self.models[reported_model_id].output_token_price
        
        reported_price = input_tokens * reported_input_token_price + reported_output_tokens * reported_output_token_price

        user_utility = rewards - reported_price
        provider_utility = reported_price * self.eta - costs
        # print(self.eta)

        result['reported_price'] = reported_price
        result['user_utility'] = user_utility
        result['provider_utility'] = provider_utility
 
        return result



    
class GameManager:
    game_config: Dict
    providers: List[Provider]
    providers_his: List[ProviderHisotryManager]
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
        self.input_tokens = np.load("data/local_records/nlgraph_new/input_tokens.npz")['data']
        self.t = 0
        self.L = None
        self.delta_1 = None
        self.delta_2 = None
        self.delta_3 = None
        # self.delta = None
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

            self.providers_his.append(ProviderHisotryManager(provider_config['id']))

    
        self.L = max([provider.max_tokens for provider in self.providers])
                
        # min_mu_r = min(mu_r)
        # max_mu_l = max(mu_l)
        mu_r = np.array(mu_r)
        mu_l = np.array(mu_l)
        max_mu_l = max(mu_l)
        mu_r_div_mu_l = mu_r / mu_l

        min_murl = min(mu_r_div_mu_l)

        # self.logger.log(f"min_mu_r={min_mu_r:.4f}, max_mu_l={max_mu_l:.4f}, L: {self.L}")
        self.delta_1 = -math.log(min_murl)  + max_mu_l / self.L + 1
        self.delta_2 = math.log(self.reward_param)

        
        
        self.logger.log(f"delta_1={self.delta_1:.4f}, delta_2={self.delta_2:.4f}")
     
    def _init_output_dir_logger(self):
        os.makedirs(self.output_dir, exist_ok=True)
        log_path = self.output_dir / 'log.txt'
        self.logger = Logger(self.output_dir.name, log_path)
        self.logger.log(f'log to file {log_path}, logger name is {self.output_dir.name}')
        

    def _delegate_task(self, provider_idx: int, phase: int, batch_size: int, log_flag=False):
        if log_flag:
            self.logger.log(f'Delegate provider {provider_idx} {batch_size} tasks.')

        if self.t >= self.T:
            return None
        
        task_ids = np.array(list(range(self.t, self.t+batch_size))).astype(int) % len(self.input_tokens)
    
        input_tokens = self.input_tokens[task_ids]
        result = self.providers[provider_idx].run_task(task_ids, input_tokens, phase, self.L, self.second_user_utility, batch_size)
        
        self.providers_his[provider_idx].append(result)
        self.t += batch_size
        return True


    def log_provider_info(self):
        for i in range(self.K):
            self.providers_his[i].log_info(self.logger)
      

    def phase1_exploration(self):

        self.logger.log('='*20+f'Stage1: Delegate each provider {self.B} Times'+ '='*20)
        for i in range(len(self.providers)):
            self._delegate_task(i, 1, self.B, True)


        sorted_provider_idx = sorted(list(range(self.K)), key=lambda idx: self.providers_his[idx].get_avg_user_utility())
        self.logger.log(f"Providers sorted by avg_user_utility: {sorted_provider_idx}")

        for i in range(self.K):
            avg_provider_utility = self.providers_his[i].get_avg_provider_utility()
            avg_user_utility = self.providers_his[i].get_avg_user_utility()

            self.logger.log(f"Provider {i} avg_user_utility: {avg_user_utility:.4f}")
            self.logger.log(f"Provider {i} avg_provider_utility: {avg_provider_utility:.4f}")
        
        self.best_provider_idx = sorted_provider_idx[-1]
        second_provider_idx = sorted_provider_idx[-2]
       
        best_user_utility = self.providers_his[self.best_provider_idx].get_avg_user_utility()
        self.second_user_utility = self.providers_his[second_provider_idx].get_avg_user_utility()

        self.logger.log(f"  阶段1完成，最佳服务商：{self.best_provider_idx}，平均效用：{best_user_utility:.4f}")
        self.logger.log(f"  第二好服务商{second_provider_idx}\t 第二好效用：{self.second_user_utility:.4f}")

    def phase2_exploitation(self):
        pi_max = self.providers[self.best_provider_idx].models[0].output_token_price
        threshold = self.second_user_utility - self.M * (self.reward_param + self.L * pi_max) / self.gamma
        self.logger.log(f'后面的东西是：{self.M * (self.reward_param + self.L * pi_max) / self.gamma}')

        delta_3 = 0
        for i in range(self.K):
            if i == self.best_provider_idx:
                continue
            avg_v = self.providers_his[i].get_avg_v(self.B)
            avg_tau = self.providers_his[i].get_avg_tau(self.B)

            self.logger.log(f"Phase2 delta_3 components | provider {i}: avg_reward={avg_v:.4f}, avg_output_tokens={avg_tau:.2f}")
            delta_3 += math.log(avg_v) - math.log(avg_tau) - avg_tau / self.L

        # R = int(max(0, self.T - (max(self.delta_1, self.delta_2) + 3) * self.B * self.K))
        self.logger.log(f'delta_1 = {self.delta_1}')
        self.logger.log(f'delta_2 = {self.delta_2}')
        self.logger.log(f'delta_3 = {delta_3}')
        
        R = self.T - ((self.delta_1 +3) * self.K + self.delta_2 + delta_3)*self.B
        self.logger.log(f'R = T - ((delta_1 + 3) * K + delta_2 + delta_3) * B = {self.T} - (({self.delta_1} + 3) * {self.K} + {self.delta_2} + {delta_3}) * {self.B} = {R}')
        R = int(R)
        nums_remain_tasks = min(R, self.T - self.t)
        self.logger.log(f"  计划委托{nums_remain_tasks}次，阈值：{threshold:.4f}")
     
        early_stop_flag = False
        phase2_sum_provider_utility = 0
        self._delegate_task(self.best_provider_idx, phase=2, batch_size=self.B)
        nums_remain_tasks = nums_remain_tasks - self.B
        remain_batch_delegation_nums = nums_remain_tasks // self.B
        last_num = nums_remain_tasks % self.B
        delegation_count = self.B

        for _ in range(remain_batch_delegation_nums):
            if self.t >= self.T:
                break
            
            self._delegate_task(self.best_provider_idx, phase=2, batch_size=self.B)
            delegation_count += self.B
            phase2_avg_user_utility = self.providers_his[self.best_provider_idx].get_recent_user_utility(delegation_count)
            
            if phase2_avg_user_utility < threshold:
                    
                early_stop_flag = True
                self.logger.log(f"  早停触发：最近{delegation_count}次平均用户效用 {phase2_avg_user_utility:.4f} < 阈值 {threshold:.4f}")
                break
        else:
            self._delegate_task(self.best_provider_idx, phase=2, batch_size=last_num)
            delegation_count += last_num
            
        phase2_sum_provider_utility = np.sum(self.providers_his[self.best_provider_idx].get_recent_sum_provider_utility(delegation_count))

        self.logger.log(f"  在{delegation_count}次委托后停止，总provider_utility：{phase2_sum_provider_utility:.4f}")
        if not early_stop_flag:

            if self.t < self.T:
                bonust_batch = min(self.B, self.T - self.t)
                # print(f"bonust_batch: {bonust_batch}")
                self._delegate_task(provider_idx=self.best_provider_idx, phase=3, batch_size=bonust_batch)
                delegation_count = bonust_batch
                phase2_sum_provider_utility = np.sum(self.providers_his[self.best_provider_idx].get_recent_sum_provider_utility(delegation_count))
                self.logger.log(f" 奖励在{delegation_count}次委托后停止，总provider utility：{phase2_sum_provider_utility:.4f}")


    def phase2_incentive(self):
        self.logger.log('='*20+f'Stage2 Incentive: Delegate non-best providers {self.B} Times each'+ '='*20)
        for i in range(self.K):
            if i == self.best_provider_idx:
                continue


            self._delegate_task(i, phase=3, batch_size=self.B, log_flag=True)

    def phase3_utility_based(self):
        for i in range(self.K):
            if self.t >= self.T:
                return
            avg_reward = self.providers_his[i].get_avg_v(self.B)
            avg_output_tokens = self.providers_his[i].get_avg_tau(self.B)
            self.logger.log(f'for provider {i}: log_avg_reward={math.log(avg_reward)}, log_avg_output_tokens={math.log(avg_output_tokens)}, avg_output_tokens/L = {avg_output_tokens/self.L}, L={self.L}')
            self.logger.log(f'delta_1={self.delta_1}')
            delta = self.delta_1 + math.log(avg_reward) - math.log(avg_output_tokens) - (avg_output_tokens / self.L)

            self.providers_his[i].set_delta(delta)
            self.logger.log(f'for provider {i}: avg_reward:{avg_reward}, avg_output_tokens: {avg_output_tokens}, delta: {delta}')
            self.logger.log(f"delta: {delta}")
            if delta > 0:
                
                num_delegations = int(delta * self.B)
                num_delegations = min(self.T- self.t, num_delegations)
                self.logger.log(f'provider {i}: 计划委托次数 {num_delegations} (整数部分)')

                delegation_count = 0
                self._delegate_task(i, phase=4, batch_size=num_delegations)
                delegation_count = num_delegations
                phase3_sum_provider_utility = np.sum(self.providers_his[i].get_recent_sum_provider_utility(num_delegations))

                self.logger.log(f'provider {i}: 被委托了{delegation_count}次: 收了provider_utility:{phase3_sum_provider_utility if delegation_count>0 else 0}')

                frac_part = delta * self.B - num_delegations
                if random.random() < frac_part:
                    num_delegations = min(1, self.T - self.t)
                    self.logger.log(f'provider {i}: 小数部分触发，追加委托 {num_delegations} 次')
                    self._delegate_task(i, phase=4, batch_size=num_delegations)
                    phase3_sum_provider_utility = np.sum(self.providers_his[i].get_recent_sum_provider_utility(num_delegations))
                    self.logger.log(f'狗运：provider {i}  被委托了{delegation_count}次: 收了provider_utility:{phase3_sum_provider_utility if delegation_count>0 else 0} ')
            else:
                self.logger.log(f'provider {i}: delta<=0，不进行委托')
    
    def get_result(self):
        game_result = {
            'total_time': self.T,
            'total_delegations': self.t,
            'best_provider_idx': self.best_provider_idx,
            'providers': []
        }
        for i in range(self.K):
            provider_result = self.providers_his[i].get_result()
            game_result['providers'].append(provider_result)
        json.dump(game_result, open(self.output_dir / 'result.json', 'w'), indent=2)



    def run_game(self):
        self.phase1_exploration()
        self.logger.log("\n\n\n"+"="*20+"phase1_exploration end"+"="*20)
        self.log_provider_info()

        self.logger.log("\n\n\n"+"="*20+"phase2_exploitation start"+"="*20)
        self.phase2_exploitation()
        self.logger.log("\n\n\n"+"="*20+"phase2_exploitation end"+"="*20)
        self.log_provider_info()

        self.logger.log("\n\n\n"+"="*20+"phase2_incentive start"+"="*20)
        self.phase2_incentive()
        self.log_provider_info()
        self.logger.log("\n\n\n"+"="*20+"phase2_incentive end"+"="*20)

        self.logger.log("\n\n\n"+"="*20+"phase3_utility_based start"+"="*20)
        self.phase3_utility_based()
        self.log_provider_info()
        self.get_result()
        self.logger.log("\n\n\n"+"="*20+"phase3_utility_based end"+"="*20)

