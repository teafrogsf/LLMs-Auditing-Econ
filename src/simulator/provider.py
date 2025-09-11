import random
import numpy as np
from src.utils import load_jsonl
from src.model import MODEL_PRICING

class ProviderManager:
    def __init__(self, config):
        self.id = config['id']
        self.models = config['models']
        self.strategy = config['strategy']
        self.eta = config['eta']
        self.reward_param = config['reward_param']
        self.task_history = []
        self.cost_hisotry = []
        self.price_history = []
        self.utility_history = []
        self.reward_history = []
        self.cumulative_reward = 0
        self._load_local_records()
        self._init_models_info()
        
 

    def _init_models_info(self):
        self.best_model = max(self.models, key=lambda x: MODEL_PRICING[x]['input'] + MODEL_PRICING[x]['output'])
        self.best_model_price = MODEL_PRICING[self.best_model]
        self.worst_model = min(self.models, key=lambda x: MODEL_PRICING[x]['input'] + MODEL_PRICING[x]['output'])
        self.worst_model_price = MODEL_PRICING[self.worst_model]

    def _load_local_records(self):
        self.records = {}
        for model in self.models:
            self.records[model] = load_jsonl(f'data/local_records/{model}_test_result.jsonl')


    def get_money(self, result, model):
        model_price = MODEL_PRICING[model]
        return model_price['input'] * result['input_tokens'] + model_price['output'] * result['output_tokens']

    def select_model(self, phase, second_utility, R):
        
        if self.strategy == 'honest':
            return self.best_model
        elif self.strategy == 'worst':
            return self.worst_model
        elif self.strategy == 'random':
            return random.choice(self.models)
        elif self.strategy == 'ours':
            if phase == 1:
                model_used = self.best_model
            elif phase == 2:
                if R is None:
                    raise ValueError(f"our strategy R is None")
                if second_utility is None:
                    raise ValueError(f"our strategy R is None")
                threshold = R * second_utility
                if self.cumulative_reward < threshold:
                    model_used = self.best_model
                else:
                    model_used = self.worst_model
            else:
                model_used = self.worst_model              
        elif self.strategy == 'h1w2':
            if phase == 1:
                model_used = self.best_model
            else:
                model_used = self.worst_model
            
        elif self.strategy == 'w1h2':
            if phase == 1:
                model_used = self.worst_model
            elif phase == 2:
                model_used = self.best_model
            else:
                model_used = self.worst_model
        
        else:
            raise ValueError(f'No stragety named {self.strategy}')

        return model_used


    def run_task(self, task_id, phase, second_utility=None, R=None):
        model_used = self.select_model(phase, second_utility, R)            
        result = self.records[model_used][task_id]
        price = self.get_money(result, self.best_model)
        cost = self.get_money(result, model_used) * self.eta
        reward = self.reward_param * result['score']
        utility = reward - price
        result['model'] = model_used
        result['price'] = price
        result['cost'] = cost
        result['reward'] = reward
        result['phase'] = phase
        result['task_id'] = task_id
        result['utility'] = utility
        self.cumulative_reward += reward
        self.task_history.append(result)
        self.cost_hisotry.append(cost)
        self.price_history.append(price)
        self.utility_history.append(utility)
        self.reward_history.append(reward) 
        return result

    def get_avg_utility(self):
        return np.mean(self.utility_history)

    def get_avg_reward(self):
        return np.mean(self.reward_history)

    def get_recent_avg_utility(self, k):
        return np.mean(self.utility_history[-k:])
    


    def get_results(self):
        num_delegations = len(self.task_history)
        total_price = sum(self.price_history)
        total_reward = sum(self.reward_history)
        total_cost = sum(self.cost_hisotry)
        avg_reward = np.mean(self.reward_history)
        profit = total_reward - total_price
        total_input_token = sum([item['input_tokens'] for item in self.task_history])
        total_output_token = sum([item['output_tokens'] for item in self.task_history])
        total_tokens = total_input_token + total_output_token
        provider_utility = self.eta * total_price - total_cost
        return {
            'delegations': len(self.task_history),
            'total_price': total_price,
            'total_reward': total_reward,
            'total_cost': total_cost,
            'avg_reward': avg_reward,  
            'profit': profit,
            'provider_utility': provider_utility,
            'total_prompt_tokens': total_input_token,   
            'total_completion_tokens': total_output_token,
            'total_tokens': total_tokens,
            'avg_tokens_per_delegation': total_tokens / num_delegations if num_delegations > 0 else 0
            }

