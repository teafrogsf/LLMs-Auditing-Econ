

import numpy as np


class ProviderHisotryManager:
    def __init__(self, idx) -> None:
        self.id = idx
        self.avg_v = None
        self.avg_tau = None
        self.data = {
            'model_id': [],
            'reported_model_id': [],
            'input_tokens': np.array([]) ,             # 真实的
            'output_tokens':np.array([]),             # 真实的
            'reported_output_tokens': np.array([]),   # 汇报的
            'reported_price': np.array([]),           # 汇报的
            'costs': np.array([]),                    # 真实的
            'rewards': np.array([]),                  # 真实的
            'user_utility': np.array([]),             
            'provider_utility': np.array([]),
            'avg_user_utility': None,
            'avg_provider_utility': None,
            'delta': None
            }

        self.iter_key = ['input_tokens','output_tokens', 'reported_output_tokens',  'reported_price', 'costs',
                    'rewards', 'user_utility', 'provider_utility',]
    
    def append(self, result):
        self.data['model_id'].append(
            {
                "id": len(self.data['model_id']),
                "batch_size": result['batch_size'],
                "model_id": result['model_id']
            }
        )

        self.data['reported_model_id'].append(
            {
                "id": len(self.data['reported_model_id']),
                "batch_size": result['batch_size'],
                "model_id": result['reported_model_id']
            }
        )

        for key in self.iter_key:
           
            try: 
                self.data[key] = np.concatenate([self.data[key], result[key]])
            except Exception as e:
                print(key)
                print(result[key])
                raise e


        self.data['avg_user_utility'] = np.mean(self.data['user_utility'])
        self.data['avg_provider_utility'] = np.mean(self.data['provider_utility'])

    
    def get_recent_user_utility(self, k):
        return np.mean(self.data['user_utility'][-k:])


    def get_avg_user_utility(self):
        return float(self.data['avg_user_utility'])

    def get_avg_provider_utility(self):
        return float(self.data['avg_provider_utility'])

    
    def get_avg_reward(self):
        return float(np.mean(self.data['rewards']))

    
    def get_avg_reported_output_tokens(self):
        return float(np.mean(self.data['reported_output_tokens']))

    def get_recent_avg_provider_utility(self, k):
        return float(np.mean(self.data['provider_utility'][-k:]))
    
    def get_recent_sum_provider_utility(self, k):
        return float(np.sum(self.data['provider_utility'][-k:]))

    def get_delegation_nums(self):
        return len(self.data['input_tokens'])

    
    def get_result(self):
        result = {
            'delegations': self.get_delegation_nums(),
            'delta': self.data['delta']
            }
        for key in self.iter_key:
            result[f'total_{key}'] = np.sum(self.data[key])
        
        return result
    
    def log_info(self, logger):
        logger.log(f'\n\n====================Provider-{self.id} History Info:==================')
        for key in self.iter_key:
            logger.log(f'total {key}: {np.sum(self.data[key])}')
            
            logger.log(f'avg {key}: {np.mean(self.data[key])}')
    def set_delta(self, delta):
        self.data['delta'] = delta


    def get_avg_v(self, num):
        avg_v = np.mean(self.data['rewards'][:num])
        return avg_v

    def get_avg_tau(self, num):
        avg_tau = np.mean(self.data['reported_output_tokens'][:num])
        return avg_tau