import random
import numpy as np
import json
import yaml
import argparse
import os
import math
import itertools
from src.simulator.toy_game_manager import GameManager
from src.utils import Logger


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def init_config(path, T, eps):
    config = yaml.safe_load(open(path))
    config['num_tasks'] = T
    config['epsilon'] = eps
    K = len(config['providers'])
    B = int(T ** (2 * eps))  # B = T^(2ϵ)
    M = (T ** (-eps)) * math.log(K * T)  # M = T^(-ϵ)ln(KT)
    config['B'] = B
    config['K'] = K
    config['M'] = M
    print(config)
    return config


    
def check_results(results, config):
    B = config['B']
    R = results[1]['R']
    delegations = [results[i]['providers'][0]['delegations'] for i in range(6)]
    group_big = [delegations[0], delegations[1], delegations[5]]
    group_small = [delegations[2], delegations[3], delegations[4]]

    if max(group_big) / min(group_big) > 1.1:
        return False
    
    if min(group_big) <= R/2 + max(group_small):
        return False
    
    return True


def run_one_game(cfg_path, T, eps, final=False):
    config = init_config(cfg_path, T, eps)
    output_dir = f'./outputs/toy_game/T-eps/T{T}/eps{eps}{"_final" if final else ""}'
    os.makedirs(output_dir, exist_ok=True)
    logger = Logger(name=f'llm-game-{T}-{eps}', path=os.path.join(output_dir, 'log.txt'))
    logger.log('='*20+'Config'+"="*20)
 
    logger.log(json.dumps(config, indent=2))
    MODEL_CHOICES = list(range(6))
    scenarios = [(i,1,1) for i in MODEL_CHOICES]
    print(scenarios)
    
    logger.log(f'{len(scenarios)}')
    results = []
    for sc in scenarios:
        logger.log(f'run {sc}')
        config['output_dir'] = os.path.join(output_dir, "-".join([str(item) for item in sc]))
        for i in range(len(config['providers'])):
            config['providers'][i]['strategy'] = sc[i]
        toy_manager = GameManager(config)
        result = toy_manager.run_game()
        # print(result)
        results.append(result)
    return check_results(results, config)



def main():
    
    cfg = 'config/toy_game/default.yaml'
    T_cand = [10**i for i in range(2, 6)]
    t_eps = {}
    for T in T_cand:
        # T = in/
    
        left = 0
        right = 0.5
        eps = 0.25
        last_eps = eps
        while True:
            if run_one_game(cfg, T, eps):
                right = eps
            else:
                left = eps
            last_eps = eps
            eps = (right + left) / 2
            if abs(last_eps - eps) < 1e-3:
                t_eps[T] = eps
                break
        
        
        run_one_game(cfg, T, eps, True)
        json.dump(t_eps, open('outputs/toy_game/T-eps/map.json', 'w'), indent=2)


            

                
            












if __name__ == '__main__':
    main()


