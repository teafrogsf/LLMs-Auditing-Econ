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


def init_config(args):
    config = yaml.safe_load(open(args.config))
    T = args.T
    epsilon = args.eps
    K = len(config['providers'])
    B = int(T ** (2 * epsilon))  # B = T^(2ϵ)
    M = (T ** (-epsilon)) * math.log(K * T)  # M = T^(-ϵ)ln(KT)
    config['eps'] = args.eps
    config['num_tasks'] = args.T
    config['reward_param'] = args.reward_param
    config['gamma'] = args.gamma
    config['B'] = B
    config['K'] = K
    config['M'] = M
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', type=int, default=1e6)
    parser.add_argument('--eps', '-e', type=float, default=0.3)
    parser.add_argument('--reward_param', '-r', type=float, default=5)
    parser.add_argument('--gamma', '-g', type=float, default=2)
    parser.add_argument('--config', '-c', type=str, default='config/toy_game/default.yaml')
    parser.add_argument('--output-dir', type=str, default='./outputs/nl_graph/vtest')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = init_config(args)
    config['output_dir'] = args.output_dir
    
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(name=f'llm-game-{args.T}', path=os.path.join(args.output_dir, 'log.txt'))
    
    


    ### step3: hyper para init
    config = init_config(args)
    logger.log('='*20+'Config'+"="*20)
 
    logger.log(json.dumps(config, indent=2))

    ### step4: scenarios making
    MODEL_CHOICES = list(range(6))
    # scenarios = [(item, 0, 0) for item in range(11)]
    scenarios = list(itertools.product(MODEL_CHOICES, repeat=3))
    print(scenarios)
    json.dump(config, open(os.path.join(args.output_dir, 'config.json'), 'w'), indent=2)
    
    logger.log(f'{len(scenarios)}')
    for sc in scenarios:
        logger.log(f'run {sc}')
        config['output_dir'] = os.path.join(args.output_dir, "-".join([str(item) for item in sc]))
        for i in range(len(config['providers'])):
            config['providers'][i]['strategy'] = sc[i]
        toy_manager = GameManager(config)
        toy_manager.run_game()
    



if __name__ == '__main__':
    main()


