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


def init_config(path):
    config = yaml.safe_load(open(path))
    T = int(config['num_tasks'])
    epsilon = float(config['epsilon'])
    K = len(config['providers'])
    B = int(T ** (2 * epsilon))  # B = T^(2ϵ)
    M = (T ** (-epsilon)) * math.log(K * T)  # M = T^(-ϵ)ln(KT)
    config['B'] = B
    config['K'] = K
    config['M'] = M
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config/toy_game/default.yaml')
    parser.add_argument('--output-dir', type=str, default='./outputs/toy_game/v1')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = init_config(args.config)
    config['output_dir'] = args.output_dir
    
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(name='llm-game', path=os.path.join(args.output_dir, 'log.txt'))
    
    


    ### step3: hyper para init
    config = init_config(args.config)
    logger.log('='*20+'Config'+"="*20)
 
    logger.log(json.dumps(config, indent=2))

    ### step4: scenarios making
    MODEL_CHOICES = list(range(10))
    scenarios = [(item, 0, 0) for item in range(10)]
    
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


