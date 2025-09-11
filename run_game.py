import random
from tkinter import N
import numpy as np
import itertools
import json
import yaml
import argparse
import os
import math
from src.utils import Logger
from src.simulator import GameManager

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def init_config(path):
    config = yaml.safe_load(open(path))
    T = config['num_tasks']
    epsilon = config['epsilon']
    K = len(config['providers'])
    B = int(T ** (2 * epsilon))  # B = T^(2ϵ)
    M = (T ** (-epsilon)) * math.log(K * T)  # M = T^(-ϵ)ln(KT)
    config['B'] = B
    config['K'] = K
    config['M'] = M

    return config

def main():
    
    ### step1: load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='./config/default.yaml')
    parser.add_argument('--output-dir', type=str, default='./outputs/default/')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    ### step2: init logger and output dir
    # if not args.debug:
    #     if os.path.exists(args.output_dir):
    #         raise IOError(f'{args.output_dir} exists')
    #     os.makedirs(args.output_dir)
    # else:
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(name='llm-game', path=os.path.join(args.output_dir, 'log.txt'))
    
    


    ### step3: hyper para init
    config = init_config(args.config)
    logger.log('='*20+'Config'+"*"*20)
    logger.log(json.dumps(config, indent=2))

    ### step4: scenarios making
    CHOICES = ['honest', 'ours', 'worst', 'random', 'h1w2', 'w1h2']

    scenarios = list(itertools.product(CHOICES, repeat=3))
    logger.log(f'{len(scenarios)}')
    for sc in scenarios:
        logger.log(f'run {sc}')
        config['output_dir'] = os.path.join(args.output_dir, "-".join(sc))
        for i in range(len(config['providers'])):
            config['providers'][i]['strategy'] = sc[i]
        game_manager = GameManager(config)
        game_manager.run_game()
    


if __name__ == "__main__":
    # create_example_scenario()
    main()