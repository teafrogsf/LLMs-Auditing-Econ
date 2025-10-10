
import itertools
import json
import numpy as np


def get_result_all_ours(i, result_path):
    CHOICES = ['0']
    provider_cost = {k: [] for k in CHOICES}
    provider_reward = {k: [] for k in CHOICES}
    provider_price = {k: [] for k in CHOICES}
    provider_utility = {k: [] for k in CHOICES}
    user_utility = {k: [] for k in CHOICES}
    provider_results = {k: {} for k in CHOICES}
    provider_delegations = {k: [] for k in CHOICES}
    provider_delta = {k: [] for k in CHOICES}

    
    strategy = '0'
    results =[ json.load(open(result_path / '0-0-0' / 'result.json'))]

    for result in results:
        provider_cost[strategy].append(result['providers'][i]['total_costs'])
        provider_price[strategy].append(result['providers'][i]['total_reported_price'])
        provider_reward[strategy].append(result['providers'][i]['total_rewards'])
        provider_utility[strategy].append(result['providers'][i]['total_provider_utility'])
        user_utility[strategy].append(result['providers'][i]['total_user_utility'])
        provider_delegations[strategy].append(result['providers'][i]['delegations'])
        provider_delta[strategy].append(result['providers'][i]['delta'])

    provider_results[strategy]['avg_cost'] = np.mean(provider_cost[strategy])
    provider_results[strategy]['avg_price'] = np.mean(provider_price[strategy])
    provider_results[strategy]['avg_reward'] = np.mean(provider_reward[strategy])
    provider_results[strategy]['provider_utility'] = np.mean(provider_utility[strategy])
    provider_results[strategy]['user_utility'] = np.mean(user_utility[strategy])
    provider_results[strategy]['delegations'] = np.mean(provider_delegations[strategy])
    provider_results[strategy]['delta'] = np.mean(provider_delta[strategy])

    return provider_results

def get_result(i, result_path):
    CHOICES = [str(item) for item in list(range(6))]
    provider_cost = {k: [] for k in CHOICES}
    provider_reward = {k: [] for k in CHOICES}
    provider_price = {k: [] for k in CHOICES}
    provider_utility = {k: [] for k in CHOICES}
    user_utility = {k: [] for k in CHOICES}
    provider_results = {k: {} for k in CHOICES}
    provider_delegations = {k: [] for k in CHOICES}
    provider_delta = {k: [] for k in CHOICES}

    for strategy in CHOICES:
        others_sc = CHOICES
        if i == 1:
            all_scenarios = [f'0-{strategy}-{s}' for s in others_sc]
        elif i == 2:
            all_scenarios = [f'0-{s}-{strategy}' for s in others_sc]
        elif i == 0:
            others_sc = itertools.product(CHOICES, repeat=2)
            all_scenarios = [f'{strategy}-{s1}-{s2}' for s1, s2 in others_sc]
            # all_scenarios = [f'{0}-{0}-{0}' for s1, s2 in others_sc]
        results = [json.load(open(result_path / item / 'result.json')) for item in all_scenarios]

        for result in results:
            provider_cost[strategy].append(result['providers'][i]['total_costs'])
            provider_price[strategy].append(result['providers'][i]['total_reported_price'])
            provider_reward[strategy].append(result['providers'][i]['total_rewards'])
            provider_utility[strategy].append(result['providers'][i]['total_provider_utility'])
            user_utility[strategy].append(result['providers'][i]['total_user_utility'])
            provider_delegations[strategy].append(result['providers'][i]['delegations'])
            provider_delta[strategy].append(result['providers'][i]['delta'])

        provider_results[strategy]['avg_cost'] = np.mean(provider_cost[strategy])
        provider_results[strategy]['avg_price'] = np.mean(provider_price[strategy])
        provider_results[strategy]['avg_reward'] = np.mean(provider_reward[strategy])
        provider_results[strategy]['provider_utility'] = np.mean(provider_utility[strategy])
        provider_results[strategy]['user_utility'] = np.mean(user_utility[strategy])
        provider_results[strategy]['delegations'] = np.mean(provider_delegations[strategy])
        provider_results[strategy]['delta'] = np.mean(provider_delta[strategy])

    return provider_results