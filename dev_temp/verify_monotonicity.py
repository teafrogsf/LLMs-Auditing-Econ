import yaml
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dev_temp.secant_check import provider_models_with_metrics

def verify_monotonicity(default_config_path, model_config_path, p_ref_mode='max'):
    """验证配置中每个provider的模型是否满足效用随成本单调递增"""
    
    with open(default_config_path, 'r') as f:
        default_cfg = yaml.safe_load(f)
    
    with open(model_config_path, 'r') as f:
        model_cfg = yaml.safe_load(f)
    
    reward_param = float(default_cfg['reward_param'])
    
    print(f'配置文件: {default_config_path}')
    print(f'reward_param: {reward_param}')
    print(f'p_ref_mode: {p_ref_mode}')
    print('='*80)
    
    all_pass = True
    
    for provider in default_cfg.get('providers', []):
        provider_id, eta, mu_r_star, mu_l_star, items = provider_models_with_metrics(
            provider, reward_param, model_cfg
        )
        
        print(f'\nProvider {provider_id} (eta={eta}):')
        print(f'  mu_r*={mu_r_star:.6f}, mu_l*={mu_l_star:.6f}')
        print(f'  模型列表（按成本c排序）:')
        print(f'  {"模型名称":<25} {"h":>8} {"g":>8} {"p_out":>12} {"c=eta*p":>12} {"效用":>12}')
        print('  ' + '-'*90)
        
        # 计算每个模型的效用值
        for item in items:
            if p_ref_mode == 'max':
                p_ref = item['p']
            elif p_ref_mode == 'min':
                p_ref = item['p']
            else:  # mean
                p_ref = item['p']
            
            utility = item['h'] * mu_r_star - item['g'] * p_ref * mu_l_star
            item['utility'] = utility
            
            print(f'  {item["name"]:<25} {item["h"]:>8.6f} {item["g"]:>8.6f} {item["p"]:>12.2e} {item["c"]:>12.2e} {utility:>12.6f}')
        
        # 检查效用是否单调递增
        print(f'\n  检查效用单调性（效用应随成本c增加而增加）:')
        provider_pass = True
        for i in range(len(items) - 1):
            a = items[i]
            b = items[i + 1]
            
            if b['c'] == a['c']:
                continue
            
            utility_increase = b['utility'] - a['utility']
            c_increase = b['c'] - a['c']
            
            # 效用应该随成本增加而增加
            is_monotonic = utility_increase >= -1e-6  # 允许小的数值误差
            
            status = '✓' if is_monotonic else '✗'
            print(f'  {status} {a["name"]} -> {b["name"]}: Δc={c_increase:.2e}, Δutility={utility_increase:.6f}')
            
            if not is_monotonic:
                provider_pass = False
                all_pass = False
        
        if provider_pass:
            print(f'  ✓ Provider {provider_id} 满足单调性')
        else:
            print(f'  ✗ Provider {provider_id} 不满足单调性')
    
    print('\n' + '='*80)
    if all_pass:
        print('✓ 所有provider都满足效用随成本单调递增')
    else:
        print('✗ 存在provider不满足效用随成本单调递增')
    
    return all_pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--default-config', type=str, required=True)
    parser.add_argument('--model-config', type=str, default='config/nl_graph/model_config.yaml')
    parser.add_argument('--p-ref-mode', type=str, choices=['mean', 'min', 'max'], default='max')
    args = parser.parse_args()
    
    verify_monotonicity(args.default_config, args.model_config, args.p_ref_mode)


