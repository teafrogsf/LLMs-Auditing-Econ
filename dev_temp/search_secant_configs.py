import argparse
import os
import sys
import random
import glob
from typing import Dict, List, Tuple, Optional

# ensure project root on sys.path for absolute import 'dev_temp.*'
_CURRENT_DIR = os.path.dirname(__file__)
_PARENT_DIR = os.path.dirname(_CURRENT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import yaml
from dev_temp.secant_check import (
    load_yaml as sec_load_yaml,
    compute_global_L as sec_compute_global_L,
    provider_models_with_metrics as sec_provider_models_with_metrics,
    secant_check_for_provider as sec_secant_check_for_provider,
    derivative_combo_check_for_provider as sec_derivative_combo_check_for_provider,
)


def load_yaml(path: str) -> Dict:
    return sec_load_yaml(path)


def compute_global_L_from_selection(providers: List[Dict], model_cfg: Dict) -> int:
    default_cfg = {'providers': providers}
    return sec_compute_global_L(default_cfg, model_cfg)


def provider_models_with_metrics(provider: Dict, reward_param: float, model_cfg: Dict) -> Tuple[int, float, float, float, List[Dict]]:
    return sec_provider_models_with_metrics(provider, reward_param, model_cfg)


def secant_check_for_provider(provider_id: int, eta: float, mu_l_star: float, L: int, items: List[Dict], p_ref_mode: str) -> List[Dict]:
    return sec_secant_check_for_provider(provider_id, eta, mu_l_star, L, items, p_ref_mode)


def derivative_combo_check_for_provider(provider_id: int, eta: float, mu_r_star: float, mu_l_star: float, items: List[Dict], p_ref_mode: str) -> List[Dict]:
    return sec_derivative_combo_check_for_provider(provider_id, eta, mu_r_star, mu_l_star, items, p_ref_mode)


def check_provider_best_models_decreasing(providers: List[Dict], model_cfg: Dict, reward_param: float) -> bool:
    """
    检查各provider的最佳模型是否按递减顺序排列。
    provider 1的最佳模型 > provider 2的最佳模型 > provider 3的最佳模型
    (按 utility_mu 比较)
    """
    if len(providers) <= 1:
        return True
    
    best_utilities: List[Tuple[int, float]] = []
    for p in providers:
        _, _, _, _, items = provider_models_with_metrics(p, reward_param, model_cfg)
        # items中已经包含每个模型的utility_mu，找出最大值
        best_utility = max(item['utility_mu'] for item in items)
        best_utilities.append((p['id'], best_utility))
    
    # 检查是否严格递减
    for i in range(len(best_utilities) - 1):
        if best_utilities[i][1] <= best_utilities[i + 1][1]:
            return False
    return True


def providers_pass_secant(providers: List[Dict], model_cfg: Dict, reward_param: float, p_ref_mode: str) -> bool:
    L = compute_global_L_from_selection(providers, model_cfg)
    for p in providers:
        _, eta, mu_r_star, mu_l_star, items = provider_models_with_metrics(p, reward_param, model_cfg)
        res1 = secant_check_for_provider(p['id'], eta, mu_l_star, L, items, p_ref_mode)
        res2 = derivative_combo_check_for_provider(p['id'], eta, mu_r_star, mu_l_star, items, p_ref_mode)
        if not res1 or not res2:
            return False
        if not all(r['ineq1_passed'] for r in res1):
            return False
        if not all(r['ineq2_passed'] for r in res2):
            return False
    return True


def all_model_names(model_cfg: Dict, exclude_models: Optional[List[str]] = None) -> List[str]:
    """
    获取所有模型名称，可选择排除某些模型
    """
    all_names = list(model_cfg.keys())
    if exclude_models:
        all_names = [name for name in all_names if name not in exclude_models]
    return all_names


def generate_provider_blocks(model_names: List[str], num_providers: int, models_per_provider: int, eta: float, allow_overlap: bool, seed: int, num_candidates: int) -> List[List[Dict]]:
    rng = random.Random(seed)
    providers_list: List[List[Dict]] = []
    # basic strategy: sample many candidate provider blocks
    pool = model_names[:]
    for _ in range(num_candidates):
        chosen: List[Dict] = []
        used = set()
        for pid in range(num_providers):
            if allow_overlap:
                models = rng.sample(pool, k=min(models_per_provider, len(pool)))
            else:
                available = [m for m in pool if m not in used]
                if len(available) < models_per_provider:
                    break
                models = rng.sample(available, k=models_per_provider)
                used.update(models)
            chosen.append({'id': pid, 'num_models': models_per_provider, 'eta': eta, 'models': models})
        if len(chosen) == num_providers:
            providers_list.append(chosen)
    return providers_list


def search_configs(model_cfg: Dict,
                   reward_param: float,
                   p_ref_mode: str,
                   num_providers: int,
                   models_per_provider: int,
                   eta: float,
                   allow_overlap: bool,
                   seed: int,
                   max_results: int,
                   base_default_path: Optional[str],
                   num_tasks_override: Optional[int],
                   epsilon_override: Optional[float],
                   gamma_override: Optional[float],
                   exclude_models: Optional[List[str]] = None,
                   num_candidates: int = 10000) -> Tuple[List[Dict], int]:
    """
    返回 (results, total_candidates) 
    """
    names = all_model_names(model_cfg, exclude_models)
    candidates = generate_provider_blocks(names, num_providers, models_per_provider, eta, allow_overlap, seed, num_candidates)
    results: List[Dict] = []
    base_cfg: Dict = {}
    if base_default_path:
        try:
            base_cfg = load_yaml(base_default_path) or {}
        except Exception:
            base_cfg = {}
    for providers in candidates:
        # 首先检查provider最佳模型是否递减
        if not check_provider_best_models_decreasing(providers, model_cfg, reward_param):
            continue
        # 然后检查secant约束
        if providers_pass_secant(providers, model_cfg, reward_param, p_ref_mode):
            # merge base cfg and overrides; providers and reward_param come from search result
            merged = dict(base_cfg)
            merged['providers'] = providers
            merged['reward_param'] = reward_param
            if num_tasks_override is not None:
                merged['num_tasks'] = num_tasks_override
            else:
                merged.setdefault('num_tasks', 1000000)
            if epsilon_override is not None:
                merged['epsilon'] = epsilon_override
            else:
                merged.setdefault('epsilon', 0.3)
            if gamma_override is not None:
                merged['gamma'] = gamma_override
            else:
                merged.setdefault('gamma', 5)
            results.append(merged)
            if len(results) >= max_results:
                break
    return results, len(candidates)


def get_next_available_index(out_dir: str, config_name: str) -> int:
    """
    查找输出目录中已存在的 {config_name}_*.yaml 文件，
    返回下一个可用的索引号（避免覆盖）
    """
    if not os.path.exists(out_dir):
        return 0
    
    pattern = os.path.join(out_dir, f'{config_name}_*.yaml')
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return 0
    
    # 提取所有现有的索引号
    indices = []
    for filepath in existing_files:
        filename = os.path.basename(filepath)
        # {config_name}_XXX.yaml
        try:
            idx_str = filename.replace(f'{config_name}_', '').replace('.yaml', '')
            indices.append(int(idx_str))
        except ValueError:
            continue
    
    if not indices:
        return 0
    
    return max(indices) + 1


def save_default_yaml(cfg: Dict, out_dir: str, index: int, config_name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{config_name}_{index:03d}.yaml')
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path


def main():
    parser = argparse.ArgumentParser(description='搜索可通过 secant 检验的 default.yaml 组装配置')
    parser.add_argument('--model-config', type=str, default='config/nl_graph/model_config.yaml', help='模型配置文件路径')
    parser.add_argument('--reward-param', type=float, default=6.0, help='奖励参数')
    parser.add_argument('--p-ref-mode', type=str, choices=['mean', 'min', 'max'], default='mean', help='参考价格模式')
    parser.add_argument('--num-providers', type=int, default=3, help='provider数量')
    parser.add_argument('--models-per-provider', type=int, default=3, help='每个provider的模型数量')
    parser.add_argument('--eta', type=float, default=0.9, help='eta参数')
    parser.add_argument('--allow-overlap', action='store_true', help='是否允许不同provider使用相同模型')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--max-results', type=int, default=5, help='最多返回多少个配置')
    parser.add_argument('--out-dir', type=str, default='dev_temp/secant_configs', help='输出目录')
    parser.add_argument('--base-default', type=str, default='config/nl_graph/default.yaml', help='作为模板的 default.yaml')
    parser.add_argument('--num-tasks', type=int, default=None, help='覆盖任务数量')
    parser.add_argument('--epsilon', type=float, default=None, help='覆盖epsilon参数')
    parser.add_argument('--gamma', type=float, default=None, help='覆盖gamma参数')
    parser.add_argument('--config-name', type=str, default='default_secant', help='配置文件名前缀（例如：my_config 会生成 my_config_000.yaml）')
    parser.add_argument('--exclude-models', type=str, nargs='*', default=[], help='要排除的模型名称列表（例如：--exclude-models gpt-4 o1 claude-4-0）')
    parser.add_argument('--num-candidates', type=int, default=10000, help='生成候选配置的数量（默认10000，增加此值可能找到更多配置但会更慢）')
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)

    # 如果指定了排除模型，显示信息
    if args.exclude_models:
        print(f'排除的模型: {", ".join(args.exclude_models)}')
        available_models = all_model_names(model_cfg, args.exclude_models)
        print(f'可用模型数量: {len(available_models)} (总共 {len(model_cfg)} 个模型)')
    
    # 显示候选配置生成信息
    print(f'正在生成 {args.num_candidates} 个候选配置进行筛选...')

    found, total_candidates = search_configs(
        model_cfg=model_cfg,
        reward_param=args.reward_param,
        p_ref_mode=args.p_ref_mode,
        num_providers=args.num_providers,
        models_per_provider=args.models_per_provider,
        eta=args.eta,
        allow_overlap=args.allow_overlap,
        seed=args.seed,
        max_results=args.max_results,
        base_default_path=args.base_default,
        num_tasks_override=args.num_tasks,
        epsilon_override=args.epsilon,
        gamma_override=args.gamma,
        exclude_models=args.exclude_models,
        num_candidates=args.num_candidates,
    )

    if not found:
        print('未找到通过 secant 检验的配置。可尝试调大随机性（允许重叠/改变种子）、或调整参数。')
        print(f'已生成 {total_candidates} 个候选配置进行测试。')
        return

    # 获取下一个可用的起始索引（避免覆盖已存在的文件）
    start_index = get_next_available_index(args.out_dir, args.config_name)
    
    print(f'成功找到 {len(found)} 个通过检验的配置（从 {total_candidates} 个候选中筛选）：')
    for i, cfg in enumerate(found):
        path = save_default_yaml(cfg, args.out_dir, start_index + i, args.config_name)
        print(f'  [{start_index + i:03d}] SAVED: {path}')
        # 打印每个provider的最佳模型信息
        for p in cfg['providers']:
            _, _, _, _, items = provider_models_with_metrics(p, cfg['reward_param'], model_cfg)
            best_item = max(items, key=lambda x: x['utility_mu'])
            print(f'      Provider {p["id"]}: 最佳模型 = {best_item["name"]}, utility = {best_item["utility_mu"]:.6f}')


if __name__ == '__main__':
    main() 