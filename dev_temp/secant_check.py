import argparse
import math
import yaml
from typing import Dict, List, Tuple


def load_yaml(path: str) -> Dict:
    """
    中文：读取 YAML 配置文件并返回为字典。
    English: Load a YAML file and return its content as a dict.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def compute_global_L(default_cfg: Dict, model_cfg: Dict) -> int:
    """
    中文：计算全局 L（各模型 max_output_tokens 的最大值）。
    English: Compute global L as the maximum of max_output_tokens across all models.
    """
    L = 0
    for provider in default_cfg.get('providers', []):
        for name in provider.get('models', []):
            L = max(L, int(model_cfg[name]['max_output_tokens']))
    return L


def provider_models_with_metrics(provider: Dict, reward_param: float, model_cfg: Dict) -> Tuple[int, float, float, float, List[Dict]]:
    """
    中文：
    - 依据 provider 的模型列表，计算每个模型的均值参数与效用，并选出按效用排序的“基准模型”（utility 最大）。
    - 定义：mu_r* = reward_param * score_mu*，mu_l* = out_mu*。
    - 归一化：h_j = score_mu_j / score_mu*；g_j = out_mu_j / out_mu*。
    - 成本轴：c_j = eta * p_j，其中 p_j 为 output_token_price。

    English:
    - For a provider, compute per-model means and utility; pick the best (max utility) as the baseline.
    - Define mu_r* = reward_param * score_mu* and mu_l* = out_mu*.
    - Normalize: h_j = score_mu_j / score_mu*; g_j = out_mu_j / out_mu*.
    - Cost axis: c_j = eta * p_j, where p_j is the model's output_token_price.
    """
    eta = float(provider['eta'])
    names: List[str] = provider['models']
    items: List[Dict] = []
    # build items and pick best by utility_mu
    for name in names:
        cfg = model_cfg[name]
        score_mu = float(cfg['score_mu'])
        out_mu = float(cfg['output_tokens_mu'])
        p_out = float(cfg['output_token_price'])
        utility_mu = reward_param * score_mu - out_mu * p_out
        items.append({
            'name': name,
            'score_mu': score_mu,
            'out_mu': out_mu,
            'p_out': p_out,
            'utility_mu': utility_mu,
        })
    # best by utility
    best = max(items, key=lambda x: x['utility_mu'])
    mu_r_star = reward_param * best['score_mu']
    mu_l_star = best['out_mu']
    # normalize h, g; compute prices and internal cost c = eta * p
    for it in items:
        it['h'] = it['score_mu'] / best['score_mu']
        it['g'] = it['out_mu'] / best['out_mu']
        it['p'] = it['p_out']
        it['c'] = eta * it['p']
    # sort by c
    items.sort(key=lambda x: x['c'])
    return provider['id'], eta, mu_r_star, mu_l_star, items


def secant_check_for_provider(provider_id: int, eta: float, mu_l_star: float, L: int, items: List[Dict], p_ref_mode: str) -> List[Dict]:
    """
    不等式 1 的割线检验：
    - 检测 d/dc ln h(c) - d/dc ln g(c) >= mu_l*/(eta * p_ref * L)。
    - 离散区间 (a,b) 上，用割线近似：s_ab = [ln h_b - ln h_a - (ln g_b - ln g_a)] / (c_b - c_a)。
    - p_ref 取 (p_a+p_b)/2 或 min/max（由参数控制）。
    """
    results: List[Dict] = []
    if len(items) < 2:
        return results
    for i in range(len(items) - 1):
        a = items[i]
        b = items[i + 1]
        # y = ln h - ln g; secant slope s_ab = [y(b)-y(a)] / (c_b - c_a)
        y_a = math.log(max(a['h'], 1e-12)) - math.log(max(a['g'], 1e-12))
        y_b = math.log(max(b['h'], 1e-12)) - math.log(max(b['g'], 1e-12))
        dc = b['c'] - a['c']
        if dc == 0:
            # skip degenerate
            continue
        s_ab = (y_b - y_a) / dc
        if p_ref_mode == 'min':
            p_ref = min(a['p'], b['p'])
        elif p_ref_mode == 'max':
            p_ref = max(a['p'], b['p'])
        else:
            p_ref = 0.5 * (a['p'] + b['p'])
        T_ab = mu_l_star / (eta * p_ref * L)
        passed = s_ab >= T_ab
        results.append({
            'provider_id': provider_id,
            'pair': (a['name'], b['name']),
            'c_a': a['c'],
            'c_b': b['c'],
            's_ab': s_ab,
            'threshold_T_ab': T_ab,
            'p_ref': p_ref,
            'ineq1_passed': passed,
        })
    return results


def derivative_combo_check_for_provider(provider_id: int, eta: float, mu_r_star: float, mu_l_star: float, items: List[Dict], p_ref_mode: str) -> List[Dict]:
    """
    不等式 2 的割线检验：
    - 检测 (dh/dc)*mu_r* - (dg/dc)*p_ref*mu_l* >= 0。
    - 离散区间 (a,b) 上，dh/dc ≈ (h_b-h_a)/(c_b-c_a)，dg/dc 同理。
    """
    results: List[Dict] = []
    if len(items) < 2:
        return results
    for i in range(len(items) - 1):
        a = items[i]
        b = items[i + 1]
        dc = b['c'] - a['c']
        if dc == 0:
            continue
        dh_dc = (b['h'] - a['h']) / dc
        dg_dc = (b['g'] - a['g']) / dc
        if p_ref_mode == 'min':
            p_ref = min(a['p'], b['p'])
        elif p_ref_mode == 'max':
            p_ref = max(a['p'], b['p'])
        else:
            p_ref = 0.5 * (a['p'] + b['p'])
        combo_val = dh_dc * mu_r_star - dg_dc * p_ref * mu_l_star
        passed = combo_val >= 0.0
        results.append({
            'provider_id': provider_id,
            'pair': (a['name'], b['name']),
            'c_a': a['c'],
            'c_b': b['c'],
            'dh_dc': dh_dc,
            'dg_dc': dg_dc,
            'p_ref': p_ref,
            'combo_val': combo_val,
            'ineq2_passed': passed,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--default-config', type=str, default='config/nl_graph/default.yaml')
    parser.add_argument('--model-config', type=str, default='config/nl_graph/model_config.yaml')
    parser.add_argument('--p-ref-mode', type=str, choices=['mean', 'min', 'max'], default='mean',
                        help='阈值中的参考价格 p_ref 取法 (mean=(p_a+p_b)/2, min, max).')
    args = parser.parse_args()

    default_cfg = load_yaml(args.default_config)
    model_cfg = load_yaml(args.model_config)

    reward_param = float(default_cfg['reward_param'])
    L = compute_global_L(default_cfg, model_cfg)

    all_results: List[Dict] = []
    for provider in default_cfg.get('providers', []):
        provider_id, eta, mu_r_star, mu_l_star, items = provider_models_with_metrics(provider, reward_param, model_cfg)
        res1 = secant_check_for_provider(provider_id, eta, mu_l_star, L, items, args.p_ref_mode)
        res2 = derivative_combo_check_for_provider(provider_id, eta, mu_r_star, mu_l_star, items, args.p_ref_mode)
        # index res2 by pair for merging
        res2_map = {tuple(r['pair']): r for r in res2}
        for r1 in res1:
            pair = tuple(r1['pair'])
            r2 = res2_map.get(pair)
            combined = {
                'provider_id': provider_id,
                'pair': r1['pair'],
                'c_a': r1['c_a'],
                'c_b': r1['c_b'],
                's_ab': r1['s_ab'],
                'threshold_T_ab': r1['threshold_T_ab'],
                'p_ref': r1['p_ref'],
                'ineq1_passed': r1['ineq1_passed'],
                # provider-level params for clearer reporting
                'eta': eta,
                'mu_r_star': mu_r_star,
                'mu_l_star': mu_l_star,
            }
            if r2 is not None:
                combined.update({
                    'dh_dc': r2['dh_dc'],
                    'dg_dc': r2['dg_dc'],
                    'combo_val': r2['combo_val'],
                    'ineq2_passed': r2['ineq2_passed'],
                })
            all_results.append(combined)

    # pretty print (detailed, multi-line, English description)
    if not all_results:
        print('No intervals to check.')
        return
    print(f'L (global max_output_tokens) = {L}')
    for r in all_results:
        pid = r['provider_id']
        a, b = r['pair']
        c_a = r['c_a']
        c_b = r['c_b']
        s_ab = r['s_ab']
        T_ab = r['threshold_T_ab']
        p_ref = r.get('p_ref', float('nan'))
        flag1 = 'PASS' if r['ineq1_passed'] else 'FAIL'
        # provider-level params
        eta = r.get('eta')
        mu_r_star = r.get('mu_r_star')
        mu_l_star = r.get('mu_l_star')
        lines: List[str] = []
        lines.append(f'Provider {pid} | Interval {a} -> {b}')
        if eta is not None and mu_r_star is not None and mu_l_star is not None:
            lines.append(f'  provider params: eta={eta:.6g}, mu_r*={mu_r_star:.6g}, mu_l*={mu_l_star:.6g}')
        lines.append(f'  c interval: [{c_a:.6g}, {c_b:.6g}]  (c = eta * p)')
        lines.append(f'  p_ref: {p_ref:.6g}  (reference output-token price for thresholds)')
        lines.append('  Inequality 1: d ln h / dc - d ln g / dc >= mu_l* / (eta * p_ref * L)')
        lines.append(f'    secant slope s_ab = {s_ab:.6g}')
        lines.append(f'    threshold T_ab = {T_ab:.6g}')
        lines.append(f'    Result: {flag1}')
        if 'combo_val' in r:
            combo = r['combo_val']
            dh_dc = r['dh_dc']
            dg_dc = r['dg_dc']
            flag2 = 'PASS' if r.get('ineq2_passed', False) else 'FAIL'
            lines.append('  Inequality 2: (dh/dc) * mu_r* - (dg/dc) * p_ref * mu_l* >= 0')
            lines.append(f'    dh/dc ≈ {dh_dc:.6g}, dg/dc ≈ {dg_dc:.6g}')
            lines.append(f'    combo = {combo:.6g}')
            lines.append(f'    Result: {flag2}')
        print('\n'.join(lines))


if __name__ == '__main__':
    main() 