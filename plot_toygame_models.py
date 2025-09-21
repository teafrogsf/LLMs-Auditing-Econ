import argparse
import os
import yaml
import matplotlib.pyplot as plt


def compute_expected_metrics(model_cfg):
    score_mu = float(model_cfg['score_mu'])
    reward_param = float(model_cfg['reward_param'])
    token_mu = float(model_cfg['token_mu'])
    token_price = float(model_cfg['token_price'])

    expected_price = token_price * token_mu
    expected_utility = score_mu * reward_param - expected_price
    return expected_price, expected_utility


def plot_models(config_path: str, output_path: str) -> None:
    config = yaml.safe_load(open(config_path))

    xs = []  # expected price
    ys = []  # expected utility
    labels = []

    for provider in config['providers']:
        provider_id = provider.get('id', 'NA')
        for model in provider['models']:
            model['reward_param'] = config['reward_param']
            x, y = compute_expected_metrics(model)
            xs.append(x)
            ys.append(y)
            labels.append(f"P{provider_id}-{model.get('name', 'model')}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, c='tab:blue', alpha=0.8)
    for i, label in enumerate(labels):
        plt.annotate(label, (xs[i], ys[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel('Expected Price')
    plt.ylabel('Expected Utility')
    plt.title('Toy Models: Expected Utility vs Expected Price')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config/toy_game/default.yaml')
    parser.add_argument('--output', '-o', type=str, default='outputs/toy_default/figs/model_utility_price_scatter.png')
    args = parser.parse_args()

    plot_models(args.config, args.output)


if __name__ == '__main__':
    main()


