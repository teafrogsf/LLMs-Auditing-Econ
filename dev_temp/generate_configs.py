from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path


def read_template(template_path: Path) -> str:
    return template_path.read_text(encoding="utf-8")


def format_decimal_for_yaml(value: Decimal) -> str:
    # Normalize decimal to strip trailing zeros; ensure plain string without exponent
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def replace_values_preserving_format(template_text: str, epsilon: Decimal, reward_param: Decimal, gamma: int) -> str:
    epsilon_text = format_decimal_for_yaml(epsilon)
    reward_text = format_decimal_for_yaml(reward_param)
    gamma_text = str(int(gamma))

    # Replace entire lines to avoid backreference pitfalls and keep formatting simple
    new_text = re.sub(r"^epsilon:\s*.*$", f"epsilon: {epsilon_text}", template_text, flags=re.MULTILINE, count=1)
    new_text = re.sub(r"^reward_param:\s*.*$", f"reward_param: {reward_text}", new_text, flags=re.MULTILINE, count=1)
    new_text = re.sub(r"^gamma:\s*.*$", f"gamma: {gamma_text}", new_text, flags=re.MULTILINE, count=1)
    return new_text


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    template_path = workspace_root / "config" / "default.yaml"
    output_dir = workspace_root / "config"

    template_text = read_template(template_path)

    epsilons = [Decimal("0.1"), Decimal("0.2"), Decimal("0.3")]
    # reward_param: 3 to 10 inclusive, step 0.5
    reward_params = [Decimal("3") + Decimal("0.5") * Decimal(i) for i in range(int((10 - 3) / 0.5) + 1)]
    gammas = list(range(2, 21))

    count = 0
    for eps in epsilons:
        for r in reward_params:
            for g in gammas:
                filled_text = replace_values_preserving_format(template_text, eps, r, g)
                eps_str = format_decimal_for_yaml(eps)
                r_str = format_decimal_for_yaml(r)
                g_str = str(g)
                output_path = output_dir / f"epsilon{eps_str}_r{r_str}_gamma{g_str}.yaml"
                output_path.write_text(filled_text, encoding="utf-8")
                count += 1

    print(f"Generated {count} config files in {output_dir}")


if __name__ == "__main__":
    main()


