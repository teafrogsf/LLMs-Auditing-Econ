from __future__ import annotations

import sys
import subprocess
from pathlib import Path


def find_generated_configs(config_dir: Path) -> list[Path]:
    return sorted(config_dir.glob("epsilon*_r*_gamma*.yaml"))


def run_command(command: list[str], stdout_path: Path | None = None, stderr_path: Path | None = None) -> int:
    stdout_handle = stdout_path.open("a", encoding="utf-8") if stdout_path else None
    stderr_handle = stderr_path.open("a", encoding="utf-8") if stderr_path else None
    try:
        result = subprocess.run(command, check=False, stdout=stdout_handle or sys.stdout, stderr=stderr_handle or sys.stderr)
        return result.returncode
    finally:
        if stdout_handle:
            stdout_handle.close()
        if stderr_handle:
            stderr_handle.close()


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    config_dir = workspace_root / "config"
    outputs_root = workspace_root / "outputs"

    configs = find_generated_configs(config_dir)
    if not configs:
        print(f"No generated configs found in {config_dir} matching 'epsilon*_r*_gamma*.yaml'.")
        sys.exit(1)

    print(f"Discovered {len(configs)} config files. Starting runs...")

    python_exe = sys.executable
    run_game_script = workspace_root / "run_game.py"
    plot_script = workspace_root / "plots" / "plot_main.py"

    failures: list[tuple[Path, str, int]] = []

    for idx, cfg_path in enumerate(configs, start=1):
        config_name = cfg_path.stem  # e.g., epsilon0.1_r3_gamma2
        run_output_dir = outputs_root / config_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(configs)}] Running for config: {cfg_path.name}")

        # 1) Run simulation
        sim_cmd = [
            python_exe,
            str(run_game_script),
            "--config",
            str(cfg_path),
            "--output-dir",
            str(run_output_dir),
        ]
        sim_rc = run_command(
            sim_cmd,
            stdout_path=run_output_dir / "run_game.stdout.log",
            stderr_path=run_output_dir / "run_game.stderr.log",
        )
        if sim_rc != 0:
            print(f"  Simulation failed (rc={sim_rc}) for {cfg_path.name}")
            failures.append((cfg_path, "run_game", sim_rc))
            # Continue to plotting attempt anyway

        # 2) Run plotting
        plot_cmd = [
            python_exe,
            str(plot_script),
            "--config",
            str(cfg_path),
        ]
        plot_rc = run_command(
            plot_cmd,
            stdout_path=run_output_dir / "plot.stdout.log",
            stderr_path=run_output_dir / "plot.stderr.log",
        )
        if plot_rc != 0:
            print(f"  Plotting failed (rc={plot_rc}) for {cfg_path.name}")
            failures.append((cfg_path, "plot_main", plot_rc))

    if failures:
        print("\nCompleted with failures:")
        for cfg_path, step, rc in failures:
            print(f"- {cfg_path.name}: {step} exited with {rc}")
        sys.exit(2)

    print("\nAll runs completed successfully.")


if __name__ == "__main__":
    main()



