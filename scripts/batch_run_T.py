import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

def run_single_task(T: int) -> None:
    output_dir = f"./outputs/nl_graph/aba_T_yuhan1_larger/T{T}"
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "python",
        "run_game.py",
        "-T",
        str(T),
        "--output-dir",
        output_dir,
        "--config",
        "config/nl_graph/yuhan1.yaml"
    ]
    subprocess.run(cmd, check=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple run_game tasks in parallel.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Maximum number of parallel workers (default: number of CPUs).",
    )
    args = parser.parse_args()

    # Create exactly 100 tasks: 1,000,000 to 10,900,000 inclusive, step 100,000
    t_values = list(range(int(3e6), int(1e7), int(1e6)))
    # print(len(t_values))
    # exit()
    # t_values.reverse()

    print(f"Launching {len(t_values)} tasks with max_workers={args.max_workers}...")
    failures = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_t = {executor.submit(run_single_task, t): t for t in t_values}
        for future in as_completed(future_to_t):
            t = future_to_t[future]
            try:
                future.result()
                print(f"[OK] T={t}")
            except subprocess.CalledProcessError as e:
                print(f"[FAIL] T={t} exited with code {e.returncode}")
                failures.append((t, e.returncode))
            except Exception as e:
                print(f"[ERROR] T={t} error: {e}")
                failures.append((t, None))

    if failures:
        print(f"Completed with {len(failures)} failures: {failures}")
    else:
        print("All tasks completed successfully.")


if __name__ == "__main__":
    main()