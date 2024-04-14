from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from subprocess import run
from pathlib import Path


script_path = str(Path(__file__).parent / "exploration.py")
root_output_dir = Path("outputs/path_integration/random_exploration")
num_workers = 15


def run_command(seed, running_time, gait, trial_output_dir):
    thread_id = get_ident()
    command = [
        "python",
        script_path,
        "--seed",
        str(seed),
        "--running_time",
        str(running_time),
        "--gait",
        gait,
        "--output_dir",
        str(trial_output_dir),
    ]
    print(f"TID {thread_id} @ {datetime.now()} - Running: {' '.join(command)}")
    st = datetime.now()
    proc = run(command, check=True, capture_output=True)
    exit_code = proc.returncode
    with open(trial_output_dir / "stdout.txt", "wb") as f:
        f.write(proc.stdout)
    with open(trial_output_dir / "stderr.txt", "wb") as f:
        f.write(proc.stderr)
        walltime = datetime.now() - st
    print(
        f"TID {thread_id} @ {datetime.now()} - Done, exit code: {exit_code}. "
        f"Walltime: {walltime}"
    )


if __name__ == "__main__":
    gaits = ["tripod", "tetrapod", "wave"]
    n_trials = 15
    running_time = 20

    tasks = []
    for gait in gaits:
        for seed in range(n_trials):
            trial_output_dir = root_output_dir / f"seed={seed}_gait={gait}"
            tasks.append(
                (
                    seed,
                    running_time,
                    gait,
                    trial_output_dir,
                )
            )

    print(f"Running {len(tasks)} tasks with {num_workers} workers...")

    with ThreadPoolExecutor(num_workers) as executor:
        executor.map(lambda args: run_command(*args), tasks)
