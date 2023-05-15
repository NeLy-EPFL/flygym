import pkg_resources
import pandas as pd
import subprocess
import re
from pathlib import Path


if __name__ == "__main__":
    data_path = Path(pkg_resources.resource_filename('flygym', 'data'))
    stats_path = data_path / "stats/ig_benchmark_stats.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    kin_replay_script = Path(__file__).parent / "ig_kin_replay.py"
    
    num_envs_li = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    walltimes = []
    for num_envs in num_envs_li:
        print(f"Simulating {num_envs} environments in parallel...")
        output = subprocess.check_output(
            [
                "python",
                "ig_kin_replay.py",
                "-n",
                str(num_envs),
                "-t",
                "0.5",
                "-r",
                "headless",
            ],
            text=True,
        )
        match = re.search(r"Walltime: (\d+\.\d+) s", output)
        if match:
            walltime = float(match.group(1))
            print(f"  Walltime: {walltime:.4f} s")
            walltimes.append(walltime)
        else:
            print("  No walltime printout detected.")
            walltimes.append(None)
    stats = pd.DataFrame(data={"num_envs": num_envs_li, "walltime": walltimes})
    stats.to_csv(stats_path, index=False)