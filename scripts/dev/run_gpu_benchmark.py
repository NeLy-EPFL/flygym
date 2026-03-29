from subprocess import run
from pathlib import Path

import pandas as pd

from flygym.warp.utils import check_gpu
from flygym_demo.benchmark import run_benchmark


if __name__ == "__main__":
    SIM_TIMESTEP = 1e-4
    OUTPUT_DIR = Path("gpu_benchmark_results/")

    check_gpu()

    dfs = []
    for simplify_geom in [False, True]:
        df = run_benchmark(
            enable_rendering=False,
            min_worlds=2**4,  # 16
            max_worlds=2**6,  # 16384
            factor=2,
            sim_timestep=SIM_TIMESTEP,
            sim_steps=1000,
            simplify_geom=simplify_geom,
        )
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / "gpu_simulation_benchmark_results.csv", index=False)
    with open(OUTPUT_DIR / "gpu_info.csv", "w") as f:
        run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], stdout=f)
