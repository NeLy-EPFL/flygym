import numpy as np
import pkg_resources
import pickle
import pandas as pd
from multiprocessing import Pool
from time import time
from pathlib import Path
from tqdm import trange
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo
from flygym.util.config import leg_dofs_fused_tarsi

data_path = Path(pkg_resources.resource_filename('flygym', 'data'))
stats_path = data_path / "stats/mujoco_benchmark_stats.csv"
stats_path.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Initialize simulation
    run_time = 0.5
    num_envs_li = [1, 2, 4, 8, 16]
    walltimes = []

    # Load recorded data
    data_path = Path(pkg_resources.resource_filename("flygym", "data"))
    with open(data_path / "behavior" / "210902_pr_fly1.pkl", "rb") as f:
        data = pickle.load(f)

    # Interpolate 5x
    target_timestep = 1e-4
    num_steps = int(run_time / target_timestep)
    data_block = np.zeros((len(leg_dofs_fused_tarsi), num_steps))
    measure_t = np.arange(len(data["joint_LFCoxa"])) * data["meta"]["timestep"]
    interp_t = np.arange(num_steps) * target_timestep
    for i, joint in enumerate(leg_dofs_fused_tarsi):
        data_block[i, :] = np.interp(interp_t, measure_t, data[joint])


    def run_sim(args):
        nmf = NeuroMechFlyMuJoCo(
            render_mode="headless",
            timestep=1e-4,
            render_config={"playspeed": 0.1},
            init_pose="stretch",
            actuated_joints=leg_dofs_fused_tarsi,
        )
        
        st = time()
        obs_list = []
        for i in trange(num_steps):
            joint_pos = data_block[:, i]
            action = {"joints": joint_pos}
            obs, info = nmf.step(action)
            nmf.render()
            obs_list.append(obs)
        nmf.close()
        walltime = time() - st
        return walltime


    for num_envs in num_envs_li:
        print(f"Running {num_envs} environments...")
        with Pool(processes=num_envs) as pool:
            my_walltimes = pool.map(run_sim, [None] * num_envs)
        walltime = max(my_walltimes)
        print(f"  Walltime for: {walltime:.2f} s")
        walltimes.append(walltime)

    print(walltimes)
    stats = pd.DataFrame(data={"num_envs": num_envs_li, "walltime": walltimes})
    stats.to_csv(stats_path, index=False)