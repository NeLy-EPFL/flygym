from time import time
from tqdm import trange
from argparse import ArgumentParser

from flygym.envs.nmf_isaacgym import NeuroMechFlyIsaacGym
from flygym.util.data import load_kin_replay_data

import torch


if __name__ == "__main__":
    parser = ArgumentParser(prog="Isaac Gym kinematic replay")
    parser.add_argument(
        "-dt",
        "--timestep",
        type=float,
        default=1e-3,
        help="timestep for the physics simulation",
    )
    parser.add_argument(
        "-n",
        "--num_envs",
        type=int,
        default=100,
        help="number of environments to simulate in parallel",
    )
    parser.add_argument(
        "-r",
        "--render_mode",
        type=str,
        default="viewer",
        help="'viewer' or 'headless'",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=float,
        default=1.0,
        help="total duration to simulate (in seconds)",
    )
    args = parser.parse_args()

    kin_data = load_kin_replay_data(
        target_timestep=args.timestep, run_time=args.time
    )
    env = NeuroMechFlyIsaacGym(
        num_envs=args.num_envs,
        timestep=args.timestep,
        render_mode=args.render_mode,
    )
    env.reset()

    st = time()
    for i in trange(kin_data.shape[1]):
        curr_ref_state = torch.tensor(kin_data[:, i], device=env.compute_device)
        curr_states = curr_ref_state.unsqueeze(0).expand(args.num_envs, -1)
        env.step(curr_states)
    walltime = time() - st
    print(f'Walltime: {walltime:.4f} s')