"""CLI entry: train a PPO policy on the muscle imitation env.

Usage:
    python -m flygym_demo.muscle_imitation --help
    python -m flygym_demo.muscle_imitation --total-timesteps 200000

Requires `stable-baselines3` (and `torch`); not installed by FlyGym by
default. Falls back to a random-policy rollout if PPO can't be imported, so
you can at least confirm the env works.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from flygym_demo.muscle_imitation.data import MoCapDataset
from flygym_demo.muscle_imitation.env import ImitationConfig
from flygym_demo.muscle_imitation.fly import make_imitation_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", default="0002")
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training; just roll out a random policy as a smoke test.",
    )
    return p.parse_args()


def random_rollout(env, n_steps: int = 200) -> None:
    rng = np.random.default_rng(0)
    obs, _ = env.reset()
    rewards = []
    for _ in range(n_steps):
        action = rng.uniform(0.05, 0.6, size=env.n_muscles).astype(np.float32)
        obs, rew, term, _, _ = env.step(action)
        rewards.append(rew)
        if term:
            obs, _ = env.reset()
    print(
        f"Random rollout: {len(rewards)} steps, "
        f"mean reward {np.mean(rewards):.4f}, max {np.max(rewards):.4f}"
    )


def main() -> int:
    args = parse_args()
    config = ImitationConfig(clip=args.clip)
    env = make_imitation_env(config=config, dataset=MoCapDataset.default())

    if args.no_train:
        random_rollout(env)
        return 0

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print(
            "stable_baselines3 not available; running a random-policy rollout. "
            "Install with `pip install stable-baselines3`.",
            file=sys.stderr,
        )
        random_rollout(env)
        return 0

    env = Monitor(env)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        policy_kwargs={"net_arch": [512, 512, 256]},
        verbose=1,
    )
    model.learn(total_timesteps=args.total_timesteps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
