"""CLI: train / evaluate the muscle imitation policy.

Examples:
    # Train with TensorBoard + Monitor logs + periodic checkpoints, then
    # record a video of the trained policy.
    python -m flygym_demo.muscle_imitation \
        --total-timesteps 200000 \
        --log-dir runs/0002 --video-path runs/0002/rollout.mp4

    # Just render a saved policy (no training).
    python -m flygym_demo.muscle_imitation --no-train \
        --model-path runs/0002/final_model.zip --video-path rollout.mp4

    # Smoke test with no ML dependencies: random-policy rollout (+ optional video).
    python -m flygym_demo.muscle_imitation --no-train --video-path random.mp4

Inspect training curves with:  tensorboard --logdir <log-dir>/tb

Training requires `stable-baselines3` (and `torch`); if they are missing the
script prints a note and falls back to a random-policy rollout so the example
still runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from flygym_demo.muscle_imitation.data import MoCapDataset
from flygym_demo.muscle_imitation.env import ImitationConfig
from flygym_demo.muscle_imitation.fly import make_imitation_env
from flygym_demo.muscle_imitation.record import (
    load_policy,
    random_policy,
    record_rollout,
    run_rollout,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", default="0002")
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument(
        "--log-dir",
        default=None,
        help="Directory for logs/checkpoints/final model. Default: runs/<clip>.",
    )
    p.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Env steps between checkpoints (0 disables).",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training (roll out --model-path, or a random policy).",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Saved PPO policy to roll out / record (defaults to the freshly "
        "trained model when training).",
    )
    p.add_argument(
        "--video-path",
        default=None,
        help="If set, record a deterministic rollout to this mp4.",
    )
    p.add_argument("--camera", default="scene", help="Scene camera name to render.")
    p.add_argument(
        "--camera-res",
        type=int,
        nargs=2,
        default=(480, 640),
        metavar=("H", "W"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log_dir = Path(args.log_dir) if args.log_dir else Path("runs") / args.clip
    model_path: str | None = args.model_path

    # --- training ---
    if not args.no_train:
        try:
            import stable_baselines3  # noqa: F401

            have_sb3 = True
        except ImportError:
            have_sb3 = False

        if not have_sb3:
            print(
                "stable_baselines3 not available; skipping training. "
                "Install with `pip install stable-baselines3`.",
                file=sys.stderr,
            )
        else:
            from flygym_demo.muscle_imitation.train import TrainConfig, train

            _, final_path = train(
                log_dir,
                config=TrainConfig(
                    clip=args.clip,
                    total_timesteps=args.total_timesteps,
                    learning_rate=args.learning_rate,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    n_epochs=args.n_epochs,
                    checkpoint_freq=args.checkpoint_freq,
                    seed=args.seed,
                ),
            )
            model_path = str(final_path)
            print(f"Saved final policy to {final_path}")
            print(f"Logs under {log_dir} (view: tensorboard --logdir {log_dir / 'tb'})")

    # --- evaluation / recording ---
    if args.video_path or args.no_train:
        env = make_imitation_env(
            config=ImitationConfig(clip=args.clip, test=bool(args.video_path)),
            dataset=MoCapDataset.default(),
        )
        if model_path is not None:
            policy = load_policy(model_path)
            label = f"policy {model_path}"
        else:
            policy = random_policy(env)
            label = "random policy"

        if args.video_path:
            stats = record_rollout(
                env,
                policy,
                args.video_path,
                camera=args.camera,
                camera_res=tuple(args.camera_res),
            )
            print(
                f"Recorded {label}: {stats['n_steps']} steps, "
                f"mean reward {stats['mean_reward']:.4f} -> {stats['video_path']}"
            )
        else:
            rewards = run_rollout(env, policy)
            print(
                f"Rollout ({label}): {len(rewards)} steps, "
                f"mean reward {np.mean(rewards):.4f}, max {np.max(rewards):.4f}"
            )
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
