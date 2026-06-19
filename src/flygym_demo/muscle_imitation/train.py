"""PPO training for the muscle imitation env, with logging + checkpointing.

`train()` builds the imitation env, wraps it in a `Monitor` (per-episode reward
and length to a CSV), trains a PPO policy with TensorBoard logging and periodic
checkpoints, and saves the final policy. All artifacts are written under a
single ``log_dir``:

    <log_dir>/
        monitor.csv              # per-episode reward/length (read with pandas)
        tb/                       # TensorBoard event files (rollout/ep_rew_mean, ...)
        checkpoints/ppo_muscle_*_steps.zip
        final_model.zip           # the policy at the end of training

`stable-baselines3` (and `torch`) are required and imported lazily, so this
module is safe to import without them — the ImportError only surfaces when
`train()` is actually called.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from flygym_demo.muscle_imitation.data import MoCapDataset
from flygym_demo.muscle_imitation.env import ImitationConfig
from flygym_demo.muscle_imitation.fly import make_imitation_env


@dataclass
class TrainConfig:
    """Hyperparameters for a PPO training run (FlyMimic's defaults)."""

    clip: str = "0002"
    total_timesteps: int = 200_000
    learning_rate: float = 1e-5
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    net_arch: tuple[int, ...] = (512, 512, 256)
    checkpoint_freq: int = 50_000
    """Env steps between checkpoints. Set to 0 to disable checkpointing."""
    seed: int | None = None


def train(
    log_dir: PathLike,
    *,
    config: TrainConfig | None = None,
    dataset: MoCapDataset | None = None,
):
    """Train a PPO policy on the muscle imitation env.

    Args:
        log_dir: Directory for all training artifacts (created if missing).
        config: `TrainConfig` hyperparameters; defaults to FlyMimic's.
        dataset: Mocap dataset; defaults to the bundled clips.

    Returns:
        ``(model, final_model_path)`` — the trained ``PPO`` model and the
        `Path` it was saved to (``<log_dir>/final_model.zip``).
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    config = config or TrainConfig()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = make_imitation_env(
        config=ImitationConfig(clip=config.clip),
        dataset=dataset or MoCapDataset.default(),
    )
    env = Monitor(env, str(log_dir / "monitor.csv"))

    # TensorBoard logging is optional: SB3 raises if `tensorboard` is missing,
    # so only enable it when importable. The Monitor CSV is always written.
    try:
        import tensorboard  # noqa: F401

        tensorboard_log = str(log_dir / "tb")
    except ImportError:
        tensorboard_log = None
        print(
            "tensorboard not installed; skipping TensorBoard logs "
            "(per-episode reward/length still written to monitor.csv). "
            "Install with `pip install tensorboard`."
        )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        policy_kwargs={"net_arch": list(config.net_arch)},
        tensorboard_log=tensorboard_log,
        seed=config.seed,
        verbose=1,
    )

    callbacks = []
    if config.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=config.checkpoint_freq,
                save_path=str(log_dir / "checkpoints"),
                name_prefix="ppo_muscle",
            )
        )

    model.learn(total_timesteps=config.total_timesteps, callback=callbacks or None)

    final_path = log_dir / "final_model.zip"
    model.save(str(final_path))
    return model, final_path
