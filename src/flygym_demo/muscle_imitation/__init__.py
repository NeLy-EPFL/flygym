"""Demo: muscle-driven imitation learning for the left-front leg.

When muscles are needed, the body model is *switched* to FlyMimic's
musculoskeletal MJCF (`flygym.compose.MusculoskeletalFly`) rather than
overlaying muscles on FlyGym's default rigid body. This demo wires that model
into an `ImitationEnv`.

* `make_imitation_env`: build the muscle Simulation + ImitationEnv in one call.
* `__main__`: train a PPO policy (stable-baselines3 optional; falls back to a
  random-policy rollout so the example always runs).

This submodule owns the imitation-learning stack end to end: the mocap clips
(under ``assets/mocap/``), the dataset loader, the reward/env, and the training
entry point. FlyGym's core only provides the musculoskeletal *body model*
(`flygym.compose.MusculoskeletalFly`); everything task-specific lives here.

Run as a script:

    python -m flygym_demo.muscle_imitation --total-timesteps 200000
"""

from flygym_demo.muscle_imitation.data import (
    DEFAULT_MOCAP_DIR,
    MoCapClip,
    MoCapDataset,
    TRACKED_BODY_NAMES,
    TRACKED_JOINT_NAMES,
    TRACKED_JOINT_NAMES_BY_NCOLS,
    tracked_joint_names_for_ncols,
)
from flygym_demo.muscle_imitation.env import ImitationConfig, ImitationEnv
from flygym_demo.muscle_imitation.fly import make_imitation_env

__all__ = [
    "make_imitation_env",
    "ImitationConfig",
    "ImitationEnv",
    "MoCapClip",
    "MoCapDataset",
    "DEFAULT_MOCAP_DIR",
    "TRACKED_JOINT_NAMES",
    "TRACKED_JOINT_NAMES_BY_NCOLS",
    "TRACKED_BODY_NAMES",
    "tracked_joint_names_for_ncols",
]
