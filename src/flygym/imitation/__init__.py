"""Imitation-learning environment + reward built on FlyGym's simulation API.

Public objects:

* `MoCapDataset`: lazy loader for the bundled FlyMimic mocap clips (qpos,
  qvel, xipos, xivel).
* `ImitationConfig`: hyperparameters (reward weights, init noise, tracked
  joint/body names, etc.).
* `ImitationEnv`: gymnasium-compatible environment that drives a
  `flygym.Simulation` built from the musculoskeletal model, applies muscle
  activations as actions, tracks a mocap clip, and emits the FlyMimic reward.

Anything FlyGym already exposes via the `Simulation` API (contact sensors,
body kinematics, proprioception, optional vision) remains available; the env
just adds task-specific reward bookkeeping.
"""

from flygym.imitation.data import (
    DEFAULT_MOCAP_DIR,
    MoCapDataset,
    TRACKED_BODY_NAMES,
    TRACKED_JOINT_NAMES,
    TRACKED_JOINT_NAMES_BY_NCOLS,
    tracked_joint_names_for_ncols,
)
from flygym.imitation.env import ImitationConfig, ImitationEnv

__all__ = [
    "MoCapDataset",
    "DEFAULT_MOCAP_DIR",
    "TRACKED_JOINT_NAMES",
    "TRACKED_JOINT_NAMES_BY_NCOLS",
    "TRACKED_BODY_NAMES",
    "tracked_joint_names_for_ncols",
    "ImitationConfig",
    "ImitationEnv",
]
