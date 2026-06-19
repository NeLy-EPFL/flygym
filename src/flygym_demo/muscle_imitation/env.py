"""Gymnasium-compatible imitation-learning env wrapping `flygym.Simulation`.

All tracked quantities (joint qpos/qvel addresses, body ids, muscle actuator
ids) are resolved directly from the compiled MuJoCo model by name via
``mj_name2id``, so the env is decoupled from any particular fly wrapper. It
works with `flygym.compose.MusculoskeletalFly` out of the box and with any other
model that exposes the same MJCF element names.
"""

from dataclasses import dataclass, field
from typing import Any

import mujoco as mj
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover - clearer error than the import trace
    raise ImportError(
        "flygym_demo.muscle_imitation.env requires the optional dependency "
        "'gymnasium'. Install with `pip install gymnasium`."
    ) from e

from flygym.compose.fly import ActuatorType
from flygym.simulation import Simulation

from flygym_demo.muscle_imitation.data import (
    MoCapClip,
    MoCapDataset,
    TRACKED_BODY_NAMES,
    tracked_joint_names_for_ncols,
)


@dataclass
class ImitationConfig:
    """Hyperparameters for the imitation-tracking reward and episodes."""

    clip: str = "0002"
    """Mocap clip to track. Defaults to the bundled clip "0002"."""
    pose_rew_weight: float = 5.0
    vel_rew_weight: float = 3.0
    rew_threshold: float = 0.01
    """Episodes terminate if reward drops below this (training only)."""
    min_episode_steps: int = 20
    init_noise_scale: float = 0.02
    """Std dev of Gaussian noise applied to initial qpos in training mode."""
    control_timestep: float = 0.002
    """Seconds per env.step(). Must be a multiple of the MuJoCo dt."""
    test: bool = False
    """If True, disable early termination and noise; useful for evaluation."""
    tracked_joint_names: tuple[str, ...] | None = None
    """MJCF joint names matching the clip's qpos columns. If None (default),
    they are inferred from the clip's qpos width at env construction (the
    shipped clip has 7 DoFs)."""
    tracked_body_names: tuple[str, ...] = field(
        default_factory=lambda: TRACKED_BODY_NAMES
    )


class ImitationEnv(gym.Env):
    """Mocap-tracking RL env using a pre-built muscle `Simulation`.

    The action is a vector of muscle activations (one per muscle, in
    ``[0, 1]``). Observation defaults to tracked qpos + qvel + muscle
    activations + muscle forces + a time-left scalar. The reward is the
    FlyMimic compound tracking reward
    ``clip((qpos_rew + xpos_rew + qvel_rew) / 3, 0, 1)``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        simulation: Simulation,
        fly_name: str,
        dataset: MoCapDataset | None = None,
        config: ImitationConfig | None = None,
    ) -> None:
        super().__init__()
        self.sim = simulation
        self.fly_name = fly_name
        self.dataset = dataset or MoCapDataset.default()
        self.config = config or ImitationConfig()
        model = self.sim.mj_model

        # Determine which joints the clip's qpos columns correspond to. If the
        # user didn't pin them explicitly, infer from the configured clip's
        # qpos width (the shipped clip has 7 DoFs).
        if self.config.tracked_joint_names is not None:
            self.tracked_joint_names = tuple(self.config.tracked_joint_names)
        else:
            ncols = self.dataset.load(self.config.clip).qpos.shape[1]
            self.tracked_joint_names = tracked_joint_names_for_ncols(ncols)

        # --- resolve tracked joint qpos/qvel addresses by name ---
        qposadrs, qveladrs = [], []
        for jname in self.tracked_joint_names:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"Tracked joint '{jname}' not found in the model.")
            qposadrs.append(model.jnt_qposadr[jid])
            qveladrs.append(model.jnt_dofadr[jid])
        self._tracked_qposadrs = np.array(qposadrs, dtype=np.int32)
        self._tracked_qveladrs = np.array(qveladrs, dtype=np.int32)

        # --- resolve tracked body ids by name ---
        body_ids = []
        for bname in self.config.tracked_body_names:
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, bname)
            if bid < 0:
                raise ValueError(f"Tracked body '{bname}' not found in the model.")
            body_ids.append(bid)
        self._tracked_body_ids = np.array(body_ids, dtype=np.int32)

        # --- muscle actuator ids + activation-vector addresses ---
        fly = self.sim.world.fly_lookup[fly_name]
        self.muscle_names: list[str] = list(
            fly.jointdof_to_mjcfactuator_by_type[ActuatorType.MUSCLE].keys()
        )
        if not self.muscle_names:
            raise ValueError(
                f"Fly '{fly_name}' has no muscle actuators; build it from the "
                "musculoskeletal model (flygym.compose.MusculoskeletalFly) "
                "before making the env."
            )
        muscle_ids = []
        for mname in self.muscle_names:
            aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, mname)
            if aid < 0:
                raise ValueError(f"Muscle actuator '{mname}' not found in the model.")
            muscle_ids.append(aid)
        self._muscle_actuator_ids = np.array(muscle_ids, dtype=np.int32)
        # mj_data.act is indexed by actuator_actadr (per stateful actuator).
        self._muscle_actadrs = model.actuator_actadr[self._muscle_actuator_ids]
        self.n_muscles = len(self.muscle_names)

        # --- spaces ---
        self.action_space = spaces.Box(
            low=np.zeros(self.n_muscles, dtype=np.float32),
            high=np.ones(self.n_muscles, dtype=np.float32),
            dtype=np.float32,
        )
        obs_dim = 2 * len(self.tracked_joint_names) + 2 * self.n_muscles + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # --- substeps per env.step ---
        physics_dt = model.opt.timestep
        substeps = self.config.control_timestep / physics_dt
        if substeps < 1 or abs(substeps - round(substeps)) > 1e-6:
            raise ValueError(
                f"control_timestep {self.config.control_timestep}s is not a "
                f"positive integer multiple of physics dt {physics_dt}s."
            )
        self.substeps_per_action = int(round(substeps))

        self._clip: MoCapClip | None = None
        self._mocap_idx = 0
        self._last_reward = 0.0
        self._np_random: np.random.Generator | None = None

    # ---- gymnasium API ----

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self._np_random is None or seed is not None:
            self._np_random = np.random.default_rng(seed)

        clip_name = (options or {}).get("clip", self.config.clip)
        self._clip = self.dataset.load(clip_name)

        if self.config.test:
            self._mocap_idx = 0
        else:
            high = max(1, self._clip.n_frames - self.config.min_episode_steps)
            self._mocap_idx = int(self._np_random.integers(0, high))

        self.sim.reset()
        qpos0 = self._clip.qpos[self._mocap_idx].astype(np.float64)
        if not self.config.test and self.config.init_noise_scale > 0:
            qpos0 = qpos0 + self._np_random.normal(
                0, self.config.init_noise_scale, size=qpos0.shape
            )
        self.sim.mj_data.qpos[self._tracked_qposadrs] = qpos0
        self.sim.mj_data.qvel[self._tracked_qveladrs] = self._clip.qvel[self._mocap_idx]
        mj.mj_forward(self.sim.mj_model, self.sim.mj_data)

        self._last_reward = 0.0
        return self._get_observation(), {"clip": clip_name, "start_idx": self._mocap_idx}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.n_muscles,):
            raise ValueError(
                f"Expected action shape ({self.n_muscles},), got {action.shape}"
            )
        self.sim.set_actuator_inputs(self.fly_name, ActuatorType.MUSCLE, action)
        for _ in range(self.substeps_per_action):
            self.sim.step()

        self._mocap_idx += 1
        self._last_reward, terminated = self._compute_reward_and_done()

        obs = self._get_observation()
        info: dict[str, Any] = {"mocap_idx": self._mocap_idx, "reward": self._last_reward}
        return obs, self._last_reward, terminated, False, info

    # ---- reward + observation ----

    def _compute_reward_and_done(self) -> tuple[float, bool]:
        assert self._clip is not None
        clip = self._clip
        if self._mocap_idx >= clip.n_frames:
            return self._last_reward, True

        # Distance metrics mirror FlyMimic's mocap_tracking task exactly:
        # qpos/qvel use the L2 norm of the joint-error vector (FlyMimic writes
        # np.mean(np.linalg.norm(..., axis=-1)) on a 1-D array, which is just
        # the vector norm); xpos uses the mean over tracked bodies of the
        # per-body L2 distance.
        target_qpos = clip.qpos[self._mocap_idx]
        actual_qpos = self.sim.mj_data.qpos[self._tracked_qposadrs]
        qpos_dist = float(np.linalg.norm(target_qpos - actual_qpos))
        qpos_rew = float(np.exp(-self.config.pose_rew_weight * qpos_dist))

        target_qvel = clip.qvel[self._mocap_idx]
        actual_qvel = self.sim.mj_data.qvel[self._tracked_qveladrs]
        qvel_dist = float(np.linalg.norm(target_qvel - actual_qvel))
        qvel_rew = float(np.exp(-self.config.vel_rew_weight * qvel_dist))

        target_xpos = clip.xipos[self._mocap_idx]
        actual_xpos = self.sim.mj_data.xpos[self._tracked_body_ids]
        xpos_dist = float(np.mean(np.linalg.norm(target_xpos - actual_xpos, axis=-1)))
        xpos_rew = float(np.exp(-self.config.pose_rew_weight * xpos_dist))

        reward = float(np.clip((qpos_rew + xpos_rew + qvel_rew) / 3.0, 0.0, 1.0))

        terminated = False
        if not self.config.test and reward < self.config.rew_threshold:
            terminated = True
        if self._mocap_idx + 1 >= clip.n_frames:
            terminated = True
        return reward, terminated

    def _get_observation(self) -> np.ndarray:
        data = self.sim.mj_data
        qpos = data.qpos[self._tracked_qposadrs].astype(np.float32)
        qvel = data.qvel[self._tracked_qveladrs].astype(np.float32)
        if data.act.size > 0:
            muscle_acts = data.act[self._muscle_actadrs].astype(np.float32)
        else:
            muscle_acts = np.zeros(self.n_muscles, dtype=np.float32)
        muscle_forces = data.actuator_force[self._muscle_actuator_ids].astype(np.float32)
        if self._clip is not None and self._clip.n_frames > 1:
            time_left = np.float32(1.0 - self._mocap_idx / (self._clip.n_frames - 1))
        else:
            time_left = np.float32(0.0)
        return np.concatenate(
            [qpos, qvel, muscle_acts, muscle_forces, [time_left]]
        ).astype(np.float32)

    def close(self) -> None:
        self.sim.close()
