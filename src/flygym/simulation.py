from os import PathLike
from abc import ABC
from time import time_ns
from typing import Callable, TypeVar, Generic, TypedDict

import mujoco
import numpy as np
import yaml
from jaxtyping import Float
from numpy.typing import NDArray

from flygym.compose.fly.anatomy import JointDOF
from flygym.utils.viewer import FlyGymRenderer
from flygym.utils.profiling import print_perf_report

ObservationType = TypeVar("ObservationType")


class BaseSimulation(ABC, Generic[ObservationType]):
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        *args,
        **kwargs,
    ) -> None:
        self.mj_model: mujoco.MjModel = mj_model
        self.mj_data: mujoco.MjData = mj_data
        self.renderers: dict[str, FlyGymRenderer] = {}
        self.curr_step: int = 0

        # Profiling counters
        self._total_physics_ns: int = 0
        self._total_poststep_ns: int = 0
        self._total_prestep_ns: int = 0
        self._total_render_ns: int = 0
        self._frames_rendered: int = 0

    def _compute_control_input_pre_step(self) -> Float[NDArray, "nu"]:
        """Compute control inputs for the current simulation step.

        Returns:
            A numpy array of shape (nu,) containing the control inputs to
            be applied at the next physics step.
        """
        pass

    def _process_updated_state_post_step(self) -> ObservationType:
        pass

    def step(
        self, *control_input_compute_args, **control_input_compute_kwargs
    ) -> ObservationType:
        prestep_start = time_ns()
        control_input = self._compute_control_input_pre_step(
            *control_input_compute_args, **control_input_compute_kwargs
        )
        physics_start = time_ns()
        self.mj_data.ctrl[:] = control_input
        mujoco.mj_step(self.mj_model, self.mj_data)
        poststep_start = time_ns()
        observation = self._process_updated_state_post_step()

        render_start = time_ns()
        for _, renderer in self.renderers.items():
            renderer_output = renderer.render_as_needed(self.mj_data)
            if renderer_output is not None:
                self._frames_rendered += 1
        render_end = time_ns()

        self.curr_step += 1

        self._total_prestep_ns += physics_start - prestep_start
        self._total_physics_ns += poststep_start - physics_start
        self._total_poststep_ns += render_start - poststep_start
        self._total_render_ns += render_end - render_start

        return observation

    def add_renderer(
        self,
        name: str | None = None,
        *,
        height: int = 240,
        width: int = 320,
        play_speed: float = 0.2,
        out_fps: int = 25,
        camera: str | int | mujoco.MjvCamera = -1,
        frame_capture_callback: Callable[[np.ndarray], None] | None = None,
        buffer_frames: bool = True,
        **kwargs,
    ) -> None:
        renderer = FlyGymRenderer(
            self.mj_model,
            height=height,
            width=width,
            play_speed=play_speed,
            out_fps=out_fps,
            camera=camera,
            frame_capture_callback=frame_capture_callback,
            buffer_frames=buffer_frames,
            **kwargs,
        )
        name = f"renderer{len(self.renderers) + 1}" if name is None else name
        self.renderers[name] = renderer

    def reset(self) -> None:
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.curr_step = 0
        self._total_physics_ns = 0
        self._total_poststep_ns = 0
        self._total_prestep_ns = 0
        self._total_render_ns = 0
        self._frames_rendered = 0

        # Reset renderers' internal timers and clear buffered frames too
        for _, renderer in self.renderers.items():
            renderer.last_render_time_sec = -np.inf
            if renderer.frames is not None:
                renderer.frames.clear()

    def print_perf_report(self):
        print_perf_report(
            self._total_prestep_ns,
            self._total_physics_ns,
            self._total_poststep_ns,
            self._total_render_ns,
            self.curr_step,
            self._frames_rendered,
        )

    @property
    def time(self) -> float:
        return self.mj_data.time


class SingleFlyObservation(TypedDict):
    joint_angles: Float[NDArray, "n_flydofs_readout"]
    joint_velocities: Float[NDArray, "n_flydofs_readout"]


class SingleFlySimulation(BaseSimulation[SingleFlyObservation]):
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        joint_readout_order: list[JointDOF | str],
        joint_actuator_order: list[JointDOF | str],
    ) -> None:
        super().__init__(mj_model, mj_data)

        self.joint_readout_order = [
            j.name if isinstance(j, JointDOF) else j for j in joint_readout_order
        ]
        self.joint_actuator_order = [
            j.name if isinstance(j, JointDOF) else j for j in joint_actuator_order
        ]
        self._map_internal_and_exposed_ids()

    def _compute_control_input_pre_step(
        self, actuator_ctrl_input: Float[NDArray, "nu"]
    ) -> Float[NDArray, "nu"]:
        return actuator_ctrl_input[self._actuator_write_indexer]

    def _process_updated_state_post_step(self) -> SingleFlyObservation:
        return {
            "joint_angles": self.joint_angles,
            "joint_velocities": self.joint_velocities,
        }

    @property
    def joint_angles(self) -> Float[NDArray, "n_flydofs_readout"]:
        return self.mj_data.qpos[self._joint_qpos_read_indexer]

    @property
    def joint_velocities(self) -> Float[NDArray, "n_flydofs_readout"]:
        return self.mj_data.qvel[self._joint_qvel_read_indexer]

    def set_pose(
        self,
        *,
        fly_dof_angles_rad: dict[str, float] | None = None,
        fly_dof_angles_deg: dict[str, float] | None = None,
        pose_file: PathLike | None = None,
    ) -> None:
        if (
            int(fly_dof_angles_rad is None)
            + int(fly_dof_angles_deg is None)
            + int(pose_file is None)
            != 1
        ):
            raise ValueError(
                "Exactly one of fly_dof_angles_rad, fly_dof_angles_deg, or pose_file "
                "must be provided."
            )

        if pose_file:
            fly_dof_angles_rad = self._parse_pose_file(pose_file)
        if fly_dof_angles_deg:
            fly_dof_angles_rad = {
                k: np.deg2rad(v) for k, v in fly_dof_angles_deg.items()
            }

        internal_flydofname2qposadr_noprefix = self._strip_name_prefix(
            self._internal_flydofname2qposadr
        )
        for dof_name, angle_rad in fly_dof_angles_rad.items():
            qposadr = internal_flydofname2qposadr_noprefix.get(dof_name, None)
            if qposadr is None:
                raise ValueError(f"Unknown fly DoF name '{dof_name}'.")
            self.mj_data.qpos[qposadr] = angle_rad

    def _map_internal_and_exposed_ids(self):
        self._map_internal_joint_ids()
        self._map_internal_actuator_ids()
        self._map_internal_flydof2qposqvel_adrs()

        # When we return qpos to the user, how do we sort internal qpos for it to match
        # the fly DoF order specified upon init?
        # I.e., build an numpy fancy indexer (list of indices) so that we can do:
        #   retval_to_user = mj_data.qpos[indexer]
        internal_flydofname2qposadr_noprefix = self._strip_name_prefix(
            self._internal_flydofname2qposadr
        )
        self._joint_qpos_read_indexer = [
            internal_flydofname2qposadr_noprefix[joint]
            for joint in self.joint_readout_order
        ]

        # Do the same for qvel
        internal_flydofname2qveladr_noprefix = self._strip_name_prefix(
            self._internal_flydofname2qveladr
        )
        self._joint_qvel_read_indexer = [
            internal_flydofname2qveladr_noprefix[joint]
            for joint in self.joint_readout_order
        ]

        # When we write actuator commands from the user, how do we sort external
        # actuator commands to match internal actuator order? (This is the opposite.)
        # I.e., build an indexer so we can do:
        #   mj_data.ctrl[:] = user_specified_command_inputs[indexer]
        internal_id2actuatorname_noprefix = self._strip_name_prefix(
            self._internal_id2actuatorname
        )
        if not (
            set(self.joint_actuator_order)
            == set(internal_id2actuatorname_noprefix.values())
        ):
            print("joint_actuator_order:", self.joint_actuator_order)
            print("internal_id2actuatorname_noprefix:", set(internal_id2actuatorname_noprefix.values()))
            raise ValueError(
                "joint_actuator_order does not match the set of actuator names in the "
                "MuJoCo model."
            )
        self._actuator_write_indexer = [
            self.joint_actuator_order.index(internal_id2actuatorname_noprefix[aid])
            for aid in range(self.mj_model.nu)
        ]

    def _map_internal_joint_ids(self):
        self._internal_jointname2id = {
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid): jid
            for jid in range(self.mj_model.njnt)
        }
        self._internal_id2jointname = {
            v: k for k, v in self._internal_jointname2id.items()
        }

    def _map_internal_actuator_ids(self):
        self._internal_actuatorname2id = {
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid): aid
            for aid in range(self.mj_model.nu)
        }
        self._internal_id2actuatorname = {
            v: k for k, v in self._internal_actuatorname2id.items()
        }

    def _map_internal_flydof2qposqvel_adrs(self):
        """Note that mj_data.qpos and mj_data.qvel may have different sizes
        because rotational DoFs are represented as quaternions in qpos
        (4 values) but angular velocities in qvel (3 values)."""
        self._internal_flydofname2qposadr = {}
        self._internal_qposadr2flydofname = {}
        self._internal_flydofname2qveladr = {}
        self._internal_qveladr2flydofname = {}

        for name, jid in self._internal_jointname2id.items():
            if self._strip_name_prefix(name).startswith("joint-"):
                qposadr = self.mj_model.jnt_qposadr[jid]

                self._internal_flydofname2qposadr[name] = qposadr
                self._internal_qposadr2flydofname[qposadr] = name

                qveladr = self.mj_model.jnt_dofadr[jid]
                self._internal_flydofname2qveladr[name] = qveladr
                self._internal_qveladr2flydofname[qveladr] = name

    @staticmethod
    def _strip_name_prefix(x, /):
        if isinstance(x, str):
            return x.split("/")[-1]
        elif isinstance(x, list | tuple):
            return [name.split("/")[-1] for name in x]
        elif isinstance(x, dict):
            new_dict = {}
            for k, v in x.items():
                new_k = k.split("/")[-1] if isinstance(k, str) else k
                new_v = v.split("/")[-1] if isinstance(v, str) else v
                new_dict[new_k] = new_v
            return new_dict

    @staticmethod
    def _parse_pose_file(pose_file: PathLike) -> dict[str, float]:
        with open(pose_file, "r") as f:
            pose_data = yaml.safe_load(f)
        if "joint_angles" not in pose_data:
            raise ValueError(f"Pose file {pose_file} missing 'joint_angles' key.")
        if "angle_unit" not in pose_data:
            raise ValueError(f"Pose file {pose_file} missing 'angle_unit' key.")
        angle_unit = pose_data["angle_unit"].lower()
        if angle_unit in ("radians", "rads", "radian", "rad"):
            joint_angles_rad = pose_data["joint_angles"]
        elif angle_unit in ("degrees", "deg", "degree", "degs"):
            joint_angles_rad = {
                k: np.deg2rad(v) for k, v in pose_data["joint_angles"].items()
            }
        else:
            raise ValueError(f"Unknown angle unit '{angle_unit}' in pose file.")

        return joint_angles_rad
