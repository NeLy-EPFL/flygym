from collections import defaultdict
from typing import Any

import mujoco
import dm_control.mjcf as mjcf
import numpy as np
from jaxtyping import Float
from time import perf_counter_ns
from collections.abc import Sequence

from flygym.compose.fly import ActuatorType
from flygym.compose.world import BaseWorld
from flygym.rendering import Renderer, CameraSpec
from flygym.utils.profiling import print_perf_report


class Simulation:
    def __init__(self, world: BaseWorld) -> None:
        if len(world.fly_lookup) == 0:
            raise ValueError("The world must contain at least one fly.")
        self.renderer: Renderer | None = None
        self.world = world
        self.mj_model, self.mj_data = world.compile()

        # Map internal IDs in the compiled MuJoCo model. This allows users to read from
        # or write to body/joint/actuator in orders defined by Fly objects.
        self._map_internal_bodyids()
        self._map_internal_qposqveladrs()
        self._map_internal_actuator_ids()

        # For performance profiling
        self._curr_step = 0
        self._frames_rendered = 0
        self._total_physics_time_ns = 0
        self._total_render_time_ns = 0

        # Reset everything (physics, renderers, and profiling stats)
        self.reset()

    def reset(self) -> None:
        # Reset physics
        neutral_keyframe_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "neutral"
        )
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, neutral_keyframe_id)

        # Reset renderers
        if self.renderer is not None:
            self.renderer.reset()

        # Stuff for performance profiling
        self._curr_step = 0
        self._frames_rendered = 0
        self._total_physics_time_ns = 0
        self._total_render_time_ns = 0

    def step(self) -> None:
        physics_start_ns = perf_counter_ns()
        mujoco.mj_step(self.mj_model, self.mj_data)
        physics_finish_ns = perf_counter_ns()
        self._total_physics_time_ns += physics_finish_ns - physics_start_ns
        self._curr_step += 1

    def set_renderer(
        self,
        camera: CameraSpec | Sequence[CameraSpec],
        *,
        camera_res: tuple[int, int] = (240, 320),
        playback_speed: float = 0.2,
        output_fps: int = 25,
        **kwargs: Any,
    ) -> Renderer:
        self.renderer = Renderer(
            self.mj_model,
            camera,
            camera_res=camera_res,
            playback_speed=playback_speed,
            output_fps=output_fps,
            **kwargs,
        )
        return self.renderer

    def render_as_needed(self) -> bool:
        render_start_ns = perf_counter_ns()
        render_done = self.renderer.render_as_needed(self.mj_data)
        render_finish_ns = perf_counter_ns()
        self._total_render_time_ns += render_finish_ns - render_start_ns
        if render_done:
            self._frames_rendered += 1
        return render_done

    def get_joint_angles(self, fly_name: str) -> Float[np.ndarray, "n_jointdofs"]:
        internal_ids = self._intern_qposadrs_by_fly[fly_name]
        return self.mj_data.qpos[internal_ids]

    def get_joint_velocities(self, fly_name: str) -> Float[np.ndarray, "n_jointdofs"]:
        internal_ids = self._intern_qveladrs_by_fly[fly_name]
        return self.mj_data.qvel[internal_ids]

    def get_body_positions(self, fly_name: str) -> Float[np.ndarray, "n_bodies 3"]:
        internal_ids = self._internal_bodyids_by_fly[fly_name]
        return self.mj_data.xpos[internal_ids, :]

    def get_body_rotations(self, fly_name: str) -> Float[np.ndarray, "n_bodies 4"]:
        internal_ids = self._internal_bodyids_by_fly[fly_name]
        return self.mj_data.xquat[internal_ids, :]

    def set_actuator_inputs(
        self,
        fly_name: str,
        actuator_type: ActuatorType,
        inputs: Float[np.ndarray, "n_actuators"],
    ):
        internal_ids = self._intern_actuatorids_by_type_by_fly[actuator_type][fly_name]
        if len(inputs) != len(internal_ids):
            raise ValueError(
                f"Expected {len(internal_ids)} inputs for actuator type "
                f"'{actuator_type.name}', but got {len(inputs)}"
            )
        self.mj_data.ctrl[internal_ids] = inputs

    def _map_internal_bodyids(self) -> None:
        internal_bodyids_by_fly = defaultdict(list)

        for fly_name, fly in self.world.fly_lookup.items():
            for bodyseg, mjcf_body_element in fly.bodyseg_to_mjcfbody.items():
                internal_body_id = mujoco.mj_name2id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    mjcf_body_element.full_identifier,
                )
                internal_bodyids_by_fly[fly_name].append(internal_body_id)

        self._internal_bodyids_by_fly = {
            k: np.array(v, dtype=np.int32) for k, v in internal_bodyids_by_fly.items()
        }

    def _map_internal_qposqveladrs(self) -> None:
        internal_qposadrs_by_fly = defaultdict(list)
        internal_qveladrs_by_fly = defaultdict(list)

        for fly_name, fly in self.world.fly_lookup.items():
            for jointdof, mjcf_joint_element in fly.jointdof_to_mjcfjoint.items():
                internal_joint_id = mujoco.mj_name2id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    mjcf_joint_element.full_identifier,
                )
                qposadr = self.mj_model.jnt_qposadr[internal_joint_id]
                qveladr = self.mj_model.jnt_dofadr[internal_joint_id]
                internal_qposadrs_by_fly[fly_name].append(qposadr)
                internal_qveladrs_by_fly[fly_name].append(qveladr)

        self._intern_qposadrs_by_fly = {
            k: np.array(v, dtype=np.int32) for k, v in internal_qposadrs_by_fly.items()
        }
        self._intern_qveladrs_by_fly = {
            k: np.array(v, dtype=np.int32) for k, v in internal_qveladrs_by_fly.items()
        }

    def _map_internal_actuator_ids(self) -> None:
        internal_actuatorids_by_fly_by_type = defaultdict(lambda: defaultdict(list))

        for fly_name, fly in self.world.fly_lookup.items():
            for actuator_ty, actuators in fly.jointdof_to_mjcfactuator_by_type.items():
                for jointdof, actuator_element in actuators.items():
                    internal_actuator_id = mujoco.mj_name2id(
                        self.mj_model,
                        mujoco.mjtObj.mjOBJ_ACTUATOR,
                        actuator_element.full_identifier,
                    )
                    internal_actuatorids_by_fly_by_type[actuator_ty][fly_name].append(
                        internal_actuator_id
                    )

        self._intern_actuatorids_by_type_by_fly = {
            actuator_ty: {
                fly_name: np.array(ids, dtype=np.int32)
                for fly_name, ids in ids_by_fly.items()
            }
            for actuator_ty, ids_by_fly in internal_actuatorids_by_fly_by_type.items()
        }

    @property
    def time(self) -> float:
        return self.mj_data.time

    def print_performance_report(self) -> None:
        print_perf_report(
            n_steps=self._curr_step,
            n_frames_rendered=self._frames_rendered,
            total_physics_time_ns=self._total_physics_time_ns,
            total_render_time_ns=self._total_render_time_ns,
            timestep=self.mj_model.opt.timestep,
        )
