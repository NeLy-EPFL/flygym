from collections import defaultdict

import mujoco
import dm_control.mjcf as mjcf
import numpy as np
from numpy.typing import NDArray
from jaxtyping import Float
from time import perf_counter_ns

from flygym.compose.fly import ActuatorType
from flygym.compose.world import BaseWorld
from flygym.rendering import Renderer
from flygym.utils.profiling import print_perf_report


class Simulation:
    def __init__(self, world: BaseWorld) -> None:
        if len(world.fly_lookup) == 0:
            raise ValueError("The world must contain at least one fly.")
        self.renderers: dict[str, Renderer] = {}
        self.world = world
        self.mj_model, self.mj_data = world.compile()

        # Map internal IDs in compiled MuJoCo model. This allows user to read to/write
        # from body/joint/actuator in orders defined by Fly objects.
        self._map_internal_bodyids()
        self._map_internal_qposqveladrs()
        self._map_internal_actuator_ids()

        # Reset everything (physics, renderers, and profiling stats)
        self.reset()

    def reset(self) -> None:
        # Reset physics
        neutral_keyframe_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "neutral"
        )
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, neutral_keyframe_id)

        # Reset renderers
        for renderer in self.renderers.values():
            renderer.reset()

        # Stuff for performance profiling
        self._curr_step = 0
        self._frames_rendered = 0
        self._total_physics_time_ns = 0
        self._total_render_time_ns = 0

    def step(self) -> None:
        # Step physics forward
        physics_start_ns = perf_counter_ns()
        mujoco.mj_step(self.mj_model, self.mj_data)
        poststep_start_ns = perf_counter_ns()
        self._total_physics_time_ns += poststep_start_ns - physics_start_ns
        self._curr_step += 1

    def render_as_needed(self) -> dict[str, Float[NDArray, "height width 3"]]:
        render_start_ns = perf_counter_ns()
        images = {}
        for name, renderer in self.renderers.items():
            image = renderer.render_as_needed(self.mj_data)
            if image is not None:
                images[name] = image
        render_finish_ns = perf_counter_ns()

        # Update profiling stats
        self._total_render_time_ns += render_finish_ns - render_start_ns
        if images:
            self._frames_rendered += 1

        return images

    def add_renderer(
        self,
        name: str | None = None,
        *,
        height: int = 240,
        width: int = 320,
        play_speed: float = 0.2,
        out_fps: int = 25,
        camera: mjcf.Element | str | int | mujoco.MjvCamera = -1,
        **kwargs,
    ) -> None:
        if isinstance(camera, mjcf.Element) and camera.tag == "camera":
            camera = camera.full_identifier

        renderer = Renderer(
            self.mj_model,
            height=height,
            width=width,
            play_speed=play_speed,
            out_fps=out_fps,
            camera=camera,
            **kwargs,
        )
        name = f"renderer{len(self.renderers) + 1}" if name is None else name
        self.renderers[name] = renderer

    def get_joint_angles(self, fly_name: str) -> Float[NDArray, "n_jointdofs"]:
        internal_ids = self._intern_qposadrs_by_fly[fly_name]
        return self.mj_data.qpos[internal_ids]

    def get_joint_velocities(self, fly_name: str) -> Float[NDArray, "n_jointdofs"]:
        internal_ids = self._intern_qveladrs_by_fly[fly_name]
        return self.mj_data.qvel[internal_ids]

    def get_body_positions(self, fly_name: str) -> Float[NDArray, "n_bodies 3"]:
        internal_ids = self._internal_bodyids_by_fly[fly_name]
        return self.mj_data.xpos[internal_ids, :]

    def get_body_rotations(self, fly_name: str) -> Float[NDArray, "n_bodies 4"]:
        internal_ids = self._internal_bodyids_by_fly[fly_name]
        return self.mj_data.xquat[internal_ids, :]

    def set_actuator_inputs(
        self,
        fly_name: str,
        actuator_type: ActuatorType,
        inputs: Float[NDArray, "n_actuators"],
    ):
        internal_ids = self._intern_actuatorids_by_type_by_fly[fly_name][actuator_type]
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
        self._intern_actuatorids_by_type_by_fly = {}

        for fly_name, fly in self.world.fly_lookup.items():
            ids_thisfly = defaultdict(list)
            for actuator_ty, actuators in fly.jointdof_to_mjcfactuator_by_type.items():
                for jointdof, actuator_element in actuators.items():
                    internal_actuator_id = mujoco.mj_name2id(
                        self.mj_model,
                        mujoco.mjtObj.mjOBJ_ACTUATOR,
                        actuator_element.full_identifier,
                    )
                    ids_thisfly[actuator_ty].append(internal_actuator_id)

            ids_thisfly = {
                ty: np.array(ids, dtype=np.int32) for ty, ids in ids_thisfly.items()
            }
            self._intern_actuatorids_by_type_by_fly[fly_name] = ids_thisfly

    @property
    def time(self) -> float:
        return self.mj_data.time

    def print_performance_report(self) -> None:
        print_perf_report(
            n_steps=self._curr_step,
            n_frames_rendered=self._frames_rendered,
            total_physics_time_ns=self._total_physics_time_ns,
            total_render_time_ns=self._total_render_time_ns,
        )
