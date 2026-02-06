from collections import defaultdict
from typing import Literal

import mujoco
import dm_control.mjcf as mjcf
import numpy as np
from numpy.typing import NDArray
from jaxtyping import Float
from time import perf_counter_ns

from flygym.anatomy import JointDOF
from flygym.compose.fly import Fly, ActuatorType, PoseDict
from flygym.compose.world import BaseWorld
from flygym.rendering import Renderer
from flygym.utils.profiling import print_perf_report

StaticPoseType = dict[str | JointDOF, float] | Float[NDArray, "n_jointdofs"]


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
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # Reset renderers
        for renderer in self.renderers.values():
            renderer.reset()

        # Stuff for performance profiling
        self._curr_step = 0
        self._frames_rendered = 0
        self._total_prestep_time_ns = 0
        self._total_physics_time_ns = 0
        self._total_poststep_time_ns = 0
        self._total_render_time_ns = 0

    def _set_actuator_input_prestep(self, *args, **kwargs) -> None:
        return

    def _process_updated_state_poststep(self) -> any:
        return

    def step(self, *actuator_input_args, **actuator_input_kwargs) -> any:
        # Prestep: set control input
        prestep_start_ns = perf_counter_ns()
        self._set_actuator_input_prestep(*actuator_input_args, **actuator_input_kwargs)

        # Step physics forward
        physics_start_ns = perf_counter_ns()
        mujoco.mj_step(self.mj_model, self.mj_data)

        # Poststep: process updated state
        poststep_start_ns = perf_counter_ns()
        observation = self._process_updated_state_poststep()

        # Render image (optional)
        render_start_ns = perf_counter_ns()
        is_rendered = False
        for renderer in self.renderers.values():
            image = renderer.render_as_needed(self.mj_data)
            is_rendered |= image is not None
        render_finish_ns = perf_counter_ns()

        # Update profiling stats
        self._total_prestep_time_ns += physics_start_ns - prestep_start_ns
        self._total_physics_time_ns += poststep_start_ns - physics_start_ns
        self._total_poststep_time_ns += render_start_ns - poststep_start_ns
        self._total_render_time_ns += render_finish_ns - render_start_ns
        self._curr_step += 1
        if is_rendered:
            self._frames_rendered += 1

        return observation

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

    def get_joint_angles(
        self, fly: Fly | str | None = None
    ) -> Float[NDArray, "n_jointdofs"]:
        fly_name = self._get_fly_name(self, fly)
        internal_ids = self._intern_qposadrs_by_fly[fly_name]
        return self.mj_data.qpos[internal_ids]

    def get_joint_velocities(
        self, fly: Fly | str | None = None
    ) -> Float[NDArray, "n_jointdofs"]:
        fly_name = self._get_fly_name(self, fly)
        internal_ids = self._intern_qveladrs_by_fly[fly_name]
        return self.mj_data.qvel[internal_ids]

    def get_body_positions(
        self, fly: Fly | str | None = None
    ) -> Float[NDArray, "n_bodies 3"]:
        fly_name = self._get_fly_name(self, fly)
        internal_ids = self._internal_bodyids_by_fly[fly_name]
        return self.mj_data.xpos[internal_ids, :]

    def get_body_rotations(
        self, fly: Fly | str | None = None
    ) -> Float[NDArray, "n_bodies 4"]:
        fly_name = self._get_fly_name(self, fly)
        internal_ids = self._internal_bodyids_by_fly[fly_name]
        return self.mj_data.xquat[internal_ids, :]

    def set_actuator_inputs(
        self,
        actuator_type: ActuatorType,
        inputs: Float[NDArray, "n_actuators"],
        fly: Fly | str | None = None,
    ):
        fly_name = self._get_fly_name(self, fly)
        internal_ids = self._intern_actuatorids_by_type_by_fly[fly_name][actuator_type]
        if len(inputs) != len(internal_ids):
            raise ValueError(
                f"Expected {len(internal_ids)} inputs for actuator type "
                f"'{actuator_type.name}', but got {len(inputs)}"
            )
        self.mj_data.ctrl[internal_ids] = inputs

    def set_state(
        self, pose_dicts: dict[Fly | str, PoseDict] | PoseDict, as_default: bool = False
    ) -> None:
        if isinstance(pose_dicts, PoseDict):
            pose_dicts = {self.fly: pose_dicts}

        # mujoco.mj_resetData(self.mj_model, self.mj_data)
        # qpos = self.mj_data.qpos
        qpos0 = self.mj_model.qpos0.copy()
        for fly, pose_dict in pose_dicts.items():
            fly = fly if isinstance(fly, Fly) else self.world.fly_lookup[fly]
            for jointdof_name, angle in pose_dict.items():
                jointdof = JointDOF.from_name(jointdof_name)
                internal_jointid = mujoco.mj_name2id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    fly.jointdof_to_mjcfjoint[jointdof].full_identifier,
                )
                qposadr = self.mj_model.jnt_qposadr[internal_jointid]
                qveladr = self.mj_model.jnt_dofadr[internal_jointid]
                self.mj_data.qpos[qposadr] = angle
                self.mj_data.qvel[qveladr] = 0.0
        # mujoco.mj_forward(self.mj_model, self.mj_data)

        if as_default:
            print(self.mj_model.qpos0)
            self.mj_model.qpos0[:] = qpos0
            # self.mj_model.qpos_spring[:] = self.mj_data.qpos[:]
            # Reinitialize MjData using the updated MjModel and reset derived constants
            self.mj_data = mujoco.MjData(self.mj_model)
            mujoco.mj_setConst(self.mj_model, self.mj_data)
            # Reset compiled model and data to apply the new qpos0
            mujoco.mj_resetData(self.mj_model, self.mj_data)
            mujoco.mj_forward(self.mj_model, self.mj_data)

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
            total_prestep_time_ns=self._total_prestep_time_ns,
            total_physics_time_ns=self._total_physics_time_ns,
            total_poststep_time_ns=self._total_poststep_time_ns,
            total_render_time_ns=self._total_render_time_ns,
        )

    @property
    def fly(self) -> Fly | None:
        """Shortcut to access the fly if there is only one fly. Raises
        ValueError if there are multiple flies."""
        if len(self.world.fly_lookup) == 1:
            return next(iter(self.world.fly_lookup.values()))
        raise ValueError(
            "World contains more than one fly. Fly must be specified explicitly."
        )

    def _get_fly_name(self, fly: Fly | str | None) -> str:
        if isinstance(fly, str):
            return fly
        elif isinstance(fly, Fly):
            return fly.name
        elif fly is None:
            return self.fly.name
        else:
            raise ValueError(
                "fly must be of type Fly, str, or None (if only one fly exists)."
            )
