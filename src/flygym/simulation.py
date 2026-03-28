from collections import defaultdict
from time import perf_counter_ns
from typing import Any

import mujoco as mj
import dm_control.mjcf as mjcf
import numpy as np
from jaxtyping import Float

from flygym.compose.fly import ActuatorType
from flygym.compose.world import BaseWorld
from flygym.rendering import Renderer
from flygym.utils.profiling import print_perf_report


class Simulation:
    """CPU-based single-world physics simulation.

    Wraps a compiled MuJoCo model and provides methods for stepping physics,
    reading state, and writing control inputs.

    Args:
        world: A fully configured world with at least one fly attached.

    Attributes:
        world: The world used to construct this simulation.
        renderer: The attached `Renderer`, or None if not set.
        mj_model: Compiled MuJoCo model.
        mj_data: Associated MuJoCo data.
    """

    def __init__(self, world: BaseWorld) -> None:
        if len(world.fly_lookup) == 0:
            raise ValueError("The world must contain at least one fly.")
        self.renderer = None
        self.world = world
        self.mj_model, self.mj_data = world.compile()
        self._neutral_keyframe_id = mj.mj_name2id(
            self.mj_model, mj.mjtObj.mjOBJ_KEY, "neutral"
        )
        mj.mj_resetDataKeyframe(self.mj_model, self.mj_data, self._neutral_keyframe_id)

        # Map internal IDs in the compiled MuJoCo model. This allows users to read from
        # or write to body/joint/actuator in orders defined by Fly objects.
        self._map_internal_bodyids()
        self._map_internal_qposqveladrs()
        self._map_internal_actuator_ids()
        self._map_internal_adhesionactuator_ids()
        self._map_internal_jointids()
        self._map_internal_groundcontactsensor_ids()

        # For performance profiling
        self._curr_step = 0
        self._frames_rendered = 0
        self._total_physics_time_ns = 0
        self._total_render_time_ns = 0

    def reset(self) -> None:
        """Reset simulation and renderer to the neutral keyframe."""
        # Reset physics
        mj.mj_resetDataKeyframe(self.mj_model, self.mj_data, self._neutral_keyframe_id)

        # Reset renderers
        if self.renderer is not None:
            self.renderer.reset()

        # Stuff for performance profiling
        self._curr_step = 0
        self._frames_rendered = 0
        self._total_physics_time_ns = 0
        self._total_render_time_ns = 0

    def step(self) -> None:
        """Advance physics by one timestep."""
        mj.mj_step(self.mj_model, self.mj_data)

    def step_with_profile(self) -> None:
        """Advance physics by one timestep, accumulating timing data for profiling."""
        physics_start_ns = perf_counter_ns()
        self.step()
        physics_finish_ns = perf_counter_ns()
        self._total_physics_time_ns += physics_finish_ns - physics_start_ns
        self._curr_step += 1

    def set_renderer(
        self,
        cameras: str | mjcf.Element | list[str | mjcf.Element],
        *,
        camera_res: tuple[int, int] = (240, 320),
        playback_speed: float = 0.2,
        output_fps: int = 25,
        buffer_frames: bool = True,
        scene_option: mj.MjvOption | None = None,
        **kwargs: Any,
    ) -> Renderer:
        """Attach a renderer to this simulation.

        Args:
            cameras: Camera(s) to render. Can be a camera name, MJCF camera element,
                or a sequence of either.
            camera_res: ``(height, width)`` in pixels.
            playback_speed: Video playback speed relative to real time.
            output_fps: Output video frame rate.
            buffer_frames: If True, store rendered frames in memory.
            scene_option: MuJoCo scene options. Uses defaults if None.
            **kwargs: Passed to ``mujoco.Renderer``.

        Returns:
            The created `Renderer` instance.
        """
        self.renderer = Renderer(
            self.mj_model,
            cameras,
            camera_res=camera_res,
            playback_speed=playback_speed,
            output_fps=output_fps,
            buffer_frames=buffer_frames,
            scene_option=scene_option,
            **kwargs,
        )
        return self.renderer

    def render_as_needed(self) -> bool:
        """Render a frame if enough simulation time has elapsed since the last render.

        Returns:
            True if a frame was rendered, False otherwise.
        """
        return self.renderer.render_as_needed(self.mj_data)

    def render_as_needed_with_profile(self) -> bool:
        """Like `render_as_needed`, but also accumulates render timing data."""
        render_start_ns = perf_counter_ns()
        render_done = self.render_as_needed()
        render_finish_ns = perf_counter_ns()
        self._total_render_time_ns += render_finish_ns - render_start_ns
        if render_done:
            self._frames_rendered += 1
        return render_done

    def get_joint_angles(self, fly_name: str) -> Float[np.ndarray, "n_jointdofs"]:
        """Get current joint angles ordered by the fly's skeleton.

        Args:
            fly_name: Name of the fly.

        Returns:
            Joint angles in radians, shape ``(n_jointdofs,)``, ordered as in
            ``fly.get_jointdofs_order()``.
        """
        internal_ids = self._intern_qposadrs_by_fly[fly_name]
        return self.mj_data.qpos[internal_ids]

    def get_joint_velocities(self, fly_name: str) -> Float[np.ndarray, "n_jointdofs"]:
        """Get current joint angular velocities ordered by the fly's skeleton.

        Args:
            fly_name: Name of the fly.

        Returns:
            Joint velocities in radians per second, shape ``(n_jointdofs,)``, ordered
            as in ``fly.get_jointdofs_order()``.
        """
        internal_ids = self._intern_qveladrs_by_fly[fly_name]
        return self.mj_data.qvel[internal_ids]

    def get_body_positions(self, fly_name: str) -> Float[np.ndarray, "n_bodies 3"]:
        """Get global 3D positions of all body segments.

        Args:
            fly_name: Name of the fly.

        Returns:
            Body positions in mm, shape ``(n_bodies, 3)``, ordered as in
            ``fly.get_bodysegs_order()``.
        """
        internal_ids = self._internal_bodyids_by_fly[fly_name]
        return self.mj_data.xpos[internal_ids, :]

    def get_body_rotations(self, fly_name: str) -> Float[np.ndarray, "n_bodies 4"]:
        """Get global orientations of all body segments as quaternions (w, x, y, z).

        Args:
            fly_name: Name of the fly.

        Returns:
            Body quaternions, shape ``(n_bodies, 4)``, ordered as in
            ``fly.get_bodysegs_order()``.
        """
        internal_ids = self._internal_bodyids_by_fly[fly_name]
        return self.mj_data.xquat[internal_ids, :]

    def get_actuator_forces(
        self, fly_name: str, actuator_type: ActuatorType
    ) -> Float[np.ndarray, "n_actuators"]:
        """Get actuator forces for the given actuator type.

        Args:
            fly_name: Name of the fly.
            actuator_type: Type of actuator to query.

        Returns:
            Actuator forces, shape ``(n_actuators,)``, ordered as in
            ``fly.get_actuated_jointdofs_order(actuator_type)``.
        """
        internal_ids = self._intern_actuatorids_by_type_by_fly[actuator_type][fly_name]
        return self.mj_data.actuator_force[internal_ids]

    def get_ground_contact_info(self, fly_name: str) -> tuple[
        Float[np.ndarray, "6"],  # contact/no contact flag
        Float[np.ndarray, "6 3"],  # force (in contact frame)
        Float[np.ndarray, "6 3"],  # torque (in contact frame)
        Float[np.ndarray, "6 3"],  # pos (in global frame)
        Float[np.ndarray, "6 3"],  # normal (in global frame)
        Float[np.ndarray, "6 3"],  # tangent (in global frame)
    ]:
        """Get ground contact information for all six legs.

        Args:
            fly_name: Name of the fly.

        Returns:
            A 6-tuple, one entry per leg ordered as in ``fly.get_legs_order()``:

            - ``contact_active``: shape ``(6,)`` — 1 if in contact, 0 otherwise.
            - ``forces``: shape ``(6, 3)`` — contact force in contact frame.
            - ``torques``: shape ``(6, 3)`` — contact torque in contact frame.
            - ``positions``: shape ``(6, 3)`` — contact position in global frame.
            - ``normals``: shape ``(6, 3)`` — contact normal in global frame.
            - ``tangents``: shape ``(6, 3)`` — contact tangent in global frame.
        """
        internal_ids = self._intern_groundcontactsensorids_by_fly[fly_name]
        sensor_data = self.mj_data.sensordata[internal_ids]
        # Reshape (6 legs * 16 dims per sensor,) to (6 legs, 16 dim per sensor)
        sensor_data = sensor_data.reshape(6, 16)
        contact_active = sensor_data[:, 0]
        forces = sensor_data[:, 1:4]
        torques = sensor_data[:, 4:7]
        positions = sensor_data[:, 7:10]
        normals = sensor_data[:, 10:13]
        tangents = sensor_data[:, 13:]
        return contact_active, forces, torques, positions, normals, tangents

    def set_actuator_inputs(
        self,
        fly_name: str,
        actuator_type: ActuatorType,
        inputs: Float[np.ndarray, "n_actuators"],
    ) -> None:
        """Set control inputs for the given actuator type.

        Args:
            fly_name: Name of the fly.
            actuator_type: Type of actuator to control.
            inputs: Control inputs, shape ``(n_actuators,)``, ordered as in
                ``fly.get_actuated_jointdofs_order(actuator_type)``.
        """
        internal_ids = self._intern_actuatorids_by_type_by_fly[actuator_type][fly_name]
        if len(inputs) != len(internal_ids):
            raise ValueError(
                f"Expected {len(internal_ids)} inputs for actuator type "
                f"'{actuator_type.name}', but got {len(inputs)}"
            )
        self.mj_data.ctrl[internal_ids] = inputs

    def set_leg_adhesion_states(
        self, fly_name: str, leg_to_adhesion_state: Float[np.ndarray, "6"]
    ) -> None:
        """Set adhesion states for each leg.

        Args:
            fly_name: Name of the fly.
            leg_to_adhesion_state: Adhesion gain per leg, shape ``(6,)``, ordered as in
                ``fly.get_legs_order()``. Values should be in the range ``[1, 100]``.
        """
        internal_ids = self._intern_adhesionactuatorids_by_fly[fly_name]
        if len(leg_to_adhesion_state) != len(internal_ids):
            raise ValueError(
                "Unexpected number of adhesion states: "
                f"expected {len(internal_ids)}, got {len(leg_to_adhesion_state)}"
            )
        self.mj_data.ctrl[internal_ids] = leg_to_adhesion_state

    def warmup(self, duration_s: float = 0.05) -> None:
        """Step the simulation for a short period to settle initialization transients.

        Call after `reset` and before the main simulation loop to allow the fly to
        settle onto the ground.

        Args:
            duration_s: Duration of the warmup period in seconds.
        """
        n_steps = int(duration_s / self.mj_model.opt.timestep)
        for _ in range(n_steps):
            self.step()

    def _map_internal_bodyids(self) -> None:
        internal_bodyids_by_fly = defaultdict(list)

        for fly_name, fly in self.world.fly_lookup.items():
            for bodyseg, mjcf_body_element in fly.bodyseg_to_mjcfbody.items():
                internal_body_id = mj.mj_name2id(
                    self.mj_model,
                    mj.mjtObj.mjOBJ_BODY,
                    mjcf_body_element.full_identifier,
                )
                internal_bodyids_by_fly[fly_name].append(internal_body_id)

        self._internal_bodyids_by_fly = {
            k: np.array(v, dtype=np.int32) for k, v in internal_bodyids_by_fly.items()
        }

    def _map_internal_jointids(self) -> None:
        internal_jointids_by_fly = defaultdict(list)

        for fly_name, fly in self.world.fly_lookup.items():
            for jointdof, mjcf_joint_element in fly.jointdof_to_mjcfjoint.items():
                internal_joint_id = mj.mj_name2id(
                    self.mj_model,
                    mj.mjtObj.mjOBJ_JOINT,
                    mjcf_joint_element.full_identifier,
                )
                internal_jointids_by_fly[fly_name].append(internal_joint_id)

        self._internal_jointids_by_fly = {
            k: np.array(v, dtype=np.int32) for k, v in internal_jointids_by_fly.items()
        }

    def _map_internal_qposqveladrs(self) -> None:
        internal_qposadrs_by_fly = defaultdict(list)
        internal_qveladrs_by_fly = defaultdict(list)

        for fly_name, fly in self.world.fly_lookup.items():
            for jointdof, mjcf_joint_element in fly.jointdof_to_mjcfjoint.items():
                internal_joint_id = mj.mj_name2id(
                    self.mj_model,
                    mj.mjtObj.mjOBJ_JOINT,
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
                    internal_actuator_id = mj.mj_name2id(
                        self.mj_model,
                        mj.mjtObj.mjOBJ_ACTUATOR,
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

    def _map_internal_adhesionactuator_ids(self) -> None:
        internal_adhesionactuatorids_by_fly = defaultdict(list)
        for fly_name, fly in self.world.fly_lookup.items():
            if len(fly.leg_to_adhesionactuator) == 0:
                continue  # This fly doesn't have leg adhesion actuators
            for leg in fly.get_legs_order():
                actuator_element = fly.leg_to_adhesionactuator[leg]
                internal_actuator_id = mj.mj_name2id(
                    self.mj_model,
                    mj.mjtObj.mjOBJ_ACTUATOR,
                    actuator_element.full_identifier,
                )
                internal_adhesionactuatorids_by_fly[fly_name].append(
                    internal_actuator_id
                )
        self._intern_adhesionactuatorids_by_fly = {
            fly_name: np.array(ids, dtype=np.int32)
            for fly_name, ids in internal_adhesionactuatorids_by_fly.items()
        }

    def _map_internal_groundcontactsensor_ids(self) -> None:
        if self.world.legpos_to_groundcontactsensors_by_fly is None:
            self._intern_groundcontactsensorids_by_fly = None
            return
        else:
            self._intern_groundcontactsensorids_by_fly = {}

        for fly_name, fly in self.world.fly_lookup.items():
            indices_thisfly = []
            for leg in fly.get_legs_order():
                sensor = self.world.legpos_to_groundcontactsensors_by_fly[fly_name][leg]
                internal_id = mj.mj_name2id(
                    self.mj_model, mj.mjtObj.mjOBJ_SENSOR, sensor.full_identifier
                )
                start_idx = self.mj_model.sensor_adr[internal_id]
                sensor_dim = self.mj_model.sensor_dim[internal_id]
                # Sensor should be 16-dim: found (1), force (3), torque (3), pos (3),
                # normal (3), tangent (3)
                assert sensor_dim == 16, "unexpected ground contact sensor dimension"
                indices_thisfly.extend(list(range(start_idx, start_idx + sensor_dim)))
            indices_arr = np.array(indices_thisfly, dtype=np.int32)
            self._intern_groundcontactsensorids_by_fly[fly_name] = indices_arr

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        return self.mj_data.time

    def print_performance_report(self) -> None:
        """Print a summary of physics and rendering performance.

        Requires that `step_with_profile` and `render_as_needed_with_profile` were
        used during the simulation loop.
        """
        print_perf_report(
            n_steps=self._curr_step,
            n_frames_rendered=self._frames_rendered,
            total_physics_time_ns=self._total_physics_time_ns,
            total_render_time_ns=self._total_render_time_ns,
            timestep=self.mj_model.opt.timestep,
        )
