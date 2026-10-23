import warnings
from typing import Any, Literal, override

import warp as wp
import mujoco as mj
import mujoco_warp as mjw
import numpy as np
from jaxtyping import Float

from flygym.compose.fly import ActuatorType
from flygym.compose.world import BaseWorld
from flygym.simulation import Simulation
from flygym.utils.profiling import print_perf_report_parallel
from flygym.warp.recording import WarpTrajectoryRecorder
from flygym.warp.rendering import (
    WarpGPUBatchRenderer,
    WarpCPURenderer,
    modify_world_for_batch_rendering,
)
from flygym.warp.utils import (
    wp_scatter_indexed_cols_2d,
    wp_gather_indexed_cols_2d,
    wp_gather_indexed_rows_vec3f,
    wp_gather_offset_body_positions,
    wp_gather_offset_body_rotations,
    wp_masked_copy_rows_2d,
)


class GPUSimulation(Simulation):
    """GPU-accelerated parallel simulation using MuJoCo-Warp.

    Runs ``n_worlds`` copies of the same simulation in parallel on the GPU.
    State queries return Warp arrays of shape ``(n_worlds, ...)``, and control
    inputs accept arrays of the same batch shape.

    Requires an NVIDIA GPU with the ``[warp]`` extra installed.

    MuJoCo Warp's warnings for reaching the solver or line search iteration limit
    are disabled, as in MuJoCo Playground, because they are printed once per world
    per step and can flood the output. To re-enable them, set
    `sim.mjw_model.opt.warn_overflow = mjw.OverflowType.ALL` before the first step.

    Args:
        world: A fully configured world with at least one fly attached.
        n_worlds: Number of parallel simulation instances.
        timestep: Physics timestep in seconds. If None, the model's compiled-in
            timestep (from ``mujoco_globals.yaml``) is used.
        max_constraints: Maximum number of constraints per world.
        max_contacts: Maximum number of contacts per world.
        solver_tolerance: Convergence tolerance of the constraint solver
            (`option/tolerance`). The default, 1e-5, is looser than MuJoCo Warp's
            1e-6 (default is 1e-8 in MuJoCo but MuJoCo Warp raises it to 1e-6
            for float32 computation). With 1e-6, a small fraction of worlds can
            stall at float32 precision until they hit the iteration limit. Since
            all worlds wait for the slowest one, this can reduce throughput by
            ~25% with 1000 worlds, without measurably improving accuracy. If None,
            the tolerance from the world's MJCF model is used.

    Attributes:
        n_worlds: Number of parallel worlds.
        mjw_model: GPU-side MuJoCo-Warp model.
        mjw_data: GPU-side MuJoCo-Warp data, batched over ``n_worlds``.
    """

    @override
    def __init__(
        self,
        world: BaseWorld,
        n_worlds: int,
        *,
        timestep: float | None = None,
        max_constraints: int = 500,
        max_contacts: int = 500,
        solver_tolerance: float | None = 1e-5,
    ) -> None:
        self._strip_unsupported_options_for_mjwarp(world)
        super().__init__(world)
        # Apply overrides to the CPU model before the GPU structs are built below, so
        # they propagate into the GPU-side model.
        self._apply_option_overrides(timestep=timestep, tolerance=solver_tolerance)
        self.n_worlds = n_worlds
        self.max_constraints = max_constraints
        self.max_contacts = max_contacts
        self.mjw_model, self.mjw_data = self._mj_structs_to_mjw_structs()

    @override
    def reset(self) -> None:
        """Reset all parallel worlds to the neutral keyframe.

        !!! warning

            `reset()` does not update derived kinematic quantities (`xpos`,
            `xquat`, `site_xpos`, ...); it only restores state fields
            (`qpos`, `qvel`, `act`, `ctrl`, `mocap`, `time`). Reading
            derived quantities right after `reset()`, before calling `forward()` or
            `step()`, returns stale values from before the reset. This is consistent
            with the behavior of MuJoCo Warp's native `reset_data()` function.
        """
        super().reset()
        mjw.reset_data_keyframe(
            self.mjw_model, self.mjw_data, self._neutral_keyframe_id
        )

    def set_state(
        self,
        qpos: Float[np.ndarray | wp.array, "n_worlds nq"] | None = None,
        qvel: Float[np.ndarray | wp.array, "n_worlds nv"] | None = None,
        worlds_mask: wp.array | np.ndarray | None = None,
    ) -> None:
        """Directly set generalized positions and/or velocities for parallel worlds.

        Unlike `reset`, which restores the single neutral keyframe to every (masked)
        world, this writes a possibly different `qpos`/`qvel` per world. It is the
        building block for reference-state initialization, where each world starts an
        episode from a different reference pose.

        Args:
            qpos: Optional generalized positions, shape ``(n_worlds, nq)``. Accepts
                numpy or Warp arrays. If None, positions are left unchanged.
            qvel: Optional generalized velocities, shape ``(n_worlds, nv)``. Accepts
                numpy or Warp arrays. If None, velocities are left unchanged.
            worlds_mask: Optional per-world boolean mask, shape ``(n_worlds,)``. Only
                masked worlds are written; the rest keep their current state. Defaults
                to all worlds.

        !!! warning

            Like `reset`, this does not update derived kinematic quantities (`xpos`,
            `xquat`, `site_xpos`, velocities, ...). Call `forward` (or `step`) before
            reading them, otherwise getters return stale pre-write values.
        """
        if qpos is None and qvel is None:
            return
        if worlds_mask is None:
            mask = wp.ones(self.n_worlds, dtype=wp.bool)
        elif isinstance(worlds_mask, wp.array):
            mask = worlds_mask
        else:
            mask = wp.array(worlds_mask, dtype=wp.bool)
        if mask.shape != (self.n_worlds,):
            raise ValueError(
                f"`worlds_mask` must have shape ({self.n_worlds},), got {mask.shape}."
            )
        for name, src, dst in (
            ("qpos", qpos, self.mjw_data.qpos),
            ("qvel", qvel, self.mjw_data.qvel),
        ):
            if src is None:
                continue
            if isinstance(src, wp.array):
                if src.dtype != wp.float32:
                    raise TypeError(
                        f"`{name}` must be a float32 Warp array, got dtype "
                        f"{wp.types.type_repr(src.dtype)}."
                    )
            else:
                src = wp.array(src, dtype=wp.float32)
            if src.shape != dst.shape:
                raise ValueError(
                    f"`{name}` must have shape {dst.shape}, got {src.shape}."
                )
            wp.launch(
                wp_masked_copy_rows_2d,
                dim=dst.shape,
                inputs=[src, dst, mask],
            )

    @override
    def get_joint_angles(
        self, fly_name: str, dst: wp.array | None = None
    ) -> Float[wp.array, "n_worlds n_jointdofs"]:
        """Get joint angles for all parallel worlds.

        Args:
            fly_name: Name of the fly.
            dst: Optional warp array to store the result. If not specified, a new array
                is allocated.

        Returns:
            Warp array of shape ``(n_worlds, n_jointdofs)`` in radians, ordered as in
            ``fly.get_jointdofs_order()``.
        """
        indices = self._wp_intern_qposadrs_by_fly[fly_name]
        if dst is None:
            dst = wp.zeros((self.n_worlds, indices.size), dtype=wp.float32)
        wp.launch(
            wp_gather_indexed_cols_2d,
            dim=(self.n_worlds, indices.size),
            inputs=[self.mjw_data.qpos, dst, indices],
        )
        return dst

    @override
    def get_joint_velocities(
        self, fly_name: str, dst: wp.array | None = None
    ) -> Float[wp.array, "n_worlds n_jointdofs"]:
        """Get joint velocities for all parallel worlds.

        Args:
            fly_name: Name of the fly.
            dst: Optional warp array to store the result. If not specified, a new array
            is allocated.

        Returns:
            Warp array of shape ``(n_worlds, n_jointdofs)`` in radians per second,
            ordered as in ``fly.get_jointdofs_order()``.
        """
        indices = self._wp_intern_qveladrs_by_fly[fly_name]
        if dst is None:
            dst = wp.zeros((self.n_worlds, indices.size), dtype=wp.float32)
        wp.launch(
            wp_gather_indexed_cols_2d,
            dim=(self.n_worlds, indices.size),
            inputs=[self.mjw_data.qvel, dst, indices],
        )
        return dst

    def get_base_linear_velocity(
        self, fly_name: str, dst: wp.array | None = None
    ) -> Float[wp.array, "n_worlds 3"]:
        """Get the linear velocity of the fly's base (free joint) for all worlds.

        The base velocity is not part of ``get_joint_velocities`` (which covers only the
        fly's articulated DOFs). It is only defined for a fly attached to the world by a
        free joint (the usual walking setup); tethered flies have none.

        Args:
            fly_name: Name of the fly.
            dst: Optional warp array to store the result. If not specified, a new array
                is allocated.

        Returns:
            Warp array of shape ``(n_worlds, 3)`` in mm/s, in the **world** frame.

        Raises:
            KeyError: If the fly has no free (base) joint.
        """
        indices = self._require_base_veladrs(fly_name, self._wp_base_linveladrs_by_fly)
        if dst is None:
            dst = wp.zeros((self.n_worlds, 3), dtype=wp.float32)
        wp.launch(
            wp_gather_indexed_cols_2d,
            dim=(self.n_worlds, 3),
            inputs=[self.mjw_data.qvel, dst, indices],
        )
        return dst

    def get_base_angular_velocity(
        self, fly_name: str, dst: wp.array | None = None
    ) -> Float[wp.array, "n_worlds 3"]:
        """Get the angular velocity of the fly's base (free joint) for all worlds.

        See ``get_base_linear_velocity`` for when the base velocity is defined.

        Args:
            fly_name: Name of the fly.
            dst: Optional warp array to store the result. If not specified, a new array
                is allocated.

        Returns:
            Warp array of shape ``(n_worlds, 3)`` in rad/s, in the **root-body** frame
            (MuJoCo's native free-joint angular-velocity convention).

        Raises:
            KeyError: If the fly has no free (base) joint.
        """
        indices = self._require_base_veladrs(fly_name, self._wp_base_angveladrs_by_fly)
        if dst is None:
            dst = wp.zeros((self.n_worlds, 3), dtype=wp.float32)
        wp.launch(
            wp_gather_indexed_cols_2d,
            dim=(self.n_worlds, 3),
            inputs=[self.mjw_data.qvel, dst, indices],
        )
        return dst

    @staticmethod
    def _require_base_veladrs(fly_name: str, adrs_by_fly: dict) -> wp.array:
        if fly_name not in adrs_by_fly:
            raise KeyError(
                f"Fly '{fly_name}' has no free (base) joint, so its base velocity is "
                "undefined (is it tethered?)."
            )
        return adrs_by_fly[fly_name]

    @override
    def get_body_positions(
        self, fly_name: str, dst: wp.array | None = None
    ) -> Float[wp.array, "n_worlds n_bodies 3"]:
        """Get global body positions for all parallel worlds.

        Args:
            fly_name: Name of the fly.
            dst: Optional warp array to store the result. If not specified, a new array
                is allocated.

        Returns:
            Warp array of shape ``(n_worlds, n_bodies, 3)`` in mm, ordered as in
            ``fly.get_bodysegs_order()``.
        """
        indices = self._wp_internal_bodyids_by_fly[fly_name]
        if dst is None:
            dst = wp.zeros((self.n_worlds, indices.size, 3), dtype=wp.float32)
        wp.launch(
            wp_gather_offset_body_positions,
            dim=(self.n_worlds, indices.size),
            inputs=[
                self.mjw_data.xpos,
                self.mjw_data.xmat,
                dst,
                indices,
                self._wp_body_relpos_by_fly[fly_name],
            ],
        )
        return dst

    @override
    def get_body_rotations(
        self, fly_name: str, dst: wp.array | None = None
    ) -> Float[wp.array, "n_worlds n_bodies 4"]:
        """Get global body orientations as quaternions for all parallel worlds.

        Args:
            fly_name: Name of the fly.
            dst: Optional warp array to store the result. If not specified, a new array
                is allocated.

        Returns:
            Warp array of shape ``(n_worlds, n_bodies, 4)`` (w, x, y, z), ordered as
            in ``fly.get_bodysegs_order()``.
        """
        indices = self._wp_internal_bodyids_by_fly[fly_name]
        if dst is None:
            dst = wp.zeros((self.n_worlds, indices.size, 4), dtype=wp.float32)
        wp.launch(
            wp_gather_offset_body_rotations,
            dim=(self.n_worlds, indices.size),
            inputs=[
                self.mjw_data.xquat,
                dst,
                indices,
                self._wp_body_relquat_by_fly[fly_name],
            ],
        )
        return dst

    @override
    def get_site_positions(
        self, fly_name: str, dst: wp.array | None = None
    ) -> Float[wp.array, "n_worlds n_sites 3"]:
        """Get global anatomical-joint site positions for all parallel worlds.

        Args:
            fly_name: Name of the fly.
            dst: Optional warp array to store the result. If not specified, a new array
                is allocated.

        Returns:
            Warp array of shape ``(n_worlds, n_sites, 3)`` in mm, ordered as in
            ``fly.get_sites_order()``.
        """
        indices = self._wp_internal_siteids_by_fly[fly_name]
        if dst is None:
            dst = wp.zeros((self.n_worlds, indices.size, 3), dtype=wp.float32)
        wp.launch(
            wp_gather_indexed_rows_vec3f,
            dim=(self.n_worlds, indices.size),
            inputs=[self.mjw_data.site_xpos, dst, indices],
        )
        return dst

    @property  # type: ignore[override]
    def time(self) -> float:
        """Current simulation time in seconds (from world 0)."""
        return float(self.mjw_data.time.numpy()[0])

    @override
    def get_actuator_forces(
        self, fly_name: str, actuator_type: ActuatorType, dst: wp.array | None = None
    ) -> Float[wp.array, "n_worlds n_actuators"]:
        """Get actuator forces for all parallel worlds.

        Args:
            fly_name: Name of the fly.
            actuator_type: Type of actuator to query.
            dst: Optional warp array to store the result. If not specified, a new array
                is allocated.

        Returns:
            Warp array of shape ``(n_worlds, n_actuators)``, ordered as in
            ``fly.get_actuated_jointdofs_order(actuator_type)``.
        """
        indices = self._wp_intern_actuatorids_by_type_by_fly[actuator_type][fly_name]
        if dst is None:
            dst = wp.zeros((self.n_worlds, indices.size), dtype=wp.float32)
        wp.launch(
            wp_gather_indexed_cols_2d,
            dim=(self.n_worlds, indices.size),
            inputs=[self.mjw_data.actuator_force, dst, indices],
        )
        return dst

    @override
    def set_leg_adhesion_states(
        self,
        fly_name: str,
        leg_to_adhesion_state: Float[np.ndarray | wp.array, "n_worlds 6"],
    ) -> None:
        """Set adhesion states for each leg across all parallel worlds.

        Args:
            fly_name: Name of the fly.
            leg_to_adhesion_state: Adhesion control array, shape ``(n_worlds, 6)``,
                ordered as in ``fly.get_legs_order()``. Values should be in the range
                ``[0, 1]``. Accepts numpy or Warp arrays.
        """
        if not isinstance(leg_to_adhesion_state, wp.array):
            leg_to_adhesion_state = wp.array(leg_to_adhesion_state, dtype=wp.float32)
        indices = self._wp_intern_adhesionactuatorids_by_fly[fly_name]
        wp.launch(
            wp_scatter_indexed_cols_2d,
            dim=(self.n_worlds, indices.size),
            inputs=[leg_to_adhesion_state, self.mjw_data.ctrl, indices],
        )

    @override
    def set_actuator_inputs(
        self,
        fly_name: str,
        actuator_type: ActuatorType,
        inputs: Float[np.ndarray | wp.array, "n_worlds n_actuators"],
    ) -> None:
        """Set control inputs for all parallel worlds.

        Args:
            fly_name: Name of the fly.
            actuator_type: Type of actuator to control.
            inputs: Control inputs, shape ``(n_worlds, n_actuators)``, ordered as in
                ``fly.get_actuated_jointdofs_order(actuator_type)``. Accepts numpy or
                Warp arrays.
        """
        if not isinstance(inputs, wp.array):
            inputs = wp.array(inputs, dtype=wp.float32)
        indices = self._wp_intern_actuatorids_by_type_by_fly[actuator_type][fly_name]
        wp.launch(
            wp_scatter_indexed_cols_2d,
            dim=(self.n_worlds, indices.size),
            inputs=[inputs, self.mjw_data.ctrl, indices],
        )

    @override
    def step(self) -> None:
        """Advance all parallel worlds by one timestep on the GPU."""
        mjw.step(self.mjw_model, self.mjw_data)

    @override
    def forward(self) -> None:
        """Recompute derived quantities for all parallel worlds on the GPU, without
        integrating. Call this after writing state directly (e.g. via `set_state` or
        `reset`) so that subsequent getters return up-to-date values.
        """
        mjw.forward(self.mjw_model, self.mjw_data)

    @override
    def set_renderer(
        self,
        cameras: str | mj.MjsCamera | list[str | mj.MjsCamera],
        *,
        camera_res: tuple[int, int] = (240, 320),
        playback_speed: float = 0.2,
        output_fps: int = 25,
        buffer_frames: bool = True,
        scene_option: mj.MjvOption | None = None,
        worlds: list[int] | None = None,
        use_gpu_batch_rendering: bool = False,
        **kwargs: Any,
    ) -> WarpGPUBatchRenderer | WarpCPURenderer:
        """Attach a renderer to this GPU simulation.

        Args:
            cameras: Camera(s) to render.
            camera_res: ``(height, width)`` in pixels.
            playback_speed: Video playback speed relative to real time.
            output_fps: Output video frame rate.
            buffer_frames: If True, store rendered frames in memory.
            scene_option: MuJoCo scene options. Uses defaults if None.
            worlds: Indices of worlds to render. Defaults to all worlds.
            use_gpu_batch_rendering: If True, use `WarpGPUBatchRenderer`;
                otherwise use `WarpCPURenderer`.
            **kwargs: Passed to the renderer.

        Returns:
            The created renderer instance.
        """
        if worlds is None:
            worlds = list(range(self.n_worlds))
        self.use_gpu_batch_rendering = use_gpu_batch_rendering

        renderer_kwargs = {
            "mj_model": self.mj_model,
            "n_worlds_total": self.n_worlds,
            "cameras": cameras,
            "camera_res": camera_res,
            "playback_speed": playback_speed,
            "output_fps": output_fps,
            "buffer_frames": buffer_frames,
            "scene_option": scene_option,
            "worlds": worlds,
            **kwargs,
        }
        if use_gpu_batch_rendering:
            is_model_modified = modify_world_for_batch_rendering(self.world)
            if is_model_modified:
                warnings.warn(
                    "The world was modified to be compatible with GPU batch rendering. "
                    "Recompiling the model."
                )
                # Carry option overrides over to the recompiled model
                timestep = self.mj_model.opt.timestep
                tolerance = self.mj_model.opt.tolerance
                self.mj_model, self.mj_data = self.world.compile()
                self._apply_option_overrides(timestep=timestep, tolerance=tolerance)
                self._neutral_keyframe_id = mj.mj_name2id(
                    self.mj_model, mj.mjtObj.mjOBJ_KEY, "neutral"
                )
                mj.mj_resetDataKeyframe(
                    self.mj_model, self.mj_data, self._neutral_keyframe_id
                )
                self.mjw_model, self.mjw_data = self._mj_structs_to_mjw_structs()
                renderer_kwargs["mj_model"] = self.mj_model
            self.renderer = WarpGPUBatchRenderer(**renderer_kwargs)
        else:
            self.renderer = WarpCPURenderer(**renderer_kwargs)

        return self.renderer

    @override
    def render_as_needed(self) -> bool:
        """Render frames for all configured cameras if enough time has elapsed.

        Returns:
            True if a render occurred, False otherwise.
        """
        return self.renderer.render_as_needed(self.mjw_data)

    @override
    def set_trajectory_recorder(
        self,
        *,
        playback_speed: float = 0.2,
        output_fps: int = 25,
        worlds: list[int] | None = None,
    ) -> WarpTrajectoryRecorder:
        """Attach a recorder that stores the kinematic state at the video frame rate.

        The recorded trajectories can be rendered to video later with
        `flygym.warp.recording.render_trajectories_gpu` or
        `flygym.recording.render_trajectories`. See `flygym.recording` for details. The
        recorder is independent of the renderer: either or both can be attached.

        Args:
            playback_speed: Video playback speed relative to real time.
            output_fps: Output video frame rate.
            worlds: Indices of worlds to record. Defaults to all worlds.

        Returns:
            The created `WarpTrajectoryRecorder` instance.
        """
        self.recorder = WarpTrajectoryRecorder(
            self.mj_model,
            self.n_worlds,
            worlds=worlds,
            playback_speed=playback_speed,
            output_fps=output_fps,
        )
        return self.recorder

    @override
    def record_as_needed(self) -> bool:
        """Record the kinematic state of the selected worlds if enough simulation time
        has elapsed since the last recorded frame.

        Returns:
            True if a frame was recorded, False otherwise.
        """
        return self.recorder.record_as_needed(self.mjw_data)

    @override
    def print_performance_report(
        self, show_in_notebook: bool | Literal["auto"] = "auto"
    ) -> None:
        """Print a parallel-simulation performance report.

        Requires that `step_with_profile` and `render_as_needed_with_profile` were
        used during the simulation loop.

        Args:
            show_in_notebook: If True, render the report as an HTML table suitable for
                display in a Jupyter notebook. If "auto", will attempt to detect if
                we're in a notebook environment and choose accordingly.
        """
        print_perf_report_parallel(
            n_steps=self._curr_step,
            n_frames_rendered=self._frames_rendered,
            total_physics_time_ns=self._total_physics_time_ns,
            total_render_time_ns=self._total_render_time_ns,
            timestep=self.timestep,
            n_worlds=self.n_worlds,
            n_worlds_rendered=len(self.renderer.world_ids),
            show_in_notebook=show_in_notebook,
        )

    @override
    def _map_internal_bodyids(self) -> None:
        super()._map_internal_bodyids()
        self._wp_internal_bodyids_by_fly = {
            k: wp.array(v, dtype=wp.int32)
            for k, v in self._internal_bodyids_by_fly.items()
        }
        self._wp_body_relpos_by_fly = {
            k: wp.array(v, dtype=wp.vec3) for k, v in self._body_relpos_by_fly.items()
        }
        self._wp_body_relquat_by_fly = {
            k: wp.array(v, dtype=wp.quat) for k, v in self._body_relquat_by_fly.items()
        }

    @override
    def _map_internal_qposqveladrs(self) -> None:
        super()._map_internal_qposqveladrs()
        self._wp_intern_qposadrs_by_fly = {
            k: wp.array(v, dtype=wp.int32)
            for k, v in self._intern_qposadrs_by_fly.items()
        }
        self._wp_intern_qveladrs_by_fly = {
            k: wp.array(v, dtype=wp.int32)
            for k, v in self._intern_qveladrs_by_fly.items()
        }
        # Base ("free") joint velocity addresses, used by get_base_*_velocity. The world
        # attaches each walking fly's root body to the world with a free joint named
        # after the fly; its 6 velocity DOFs are laid out as [linear xyz, angular xyz].
        # Flies without a free joint (e.g. tethered) are simply left out of these maps.
        self._wp_base_linveladrs_by_fly = {}
        self._wp_base_angveladrs_by_fly = {}
        for fly_name in self._intern_qveladrs_by_fly:
            jid = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, fly_name)
            if jid < 0 or self.mj_model.jnt_type[jid] != mj.mjtJoint.mjJNT_FREE:
                continue
            vadr = int(self.mj_model.jnt_dofadr[jid])
            self._wp_base_linveladrs_by_fly[fly_name] = wp.array(
                [vadr, vadr + 1, vadr + 2], dtype=wp.int32
            )
            self._wp_base_angveladrs_by_fly[fly_name] = wp.array(
                [vadr + 3, vadr + 4, vadr + 5], dtype=wp.int32
            )

    @override
    def _map_internal_actuator_ids(self) -> None:
        super()._map_internal_actuator_ids()
        self._wp_intern_actuatorids_by_type_by_fly = {
            ty: {
                fly_name: wp.array(ids, dtype=wp.int32)
                for fly_name, ids in ids_by_fly.items()
            }
            for ty, ids_by_fly in self._intern_actuatorids_by_type_by_fly.items()
        }

    @override
    def _map_internal_adhesionactuator_ids(self) -> None:
        super()._map_internal_adhesionactuator_ids()
        self._wp_intern_adhesionactuatorids_by_fly = {
            k: wp.array(v, dtype=wp.int32)
            for k, v in self._intern_adhesionactuatorids_by_fly.items()
        }

    @override
    def _map_internal_site_ids(self) -> None:
        super()._map_internal_site_ids()
        self._wp_internal_siteids_by_fly = {
            k: wp.array(v, dtype=wp.int32)
            for k, v in self._internal_siteids_by_fly.items()
        }

    def _apply_option_overrides(
        self, *, timestep: float | None, tolerance: float | None
    ) -> None:
        """Override options in `mj_model.opt`. None leaves an option unchanged."""
        if timestep is not None:
            self.mj_model.opt.timestep = timestep
        if tolerance is not None:
            self.mj_model.opt.tolerance = tolerance

    def _mj_structs_to_mjw_structs(self) -> tuple[mjw.Model, mjw.Data]:
        mjw_model = mjw.put_model(self.mj_model)
        # Silence per-world solver and line search iteration limit warnings, as in
        # MuJoCo Playground (https://github.com/google-deepmind/mujoco_playground/commit/fb246cd).
        # Buffer overflow warnings (contacts, constraints) remain.
        mjw_model.opt.warn_overflow &= ~(
            mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS
        )
        mjw_data = mjw.put_data(
            self.mj_model,
            self.mj_data,
            nworld=self.n_worlds,
            njmax=self.max_constraints,
            nconmax=self.max_contacts,
        )
        return mjw_model, mjw_data

    @staticmethod
    def _strip_unsupported_options_for_mjwarp(world: BaseWorld) -> bool:
        """Remove specs in world MJCF model that are unsupported in MJWarp.

        Modification happens in place. Returns True if any modifications were made,
        False otherwise.

        Note for developers: Check if anything here can be dropped upon new MJWarp
        releases.
        """
        is_modified = False

        # Noslip solver not supported
        if (noslip_iters := world.mjcf_root.option.noslip_iterations) > 0:
            warnings.warn(
                "MJWarp does not support noslip iterations. Changing "
                f"option/noslip_iterations from {noslip_iters} to 0."
            )
            world.mjcf_root.option.noslip_iterations = 0
            is_modified = True

        return is_modified

    @property
    def timestep(self) -> float:
        """Simulation timestep in seconds."""
        return self.mj_model.opt.timestep
