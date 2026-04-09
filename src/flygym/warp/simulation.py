import warnings
from typing import Any, Literal, override

import warp as wp
import mujoco as mj
import mujoco_warp as mjw
import dm_control.mjcf as mjcf
import numpy as np
from jaxtyping import Float

from flygym.compose.fly import ActuatorType
from flygym.compose.world import BaseWorld
from flygym.simulation import Simulation
from flygym.utils.profiling import print_perf_report_parallel
from flygym.warp.rendering import (
    WarpGPUBatchRenderer,
    WarpCPURenderer,
    modify_world_for_batch_rendering,
)
from flygym.warp.utils import (
    wp_scatter_indexed_cols_2d,
    wp_gather_indexed_cols_2d,
    wp_gather_indexed_rows_vec3f,
    wp_gather_indexed_rows_quatf,
)


class GPUSimulation(Simulation):
    """GPU-accelerated parallel simulation using MuJoCo-Warp.

    Runs ``n_worlds`` copies of the same simulation in parallel on the GPU.
    State queries return Warp arrays of shape ``(n_worlds, ...)``, and control
    inputs accept arrays of the same batch shape.

    Requires an NVIDIA GPU with the ``[warp]`` extra installed.

    Args:
        world: A fully configured world with at least one fly attached.
        n_worlds: Number of parallel simulation instances.
        max_constraints: Maximum number of constraints per world.
        max_contacts: Maximum number of contacts per world.

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
        max_constraints: int = 500,
        max_contacts: int = 500,
    ) -> None:
        self._strip_unsupported_options_for_mjwarp(world)
        super().__init__(world)
        self.n_worlds = n_worlds
        self.max_constraints = max_constraints
        self.max_contacts = max_contacts
        self.mjw_model, self.mjw_data = self._mj_structs_to_mjw_structs()

    @override
    def reset(self) -> None:
        """Reset all parallel worlds to the neutral keyframe."""
        super().reset()
        # The superclass call resets CPU-side MuJoCo structs to the neutral keyframe,
        # so we need to recreate GPU-side structs to reflect that reset.
        self.mjw_model, self.mjw_data = self._mj_structs_to_mjw_structs()
        # ... don't call mjw.reset_data() here! That loses the keyframe reset.

    @override
    def get_joint_angles(
        self, fly_name: str
    ) -> Float[wp.array, "n_worlds n_jointdofs"]:
        """Get joint angles for all parallel worlds.

        Args:
            fly_name: Name of the fly.

        Returns:
            Warp array of shape ``(n_worlds, n_jointdofs)`` in radians, ordered as in
            ``fly.get_jointdofs_order()``.
        """
        indices = self._wp_intern_qposadrs_by_fly[fly_name]
        dst = wp.zeros((self.n_worlds, indices.size), dtype=wp.float32)
        wp.launch(
            wp_gather_indexed_cols_2d,
            dim=(self.n_worlds, indices.size),
            inputs=[self.mjw_data.qpos, dst, indices],
        )
        return dst

    @override
    def get_joint_velocities(
        self, fly_name: str
    ) -> Float[wp.array, "n_worlds n_jointdofs"]:
        """Get joint velocities for all parallel worlds.

        Args:
            fly_name: Name of the fly.

        Returns:
            Warp array of shape ``(n_worlds, n_jointdofs)`` in radians per second,
            ordered as in ``fly.get_jointdofs_order()``.
        """
        indices = self._wp_intern_qveladrs_by_fly[fly_name]
        dst = wp.zeros((self.n_worlds, indices.size), dtype=wp.float32)
        wp.launch(
            wp_gather_indexed_cols_2d,
            dim=(self.n_worlds, indices.size),
            inputs=[self.mjw_data.qvel, dst, indices],
        )
        return dst

    @override
    def get_body_positions(
        self, fly_name: str
    ) -> Float[wp.array, "n_worlds n_bodies 3"]:
        """Get global body positions for all parallel worlds.

        Args:
            fly_name: Name of the fly.

        Returns:
            Warp array of shape ``(n_worlds, n_bodies, 3)`` in mm, ordered as in
            ``fly.get_bodysegs_order()``.
        """
        indices = self._wp_internal_bodyids_by_fly[fly_name]
        dst = wp.zeros((self.n_worlds, indices.size, 3), dtype=wp.float32)
        wp.launch(
            wp_gather_indexed_rows_vec3f,
            dim=(self.n_worlds, indices.size),
            inputs=[self.mjw_data.xpos, dst, indices],
        )
        return dst

    @override
    def get_body_rotations(
        self, fly_name: str
    ) -> Float[wp.array, "n_worlds n_bodies 4"]:
        """Get global body orientations as quaternions for all parallel worlds.

        Args:
            fly_name: Name of the fly.

        Returns:
            Warp array of shape ``(n_worlds, n_bodies, 4)`` (w, x, y, z), ordered as
            in ``fly.get_bodysegs_order()``.
        """
        indices = self._wp_internal_bodyids_by_fly[fly_name]
        dst = wp.zeros((self.n_worlds, indices.size, 4), dtype=wp.float32)
        wp.launch(
            wp_gather_indexed_rows_quatf,
            dim=(self.n_worlds, indices.size),
            inputs=[self.mjw_data.xquat, dst, indices],
        )
        return dst

    @override
    def get_site_positions(
        self, fly_name: str
    ) -> Float[wp.array, "n_worlds n_sites 3"]:
        """Get global anatomical-joint site positions for all parallel worlds.

        Args:
            fly_name: Name of the fly.

        Returns:
            Warp array of shape ``(n_worlds, n_sites, 3)`` in mm, ordered as in
            ``fly.get_sites_order()``.
        """
        indices = self._wp_internal_siteids_by_fly[fly_name]
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
        self,
        fly_name: str,
        actuator_type: ActuatorType,
    ) -> Float[wp.array, "n_worlds n_actuators"]:
        """Get actuator forces for all parallel worlds.

        Args:
            fly_name: Name of the fly.
            actuator_type: Type of actuator to query.

        Returns:
            Warp array of shape ``(n_worlds, n_actuators)``, ordered as in
            ``fly.get_actuated_jointdofs_order(actuator_type)``.
        """
        indices = self._wp_intern_actuatorids_by_type_by_fly[actuator_type][fly_name]
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
            leg_to_adhesion_state: Adhesion gain array, shape ``(n_worlds, 6)``,
                ordered as in ``fly.get_legs_order()``. Accepts numpy or Warp arrays.
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
    def set_renderer(
        self,
        cameras: str | mjcf.Element | list[str | mjcf.Element],
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
                self.mj_model, self.mj_data = self.world.compile()
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
    def render_as_needed(self) -> dict[str, Float[np.ndarray, "height width 3"]]:
        """Render frames for all configured cameras if enough time has elapsed.

        Returns:
            Dict mapping camera name to rendered frame array ``(height, width, 3)``,
            or an empty dict if no render occurred.
        """
        return self.renderer.render_as_needed(self.mjw_data)

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

    def _mj_structs_to_mjw_structs(self) -> tuple[mjw.Model, mjw.Data]:
        mjw_model = mjw.put_model(self.mj_model)
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
