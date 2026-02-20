import warnings
from typing import Any, override
from collections.abc import Sequence
from time import perf_counter_ns

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
from flygym.warp.rendering import GPURenderer
from flygym.warp.utils import wp_scatter_indexed_cols_2d, wp_gather_indexed_cols_2d


class GPUSimulation(Simulation):
    @override
    def __init__(self, world, n_worlds, max_constraints=100, max_contacts=100):
        strip_unsupported_options_for_mjwarp(world.mjcf_root)
        super().__init__(world)
        self.n_worlds = n_worlds
        self.max_constraints = max_constraints
        self.max_contacts = max_contacts
        self.mjw_model, self.mjw_data = self._mj_structs_to_mjw_structs()

    @override
    def reset(self):
        super().reset()
        # The superclass call resets CPU-side MuJoCo structs to the neutral keyframe,
        # so we need to recreate GPU-side structs to reflect that reset.
        self.mjw_model, self.mjw_data = self._mj_structs_to_mjw_structs()
        # ... don't call mjw.reset_data() here! That loses the keyframe reset.

    @override
    def get_joint_angles(
        self, fly_name: str
    ) -> Float[np.ndarray | wp.array, "n_jointdofs"]:
        raise NotImplementedError

    @override
    def get_joint_velocities(
        self, fly_name: str
    ) -> Float[np.ndarray | wp.array, "n_jointdofs"]:
        raise NotImplementedError

    @override
    def get_body_positions(
        self, fly_name: str
    ) -> Float[np.ndarray | wp.array, "n_bodies 3"]:
        raise NotImplementedError

    @override
    def get_body_rotations(
        self, fly_name: str
    ) -> Float[np.ndarray | wp.array, "n_bodies 4"]:
        raise NotImplementedError

    @override
    def set_actuator_inputs(
        self,
        fly_name: str,
        actuator_type: ActuatorType,
        inputs: Float[np.ndarray | wp.array, "n_worlds n_actuators"],
    ):
        if not isinstance(inputs, wp.array):
            inputs = wp.array(inputs, dtype=wp.float32)
        indices = self._wp_intern_actuatorids_by_type_by_fly[actuator_type][fly_name]
        wp.launch(
            wp_scatter_indexed_cols_2d,
            dim=(self.n_worlds, indices.size),
            inputs=[inputs, self.mjw_data.ctrl, indices],
        )

    @override
    def step(self):
        physics_start_ns = perf_counter_ns()
        mjw.step(self.mjw_model, self.mjw_data)
        poststep_start_ns = perf_counter_ns()
        self._total_physics_time_ns += poststep_start_ns - physics_start_ns
        self._curr_step += 1

    def set_renderer(
        self,
        camera: str | mjcf.Element | Sequence[str | mjcf.Element],
        *,
        camera_res: tuple[int, int] = (240, 320),
        playback_speed: float = 0.2,
        output_fps: int = 25,
        **kwargs: Any,
    ) -> GPURenderer:
        self.renderer = GPURenderer(
            self.mjw_model,
            self.mjw_data,
            self.mj_model,
            camera,
            camera_res=camera_res,
            playback_speed=playback_speed,
            output_fps=output_fps,
            **kwargs,
        )
        return self.renderer

    def render_as_needed(self) -> dict[str, Float[np.ndarray, "height width 3"]]:
        render_start_ns = perf_counter_ns()
        render_done = self.renderer.render_as_needed(self.mjw_data)
        render_finish_ns = perf_counter_ns()
        self._total_render_time_ns += render_finish_ns - render_start_ns
        if render_done:
            self._frames_rendered += 1
        return render_done

    @override
    def print_performance_report(self) -> None:
        print_perf_report_parallel(
            n_steps=self._curr_step,
            n_frames_rendered=self._frames_rendered,
            total_physics_time_ns=self._total_physics_time_ns,
            total_render_time_ns=self._total_render_time_ns,
            timestep=self.mj_model.opt.timestep,
            n_worlds=self.n_worlds,
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


def strip_unsupported_options_for_mjwarp(mjcf_root: mjcf.RootElement) -> None:
    """Remove options from dm_control.mjcf.RootElement that are unsupported in MJWarp.

    With every new release of MJWarp, check if anything here can be removed.
    """
    # Noslip solver not supported
    if (noslip_iters := mjcf_root.option.noslip_iterations) > 0:
        warnings.warn(
            "MJWarp does not support noslip iterations. Changing "
            f"option/noslip_iterations from {noslip_iters} to 0."
        )
        mjcf_root.option.noslip_iterations = 0

    # Texture not supported for GPU rendering context until warp 1.12
    # (see https://github.com/google-deepmind/mujoco_warp/pull/1113)
    for material in mjcf_root.asset.find_all("material"):
        if material.texture is not None:
            warnings.warn(
                "MJWarp GPU renderer does not support textures. Removing texture and "
                "adding texture's rgb1 to material directly."
            )
            if material.texture.rgb1 is not None:
                alpha = material.rgba[3] if material.rgba is not None else 1
                material.rgba = (*material.texture.rgb1, alpha)
            material.texture = None
    for texture in mjcf_root.asset.find_all("texture"):
        texture.remove()

    # Add light above each fly explicitly
    for body in mjcf_root.find_all("body"):
        if hasattr(body, "name") and body.name == "rootbody":
            warnings.warn(f"Adding overhead light for body {body.full_identifier}")
            body.add(
                "light",
                name=f"overheadlight",
                mode="track",
                target="c_thorax",
                pos=(0, 0, 7),
                dir=(0, 0, -1),
                directional=True,
                ambient=(1, 1, 1),
                diffuse=(1, 1, 1),
                specular=(1, 1, 1),
            )
