"""Live multi-world MuJoCo-Warp rendering (GPU batch and per-world CPU).

For recording qpos trajectories on GPU and replaying them, see
`flygym.warp.rendering.recorded_trajectory`.
"""

import warnings
from typing import Any, override

import mujoco as mj
import mujoco_warp as mjw
import warp as wp
import numpy as np

from flygym.compose import BaseWorld
from flygym.warp.rendering.base import _BaseWarpRenderer
from flygym.warp.utils import get_rgb_selected_worlds_and_cameras

__all__ = [
    "WarpGPUBatchRenderer",
    "WarpCPURenderer",
    "modify_world_for_batch_rendering",
]


class WarpGPUBatchRenderer(_BaseWarpRenderer):
    """GPU-side renderer using MJWarp's GPU batch rendering functionality."""

    def _render_setup_impl(self, **kwargs: Any) -> None:
        if not self._is_scene_option_default(self.scene_option):
            raise RuntimeError(
                "Custom scene options are not supported with WarpGPUBatchRenderer "
                "because it is not implemented in MJWarp batch rendering."
            )

        self._world_ids_gpu = wp.array(self.world_ids, dtype=wp.int32)
        self._enabled_cam_ids_gpu = wp.array(
            [self._cameras_names2id[n] for n in self.enabled_cam_names], dtype=wp.int32
        )
        cam_mask = [
            self._cameras_id2name[cid] in self.enabled_cam_names
            for cid in range(self.mj_model.ncam)
        ]

        # Create batch rendering context
        self._rendering_context = mjw.create_render_context(
            mjm=self.mj_model,
            nworld=self._n_worlds_total,
            cam_active=cam_mask,
            cam_res=self.camera_res[::-1],  # MJWarp expects (W, H); we use (H, W)
            **kwargs,
        )

        # Remove normal MjRenderer inherited from CPU Renderer
        self.scene_option = None
        self.mj_renderer = None

    def _render_impl(self, mjw_data: mjw.Data) -> np.ndarray | wp.array:
        mjw.refit_bvh(self.mjw_model, mjw_data, self._rendering_context)
        mjw.render(self.mjw_model, mjw_data, self._rendering_context)
        rgb_out = wp.zeros(self._buf_dim_per_frame, dtype=wp.vec3f)
        get_rgb_selected_worlds_and_cameras(
            self._rendering_context,
            self._world_ids_gpu,
            self._enabled_cam_ids_gpu,
            rgb_out,
        )
        return rgb_out

    def _fetch_frames_to_cpu_impl(
        self, world_id_among_rendered: int, cam_id_among_rendered: int
    ) -> list[np.ndarray]:
        frames = []
        for frame_buffer in self._frames:
            frame = frame_buffer[world_id_among_rendered, cam_id_among_rendered, :, :]
            frame = (frame * 255.0).numpy().astype(np.uint8)
            frames.append(frame)
        return frames

    @override
    def close(self):
        return  # nothing to do since we are not using a mj.Renderer context

    @staticmethod
    def _is_scene_option_default(scene_option: mj.MjvOption) -> bool:
        default_option = mj.MjvOption()
        mj.mjv_defaultOption(default_option)
        return scene_option == default_option


class WarpCPURenderer(_BaseWarpRenderer):
    """CPU-side renderer for multi-world MJWarp simulation."""

    def _render_setup_impl(self, **kwargs: Any) -> None:
        self._mj_data_buffer = mj.MjData(self.mj_model)
        # Nothing else to do - just use mjRenderer inherited from CPU Renderer

    def _render_impl(self, mjw_data: mjw.Data) -> np.ndarray | wp.array:
        rendered_images = np.zeros((*self._buf_dim_per_frame, 3), dtype=np.uint8)

        for world_id in self.world_ids:
            wid_among_rendered = self.world_ids.index(world_id)

            # Copy data into CPU MjData struct
            mj.mj_resetData(self.mj_model, self._mj_data_buffer)
            mjw.get_data_into(self._mj_data_buffer, self.mj_model, mjw_data, world_id)

            # Render each enabled camera and store frames
            for cam_name, internal_cam_id in self._cameras_names2id.items():
                cid_among_rendered = self.enabled_cam_names.index(cam_name)

                self.mj_renderer.update_scene(
                    self._mj_data_buffer, internal_cam_id, self.scene_option
                )
                frame = self.mj_renderer.render()

                if self.buffer_frames:
                    rendered_images[wid_among_rendered, cid_among_rendered] = frame

        return rendered_images

    def _fetch_frames_to_cpu_impl(
        self, world_id_among_rendered: int, cam_id_among_rendered: int
    ) -> list[np.ndarray]:
        frames = []
        for rendered_images in self._frames:
            frame = rendered_images[world_id_among_rendered, cam_id_among_rendered, ...]
            frames.append(frame)
        return frames


def modify_world_for_batch_rendering(world: BaseWorld) -> bool:
    """Modify world MJCF model to make it compatible with MJWarp's GPU batch rendering.

    This may reduce texture and lighting realism.

    Modification happens in place on ``world.mjcf_root``. Returns True if any
    modifications were made, False otherwise. Only ``world.mjcf_root`` (the `MjSpec`)
    and ``world.fly_lookup`` (the fly names) are used, so this can be called on any
    object exposing those two attributes -- see `render_trajectories_gpu`, which
    applies it to a spec reconstructed from a saved trajectory.

    Note: these are material/texture/light edits only -- they do not change the joint
    structure, so a model recompiled afterward keeps the same ``qpos`` layout and a
    recorded trajectory stays valid against it.

    Note for developers: Check if anything here can be dropped upon new MJWarp releases.
    """
    is_modified = False

    rgb_role = int(mj.mjtTextureRole.mjTEXROLE_RGB)

    # Strip textures from fly body materials
    # (rendering textures on complex meshes causes MJWarp memory corruption)
    for material in world.mjcf_root.materials:
        # Don't touch things that are not part of a Fly
        if material.name.split("/")[0] not in world.fly_lookup:
            continue
        # Make wings half transparent
        if "wing" in material.name:
            material.rgba[3] = 0.5
        # If material has a texture, remove it to reduce memory use
        texture_name = material.textures[rgb_role]
        if texture_name:
            texture_element = world.mjcf_root.texture(texture_name)
            primary_color_rgb = texture_element.rgb1
            material.textures[rgb_role] = ""
            material.rgba[:3] = primary_color_rgb
            is_modified = True

    # Adjust scale of checker materials (e.g., ground): texrepeat needs to be scaled
    # down by 1000x to get the same pattern - unclear why. Only materials that still
    # reference a texture (e.g. the ground checker) need this.
    for material in world.mjcf_root.materials:
        if material.textures[rgb_role]:
            material.texrepeat = tuple(tr / 1000 for tr in material.texrepeat)
            is_modified = True

    # Add light above each fly explicitly (only until MuJoCo Warp 3.9)
    mujoco_warp_version = tuple(int(x) for x in mjw.__version__.split(".")[:2])
    if mujoco_warp_version < (3, 10):
        for body in world.mjcf_root.bodies:
            if body.name.split("/")[-1] == "c_thorax":
                warnings.warn(f"Adding overhead light for body {body.name}")
                body.add_light(
                    name=body.name.replace("/", "-") + "-overheadlight",
                    mode=mj.mjtCamLight.mjCAMLIGHT_TRACK,
                    targetbody=body.name,
                    pos=(0, 0, 30),
                    dir=(0, 0, -1),
                    type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
                    ambient=(10, 10, 10),
                    diffuse=(10, 10, 10),
                    specular=(0.3, 0.3, 0.3),
                )
                is_modified = True

    return is_modified
