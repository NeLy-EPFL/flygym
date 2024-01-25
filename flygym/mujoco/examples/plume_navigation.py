import numpy as np
import logging
import tempfile
import imageio
import skimage
import shutil
from numba import njit
from typing import Dict, Tuple, List, Optional, Callable
from dm_control import mjcf
from pathlib import Path
from tqdm import trange

from flygym.mujoco.util import load_config
from flygym.mujoco.arena import BaseArena
from flygym.mujoco import Parameters
from flygym.mujoco.examples.turning_controller import HybridTurningNMF


class OdorPlumeArena(BaseArena):
    def __init__(
        self,
        plume_data_path: Optional[Path] = None,
        dimension_scale_factor: float = 1.0,
        plume_simulation_fps: float = 100,
        intensity_scale_factor: float = 1.0,
        friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
        num_sensors: int = 4,
    ):
        self.dimension_scale_factor = dimension_scale_factor
        self.plume_simulation_fps = plume_simulation_fps
        self.intensity_scale_factor = intensity_scale_factor
        self.friction = friction
        self.num_sensors = num_sensors
        self.curr_time = 0
        self._0th_plume_img_path = Path(
            tempfile.mkstemp(prefix="flygym_cdf_cache_", suffix=".png")[1]
        )
        self._plume_update_interval = 1 / plume_simulation_fps
        self._plume_last_update_time = 0

        # Load plume data
        if plume_data_path is None:
            raise NotImplementedError("TODO: download from some URL automatically")
        self.plume_data = np.load(plume_data_path)
        self.plume_grid = self.plume_data["plume"].copy()
        self.arena_size = (
            np.array(self.plume_grid.shape[1:]) * self.dimension_scale_factor
        )
        # self._plume_display_dir = self._write_texture_files(
        #     self.plume_grid, self.arena_size
        # )

        # Set up floor
        self.root_element = mjcf.RootElement()
        zeroth_plume_img = self._plume_grid_to_rgb(self.plume_grid[0], self.arena_size)
        imageio.imwrite(self._0th_plume_img_path, zeroth_plume_img)
        floor_texture = self.root_element.asset.add(
            "texture",
            name="floor_texture",
            type="cube",
            filefront=str(self._0th_plume_img_path),
            width=self.arena_size[0],
            height=self.arena_size[1],
        )
        floor_material = self.root_element.asset.add(
            "material",
            name="floor_material",
            texture=floor_texture,
        )
        self.root_element.worldbody.add(
            "geom",
            name="floor",
            type="box",
            size=(self.arena_size[0] / 2, self.arena_size[1] / 2, 1),
            material=floor_material,
            pos=(self.arena_size[0] / 2, 0, -1),
        )

        # Add birdeye camera
        self.birdeye_cam = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam",
            mode="fixed",
            pos=(30, 0, 300),
            euler=(0, 0, 0),
            fovy=60,
        )

    @staticmethod
    def _plume_grid_to_rgb(
        plume_grid, arena_size, spatial_interpolation_factor=2, max_intensity=0.7
    ):
        plume_image = np.clip(plume_grid, 0, max_intensity)
        plume_image = 255 - 255 * plume_image / max_intensity
        size = np.array(arena_size) * spatial_interpolation_factor
        plume_image = skimage.transform.resize(plume_image, size, order=0)
        plume_image = plume_image.astype(np.uint8)
        # plume_grid = _resample_image_nearest_neighbor(
        #     plume_grid, spatial_interpolation_factor
        # )
        # texture_path = plume_display_dir / f"plume_frame{i:08d}.png"
        # imageio.imwrite(texture_path, plume_grid.astype(np.uint8))
        return plume_image

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle

    def mm_position_to_grid_position(
        self, x_mm: float, y_mm: float
    ) -> Tuple[float, float]:
        x_idx = int((self.arena_size[0] - x_mm) / self.dimension_scale_factor)
        y_idx = int((y_mm + self.arena_size[1] / 2) / self.dimension_scale_factor)
        x_idx = int(min(max(0, x_idx), self.arena_size[0] - 1))
        y_idx = int(min(max(0, y_idx), self.arena_size[1] - 1))
        return x_idx, y_idx

    def get_olfaction(self, antennae_pos: np.ndarray) -> np.ndarray:
        """
        Returns the olfactory input for the given antennae positions
        """
        frame_num = int(self.curr_time * self.plume_simulation_fps)
        assert self.num_sensors == antennae_pos.shape[0]
        intensities = np.zeros((self.odor_dimensions, self.num_sensors))
        for i_sensor in range(self.num_sensors):
            x_mm, y_mm, _ = antennae_pos[i_sensor, :]
            x_idx, y_idx = self.mm_position_to_grid_position(x_mm, y_mm)
            intensities[0, i_sensor] = self.plume_grid[frame_num, x_idx, y_idx]
        return intensities * self.intensity_scale_factor

    @property
    def odor_dimensions(self) -> int:
        return 1

    def step(self, dt: float, physics: mjcf.Physics = None, *args, **kwargs) -> None:
        self.curr_time += dt
        if self._plume_last_update_time + self._plume_update_interval < self.curr_time:
            # Update plume display
            self._plume_last_update_time = self.curr_time
            frame_num = int(self.curr_time * self.plume_simulation_fps)
            texture_img = self._plume_grid_to_rgb(
                self.plume_grid[frame_num, :, :], self.arena_size
            )
            # texture_img_rgb = np.repeat(texture_img[:, :, np.newaxis], 3, axis=2)
            # texture_img_6views = np.zeros((texture_img.shape[0] * 6, texture_img.shape[1], 3))
            # row_min = texture_img.shape[0] * 4
            # row_max = texture_img.shape[0] * 5
            # texture_img_6views[row_min:row_max, :, :] = texture_img_rgb
            # physics.model.tex(0).rgb = texture_img_6views[:, :, 0]
            
            
            
            texture_img_rgb = np.repeat(texture_img[:, :, np.newaxis], 3, axis=2)
            print(texture_img.shape, texture_img_rgb.shape)
            # physics.model.tex(0).rgb = texture_img_rgb
            
            cached_width = physics.named.model.tex_width["floor_texture"]
            cached_height = physics.named.model.tex_height["floor_texture"]
            cached_addr = physics.named.model.tex_adr["floor_texture"]
            cached_rgb = np.array(physics.model.tex_rgb)
            cached_rgb_6views_sel = cached_rgb[
                cached_addr : cached_addr + cached_width * cached_height * 3
            ]
            cached_rgb_6views_sel = cached_rgb_6views_sel.reshape((cached_height, cached_width, 3))
            row_min = int(cached_height / 6) * 4
            row_max = int(cached_height / 6) * 5
            cached_rgb_6views_sel[row_min:row_max, :, :] = texture_img[:, :, np.newaxis]
            # # physics.named.model.tex_rgb = cached_rgb
            # physics.model.tex(0).rgb = cached_rgb_6views_sel[row_min:row_max, :, :]
            
            mat_texid = physics.named.model.mat_texid["floor_material"]
            num_textures = physics.model.ntex
            physics.named.model.mat_texid["floor_material"] = num_textures - 1 if mat_texid == 0 else 0
            physics.named.model.mat_texid["floor_material"] = mat_texid
            # mat_texid = phy
            # import matplotlib.pyplot as plt
            # plt.imshow(physics.model.tex_rgb[
            #     cached_addr : cached_addr + cached_width * cached_height * 3
            # ].reshape((cached_height, cached_width, 3)))
            # plt.show()
            # texture_id = self._texture_id_lookup[frame_num]
            # physics.named.model.mat_texid["floor_material"] = texture_id


if __name__ == "__main__":
    arena = OdorPlumeArena(
        Path("/home/sibwang/Projects/flygym/outputs/complex_plume/plume.npy.npz")
    )

    # Define the fly
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]
    sim_params = Parameters(
        timestep=1e-4,
        render_mode="saved",
        render_playspeed=0.5,
        render_window_size=(800, 608),
        enable_olfaction=True,
        enable_adhesion=True,
        draw_adhesion=False,
        render_camera="birdeye_cam",
    )
    sim = HybridTurningNMF(
        sim_params=sim_params,
        arena=arena,
        spawn_pos=(30, 0, 0.2),
        contact_sensor_placements=contact_sensor_placements,
    )
    
# sim.physics.model.ntex
# 12
# sim.physics.model.ntexdata
# 23769000
# sim.physics.named.model.tex_rgb.shape
# (23769000,)
# sim.physics.model.tex_rgb.shape
# (23769000,)
# sim.physics.model.tex_adr.shape
# (12,)

    # Run the simulation
    attractive_gain = -500
    aversive_gain = 80
    run_time = 1

    obs_hist = []
    odor_history = []
    obs, _ = sim.reset()
    for i in trange(int(run_time / sim_params.timestep)):
        obs, _, _, _, _ = sim.step(np.array([1, 1]))
        rendered_img = sim.render()
        if rendered_img is not None:
            import cv2
            cv2.imshow("rendered_img", rendered_img)
            cv2.waitKey(1)
            
        obs_hist.append(obs)

    sim.save_video("./outputs/plume_navigation.mp4")
    # import pickle
    # with open("./outputs/obs_hist.pkl", "wb") as f:
    #     pickle.dump(obs_hist, f)

    # odor_history = [x["odor_intensity"].mean() for x in obs_hist]
    # import matplotlib.pyplot as plt
    # plt.plot(odor_history)
    # plt.show()


# import numpy as np
# import logging
# import tempfile
# import imageio
# import skimage
# import shutil
# from numba import njit
# from typing import Dict, Tuple, List, Optional, Callable
# from dm_control import mjcf
# from pathlib import Path
# from tqdm import trange

# from flygym.mujoco.util import load_config
# from flygym.mujoco.arena import BaseArena
# from flygym.mujoco import Parameters
# from flygym.mujoco.examples.turning_controller import HybridTurningNMF


# class OdorPlumeArena(BaseArena):
#     def __init__(
#         self,
#         plume_data_path: Optional[Path] = None,
#         dimension_scale_factor: float = 1.0,
#         plume_simulation_fps: float = 100,
#         intensity_scale_factor: float = 1.0,
#         friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
#         num_sensors: int = 4,
#     ):
#         self.dimension_scale_factor = dimension_scale_factor
#         self.plume_simulation_fps = plume_simulation_fps
#         self.intensity_scale_factor = intensity_scale_factor
#         self.friction = friction
#         self.num_sensors = num_sensors
#         self.curr_time = 0
#         self._plume_display_dir = None
#         self._plume_update_interval = 1 / plume_simulation_fps
#         self._plume_last_update_time = 0

#         # Load plume data
#         if plume_data_path is None:
#             raise NotImplementedError("TODO: download from some URL automatically")
#         self.plume_data = np.load(plume_data_path)
#         self.plume_grid = self.plume_data["plume"].copy()[:30, :, :]
#         self.arena_size = (
#             np.array(self.plume_grid.shape[1:]) * self.dimension_scale_factor
#         )
#         self._plume_display_dir = self._write_texture_files(
#             self.plume_grid, self.arena_size
#         )

#         # Set up floor
#         self.root_element = mjcf.RootElement()
#         floor_textures = []
#         for i in range(self.plume_grid.shape[0]):
#             texture = self.root_element.asset.add(
#                 "texture",
#                 name=f"floor_texture_{i:08d}",
#                 type="cube",
#                 filefront=str(self._plume_display_dir / f"plume_frame{i:08d}.png"),
#                 width=self.arena_size[0],
#                 height=self.arena_size[1],
#             )
#             floor_textures.append(texture)
#         self._texture_id_lookup = {
#             int(texture.name.split("_")[-1]): i
#             for i, texture in enumerate(self.root_element.asset.texture)
#         }
#         floor_material = self.root_element.asset.add(
#             "material",
#             name="floor_material",
#             texture=floor_textures[0],
#         )
#         self.root_element.worldbody.add(
#             "geom",
#             name="floor",
#             type="box",
#             size=(self.arena_size[0] / 2, self.arena_size[1] / 2, 1),
#             material=floor_material,
#             pos=(self.arena_size[0] / 2, 0, -1),
#         )

#         # Add birdeye camera
#         self.birdeye_cam = self.root_element.worldbody.add(
#             "camera",
#             name="birdeye_cam",
#             mode="fixed",
#             pos=(30, 0, 300),
#             euler=(0, 0, 0),
#             fovy=60,
#         )

#     @staticmethod
#     def _write_texture_files(
#         plume_grid: np.ndarray,
#         arena_size: Tuple[int, int],
#         max_intensity: float = 0.7,
#         spatial_interpolation_factor: int = 1,
#     ) -> Path:
#         plume_display_dir = Path(tempfile.gettempdir()) / "flygym_cfd_cache"
#         plume_display_dir.mkdir(exist_ok=True, parents=True)
#         assert plume_grid.ndim == 3
#         assert plume_grid.shape[0] < 1e9
#         for i in trange(plume_grid.shape[0], desc="Creating plume display buffer"):
#             texture = plume_grid[i, :, :]
#             # texture = np.flipud(texture)
#             # texture = np.rot90(texture)
#             texture = np.clip(texture, 0, max_intensity)
#             texture = 255 - 255 * texture / max_intensity
#             size = np.array(arena_size) * spatial_interpolation_factor
#             texture = skimage.transform.resize(texture, size, order=0)
#             # texture = _resample_image_nearest_neighbor(
#             #     texture, spatial_interpolation_factor
#             # )
#             texture_path = plume_display_dir / f"plume_frame{i:08d}.png"
#             imageio.imwrite(texture_path, texture.astype(np.uint8))
#         return plume_display_dir

#     def __del__(self):
#         # remove the temporary directory
#         # if self._plume_display_dir is not None:
#         #     shutil.rmtree(self._plume_display_dir)
#         pass

#     def get_spawn_position(
#         self, rel_pos: np.ndarray, rel_angle: np.ndarray
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         return rel_pos, rel_angle

#     def mm_position_to_grid_position(
#         self, x_mm: float, y_mm: float
#     ) -> Tuple[float, float]:
#         x_idx = int((self.arena_size[0] - x_mm) / self.dimension_scale_factor)
#         y_idx = int((y_mm + self.arena_size[1] / 2) / self.dimension_scale_factor)
#         x_idx = int(min(max(0, x_idx), self.arena_size[0] - 1))
#         y_idx = int(min(max(0, y_idx), self.arena_size[1] - 1))
#         return x_idx, y_idx

#     def get_olfaction(self, antennae_pos: np.ndarray) -> np.ndarray:
#         """
#         Returns the olfactory input for the given antennae positions
#         """
#         frame_num = int(self.curr_time * self.plume_simulation_fps)
#         assert self.num_sensors == antennae_pos.shape[0]
#         intensities = np.zeros((self.odor_dimensions, self.num_sensors))
#         for i_sensor in range(self.num_sensors):
#             x_mm, y_mm, _ = antennae_pos[i_sensor, :]
#             x_idx, y_idx = self.mm_position_to_grid_position(x_mm, y_mm)
#             intensities[0, i_sensor] = self.plume_grid[frame_num, x_idx, y_idx]
#         return intensities * self.intensity_scale_factor

#     @property
#     def odor_dimensions(self) -> int:
#         return 1

#     def step(self, dt: float, physics: mjcf.Physics = None, *args, **kwargs) -> None:
#         self.curr_time += dt
#         if self._plume_last_update_time + self._plume_update_interval < self.curr_time:
#             # Update plume display
#             self._plume_last_update_time = self.curr_time
#             frame_num = int(self.curr_time * self.plume_simulation_fps)
#             texture_id = self._texture_id_lookup[frame_num]
#             physics.named.model.mat_texid["floor_material"] = texture_id


# @njit(parallel=True)
# def _resample_image_nearest_neighbor(image, resize_factor):
#     out = np.empty((image.shape[0] * resize_factor, image.shape[1] * resize_factor))
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             for k in range(resize_factor):
#                 for l in range(resize_factor):
#                     out[i * resize_factor + k, j * resize_factor + l] = image[i, j]
#     return out


# if __name__ == "__main__":
#     arena = OdorPlumeArena(
#         Path("/home/sibwang/Projects/flygym/outputs/complex_plume/plume.npy.npz")
#     )

#     # Define the fly
#     contact_sensor_placements = [
#         f"{leg}{segment}"
#         for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
#         for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
#     ]
#     sim_params = Parameters(
#         timestep=1e-4,
#         render_mode="saved",
#         render_playspeed=0.5,
#         render_window_size=(800, 608),
#         enable_olfaction=True,
#         enable_adhesion=True,
#         draw_adhesion=False,
#         render_camera="birdeye_cam",
#     )
#     sim = HybridTurningNMF(
#         sim_params=sim_params,
#         arena=arena,
#         spawn_pos=(30, 0, 0.2),
#         contact_sensor_placements=contact_sensor_placements,
#     )

#     # Run the simulation
#     attractive_gain = -500
#     aversive_gain = 80
#     run_time = 1

#     obs_hist = []
#     odor_history = []
#     obs, _ = sim.reset()
#     for i in trange(int(run_time / sim_params.timestep)):
#         obs, _, _, _, _ = sim.step(np.array([1, 1]))
#         rendered_img = sim.render()
#         obs_hist.append(obs)

#     sim.save_video("./outputs/plume_navigation.mp4")
#     # import pickle
#     # with open("./outputs/obs_hist.pkl", "wb") as f:
#     #     pickle.dump(obs_hist, f)

#     # odor_history = [x["odor_intensity"].mean() for x in obs_hist]
#     # import matplotlib.pyplot as plt
#     # plt.plot(odor_history)
#     # plt.show()
