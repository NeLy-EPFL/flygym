import numpy as np
from scipy.interpolate import LinearNDInterpolator
from numba import njit, prange
from flygym.examples.turning_controller import HybridTurningNMF
from flygym.examples.plume_tracking.arena import OdorPlumeArena


class PlumeNavigationTask(HybridTurningNMF):
    def __init__(
        self,
        render_plume_alpha: float = 0.75,
        intensity_display_vmax: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.arena: OdorPlumeArena = kwargs["arena"]
        self._plume_last_update_time = -np.inf
        self._cached_plume_img = None
        self._render_plume_alpha = render_plume_alpha
        self._intensity_display_vmax = intensity_display_vmax

        # Find out where on the displayed images the plume simulation grid
        # should be overlaid. In other words, interpolate the mapping from
        # displayed pixel position to simulated physical position.
        pos_display_sample, pos_physical_sample = self.arena.get_position_mapping(
            self, camera_id="birdeye_cam"
        )
        pos_display_sample = pos_display_sample.reshape(-1, 2)
        pos_physical_sample = pos_physical_sample.reshape(-1, 2)
        interp = LinearNDInterpolator(
            pos_display_sample, pos_physical_sample, fill_value=np.nan
        )
        xs_display, ys_display = np.meshgrid(
            np.arange(self.cameras[0].window_size[0]),
            np.arange(self.cameras[0].window_size[1]),
        )
        pos_display_all = np.vstack([xs_display.flatten(), ys_display.flatten()]).T
        pos_physical_all = interp(pos_display_all)
        pos_physical_all = pos_physical_all.reshape(
            *self.cameras[0].window_size[::-1], 2
        )
        grid_idx_all = pos_physical_all / self.arena.dimension_scale_factor
        grid_idx_all[np.isnan(grid_idx_all)] = -1
        # self.grid_idx_all has the shape (cam_nrows, cam_ncols, 2) and
        # indicates the (x, y) indices of the plume simulation grid cell.
        # When the index is -1, this point on the displayed image is out of
        # the simulated arena.
        self.grid_idx_all = grid_idx_all.astype(np.int16)

    def render(self, *args, **kwargs):
        rendered_img = super().render(*args, **kwargs)[0]
        if rendered_img is None:
            return [rendered_img]  # no image rendered

        # Overlay plume
        time_since_last_update = self.curr_time - self._plume_last_update_time
        update_needed = time_since_last_update > self.arena.plume_update_interval
        if update_needed or self._cached_plume_img is None:
            t_idx = int(self.curr_time * self.arena.plume_simulation_fps)
            self._cached_plume_img = _resample_plume_image(
                self.grid_idx_all, self.arena.plume_grid[t_idx, :, :].astype(np.float32)
            )
            self._plume_last_update_time = self.curr_time
        plume_img = self._cached_plume_img[:, :, np.newaxis] * self._render_plume_alpha
        plume_img[np.isnan(plume_img)] = 0
        rendered_img = np.clip(rendered_img - plume_img * 255, 0, 255).astype(np.uint8)

        # Add intensity indicator
        mean_intensity = self.get_observation()["odor_intensity"].mean()
        mean_intensity_relative = np.clip(
            mean_intensity / self._intensity_display_vmax, 0, 1
        )
        rmin = self.cameras[0].window_size[1] - 10
        rmax = self.cameras[0].window_size[1]
        cmin = 0
        cmax = int(self.cameras[0].window_size[0] * mean_intensity_relative)
        rendered_img[rmin:rmax, cmin:cmax] = (255, 0, 0)

        # Replace recorded image with modified one
        self.cameras[0]._frames[-1] = rendered_img
        return [rendered_img]

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if np.isnan(obs["odor_intensity"]).any():
            truncated = True
        # if self.arena.is_in_target(*obs["fly"][0, :2]):
        #     terminated = True
        #     reward = 1
        return obs, reward, terminated, truncated, info


@njit(parallel=True)
def _resample_plume_image(grid_idx_all, plume_grid):
    plume_img = np.zeros(grid_idx_all.shape[:2])
    for i in prange(grid_idx_all.shape[0]):
        for j in prange(grid_idx_all.shape[1]):
            x_idx = grid_idx_all[i, j, 0]
            y_idx = grid_idx_all[i, j, 1]
            if x_idx != -1:
                plume_img[i, j] = plume_grid[y_idx, x_idx]
    return plume_img
