import numpy as np
from scipy.interpolate import LinearNDInterpolator
from numba import njit, prange
from flygym import Fly
from flygym.examples.locomotion import HybridTurningController
from flygym.examples.olfaction import OdorPlumeArena

from dm_control.mujoco import Camera as DMCamera


class PlumeNavigationTask(HybridTurningController):
    """
    A wrapper around the ``HybridTurningController`` that implements logics
    and utilities related to plume tracking such as overlaying the plume on
    the rendered images. It also checks if the fly is within the plume
    simulation grid and truncates the simulation accordingly.

    Notes
    -----
    Please refer to the `"MPD Task Specifications" page
    <https://neuromechfly.org/api_ref/mdp_specs.html#plume-tracking-task-plumenavigationtask>`_
    of the API references for the detailed specifications of the action
    space, the observation space, the reward, the "terminated" and
    "truncated" flags, and the "info" dictionary.
    """

    def __init__(
        self,
        fly: Fly,
        arena: OdorPlumeArena,
        render_plume_alpha: float = 0.75,
        intensity_display_vmax: float = 1.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        fly: Fly
            The fly object to be used. See
            ``flygym.example.locomotion.HybridTurningController``.
        arena: OdorPlumeArena
            The odor plume arena object to be used. Initialize it before
            creating the ``PlumeNavigationTask`` object.
        render_plume_alpha : float
            The transparency of the plume overlay on the rendered images.
        intensity_display_vmax : float
            The maximum intensity value to be displayed on the rendered
            images.
        """
        super().__init__(fly=fly, arena=arena, **kwargs)
        self.arena = arena
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

        self.focus_cam = self.cameras[1] if len(self.cameras) > 1 else None
        if self.focus_cam is not None:
            self.fc_width, self.fc_height = self.focus_cam.window_size
            pixel_meshgrid = np.meshgrid(
                np.arange(self.fc_width), np.arange(self.fc_height)
            )
            self.pixel_idxs = np.stack(
                [pixel_meshgrid[0].flatten(), pixel_meshgrid[1].flatten()], axis=1
            )

    def render(self, *args, **kwargs):
        imgs = super().render(*args, **kwargs)
        rendered_img = imgs[0]

        if rendered_img is None:
            return [None, None]  # no image rendered

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

        # project the plume on the focused_img
        if self.focus_cam is not None:
            focus_img = imgs[1]
            plume_focus = self.overlay_focused_plume(focus_img, t_idx)
            # overlay plume focus on the focused image
            plume_focus = plume_focus[:, :, np.newaxis] * self._render_plume_alpha
            plume_focus[np.isnan(plume_focus)] = 0
            focus_img = np.clip(focus_img - plume_focus * 255, 0, 255).astype(np.uint8)

        else:
            focus_img = None

        return [rendered_img, focus_img]

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if np.isnan(obs["odor_intensity"]).any():
            truncated = True
        return obs, reward, terminated, truncated, info

    def overlay_focused_plume(self, focus_img, t_idx):
        # get the camera field of view in mm
        fc_fov = np.deg2rad(self.physics.model.camera(self.cameras[1].camera_id).fovy)
        fc_pos = self.physics.data.camera(self.cameras[1].camera_id).xpos
        fc_y_fov = np.tan(fc_fov / 2) * fc_pos[2] * 2
        fc_x_fov = fc_y_fov * self.fc_width / self.fc_height

        # get a grid of points in the physical flygym space centered around the fly
        xs_physical_fov = (
            np.arange(
                0, np.ceil(fc_x_fov).astype(int) + 5, self.arena.dimension_scale_factor
            )
            - int(fc_y_fov / 2)
            + fc_pos[0]
            - 3
        )
        ys_physical_fov = (
            np.arange(
                0, np.ceil(fc_y_fov).astype(int) + 5, self.arena.dimension_scale_factor
            )
            - int(fc_y_fov / 2)
            + fc_pos[1]
            - 3
        )

        # get the invalid plume simulation indexes
        invalid_xs = np.logical_or(
            xs_physical_fov / self.arena.dimension_scale_factor < 0,
            xs_physical_fov / self.arena.dimension_scale_factor
            >= self.arena.plume_grid.shape[2],
        )
        invalid_ys = np.logical_or(
            ys_physical_fov / self.arena.dimension_scale_factor < 0,
            ys_physical_fov / self.arena.dimension_scale_factor
            >= self.arena.plume_grid.shape[1],
        )
        # remove them from the xs and ys
        xs_physical_fov = xs_physical_fov[~invalid_xs]
        ys_physical_fov = ys_physical_fov[~invalid_ys]
        # get the plume intensities at the physical points

        xs_physical_fov, ys_physical_fov = np.meshgrid(xs_physical_fov, ys_physical_fov)
        focus_dm_cam = DMCamera(
            self.physics,
            camera_id=self.cameras[1].camera_id,
            width=self.cameras[1].window_size[0],
            height=self.cameras[1].window_size[1],
        )
        camera_matrix = focus_dm_cam.matrix
        xyz1_vecs = np.ones((xs_physical_fov.size, 4))
        xyz1_vecs[:, 0] = xs_physical_fov.flatten()
        xyz1_vecs[:, 1] = ys_physical_fov.flatten()
        xyz1_vecs[:, 2] = 0
        xyz1_vecs = xyz1_vecs
        xs_display, ys_display, display_scale = camera_matrix @ xyz1_vecs.T
        xs_display /= display_scale
        ys_display /= display_scale
        pos_display = np.vstack((xs_display, ys_display))
        pos_display = pos_display.T.reshape(*xs_physical_fov.shape, 2)

        # get the plume intensities at the physical points
        x_plume_idxs = (
            xs_physical_fov.flatten() / self.arena.dimension_scale_factor
        ).astype(int)
        y_plume_idxs = (
            ys_physical_fov.flatten() / self.arena.dimension_scale_factor
        ).astype(int)

        plume_values = self.arena.plume_grid[t_idx][y_plume_idxs, x_plume_idxs]
        interp = LinearNDInterpolator(
            np.stack([xs_display, ys_display], axis=1), plume_values, fill_value=np.nan
        )

        # interp to match display meshgrid
        plume_display = interp(self.pixel_idxs).reshape((self.fc_height, self.fc_width))

        return plume_display


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
