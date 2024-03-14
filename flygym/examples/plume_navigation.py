import numpy as np
import cv2
from enum import Enum
from scipy.interpolate import LinearNDInterpolator
from dm_control.mujoco import Camera
from numba import njit, prange
from typing import Dict, Tuple, List, Optional, Callable, Union
from dm_control import mjcf
from pathlib import Path
from tqdm import trange
from flygym.mujoco.core import Parameters
from flygym.mujoco.preprogrammed import all_leg_dofs, all_tarsi_links
from flygym.mujoco.state.kinematic_pose import KinematicPose

from flygym.mujoco.util import load_config, get_data_path
from flygym.mujoco.arena import BaseArena
from flygym.mujoco import Parameters, NeuroMechFly
from flygym.mujoco.examples.turning_controller import HybridTurningNMF


class OdorPlumeArena(BaseArena):
    def __init__(
        self,
        plume_data_path: Optional[Path] = None,
        dimension_scale_factor: float = 0.5,
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
        self.plume_update_interval = 1 / plume_simulation_fps

        # Load plume data
        if plume_data_path is None:
            raise NotImplementedError("TODO: download from some URL automatically")
        self.plume_data = np.load(plume_data_path)
        self.plume_grid = self.plume_data["plume"].copy()
        print(self.plume_grid.shape, self.dimension_scale_factor)
        self.arena_size = (
            np.array(self.plume_grid.shape[1:])[::-1] * self.dimension_scale_factor
        )

        # Set up floor
        self.root_element = mjcf.RootElement()
        floor_material = self.root_element.asset.add(
            "material",
            name="floor_material",
            reflectance=0.0,
            shininess=0.0,
            specular=0.0,
            rgba=[0.6, 0.6, 0.6, 1],
        )
        self.root_element.worldbody.add(
            "geom",
            name="floor",
            type="box",
            size=(self.arena_size[0] / 2, self.arena_size[1] / 2, 1),
            pos=(self.arena_size[0] / 2, self.arena_size[1] / 2, -1),
            material=floor_material,
        )

        # Add birdeye camera
        self.birdeye_cam = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam",
            mode="fixed",
            pos=(
                0.50 * self.arena_size[0],
                0.15 * self.arena_size[1],
                1.00 * self.arena_size[1],
            ),
            euler=(np.deg2rad(15), 0, 0),
            fovy=60,
        )

        # # Add second camera
        # self.birdeye_cam = self.root_element.worldbody.add(
        #     "camera",
        #     name="birdeye_cam",
        #     mode="fixed",
        #     pos=(158, 93, 20),
        #     euler=(0, 0, 0),
        #     fovy=45,
        # )

    def get_position_mapping(
        self, sim: NeuroMechFly, camera_id: str = "birdeye_cam"
    ) -> np.ndarray:
        """Get the display location (row-col coordinates) of each pixel on
        the fluid dynamics simulation.

        Parameters
        ----------
        sim : NeuroMechFly
            NeuroMechFly simulation object.
        camera_id : str, optional
            Camera to build position mapping for, by default "birdeye_cam"

        Returns
        -------
        pos_display: np.ndarray
            Array of shape (n_row_pxls_plume, n_row_pxls_plume, 2)
            containing the row-col coordinates of each plume simulation
            cell on the **display** image (in pixels).
        pos_physical: np.ndarray
            Array of shape (n_row_pxls_plume, n_row_pxls_plume, 2)
            containing the row-col coordinates of each plume simulation
            cell on the **physical** simulated grid (in mm). This is a
            regular lattice grid marking the physical position of the
            *centers* of the fluid simulation cells.
        """
        birdeye_cam_dm_control_obj = Camera(
            sim.physics,
            camera_id=camera_id,
            width=sim.sim_params.render_window_size[0],
            height=sim.sim_params.render_window_size[1],
        )
        camera_matrix = birdeye_cam_dm_control_obj.matrix
        xs_physical, ys_physical = np.meshgrid(
            np.arange(self.arena_size[0]) + 0.5,
            np.arange(self.arena_size[1]) + 0.5,
        )
        xyz1_vecs = np.ones((xs_physical.size, 4))
        xyz1_vecs[:, 0] = xs_physical.flatten()
        xyz1_vecs[:, 1] = ys_physical.flatten()
        xyz1_vecs[:, 2] = 0
        pos_physical = xyz1_vecs[:, :2].reshape(*xs_physical.shape, 2)
        xs_display, ys_display, display_scale = camera_matrix @ xyz1_vecs.T
        xs_display /= display_scale
        ys_display /= display_scale
        pos_display = np.vstack((xs_display, ys_display))
        pos_display = pos_display.T.reshape(*xs_physical.shape, 2)
        return pos_display, pos_physical

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle

    def get_olfaction(self, antennae_pos: np.ndarray) -> np.ndarray:
        """
        Returns the olfactory input for the given antennae positions. If
        the fly is outside the plume simulation grid, returns np.nan.
        """
        frame_num = int(self.curr_time * self.plume_simulation_fps)
        assert self.num_sensors == antennae_pos.shape[0]
        intensities = np.zeros((self.odor_dimensions, self.num_sensors))
        for i_sensor in range(self.num_sensors):
            x_mm, y_mm, _ = antennae_pos[i_sensor, :]
            x_idx = int(x_mm / self.dimension_scale_factor)
            y_idx = int(y_mm / self.dimension_scale_factor)
            if (
                x_idx < 0
                or y_idx < 0
                or x_idx >= self.plume_grid.shape[2]
                or y_idx >= self.plume_grid.shape[1]
            ):
                intensities[0, i_sensor] = np.nan
            else:
                intensities[0, i_sensor] = self.plume_grid[frame_num, y_idx, x_idx]
        return intensities * self.intensity_scale_factor

    @property
    def odor_dimensions(self) -> int:
        return 1

    def step(self, dt: float, physics: mjcf.Physics = None, *args, **kwargs) -> None:
        self.curr_time += dt


class WalkingState(Enum):
    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STOP = 3


class TurningObjective(Enum):
    UPWIND = 0
    DOWNWIND = 1


class PlumeNavigationTask(HybridTurningNMF):
    def __init__(
        self,
        render_plume_alpha: float = 0.75,
        intensity_display_vmax: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.arena = kwargs["arena"]
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
            np.arange(self.sim_params.render_window_size[0]),
            np.arange(self.sim_params.render_window_size[1]),
        )
        pos_display_all = np.vstack([xs_display.flatten(), ys_display.flatten()]).T
        pos_physical_all = interp(pos_display_all)
        pos_physical_all = pos_physical_all.reshape(
            *self.sim_params.render_window_size[::-1], 2
        )
        grid_idx_all = pos_physical_all / self.arena.dimension_scale_factor
        grid_idx_all[np.isnan(grid_idx_all)] = -1
        # self.grid_idx_all has the shape (cam_nrows, cam_ncols, 2) and
        # indicates the (x, y) indices of the plume simulation grid cell.
        # When the index is -1, this point on the displayed image is out of
        # the simulated arena.
        self.grid_idx_all = grid_idx_all.astype(np.int16)

    def render(self, *args, **kwargs):
        res = super().render(*args, **kwargs)
        if res is None:
            return  # no image rendered

        # Overlay plume
        time_since_last_update = self.curr_time - self._plume_last_update_time
        update_needed = time_since_last_update > self.arena.plume_update_interval
        if update_needed or self._cached_plume_img is None:
            t_idx = int(self.curr_time * self.arena.plume_simulation_fps)
            self._cached_plume_img = _resample_plume_image(
                self.grid_idx_all, self.arena.plume_grid[t_idx, :, :]
            )
            self._plume_last_update_time = self.curr_time
        plume_img = self._cached_plume_img[:, :, np.newaxis] * self._render_plume_alpha
        plume_img[np.isnan(plume_img)] = 0
        res = np.clip(res - plume_img * 255, 0, 255).astype(np.uint8)

        # Add intensity indicator
        mean_intensity = obs["odor_intensity"].mean()
        mean_intensity_relative = np.clip(
            mean_intensity / self._intensity_display_vmax, 0, 1
        )
        rmin = self.sim_params.render_window_size[1] - 10
        rmax = self.sim_params.render_window_size[1]
        cmin = 0
        cmax = int(self.sim_params.render_window_size[0] * mean_intensity_relative)
        res[rmin:rmax, cmin:cmax] = (255, 0, 0)

        # Replace recorded image with modified one
        self._frames[-1] = res
        return res

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if np.isnan(obs["odor_intensity"]).any():
            truncated = True
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


class PlumeNavigationController:
    def __init__(
        self,
        forward_dn_drive: Tuple[float, float] = (1.0, 1.0),
        left_turn_dn_drive: Tuple[float, float] = (-0.4, 1.2),
        right_turn_dn_drive: Tuple[float, float] = (1.2, -0.4),
        stop_dn_drive: Tuple[float, float] = (0.0, 0.0),
        inter_turn_interval: float = 0.5,
        turn_duration: float = 0.2,
    ) -> None:
        # DN drives
        self.dn_drives = {
            WalkingState.FORWARD: np.array(forward_dn_drive),
            WalkingState.TURN_LEFT: np.array(left_turn_dn_drive),
            WalkingState.TURN_RIGHT: np.array(right_turn_dn_drive),
            WalkingState.STOP: np.array(stop_dn_drive),
        }

        self.inter_turn_interval = inter_turn_interval
        self.turn_duration = turn_duration

        self.current_state = WalkingState.FORWARD
        self.current_state_start_time = 0.0

    def decide_state(
        self, encounter_flag: bool, curr_time: float, fly_heading: np.ndarray
    ):
        # Update integration state
        if self.current_state == WalkingState.STOP:
            # TODO
            ...

        # Forward -> turn transition
        if (
            self.current_state == WalkingState.FORWARD
            and curr_time - self.current_state_start_time > self.inter_turn_interval
        ):
            turn_objective = TurningObjective.UPWIND  # TODO

            if fly_heading[1] >= 0:  # upwind == left turn
                if turn_objective == TurningObjective.UPWIND:
                    self.current_state = WalkingState.TURN_LEFT
                else:
                    self.current_state = WalkingState.TURN_RIGHT
            else:
                if turn_objective == TurningObjective.UPWIND:
                    self.current_state = WalkingState.TURN_RIGHT
                else:
                    self.current_state = WalkingState.TURN_LEFT
            self.current_state_start_time = curr_time

        # Forward -> stop transition
        # TODO
        ...

        # Turn -> forward transition
        if (
            self.current_state in (WalkingState.TURN_LEFT, WalkingState.TURN_RIGHT)
            and curr_time - self.current_state_start_time > self.turn_duration
        ):
            self.current_state = WalkingState.FORWARD
            self.current_state_start_time = curr_time

        # Stop -> forward transition
        # TODO
        ...

        return self.current_state, self.dn_drives[self.current_state]


def get_walking_icons():
    icons_dir = get_data_path("flygym", "data") / "etc/locomotion_icons"
    icons = {}
    for key in ["forward", "left", "right", "stop"]:
        icon_path = icons_dir / f"{key}.png"
        icons[key] = cv2.imread(str(icon_path), cv2.IMREAD_UNCHANGED)
    return {
        WalkingState.FORWARD: icons["forward"],
        WalkingState.TURN_LEFT: icons["left"],
        WalkingState.TURN_RIGHT: icons["right"],
        WalkingState.STOP: icons["stop"],
    }


def add_icon_to_image(image, icon):
    sel = image[: icon.shape[0], -icon.shape[1] :, :]
    mask = icon[:, :, 3] > 0
    sel[mask] = icon[mask, :3]


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
        render_window_size=(720, 480),
        enable_olfaction=True,
        enable_adhesion=True,
        draw_adhesion=False,
        render_camera="birdeye_cam",
        render_timestamp_text=True,
    )
    sim = PlumeNavigationTask(
        sim_params=sim_params,
        arena=arena,
        spawn_pos=(arena.arena_size[0] * 0.75, arena.arena_size[1] / 2, 0.2),
        spawn_orientation=(0, 0, -np.pi / 2),
        contact_sensor_placements=contact_sensor_placements,
    )
    controller = PlumeNavigationController()
    icons = get_walking_icons()
    encounter_threshold = 0.05

    # Run the simulation
    run_time = 5

    obs_hist = []
    odor_history = []
    obs, _ = sim.reset()
    for i in trange(int(run_time / sim_params.timestep)):
        obs = sim.get_observation()
        walking_state, dn_drive = controller.decide_state(
            encounter_flag=obs["odor_intensity"].max() > encounter_threshold,
            curr_time=sim.curr_time,
            fly_heading=obs["fly_orientation"],
        )
        obs, reward, terminated, truncated, info = sim.step(dn_drive)
        if terminated or truncated:
            break
        rendered_img = sim.render()
        if rendered_img is not None:
            add_icon_to_image(rendered_img, icons[walking_state])
            cv2.imshow("rendered_img", rendered_img[:, :, ::-1])
            cv2.waitKey(1)

        obs_hist.append(obs)

    # sim.save_video("./outputs/plume_navigation.mp4")
