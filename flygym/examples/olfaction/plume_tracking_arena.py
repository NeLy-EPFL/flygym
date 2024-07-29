import numpy as np
import h5py
from dm_control.mujoco import Camera
from dm_control import mjcf
from pathlib import Path

from flygym.arena import BaseArena
from flygym import Simulation


class OdorPlumeArena(BaseArena):
    """
    This Arena class provides an interface to the separately simulated
    odor plume. The plume simulation is stored in an HDF5 file. In this
    class, we implement logics that calculate the intensity of the odor
    at the fly's location at the correct time.
    """

    def __init__(
        self,
        plume_data_path: Path,
        dimension_scale_factor: float = 0.5,
        plume_simulation_fps: float = 200,
        intensity_scale_factor: float = 1.0,
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
        num_sensors: int = 4,
    ):
        """
        Parameters
        ----------
        plume_data_path : Path
            Path to the HDF5 file containing the plume simulation data.
        dimension_scale_factor : float, optional
            Scaling factor for the plume simulation grid. Each cell in the
            plume grid is this many millimeters in the simulation. By
            default 0.5.
        plume_simulation_fps : float, optional
            Frame rate of the plume simulation. Each frame in the plume
            dataset is ``1 / plume_simulation_fps`` seconds in the physics
            simulation. By default 200.
        intensity_scale_factor : float, optional
            Scaling factor for the intensity of the odor. By default 1.0.
        friction : tuple[float, float, float], optional
            Friction parameters for the floor geom. By default (1, 0.005,
            0.0001).
        num_sensors : int, optional
            Number of olfactory sensors on the fly. By default 4.
        """
        super().__init__()

        self.dimension_scale_factor = dimension_scale_factor
        self.plume_simulation_fps = plume_simulation_fps
        self.intensity_scale_factor = intensity_scale_factor
        self.friction = friction
        self.num_sensors = num_sensors
        self.curr_time = 0
        self.plume_update_interval = 1 / plume_simulation_fps

        # Load plume data
        self.plume_dataset = h5py.File(plume_data_path, "r")
        self.plume_grid = self.plume_dataset["plume"]
        self.arena_size = (
            np.array(self.plume_grid.shape[1:][::-1]) * dimension_scale_factor
        )

        # Set up floor
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
            size=(self.arena_size[0] / 2, self.arena_size[1], 1),
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

    def get_position_mapping(
        self, sim: Simulation, camera_id: str = "birdeye_cam"
    ) -> np.ndarray:
        """Get the display location (row-col coordinates) of each pixel on
        the fluid dynamics simulation.

        Parameters
        ----------
        sim : Simulation
            Simulation simulation object.
        camera_id : str, optional
            Camera to build position mapping for, by default "birdeye_cam"

        Returns
        -------
        pos_display: np.ndarray
            Array of shape (n_row_pxls_plume, n_col_pxls_plume, 2)
            containing the row-col coordinates of each plume simulation
            cell on the **display** image (in pixels).
        pos_physical: np.ndarray
            Array of shape (n_row_pxls_plume, n_col_pxls_plume, 2)
            containing the row-col coordinates of each plume simulation
            cell on the **physical** simulated grid (in mm). This is a
            regular lattice grid marking the physical position of the
            *centers* of the fluid simulation cells.
        """
        birdeye_cam_dm_control_obj = Camera(
            sim.physics,
            camera_id=camera_id,
            width=sim.cameras[0].window_size[0],
            height=sim.cameras[0].window_size[1],
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
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def __del__(self):
        self.plume_dataset.close()
