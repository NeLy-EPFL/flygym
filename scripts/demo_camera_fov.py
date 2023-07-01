import numpy as np
import matplotlib.pyplot as plt
from dm_control import mjcf
from typing import Tuple

from flygym.arena import BaseArena
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo, MuJoCoParameters
from flygym.state import stretched_pose
from flygym.util.config import all_leg_dofs


class FovCalibrationArena(BaseArena):
    def __init__(
        self,
        size: Tuple[float, float] = (50, 50),
        friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
    ):
        self.root_element = mjcf.RootElement()
        ground_size = [*size, 1]
        chequered = self.root_element.asset.add(
            "texture",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=(0.2, 0.3, 0.4),
            rgb2=(0.3, 0.4, 0.5),
        )
        grid = self.root_element.asset.add(
            "material",
            name="grid",
            texture=chequered,
            texrepeat=(10, 10),
            reflectance=0.1,
        )
        self.root_element.worldbody.add(
            "geom",
            type="plane",
            name="ground",
            material=grid,
            size=ground_size,
            friction=friction,
        )

        # Add FOV limit markers
        left_points = [(30.3843, -4.1757), (14.2417, 20.0799), (-20.4932, 21.5132)]
        colors = [
            # left eye: anterior up to red, posterior down to blue, green in the middle
            [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)],
            # right eye: ant up to yellow, post down to cyan, magenta in the middle
            [(1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1)],
        ]
        for i in range(3):
            x, left_y = left_points[i]
            self.root_element.worldbody.add(
                "geom",
                type="cylinder",
                size=(1, 50),
                pos=(x, left_y, 25),
                rgba=colors[0][i],
            )
            self.root_element.worldbody.add(
                "geom",
                type="cylinder",
                size=(1, 50),
                pos=(x, -left_y, 25),
                rgba=colors[1][i],
            )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle


run_time = 1e-4
sim_params = MuJoCoParameters(
    timestep=1e-4,
    render_mode="saved",
    render_playspeed=0.1,
    render_camera="Animat/camera_LEye",
    # render_camera="Animat/camera_left_top_zoomout",
)
arena = FovCalibrationArena()
nmf = NeuroMechFlyMuJoCo(
    sim_params=sim_params,
    arena=arena,
    init_pose=stretched_pose,
    actuated_joints=all_leg_dofs,
)
nmf.render()
image = nmf._frames[0]
plt.imshow(image)
plt.show()
