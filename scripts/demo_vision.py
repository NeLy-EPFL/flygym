import numpy as np
import pkg_resources
import pickle
import matplotlib.pyplot as plt
from dm_control import mjcf
from typing import Tuple
from tqdm import trange
from pathlib import Path

from flygym.arena import BaseArena
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo, MuJoCoParameters
from flygym.state import stretched_pose
from flygym.util.config import all_leg_dofs
from flygym.util.vision import visualize_visual_input


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
            rgb1=(0.3, 0.3, 0.3),
            rgb2=(0.4, 0.4, 0.4),
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
        self.friction = friction

        # Add camera
        # <camera name="camera_top_zoomout" class="nmf" mode="fixed"  ipd="0.068" pos="0 0 100" euler="0 0 0" fovy="20"/>
        self.root_element.worldbody.add(
            "camera",
            name="birdseye_cam",
            mode="fixed",
            pos=(0, 0, 300),
            euler=(0, 0, 0),
            fovy=10,
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
                size=(0.5, 50),
                pos=(x, left_y, 25),
                rgba=colors[0][i],
            )
            self.root_element.worldbody.add(
                "geom",
                type="cylinder",
                size=(0.5, 50),
                pos=(x, -left_y, 25),
                rgba=colors[1][i],
            )

        for i in range(36):
            x = np.sin(i * np.pi / 18) * 20
            y = np.cos(i * np.pi / 18) * 20
            self.root_element.worldbody.add(
                "geom",
                type="cylinder",
                size=(0.1, 50),
                pos=(x, y, 25),
                rgba=(1, 1, 1, 0.5),
            )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle


# Initialize simulation
run_time = 1
sim_params = MuJoCoParameters(
    timestep=1e-4,
    render_mode="saved",
    render_playspeed=0.1,
    enable_vision=True,
    render_raw_vision=True,
    render_camera="birdseye_cam",
)
arena = FovCalibrationArena()
nmf = NeuroMechFlyMuJoCo(
    sim_params=sim_params,
    arena=arena,
    init_pose=stretched_pose,
    actuated_joints=all_leg_dofs,
)

# Load recorded data
data_path = Path(pkg_resources.resource_filename("flygym", "data"))
with open(data_path / "behavior" / "210902_pr_fly1.pkl", "rb") as f:
    data = pickle.load(f)

# Interpolate 5x
num_steps = int(run_time / nmf.timestep)
data_block = np.zeros((len(nmf.actuated_joints), num_steps))
measure_t = np.arange(len(data["joint_LFCoxa"])) * data["meta"]["timestep"]
interp_t = np.arange(num_steps) * nmf.timestep
for i, joint in enumerate(nmf.actuated_joints):
    data_block[i, :] = np.interp(interp_t, measure_t, data[joint])

obs, reward, terminated, truncated, info = nmf.step({"joints": data_block[:, 0]})
for i in trange(100):
    joint_pos = data_block[:, 0]
    action = {"joints": joint_pos}
    obs, reward, terminated, truncated, info = nmf.step(action)
    nmf.render()

# Visualize static camera views upon initialization
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(nmf.physics.render(700, 700, camera_id="Animat/camera_left"))
axs[0, 0].axis("off")
axs[0, 1].imshow(nmf.physics.render(700, 700, camera_id="birdseye_cam"))
axs[0, 1].axis("off")
axs[1, 0].imshow(nmf.curr_raw_visual_input[0])
axs[1, 0].axis("off")
axs[1, 1].imshow(nmf.curr_raw_visual_input[1])
axs[1, 1].axis("off")
plt.tight_layout()
plt.show()
nmf.close()

# Visualize camera views during simulation
sim_params = MuJoCoParameters(
    timestep=1e-4,
    render_mode="saved",
    render_playspeed=0.1,
    enable_vision=True,
    render_raw_vision=True,
)
arena = FovCalibrationArena()
nmf = NeuroMechFlyMuJoCo(
    sim_params=sim_params,
    arena=arena,
    init_pose=stretched_pose,
    actuated_joints=all_leg_dofs,
)
obs_list = []
for i in trange(num_steps):
    joint_pos = data_block[:, i]
    action = {"joints": joint_pos}
    obs, reward, terminated, truncated, info = nmf.step(action)
    nmf.render()
    obs_list.append(obs)
nmf.close()
nmf.save_video(Path("vision_arena.mp4"))

visualize_visual_input(
    output_path=Path("eyes.mp4"),
    vision_data_li=[x["vision"] for x in obs_list],
    raw_vision_data_li=[x["raw_vision"] for x in obs_list],
    vision_update_mask=nmf.vision_update_mask,
    vision_refresh_rate=sim_params.vision_refresh_rate,
    playback_speed=0.1,
)
