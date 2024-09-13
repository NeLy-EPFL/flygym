import cv2
import numpy as np
from tqdm import trange
from flygym import Camera, PhysicsError
from flygym import SingleFlySimulation
from flygym.arena import FlatTerrain
from flygym.examples.locomotion import HybridTurningFly


class MarkedArena(FlatTerrain):
    """Arena with small spheres marking x and y directions.
    +x: red, big
    -x: red, small
    +y: green, big
    -y: green, small
    With default orientation, the fly initially faces the +x direction and
    the +y direction is to the left.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        positions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        colors = [(1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0)]
        sizes = [0.15, 0.05, 0.15, 0.05]
        for i, (position, color, size) in enumerate(zip(positions, colors, sizes)):
            self._add_marker(position, color, size, f"marker_{i}")

    def _add_marker(self, position, color, size, name):
        body = self.root_element.worldbody.add(
            "body", name=name, mocap=True, pos=(*position, size)
        )
        body.add("geom", name=name, type="sphere", size=(size,), rgba=(*color, 1))


class CameraWithCardinalVectors(Camera):
    """Render cardinal vectors pointing in front (idx 0, red), up (idx 1, green), and
    right (idx 2, blue).
    """

    def attach_simulation(self, sim):
        self.sim = sim

    def render(self, *args, **kwargs):
        img = super().render(*args, **kwargs)

        # Compute where the fly's thorax is and where the tips of the
        # cardinal vector arrows should be on the rendered view
        camera_matrix = self.dm_camera.matrix
        obs = self.fly.get_observation(sim)
        pos_thorax = obs["fly"][0, :]
        cardinal_vectors = obs["cardinal_vectors"]
        assert np.allclose(np.linalg.norm(obs["cardinal_vectors"], axis=1), 1)
        arrow_tips = pos_thorax + cardinal_vectors
        disp_x_thorax, disp_y_thorax, disp_scale_thorax = (
            camera_matrix @ np.hstack([*pos_thorax, np.ones(1)])[np.newaxis, :].T
        )
        disp_xs_arrowtips, disp_ys_arrowtips, disp_scales_arrowtips = (
            camera_matrix @ np.hstack([arrow_tips, np.ones((3, 1))]).T
        )
        disp_x_thorax = disp_x_thorax[0] / disp_scale_thorax
        disp_y_thorax = disp_y_thorax[0] / disp_scale_thorax
        disp_xs_arrowtips = disp_xs_arrowtips / disp_scales_arrowtips
        disp_ys_arrowtips = disp_ys_arrowtips / disp_scales_arrowtips

        # Draw arrows
        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            img = cv2.arrowedLine(
                img,
                (int(disp_x_thorax), int(disp_y_thorax)),
                (int(disp_xs_arrowtips[i]), int(disp_ys_arrowtips[i])),
                color,
                3,
            )
        return img


run_time = 2.0
timestep = 1e-4
contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

np.random.seed(1)

fly = HybridTurningFly(
    enable_adhesion=True,
    draw_adhesion=True,
    contact_sensor_placements=contact_sensor_placements,
    seed=0,
    draw_corrections=True,
    timestep=timestep,
)
print(f"Spawn orientation: {fly.spawn_orientation}")

cameras = {
    pos: CameraWithCardinalVectors(
        fly=fly, camera_id=f"Animat/camera_{pos}", play_speed=0.1
    )
    for pos in ["left", "right", "top", "front", "back"]
}

sim = SingleFlySimulation(
    fly=fly,
    cameras=list(cameras.values()),
    timestep=timestep,
    arena=MarkedArena(),
)
[cam.attach_simulation(sim) for cam in cameras.values()]

obs_list = []

obs, info = sim.reset(seed=0)
print(f"Spawning fly at {obs['fly'][0]} mm")

for i in trange(int(run_time / sim.timestep)):
    curr_time = i * sim.timestep
    action = np.array([0.4, 1])

    try:
        obs, reward, terminated, truncated, info = sim.step(action)
        obs_list.append(obs)
        sim.render()
    except PhysicsError:
        print("Simulation was interrupted because of a physics error")
        break

x_pos = obs_list[-1]["fly"][0][0]
print(f"Final x position: {x_pos:.4f} mm")
print(f"Simulation terminated: {obs_list[-1]['fly'][0] - obs_list[0]['fly'][0]}")

for pos, cam in cameras.items():
    cam.save_video(f"./outputs/cardinal_vectors/camera_{pos}.mp4", 0)
