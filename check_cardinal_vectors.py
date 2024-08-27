import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from flygym import Fly, Camera, PhysicsError
from flygym import SingleFlySimulation
from flygym.arena import FlatTerrain
from flygym.examples.locomotion import HybridTurningFly


run_time = 1.0
timestep = 1e-4
contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

np.random.seed(0)

fly = HybridTurningFly(
    enable_adhesion=True,
    draw_adhesion=True,
    contact_sensor_placements=contact_sensor_placements,
    seed=0,
    draw_corrections=True,
    timestep=timestep,
)

cam = Camera(fly=fly, camera_id="Animat/camera_right", play_speed=0.1)
sim = SingleFlySimulation(
    fly=fly,
    cameras=[cam],
    timestep=timestep,
    arena=FlatTerrain(),
)

obs_list = []

obs, info = sim.reset(seed=0)
print(f"Spawning fly at {obs['fly'][0]} mm")

for i in trange(int(run_time / sim.timestep)):
    curr_time = i * sim.timestep

    # # To demonstrate left and right turns:
    # if curr_time < run_time / 2:
    #     action = np.array([1.2, 0.4])
    # else:
    action = np.array([0.2, 1.2])

    # To demonstrate that the result is identical with the hybrid controller without
    # turning:
    # action = np.array([1.0, 1.0])

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

cam.save_video("./outputs/hybrid_turning_fly.mp4", 0)

cardinal_vectors = np.array([obs["cardinal_vectors"] for obs in obs_list])
print(cardinal_vectors.shape)
fig, axs = plt.subplots(3, 1, figsize=(4, 6), tight_layout=True)
t_grid = np.arange(0, run_time, sim.timestep)
for i in range(3):
    for j in range(3):
        axs[i].plot(t_grid, cardinal_vectors[:, i, j], label=["x", "y", "z"][j])
    axs[i].set_title(["Front", "Side", "Up"][i])
    axs[i].legend()
    axs[i].set_xlabel("Time (s)")
fig.savefig("./outputs/cardinal_vectors.png")