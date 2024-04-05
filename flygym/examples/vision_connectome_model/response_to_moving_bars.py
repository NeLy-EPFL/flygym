import pickle
from pathlib import Path

import numpy as np
import torch
from flygym import Camera, Fly
from flygym.examples.vision_connectome_model import (
    MovingBarArena,
    get_azimuth_func,
    visualize_vision,
)
from tqdm import trange

from response_to_fly import NMFRealisticVison

torch.manual_seed(0)


if __name__ == "__main__":
    for speed in 360 * np.power(2.0, np.arange(-7, 5)[::-1]):
        regenerate_walking = True
        output_dir = Path("./outputs/moving_bars_4deg/")
        output_dir.mkdir(parents=True, exist_ok=True)
        start_angle = -180
        end_angle = 180
        run_time = abs(end_angle - start_angle) / speed
        vision_refresh_rate = 500  # Hz

        arena = MovingBarArena(
            azimuth_func=get_azimuth_func(start_angle, end_angle, run_time, 0.2),
            ground_alpha=0,
            visual_angle=(4, 150),
            distance=10,
        )

        run_time += 0.4

        contact_sensor_placements = [
            f"{leg}{segment}"
            for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
            for segment in [
                "Tibia",
                "Tarsus1",
                "Tarsus2",
                "Tarsus3",
                "Tarsus4",
                "Tarsus5",
            ]
        ]

        fly = Fly(
            render_raw_vision=True,
            actuator_kp=0,
            enable_vision=True,
            vision_refresh_rate=vision_refresh_rate,
            contact_sensor_placements=contact_sensor_placements,
        )

        cam = Camera(
            fly=fly,
            camera_id="birdeye_cam",
            play_speed=0.2,
            window_size=(800, 608),
            fps=24,
            play_speed_text=False,
        )

        sim = NMFRealisticVison(
            fly=fly,
            cameras=[cam],
            arena=arena,
            timestep=2e-4,
        )

        for i in fly.model.find_all("geom"):
            sim.physics.named.model.geom_rgba[f"{fly.name}/{i.name}"] = [
                0.5,
                0.5,
                0.5,
                0,
            ]

        num_physics_steps = int(run_time / sim.timestep)

        obs_hist = []
        rendered_image_hist = []
        vision_observation_hist = []
        nn_activities_hist = []

        # Main simulation loop
        for i in trange(num_physics_steps):
            obs, _, _, _, info = sim.step(np.zeros(2))
            rendered_img = sim.render()[0]
            obs_hist.append(obs)
            if rendered_img is not None:
                rendered_image_hist.append(rendered_img)
                vision_observation_hist.append(obs["vision"])
                nn_activities_hist.append(obs["nn_activities"])

        visualize_vision(
            Path(output_dir / f"bar_simulation_{speed}.mp4"),
            fly.retina,
            sim.retina_mapper,
            rendered_image_hist,
            vision_observation_hist,
            nn_activities_hist,
            fps=cam.fps,
        )

        np.save(
            output_dir / f"nn_hist_{speed}.npy",
            np.array([i[:] for i in nn_activities_hist]),
        )

        obs_hist = [
            obs
            for obs, vision_updated in zip(obs_hist, fly.vision_update_mask)
            if vision_updated
        ]
        for obs in obs_hist:
            del obs["nn_activities"]

        with open(output_dir / f"obs_hist_{speed}.npy", "wb") as f:
            pickle.dump(obs_hist, f)
