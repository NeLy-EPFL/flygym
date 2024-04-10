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
from scipy.ndimage import gaussian_filter1d
from tqdm import trange

from response_to_fly import NMFRealisticVison

torch.manual_seed(0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.rcParams["figure.facecolor"] = (0.95,) * 3
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42

    # experiment parameters
    vision_refresh_rate = 500  # Hz
    timestep = 1 / vision_refresh_rate
    start_angle = -180
    end_angle = 180
    speeds = 360 * np.power(2.0, np.arange(-7, 5)[::-1])
    time_before = 0.2
    time_after = 0.2
    bar_width_deg = 4
    bar_height_deg = 150

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
        neck_kp=0,
        init_pose="zero",
    )

    cam = Camera(
        fly=fly,
        camera_id="birdeye_cam",
        play_speed=0.2,
        window_size=(800, 608),
        fps=24,
        play_speed_text=False,
    )

    arena = MovingBarArena(
        azimuth_func=lambda t: start_angle,
        ground_alpha=0,
        visual_angle=(bar_width_deg, bar_height_deg),
        distance=10,
    )

    sim = NMFRealisticVison(
        fly=fly,
        cameras=[cam],
        arena=arena,
        timestep=timestep,
        gravity=(0, 0, 0),
    )

    output_dir = Path(f"./outputs/moving_bars_{bar_width_deg}deg/")
    output_dir.mkdir(parents=True, exist_ok=True)

    for speed in speeds:
        run_time = abs(end_angle - start_angle) / speed
        arena.azimuth_func = get_azimuth_func(
            start_angle, end_angle, run_time, time_before
        )
        arena.reset(sim.physics)
        run_time += time_before + time_after
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

        cam.reset()

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

    # Plot the activity of a single neuron type
    def get_activity(x, neuron_type):
        from flyvision.utils.activity_utils import LayerActivity

        layer_activity = LayerActivity(
            x,
            sim.vision_network.connectome,
            keepref=True,
            use_central=False,
        )
        neuron_type_activity = getattr(layer_activity, neuron_type)
        right_center_ommatidia = sim.retina_mapper.flyvis_to_flygym(
            neuron_type_activity
        )[..., 1, 360]
        return right_center_ommatidia

    neuron_type = "T4a"

    nn_activities = [
        get_activity(np.load(output_dir / f"nn_hist_{i}.npy"), neuron_type)
        for i in speeds
    ]
    steepest_idx = [gaussian_filter1d(i, 2, order=1).argmax() for i in nn_activities]

    plt.rcParams["figure.facecolor"] = "none"

    # plot activity over time
    n_cols = len(speeds)
    fig, axs = plt.subplots(
        1, n_cols, figsize=(n_cols * 1.2, 1), sharex=False, sharey=True, dpi=300
    )

    vmax = 2

    for i, ax in enumerate(axs):
        speed = speeds[i]

        if int(speed) == speed:
            speed = int(speed)

        y = nn_activities[i]
        t = np.arange(len(y)) / 120
        ax.plot(t, y, color="k")
        j = steepest_idx[i]
        t0 = t[y[:j].argmin()]
        t1 = t[y[j:].argmax() + j]
        dur = t1 - t0
        ax.set_xlim(t0 - dur * 2, t1 + dur * 2)
        ax.set_title(f"{speed}°/s")

        x_trans = ax.get_xaxis_transform()

        ax.plot([t0, t1], [0, 0], transform=x_trans, c="k", clip_on=False)
        ax.text(
            (t0 + t1) / 2,
            -0.05,
            f"{dur:.2f} s",
            color="k",
            ha="center",
            va="top",
            transform=x_trans,
        )

        ax.set_xticks([])
        ax.set_ylim(-vmax, vmax)
        ax.set_yticks([-vmax, 0, vmax])

        for spine in ax.spines.values():
            spine.set_visible(False)

        if i == 0:
            ax.spines["left"].set_visible(True)
            ax.spines["left"].set_bounds(-vmax, vmax)
        else:
            ax.yaxis.set_tick_params(size=0)

    axs[0].set_ylabel(f"{neuron_type} activity (a.u.)", labelpad=1)

    plt.savefig(output_dir / f"{neuron_type}_activity.pdf", bbox_inches="tight")

    # plot the tuning curve
    tuning_curve = [nn_activities[i][j:].max() for i, j in enumerate(steepest_idx)]

    fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=300)
    ax.plot(speeds, tuning_curve, c="k", marker=".", markeredgewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(speeds)

    xlabels = [int(s) if int(s) == s else f"{s:0.1f}" for s in speeds]
    ax.set_xticklabels(
        xlabels, rotation=45, ha="right", rotation_mode="anchor", va="top"
    )
    ax.tick_params(axis="both", pad=1)
    ax.set_xlabel("Speed (°/s)")
    ax.set_ylabel(f"{neuron_type} max. activity (a.u.)", labelpad=1)
    ax.set_ylim(0, vmax)
    ax.set_yticks([0, vmax])

    for sides in ["top", "right"]:
        ax.spines[sides].set_visible(False)

    plt.savefig(output_dir / f"{neuron_type}_tuning_curve.pdf", bbox_inches="tight")
