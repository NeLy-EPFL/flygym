import numpy as np
import cv2
import pickle
from sys import stderr
from datetime import datetime
from pathlib import Path

from flygym.util import get_data_path
from flygym import Fly, Camera
from flygym.examples.olfaction import (
    PlumeNavigationController,
    WalkingState,
    OdorPlumeArena,
    PlumeNavigationTask,
)


def eprint(*args, **kwargs):
    """Print log to stderr so that the buffer gets flushed immediately."""
    print(*args, file=stderr, **kwargs)


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


def run_simulation(
    plume_dataset_path,
    output_dir,
    seed,
    initial_position=(180, 80),
    live_display=False,
    is_control=False,
    run_time=60,
):
    arena = OdorPlumeArena(plume_dataset_path)

    # Define the fly
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]
    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        enable_olfaction=True,
        enable_vision=False,
        contact_sensor_placements=contact_sensor_placements,
        spawn_pos=(*initial_position, 0.25),
        spawn_orientation=(0, 0, -np.pi / 2),
    )
    cam = Camera(fly=fly, camera_id="birdeye_cam", play_speed=0.5, timestamp_text=True)
    sim = PlumeNavigationTask(
        fly=fly,
        arena=arena,
        cameras=[cam],
    )
    if is_control:
        controller = PlumeNavigationController(
            dt=sim.timestep,
            alpha=0,
            delta_lambda_sw=0,
            delta_lambda_ws=0,
            random_seed=seed,
        )
    else:
        controller = PlumeNavigationController(sim.timestep)
    icons = get_walking_icons()
    encounter_threshold = 0.001

    # Run the simulation
    obs_hist = []
    obs, _ = sim.reset()
    for i in range(int(run_time / sim.timestep)):
        if i % int(1 / sim.timestep) == 0:
            sec = i * sim.timestep
            eprint(f"{datetime.now()} - seed {seed}: {sec:.1f} / {run_time:.1f} sec")
        obs = sim.get_observation()
        walking_state, dn_drive, debug_str = controller.decide_state(
            encounter_flag=obs["odor_intensity"].max() > encounter_threshold,
            fly_heading=obs["fly_orientation"],
        )
        obs, reward, terminated, truncated, info = sim.step(dn_drive)
        if terminated or truncated:
            break
        rendered_img = sim.render()[0]
        if rendered_img is not None:
            add_icon_to_image(rendered_img, icons[walking_state])
            cv2.putText(
                rendered_img,
                debug_str,
                (20, cam.window_size[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            # if obs["odor_intensity"].max() > encounter_threshold:
            #     cv2.putText(
            #         rendered_img,
            #         f"Encounter {obs['odor_intensity'].max()}",
            #         (20, sim_params.render_window_size[1] - 80),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.5,
            #         (0, 0, 0),
            #         1,
            #         cv2.LINE_AA,
            #     )
            if live_display:
                cv2.imshow("rendered_img", rendered_img[:, :, ::-1])
                cv2.waitKey(1)

        obs_hist.append(obs)

    filename_stem = f"plume_navigation_seed{seed}_control{is_control}"
    cam.save_video(output_dir / (filename_stem + ".mp4"))
    with open(output_dir / (filename_stem + ".pkl"), "wb") as f:
        pickle.dump({"obs_hist": obs_hist, "reward": reward}, f)
    return sim


def process_trial(plume_dataset_path, output_dir, seed, initial_position, is_control):
    try:
        # Run_simulation returns a Simulation object, which is not
        # pickle-able. Do not return it here in this wrapper.
        run_simulation(
            plume_dataset_path,
            output_dir,
            seed,
            initial_position,
            is_control=is_control,
            live_display=False,
        )
    except Exception as e:
        eprint(f"Error in seed {seed}: {e}")
        raise e


if __name__ == "__main__":
    from shutil import copy
    from joblib import Parallel, delayed

    plume_dataset_path = Path("./outputs/plume_tracking/plume_dataset/plume.hdf5")
    output_dir = Path("./outputs/plume_tracking/sim_results/")

    # Copy the plume dataset to the shared memory for parallel access
    plume_dataset_shm_path = Path("/dev/shm/") / plume_dataset_path.name
    copy(plume_dataset_path, plume_dataset_shm_path)

    try:
        xx, yy = np.meshgrid(np.linspace(155, 200, 10), np.linspace(57.5, 102.5, 10))
        points = np.vstack((xx.flat, yy.flat)).T
        configs = [
            (plume_dataset_shm_path, output_dir, seed, initial_position, False)
            for seed, initial_position in enumerate(points)
        ]

        # Run the simulations in parallel
        results = Parallel(n_jobs=-2)(
            delayed(process_trial)(*config) for config in configs
        )
    finally:
        # Clean up the shared memory
        plume_dataset_shm_path.unlink()
