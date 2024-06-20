import numpy as np
import pickle
import cv2
from tqdm import trange
from pathlib import Path
from typing import Optional

from flygym import Fly, Camera, is_rendering_skipped
from flygym.util import get_data_path
from flygym.preprogrammed import get_cpg_biases
from flygym.examples.path_integration.arena import (
    PathIntegrationArenaFlat,
    PathIntegrationArenaBlocks,
)
from flygym.examples.path_integration.controller import (
    WalkingState,
    RandomExplorationController,
    PathIntegrationController,
)


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


def add_heading_to_image(
    image, real_heading, estimated_heading, real_position, estimated_position
):
    cv2.putText(
        image,
        f"Real position: ({int(real_position[0]): 3d}, {int(real_position[1]): 3d})",
        org=(20, 390),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.5,
        color=(0, 0, 0),
        lineType=cv2.LINE_AA,
        thickness=1,
    )
    cv2.putText(
        image,
        (
            f"Estm position: ({int(estimated_position[0]): 3d}, "
            f"{int(estimated_position[1]): 3d})"
        ),
        org=(20, 410),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.5,
        color=(0, 0, 0),
        lineType=cv2.LINE_AA,
        thickness=1,
    )
    cv2.putText(
        image,
        f"Real heading: {int(np.rad2deg(real_heading)): 4d} deg",
        org=(20, 430),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.5,
        color=(0, 0, 0),
        lineType=cv2.LINE_AA,
        thickness=1,
    )
    cv2.putText(
        image,
        f"Estm heading: {int(np.rad2deg(estimated_heading)): 4d} deg",
        org=(20, 450),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.5,
        color=(0, 0, 0),
        lineType=cv2.LINE_AA,
        thickness=1,
    )


def wrap_angle(angle):
    """Wrap angle to [-π, π]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def run_simulation(
    seed: int = 0,
    running_time: float = 20.0,
    terrain_type: str = "flat",
    gait: str = "tripod",
    live_display: bool = False,
    enable_rendering: bool = True,
    pbar: bool = False,
    output_dir: Optional[Path] = None,
):
    if (not enable_rendering) and live_display:
        raise ValueError("Cannot enable live display without rendering.")
    icons = get_walking_icons()
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]

    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        contact_sensor_placements=contact_sensor_placements,
        spawn_pos=(0, 0, 0.25),
    )

    if terrain_type == "flat":
        arena = PathIntegrationArenaFlat()
    elif terrain_type == "blocks":
        arena = PathIntegrationArenaBlocks(
            height_range=(0.2, 0.2), x_range=(-50, 50), y_range=(-50, 50)
        )
    else:
        raise ValueError(f"Unknown terrain type: {terrain_type}")

    cam = Camera(fly=fly, camera_id="birdeye_cam", play_speed=0.5, timestamp_text=True)
    sim = PathIntegrationController(
        phase_biases=get_cpg_biases(gait),
        fly=fly,
        arena=arena,
        cameras=[] if is_rendering_skipped else [cam],
        timestep=1e-4,
        correction_rates={"retraction": (0, 0), "stumbling": (0, 0)},
    )

    random_exploration_controller = RandomExplorationController(
        dt=sim.timestep,
        lambda_turn=2,
        seed=seed,
        forward_dn_drive=(1.0, 1.0),
        left_turn_dn_drive=(0.2, 1.0) if gait == "wave" else (-0.2, 1.0),
        right_turn_dn_drive=(1.0, 0.2) if gait == "wave" else (1.0, -0.2),
    )

    obs, info = sim.reset(0)
    obs_hist, info_hist, action_hist = [], [], []
    _real_heading_buffer = []
    iterator = trange if pbar else range
    for i in iterator(int(running_time / sim.timestep)):
        walking_state, dn_drive = random_exploration_controller.step()
        action_hist.append(dn_drive)
        obs, reward, terminated, truncated, info = sim.step(dn_drive)
        if enable_rendering:
            rendered_img = sim.render()[0]
        else:
            rendered_img = None

        # Get real heading
        orientation_x, orientation_y = obs["fly_orientation"][:2]
        real_heading = np.arctan2(orientation_y, orientation_x)
        _real_heading_buffer.append(real_heading)

        if rendered_img is not None:
            # Add walking state icon
            add_icon_to_image(rendered_img, icons[walking_state])

            # Display rendered image live
            if live_display:
                cv2.imshow("rendered_img", rendered_img[:, :, ::-1])
                cv2.waitKey(1)

        obs_hist.append(obs)
        info_hist.append(info)

    # Save data if output_dir is provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if enable_rendering:
            cam.save_video(output_dir / "rendering.mp4")
        with open(output_dir / "sim_data.pkl", "wb") as f:
            data = {
                "obs_hist": obs_hist,
                "info_hist": info_hist,
                "action_hist": action_hist,
            }
            pickle.dump(data, f)


if __name__ == "__main__":
    from joblib import Parallel, delayed

    root_output_dir = Path("./outputs/path_integration/random_exploration")
    gaits = ["tripod", "tetrapod", "wave"]
    seeds = list(range(15))

    configs = [(gait, seed) for gait in gaits for seed in seeds]

    def wrapper(gait, seed):
        run_simulation(
            seed=seed,
            running_time=20.0,
            terrain_type="flat",
            gait=gait,
            live_display=False,
            pbar=True,
            output_dir=root_output_dir / f"seed={seed}_gait={gait}",
        )

    Parallel(n_jobs=-2)(delayed(wrapper)(*config) for config in configs)
