import argparse
import numpy as np
import pickle
import cv2
from tqdm import trange
from pathlib import Path
from typing import Tuple, Union, Optional, Callable

from flygym import Fly, Camera
from flygym.arena import BaseArena
from flygym.util import get_data_path
from flygym.examples.turning_controller import HybridTurningNMF
from flygym.preprogrammed import get_cpg_biases
from flygym.examples.path_integration import (
    PathIntegrationArena,
    WalkingState,
    PathIntegrationNMF,
    RandomExplorationController,
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
    gait: str = "tripod",
    live_display: bool = False,
    do_path_integration: bool = False,
    heading_model: Optional[Callable] = None,
    displacement_model: Optional[Callable] = None,
    time_scale: Optional[float] = None,
    output_dir: Optional[Path] = None,
):
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
    arena = PathIntegrationArena()
    # cam = Camera(
    #     fly=fly, camera_id="Animat/camera_left", play_speed=0.1, timestamp_text=True
    # )
    cam = Camera(fly=fly, camera_id="birdeye_cam", play_speed=0.5, timestamp_text=True)
    sim = PathIntegrationNMF(
        phase_biases=get_cpg_biases(gait),
        fly=fly,
        arena=arena,
        cameras=[cam],
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
    _estimated_heading_buffer = []
    for i in trange(int(running_time / sim.timestep)):
        walking_state, dn_drive = random_exploration_controller.step()
        action_hist.append(dn_drive)
        obs, reward, terminated, truncated, info = sim.step(dn_drive)
        rendered_img = sim.render()[0]

        # Get real heading
        orientation_x, orientation_y = obs["fly_orientation"][:2]
        real_heading = np.arctan2(orientation_y, orientation_x)
        _real_heading_buffer.append(real_heading)

        # Get estimated heading
        if heading_model is not None:
            estimated_heading = info["heading_estimate"]
            _estimated_heading_buffer.append(wrap_angle(estimated_heading))

        if rendered_img is not None:
            # Add walking state icon
            add_icon_to_image(rendered_img, icons[walking_state])

            # Add heading info
            if do_path_integration:
                real_heading = np.mean(_real_heading_buffer)
                estimated_heading = np.mean(_estimated_heading_buffer)
                _real_heading_buffer = []
                _estimated_heading_buffer = []
                add_heading_to_image(
                    rendered_img,
                    real_heading=real_heading,
                    estimated_heading=estimated_heading,
                    real_position=obs["fly"][0, :2],
                    estimated_position=info["position_estimate"],
                )

            # Display rendered image live
            if live_display:
                cv2.imshow("rendered_img", rendered_img[:, :, ::-1])
                cv2.waitKey(1)

        obs_hist.append(obs)
        info_hist.append(info)

    # Save data if output_dir is provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        cam.save_video(output_dir / "rendering.mp4")
        with open(output_dir / "sim_data.pkl", "wb") as f:
            data = {
                "obs_hist": obs_hist,
                "info_hist": info_hist,
                "action_hist": action_hist,
            }
            pickle.dump(data, f)


def main():
    parser = argparse.ArgumentParser(
        description="Run path integration experiment with random exploration."
    )

    # Adding arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for random exploration. Defaults to 0.",
    )
    parser.add_argument(
        "--running_time",
        type=float,
        default=20.0,
        help="Running time in seconds. Defaults to 20.",
    )
    parser.add_argument(
        "--live_display",
        action="store_true",
        help="Enable live display. Defaults to False.",
    )
    parser.add_argument(
        "--gait",
        type=str,
        choices=["tripod", "tetrapod", "wave"],
        default="tripod",
        help=(
            "Hexapod gait type. Choose from ['tripod', 'tetrapod', 'wave']. "
            "Defaults to 'tripod'."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to which output files are saved. Defaults to '.'.",
    )
    args = parser.parse_args()

    run_simulation(
        seed=args.seed,
        running_time=args.running_time,
        gait=args.gait,
        live_display=args.live_display,
        do_path_integration=False,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
