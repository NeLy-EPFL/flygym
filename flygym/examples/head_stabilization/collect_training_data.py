import argparse
import numpy as np
import pickle
import cv2
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from dm_control.utils import transformations

from flygym import Fly, Camera
from flygym.preprogrammed import get_cpg_biases
from flygym.examples.turning_controller import HybridTurningNMF


def run_simulation(
    gait: str = "tripod",
    live_display: bool = False,
    output_dir: Optional[Path] = None,
):
    dn_drive_seq = np.array(
        [[l, r] for l in np.linspace(-0.2, 1.2, 11) for r in np.linspace(-0.2, 1.2, 11)]
    )
    dn_drive_seq = np.repeat(dn_drive_seq, int(0.5 / 1e-4), axis=0)

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
    cam = Camera(
        fly=fly, camera_id="Animat/camera_left", play_speed=0.1, timestamp_text=True
    )
    # cam = Camera(fly=fly, camera_id="birdeye_cam", play_speed=0.5, timestamp_text=True)
    sim = HybridTurningNMF(
        phase_biases=get_cpg_biases(gait),
        fly=fly,
        cameras=[cam],
        timestep=1e-4,
        correction_rates={"retraction": (0, 0), "stumbling": (0, 0)},
    )

    obs, info = sim.reset(0)
    obs_hist, info_hist, action_hist = [], [], []
    for dn_drive in tqdm(dn_drive_seq):
        action_hist.append(dn_drive)
        obs, reward, terminated, truncated, info = sim.step(dn_drive)
        rendered_img = sim.render()[0]

        # Get necessary angles
        quat = sim.physics.bind(sim.fly.thorax).xquat
        quat_inv = transformations.quat_inv(quat)
        roll, pitch, yaw = transformations.quat_to_euler(quat_inv, ordering="XYZ")
        info["roll"], info["pitch"], info["yaw"] = roll, pitch, yaw

        obs_hist.append(obs)
        info_hist.append(info)

        # Live display
        if live_display and rendered_img is not None:
            cv2.imshow("rendered_img", rendered_img[:, :, ::-1])
            cv2.waitKey(1)

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
        gait=args.gait,
        live_display=args.live_display,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
