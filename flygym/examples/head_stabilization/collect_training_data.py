import numpy as np
import pickle
import cv2
from tqdm import trange
from pathlib import Path
from typing import Optional
from dm_control.utils import transformations
from dm_control.rl.control import PhysicsError

from flygym import Fly, Camera
from flygym.arena import FlatTerrain, BlocksTerrain
from flygym.preprogrammed import get_cpg_biases
from flygym.examples.locomotion import HybridTurningController


def run_simulation(
    gait: str = "tripod",
    terrain: str = "flat",
    spawn_xy: tuple[float, float] = (0, 0),
    dn_drive: tuple[float, float] = (1, 1),
    sim_duration: float = 0.5,
    enable_rendering: bool = False,
    live_display: bool = False,
    output_dir: Optional[Path] = None,
    pbar: bool = False,
):
    """Simulate locomotion and collect proprioceptive information to train
    a neural network for head stabilization.

    Parameters
    ----------
    gait : str, optional
        The type of gait for the fly. Choose from ['tripod', 'tetrapod',
        'wave']. Defaults to "tripod".
    terrain : str, optional
        The type of terrain for the fly. Choose from ['flat', 'blocks'].
        Defaults to "flat".
    spawn_xy : tuple[float, float], optional
        The x and y coordinates of the fly's spawn position. Defaults to
        (0, 0).
    dn_drive : tuple[float, float], optional
        The DN drive values for the left and right wings. Defaults to
        (1, 1).
    sim_duration : float, optional
        The duration of the simulation in seconds. Defaults to 0.5.
    enable_rendering: bool, optional
        If True, enables rendering. Defaults to False.
    live_display : bool, optional
        If True, enables live display. Defaults to False.
    output_dir : Path, optional
        The directory to which output files are saved. Defaults to None.
    pbar : bool, optional
        If True, enables progress bar. Defaults to False.

    Raises
    ------
    ValueError
        Raised when an unknown terrain type is provided.
    """
    if (not enable_rendering) and live_display:
        raise ValueError("Cannot enable live display without rendering.")

    # Set up arena
    if terrain == "flat":
        arena = FlatTerrain()
    elif terrain == "blocks":
        arena = BlocksTerrain(height_range=(0.2, 0.2))
    else:
        raise ValueError(f"Unknown terrain type: {terrain}")

    # Set up simulation
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]
    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        detect_flip=True,
        contact_sensor_placements=contact_sensor_placements,
        spawn_pos=(*spawn_xy, 0.25),
    )
    if enable_rendering:
        cam = Camera(
            fly=fly, camera_id="Animat/camera_left", play_speed=0.1, timestamp_text=True
        )
    else:
        cam = None
    sim = HybridTurningController(
        arena=arena,
        phase_biases=get_cpg_biases(gait),
        fly=fly,
        cameras=[cam] if enable_rendering else [],
        timestep=1e-4,
    )

    # Main simulation loop
    obs, info = sim.reset(0)
    obs_hist, info_hist, action_hist = [], [], []
    dn_drive = np.array(dn_drive)
    physics_error, fly_flipped = False, False
    iterator = trange if pbar else range
    for _ in iterator(int(sim_duration / sim.timestep)):
        action_hist.append(dn_drive)

        try:
            obs, reward, terminated, truncated, info = sim.step(dn_drive)
        except PhysicsError:
            print("Physics error detected!")
            physics_error = True
            break

        if enable_rendering:
            rendered_img = sim.render()[0]

        # Get necessary angles
        quat = sim.physics.bind(sim.fly.thorax).xquat
        quat_inv = transformations.quat_inv(quat)
        roll, pitch, yaw = transformations.quat_to_euler(quat_inv, ordering="XYZ")
        info["roll"], info["pitch"], info["yaw"] = roll, pitch, yaw

        obs_hist.append(obs)
        info_hist.append(info)

        if info["flip"]:
            print("Flip detected!")
            break

        # Live display
        if enable_rendering and live_display and rendered_img is not None:
            cv2.imshow("rendered_img", rendered_img[:, :, ::-1])
            cv2.waitKey(1)

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
                "errors": {
                    "fly_flipped": fly_flipped,
                    "physics_error": physics_error,
                },
            }
            pickle.dump(data, f)


if __name__ == "__main__":
    # run_simulation(live_display=True, terrain="blocks", dn_drive=(1, 1), pbar=True)

    from joblib import Parallel, delayed
    from numpy.random import RandomState

    random_state = RandomState(0)
    output_basedir = Path("outputs/head_stabilization/random_exploration/")

    job_specs = []
    for gait in ["tripod", "tetrapod", "wave"]:
        for terrain in ["flat", "blocks"]:
            for test_set in [True, False]:
                # Get DN drives
                if test_set:
                    turning_drives = np.linspace(-0.9, 0.9, 10)
                else:
                    turning_drives = np.linspace(-1, 1, 11)  # staggered from test set
                amp_lower = np.maximum(1 - 0.6 * np.abs(turning_drives), 0.4)
                amp_upper = np.minimum(1 + 0.2 * np.abs(turning_drives), 1.2)
                dn_drives_left = np.where(turning_drives > 0, amp_upper, amp_lower)
                dn_drives_right = np.where(turning_drives > 0, amp_lower, amp_upper)

                set_tag = "test_set" if test_set else "train_set"
                for dn_left, dn_right in zip(dn_drives_left, dn_drives_right):
                    spawn_xy = random_state.uniform(-1.3, 1.3, size=2)
                    dn_drive = np.array([dn_left, dn_right])
                    output_dir = (
                        output_basedir
                        / f"{gait}_{terrain}_{set_tag}_{dn_left:.2f}_{dn_right:.2f}"
                    )
                    job_specs.append(
                        (
                            gait,
                            terrain,
                            spawn_xy,
                            dn_drive,
                            1.5,  # sim duration
                            True,  # enable rendering
                            False,  # live display
                            output_dir,
                            True,  # enable progress bar
                        )
                    )

    Parallel(n_jobs=-2)(delayed(run_simulation)(*job_spec) for job_spec in job_specs)
