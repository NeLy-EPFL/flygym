import pickle
import numpy as np
from pathlib import Path
from tqdm import trange
from typing import Optional
from flygym import Camera, SingleFlySimulation
from dm_control.rl.control import PhysicsError

from flygym.examples.vision import MovingFlyArena, RealisticVisionFly
from flygym.examples.vision import viz
from flygym.examples.head_stabilization import HeadStabilizationInferenceWrapper
from flygym.examples.head_stabilization import get_head_stabilization_model_paths


contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

# fmt: off
neurons_txall = [
    "T1", "T2", "T2a", "T3", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
    "Tm1", "Tm2", "Tm3", "Tm4", "Tm5Y", "Tm5a", "Tm5b", "Tm5c", "Tm9", "Tm16", "Tm20",
    "Tm28", "Tm30", "TmY3", "TmY4", "TmY5a", "TmY9", "TmY10", "TmY13", "TmY14", "TmY15",
    "TmY18"
]
neurons_lc910_inputs = [
    "T2", "T2a", "T3",
    "Tm1", "Tm2", "Tm3", "Tm4", "Tm5Y", "Tm5a", "Tm5b", "Tm5c", "Tm9", "Tm16", "Tm20",
    "Tm28", "Tm30", "TmY3", "TmY4", "TmY5a", "TmY9", "TmY10", "TmY13", "TmY14", "TmY15",
    "TmY18"
]
# fmt: on

leading_fly_speeds = {"blocks": 13, "flat": 15}
leading_fly_radius = 10

baseline_dir = Path("./outputs/connectome_constrained_vision/baseline_response/")
output_dir = Path("./outputs/connectome_constrained_vision/closed_loop_control/")

# If you trained the models yourself (by running ``collect_training_data.py``
# followed by ``train_proprioception_model.py``), you can use the following
# paths to load the models that you trained. Modify the paths if saved the
# model checkpoints elsewhere.
stabilization_model_dir = Path("./outputs/head_stabilization/models/")
stabilization_model_path = stabilization_model_dir / "All.ckpt"
scaler_param_path = stabilization_model_dir / "joint_angle_scaler_params.pkl"

# Alternatively, you can use the pre-trained models that come with the
# package. To do so, comment out the three lines above and uncomment the
# following line.
# stabilization_model_path, scaler_param_path = get_head_stabilization_model_paths()

if not stabilization_model_path.exists() or not scaler_param_path.exists():
    import warnings

    warnings.warn(
        "Head stabilization model not found. " "Pre-trained model will be used instead."
    )

    stabilization_model_path, scaler_param_path = get_head_stabilization_model_paths()


def run_simulation(
    arena: MovingFlyArena,
    tracking_cells: list[str],
    run_time: float,
    baseline_response: np.ndarray,
    z_score_threshold: float,
    tracking_gain: float,
    head_stabilization_model: Optional[HeadStabilizationInferenceWrapper] = None,
    spawn_xy: tuple[float, float] = (0, 0),
) -> dict:
    # Setup NMF simulation
    fly = RealisticVisionFly(
        contact_sensor_placements=contact_sensor_placements,
        enable_adhesion=True,
        vision_refresh_rate=500,
        neck_kp=500,
        head_stabilization_model=head_stabilization_model,
        spawn_pos=(*spawn_xy, 0.3),
    )
    cam = Camera(
        fly=fly,
        camera_id="birdeye_cam",
        play_speed=0.2,
        window_size=(800, 608),
        fps=24,
        play_speed_text=False,
    )
    sim = SingleFlySimulation(fly=fly, cameras=[cam], arena=arena)

    # Calculate center-of-mass of each ommatidium
    ommatidia_coms = np.empty((fly.retina.num_ommatidia_per_eye, 2))
    for i in range(fly.retina.num_ommatidia_per_eye):
        mask = fly.retina.ommatidia_id_map == i + 1
        ommatidia_coms[i, :] = np.argwhere(mask).mean(axis=0)

    # Run simulation
    obs, info = sim.reset(seed=0)
    obs_hist = []
    info_hist = []
    viz_data_all = []

    dn_drive = np.array([1, 1])
    for _ in trange(int(run_time / sim.timestep)):
        if info["vision_updated"]:
            # Estimate object mask
            nn_activities = info["nn_activities"]
            zscores = []
            for cell in tracking_cells:
                activities = fly.retina_mapper.flyvis_to_flygym(nn_activities[cell])
                response_mean = baseline_response[cell]["mean"]
                response_std = baseline_response[cell]["std"]
                abs_zscore = np.abs((activities - response_mean) / response_std)
                zscores.append(abs_zscore)
            zscores = np.array(zscores)
            mean_zscore = zscores.mean(axis=0)
            obj_mask = zscores.mean(axis=0) > z_score_threshold

            # Calculate turning bias based on object mask
            size_per_eye = obj_mask.sum(axis=1)
            com_per_eye = np.full((2, 2), np.nan)
            for eye_idx in range(2):
                if size_per_eye[eye_idx] > 0:
                    masked_xy_coords = ommatidia_coms[obj_mask[eye_idx], :]
                    com_per_eye[eye_idx, :] = masked_xy_coords.mean(axis=0)
            com_per_eye /= np.array([fly.retina.nrows, fly.retina.ncols])
            size_per_eye = size_per_eye / fly.retina.num_ommatidia_per_eye
            center_deviation = com_per_eye[:, 1].copy()
            center_deviation[0] = 1 - center_deviation[0]
            _center_deviation = center_deviation.copy()
            _center_deviation[size_per_eye == 0] = 1e9  # make sure it will break
            if size_per_eye.sum() == 0:
                turning_bias = 0
            else:
                turning_bias = (
                    -size_per_eye[0] * _center_deviation[0]
                    + size_per_eye[1] * _center_deviation[1]
                ) / size_per_eye.sum()

            # Calculate DN drive based on turning bias
            dn_inner = max(0.4, 1 - np.abs(turning_bias * tracking_gain) * 0.6)
            dn_outer = min(1.2, 1 + np.abs(turning_bias * tracking_gain) * 0.2)
            if turning_bias < 0:
                dn_drive = np.array([dn_inner, dn_outer])
            else:
                dn_drive = np.array([dn_outer, dn_inner])

        try:
            obs, _, _, _, info = sim.step(action=dn_drive)
        except PhysicsError:
            print("Physics error, breaking simulation")
            break

        # Stop simulation if the fly has fallen off the edge of the arena
        if arena.terrain_type == "blocks":
            x_ok = arena.x_range[0] < obs["fly"][0, 0] < arena.x_range[1]
            y_ok = arena.y_range[0] < obs["fly"][0, 1] < arena.y_range[1]
            if not (x_ok and y_ok):
                print("Fly has fallen off the arena, ending simulation early")
                break

        rendered_img = sim.render()[0]
        info["com_per_eye"] = com_per_eye
        info["size_per_eye"] = size_per_eye
        info["center_deviation"] = center_deviation
        info["turning_bias"] = turning_bias
        obs_hist.append(obs)
        info_hist.append(info)
        if rendered_img is not None:
            viz_data = {
                "rendered_image": rendered_img,
                "vision_observation": obs["vision"],
                "nn_activities": info["nn_activities"],
                "mean_zscore": mean_zscore,
            }
            viz_data_all.append(viz_data)

    return {
        "sim": sim,
        "fly": fly,
        "obs_hist": obs_hist,
        "info_hist": info_hist,
        "viz_data_all": viz_data_all,
    }


def process_trial(
    terrain_type: str,
    stabilization_on: bool,
    cells_selection: str,
    spawn_xy: tuple[float, float],
):
    variation_name = f"{terrain_type}terrain_stabilization{stabilization_on}"
    trial_name = f"{cells_selection}_x{spawn_xy[0]:.4f}y{spawn_xy[1]:.4f}"

    output_path = output_dir / f"sim_data/{variation_name}_{trial_name}.pkl"
    if output_path.is_file():
        with open(output_path, "rb") as f:
            try:
                _ = pickle.load(f)
                print(f"Skipping {variation_name}_{trial_name}")
                return
            except Exception as e:
                print(f"Failed to load {output_path} Rerunning.")

    with open(baseline_dir / f"{variation_name}_response_stats.pkl", "rb") as f:
        response_stats = pickle.load(f)

    if terrain_type == "flat":
        arena = MovingFlyArena(
            move_speed=leading_fly_speeds[terrain_type],
            radius=leading_fly_radius,
            terrain_type=terrain_type,
        )
    elif terrain_type == "blocks":
        arena = MovingFlyArena(
            move_speed=leading_fly_speeds[terrain_type],
            radius=leading_fly_radius,
            terrain_type=terrain_type,
        )
    else:
        raise ValueError("Invalid terrain type")

    if stabilization_on:
        stabilization_model = HeadStabilizationInferenceWrapper(
            model_path=stabilization_model_path,
            scaler_param_path=scaler_param_path,
        )
    else:
        stabilization_model = None

    if cells_selection == "txall":
        cells = neurons_txall
    elif cells_selection == "lc910_inputs":
        cells = neurons_lc910_inputs
    else:
        raise ValueError("Invalid cell selection")

    # Run simulation
    res = run_simulation(
        arena=arena,
        tracking_cells=cells,
        run_time=3.0,
        baseline_response=response_stats,
        z_score_threshold=5,
        tracking_gain=6,
        head_stabilization_model=stabilization_model,
        spawn_xy=spawn_xy,
    )

    # Save visualization
    viz.visualize_vision(
        Path(output_dir / f"videos/{variation_name}_{trial_name}.mp4"),
        res["fly"].retina,
        res["fly"].retina_mapper,
        viz_data_all=res["viz_data_all"],
        fps=res["sim"].cameras[0].fps,
    )

    # Save sim data for diagnostics
    try:
        with open(output_path, "wb") as f:
            # Remove items that work poorly with pickle
            del res["sim"]
            del res["fly"]
            for viz_data in res["viz_data_all"]:
                del viz_data["nn_activities"]
            for info in res["info_hist"]:
                del info["nn_activities"]
            pickle.dump(res, f)
    except Exception as e:
        print(f"Failed to save sim data for {variation_name}_{trial_name}: {e}")


if __name__ == "__main__":
    from joblib import Parallel, delayed

    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "videos").mkdir(exist_ok=True)
    (output_dir / "sim_data").mkdir(exist_ok=True)
    (output_dir / "figs").mkdir(exist_ok=True)

    # Run trials in parallel
    configs = [
        (terrain_type, stabilization_on, cells_selection, (-5, y_pos))
        for terrain_type in ["flat", "blocks"]
        for stabilization_on in [True, False]
        for cells_selection in ["txall", "lc910_inputs"]
        for y_pos in np.linspace(10 - 0.13, 10 + 0.13, 11)
    ]
    Parallel(n_jobs=8)(delayed(process_trial)(*config) for config in configs)
    # process_trial("flat", True, "lc910_inputs", (-5, 10))

    # Visualize trajectories
    trajectories = {
        (terrain_type, stabilization_on, cells_selection): []
        for terrain_type in ["flat", "blocks"]
        for stabilization_on in [True, False]
        for cells_selection in ["txall", "lc910_inputs"]
    }
    for terrain_type, stabilization_on, cells_selection, spawn_xy in configs:
        variation_name = f"{terrain_type}terrain_stabilization{stabilization_on}"
        trial_name = f"{cells_selection}_x{spawn_xy[0]:.4f}y{spawn_xy[1]:.4f}"
        data_path = output_dir / f"sim_data/{variation_name}_{trial_name}.pkl"
        with open(data_path, "rb") as f:
            sim_data = pickle.load(f)
            fly_traj = np.array([obs["fly"][0, :2] for obs in sim_data["obs_hist"]])
        key = (terrain_type, stabilization_on, cells_selection)
        trajectories[key].append(fly_traj.copy())

    viz.plot_fly_following_trajectories(
        trajectories,
        leading_fly_radius,
        leading_fly_speeds,
        output_dir / f"figs/trajectories.pdf",
    )
