import numpy as np
from pathlib import Path
from tqdm import trange
from flygym import Camera, NeckCamera, SingleFlySimulation
from flygym.vision import Retina
from flygym.arena import BaseArena, FlatTerrain, BlocksTerrain
from typing import Optional
from dm_control.rl.control import PhysicsError
from sklearn.metrics import r2_score
from dm_control.utils import transformations

import flygym.examples.head_stabilization.viz as viz
from flygym.examples.vision import RealisticVisionFly, RetinaMapper
from flygym.examples.head_stabilization import HeadStabilizationInferenceWrapper


contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]
output_dir = Path("./outputs/head_stabilization/")
(output_dir / "videos").mkdir(exist_ok=True, parents=True)

# If you trained the models yourself (by running ``collect_training_data.py``
# followed by ``train_proprioception_model.py``), you can use the following
# paths to load the models that you trained. Modify the paths if saved the
# model checkpoints elsewhere.
stabilization_model_dir = Path("./outputs/head_stabilization/models/")
stabilization_model_path = stabilization_model_dir / "All.ckpt"
scaler_param_path = stabilization_model_dir / "joint_angle_scaler_params.pkl"

# Alternatively, you can use the pre-trained models that come with the
# package. To do so, comment out the three lines above and uncomment the
# following 2 lines.
# from flygym.examples.head_stabilization import get_head_stabilization_model_paths
# stabilization_model_path, scaler_param_path = get_head_stabilization_model_paths()

# Simulation parameters

run_time = 1.15  # seconds


def run_simulation(
    arena: BaseArena,
    run_time: float = 0.5,
    head_stabilization_model: Optional[HeadStabilizationInferenceWrapper] = None,
):
    fly = RealisticVisionFly(
        contact_sensor_placements=contact_sensor_placements,
        enable_adhesion=True,
        vision_refresh_rate=500,
        neck_kp=500,
        head_stabilization_model=head_stabilization_model,
    )

    birdeye_camera = Camera(
        fly=fly,
        camera_id="Animat/camera_top_zoomout",
        play_speed=0.2,
        window_size=(600, 600),
        fps=24,
        play_speed_text=False,
    )
    birdeye_camera._cam.pos -= np.array([0, 0, 20.0])

    neck_camera = NeckCamera(
        fly=fly,
        play_speed=0.2,
        fps=24,
        window_size=(600, 600),
        camera_follows_fly_orientation=True,
        play_speed_text=False,
    )

    sim = SingleFlySimulation(
        fly=fly, cameras=[birdeye_camera, neck_camera], arena=arena
    )

    sim.reset(seed=0)

    # These are only updated when a frame is rendered. They are used for
    # generating the summary video at the end of the simulation. Each
    # element in the list corresponds to a frame in the video.
    birdeye_snapshots = []
    zoomin_snapshots = []
    raw_vision_snapshots = []
    nn_activities_snapshots = []
    neck_actuation_viz_vars = []

    # These are updated at every time step and are used for generating
    # statistics and plots (except vision_all, which is updated every
    # time step where the visual input is updated. Visual updates are less
    # frequent than physics steps).
    head_rotation_hist = []
    thorax_rotation_hist = []
    neck_actuation_pred_hist = []
    neck_actuation_true_hist = []
    vision_all = []  # (only updated when visual input is updated)

    thorax_body = fly.model.find("body", "Thorax")
    head_body = fly.model.find("body", "Head")

    # Main simulation loop
    for i in trange(int(run_time / sim.timestep)):
        try:
            obs, _, _, _, info = sim.step(action=np.array([1, 1]))
        except PhysicsError:
            print("Physics error, ending simulation early")
            break

        # Record neck actuation for stats at the end of the simulation
        if head_stabilization_model is not None:
            neck_actuation_pred_hist.append(info["neck_actuation"])
        quat = sim.physics.bind(fly.thorax).xquat
        quat_inv = transformations.quat_inv(quat)
        roll, pitch, _ = transformations.quat_to_euler(quat_inv, ordering="XYZ")
        neck_actuation_true_hist.append(np.array([roll, pitch]))

        # Record head and thorax orientation
        thorax_rotation_quat = sim.physics.bind(thorax_body).xquat
        thorax_roll, thorax_pitch, _ = transformations.quat_to_euler(
            thorax_rotation_quat, ordering="XYZ"
        )
        thorax_rotation_hist.append([thorax_roll, thorax_pitch])
        head_rotation_quat = sim.physics.bind(head_body).xquat
        head_roll, head_pitch, _ = transformations.quat_to_euler(
            head_rotation_quat, ordering="XYZ"
        )
        head_rotation_hist.append([head_roll, head_pitch])

        rendered_images = sim.render()
        if rendered_images[0] is not None:
            birdeye_snapshots.append(rendered_images[0])
            zoomin_snapshots.append(rendered_images[1])
            raw_vision_snapshots.append(obs["vision"])
            nn_activities_snapshots.append(info["nn_activities"])
            neck_act = np.zeros(2)
            if head_stabilization_model is not None:
                neck_act = info["neck_actuation"]
            neck_signals = np.hstack(
                [np.rad2deg([roll, pitch]), np.rad2deg(neck_act), [sim.curr_time]]
            )
            neck_actuation_viz_vars.append(neck_signals)

        if info["vision_updated"]:
            vision_all.append(obs["vision"])

    # Generate performance stats on head stabilization
    if head_stabilization_model is not None:
        neck_actuation_true_hist = np.array(neck_actuation_true_hist)
        neck_actuation_pred_hist = np.array(neck_actuation_pred_hist)
        r2_scores = {
            "roll": r2_score(
                neck_actuation_true_hist[:, 0], neck_actuation_pred_hist[:, 0]
            ),
            "pitch": r2_score(
                neck_actuation_true_hist[:, 1], neck_actuation_pred_hist[:, 1]
            ),
        }
    else:
        r2_scores = None
        neck_actuation_true_hist = np.array(neck_actuation_true_hist)
        neck_actuation_pred_hist = np.zeros_like(neck_actuation_true_hist)

    # Compute standard deviation of each ommatidium's intensity
    vision_all = np.array(vision_all).sum(axis=-1)  # sum over both channels
    vision_std = np.std(vision_all, axis=0)
    vision_std_raster = fly.retina.hex_pxls_to_human_readable(vision_std.T)
    vision_std_raster[fly.retina.ommatidia_id_map == 0, :] = np.nan

    return {
        "sim": sim,
        "birdeye": birdeye_snapshots,
        "zoomin": zoomin_snapshots,
        "raw_vision": raw_vision_snapshots,
        "nn_activities": nn_activities_snapshots,
        "neck_true": neck_actuation_true_hist,
        "neck_pred": neck_actuation_pred_hist,
        "neck_actuation": neck_actuation_viz_vars,
        "r2_scores": r2_scores,
        "head_rotation_hist": np.array(head_rotation_hist),
        "thorax_rotation_hist": np.array(thorax_rotation_hist),
        "vision_std": vision_std_raster,
    }


def raw_vision_to_human_readable(retina: Retina, raw_vision: np.ndarray):
    left_raw = raw_vision[0, :, :].max(axis=-1)
    right_raw = raw_vision[1, :, :].max(axis=-1)
    left_img = retina.hex_pxls_to_human_readable(left_raw, color_8bit=False)
    right_img = retina.hex_pxls_to_human_readable(right_raw, color_8bit=False)
    return np.concatenate([left_img[None, :], right_img[None, :]], axis=0)


def cell_response_to_human_readable(
    retina: Retina, retina_mapper: RetinaMapper, nn_activities: np.ndarray, cell: str
):
    left_raw = nn_activities[cell][0, :]
    right_raw = nn_activities[cell][1, :]
    left_mapped = retina_mapper.flyvis_to_flygym(left_raw)
    right_mapped = retina_mapper.flyvis_to_flygym(right_raw)
    left_img = retina.hex_pxls_to_human_readable(left_mapped, color_8bit=False)
    right_img = retina.hex_pxls_to_human_readable(right_mapped, color_8bit=False)
    return np.concatenate([left_img[None, :], right_img[None, :]], axis=0)


def process_trial(terrain_type: str, stabilization_on: bool, cell: str):
    # Set up arena
    if terrain_type == "flat":
        arena = FlatTerrain()
    elif terrain_type == "blocks":
        arena = BlocksTerrain(height_range=(0.2, 0.2), x_range=(-5, 35))
    else:
        raise ValueError("Invalid terrain type")

    # Set up head stabilization model
    if stabilization_on:
        stabilization_model = HeadStabilizationInferenceWrapper(
            model_path=stabilization_model_path,
            scaler_param_path=scaler_param_path,
        )
    else:
        stabilization_model = None

    # Run simulation
    sim_res = run_simulation(
        arena=arena, run_time=run_time, head_stabilization_model=stabilization_model
    )
    print(
        f"Terrain type {terrain_type}, stabilization {stabilization_on} completed "
        f"with R2 scores: {sim_res['r2_scores']}"
    )
    sim: SingleFlySimulation = sim_res["sim"]
    raw_vision_hist = [
        raw_vision_to_human_readable(sim.fly.retina, x) for x in sim_res["raw_vision"]
    ]
    cell_response_hist = [
        cell_response_to_human_readable(sim.fly.retina, sim.fly.retina_mapper, x, cell)
        for x in sim_res["nn_activities"]
    ]

    return {
        "birdeye": sim_res["birdeye"],
        "zoomin": sim_res["zoomin"],
        "raw_vision": raw_vision_hist,
        "cell_response": cell_response_hist,
        "head_rotation": sim_res["head_rotation_hist"],
        "thorax_rotation": sim_res["thorax_rotation_hist"],
        "neck_actuation": sim_res["neck_actuation"],
        "vision_std": sim_res["vision_std"],
    }


if __name__ == "__main__":
    from joblib import Parallel, delayed

    # Run simulation for all configurations
    configs = [
        (terrain_type, stabilization_on, "T4a")
        for terrain_type in ["flat", "blocks"]
        for stabilization_on in [True, False]
    ]
    res_all = Parallel(n_jobs=4)(delayed(process_trial)(*config) for config in configs)
    res_all = {k[:2]: v for k, v in zip(configs, res_all)}
    # res_all = {config[:2]: process_trial(*config) for config in configs}

    # Make summary video
    data = {}
    for stabilization_on in [True, False]:
        for view in ["birdeye", "zoomin", "raw_vision", "neck_actuation"]:
            # Start with flat terrain
            frames = res_all[("flat", stabilization_on)][view]

            # Pause for 0.5s
            for _ in range(int(24 * 0.5)):
                frames.append(frames[-1])

            # Switch to blocks terrain
            frames += res_all[("blocks", stabilization_on)][view]
            data[(stabilization_on, view)] = frames
    viz.closed_loop_comparison_video(
        data, 24, output_dir / "videos/closed_loop_comparison.mp4", run_time
    )

    # Plot example head and thorax rotation time series
    rotation_data = {}
    for terrain_type in ["flat", "blocks"]:
        rotation_data[terrain_type] = {
            body: res_all[(terrain_type, True)][f"{body}_rotation"]
            for body in ["head", "thorax"]
        }
    viz.plot_rotation_time_series(
        rotation_data, output_dir / "figs/rotation_time_series.pdf"
    )

    # Plot standard deviation of intensity per ommatidium with and
    # without head stabilization
    std_data = {}
    for terrain_type in ["flat", "blocks"]:
        for stabilization_on in [True, False]:
            std_data[(terrain_type, stabilization_on)] = res_all[
                (terrain_type, stabilization_on)
            ]["vision_std"]
    viz.plot_activities_std(std_data, output_dir / "figs/vision_std.pdf")
