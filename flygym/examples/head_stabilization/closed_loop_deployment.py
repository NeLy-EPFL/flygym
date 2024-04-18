import pickle
import numpy as np
from pathlib import Path
from tqdm import trange
from flygym import Fly, Camera
from flygym.vision import Retina
from flygym.arena import BaseArena, FlatTerrain, BlocksTerrain
from typing import Optional
from dm_control.rl.control import PhysicsError
from sklearn.metrics import r2_score
from dm_control.utils import transformations

import flygym.examples.head_stabilization.viz as viz
from flygym.examples.vision_connectome_model import NMFRealisticVison, RetinaMapper
from flygym.examples.head_stabilization import HeadStabilizationInferenceWrapper


contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]
stabilization_model_dir = Path("./outputs/head_stabilization/models/")
output_dir = Path("./outputs/head_stabilization/videos/")
output_dir.mkdir(exist_ok=True, parents=True)


def run_simulation(
    arena: BaseArena,
    run_time: float = 1.0,
    head_stabilization_model: Optional[HeadStabilizationInferenceWrapper] = None,
):
    fly = Fly(
        contact_sensor_placements=contact_sensor_placements,
        enable_adhesion=True,
        enable_vision=True,
        vision_refresh_rate=500,
        neck_kp=100,
        head_stabilization_model=head_stabilization_model,
    )

    cameras = [
        Camera(
            fly=fly,
            camera_id="Animat/camera_top_zoomout",
            play_speed=0.2,
            window_size=(600, 600),
            fps=24,
            play_speed_text=False,
        ),
        Camera(
            fly=fly,
            camera_id="Animat/camera_neck_zoomin",
            play_speed=0.2,
            window_size=(600, 600),
            fps=24,
            play_speed_text=False,
        ),
    ]

    sim = NMFRealisticVison(
        fly=fly,
        cameras=cameras,
        arena=arena,
    )

    sim.reset(seed=0)
    birdeye_snapshots = []
    zoomin_snapshots = []
    raw_vision_snapshots = []
    nn_activities_snapshots = []
    neck_actuation_pred_hist = []
    neck_actuation_true_hist = []

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

        rendered_images = sim.render()
        if rendered_images[0] is not None:
            birdeye_snapshots.append(rendered_images[0])
            zoomin_snapshots.append(rendered_images[1])
            raw_vision_snapshots.append(obs["vision"])
            nn_activities_snapshots.append(info["nn_activities"])

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

    return {
        "sim": sim,
        "birdeye": birdeye_snapshots,
        "zoomin": zoomin_snapshots,
        "raw_vision": raw_vision_snapshots,
        "nn_activities": nn_activities_snapshots,
        "r2_scores": r2_scores,
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
        arena = BlocksTerrain(height_range=(0.2, 0.2))
    else:
        raise ValueError("Invalid terrain type")

    # Set up head stabilization model
    if stabilization_on:
        stablization_model = HeadStabilizationInferenceWrapper(
            model_path=stabilization_model_dir / "All.ckpt",
            scaler_param_path=stabilization_model_dir / "joint_angle_scaler_params.pkl",
        )
    else:
        stablization_model = None

    # Run simulation
    sim_res = run_simulation(
        arena=arena, run_time=2.0, head_stabilization_model=stablization_model
    )
    print(
        f"Terrain type {terrain_type}, stabilization {stabilization_on} completed "
        f"with R2 scores: {sim_res['r2_scores']}"
    )
    sim: NMFRealisticVison = sim_res["sim"]
    raw_vision_hist = [
        raw_vision_to_human_readable(sim.fly.retina, x) for x in sim_res["raw_vision"]
    ]
    cell_response_hist = [
        cell_response_to_human_readable(sim.fly.retina, sim.retina_mapper, x, cell)
        for x in sim_res["nn_activities"]
    ]

    return {
        "birdeye": sim_res["birdeye"],
        "zoomin": sim_res["zoomin"],
        "raw_vision": raw_vision_hist,
        "cell_response": cell_response_hist,
    }


if __name__ == "__main__":
    from joblib import Parallel, delayed

    # Run simulation for all configurations
    configs = [
        (terrain_type, stabilization_on, "T4a")
        for terrain_type in ["flat", "blocks"]
        for stabilization_on in [True, False]
    ]
    res_all = Parallel(n_jobs=-1)(delayed(process_trial)(*config) for config in configs)
    res_all = {k[:2]: v for k, v in zip(configs, res_all)}

    # Make summary video
    data = {}
    for stabilization_on in [True, False]:
        for view in ["birdeye", "zoomin", "raw_vision", "cell_response"]:
            # Start with flat terrain
            frames = res_all[("flat", stabilization_on)][view]

            # Pause for 0.5s
            for _ in range(int(24 * 0.5)):
                frames.append(frames[-1])

            # Switch to blocks terrain
            frames += res_all[("blocks", stabilization_on)][view]

            data[(stabilization_on, view)] = frames
    viz.closed_loop_comparison_video(
        data, "T4a", 24, output_dir / "closed_loop_comparison.mp4"
    )
