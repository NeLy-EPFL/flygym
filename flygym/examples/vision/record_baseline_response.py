import pickle
import numpy as np
from pathlib import Path
from tqdm import trange
from flygym import YawOnlyCamera, SingleFlySimulation
from flygym.arena import BaseArena, FlatTerrain, BlocksTerrain
from typing import Optional
from dm_control.rl.control import PhysicsError

from flygym.examples.vision import RealisticVisionFly, viz
from flygym.examples.head_stabilization import HeadStabilizationInferenceWrapper
from flygym.examples.head_stabilization import get_head_stabilization_model_paths


contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

# fmt: off
cells = [
    "T1", "T2", "T2a", "T3", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
    "Tm1", "Tm2", "Tm3", "Tm4", "Tm5Y", "Tm5a", "Tm5b", "Tm5c", "Tm9", "Tm16", "Tm20",
    "Tm28", "Tm30", "TmY3", "TmY4", "TmY5a", "TmY9", "TmY10", "TmY13", "TmY14", "TmY15",
    "TmY18"
]
# fmt: on

output_dir = Path("./outputs/connectome_constrained_vision/baseline_response/")

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
stabilization_model_path, scaler_param_path = get_head_stabilization_model_paths()

if not stabilization_model_path.exists() or not scaler_param_path.exists():
    import warnings

    warnings.warn(
        "Head stabilization model not found. " "Pre-trained model will be used instead."
    )

    stabilization_model_path, scaler_param_path = get_head_stabilization_model_paths()


def run_simulation(
    arena: BaseArena,
    run_time: float = 1.0,
    head_stabilization_model: Optional[HeadStabilizationInferenceWrapper] = None,
):
    fly = RealisticVisionFly(
        contact_sensor_placements=contact_sensor_placements,
        enable_adhesion=True,
        vision_refresh_rate=500,
        neck_kp=1000,
        head_stabilization_model=head_stabilization_model,
    )

    cam = YawOnlyCamera(
        attachment_point=fly.model.worldbody,
        camera_name="camera_top",
        targeted_fly_names=fly.name,
        play_speed=0.1,
    )

    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        arena=arena,
    )

    sim.reset(seed=0)
    obs_hist = []
    info_hist = []
    viz_data_all = []

    # Main simulation loop
    for _ in trange(int(run_time / sim.timestep)):
        try:
            obs, _, _, _, info = sim.step(action=np.array([1, 1]))
        except PhysicsError:
            print("Physics error, ending simulation early")
            break
        obs_hist.append(obs)
        info_hist.append(info)
        rendered_img = sim.render()[0]
        if rendered_img is not None:
            viz_data = {
                "rendered_image": rendered_img,
                "vision_observation": obs["vision"],
                "nn_activities": info["nn_activities"],
            }
            viz_data_all.append(viz_data)

    return {
        "sim": sim,
        "fly": fly,
        "obs_hist": obs_hist,
        "info_hist": info_hist,
        "viz_data_all": viz_data_all,
    }


def process_trial(terrain_type: str, stabilization_on: bool):
    variation_name = f"{terrain_type}terrain_stabilization{stabilization_on}"

    if terrain_type == "flat":
        arena = FlatTerrain()
    elif terrain_type == "blocks":
        arena = BlocksTerrain(
            height_range=(0.2, 0.2), x_range=(-5, 45), y_range=(-40, 40)
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

    # Run simulation
    res = run_simulation(
        arena=arena, run_time=3.0, head_stabilization_model=stabilization_model
    )

    # Save visualization
    viz.visualize_vision(
        Path(output_dir / f"{variation_name}_vision_simulation.mp4"),
        res["fly"].retina,
        res["fly"].retina_mapper,
        viz_data_all=res["viz_data_all"],
        fps=res["sim"].cameras[0].fps,
    )

    # Save median and std of response for each cell
    response_stats = {}
    for cell in cells:
        response_all = np.array(
            [info["nn_activities"][cell] for info in res["info_hist"]]
        )
        response_mean = np.mean(response_all, axis=0)
        response_std = np.std(response_all, axis=0)
        response_stats[cell] = {
            "mean": res["fly"].retina_mapper.flyvis_to_flygym(response_mean),
            "std": res["fly"].retina_mapper.flyvis_to_flygym(response_std),
        }
    with open(output_dir / f"{variation_name}_response_stats.pkl", "wb") as f:
        pickle.dump(response_stats, f)


if __name__ == "__main__":
    from joblib import Parallel, delayed

    output_dir.mkdir(exist_ok=True, parents=True)

    configs = [
        (terrain_type, stabilization_on)
        for terrain_type in ["flat", "blocks"]
        for stabilization_on in [True, False]
    ]

    Parallel(n_jobs=-2)(delayed(process_trial)(*config) for config in configs)
