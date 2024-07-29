import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
import lightning as pl
import pickle
from torch.utils.data import DataLoader, ConcatDataset, random_split
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from shutil import copyfile
from sklearn.metrics import r2_score, root_mean_squared_error
from pathlib import Path
from copy import deepcopy

import flygym
import flygym.examples.head_stabilization.viz as viz
from flygym.examples.head_stabilization import WalkingDataset, ThreeLayerMLP


base_dir = Path("./outputs/head_stabilization/")


def subset_to_mask(dof_subset):
    _dof_name_lookup = {
        "ThC_pitch": "Coxa",
        "ThC_roll": "Coxa_roll",
        "ThC_yaw": "Coxa_yaw",
        "CTr_pitch": "Femur",
        "CTr_roll": "Femur_roll",
        "FTi_pitch": "Tibia",
        "TiTa_pitch": "Tarsus1",
    }
    dof_subset = [_dof_name_lookup[dof] for dof in dof_subset]
    mask = []
    for dof in flygym.preprogrammed.all_leg_dofs:
        to_include = False
        for dof_to_include in dof_subset:
            if dof.endswith(dof_to_include):
                to_include = True
                break
        mask.append(to_include)
    return np.array(mask)


def make_concat_subdataset(individual_subdatasets, dofs):
    joint_mask = subset_to_mask(dofs)
    dataset_list = []
    for gait, dict_ in individual_subdatasets.items():
        for terrain, dict_ in dict_.items():
            for dn_drive, ds in dict_.items():
                ds = deepcopy(ds)
                ds.joint_mask = joint_mask
                dataset_list.append(ds)
    return ConcatDataset(dataset_list)


def train_model(
    train_ds: WalkingDataset,
    dofs: list[str],
    trial_name: str,
    max_epochs: int = 20,
    num_workers: int = 8,
):
    pl.pytorch.seed_everything(0, workers=True)

    # Mask out dofs in features
    train_ds = deepcopy(train_ds)
    train_ds.joint_mask = subset_to_mask(dofs)

    # Subdivide training set into training and validation sets
    train_ds, val_ds = random_split(train_ds, [0.8, 0.2])
    train_loader = DataLoader(
        train_ds, batch_size=256, num_workers=num_workers, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1028, num_workers=num_workers, shuffle=False
    )

    # Train model
    logger = TensorBoardLogger(base_dir / "logs", name=trial_name)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=base_dir / "models/checkpoints",
        filename="%s-{epoch:02d}-{val_loss:.2f}" % trial_name,
        save_top_k=1,  # Save only the best checkpoint
        mode="min",  # `min` for minimizing the validation loss
    )
    model = ThreeLayerMLP()
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)

    return model, checkpoint_callback.best_model_path


def evaluate_model(
    individual_test_datasets: list[WalkingDataset],
    dofs: list[str],
    model: ThreeLayerMLP,
):
    stats = []
    for i, ds in enumerate(individual_test_datasets):
        joint_angles_mask = subset_to_mask(dofs)
        joint_angles = ds.joint_angles.copy()
        joint_angles[:, ~joint_angles_mask] = 0
        x = torch.tensor(
            np.concatenate([joint_angles, ds.contact_mask], axis=1),
            device=model.device,
        )
        y = ds.roll_pitch_ts
        y_pred = model(x).detach().cpu().numpy()
        r2_roll = r2_score(y[:, 0], y_pred[:, 0])
        r2_pitch = r2_score(y[:, 1], y_pred[:, 1])
        rmse_roll = root_mean_squared_error(y[:, 0], y_pred[:, 0])
        rmse_pitch = root_mean_squared_error(y[:, 1], y_pred[:, 1])
        stats.append(
            {
                "r2_roll": r2_roll,
                "r2_pitch": r2_pitch,
                "rmse_roll": rmse_roll,
                "rmse_pitch": rmse_pitch,
                "gait": ds.gait,
                "terrain": ds.terrain,
                "subset": ds.subset,
                "dn_drive": ds.dn_drive,
            }
        )
    return pd.DataFrame.from_dict(stats)


def load_datasets(base_dir, excluded_videos, joint_angle_scaler):
    individual_datasets = {}
    for subset in ["train", "test"]:
        individual_datasets[subset] = {}
        for gait in ["tripod", "tetrapod", "wave"]:
            individual_datasets[subset][gait] = {}
            for terrain in ["flat", "blocks"]:
                individual_datasets[subset][gait][terrain] = {}
                paths = base_dir.glob(
                    f"random_exploration/{gait}_{terrain}_{subset}_set_*"
                )
                dn_drives = ["_".join(p.name.split("_")[-2:]) for p in paths]
                for dn_drive in dn_drives:
                    if (gait, terrain, subset, dn_drive) in excluded_videos:
                        print("skipping dataset because fly flipped")
                        continue
                    sim = f"{gait}_{terrain}_{subset}_set_{dn_drive}"
                    path = base_dir / f"random_exploration/{sim}/sim_data.pkl"
                    ds = WalkingDataset(path, joint_angle_scaler=joint_angle_scaler)
                    if ds.contains_fly_flip or ds.contains_physics_error:
                        continue
                    individual_datasets[subset][gait][terrain][dn_drive] = ds
    return individual_datasets


# fmt: off
dof_subsets = {
    "All": ["ThC_pitch", "ThC_roll", "ThC_yaw", "CTr_pitch", "CTr_roll", "FTi_pitch", "TiTa_pitch"],
    "~(ThC pitch)": ["ThC_roll", "ThC_yaw", "CTr_pitch", "CTr_roll", "FTi_pitch", "TiTa_pitch"],
    "~(ThC roll)": ["ThC_pitch", "ThC_yaw", "CTr_pitch", "CTr_roll", "FTi_pitch", "TiTa_pitch"],
    "~(ThC yaw)": ["ThC_pitch", "ThC_roll", "CTr_pitch", "CTr_roll", "FTi_pitch", "TiTa_pitch"],
    "~(CTr pitch)": ["ThC_pitch", "ThC_roll", "ThC_yaw", "CTr_roll", "FTi_pitch", "TiTa_pitch"],
    "~(CTr roll)": ["ThC_pitch", "ThC_roll", "ThC_yaw", "CTr_pitch", "FTi_pitch", "TiTa_pitch"],
    "~(FTi pitch)": ["ThC_pitch", "ThC_roll", "ThC_yaw", "CTr_pitch", "CTr_roll", "TiTa_pitch"],
    "~(TiTa pitch)": ["ThC_pitch", "ThC_roll", "ThC_yaw", "CTr_pitch", "CTr_roll", "FTi_pitch"],
    "~(ThC all)": ["CTr_pitch", "CTr_roll", "FTi_pitch", "TiTa_pitch"],
    "~(CTr both)": ["ThC_pitch", "ThC_roll", "ThC_yaw", "FTi_pitch", "TiTa_pitch"],
    "ThC pitch": ["ThC_pitch"],
    "ThC roll": ["ThC_roll"],
    "ThC yaw": ["ThC_yaw"],
    "CTr pitch": ["CTr_pitch"],
    "CTr roll": ["CTr_roll"],
    "FTi pitch": ["FTi_pitch"],
    "TiTa pitch": ["TiTa_pitch"],
    "ThC all": ["ThC_pitch", "ThC_roll", "ThC_yaw"],
    "CTr both": ["CTr_pitch", "CTr_roll"],
    "None": [],
}
# fmt: on


# Exclude these videos: fly flips
excluded_videos = [
    # ("wave", "blocks", "train", "1.12_0.64"),
    # ("tripod", "blocks", "test", "1.14_0.58"),
]


if __name__ == "__main__":
    # Setups
    retrain_base = True
    retrain_feature_selection = True

    # Setup paths etc
    (base_dir / "logs").mkdir(exist_ok=True, parents=True)
    (base_dir / "models").mkdir(exist_ok=True, parents=True)
    (base_dir / "models/checkpoints").mkdir(exist_ok=True, parents=True)
    (base_dir / "models/stats").mkdir(exist_ok=True, parents=True)
    (base_dir / "figs").mkdir(exist_ok=True, parents=True)

    # Torch setup
    torch.set_float32_matmul_precision("medium")

    # Get joint angle scaler (use any one dataset as this doesn't have to be precise)
    _ds = WalkingDataset(
        base_dir / "random_exploration/tripod_flat_train_set_1.00_1.00/sim_data.pkl"
    )
    joint_angle_scaler = _ds.joint_angle_scaler
    with open(base_dir / "models/joint_angle_scaler_params.pkl", "wb") as f:
        pickle.dump({"mean": joint_angle_scaler.mean, "std": joint_angle_scaler.std}, f)

    # Load datasets
    individual_datasets = load_datasets(base_dir, excluded_videos, joint_angle_scaler)

    # Train or load model
    if retrain_base:
        concat_training_set = make_concat_subdataset(
            individual_datasets["train"], dof_subsets["All"]
        )
        model, best_ckpt = train_model(
            concat_training_set, dof_subsets["All"], "three_layer_mlp"
        )
        copyfile(best_ckpt, base_dir / "models" / f"three_layer_mlp.ckpt")
    else:
        model = ThreeLayerMLP.load_from_checkpoint(
            base_dir / "models/three_layer_mlp.ckpt",
        )

    # Visualize results
    viz.visualize_one_dataset(
        model,
        individual_datasets["test"],
        output_path=base_dir / "figs/three_layer_mlp.pdf",
        dn_drive="0.94_1.02",
    )
    all_test_datasets = [
        ds
        for gait, dict_ in individual_datasets["test"].items()
        for terrain, dict_ in dict_.items()
        for dn_drive, ds in dict_.items()
    ]
    test_perf = evaluate_model(all_test_datasets, dof_subsets["All"], model)
    test_perf.to_csv(
        base_dir / "models/stats/three_layer_mlp_test_perf.csv", index=False
    )

    # Feature selection
    for dof_subset_tag, dofs in dof_subsets.items():
        if retrain_feature_selection:
            print(f"Training model for {dof_subset_tag}")
            concat_training_set = make_concat_subdataset(
                individual_datasets["train"], dofs
            )
            model, best_ckpt = train_model(concat_training_set, dofs, dof_subset_tag)
            copyfile(best_ckpt, base_dir / "models" / f"{dof_subset_tag}.ckpt")
        else:
            model = ThreeLayerMLP.load_from_checkpoint(
                base_dir / "models" / f"{dof_subset_tag}.ckpt"
            )

        viz.visualize_one_dataset(
            model,
            individual_datasets["test"],
            output_path=base_dir / f"figs/{dof_subset_tag}.pdf",
            dn_drive="0.94_1.02",
            dof_subset_tag=dof_subset_tag,
            joint_angles_mask=subset_to_mask(dofs),
        )
        test_perf = evaluate_model(all_test_datasets, dofs, model)
        test_perf.to_csv(base_dir / f"models/stats/{dof_subset_tag}.csv", index=False)

    # Make bar plot for feature selection results
    perf_dfs = []
    for dof_subset_tag in dof_subsets.keys():
        test_perf = pd.read_csv(base_dir / f"models/stats/{dof_subset_tag}.csv")
        test_perf["dof_subset_tag"] = dof_subset_tag
        perf_dfs.append(test_perf)
    all_test_perf = pd.concat(perf_dfs, ignore_index=True)
    assert (all_test_perf["subset"] == "test").all()  # Ensure this is entirely test set
    viz.make_feature_selection_summary_plot(
        all_test_perf[all_test_perf["terrain"] == "flat"],
        base_dir / "figs/feature_selection_flat.pdf",
        title="Flat terrain",
    )
    viz.make_feature_selection_summary_plot(
        all_test_perf[all_test_perf["terrain"] == "blocks"],
        base_dir / "figs/feature_selection_blocks.pdf",
        title="Blocks terrain",
    )
