import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import lightning as pl
from torch.utils.data import DataLoader, ConcatDataset, random_split
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import r2_score
from pathlib import Path
from copy import deepcopy

import flygym
from flygym.examples.head_stabilization import WalkingDataset, ThreeLayerMLP


# Setups
base_dir = sim_data_dir = Path(
    "/home/sibwang/Projects/flygym/outputs/head_stabilization/"
)
retrain = True
feature_selection = True
max_epochs = 3


# Setup paths etc
(base_dir / "logs").mkdir(exist_ok=True, parents=True)
(base_dir / "models").mkdir(exist_ok=True, parents=True)
(base_dir / "figs").mkdir(exist_ok=True, parents=True)


# Torch setup
pl.pytorch.seed_everything(0, workers=True)
torch.set_float32_matmul_precision("medium")


# Matplotlib setup
plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42


# Load datasets
sim_data_pickles = {
    gait: sim_data_dir / "random_exploration" / gait / "sim_data.pkl"
    for gait in ["tripod", "tetrapod", "wave"]
}
datasets = {}
joint_angle_scaler = None
for gait, path in sim_data_pickles.items():
    ds = WalkingDataset(path, joint_angle_scaler=joint_angle_scaler)
    joint_angle_scaler = ds.joint_angle_scaler
    datasets[gait] = ds


# Train model
if retrain:
    logger = TensorBoardLogger(base_dir / "logs", name="three_layer_mlp")

    ds_all_gaits = ConcatDataset(list(datasets.values()))
    train_ds, val_ds = random_split(ds_all_gaits, [0.8, 0.2])
    train_loader = DataLoader(train_ds, batch_size=256, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1028, num_workers=4)

    model = ThreeLayerMLP(input_size=42 + 6, hidden_size=8, output_size=2)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)
    torch.save(model.state_dict(), base_dir / "models/three_layer_mlp.pth")
else:
    model = ThreeLayerMLP(input_size=42 + 6, hidden_size=8, output_size=2)
    model.load_state_dict(torch.load(base_dir / "models/three_layer_mlp.pth"))


# Make predictions
predictions = {}
performances = {}
for gait, ds in datasets.items():
    x_all = torch.tensor(np.concatenate([ds.joint_angles, ds.contact_mask], axis=1))
    y_pred = model(x_all).detach().numpy()
    predictions[gait] = y_pred
    y_true = ds.roll_pitch_ts
    r2_roll = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_pitch = r2_score(y_true[:, 1], y_pred[:, 1])
    performances[gait] = {"roll": r2_roll, "pitch": r2_pitch}


# Visualize predictions
fig, axs = plt.subplots(
    3, 2, figsize=(8, 6), tight_layout=True, sharex=True, sharey=True
)
for i, (gait, ds) in enumerate(datasets.items()):
    t_grid = (np.arange(len(ds)) + ds.ignore_first_n) * 1e-4
    for j, dof in enumerate(["roll", "pitch"]):
        axs[i, j].plot(
            t_grid,
            np.rad2deg(datasets[gait].roll_pitch_ts[:, j]),
            lw=1,
            color="black",
            label="Actual",
        )
        axs[i, j].plot(
            t_grid,
            np.rad2deg(predictions[gait][:, j]),
            lw=1,
            color="tab:red",
            label="Predicted",
            linestyle="--",
        )
        axs[i, j].text(
            1.0,
            0.01,
            f"$R^2$={performances[gait][dof]:.2f}",
            ha="right",
            va="bottom",
            transform=axs[i, j].transAxes,
        )
        axs[i, j].set_ylim(-15, 15)
        axs[i, j].set_xlim(51, 52)
        if i == 0:
            axs[i, j].set_title(rf"Thorax {dof}", size=12)
        if j == 0:
            axs[i, j].text(
                -0.3,
                0.5,
                f"{gait.title()} gait",
                size=12,
                va="center",
                rotation=90,
                transform=axs[i, j].transAxes,
            )
        if i == 2:
            axs[i, j].set_xlabel("Time [s]")
        if j == 0:
            axs[i, j].set_ylabel(r"Angle [$^\circ$]")
        if i == 0 and j == 0:
            axs[i, j].legend(frameon=False)
        sns.despine(ax=axs[i, j])
fig.savefig(base_dir / f"figs/three_layer_mlp.pdf")


# Select features based on importance
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
    mask = [dof in dof_subset for dof in flygym.preprogrammed.all_leg_dofs]
    mask = []
    for dof in flygym.preprogrammed.all_leg_dofs:
        to_include = False
        for dof_to_include in dof_subset:
            if dof.endswith(dof_to_include):
                to_include = True
                break
        mask.append(to_include)

    return np.array(mask)


if feature_selection:
    models_all = {}
    predictions_all = {}
    performances_all = {}

    for dof_subset_tag, dofs in dof_subsets.items():
        print(f"Processing {dof_subset_tag} model...")

        # Load and mask out dataset
        dof_mask = subset_to_mask(dofs)
        masked_datasets = deepcopy(datasets)
        for ds in masked_datasets.values():
            ds.joint_angles[:, ~dof_mask] = 0

        # Train model
        ds_all_gaits = ConcatDataset(list(datasets.values()))
        train_ds, val_ds = random_split(ds_all_gaits, [0.8, 0.2])
        train_loader = DataLoader(train_ds, batch_size=256, num_workers=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1028, num_workers=4)
        model = ThreeLayerMLP(input_size=42 + 6, hidden_size=8, output_size=2)
        logger = TensorBoardLogger(
            base_dir / "logs", name="three_layer_mlp_ablation", version=dof_subset_tag
        )
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=max_epochs,
            check_val_every_n_epoch=1,
            deterministic=True,
        )
        trainer.fit(model, train_loader, val_loader)
        models_all[dof_subset_tag] = model
        torch.save(
            model.state_dict(),
            base_dir / f"models/three_layer_mlp_{dof_subset_tag}.pth",
        )
        print(f"Done processing {dof_subset_tag} model.")

        # Evaluate model
        predictions_all[dof_subset_tag] = {}
        performances_all[dof_subset_tag] = {}
        for gait, ds in masked_datasets.items():
            x_all = torch.tensor(
                np.concatenate([ds.joint_angles, ds.contact_mask], axis=1)
            )
            y_pred = model(x_all).detach().numpy()
            predictions_all[dof_subset_tag][gait] = y_pred
            y_true = ds.roll_pitch_ts
            r2_roll = r2_score(y_true[:, 0], y_pred[:, 0])
            r2_pitch = r2_score(y_true[:, 1], y_pred[:, 1])
            performances_all[dof_subset_tag][gait] = {
                "roll": r2_roll,
                "pitch": r2_pitch,
            }
            print(f"  {gait} gait: r2_roll: {r2_roll:.3f}, r2_pitch: {r2_pitch:.3f}")

    # Visualize results
    for dof_subset_tag in dof_subsets.keys():
        fig, axs = plt.subplots(
            3, 2, figsize=(8, 6), tight_layout=True, sharex=True, sharey=True
        )
        for i, (gait, ds) in enumerate(datasets.items()):
            t_grid = (np.arange(len(ds)) + ds.ignore_first_n) * 1e-4
            for j, dof in enumerate(["roll", "pitch"]):
                axs[i, j].plot(
                    t_grid,
                    np.rad2deg(datasets[gait].roll_pitch_ts[:, j]),
                    lw=1,
                    color="black",
                    label="Actual",
                )
                axs[i, j].plot(
                    t_grid,
                    np.rad2deg(predictions_all[dof_subset_tag][gait][:, j]),
                    lw=1,
                    color="tab:red",
                    label="Predicted",
                    linestyle="--",
                )
                axs[i, j].text(
                    1.0,
                    0.01,
                    f"$R^2$={performances_all[dof_subset_tag][gait][dof]:.2f}",
                    ha="right",
                    va="bottom",
                    transform=axs[i, j].transAxes,
                )
                # axs[i, j].set_title(rf"{gait.title()} gait, thorax {dof}")
                axs[i, j].set_ylim(-15, 15)
                axs[i, j].set_xlim(51, 52)
                if i == 0:
                    axs[i, j].set_title(rf"Thorax {dof}", size=12)
                if j == 0:
                    axs[i, j].text(
                        -0.3,
                        0.5,
                        f"{gait.title()} gait",
                        size=12,
                        va="center",
                        rotation=90,
                        transform=axs[i, j].transAxes,
                    )
                if i == 2:
                    axs[i, j].set_xlabel("Time [s]")
                if j == 0:
                    axs[i, j].set_ylabel(r"Angle [$^\circ$]")
                if i == 0 and j == 0:
                    axs[i, j].legend(frameon=False)
                sns.despine(ax=axs[i, j])
        fig.suptitle(f"DoF selection: {dof_subset_tag}", fontweight="bold")
        fig.savefig(base_dir / f"figs/three_layer_mlp_{dof_subset_tag}.pdf")
