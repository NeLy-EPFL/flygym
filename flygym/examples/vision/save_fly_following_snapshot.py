# Let's save a snapshot of the simulation (the arena as well as visual
# neuron activities) for the paper.

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

from flygym.examples.vision import MovingFlyArena
from flygym.examples.head_stabilization import HeadStabilizationInferenceWrapper
from flygym.examples.vision.follow_fly_closed_loop import (
    leading_fly_speeds,
    leading_fly_radius,
    baseline_dir,
    output_dir,
    stabilization_model_path,
    scaler_param_path,
    run_simulation,
)


plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

# fmt: off
neurons_txall = [
    "T1", "T2", "T2a", "T3", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
    "Tm1", "Tm2", "Tm3", "Tm4", "Tm5Y", "Tm5a", "Tm5b", "Tm5c", "Tm9", "Tm16", "Tm20",
    "Tm28", "Tm30", "TmY3", "TmY4", "TmY5a", "TmY9", "TmY10", "TmY13", "TmY14", "TmY15",
    "TmY18"
]
# fmt: on


# Run a very short simulation
arena = MovingFlyArena(
    move_speed=leading_fly_speeds["blocks"],
    radius=leading_fly_radius,
    terrain_type="blocks",
)
stabilization_model = HeadStabilizationInferenceWrapper(
    model_path=stabilization_model_path,
    scaler_param_path=scaler_param_path,
)
variation_name = "flatterrain_stabilizationTrue"
with open(baseline_dir / f"{variation_name}_response_stats.pkl", "rb") as f:
    response_stats = pickle.load(f)
res = run_simulation(
    arena,
    tracking_cells=neurons_txall,
    run_time=0.2,
    baseline_response=response_stats,
    z_score_threshold=10,
    tracking_gain=5,
    head_stabilization_model=stabilization_model,
    spawn_xy=(-5, 10),
)


# Plot the arena
img = res["viz_data_all"][-1]["rendered_image"]
cv2.imwrite(
    str(output_dir / "figs/arena_snapshot.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
)


# Plot the system activities
nn_activities = res["viz_data_all"][-1]["nn_activities"]
retina = res["sim"].flies[0].retina
cell_order_str = """
    T1    T2    T2a   T3    T4a   T4b   T4c   T4d
    T5a   T5b   T5c   T5d   Tm1   Tm2   Tm3   Tm4
    Tm5Y  Tm5a  Tm5b  Tm5c  Tm9   Tm16  Tm20  Tm28
    Tm30  TmY3  TmY4  TmY5a TmY9  TmY10 TmY13 TmY14
    TmY15 TmY18
    """
cells = cell_order_str.split()
images = {}
images["obj_score"] = retina.hex_pxls_to_human_readable(
    res["viz_data_all"][-1]["mean_zscore"].T
)
images["obj_score"][retina.ommatidia_id_map == 0] = np.nan
for cell in cells:
    nn_activity = res["sim"].retina_mapper.flyvis_to_flygym(nn_activities[cell])
    img = retina.hex_pxls_to_human_readable(nn_activity.T)
    img[retina.ommatidia_id_map == 0] = np.nan
    images[cell] = img
images["raw"] = retina.hex_pxls_to_human_readable(
    res["viz_data_all"][-1]["vision_observation"].sum(axis=-1).T
)
images["raw"][retina.ommatidia_id_map == 0] = np.nan

fig, axs = plt.subplots(8, 5, figsize=(11, 16), tight_layout=True)
for i, (cell, img) in enumerate(images.items()):
    ax = axs.flat[i]
    if cell == "raw":
        ax.imshow(img[:, :, 1], cmap="gray", vmin=0, vmax=1)
        label = "Raw"
    elif cell == "obj_score":
        ax.imshow(img[:, :, 1], cmap="viridis", vmin=0, vmax=20)
        label = "Obj. score"
    else:
        ax.imshow(img[:, :, 1], cmap="seismic", vmin=-3, vmax=3)
        label = cell
    ax.text(
        0,
        1,
        label,
        size=20,
        va="center",
        transform=ax.transAxes,
    )
    ax.axis("off")
    ax.set_aspect("equal")

for i in range(5):
    axs[7, i].axis("off")

ax = axs[7, 3]
norm = plt.Normalize(vmin=0, vmax=20)
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="horizontal")
cbar.set_label("Object score")

ax = axs[7, 4]
norm = plt.Normalize(vmin=-3, vmax=3)
sm = plt.cm.ScalarMappable(cmap="seismic", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="horizontal")
cbar.set_label("Cell activity")

fig.savefig(output_dir / "figs/visual_neurons_snapshot.pdf", dpi=300)
