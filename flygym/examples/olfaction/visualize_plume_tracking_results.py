import numpy as np
import pickle
import h5py
import gc
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

data_dir = Path("./outputs/plume_tracking/")
(data_dir / "figs").mkdir(exist_ok=True)

dimension_scale_factor = 0.5
plume_simulation_fps = 200
success_radius = 15

# Load the plume data
plume_data_path = data_dir / "plume_dataset/plume.hdf5"
with h5py.File(plume_data_path, "r") as f:
    mean_intensity = np.mean(f["plume"], axis=0)
    inflow_pos = f["inflow_pos"][:] / dimension_scale_factor

# Load fly trajectories
all_res_files = list(data_dir.glob("sim_results/plume_navigation_*_controlFalse.pkl"))
trajectories_all = {}
for file in tqdm(all_res_files, desc="Loading trajectories"):
    with open(file, "rb") as f:
        data = pickle.load(f)
    traj = np.array([obs["fly"][0, :2] for obs in data["obs_hist"]])
    trajectories_all[file.stem] = traj
    del data
    gc.collect()  # Force garbage collection to avoid memory fragmentation
with open(data_dir / "all_trajectories.pkl", "wb") as f:
    pickle.dump(trajectories_all, f)
# with open(data_dir / "all_trajectories.pkl", "rb") as f:
#     trajectories_all = pickle.load(f)

# Check which trials are successful
successful_trials = {}
for trial, traj in trajectories_all.items():
    dist_to_target = np.linalg.norm(traj - inflow_pos, axis=1)
    dist_argmin = np.argmin(dist_to_target)
    if dist_to_target[dist_argmin] < success_radius:
        successful_trials[trial] = traj[: dist_argmin + 1]
print(f"Number of successful trials: {len(successful_trials)}")
print("Successful trials:")
for key in successful_trials.keys():
    print(f" * {key}")


# Plot the mean intensity
arena_height = mean_intensity.shape[0] * dimension_scale_factor
arena_width = mean_intensity.shape[1] * dimension_scale_factor
fig, ax = plt.subplots(figsize=(8, 5))
ax.imshow(
    mean_intensity,
    origin="lower",
    cmap="Blues",
    extent=[0, arena_width, 0, arena_height],
    vmin=0,
    vmax=0.3,
)

# Plot inflow / target position
ax.plot([inflow_pos[0]], [inflow_pos[1]], marker="o", markersize=5, color="black")
thetas = np.linspace(0, 2 * np.pi, 100)
ax.plot(
    inflow_pos[0] + success_radius * np.cos(thetas),
    inflow_pos[1] + success_radius * np.sin(thetas),
    color="black",
    linestyle="--",
    lw=1,
)

for trial, traj in successful_trials.items():
    ax.plot(traj[:, 0], traj[:, 1], label=trial, lw=1)
# ax.legend(frameon=False)
ax.set_xlim(0, arena_width)
ax.set_ylim(0, arena_height)
fig.savefig(data_dir / "figs/trajectory_plot.pdf", dpi=300)


# For a chosen example, trim the video so that it ends at the target
def trim_video_by_fraction(input_file: Path, output_file: Path, fraction: float):
    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_keep = int(total_frames * fraction)
    print(f"Total frames: {total_frames}, keeping {frames_to_keep}")

    out = cv2.VideoWriter(str(output_file), codec, fps, (frame_width, frame_height))
    count = 0
    while count < frames_to_keep:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            count += 1
        else:
            break
    cap.release()
    out.release()


chosen_trial = input("Enter the trial to visualize (one of the trials above): ")
assert chosen_trial in successful_trials, "Chosen trial is not successful"
trimmed_len = successful_trials[chosen_trial].shape[0]
total_len = trajectories_all[chosen_trial].shape[0]
fraction_to_keep = trimmed_len / total_len
trim_video_by_fraction(
    data_dir / f"sim_results/{chosen_trial}.mp4",
    data_dir / f"figs/{chosen_trial}_trimmed.mp4",
    fraction_to_keep,
)
