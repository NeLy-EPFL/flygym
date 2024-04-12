import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from pathlib import Path

from flygym.vision.retina import Retina
from flygym.examples.head_stabilization import HeadStabilizationInferenceWrapper
from flygym.examples.head_stabilization import ThreeLayerMLP, JointAngleScaler, WalkingDataset


def left_side_median_filter(data, window_size):
    # Ensure window size is odd to have a symmetric filter around the current point when possible
    if window_size % 2 == 0:
        window_size += 1

    # Length of the input data
    n = len(data)

    # Initialize the filtered output array
    filtered_data = np.empty(n)

    # Pad the data at the beginning with NaNs or zeros, depending on your use case
    # Here, NaN is used to denote that there's no data to compute the median in these cases
    padded_data = np.pad(data, (window_size//2, 0), mode='constant', constant_values=np.nan)

    # Compute the median filtering considering only the left side
    for i in range(n):
        # Calculate median from the current point and the left side values only
        filtered_data[i] = np.nanmedian(padded_data[i:i + window_size//2 + 1])

    return filtered_data


with open(
    "outputs/connectome_constrained_vision/complex_terrain/fly_tracking_sim_data.pkl",
    "rb",
) as f:
    data = pickle.load(f)

plt.plot(data["diff_hist"])
plt.plot(left_side_median_filter(data["diff_hist"], 100))
plt.show()

# t3_zscore_hist = np.array(data["t3_zscore_hist"])

# retina = Retina()
# for i in range(10):
#     img = retina.hex_pxls_to_human_readable(t3_zscore_hist[i * 10, 1])
#     plt.imshow(img, vmin=-5, vmax=5, cmap="seismic")
#     # plt.savefig(f"temp/{i}.png")
#     if i == 8:
#         plt.show()
#     # plt.close()

# data["obs_hist"][0].keys()
# joint_obs = np.array([x["joints"][0, -2:] for x in data["obs_hist"]])
# joint_act = np.array(data["neck_actuation_hist"])
# fly_angles = np.array([x["fly"][2, :] for x in data["obs_hist"]])
# tgt_angles = np.array(data["target"])

# print(
#     r2_score(tgt_angles[:, 0], joint_act[:, 0]),
#     r2_score(tgt_angles[:, 1], joint_act[:, 1]),
# )

# # plt.plot(joint_obs)
# plt.plot(joint_act)
# # plt.plot(fly_angles)
# plt.plot(tgt_angles)
# plt.show()


# model_path = "outputs/head_stabilization/models/All.ckpt"
# scaler_path = "outputs/head_stabilization/models/joint_angle_scaler_params.pkl"
# data_path = Path("outputs/head_stabilization/random_exploration/tripod_flat_train_set_1.00_1.00/sim_data.pkl")

# mlp = ThreeLayerMLP.load_from_checkpoint(model_path).cpu()
# with open(scaler_path, "rb") as f:
#     params = pickle.load(f)
#     scaler = JointAngleScaler.from_params(mean=params["mean"], std=params["std"])

# wrapper = HeadStabilizationInferenceWrapper(model_path, scaler_path)

# with open(data_path, "rb") as f:
#     train_sample = pickle.load(f)
# ds = WalkingDataset(data_path, joint_angle_scaler=scaler)

# y_pred_all = []
# for sample in ds:
#     x = np.concatenate([sample["joint_angles"], sample["contact_mask"]])
#     y_pred = mlp(torch.tensor(x[None, :])).detach().cpu().numpy().squeeze()
#     y_pred_all.append(y_pred)
# y_pred_all = np.array(y_pred_all)

1