import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from flygym.util.data import ommatidia_id_map_path
from typing import List
from pathlib import Path
from matplotlib.animation import FuncAnimation


ommatidia_id_map = np.load(ommatidia_id_map_path)
num_pixels_per_ommatidia = np.unique(ommatidia_id_map, return_counts=True)[1][1:]


@nb.njit(parallel=True)
def raw_image_to_hex_pxls(img_arr, num_pixels_per_ommatidia, ommatidia_id_map):
    vals = np.zeros((len(num_pixels_per_ommatidia), 2))
    img_arr_flat = img_arr.reshape((-1, 3))
    hex_id_map_flat = ommatidia_id_map.flatten()
    for i in nb.prange(hex_id_map_flat.size):
        hex_pxl_id = hex_id_map_flat[i] - 1
        if hex_pxl_id != -1:
            ch_idx = hex_pxl_id % 2
            vals[hex_pxl_id, ch_idx] += (
                img_arr_flat[i, ch_idx + 1] / num_pixels_per_ommatidia[hex_pxl_id]
            )
    return vals


@nb.njit(parallel=True)
def hex_pxls_to_human_readable(vals, ommatidia_id_map):
    processed_image_flat = np.zeros(ommatidia_id_map.size, dtype=np.uint8) + 255
    hex_id_map_flat = ommatidia_id_map.flatten().astype(np.int16)
    for i in nb.prange(hex_id_map_flat.size):
        hex_pxl_id = hex_id_map_flat[i] - 1
        if hex_pxl_id != -1:
            hex_pxl_val = vals[hex_pxl_id, :].max()
            processed_image_flat[i] = hex_pxl_val
    return processed_image_flat.reshape(ommatidia_id_map.shape)


def visualize_visual_input(
    output_path: Path,
    vision_data_li: List[np.ndarray],
    raw_vision_data_li: List[np.ndarray],
    vision_update_mask: np.ndarray,
    vision_refresh_rate: float = 500,
    playback_speed: float = 0.1,
):
    vision_data_key_frames = np.array(vision_data_li)[vision_update_mask, :, :, :]
    raw_vision_key_frames = np.array(raw_vision_data_li)[vision_update_mask, :, :, :]
    num_frames = vision_data_key_frames.shape[0]

    # Convert hex pixels back to human readable pixels
    readable_processed_images = []
    for i in range(num_frames):
        frame_data = vision_data_key_frames[i, :, :]
        left_img = hex_pxls_to_human_readable(frame_data[0, :], ommatidia_id_map)
        right_img = hex_pxls_to_human_readable(frame_data[1, :], ommatidia_id_map)
        readable_processed_images.append([left_img, right_img])
    readable_processed_images = np.array(readable_processed_images)

    # Compile video
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    def update(frame):
        for i, side in enumerate(["Left", "Right"]):
            axs[0, i].cla()
            axs[0, i].imshow(raw_vision_key_frames[frame, i, :, :, :])
            axs[0, i].axis("off")
            axs[0, i].set_title(f"{side} eye")

            axs[1, i].cla()
            axs[1, i].imshow(
                readable_processed_images[frame, i, :, :], cmap="gray", vmin=0, vmax=255
            )
            axs[1, i].axis("off")

    playback_speed = playback_speed
    interval_1x_speed = 1000 / vision_refresh_rate
    interval = interval_1x_speed / playback_speed
    animation = FuncAnimation(fig, update, frames=num_frames, interval=interval)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, dpi=100, writer="ffmpeg")
