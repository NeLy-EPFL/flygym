import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from flygym.util.data import ommatidia_id_map_path
from typing import List
from pathlib import Path
from matplotlib.animation import FuncAnimation


# ommatidia_id_map: 2D array of shape (raw_img_height, raw_img_width) where
# the integer pixel value (starting from 1) indicates which ommatidium it
# belongs to. 0 indicates that the pixel is not part of any ommatidium.
ommatidia_id_map = np.load(ommatidia_id_map_path)
# num_pixels_per_ommatidia: 1D array of shape (num_ommatidia,) where the
# numbers indicate how many raw pixels each ommatidium contains. This is
# helpful for calculating the mean per ommatidium.
num_pixels_per_ommatidia = np.unique(ommatidia_id_map, return_counts=True)[1][1:]


@nb.njit(parallel=True)
def raw_image_to_hex_pxls(raw_img, num_pixels_per_ommatidia, ommatidia_id_map):
    """Given a raw image from an eye (one camera), simulate what the fly
    would see.

    Parameters
    ----------
    raw_img : np.ndarray
        RGB image with the shape (H, W, 3) returned by the camera.
    num_pixels_per_ommatidia : np.ndarray
        1D array of shape (num_ommatidia,) where the numbers indicate how
        many raw pixels each ommatidium contains.
    ommatidia_id_map : np.ndarray
        2D array of shape (raw_img_height, raw_img_width) where the integer
        pixel value (starting from 1) indicates which ommatidium it belongs
        to. 0 indicates that the pixel is not part of any ommatidium.

    Returns
    -------
    np.ndarray
        Our simulation of what the fly might see through its compound eyes.
        It is a (N, 2) array where the first dimension is for the N
        ommatidia, and the third dimension is for the two channels.
    """
    vals = np.zeros((len(num_pixels_per_ommatidia), 2))
    img_arr_flat = raw_img.reshape((-1, 3))
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
def hex_pxls_to_human_readable(ommatidia_reading, ommatidia_id_map):
    """Given the intensity readings for all ommatidia in one eye, convert
    them to an (H, W) image with hexagonal blocks that can be visualized as
    a human-readable image.

    Parameters
    ----------
    ommatidia_reading : np.ndarray
        Our simulation of what the fly might see through its compound eyes.
        It is a (N, 2) array where the first dimension is for the N
        ommatidia, and the third dimension is for the two channels.
    ommatidia_id_map : np.ndarray
        2D array of shape (raw_img_height, raw_img_width) where the integer
        pixel value (starting from 1) indicates which ommatidium it belongs
        to. 0 indicates that the pixel is not part of any ommatidium.

    Returns
    -------
    np.ndarray
        An (H, W) grayscale image with hexagonal blocks that can be
        visualized as a human-readable image.
    """
    processed_image_flat = np.zeros(ommatidia_id_map.size, dtype=np.uint8) + 255
    hex_id_map_flat = ommatidia_id_map.flatten().astype(np.int16)
    for i in nb.prange(hex_id_map_flat.size):
        hex_pxl_id = hex_id_map_flat[i] - 1
        if hex_pxl_id != -1:
            hex_pxl_val = ommatidia_reading[hex_pxl_id, :].max()
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
    """Convert lists of vision readings into a video and save it to disk.

    Parameters
    ----------
    output_path : Path
        Path of the output video will be saved. Should end with ".mp4".
    vision_data_li : List[np.ndarray]
        List of ommatidia readings. Each element is an array of shape
        (2, N, 2) where the first dimension is for the left and right eyes,
        the second dimension is for the N ommatidia, and the third
        dimension is for the two channels. The length of this list is the
        number of simulation steps.
    raw_vision_data_li : List[np.ndarray]
        Same as ``vision_data_li`` but with the raw RGB images from the
        cameras instead of the simulated ommatidia readings. The shape of
        each element is therefore (2, H, W, 3) where the first dimension is
        for the left and right eyes, and the remaining dimensions are for
        the RGB image.
    vision_update_mask : np.ndarray
        Mask indicating which simulation steps have vision updates. This
        should be taken from ``NeuroMechFlyMuJoCo.vision_update_mask``.
    vision_refresh_rate : float, optional
        The refresh rate of visual inputs in Hz. This should be consistent
        with ``MuJoCoParameters.vision_refresh_rate`` that is given to the
        simulation. By default 500.
    playback_speed : float, optional
        Speed, as a multiple of the 1x speed, at which the video should be
        rendred, by default 0.1
    """
    vision_data_key_frames = np.array(vision_data_li)[vision_update_mask, :, :, :]
    raw_vision_key_frames = np.array(raw_vision_data_li)[vision_update_mask, :, :, :]
    num_frames = vision_data_key_frames.shape[0]

    # Convert hex pixels back to human readable pixels
    readable_processed_images = []
    for i in range(num_frames):
        frame_data = vision_data_key_frames[i, :, :]
        left_img = hex_pxls_to_human_readable(frame_data[0, :, :], ommatidia_id_map)
        right_img = hex_pxls_to_human_readable(frame_data[1, :, :], ommatidia_id_map)
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
