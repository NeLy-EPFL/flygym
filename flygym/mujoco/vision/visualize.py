import numpy as np
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
from matplotlib.animation import FuncAnimation

from .retina import Retina


def visualize_visual_input(
    retina: Retina,
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
        left_img = retina.hex_pxls_to_human_readable(frame_data[0, :, :])
        right_img = retina.hex_pxls_to_human_readable(frame_data[1, :, :])
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
                readable_processed_images[frame, i, :, :],
                cmap="gray",
                vmin=0,
                vmax=255,
            )
            axs[1, i].axis("off")

    playback_speed = playback_speed
    interval_1x_speed = 1000 / vision_refresh_rate
    interval = interval_1x_speed / playback_speed
    animation = FuncAnimation(fig, update, frames=num_frames, interval=interval)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, dpi=100, writer="ffmpeg")