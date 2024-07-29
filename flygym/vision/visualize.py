import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
import logging
from pathlib import Path
from matplotlib.animation import FuncAnimation

from .retina import Retina


def add_insets(retina, viz_frame, visual_input, panel_height=150):
    """Add insets to the visualization frame.

    Parameters
    ----------
    retina : Retina
        The retina object used to generate the visual input.
    viz_frame : np.ndarray
        The visualization frame to add insets to.
    visual_input : np.ndarray
        The visual input to the retina. Should be of shape (2, N, 2) as
        returned in the observation of the environment (``obs["vision"]``).
    panel_height : int, optional
        Height of the panel that contains the insets, by default 150.

    Returns
    -------
    np.ndarray
        The visualization frame with insets added.
    """
    final_frame = np.zeros(
        (viz_frame.shape[0] + panel_height + 5, viz_frame.shape[1], 3), dtype=np.uint8
    )
    final_frame[: viz_frame.shape[0], :, :] = viz_frame

    img_l = (
        retina.hex_pxls_to_human_readable(visual_input[0, :, :], color_8bit=True)
        .astype(np.uint8)
        .max(axis=-1)
    )
    img_r = (
        retina.hex_pxls_to_human_readable(visual_input[1, :, :], color_8bit=True)
        .astype(np.uint8)
        .max(axis=-1)
    )
    vision_inset_size = np.array(
        [panel_height, panel_height * (img_l.shape[1] / img_l.shape[0])]
    ).astype(np.uint16)

    img_l = cv2.resize(img_l, vision_inset_size[::-1])
    img_r = cv2.resize(img_r, vision_inset_size[::-1])
    mask = cv2.resize(
        (retina.ommatidia_id_map > 0).astype(np.uint8), vision_inset_size[::-1]
    ).astype(bool)
    img_l[~mask] = 0
    img_r[~mask] = 0
    img_l = np.repeat(img_l[:, :, np.newaxis], 3, axis=2)
    img_r = np.repeat(img_r[:, :, np.newaxis], 3, axis=2)
    vision_inset = np.zeros(
        (panel_height, vision_inset_size[1] * 2 + 10, 3), dtype=np.uint8
    )
    vision_inset[:, : vision_inset_size[1], :] = img_l
    vision_inset[:, vision_inset_size[1] + 10 :, :] = img_r
    col_start = int((viz_frame.shape[1] - vision_inset.shape[1]) / 2)
    final_frame[
        -panel_height:, col_start : col_start + vision_inset.shape[1], :
    ] = vision_inset

    cv2.putText(
        final_frame,
        f"L",
        org=(col_start, viz_frame.shape[0] + 27),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.8,
        color=(255, 255, 255),
        lineType=cv2.LINE_AA,
        thickness=1,
    )
    cv2.putText(
        final_frame,
        f"R",
        org=(col_start + vision_inset.shape[1] - 17, viz_frame.shape[0] + 27),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.8,
        color=(255, 255, 255),
        lineType=cv2.LINE_AA,
        thickness=1,
    )

    return final_frame


def save_video_with_vision_insets(
    sim, cam, path, visual_input_hist, stabilization_time=0.02
):
    """Save a list of frames as a video with insets showing the visual
    experience of the fly. This is almost a drop-in replacement of
    ``NeuroMechFly.save_video`` but as a static function (instead of a
    class method) and with an extra argument ``visual_input_hist``.

    Parameters
    ----------
    sim : Simulation
        The Simulation object.
    cam : Camera
        The Camera object that has been used to generate the frames.
    path : Path
        Path of the output video will be saved. Should end with ".mp4".
    visual_input_hist : list[np.ndarray]
        List of ommatidia readings. Each element is an array of shape
        (2, N, 2) where N is the number of ommatidia per eye.
    stabilization_time : float, optional
        Time (in seconds) to wait before starting to render the video.
        This might be wanted because it takes a few frames for the
        position controller to move the joints to the specified angles
        from the default, all-stretched position. By default 0.02s
    """
    if len(visual_input_hist) != len(cam._frames):
        raise ValueError(
            "Length of `visual_input_hist` must match the number of "
            "frames in the `NeuroMechFly` object. Save the visual input "
            "every time a frame is rendered, i.e. when `.render()` returns "
            "a non-`None` value."
        )

    num_stab_frames = int(np.ceil(stabilization_time / cam._eff_render_interval))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving video to {path}")
    with imageio.get_writer(path, fps=cam.fps) as writer:
        for i, (frame, visual_input) in enumerate(zip(cam._frames, visual_input_hist)):
            if i < num_stab_frames:
                continue
            frame = add_insets(sim.fly.retina, frame, visual_input)
            writer.append_data(frame)


def visualize_visual_input(
    retina: Retina,
    output_path: Path,
    vision_data_li: list[np.ndarray],
    raw_vision_data_li: list[np.ndarray],
    vision_update_mask: np.ndarray,
    vision_refresh_rate: float = 500,
    playback_speed: float = 0.1,
):
    """Convert lists of vision readings into a video and save it to disk.

    Parameters
    ----------
    retina : Retina
        The retina object used to generate the visual input.
    output_path : Path
        Path of the output video will be saved. Should end with ".mp4".
    vision_data_li : list[np.ndarray]
        List of ommatidia readings. Each element is an array of shape
        (2, N, 2) where the first dimension is for the left and right eyes,
        the second dimension is for the N ommatidia, and the third
        dimension is for the two channels. The length of this list is the
        number of simulation steps.
    raw_vision_data_li : list[np.ndarray]
        Same as ``vision_data_li`` but with the raw RGB images from the
        cameras instead of the simulated ommatidia readings. The shape of
        each element is therefore (2, H, W, 3) where the first dimension is
        for the left and right eyes, and the remaining dimensions are for
        the RGB image.
    vision_update_mask : np.ndarray
        Mask indicating which simulation steps have vision updates. This
        should be taken from ``NeuroMechFly.vision_update_mask``.
    vision_refresh_rate : float, optional
        The refresh rate of visual inputs in Hz. This should be consistent
        with ``MuJoCoParameters.vision_refresh_rate`` that is given to the
        simulation. By default 500.
    playback_speed : float, optional
        Speed, as a multiple of the 1x speed, at which the video should be
        rendered, by default 0.1.
    """
    vision_data_key_frames = np.array(vision_data_li)[vision_update_mask, :, :, :]
    raw_vision_key_frames = np.array(raw_vision_data_li)[vision_update_mask, :, :, :]
    num_frames = vision_data_key_frames.shape[0]

    # Convert hex pixels back to human readable pixels
    readable_processed_images = []
    for i in range(num_frames):
        frame_data = vision_data_key_frames[i, :, :]
        left_img = retina.hex_pxls_to_human_readable(
            frame_data[0, :, :], color_8bit=True
        )
        left_img = left_img.max(axis=-1)
        right_img = retina.hex_pxls_to_human_readable(
            frame_data[1, :, :], color_8bit=True
        )
        right_img = right_img.max(axis=-1)
        readable_processed_images.append([left_img, right_img])
    readable_processed_images = np.array(readable_processed_images)

    # Compile video
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    def update(frame):
        for i, side in enumerate(["Left", "Right"]):
            axs[0, i].cla()
            raw_img = raw_vision_key_frames[frame, i, :, :, :]
            raw_img = np.clip(raw_img, 0, 255).astype(np.uint8)
            axs[0, i].imshow(raw_img)
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
