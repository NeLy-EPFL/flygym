import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
from matplotlib.animation import FuncAnimation


class Retina:
    def __init__(
        self,
        ommatidia_id_map: np.ndarray,
        pale_type_mask: np.ndarray,
        distortion_coefficient: float,
        zoom: float,
        nrows: int,
        ncols: int,
    ) -> None:
        self.ommatidia_id_map = ommatidia_id_map
        _unique_count = np.unique(ommatidia_id_map, return_counts=True)
        self.num_pixels_per_ommatidia = _unique_count[1][1:]
        self.pale_type_mask = pale_type_mask
        self.distortion_coefficient = distortion_coefficient
        self.zoom = zoom
        self.nrows = nrows
        self.ncols = ncols

    def raw_image_to_hex_pxls(self, raw_img: np.ndarray) -> np.ndarray:
        return self._raw_image_to_hex_pxls(
            raw_img,
            self.ommatidia_id_map,
            self.num_pixels_per_ommatidia,
            self.pale_type_mask,
        )

    def hex_pxls_to_human_readable(self, ommatidia_reading: np.ndarray) -> np.ndarray:
        return self._hex_pxls_to_human_readable(
            ommatidia_reading, self.ommatidia_id_map
        )

    def correct_fisheye(self, img: np.ndarray) -> np.ndarray:
        return self._correct_fisheye(
            img, self.nrows, self.ncols, self.zoom, self.distortion_coefficient
        )

    @staticmethod
    @nb.njit(parallel=True)
    def _raw_image_to_hex_pxls(
        raw_img, ommatidia_id_map, num_pixels_per_ommatidia, pale_type_mask
    ):
        vals = np.zeros((len(num_pixels_per_ommatidia), 2))
        img_arr_flat = raw_img.reshape((-1, 3))
        hex_id_map_flat = ommatidia_id_map.flatten()
        for i in nb.prange(hex_id_map_flat.size):
            hex_pxl_id = hex_id_map_flat[i] - 1
            hex_pxl_size = num_pixels_per_ommatidia[hex_pxl_id]  # num raw pxls
            if hex_pxl_id != -1:
                ch_idx = pale_type_mask[hex_pxl_id]
                vals[hex_pxl_id, ch_idx] += img_arr_flat[i, ch_idx + 1] / hex_pxl_size
        return vals

    @staticmethod
    @nb.njit(parallel=True)
    def _hex_pxls_to_human_readable(ommatidia_reading, ommatidia_id_map):
        processed_image_flat = np.zeros(ommatidia_id_map.size, dtype=np.uint8) + 255
        hex_id_map_flat = ommatidia_id_map.flatten().astype(np.int16)
        for i in nb.prange(hex_id_map_flat.size):
            hex_pxl_id = hex_id_map_flat[i] - 1
            if hex_pxl_id != -1:
                hex_pxl_val = ommatidia_reading[hex_pxl_id, :].max()
                processed_image_flat[i] = hex_pxl_val
        return processed_image_flat.reshape(ommatidia_id_map.shape)

    @staticmethod
    @nb.njit(parallel=True)
    def _correct_fisheye(img, nrows, ncols, zoom, distortion_coefficient):
        """Based on https://github.com/Gil-Mor/iFish, MIT License."""
        dst_img = np.zeros((nrows, ncols, 3), dtype="uint8")

        # easier to calculate if we traverse x, y in dst image
        for dst_row in nb.prange(nrows):
            for dst_col in nb.prange(ncols):
                # normalize row and col to be in interval of [-1, 1] and apply zoom
                dst_row_norm = ((2 * dst_row - nrows) / nrows) / zoom
                dst_col_norm = ((2 * dst_col - ncols) / ncols) / zoom

                # get normalized row and col dist from center, +1e-6 to avoid div by 0
                dst_radius_norm = np.sqrt(dst_col_norm**2 + dst_row_norm**2)
                denom = 1 - (distortion_coefficient * (dst_radius_norm**2)) + 1e-6
                src_row_norm = dst_row_norm / denom
                src_col_norm = dst_col_norm / denom

                # convert the normalized distorted row and col back to image pixels
                src_row = int(((src_row_norm + 1) * nrows) / 2)
                src_col = int(((src_col_norm + 1) * ncols) / 2)

                # if new pixel is in bounds copy from source pixel to destination pixel
                if (
                    0 <= src_row
                    and src_row < nrows
                    and 0 <= src_col
                    and src_col < ncols
                ):
                    dst_img[dst_row][dst_col] = img[src_row][src_col]

        return dst_img


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