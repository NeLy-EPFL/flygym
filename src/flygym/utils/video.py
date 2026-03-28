from pathlib import Path

import numpy as np
import imageio.v3 as iio
from PIL import Image


def write_video_from_frames(path: str | Path, frames: list[np.ndarray], **kwargs):
    """Writes a video to the given path from a list of frames (numpy arrays).

    The image size is automatically scaled up to multiples of 16 to enhance
    compatibility with most codecs and players. Additional keyword arguments are passed
    to `imageio.v3.imwrite`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Scale image size up to multiples of 16 to enhance compatibility with most codecs
    # and players
    height, width = frames[0].shape[:2]
    new_width = (width + 15) // 16 * 16
    new_height = (height + 15) // 16 * 16
    if (new_width, new_height) != (width, height):
        for i, frame in enumerate(frames):
            pil_frame = Image.fromarray(frame)
            pil_frame_resized = pil_frame.resize(
                (new_width, new_height), resample=Image.Resampling.BICUBIC
            )
            frame_resized = np.array(pil_frame_resized)
            frames[i] = frame_resized

    iio.imwrite(path, frames, **kwargs)
