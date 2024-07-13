import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import flygym.util as util


import cv2
import dm_control.mujoco
import imageio
import numpy as np
from dm_control import mjcf
from dm_control.utils import transformations
from flygym.fly import Fly
from scipy.spatial.transform import Rotation as R

from typing import Tuple, List, Dict, Any, Optional

from abc import ABC, abstractmethod



class Camera():
    def __init__(
        self,
        attachement_point: Any,
        camera_name: str,
        attachement_name: str = None,
        window_size: Tuple[int, int] = (640, 480),
        play_speed: float = 0.2,
        fps: int = 30,
        timestamp_text: bool = False,
        play_speed_text: bool = True,
        camera_parameters: Optional[Dict[str, Any]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize a Camera.

        Parameters
        ----------
        attachement_point: Any
        attachement_name : str
        camera_name : str
        window_size : Tuple[int, int]
            Size of the rendered images in pixels, by default (640, 480).
        play_speed : float
            Play speed of the rendered video, by default 0.2.
        fps: int
            FPS of the rendered video when played at ``play_speed``, by
            default 30.
        timestamp_text : bool
            If True, text indicating the current simulation time will be
            added to the rendered video.
        play_speed_text : bool
            If True, text indicating the play speed will be added to the
            rendered video.
        camera_parameters : Optional[Dict[str, Any]]
            Parameters of the camera to be added to the model. If None, the
        output_path : str or Path, optional
            Path to which the rendered video should be saved. If None, the
            video will not be saved. By default None.
        """
        self.attachement_point = attachement_point

        config = util.load_config()

        self.iscustom = True
        if camera_parameters is None and camera_name in config["cameras"]:
            camera_parameters = config["cameras"][camera_name]
            self.iscustom = False

        assert camera_parameters is not None, ("Camera parameters must be provided "
        "if the camera name is not a predefined camera")

        camera_parameters["name"] = camera_name

        self._cam, self.camera_id = self._add_camera(attachement_point,
                                                     camera_parameters,
                                                     attachement_name)

        self._initialize_custom_camera_handling()
        
        self.window_size = window_size
        self.play_speed = play_speed
        self.fps = fps
        self.timestamp_text = timestamp_text
        self.play_speed_text = play_speed_text

        if output_path is not None:
            self.output_path = Path(output_path)
        else:
            self.output_path = None

        self._last_render_time = -np.inf
        self._eff_render_interval = self.play_speed / self.fps
        self._frames: list[np.ndarray] = []

    def _add_camera(self, attachement, camera_parameters, attachement_name):
        # Add camera to the model
        camera = attachement.add("camera", **camera_parameters)
        if attachement_name is None:
            camera_id = camera.name
        else:
            camera_id = attachement_name + "/" + camera.name

        return camera, camera_id
         
    def render(
        self, physics: mjcf.Physics, floor_height: float, curr_time: float
    ) -> Union[np.ndarray, None]:
        """Call the ``render`` method to update the renderer. It should be
        called every iteration; the method will decide by itself whether
        action is required.

        Returns
        -------
        np.ndarray
            The rendered image is one is rendered.
        """
        if curr_time < len(self._frames) * self._eff_render_interval:
            return None

        self.update_camera(physics)

        width, height = self.window_size
        img = physics.render(width=width, height=height, camera_id=self.camera_id)
        img = img.copy()

        render_playspeed_text = self.play_speed_text
        render_time_text = self.timestamp_text
        # if render_playspeed_text or render_time_text:
        if render_playspeed_text and render_time_text:
            text = f"{curr_time:.2f}s ({self.play_speed}x)"
        elif render_playspeed_text:
            text = f"{self.play_speed}x"
        elif render_time_text:
            text = f"{curr_time:.2f}s"
        else:
            text = ""

        if text:
            img = cv2.putText(
                img,
                text,
                org=(20, 30),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.8,
                color=(0, 0, 0),
                lineType=cv2.LINE_AA,
                thickness=1,
            )

        self._frames.append(img)
        self._last_render_time = curr_time
        return img


    def reset(self):
        self._frames.clear()
        self._last_render_time = -np.inf

    def save_video(self, path: Union[str, Path], stabilization_time=0.02):
        """Save rendered video since the beginning or the last ``reset()``,
        whichever is the latest. Only useful if ``render_mode`` is 'saved'.

        Parameters
        ----------
        path : str or Path
            Path to which the video should be saved.
        stabilization_time : float, optional
            Time (in seconds) to wait before starting to render the video.
            This might be wanted because it takes a few frames for the
            position controller to move the joints to the specified angles
            from the default, all-stretched position. By default 0.02s
        """
        if len(self._frames) == 0:
            logging.warning(
                "No frames have been rendered yet; no video will be saved despite "
                "`save_video()` call. Be sure to call `.render()` in your simulation "
                "loop."
            )

        num_stab_frames = int(np.ceil(stabilization_time / self._eff_render_interval))

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving video to {path}")
        with imageio.get_writer(path, fps=self.fps) as writer:
            for frame in self._frames[num_stab_frames:]:
                writer.append_data(frame)
    
    def _initialize_custom_camera_handling(self):
        pass

    def update_camera(self, physics: mjcf.Physics):
        pass


class ZStabCamera(Camera):

    def __init__(self, floor_height: float, *args, **kwargs):
        self.floor_height = floor_height
        super().__init__(*args, **kwargs)
        self.cam_offset = self._cam.pos.copy()
    
    def _update_cam_pos(self, physics: mjcf.Physics):
        cam = physics.bind(self._cam)
        cam_pos = cam.xpos.copy()
        cam_pos[2] = self.cam_offset[2] + self.floor_height
        cam.xpos = cam_pos    

    def update_camera(self, physics: mjcf.Physics):
        self._update_cam_pos(physics)
        return


class GravityAlignedCamera(Camera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cam_offset = self._cam.pos.copy()
        self.cam_euler = self._cam.euler.copy()

    def _update_cam_pos(self, physics: mjcf.Physics):
        cam = physics.bind(self._cam)
        cam_pos = cam.xpos.copy()
        cam_pos[2] = self.cam_offset[2]
        cam.xpos = cam_pos

    def _update_cam_euler(self, physics: mjcf.Physics):
        cam = physics.bind(self._cam)
        cam_euler = cam.xmat.copy()
        cam_euler[:3, :3] = np.eye(3)
        cam.xmat = cam_euler

    def update_camera(self, physics: mjcf.Physics):
        self._update_cam_pos(physics)
        self._update_cam_euler(physics)
        return