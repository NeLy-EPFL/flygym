import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Union, List

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

# Would like it to always draw gravity in the upper right corner
# Check if contact need to be drawn (outside of the image)
# New gravity camera


_roll_eye = np.roll(np.eye(4, 3), -1)


class Camera:
    def __init__(
        self,
        attachment_point: mjcf.element._AttachableElement,
        camera_name: str,
        attachment_name: str = None,
        targeted_flies_id: int = [],
        window_size: Tuple[int, int] = (640, 480),
        play_speed: float = 0.2,
        fps: int = 30,
        timestamp_text: bool = False,
        play_speed_text: bool = True,
        camera_parameters: Optional[Dict[str, Any]] = None,
        draw_contacts: bool = False,
        decompose_contacts: bool = True,
        decompose_colors: Tuple[
            Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]
        ] = ((0, 0, 255), (0, 255, 0), (255, 0, 0)),
        force_arrow_scaling: float = float("nan"),
        tip_length: float = 10.0,  # number of pixels
        contact_threshold: float = 0.1,
        perspective_arrow_length: bool = False,
        draw_gravity: bool = False,
        gravity_arrow_scaling: float = 1e-4,
        output_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize a Camera that can be attached to any attachable element and take any mujoco inbuilt parameters.
        A set of preset configurations are available in the config file:
        - Simple cameras like: "camera_top" "camera_right", "camera_left",
        "camera_front", "camera_back", "camera_bottom"
        - Compound rotated cameras with different zoom levels: "camera_top_right", "camera_top_zoomout"
        "camera_right_front", "camera_left_top_zoomout", "camera_neck_zoomin",
        "camera_head_zoomin", "camera_front_zoomin", "camera_LFTarsus1_zoomin"
        - "camera_LFTarsus1_zoomin": Camera looking at the left tarsus of the first leg
        - "camera_back_track": 3rd person camera following the fly

        This camera can also be set with custom parameters by providing a dictionary of parameters.

        Parameters
        ----------
        attachment_point: dm_control.mjcf.element._AttachableElement
            Attachment point pf the camera
        attachment_name : str
            Name of the attachment point
        targeted_flies_id: List(int)
            Index of the flies the camera is looking at. The first index is the focused fly that is tracked if using a
            complex camera. The rest of the indices are used to draw the contact forces.
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
        draw_contacts : bool
            If True, arrows will be drawn to indicate contact forces between
            the legs and the ground.
        decompose_contacts : bool
            If True, the arrows visualizing contact forces will be decomposed
            into x-y-z components.
        decompose_colors
            Colors for the x, y, and z components of the contact force arrows.
        force_arrow_scaling : float
            Scaling factor determining the length of arrows visualizing contact
            forces.
        tip_length : float
            Size of the arrows indicating the contact forces in pixels.
        contact_threshold : float
            The threshold for contact detection in mN (forces below this
            magnitude will be ignored).
        perspective_arrow_length : bool
            If true, the length of the arrows indicating the contact forces
            will be determined by the perspective.
        draw_gravity : bool
            If True, an arrow will be drawn indicating the direction of
            gravity. This is useful during climbing simulations.
        gravity_arrow_scaling : float
            Scaling factor determining the size of the arrow indicating
            gravity.
        output_path : str or Path, optional
            Path to which the rendered video should be saved. If None, the
            video will not be saved. By default None.
        """
        self.attachment_point = attachment_point
        self.targeted_flies_id = targeted_flies_id

        config = util.load_config()

        self.iscustom = True
        if camera_parameters is None and camera_name in config["cameras"]:
            camera_parameters = config["cameras"][camera_name]
            self.iscustom = False

        assert camera_parameters is not None, (
            "Camera parameters must be provided "
            "if the camera name is not a predefined camera"
        )

        camera_parameters["name"] = camera_name

        # get a first value before spawning: useful for the zstab cam
        self.camera_base_offset = np.array(camera_parameters.get("pos", np.zeros(3)))

        self._cam, self.camera_id = self._add_camera(
            attachment_point, camera_parameters, attachment_name
        )

        self.window_size = window_size
        self.play_speed = play_speed
        self.fps = fps
        self.timestamp_text = timestamp_text
        self.play_speed_text = play_speed_text

        self.draw_contacts = draw_contacts
        self.decompose_contacts = decompose_contacts
        self.decompose_colors = decompose_colors
        if not np.isfinite(force_arrow_scaling):
            self.force_arrow_scaling = 1.0 if perspective_arrow_length else 10.0
        else:
            self.force_arrow_scaling = force_arrow_scaling
        self.tip_length = tip_length
        self.contact_threshold = contact_threshold
        self.perspective_arrow_length = perspective_arrow_length

        if self.draw_contacts and len(self.targeted_flies_id) <= 0:
            logging.warning(
                "Overriding `draw_contacts` to False because no flies are targeted."
            )
            self.draw_contacts = False

        if self.draw_contacts and "cv2" not in sys.modules:
            logging.warning(
                "Overriding `draw_contacts` to False because OpenCV is required "
                "to draw the arrows but it is not installed."
            )
            self.draw_contacts = False

        self.draw_gravity = draw_gravity
        self.gravity_arrow_scaling = gravity_arrow_scaling

        if self.draw_gravity:
            self._gravity_rgba = [1 - 213 / 255, 1 - 90 / 255, 1 - 255 / 255, 1.0]
            self._grav_arrow_start = (self.window_size[0] - 100, 100)

        if output_path is not None:
            self.output_path = Path(output_path)
        else:
            self.output_path = None

        self._eff_render_interval = self.play_speed / self.fps
        self._frames: list[np.ndarray] = []
        self._timestamp_per_frame: list[float] = []

    def _add_camera(self, attachment, camera_parameters, attachment_name):
        """Add a camera to the model."""
        camera = attachment.add("camera", **camera_parameters)
        if attachment_name is None:
            camera_id = camera.name
        else:
            camera_id = attachment_name + "/" + camera.name

        return camera, camera_id

    def init_camera_orientation(self, physics: mjcf.Physics):
        """Initialize the camera handling by storing the base camera position
        and rotation. This is useful for cameras that need to be updated
        during the simulation beyond the default behavior of the camera.
        """
        bound_cam = physics.bind(self._cam)
        self.camera_base_offset = bound_cam.xpos.copy()
        self.camera_base_rot = R.from_matrix(bound_cam.xmat.reshape(3, 3))

    def render(
        self,
        physics: mjcf.Physics,
        floor_height: float,
        curr_time: float,
        last_obs: List[dict],
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

        self.update_camera(physics, floor_height, last_obs[0])

        width, height = self.window_size
        img = physics.render(width=width, height=height, camera_id=self.camera_id)
        img = img.copy()
        if self.draw_contacts:
            for i in range(len(self.targeted_flies_id)):
                img = self._draw_contacts(img, physics, last_obs[i])
        if self.draw_gravity:
            img = self._draw_gravity(img, physics, last_obs[0]["pos"])

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
        self._timestamp_per_frame.append(curr_time)
        return img

    def reset(self):
        self._frames.clear()
        self._timestamp_per_frame = []

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

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving video to {path}")
        with imageio.get_writer(path, fps=self.fps) as writer:
            for frame, timestamp in zip(self._frames, self._timestamp_per_frame):
                if timestamp >= stabilization_time:
                    writer.append_data(frame)

    def update_camera(self, physics: mjcf.Physics, floor_height: float, obs: dict):
        """Update the camera position and rotation based on the fly position and orientation.
        Used only for the complex camera that require updating the camera position and rotation
        on top of the default behavior"""
        pass

    def _compute_camera_matrices(self, physics: mjcf.Physics):
        """Compute the camera matrices needed to project world coordinates into
        pixel space. The matrices are computed based on the camera's position
        and orientation in the world.
        With this there is no need for using dm_control's camera"""

        cam_bound = physics.bind(self._cam)

        width, height = self.window_size

        image = np.eye(3)
        image[0, 2] = (width - 1) / 2.0
        image[1, 2] = (height - 1) / 2.0

        focal_scaling = (1.0 / np.tan(np.deg2rad(cam_bound.fovy) / 2)) * height / 2.0
        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

        # Rotation matrix (4x4).
        rotation = np.eye(4)
        rotation[0:3, 0:3] = cam_bound.xmat.reshape(3, 3).T

        # Translation matrix (4x4).
        translation = np.eye(4)
        translation[0:3, 3] = -cam_bound.xpos

        return image, focal, rotation, translation

    def _draw_gravity(
        self,
        img: np.ndarray,
        physics: mjcf.Physics,
        fly_pos: list[float],
        thickness: float = 5,
    ) -> np.ndarray:
        """Draw gravity as an arrow. The arrow is drawn at the top right
        of the frame.
        """

        image, focal, rotation, translation = self._compute_camera_matrices(physics)
        camera_matrix = image @ focal @ rotation @ translation

        # Camera matrices multiply homogenous [x, y, z, 1] vectors.
        grav_homogeneous = np.ones((4, 2), dtype=float)
        grav_homogeneous[:3, :] = np.hstack(
            [
                np.expand_dims(fly_pos, -1),
                np.expand_dims(
                    fly_pos + physics.model.opt.gravity * self.gravity_arrow_scaling, -1
                ),
            ]
        )

        # Project world coordinates into pixel space. See:
        # https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula
        xs, ys, s = camera_matrix @ grav_homogeneous
        # x and y are in the pixel coordinate system.
        x = xs / s
        y = ys / s
        grav_vector = np.array(
            [
                x[1] - x[0],
                y[1] - y[0],
            ]
        ).astype(int)

        # Draw the vector on the image
        cv2.arrowedLine(
            img,
            self._grav_arrow_start,
            self._grav_arrow_start + grav_vector,
            self._gravity_rgba,
            thickness,
            cv2.LINE_AA,
        )

        return img

    def _draw_contacts(
        self, img: np.ndarray, physics: mjcf.Physics, last_obs: dict, thickness=2
    ) -> np.ndarray:
        """Draw contacts as arrow which length is proportional to the force
        magnitude. The arrow is drawn at the center of the body. It uses
        the camera matrix to transfer from the global space to the pixels
        space.
        """

        def clip(p_in, p_out, z_clip):
            t = (z_clip - p_out[-1]) / (p_in[-1] - p_out[-1])
            return t * p_in + (1 - t) * p_out

        forces = last_obs["contact_forces"]
        pos = last_obs["contact_pos"]

        magnitudes = np.linalg.norm(forces, axis=1)
        contact_indices = np.nonzero(magnitudes > self.contact_threshold)[0]

        n_contacts = len(contact_indices)
        # Build an array of start and end points for the force arrows
        if n_contacts == 0:
            return img

        contact_forces = forces[contact_indices] * self.force_arrow_scaling

        if self.decompose_contacts:
            contact_pos = pos[:, None, contact_indices]
            Xw = contact_pos + (contact_forces[:, None] * _roll_eye).T
        else:
            contact_pos = pos[:, contact_indices]
            Xw = np.stack((contact_pos, contact_pos + contact_forces.T), 1)

        # Convert to homogeneous coordinates
        Xw = np.concatenate((Xw, np.ones((1, *Xw.shape[1:]))))

        im_mat, foc_mat, rot_mat, trans_mat = self._compute_camera_matrices(physics)

        # Project to camera space
        Xc = np.tensordot(rot_mat @ trans_mat, Xw, 1)
        Xc = Xc[:3, :] / Xc[-1, :]

        z_near = -physics.model.vis.map.znear * physics.model.stat.extent

        is_behind_cam = Xc[2] >= z_near
        is_visible = ~(is_behind_cam[0] & is_behind_cam[1:])

        is_out = is_visible & is_behind_cam[1:]
        is_in = np.where(is_visible & is_behind_cam[0])

        if self.decompose_contacts:
            lines = np.stack((np.stack([Xc[:, 0]] * 3, axis=1), Xc[:, 1:]), axis=1)
        else:
            lines = Xc[:, :, None]

        lines[:, 1, is_out] = clip(lines[:, 0, is_out], lines[:, 1, is_out], z_near)
        lines[:, 0, is_in] = clip(lines[:, 1, is_in], lines[:, 0, is_in], z_near)

        # Project to pixel space
        lines = np.tensordot((im_mat @ foc_mat)[:, :3], lines, axes=1)
        lines2d = lines[:2] / lines[-1]
        lines2d = lines2d.T

        if not self.perspective_arrow_length:
            unit_vectors = lines2d[:, :, 1] - lines2d[:, :, 0]
            length = np.linalg.norm(unit_vectors, axis=-1, keepdims=True)
            # avoid division by small number
            length = np.clip(length, 1e-8, 1e8)
            unit_vectors /= length
            if self.decompose_contacts:
                lines2d[:, :, 1] = (
                    lines2d[:, :, 0] + np.abs(contact_forces[:, :, None]) * unit_vectors
                )
            else:
                lines2d[:, :, 1] = (
                    lines2d[:, :, 0]
                    + np.linalg.norm(contact_forces, axis=1)[:, None, None]
                    * unit_vectors
                )

        lines2d = np.rint(lines2d.reshape((-1, 2, 2))).astype(int)

        argsort = lines[2, 0].T.ravel().argsort()
        color_indices = np.tile(np.arange(3), lines.shape[-1])

        img = img.astype(np.uint8)

        for j in argsort:
            if not is_visible.ravel()[j]:
                continue

            color = self.decompose_colors[color_indices[j]]
            p1, p2 = lines2d[j].astype(int)
            arrow_length = np.linalg.norm(p2 - p1)

            if arrow_length > 1e-2:
                r = self.tip_length / arrow_length
            else:
                r = 1e-4

            if is_out.ravel()[j] and self.perspective_arrow_length:
                cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
            else:
                p1 = np.clip(p1, -1e5, 1e5).astype(int)
                p2 = np.clip(p2, -1e5, 1e5).astype(int)
                cv2.arrowedLine(img, p1, p2, color, thickness, cv2.LINE_AA, tipLength=r)
        return img


class ZStabCamera(Camera):
    """Camera that stabilizes the z-axis of the camera to the floor height."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Raise error if targeted flies are empty
        if len(self.targeted_flies_id) == 0:
            raise ValueError(
                "No flies are targeted by the camera. "
                "Stabilized cameras require at least one fly to target."
            )

    def init_camera_orientation(self, physics: mjcf.Physics):
        """Initialize the camera handling by storing the base camera position
        and rotation. This is useful for cameras that need to be updated
        during the simulation beyond the default behavior of the camera.
        """
        bound_cam = physics.bind(self._cam)
        # only update x and y as z is already set to floor height
        self.camera_base_offset[:2] = bound_cam.xpos[:2].copy()
        self.camera_base_rot = R.from_matrix(bound_cam.xmat.reshape(3, 3))

    def _update_cam_pos(self, physics: mjcf.Physics, floor_height: float):
        cam = physics.bind(self._cam)
        cam_pos = cam.xpos.copy()
        cam_pos[2] = floor_height + self.camera_base_offset[2]
        cam.xpos = cam_pos

    def update_camera(self, physics: mjcf.Physics, floor_height: float, obs: dict):
        self._update_cam_pos(physics, floor_height)
        return


class YawOnlyCamera(ZStabCamera):
    """Camera that stabilizes the z-axis of the camera to the floor height and
    only changes the yaw of the camera to follow the fly hereby preventing unnecessary
    camera rotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.smoothing = 0.99995  # tested empirically
        self.prev_yaw = None
        self.init_yaw = None

    def update_camera(self, physics: mjcf.Physics, floor_height: float, obs: dict):
        smoothed_yaw = self._smooth_yaw(obs["rot"][0])
        correction = R.from_euler("xyz", [0, 0, smoothed_yaw - self.init_yaw])
        self._update_cam_pos(physics, floor_height, correction, obs["pos"])
        self.update_cam_rot(physics, correction)

        self.prev_yaw = obs["rot"][0]

    def _smooth_yaw(self, yaw: float):
        if self.prev_yaw is None:
            self.prev_yaw = yaw
            self.init_yaw = yaw
        return np.arctan2(
            self.smoothing * np.sin(self.prev_yaw) + (1 - self.smoothing) * np.sin(yaw),
            self.smoothing * np.cos(self.prev_yaw) + (1 - self.smoothing) * np.cos(yaw),
        )

    def update_cam_rot(self, physics: mjcf.Physics, yaw_correction: R):
        physics.bind(self._cam).xmat = (
            (yaw_correction * self.camera_base_rot).as_matrix().flatten()
        )
        return

    def _update_cam_pos(
        self,
        physics: mjcf.Physics,
        floor_height: float,
        yaw_correction: R,
        fly_pos: List[float],
    ):
        # position the camera some distance behind the fly, at a fixed height
        physics.bind(self._cam).xpos = (
            # only add floor offset to z as camera base offset is added in the next line
            np.hstack([fly_pos[:2], floor_height])
            + (yaw_correction.as_matrix() @ self.camera_base_offset).flatten()
        )


class GravityAlignedCamera(Camera):
    """Camera that keeps the camera aligned with the original direction of the gravity
    while following the fly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # get the yaw_pitch roll of gravity vector
        self.gravity = None
        self.grav_rot = None
        self.cam_matrix = np.zeros((3, 3))

    def update_gravrot(self, gravity):
        """
        Update the rotation matrix that aligns the gravity vector with the z-axis
        """
        self.gravity = gravity
        gravity_norm = gravity / np.linalg.norm(gravity)
        self.grav_rot = R.align_vectors(gravity_norm, [0, 0, -1])[0]

    def update_camera(self, physics: mjcf.Physics, floor_height: float, obs: dict):
        if self.gravity is None or np.any(physics.model.opt.gravity != self.gravity):
            print("updating gravity")
            self.update_gravrot(physics.model.opt.gravity.copy())
        self.update_cam_rot(physics)
        self._update_cam_pos(physics, obs["pos"])

    def update_cam_rot(self, physics: mjcf.Physics):
        self.cam_matrix = self.grav_rot * self.camera_base_rot
        physics.bind(self._cam).xmat = self.cam_matrix.as_matrix().flatten()
        return

    def _update_cam_pos(self, physics: mjcf.Physics, fly_pos: List[float]):
        # position the camera some distance behind the fly, at a fixed height
        fly_pos[2] = 0
        self.cam_pos = (
            fly_pos + (self.grav_rot.as_matrix() @ self.camera_base_offset).flatten()
        )
        physics.bind(self._cam).xpos = self.cam_pos
