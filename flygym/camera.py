import logging
import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import dm_control.mujoco
import imageio
import numpy as np
from dm_control import mjcf
from dm_control.utils import transformations
from flygym.fly import Fly
from scipy.spatial.transform import Rotation as R


_roll_eye = np.roll(np.eye(4, 3), -1)


class Camera:
    """Camera associated with a fly.

    Attributes
    ----------
    fly : Fly
        The fly to which the camera is associated.
    window_size : tuple[int, int]
        Size of the rendered images in pixels.
    play_speed : float
        Play speed of the rendered video.
    fps: int
        FPS of the rendered video when played at ``play_speed``.
    timestamp_text : bool
        If True, text indicating the current simulation time will be added
        to the rendered video.
    play_speed_text : bool
        If True, text indicating the play speed will be added to the
        rendered video.
    dm_camera : dm_control.mujoco.Camera
        The ``dm_control`` camera instance associated with the camera.
        Only available after calling ``initialize_dm_camera(physics)``.
        Useful for mapping the rendered location to the physical location
        in the simulation.
    draw_contacts : bool
        If True, arrows will be drawn to indicate contact forces between
        the legs and the ground.
    decompose_contacts : bool
        If True, the arrows visualizing contact forces will be decomposed
        into x-y-z components.
    force_arrow_scaling : float
        Scaling factor determining the length of arrows visualizing contact
        forces.
    tip_length : float
        Size of the arrows indicating the contact forces in pixels.
    contact_threshold : float
        The threshold for contact detection in mN (forces below this
        magnitude will be ignored).
    draw_gravity : bool
        If True, an arrow will be drawn indicating the direction of
        gravity. This is useful during climbing simulations.
    gravity_arrow_scaling : float
        Scaling factor determining the size of the arrow indicating
        gravity.
    align_camera_with_gravity : bool
        If True, the camera will be rotated such that gravity points down.
        This is
        useful during climbing simulations.
    camera_follows_fly_orientation : bool
        If True, the camera will be rotated so that it aligns with the
        fly's orientation.
    decompose_colors
        Colors for the x, y, and z components of the contact force arrows.
    output_path : Optional[Union[str, Path]]
        Path to which the rendered video should be saved. If None, the
        video will not be saved.
    """

    dm_camera: dm_control.mujoco.Camera

    def __init__(
        self,
        fly: Fly,
        camera_id: str = "Animat/camera_left",
        window_size: tuple[int, int] = (640, 480),
        play_speed: float = 0.2,
        fps: int = 30,
        timestamp_text: bool = False,
        play_speed_text: bool = True,
        draw_contacts: bool = False,
        decompose_contacts: bool = True,
        force_arrow_scaling: float = float("nan"),
        tip_length: float = 10.0,  # number of pixels
        contact_threshold: float = 0.1,
        draw_gravity: bool = False,
        gravity_arrow_scaling: float = 1e-4,
        align_camera_with_gravity: bool = False,
        camera_follows_fly_orientation: bool = False,
        decompose_colors: tuple[
            tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]
        ] = ((255, 0, 0), (0, 255, 0), (0, 0, 255)),
        output_path: Optional[Union[str, Path]] = None,
        perspective_arrow_length=False,
    ):
        """Initialize a Camera.

        Parameters
        ----------
        fly : Fly
            The fly to which the camera is associated.
        camera_id : str
            The camera that will be used for rendering, by default
            "Animat/camera_left".
        window_size : tuple[int, int]
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
        draw_contacts : bool
            If True, arrows will be drawn to indicate contact forces
            between the legs and the ground. By default False.
        decompose_contacts : bool
            If True, the arrows visualizing contact forces will be
            decomposed into x-y-z components. By default True.
        force_arrow_scaling : float, optional
            Scaling factor determining the length of arrows visualizing
            contact forces. By default 1.0 if perspective_arrow_length
            is True and 10.0 otherwise.
        tip_length : float
            Size of the arrows indicating the contact forces in pixels. By
            default 10.
        contact_threshold : float
            The threshold for contact detection in mN (forces below this
            magnitude will be ignored). By default 0.1.
        draw_gravity : bool
            If True, an arrow will be drawn indicating the direction of
            gravity. This is useful during climbing simulations. By default
            False.
        gravity_arrow_scaling : float
            Scaling factor determining the size of the arrow indicating
            gravity. By default 0.0001.
        align_camera_with_gravity : bool
            If True, the camera will be rotated such that gravity points
            down. This is useful during climbing simulations. By default
            False.
        camera_follows_fly_orientation : bool
            If True, the camera will be rotated so that it aligns with the
            fly's orientation. By default False.
        decompose_colors
            Colors for the x, y, and z components of the contact force
            arrows. By default ((255, 0, 0), (0, 255, 0), (0, 0, 255)).
        output_path : str or Path, optional
            Path to which the rendered video should be saved. If None, the
            video will not be saved. By default None.
        perspective_arrow_length : bool
            If true, the length of the arrows indicating the contact forces
            will be determined by the perspective.
        """
        self.fly = fly
        self.window_size = window_size
        self.play_speed = play_speed
        self.fps = fps
        self.timestamp_text = timestamp_text
        self.play_speed_text = play_speed_text
        self.draw_contacts = draw_contacts
        self.decompose_contacts = decompose_contacts
        self.tip_length = tip_length
        self.contact_threshold = contact_threshold
        self.draw_gravity = draw_gravity
        self.gravity_arrow_scaling = gravity_arrow_scaling
        self.align_camera_with_gravity = align_camera_with_gravity
        self.camera_follows_fly_orientation = camera_follows_fly_orientation
        self.decompose_colors = decompose_colors
        self.camera_id = camera_id.replace("Animat", fly.name)
        self.perspective_arrow_length = perspective_arrow_length

        if not np.isfinite(force_arrow_scaling):
            self.force_arrow_scaling = 1.0 if perspective_arrow_length else 10.0
        else:
            self.force_arrow_scaling = force_arrow_scaling

        if output_path is not None:
            self.output_path = Path(output_path)
        else:
            self.output_path = None

        if self.draw_contacts and "cv2" not in sys.modules:
            logging.warning(
                "Overriding `draw_contacts` to False because OpenCV is required "
                "to draw the arrows but it is not installed."
            )
            self.draw_contacts = False

        if self.draw_gravity:
            fly._last_fly_pos = self.fly.spawn_pos
            self._gravity_rgba = [1 - 213 / 255, 1 - 90 / 255, 1 - 255 / 255, 1.0]
            self._arrow_offset = np.zeros(3)
            if "bottom" in camera_id or "top" in camera_id:
                self._arrow_offset[0] = -3
                self._arrow_offset[1] = 2
            elif "left" in camera_id or "right" in camera_id:
                self._arrow_offset[2] = 2
                self._arrow_offset[0] = -3
            elif "front" in camera_id or "back" in camera_id:
                self._arrow_offset[2] = 2
                self._arrow_offset[1] = 3

        if self.align_camera_with_gravity:
            self._camera_rot = np.eye(3)

        self._cam = self.fly.model.find("camera", camera_id.split("/")[-1])
        self._initialize_custom_camera_handling(camera_id)
        self._eff_render_interval = self.play_speed / self.fps
        self._frames: list[np.ndarray] = []
        self._timestamp_per_frame: list[float] = []

    def _initialize_custom_camera_handling(self, camera_name: str):
        """
        This function is called when the camera is initialized. It can be
        used to customize the camera behavior. I case update_camera_pos is
        True and the camera is within the animat and not a head camera, the
        z position will be fixed to avoid oscillations. If
        self.camera_follows_fly_orientation is True, the camera
        will be rotated to follow the fly orientation (i.e. the front
        camera will always be in front of the fly).
        """

        is_animat = camera_name.startswith("Animat") or camera_name.startswith(
            self.fly.name
        )
        is_visualization_camera = (
            "head" in camera_name
            or "Tarsus" in camera_name
            or "camera_front_zoomin" in camera_name
        )

        canonical_cameras = [
            "camera_front",
            "camera_back",
            "camera_top",
            "camera_bottom",
            "camera_left",
            "camera_right",
            "camera_neck_zoomin",
        ]
        if "/" not in camera_name:
            is_canonical_camera = False
        else:
            is_canonical_camera = camera_name.split("/")[-1] in canonical_cameras

        # always add pos update if it is a head camera
        if is_animat and not is_visualization_camera:
            self.update_camera_pos = True
            self.cam_offset = self._cam.pos
            if (not is_canonical_camera) and self.camera_follows_fly_orientation:
                self.camera_follows_fly_orientation = False
                logging.warning(
                    "Overriding `camera_follows_fly_orientation` to False because"
                    "the rendering camera is not a simple camera from a canonical "
                    "angle (front, back, top, bottom, left, right, neck_zoomin)."
                )
            elif self.camera_follows_fly_orientation:
                # Why would that be xyz and not XYZ ? DOES NOT MAKE SENSE BUT IT WORKS
                self.base_camera_rot = R.from_euler(
                    "xyz", self._cam.euler + self.fly.spawn_orientation
                ).as_matrix()
                # THIS SOMEHOW REPLICATES THE CAMERA XMAT OBTAINED BY MUJOCO WHE USING
                # TRACKED CAMERA
            else:
                # if not camera_follows_fly_orientation need to change the camera mode
                # to track
                self._cam.mode = "track"
            return
        else:
            self.update_camera_pos = False
            if self.camera_follows_fly_orientation:
                self.camera_follows_fly_orientation = False
                logging.warning(
                    "Overriding `camera_follows_fly_orientation` to False because"
                    "it is never applied to visualization cameras (head, tarsus, ect)"
                    "or non Animat camera."
                )
            return

    def initialize_dm_camera(self, physics: mjcf.Physics):
        """
        ``dm_control`` comes with its own camera class that contains a
        number of useful utilities, including in particular tools for
        mapping the rendered location (row-column coordinate on the
        rendered image) to the physical location in the simulation. Given
        the physics instance of the simulation, this method initializes a
        "shadow" ``dm_control`` camera instance.

        Parameters
        ----------
        physics : mjcf.Physics
            Physics instance of the simulation.
        """
        self.dm_camera = dm_control.mujoco.Camera(
            physics,
            camera_id=self.camera_id,
            width=self.window_size[0],
            height=self.window_size[1],
        )

    def set_gravity(self, gravity: np.ndarray, rot_mat: np.ndarray = None) -> None:
        """Set the gravity of the environment. Changing the gravity vector
        might be useful during climbing simulations. The change in the
        camera point of view has been extensively tested for the simple
        cameras (left right top bottom front back) but not for the composed
        ones.

        Parameters
        ----------
        gravity : np.ndarray
            The gravity vector.
        rot_mat : np.ndarray, optional
            The rotation matrix to align the camera with the gravity vector
             by default None.
        """
        # Only change the angle of the camera if the new gravity vector and the camera
        # angle are compatible
        camera_is_compatible = False
        if "left" in self.camera_id or "right" in self.camera_id:
            if not gravity[1] > 0:
                camera_is_compatible = True
        # elif "top" in self.camera_name or "bottom" in
        # self.camera_name:
        elif "front" in self.camera_id or "back" in self.camera_id:
            if not gravity[1] > 0:
                camera_is_compatible = True

        if rot_mat is not None and self.align_camera_with_gravity:
            self._camera_rot = rot_mat
        elif camera_is_compatible:
            normalised_gravity = (np.array(gravity) / np.linalg.norm(gravity)).reshape(
                (1, 3)
            )
            downward_ref = np.array([0.0, 0.0, -1.0]).reshape((1, 3))

            if (
                not np.all(normalised_gravity == downward_ref)
                and self.align_camera_with_gravity
            ):
                # Generate a bunch of vectors to help the optimisation algorithm

                random_vectors = np.tile(np.random.rand(10_000), (3, 1)).T
                downward_refs = random_vectors + downward_ref
                gravity_vectors = random_vectors + normalised_gravity
                downward_refs = downward_refs
                gravity_vectors = gravity_vectors
                rot_mult = R.align_vectors(downward_refs, gravity_vectors)[0]

                rot_simple = R.align_vectors(
                    np.reshape(normalised_gravity, (1, 3)),
                    downward_ref.reshape((1, 3)),
                )[0]

                diff_mult = np.linalg.norm(
                    np.dot(rot_mult.as_matrix(), normalised_gravity.T) - downward_ref.T
                )
                diff_simple = np.linalg.norm(
                    np.dot(rot_simple.as_matrix(), normalised_gravity.T)
                    - downward_ref.T
                )
                if diff_mult < diff_simple:
                    rot = rot_mult
                else:
                    rot = rot_simple

                logging.info(
                    f"{normalised_gravity}, "
                    f"{rot.as_euler('xyz')}, "
                    f"{np.dot(rot.as_matrix(), normalised_gravity.T).T}, ",
                    f"{downward_ref}",
                )

                # check if rotation has effect if not remove it
                euler_rot = rot.as_euler("xyz")
                new_euler_rot = np.zeros(3)
                last_rotated_vector = normalised_gravity
                for i in range(0, 3):
                    new_euler_rot[: i + 1] = euler_rot[: i + 1].copy()

                    rotated_vector = (
                        R.from_euler("xyz", new_euler_rot).as_matrix()
                        @ normalised_gravity.T
                    ).T
                    logging.info(
                        f"{euler_rot}, "
                        f"{new_euler_rot}, "
                        f"{rotated_vector}, "
                        f"{last_rotated_vector}"
                    )
                    if np.linalg.norm(rotated_vector - last_rotated_vector) < 1e-2:
                        logging.info("Removing component {i}")
                        euler_rot[i] = 0
                    last_rotated_vector = rotated_vector

                logging.info(str(euler_rot))
                rot = R.from_euler("xyz", euler_rot)
                rot_mat = rot.as_matrix()

                self._camera_rot = rot_mat.T

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

        width, height = self.window_size
        if self.update_camera_pos:
            self._update_cam_pos(physics, floor_height)
        if self.camera_follows_fly_orientation:
            self._update_cam_rot(physics)
        if self.align_camera_with_gravity:
            self._rotate_camera(physics)
        img = physics.render(width=width, height=height, camera_id=self.camera_id)
        img = img.copy()
        if self.draw_contacts:
            img = self._draw_contacts(img, physics)
        if self.draw_gravity:
            img = self._draw_gravity(img, physics)

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

    def _update_cam_pos(self, physics: mjcf.Physics, floor_height: float):
        cam = physics.bind(self._cam)
        cam_pos = cam.xpos.copy()
        cam_pos[2] = self.cam_offset[2] + floor_height
        cam.xpos = cam_pos

    def _update_cam_rot(self, physics: mjcf.Physics):
        cam = physics.bind(self._cam)
        cam_name = self._cam.name
        fly_z_rot_euler = (
            np.array([self.fly.last_obs["rot"][0], 0.0, 0.0])
            - self.fly.spawn_orientation[::-1]
            - [np.pi / 2, 0, 0]
        )
        # This compensates both for the scipy to mujoco transform (align with y is
        # [0, 0, 0] in mujoco but [pi/2, 0, 0] in scipy) and the fact that the fly
        # orientation is already taken into account in the base_camera_rot (see below)
        # camera is always looking along its -z axis
        if cam_name in ["camera_top", "camera_bottom"]:
            # if camera is top or bottom always keep rotation around z only
            cam_matrix = R.from_euler("zyx", fly_z_rot_euler).as_matrix()
        elif cam_name in ["camera_front", "camera_back", "camera_left", "camera_right"]:
            # if camera is front, back, left or right apply the rotation around y
            cam_matrix = R.from_euler("yzx", fly_z_rot_euler).as_matrix()
        else:
            cam_matrix = np.eye(3)

        if cam_name in ["camera_bottom"]:
            cam_matrix = cam_matrix.T
            # z axis is inverted

        cam_matrix = self.base_camera_rot @ cam_matrix
        cam.xmat = cam_matrix.flatten()

    def _rotate_camera(self, physics: mjcf.Physics):
        # get camera
        cam = physics.bind(self._cam)
        # rotate the cam
        cam_matrix_base = getattr(cam, "xmat").copy()
        cam_matrix = self._camera_rot @ cam_matrix_base.reshape(3, 3)
        setattr(cam, "xmat", cam_matrix.flatten())

        return 0

    def _draw_gravity(self, img: np.ndarray, physics: mjcf.Physics) -> np.ndarray:
        """Draw gravity as an arrow. The arrow is drawn at the top right
        of the frame.
        """

        camera_matrix = self.dm_camera.matrix
        last_fly_pos = self.fly.last_obs["pos"]

        if self.align_camera_with_gravity:
            arrow_start = last_fly_pos + self._camera_rot @ self._arrow_offset
        else:
            arrow_start = last_fly_pos + self._arrow_offset

        arrow_end = arrow_start + physics.model.opt.gravity * self.gravity_arrow_scaling
        xyz_global = np.array([arrow_start, arrow_end]).T

        # Camera matrices multiply homogenous [x, y, z, 1] vectors.
        corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
        corners_homogeneous[:3, :] = xyz_global

        # Project world coordinates into pixel space. See:
        # https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula
        xs, ys, s = camera_matrix @ corners_homogeneous

        # x and y are in the pixel coordinate system.
        x = np.rint(xs / s).astype(int)
        y = np.rint(ys / s).astype(int)

        img = img.astype(np.uint8)
        img = cv2.arrowedLine(img, (x[0], y[0]), (x[1], y[1]), self._gravity_rgba, 10)

        return img

    def _draw_contacts(
        self, img: np.ndarray, physics: mjcf.Physics, thickness=2
    ) -> np.ndarray:
        """Draw contacts as arrow which length is proportional to the force
        magnitude. The arrow is drawn at the center of the body. It uses
        the camera matrix to transfer from the global space to the pixels
        space.
        """

        def clip(p_in, p_out, z_clip):
            t = (z_clip - p_out[-1]) / (p_in[-1] - p_out[-1])
            return t * p_in + (1 - t) * p_out

        forces = self.fly.last_obs["contact_forces"]
        pos = self.fly.last_obs["contact_pos"]

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

        mat = self.dm_camera.matrices()

        # Project to camera space
        Xc = np.tensordot(mat.rotation @ mat.translation, Xw, 1)
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
        lines = np.tensordot((mat.image @ mat.focal)[:, :3], lines, axes=1)
        lines2d = lines[:2] / lines[-1]
        lines2d = lines2d.T

        if not self.perspective_arrow_length:
            unit_vectors = lines2d[:, :, 1] - lines2d[:, :, 0]
            length = np.linalg.norm(unit_vectors, axis=-1, keepdims=True)
            length[length == 0] = 1
            unit_vectors /= length
            lines2d[:, :, 1] = (
                lines2d[:, :, 0] + np.abs(contact_forces[:, :, None]) * unit_vectors
            )

        lines2d = np.rint(lines2d.reshape((-1, 2, 2))).astype(int)

        argsort = lines[2, 0].T.ravel().argsort()
        color_indices = np.tile(np.arange(3), lines.shape[-1])

        img = img.astype(np.uint8)

        for j in argsort:
            if not is_visible.ravel()[j]:
                continue

            color = self.decompose_colors[color_indices[j]]
            p1, p2 = lines2d[j]
            arrow_length = np.linalg.norm(p2 - p1)

            if arrow_length > 1e-2:
                r = self.tip_length / arrow_length
            else:
                r = 1e-4

            if is_out.ravel()[j] and self.perspective_arrow_length:
                cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
            else:
                cv2.arrowedLine(img, p1, p2, color, thickness, cv2.LINE_AA, tipLength=r)
        return img

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

    def reset(self):
        self._frames.clear()
        self._timestamp_per_frame = []

    def _correct_camera_orientation(self, camera_name: str):
        # Correct the camera orientation by incorporating the spawn rotation
        # of the arena

        # Get the camera
        fly = self.fly
        camera = fly.model.find("camera", camera_name)

        if camera is None or camera.mode in ["targetbody", "targetbodycom"]:
            return 0

        if "head" in camera_name or "front_zoomin" in camera_name:
            # Don't correct the head camera
            return camera

        # Add the spawn rotation (keep horizon flat)
        spawn_quat = np.array(
            [
                np.cos(fly.spawn_orientation[-1] / 2),
                fly.spawn_orientation[0] * np.sin(fly.spawn_orientation[-1] / 2),
                fly.spawn_orientation[1] * np.sin(fly.spawn_orientation[-1] / 2),
                fly.spawn_orientation[2] * np.sin(fly.spawn_orientation[-1] / 2),
            ]
        )

        # Change camera euler to quaternion
        camera_quat = transformations.euler_to_quat(camera.euler)
        new_camera_quat = transformations.quat_mul(
            transformations.quat_inv(spawn_quat), camera_quat
        )
        camera.euler = transformations.quat_to_euler(new_camera_quat)

        # Elevate the camera slightly gives a better view of the arena
        if "zoomin" not in camera_name:
            camera.pos = camera.pos + [0.0, 0.0, 0.5]
        if "front" in camera_name:
            camera.pos[2] = camera.pos[2] + 1.0

        return camera


class NeckCamera(Camera):
    def __init__(self, **kwargs):
        assert "camera_id" not in kwargs, "camera_id should not be passed to NeckCamera"
        kwargs["camera_id"] = "Animat/camera_neck_zoomin"
        super().__init__(**kwargs)

    def _update_cam_pos(self, physics: mjcf.Physics, floor_height: float):
        pass
        # cam = physics.bind(self._cam)
        # cam_pos = cam.xpos.copy()
        # cam_pos[2] += floor_height
        # cam.xpos = cam_pos

    def _update_cam_rot(self, physics: mjcf.Physics):
        pass
        # cam = physics.bind(self._cam)

        # fly_z_rot_euler = (
        #     np.array([self.fly.last_obs["rot"][0], 0.0, 0.0])
        #     - self.fly.spawn_orientation[::-1]
        #     - [np.pi / 2, 0, 0]
        # )
        # # This compensates both for the scipy to mujoco transform (align with y is
        # # [0, 0, 0] in mujoco but [pi/2, 0, 0] in scipy) and the fact that the fly
        # # orientation is already taken into account in the base_camera_rot (see below)
        # # camera is always looking along its -z axis
        # cam_matrix = R.from_euler(
        #     "yxz", fly_z_rot_euler
        # ).as_matrix()  # apply the rotation along the y axis of the cameras
        # cam_matrix = self.base_camera_rot @ cam_matrix
        # cam.xmat = cam_matrix.flatten()

    def render(self, physics: mjcf.Physics, floor_height: float, curr_time: float):
        return super().render(physics, floor_height, curr_time)
