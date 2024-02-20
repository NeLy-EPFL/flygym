import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import dm_control.mujoco
import imageio
import numpy as np
from dm_control import mjcf
from flygym.mujoco.fly import Fly
from scipy.spatial.transform import Rotation as R


class Camera:
    """Camera associated with a fly.

    Attributes
    ----------
    fly : Fly
        The fly to which the camera is associated.
    window_size : Tuple[int, int]
        Size of the rendered images in pixels.
    play_speed : float
        Play speed of the rendered video.
    fps: int
        FPS of the rendered video when played at ``play_speed``.
    timestamp_text : bool
        If True, text indicating the current simulation time will be added to the
        rendered video.
    play_speed_text : bool
        If True, text indicating the play speed will be added to the rendered video.
    draw_contacts : bool
        If True, arrows will be drawn to indicate contact forces between the legs and
        the ground.
    decompose_contacts : bool
        If True, the arrows visualizing contact forces will be decomposed into x-y-z
        components.
    force_arrow_scaling : float
        Scaling factor determining the length of arrows visualizing contact forces.
    tip_length : float
        Size of the arrows indicating the contact forces in pixels.
    contact_threshold : float
        The threshold for contact detection in mN (forces below this magnitude will be
        ignored).
    draw_gravity : bool
        If True, an arrow will be drawn indicating the direction of gravity. This is
        useful during climbing simulations.
    gravity_arrow_scaling : float
        Scaling factor determining the size of the arrow indicating gravity.
    align_camera_with_gravity : bool
        If True, the camera will be rotated such that gravity points down. This is
        useful during climbing simulations.
    camera_follows_fly_orientation : bool
        If True, the camera will be rotated so that it aligns with the fly's
        orientation.
    decompose_colors : Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]
        Colors for the x, y, and z components of the contact force arrows.
    output_path : Optional[Union[str, Path]]
        Path to which the rendered video should be saved. If None, the video will not
        be saved.
    """

    _dm_camera: dm_control.mujoco.Camera

    def __init__(
        self,
        fly: Fly,
        camera_id: str = "Animat/camera_left",
        window_size: Tuple[int, int] = (640, 480),
        play_speed: float = 0.2,
        fps: int = 30,
        timestamp_text: bool = False,
        play_speed_text: bool = True,
        draw_contacts: bool = False,
        decompose_contacts: bool = True,
        force_arrow_scaling: float = 1.0,
        tip_length: float = 10.0,  # number of pixels
        contact_threshold: float = 0.1,
        draw_gravity: bool = False,
        gravity_arrow_scaling: float = 1e-4,
        align_camera_with_gravity: bool = False,
        camera_follows_fly_orientation: bool = False,
        decompose_colors: Tuple[
            Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]
        ] = ((255, 0, 0), (0, 255, 0), (0, 0, 255)),
        output_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize a Camera.

        Parameters
        ----------
        fly : Fly
            The fly to which the camera is associated.
        camera_id : str
            The camera that will be used for rendering, by default
            "Animat/camera_left".
        window_size : Tuple[int, int]
            Size of the rendered images in pixels, by default (640, 480).
        play_speed : float
            Play speed of the rendered video, by default 0.2.
        fps: int
            FPS of the rendered video when played at ``play_speed``, by
            default 30.
        timestamp_text : bool
            If True, text indicating the current simulation time will be added
            to the rendered video.
        play_speed_text : bool
            If True, text indicating the play speed will be added to the
            rendered video.
        draw_contacts : bool
            If True, arrows will be drawn to indicate contact forces between
            the legs and the ground. By default False.
        decompose_contacts : bool
            If True, the arrows visualizing contact forces will be decomposed
            into x-y-z components. By default True.
        force_arrow_scaling : float
            Scaling factor determining the length of arrows visualizing contact
            forces. By default 1.0.
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
            If True, the camera will be rotated such that gravity points down.
            This is useful during climbing simulations. By default False.
        camera_follows_fly_orientation : bool
            If True, the camera will be rotated so that it aligns with the fly's
            orientation. By default False.
        decompose_colors : Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]
            Colors for the x, y, and z components of the contact force arrows.
            By default ((255, 0, 0), (0, 255, 0), (0, 0, 255)).
        output_path : str or Path, optional
            Path to which the rendered video should be saved. If None, the video
            will not be saved. By default None.
        """
        self.fly = fly
        self.window_size = window_size
        self.play_speed = play_speed
        self.fps = fps
        self.timestamp_text = timestamp_text
        self.play_speed_text = play_speed_text
        self.draw_contacts = draw_contacts
        self.decompose_contacts = decompose_contacts
        self.force_arrow_scaling = force_arrow_scaling
        self.tip_length = tip_length
        self.contact_threshold = contact_threshold
        self.draw_gravity = draw_gravity
        self.gravity_arrow_scaling = gravity_arrow_scaling
        self.align_camera_with_gravity = align_camera_with_gravity
        self.camera_follows_fly_orientation = camera_follows_fly_orientation
        self.decompose_colors = decompose_colors
        self.camera_id = camera_id.replace("Animat", fly.name)

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
        self._last_render_time = -np.inf
        self._eff_render_interval = self.play_speed / self.fps
        self._frames: list[np.ndarray] = []

    def _initialize_custom_camera_handling(self, camera_name: str):
        """
        This function is called when the camera is initialized. It can be
        used to customize the camera behavior. I case update_camera_pos is
        True and the camera is within the animat and not a head camera, the
        z position will be fixed to avoid oscillations. If
        self.camera_follows_fly_orientation is True, the camera
        will be rotated to follow the fly orientation (i.e. the front camera
        will always be in front of the fly).
        """

        is_animat = "Animat" in camera_name
        is_visualization_camera = (
            "head" in camera_name
            or "Tarsus" in camera_name
            or "camera_front_zoomin" in camera_name
        )

        is_compound_camera = camera_name not in [
            "Animat/camera_front",
            "Animat/camera_top",
            "Animat/camera_bottom",
            "Animat/camera_back",
            "Animat/camera_right",
            "Animat/camera_left",
        ]

        # always add pos update if it is a head camera
        if is_animat and not is_visualization_camera:
            self.update_camera_pos = True
            self.cam_offset = self._cam.pos
            if is_compound_camera and self.camera_follows_fly_orientation:
                self.camera_follows_fly_orientation = False
                logging.warning(
                    "Overriding `camera_follows_fly_orientation` to False because"
                    "it is never applied to visualization cameras (head, tarsus, ect)"
                    "or non Animat camera."
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
        if self.draw_contacts or self.draw_gravity:
            width, height = self.window_size
            self._dm_camera = dm_control.mujoco.Camera(
                physics,
                camera_id=self.camera_id,
                width=width,
                height=height,
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
            img = self._draw_contacts(img)
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
        self._last_render_time = curr_time
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
        """Draw gravity as an arrow. The arrow is drawn at the top right of the frame."""

        camera_matrix = self._dm_camera.matrix
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

    def _draw_contacts(self, img: np.ndarray) -> np.ndarray:
        """Draw contacts as arrow which length is proportional to the force
        magnitude. The arrow is drawn at the center of the body. It uses the
        camera matrix to transfer from the global space to the pixels space."""

        forces = self.fly.last_obs["contact_forces"]
        pos = self.fly.last_obs["contact_pos"]
        magnitudes = np.linalg.norm(forces, axis=1)
        contact_indices = np.nonzero(magnitudes > self.contact_threshold)[0]

        n_contacts = len(contact_indices)
        # Build an array of start and end points for the force arrows
        if n_contacts == 0:
            return img

        if not self.decompose_contacts:
            arrow_points = np.tile(pos[:, contact_indices], (1, 2)).squeeze()

            arrow_points[:, n_contacts:] += (
                forces[:, contact_indices] * self.force_arrow_scaling
            )
        else:
            arrow_points = np.tile(pos[:, contact_indices], (1, 4)).squeeze()
            for j in range(3):
                arrow_points[j, (j + 1) * n_contacts : (j + 2) * n_contacts] += (
                    forces[contact_indices, j] * self.force_arrow_scaling
                )

        camera_matrix = self._dm_camera.matrix

        # code sample from dm_control demo notebook
        xyz_global = arrow_points

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

        # Draw the contact forces
        for i in range(n_contacts):
            pts1 = [x[i], y[i]]
            if self.decompose_contacts:
                for j in range(3):
                    pts2 = np.array(
                        [x[i + (j + 1) * n_contacts], y[i + (j + 1) * n_contacts]]
                    )
                    if (
                        np.linalg.norm(
                            arrow_points[:, i]
                            - arrow_points[:, i + (j + 1) * n_contacts]
                        )
                        > self.contact_threshold
                    ):
                        arrow_length = np.linalg.norm(pts2 - pts1)
                        if arrow_length > 1e-2:
                            r = self.tip_length / arrow_length
                        else:
                            r = 1e-4
                        img = cv2.arrowedLine(
                            img,
                            pts1,
                            pts2,
                            color=self.decompose_colors[j],
                            thickness=2,
                            tipLength=r,
                        )
            else:
                pts2 = np.array([x[i + n_contacts], y[i + n_contacts]])
                r = self.tip_length / np.linalg.norm(pts2 - pts1)
                img = cv2.arrowedLine(
                    img,
                    pts1,
                    pts2,
                    color=(255, 0, 0),
                    thickness=2,
                    tipLength=r,
                )
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

        num_stab_frames = int(np.ceil(stabilization_time / self._eff_render_interval))

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving video to {path}")
        with imageio.get_writer(path, fps=self.fps) as writer:
            for frame in self._frames[num_stab_frames:]:
                writer.append_data(frame)
