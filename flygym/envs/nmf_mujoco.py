import numpy as np
import imageio
import logging
import sys
from typing import List, Tuple, Dict, Any, Optional, SupportsFloat, Union
from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType

try:
    import mujoco
    import dm_control
    from dm_control import mjcf
    from dm_control.utils import transformations
except ImportError:
    raise ImportError(
        "MuJoCo prerequisites not installed. Please install the prerequisites "
        "by running `pip install flygym[mujoco]` or "
        '`pip install -e ."[mujoco]"` if installing locally.'
    )
try:
    import cv2
except ImportError:
    pass

import flygym.util.vision as vision
import flygym.util.config as config
import flygym.util.data as data
from flygym.arena import BaseArena
from flygym.arena.mujoco_arena import FlatTerrain
from flygym.state import BaseState, stretched_pose


@dataclass
class MuJoCoParameters:
    """Parameters of the MuJoCo simulation.

    Attributes
    ----------
    timestep : float
        Simulation timestep in seconds.
    joint_stiffness : float, optional
        Stiffness of actuated joints, by default 0.05.
    joint_damping : float, optional
        Damping coefficient of actuated joints, by default 0.06.
    actuator_kp : float, optional
        Position gain of the actuators, by default 18.0.
    tarsus_stiffness : float, optional
        Stiffness of the passive, compliant tarsus joints, by default 2.2.
    tarsus_damping : float, optional
        Damping coefficient of the passive, compliant tarsus joints, by
        default 0.126.
    friction : float, optional
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001)
    gravity : Tuple[float, float, float], optional
        Gravity in (x, y, z) axes, by default (0., 0., -9.81e3). Note that
        the gravity is -9.81 * 1000 due to the scaling of the model.
    contact_solref: Tuple[float, float], optional
        Contact reference parameter as defined in
        https://mujoco.readthedocs.io/en/stable/modeling.html#impedance,
        by default (9.99e-01, 9.999e-01, 1.0e-03, 5.0e-01, 2.0e+00) contacts
        are very stiff (avoid penetration with adhesion
        stiffness could be decreased if instability is a problem)
    contact_solimp: Tuple[float, float, float, float, float], optional
        Contact impedance parameter as defined in
        https://mujoco.readthedocs.io/en/stable/modeling.html#reference,
        by default (9.99e-01, 9.999e-01, 1.0e-03, 5.0e-01, 2.0e+00) contacts
        are very stiff (avoid penetration with adhesion
        stiffness could be decreased if instability is a problem)
    enable_olfaction : bool, optional
        Whether to enable olfaction, by default False.
    enable_vision : bool, optional
        Whether to enable vision, by default False.
    render_raw_vision : bool, optional
        If ``enable_vision`` is True, whether to render the raw vision
        (raw pixel values before binning by ommatidia), by default False.
    render_mode : str, optional
        The rendering mode. Can be "saved" or "headless", by default
        "saved".
    render_window_size : Tuple[int, int], optional
        Size of the rendered images in pixels, by default (640, 480).
    render_playspeed : SupportsFloat, optional
        Play speed of the rendered video, by default 1.0.
    render_fps : int, optional
        FPS of the rendered video when played at ``render_playspeed``, by
        default 60.
    render_camera : str, optional
        The camera that will be used for rendering, by default
        "Animat/camera_left_top".
    vision_refresh_rate : int, optional
        The rate at which the vision sensor is updated, in Hz, by default
        500.
    draw_contacts : bool, optional
        Whether to draw contacts in the simulation. By default False.
    decompose_contacts : bool, optional
        Whether to decompose the contact forces into normal and tangential
    contact_threshold : float, optional
        The threshold for contact detection. By default 0.1.
    tip_length : float, optional
        The length of the contact force arrows in pixels. By default 10.
    enable_adhesion : bool, optional
        Whether to enable adhesion. By default False.
    draw_adhesion : bool, optional
        Whether to signal that adhesion is on by changing the color of the
        concerned leg. By default False.
    adhesion_gain : float, optional
        The gain of the adhesion force. By default 40.
    adhesion_off_duration: float, optional
        The duration of the adhesion off phase in seconds. By default 0.1.
        Used in the get_adheion_vector method. When the leg is predicted to swing
        thw adhesion will be switched off for this duration. Increasing this
        value will might make the swing longer if it was interrupted by the adhesion)
    adhesion_off_refractory_duration: float, optional
        The duration of the adhesion off refractory phase in seconds. By default 0.1.
        (Used in the get_adheion_vector method. When the adhesion was switched
        off for adhesion_off_duration, it will not be switched off again for this
        period of time. This might be able to compensate for noise in the joint
        angles predicting a swing)
    draw_gravtiy : bool, optional
            Whether to draw the gravity vector in the simulation. By default
            False,
    align_camera_with_gravity : bool, optional
        Whether to align the camera with the gravity vector: When adding a
        slope the fly will appear to climb this slope. By default
        False.
    """

    timestep: float = 0.0001
    joint_stiffness: float = 0.05
    joint_damping: float = 0.06
    actuator_kp: float = 18.0
    tarsus_stiffness: float = 2.2
    tarsus_damping: float = 0.05
    friction: float = (1.0, 0.005, 0.0001)
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81e3)
    contact_solref: Tuple[float, float] = (2e-4, 1e3)
    contact_solimp: Tuple[float, float, float, float, float] = (
        9.99e-01,
        9.999e-01,
        1.0e-03,
        5.0e-01,
        2.0e00,
    )
    enable_olfaction: bool = False
    enable_vision: bool = False
    render_raw_vision: bool = False
    render_mode: str = "saved"
    render_window_size: Tuple[int, int] = (640, 480)
    render_playspeed: float = 1.0
    render_fps: int = 60
    render_camera: str = "Animat/camera_left"
    vision_refresh_rate: int = 500
    enable_adhesion: bool = False
    adhesion_gain: float = 20
    adhesion_off_duration: float = 0.02
    adhesion_off_refractory_duration: float = 0.05
    draw_adhesion: bool = False
    draw_markers: bool = False
    draw_contacts: bool = False
    force_arrow_scaling: float = 1.0
    tip_length: float = 10.0  # number of pixels
    decompose_contacts: bool = True
    contact_threshold: float = 0.1
    draw_gravity: bool = False
    gravity_arrow_scaling: float = 1e-4
    align_camera_with_gravity: bool = False
    add_ruler: bool = True


class NeuroMechFlyMuJoCo(gym.Env):
    """A NeuroMechFly environment using MuJoCo as the physics engine.

    Attributes
    ----------
    sim_params : flygym.envs.nmf_mujoco.MuJoCoParameters
        Parameters of the MuJoCo simulation.
    actuated_joints : List[str]
        List of names of actuated joints.
    contact_sensor_placements : List[str]
        List of body parts where contact sensors are placed.
    timestep: float
        Simulation timestep in seconds.
    output_dir : Path
        Directory to save simulation data.
    arena : flygym.arena.BaseWorld
        The arena in which the robot is placed.
    spawn_pos : Tuple[froot_elementloat, float, float], optional
        The (x, y, z) position in the arena defining where the fly will
        be spawn.
    spawn_orient : Tuple[float, float, float, float], optional
        The spawn orientation of the fly, in the "axisangle" format
        (x, y, z, a) where x, y, z define the rotation axis and a
        defines the angle of rotation.
    control : str
        The joint controller type. Can be "position", "velocity", or
        "torque".
    init_pose : flygym.state.BaseState
        Which initial pose to start the simulation from.
    render_mode : str
        The rendering mode. Can be "saved" or "headless".
    last_tarsalseg_names : List[str]
        List of names of end effectors; matches the order of end effector
        sensor readings.
    floor_collisions : List[str]
        List of body parts that can collide with the floor.
    self_collisions : List[str]
        List of body parts that can collide with each other.
    action_space : gymnasium.core.ObsType
        Definition of the simulation's action space as a Gym environment.
    observation_space : gymnasium.core.ObsType
        Definition of the simulation's observation space as a Gym
        environment.
    actuators : List[dm_control.mjcf.Element]
        The MuJoCo actuators.
    model : dm_control.mjcf.RootElement
        The MuJoCo model.
    arena_root = dm_control.mjcf.RootElement
        The root element of the arena.
    floor_contacts : List[dm_control.mjcf.Element]
        The MuJoCo geom pairs that can collide with the floor.
    floor_contact_names : List[str]
        The names of the MuJoCo geom pairs that can collide with the floor.
    self_contacts : List[dm_control.mjcf.Element]
        The MuJoCo geom pairs within the fly model that can collide with
        each other.
    self_contact_names : List[str]
        The names of MuJoCo geom pairs within the fly model that can
        collide with each other.
    joint_sensors : List[dm_control.mjcf.Element]
        The MuJoCo sensors on joint positions, velocities, and forces.
    body_sensors : List[dm_control.mjcf.Element]
        The MuJoCo sensors on the root (thorax) position and orientation.
    end_effector_sensors : List[dm_control.mjcf.Element]
        The position sensors on the end effectors.
    physics: dm_control.mjcf.Physics
        The MuJoCo Physics object built from the arena's MJCF model with
        the fly in it.
    curr_time : float
        The (simulated) time elapsed since the last reset (in seconds).
    curr_visual_input : np.ndarray
        The current visual input from the fly's eyes. This is our
        simulation of what the fly might see through its compound eyes.
        It is a (2, N, 2) array where the first dimension is for the left
        and right eyes, the second dimension is for the N ommatidia, and
        the third dimension is for the two channels.
    curr_raw_visual_input : np.ndarray
        The current raw visual input from the fly's eyes. This is the raw
        camera reading before we simulate what the fly actually sees with
        its compound eyes. It is a (2, H, W, 3) array where the first
        dimension is for the left and right eyes, and the remaining
        dimensions are for the RGB image.
    vision_update_mask : np.ndarray
        The refresh frequency of the visual system is often loser than the
        same as the physics simulation time step. This 1D mask, whose
        size is the same as the number of simulation time steps, indicates
        in which time steps the visual inputs have been refreshed. In other
        words, the visual input frames where this mask is False are
        repititions of the previous updated visual input frames.
    antennae_sensors : Union[None, List[dm_control.mjcf.Element], None]
        If olfaction is enabled, this is a list of the MuJoCo sensors on
        the antenna positions.
    """

    def __init__(
        self,
        sim_params: MuJoCoParameters = None,
        actuated_joints: List = config.all_leg_dofs,
        contact_sensor_placements: List = config.all_tarsi_links,
        output_dir: Optional[Path] = None,
        arena: BaseArena = None,
        spawn_pos: Tuple[float, float, float] = (0.0, 0.0, 0.5),
        spawn_orient: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.1),
        control: str = "position",
        init_pose: BaseState = stretched_pose,
        floor_collisions: Union[str, List[str]] = "legs",
        self_collisions: Union[str, List[str]] = "legs",
        camera_correction: bool = True,
    ) -> None:
        """Initialize a NeuroMechFlyMuJoCo environment.

        Parameters
        ----------
        sim_params : MuJoCoParameters, optional
            Parameters of the MuJoCo simulation. Default parameters of
            ``MuJoCoParameters`` will be used if not specified.
        actuated_joints : List, optional
            List of actuated joint DoFs, by default all leg DoFs.
        contact_sensor_placements : List, optional
            List of geometries on each leg where a contact sensor should be
            placed. By default all tarsi.
        output_dir : Path, optional
            Directory to save simulation data. If ``None``, no data will be
            saved. By default None.
        arena : BaseWorld, optional
            The arena in which the robot is placed. ``FlatTerrain`` will be
            used if not specified.
        spawn_pos : Tuple[float, float, float], optional
            The (x, y, z) position in the arena defining where the fly will
            be spawn, by default (0., 0., 300.).
        spawn_orient : Tuple[float, float, float, float], optional
            The spawn orientation of the fly, in the "axisangle" format
            (x, y, z, a) where x, y, z define the rotation axis and a
            defines the angle of rotation, by default (0., 1., 0., 0.1).
        control : str, optional
            The joint controller type. Can be "position", "velocity", or
            "torque", by default "position".
        init_pose : BaseState, optional
            Which initial pose to start the simulation from. By default
            "stretched" kinematic pose with all legs fully stretched.
        floor_collisions :str
            Which set of collisions should collide with the floor. Can be
            "all", "legs", "tarsi" or a list of body names. By default
            "legs".
        self_collisions : str
            Which set of collisions should collide with each other. Can be
            "all", "legs", "legs-no-coxa", "tarsi", "none", or a list of
            body names. By default "legs".
        """
        if sim_params is None:
            sim_params = MuJoCoParameters()
        if arena is None:
            arena = FlatTerrain()
        self.sim_params = deepcopy(sim_params)
        self.actuated_joints = actuated_joints
        self.contact_sensor_placements = contact_sensor_placements
        self.timestep = sim_params.timestep
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.arena = arena
        self.spawn_pos = spawn_pos
        self.spawn_orient = spawn_orient
        self.control = control
        self.init_pose = init_pose
        self.render_mode = sim_params.render_mode
        self.last_tarsalseg_names = [
            f"{side}{pos}Tarsus5" for side in "LR" for pos in "FMH"
        ]

        if self.sim_params.draw_contacts and "cv2" not in sys.modules:
            logging.warning(
                "Overriding `draw_contacts` to False because OpenCV is required "
                "to draw the arrows but it is not installed."
            )
            self.sim_params.draw_contacts = False

        # Parse collisions specs
        if isinstance(floor_collisions, str):
            self.floor_collisions = config.get_collision_geoms(floor_collisions)
        else:
            self.floor_collisions = floor_collisions
        if isinstance(self_collisions, str):
            self.self_collisions = config.get_collision_geoms(self_collisions)
        else:
            self.self_collisions = self_collisions

        self.n_legs = 6

        self.adhesion_off_duration_steps = int(
            self.sim_params.adhesion_off_duration / self.timestep
        )
        self.adhesion_off_refractory_duration_steps = int(
            self.sim_params.adhesion_off_refractory_duration / self.timestep
        )
        self.adhesion_counter = np.zeros(self.n_legs)
        self.adhesion_refractory_counter = np.zeros(self.n_legs)

        # Order is based on the self.last_tarsalseg_names
        leglift_reference_joint = [
            "Tibia",
            "Femur_roll",
            "Femur_roll",
            "Tibia",
            "Femur_roll",
            "Femur_roll",
        ]
        self.leglift_ref_jnt_id = [
            self.actuated_joints.index("joint_" + tarsus_joint[:2] + joint)
            for tarsus_joint, joint in zip(
                self.last_tarsalseg_names, leglift_reference_joint
            )
        ]
        adhesion_comparison_dir = ["inf", "inf", "inf", "inf", "sup", "sup"]
        self.adhesion_sup_id = np.array(
            [c_dir == "sup" for c_dir in adhesion_comparison_dir]
        )
        self.last_refjnt_angvel = np.zeros(self.n_legs)
        self._last_adhesion = np.zeros(self.n_legs)
        self._active_adhesion = np.zeros(self.n_legs)

        if self.sim_params.draw_adhesion and not self.sim_params.enable_adhesion:
            logging.warning(
                "Overriding `draw_adhesion` to False because adhesion is not enabled."
            )
            self.sim_params.draw_adhesion = False

        if self.sim_params.draw_adhesion:
            self.leg_adhesion_drawing_segments = np.array(
                [
                    [
                        "Animat/" + tarsus5.replace("5", str(i)) + "_visual"
                        for i in range(1, 6)
                    ]
                    for tarsus5 in self.last_tarsalseg_names
                ]
            )
            self.adhesion_rgba = [1.0, 0.0, 0.0, 0.8]
            self.active_adhesion_rgba = [0.0, 0.0, 1.0, 0.8]
            self.base_rgba = [0.5, 0.5, 0.5, 1.0]

        if self.sim_params.draw_gravity:
            self.last_fly_pos = spawn_pos
            self.gravity_rgba = [1 - 213 / 255, 1 - 90 / 255, 1 - 255 / 255, 1.0]
            cam_name = self.sim_params.render_camera
            self.arrow_offset = np.zeros(3)
            if "bottom" in cam_name or "top" in cam_name:
                self.arrow_offset[0] = -3
                self.arrow_offset[1] = 2
            elif "left" in cam_name or "right" in cam_name:
                self.arrow_offset[2] = 2
                self.arrow_offset[0] = -3
            elif "front" in cam_name or "back" in cam_name:
                self.arrow_offset[2] = 2
                self.arrow_offset[1] = 3

        if self.sim_params.align_camera_with_gravity:
            self.camera_rot = np.eye(3)

        # Define action and observation spaces
        num_dofs = len(actuated_joints)
        action_bound = np.pi if self.control == "position" else np.inf
        num_contacts = len(self.contact_sensor_placements)
        self.action_space, self.observation_space = self._define_spaces(
            num_dofs, action_bound, num_contacts
        )

        # Load NMF model
        self.model = mjcf.from_path(data.mujoco_groundwalking_model_path)
        self._set_geom_colors()

        # Add cameras imitating the fly's eyes
        self.curr_visual_input = None
        self.curr_raw_visual_input = None
        self._last_vision_update_time = -np.inf
        self._eff_visual_render_interval = 1 / self.sim_params.vision_refresh_rate
        self._vision_update_mask = []
        if self.sim_params.enable_vision:
            self._configure_eyes()

        # Define list of actuated joints
        self.actuators = [
            self.model.find("actuator", f"actuator_{control}_{joint}")
            for joint in actuated_joints
        ]
        self._set_actuators_gain()
        self._set_geoms_friction()
        self._set_joints_stiffness_and_damping()
        self._set_compliant_tarsus()

        print("Before")
        if self.sim_params.add_ruler:
            print("Adding ruler")
            self._add_ruler()

        # Add arena and put fly in it
        arena.spawn_entity(self.model, self.spawn_pos, self.spawn_orient)
        self.arena_root = arena.root_element
        self.arena_root.option.timestep = self.timestep

        if camera_correction:
            self._correct_camera_orientation(
                self.sim_params.render_camera.replace("Animat/", "")
            )

        # Add collision/contacts
        floor_collision_geoms = self._parse_collision_specs(floor_collisions)
        self.floor_contacts, self.floor_contact_names = self._define_floor_contacts(
            floor_collision_geoms
        )
        self_collision_geoms = self._parse_collision_specs(self_collisions)
        self.self_contacts, self.self_contact_names = self._define_self_contacts(
            self_collision_geoms
        )

        # Add sensors
        self.joint_sensors = self._add_joint_sensors()
        self.body_sensors = self._add_body_sensors()
        self.end_effector_sensors = self._add_end_effector_sensors()
        self.antennae_sensors = (
            self._add_odor_sensors() if sim_params.enable_olfaction else None
        )
        self._add_force_sensors()
        self.contact_sensor_placements = [
            f"Animat/{body}" for body in self.contact_sensor_placements
        ]
        self.adhesion_actuators = self._add_adhesion_actuators(
            self.sim_params.adhesion_gain
        )
        # Those need to be in the same order as the adhesion sensor
        # (due to comparison with the last adhesion_signal)
        adhesion_sensor_indices = []
        for adhesion_actuator in self.adhesion_actuators:
            for index, contact_sensor in enumerate(self.contact_sensor_placements):
                if f"{contact_sensor}_adhesion" in f"Animat/{adhesion_actuator.name}":
                    adhesion_sensor_indices.append(index)
        self.adhesion_bodies_with_contact_sensors = np.array(adhesion_sensor_indices)

        # Set up physics and apply ad hoc changes to gravity, stiffness, and friction
        self.physics = mjcf.Physics.from_mjcf_model(self.arena_root)
        self.adhesion_actuator_geomid = np.array(
            [
                self.physics.model.geom(
                    "Animat/" + adhesion_actuator.body + "_collision"
                ).id
                for adhesion_actuator in self.adhesion_actuators
            ]
        )

        # Set gravity
        self.set_gravity(self.sim_params.gravity)

        # Apply initial pose.(TARSI MUST HAVE MADE COMPLIANT BEFORE)!
        self._set_init_pose(self.init_pose)

        # Set up a few things for rendering
        self.curr_time = 0
        self._last_render_time = -np.inf
        if sim_params.render_mode != "headless":
            self._eff_render_interval = (
                sim_params.render_playspeed / self.sim_params.render_fps
            )
        self._frames = []

        if self.sim_params.draw_contacts:
            self._last_contact_force = []
            self._last_contact_pos = []

        if self.sim_params.draw_contacts or self.sim_params.draw_gravity:
            width, height = self.sim_params.render_window_size
            self.dm_camera = dm_control.mujoco.Camera(
                self.physics,
                camera_id=self.sim_params.render_camera,
                width=width,
                height=height,
            )
        self.decompose_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

    def _configure_eyes(self):
        for name in ["LEye_cam", "REye_cam"]:
            parent_name, position, euler_angle, rgba = config.sensor_positions[name]
            parent_body = self.model.find("body", parent_name)
            sensor_body = parent_body.add("body", name=f"{name}_body", pos=position)
            sensor_body.add(
                "camera",
                name=name,
                dclass="nmf",
                mode="track",
                euler=euler_angle,
                fovy=config.fovy_per_eye,
            )
            if self.sim_params.draw_markers:
                sensor_body.add(
                    "geom",
                    name=f"{name}_marker",
                    type="sphere",
                    size=[0.06],
                    rgba=rgba,
                )

        # Make list of geometries that are hidden during visual input rendering
        self._geoms_to_hide = []
        for segment in config.hidden_segments_for_vision:
            # all body segments have a visual geom - add this first
            visual_geom_name = f"{segment}_visual"
            self._geoms_to_hide.append(visual_geom_name)
            # some body segments also have a collision geom - add this if it exists
            collision_geom_name = f"{segment}_collision"
            collision_geom = self.model.find("geom", collision_geom_name)
            if collision_geom is not None:
                self._geoms_to_hide.append(collision_geom_name)

    def _parse_collision_specs(self, collision_spec: Union[str, List[str]]):
        if collision_spec == "all":
            return [
                geom.name
                for geom in self.model.find_all("geom")
                if "collision" in geom.name
            ]
        elif isinstance(collision_spec, str):
            return config.get_collision_geoms(collision_spec)
        elif isinstance(collision_spec, list):
            return collision_spec
        else:
            raise ValueError(f"Unrecognized collision spec {collision_spec}")

    def _correct_camera_orientation(self, camera_name: str):
        # Correct the camera orientation by incorporating the spawn rotation
        # of the arena

        # Get the camera
        camera = self.model.find("camera", camera_name)
        if camera is None or camera.mode in ["targetbody", "targetbodycom"]:
            return 0
        if "head" in camera_name or "front_zoomin" in camera_name:
            # Don't correct the head camera
            return camera

        # Add the spawn rotation (keep horizon flat)
        spawn_quat = np.array(
            [
                np.cos(self.spawn_orient[-1] / 2),
                self.spawn_orient[0] * np.sin(self.spawn_orient[-1] / 2),
                self.spawn_orient[1] * np.sin(self.spawn_orient[-1] / 2),
                self.spawn_orient[2] * np.sin(self.spawn_orient[-1] / 2),
            ]
        )

        # Change camera euler to quaternion
        camera_quat = transformations.euler_to_quat(camera.euler)
        new_camera_quat = transformations.quat_mul(
            transformations.quat_inv(spawn_quat), camera_quat
        )
        camera.euler = transformations.quat_to_euler(new_camera_quat)

        # Elevate the camera slightly gives a better view of the arena
        if not "zoomin" in camera_name:
            camera.pos = camera.pos + [0.0, 0.0, 0.5]
        if "front" in camera_name:
            camera.pos[2] = camera.pos[2] + 1.0

        return camera

    def _set_geom_colors(self):
        for bodypart in config.colors.keys():
            if bodypart in ["A12345", "A6"]:
                self.model.asset.add(
                    "texture",
                    name=f"{bodypart}_texture",
                    type="cube",
                    builtin="gradient",
                    mark="random",
                    random=0.3,
                    markrgb=config.colors[bodypart][2],
                    rgb1=config.colors[bodypart][0],
                    rgb2=config.colors[bodypart][1],
                    width=200,
                    height=200,
                )
                self.model.asset.add(
                    "material",
                    name=f"{bodypart}_material",
                    texture=f"{bodypart}_texture",
                    specular=0.0,
                    shininess=0.0,
                    reflectance=0.0,
                    texuniform=True,
                    texrepeat=[1, 1],
                )
            elif bodypart in [
                "thorax",
                "coxa",
                "femur",
                "tibia",
                "tarsus",
                "head",
                "antennas",
                "proboscis",
            ]:
                size = 500
                random = 0.05

                if bodypart in ["thorax", "head"]:
                    size = 50
                    random = 0.3
                elif bodypart in ["antennas", "proboscis"]:
                    size = 50
                    random = 0.1

                self.model.asset.add(
                    "texture",
                    name=f"{bodypart}_texture",
                    type="cube",
                    builtin="flat",
                    rgb1=config.colors[bodypart][0],
                    rgb2=config.colors[bodypart][0],
                    markrgb=config.colors[bodypart][1],
                    mark="random",
                    random=random,
                    width=size,
                    height=size,
                )
                self.model.asset.add(
                    "material",
                    name=f"{bodypart}_material",
                    texture=f"{bodypart}_texture",
                    rgba=config.colors[bodypart][2],
                    specular=0.0,
                    shininess=0.0,
                    reflectance=0.0,
                    texuniform=True,
                    texrepeat=[1, 1],
                )
            else:
                self.model.asset.add(
                    "material",
                    name=f"{bodypart}_material",
                    specular=0.0,
                    shininess=0.0,
                    reflectance=0.0,
                    rgba=config.colors[bodypart],
                )

        for geom in self.model.find_all("geom"):
            if "visual" in geom.name:
                if geom.name[1:-7] == "Eye":
                    geom.material = "eyes_material"
                elif geom.name[2:-7] == "Coxa":
                    geom.material = "coxa_material"
                elif geom.name[2:-7] == "Femur":
                    geom.material = "femur_material"
                elif geom.name[2:-7] == "Tibia":
                    geom.material = "tibia_material"
                elif geom.name[2:-8] == "Tarsus":
                    geom.material = "tarsus_material"
                elif geom.name[1:-7] == "Wing":
                    geom.material = "wings_material"
                elif geom.name[0:-7] in ["A1A2", "A3", "A4", "A5"]:
                    geom.material = "A12345_material"
                elif geom.name[0:-7] == "A6":
                    geom.material = "A6_material"
                elif geom.name[0:-7] == "Thorax":
                    geom.material = "thorax_material"
                elif geom.name[0:-7] in ["Haustellum", "Rostrum"]:
                    geom.material = "proboscis_material"
                elif geom.name[1:-7] == "Arista":
                    geom.material = "aristas_material"
                elif geom.name[1:-7] in ["Pedicel", "Funiculus"]:
                    geom.material = "antennas_material"
                elif geom.name[1:-7] == "Haltere":
                    geom.material = "halteres_material"
                elif geom.name[:-7] == "Head":
                    geom.material = "head_material"
                else:
                    geom.material = "body_material"

    def _add_ruler(self):
        pos, orient = self.arena.get_spawn_position(self.spawn_pos, self.spawn_orient)
        pos = np.array(pos)
        pos[1] -= 5.0

        size = 100.0

        ruler_path = "../flygym/data/background/white-ruler_1-8c.png"
        gradient=False
        try:
            ruler_texture = self.arena.root_element.asset.add(
                "texture",
                name=f"ruler_texture",
                type="cube",
                rgb1=[1.0, 1.0, 1.0],
                fileup=ruler_path,
                filedown=ruler_path,
                width=size,
                height=size,
            )
        except FileNotFoundError:
            print("Ruler image not found, use gradient instead")
            ruler_texture = self.arena.root_element.asset.add(
                "texture",
                name=f"ruler_texture",
                type="cube",
                builtin="gradient",
                rgb1=[0.0, 0.0, 0.0],
                rgb2=[1.0, 1.0, 1.0],
                width=size,
                height=size,
            )
            gradient = True
        ruler_material = self.arena.root_element.asset.add(
            "material",
            name="ruler_material",
            texture="ruler_texture",
        )

        if gradient:
            self.ruler = self.arena.root_element.worldbody.add(
                "geom",
                name="ruler",
                type="box",
                pos=pos,
                size=[10.0, 1.0, 1.0],
                mass=0.0,
                contype=0,
                conaffinity=0,
                material="ruler_material",
            )
        else:
            print("Add multi ruler")
            pos[0] -= 5.0
            for i in range(10):
                self.ruler = self.arena.root_element.worldbody.add(
                    "geom",
                    name=f"ruler_{i}",
                    type="box",
                    pos=pos + np.array([i * 2.0, 0.0, 0.0]),
                    size=[2.0, 1.0, 1.0],
                    mass=0.0,
                    contype=0,
                    conaffinity=0,
                    material="ruler_material",
                )

    def _define_spaces(self, num_dofs, action_bound, num_contacts):
        action_space = {
            "joints": spaces.Box(
                low=-action_bound, high=action_bound, shape=(num_dofs,)
            ),
            "adhesion": spaces.Discrete(n=2, start=0),  # 0: no adhesion, 1: adhesion
        }
        observation_space = {
            # joints: shape (3, num_dofs): (pos, vel, torque) of each DoF
            "joints": spaces.Box(low=-np.inf, high=np.inf, shape=(3, num_dofs)),
            # fly: shape (4, 3):
            # 0th row: x, y, z position of the fly in arena
            # 1st row: x, y, z velocity of the fly in arena
            # 2nd row: orientation of fly around x, y, z axes
            # 3rd row: rate of change of fly orientation
            "fly": spaces.Box(low=-np.inf, high=np.inf, shape=(4, 3)),
            # contact forces: readings of the touch contact sensors, one
            # placed for each of the ``contact_sensor_placements``
            "contact_forces": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3, num_contacts),
            ),
            # x, y, z positions of the end effectors (tarsus-5 segments)
            "end_effectors": spaces.Box(low=-np.inf, high=np.inf, shape=(3 * 6,)),
        }
        return action_space, observation_space

    def _set_actuators_gain(self):
        for actuator in self.actuators:
            actuator.kp = self.sim_params.actuator_kp

    def _set_geoms_friction(self):
        for geom in self.model.find_all("geom"):
            if "collision" in geom.name:
                geom.friction = self.sim_params.friction

    def _set_joints_stiffness_and_damping(self):
        for joint in self.model.find_all("joint"):
            if joint.name in self.actuated_joints:
                joint.stiffness = self.sim_params.joint_stiffness
                joint.damping = self.sim_params.joint_damping

    def _define_self_contacts(self, self_collisions_geoms):
        self_contact_pairs = []
        self_contact_pairs_names = []
        for geom1 in self_collisions_geoms:
            for geom2 in self_collisions_geoms:
                is_duplicate = f"{geom1}_{geom2}" in self_contact_pairs_names
                if geom1 != geom2 and not is_duplicate:
                    # Do not add contact if the parent bodies have a child parent
                    # relationship
                    body1 = self.model.find("geom", geom1).parent
                    body2 = self.model.find("geom", geom2).parent
                    body1_children = [
                        child.name
                        for child in body1.all_children()
                        if child.tag == "body"
                    ]
                    body2_children = [
                        child.name
                        for child in body2.all_children()
                        if child.tag == "body"
                    ]

                    if not (
                        body1.name == body2.name
                        or body1.name in body2_children
                        or body2.name in body1_children
                        or body1.name in body2.parent.name
                        or body2.name in body1.parent.name
                    ):
                        contact_pair = self.model.contact.add(
                            "pair",
                            name=f"{geom1}_{geom2}",
                            geom1=geom1,
                            geom2=geom2,
                            solref=self.sim_params.contact_solref,
                            solimp=self.sim_params.contact_solimp,
                            margin=0.0,  # change margin to avoid penetration
                        )
                        self_contact_pairs.append(contact_pair)
                        self_contact_pairs_names.append(f"{geom1}_{geom2}")
        return self_contact_pairs, self_contact_pairs_names

    def _define_floor_contacts(self, floor_collisions_geoms):
        floor_contact_pairs = []
        floor_contact_pairs_names = []
        ground_id = 0

        for geom in self.arena_root.find_all("geom"):
            if geom.name is None:
                is_ground = True
            elif "visual" in geom.name or "collision" in geom.name:
                is_ground = False
            elif "cam" in geom.name or "sensor" in geom.name:
                is_ground = False
            else:
                is_ground = True
            if is_ground:
                for animat_geom_name in floor_collisions_geoms:
                    if geom.name is None:
                        geom.name = f"groundblock_{ground_id}"
                        ground_id += 1
                    mean_friction = np.mean(
                        [
                            self.sim_params.friction,  # fly friction
                            self.arena.friction,  # arena ground friction
                        ],
                        axis=0,
                    )
                    floor_contact_pair = self.arena_root.contact.add(
                        "pair",
                        name=f"{geom.name}_{animat_geom_name}",
                        geom1=f"Animat/{animat_geom_name}",
                        geom2=f"{geom.name}",
                        solref=self.sim_params.contact_solref,
                        solimp=self.sim_params.contact_solimp,
                        margin=0.0,  # change margin to avoid penetration
                        friction=np.repeat(
                            mean_friction,
                            (2, 1, 2),
                        ),
                    )
                    floor_contact_pairs.append(floor_contact_pair)
                    floor_contact_pairs_names.append(f"{geom.name}_{animat_geom_name}")

        return floor_contact_pairs, floor_contact_pairs_names

    def _add_joint_sensors(self):
        joint_sensors = []
        for joint in self.actuated_joints:
            joint_sensors.extend(
                [
                    self.model.sensor.add(
                        "jointpos", name=f"jointpos_{joint}", joint=joint
                    ),
                    self.model.sensor.add(
                        "jointvel", name=f"jointvel_{joint}", joint=joint
                    ),
                    self.model.sensor.add(
                        "actuatorfrc",
                        name=f"actuatorfrc_position_{joint}",
                        actuator=f"actuator_position_{joint}",
                    ),
                    self.model.sensor.add(
                        "actuatorfrc",
                        name=f"actuatorfrc_velocity_{joint}",
                        actuator=f"actuator_velocity_{joint}",
                    ),
                    self.model.sensor.add(
                        "actuatorfrc",
                        name=f"actuatorfrc_motor_{joint}",
                        actuator=f"actuator_torque_{joint}",
                    ),
                ]
            )
        return joint_sensors

    def _add_body_sensors(self):
        lin_pos_sensor = self.model.sensor.add(
            "framepos", name="thorax_pos", objtype="body", objname="Thorax"
        )
        lin_vel_sensor = self.model.sensor.add(
            "framelinvel", name="thorax_linvel", objtype="body", objname="Thorax"
        )
        ang_pos_sensor = self.model.sensor.add(
            "framequat", name="thorax_quat", objtype="body", objname="Thorax"
        )
        ang_vel_sensor = self.model.sensor.add(
            "frameangvel", name="thorax_angvel", objtype="body", objname="Thorax"
        )
        return [lin_pos_sensor, lin_vel_sensor, ang_pos_sensor, ang_vel_sensor]

    def _add_end_effector_sensors(self):
        end_effector_sensors = []
        for name in self.last_tarsalseg_names:
            sensor = self.model.sensor.add(
                "framepos",
                name=f"{name}_pos",
                objtype="body",
                objname=name,
            )
            end_effector_sensors.append(sensor)
        return end_effector_sensors

    def _add_odor_sensors(self):
        sensor_names = [
            "LAntenna_sensor",
            "RAntenna_sensor",
            "LMaxillaryPalp_sensor",
            "RMaxillaryPalp_sensor",
        ]
        antennae_sensors = []
        for name in sensor_names:
            parent_name, position, rgba = config.sensor_positions[name]
            parent_body = self.model.find("body", parent_name)
            sensor_body = parent_body.add("body", name=f"{name}_body", pos=position)
            sensor = self.model.sensor.add(
                "framepos",
                name=f"{name}_pos_sensor",
                objtype="body",
                objname=f"{name}_body",
            )
            antennae_sensors.append(sensor)
            if self.sim_params.draw_markers:
                sensor_body.add(
                    "geom",
                    name=f"{name}_marker",
                    type="sphere",
                    size=[0.06],
                    rgba=rgba,
                )
            # sensor = self.model.sensor.add(
            #     "framepos",
            #     name=f"{name}_pos",
            #     objtype="body",
            #     objname=name,
            # )

        return antennae_sensors

    def _add_force_sensors(self):
        """
        Add force sensors to the tracked bodies
        Without them the cfrc_ext is zero
        Returns
        -------
        All force sensors
        """
        force_sensors = []
        for tracked_geom in self.contact_sensor_placements:
            body = self.model.find("body", tracked_geom)
            site = body.add(
                "site",
                name=f"{tracked_geom}_site",
                pos=[0, 0, 0],
                size=np.ones(3) * 0.005,
            )
            force_sensor = self.model.sensor.add(
                "force", name=f"force_{body.name}", site=site.name
            )
            force_sensors.append(force_sensor)

        return force_sensors

    def _add_adhesion_actuators(self, gain):
        adhesion_actuators = []
        for name in self.last_tarsalseg_names:
            adhesion_actuators.append(
                self.model.actuator.add(
                    "adhesion",
                    name=f"{name}_adhesion",
                    gain=f"{gain}",
                    body=name,
                    ctrlrange="0 1000000",
                    forcerange="-inf inf",
                )
            )
        return adhesion_actuators

    def _set_init_pose(self, init_pose: Dict[str, float]):
        with self.physics.reset_context():
            for i in range(len(self.actuated_joints)):
                curr_joint = self.actuators[i].joint.name
                if (curr_joint in self.actuated_joints) and (curr_joint in init_pose):
                    animat_name = f"Animat/{curr_joint}"
                    self.physics.named.data.qpos[animat_name] = init_pose[curr_joint]

    def _set_compliant_tarsus(self):
        """Set the Tarsus2/3/4/5 to be compliant by setting the stiffness
        and damping to a low value"""
        stiffness = self.sim_params.tarsus_stiffness
        damping = self.sim_params.tarsus_damping
        for side in "LR":
            for pos in "FMH":
                for tarsus_link in range(2, 5 + 1):
                    joint = self.model.find(
                        "joint", f"joint_{side}{pos}Tarsus{tarsus_link}"
                    )
                    joint.stiffness = stiffness
                    joint.damping = damping

    def set_gravity(self, gravity: List[float], rot_mat: np.ndarray = None):
        """Set the gravity of the environment.
        The change in the point of view has been extensively tested for the simple cameras
        (left right top bottom front back) but not for the composed ones

        In any case set slope is the way to go to change the slope of the floor

        Parameters
        ----------
        gravity : List[float]
            The gravity vector.
        rot_mat : np.ndarray, optional
            The rotation matrix to align the camera with the gravity vector, by default None
        """
        # Only change the angle of the camera if the new gravity vector and the camera angle are compatible
        camera_is_compatible = False
        if (
            "left" in self.sim_params.render_camera
            or "right" in self.sim_params.render_camera
        ):
            if not gravity[1] > 0:
                camera_is_compatible = True
        # elif "top" in self.sim_params.camera_name or "bottom" in self.sim_params.camera_name:
        elif (
            "front" in self.sim_params.render_camera
            or "back" in self.sim_params.render_camera
        ):
            if not gravity[1] > 0:
                camera_is_compatible = True

        if rot_mat is not None and self.sim_params.align_camera_with_gravity:
            self.camera_rot = rot_mat
        elif camera_is_compatible:
            normalised_gravity = (np.array(gravity) / np.linalg.norm(gravity)).reshape(
                (1, 3)
            )
            downward_ref = np.array([0.0, 0.0, -1.0]).reshape((1, 3))

            if (
                not (normalised_gravity == downward_ref).all()
                and self.sim_params.align_camera_with_gravity
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

                self.camera_rot = rot_mat.T

        self.physics.model.opt.gravity[:] = gravity

    def set_slope(self, slope: float, rot_axis="y"):
        """Set the slope of the environment. And modify the camera orientation
        so that gravity is always pointing down.
        Always use as reference a slope of zeros and not the previous slope.

        Parameters
        ----------
        slope : float
            The desired_slope of the environment in degrees.
        rot_axis : str, optional
            The axis about which the slope is applied, by default "y".
        """
        rot_mat = np.eye(3)
        if rot_axis == "x":
            rot_mat = transformations.rotation_x_axis(np.deg2rad(slope))
        elif rot_axis == "y":
            rot_mat = transformations.rotation_y_axis(np.deg2rad(slope))
        elif rot_axis == "z":
            rot_mat = transformations.rotation_z_axis(np.deg2rad(slope))
        new_gravity = np.dot(rot_mat, self.sim_params.gravity)
        self.set_gravity(new_gravity, rot_mat)

        return 0

    def reset(self) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset the Gym environment.

        Returns
        -------
        ObsType
            The observation as defined by the environment.
        Dict[str, Any]
            Any additional information that is not part of the observation.
            This is an empty dictionary by default but the user can
            override this method to return additional information.
        """
        self.physics.reset()
        if np.any(self.physics.model.opt.gravity[:] - self.sim_params.gravity > 1e-3):
            self.set_gravity(self.sim_params.gravity)
            if self.sim_params.align_camera_with_gravity:
                self.camera_rot = np.eye(3)
        self.curr_time = 0
        self._set_init_pose(self.init_pose)
        self._frames = []
        self._last_render_time = -np.inf
        self._last_vision_update_time = -np.inf
        self.curr_raw_visual_input = None
        self.curr_visual_input = None
        self._vision_update_mask = []
        return self.get_observation(), self.get_info()

    def step(
        self, action: ObsType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Step the Gym environment.

        Parameters
        ----------
        action : ObsType
            Action dictionary as defined by the environment's action space.

        Returns
        -------
        ObsType
            The observation as defined by the environment.
        SupportsFloat
            The reward as defined by the environment.
        bool
            Whether the episode has terminated due to factors that are
            defined within the Markov Decision Process (eg. task
            completion/failure, etc).
        bool
            Whether the episode has terminated due to factors beyond the
            Markov Decision Process (eg. time limit, etc).
        Dict[str, Any]
            Any additional information that is not part of the observation.
            This is an empty dictionary by default but the user can
            override this method to return additional information.
        """
        self.arena.step(dt=self.timestep, physics=self.physics)
        self.physics.bind(self.actuators).ctrl = action["joints"]
        if self.sim_params.enable_adhesion:
            self.physics.bind(self.adhesion_actuators).ctrl = action["adhesion"]
            self._last_adhesion = action["adhesion"]

        self.physics.step()
        self.curr_time += self.timestep
        observation = self.get_observation()
        reward = self.get_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.get_info()
        return observation, reward, terminated, truncated, info

    def get_adhesion_vector(self):
        adhesion = np.ones(len(self.last_tarsalseg_names))
        adhesion[
            np.logical_and(
                self.adhesion_sup_id,
                self.last_refjnt_angvel > config.adhesion_speed_thresholds,
            )
        ] = 0
        adhesion[
            np.logical_and(
                ~self.adhesion_sup_id,
                self.last_refjnt_angvel < config.adhesion_speed_thresholds,
            )
        ] = 0
        adhesion[self.adhesion_counter > 0] = 0
        # During the refractory period, adhesion is ON; by defualt adhesion
        # is on only switch it off when lifting the leg
        adhesion[self.adhesion_refractory_counter > 0] = 1

        self.adhesion_counter[
            np.logical_or(adhesion <= 0, self.adhesion_counter > 0)
        ] += 1
        self.adhesion_refractory_counter[
            np.logical_or(
                self.adhesion_counter > self.adhesion_off_duration_steps,
                self.adhesion_refractory_counter > 0,
            )
        ] += 1
        self.adhesion_refractory_counter[
            self.adhesion_refractory_counter
            > self.adhesion_off_refractory_duration_steps
        ] = 0
        self.adhesion_counter[
            self.adhesion_counter > self.adhesion_off_duration_steps
        ] = 0
        return adhesion

    def render(self):
        """Call the ``render`` method to update the renderer. It should be
        called every iteration; the method will decide by itself whether
        action is required."""
        if self.render_mode == "headless":
            return None
        if self.curr_time < self._last_render_time + self._eff_render_interval:
            return None
        if self.render_mode == "saved":
            width, height = self.sim_params.render_window_size
            camera = self.sim_params.render_camera
            if self.sim_params.draw_adhesion:
                self._draw_adhesion()
            if self.sim_params.align_camera_with_gravity:
                self._rotate_camera()
            img = self.physics.render(width=width, height=height, camera_id=camera)
            if self.sim_params.draw_contacts:
                img = self._draw_contacts(img)
            if self.sim_params.draw_gravity:
                img = self._draw_gravity(img)

            self._frames.append(img.copy())
            self._last_render_time = self.curr_time
            return self._frames
        else:
            raise NotImplementedError

    def _rotate_camera(self):
        # get camera
        cam = self.model.find("camera", self.sim_params.render_camera.split("/")[-1])
        cam = self.physics.bind(cam)
        # rotate the cam
        cam_matrix_base = getattr(cam, "xmat").copy()
        cam_matrix = self.camera_rot @ cam_matrix_base.reshape(3, 3)
        setattr(cam, "xmat", cam_matrix.flatten())

        return 0

    def _draw_adhesion(self):
        """Highlight the tarsal segments of the leg having adhesion"""
        if np.any(self._last_adhesion == 1):
            self.physics.named.model.geom_rgba[
                self.leg_adhesion_drawing_segments[self._last_adhesion == 1].flatten()
            ] = self.adhesion_rgba
        if np.any(self._active_adhesion):
            self.physics.named.model.geom_rgba[
                self.leg_adhesion_drawing_segments[self._active_adhesion].flatten()
            ] = self.active_adhesion_rgba
        if np.any(self._last_adhesion == 0):
            self.physics.named.model.geom_rgba[
                self.leg_adhesion_drawing_segments[self._last_adhesion == 0].flatten()
            ] = self.base_rgba
        return

    def _draw_gravity(self, img: np.ndarray) -> np.ndarray:
        """Draw gravity as an arrow. The arrow is drown at the top right of the frame."""

        camera_matrix = self.dm_camera.matrix

        if self.sim_params.align_camera_with_gravity:
            arrow_start = self.last_fly_pos + self.camera_rot @ self.arrow_offset
        else:
            arrow_start = self.last_fly_pos + self.arrow_offset

        arrow_end = (
            arrow_start
            + self.physics.model.opt.gravity * self.sim_params.gravity_arrow_scaling
        )

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
        img = cv2.arrowedLine(img, (x[0], y[0]), (x[1], y[1]), self.gravity_rgba, 10)

        return img

    def _draw_contacts(self, img: np.ndarray) -> np.ndarray:
        """Draw contacts as arrow wich length is proportional to the force
        magnitude. The arrow is drown at the center of the body. It uses the
        camera matrix to transfer from the global space to the pixels space."""
        contact_forces = np.linalg.norm(self._last_contact_force, axis=0)
        contact_indexes = np.nonzero(
            contact_forces > self.sim_params.contact_threshold
        )[0]

        n_contacts = len(contact_indexes)
        # Build an array of start and end points for the force arrows
        if n_contacts > 0:
            if not self.sim_params.decompose_contacts:
                force_arrow_points = np.tile(
                    self._last_contact_pos[:, contact_indexes], (1, 2)
                ).squeeze()

                force_arrow_points[:, n_contacts:] += (
                    self._last_contact_force[:, contact_indexes]
                    * self.sim_params.force_arrow_scaling
                )
            else:
                force_arrow_points = np.tile(
                    self._last_contact_pos[:, contact_indexes], (1, 4)
                ).squeeze()
                for j in range(3):
                    force_arrow_points[
                        j, (j + 1) * n_contacts : (j + 2) * n_contacts
                    ] += (
                        self._last_contact_force[j, contact_indexes]
                        * self.sim_params.force_arrow_scaling
                    )

            camera_matrix = self.dm_camera.matrix

            # code sample from dm_control demo notebook
            xyz_global = force_arrow_points

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
                if self.sim_params.decompose_contacts:
                    for j in range(3):
                        pts2 = np.array(
                            [x[i + (j + 1) * n_contacts], y[i + (j + 1) * n_contacts]]
                        )
                        if (
                            np.linalg.norm(
                                force_arrow_points[:, i]
                                - force_arrow_points[:, i + (j + 1) * n_contacts]
                            )
                            > self.sim_params.contact_threshold
                        ):
                            arrow_length = np.linalg.norm(pts2 - pts1)
                            if arrow_length > 1e-2:
                                r = self.sim_params.tip_length / arrow_length
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
                    r = self.sim_params.tip_length / np.linalg.norm(pts2 - pts1)
                    img = cv2.arrowedLine(
                        img,
                        pts1,
                        pts2,
                        color=(255, 0, 0),
                        thickness=2,
                        tipLength=r,
                    )
        return img

    def _update_vision(self) -> np.ndarray:
        next_render_time = (
            self._last_vision_update_time + self._eff_visual_render_interval
        )
        # avoid floating point errors: when too close, update anyway
        if self.curr_time + 0.5 * self.timestep < next_render_time:
            self._vision_update_mask.append(False)
            return
        self._vision_update_mask.append(True)
        raw_visual_input = []
        ommatidia_readouts = []
        for geom in self._geoms_to_hide:
            self.physics.named.model.geom_rgba[f"Animat/{geom}"] = [0.5, 0.5, 0.5, 0]
        for side in ["L", "R"]:
            raw_img = self.physics.render(
                width=config.raw_img_width_px,
                height=config.raw_img_height_px,
                camera_id=f"Animat/{side}Eye_cam",
            )
            fish_img = vision.correct_fisheye(
                raw_img,
                config.fisheye_distortion_coefficient,
                config.fisheye_zoom,
                config.raw_img_height_px,
                config.raw_img_width_px,
            )
            readouts_per_eye = vision.raw_image_to_hex_pxls(
                np.ascontiguousarray(fish_img),
                vision.num_pixels_per_ommatidia,
                vision.ommatidia_id_map,
            )
            ommatidia_readouts.append(readouts_per_eye)
            raw_visual_input.append(fish_img)
        for geom in self._geoms_to_hide:
            self.physics.named.model.geom_rgba[f"Animat/{geom}"] = [0.5, 0.5, 0.5, 1]
        self.curr_visual_input = np.array(ommatidia_readouts)
        if self.sim_params.render_raw_vision:
            self.curr_raw_visual_input = np.array(raw_visual_input)
        self._last_vision_update_time = self.curr_time

    @property
    def vision_update_mask(self) -> np.ndarray:
        return np.array(self._vision_update_mask[1:])

    def get_observation(self) -> Tuple[ObsType, Dict[str, Any]]:
        """Get observation without stepping the physics simulation.

        Returns
        -------
        ObsType
            The observation as defined by the environment.
        """
        # joint sensors
        joint_obs = np.zeros((3, len(self.actuated_joints)))
        joint_sensordata = self.physics.bind(self.joint_sensors).sensordata
        for i, joint in enumerate(self.actuated_joints):
            base_idx = i * 5
            # pos and vel
            joint_obs[:2, i] = joint_sensordata[base_idx : base_idx + 2]
            # torque from pos/vel/motor actuators
            joint_obs[2, i] = joint_sensordata[base_idx + 2 : base_idx + 5].sum()
        joint_obs[2, :] *= 1e-9  # convert to N

        if self.sim_params.enable_adhesion:
            self.last_refjnt_angvel = joint_obs[1, self.leglift_ref_jnt_id]
        # fly position and orientation
        cart_pos = self.physics.bind(self.body_sensors[0]).sensordata
        cart_vel = self.physics.bind(self.body_sensors[1]).sensordata
        quat = self.physics.bind(self.body_sensors[2]).sensordata
        # ang_pos = transformations.quat_to_euler(quat)
        ang_pos = R.from_quat(quat).as_euler("xyz")  # explicitly use intrinsic
        ang_pos[0] *= -1  # flip roll??
        ang_vel = self.physics.bind(self.body_sensors[3]).sensordata
        fly_pos = np.array([cart_pos, cart_vel, ang_pos, ang_vel])

        if self.sim_params.draw_gravity:
            self.last_fly_pos = cart_pos

        # contact forces from crf_ext (first three componenents are rotational (torque ?))
        contact_forces = (
            self.physics.named.data.cfrc_ext[self.contact_sensor_placements][
                :, 3:
            ].copy()
        ).T
        if self.sim_params.enable_adhesion:
            self.last_refjnt_angvel = joint_obs[1, self.leglift_ref_jnt_id]

            # Adhesion inputs force in the contact. Lets compute this force
            # and remove it from the contact forces
            contactid_normal = {}
            self._active_adhesion = np.zeros(self.n_legs, dtype=bool)
            for contact in self.physics.data.contact:
                id = np.where(self.adhesion_actuator_geomid == contact.geom1)
                if len(id[0]) > 0 and contact.exclude == 0:
                    contact_sensor_id = self.adhesion_bodies_with_contact_sensors[id][0]
                    if contact_sensor_id in contactid_normal:
                        contactid_normal[contact_sensor_id].append(contact.frame[:3])
                    else:
                        contactid_normal[contact_sensor_id] = [contact.frame[:3]]
                    self._active_adhesion[id] = True
                id = np.where(self.adhesion_actuator_geomid == contact.geom2)
                if len(id[0]) > 0 and contact.exclude == 0:
                    contact_sensor_id = self.adhesion_bodies_with_contact_sensors[id][0]
                    if contact_sensor_id in contactid_normal:
                        contactid_normal[contact_sensor_id].append(contact.frame[:3])
                    else:
                        contactid_normal[contact_sensor_id] = [contact.frame[:3]]
                    self._active_adhesion[id] = True

            for contact_sensor_id, normal in contactid_normal.items():
                adh_actuator_id = (
                    self.adhesion_bodies_with_contact_sensors == contact_sensor_id
                )
                if self._last_adhesion[adh_actuator_id] > 0:
                    if len(np.shape(normal)) > 1:
                        normal = np.mean(normal, axis=0)
                    contact_forces[:, contact_sensor_id] -= (
                        self.sim_params.adhesion_gain * normal
                    )

        # if draw contacts same last contact forces and positiions
        if self.sim_params.draw_contacts:
            self._last_contact_force = contact_forces
            self._last_contact_pos = (
                self.physics.named.data.xpos[self.contact_sensor_placements].copy().T
            )

        # end effector position
        ee_pos = self.physics.bind(self.end_effector_sensors).sensordata.copy()

        obs = {
            "joints": joint_obs,
            "fly": fly_pos,
            "contact_forces": contact_forces,
            "end_effectors": ee_pos,
        }

        # olfaction
        if self.sim_params.enable_olfaction:
            antennae_pos = self.physics.bind(self.antennae_sensors).sensordata
            odor_intensity = self.arena.get_olfaction(antennae_pos.reshape(4, 3))
            obs["odor_intensity"] = odor_intensity

        # vision
        if self.sim_params.enable_vision:
            self._update_vision()
            obs["vision"] = self.curr_visual_input
            if self.sim_params.render_raw_vision:
                obs["raw_vision"] = self.curr_raw_visual_input

        return obs

    def get_reward(self):
        """Get the reward for the current state of the environment. This
        method always returns 0 unless extended by the user.

        Returns
        -------
        SupportsFloat
            The reward.
        """
        return 0

    def is_terminated(self):
        """Whether the episode has terminated due to factors that are
        defined within the Markov Decision Process (eg. task completion/
        failure, etc). This method always returns False unless extended by
        the user.

        Returns
        -------
        bool
            Whether the simulation is terminated.
        """
        return False

    def is_truncated(self):
        """Whether the episode has terminated due to factors beyond the
            Markov Decision Process (eg. time limit, etc). This method
            always returns False unless extended by the user.

        Returns
        -------
        bool
            Whether the simulation is truncated.
        """
        return False

    def get_info(self):
        """Any additional information that is not part of the observation.
        This method always returns an empty dictionary unless extended by
        the user.

        Returns
        -------
        Dict[str, Any]
            The dictionary containing additional information.
        """
        return {}

    def save_video(self, path: Path, stabilization_time=0.02):
        """Save rendered video since the beginning or the last ``reset()``,
        whichever is the latest. Only useful if ``render_mode`` is 'saved'.

        Parameters
        ----------
        path : Path
            Path to which the video should be saved.
        stabilization_time : float, optional
            Time (in seconds) to wait before starting to render the video.
            This might be wanted because it takes a few frames for the
            position controller to move the joints to the specified angles
            from the default, all-stretched position. By default 0.02s
        """
        if self.render_mode != "saved":
            logging.warning(
                'Render mode is not "saved"; no video will be '
                "saved despite `save_video()` call."
            )

        num_stab_frames = int(np.ceil(stabilization_time / self._eff_render_interval))

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving video to {path}")
        with imageio.get_writer(path, fps=self.sim_params.render_fps) as writer:
            for frame in self._frames[num_stab_frames:]:
                writer.append_data(frame)

    def get_last_frame(self):
        """Get the last rendered frame. Only useful if ``render_mode`` is
        'saved'.
        Returns
        -------
        np.ndarray
            The last rendered frame.
        """

        return self._frames[-1]

    def get_COM(self):
        """Get the center of mass of the fly.
        (subtree com weighted by mass) STILL NEEDS TO BE TESTED MORE THOROUGHLY
        Returns
        -------
        np.ndarray
            The center of mass of the fly.
        """
        return np.average(
            self.physics.data.subtree_com, axis=0, weights=self.physics.data.crb[:, 0]
        )

    def get_energy(self):
        """Get the energy of the system (kinetic, potential). Need to activate
        the energy flag in the mujoco xml file.
        Returns
        -------
        np.ndarray
            The energy of the system (kinetic, potential).
        """
        if not self.model.option.flag.energy == "enable":
            raise ValueError("Energy flag not activated in the mujoco xml file. ")
        return self.data.energy

    def close(self):
        """Close the environment, save data, and release any resources."""
        if self.render_mode == "saved" and self.output_dir is not None:
            self.save_video(self.output_dir / "video.mp4")
