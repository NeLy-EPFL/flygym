import numpy as np
import yaml
import imageio
import copy
import logging
from typing import List, Tuple, Dict, Any, Optional, SupportsFloat, Union
from pathlib import Path
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

from flygym.arena import BaseArena
from flygym.arena.mujoco_arena import FlatTerrain
from flygym.state import BaseState, stretched_pose
from flygym.util.data import mujoco_groundwalking_model_path
from flygym.util.config import all_leg_dofs, all_tarsi_links, get_collision_geoms

try:
    import cv2
except ImportError:
    print("OpenCV not installed. Rendering of contact forces is disabled.")


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
    """

    def __init__(
        self,
        timestep: float = 0.0001,
        joint_stiffness: float = 0.05,
        joint_damping: float = 0.06,
        actuator_kp: float = 18.0,
        tarsus_stiffness: float = 2.2,
        tarsus_damping: float = 0.126,
        friction: float = (1.0, 0.005, 0.0001),
        gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81e3),
        render_mode: str = "saved",
        render_window_size: Tuple[int, int] = (640, 480),
        render_playspeed: float = 1.0,
        render_fps: int = 60,
        render_camera: str = "Animat/camera_left_top",
    ) -> None:
        self.timestep = timestep
        self.joint_stiffness = joint_stiffness
        self.joint_damping = joint_damping
        self.actuator_kp = actuator_kp
        self.tarsus_stiffness = tarsus_stiffness
        self.tarsus_damping = tarsus_damping
        self.friction = friction
        self.gravity = gravity
        self.render_mode = render_mode
        self.render_window_size = render_window_size
        self.render_playspeed = render_playspeed
        self.render_fps = render_fps
        self.render_camera = render_camera

    def __str__(self) -> str:
        attributes = vars(self)
        attributes_str = [f"{key}: {value}" for key, value in attributes.items()]
        return "MuJoCo Parameters:\n  " + "\n  ".join(attributes_str)

    def __repr__(self) -> str:
        return str(self)


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
    end_effector_names : List[str]
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
    """

    def __init__(
        self,
        sim_params: MuJoCoParameters = None,
        actuated_joints: List = all_leg_dofs,
        contact_sensor_placements: List = all_tarsi_links,
        output_dir: Optional[Path] = None,
        arena: BaseArena = None,
        spawn_pos: Tuple[float, float, float] = (0.0, 0.0, 0.5),
        spawn_orient: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.1),
        control: str = "position",
        init_pose: BaseState = stretched_pose,
        floor_collisions: Union[str, List[str]] = "legs",
        self_collisions: Union[str, List[str]] = "legs",
        draw_contacts: bool = False,
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
        spawn_pos : Tuple[froot_elementloat, float, float], optional
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
        draw_contacts : bool, optional
            Whether to draw contacts in the simulation. By default False.
        """
        from time import time

        st = time()
        if sim_params is None:
            sim_params = MuJoCoParameters()
        if arena is None:
            arena = FlatTerrain()
        self.sim_params = sim_params
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
        self.end_effector_names = [
            f"{side}{pos}Tarsus5" for side in "LR" for pos in "FMH"
        ]

        # Parse collisions specs
        if isinstance(floor_collisions, str):
            self.floor_collisions = get_collision_geoms(floor_collisions)
        else:
            self.floor_collisions = floor_collisions
        if isinstance(self_collisions, str):
            self.self_collisions = get_collision_geoms(self_collisions)
        else:
            self.self_collisions = self_collisions

        # Define action and observation spaces
        num_dofs = len(actuated_joints)
        action_bound = np.pi if self.control == "position" else np.inf
        num_contacts = len(self.contact_sensor_placements)
        self.action_space, self.observation_space = self._define_spaces(
            num_dofs, action_bound, num_contacts
        )

        # Load NMF model
        self.model = mjcf.from_path(mujoco_groundwalking_model_path)

        # Define list of actuated joints
        self.actuators = [
            self.model.find("actuator", f"actuator_{control}_{joint}")
            for joint in actuated_joints
        ]
        for actuator in self.actuators:
            actuator.kp = self.sim_params.actuator_kp

        # Add arena and put fly in it
        arena.spawn_entity(self.model, self.spawn_pos, self.spawn_orient)
        self.arena_root = arena.root_element
        self.arena_root.option.timestep = self.timestep

        # Add collision/contacts
        floor_collision_geoms = self._parse_collision_specs(floor_collisions)
        self.floor_contacts, self.floor_contact_names = self._define_floor_contacts(
            floor_collision_geoms
        )
        self_collision_geoms = self._parse_collision_specs(self_collisions)
        self.self_contacts, self.self_contact_names = self._define_self_contacts(
            self_collision_geoms
        )
        self.contact_sensor_placements = [
            f"Animat/{body}" for body in self.contact_sensor_placements
        ]

        # Add sensors
        self.joint_sensors = self._add_joint_sensors()
        self.body_sensors = self._add_body_sensors()
        self.end_effector_sensors = self._add_end_effector_sensors()
        self.antennae_sensors = self._add_antennae_sensors()

        # Set up physics and apply ad hoc changes to gravity, stiffness, and friction
        self.physics = mjcf.Physics.from_mjcf_model(self.arena_root)
        for geom in [geom.name for geom in self.arena_root.find_all("geom")]:
            if "collision" in geom:
                self.physics.model.geom(
                    f"Animat/{geom}"
                ).friction = self.sim_params.friction
        for joint in self.actuated_joints:
            if joint is not None:
                self.physics.model.joint(
                    f"Animat/{joint}"
                ).stiffness = self.sim_params.joint_stiffness
                self.physics.model.joint(
                    f"Animat/{joint}"
                ).damping = self.sim_params.joint_damping

        self.physics.model.opt.gravity = self.sim_params.gravity

        # Make tarsi compliant and apply initial pose. MUST BE IN THIS ORDER!
        self._set_compliant_tarsus()
        self._set_init_pose(self.init_pose)

        # Set up a few things for rendering
        self.curr_time = 0
        self._last_render_time = -np.inf
        if sim_params.render_mode != "headless":
            self._eff_render_interval = (
                sim_params.render_playspeed / self.sim_params.render_fps
            )
        self._frames = []

        self.draw_contacts = draw_contacts
        if self.draw_contacts:
            try:
                import cv2
            except ImportError:
                raise ImportError(
                    "Cannot draw contacts without OpenCV. Please switch to " "dev mode."
                )
            self.contact_bodies_ids = set(
                [self.physics.model.body(b) for b in self.contact_sensor_placements]
            )
            self._last_contact_force = []
            self._last_contact_pos = []
            width, height = self.sim_params.render_window_size
            self.contact_camera = dm_control.mujoco.Camera(
                self.physics,
                camera_id=self.sim_params.render_camera,
                width=width,
                height=height,
            )

    def _parse_collision_specs(self, collision_spec: Union[str, List[str]]):
        if collision_spec == "all":
            return [
                geom.name
                for geom in self.model.find_all("geom")
                if "collision" in geom.name
            ]
        elif isinstance(collision_spec, str):
            return get_collision_geoms(collision_spec)
        elif isinstance(collision_spec, list):
            return collision_spec
        else:
            raise ValueError(f"Unrecognized collision spec {collision_spec}")

    def _define_spaces(self, num_dofs, action_bound, num_contacts):
        action_space = {
            "joints": spaces.Box(
                low=-action_bound, high=action_bound, shape=(num_dofs,)
            )
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
                shape=(6, num_contacts),
            ),
            # x, y, z positions of the end effectors (tarsus-5 segments)
            "end_effectors": spaces.Box(low=-np.inf, high=np.inf, shape=(3 * 6,)),
        }
        return action_space, observation_space

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
                            solref="-1000000 -10000",
                            margin=0.0,
                        )
                        self_contact_pairs.append(contact_pair)
                        self_contact_pairs_names.append(f"{geom1}_{geom2}")
        return self_contact_pairs, self_contact_pairs_names

    def _define_floor_contacts(self, floor_collisions_geoms):
        floor_contact_pairs = []
        floor_contact_pairs_names = []
        ground_id = 0

        for geom in self.arena_root.find_all("geom"):
            is_ground = geom.name is None or not (
                "visual" in geom.name or "collision" in geom.name
            )
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
                        solref="-1000000 -10000",
                        margin=0.0,
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
        for name in self.end_effector_names:
            sensor = self.model.sensor.add(
                "framepos",
                name=f"{name}_pos",
                objtype="body",
                objname=name,
            )
            end_effector_sensors.append(sensor)
        return end_effector_sensors

    def _add_antennae_sensors(self):
        antennae_sensors = []
        for name in ["LFuniculus", "RFuniculus"]:
            sensor = self.model.sensor.add(
                "framepos",
                name=f"{name}_pos",
                objtype="body",
                objname=name,
            )
            antennae_sensors.append(sensor)
        return antennae_sensors

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
                    joint = f"joint_{side}{pos}Tarsus{tarsus_link}"
                    self.physics.model.joint(f"Animat/{joint}").stiffness = stiffness
                    self.physics.model.joint(f"Animat/{joint}").damping = damping

        self.physics.reset()

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
        self.curr_time = 0
        self._set_init_pose(self.init_pose)
        self._frames = []
        self._last_render_time = -np.inf
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
        self.physics.bind(self.actuators).ctrl = action["joints"]
        self.physics.step()
        self.curr_time += self.timestep
        observation = self.get_observation()
        reward = self.get_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        """Call the ``render`` method to update the renderer. It should be
        called every iteration; the method will decide by itself whether
        action is required."""
        if self.render_mode == "headless":
            return
        if self.curr_time < self._last_render_time + self._eff_render_interval:
            return
        if self.render_mode == "saved":
            width, height = self.sim_params.render_window_size
            camera = self.sim_params.render_camera
            img = self.physics.render(width=width, height=height, camera_id=camera)
            if self.draw_contacts:
                img = self._draw_contacts(img)
            self._frames.append(img.copy())
            self._last_render_time = self.curr_time

        else:
            raise NotImplementedError

    def _draw_contacts(self, img: np.ndarray) -> np.ndarray:
        """Draw contacts as arrow wich length is proportional to the force
        magnitude. The arrow is drown at the center of the body. It uses the
        camera matrix to transfer from the global space to the pixels space."""

        contact_indexes = np.nonzero(np.sum(self._last_contact_force, axis=1))[0]
        n_contacts = len(contact_indexes)
        force_arrow_points = np.tile(
            self._last_contact_pos[contact_indexes, :], (2, 1)
        ).squeeze()
        force_arrow_points[n_contacts:] += self._last_contact_force[contact_indexes, :]

        camera_matrix = self.contact_camera.matrix

        # code sample from dm_control demo notebook
        xyz_global = force_arrow_points.T

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
            img = cv2.arrowedLine(
                img,
                [x[i], y[i]],
                [x[i + n_contacts], y[i + n_contacts]],
                color=(255, 0, 0),
                thickness=2,
            )

        return img

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

        # fly position and orientation
        cart_pos = self.physics.bind(self.body_sensors[0]).sensordata
        cart_vel = self.physics.bind(self.body_sensors[1]).sensordata
        quat = self.physics.bind(self.body_sensors[2]).sensordata
        # ang_pos = transformations.quat_to_euler(quat)
        ang_pos = R.from_quat(quat).as_euler("xyz")  # explicitly use intrinsic
        ang_pos[0] *= -1  # flip roll??
        ang_vel = self.physics.bind(self.body_sensors[3]).sensordata
        fly_pos = np.array([cart_pos, cart_vel, ang_pos, ang_vel])

        # tarsi contact forces
        """contact_forces = np.zeros((3, len(self.contact_sensor_placements)))
        if self.physics.named.data.ncon > 0:
            tracked_geom1 = self.contact_bodies_ids.intersection(self.physics.named.data.contact.geom1)
            tracked_geom2 = self.contact_bodies_ids.intersection(self.physics.named.data.contact.geom2)
            for i in """

        contact_forces = self.physics.named.data.cfrc_ext[
            self.contact_sensor_placements
        ].copy()

        if self.draw_contacts:
            self._last_contact_force = contact_forces[:, 1:][:, ::2]
            self._last_contact_pos = self.physics.named.data.xpos[
                self.contact_sensor_placements
            ].copy()

        # end effector position
        ee_pos = self.physics.bind(self.end_effector_sensors).sensordata

        # olfaction
        antennae_pos = self.physics.bind(self.antennae_sensors).sensordata.reshape(2, 3)
        odor_intensity = self.arena.get_olfaction(antennae_pos)

        return {
            "joints": joint_obs,
            "fly": fly_pos,
            "contact_forces": contact_forces,
            "end_effectors": ee_pos,
            "odor_intensity": odor_intensity,
        }

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

    def save_video(self, path: Path):
        """Save rendered video since the beginning or the last ``reset()``,
        whichever is the latest. Only useful if ``render_mode`` is 'saved'.

        Parameters
        ----------
        path : Path
            Path to which the video should be saved.
        """
        if self.render_mode != "saved":
            logging.warning(
                'Render mode is not "saved"; no video will be '
                "saved despite `save_video()` call."
            )

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving video to {path}")
        with imageio.get_writer(path, fps=self.sim_params.render_fps) as writer:
            for frame in self._frames:
                writer.append_data(frame)

    def close(self):
        """Close the environment, save data, and release any resources."""
        if self.render_mode == "saved" and self.output_dir is not None:
            self.save_video(self.output_dir / "video.mp4")
