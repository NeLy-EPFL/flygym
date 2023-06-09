import numpy as np
import yaml
import imageio
import copy
import logging
from typing import List, Tuple, Dict, Any, Optional, SupportsFloat
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
from flygym.arena.mujoco_terrain import FlatTerrain
from flygym.state import BaseState, stretched_pose
from flygym.util.data import mujoco_groundwalking_model_path
from flygym.util.config import (
    all_leg_dofs,
    all_tarsi_collisions_geoms,
    all_legs_collisions_geoms,
    all_legs_collisions_geoms_no_coxa,
    all_tarsi_links,
)


_collision_lookup = {
    "all": "all",
    "legs": all_legs_collisions_geoms,
    "legs-no-coxa": all_legs_collisions_geoms_no_coxa,
    "tarsi": all_tarsi_collisions_geoms,
    "none": [],
}


class MuJoCoParameters:
    def __init__(
        self,
        timestep: float = 0.0001,
        joint_stiffness: float = 2500,
        friction: float = (1.0, 0.005, 0.0001),
        gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81e5),
        render_mode: str = "saved",
        render_window_size: Tuple[int, int] = (640, 480),
        render_playspeed: float = 1.0,
        render_fps: int = 60,
        render_camera: str = "Animat/camera_left_top",
    ) -> None:
        """Parameters of the MuJoCo simulation.

        Parameters
        ----------
        timestep : float
            Simulation timestep in seconds.
        joint_stiffness : float, optional
            Stiffness of actuated joints, by default 2500
        friction : float, optional
            , by default (1., 0.005., 0.0001)
        gravity : Tuple[float, float, float], optional
            _description_, by default (0., 0., -9.81e5)
        render_mode : str, optional
            _description_, by default "saved"
        render_window_size : Tuple[int, int], optional
            _description_, by default (640, 480)
        render_playspeed : SupportsFloat, optional
            _description_, by default 1.0
        render_fps : int, optional
            _description_, by default 60
        render_camera : str, optional
            _description_, by default "Animat/camera_left_top"
        """
        self.timestep = timestep
        self.joint_stiffness = joint_stiffness
        self.friction = friction
        self.gravity = gravity
        self.render_mode = render_mode
        self.render_window_size = render_window_size
        self.render_playspeed = render_playspeed
        self.render_fps = render_fps
        self.render_camera = render_camera


class NeuroMechFlyMuJoCo(gym.Env):
    """A NeuroMechFly environment using MuJoCo as the physics engine.

    Attributes
    ----------
    actuated_joints : List[str]
        List of actuated joints.
    timestep : float
        Simulation timestep in seconds.
    output_dir : Path
        Directory to save simulation data.
    control : str
        The joint controller type. Can be 'position', 'velocity', or
        'torque'.
    init_pose : str
        Which initial pose to start the simulation from. Currently only
        'default' is implemented.
    action_space : Dict[str, gym.spaces.Box]
        Definition of the simulation's action space as a Gym
        environment.
    observation_space : Dict[str, gym.spaces.Box]
        Definition of the simulation's observation space as a Gym
        environment.
    model : dm_control.mjcf.RootElement
        The MuJoCo model.
    physics : dm_control.mujoco.Physics
        The MuJoCo physics simulation.
    actuators : Dict[str, dm_control.mjcf.Element]
        The MuJoCo actuators.
    joint_sensors : Dict[str, dm_control.mjcf.Element]
        The MuJoCo sensors on joint positions, velocities, and forces.
    body_sensors : Dict[str, dm_control.mjcf.Element]
        The MuJoCo sensors on thorax position and orientation.
    curr_time : float
        The (simulated) time elapsed since the last reset (in seconds).
    self_contact_pairs: List[dm_control.mjcf.Element]
        The MuJoCo geom pairs that can be in contact with each other
    self_contact_pair_names: List[str]
        The names of the MuJoCo geom pairs that can be in contact with
        each other
    floor_contact_pairs: List[dm_control.mjcf.Element]
        The MuJoCo geom pairs that can be in contact with the floor
    floor_contact_pair_names: List[str]
        The names of the MuJoCo geom pairs that can be in contact with
        the floor
    touch_sensors: List[dm_control.mjcf.Element]
        The MuJoCo touch sesnor used to conpute contact forces
    end_effector_sensors: List[dm_control.mjcf.Element]
        The set of position sensors on the end effectors

    """

    def __init__(
        self,
        sim_params: MuJoCoParameters = None,
        actuated_joints: List = all_leg_dofs,
        contact_sensor_placements: List = all_tarsi_links,
        output_dir: Optional[Path] = None,
        arena: BaseArena = None,
        spawn_pos: Tuple[float, float, float] = (0.0, 0.0, 1500.0),
        spawn_orient: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.1),
        control: str = "position",
        init_pose: BaseState = stretched_pose,
        floor_collisions_geoms: str = "legs",
        self_collisions_geoms: str = "legs",
    ) -> None:
        """Initialize a MuJoCo-based NeuroMechFly environment.

        Parameters
        ----------
        actuated_joints : List, optional
            List of actuated joint DoFs, by default all leg DoFs
        contact_sensor_placements : List, optional
            List of geometries on each leg where a contact sensor should
            be placed. By default all tarsi.
            Simulation timestep in seconds, by default 0.0001
        output_dir : Path, optional
            Directory to save simulation data (by default just the video,
            but you can extend this class to save additional data).
            If ``None``, no data will be saved. By default None
        arena : BaseWorld, optional
            XXX
        spawn_pos : Tuple[froot_elementloat, float, float], optional
            XXX, by default (0., 0., 300.)
        spawn_orient : Tuple[float, float, float, float], optional
            XXX, by default (0., 1., 0., 0.1)
        control : str, optional
            The joint controller type. Can be 'position', 'velocity', or
            'torque'., by default 'position'
        init_pose : BaseState, optional
            Which initial pose to start the simulation from. By default
            "stretched" kinematic pose with all legs fully stretched.
        floor_collisions_geoms :str
            Which set of collisions should collide with the floor. Can be
            'all', 'legs', or 'tarsi'.
        self_collisions_geoms : str
            Which set of collisions should collide with each other. Can be
            'all', 'legs', 'legs-no-coxa', 'tarsi', or 'none'.
        """
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

        # Define action and observation spaces
        num_dofs = len(actuated_joints)
        bound = np.pi if control == "position" else np.inf
        self.action_space = {
            "joints": spaces.Box(low=-bound, high=bound, shape=(num_dofs,))
        }
        self.observation_space = {
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
                shape=(
                    6,
                    len(contact_sensor_placements),
                ),
            ),
            # x, y, z positions of the end effectors (tarsus-5 segments)
            "end_effectors": spaces.Box(low=-np.inf, high=np.inf, shape=(3 * 6,)),
        }

        # Load NMF model
        self.model = mjcf.from_path(mujoco_groundwalking_model_path)

        # Fix unactuated joints and define list of actuated joints
        # for joint in model.find_all('joint'):
        #     if joint.name not in actuated_joints:
        #         joint.type = 'fixed'
        self.actuators = [
            self.model.find("actuator", f"actuator_{control}_{joint}")
            for joint in actuated_joints
        ]

        # Add sensors
        self.joint_sensors = []
        for joint in actuated_joints:
            self.joint_sensors.extend(
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

        self.body_sensors = [
            self.model.sensor.add(
                "framepos", name="thorax_pos", objtype="body", objname="Thorax"
            ),
            self.model.sensor.add(
                "framelinvel", name="thorax_linvel", objtype="body", objname="Thorax"
            ),
            self.model.sensor.add(
                "framequat", name="thorax_quat", objtype="body", objname="Thorax"
            ),
            self.model.sensor.add(
                "frameangvel", name="thorax_angvel", objtype="body", objname="Thorax"
            ),
        ]

        self_collisions_geoms = _collision_lookup[self_collisions_geoms]
        if self_collisions_geoms == "all":
            self_collisions_geoms = []
            for geom in self.model.find_all("geom"):
                if "collision" in geom.name:
                    self_collisions_geoms.append(geom.name)

        self.self_contact_pairs = []
        self.self_contact_pairs_names = []

        for geom1 in self_collisions_geoms:
            for geom2 in self_collisions_geoms:
                is_duplicate = f"{geom1}_{geom2}" in self.self_contact_pairs_names
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
                        self.self_contact_pairs.append(contact_pair)
                        self.self_contact_pairs_names.append(f"{geom1}_{geom2}")

        self.end_effector_sensors = []
        self.end_effector_names = []
        for body in self.model.find_all("body"):
            if "Tarsus5" in body.name:
                self.end_effector_names.append(body.name)
                end_effector_sensor = self.model.sensor.add(
                    "framepos",
                    name=f"{body.name}_pos",
                    objtype="body",
                    objname=body.name,
                )
                self.end_effector_sensors.append(end_effector_sensor)

        ## Add sites and touch sensors
        self.touch_sensors = []
        for side in "LR":
            for pos in "FMH":
                for tracked_geom in contact_sensor_placements:
                    geom = self.model.find(
                        "geom", f"{side}{pos}{tracked_geom}_collision"
                    )
                    body = geom.parent
                    site = body.add(
                        "site",
                        name=f"site_{geom.name}",
                        size=np.ones(3) * 1000,
                        pos=geom.pos,
                        quat=geom.quat,
                        type="sphere",
                        group=3,
                    )
                    touch_sensor = self.model.sensor.add(
                        "touch", name=f"touch_{geom.name}", site=site.name
                    )
                    self.touch_sensors.append(touch_sensor)

        # Add arena and put fly in it
        arena.spawn_entity(self.model, self.spawn_pos, self.spawn_orient)
        root_element = arena.root_element

        # Add collision between the ground and the fly
        floor_collisions_geoms = _collision_lookup[floor_collisions_geoms]

        self.floor_contact_pairs = []
        self.floor_contact_pairs_names = []
        ground_id = 0

        if floor_collisions_geoms == "all":
            floor_collisions_geoms = []
            for geom in self.model.find_all("geom"):
                if "collision" in geom.name:
                    floor_collisions_geoms.append(geom.name)

        for geom in root_element.find_all("geom"):
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
                    floor_contact_pair = root_element.contact.add(
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
                    self.floor_contact_pairs.append(floor_contact_pair)
                    self.floor_contact_pairs_names.append(
                        f"{geom.name}_{animat_geom_name}"
                    )

        root_element.option.timestep = self.timestep
        self.physics = mjcf.Physics.from_mjcf_model(root_element)
        self.curr_time = 0
        self._last_render_time = -np.inf
        if sim_params.render_mode != "headless":
            self._eff_render_interval = (
                sim_params.render_playspeed / self.sim_params.render_fps
            )
        self._frames = []

        # Ad hoc changes to gravity, stiffness, and friction
        for geom in [geom.name for geom in root_element.find_all("geom")]:
            if "collision" in geom:
                self.physics.model.geom(
                    f"Animat/{geom}"
                ).friction = self.sim_params.friction

        for joint in self.actuated_joints:
            if joint is not None:
                self.physics.model.joint(
                    f"Animat/{joint}"
                ).stiffness = self.sim_params.joint_stiffness

        self.physics.model.opt.gravity = self.sim_params.gravity

        # set complaint tarsus
        all_joints = [joint.name for joint in root_element.find_all("joint")]
        self._set_compliant_Tarsus(all_joints, stiff=3.5e5, damping=100)
        # set init pose
        self._set_init_pose(self.init_pose)

    def _set_init_pose(self, init_pose: Dict[str, float]):
        with self.physics.reset_context():
            for i in range(len(self.actuated_joints)):
                curr_joint = self.actuators[i].joint.name
                if (curr_joint in self.actuated_joints) and (curr_joint in init_pose):
                    animat_name = f"Animat/{curr_joint}"
                    self.physics.named.data.qpos[animat_name] = init_pose[curr_joint]

    def _set_compliant_Tarsus(
        self, all_joints: List, stiff: float = 0.0, damping: float = 100
    ):
        """Set the Tarsus2/3/4/5 to be compliant by setting the
        stifness and damping to a low value"""
        for joint in all_joints:
            if joint is None:
                continue
            if ("Tarsus" in joint) and (not "Tarsus1" in joint):
                self.physics.model.joint(f"Animat/{joint}").stiffness = stiff
                # self.physics.model.joint(f'Animat/{joint}').springref = 0.0
                self.physics.model.joint(f"Animat/{joint}").damping = damping

        self.physics.reset()

    def set_output_dir(self, output_dir: str):
        """Set the output directory for the environment.
        Allows the user to change the output directory after the
        environment has been initialized. If the directory does not
        exist, it will be created.
        Parameters
        ----------
        output_dir : str
            The output directory.
        """
        output_dir = Path.cwd() / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    def reset(self) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset the Gym environment.

        Returns
        -------
        ObsType
            The observation as defined by the environment.
        Dict[str, Any]
            Any additional information that is not part of the
            observation. This is an empty dictionary by default but
            the user can override this method to return additional
            information.
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
            Action dictionary as defined by the environment's
            action space.

        Returns
        -------
        ObsType
            The observation as defined by the environment.
        SupportsFloat
            The reward as defined by the environment.
        bool
            Whether the episode has terminated due to factors that
            are defined within the Markov Decision Process (eg. task
            completion/failure, etc).
        bool
            Whether the episode has terminated due to factors beyond
            the Markov Decision Process (eg. time limit, etc).
        Dict[str, Any]
            Any additional information that is not part of the
            observation. This is an empty dictionary by default but
            the user can override this method to return additional
            information.
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
        """Call the ``render`` method to update the renderer. It should
        be called every iteration; the method will decide by itself
        whether action is required."""
        if self.render_mode == "headless":
            return
        if self.curr_time < self._last_render_time + self._eff_render_interval:
            return
        if self.render_mode == "saved":
            width, height = self.sim_params.render_window_size
            camera = self.sim_params.render_camera
            img = self.physics.render(width=width, height=height, camera_id=camera)
            self._frames.append(img.copy())
            self._last_render_time = self.curr_time
        else:
            raise NotImplementedError

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
        touch_sensordata = self.physics.bind(self.touch_sensors).sensordata
        contact_forces = touch_sensordata.copy().reshape(
            (6, len(self.contact_sensor_placements))
        )

        # end effector position
        ee_pos = self.physics.bind(self.end_effector_sensors).sensordata

        return {
            "joints": joint_obs,
            "fly": fly_pos,
            "contact_forces": contact_forces,
            "end_effectors": ee_pos,
        }

    def get_reward(self):
        return 0

    def is_terminated(self):
        return False

    def is_truncated(self):
        return False

    def get_info(self):
        return {}

    def save_video(self, path: Path):
        """Save rendered video since the beginning or the last
        ``reset()``, whichever is the latest.
        Only useful if ``render_mode`` is 'saved'.

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
