import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
from dm_control import mjcf
from dm_control.utils import transformations
from gymnasium import spaces
from gymnasium.core import ObsType
from scipy.spatial.transform import Rotation as R

import flygym.preprogrammed as preprogrammed
import flygym.state as state
import flygym.util as util
import flygym.vision as vision
from flygym.arena import BaseArena
from flygym.util import get_data_path

if TYPE_CHECKING:
    from flygym.simulation import Simulation


class Fly:
    """A NeuroMechFly environment using MuJoCo as the physics engine.

    Attributes
    ----------
    name : str
        The name of the fly model.
    actuated_joints : list[str]
        List of names of actuated joints.
    contact_sensor_placements : list[str]
        List of body segments where contact sensors are placed.
    spawn_pos : tuple[float, float, float]
        The (x, y, z) position in the arena defining where the fly will be
        spawn, in mm.
    spawn_orientation : tuple[float, float, float, float]
        The spawn orientation of the fly in the Euler angle format: (x, y,
        z), where x, y, z define the rotation around x, y and z in radian.
    control : str
        The joint controller type. Can be "position", "velocity", or
        "torque".
    init_pose : flygym.state.BaseState
        Which initial pose to start the simulation from.
    floor_collisions :str
        Which set of collisions should collide with the floor. Can be
        "all", "legs", "tarsi" or a list of body names.
    self_collisions : str
        Which set of collisions should collide with each other. Can be
        "all", "legs", "legs-no-coxa", "tarsi", "none", or a list of
        body names.
    detect_flip : bool
        If True, the simulation will indicate whether the fly has flipped
        in the ``info`` returned by ``.step(...)``. Flip detection is
        achieved by checking whether the leg tips are free of any contact
        for a duration defined in the configuration file. Flip detection is
        disabled for a period of time at the beginning of the simulation as
        defined in the configuration file. This avoids spurious detection
        when the fly is not standing reliably on the ground yet.
    joint_stiffness : float
        Stiffness of actuated joints.
        joint_stiffness : float
        Stiffness of actuated joints.
    joint_damping : float
        Damping coefficient of actuated joints.
    non_actuated_joint_stiffness : float
        Stiffness of non-actuated joints.
    non_actuated_joint_damping : float
        Damping coefficient of non-actuated joints.
    neck_stiffness : Union[float, None]
        Stiffness of the neck joints (``joint_Head``, ``joint_Head_roll``,
        and ``joint_Head_yaw``), by default 10.0. The head joints have
        their stiffness set separately, typically to a higher value than
        the other non-actuated joints, to ensure that the visual input is
        not perturbed by unintended passive head movements. If set, this
        value overrides ``non_actuated_joint_stiffness``.
    control: str
        The joint controller type. Can be "position", "velocity", or
        "motor".
    tarsus_stiffness : float
        Stiffness of the passive, compliant tarsus joints.
    tarsus_damping : float
        Damping coefficient of the passive, compliant tarsus joints.
    friction : float
        Sliding, torsional, and rolling friction coefficients.
    contact_solref: tuple[float, float]
        MuJoCo contact reference parameters (see `MuJoCo documentation
        <https://mujoco.readthedocs.io/en/stable/modeling.html#impedance>`_
        for details). Under the default configuration, contacts are very
        stiff. This is to avoid penetration of the leg tips into the ground
        when leg adhesion is enabled. The user might want to decrease the
        stiffness if the stability becomes an issue.
    contact_solimp: tuple[float, float, float, float, float]
        MuJoCo contact reference parameters (see `MuJoCo docs
        <https://mujoco.readthedocs.io/en/stable/modeling.html#reference>`_
        for details). Under the default configuration, contacts are very
        stiff. This is to avoid penetration of the leg tips into the ground
        when leg adhesion is enabled. The user might want to decrease the
        stiffness if the stability becomes an issue.
    enable_olfaction : bool
        Whether to enable olfaction.
    enable_vision : bool
        Whether to enable vision.
    render_raw_vision : bool
        If ``enable_vision`` is True, whether to render the raw vision
        (raw pixel values before binning by ommatidia).
    vision_refresh_rate : int
        The rate at which the vision sensor is updated, in Hz.
    enable_adhesion : bool
        Whether to enable adhesion.
    adhesion_force : float
        The magnitude of the adhesion force.
    draw_adhesion : bool
        Whether to signal that adhesion is on by changing the color of the
        concerned leg.
    draw_sensor_markers : bool
        If True, colored spheres will be added to the model to indicate the
        positions of the cameras (for vision) and odor sensors.
    head_stabilization_model : Callable or str optional
        If callable, it should be a function that, given the observation,
        predicts signals that need to be applied to the neck DoFs to
        stabilizes the head of the fly. If "thorax", the rotation (roll
        and pitch) of the thorax is inverted and applied to the head by
        the neck actuators. If None, no head stabilization is applied.
    retina : flygym.vision.Retina
        The retina simulation object used to render the fly's visual
        inputs.
    arena_root = dm_control.mjcf.RootElement
        The root element of the arena.
    action_space : gymnasium.core.ObsType
        Definition of the fly's action space.
    observation_space : gymnasium.core.ObsType
        Definition of the fly's observation space.
    model : dm_control.mjcf.RootElement
        The MuJoCo model.
    vision_update_mask : np.ndarray
        The refresh frequency of the visual system is often loser than the
        same as the physics simulation time step. This 1D mask, whose
        size is the same as the number of simulation time steps, indicates
        in which time steps the visual inputs have been refreshed. In other
        words, the visual input frames where this mask is False are
        repetitions of the previous updated visual input frames.
    """

    config = util.load_config()
    _last_tarsal_seg_names = tuple(
        f"{side}{pos}Tarsus5" for side in "LR" for pos in "FMH"
    )
    n_legs = 6
    _floor_contacts: dict[str, mjcf.Element]
    _self_contacts: dict[str, mjcf.Element]
    _adhesion_actuator_geom_id: np.ndarray
    _default_fly_name = 0
    _existing_fly_names = set()
    observation_space: spaces.Dict

    def __init__(
        self,
        name: Optional[str] = None,
        actuated_joints: list = preprogrammed.all_leg_dofs,
        contact_sensor_placements: list = preprogrammed.all_tarsi_links,
        xml_variant: Union[str, Path] = "seqik",
        spawn_pos: tuple[float, float, float] = (0.0, 0.0, 0.5),
        spawn_orientation: tuple[float, float, float] = (0.0, 0.0, np.pi / 2),
        control: str = "position",
        init_pose: Union[str, state.KinematicPose] = "stretch",
        floor_collisions: Union[str, list[str]] = "legs",
        self_collisions: Union[str, list[str]] = "legs",
        detect_flip: bool = False,
        joint_stiffness: float = 0.05,
        joint_damping: float = 0.06,
        non_actuated_joint_stiffness: float = 1.0,
        non_actuated_joint_damping: float = 1.0,
        neck_stiffness: Union[float, None] = 10.0,
        actuator_gain: Union[float, list] = 45.0,
        actuator_forcerange: Union[float, tuple[float, float], list] = 65.0,
        tarsus_stiffness: float = 7.5,
        tarsus_damping: float = 1e-2,
        friction: float = (1.0, 0.005, 0.0001),
        contact_solref: tuple[float, float] = (2e-4, 1e3),
        contact_solimp: tuple[float, float, float, float, float] = (
            9.99e-01,
            9.999e-01,
            1.0e-03,
            5.0e-01,
            2.0e00,
        ),
        enable_olfaction: bool = False,
        enable_vision: bool = False,
        render_raw_vision: bool = False,
        vision_refresh_rate: int = 500,
        enable_adhesion: bool = False,
        adhesion_force: float = 40,
        draw_adhesion: bool = False,
        draw_sensor_markers: bool = False,
        neck_kp: Optional[float] = None,
        head_stabilization_model: Optional[Union[Callable, str]] = None,
    ) -> None:
        """Initialize a NeuroMechFly environment.

        Parameters
        ----------
        name : str, optional
            The name of the fly model. Will be automatically generated if
            not provided.
        actuated_joints : list[str], optional
            List of names of actuated joints. By default all active leg
            DoFs.
        contact_sensor_placements : list[str], optional
            List of body segments where contact sensors are placed. By
            default all tarsus segments.
        xml_variant: str or Path, optional
            The variant of the fly model to use. Multiple variants exist
            because when replaying experimentally recorded behavior, the
            ordering of DoF angles in multi-DoF joints depends on how they
            are configured in the upstream inverse kinematics program. Two
            variants are provided: "seqik" (default) and "deepfly3d" (for
            legacy data produced by DeepFly3D, Gunel et al., eLife, 2019).
            The ordering of DoFs can be seen from the XML files under
            ``flygym/data/mjcf/``.
        spawn_pos : tuple[float, float, float], optional
            The (x, y, z) position in the arena defining where the fly
            will be spawn, in mm. By default (0, 0, 0.5).
        spawn_orientation : tuple[float, float, float], optional
            The spawn orientation of the fly in the Euler angle format:
            (x, y, z), where x, y, z define the rotation around x, y and
            z in radian. By default (0.0, 0.0, pi/2), which leads to a
            position facing the positive direction of the x-axis.
        control : str, optional
            The joint controller type. Can be "position", "velocity", or
            "motor", by default "position".
        init_pose : BaseState, optional
            Which initial pose to start the simulation from. By default
            "stretch" kinematic pose with all legs fully stretched.
        floor_collisions :str
            Which set of collisions should collide with the floor. Can be
            "all", "legs", "tarsi" or a list of body names. By default
            "legs".
        self_collisions : str
            Which set of collisions should collide with each other. Can be
            "all", "legs", "legs-no-coxa", "tarsi", "none", or a list of
            body names. By default "legs".
        detect_flip : bool
            If True, the simulation will indicate whether the fly has
            flipped in the ``info`` returned by ``.step(...)``. Flip
            detection is achieved by checking whether the leg tips are free
            of any contact for a duration defined in the configuration
            file. Flip detection is disabled for a period of time at the
            beginning of the simulation as defined in the configuration
            file. This avoids spurious detection when the fly is not
            standing reliably on the ground yet. By default False.
        joint_stiffness : float
            Stiffness of actuated joints, by default 0.05.
            joint_stiffness : float
            Stiffness of actuated joints, by default 0.05.
        joint_damping : float
            Damping coefficient of actuated joints, by default 0.06.
        non_actuated_joint_stiffness : float
            Stiffness of non-actuated joints, by default 1.0. If set to 0,
            the DoF would passively drift over time. Therefore it is set
            explicitly here for better stability.
        non_actuated_joint_damping : float
            Damping coefficient of non-actuated joints, by default 1.0.
            Similar to ``non_actuated_joint_stiffness``, it is set
            explicitly here for better stability.
        neck_stiffness : Union[float, None]
            Stiffness of the neck joints (``joint_Head``,
            ``joint_Head_roll``, and ``joint_Head_yaw``), by default 10.0.
            The head joints have their stiffness set separately, typically
            to a higher value than the other non-actuated joints, to ensure
            that the visual input is not perturbed by unintended passive
            head movements. If set, this value overrides
            ``non_actuated_joint_stiffness``.
        actuator_gain : Union[float, list[float]]
            Gain of the actuator:
            If ``control`` is "position", it is the position gain of the
            actuators.
            If ``control`` is "velocity", it is the velocity gain of the
            actuators.
            If ``control`` is "motor", it is not used
            if the actuator gain is a list, it needs to be of same length as
            the number of actuated joints and will be applied to every joint
        actuator_forcerange : Union[float, tuple[float, float], list]
            The force limit of the actuators. If a single value is
            provided, it will be symmetrically applied to all actuators
            (-a, a). If a tuple is provided, the first value is the lower
            limit and the second value is the upper limit. If a list is
            provided, it should have the same length as the number of
            actuators. By default 65.0.
        tarsus_stiffness : float
            Stiffness of the passive, compliant tarsus joints, by default
            7.5.
        tarsus_damping : float
            Damping coefficient of the passive, compliant tarsus joints, by
            default 1e-2.
        friction : float
            Sliding, torsional, and rolling friction coefficients, by
            default (1, 0.005, 0.0001)
        contact_solref: tuple[float, float]
            MuJoCo contact reference parameters (see `MuJoCo documentation
            <https://mujoco.readthedocs.io/en/stable/modeling.html#impedance>`_
            for details). By default (9.99e-01, 9.999e-01, 1.0e-03,
            5.0e-01, 2.0e+00). Under the default configuration, contacts
            are very stiff. This is to avoid penetration of the leg tips
            into the ground when leg adhesion is enabled. The user might
            want to decrease the stiffness if the stability becomes an
            issue.
        contact_solimp: tuple[float, float, float, float, float]
            MuJoCo contact reference parameters (see `MuJoCo docs
            <https://mujoco.readthedocs.io/en/stable/modeling.html#reference>`_
            for details). By default (9.99e-01, 9.999e-01, 1.0e-03,
            5.0e-01, 2.0e+00). Under the default configuration, contacts
            are very stiff. This is to avoid penetration of the leg tips
            into the ground when leg adhesion is enabled. The user might
            want to decrease the stiffness if the stability becomes an
            issue.
        enable_olfaction : bool
            Whether to enable olfaction, by default False.
        enable_vision : bool
            Whether to enable vision, by default False.
        render_raw_vision : bool
            If ``enable_vision`` is True, whether to render the raw vision
            (raw pixel values before binning by ommatidia), by default
            False.
        vision_refresh_rate : int
            The rate at which the vision sensor is updated, in Hz, by
            default
            500.
        enable_adhesion : bool
            Whether to enable adhesion. By default False.
        adhesion_force : float
            The magnitude of the adhesion force. By default 20.
        draw_adhesion : bool
            Whether to signal that adhesion is on by changing the color of
            the concerned leg. By default False.
        draw_sensor_markers : bool
            If True, colored spheres will be added to the model to indicate
            the positions of the cameras (for vision) and odor sensors. By
            default False.
        neck_kp : float, optional
            Position gain of the neck position actuators. If supplied, this
            will overwrite ``actuator_kp`` for the neck actuators.
            Otherwise, ``actuator_kp`` will be used.
        head_stabilization_model : Callable or str optional
            If callable, it should be a function that, given the observation,
            predicts signals that need to be applied to the neck DoFs to
            stabilizes the head of the fly. If "thorax", the rotation (roll
            and pitch) of the thorax is inverted and applied to the head by
            the neck actuators. If None (default), no head stabilization is
            applied.
        """
        actuated_joints = list(actuated_joints)

        # Check neck actuation if head stabilization is enabled
        if head_stabilization_model is not None:
            if "joint_Head_yaw" in actuated_joints or "joint_Head" in actuated_joints:
                raise ValueError(
                    "The head joints are actuated by a preset algorithm. "
                    "However, the head joints are already included in the "
                    "provided Fly instance. Please remove the head joints from "
                    "the list of actuated joints."
                )
            self._last_observation = None  # tracked only for head stabilization
            self._last_neck_actuation = None  # tracked only for head stabilization

        self.actuated_joints = actuated_joints
        self.contact_sensor_placements = contact_sensor_placements
        self.detect_flip = detect_flip
        self.joint_stiffness = joint_stiffness
        self.joint_damping = joint_damping
        self.non_actuated_joint_stiffness = non_actuated_joint_stiffness
        self.non_actuated_joint_damping = non_actuated_joint_damping
        self.tarsus_stiffness = tarsus_stiffness
        self.tarsus_damping = tarsus_damping
        self.neck_stiffness = neck_stiffness
        self.friction = friction
        self.contact_solref = contact_solref
        self.contact_solimp = contact_solimp
        self.enable_olfaction = enable_olfaction
        self.enable_vision = enable_vision
        self.render_raw_vision = render_raw_vision
        self.vision_refresh_rate = vision_refresh_rate
        self.enable_adhesion = enable_adhesion
        self.adhesion_force = adhesion_force
        self.draw_adhesion = draw_adhesion
        self.draw_sensor_markers = draw_sensor_markers
        self.floor_collisions = floor_collisions
        self.self_collisions = self_collisions
        self.head_stabilization_model = head_stabilization_model

        # Load NMF model
        if isinstance(xml_variant, str):
            xml_variant = (
                get_data_path("flygym", "data")
                / self.config["paths"]["mjcf"][xml_variant]
            )
        self.model = mjcf.from_path(xml_variant)

        if name is None:
            name = f"{self._default_fly_name}"
            self._default_fly_name += 1

            while name in self._existing_fly_names:
                name = f"{self._default_fly_name}"
                self._default_fly_name += 1

        self._existing_fly_names.add(str(name))
        self.model.model = str(name)

        self.spawn_pos = np.array(spawn_pos)
        # convert to mujoco orientation format [0, 0, 0] would orient along the x-axis
        # but the output fly_orientation from framequat would be [0, 0, pi/2] for
        # spawn_orient = [0, 0, 0]
        self.spawn_orientation = spawn_orientation - np.array((0, 0, np.pi / 2))
        self.control = control
        if isinstance(init_pose, str):
            self.init_pose = preprogrammed.get_preprogrammed_pose(init_pose)
        else:
            self.init_pose = init_pose

        # Parse collisions specs
        if isinstance(floor_collisions, str):
            self._floor_collisions = preprogrammed.get_collision_geometries(
                floor_collisions
            )
        else:
            self._floor_collisions = floor_collisions
        if isinstance(self_collisions, str):
            self._self_collisions = preprogrammed.get_collision_geometries(
                self_collisions
            )
        else:
            self._self_collisions = self_collisions

        self._last_adhesion = np.zeros(self.n_legs)
        self._active_adhesion = np.zeros(self.n_legs)

        if self.draw_adhesion and not self.enable_adhesion:
            logging.warning(
                "Overriding `draw_adhesion` to False because adhesion is not enabled."
            )
            self.draw_adhesion = False

        if self.draw_adhesion:
            self._leg_adhesion_drawing_segments = np.array(
                [
                    [f"{self.name}/{tarsus5.replace('5', str(i))}" for i in range(1, 6)]
                    for tarsus5 in self._last_tarsal_seg_names
                ]
            ).astype("U64")
            self._adhesion_rgba = [1.0, 0.0, 0.0, 0.8]
            self._active_adhesion_rgba = [0.0, 0.0, 1.0, 0.8]
            self._base_rgba = [0.5, 0.5, 0.5, 1.0]

        self._set_geom_colors()

        # Add cameras imitating the fly's eyes
        self._curr_visual_input = None
        self._curr_raw_visual_input = None
        self._last_vision_update_time = -np.inf
        self._eff_visual_render_interval = 1 / self.vision_refresh_rate
        self._vision_update_mask: list[bool] = []
        if self.enable_vision:
            self._configure_eyes()
            self.retina = vision.Retina()

        # Define list of actuated joints
        self.actuators = self._add_joint_actuators(actuator_gain, actuator_forcerange)
        if self.head_stabilization_model is not None:
            self.neck_actuators = [
                self.model.actuator.add(
                    self.control,
                    name=f"actuator_position_{joint}",
                    joint=joint,
                    kp=neck_kp,
                    ctrlrange="-1000000 1000000",
                    forcelimited=False,
                )
                for joint in ["joint_Head_yaw", "joint_Head"]
            ]

        self._set_geoms_friction()
        self._set_joints_stiffness_and_damping()
        self._set_compliant_tarsus()

        self.thorax = self.model.find("body", "Thorax")

        # Add self collisions
        self._init_self_contacts()

        # Add sensors
        self._joint_sensors = self._add_joint_sensors()
        self._body_sensors = self._add_body_sensors()
        self._end_effector_sensors = self._add_end_effector_sensors()
        self._antennae_sensors = (
            self._add_odor_sensors() if self.enable_olfaction else None
        )
        self._add_force_sensors()
        self.contact_sensor_placements = [
            f"{self.name}/{body}" for body in self.contact_sensor_placements
        ]
        self.adhesion_actuators = self._add_adhesion_actuators(self.adhesion_force)
        # Those need to be in the same order as the adhesion sensor
        # (due to comparison with the last adhesion_signal)
        adhesion_sensor_indices = []
        for adhesion_actuator in self.adhesion_actuators:
            for index, contact_sensor in enumerate(self.contact_sensor_placements):
                if (
                    f"{contact_sensor}_adhesion"
                    in f"{self.name}/{adhesion_actuator.name}"
                ):
                    adhesion_sensor_indices.append(index)
        self._adhesion_bodies_with_contact_sensors = np.array(adhesion_sensor_indices)

        # flip detection
        self._flip_counter = 0

        # Define action and observation spaces
        action_bound = np.pi if self.control == "position" else np.inf
        self.action_space = self._define_action_space(action_bound)

        # Add metadata as specified by Gym
        self.metadata = {}

        self.last_obs = {
            "contact_forces": [],
            "contact_pos": [],
        }

    @property
    def name(self) -> str:
        return self.model.model

    def post_init(self, sim: "Simulation"):
        """Initialize attributes that depend on the arena or physics of the
        simulation.

        Parameters
        ----------
        sim : Simulation
            Simulation object.
        """
        self._adhesion_actuator_geom_id = np.array(
            [
                sim.physics.model.geom(f"{self.name}/{actuator.body}").id
                for actuator in self.adhesion_actuators
            ]
        )

        self.observation_space = self._define_observation_space(sim.arena)

    def _configure_eyes(self):
        for name in ["LEye_cam", "REye_cam"]:
            sensor_config = self.config["vision"]["sensor_positions"][name]
            parent_body = self.model.find("body", sensor_config["parent"])
            sensor_body = parent_body.add(
                "body", name=f"{name}_body", pos=sensor_config["rel_pos"]
            )

            sensor_body.add(
                "camera",
                name=name,
                dclass="nmf",
                mode="fixed",
                euler=sensor_config["orientation"],
                fovy=self.config["vision"]["fovy_per_eye"],
            )
            if self.draw_sensor_markers:
                sensor_body.add(
                    "geom",
                    name=f"{name}_marker",
                    type="sphere",
                    size=[0.06],
                    rgba=sensor_config["marker_rgba"],
                )

        # Make list of geometries that are hidden during visual input rendering
        self._geoms_to_hide = self.config["vision"]["hidden_segments"]

    def _parse_collision_specs(self, collision_spec: Union[str, list[str]]):
        if collision_spec == "all":
            return [geom.name for geom in self.model.find_all("geom")]
        elif isinstance(collision_spec, str):
            return preprogrammed.get_collision_geometries(collision_spec)
        elif isinstance(collision_spec, list):
            return collision_spec
        else:
            raise TypeError(
                "Collision specs must be a string ('legs', 'legs-no-coxa', 'tarsi', "
                "'none'), or a list of body segment names."
            )

    def _set_geom_colors(self):
        for type_, specs in self.config["appearance"].items():
            # Define texture and material
            if specs["texture"] is not None:
                self.model.asset.add(
                    "texture",
                    name=f"{type_}_texture",
                    builtin=specs["texture"]["builtin"],
                    mark="random",
                    width=specs["texture"]["size"],
                    height=specs["texture"]["size"],
                    random=specs["texture"]["random"],
                    rgb1=specs["texture"]["rgb1"],
                    rgb2=specs["texture"]["rgb2"],
                    markrgb=specs["texture"]["markrgb"],
                )
            self.model.asset.add(
                "material",
                name=f"{type_}_material",
                texture=f"{type_}_texture" if specs["texture"] is not None else None,
                rgba=specs["material"]["rgba"],
                specular=0.0,
                shininess=0.0,
                reflectance=0.0,
                texuniform=True,
            )
            # Apply to geoms
            for segment in specs["apply_to"]:
                geom = self.model.find("geom", segment)
                if geom is None:
                    geom = self.model.find("geom", f"{segment}")
                geom.material = f"{type_}_material"

    def _define_action_space(self, action_bound):
        _action_space = {
            "joints": spaces.Box(
                low=-action_bound, high=action_bound, shape=(len(self.actuated_joints),)
            )
        }
        if self.enable_adhesion:
            # continuous actuation between 0 and 1
            _action_space["adhesion"] = spaces.Box(
                low=0, high=1, shape=(len(self.adhesion_actuators),)
            )
        return spaces.Dict(_action_space)

    def _define_observation_space(self, arena: BaseArena):
        _observation_space = {
            "joints": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3, len(self.actuated_joints))
            ),
            "fly": spaces.Box(low=-np.inf, high=np.inf, shape=(4, 3)),
            "contact_forces": spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(self.contact_sensor_placements), 3)
            ),
            # x, y, z positions of the end effectors (tarsus-5 segments)
            "end_effectors": spaces.Box(low=-np.inf, high=np.inf, shape=(6, 3)),
            "fly_orientation": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        }
        if self.enable_vision:
            _observation_space["vision"] = spaces.Box(
                low=0,
                high=255,
                shape=(2, self.config["vision"]["num_ommatidia_per_eye"], 2),
            )
        if self.enable_olfaction:
            _observation_space["odor_intensity"] = spaces.Box(
                low=0,
                high=np.inf,
                shape=(arena.odor_dimensions, len(self._antennae_sensors)),
            )
        return spaces.Dict(_observation_space)

    def _set_geoms_friction(self):
        for geom in self.model.find_all("geom"):
            geom.friction = self.friction

    def _set_joints_stiffness_and_damping(self):
        for joint in self.model.find_all("joint"):
            if joint.name in self.actuated_joints:
                joint.stiffness = self.joint_stiffness
                joint.damping = self.joint_damping
            elif joint.name in ("joint_Head", "joint_Head_roll", "joint_Head_yaw"):
                if self.neck_stiffness is not None:
                    joint.stiffness = self.neck_stiffness
                else:
                    joint.stiffness = self.non_actuated_joint_stiffness
                joint.damping = self.non_actuated_joint_damping
            else:
                joint.stiffness = self.non_actuated_joint_stiffness
                joint.damping = self.non_actuated_joint_damping

    def _get_real_parent(self, child):
        child_name = child.name.split("_")[0]
        parent = child.parent

        if child_name in parent.name:
            real_parent = self._get_real_parent(parent)
        else:
            real_parent = parent.name.split("_")[0]

        assert (
            real_parent is not None
        ), f"Real parent not found for {child_name} but this cannot be"
        return real_parent

    def _get_real_children(self, parent):
        real_children = []
        parent_name = parent.name.split("_")[0]
        for child in parent.get_children("body"):
            if parent_name in child.name:
                real_children.extend(self._get_real_children(child))

            else:
                real_children.extend([child.name.split("_")[0]])

        return real_children

    def _init_self_contacts(self):
        self_collision_geoms = self._parse_collision_specs(self.self_collisions)
        self_contacts: dict[str, mjcf.Element] = {}

        for geom1 in self_collision_geoms:
            for geom2 in self_collision_geoms:
                is_duplicate = f"{geom1}_{geom2}" in self_contacts
                if geom1 != geom2 and not is_duplicate:
                    # Do not add contact if the parent bodies have a child parent
                    # relationship
                    body1 = self.model.find("geom", geom1).parent
                    body2 = self.model.find("geom", geom2).parent
                    simple_body1_name = body1.name.split("_")[0]
                    simple_body2_name = body2.name.split("_")[0]

                    body1_children = self._get_real_children(body1)
                    body2_children = self._get_real_children(body2)

                    body1_parent = self._get_real_parent(body1)
                    body2_parent = self._get_real_parent(body2)

                    if not (
                        body1.name == body2.name
                        or simple_body1_name in body2_children
                        or simple_body2_name in body1_children
                        or simple_body1_name == body2_parent
                        or simple_body2_name == body1_parent
                    ):
                        contact_pair = self.model.contact.add(
                            "pair",
                            name=f"{geom1}_{geom2}",
                            geom1=geom1,
                            geom2=geom2,
                            solref=self.contact_solref,
                            solimp=self.contact_solimp,
                            margin=0.0,  # change margin to avoid penetration
                        )
                        self_contacts[f"{geom1}_{geom2}"] = contact_pair
        self._self_contacts = self_contacts

    def init_floor_contacts(self, arena: BaseArena):
        """Initialize contacts between the fly and the floor. This is
        called by the Simulation after the fly is placed in the arena and
        before setting up the physics engine.

        Parameters
        ----------
        arena : BaseArena
            The arena in which the fly is placed.
        """
        floor_collision_geoms = self._parse_collision_specs(self.floor_collisions)

        floor_contacts: dict[str, mjcf.Element] = {}
        ground_id = 0

        arena_root = arena.root_element

        for geom in arena_root.find_all("geom"):
            if geom.name is None:
                is_ground = True
            elif geom.dclass is not None and geom.dclass.dclass == "nmf":
                is_ground = False
            elif "cam" in geom.name or "sensor" in geom.name:
                is_ground = False
            else:
                is_ground = True
            if is_ground:
                for animat_geom_name in floor_collision_geoms:
                    if geom.name is None:
                        geom.name = f"groundblock_{ground_id}"
                        ground_id += 1
                    mean_friction = np.mean(
                        [
                            self.friction,  # fly friction
                            arena.friction,  # arena ground friction
                        ],
                        axis=0,
                    )
                    floor_contact_pair = arena_root.contact.add(
                        "pair",
                        name=f"{geom.name}_{self.name}_{animat_geom_name}",
                        geom1=f"{self.name}/{animat_geom_name}",
                        geom2=f"{geom.name}",
                        solref=self.contact_solref,
                        solimp=self.contact_solimp,
                        margin=0.0,  # change margin to avoid penetration
                        friction=np.repeat(
                            mean_friction,
                            (2, 1, 2),
                        ),
                    )
                    floor_contacts[
                        f"{geom.name}_{animat_geom_name}"
                    ] = floor_contact_pair

        self._floor_contacts = floor_contacts

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
                        name=f"actuatorfrc_{self.control}_{joint}",
                        actuator=f"actuator_{self.control}_{joint}",
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
        orient_sensor = self.model.sensor.add(
            "framezaxis", name="thorax_orient", objtype="body", objname="Thorax"
        )
        return [
            lin_pos_sensor,
            lin_vel_sensor,
            ang_pos_sensor,
            ang_vel_sensor,
            orient_sensor,
        ]

    def _add_end_effector_sensors(self):
        end_effector_sensors = []
        for name in self._last_tarsal_seg_names:
            sensor = self.model.sensor.add(
                "framepos",
                name=f"{name}_pos",
                objtype="body",
                objname=name,
            )
            end_effector_sensors.append(sensor)
        return end_effector_sensors

    def _add_odor_sensors(self):
        antennae_sensors = []
        for name, specs in self.config["olfaction"]["sensor_positions"].items():
            parent_body = self.model.find("body", specs["parent"])
            sensor_body = parent_body.add(
                "body", name=f"{name}_body", pos=specs["rel_pos"]
            )
            sensor = self.model.sensor.add(
                "framepos",
                name=f"{name}_pos_sensor",
                objtype="body",
                objname=f"{name}_body",
            )
            antennae_sensors.append(sensor)
            if self.draw_sensor_markers:
                sensor_body.add(
                    "geom",
                    name=f"{name}_marker",
                    type="sphere",
                    size=[0.06],
                    rgba=specs["marker_rgba"],
                )
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

    def _add_joint_actuators(self, gain, forcerange):
        # if self control is "motor" check that the gain is not provided
        if self.control == "motor" and gain is not None:
            # print warning
            logging.warning(
                "Motor control is selected, the gain parameter will not be used"
            )

        ## Need to deal with the kp, force range, and neck actuators
        if not isinstance(gain, list):
            gain = [gain] * len(self.actuated_joints)
        if not isinstance(forcerange, list):
            if isinstance(forcerange, tuple):
                forcerange = [forcerange] * len(self.actuated_joints)
            else:
                forcerange = [(-forcerange, forcerange)] * len(self.actuated_joints)

        actuators = []

        for joint, g, forcerange in zip(self.actuated_joints, gain, forcerange):
            if self.control == "position":
                actuator = self.model.actuator.add(
                    self.control,
                    name=f"actuator_{self.control}_{joint}",
                    joint=joint,
                    kp=g,
                    ctrlrange="-1000000 1000000",
                    forcerange=forcerange,
                    forcelimited=True,
                )
            elif self.control == "velocity":
                actuator = self.model.actuator.add(
                    self.control,
                    name=f"actuator_{self.control}_{joint}",
                    joint=joint,
                    kv=g,
                    ctrlrange="-1000000 1000000",
                    forcerange=forcerange,
                    forcelimited=True,
                )
            elif self.control == "motor":
                actuator = self.model.actuator.add(
                    self.control,
                    name=f"actuator_{self.control}_{joint}",
                    joint=joint,
                    ctrlrange="-1000000 1000000",
                    forcerange=forcerange,
                    forcelimited=True,
                )
            else:
                raise ValueError(f"Invalid control type {self.control}.")

            actuators.append(actuator)

        return actuators

    def _add_adhesion_actuators(self, gain):
        adhesion_actuators = []
        for name in self._last_tarsal_seg_names:
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

    def set_pose(self, pose: state.KinematicPose, physics: mjcf.Physics):
        for i in range(len(self.actuated_joints)):
            curr_joint = self.actuators[i].joint
            if (curr_joint in self.actuated_joints) and (curr_joint in pose):
                animat_name = f"{self.name}/{curr_joint}"
                physics.named.data.qpos[animat_name] = pose[curr_joint]

    def _set_compliant_tarsus(self):
        """Set the Tarsus2/3/4/5 to be compliant by setting the stiffness
        and damping to a low value"""
        stiffness = self.tarsus_stiffness
        damping = self.tarsus_damping
        for side in "LR":
            for pos in "FMH":
                for tarsus_link in range(2, 5 + 1):
                    joint = self.model.find(
                        "joint", f"joint_{side}{pos}Tarsus{tarsus_link}"
                    )
                    joint.stiffness = stiffness
                    joint.damping = damping

    def update_colors(self, physics: mjcf.Physics):
        """Update the colors of the fly's body segments. This is typically
        called by Simulation.render to update the colors of the fly before
        the cameras do the rendering.

        Parameters
        ----------
        physics : mjcf.Physics
            The physics object of the simulation.
        """
        if self.draw_adhesion:
            self._draw_adhesion(physics)

    def _draw_adhesion(self, physics: mjcf.Physics):
        """Highlight the tarsal segments of the leg having adhesion"""
        if np.any(self._last_adhesion == 1):
            physics.named.model.geom_rgba[
                self._leg_adhesion_drawing_segments[self._last_adhesion == 1].ravel()
            ] = self._adhesion_rgba
        if np.any(self._active_adhesion):
            physics.named.model.geom_rgba[
                self._leg_adhesion_drawing_segments[self._active_adhesion].ravel()
            ] = self._active_adhesion_rgba
        if np.any(self._last_adhesion == 0):
            physics.named.model.geom_rgba[
                self._leg_adhesion_drawing_segments[self._last_adhesion == 0].ravel()
            ] = self._base_rgba
        return

    def _update_vision(self, sim: "Simulation") -> None:
        """Check if the visual input needs to be updated (because the
        vision update freq does not necessarily match the physics
        simulation timestep). If needed, update the visual input of the fly
        and buffer it to ``self._curr_raw_visual_input``.
        """
        physics = sim.physics

        vision_config = self.config["vision"]
        next_render_time = (
            self._last_vision_update_time + self._eff_visual_render_interval
        )

        # avoid floating point errors: when too close, update anyway
        if sim.curr_time + 0.5 * sim.timestep < next_render_time:
            return

        raw_visual_input = []
        ommatidia_readouts = []

        for geom in self._geoms_to_hide:
            physics.named.model.geom_rgba[f"{self.name}/{geom}"] = [0.5, 0.5, 0.5, 0]

        sim.arena.pre_visual_render_hook(physics)

        for side in ["L", "R"]:
            raw_img = physics.render(
                width=vision_config["raw_img_width_px"],
                height=vision_config["raw_img_height_px"],
                camera_id=f"{self.name}/{side}Eye_cam",
            )
            fish_img = np.ascontiguousarray(self.retina.correct_fisheye(raw_img))
            readouts_per_eye = self.retina.raw_image_to_hex_pxls(fish_img)
            ommatidia_readouts.append(readouts_per_eye)
            raw_visual_input.append(fish_img)

        for geom in self._geoms_to_hide:
            physics.named.model.geom_rgba[f"{self.name}/{geom}"] = [0.5, 0.5, 0.5, 1]

        sim.arena.post_visual_render_hook(physics)
        self._curr_visual_input = np.array(ommatidia_readouts)

        if self.render_raw_vision:
            self._curr_raw_visual_input = np.array(raw_visual_input)

        self._last_vision_update_time = sim.curr_time

    def change_segment_color(self, physics: mjcf.Physics, segment: str, color):
        """Change the color of a segment of the fly.

        Parameters
        ----------
        physics : mjcf.Physics
            The physics object of the simulation.
        segment : str
            The name of the segment to change the color of.
        color : tuple[float, float, float, float]
            Target color as RGBA values normalized to [0, 1].
        """
        physics.named.model.geom_rgba[f"{self.name}/{segment}"] = color

    @property
    def vision_update_mask(self) -> np.ndarray:
        """
        The refresh frequency of the visual system is often loser than the
        same as the physics simulation time step. This 1D mask, whose
        size is the same as the number of simulation time steps, indicates
        in which time steps the visual inputs have been refreshed. In other
        words, the visual input frames where this mask is False are
        repetitions of the previous updated visual input frames.
        """
        return np.array(self._vision_update_mask)

    def get_observation(self, sim: "Simulation") -> ObsType:
        """Get observation without stepping the physics simulation.

        Returns
        -------
        ObsType
            The observation as defined by the environment.
        """
        physics = sim.physics

        # joint sensors
        joint_obs = np.zeros((3, len(self.actuated_joints)))
        joint_sensordata = physics.bind(self._joint_sensors).sensordata
        for i, joint in enumerate(self.actuated_joints):
            base_idx = i * 3
            # pos and vel and torque from the joint sensors
            joint_obs[:3, i] = joint_sensordata[base_idx : base_idx + 3]
        joint_obs[2, :] *= 1e-9  # convert to N

        # fly position and orientation
        cart_pos = physics.bind(self._body_sensors[0]).sensordata
        cart_vel = physics.bind(self._body_sensors[1]).sensordata

        quat = physics.bind(self._body_sensors[2]).sensordata
        # ang_pos = transformations.quat_to_euler(quat)
        ang_pos = R.from_quat(quat[[1, 2, 3, 0]]).as_euler(
            "ZYX"
        )  # explicitly use extrinsic ZYX
        # ang_pos[0] *= -1  # flip roll??
        ang_vel = physics.bind(self._body_sensors[3]).sensordata
        fly_pos = np.array([cart_pos, cart_vel, ang_pos, ang_vel])

        self.last_obs["rot"] = ang_pos
        self.last_obs["pos"] = cart_pos

        # contact forces from crf_ext (first three components are rotational)
        contact_forces = physics.named.data.cfrc_ext[self.contact_sensor_placements][
            :, 3:
        ].copy()
        if self.enable_adhesion:
            # Adhesion inputs force in the contact. Let's compute this force
            # and remove it from the contact forces
            contactid_normal = {}
            self._active_adhesion = np.zeros(self.n_legs, dtype=bool)
            for contact in physics.data.contact:
                id_ = np.where(self._adhesion_actuator_geom_id == contact.geom1)
                if len(id_[0]) > 0 and contact.exclude == 0:
                    contact_sensor_id = self._adhesion_bodies_with_contact_sensors[id_][
                        0
                    ]
                    if contact_sensor_id in contactid_normal:
                        contactid_normal[contact_sensor_id].append(contact.frame[:3])
                    else:
                        contactid_normal[contact_sensor_id] = [contact.frame[:3]]
                    self._active_adhesion[id_] = True
                id_ = np.where(self._adhesion_actuator_geom_id == contact.geom2)
                if len(id_[0]) > 0 and contact.exclude == 0:
                    contact_sensor_id = self._adhesion_bodies_with_contact_sensors[id_][
                        0
                    ]
                    if contact_sensor_id in contactid_normal:
                        contactid_normal[contact_sensor_id].append(contact.frame[:3])
                    else:
                        contactid_normal[contact_sensor_id] = [contact.frame[:3]]
                    self._active_adhesion[id_] = True

            for contact_sensor_id, normal in contactid_normal.items():
                adh_actuator_id = (
                    self._adhesion_bodies_with_contact_sensors == contact_sensor_id
                )
                if self._last_adhesion[adh_actuator_id] > 0:
                    if len(np.shape(normal)) > 1:
                        normal = np.mean(normal, axis=0)
                    contact_forces[contact_sensor_id, :] -= self.adhesion_force * normal

        # if draw contacts same last contact forces and positions
        # if self.draw_contacts:
        self.last_obs["contact_forces"] = contact_forces
        self.last_obs["contact_pos"] = (
            physics.named.data.xpos[self.contact_sensor_placements].copy().T
        )

        # end effector position
        ee_pos = physics.bind(self._end_effector_sensors).sensordata.copy()
        ee_pos = ee_pos.reshape((self.n_legs, 3))

        orientation_vec = physics.bind(self._body_sensors[4]).sensordata.copy()

        obs = {
            "joints": joint_obs.astype(np.float32),
            "fly": fly_pos.astype(np.float32),
            "contact_forces": contact_forces.astype(np.float32),
            "end_effectors": ee_pos.astype(np.float32),
            "fly_orientation": orientation_vec.astype(np.float32),
        }

        # olfaction
        if self.enable_olfaction:
            antennae_pos = physics.bind(self._antennae_sensors).sensordata
            odor_intensity = sim.arena.get_olfaction(antennae_pos.reshape(4, 3))
            obs["odor_intensity"] = odor_intensity.astype(np.float32)

        # vision
        if self.enable_vision:
            self._update_vision(sim)
            obs["vision"] = self._curr_visual_input.astype(np.float32)

        return obs

    def get_reward(self):
        """Get the reward for the current state of the environment. This
        method always returns 0 unless extended by the user.

        Returns
        -------
        float
            The reward.
        """
        return 0

    def is_terminated(self):
        """Whether the episode has terminated due to factors that are
        defined within the Markov Decision Process (e.g. task completion/
        failure, etc.). This method always returns False unless extended by
        the user.

        Returns
        -------
        bool
            Whether the simulation is terminated.
        """
        return False

    def is_truncated(self):
        """Whether the episode has terminated due to factors beyond the
            Markov Decision Process (e.g. time limit, etc.). This method
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
        dict[str, Any]
            The dictionary containing additional information.
        """
        info = {}
        if self.enable_vision:
            if self.render_raw_vision:
                info["raw_vision"] = self._curr_raw_visual_input.astype(np.float32)
        return info

    def reset(self, sim: "Simulation", **kwargs):
        self._last_vision_update_time = -np.inf
        self._curr_raw_visual_input = None
        self._curr_visual_input = None
        self._vision_update_mask = []
        self._flip_counter = 0

        obs = self.get_observation(sim)
        info = self.get_info()

        if self.enable_vision:
            info["vision_updated"] = True

        return obs, info

    def pre_step(self, action, sim: "Simulation"):
        physics = sim.physics
        joint_action = action["joints"]

        # estimate necessary neck actuation signals for head stabilization
        if self.head_stabilization_model is not None:
            if callable(self.head_stabilization_model):
                if self._last_observation is not None:
                    leg_joint_angles = self._last_observation["joints"][0, :]
                    leg_contact_forces = self._last_observation["contact_forces"]
                    neck_actuation = self.head_stabilization_model(
                        leg_joint_angles, leg_contact_forces
                    )
                else:
                    neck_actuation = np.zeros(2)
            elif self.head_stabilization_model == "thorax":
                quat = physics.bind(self.thorax).xquat
                quat_inv = transformations.quat_inv(quat)
                roll, pitch, _ = transformations.quat_to_euler(quat_inv, ordering="XYZ")
                neck_actuation = np.array([roll, pitch])
            else:
                raise NotImplementedError(
                    "Unknown head stabilization model"
                    "Available options are 'thorax' or a callable function."
                )

            joint_action = np.concatenate((joint_action, neck_actuation))
            self._last_neck_actuation = neck_actuation
            physics.bind(self.actuators + self.neck_actuators).ctrl = joint_action
        else:
            physics.bind(self.actuators).ctrl = joint_action

        if self.enable_adhesion:
            physics.bind(self.adhesion_actuators).ctrl = action["adhesion"]
            self._last_adhesion = action["adhesion"]

    def post_step(self, sim: "Simulation"):
        obs = self.get_observation(sim)
        reward = self.get_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.get_info()

        if self.enable_vision:
            vision_updated_this_step = sim.curr_time == self._last_vision_update_time
            self._vision_update_mask.append(vision_updated_this_step)
            info["vision_updated"] = vision_updated_this_step

        if self.detect_flip:
            if obs["contact_forces"].sum() < 1:
                self._flip_counter += 1
            else:
                self._flip_counter = 0

            flip_config = self.config["flip_detection"]
            has_passed_init = sim.curr_time > flip_config["ignore_period"]
            contact_lost_time = self._flip_counter * sim.timestep
            lost_contact_long_enough = (
                contact_lost_time > flip_config["min_flip_duration"]
            )
            info["flip"] = has_passed_init and lost_contact_long_enough
            info["flip_counter"] = self._flip_counter
            info["contact_forces"] = obs["contact_forces"].copy()

        if self.head_stabilization_model is not None:
            # this is tracked to decide neck actuation for the next step
            self._last_observation = obs
            info["neck_actuation"] = self._last_neck_actuation

        return obs, reward, terminated, truncated, info

    def close(self):
        """Release resources allocated by the environment."""
        pass
