import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

import flygym.preprogrammed as preprogrammed
import flygym.state as state
from flygym.arena import BaseArena
from flygym.camera import Camera
from flygym.fly import Fly
from flygym.simulation import SingleFlySimulation


@dataclass
class Parameters:
    """Parameters of the MuJoCo simulation.

    Attributes
    ----------
    timestep : float
        Simulation timestep in seconds, by default 0.0001.
    joint_stiffness : float
        Stiffness of actuated joints, by default 0.05.
    joint_damping : float
        Damping coefficient of actuated joints, by default 0.06.
    non_actuated_joint_stiffness : float
        Stiffness of non-actuated joints, by default 1.0. (made stiff for better stability)
    non_actuated_joint_damping : float
        Damping coefficient of non-actuated joints, by default 1.0. (made stiff for better stability)
    actuator_gain : float
        if control is "position", this is the position gain of the
        position actuators. If control is "velocity", this is the velocity
        gain of the velocity actuators. If control is "torque", this is ignored.
        By default 40.0.
    tarsus_stiffness : float
        Stiffness of the passive, compliant tarsus joints, by default 2.2.
    tarsus_damping : float
        Damping coefficient of the passive, compliant tarsus joints, by
        default 0.126.
    friction : float
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001)
    gravity : tuple[float, float, float]
        Gravity in (x, y, z) axes, by default (0., 0., -9.81e3). Note that
        the gravity is -9.81 * 1000 due to the scaling of the model.
    contact_solref: tuple[float, float]
        MuJoCo contact reference parameters (see `MuJoCo documentation
        <https://mujoco.readthedocs.io/en/stable/modeling.html#impedance>`_
        for details). By default (9.99e-01, 9.999e-01, 1.0e-03, 5.0e-01,
        2.0e+00). Under the default configuration, contacts are very stiff.
        This is to avoid penetration of the leg tips into the ground when
        leg adhesion is enabled. The user might want to decrease the
        stiffness if the stability becomes an issue.
    contact_solimp: tuple[float, float, float, float, float]
        MuJoCo contact reference parameters (see `MuJoCo docs
        <https://mujoco.readthedocs.io/en/stable/modeling.html#reference>`_
        for details). By default (9.99e-01, 9.999e-01, 1.0e-03, 5.0e-01,
        2.0e+00). Under the default configuration, contacts are very stiff.
        This is to avoid penetration of the leg tips into the ground when
        leg adhesion is enabled. The user might want to decrease the
        stiffness if the stability becomes an issue.
    enable_olfaction : bool
        Whether to enable olfaction, by default False.
    enable_vision : bool
        Whether to enable vision, by default False.
    render_raw_vision : bool
        If ``enable_vision`` is True, whether to render the raw vision
        (raw pixel values before binning by ommatidia), by default False.
    render_mode : str
        The rendering mode. Can be "saved" or "headless", by default
        "saved".
    render_window_size : tuple[int, int]
        Size of the rendered images in pixels, by default (640, 480).
    render_playspeed : float
        Play speed of the rendered video, by default 0.2.
    render_fps : int
        FPS of the rendered video when played at ``render_playspeed``, by
        default 30.
    render_camera : str
        The camera that will be used for rendering, by default
        "Animat/camera_left".
    render_timestamp_text : bool
        If True, text indicating the current simulation time will be added
        to the rendered video.
    render_playspeed_text : bool
        If True, text indicating the play speed will be added to the
        rendered video.
    vision_refresh_rate : int
        The rate at which the vision sensor is updated, in Hz, by default
        500.
    enable_adhesion : bool
        Whether to enable adhesion. By default False.
    draw_adhesion : bool
        Whether to signal that adhesion is on by changing the color of the
        concerned leg. By default False.
    adhesion_force : float
        The magnitude of the adhesion force. By default 20.
    draw_sensor_markers : bool
        If True, colored spheres will be added to the model to indicate the
        positions of the cameras (for vision) and odor sensors. By default
        False.
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
        If True, the camera will be rotated so that it aligns with the
        fly's orientation. By default False.
    perspective_arrow_length : bool
        If true, the length of the arrows indicating the contact forces
        will be determined by the perspective.
    neck_kp : float, optional
        Position gain of the neck position actuators. If supplied, this
        will overwrite ``actuator_gain`` for the neck actuators.
        Otherwise, ``actuator_gain`` will be used.
    """

    timestep: float = 0.0001
    joint_stiffness: float = 0.05
    joint_damping: float = 0.06
    non_actuated_joint_stiffness: float = 1.0
    non_actuated_joint_damping: float = 1.0
    actuator_gain: float = 40.0
    tarsus_stiffness: float = 10.0
    tarsus_damping: float = 10.0
    friction: float = (1.0, 0.005, 0.0001)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81e3)
    contact_solref: tuple[float, float] = (2e-4, 1e3)
    contact_solimp: tuple[float, float, float, float, float] = (
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
    render_window_size: tuple[int, int] = (640, 480)
    render_playspeed: float = 0.2
    render_fps: int = 30
    render_camera: str = "Animat/camera_left"
    render_timestamp_text: bool = False
    render_playspeed_text: bool = True
    vision_refresh_rate: int = 500
    enable_adhesion: bool = False
    adhesion_force: float = 40
    draw_adhesion: bool = False
    draw_sensor_markers: bool = False
    draw_contacts: bool = False
    decompose_contacts: bool = True
    force_arrow_scaling: float = float("nan")
    tip_length: float = 10.0  # number of pixels
    contact_threshold: float = 0.1
    draw_gravity: bool = False
    gravity_arrow_scaling: float = 1e-4
    align_camera_with_gravity: bool = False
    camera_follows_fly_orientation: bool = False
    perspective_arrow_length = False
    neck_kp: Optional[float] = None

    def __post_init__(self):
        if not np.isfinite(self.force_arrow_scaling):
            self.force_arrow_scaling = 1.0 if self.perspective_arrow_length else 10.0


class NeuroMechFly(SingleFlySimulation):
    """A NeuroMechFly environment using MuJoCo as the physics engine. This
    class is a wrapper around the SingleFlySimulation and is provided for
    backward compatibility.

    Attributes
    ----------
    sim_params : flygym.Parameters
        Parameters of the MuJoCo simulation.
    timestep: float
        Simulation timestep in seconds.
    output_dir : Path
        Directory to save simulation data.
    arena : flygym.arena.BaseArena
        The arena in which the fly is placed.
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
    render_mode : str
        The rendering mode. Can be "saved" or "headless".
    actuated_joints : list[str]
            List of names of actuated joints.
    contact_sensor_placements : list[str]
        List of body segments where contact sensors are placed. By
        default all tarsus segments.
    detect_flip : bool
        If True, the simulation will indicate whether the fly has flipped
        in the ``info`` returned by ``.step(...)``. Flip detection is
        achieved by checking whether the leg tips are free of any contact
        for a duration defined in the configuration file. Flip detection is
        disabled for a period of time at the beginning of the simulation as
        defined in the configuration file. This avoids spurious detection
        when the fly is not standing reliably on the ground yet.
    retina : flygym.vision.Retina
        The retina simulation object used to render the fly's visual
        inputs.
    arena_root = dm_control.mjcf.RootElement
        The root element of the arena.
    physics: dm_control.mjcf.Physics
        The MuJoCo Physics object built from the arena's MJCF model with
        the fly in it.
    curr_time : float
        The (simulated) time elapsed since the last reset (in seconds).
    action_space : gymnasium.core.ObsType
        Definition of the simulation's action space as a Gym environment.
    observation_space : gymnasium.core.ObsType
        Definition of the simulation's observation space as a Gym
        environment.
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

    def __init__(
        self,
        sim_params: Parameters = None,
        actuated_joints: list = preprogrammed.all_leg_dofs,
        contact_sensor_placements: list = preprogrammed.all_tarsi_links,
        output_dir: Optional[Path] = None,
        arena: BaseArena = None,
        xml_variant: Union[str, Path] = "seqik",
        spawn_pos: tuple[float, float, float] = (0.0, 0.0, 0.5),
        spawn_orientation: tuple[float, float, float] = (0.0, 0.0, np.pi / 2),
        control: str = "position",
        init_pose: Union[str, state.KinematicPose] = "stretch",
        floor_collisions: Union[str, list[str]] = "legs",
        self_collisions: Union[str, list[str]] = "legs",
        detect_flip: bool = False,
    ) -> None:
        """Initialize a NeuroMechFly environment.

        Parameters
        ----------
        sim_params : flygym.Parameters
            Parameters of the MuJoCo simulation.
        actuated_joints : list[str], optional
            List of names of actuated joints. By default all active leg
            DoFs.
        contact_sensor_placements : list[str], optional
            List of body segments where contact sensors are placed. By
            default all tarsus segments.
        output_dir : Path, optional
            Directory to save simulation data. If ``None``, no data will
            be saved. By default None.
        arena : flygym.arena.BaseArena, optional
            The arena in which the fly is placed. ``FlatTerrain`` will be
            used if not specified.
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
            "torque", by default "position".
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
        """

        warnings.warn(
            "Deprecation warning: The `NeuroMechFly` class has been "
            "restructured into `Simulation`, `Fly`, and `Camera`."
            "`NeuroMechFly` will be removed in future versions."
        )

        if sim_params is None:
            sim_params = Parameters()

        self.sim_params = sim_params

        fly = Fly(
            name="Animat",
            actuated_joints=actuated_joints,
            contact_sensor_placements=contact_sensor_placements,
            xml_variant=xml_variant,
            spawn_pos=spawn_pos,
            spawn_orientation=spawn_orientation,
            control=control,
            init_pose=init_pose,
            floor_collisions=floor_collisions,
            self_collisions=self_collisions,
            detect_flip=detect_flip,
            joint_stiffness=sim_params.joint_stiffness,
            joint_damping=sim_params.joint_damping,
            non_actuated_joint_stiffness=sim_params.non_actuated_joint_stiffness,
            non_actuated_joint_damping=sim_params.non_actuated_joint_damping,
            actuator_gain=sim_params.actuator_gain,
            tarsus_stiffness=sim_params.tarsus_stiffness,
            tarsus_damping=sim_params.tarsus_damping,
            friction=sim_params.friction,
            contact_solref=sim_params.contact_solref,
            contact_solimp=sim_params.contact_solimp,
            enable_olfaction=sim_params.enable_olfaction,
            enable_vision=sim_params.enable_vision,
            render_raw_vision=sim_params.render_raw_vision,
            vision_refresh_rate=sim_params.vision_refresh_rate,
            enable_adhesion=sim_params.enable_adhesion,
            adhesion_force=sim_params.adhesion_force,
            draw_adhesion=sim_params.draw_adhesion,
            draw_sensor_markers=sim_params.draw_sensor_markers,
            neck_kp=sim_params.neck_kp,
        )

        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = Path(output_dir) / "video.mp4"
        else:
            output_path = None

        self.output_dir = output_dir

        if sim_params.render_mode == "saved":
            cameras = [
                Camera(
                    fly,
                    window_size=sim_params.render_window_size,
                    play_speed=sim_params.render_playspeed,
                    fps=sim_params.render_fps,
                    camera_id=sim_params.render_camera,
                    timestamp_text=sim_params.render_timestamp_text,
                    play_speed_text=sim_params.render_playspeed_text,
                    draw_contacts=sim_params.draw_contacts,
                    decompose_contacts=sim_params.decompose_contacts,
                    force_arrow_scaling=sim_params.force_arrow_scaling,
                    tip_length=sim_params.tip_length,
                    contact_threshold=sim_params.contact_threshold,
                    draw_gravity=sim_params.draw_gravity,
                    gravity_arrow_scaling=sim_params.gravity_arrow_scaling,
                    align_camera_with_gravity=sim_params.align_camera_with_gravity,
                    camera_follows_fly_orientation=sim_params.camera_follows_fly_orientation,
                    output_path=output_path,
                )
            ]
        else:
            cameras = ()

        super().__init__(
            fly=fly,
            cameras=cameras,
            arena=arena,
            timestep=sim_params.timestep,
            gravity=sim_params.gravity,
        )

    def render(self):
        if self.sim_params.render_mode == "saved":
            return super().render()[0]

    def save_video(self, path: Union[str, Path], stabilization_time=0.02):
        if self.cameras:
            return self.cameras[0].save_video(path, stabilization_time)

    def __getattr__(self, item):
        try:
            return getattr(self.fly, item)
        except AttributeError:
            return getattr(self.cameras[0], item)
