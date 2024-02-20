from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from dm_control import mjcf
from dm_control.utils import transformations
from flygym.mujoco import preprogrammed, state
from flygym.mujoco.arena import BaseArena, FlatTerrain
from flygym.mujoco.camera import Camera
from flygym.mujoco.core import Parameters
from flygym.mujoco.fly import Fly
from gymnasium.core import ObsType, spaces


class Simulation(gym.Env):
    """A multi-fly simulation environment using MuJoCo as the physics engine.

    Attributes
    ----------
    flies : List[flygym.mujoco.fly.Fly]
        List of flies in the simulation.
    cameras : List[flygym.mujoco.camera.Camera]
        List of cameras in the simulation.
    arena : flygym.mujoco.arena.BaseArena
        The arena in which the fly is placed.
    timestep: float
        Simulation timestep in seconds.
    gravity : Tuple[float, float, float]
        Gravity in (x, y, z) axes. Note that the gravity is -9.81 * 1000
        due to the scaling of the model.
    curr_time : float
        The (simulated) time elapsed since the last reset (in seconds).
    physics: dm_control.mjcf.Physics
        The MuJoCo Physics object built from the arena's MJCF model with
        the fly in it.
    """

    def __init__(
        self,
        flies: Union[Iterable[Fly], Fly],
        cameras: Union[Iterable[Fly], Camera, None],
        arena: BaseArena = None,
        timestep: float = 0.0001,
        gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81e3),
    ):
        """
        Parameters
        ----------
        flies : Iterable[Fly] or Fly
            List of flies in the simulation.
        cameras : Iterable[Fly] or Camera, optional
            List of cameras in the simulation. Defaults to the left camera
            of the first fly.
        arena : flygym.mujoco.arena.BaseArena, optional
            The arena in which the fly is placed. ``FlatTerrain`` will be
            used if not specified.
        timestep : float
            Simulation timestep in seconds, by default 0.0001.
        gravity : Tuple[float, float, float]
            Gravity in (x, y, z) axes, by default (0., 0., -9.81e3). Note that
            the gravity is -9.81 * 1000 due to the scaling of the model.
        """
        if isinstance(flies, Iterable):
            self.flies = list(flies)
        else:
            self.flies = [flies]

        if cameras is None:
            self.cameras = [Camera(self.flies[0], render_camera="Animat/camera_left")]
        elif isinstance(cameras, Iterable):
            self.cameras = list(cameras)
        else:
            self.cameras = [cameras]

        self.arena = arena if arena is not None else FlatTerrain()
        self.timestep = timestep
        self.curr_time = 0.0

        self._floor_height = self._get_max_floor_height(self.arena)

        for fly in self.flies:
            self.arena.spawn_entity(fly.model, fly.spawn_pos, fly.spawn_orientation)

        arena_root = self.arena.root_element
        arena_root.option.timestep = timestep

        for fly in self.flies:
            fly.init_floor_contacts(self.arena)

        self.physics = mjcf.Physics.from_mjcf_model(self.arena.root_element)

        for camera in self.cameras:
            camera.initialize_dm_camera(self.physics)

        self.gravity = gravity

        self._set_init_pose()

        for fly in self.flies:
            fly.post_init(self.arena, self.physics)

    def _set_init_pose(self):
        """Set the initial pose of all flies."""
        with self.physics.reset_context():
            for fly in self.flies:
                fly.set_pose(fly.init_pose, self.physics)

    @property
    def gravity(self):
        return np.array(self.physics.model.opt.gravity)

    @gravity.setter
    def gravity(self, value):
        self.physics.model.opt.gravity[:] = value

        for camera in self.cameras:
            camera.set_gravity(value)

    @property
    def action_space(self):
        return spaces.Dict({fly.name: fly.action_space for fly in self.flies})

    @property
    def observation_space(self):
        return spaces.Dict({fly.name: fly.observation_space for fly in self.flies})

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset the Simulation.

        Parameters
        ----------
        seed : int
            Random seed for the simulation. The provided base simulation
            is deterministic, so this does not have an effect unless
            extended by the user.
        options : Dict
            Additional parameter for the simulation. There is none in the
            provided base simulation, so this does not have an effect
            unless extended by the user.

        Returns
        -------
        ObsType
            The observation as defined by the simulation environment.
        Dict[str, Any]
            Any additional information that is not part of the observation.
            This is an empty dictionary by default but the user can
            override this method to return additional information.
        """
        super().reset(seed=seed, options=options)
        self.physics.reset()

        self.curr_time = 0

        self._set_init_pose()

        obs = {}
        info = {}

        for fly in self.flies:
            fly._frames = []
            fly._last_render_time = -np.inf
            fly.last_vision_update_time = -np.inf
            fly._curr_raw_visual_input = None
            fly._curr_visual_input = None
            fly._vision_update_mask = []
            fly.flip_counter = 0
            obs[fly.name] = fly.get_observation(
                self.physics, self.arena, self.timestep, self.curr_time
            )
            info[fly.name] = fly.get_info()
            if fly.enable_vision:
                info[fly.name]["vision_updated"] = True

        return obs, info

    def step(
        self, action: ObsType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Step the simulation.

        Parameters
        ----------
        action : ObsType
            Action dictionary as defined by the simulation environment's
            action space.

        Returns
        -------
        ObsType
            The observation as defined by the simulation environment.
        float
            The reward as defined by the simulation environment.
        bool
            Whether the episode has terminated due to factors that are
            defined within the Markov Decision Process (e.g. task
            completion/failure, etc.).
        bool
            Whether the episode has terminated due to factors beyond the
            Markov Decision Process (e.g. time limit, etc.).
        Dict[str, Any]
            Any additional information that is not part of the observation.
            This is an empty dictionary by default (except when vision is
            enabled; in this case a "vision_updated" boolean variable
            indicates whether the visual input to the fly was refreshed at
            this step) but the user can override this method to return
            additional information.
        """
        self.arena.step(dt=self.timestep, physics=self.physics)

        for fly in self.flies:
            self.physics.bind(fly.actuators).ctrl = action[fly.name]["joints"]

        for fly in self.flies:
            if fly.enable_adhesion:
                self.physics.bind(fly.adhesion_actuators).ctrl = action[fly.name][
                    "adhesion"
                ]
                fly._last_adhesion = action[fly.name]["adhesion"]

        self.physics.step()
        self.curr_time += self.timestep

        obs, reward, terminated, truncated, info = {}, {}, {}, {}, {}

        for fly in self.flies:
            key = fly.name
            obs[key] = fly.get_observation(
                self.physics, self.arena, self.timestep, self.curr_time
            )
            reward[key] = fly.get_reward()
            terminated[key] = fly.is_terminated()
            truncated[key] = fly.is_truncated()
            info[key] = fly.get_info()

            if fly.enable_vision:
                vision_updated_this_step = self.curr_time == fly.last_vision_update_time
                fly._vision_update_mask.append(vision_updated_this_step)
                info[key]["vision_updated"] = vision_updated_this_step

            if fly.detect_flip:
                if obs[key]["contact_forces"].sum() < 1:
                    fly.flip_counter += 1
                else:
                    fly.flip_counter = 0
                flip_config = fly.mujoco_config["flip_detection"]
                has_passed_init = self.curr_time > flip_config["ignore_period"]
                contact_lost_time = fly.flip_counter * self.timestep
                lost_contact_long_enough = (
                    contact_lost_time > flip_config["flip_threshold"]
                )
                info[key]["flip"] = has_passed_init and lost_contact_long_enough
                info[key]["flip_counter"] = fly.flip_counter
                info[key]["contact_forces"] = obs[key]["contact_forces"].copy()

        return (
            obs,
            sum(reward.values()),
            any(terminated.values()),
            any(truncated.values()),
            info,
        )

    def render(self):
        for fly in self.flies:
            fly.update_colors(self.physics)
        return [
            camera.render(self.physics, self._floor_height, self.curr_time)
            for camera in self.cameras
        ]

    def _get_max_floor_height(self, arena):
        max_floor_height = -1 * np.inf
        for geom in arena.root_element.find_all("geom"):
            name = geom.name
            if name is None or (
                "floor" in name or "ground" in name or "treadmill" in name
            ):
                if geom.type == "box":
                    block_height = geom.pos[2] + geom.size[2]
                    max_floor_height = max(max_floor_height, block_height)
                elif geom.type == "plane":
                    try:
                        plane_height = geom.pos[2]
                    except TypeError:
                        plane_height = 0.0
                    max_floor_height = max(max_floor_height, plane_height)
                elif geom.type == "sphere":
                    sphere_height = geom.parent.pos[2] + geom.size[0]
                    max_floor_height = max(max_floor_height, sphere_height)
        if np.isinf(max_floor_height):
            max_floor_height = (fly.spawn_pos[2] for fly in self.flies)
        return max_floor_height

    def set_slope(self, slope: float, rot_axis="y"):
        """Set the slope of the simulation environment and modify the camera
        orientation so that gravity is always pointing down. Changing the
        gravity vector might be useful during climbing simulations. The
        change in the camera angle has been extensively tested for the
        simple cameras (left, right, top, bottom, front, back) but not for
        the composed ones.

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
        new_gravity = np.dot(rot_mat, self.gravity)

        self.physics.model.opt.gravity[:] = new_gravity

        for camera in self.cameras:
            camera.set_gravity(new_gravity, rot_mat)

    def _get_center_of_mass(self):
        """Get the center of mass of the flies.
        (subtree com weighted by mass) STILL NEEDS TO BE TESTED MORE THOROUGHLY

        Returns
        -------
        np.ndarray
            The center of mass of the fly.
        """
        return np.average(
            self.physics.data.subtree_com, axis=0, weights=self.physics.data.crb[:, 0]
        )

    def close(self) -> None:
        """Close the simulation, save data, and release any resources."""

        for camera in self.cameras:
            if camera.output_path is not None:
                camera.output_path.parent.mkdir(parents=True, exist_ok=True)
                camera.save_video(camera.output_path)


class SingleFlySimulation(Simulation):
    """A single fly simulation environment using MuJoCo as the physics engine.

    This class is a wrapper around the Simulation class with a single fly.
    It is provided for convenience, so that the action and observation spaces
    do not have to be keyed by the fly's name.

    Attributes
    ----------
    fly : flygym.mujoco.fly.Fly
        The fly in the simulation.
    cameras : List[flygym.mujoco.camera.Camera]
        List of cameras in the simulation.
    arena : flygym.mujoco.arena.BaseArena
        The arena in which the fly is placed.
    timestep: float
        Simulation timestep in seconds.
    gravity : Tuple[float, float, float]
        Gravity in (x, y, z) axes. Note that the gravity is -9.81 * 1000
        due to the scaling of the model.
    curr_time : float
        The (simulated) time elapsed since the last reset (in seconds).
    physics: dm_control.mjcf.Physics
        The MuJoCo Physics object built from the arena's MJCF model with
        the fly in it.
    """

    def __init__(
        self,
        fly: Fly,
        cameras: Union[Camera, Iterable[Camera], None] = None,
        arena: BaseArena = None,
        timestep: float = 0.0001,
        gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81e3),
    ):
        """
        Parameters
        ----------
        fly : Fly
            The fly in the simulation.
        cameras : Iterable[Fly] or Camera, optional
            List of cameras in the simulation. Defaults to the left camera
            of the first fly.
        arena : flygym.mujoco.arena.BaseArena, optional
            The arena in which the fly is placed. ``FlatTerrain`` will be
            used if not specified.
        timestep : float
            Simulation timestep in seconds, by default 0.0001.
        gravity : Tuple[float, float, float]
            Gravity in (x, y, z) axes, by default (0., 0., -9.81e3). Note that
            the gravity is -9.81 * 1000 due to the scaling of the model.
        """
        self.fly = fly
        super().__init__(
            flies=[fly],
            cameras=cameras,
            arena=arena,
            timestep=timestep,
            gravity=gravity,
        )

    @property
    def action_space(self):
        return spaces.Dict(super().action_space[self.flies[0].name])

    @property
    def observation_space(self):
        return spaces.Dict(super().observation_space[self.flies[0].name])

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset the simulation environment.

        Parameters
        ----------
        seed : int
            Random seed for the environment. The provided base simulation
            is deterministic, so this does not have an effect unless
            extended by the user.
        options : Dict
            Additional parameter for the simulation. There is none in the
            provided base simulation, so this does not have an effect
            unless extended by the user.

        Returns
        -------
        ObsType
            The observation as defined by the environment.
        Dict[str, Any]
            Any additional information that is not part of the observation.
            This is an empty dictionary by default but the user can
            override this method to return additional information.
        """
        key = self.flies[0].name
        obs, info = super().reset(seed=seed, options=options)
        return obs[key], info[key]

    def step(
        self, action: ObsType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Step the simulation environment.

        Parameters
        ----------
        action : ObsType
            Action dictionary as defined by the environment's action space.

        Returns
        -------
        ObsType
            The observation as defined by the environment.
        float
            The reward as defined by the environment.
        bool
            Whether the episode has terminated due to factors that are
            defined within the Markov Decision Process (e.g. task
            completion/failure, etc.).
        bool
            Whether the episode has terminated due to factors beyond the
            Markov Decision Process (e.g. time limit, etc.).
        Dict[str, Any]
            Any additional information that is not part of the observation.
            This is an empty dictionary by default (except when vision is
            enabled; in this case a "vision_updated" boolean variable
            indicates whether the visual input to the fly was refreshed at
            this step) but the user can override this method to return
            additional information.
        """
        key = self.flies[0].name
        obs, reward, terminated, truncated, info = super().step({key: action})
        return obs[key], reward, terminated, truncated, info[key]


class NeuroMechFly(SingleFlySimulation):
    """A NeuroMechFly environment using MuJoCo as the physics engine. This
    class is a wrapper around the SingleFlySimulation and is provided for
    backward compatibility.

    Attributes
    ----------
    sim_params : flygym.mujoco.Parameters
        Parameters of the MuJoCo simulation.
    timestep: float
        Simulation timestep in seconds.
    output_dir : Path
        Directory to save simulation data.
    arena : flygym.mujoco.arena.BaseArena
        The arena in which the fly is placed.
    spawn_pos : Tuple[float, float, float]
        The (x, y, z) position in the arena defining where the fly will be
        spawn, in mm.
    spawn_orientation : Tuple[float, float, float, float]
        The spawn orientation of the fly in the Euler angle format: (x, y,
        z), where x, y, z define the rotation around x, y and z in radian.
    control : str
        The joint controller type. Can be "position", "velocity", or
        "torque".
    init_pose : flygym.state.BaseState
        Which initial pose to start the simulation from.
    render_mode : str
        The rendering mode. Can be "saved" or "headless".
    actuated_joints : List[str]
            List of names of actuated joints.
    contact_sensor_placements : List[str]
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
    retina : flygym.mujoco.vision.Retina
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
        actuated_joints: List = preprogrammed.all_leg_dofs,
        contact_sensor_placements: List = preprogrammed.all_tarsi_links,
        output_dir: Optional[Path] = None,
        arena: BaseArena = None,
        xml_variant: Union[str, Path] = "seqik",
        spawn_pos: Tuple[float, float, float] = (0.0, 0.0, 0.5),
        spawn_orientation: Tuple[float, float, float] = (0.0, 0.0, np.pi / 2),
        control: str = "position",
        init_pose: Union[str, state.KinematicPose] = "stretch",
        floor_collisions: Union[str, List[str]] = "legs",
        self_collisions: Union[str, List[str]] = "legs",
        detect_flip: bool = False,
    ) -> None:
        """Initialize a NeuroMechFly environment.

        Parameters
        ----------
        sim_params : flygym.mujoco.Parameters
            Parameters of the MuJoCo simulation.
        actuated_joints : List[str], optional
            List of names of actuated joints. By default all active leg
            DoFs.
        contact_sensor_placements : List[str], optional
            List of body segments where contact sensors are placed. By
            default all tarsus segments.
        output_dir : Path, optional
            Directory to save simulation data. If ``None``, no data will
            be saved. By default None.
        arena : flygym.mujoco.arena.BaseArena, optional
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
        spawn_pos : Tuple[float, float, float], optional
            The (x, y, z) position in the arena defining where the fly
            will be spawn, in mm. By default (0, 0, 0.5).
        spawn_orientation : Tuple[float, float, float], optional
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

        if sim_params is None:
            sim_params = Parameters()

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
            actuator_kp=sim_params.actuator_kp,
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
        )

        if output_dir is not None:
            output_path = Path(output_dir) / "video.mp4"
        else:
            output_path = None

        camera = Camera(
            fly,
            render_window_size=sim_params.render_window_size,
            render_playspeed=sim_params.render_playspeed,
            render_fps=sim_params.render_fps,
            render_camera=sim_params.render_camera,
            render_timestamp_text=sim_params.render_timestamp_text,
            render_playspeed_text=sim_params.render_playspeed_text,
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

        super().__init__(
            fly=fly,
            cameras=camera,
            arena=arena,
            timestep=sim_params.timestep,
            gravity=sim_params.gravity,
        )
