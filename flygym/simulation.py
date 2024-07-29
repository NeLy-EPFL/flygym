from typing import Any, Iterable, Optional, Union

import gymnasium as gym
import numpy as np
from dm_control import mjcf
from dm_control.utils import transformations
from flygym.arena import BaseArena, FlatTerrain
from flygym.camera import Camera
from flygym.fly import Fly
from gymnasium.core import ObsType, spaces


class Simulation(gym.Env):
    """A multi-fly simulation environment using MuJoCo as the physics
    engine.

    Attributes
    ----------
    flies : list[flygym.fly.Fly]
        List of flies in the simulation.
    cameras : list[flygym.camera.Camera]
        List of cameras in the simulation.
    arena : flygym.arena.BaseArena
        The arena in which the fly is placed.
    timestep: float
        Simulation timestep in seconds.
    gravity : tuple[float, float, float]
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
        cameras: Union[Iterable[Camera], Camera, None],
        arena: BaseArena = None,
        timestep: float = 0.0001,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81e3),
    ):
        """
        Parameters
        ----------
        flies : Iterable[flygym.fly.Fly] or Fly
            List of flies in the simulation.
        cameras : Iterable[flygym.camera.Camera] or Camera, optional
            List of cameras in the simulation. Defaults to the left camera
            of the first fly.
        arena : flygym.arena.BaseArena, optional
            The arena in which the fly is placed. ``FlatTerrain`` will be
            used if not specified.
        timestep : float
            Simulation timestep in seconds, by default 0.0001.
        gravity : tuple[float, float, float]
            Gravity in (x, y, z) axes, by default (0., 0., -9.81e3). Note
            that the gravity is -9.81 * 1000 due to the scaling of the
            model.
        """
        if isinstance(flies, Iterable):
            self.flies = list(flies)
        else:
            self.flies = [flies]

        if cameras is None:
            self.cameras = [Camera(self.flies[0], camera_id="Animat/camera_left")]
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
            fly.post_init(self)

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

    @action_space.setter
    def action_space(self, value):
        for fly in self.flies:
            fly.action_space = value[fly.name]

    @property
    def observation_space(self):
        return spaces.Dict({fly.name: fly.observation_space for fly in self.flies})

    @observation_space.setter
    def observation_space(self, value):
        for fly in self.flies:
            fly.observation_space = value[fly.name]

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the Simulation.

        Parameters
        ----------
        seed : int
            Random seed for the simulation. The provided base simulation
            is deterministic, so this does not have an effect unless
            extended by the user.
        options : dict
            Additional parameter for the simulation. There is none in the
            provided base simulation, so this does not have an effect
            unless extended by the user.

        Returns
        -------
        ObsType
            The observation as defined by the simulation environment.
        dict[str, Any]
            Any additional information that is not part of the observation.
            This is an empty dictionary by default but the user can
            override this method to return additional information.
        """
        super().reset(seed=seed, options=options)
        self.physics.reset()
        self.curr_time = 0
        self._set_init_pose()

        for camera in self.cameras:
            camera.reset()

        obs = {}
        info = {}

        for fly in self.flies:
            obs[fly.name], info[fly.name] = fly.reset(self, seed=seed)

        return obs, info

    def step(
        self, action: ObsType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
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
        dict[str, Any]
            Any additional information that is not part of the observation.
            This is an empty dictionary by default (except when vision is
            enabled; in this case a "vision_updated" boolean variable
            indicates whether the visual input to the fly was refreshed at
            this step) but the user can override this method to return
            additional information.
        """
        self.arena.step(dt=self.timestep, physics=self.physics)

        for fly in self.flies:
            fly.pre_step(action[fly.name], self)

        self.physics.step()
        self.curr_time += self.timestep

        obs, reward, terminated, truncated, info = {}, {}, {}, {}, {}

        for fly in self.flies:
            key = fly.name
            obs_, reward_, terminated_, truncated_, info_ = fly.post_step(self)
            obs[key] = obs_
            reward[key] = reward_
            terminated[key] = terminated_
            truncated[key] = truncated_
            info[key] = info_

        return (
            obs,
            sum(reward.values()),
            any(terminated.values()),
            any(truncated.values()),
            info,
        )

    def render(self, *args, **kwargs):
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
            max_floor_height = min(fly.spawn_pos[2] for fly in self.flies)

        return max_floor_height

    def set_slope(self, slope: float, rot_axis="y"):
        """Set the slope of the simulation environment and modify the
        camera orientation so that gravity is always pointing down.
        Changing the gravity vector might be useful during climbing
        simulations. The change in the camera angle has been extensively
        tested for the simple cameras (left, right, top, bottom, front,
        back) but not for the composed ones.

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
        (subtree com weighted by mass) STILL NEEDS TO BE TESTED MORE
        THOROUGHLY

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

        for fly in self.flies:
            fly.close()


class SingleFlySimulation(Simulation):
    """A single fly simulation environment using MuJoCo as the physics
    engine.

    This class is a wrapper around the Simulation class with a single fly.
    It is provided for convenience, so that the action and observation
    spaces do not have to be keyed by the fly's name.

    Attributes
    ----------
    fly : flygym.fly.Fly
        The fly in the simulation.
    cameras : list[flygym.camera.Camera]
        List of cameras in the simulation.
    arena : flygym.arena.BaseArena
        The arena in which the fly is placed.
    timestep: float
        Simulation timestep in seconds.
    gravity : tuple[float, float, float]
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
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81e3),
    ):
        """
        Parameters
        ----------
        fly : Fly
            The fly in the simulation.
        cameras : Iterable[Fly] or Camera, optional
            List of cameras in the simulation. Defaults to the left camera
            of the first fly.
        arena : flygym.arena.BaseArena, optional
            The arena in which the fly is placed. ``FlatTerrain`` will be
            used if not specified.
        timestep : float
            Simulation timestep in seconds, by default 0.0001.
        gravity : tuple[float, float, float]
            Gravity in (x, y, z) axes, by default (0., 0., -9.81e3). Note
            that the gravity is -9.81 * 1000 due to the scaling of the
            model.
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
        return self.flies[0].action_space

    @action_space.setter
    def action_space(self, value):
        self.flies[0].action_space = value

    @property
    def observation_space(self):
        return self.flies[0].observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.flies[0].observation_space = value

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the simulation environment.

        Parameters
        ----------
        seed : int
            Random seed for the environment. The provided base simulation
            is deterministic, so this does not have an effect unless
            extended by the user.
        options : dict
            Additional parameter for the simulation. There is none in the
            provided base simulation, so this does not have an effect
            unless extended by the user.

        Returns
        -------
        ObsType
            The observation as defined by the environment.
        dict[str, Any]
            Any additional information that is not part of the observation.
            This is an empty dictionary by default but the user can
            override this method to return additional information.
        """
        key = self.flies[0].name
        obs, info = super().reset(seed=seed, options=options)
        return obs[key], info[key]

    def step(
        self, action: ObsType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
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
        dict[str, Any]
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

    def get_observation(self) -> ObsType:
        return self.fly.get_observation(self)
