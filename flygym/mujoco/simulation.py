from typing import Optional, Dict, Any, Tuple
from gymnasium.core import ObsType
import gymnasium as gym
from flygym.mujoco.fly import Fly
from flygym.mujoco.arena import BaseArena, FlatTerrain
import numpy as np
from dm_control.utils import transformations
from dm_control import mjcf


class Simulation(gym.Env):
    """
    Attributes
    ----------
    arena : flygym.arena.BaseWorld
        The arena in which the fly is placed.
    timestep: float
        Simulation timestep in seconds.
    gravity : Tuple[float, float, float]
        Gravity in (x, y, z) axes. Note that the gravity is -9.81 * 1000
        due to the scaling of the model.
    """

    def __init__(
        self,
        fly: Fly,
        arena: BaseArena = None,
        timestep: float = 0.0001,
        gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81e3),
    ):
        """
        Parameters
        ----------
        arena : flygym.mujoco.arena.BaseArena, optional
            The arena in which the fly is placed. ``FlatTerrain`` will be
            used if not specified.
        timestep : float
            Simulation timestep in seconds, by default 0.0001.
        gravity : Tuple[float, float, float]
            Gravity in (x, y, z) axes, by default (0., 0., -9.81e3). Note that
            the gravity is -9.81 * 1000 due to the scaling of the model.
        """
        self.arena = arena if arena is not None else FlatTerrain()
        self.timestep = timestep
        self.fly = fly
        self.curr_time = 0

        self._floor_height = self._get_max_floor_height(self.arena)

        fly = self.fly
        self.arena.spawn_entity(fly.model, fly.spawn_pos, fly.spawn_orientation)
        arena_root = self.arena.root_element
        arena_root.option.timestep = timestep

        self.fly.init_floor_collisions(self.arena)
        self.physics = mjcf.Physics.from_mjcf_model(self.arena.root_element)

        self.gravity = gravity

        # Apply initial pose.(TARSI MUST HAVE MADE COMPLIANT BEFORE)!
        fly.set_pose(fly.init_pose, self.physics)

        self.fly.post_init(self.arena, self.physics, self.gravity)

    @property
    def gravity(self):
        return np.array(self.physics.model.opt.gravity)

    @gravity.setter
    def gravity(self, value):
        self.physics.model.opt.gravity[:] = value
        self.fly.set_gravity(value)

    @property
    def action_space(self):
        return self.fly.action_space

    @property
    def observation_space(self):
        return self.fly.observation_space

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset the Gym environment.

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
        super().reset(seed=seed)
        fly = self.fly
        self.physics.reset()
        if np.any(self.physics.model.opt.gravity[:] - self.gravity > 1e-3):
            fly.set_gravity(self.gravity)
            if fly.align_camera_with_gravity:
                fly._camera_rot = np.eye(3)
        self.curr_time = 0
        fly.set_pose(fly.init_pose, self.physics)
        fly._frames = []
        fly._last_render_time = -np.inf
        fly._last_vision_update_time = -np.inf
        fly._curr_raw_visual_input = None
        fly._curr_visual_input = None
        fly._vision_update_mask = []
        fly._flip_counter = 0
        obs = fly.get_observation(
            self.physics, self.arena, self.timestep, self.curr_time
        )
        info = fly.get_info()
        if fly.enable_vision:
            info["vision_updated"] = True
        return obs, info

    def step(
        self, action: ObsType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Step the Gym environment.

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
        self.arena.step(dt=self.timestep, physics=self.physics)

        fly = self.fly

        self.physics.bind(fly._actuators).ctrl = action["joints"]
        if fly.enable_adhesion:
            self.physics.bind(fly._adhesion_actuators).ctrl = action["adhesion"]
            fly._last_adhesion = action["adhesion"]

        self.physics.step()
        self.curr_time += self.timestep
        observation = fly.get_observation(
            self.physics, self.arena, self.timestep, self.curr_time
        )
        reward = fly.get_reward()
        terminated = fly.is_terminated()
        truncated = fly.is_truncated()
        info = fly.get_info()
        if fly.enable_vision:
            vision_updated_this_step = self.curr_time == fly._last_vision_update_time
            fly._vision_update_mask.append(vision_updated_this_step)
            info["vision_updated"] = vision_updated_this_step

        if fly.detect_flip:
            if observation["contact_forces"].sum() < 1:
                fly._flip_counter += 1
            else:
                fly._flip_counter = 0
            flip_config = fly._mujoco_config["flip_detection"]
            has_passed_init = self.curr_time > flip_config["ignore_period"]
            contact_lost_time = fly._flip_counter * self.timestep
            lost_contact_long_enough = contact_lost_time > flip_config["flip_threshold"]
            info["flip"] = has_passed_init and lost_contact_long_enough
            info["flip_counter"] = fly._flip_counter
            info["contact_forces"] = observation["contact_forces"].copy()

        return observation, reward, terminated, truncated, info

    def render(self):
        return self.fly.render(self.physics, self._floor_height, self.curr_time)

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
            max_floor_height = self.spawn_pos[2]
        return max_floor_height

    def set_slope(self, slope: float, rot_axis="y"):
        """Set the slope of the environment and modify the camera
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

        self.gravity = new_gravity
        self.fly.set_gravity(new_gravity, rot_mat)

        return 0

    def _get_center_of_mass(self):
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
