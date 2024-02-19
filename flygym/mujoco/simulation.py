from typing import Optional, Dict, Any, Tuple
from gymnasium.core import ObsType
import gymnasium as gym
from flygym.mujoco.fly import Fly
from flygym.mujoco.arena import BaseArena, FlatTerrain
import numpy as np
from dm_control.utils import transformations


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
        self.gravity = gravity
        self.fly = fly

        self._floor_height = self._get_max_floor_height(self.arena)

        self.fly.post_init(self.arena, self.timestep, self.gravity)

    # get undefined methods or properties from fly
    def __getattr__(self, name):
        return getattr(self.fly, name)

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
        return self.fly.reset(self.arena, self.gravity)

    def step(
        self, action: ObsType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        return self.fly.step(action, self.arena)

    def render(self):
        return self.fly.render(self._floor_height)

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
        self._set_gravity(new_gravity, rot_mat)

        return 0
