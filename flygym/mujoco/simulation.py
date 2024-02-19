from typing import Optional, Dict, Any, Tuple
from gymnasium.core import ObsType
import gymnasium as gym
from flygym.mujoco.fly import Fly
from flygym.mujoco.arena import BaseArena, FlatTerrain


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
        return self.fly.reset()

    def step(
        self, action: ObsType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        return self.fly.step(action)

    def render(self):
        return self.fly.render()
