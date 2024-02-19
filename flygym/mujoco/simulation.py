from typing import Optional, Dict, Any, Tuple
from gymnasium.core import ObsType
import gymnasium as gym
from flygym.mujoco.fly import Fly


class Simulation(gym.Env):
    def __init__(
        self,
        fly: Fly,
    ):
        self.fly = fly

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
