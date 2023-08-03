import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
from dm_control import mjcf


class BaseArena(ABC):
    """Base class for all arenas."""

    friction = (100.0, 0.005, 0.0001)

    @abstractmethod
    def __init__(self, *args: List, **kwargs: Dict):
        """Create a new terrain object.

        Attributes
        ----------
        arena : Any
            The arena object that the terrain is built on. Exactly what it
            is depends on the physics simulator.
        """
        self.root_element = mjcf.RootElement()

    @abstractmethod
    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Given a relative entity spawn position and orientation (as if it
        was a simple flat terrain), return the abjusted position and
        orientation. This is useful for environments that have complex
        terrain (eg. with obstacles) where the entity's spawn position
        needs to be shifted accordingly.

        Parameters
        ----------
        rel_pos : np.ndarray
            (x, y, z) position of the entity if it were spawned on a
            simple flat environment.
        rel_angle : np.ndarray
            Axis-angle representation (x, y, z, a) of the entity's
            orientation if it were spawned on a simple flat terrain.
            (x, y, z) define the 3D vector that is the rotation axis;
            a is the rotation angle in unit as configured in the model.

        Returns
        -------
        np.ndarray
            Adjusted (x, y, z) position of the entity.
        np.ndarray
            Adjusted axis-angle representation (x, y, z, a).
        """
        pass

    def spawn_entity(
        self, entity: Any, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> None:
        """Add an entity (eg. the fly) to the arena.

        Parameters
        ----------
        entity : mjcf.RootElement
            The entity to be added to the arena.
        rel_pos : np.ndarray
            (x, y, z) position of the entity if it were spawned on a simple
            flat environment.
        rel_angle : np.ndarray
            Axis-angle representation (x, y, z, a) of the entity's
            orientation if it were spawned on a simple flat terrain.
            (x, y, z) define the 3D vector that is the rotation axis; a is
            the rotation angle in unit as configured in the model.
        """
        adj_pos, adj_angle = self.get_spawn_position(rel_pos, rel_angle)
        spawn_site = self.root_element.worldbody.add(
            "site", pos=adj_pos, axisangle=adj_angle
        )
        spawn_site.attach(entity).add("freejoint")

    def get_olfaction(self, antennae_pos: np.ndarray) -> np.ndarray:
        """Get the odor intensity readings from the environment.

        Parameters
        ----------
        antennae_pos : np.ndarray
            The Cartesian coordinates of the antennae of the fly as a
            (2, 3) NumPy array.

        Returns
        -------
        np.ndarray
            The odor intensity readings from the environment as a
            (k, 2) NumPy array where k is the dimension of the odor
            signal.
        """
        return np.zeros((0, 2))

    def step(self, dt: float, physics: mjcf.Physics, *args, **kwargs) -> None:
        return
