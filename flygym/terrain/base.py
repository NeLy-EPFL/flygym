import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any


class BaseTerrain(ABC):
    arena = None
    
    @abstractmethod
    def __init__(self, *args: List, **kwargs: Dict):
        """Create a new terrain object.
        
        Attributes
        ----------
        arena : Any
            The arena object that the terrain is built on. Exactly
            what it is depends on the physics simulator.
        """
        pass
        
    @abstractmethod
    def get_spawn_position(self,
                           rel_pos: np.ndarray,
                           rel_angle: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """Given a relative entity spawn position and orientation (as
        if it was a simple flat terrain), return the abjusted position
        and orientation. This is useful for environments that have
        complex terrain (eg. with obstacles) where the entity's spawn
        position needs to be shifted accordingly.

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
    
    def spawn_entity(self,
                     entity: Any,
                     rel_pos: np.ndarray,
                     rel_angle: np.ndarray) -> None:
        """Add an entity (eg. the fly) to the arena.

        Parameters
        ----------
        entity : mjcf.RootElement
            The entity to be added to the arena.
        rel_pos : np.ndarray
            (x, y, z) position of the entity if it were spawned on a
            simple flat environment.
        rel_angle : np.ndarray
            Axis-angle representation (x, y, z, a) of the entity's
            orientation if it were spawned on a simple flat terrain.
            (x, y, z) define the 3D vector that is the rotation axis;
            a is the rotation angle in unit as configured in the model.
        """
        adj_pos, adj_angle = self.get_spawn_position(rel_pos, rel_angle)
        spawn_site = self.arena.worldbody.add('site',
                                              pos=adj_pos, axisangle=adj_angle)
        spawn_site.attach(entity).add('freejoint')