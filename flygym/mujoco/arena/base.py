import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional
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
            Euler representation (x, y, z) of the entity's
            orientation if it were spawned on a simple flat terrain.

        Returns
        -------
        np.ndarray
            Adjusted (x, y, z) position of the entity.
        np.ndarray
            Adjusted euler representation (x, y, z).
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
            euler angle representation (rot around x, y, z) of the entity's
            orientation if it were spawned on a simple flat terrain.
        """
        adj_pos, adj_angle = self.get_spawn_position(rel_pos, rel_angle)
        spawn_site = self.root_element.worldbody.add(
            "site", pos=adj_pos, euler=adj_angle
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

    def pre_visual_render_hook(self, physics: mjcf.Physics, *args, **kwargs) -> None:
        """Make necessary changes (eg. hide certain visualization markers)
        before rendering the visual inputs.
        """
        pass

    def post_visual_render_hook(self, physics: mjcf.Physics, *args, **kwargs) -> None:
        """Make necessary changes (eg. show certain visualization markers)
        after rendering the visual inputs.
        """
        pass

    def step(self, dt: float, physics: mjcf.Physics, *args, **kwargs) -> None:
        return


class FlatTerrain(BaseArena):
    """Flat terrain with no obstacles.

    Attributes
    ----------
    arena : mjcf.RootElement
        The arena object that the terrain is built on.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of the terrain in (x, y) dimensions.
    friction : Tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001)
    """

    def __init__(
        self,
        size: Tuple[float, float] = (50, 50),
        friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
        ground_alpha: float = 0.8,
        scale_bar_pos: Optional[Tuple[float, float, float]] = None,
    ):
        self.root_element = mjcf.RootElement()
        ground_size = [*size, 1]
        chequered = self.root_element.asset.add(
            "texture",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=(0.3, 0.3, 0.3),
            rgb2=(0.4, 0.4, 0.4),
        )
        grid = self.root_element.asset.add(
            "material",
            name="grid",
            texture=chequered,
            texrepeat=(10, 10),
            reflectance=0.1,
            rgba=(1.0, 1.0, 1.0, ground_alpha),
        )
        self.root_element.worldbody.add(
            "geom",
            type="plane",
            name="ground",
            material=grid,
            size=ground_size,
            friction=friction,
        )
        self.friction = friction
        if scale_bar_pos:
            self.root_element.worldbody.add(
                "geom",
                type="cylinder",
                size=(0.05, 0.5),
                pos=scale_bar_pos,
                rgba=(0, 0, 0, 1),
                euler=(0, np.pi / 2, 0),
            )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle
