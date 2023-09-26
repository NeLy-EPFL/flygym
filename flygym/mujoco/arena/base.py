import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional
from dm_control import mjcf


class BaseArena(ABC):
    """Base class for all arenas.

    Attributes
    ----------
    arena : Any
        The arena object that the terrain is built on. Exactly what it
        is depends on the physics simulator.
    friction : Tuple [float]
        Default sliding, torsional, and rolling friction coefficients of
        surfaces. This is provided for the user's convinience but can be
        overriden for either all or some surfaces.
    """

    friction = (100.0, 0.005, 0.0001)

    @abstractmethod
    def __init__(self, *args: List, **kwargs: Dict):
        """Create a new terrain object."""
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

        For example, if the arena has flat terrain, this method can simply
        return ``rel_pos``, ``rel_angle`` unchanged (as is the case by
        default). If there is are featues on the ground that are 0.1 mm in
        height, then this method should return ``rel_pos + [0, 0, 0.1],
        rel_angle``.

        Parameters
        ----------
        rel_pos : np.ndarray
            (x, y, z) position of the entity in mm as supplied by the user
            (before any transformation).
        rel_angle : np.ndarray
            Euler angle (rotation along x, y, z in radian) of the fly's
            orientation as supplied by the user (before any
            transformation).
        *args
            User defined arguments and keyword arguments.
        **kwargs
            User defined arguments and keyword arguments.

        Returns
        -------
        np.ndarray
            Adjusted (x, y, z) position of the entity.
        np.ndarray
            Adjusted euler angle (rotation along x, y, z in raidan) of the
            fly's oreintation.
        """
        pass

    def spawn_entity(
        self, entity: Any, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> None:
        """Add the fly to the arena.

        Parameters
        ----------
        entity : mjcf.RootElement
            The entity to be added to the arena (this should be the fly).
        rel_pos : np.ndarray
            (x, y, z) position of the entity.
        rel_angle : np.ndarray
            euler angle representation (rot around x, y, z) of the entity's
            orientation if it were spawned on a simple flat terrain.
        """
        adj_pos, adj_angle = self.get_spawn_position(rel_pos, rel_angle)
        spawn_site = self.root_element.worldbody.add(
            "site", pos=adj_pos, euler=adj_angle
        )
        spawn_site.attach(entity).add("freejoint")

    def get_olfaction(self, sensor_pos: np.ndarray) -> np.ndarray:
        """Get the odor intensity readings from the environment.

        Parameters
        ----------
        sensor_pos : np.ndarray
            The Cartesian coordinates of the antennae of the fly as a
            (n, 3) NumPy array where n is the number of sensors (usually
            n=4: 2 antennae + 2 maxillary palps), and the second dimension
            gives the corrdinates in (x, y, z).

        Returns
        -------
        np.ndarray
            The odor intensity readings from the environment as a (k, n)
            NumPy array where k is the dimension of the odor signal and n
            is the number of odor sensors (usally n=4: 2 antennae + 2
            maxillary palps).
        """
        return np.zeros((0, 2))

    @property
    def odor_dimensions(self) -> int:
        """The dimension of the odor signal. This can be used to emulate
        multiple monomolecular chemical concentrations or multiple
        composite ordor intensities.

        Returns
        -------
        int
            The dimension of the odor space.
        """
        return 0

    def pre_visual_render_hook(self, physics: mjcf.Physics, *args, **kwargs) -> None:
        """Make necessary changes (eg. make certain visualization markers
        transparent) before rendering the visual inputs. By default, this
        does nothing.
        """
        pass

    def post_visual_render_hook(self, physics: mjcf.Physics, *args, **kwargs) -> None:
        """Make necessary changes (eg. make certain visualization markers
        opaque) after rendering the visual inputs. By default, this does
        nothing.
        """
        pass

    def step(self, dt: float, physics: mjcf.Physics, *args, **kwargs) -> None:
        """Advance the arena by one step. This is useful for interactive
        environments (eg. moving object). Typically, this method is called
        from the core simulation class (eg. ``NeuroMechFlyMuJoCo``).

        Parameters
        ----------
        dt : float
            The time step in seconds since the last update. Typically, this
            is the same as the time step of the physics simulation
            (provided that this method is called by the core simulation
            every time the simulation steps).
        physics : mjcf.Physics
            The physics object of the simulation. This is typically
            provided by the core simulation class (eg.
            ``NeuroMechFlyMuJoCo.physics``) when the core simulation calls
            this method.
        *args
            User defined arguments and keyword arguments.
        **kwargs
            User defined arguments and keyword arguments.
        """
        return


class FlatTerrain(BaseArena):
    """Flat terrain with no obstacles.

    Attributes
    ----------
    root_element : mjcf.RootElement
        The root MJCF element of the arena.
    friction : Tuple[float, float, float]
        The sliding, torsional, and rolling friction coefficients of the
        ground, by default (1, 0.005, 0.0001).

    Parameters
    ----------
    size : Tuple[float, float], optional
        The size of the arena in mm, by default (50, 50).
    friction : Tuple[float, float, float]
        The sliding, torsional, and rolling friction coefficients of the
        ground, by default (1, 0.005, 0.0001).
    ground_alpha : float
        Opacity of the ground, by default 1 (fully opaque).
    scale_bar_pos : Tuple[float, float, float], optional
        If supplied, a 1 mm scale bar will be placed at this location.
    """

    def __init__(
        self,
        size: Tuple[float, float] = (50, 50),
        friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
        ground_alpha: float = 1.0,
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
