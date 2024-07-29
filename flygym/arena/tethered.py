import numpy as np
from typing import Any
from dm_control import mjcf

from .base import BaseArena


class Tethered(BaseArena):
    """Fly tethered in the air.

    Attributes
    ----------
    root_element : Any
        The arena object that the terrain is built on. Exactly what it
        is depends on the physics simulator.
    friction : tuple[float, float, float]
        The sliding, torsional, and rolling friction coefficients of the
        ground, by default (1, 0.005, 0.0001).
    """

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle

    def spawn_entity(
        self, entity: Any, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> None:
        """Add an entity (e.g. the fly) to the arena.

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
        spawn_site.attach(entity).add(
            "joint", name="prismatic_support_1", limited=True, range=(0, 1e-10)
        )


class Ball(Tethered):
    """Fly tethered on a spherical treadmill.

    Attributes
    ----------
    root_element : mjcf.RootElement
        The arena object that the terrain is built on.
    friction : tuple[float, float, float]
        The sliding, torsional, and rolling friction coefficients of the
        ground, by default (1, 0.005, 0.0001).

    Parameters
    ----------
    radius : float, optional
        Radius of the ball, by default 5.390852782067457
    ball_pos : tuple[float, float, float], optional
        (x, y, z) mounting position of the ball, by default
        (-0.09867235483, -0.05435809692, -5.20309506806)
    mass : float, optional
        Mass of the ball, by default 0.05456
    sliding_friction : float, optional
        Sliding friction coefficient of the ball, by default 1.3
    torsional_friction : float, optional
        Torsional friction coefficient of the ball, by default 0.005
    rolling_friction : float, optional
        Rolling friction coefficient of the ball, by default 0.0001
    """

    def __init__(
        self,
        radius: float = 5.390852782067457,
        ball_pos: tuple[float, float, float] = (
            -0.09867235483,
            -0.05435809692,
            -5.20309506806,
        ),
        mass: float = 0.05456,
        sliding_friction: float = 1.3,
        torsional_friction: float = 0.005,
        rolling_friction: float = 0.0001,
    ):
        super().__init__()

        chequered = self.root_element.asset.add(
            "texture",
            type="2d",
            builtin="checker",
            width=50,
            height=50,
            rgb1=(0.3, 0.3, 0.3),
            rgb2=(0.4, 0.4, 0.4),
        )
        grid = self.root_element.asset.add(
            "material",
            name="grid",
            texture=chequered,
            texrepeat=(3, 3),
            reflectance=0.1,
        )

        treadmill_body = self.root_element.worldbody.add(
            "body", name="treadmill", pos=ball_pos
        )

        treadmill_body.add(
            "geom",
            name="treadmill",
            type="sphere",
            size=[radius],
            mass=mass,
            friction=[sliding_friction, torsional_friction, rolling_friction],
            material=grid,
        )

        treadmill_body.add(
            "joint", name="treadmill_joint", type="ball", limited="false"
        )
        treadmill_body.add("inertial", pos=[0, 0, 0], mass=mass)
