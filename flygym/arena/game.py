import numpy as np
from typing import Optional, Callable
from dm_control import mjcf

from .base import BaseArena


class SlalomArena(BaseArena):
    """Flat terrain with no obstacles.

    Attributes
    ----------
    root_element : mjcf.RootElement
        The root MJCF element of the arena.
    friction : tuple[float, float, float]
        The sliding, torsional, and rolling friction coefficients of the
        ground, by default (1, 0.005, 0.0001).

    Parameters
    ----------
    size : tuple[float, float], optional
        The size of the arena in mm, by default (50, 50).
    friction : tuple[float, float, float]
        The sliding, torsional, and rolling friction coefficients of the
        ground, by default (1, 0.005, 0.0001).
    ground_alpha : float
        Opacity of the ground, by default 1 (fully opaque).
    scale_bar_pos : tuple[float, float, float], optional
        If supplied, a 1 mm scale bar will be placed at this location.
    """

    def __init__(
        self,
        size: tuple[float, float] = (100, 100),
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
        ground_alpha: float = 1.0,
        gate_offset: float = -3.0,
        gate_width: float = 10.0,
        gate_height: float = 5.0,
        gate_spacing: float = 10.0,
        pole_radius: float = 0.05,
        n_gates: int = 5,
    ):
        super().__init__()

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

        # Start poles
        self.root_element.worldbody.add(
            "geom",
            type="cylinder",
            name="start_pole_left",
            size=[pole_radius, gate_height / 2],
            pos=[0, -gate_width / 2, 0],
            rgba=[0.0, 0.0, 0.0, 1.0],
            contype=0,
            conaffinity=0,
        )
        self.root_element.worldbody.add(
            "geom",
            type="cylinder",
            name="start_pole_right",
            size=[pole_radius, gate_height / 2],
            pos=[0, gate_width / 2, 0],
            rgba=[0.0, 0.0, 0.0, 1.0],
            contype=0,
            conaffinity=0,
        )

        for i in range(n_gates):
            offset_factor = 1 if i % 2 == 0 else -1

            if i == n_gates - 1:
                color = [1.0, 1.0, 1.0, 1.0]
                self.finish_line_points = np.array(
                    [
                        [(i + 1) * gate_spacing, offset_factor * gate_offset],
                        [
                            (i + 1) * gate_spacing,
                            offset_factor * (gate_offset + gate_width),
                        ],
                    ]
                )
            else:
                color = [1.0, 0.0, 0.0, 1.0] if i % 2 == 0 else [0.0, 0.0, 1.0, 1.0]
            self.root_element.worldbody.add(
                "geom",
                type="cylinder",
                name=f"gate{i}_inside",
                size=[pole_radius, gate_height / 2 + 0.05],
                pos=[(i + 1) * gate_spacing, offset_factor * gate_offset, -0.05],
                rgba=color,
                contype=0,
                conaffinity=0,
            )
            self.root_element.worldbody.add(
                "geom",
                type="cylinder",
                name=f"gate{i}_outside",
                size=[pole_radius, gate_height / 2 + 0.05],
                pos=[
                    (i + 1) * gate_spacing,
                    offset_factor * (gate_offset + gate_width),
                    -0.05,
                ],
                rgba=color,
                contype=0,
                conaffinity=0,
            )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle
