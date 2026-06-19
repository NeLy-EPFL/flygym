from typing import override

import numpy as np
import dm_control.mjcf as mjcf

from flygym.compose.fly import BaseFly
from flygym.utils.math import Vec3, Rotation3D
from flygym.compose.world.base_world import (
    _GroundContactMixin,
    BaseWorld,
    _format_name_number,
)

__all__ = [
    "GappedTerrainWorld",
    "BlocksTerrainWorld",
    "MixedTerrainWorld",
    "TetheredWorld",
]


class _ComplexTerrainWorld(_GroundContactMixin, BaseWorld):
    """Base for terrain worlds built from explicit ground geoms.

    Subclasses should call ``super().__init__(name)`` and then use
    ``_add_ground_box`` / ``_add_ground_plane`` to populate ``ground_geoms``.
    Contact-pair generation is handled automatically by ``_GroundContactMixin``
    when ``add_fly`` is called.

    !!! warning

        Per-leg ground contact sensors are **not** added for multi-geom worlds as is,
        even when ``add_fly(..., add_ground_contact_sensors=True)`` is requested.
        This is because it is ambiguous which of the many ground geoms a sensor should
        be anchored to for each leg. Calling ``Simulation.get_ground_contact_info()`` on
        such a world will raise a ``TypeError`` at runtime.

        If you need per-leg contact sensors, implement them explicitly in a custom
        ``_attach_fly_mjcf`` override that hard-codes which geom each sensor monitors.
        Alternatively, use ``Simulation.get_bodysegment_contact_forces()`` to query
        contact forces from the raw contact list.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.ground_geoms = []

    def _add_ground_box(
        self,
        name: str,
        size: tuple[float, float, float],
        pos: tuple[float, float, float],
        *,
        rgba: tuple[float, float, float, float],
    ) -> mjcf.Element:
        geom = self.mjcf_root.worldbody.add(
            "geom",
            type="box",
            name=name,
            size=size,
            pos=pos,
            rgba=rgba,
            contype=0,
            conaffinity=0,
        )
        self.ground_geoms.append(geom)
        return geom

    def _add_ground_plane(
        self,
        name: str,
        size: tuple[float, float, float],
        pos: tuple[float, float, float],
        *,
        rgba: tuple[float, float, float, float],
    ) -> mjcf.Element:
        geom = self.mjcf_root.worldbody.add(
            "geom",
            type="plane",
            name=name,
            size=size,
            pos=pos,
            rgba=rgba,
            contype=0,
            conaffinity=0,
        )
        self.ground_geoms.append(geom)
        return geom


class GappedTerrainWorld(_ComplexTerrainWorld):
    """World with alternating floor blocks and transverse gaps."""

    def __init__(
        self,
        name: str = "gapped_terrain_world",
        *,
        x_range: tuple[float, float] = (-10, 25),
        y_range: tuple[float, float] = (-20, 20),
        gap_width: float = 0.3,
        block_width: float = 1.0,
        gap_depth: float = 2.0,
        ground_alpha: float = 1.0,
    ) -> None:

        super().__init__(name=name)
        y_halfwidth = (y_range[1] - y_range[0]) / 2
        block_centers = np.arange(
            x_range[0] + block_width / 2,
            x_range[1],
            block_width + gap_width,
        )
        for x_pos in block_centers:
            self._add_ground_box(
                name=f"ground_block_x{_format_name_number(x_pos)}",
                size=(block_width / 2, y_halfwidth, gap_depth / 2),
                pos=(x_pos, 0, -gap_depth / 2),
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )
        self._add_ground_plane(
            name="ground_base",
            size=((x_range[1] - x_range[0]) / 2, y_halfwidth, 1),
            pos=(np.mean(x_range), 0, -gap_depth),
            rgba=(0.3, 0.3, 0.3, ground_alpha),
        )


class BlocksTerrainWorld(_ComplexTerrainWorld):
    """World tiled by blocks with alternating heights."""

    def __init__(
        self,
        name: str = "blocks_terrain_world",
        *,
        x_range: tuple[float, float] = (-10, 25),
        y_range: tuple[float, float] = (-20, 20),
        block_size: float = 1.3,
        height_range: tuple[float, float] = (0.35, 0.35),
        ground_alpha: float = 1.0,
        rand_seed: int = 0,
    ) -> None:
        super().__init__(name=name)
        rand_state = np.random.RandomState(rand_seed)
        x_centers = np.arange(x_range[0] + block_size / 2, x_range[1], block_size)
        y_centers = np.arange(y_range[0] + block_size / 2, y_range[1], block_size)
        for i, x_pos in enumerate(x_centers):
            for j, y_pos in enumerate(y_centers):
                if (i % 2 == 1) != (j % 2 == 1):
                    height = 0.1
                else:
                    height = 0.1 + rand_state.uniform(*height_range)
                self._add_ground_box(
                    name=(
                        "ground_block_"
                        f"x{_format_name_number(x_pos)}_y{_format_name_number(y_pos)}"
                    ),
                    size=(0.55 * block_size, 0.55 * block_size, height / 2),
                    pos=(x_pos, y_pos, height / 2),
                    rgba=(0.3, 0.3, 0.3, ground_alpha),
                )


class MixedTerrainWorld(_ComplexTerrainWorld):
    """World with repeated blocks, gaps, and flat floor sections."""

    def __init__(
        self,
        name: str = "mixed_terrain_world",
        *,
        x_ranges: tuple[tuple[float, float], ...] = ((-4, 5), (5, 14), (14, 23)),
        y_range: tuple[float, float] = (-20, 20),
        gap_width: float = 0.3,
        gapped_block_width: float = 1.0,
        gap_depth: float = 2.0,
        block_size: float = 1.3,
        height_range: tuple[float, float] = (0.35, 0.35),
        ground_alpha: float = 1.0,
        rand_seed: int = 0,
    ) -> None:
        super().__init__(name=name)
        y_halfwidth = (y_range[1] - y_range[0]) / 2
        rand_state = np.random.RandomState(rand_seed)
        height_expected_value = np.mean(height_range)

        for range_idx, x_range in enumerate(x_ranges):
            x_centers = np.arange(
                x_range[0] + block_size / 2,
                x_range[0] + block_size * 3,
                block_size,
            )
            y_centers = np.arange(y_range[0] + block_size / 2, y_range[1], block_size)
            for i, x_pos in enumerate(x_centers):
                for j, y_pos in enumerate(y_centers):
                    if (i % 2 == 1) != (j % 2 == 1):
                        height = 0.1
                    else:
                        height = 0.1 + rand_state.uniform(*height_range)
                    self._add_ground_box(
                        name=(
                            f"ground_mixed_block{range_idx}_"
                            f"x{_format_name_number(x_pos)}_"
                            f"y{_format_name_number(y_pos)}"
                        ),
                        size=(
                            0.55 * block_size,
                            0.55 * block_size,
                            height / 2 + block_size / 2,
                        ),
                        pos=(
                            x_pos,
                            y_pos,
                            height / 2 - block_size / 2 - height_expected_value - 0.1,
                        ),
                        rgba=(0.3, 0.3, 0.3, ground_alpha),
                    )

            curr_x_pos = x_range[0] + block_size * 3
            self._add_ground_box(
                name=f"ground_gap_pre{range_idx}",
                size=(gapped_block_width / 4, y_halfwidth, gap_depth / 2),
                pos=(curr_x_pos + gapped_block_width / 4, 0, -gap_depth / 2),
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )
            curr_x_pos += gapped_block_width / 2 + gap_width
            self._add_ground_box(
                name=f"ground_gap_post{range_idx}",
                size=(gapped_block_width / 2, y_halfwidth, gap_depth / 2),
                pos=(curr_x_pos + gapped_block_width / 2, 0, -gap_depth / 2),
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )
            curr_x_pos += gapped_block_width + gap_width
            remaining_space = x_range[1] - curr_x_pos
            if remaining_space <= 0:
                raise ValueError(
                    "Each mixed-terrain x-range must leave positive flat-floor "
                    "space after the block and gap sections."
                )
            self._add_ground_box(
                name=f"ground_flat{range_idx}",
                size=(remaining_space / 2, y_halfwidth, gap_depth / 2),
                pos=(curr_x_pos + remaining_space / 2, 0, -gap_depth / 2),
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )
            self._add_ground_plane(
                name=f"ground_base{range_idx}",
                size=((x_range[1] - x_range[0]) / 2, y_halfwidth, 1),
                pos=(np.mean(x_range), 0, -gap_depth / 2),
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )


class TetheredWorld(BaseWorld):
    """World where the fly body is fixed in space via a weld constraint.

    The fly's appendages (legs, wings, etc.) can still move. Useful for motor control
    experiments without locomotion.

    Args:
        name: Name of the world.
    """

    @override
    def __init__(self, name: str = "tethered_world") -> None:
        super().__init__(name=name)

    @override
    def _attach_fly_mjcf(
        self, fly: BaseFly, spawn_position: Vec3, spawn_rotation: Rotation3D
    ) -> mjcf.Element:
        spawn_site = self.mjcf_root.worldbody.add(
            "site", name=fly.name, pos=spawn_position, **spawn_rotation.as_kwargs()
        )
        spawn_site.attach(fly.mjcf_root)
        return {}
