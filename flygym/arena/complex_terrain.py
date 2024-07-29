import numpy as np
from typing import Optional
from dm_control import mjcf

from .base import BaseArena


class GappedTerrain(BaseArena):
    """Terrain with horizontal gaps.

    Attributes
    ----------
    root_element : mjcf.RootElement
        The root MJCF element of the arena.
    friction : tuple[float, float, float]
        The sliding, torsional, and rolling friction coefficients of the
        ground, by default (1, 0.005, 0.0001).
    x_range : tuple[float, float]
        Range of the arena in the x direction (anterior-posterior axis of
        the fly) over which the block-gap pattern should span, by default
        (-10, 20).
    y_range : tuple[float, float]Í
        Same as above in y, by default (-10, 10)
    gap_width : float
        Width of each gap, by default 0.4
    block_width : float
        Width of each block (piece of floor), by default 1
    gap_depth : float
        Height of the gaps, by default 2

    Parameters
    ----------
    x_range : tuple[float, float], optional
        Range of the arena in the x direction (anterior-posterior axis of
        the fly) over which the block-gap pattern should span, by default
        (-10, 20)
    y_range : tuple[float, float], optional
        Same as above in y, by default (-10, 10).
    friction : tuple[float, float, float], optional
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001).
    gap_width : float, optional
        Width of each gap, by default 0.3.
    block_width : float, optional
        Width of each block (piece of floor), by default 1.
    gap_depth : float, optional
        Height of the gaps, by default 2.
    ground_alpha : float, optional
        Opacity of the ground, by default 1 (fully opaque).
    scale_bar_pos : tuple[float, float, float], optional
        If supplied, a 1 mm scale bar will be placed at this location.
    """

    def __init__(
        self,
        x_range: tuple[float, float] = (-10, 25),
        y_range: tuple[float, float] = (-20, 20),
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
        gap_width: float = 0.3,
        block_width: float = 1.0,
        gap_depth: float = 2.0,
        ground_alpha: float = 1.0,
        scale_bar_pos: Optional[tuple[float, float, float]] = None,
    ) -> None:
        super().__init__()

        self.x_range = x_range
        self.y_range = y_range
        self.friction = friction
        self.gap_width = gap_width
        self.block_width = block_width
        self.gap_depth = gap_depth

        # add blocks
        block_centers = np.arange(
            x_range[0] + block_width / 2, x_range[1], block_width + gap_width
        )
        box_size = (block_width / 2, (y_range[1] - y_range[0]) / 2, gap_depth / 2)
        for x_pos in block_centers:
            self.root_element.worldbody.add(
                "geom",
                type="box",
                name=f"ground_element_x{x_pos}",
                size=box_size,
                pos=(x_pos, 0, 0),
                friction=friction,
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )

        # add floor underneath
        ground_size = ((self.x_range[1] - self.x_range[0]) / 2, max(self.y_range), 1)
        self.root_element.worldbody.add(
            "geom",
            type="plane",
            name="ground_base",
            pos=(np.mean(x_range), 0, -gap_depth / 2),
            rgba=(0.3, 0.3, 0.3, ground_alpha),
            size=ground_size,
        )

        if scale_bar_pos:
            self.root_element.worldbody.add(
                "geom",
                type="cylinder",
                size=(0.05, 0.5),
                pos=scale_bar_pos + np.array([0, 0, self.gap_depth / 2]),
                rgba=(0, 0, 0, 1),
                euler=(0, np.pi / 2, 0),
            )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        adj_pos = rel_pos + np.array([0, 0, self.gap_depth / 2])
        return adj_pos, rel_angle


class BlocksTerrain(BaseArena):
    """Terrain formed by blocks at random heights.

    Attributes
    ----------
    root_element : mjcf.RootElement
        The root MJCF element of the arena.
    x_range : tuple[float, float], optional
        Range of the arena in the x direction (anterior-posterior axis of
        the fly) over which the block-gap pattern should span, by default
        (-10, 20).
    y_range : tuple[float, float], optional
        Same as above in y, by default (-10, 10).
    friction : tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001).
    block_size : float
        The side length of the rectangular blocks forming the terrain in
        mm, by default 1.3.
    height_range : tuple[float, float]
        Range from which the height of the extruding blocks should be
        sampled. Only half of the blocks arranged in a diagonal pattern are
        extruded, by default (0.35, 0.35).

    Parameters
    ----------
    x_range : tuple[float, float], optional
        Range of the arena in the x direction (anterior-posterior axis of
        the fly) over which the block-gap pattern should span, by default
        (-10, 20).
    y_range : tuple[float, float], optional
        Same as above in y, by default (-10, 10).
    friction : tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001).
    block_size : float, optional
        The side length of the rectangular blocks forming the terrain in
        mm, by default 1.3.
    height_range : tuple[float, float], optional
        Range from which the height of the extruding blocks should be
        sampled. Only half of the blocks arranged in a diagonal pattern are
        extruded, by default (0.35, 0.35).
    ground_alpha : float, optional
        Opacity of the ground, by default 1 (fully opaque).
    rand_seed : int, optional
        Seed for generating random block heights, by default 0.
    scale_bar_pos : tuple[float, float, float], optional
        If supplied, a 1 mm scale bar will be placed at this location.
    """

    def __init__(
        self,
        x_range: tuple[float, float] = (-10, 25),
        y_range: tuple[float, float] = (-20, 20),
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
        block_size: float = 1.3,
        height_range: tuple[float, float] = (0.35, 0.35),
        ground_alpha: float = 1.0,
        rand_seed: int = 0,
        scale_bar_pos: Optional[tuple[float, float, float]] = None,
    ):
        super().__init__()

        self.x_range = x_range
        self.y_range = y_range
        self.friction = friction
        self.block_size = block_size
        self.height_range = height_range
        rand_state = np.random.RandomState(rand_seed)

        x_centers = np.arange(x_range[0] + block_size / 2, x_range[1], block_size)
        y_centers = np.arange(y_range[0] + block_size / 2, y_range[1], block_size)
        for i, x_pos in enumerate(x_centers):
            for j, y_pos in enumerate(y_centers):
                is_i_odd = i % 2 == 1
                is_j_odd = j % 2 == 1

                if is_i_odd != is_j_odd:
                    height = 0.1
                else:
                    height = 0.1 + rand_state.uniform(*height_range)

                self.root_element.worldbody.add(
                    "geom",
                    type="box",
                    name=f"ground_element_x{x_pos}_y{y_pos}",
                    size=(
                        block_size / 2 + 0.1 * block_size / 2,
                        block_size / 2 + 0.1 * block_size / 2,
                        height / 2 + block_size / 2,
                    ),
                    pos=(
                        x_pos,
                        y_pos,
                        height / 2 - block_size / 2,
                    ),
                    rgba=(0.3, 0.3, 0.3, ground_alpha),
                    friction=friction,
                )

        if scale_bar_pos:
            self.root_element.worldbody.add(
                "geom",
                type="cylinder",
                size=(0.1, 0.5),
                pos=scale_bar_pos + np.array([0, 0, 0.1]),
                rgba=(0, 0, 0, 1),
                euler=(0, np.pi / 2, 0),
            )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        adj_pos = rel_pos + np.array([0, 0, 0.1])
        return adj_pos, rel_angle


class MixedTerrain(BaseArena):
    """A mixture of flat, blocks, and gaps terrains.

    Attributes
    ----------
    root_element : mjcf.RootElement
        The root MJCF element of the arena.
    friction : tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001).

    Parameters
    ----------
    friction : tuple[float, float, float], optional
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001)
    gap_width : float, optional
        Width of each gap, by default 0.4.
    gapped_block_width : float, optional
        Width of each block (piece of floor), by default 1.
    gap_depth : float, optional
        Height of the gaps, by default 2.
    block_size : float, optional
        The side length of the rectangular blocks forming the terrain in
        mm, by default 1.3.
    height_range : tuple[float, float], optional
        Range from which the height of the extruding blocks should be
        sampled. Only half of the blocks arranged in a diagonal pattern are
        extruded, by default (0.35, 0.35).
    ground_alpha : float, optional
        Opacity of the ground, by default 1 (fully opaque).
    rand_seed : int, optional
        Seed for generating random block heights, by default 0.
    scale_bar_pos : tuple[float, float, float], optional
        If supplied, a 1 mm scale bar will be placed at this location.
    """

    def __init__(
        self,
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
        gap_width: float = 0.3,
        gapped_block_width: float = 1.0,
        gap_depth: float = 2,
        block_size: float = 1.3,
        height_range: tuple[float, float] = (0.35, 0.35),
        ground_alpha: float = 1.0,
        rand_seed: int = 0,
        scale_bar_pos: Optional[tuple[float, float, float]] = None,
    ):
        super().__init__()

        self.friction = friction
        y_range = (-20, 20)
        rand_state = np.random.RandomState(rand_seed)

        self._height_expected_value = np.mean([*height_range])

        # 3 repetitions, each consisting of a block part, 2 gaps, and a flat part
        for x_range in [(-4, 5), (5, 14), (14, 23)]:
            # block part
            x_centers = np.arange(
                x_range[0] + block_size / 2, x_range[0] + block_size * 3, block_size
            )
            y_centers = np.arange(y_range[0] + block_size / 2, y_range[1], block_size)
            for i, x_pos in enumerate(x_centers):
                for j, y_pos in enumerate(y_centers):
                    is_i_odd = i % 2 == 1
                    is_j_odd = j % 2 == 1

                    if is_i_odd != is_j_odd:
                        height = 0.1
                    else:
                        height = 0.1 + rand_state.uniform(*height_range)

                    box_size = (
                        block_size / 2 + 0.1 * block_size / 2,
                        block_size / 2 + 0.1 * block_size / 2,
                        height / 2 + block_size / 2,
                    )
                    box_pos = (
                        x_pos,
                        y_pos,
                        height / 2 - block_size / 2 - self._height_expected_value - 0.1,
                    )
                    self.root_element.worldbody.add(
                        "geom",
                        type="box",
                        name=f"ground_element_block_x{x_pos}_x{y_pos}",
                        size=box_size,
                        pos=box_pos,
                        rgba=(0.3, 0.3, 0.3, ground_alpha),
                        friction=friction,
                    )

            # gap part
            curr_x_pos = x_range[0] + block_size * 3
            arena_width = y_range[1] - y_range[0]
            # first flat bit
            self.root_element.worldbody.add(
                "geom",
                type="box",
                name=f"ground_element_gap_x{curr_x_pos}",
                size=(gapped_block_width / 4, arena_width / 2, gap_depth / 2),
                pos=(curr_x_pos + gapped_block_width / 4, 0, -gap_depth / 2),
                friction=friction,
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )
            # second flat bit
            curr_x_pos += gapped_block_width / 2 + gap_width
            self.root_element.worldbody.add(
                "geom",
                type="box",
                name=f"ground_element_gap_x{curr_x_pos}",
                size=(gapped_block_width / 2, arena_width / 2, gap_depth / 2),
                pos=(curr_x_pos + gapped_block_width / 2, 0, -gap_depth / 2),
                friction=friction,
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )

            # flat part
            curr_x_pos += gapped_block_width + gap_width
            remaining_space = x_range[1] - curr_x_pos
            assert remaining_space > 0, "remaining space for flat part is negative"
            self.root_element.worldbody.add(
                "geom",
                type="box",
                name=f"ground_element_flat_x{curr_x_pos}",
                size=(remaining_space / 2, arena_width / 2, gap_depth / 2),
                pos=(curr_x_pos + remaining_space / 2, 0, -gap_depth / 2),
                friction=friction,
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )

            # add floor underneath
            ground_size = ((x_range[1] - x_range[0]) / 2, max(y_range), 1)
            self.root_element.worldbody.add(
                "geom",
                type="plane",
                name=f"ground_base_{x_range[0]}",
                pos=(np.mean(x_range), 0, -gap_depth / 2),
                rgba=(0.3, 0.3, 0.3, ground_alpha),
                size=ground_size,
            )

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
    ) -> tuple[np.ndarray, np.ndarray]:
        adj_pos = rel_pos + np.array([0, 0, -1 * self._height_expected_value])
        return adj_pos, rel_angle
