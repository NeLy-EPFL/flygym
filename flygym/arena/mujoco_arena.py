import numpy as np
import scipy.stats
from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple, List, Dict, Union, Optional, Any, Callable
from dm_control import mjcf

from flygym.arena.base import BaseArena


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
    ):
        self.root_element = mjcf.RootElement()
        ground_size = [*size, 1]
        chequered = self.root_element.asset.add(
            "texture",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=(0.2, 0.3, 0.4),
            rgb2=(0.3, 0.4, 0.5),
        )
        grid = self.root_element.asset.add(
            "material",
            name="grid",
            texture=chequered,
            texrepeat=(10, 10),
            reflectance=0.1,
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

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle


class GappedTerrain(BaseArena):
    """Terrain with horizontal gaps.

    Attributes
    ----------
    arena : mjcf.RootElement
        The arena object that the terrain is built on.

    Parameters
    ----------
    x_range : Tuple[float, float]
        Range of the arena in the x direction (anterior-posterior axis of
        the fly) over which the block-gap pattern should span, by default
        (-10, 20)
    y_range : Tuple[float, float]
        Same as above in y, by default (-10, 10)
    friction : Tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001)
    gap_width : float
        Width of each gap, by default 0.2
    block_width : float
        Width of each block (piece of floor), by default 1
    gap_depth : float
        Height of the gaps, by default 2
    """

    def __init__(
        self,
        x_range: Tuple[float, float] = (-10, 20),
        y_range: Tuple[float, float] = (-10, 10),
        friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
        gap_width: float = 0.2,
        block_width: float = 1,
        gap_depth: float = 2,
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.friction = friction
        self.gap_width = gap_width
        self.block_width = block_width
        self.gap_depth = gap_depth

        # add blocks
        self.root_element = mjcf.RootElement()
        block_centers = np.arange(
            x_range[0] + block_width / 2, x_range[1], block_width + gap_width
        )
        box_size = (block_width / 2, (y_range[1] - y_range[0]) / 2, gap_depth / 2)
        obstacle = self.root_element.asset.add(
            "material", name="obstacle", reflectance=0.1
        )
        for x_pos in block_centers:
            self.root_element.worldbody.add(
                "geom",
                type="box",
                size=box_size,
                pos=(x_pos, 0, 0),
                friction=friction,
                rgba=(0.3, 0.3, 0.3, 1),
                material=obstacle,
            )

        # add floor underneath
        chequered = self.root_element.asset.add(
            "texture",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=(0.2, 0.3, 0.4),
            rgb2=(0.3, 0.4, 0.5),
        )
        grid = self.root_element.asset.add(
            "material",
            name="grid",
            texture=chequered,
            texrepeat=(10, 10),
            reflectance=0.1,
        )
        ground_size = ((self.x_range[1] - self.x_range[0]) / 2, max(self.y_range), 1)
        self.root_element.worldbody.add(
            "geom",
            type="plane",
            name="ground",
            pos=(np.mean(x_range), 0, -gap_depth / 2),
            material=grid,
            size=ground_size,
        )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        adj_pos = rel_pos + np.array([0, 0, self.gap_depth / 2])
        return adj_pos, rel_angle


class BlocksTerrain(BaseArena):
    """Terrain formed by blocks at random heights.

    Attributes
    ----------
    arena : mjcf.RootElement
        The arena object that the terrain is built on.

    Parameters
    ----------
    x_range : Tuple[float, float], optional
        Range of the arena in the x direction (anterior-posterior axis of
        the fly) over which the block-gap pattern should span, by default
        (-10, 20)
    y_range : Tuple[float, float], optional
        Same as above in y, by default (-10, 10)
    friction : Tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001)
    block_size : float, optional
        The side length of the rectangular blocks forming the terrain, by
        default 1
    height_range : Tuple[float, float], optional
        Range from which the height of the extruding blocks should be
        sampled. Only half of the blocks arranged in a diagonal pattern are
        extruded, by default (0.3, 0.3)
    rand_seed : int, optional
        Seed for generating random block heights, by default 0
    """

    def __init__(
        self,
        x_range: Tuple[float, float] = (-10, 20),
        y_range: Tuple[float, float] = (-10, 10),
        friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
        block_size: float = 1,
        height_range: Tuple[float, float] = (0.3, 0.3),
        rand_seed: int = 0,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.friction = friction
        self.block_size = block_size
        self.height_range = height_range
        rand_state = np.random.RandomState(rand_seed)

        self.root_element = mjcf.RootElement()
        obstacle = self.root_element.asset.add(
            "material", name="obstacle", reflectance=0.1
        )

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
                    size=(block_size / 2, block_size / 2, height / 2),
                    pos=(x_pos, y_pos, height / 2),
                    rgba=(0.3, 0.3, 0.3, 1),
                    material=obstacle,
                    friction=friction,
                )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        adj_pos = rel_pos + np.array([0, 0, 100])
        return adj_pos, rel_angle


class MixedTerrain(BaseArena):
    """A mixture of flat, blocks, and gaps terrains.

    Parameters
    ----------
    friction : Tuple[float, float, float], optional
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001)
    gap_width : float
        Width of each gap, by default 0.2
    block_width : float
        Width of each block (piece of floor), by default 1
    gap_depth : float
        Height of the gaps, by default 2
    block_size : float, optional
        The side length of the rectangular blocks forming the terrain, by
        default 1
    height_range : Tuple[float, float], optional
        Range from which the height of the extruding blocks should be
        sampled. Only half of the blocks arranged in a diagonal pattern are
        extruded, by default (0.3, 0.3)
    rand_seed : int, optional
        Seed for generating random block heights, by default 0
    """

    def __init__(
        self,
        friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
        gap_width: float = 0.2,
        block_width: float = 1,
        gap_depth: float = 2,
        block_size: float = 1,
        height_range: Tuple[float, float] = (0.3, 0.3),
        rand_seed: int = 0,
    ):
        self.root_element = mjcf.RootElement()
        self.friction = friction
        obstacle = self.root_element.asset.add(
            "material", name="obstacle", reflectance=0.1
        )
        chequered = self.root_element.asset.add(
            "texture",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=(0.2, 0.3, 0.4),
            rgb2=(0.3, 0.4, 0.5),
        )
        grid = self.root_element.asset.add(
            "material",
            name="grid",
            texture=chequered,
            texrepeat=(10, 10),
            reflectance=0.1,
        )
        y_range = (-10, 10)
        rand_state = np.random.RandomState(rand_seed)

        # Extruding blocks near origin
        for x_range in [(-2, 2), (6, 8), (12, 14)]:
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
                        size=(block_size / 2, block_size / 2, height / 2),
                        pos=(x_pos, y_pos, height / 2 - 0.05),
                        rgba=(0.3, 0.3, 0.3, 1),
                        material=obstacle,
                        friction=friction,
                    )

        # Then gaps
        for x_range in [(2, 4), (8, 10), (14, 16)]:
            block_centers = np.arange(
                x_range[0] + block_width / 2, x_range[1], block_width + gap_width
            )
            box_size = (block_width / 2, (y_range[1] - y_range[0]) / 2, gap_depth / 2)
            for x_pos in block_centers:
                self.root_element.worldbody.add(
                    "geom",
                    type="box",
                    size=box_size,
                    pos=(x_pos, 0, -gap_depth / 2),
                    friction=friction,
                    rgba=(0.3, 0.3, 0.3, 1),
                    material=obstacle,
                )

            # add floor underneath
            ground_size = ((x_range[1] - x_range[0]) / 2, max(y_range), 1)
            self.root_element.worldbody.add(
                "geom",
                type="plane",
                name=f"ground_{x_range[0]}",
                pos=(np.mean(x_range), 0, -gap_depth / 2),
                material=grid,
                size=ground_size,
            )

        # Finally, flat areas
        for x_range in [
            (-4, -2),
            (4, 6),
            (10, 12),
            (10, 18),
        ]:
            self.root_element.worldbody.add(
                "geom",
                type="box",
                size=(2 / 2, 20 / 2, 0.001),
                pos=(np.mean(x_range), 0, -0.001),
                friction=friction,
                rgba=(0.3, 0.3, 0.3, 1),
                material=obstacle,
            )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        adj_pos = rel_pos + np.array([0, 0, 100])
        return adj_pos, rel_angle


class Ball(BaseArena):
    """Fly tethered on a spherical threadmill.

    Attributes
    ----------
    arena : mjcf.RootElement
        The arena object that the terrain is built on.

    Parameters
    ----------
    radius : float, optional
        Radius of the ball, by default 5.390852782067457
    ball_pos : Tuple[float, float, float], optional
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
        ball_pos: Tuple[float, float, float] = (
            -0.09867235483,
            -0.05435809692,
            -5.20309506806,
        ),
        mass: float = 0.05456,
        sliding_friction: float = 1.3,
        torsional_friction: float = 0.005,
        rolling_friction: float = 0.0001,
    ):
        raise NotImplementedError

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class OdorArena(BaseArena):
    """Flat terrain with an odor source.

    Attributes
    ----------
    arena : mjcf.RootElement
        The arena object that the terrain is built on.

    Parameters
    ----------
    size : Tuple[float, float]
        The size of the terrain in (x, y) dimensions.
    friction : Tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001)
    """

    def __init__(
        self,
        size: Tuple[float, float] = (50, 50),
        friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
        odor_source: np.ndarray = np.array([[10, 0, 0]]),
        peak_intensity: np.ndarray = np.array([[1]]),
        diffuse_func: Callable = lambda x: (x) ** -2,
    ):
        self.root_element = mjcf.RootElement()
        ground_size = [*size, 1]
        chequered = self.root_element.asset.add(
            "texture",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=(0.2, 0.3, 0.4),
            rgb2=(0.3, 0.4, 0.5),
        )
        grid = self.root_element.asset.add(
            "material",
            name="grid",
            texture=chequered,
            texrepeat=(10, 10),
            reflectance=0.1,
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
        self.odor_source = np.array(odor_source)
        self.peak_odor_intensity = np.array(peak_intensity)
        self.num_odor_sources = self.odor_source.shape[0]
        if self.odor_source.shape[0] != self.peak_odor_intensity.shape[0]:
            raise ValueError(
                "Number of odor source locations and peak intensities must match."
            )
        self.odor_dim = self.peak_odor_intensity.shape[1]
        self.diffuse_func = diffuse_func

        # Reshape odor source and peak intensity arrays to simplify future claculations
        _odor_source_repeated = self.odor_source[:, np.newaxis, np.newaxis, :]
        _odor_source_repeated = np.repeat(_odor_source_repeated, self.odor_dim, axis=1)
        _odor_source_repeated = np.repeat(_odor_source_repeated, 2, axis=2)
        self._odor_source_repeated = _odor_source_repeated
        _peak_intensity_repeated = self.peak_odor_intensity[:, :, np.newaxis]
        _peak_intensity_repeated = np.repeat(_peak_intensity_repeated, 2, axis=2)
        self._peak_intensity_repeated = _peak_intensity_repeated

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle

    def get_olfaction(self, antennae_pos: np.ndarray) -> np.ndarray:
        """
        Notes
        -----
        2: number of antennae
        3: spatial dimensionality
        k: data dimensionality
        n: number of odor sources

        Input - odor source position: [n, 3]
        Input - antennae position: [2, 3]
        Input - peak intensity: [n, k]
        Input - difusion function: f(dist)

        Reshape sources to S = [n, k*, 2*, 3] (* means repeated)
        Reshape antennae position to A = [n*, k*, 2, 3] (* means repeated)
        Subtract, getting an Delta = [n, k, 2, 3] array of rel difference
        Calculate Euclidean disctance: D = [n, k, 2]

        Apply pre-integrated difusion function: S = f(D) -> [n, k, 2]
        Reshape peak intensities to P = [n, k, 2*]
        Apply scaling: I = P * S -> [n, k, 2] element wise

        Output - Sum over the first axis: [k, 2]
        """
        antennae_pos_repeated = antennae_pos[np.newaxis, np.newaxis, :, :]
        dist_3d = antennae_pos_repeated - self._odor_source_repeated  # (n, k, 2, 3)
        dist_euc = np.linalg.norm(dist_3d, axis=3)  # (n, k, 2)
        scaling = self.diffuse_func(dist_euc)  # (n, k, 2)
        intensity = self._peak_intensity_repeated * scaling  # (n, k, 2)
        return intensity.sum(axis=0)  # (k, 2)
