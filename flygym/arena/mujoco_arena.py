import numpy as np
import scipy.stats
from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple, List, Dict, Union, Optional, Any, Callable
from dm_control import mjcf

from flygym.arena.base import BaseArena
from flygym.util.data import color_cycle_rgb


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
        gap_width: float = 0.5,
        block_width: float = 1.0,
        gap_depth: float = 2,
        ground_alpha: float = 0.8,
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
        for x_pos in block_centers:
            self.root_element.worldbody.add(
                "geom",
                type="box",
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
            name="ground",
            pos=(np.mean(x_range), 0, -gap_depth / 2),
            rgba=(0.3, 0.3, 0.3, ground_alpha),
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
        block_size: float = 1.3,
        height_range: Tuple[float, float] = (0.45, 0.45),
        ground_alpha: float = 0.8,
        rand_seed: int = 0,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.friction = friction
        self.block_size = block_size
        self.height_range = height_range
        rand_state = np.random.RandomState(rand_seed)

        self.root_element = mjcf.RootElement()

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

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        adj_pos = rel_pos + np.array([0, 0, 0.1])
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
        gap_width: float = 0.5,
        block_width: float = 1.0,
        gap_depth: float = 2,
        block_size: float = 1.3,
        height_range: Tuple[float, float] = (0.45, 0.45),
        ground_alpha: float = 0.8,
        rand_seed: int = 0,
    ):
        self.root_element = mjcf.RootElement()
        self.friction = friction
        y_range = (-10, 10)
        rand_state = np.random.RandomState(rand_seed)

        self.height_expected_value = np.mean([*height_range])

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
                        height / 2 - block_size / 2 - self.height_expected_value - 0.1,
                    )
                    self.root_element.worldbody.add(
                        "geom",
                        type="box",
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
                size=(block_width / 4, arena_width / 2, gap_depth / 2),
                pos=(curr_x_pos + block_width / 4, 0, -gap_depth / 2),
                friction=friction,
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )
            # second flat bit
            curr_x_pos += block_width / 2 + gap_width
            self.root_element.worldbody.add(
                "geom",
                type="box",
                size=(block_width / 2, arena_width / 2, gap_depth / 2),
                pos=(curr_x_pos + block_width / 2, 0, -gap_depth / 2),
                friction=friction,
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )

            # flat part
            curr_x_pos += block_width + gap_width
            remaining_space = x_range[1] - curr_x_pos
            assert remaining_space > 0, "remaining space for flat part is negative"
            self.root_element.worldbody.add(
                "geom",
                type="box",
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
                name=f"ground_{x_range[0]}",
                pos=(np.mean(x_range), 0, -gap_depth / 2),
                rgba=(0.3, 0.3, 0.3, ground_alpha),
                size=ground_size,
            )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        adj_pos = rel_pos + np.array([0, 0, -1 * self.height_expected_value])
        return adj_pos, rel_angle


class Tethered(BaseArena):
    """Fly tethered in the air"""

    def __init__(self, *args: List, **kwargs: Dict):
        """Create a new terrain object.

        Attributes
        ----------
        arena : Any
            The arena object that the terrain is built on. Exactly what it
            is depends on the physics simulator.
        """
        self.root_element = mjcf.RootElement()

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle

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
        spawn_site.attach(entity).add(
            "joint", name="prismatic_support_1", limited=True, range=(0, 1e-10)
        )


class Ball(Tethered):
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
        self.root_element = mjcf.RootElement()

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


class OdorArena(BaseArena):
    """Flat terrain with an odor source.

    Attributes
    ----------
    arena : mjcf.RootElement
        The arena object that the terrain is built on.
    num_sensors : int = 4
        2x antennae + 2x max. palps

    Parameters
    ----------
    size : Tuple[float, float]
        The size of the terrain in (x, y) dimensions.
    friction : Tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001)
    odor_source : np.ndarray
        The position of the odor source in (x, y, z) coordinates. The shape
        of the array is (n_sources, 3).
    peak_intensity : np.ndarray
        The peak intensity of the odor source. The shape of the array is
        (n_sources, n_dimensions). Note that the odor intensity can be
        multidimensional.
    diffuse_func : Callable
        The function that, given a distance from the odor source, returns
        the relative intensity of the odor. By default, this is a inverse
        square relationship.
    """

    num_sensors = 4

    def __init__(
        self,
        size: Tuple[float, float] = (50, 50),
        friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
        odor_source: np.ndarray = np.array([[10, 0, 0]]),
        peak_intensity: np.ndarray = np.array([[1]]),
        diffuse_func: Callable = lambda x: x**-2,
        marker_colors: Optional[List[Tuple[float, float, float, float]]] = None,
        marker_size: float = 0.1,
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

        # Add markers at the odor sources
        if marker_colors is None:
            marker_colors = []
            num_odor_sources = self.odor_source.shape[0]
            for i in range(num_odor_sources):
                rgb = np.array(color_cycle_rgb[i % num_odor_sources]) / 255
                rgba = (*rgb, 1)
                marker_colors.append(rgba)
        for i, (pos, rgba) in enumerate(zip(self.odor_source, marker_colors)):
            marker_body = self.root_element.worldbody.add(
                "body", name=f"odor_source_marker_{i}", pos=pos, mocap=True
            )
            marker_body.add(
                "geom", type="capsule", size=(marker_size, marker_size), rgba=rgba
            )

        # Reshape odor source and peak intensity arrays to simplify future claculations
        _odor_source_repeated = self.odor_source[:, np.newaxis, np.newaxis, :]
        _odor_source_repeated = np.repeat(_odor_source_repeated, self.odor_dim, axis=1)
        _odor_source_repeated = np.repeat(
            _odor_source_repeated, self.num_sensors, axis=2
        )
        self._odor_source_repeated = _odor_source_repeated
        _peak_intensity_repeated = self.peak_odor_intensity[:, :, np.newaxis]
        _peak_intensity_repeated = np.repeat(
            _peak_intensity_repeated, self.num_sensors, axis=2
        )
        self._peak_intensity_repeated = _peak_intensity_repeated

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle

    def get_olfaction(self, antennae_pos: np.ndarray) -> np.ndarray:
        """
        Notes
        -----
        w = 4: number of sensors (2x antennae + 2x max. palps)
        3: spatial dimensionality
        k: data dimensionality
        n: number of odor sources

        Input - odor source position: [n, 3]
        Input - sensor positions: [w, 3]
        Input - peak intensity: [n, k]
        Input - difusion function: f(dist)

        Reshape sources to S = [n, k*, w*, 3] (* means repeated)
        Reshape sensor position to A = [n*, k*, w, 3] (* means repeated)
        Subtract, getting an Delta = [n, k, w, 3] array of rel difference
        Calculate Euclidean disctance: D = [n, k, w]

        Apply pre-integrated difusion function: S = f(D) -> [n, k, w]
        Reshape peak intensities to P = [n, k, w*]
        Apply scaling: I = P * S -> [n, k, w] element wise

        Output - Sum over the first axis: [k, w]
        """
        antennae_pos_repeated = antennae_pos[np.newaxis, np.newaxis, :, :]
        dist_3d = antennae_pos_repeated - self._odor_source_repeated  # (n, k, w, 3)
        dist_euc = np.linalg.norm(dist_3d, axis=3)  # (n, k, w)
        scaling = self.diffuse_func(dist_euc)  # (n, k, w)
        intensity = self._peak_intensity_repeated * scaling  # (n, k, w)
        return intensity.sum(axis=0)  # (k, w)
