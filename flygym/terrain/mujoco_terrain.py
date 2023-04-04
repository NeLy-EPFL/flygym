import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Union, Optional
from dm_control import mjcf

from flygym.terrain.base import BaseTerrain


class FlatTerrain(BaseTerrain):
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
        Sliding, torsional, and rolling friction coefficients, by
        default (1, 0.005, 0.0001)
    """
    
    def __init__(self,
                 size: Tuple[float, float] = (50_000, 50_000),
                 friction: Tuple[float, float, float] = (100, 0.005, 0.0001)):
        self.arena = mjcf.RootElement()
        ground_size = [*size, 1]
        chequered = self.arena.asset.add('texture', type='2d',
                                         builtin='checker',
                                         width=300, height=300,
                                         rgb1=(.2, .3, .4), rgb2=(.3, .4, .5))
        grid = self.arena.asset.add('material', name='grid', texture=chequered,
                                    texrepeat=(10, 10), reflectance=0.1)
        self.arena.worldbody.add('geom', type='plane', name='ground',
                                 material=grid, size=ground_size,
                                 friction=friction)
    
    def get_spawn_position(self, rel_pos: np.ndarray, rel_angle: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle


class GappedTerrain(BaseTerrain):
    """Terrain with horizontal gaps.
    
    Attributes
    ----------
    arena : mjcf.RootElement
        The arena object that the terrain is built on.

    Parameters
    ----------
    x_range : Tuple[int, int]
        Range of the arena in the x direction (anterior-posterior axis
        of the fly) over which the block-gap pattern should span, by
        default (-10_000, 10_000)
    y_range : Tuple[int, int]
        Same as above in y, by default (-10_000, 10_000)
    friction : Tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by
        default (1, 0.005, 0.0001)
    gap_width : int
        Width of each gap, by default 200
    block_width : int
        Width of each block (piece of floor), by default 1000
    gap_depth : int
        Height of the gaps, by default 2000
    """
    
    def __init__(self,
                 x_range: Tuple[int, int] = (-10_000, 10_000),
                 y_range: Tuple[int, int] = (-10_000, 10_000),
                 friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
                 gap_width: int = 200,
                 block_width: int = 1000,
                 gap_depth: int = 2000
                 ) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.gap_width = gap_width
        self.block_width = block_width
        self.gap_depth = gap_depth
        
        # add blocks
        self.arena = mjcf.RootElement()
        block_centers = np.arange(x_range[0] + block_width / 2,
                                  x_range[1],
                                  block_width + gap_width)
        box_size = (block_width / 2,
                    (y_range[1] - y_range[0]) / 2,
                    gap_depth / 2)
        obstacle = self.arena.asset.add('material', name='obstacle',
                                        reflectance=0.1)
        for x_pos in block_centers:
            self.arena.worldbody.add('geom', type='box', size=box_size,
                                     pos=(x_pos, 0, 0), friction=friction,
                                     rgba=(0.3, 0.3, 0.3, 1), material=obstacle)
    
        # add floor underneath
        chequered = self.arena.asset.add('texture', type='2d',
                                         builtin='checker',
                                         width=300, height=300,
                                         rgb1=(.2, .3, .4), rgb2=(.3, .4, .5))
        grid = self.arena.asset.add('material', name='grid', texture=chequered,
                                    texrepeat=(10, 10), reflectance=0.1)
        ground_size = (max(self.x_range), max(self.y_range), 1)
        self.arena.worldbody.add('geom', type='plane', name='ground',
                                 pos=(0, 0, -gap_depth / 2),
                                 material=grid, size=ground_size)
    
    
    def get_spawn_position(self, rel_pos: np.ndarray, rel_angle: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
        adj_pos = rel_pos + np.array([0, 0, self.gap_depth / 2])
        return adj_pos, rel_angle


class ExtrudingBlocksTerrain(BaseTerrain):
    """Terrain formed by blocks at random heights.
    
    Attributes
    ----------
    arena : mjcf.RootElement
        The arena object that the terrain is built on.

    Parameters
    ----------
    x_range : Tuple[int, int], optional
        Range of the arena in the x direction (anterior-posterior axis
        of the fly) over which the block-gap pattern should span, by
        default (-10_000, 10_000)
    y_range : Tuple[int, int], optional
        Same as above in y, by default (-10_000, 10_000)
    friction : Tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by
        default (1, 0.005, 0.0001)
    block_size : int, optional
        The side length of the rectangular blocks forming the terrain,
        by default 1000
    height_range : Tuple[int, int], optional
        Range from which the height of the extruding blocks should be
        sampled. Only half of the blocks arranged in a diagonal pattern
        are extruded, by default (300, 300)
    rand_seed : int, optional
        Seed for generating random block heights, by default 0
    """
    
    def __init__(self,
                 x_range: Tuple[int, int] = (-10_000, 10_000),
                 y_range: Tuple[int, int] = (-10_000, 10_000),
                 friction: Tuple[float, float, float] = (1, 0.005, 0.0001),
                 block_size: int = 1000,
                 height_range: Tuple[int, int] = (300, 300),
                 rand_seed: int = 0):
        self.x_range = x_range
        self.y_range = y_range
        self.block_size = block_size
        self.height_range = height_range
        rand_state = np.random.RandomState(rand_seed)
        
        self.arena = mjcf.RootElement()
        obstacle = self.arena.asset.add('material', name='obstacle',
                                        reflectance=0.1)
        
        x_centers = np.arange(x_range[0] + block_size / 2, x_range[1],
                              block_size)
        y_centers = np.arange(y_range[0] + block_size / 2, y_range[1],
                              block_size)
        for i, x_pos in enumerate(x_centers):
            for j, y_pos in enumerate(y_centers):
                is_i_odd = i % 2 == 1
                is_j_odd = j % 2 == 1
                
                if is_i_odd != is_j_odd:
                    height = 100
                else:
                    height = 100 + rand_state.uniform(*height_range)
                
                self.arena.worldbody.add('geom', type='box',
                                         size=(block_size / 2,
                                               block_size / 2,
                                               height / 2),
                                         pos=(x_pos, y_pos, height / 2),
                                         rgba=(0.3, 0.3, 0.3, 1),
                                         material=obstacle,
                                         friction=friction)
    
    def get_spawn_position(self, rel_pos: np.ndarray, rel_angle: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
        adj_pos = rel_pos + np.array([0, 0, 100])
        return adj_pos, rel_angle


class Ball(BaseTerrain):
    """Fly tethered on a spherical threadmill.
    
    Attributes
    ----------
    arena : mjcf.RootElement
        The arena object that the terrain is built on.
    
    Parameters
    ----------
    radius : float, optional
        Radius of the ball, by default 5390.852782067457
    ball_pos : Tuple[float, float, float], optional
        (x, y, z) mounting position of the ball, by default
        (-98.67235483, -54.35809692, -5203.09506806)
    mass : float, optional
        Mass of the ball, by default 0.05456
    sliding_friction : float, optional
        Sliding friction coefficient of the ball, by default 1.3
    torsional_friction : float, optional
        Torsional friction coefficient of the ball, by default 0.005
    rolling_friction : float, optional
        Rolling friction coefficient of the ball, by default 0.0001
    """
    
    def __init__(self,
                 radius: float = 5390.852782067457,
                 ball_pos: Tuple[float, float, float] = (-98.67235483,
                                                         -54.35809692,
                                                         -5203.09506806),
                 mass: float = 0.05456,
                 sliding_friction: float = 1.3,
                 torsional_friction: float = 0.005,
                 rolling_friction: float = 0.0001
                 ):
        raise NotImplementedError
    
    def get_spawn_position(self, rel_pos: np.ndarray, rel_angle: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError