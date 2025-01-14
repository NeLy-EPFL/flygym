import numpy as np
from typing import Tuple
from dm_control import mjcf
from flygym.arena import FlatTerrain, BlocksTerrain


class PathIntegrationArenaBase:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add plane at the bottom to make sure the floor is rendered correctly
        self.root_element.worldbody.add(
            "geom",
            name="floor",
            type="box",
            size=(300, 300, 1),
            pos=(0, 0, -1),
        )

        # Add marker at origin
        self.root_element.worldbody.add(
            "geom",
            name="origin_marker",
            type="sphere",
            size=(1,),
            pos=(0, 0, 5),
            rgba=(1, 0, 0, 1),
        )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle


class PathIntegrationArenaFlat(PathIntegrationArenaBase, FlatTerrain):
    """
    Flat terrain for the path integration task with some fixed camera
    configurations.
    """

    pass


class PathIntegrationArenaBlocks(PathIntegrationArenaBase, BlocksTerrain):
    """
    Blocks terrain for the path integration task with some fixed camera
    configurations.
    """

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        adj_pos = rel_pos + np.array([0, 0, 0.1])
        return adj_pos, rel_angle
