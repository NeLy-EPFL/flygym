import numpy as np

from flygym import Fly, SingleFlySimulation, disable_rendering
from flygym.arena import GappedTerrain, BlocksTerrain, MixedTerrain


np.random.seed(0)


def test_gapped_terrain():
    cameras = [] if disable_rendering else None  # None = default camera
    sim = SingleFlySimulation(fly=Fly(), arena=GappedTerrain(), cameras=cameras)
    sim.close()


def test_blocks_terrain():
    cameras = [] if disable_rendering else None  # None = default camera
    sim = SingleFlySimulation(fly=Fly(), arena=BlocksTerrain(), cameras=cameras)
    sim.close()


def test_mixed_terrain():
    cameras = [] if disable_rendering else None  # None = default camera
    sim = SingleFlySimulation(fly=Fly(), arena=MixedTerrain(), cameras=cameras)
    sim.close()
