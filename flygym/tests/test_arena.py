import numpy as np

from flygym import Fly, SingleFlySimulation
from flygym.arena import GappedTerrain, BlocksTerrain, MixedTerrain


np.random.seed(0)


def test_gapped_terrain():
    sim = SingleFlySimulation(fly=Fly(), arena=GappedTerrain())
    sim.close()


def test_blocks_terrain():
    sim = SingleFlySimulation(fly=Fly(), arena=BlocksTerrain())
    sim.close()


def test_mixed_terrain():
    sim = SingleFlySimulation(fly=Fly(), arena=MixedTerrain())
    sim.close()
