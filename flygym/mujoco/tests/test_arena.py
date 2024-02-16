import numpy as np

from flygym.mujoco import NeuroMechFly
from flygym.mujoco.arena import GappedTerrain, BlocksTerrain, MixedTerrain


np.random.seed(0)


def test_gapped_terrain():
    arena = GappedTerrain()
    nmf = NeuroMechFly(arena=arena)
    nmf.close()


def test_blocks_terrain():
    arena = BlocksTerrain()
    nmf = NeuroMechFly(arena=arena)
    nmf.close()


def test_mixed_terrain():
    arena = MixedTerrain()
    nmf = NeuroMechFly(arena=arena)
    nmf.close()
