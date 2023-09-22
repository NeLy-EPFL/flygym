import numpy as np

from flygym.mujoco import NeuroMechFlyMuJoCo
from flygym.mujoco.arena import GappedTerrain, BlocksTerrain, MixedTerrain


random_state = np.random.RandomState(0)


def test_gapped_terrain():
    arena = GappedTerrain()
    nmf = NeuroMechFlyMuJoCo(arena=arena)
    nmf.close()


def test_blocks_terrain():
    arena = BlocksTerrain()
    nmf = NeuroMechFlyMuJoCo(arena=arena)
    nmf.close()


def test_mixed_terrain():
    arena = MixedTerrain()
    nmf = NeuroMechFlyMuJoCo(arena=arena)
    nmf.close()