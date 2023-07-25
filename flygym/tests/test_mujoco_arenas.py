import numpy as np

from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo
from flygym.arena.mujoco_arena import GappedTerrain, BlocksTerrain, MixedTerrain
from flygym.tests import temp_base_dir


random_state = np.random.RandomState(0)


def test_gapped_terrain():
    out_dir = temp_base_dir / "mujoco_gapped_terrain"
    arena = GappedTerrain()
    nmf = NeuroMechFlyMuJoCo(output_dir=out_dir, arena=arena, adhesion=False)
    nmf.close()


def test_blocks_terrain():
    out_dir = temp_base_dir / "mujoco_blocks_terrain"
    arena = BlocksTerrain()
    nmf = NeuroMechFlyMuJoCo(output_dir=out_dir, arena=arena)
    nmf.close()


def test_mixed_terrain():
    out_dir = temp_base_dir / "mujoco_mixed_terrain"
    arena = MixedTerrain()
    nmf = NeuroMechFlyMuJoCo(output_dir=out_dir, arena=arena)
    nmf.close()
