import numpy as np

from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo
from flygym.tests import temp_base_dir


random_state = np.random.RandomState(0)


def test_gapped_terrain():
    out_dir = temp_base_dir / "mujoco_gapped_terrain"
    nmf = NeuroMechFlyMuJoCo(
        render_mode="headless", output_dir=out_dir, terrain="gapped"
    )
    nmf.close()


def test_blocks_terrain():
    out_dir = temp_base_dir / "mujoco_blocks_terrain"
    nmf = NeuroMechFlyMuJoCo(
        render_mode="headless", output_dir=out_dir, terrain="blocks"
    )
    nmf.close()


def test_mixed_terrain():
    out_dir = temp_base_dir / "mujoco_mixed_terrain"
    nmf = NeuroMechFlyMuJoCo(
        render_mode="headless", output_dir=out_dir, terrain="mixed"
    )
    nmf.close()
