import numpy as np

from flygym.mujoco import NeuroMechFlyMuJoCo, MuJoCoParameters
from flygym.mujoco.arena import OdorArena


def test_odor_dimensions():
    num_sources = 5
    num_dims = 4
    odor_source = np.zeros((num_sources, 3))
    peak_intensity = np.ones((num_sources, num_dims))

    # Initialize simulation
    num_steps = 100
    arena = OdorArena(odor_source=odor_source, peak_intensity=peak_intensity)
    sim_params = MuJoCoParameters(enable_olfaction=True)
    nmf = NeuroMechFlyMuJoCo(sim_params=sim_params, arena=arena)

    # Run simulation
    obs_list = []
    for i in range(num_steps):
        joint_pos = np.zeros(len(nmf.actuated_joints))
        action = {"joints": joint_pos}
        obs, reward, terminated, truncated, info = nmf.step(action)
        # nmf.render()
        obs_list.append(obs)
    nmf.close()

    # Check dimensionality
    odor = np.array([obs["odor_intensity"] for obs in obs_list])
    assert odor.shape == (num_steps, num_dims, 4)
