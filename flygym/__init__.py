from gymnasium.envs.registration import register
from importlib_metadata import entry_points


register(
    id='nmf-mujoco-base-v0',
    entry_point='flygym.envs:NeuroMechFlyMuJoCo',
)