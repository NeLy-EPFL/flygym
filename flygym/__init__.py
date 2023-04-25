from gymnasium.envs.registration import register


register(
    id='nmf-mujoco-base-v0',
    entry_point='flygym.envs:NeuroMechFlyMuJoCo',
)