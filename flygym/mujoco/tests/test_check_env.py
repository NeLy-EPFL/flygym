import gymnasium.spaces as spaces
import gymnasium.utils.env_checker as env_checker

from flygym.mujoco import NeuroMechFlyMuJoCo, MuJoCoParameters
from flygym.mujoco.arena import OdorArena


def test_check_env_basic():
    sim_params = MuJoCoParameters(enable_vision=False, enable_olfaction=False)
    nmf = NeuroMechFlyMuJoCo(sim_params=sim_params)
    # Override action space so joint control signals are reasonably bound. Otherwise
    # the physics might become invalid
    nmf.action_space = spaces.Dict(
        {
            "joints": spaces.Box(low=-1, high=1, shape=(len(nmf.actuated_joints),)),
            "adhesion": spaces.Discrete(n=2, start=0),  # 0: no adhesion, 1: adhesion
        }
    )
    env_checker.check_env(nmf, skip_render_check=True)


def test_check_env_vision():
    sim_params = MuJoCoParameters(enable_vision=True, enable_olfaction=False)
    nmf = NeuroMechFlyMuJoCo(sim_params=sim_params)
    # Override action space so joint control signals are reasonably bound. Otherwise
    # the physics might become invalid
    nmf.action_space = spaces.Dict(
        {
            "joints": spaces.Box(low=-1, high=1, shape=(len(nmf.actuated_joints),)),
            "adhesion": spaces.Discrete(n=2, start=0),  # 0: no adhesion, 1: adhesion
        }
    )
    env_checker.check_env(nmf, skip_render_check=True)


def test_check_env_olfaction():
    sim_params = MuJoCoParameters(enable_vision=False, enable_olfaction=True)
    arena = OdorArena(odor_source=[[10, 0, 0]], peak_intensity=[[1, 2]])
    nmf = NeuroMechFlyMuJoCo(sim_params=sim_params, arena=arena)
    # Override action space so joint control signals are reasonably bound. Otherwise
    # the physics might become invalid
    nmf.action_space = spaces.Dict(
        {
            "joints": spaces.Box(low=-1, high=1, shape=(len(nmf.actuated_joints),)),
            "adhesion": spaces.Discrete(n=2, start=0),  # 0: no adhesion, 1: adhesion
        }
    )
    env_checker.check_env(nmf, skip_render_check=True)
