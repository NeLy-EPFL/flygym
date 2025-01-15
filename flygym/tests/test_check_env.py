import gymnasium.spaces as spaces
import gymnasium.utils.env_checker as env_checker
import pytest

from flygym import Camera, Fly, Simulation, SingleFlySimulation, is_rendering_skipped
from flygym.arena import OdorArena


def test_check_env_1fly():
    fly = Fly(
        enable_vision=False,
        enable_olfaction=False,
        enable_adhesion=False,
    )
    num_dofs_per_fly = len(fly.actuated_joints)
    sim = SingleFlySimulation(fly, cameras=[])
    # Override action space so joint control signals are reasonably bound. Otherwise,
    # the physics might become invalid
    sim.action_space = spaces.Dict(
        {
            "joints": spaces.Box(low=-0.1, high=0.1, shape=(num_dofs_per_fly,)),
        }
    )
    env_checker.check_env(sim, skip_render_check=True)


def test_check_env_2flies():
    flies = [
        Fly(
            name=f"fly{i}",
            enable_vision=False,
            enable_olfaction=False,
            enable_adhesion=False,
        )
        for i in range(2)
    ]
    num_dofs_per_fly = len(flies[0].actuated_joints)
    sim = Simulation(flies, cameras=[])
    # Override action space so joint control signals are reasonably bound. Otherwise,
    # the physics might become invalid
    sim.action_space = spaces.Dict(
        {
            "fly0": spaces.Dict(
                {
                    "joints": spaces.Box(low=-0.1, high=0.1, shape=(num_dofs_per_fly,)),
                }
            ),
            "fly1": spaces.Dict(
                {
                    "joints": spaces.Box(low=-0.1, high=0.1, shape=(num_dofs_per_fly,)),
                }
            ),
        }
    )
    env_checker.check_env(sim, skip_render_check=True)


def test_check_simulation_env_basic():
    flies = [
        Fly(
            name=f"{i}",
            enable_vision=False,
            enable_olfaction=False,
        )
        for i in range(2)
    ]
    cameras = (
        []
        if is_rendering_skipped
        else [Camera(fly.model.worldbody, "camera_left", fly.name) for fly in flies]
    )
    sim = Simulation(flies=flies, cameras=cameras)

    sim.action_space = spaces.Dict(
        {
            fly.name: spaces.Dict(
                {
                    "joints": spaces.Box(
                        low=-0.1, high=0.1, shape=(len(fly.actuated_joints),)
                    ),
                    "adhesion": spaces.Discrete(
                        n=2, start=0
                    ),  # 0: no adhesion, 1: adhesion
                }
            )
            for fly in flies
        }
    )
    env_checker.check_env(sim, skip_render_check=True)


def test_check_single_fly_simulation_env_basic():
    fly = Fly(name="0", enable_vision=False, enable_olfaction=False)
    camera = (
        []
        if is_rendering_skipped
        else Camera(fly.model.worldbody, "camera_left", fly.name)
    )
    sim = SingleFlySimulation(fly=fly, cameras=camera)  # check cam instead of [cam]
    fly.action_space = spaces.Dict(
        {
            "joints": spaces.Box(low=-0.1, high=0.1, shape=(len(fly.actuated_joints),)),
            "adhesion": spaces.Discrete(n=2, start=0),  # 0: no adhesion, 1: adhesion
        }
    )
    env_checker.check_env(sim, skip_render_check=True)


@pytest.mark.skipif(is_rendering_skipped, reason="env['SKIP_RENDERING'] == 'true'")
def test_check_env_vision():
    fly = Fly(enable_vision=True, enable_olfaction=False)
    sim = SingleFlySimulation(fly=fly)
    # Override action space so joint control signals are reasonably bound. Otherwise,
    # the physics might become invalid
    sim.action_space = spaces.Dict(
        {
            "joints": spaces.Box(low=-0.1, high=0.1, shape=(len(fly.actuated_joints),)),
            "adhesion": spaces.Discrete(n=2, start=0),  # 0: no adhesion, 1: adhesion
        }
    )
    env_checker.check_env(sim, skip_render_check=True)


def test_check_env_olfaction():
    fly = Fly(enable_vision=False, enable_olfaction=True)
    arena = OdorArena(odor_source=[[10, 0, 0]], peak_odor_intensity=[[1, 2]])
    cameras = [] if is_rendering_skipped else None  # None = default camera
    sim = SingleFlySimulation(fly=fly, arena=arena, cameras=cameras)
    # Override action space so joint control signals are reasonably bound. Otherwise,
    # the physics might become invalid
    sim.action_space = spaces.Dict(
        {
            "joints": spaces.Box(low=-0.1, high=0.1, shape=(len(fly.actuated_joints),)),
            "adhesion": spaces.Discrete(n=2, start=0),  # 0: no adhesion, 1: adhesion
        }
    )
    env_checker.check_env(sim, skip_render_check=True)
