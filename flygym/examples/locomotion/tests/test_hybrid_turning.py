import numpy as np
from gymnasium.utils.env_checker import check_env
from flygym import Fly
from flygym.arena import MixedTerrain
from flygym.examples.locomotion import HybridTurningController, HybridTurningNMF
from flygym.preprogrammed import default_leg_sensor_placements


def test_rule_based_controller_nophysics():
    run_time = 0.1
    timestep = 1e-4

    np.random.seed(0)

    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        contact_sensor_placements=default_leg_sensor_placements,
    )
    sim = HybridTurningController(
        fly=fly,
        cameras=[],
        timestep=timestep,
        seed=0,
        draw_corrections=True,
        arena=MixedTerrain(),
    )
    check_env(sim)

    for i in range(int(run_time / sim.timestep)):
        curr_time = i * sim.timestep

        # To demonstrate left and right turns:
        if curr_time < run_time / 2:
            action = np.array([1.2, 0.4])
        else:
            action = np.array([0.4, 1.2])

        # To demonstrate that the result is identical with the hybrid controller without
        # turning:
        action = np.array([1.0, 1.0])

        obs, reward, terminated, truncated, info = sim.step(action)
        sim.render()

    assert obs["fly"][0, 0] > 0  # fly has moved forward a little bit
    assert obs["fly"][0, 2] > 0  # fly is still above ground


def test_deprecation():
    timestep = 1e-4

    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        contact_sensor_placements=default_leg_sensor_placements,
    )
    sim = HybridTurningController(
        fly=fly,
        cameras=[],
        timestep=timestep,
        seed=0,
        draw_corrections=True,
        arena=MixedTerrain(),
    )

    # This should also work but with a warning
    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        contact_sensor_placements=default_leg_sensor_placements,
    )
    sim = HybridTurningNMF(
        fly=fly,
        cameras=[],
        timestep=timestep,
        seed=0,
        draw_corrections=True,
        arena=MixedTerrain(),
    )
