import pytest
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.utils.env_checker import check_env
from flygym import Fly, SingleFlySimulation
from flygym.arena import MixedTerrain
from flygym.preprogrammed import all_leg_dofs
from flygym.examples.locomotion import PreprogrammedSteps, HybridTurningController


def test_rule_based_controller_nophysics():
    run_time = 0.1
    timestep = 1e-4
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]

    np.random.seed(0)

    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        contact_sensor_placements=contact_sensor_placements,
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
