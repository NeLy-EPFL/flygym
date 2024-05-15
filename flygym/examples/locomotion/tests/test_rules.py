import pytest
import numpy as np
import matplotlib.pyplot as plt
from flygym import Fly, SingleFlySimulation
from flygym.preprogrammed import all_leg_dofs
from flygym.examples.locomotion import RuleBasedController, PreprogrammedSteps
from flygym.examples.locomotion.rule_based_controller import construct_rules_graph


def test_rule_based_controller_nophysics():
    run_time = 0.1
    timestep = 1e-4

    # Initialize preprogrammed steps
    preprogrammed_steps = PreprogrammedSteps()

    # Initialize rule-based controller
    weights = {
        "rule1": -10,
        "rule2_ipsi": 2.5,
        "rule2_contra": 1,
        "rule3_ipsi": 3.0,
        "rule3_contra": 2.0,
    }
    rules_graph = construct_rules_graph()
    controller = RuleBasedController(
        timestep=timestep,
        rules_graph=rules_graph,
        weights=weights,
        preprogrammed_steps=preprogrammed_steps,
    )

    joint_angle_hist = []
    adhesion_hist = []
    for i in range(int(run_time / timestep)):
        controller.step()
        joint_angles = []
        adhesion_onoff = []
        for leg, phase in zip(controller.legs, controller.leg_phases):
            joint_angles_arr = controller.preprogrammed_steps.get_joint_angles(
                leg, phase
            )
            joint_angles.append(joint_angles_arr.flatten())
            adhesion_onoff.append(
                controller.preprogrammed_steps.get_adhesion_onoff(leg, phase)
            )
        joint_angle_hist.append(np.concatenate(joint_angles))
        adhesion_hist.append(np.array(adhesion_onoff))

    joint_angle_hist = np.array(joint_angle_hist)
    adhesion_hist = np.array(adhesion_hist)

    # Check if the figures make sense. If so, record what the numbers are
    # supposed to be using the print statements. Then assert that they are
    # always the same.
    # plt.plot(joint_angle_hist)
    # plt.show()
    # plt.plot(adhesion_hist)
    # plt.show()
    # print(joint_angle_hist.shape, adhesion_hist.shape)
    # print(joint_angle_hist.sum(), adhesion_hist.sum())

    assert joint_angle_hist.shape == (int(run_time / timestep), 42)
    assert adhesion_hist.shape == (int(run_time / timestep), 6)
    assert joint_angle_hist.sum() == pytest.approx(-194.31692)
    assert adhesion_hist.sum() == pytest.approx(4931)


def test_rule_based_controller():
    run_time = 0.1
    timestep = 1e-4

    # Initialize preprogrammed steps
    preprogrammed_steps = PreprogrammedSteps()

    # Initialize rule-based controller
    weights = {
        "rule1": -10,
        "rule2_ipsi": 2.5,
        "rule2_contra": 1,
        "rule3_ipsi": 3.0,
        "rule3_contra": 2.0,
    }
    rules_graph = construct_rules_graph()
    controller = RuleBasedController(
        timestep=timestep,
        rules_graph=rules_graph,
        weights=weights,
        preprogrammed_steps=preprogrammed_steps,
    )
    fly = Fly(
        init_pose="stretch",
        actuated_joints=all_leg_dofs,
        control="position",
        enable_adhesion=True,
        draw_adhesion=True,
    )
    sim = SingleFlySimulation(
        fly=fly,
        cameras=[],
        timestep=timestep,
    )

    obs, _ = sim.reset()
    for i in range(int(run_time / timestep)):
        controller.step()
        joint_angles = []
        adhesion_onoff = []
        for leg, phase in zip(controller.legs, controller.leg_phases):
            joint_angles_arr = controller.preprogrammed_steps.get_joint_angles(
                leg, phase
            )
            joint_angles.append(joint_angles_arr.flatten())
            adhesion_onoff.append(
                controller.preprogrammed_steps.get_adhesion_onoff(leg, phase)
            )
        action = {
            "joints": np.concatenate(joint_angles),
            "adhesion": np.array(adhesion_onoff),
        }
        obs, reward, terminated, truncated, info = sim.step(action)

    assert obs["fly"][0, 0] > 0  # has moved forward a little bit
    assert obs["fly"][0, 2] > 0  # still above ground
