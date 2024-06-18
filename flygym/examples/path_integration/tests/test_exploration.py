import pytest
import numpy as np
from flygym import Fly
from flygym.examples.path_integration import controller
from flygym.examples.path_integration.arena import PathIntegrationArenaFlat
from flygym.preprogrammed import get_cpg_biases


def test_random_exploration_controller():
    dt = 0.001
    random_exp_controller = controller.RandomExplorationController(
        dt=dt,
        seed=100,
        init_time=1,
        lambda_turn=10,
        forward_dn_drive=(1.0, 1.0),
        left_turn_dn_drive=(0.2, 1.0),
        right_turn_dn_drive=(1.0, 0.2),
        turn_duration_mean=0.04,
        turn_duration_std=0.01,
    )
    state_hist = []
    dn_drive_hist = []
    for i in range(int(1000 / dt)):
        state, dn_drive = random_exp_controller.step()
        state_hist.append(state)
        dn_drive_hist.append(dn_drive)
    state_hist = np.array(state_hist)
    dn_drive_hist = np.array(dn_drive_hist)

    forward_mask = state_hist == controller.WalkingState.FORWARD
    left_rurn_mask = state_hist == controller.WalkingState.TURN_LEFT
    right_turn_mask = state_hist == controller.WalkingState.TURN_RIGHT
    stop_mask = state_hist == controller.WalkingState.STOP

    assert forward_mask[: int(1 / dt)].all()  # always walking forward during init_time
    assert (dn_drive_hist[forward_mask, :] == [1.0, 1.0]).all()
    assert (dn_drive_hist[left_rurn_mask, :] == [0.2, 1.0]).all()
    assert (dn_drive_hist[right_turn_mask, :] == [1.0, 0.2]).all()
    assert (~stop_mask).all()  # never stops

    num_walk_to_turn_transitions = (np.diff(forward_mask.astype(int)) == -1).sum()
    walking_time = forward_mask.sum() * dt
    turning_rate = num_walk_to_turn_transitions / (walking_time - 1)
    assert turning_rate == pytest.approx(10, rel=0.1)
    mean_turn_duration = (1000 - walking_time) / num_walk_to_turn_transitions
    assert mean_turn_duration == pytest.approx(0.04, rel=0.1)


def test_path_integration_controller():
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]

    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        contact_sensor_placements=contact_sensor_placements,
        spawn_pos=(0, 0, 0.25),
    )

    arena = PathIntegrationArenaFlat()
    sim = controller.PathIntegrationController(
        phase_biases=get_cpg_biases("tripod"),
        fly=fly,
        arena=arena,
        cameras=[],
        timestep=1e-4,
        correction_rates={"retraction": (0, 0), "stumbling": (0, 0)},
    )

    dn_drive_arr = np.array([[1, 1], [0.2, 0.2], [0.2, 1.2], [1.2, 0.2]])
    left_total_li = []
    right_total_li = []
    for dn_drive in dn_drive_arr:
        obs, _ = sim.reset(0)
        stride_diff_hist = []
        for i in range(int(0.1 / sim.timestep)):
            obs, reward, terminated, truncated, info = sim.step(dn_drive)

            # Get real heading
            orientation_x, orientation_y = obs["fly_orientation"][:2]
            stride_diff_hist.append(obs["stride_diff_unmasked"])
        stride_diff_hist = np.array(stride_diff_hist)
        left_total_li.append(np.abs(stride_diff_hist[100:, :3, :]).sum())
        right_total_li.append(np.abs(stride_diff_hist[100:, 3:, :]).sum())

    # (1, 1) and (0.2, 0.2) should both be more or less straight
    assert left_total_li[0] / right_total_li[0] == pytest.approx(1, rel=0.25)
    assert left_total_li[1] / right_total_li[1] == pytest.approx(1, rel=0.25)
    # print(left_total_li[0] / right_total_li[0], left_total_li[1] / right_total_li[1])

    # (1, 1) should be 3-5x faster than (0.2, 0.2)
    ratio_2_10 = (left_total_li[0] + right_total_li[0]) / (
        left_total_li[1] + right_total_li[1]
    )
    # print(ratio_2_10)
    assert 3 < ratio_2_10 < 5

    # (0.2, 1.2) and (1.2, 0.2) should both be more or less turning
    # print(right_total_li[2] / left_total_li[2], left_total_li[3] / right_total_li[3])
    assert right_total_li[2] / left_total_li[2] == pytest.approx(1.85, abs=0.4)
    assert left_total_li[3] / right_total_li[3] == pytest.approx(1.85, abs=0.4)
