import pytest
import numpy as np
from flygym.examples.locomotion import CPGNetwork, PreprogrammedSteps


def test_small_cpg_network():
    intrinsic_freqs = np.ones(3)
    intrinsic_amps = np.array([1.0, 1.1, 1.2])
    coupling_weights = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
    )
    phase_biases = np.deg2rad(
        np.array(
            [
                [0, 120, 0],
                [-120, 0, 120],
                [0, -120, 0],
            ]
        )
    )
    convergence_coefs = np.ones(3)

    network = CPGNetwork(
        timestep=1e-3,
        intrinsic_freqs=intrinsic_freqs,
        intrinsic_amps=intrinsic_amps,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=convergence_coefs,
    )

    num_steps = int(10 / network.timestep)
    phase_hist = np.empty((num_steps, 3))
    magnitude_hist = np.empty((num_steps, 3))

    # Simulate the network
    for i in range(num_steps):
        network.step()
        phase_hist[i, :] = network.curr_phases
        magnitude_hist[i, :] = network.curr_magnitudes

    assert phase_hist[-1, :] == pytest.approx([64.53087342, 66.62493815, 68.71905795])
    assert magnitude_hist[-1, :] == pytest.approx([1, 1.1, 1.2], abs=1e-2)


def test_preprogrammed_steps():
    preprogrammed_steps = PreprogrammedSteps()
    theta_ts = np.linspace(0, 3 * 2 * np.pi, 100)
    r_ts = np.linspace(0, 1, 100)
    joint_angles_all = []

    for side in "LR":
        for pos in "FMH":
            leg = f"{side}{pos}"
            joint_angles = preprogrammed_steps.get_joint_angles(leg, theta_ts, r_ts)
            joint_angles_all.append(joint_angles)

    joint_angles_all = np.array(joint_angles_all)
    assert joint_angles_all.shape == (6, 7, 100)
    assert joint_angles_all.sum() == pytest.approx(-23.01576, abs=1e-3)
