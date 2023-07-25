from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt

CPG_PERIOD = 2 * np.pi


def plot_phase_amp_output(phases, amps, outs, labels=None):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(phases, label=labels)
    axs[0].set_ylabel("Phase")
    axs[1].plot(amps, label=labels)
    axs[1].set_ylabel("Amplitude")
    axs[1].legend(loc="lower right")
    axs[2].plot(outs, label=labels)
    axs[2].set_ylabel("Output")
    axs[2].legend(loc="lower right")

    if labels:
        axs[0].legend(loc="lower right")
        axs[1].legend(loc="lower right")
        axs[2].legend(loc="lower right")
    plt.tight_layout()


def advancement_transfer(phases, step_dur, match_leg_to_joints):
    """From phase define what is the corresponding timepoint in the joint dataset
    In the case of the oscillator, the period is 2pi and the step duration is the period of the step
    We have to match those two"""

    # match length of step to period phases should have a period of period mathc this perios to the one of the step
    t_indices = np.round(np.mod(phases * step_dur / CPG_PERIOD, step_dur - 1)).astype(
        int
    )
    t_indices = t_indices[match_leg_to_joints]

    return t_indices


def phase_oscillator(
    _time,
    state,
    n_oscillators,
    frequencies,
    coupling_weights,
    phase_biases,
    target_amplitudes,
    rates,
):
    """Phase oscillator model used in Ijspeert et al. 2007"""

    phases = state[:n_oscillators]
    amplitudes = state[n_oscillators : 2 * n_oscillators]

    # NxN matrices with the phases of the oscillators and amplitudes
    phase_matrix = np.tile(phases, (n_oscillators, 1))
    amp_matrix = np.tile(amplitudes, (n_oscillators, 1))

    freq_contribution = 2 * np.pi * frequencies
    scaling = np.multiply(amp_matrix, coupling_weights)
    phase_shifts_contribution = np.sin(phase_matrix - phase_matrix.T - phase_biases)
    coupling_contribution = np.sum(
        np.multiply(scaling, phase_shifts_contribution), axis=1
    )
    dphases = freq_contribution + coupling_contribution

    damplitudes = np.multiply(rates, target_amplitudes - amplitudes)

    return np.concatenate([dphases, damplitudes])


def sine_output(phases, amplitudes):
    return amplitudes * (1 + np.cos(phases))


def initialize_solver(
    int_f,
    integ_name,
    t,
    n_oscillators,
    frequencies,
    coupling_weights,
    phase_biases,
    target_amplitudes,
    rates,
    int_params={},
):
    solver = ode(f=int_f)
    solver.set_integrator(integ_name, **int_params)
    initial_values = np.random.rand(2 * n_oscillators)
    solver.set_initial_value(y=initial_values, t=t).set_f_params(
        n_oscillators,
        frequencies,
        coupling_weights,
        phase_biases,
        target_amplitudes,
        rates,
    )
    return solver


# From de Angelis et al. 2019 eLife
phase_biases_tripod_measured = np.array(
    [
        [0, 0.43, 0.86, 0.5, 0.93, 1.36],
        [-0.43, 0, 0.43, 0.07, 0.5, 0.93],
        [-0.86, -0.43, 0, -0.36, 0.07, 0.5],
        [-0.5, -0.07, 0.36, 0, 0.43, 0.86],
        [-0.93, 0.5, -0.07, -0.43, 0, 0.43],
        [-1.36, -0.93, 0.5, -0.86, -0.43, 0],
    ]
)

phase_biases_tripod_idealized = np.array(
    [
        [0, 0.5, 1.0, 0.5, 1.0, 0.5],
        [0.5, 0, 0.5, 1.0, 0.5, 1.00],
        [1.0, 0.5, 0, 0.5, 1.0, 0.5],
        [0.5, 1.0, 0.5, 0, 0.5, 1.0],
        [1.0, 0.5, 1.0, 0.5, 0, 0.5],
        [0.5, 1.0, 0.5, 1.0, 0.5, 0],
    ]
)

phase_biases_metachronal_idealized = np.array(
    [
        [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6],
        [-1 / 6, 0, 1 / 6, 2 / 6, 3 / 6, 4 / 6],
        [-2 / 6, -1 / 6, 0, 1 / 6, 2 / 6, 3 / 6],
        [-3 / 6, -2 / 6, -1 / 6, 0, 1 / 6, 2 / 6],
        [-4 / 6, -3 / 6, -2 / 6, -1 / 6, 0, 1 / 6],
        [-5 / 6, -4 / 6, -3 / 6, -2 / 6, -1 / 6, 0],
    ]
)

phase_biases_ltetrapod_idealized = np.array(
    [
        [0, -1 / 3, 1 / 3, 1 / 3, 1, -1 / 3],
        [1 / 3, 0, -1 / 3, -1 / 3, 1 / 3, 1],
        [-1 / 3, 1 / 3, 0, 1, -1 / 3, 1 / 3],
        [-1 / 3, 1 / 3, 1, 0, -1 / 3, 1 / 3],
        [1, -1 / 3, 1 / 3, 1 / 3, 0, -1 / 3],
        [1 / 3, 0, -1 / 3, -1 / 3, 1 / 3, 0],
    ]
)
