from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt

CPG_PERIOD = 2 * np.pi


def plot_phase_amp_output(phases, amps, outs, labels=None, timestep=1):
    time = timestep * np.arange(len(phases))
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(time, phases, label=labels)
    axs[0].set_ylabel("Phase")
    axs[1].plot(time, amps, label=labels)
    axs[1].set_ylabel("Amplitude")
    axs[1].legend(loc="lower right")
    axs[2].plot(time, outs, label=labels)
    axs[2].set_ylabel("Output")
    axs[2].legend(loc="lower right")

    if labels:
        axs[0].legend(loc="lower right")
        axs[1].legend(loc="lower right")
        axs[2].legend(loc="lower right")
    plt.tight_layout()


def plot_phase_amp_output_rules(
    phases, amps, outs, rules, labels=None, rule_labels=None, timestep=1
):
    time = timestep * np.arange(len(phases))
    fig, axs = plt.subplots(4, 1, figsize=(12, 10))
    axs[0].plot(time, phases, label=labels)
    axs[0].set_ylabel("Phase")
    axs[1].plot(time, amps, label=labels)
    axs[1].set_ylabel("Amplitude")
    axs[1].legend(loc="lower right")
    axs[2].plot(time, outs, label=labels)
    axs[2].set_ylabel("Output")
    axs[2].legend(loc="lower right")
    for i, rule in enumerate(rules):
        axs[3].plot(time, rule, label=rule_labels[i])
    axs[3].set_ylabel("Rule active")
    axs[3].legend(loc="lower right")

    if labels:
        axs[0].legend(loc="lower right")
        axs[1].legend(loc="lower right")
        axs[2].legend(loc="lower right")
    if rule_labels:
        axs[3].legend(loc="lower right")
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


class CPG:
    """Central Pattern Generator.

    Attributes
    ----------
    phase : np.ndarray
        Current phase of each oscillator, of size (n_oscillators,).
    amplitude : np.ndarray
        Current amplitude of each oscillator, of size (n_oscillators,).
    frequencies : np.ndarray
        Target frequency of each oscillator, of size (n_oscillators,).
    phase_biases : np.ndarray
        Phase bias matrix describing the target phase relations between the oscillators.
        Dimensions (n_oscillators,n_oscillators), set to values generating a tripod gait.
    coupling_weights : np.ndarray
        Coupling weights matrix between oscillators enforcing phase relations.
        Dimensions (n_oscillators,n_oscillators).
    rates : np.ndarray
        Convergence rates for the amplitudes.
    targ_ampl : np.ndarray
        Target amplitude for each oscillator, of size (n_oscillators,).
        Default value of 1.0, modulated by input to the step function.

    Parameters
    ----------
    timestep : float
        Timestep duration for the integration.
    n_oscillators : int
        Number of individual oscillators, by default 6.
    turn_mode : string
        Describes what quantities are influenced by the turn modulation.
        Can be "amp" (only amplitudes - default), "freq" (only frequencies), or "both".
    """

    def __init__(self, timestep, n_oscillators: int = 6):
        self.n_oscillators = n_oscillators
        # Random initializaton of oscillator states
        self.phase = np.random.rand(n_oscillators)
        self.amplitude = np.repeat(np.random.randint(1), n_oscillators)
        self.timestep = timestep
        # CPG parameters
        self.frequencies = 12 * np.ones(n_oscillators)
        self.base_freq = 12 * np.ones(n_oscillators)
        self.phase_biases = 2 * np.pi * phase_biases_tripod_idealized
        self.base_ampl = 1.0 * np.ones(n_oscillators)
        self.min_ampl = 0.2 * np.ones(n_oscillators)

        self.coupling_weights = (np.abs(self.phase_biases) > 0).astype(float) * 5.0
        self.rates = 10.0 * np.ones(n_oscillators)

    def step(self, turn_modulation=[0, 0]):
        # Sign of the turn modulation changes the frequency of the leg
        # i.e negative values will lead to the reversal of the phase
        # Absolute value of the turn modulation changes the amplitude of the leg
        # i.e. higher values will lead to higher amplitudes

        turn_modulation = np.array(turn_modulation)

        # Reset the frequencies to the base frequency
        self.frequencies = (
            (np.repeat(turn_modulation >= 0, 3) - 0.5) * 2 * self.base_freq
        )
        # Need to add base amplitude as legs should always be stepping to turn on or off the adhesion
        self.targ_ampl = (
            np.repeat(np.abs(turn_modulation), 3) * self.base_ampl + self.min_ampl
        )

        if turn_modulation[0] == 0.0 and turn_modulation[1] == 0.0:
            self.targ_ampl = np.zeros(6)

        # if np.random.rand() < 0.001 and turn_modulation[0] != turn_modulation[1]:
        #    print(turn_modulation, self.frequencies, self.targ_ampl)

        # Integration step
        self.phase, self.amplitude = self.euler_int(
            self.phase, self.amplitude, timestep=self.timestep
        )

    def euler_int(self, prev_phase, prev_ampl, timestep):
        dphas, dampl = self.phase_oscillator(prev_phase, prev_ampl)
        phase = (prev_phase + timestep * dphas) % (2 * np.pi)
        ampl = prev_ampl + timestep * dampl
        return phase, ampl

    def phase_oscillator(self, phases, amplitudes):
        """Phase oscillator model used in Ijspeert et al. 2007"""
        # NxN matrix with the phases of the oscillators
        phase_matrix = np.tile(phases, (self.n_oscillators, 1))

        # NxN matrix with the amplitudes of the oscillators
        amp_matrix = np.tile(amplitudes, (self.n_oscillators, 1))

        freq_contribution = 2 * np.pi * self.frequencies

        #  scaling of the phase differences between oscillators by the amplitude of the oscillators and the coupling weights
        scaling = np.multiply(amp_matrix, self.coupling_weights)

        # phase matrix and transpose substraction are analogous to the phase differences between oscillators, those should be close to the phase biases
        phase_shifts_contribution = np.sin(
            phase_matrix - phase_matrix.T - self.phase_biases
        )

        # Here we compute the contribution of the phase biases to the derivative of the phases
        # we mulitply two NxN matrices and then sum over the columns (all j oscillators contributions) to get a vector of size N
        coupling_contribution = np.sum(
            np.multiply(scaling, phase_shifts_contribution), axis=1
        )

        # Here we compute the derivative of the phases given by the equations defined previously.
        # We are using for that matrix operations to speed up the computation
        dphases = freq_contribution + coupling_contribution
        # dphases = np.clip(dphases, 0, None)

        damplitudes = np.multiply(self.rates, self.targ_ampl - amplitudes)
        # print("targ_ampl ", targ_ampl, " ", damplitudes)

        return dphases, damplitudes

    def reset(self):
        self.phase = np.random.rand(self.n_oscillators)
        self.amplitude = np.random.rand(self.n_oscillators)
