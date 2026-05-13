from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from flygym.anatomy import JointDOF
from flygym.examples.locomotion.common import LocomotionAction
from flygym.examples.locomotion.preprogrammed import PreprogrammedSteps


def calculate_ddt(
    theta: Float[np.ndarray, "n"],
    r: Float[np.ndarray, "n"],
    w: Float[np.ndarray, "n n"],
    phi: Float[np.ndarray, "n n"],
    nu: Float[np.ndarray, "n"],
    R: Float[np.ndarray, "n"],
    alpha: Float[np.ndarray, "n"],
) -> tuple[Float[np.ndarray, "n"], Float[np.ndarray, "n"]]:
    """Compute oscillator phase and magnitude derivatives."""
    intrinsic_term = 2 * np.pi * nu
    phase_diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    coupling_term = (r * w * np.sin(phase_diff - phi)).sum(axis=1)
    dtheta_dt = intrinsic_term + coupling_term
    dr_dt = alpha * (R - r)
    return dtheta_dt, dr_dt


class CPGNetwork:
    """Euler-integrated network of coupled phase-amplitude oscillators."""

    def __init__(
        self,
        timestep: float,
        intrinsic_freqs: Float[np.ndarray, "n"],
        intrinsic_amps: Float[np.ndarray, "n"],
        coupling_weights: Float[np.ndarray, "n n"],
        phase_biases: Float[np.ndarray, "n n"],
        convergence_coefs: Float[np.ndarray, "n"],
        init_phases: Float[np.ndarray, "n"] | None = None,
        init_magnitudes: Float[np.ndarray, "n"] | None = None,
        seed: int = 0,
    ) -> None:
        self.timestep = timestep
        self.num_cpgs = intrinsic_freqs.size
        self.intrinsic_freqs = np.asarray(intrinsic_freqs, dtype=float)
        self.intrinsic_amps = np.asarray(intrinsic_amps, dtype=float)
        self.coupling_weights = np.asarray(coupling_weights, dtype=float)
        self.phase_biases = np.asarray(phase_biases, dtype=float)
        self.convergence_coefs = np.asarray(convergence_coefs, dtype=float)
        self.random_state = np.random.RandomState(seed)

        self.reset(init_phases, init_magnitudes)

        if self.intrinsic_freqs.shape != (self.num_cpgs,):
            raise ValueError("intrinsic_freqs must have shape (n,).")
        if self.intrinsic_amps.shape != (self.num_cpgs,):
            raise ValueError("intrinsic_amps must have shape (n,).")
        if self.coupling_weights.shape != (self.num_cpgs, self.num_cpgs):
            raise ValueError("coupling_weights must have shape (n, n).")
        if self.phase_biases.shape != (self.num_cpgs, self.num_cpgs):
            raise ValueError("phase_biases must have shape (n, n).")
        if self.convergence_coefs.shape != (self.num_cpgs,):
            raise ValueError("convergence_coefs must have shape (n,).")

    def reset(
        self,
        init_phases: Float[np.ndarray, "n"] | None = None,
        init_magnitudes: Float[np.ndarray, "n"] | None = None,
    ) -> None:
        if init_phases is None:
            self.curr_phases = self.random_state.random(self.num_cpgs) * 2 * np.pi
        else:
            self.curr_phases = np.asarray(init_phases, dtype=float).copy()

        if init_magnitudes is None:
            self.curr_magnitudes = np.zeros(self.num_cpgs, dtype=float)
        else:
            self.curr_magnitudes = np.asarray(init_magnitudes, dtype=float).copy()

    def step(self) -> None:
        """Integrate the oscillator state by one timestep."""
        dtheta_dt, dr_dt = calculate_ddt(
            theta=self.curr_phases,
            r=self.curr_magnitudes,
            w=self.coupling_weights,
            phi=self.phase_biases,
            nu=self.intrinsic_freqs,
            R=self.intrinsic_amps,
            alpha=self.convergence_coefs,
        )
        self.curr_phases += dtheta_dt * self.timestep
        self.curr_magnitudes += dr_dt * self.timestep


@dataclass
class CPGController:
    """Map CPG oscillator states to preprogrammed leg-step actions."""

    cpg_network: CPGNetwork
    preprogrammed_steps: PreprogrammedSteps
    output_dof_order: list[JointDOF] | None = None

    def step(self) -> LocomotionAction:
        self.cpg_network.step()
        joint_angles = self.preprogrammed_steps.get_joint_angles_by_dof_order(
            self.cpg_network.curr_phases,
            self.cpg_network.curr_magnitudes,
            self.output_dof_order,
        )
        adhesion_onoff = self.preprogrammed_steps.get_adhesion_onoff_by_phase(
            self.cpg_network.curr_phases
        )
        return LocomotionAction(
            joint_angles=joint_angles, adhesion_onoff=adhesion_onoff
        )


def get_cpg_biases(gait: str) -> Float[np.ndarray, "6 6"]:
    """Define phase biases for tripod, tetrapod, or wave gaits."""
    gait = gait.lower()
    if gait == "tripod":
        phase_biases = np.array(
            [
                [0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0],
            ],
            dtype=float,
        )
        phase_biases *= np.pi
    elif gait == "tetrapod":
        phase_biases = np.array(
            [
                [0, 1, 2, 2, 0, 1],
                [2, 0, 1, 1, 2, 0],
                [1, 2, 0, 0, 1, 2],
                [1, 2, 0, 0, 1, 2],
                [0, 1, 2, 2, 0, 1],
                [2, 0, 1, 1, 2, 0],
            ],
            dtype=float,
        )
        phase_biases *= 2 * np.pi / 3
    elif gait == "wave":
        phase_biases = np.array(
            [
                [0, 1, 2, 3, 4, 5],
                [5, 0, 1, 2, 3, 4],
                [4, 5, 0, 1, 2, 3],
                [3, 4, 5, 0, 1, 2],
                [2, 3, 4, 5, 0, 1],
                [1, 2, 3, 4, 5, 0],
            ],
            dtype=float,
        )
        phase_biases *= 2 * np.pi / 6
    else:
        raise ValueError(f"Unknown gait: {gait}")
    return phase_biases


def make_tripod_cpg_network(
    timestep: float,
    *,
    intrinsic_frequency: float = 12.0,
    intrinsic_amplitude: float = 1.0,
    coupling_strength: float = 10.0,
    convergence_coef: float = 20.0,
    seed: int = 0,
) -> CPGNetwork:
    """Create the default six-leg tripod CPG network used in v1 tutorials.

    ``intrinsic_frequency`` defaults to **12 Hz**, matching the legacy v1 CPG demos.
    Setting it to ``1 / PreprogrammedSteps().duration`` (see
    ``PreprogrammedSteps.step_cycle_frequency_hz``, ~7.4 Hz for the bundled pickle)
    matches how fast the *kinematic* spline was sampled along one recorded step cycle,
    but in MuJoCo v2 that choice typically **produces far less thorax translation** than
    12 Hz for the same wall time: floating-base walking builds thrust from dynamic
    interaction with adhesion and friction, not from kinematic time-scaling alone.
    Higher ``coupling_strength`` (with 12 Hz) tends to further increase forward drift
    by locking tripod phasing. Replay of experimental joint angles (Tutorial 2) can
    still move the body faster than this open-loop CPG because the recording encodes
    whole-animal progression, not only periodic leg templates.
    """
    phase_biases = get_cpg_biases("tripod")
    return CPGNetwork(
        timestep=timestep,
        intrinsic_freqs=np.ones(6) * intrinsic_frequency,
        intrinsic_amps=np.ones(6) * intrinsic_amplitude,
        coupling_weights=(phase_biases > 0) * coupling_strength,
        phase_biases=phase_biases,
        convergence_coefs=np.ones(6) * convergence_coef,
        seed=seed,
    )
