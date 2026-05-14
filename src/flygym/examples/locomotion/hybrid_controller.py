from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flygym.anatomy import BodySegment, JointDOF, LEGS
from flygym.examples.locomotion.common import LocomotionAction
from flygym.examples.locomotion.cpg_controller import (
    CPGNetwork,
    make_tripod_cpg_network,
)
from flygym.examples.locomotion.preprogrammed import (
    PreprogrammedSteps,
    _dof_spec_to_jointdof,
)
from flygym.simulation import Simulation

_CORRECTION_VECTORS = {
    "f": np.array([-0.03, 0.0, 0.0, -0.03, 0.0, 0.03, 0.03]),
    "m": np.array([-0.015, 0.001, 0.025, -0.02, 0.0, -0.02, 0.0]),
    "h": np.array([0.0, 0.0, 0.0, -0.02, 0.0, 0.01, -0.02]),
}
_RIGHT_LEG_CORRECTION_SIGN = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0])
_DETECTED_STUMBLING_LINKS = ("tibia", "tarsus1", "tarsus2")


@dataclass
class HybridController:
    """CPG walking controller with retraction and stumbling corrections."""

    timestep: float
    cpg_network: CPGNetwork | None = None
    preprogrammed_steps: PreprogrammedSteps = field(default_factory=PreprogrammedSteps)
    output_dof_order: list[JointDOF] | None = None
    stumbling_force_threshold: float = -1.0
    retraction_height_threshold: float = 0.05
    retraction_rates: tuple[float, float] = (800.0, 700.0)
    stumbling_rates: tuple[float, float] = (2200.0, 1800.0)
    max_correction: float = 80.0
    swing_extension: float = np.pi / 4
    retraction_persistence_steps: int = 20
    retraction_persistence_initiation_threshold: float = 20.0

    legs: tuple[str, ...] = tuple(LEGS)

    def __post_init__(self) -> None:
        if self.cpg_network is None:
            self.cpg_network = make_tripod_cpg_network(self.timestep)
        self._base_intrinsic_freqs = self.cpg_network.intrinsic_freqs.copy()
        self._base_intrinsic_amps = self.cpg_network.intrinsic_amps.copy()
        self.retraction_correction = np.zeros(6, dtype=float)
        self.stumbling_correction = np.zeros(6, dtype=float)
        self.retraction_persistence_counter = np.zeros(6, dtype=float)
        self.last_info: dict[str, np.ndarray | int | None] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        init_phases: np.ndarray | None = None,
        init_magnitudes: np.ndarray | None = None,
    ) -> None:
        if seed is not None:
            self.cpg_network.random_state = np.random.RandomState(seed)
        self.cpg_network.intrinsic_freqs = self._base_intrinsic_freqs.copy()
        self.cpg_network.intrinsic_amps = self._base_intrinsic_amps.copy()
        self.cpg_network.reset(init_phases, init_magnitudes)
        self.retraction_correction[:] = 0
        self.stumbling_correction[:] = 0
        self.retraction_persistence_counter[:] = 0
        self.last_info = {}

    def step(self, sim: Simulation, fly_name: str) -> LocomotionAction:
        """Advance controller state using the current simulation state."""
        leg_to_correct_retraction = self._select_retraction_leg(sim, fly_name)
        if leg_to_correct_retraction is not None:
            if (
                self.retraction_correction[leg_to_correct_retraction]
                > self.retraction_persistence_initiation_threshold
            ):
                self.retraction_persistence_counter[leg_to_correct_retraction] = 1.0

        self._update_persistence_counter()

        stumbling_mask = self._get_stumbling_mask(sim, fly_name)
        self.cpg_network.step()

        joint_angles_by_dof = {}
        adhesion_onoff = []
        net_corrections = np.zeros(6, dtype=float)

        for leg_idx, leg in enumerate(self.legs):
            self._update_retraction_correction(leg_idx, leg_to_correct_retraction)
            self._update_stumbling_correction(leg_idx, stumbling_mask[leg_idx])

            if self.retraction_correction[leg_idx] > 0:
                net_correction = self.retraction_correction[leg_idx]
                self.stumbling_correction[leg_idx] = 0
            else:
                net_correction = self.stumbling_correction[leg_idx]

            phase = self.cpg_network.curr_phases[leg_idx]
            magnitude = self.cpg_network.curr_magnitudes[leg_idx]
            leg_angles = self.preprogrammed_steps.get_joint_angles(
                leg, phase, magnitude
            )

            net_correction = np.clip(net_correction, 0, self.max_correction)
            phase_gain = _step_phase_gain(
                phase % (2 * np.pi),
                self.preprogrammed_steps.swing_period[leg],
                self.swing_extension,
            )
            correction_vector = _CORRECTION_VECTORS[leg[1]]
            if leg.startswith("r"):
                correction_vector = correction_vector * _RIGHT_LEG_CORRECTION_SIGN
            leg_angles = leg_angles + net_correction * phase_gain * correction_vector
            net_corrections[leg_idx] = net_correction * phase_gain

            for dof_idx, dof_spec in enumerate(self.preprogrammed_steps.dofs_per_leg):
                jointdof = _dof_spec_to_jointdof(leg, dof_spec)
                joint_angles_by_dof[jointdof] = leg_angles[dof_idx]
            adhesion_onoff.append(self._get_adhesion_onoff(leg, phase))

        if self.output_dof_order is None:
            from flygym.examples.locomotion.common import (
                get_default_locomotion_dof_order,
            )

            output_dof_order = get_default_locomotion_dof_order()
        else:
            output_dof_order = self.output_dof_order
        joint_angles = np.array(
            [joint_angles_by_dof[dof] for dof in output_dof_order],
            dtype=float,
        )
        adhesion_onoff = np.array(adhesion_onoff, dtype=bool)
        self.last_info = {
            "net_corrections": net_corrections.copy(),
            "retraction_correction": self.retraction_correction.copy(),
            "stumbling_correction": self.stumbling_correction.copy(),
            "stumbling_mask": stumbling_mask.copy(),
            "leg_to_correct_retraction": leg_to_correct_retraction,
        }
        return LocomotionAction(
            joint_angles=joint_angles, adhesion_onoff=adhesion_onoff
        )

    def _select_retraction_leg(self, sim: Simulation, fly_name: str) -> int | None:
        fly = sim.world.fly_lookup[fly_name]
        body_order = fly.get_bodysegs_order()
        positions = sim.get_body_positions(fly_name)
        thorax_z = positions[body_order.index(BodySegment("c_thorax")), 2]
        tarsus_z = np.array(
            [
                positions[body_order.index(BodySegment(f"{leg}_tarsus5")), 2]
                for leg in self.legs
            ]
        )
        end_effector_z_pos = thorax_z - tarsus_z
        sorted_idx = np.argsort(end_effector_z_pos)
        sorted_vals = end_effector_z_pos[sorted_idx]
        if sorted_vals[-1] > sorted_vals[-3] + self.retraction_height_threshold:
            return int(sorted_idx[-1])
        return None

    def _get_stumbling_mask(
        self, sim: Simulation, fly_name: str
    ) -> np.ndarray:
        detected_segments = [
            BodySegment(f"{leg}_{link}")
            for leg in self.legs
            for link in _DETECTED_STUMBLING_LINKS
        ]
        contact_forces = sim.get_bodysegment_contact_forces(
            fly_name, detected_segments, ground_only=True
        ).reshape(len(self.legs), len(_DETECTED_STUMBLING_LINKS), 3)
        heading = _get_fly_heading(sim, fly_name)
        force_proj = np.dot(contact_forces, heading)
        return (force_proj < self.stumbling_force_threshold).any(axis=1)

    def _update_persistence_counter(self) -> None:
        self.retraction_persistence_counter[
            self.retraction_persistence_counter > 0
        ] += 1.0
        self.retraction_persistence_counter[
            self.retraction_persistence_counter > self.retraction_persistence_steps
        ] = 0

    def _get_adhesion_onoff(self, leg: str, phase: float) -> bool:
        swing_start, swing_end = self.preprogrammed_steps.swing_period[leg]
        swing_end += self.swing_extension
        return not (swing_start < phase % (2 * np.pi) < swing_end)

    def _update_retraction_correction(
        self, leg_idx: int, leg_to_correct_retraction: int | None
    ) -> None:
        if (
            leg_idx == leg_to_correct_retraction
            or self.retraction_persistence_counter[leg_idx] > 0
        ):
            self.retraction_correction[leg_idx] += (
                self.retraction_rates[0] * self.timestep
            )
        else:
            self.retraction_correction[leg_idx] = max(
                0,
                self.retraction_correction[leg_idx]
                - self.retraction_rates[1] * self.timestep,
            )

    def _update_stumbling_correction(self, leg_idx: int, is_stumbling: bool) -> None:
        if is_stumbling:
            self.stumbling_correction[leg_idx] += (
                self.stumbling_rates[0] * self.timestep
            )
        else:
            self.stumbling_correction[leg_idx] = max(
                0,
                self.stumbling_correction[leg_idx]
                - self.stumbling_rates[1] * self.timestep,
            )


def _get_fly_heading(sim: Simulation, fly_name: str) -> np.ndarray:
    fly = sim.world.fly_lookup[fly_name]
    thorax_idx = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
    thorax_body_id = sim._internal_bodyids_by_fly[fly_name][thorax_idx]
    thorax_xmat = sim.mj_data.xmat[thorax_body_id].reshape(3, 3)
    return thorax_xmat[:, 0]


def _step_phase_gain(
    phase: float, swing_period: np.ndarray, swing_extension: float = np.pi / 4
) -> float:
    swing_start, swing_end = swing_period
    step_points = np.array(
        [
            swing_start,
            np.mean([swing_start, swing_end]),
            swing_end + swing_extension,
            np.mean([swing_end, 2 * np.pi]),
            2 * np.pi,
        ]
    )
    increment_vals = np.array([0.0, 0.8, 0.0, -0.1, 0.0])
    return float(np.interp(phase, step_points, increment_vals))
