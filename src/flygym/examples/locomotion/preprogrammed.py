from __future__ import annotations

import pickle

import numpy as np
from scipy.interpolate import CubicSpline

from flygym import assets_dir
from flygym.anatomy import BodySegment, JointDOF, LEGS, RotationAxis

_LEGACY_LEG = {leg: leg.upper() for leg in LEGS}
_LEG_BY_LEGACY = {v: k for k, v in _LEGACY_LEG.items()}

_LEGACY_DOF_NAMES = [
    "Coxa",
    "Coxa_roll",
    "Coxa_yaw",
    "Femur",
    "Femur_roll",
    "Tibia",
    "Tarsus1",
]

_DOFS_PER_LEG = [
    ("thorax", "coxa", "pitch"),
    ("thorax", "coxa", "roll"),
    ("thorax", "coxa", "yaw"),
    ("coxa", "trochanterfemur", "pitch"),
    ("coxa", "trochanterfemur", "roll"),
    ("trochanterfemur", "tibia", "pitch"),
    ("tibia", "tarsus1", "pitch"),
]


class PreprogrammedSteps:
    """Preprogrammed single-leg steps extracted from v1 walking recordings.

    Angles are exposed in FlyGym v2's anatomical convention. In particular, right-leg
    roll and yaw are sign-flipped relative to the legacy v1 data.
    """

    legs = LEGS
    dofs_per_leg = _DOFS_PER_LEG

    def __init__(
        self,
        path=None,
        neutral_pose_phases: tuple[float, float, float, float, float, float] = (
            np.pi,
            np.pi,
            np.pi,
            np.pi,
            np.pi,
            np.pi,
        ),
    ) -> None:
        if path is None:
            path = assets_dir / "behavior/single_steps_untethered.pkl"
        with open(path, "rb") as f:
            single_steps_data = pickle.load(f)

        self._length = len(single_steps_data["joint_LFCoxa"])
        self._timestep = single_steps_data["meta"]["timestep"]
        self.duration = self._length * self._timestep

        phase_grid = np.linspace(0, 2 * np.pi, self._length)
        self._psi_funcs = {}
        for leg in self.legs:
            legacy_leg = _LEGACY_LEG[leg]
            joint_angles = np.array(
                [
                    single_steps_data[f"joint_{legacy_leg}{dof}"]
                    for dof in _LEGACY_DOF_NAMES
                ],
                dtype=float,
            )
            if leg.startswith("r"):
                for dof_idx, (_, _, axis) in enumerate(self.dofs_per_leg):
                    if axis in ("roll", "yaw"):
                        joint_angles[dof_idx] *= -1
            self._psi_funcs[leg] = CubicSpline(
                phase_grid, joint_angles, axis=1, bc_type="periodic"
            )

        self.neutral_pos = {
            leg: self._psi_funcs[leg](theta_neutral)[:, np.newaxis]
            for leg, theta_neutral in zip(self.legs, neutral_pose_phases)
        }

        swing_stance_time_dict = single_steps_data["swing_stance_time"]
        self.swing_period = {}
        for leg in self.legs:
            legacy_leg = _LEGACY_LEG[leg]
            my_swing_period = np.array(
                [0, swing_stance_time_dict["stance"][legacy_leg]],
                dtype=float,
            )
            my_swing_period /= self.duration
            my_swing_period *= 2 * np.pi
            self.swing_period[leg] = my_swing_period

    @property
    def step_cycle_frequency_hz(self) -> float:
        """Frequency at which one oscillator cycle matches one recorded step."""
        return 1.0 / self.duration

    def get_joint_angles(
        self,
        leg: str,
        phase: float | np.ndarray,
        magnitude: float | np.ndarray = 1,
    ) -> np.ndarray:
        """Get seven per-leg joint angles at a stepping phase."""
        leg = leg.lower()
        if leg not in self.legs:
            raise ValueError(f"Unknown leg '{leg}'. Expected one of {self.legs}.")
        phase = np.asarray(phase)
        if phase.shape == ():
            phase = phase[np.newaxis]
        psi_func = self._psi_funcs[leg]
        offset = psi_func(phase) - self.neutral_pos[leg]
        joint_angles = self.neutral_pos[leg] + magnitude * offset
        return joint_angles.squeeze()

    def get_adhesion_onoff(self, leg: str, phase: float) -> bool:
        """Return whether adhesion should be on for one leg at a phase."""
        swing_start, swing_end = self.swing_period[leg.lower()]
        return not (swing_start < phase % (2 * np.pi) < swing_end)

    def get_joint_angles_by_dof_order(
        self,
        phases: np.ndarray,
        magnitudes: np.ndarray | None = None,
        output_dof_order: list[JointDOF] | None = None,
    ) -> np.ndarray:
        """Return all leg angles in a requested FlyGym v2 DOF order."""
        if output_dof_order is None:
            from flygym.examples.locomotion.common import (
                get_default_locomotion_dof_order,
            )

            output_dof_order = get_default_locomotion_dof_order()
        if magnitudes is None:
            magnitudes = np.ones(len(self.legs))

        angles_by_dof = {}
        for leg_idx, leg in enumerate(self.legs):
            leg_angles = self.get_joint_angles(
                leg, phases[leg_idx], magnitudes[leg_idx]
            )
            for dof_idx, dof_spec in enumerate(self.dofs_per_leg):
                angles_by_dof[_dof_spec_to_jointdof(leg, dof_spec)] = leg_angles[
                    dof_idx
                ]
        return np.array([angles_by_dof[dof] for dof in output_dof_order], dtype=float)

    def get_adhesion_onoff_by_phase(self, phases: np.ndarray) -> np.ndarray:
        """Return per-leg adhesion flags ordered as ``fly.get_legs_order()``."""
        return np.array(
            [
                self.get_adhesion_onoff(leg, phase)
                for leg, phase in zip(self.legs, phases)
            ],
            dtype=bool,
        )

    def default_pose_by_dof_order(
        self, output_dof_order: list[JointDOF] | None = None
    ) -> np.ndarray:
        """Return the neutral preprogrammed step pose in v2 actuator order."""
        phases = np.full(len(self.legs), np.pi)
        magnitudes = np.ones(len(self.legs))
        return self.get_joint_angles_by_dof_order(phases, magnitudes, output_dof_order)

    @property
    def default_pose(self) -> np.ndarray:
        """Default pose ordered like the default v2 active leg actuators."""
        return self.default_pose_by_dof_order()


def _dof_spec_to_jointdof(leg: str, dof_spec: tuple[str, str, str]) -> JointDOF:
    parent_link, child_link, axis = dof_spec
    if parent_link == "thorax":
        parent = BodySegment("c_thorax")
    else:
        parent = BodySegment(f"{leg}_{parent_link}")
    child = BodySegment(f"{leg}_{child_link}")
    return JointDOF(parent, child, RotationAxis(axis))
