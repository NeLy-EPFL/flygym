from __future__ import annotations

from importlib.resources import files
import pickle

import numpy as np
from scipy.interpolate import CubicSpline

from flygym.anatomy import JointDOF, LEGS
from flygym_demo.complex_terrain.common import (
    dof_spec_to_jointdof,
    get_default_locomotion_dof_order,
)

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
            path = (
                files("flygym_demo.complex_terrain")
                / "assets/single_steps_untethered.pkl"
            )
        if hasattr(path, "open"):
            with path.open("rb") as f:
                single_steps_data = pickle.load(f)
        else:
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
            # swing_period[leg] = [swing_start, swing_end] in phase units [0, 2π).
            # Phase 0 is the start of the swing. The raw data stores the duration of
            # the stance phase (i.e. the time from swing_end back to 0 = next swing_start),
            # so swing_end = stance_duration / step_duration * 2π and swing_start = 0.
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
        phase = np.asarray(phase, dtype=float)
        phase_is_scalar = phase.shape == ()
        if phase_is_scalar:
            phase = phase[np.newaxis]
        psi_func = self._psi_funcs[leg]
        offset = psi_func(phase) - self.neutral_pos[leg]
        joint_angles = self.neutral_pos[leg] + magnitude * offset
        if phase_is_scalar:
            return joint_angles[:, 0]
        return joint_angles

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
            output_dof_order = get_default_locomotion_dof_order()
        if magnitudes is None:
            magnitudes = np.ones(len(self.legs))

        angles_by_dof = {}
        for leg_idx, leg in enumerate(self.legs):
            leg_angles = self.get_joint_angles(
                leg, phases[leg_idx], magnitudes[leg_idx]
            )
            for dof_idx, dof_spec in enumerate(self.dofs_per_leg):
                angles_by_dof[dof_spec_to_jointdof(leg, dof_spec)] = leg_angles[dof_idx]
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


class FlyBodyPreprogrammedSteps(PreprogrammedSteps):
    """Preprogrammed single-leg steps tailored to the FlyBody anatomy.

    Provenance
    ----------
    The underlying joint-angle trajectories come from
    ``src/flygym_demo/ball_flybody_data/assets/ball_flybody_clip.npz`` -- a
    1-second snippet (100 frames @ 100 fps) of a fly walking on an air-supported
    ball, taken from the original NeuroMechFly v1 dataset and reconstructed
    into per-joint anatomical angles via a SeqIKPy-based inverse-kinematics
    pipeline. The clip is loaded through
    ``flygym_demo.spotlight_data.MotionSnippet``.

    Pipeline (see ``flybody_step_extraction.py``)
    ----------------------------------------------------------------
    1. Replay the clip's joint angles on a *tethered* ``FlyBody`` (LEGS_ONLY
       joints, position actuators, passive tendons, no ground); record the
       world-frame position of each claw (tarsus5) and the thorax at every
       sim step.
    2. Project each claw's position onto the thorax frame to obtain
       body-frame ``(anteroposterior, lateral, vertical)`` coordinates.
    3. For each leg position (F / M / H), hand-pick one step bounded by two
       consecutive **posterior extreme positions (PEPs)** -- local minima of
       the body-frame anteroposterior trace -- so the cycle starts with swing.
    4. Resample the picked slice's 7 leg DOFs onto a common phase grid (with
       end-of-cycle closure for a smooth loop); estimate swing fraction as the
       fraction of timesteps where the body-frame anteroposterior velocity is
       positive (``np.diff(claw_ap) > 0``).
    5. Keep the picked side's trajectory verbatim and fill the opposite side by
       mirroring it across the sagittal plane, giving one canonical
       ``(7, n_phase_bins)`` trajectory and one scalar swing fraction per leg.

    Conventions
    -----------
    Phase 0 corresponds to the PEP (start of swing). ``swing_period[leg] =
    [0, 2π * swing_fraction[leg]]`` carves swing out of the front of the
    cycle; the remainder is stance. The DOF axis matches
    ``_DOFS_PER_LEG`` (the parent class layout): no DOF reshuffling is
    needed compared to ``PreprogrammedSteps``.

    The pickled asset embeds the full provenance under ``meta["description"]``
    and ``meta["source_clip"]``; inspect it with ``pickle.load`` if you need to
    audit a specific build.
    """

    # Nominal step duration kept around so `step_cycle_frequency_hz` stays
    # meaningful for callers that pick CPG intrinsic frequencies from it. Not
    # used to map phase to time anywhere in this class.
    _NOMINAL_CYCLE_DURATION_S = 1.0 / 12.0

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
            path = (
                files("flygym_demo.complex_terrain") / "assets/single_steps_flybody.pkl"
            )
        if hasattr(path, "open"):
            with path.open("rb") as f:
                data = pickle.load(f)
        else:
            with open(path, "rb") as f:
                data = pickle.load(f)

        joint_angles = data["joint_angles"]  # dict: leg -> (7, n_phase_bins)
        swing_fractions = data["swing_fractions"]
        self._length = int(data["meta"]["n_phase_bins"])
        self.duration = self._NOMINAL_CYCLE_DURATION_S
        self._timestep = self.duration / self._length

        # CubicSpline with periodic BC requires the last sample to coincide
        # with the first. The asset stores cycles with `endpoint=False`, so
        # wrap the first sample back onto 2π before fitting.
        phase_inner = np.linspace(0, 2 * np.pi, self._length, endpoint=False)
        phase_grid = np.concatenate([phase_inner, [2 * np.pi]])
        self._psi_funcs = {}
        for leg in self.legs:
            angles = joint_angles[leg]  # (7, n_phase_bins)
            angles_periodic = np.concatenate([angles, angles[:, :1]], axis=1)
            self._psi_funcs[leg] = CubicSpline(
                phase_grid, angles_periodic, axis=1, bc_type="periodic"
            )

        self.neutral_pos = {
            leg: self._psi_funcs[leg](theta_neutral)[:, np.newaxis]
            for leg, theta_neutral in zip(self.legs, neutral_pose_phases)
        }

        # Phase 0 is AEP, so the cycle is laid out as
        #   [0, swing_end]  -> swing  (leg in air)
        #   [swing_end, 2π] -> stance (leg planted)
        self.swing_period = {
            leg: np.array([0.0, float(swing_fractions[leg]) * 2 * np.pi], dtype=float)
            for leg in self.legs
        }
