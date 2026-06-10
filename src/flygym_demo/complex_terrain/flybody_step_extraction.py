"""Helper functions for ``build_flybody_preprogrammed_steps.ipynb``.

This module is a *library only* -- no CLI, no script entrypoint. It groups the
helpers the curation notebook needs in one place so the notebook stays readable.

The notebook drives the full two-step pipeline:

1. **Replay & select.** Replay
   ``src/flygym_demo/ball_flybody_data/assets/ball_flybody_clip.npz`` (1 s at
   100 fps from the NeuroMechFly v1 walking-on-ball dataset, joint angles
   reconstructed by a SeqIKPy-based IK pipeline) on a tethered ``FlybodyFly``,
   then pick one step per leg position (F / M / H) bounded by two consecutive
   posterior extreme positions (PEPs) of the claw -- so the cycle starts with
   swing.

2. **Build the asset.** Resample each picked slice onto a phase grid, compute
   the swing fraction from the sign of the anteroposterior velocity, mirror
   the picked side onto the opposite side, and pickle the result to
   ``single_steps_flybody.pkl``.

The helpers below cover: per-timestep replay recording, per-leg slicing, PEP
candidate detection, selection JSON I/O, left-right mirroring, and asset
construction.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.signal import find_peaks
from tqdm import trange

from flygym import Simulation
from flygym.compose import ActuatorType
from flygym.compose.fly import FlybodyFly
from flygym.flybody import FlybodyBodySegment
from flygym_demo.complex_terrain.preprogrammed import _DOFS_PER_LEG
from flygym_demo.spotlight_data import MotionSnippet


LEGS: tuple[str, ...] = ("lf", "lm", "lh", "rf", "rm", "rh")
LEG_POSITIONS: tuple[str, ...] = ("F", "M", "H")  # canonical leg-type keys
N_PHASE_BINS_DEFAULT = 200
WARMUP_SEC_DEFAULT = 0.1
# PEP candidate filtering: lower bound on plausible step period (s).
MIN_STEP_PERIOD_SEC = 0.04


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

@dataclass
class ReplayRecording:
    """Per-timestep state captured during the tethered replay."""

    timestep: float
    targets: np.ndarray                 # (nsteps, n_position_actuators)
    position_dof_names: list[str]       # parent|child|axis strings
    claw_body: np.ndarray               # (nsteps, 6, 3) -- body-frame AP/lat/vert
    clip_path: str                      # absolute string path for provenance
    warmup_sec: float

    def n_steps(self) -> int:
        return self.targets.shape[0]


def _dof_name(dof) -> str:
    """Stable string representation of a JointDOF (parent|child|axis)."""
    return f"{dof.parent.name}|{dof.child.name}|{dof.axis.value}"


def replay_clip(
    clip_path: Path | str,
    *,
    fly: FlybodyFly,
    sim: Simulation,
    warmup_sec: float = WARMUP_SEC_DEFAULT,
    show_progress: bool = True,
) -> tuple[ReplayRecording, FlybodyFly, Simulation]:
    """Replay the clip on a tethered Flybody and record claw + thorax kinematics.

    The caller is responsible for building the fly + sim (so it can set up
    cameras, renderers, etc. to taste); we only drive the replay loop.
    """
    clip_path = Path(clip_path)
    timestep = float(sim.mj_model.opt.timestep)

    snippet = MotionSnippet(clip_path, angles_global2anatomical=False)
    position_dofs = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    targets = snippet.get_joint_angles(
        output_timestep=timestep,
        output_dof_order=position_dofs,
        sgfilter_window_sec=0.05,
    )
    nsteps = targets.shape[0]

    n_tendon = len(fly.get_actuated_jointdofs_order(ActuatorType.TENDON))
    zero_tendon = np.zeros(n_tendon, dtype=np.float32)

    body_order = fly.get_bodysegs_order()
    thorax_idx = body_order.index(FlybodyBodySegment("c_thorax"))
    claw_idx = np.array(
        [body_order.index(FlybodyBodySegment(f"{leg}_tarsus5")) for leg in LEGS],
        dtype=np.int64,
    )

    claw_world = np.empty((nsteps, len(LEGS), 3))
    thorax_world = np.empty((nsteps, 3))

    sim.reset()
    sim.set_tendon_actuator_inputs(fly.name, zero_tendon)
    sim.warmup()
    warmup_steps = max(1, int(warmup_sec / timestep))

    iterator = trange(nsteps, desc="Replaying clip") if show_progress else range(nsteps)
    for step_idx in iterator:
        ramp = min(1.0, step_idx / warmup_steps)
        sim.set_actuator_inputs(
            fly.name, ActuatorType.POSITION, targets[step_idx] * ramp
        )
        sim.step_with_profile()
        positions = sim.get_body_positions(fly.name)
        claw_world[step_idx] = positions[claw_idx]
        thorax_world[step_idx] = positions[thorax_idx]
        sim.render_as_needed_with_profile()

    # Project claw positions into the body frame. Body-frame columns of
    # `(claw - thorax) @ xmat` are (forward, lateral, up) by MuJoCo convention.
    thorax_body_id = sim._internal_bodyids_by_fly[fly.name][thorax_idx]
    thorax_xmat = sim.mj_data.xmat[thorax_body_id].reshape(3, 3).copy()
    claw_body = (claw_world - thorax_world[:, None, :]) @ thorax_xmat

    return (
        ReplayRecording(
            timestep=timestep,
            targets=targets,
            position_dof_names=[_dof_name(d) for d in position_dofs],
            claw_body=claw_body,
            clip_path=str(clip_path.resolve()),
            warmup_sec=warmup_sec,
        ),
        fly,
        sim,
    )


def save_replay_recording(recording: ReplayRecording, out_path: Path | str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(recording, f)


def load_replay_recording(path: Path | str) -> ReplayRecording:
    with Path(path).open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Per-leg slicing and PEP candidate detection
# ---------------------------------------------------------------------------

def find_candidate_peps(
    claw_anteroposterior: np.ndarray, timestep: float
) -> np.ndarray:
    """Return sample indices of plausible PEPs in a single leg's AP trace.

    PEPs are detected as local minima of the body-frame anteroposterior
    coordinate (equivalent: local maxima of its negation). These are
    *suggestions* to anchor the eye on the selection plot -- not the final
    segmentation.
    """
    distance = max(1, int(MIN_STEP_PERIOD_SEC / timestep))
    peaks, _ = find_peaks(
        -claw_anteroposterior, distance=distance, prominence=1e-3
    )
    return peaks


def per_leg_target_slice(
    recording: ReplayRecording, leg: str, start: int, end: int
) -> np.ndarray:
    """Return the ``(cycle_len, 7)`` target slice for one leg.

    The 7 DOFs are ordered as ``_DOFS_PER_LEG`` (matches the asset schema and
    the parent ``PreprogrammedSteps``).
    """
    indices = _per_leg_dof_indices(recording.position_dof_names, leg)
    return recording.targets[start:end, indices]


def per_leg_claw_y(
    recording: ReplayRecording, leg: str, start: int, end: int
) -> np.ndarray:
    """Body-frame anteroposterior claw coordinate over a slice."""
    leg_idx = LEGS.index(leg.lower())
    return recording.claw_body[start:end, leg_idx, 0]


# ---------------------------------------------------------------------------
# Selection JSON I/O
# ---------------------------------------------------------------------------

def save_selection(selection: dict, out_path: Path | str) -> None:
    """Save the user's leg-position picks to JSON.

    ``selection`` must have a ``"picks"`` key mapping each of ``"F"``, ``"M"``,
    ``"H"`` to ``{"side": "l"|"r", "start": int, "end": int}``. Additional keys
    are preserved verbatim (e.g. provenance notes).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_selection(selection)
    with out_path.open("w") as f:
        json.dump(selection, f, indent=2)


def _validate_selection(selection: dict) -> None:
    if "picks" not in selection:
        raise ValueError("selection must contain a 'picks' key.")
    picks = selection["picks"]
    for leg_pos in LEG_POSITIONS:
        if leg_pos not in picks:
            raise ValueError(f"selection['picks'] missing leg position '{leg_pos}'.")
        entry = picks[leg_pos]
        for key in ("side", "start", "end"):
            if key not in entry:
                raise ValueError(
                    f"selection['picks']['{leg_pos}'] missing field '{key}'."
                )
        if entry["side"] not in ("l", "r"):
            raise ValueError(
                f"selection['picks']['{leg_pos}']['side'] must be 'l' or 'r'."
            )
        if int(entry["end"]) - int(entry["start"]) < 4:
            raise ValueError(
                f"selection['picks']['{leg_pos}']: end-start must be >= 4 samples."
            )


# ---------------------------------------------------------------------------
# Left <-> right mirroring
# ---------------------------------------------------------------------------

def mirror_leg_joint_angles(angles: np.ndarray) -> np.ndarray:
    """Mirror a 7-DOF leg trajectory across the body's sagittal plane.

    The clip stores joint angles in the SeqIKPy / global convention where
    symmetric left/right motion has opposite roll and yaw signs. Mirroring is
    therefore: copy pitch DOFs verbatim, flip roll and yaw signs.

    Args:
        angles: ``(7, n_phase_bins)`` array ordered as ``_DOFS_PER_LEG``.
    Returns:
        Same-shape mirrored array. The function is its own inverse.
    """
    if angles.shape[0] != len(_DOFS_PER_LEG):
        raise ValueError(
            f"Expected first axis of length {len(_DOFS_PER_LEG)}, got {angles.shape}."
        )
    mirrored = angles.copy()
    for dof_idx, (_, _, axis) in enumerate(_DOFS_PER_LEG):
        if axis in ("roll", "yaw"):
            mirrored[dof_idx] *= 1
    return mirrored


# ---------------------------------------------------------------------------
# Asset construction
# ---------------------------------------------------------------------------

def _resample_cycle(
    leg_targets: np.ndarray,
    n_phase_bins: int,
    *,
    closure_fraction: float = 0.05,
) -> np.ndarray:
    """Resample ``(cycle_len, 7)`` onto ``(7, n_phase_bins)`` via linear interp,
    with end-of-cycle closure so the result is C0-continuous when looped.

    The PEP-to-PEP slice picked by the user is unlikely to close exactly in
    joint-angle space: the AP-velocity zero-crossing aligns one DOF (claw
    position) to its peak, but the seven leg DOFs each have their own waveform
    and discrete sample picks rarely land on every minimum. Naive resampling
    therefore produces a jump at the phase-2π / phase-0 wrap.

    Fix: discard the last ``closure_fraction`` of source samples (end of stance)
    and treat phase ``2π`` as exactly equal to sample 0. ``np.interp`` then
    linearly ramps the last kept sample back to sample 0 across the sacrificed
    tail, giving a smooth loop. Default 5 % is enough to absorb the misalignment
    while leaving the rest of the cycle untouched.
    """
    cycle_len, n_dofs = leg_targets.shape
    if closure_fraction < 0 or closure_fraction >= 1:
        raise ValueError("closure_fraction must be in the range [0, 1).")
    real_cutoff = max(2, int(np.floor(cycle_len * (1.0 - closure_fraction))))

    # Source: first `real_cutoff` samples at their natural phases + a closure
    # anchor at phase 2π whose value is leg_targets[0].
    source_phases = np.empty(real_cutoff + 1)
    source_phases[:real_cutoff] = np.arange(real_cutoff) / cycle_len * 2 * np.pi
    source_phases[real_cutoff] = 2 * np.pi

    source_values = np.empty((real_cutoff + 1, n_dofs))
    source_values[:real_cutoff] = leg_targets[:real_cutoff]
    source_values[real_cutoff] = leg_targets[0]

    phase_grid = np.linspace(0, 2 * np.pi, n_phase_bins, endpoint=False)
    out = np.empty((n_dofs, n_phase_bins))
    for dof_i in range(n_dofs):
        out[dof_i] = np.interp(phase_grid, source_phases, source_values[:, dof_i])
    return out


def _swing_fraction_from_diff(claw_y: np.ndarray) -> float:
    """Fraction of cycle where the anteroposterior velocity is positive.

    Under PEP-segmented cycles (phase 0 = PEP), positive velocity = claw
    moving anteriorly = leg-lifted swing phase.
    """
    claw_y_vel = np.diff(claw_y)
    swing_mask = claw_y_vel > 0
    return float(swing_mask.mean()) if swing_mask.size else 0.0


def build_asset_from_selection(
    recording: ReplayRecording,
    selection: dict,
    *,
    n_phase_bins: int = N_PHASE_BINS_DEFAULT,
) -> dict:
    """Assemble the per-leg cycle dict from the user's picks.

    Each leg position (F/M/H) contributes one canonical cycle. The picked side
    keeps its data verbatim; the opposite side is filled by mirroring.
    """
    _validate_selection(selection)
    joint_angles_per_leg: dict[str, np.ndarray] = {}
    swing_fractions: dict[str, float] = {}
    cycle_lengths: dict[str, int] = {}

    for leg_pos in LEG_POSITIONS:
        entry = selection["picks"][leg_pos]
        side, start, end = entry["side"], int(entry["start"]), int(entry["end"])
        picked_leg = f"{side}{leg_pos.lower()}"
        opposite_leg = f"{'r' if side == 'l' else 'l'}{leg_pos.lower()}"

        leg_targets = per_leg_target_slice(recording, picked_leg, start, end)
        claw_y = per_leg_claw_y(recording, picked_leg, start, end)
        picked_cycle = _resample_cycle(leg_targets, n_phase_bins)
        swing_frac = _swing_fraction_from_diff(claw_y)

        joint_angles_per_leg[picked_leg] = picked_cycle
        joint_angles_per_leg[opposite_leg] = mirror_leg_joint_angles(picked_cycle)
        swing_fractions[picked_leg] = swing_frac
        swing_fractions[opposite_leg] = swing_frac
        cycle_lengths[picked_leg] = end - start
        cycle_lengths[opposite_leg] = end - start

    description = (
        "Three hand-picked steps (one per leg position) from a tethered Flybody "
        "replay of the NeuroMechFly v1 walking-on-ball clip "
        f"({Path(recording.clip_path).name}, 100 frames @ 100 fps, anatomical-"
        "convention joint angles via SeqIKPy IK). Each pick supplies the chosen "
        "side; the opposite side is mirrored (roll/yaw sign flip in SeqIKPy/"
        "global convention). Phase 0 = PEP / start of swing; swing fraction "
        "computed as the fraction of timesteps where the body-frame "
        "anteroposterior velocity is positive (np.diff(claw_ap) > 0). Built by "
        "src/flygym_demo/complex_terrain/flybody_step_extraction.py from a "
        "selection JSON produced via "
        "src/flygym_demo/complex_terrain/build_flybody_preprogrammed_steps.ipynb."
    )

    return {
        "joint_angles": joint_angles_per_leg,
        "swing_fractions": swing_fractions,
        "meta": {
            "description": description,
            "source_clip": recording.clip_path,
            "n_phase_bins": n_phase_bins,
            "cycle_lengths_samples": cycle_lengths,
            "sim_timestep": recording.timestep,
            "warmup_sec": recording.warmup_sec,
            "selection": selection,
        },
    }


def save_asset(asset: dict, out_path: Path | str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(asset, f)


# ---------------------------------------------------------------------------
# Private DOF index lookup
# ---------------------------------------------------------------------------

def _per_leg_dof_indices(
    position_dof_names: Iterable[str], leg: str
) -> list[int]:
    """Map ``_DOFS_PER_LEG`` to indices in the recording's flat DOF ordering."""
    names = list(position_dof_names)
    indices = []
    for parent_link, child_link, axis in _DOFS_PER_LEG:
        parent_name = "c_thorax" if parent_link == "thorax" else f"{leg}_{parent_link}"
        child_name = f"{leg}_{child_link}"
        target = f"{parent_name}|{child_name}|{axis}"
        try:
            indices.append(names.index(target))
        except ValueError as e:
            raise RuntimeError(
                f"Could not locate DOF {target!r} in the recording's "
                f"position_dof_names."
            ) from e
    return indices
