"""Unit tests for ``flygym_demo.complex_terrain.flybody_step_extraction``.

The module is the library behind ``build_flybody_preprogrammed_steps.ipynb``.
``replay_clip`` is the only heavy part (it drives a MuJoCo ``Simulation``) and is
exercised by the notebook; everything else is pure array / I/O logic and is
covered here: per-leg slicing, PEP detection, selection validation + JSON I/O,
phase resampling with loop closure, swing-fraction computation, the DOF-index
lookup, and the asset assembly + pickle round-trips.
"""

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from flygym_demo.complex_terrain import flybody_step_extraction as fse
from flygym_demo.complex_terrain.preprogrammed import _DOFS_PER_LEG


# ---------------------------------------------------------------------------
# Helpers to build a synthetic ReplayRecording without a Simulation
# ---------------------------------------------------------------------------


def _dof_names_for_legs(legs):
    """Flat DOF-name list (parent|child|axis) matching _DOFS_PER_LEG per leg."""
    names = []
    for leg in legs:
        for parent_link, child_link, axis in _DOFS_PER_LEG:
            parent = "c_thorax" if parent_link == "thorax" else f"{leg}_{parent_link}"
            names.append(f"{parent}|{leg}_{child_link}|{axis}")
    return names


def _make_recording(nsteps=60, legs=("lf", "lm", "lh")):
    names = _dof_names_for_legs(legs)
    n_dofs = len(names)
    rng = np.random.default_rng(0)
    targets = rng.standard_normal((nsteps, n_dofs))
    # claw_body must index all six LEGS by position; fill the columns we use.
    claw_body = rng.standard_normal((nsteps, len(fse.LEGS), 3))
    return fse.ReplayRecording(
        timestep=1e-4,
        targets=targets,
        position_dof_names=names,
        claw_body=claw_body,
        clip_path="/tmp/clip.npz",
        warmup_sec=0.1,
    )


# ---------------------------------------------------------------------------
# ReplayRecording + _dof_name
# ---------------------------------------------------------------------------


def test_replay_recording_n_steps():
    rec = _make_recording(nsteps=42)
    assert rec.n_steps() == 42


def test_dof_name_format():
    dof = SimpleNamespace(
        parent=SimpleNamespace(name="c_thorax"),
        child=SimpleNamespace(name="lf_coxa"),
        axis=SimpleNamespace(value="pitch"),
    )
    assert fse._dof_name(dof) == "c_thorax|lf_coxa|pitch"


# ---------------------------------------------------------------------------
# DOF index lookup + per-leg slicing
# ---------------------------------------------------------------------------


def test_per_leg_dof_indices_maps_in_order():
    names = _dof_names_for_legs(("lf", "lm"))
    indices = fse._per_leg_dof_indices(names, "lm")
    # lm's seven DOFs are the second block, i.e. indices 7..13.
    assert indices == list(range(7, 14))


def test_per_leg_dof_indices_missing_raises():
    names = _dof_names_for_legs(("lf",))
    with pytest.raises(RuntimeError):
        fse._per_leg_dof_indices(names, "rh")


def test_per_leg_target_slice_shape_and_content():
    rec = _make_recording(nsteps=60)
    sl = fse.per_leg_target_slice(rec, "lf", 10, 25)
    assert sl.shape == (15, 7)
    expected_cols = fse._per_leg_dof_indices(rec.position_dof_names, "lf")
    np.testing.assert_array_equal(sl, rec.targets[10:25][:, expected_cols])


def test_per_leg_claw_y_selects_ap_column():
    rec = _make_recording(nsteps=60)
    y = fse.per_leg_claw_y(rec, "lm", 5, 20)
    leg_idx = fse.LEGS.index("lm")
    np.testing.assert_array_equal(y, rec.claw_body[5:20, leg_idx, 0])


# ---------------------------------------------------------------------------
# find_candidate_peps
# ---------------------------------------------------------------------------


def test_find_candidate_peps_locates_minima():
    n = 1000
    t = np.arange(n)
    n_cycles = 5
    ap = np.sin(2 * np.pi * n_cycles * t / n)
    peaks = fse.find_candidate_peps(ap, timestep=1e-3)
    # roughly one PEP (AP minimum) per cycle
    assert 3 <= len(peaks) <= n_cycles + 1
    # detected points are genuine minima (negative side of the sine)
    assert np.all(ap[peaks] < -0.5)


# ---------------------------------------------------------------------------
# Selection validation + JSON I/O
# ---------------------------------------------------------------------------


def _valid_selection():
    return {
        "picks": {
            "F": {"side": "l", "start": 0, "end": 30},
            "M": {"side": "l", "start": 5, "end": 35},
            "H": {"side": "r", "start": 10, "end": 40},
        },
        "note": "provenance preserved",
    }


def test_validate_selection_accepts_valid():
    fse._validate_selection(_valid_selection())  # no raise


def test_validate_selection_missing_picks():
    with pytest.raises(ValueError, match="picks"):
        fse._validate_selection({})


def test_validate_selection_missing_leg_position():
    sel = _valid_selection()
    del sel["picks"]["M"]
    with pytest.raises(ValueError, match="'M'"):
        fse._validate_selection(sel)


def test_validate_selection_missing_field():
    sel = _valid_selection()
    del sel["picks"]["F"]["start"]
    with pytest.raises(ValueError, match="start"):
        fse._validate_selection(sel)


def test_validate_selection_bad_side():
    sel = _valid_selection()
    sel["picks"]["F"]["side"] = "x"
    with pytest.raises(ValueError, match="side"):
        fse._validate_selection(sel)


def test_validate_selection_slice_too_short():
    sel = _valid_selection()
    sel["picks"]["F"]["end"] = sel["picks"]["F"]["start"] + 2
    with pytest.raises(ValueError, match=">= 4"):
        fse._validate_selection(sel)


def test_save_selection_roundtrip_and_preserves_extra_keys(tmp_path):
    sel = _valid_selection()
    out = tmp_path / "sub" / "selection.json"
    fse.save_selection(sel, out)
    assert out.exists()
    with out.open() as f:
        loaded = json.load(f)
    assert loaded == sel


def test_save_selection_validates_before_writing(tmp_path):
    out = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        fse.save_selection({"picks": {}}, out)
    assert not out.exists()


# ---------------------------------------------------------------------------
# _resample_cycle
# ---------------------------------------------------------------------------


def test_resample_cycle_shape_and_closure():
    cycle = np.cumsum(np.ones((40, 7)), axis=0)  # arbitrary monotone waveform
    out = fse._resample_cycle(cycle, n_phase_bins=200)
    assert out.shape == (7, 200)
    # phase 0 equals the first source sample (loop-closure anchor)
    np.testing.assert_allclose(out[:, 0], cycle[0])


def test_resample_cycle_rejects_bad_closure_fraction():
    cycle = np.ones((10, 7))
    with pytest.raises(ValueError):
        fse._resample_cycle(cycle, n_phase_bins=50, closure_fraction=1.0)
    with pytest.raises(ValueError):
        fse._resample_cycle(cycle, n_phase_bins=50, closure_fraction=-0.1)


# ---------------------------------------------------------------------------
# _swing_fraction_from_diff
# ---------------------------------------------------------------------------


def test_swing_fraction_all_increasing():
    assert fse._swing_fraction_from_diff(np.arange(10.0)) == pytest.approx(1.0)


def test_swing_fraction_all_decreasing():
    assert fse._swing_fraction_from_diff(np.arange(10.0)[::-1]) == pytest.approx(0.0)


def test_swing_fraction_half():
    # up for 5 steps, down for 5 steps -> half the velocity samples positive
    claw = np.concatenate([np.arange(6.0), np.arange(5.0)[::-1]])
    assert fse._swing_fraction_from_diff(claw) == pytest.approx(0.5)


def test_swing_fraction_single_sample_is_zero():
    assert fse._swing_fraction_from_diff(np.array([1.0])) == 0.0


# ---------------------------------------------------------------------------
# build_asset_from_selection + pickle round-trips
# ---------------------------------------------------------------------------


def test_build_asset_from_selection_structure():
    rec = _make_recording(nsteps=60)
    selection = {
        "picks": {
            "F": {"side": "l", "start": 0, "end": 30},
            "M": {"side": "l", "start": 5, "end": 35},
            "H": {"side": "l", "start": 10, "end": 40},
        }
    }
    asset = fse.build_asset_from_selection(rec, selection, n_phase_bins=120)

    # every leg (picked + mirrored opposite) is present
    assert set(asset["joint_angles"]) == set(fse.LEGS)
    assert set(asset["swing_fractions"]) == set(fse.LEGS)
    for arr in asset["joint_angles"].values():
        assert arr.shape == (7, 120)

    # opposite side reuses the picked cycle but is an independent copy
    np.testing.assert_array_equal(
        asset["joint_angles"]["lf"], asset["joint_angles"]["rf"]
    )
    assert asset["joint_angles"]["lf"] is not asset["joint_angles"]["rf"]

    meta = asset["meta"]
    assert meta["n_phase_bins"] == 120
    assert meta["sim_timestep"] == rec.timestep
    assert meta["selection"] == selection
    assert meta["cycle_lengths_samples"]["lf"] == 30


def test_build_asset_from_selection_validates():
    rec = _make_recording()
    with pytest.raises(ValueError):
        fse.build_asset_from_selection(rec, {"picks": {}})


def test_replay_recording_pickle_roundtrip(tmp_path):
    rec = _make_recording(nsteps=12)
    out = tmp_path / "nested" / "rec.pkl"
    fse.save_replay_recording(rec, out)
    loaded = fse.load_replay_recording(out)
    assert loaded.n_steps() == 12
    np.testing.assert_array_equal(loaded.targets, rec.targets)
    assert loaded.position_dof_names == rec.position_dof_names


def test_save_asset_roundtrip(tmp_path):
    rec = _make_recording()
    selection = {
        "picks": {
            "F": {"side": "l", "start": 0, "end": 30},
            "M": {"side": "l", "start": 5, "end": 35},
            "H": {"side": "l", "start": 10, "end": 40},
        }
    }
    asset = fse.build_asset_from_selection(rec, selection)
    out = tmp_path / "asset.pkl"
    fse.save_asset(asset, out)
    with out.open("rb") as f:
        loaded = pickle.load(f)
    assert set(loaded["joint_angles"]) == set(fse.LEGS)
