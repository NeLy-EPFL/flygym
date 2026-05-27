"""Tests for the flygym.muscle musculoskeletal-model wrappers."""

import platform

import mujoco as mj
import numpy as np
import pytest

from flygym.compose.fly import ActuatorType
from flygym.compose.world import BaseWorld
from flygym.muscle import (
    DEFAULT_MUSCLE_XML,
    MjWarpCompatibilityReport,
    MuscleFly,
    MuscleWorld,
    build_muscle_gpu_simulation,
    build_muscle_simulation,
    check_mjwarp_compatibility,
)
from flygym.simulation import Simulation


# -----------------------------------------------------------------------------
# MuscleFly
# -----------------------------------------------------------------------------


def test_default_muscle_xml_exists():
    assert DEFAULT_MUSCLE_XML.exists()


def test_muscle_fly_parses_and_exposes_dicts():
    fly = MuscleFly()
    # 15 LF-leg muscles registered under the MUSCLE actuator type
    assert len(fly.muscle_names) == 15
    assert "LFTibia_flex_93434" in fly.muscle_names
    # Bodies + joints are populated with FlyMimic element names
    assert "LFFemur" in fly.bodyseg_to_mjcfbody
    assert "joint_LFCoxa_yaw" in fly.jointdof_to_mjcfjoint
    # No adhesion / anatomical sites / eye cameras by default
    assert fly.leg_to_adhesionactuator == {}
    assert fly.eyecameraname_to_mjcfcamera == {}


def test_muscle_fly_renames_keyframe_to_neutral():
    fly = MuscleFly()
    mj_model, _ = fly.compile()
    neutral_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_KEY, "neutral")
    assert neutral_id >= 0


def test_muscle_fly_compiles_with_15_muscle_actuators():
    fly = MuscleFly()
    mj_model, _ = fly.compile()
    n_muscle = sum(
        1
        for i in range(mj_model.nu)
        if mj_model.actuator_dyntype[i] == mj.mjtDyn.mjDYN_MUSCLE
    )
    assert n_muscle == 15
    assert mj_model.ntendon == 15


def test_add_vision_attaches_eye_cameras():
    fly = MuscleFly()
    added = fly.add_vision()
    assert set(added) == {"LEye", "REye"}
    assert len(fly.eyecameraname_to_mjcfcamera) == 2


# -----------------------------------------------------------------------------
# MuscleWorld + Simulation integration
# -----------------------------------------------------------------------------


def test_build_muscle_simulation_returns_working_sim():
    sim, fly = build_muscle_simulation()
    assert isinstance(sim, Simulation)
    assert sim.mj_model.nu == 15
    # Sensors resolve to sensible shapes on the new body
    assert sim.get_joint_angles("nmf").shape == (14,)
    assert sim.get_body_positions("nmf").shape[1] == 3
    muscle_ids = sim._intern_actuatorids_by_type_by_fly[ActuatorType.MUSCLE]["nmf"]
    assert len(muscle_ids) == 15


def test_muscle_world_exposes_ground_geom():
    fly = MuscleFly()
    world = MuscleWorld(fly)
    assert len(world.ground_geoms) == 1
    assert world.fly_lookup == {"nmf": fly}


def test_muscle_world_is_baseworld_subclass():
    # The GPU path annotates `world: BaseWorld`; honor that contract.
    world = MuscleWorld(MuscleFly())
    assert isinstance(world, BaseWorld)


def test_muscle_world_rejects_add_fly_attachment():
    world = MuscleWorld(MuscleFly())
    with pytest.raises(NotImplementedError, match="self-contained"):
        world._attach_fly_mjcf(MuscleFly(), (0, 0, 0), None)


# -----------------------------------------------------------------------------
# GPU / MuJoCo-Warp helpers (guarded; must work without mujoco_warp)
# -----------------------------------------------------------------------------


def test_check_mjwarp_compatibility_graceful_without_mjwarp():
    # On this (non-CUDA) machine mujoco_warp is absent: the probe must report
    # unavailable rather than raise.
    try:
        import mujoco_warp  # noqa: F401

        has_mjwarp = True
    except ImportError:
        has_mjwarp = False

    report = check_mjwarp_compatibility()
    assert isinstance(report, MjWarpCompatibilityReport)
    if not has_mjwarp:
        assert report.mjwarp_available is False
        assert report.put_model_ok is None
        assert bool(report) is False
    else:
        # If mjwarp IS installed, the probe actually ran put_model.
        assert report.put_model_ok in (True, False)


def test_build_muscle_gpu_simulation_errors_clearly_without_warp():
    try:
        import mujoco_warp  # noqa: F401
        import warp  # noqa: F401

        pytest.skip("warp is installed; skip the missing-extra error path")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="warp"):
        build_muscle_gpu_simulation(n_worlds=4)


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="mujoco offscreen GL only works headlessly on Linux CI",
)
def test_vision_readout_on_muscle_model():
    sim, fly = build_muscle_simulation(add_vision=True)
    sim.warmup(0.02)
    omm = sim.get_ommatidia_readouts("nmf")
    # 2 eyes, 721 ommatidia, pale/yellow channels
    assert omm.shape == (2, 721, 2)


def test_muscle_actuation_steps_and_stays_finite():
    sim, fly = build_muscle_simulation()
    sim.reset()
    muscle_ids = sim._intern_actuatorids_by_type_by_fly[ActuatorType.MUSCLE]["nmf"]
    sim.set_actuator_inputs(
        "nmf", ActuatorType.MUSCLE, np.full(len(muscle_ids), 0.2, dtype=np.float32)
    )
    for _ in range(20):
        sim.step()
    assert np.all(np.isfinite(sim.mj_data.qpos))
    assert np.all(np.isfinite(sim.mj_data.qvel))
