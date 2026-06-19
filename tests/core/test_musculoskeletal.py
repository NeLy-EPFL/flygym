"""Tests for the musculoskeletal (FlyMimic) body-model wrappers."""

import platform

import mujoco as mj
import numpy as np
import pytest

from flygym.compose import (
    ActuatorType,
    BaseWorld,
    DEFAULT_MUSCULOSKELETAL_XML,
    MjWarpCompatibilityReport,
    MusculoskeletalFly,
    MusculoskeletalWorld,
    build_musculoskeletal_gpu_simulation,
    build_musculoskeletal_simulation,
    check_mjwarp_compatibility,
)
from flygym.simulation import Simulation


# -----------------------------------------------------------------------------
# MusculoskeletalFly
# -----------------------------------------------------------------------------


def test_default_musculoskeletal_xml_exists():
    assert DEFAULT_MUSCULOSKELETAL_XML.exists()


def test_muscle_fly_parses_and_exposes_dicts():
    fly = MusculoskeletalFly()
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
    fly = MusculoskeletalFly()
    mj_model, _ = fly.compile()
    neutral_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_KEY, "neutral")
    assert neutral_id >= 0


def test_muscle_fly_compiles_with_15_muscle_actuators():
    fly = MusculoskeletalFly()
    mj_model, _ = fly.compile()
    n_muscle = sum(
        1
        for i in range(mj_model.nu)
        if mj_model.actuator_dyntype[i] == mj.mjtDyn.mjDYN_MUSCLE
    )
    assert n_muscle == 15
    assert mj_model.ntendon == 15


def test_add_vision_attaches_eye_cameras():
    fly = MusculoskeletalFly()
    added = fly.add_vision()
    assert set(added) == {"LEye", "REye"}
    assert len(fly.eyecameraname_to_mjcfcamera) == 2


# -----------------------------------------------------------------------------
# MusculoskeletalWorld + Simulation integration
# -----------------------------------------------------------------------------


def test_build_musculoskeletal_simulation_returns_working_sim():
    sim, fly = build_musculoskeletal_simulation()
    assert isinstance(sim, Simulation)
    assert sim.mj_model.nu == 15
    # Sensors resolve to sensible shapes on the new body
    assert sim.get_joint_angles("nmf").shape == (14,)
    assert sim.get_body_positions("nmf").shape[1] == 3
    muscle_ids = sim._intern_actuatorids_by_type_by_fly[ActuatorType.MUSCLE]["nmf"]
    assert len(muscle_ids) == 15


def test_muscle_world_exposes_ground_geom():
    fly = MusculoskeletalFly()
    world = MusculoskeletalWorld(fly)
    assert len(world.ground_geoms) == 1
    assert world.fly_lookup == {"nmf": fly}


def test_muscle_world_is_baseworld_subclass():
    # The GPU path annotates `world: BaseWorld`; honor that contract.
    world = MusculoskeletalWorld(MusculoskeletalFly())
    assert isinstance(world, BaseWorld)


def test_muscle_world_rejects_add_fly_attachment():
    world = MusculoskeletalWorld(MusculoskeletalFly())
    with pytest.raises(NotImplementedError, match="self-contained"):
        world._attach_fly_mjcf(MusculoskeletalFly(), (0, 0, 0), None)


# -----------------------------------------------------------------------------
# Why MusculoskeletalWorld exists (and isn't "just a fly in a normal world")
# -----------------------------------------------------------------------------
#
# The attach-based worlds (FlatGroundWorld et al.) are written against the
# `BaseFly` contract: they spawn a fresh empty scene, attach the fly with a
# freejoint, and build ground-contact pairs/sensors keyed by `BodySegment`
# enums (via `CONTACT_BODIES_PRESET_CLASS`, `LEG_LINKS`,
# `bodyseg_to_mjcfgeom[BodySegment]`). FlyMimic's musculoskeletal model is a
# *self-contained* MJCF (own floor + lighting + anchored thorax) keyed by raw
# string names and is not a `BaseFly`, so it cannot ride the normal path —
# hence the dedicated thin `MusculoskeletalWorld` adapter.


def test_muscle_fly_is_not_a_basefly():
    # The structural reason the normal worlds can't host it: those worlds use
    # BaseFly-only API (CONTACT_BODIES_PRESET_CLASS, LEG_LINKS, BodySegment-
    # keyed geoms) that MusculoskeletalFly deliberately does not implement.
    from flygym.compose.fly import BaseFly

    fly = MusculoskeletalFly()
    assert not isinstance(fly, BaseFly)
    assert not hasattr(type(fly), "CONTACT_BODIES_PRESET_CLASS")
    # Its geom dict is keyed by FlyMimic's raw MJCF strings, not BodySegment.
    assert all(isinstance(k, str) for k in fly.bodyseg_to_mjcfgeom)


def test_muscle_fly_cannot_be_added_to_a_normal_attach_world():
    from flygym.compose.world import FlatGroundWorld
    from flygym.utils.math import Rotation3D

    world = FlatGroundWorld()
    # add_fly drives the BaseFly-shaped attachment path, which the self-
    # contained muscle model does not satisfy; it must fail rather than
    # silently produce a doubled floor / wrongly free-jointed thorax.
    with pytest.raises(Exception):
        world.add_fly(
            MusculoskeletalFly(), (0, 0, 0), Rotation3D("quat", [1, 0, 0, 0])
        )


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


def test_build_musculoskeletal_gpu_simulation_errors_clearly_without_warp():
    try:
        import mujoco_warp  # noqa: F401
        import warp  # noqa: F401

        pytest.skip("warp is installed; skip the missing-extra error path")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="warp"):
        build_musculoskeletal_gpu_simulation(n_worlds=4)


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="mujoco offscreen GL only works headlessly on Linux CI",
)
def test_vision_readout_on_muscle_model():
    sim, fly = build_musculoskeletal_simulation(add_vision=True)
    sim.warmup(0.02)
    omm = sim.get_ommatidia_readouts("nmf")
    # 2 eyes, 721 ommatidia, pale/yellow channels
    assert omm.shape == (2, 721, 2)


def test_muscle_actuation_steps_and_stays_finite():
    sim, fly = build_musculoskeletal_simulation()
    sim.reset()
    muscle_ids = sim._intern_actuatorids_by_type_by_fly[ActuatorType.MUSCLE]["nmf"]
    sim.set_actuator_inputs(
        "nmf", ActuatorType.MUSCLE, np.full(len(muscle_ids), 0.2, dtype=np.float32)
    )
    for _ in range(20):
        sim.step()
    assert np.all(np.isfinite(sim.mj_data.qpos))
    assert np.all(np.isfinite(sim.mj_data.qvel))
