"""Tests for trajectory recording, serialization, and CPU replay (flygym.rendering)."""

import os

import numpy as np
import mujoco as mj
import pytest

from flygym.anatomy import AxisOrder, JointPreset, Skeleton
from flygym.compose.fly import NeuroMechFly
from flygym.compose.pose import KinematicPosePreset
from flygym.compose.world import TetheredWorld
from flygym.utils.math import Rotation3D
from flygym.simulation import Simulation
from flygym.rendering import (
    RecordedTrajectory,
    render_trajectories,
)
from flygym.rendering.recorded_trajectory import _render_trajectory_frames


def _save_all(trajectories, folder):
    """Save a list of trajectories as traj_XXXX.npz in a folder (test helper)."""
    folder.mkdir(parents=True, exist_ok=True)
    for i, traj in enumerate(trajectories):
        traj.save(folder / f"traj_{i:04d}.npz")


def _load_all(folder):
    """Load all trajectories from a folder of traj_*.npz (test helper)."""
    return [RecordedTrajectory.from_file(p) for p in sorted(folder.glob("traj_*.npz"))]


# Rendering (rasterization) needs a headless GL context; skip those on CI runners
# that set SKIP_RENDERING_TESTS=1. Recording and serialization need no GL.
needs_gl = pytest.mark.skipif(
    os.environ.get("SKIP_RENDERING_TESTS") == "1",
    reason="SKIP_RENDERING_TESTS=1 (headless GL unavailable on this CI runner)",
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_sim(name: str) -> tuple[Simulation, str]:
    """Build a Simulation whose fly has a tracking camera; return (sim, cam_name)."""
    pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY
    )
    fly = NeuroMechFly(name=f"{name}_fly")
    fly.add_joints(skeleton, neutral_pose=pose)
    fly.add_tracking_camera(name="trackcam")
    world = TetheredWorld(name=f"{name}_world")
    world.add_fly(
        fly,
        spawn_position=[0, 0, 1.5],
        spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
    )
    sim = Simulation(world)
    return sim, fly.cameraname_to_mjcfcamera["trackcam"].name


@pytest.fixture(scope="module")
def sim_with_camera():
    return make_sim("traj")


@pytest.fixture(scope="module")
def recorded(sim_with_camera):
    """Record a short trajectory (no GL needed -- the recorder stores qpos)."""
    sim, cam_name = sim_with_camera
    rec = sim.set_renderer(
        cam_name, camera_res=(64, 64), output_fps=100, record_trajectory_only=True
    )
    sim.reset()
    for _ in range(200):
        sim.step()
        sim.render_as_needed()
    return rec.recorded_trajectory, sim, cam_name


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class TestTrajectoryRecorder:
    def test_records_frames(self, recorded):
        traj, sim, _ = recorded
        assert traj.n_frames > 0
        assert traj.qpos.shape == (traj.n_frames, sim.mj_model.nq)

    def test_world_id_is_zero(self, recorded):
        traj, _, _ = recorded
        assert traj.world_id == 0

    def test_metadata_captured(self, recorded):
        traj, _, cam_name = recorded
        assert traj.camera_res == (64, 64)
        assert traj.output_fps == 100
        assert cam_name in traj.camera_names

    def test_mocap_recorded_when_present(self, recorded):
        # TetheredWorld tethers the fly via a mocap body, so nmocap > 0 and the
        # recorder must capture mocap poses alongside qpos.
        traj, sim, _ = recorded
        if sim.mj_model.nmocap > 0:
            assert traj.has_mocap
            assert traj.mocap_pos.shape == (traj.n_frames, sim.mj_model.nmocap, 3)
            assert traj.mocap_quat.shape == (traj.n_frames, sim.mj_model.nmocap, 4)
        else:
            assert not traj.has_mocap

    def test_reset_clears_buffer(self, sim_with_camera):
        sim, cam_name = sim_with_camera
        rec = sim.set_renderer(
            cam_name, camera_res=(64, 64), output_fps=100, record_trajectory_only=True
        )
        sim.reset()
        for _ in range(50):
            sim.step()
            sim.render_as_needed()
        assert rec.recorded_trajectory.n_frames > 0
        rec.reset()
        with pytest.raises(RuntimeError):
            _ = rec.recorded_trajectory


# ---------------------------------------------------------------------------
# Serialization (individual self-describing npz files; no model)
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_from_file_roundtrip(self, recorded, tmp_path):
        traj, _, _ = recorded
        path = tmp_path / "one.npz"
        traj.save(path)
        assert path.exists()
        loaded = RecordedTrajectory.from_file(path)
        assert np.array_equal(loaded.qpos, traj.qpos)
        assert loaded.camera_res == traj.camera_res
        assert loaded.camera_names == traj.camera_names
        assert loaded.output_fps == traj.output_fps
        assert loaded.world_id == traj.world_id

    def test_mocap_roundtrip(self, recorded, tmp_path):
        traj, _, _ = recorded
        path = tmp_path / "m.npz"
        traj.save(path)
        loaded = RecordedTrajectory.from_file(path)
        assert loaded.has_mocap == traj.has_mocap
        if traj.has_mocap:
            assert np.array_equal(loaded.mocap_pos, traj.mocap_pos)
            assert np.array_equal(loaded.mocap_quat, traj.mocap_quat)

    def test_save_load_multiple(self, recorded, tmp_path):
        traj, _, _ = recorded
        _save_all([traj, traj], tmp_path)
        assert (tmp_path / "traj_0001.npz").exists()
        trajs = _load_all(tmp_path)
        assert len(trajs) == 2


# ---------------------------------------------------------------------------
# Replay validation (no GL: the guard runs before any rasterization)
# ---------------------------------------------------------------------------


class TestRenderValidation:
    def test_incompatible_model_raises(self, recorded, tmp_path):
        traj, sim, _ = recorded
        # Drop a qpos column so the trajectory no longer fits the model's nq.
        bad = RecordedTrajectory(
            qpos=traj.qpos[:, :-1],
            output_fps=traj.output_fps,
            playback_speed=traj.playback_speed,
            camera_names=traj.camera_names,
            camera_res=traj.camera_res,
        )
        with pytest.raises(ValueError, match="different model"):
            render_trajectories(sim.mj_model, bad, tmp_path / "x.mp4")


# ---------------------------------------------------------------------------
# CPU replay
# ---------------------------------------------------------------------------


@needs_gl
class TestRenderCPU:
    def test_render_from_memory_writes_video(self, recorded, tmp_path):
        traj, sim, _ = recorded
        out = tmp_path / "video.mp4"
        render_trajectories(sim.mj_model, traj, out)
        assert out.exists() and out.stat().st_size > 0

    def test_render_from_saved_file_writes_video(self, recorded, tmp_path):
        traj, sim, _ = recorded
        path = tmp_path / "traj.npz"
        traj.save(path)
        trajs = [RecordedTrajectory.from_file(path)]
        out = tmp_path / "out.mp4"
        render_trajectories(sim.mj_model, trajs, out)
        assert out.exists() and out.stat().st_size > 0

    def test_replay_matches_qpos_consistent_render(self, sim_with_camera):
        """The production CPU replay reproduces a qpos-consistent render bit-for-bit.

        Reference frames are captured in the same run at the same render timepoints,
        rendering after refreshing position kinematics from the post-step qpos (so
        the geometry is consistent with the recorded qpos rather than one step
        stale). The replay path builds its own MjData independently, so a bit-exact
        match validates the full record -> replay round trip.
        """
        sim, cam_name = sim_with_camera
        cam_id = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, cam_name)

        rec = sim.set_renderer(
            cam_name, camera_res=(64, 64), output_fps=100, record_trajectory_only=True
        )
        sim.reset()
        ref_renderer = mj.Renderer(sim.mj_model, 64, 64)
        ref_frames = []
        for _ in range(150):
            sim.step()
            if sim.render_as_needed():  # records qpos at the render cadence
                d = sim.mj_data
                mj.mj_kinematics(sim.mj_model, d)
                mj.mj_camlight(sim.mj_model, d)
                ref_renderer.update_scene(d, cam_id)
                ref_frames.append(ref_renderer.render().copy())
        ref_renderer.close()
        traj = rec.recorded_trajectory

        from flygym.rendering import Renderer

        replay_renderer = Renderer(sim.mj_model, cam_name, camera_res=(64, 64))
        mj_data = mj.MjData(sim.mj_model)
        replay = _render_trajectory_frames(
            sim.mj_model,
            mj_data,
            replay_renderer.mj_renderer,
            traj,
            {cam_name: cam_id},
            replay_renderer.scene_option,
        )[cam_name]
        replay_renderer.close()

        assert len(replay) == len(ref_frames) > 0
        for r_frame, ref_frame in zip(replay, ref_frames):
            assert np.array_equal(r_frame, ref_frame)
