"""Tests for flygym.recording (trajectory recording and CPU rendering)."""

import os

import imageio.v3 as iio
import numpy as np
import pytest

from flygym.anatomy import AxisOrder, JointPreset, Skeleton
from flygym.compose.fly import NeuroMechFly
from flygym.compose.pose import KinematicPosePreset
from flygym.compose.world import TetheredWorld
from flygym.recording import RecordedTrajectory, render_trajectories
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D

needs_gl = pytest.mark.skipif(
    os.environ.get("SKIP_RENDERING_TESTS") == "1",
    reason="SKIP_RENDERING_TESTS=1 (eg. headless GL unavailable on this CI runner)",
)

OUTPUT_FPS = 100
N_STEPS = 200


@pytest.fixture(scope="module")
def sim_and_cam():
    """Simulation of a tethered fly (which uses a mocap body) with a camera."""
    pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY
    )
    fly = NeuroMechFly(name="rec_fly")
    fly.add_joints(skeleton, neutral_pose=pose)
    cam = fly.add_tracking_camera(name="trackcam")
    world = TetheredWorld(name="rec_world")
    world.add_fly(fly, [0, 0, 1.5], Rotation3D("quat", [1, 0, 0, 0]))
    sim = Simulation(world)
    yield sim, cam.name
    sim.close()


@pytest.fixture(scope="module")
def recording(sim_and_cam):
    """Record a short trajectory, keeping the live state at each recorded frame."""
    sim, _ = sim_and_cam
    recorder = sim.set_trajectory_recorder(output_fps=OUTPUT_FPS)
    sim.reset()
    expected_time, expected_qpos = [], []
    for _ in range(N_STEPS):
        sim.step()
        if sim.record_as_needed():
            expected_time.append(sim.mj_data.time)
            expected_qpos.append(sim.mj_data.qpos.copy())
    return recorder.trajectory, np.array(expected_time), np.array(expected_qpos)


class TestTrajectoryRecorder:
    def test_records_state_at_frame_times(self, recording):
        traj, expected_time, expected_qpos = recording
        assert traj.n_frames == len(expected_time) > 1
        assert np.array_equal(traj.time, expected_time)
        assert np.array_equal(traj.qpos, expected_qpos)
        assert np.allclose(np.diff(traj.time), traj.playback_speed / OUTPUT_FPS)

    def test_records_mocap(self, recording, sim_and_cam):
        traj, _, _ = recording
        sim, _ = sim_and_cam
        assert sim.mj_model.nmocap > 0
        assert traj.mocap_pos.shape == (traj.n_frames, sim.mj_model.nmocap, 3)
        assert traj.mocap_quat.shape == (traj.n_frames, sim.mj_model.nmocap, 4)

    def test_reset_discards_frames(self, sim_and_cam):
        sim, _ = sim_and_cam
        recorder = sim.set_trajectory_recorder(output_fps=OUTPUT_FPS)
        sim.step()
        assert sim.record_as_needed()
        sim.reset()
        with pytest.raises(RuntimeError, match="No frames"):
            recorder.trajectory

    @needs_gl
    def test_same_frame_times_as_renderer(self, sim_and_cam):
        sim, cam_name = sim_and_cam
        renderer = sim.set_renderer(
            cam_name, camera_res=(64, 64), output_fps=OUTPUT_FPS
        )
        recorder = sim.set_trajectory_recorder(output_fps=OUTPUT_FPS)
        sim.reset()
        for _ in range(N_STEPS):
            sim.step()
            assert sim.render_as_needed() == sim.record_as_needed()
        assert len(renderer.frames[cam_name]) == recorder.trajectory.n_frames


def test_save_load_roundtrip(recording, tmp_path):
    traj, _, _ = recording
    traj.save(tmp_path / "traj.npz")
    loaded = RecordedTrajectory.from_file(tmp_path / "traj.npz")
    for field in ("time", "qpos", "mocap_pos", "mocap_quat"):
        assert np.array_equal(getattr(loaded, field), getattr(traj, field))
    assert loaded.output_fps == traj.output_fps
    assert loaded.playback_speed == traj.playback_speed
    assert loaded.world_id == traj.world_id


class TestRenderTrajectories:
    def test_mismatched_model_raises(self, recording, sim_and_cam, tmp_path):
        traj, _, _ = recording
        sim, cam_name = sim_and_cam
        bad_traj = RecordedTrajectory(
            time=traj.time,
            qpos=traj.qpos[:, :-1],
            output_fps=traj.output_fps,
            playback_speed=traj.playback_speed,
            mocap_pos=traj.mocap_pos,
            mocap_quat=traj.mocap_quat,
        )
        with pytest.raises(ValueError, match="does not match the model"):
            render_trajectories(sim.mj_model, bad_traj, cam_name, tmp_path / "x.mp4")

    def test_unknown_camera_raises(self, recording, sim_and_cam, tmp_path):
        traj, _, _ = recording
        sim, _ = sim_and_cam
        with pytest.raises(ValueError, match="not found"):
            render_trajectories(sim.mj_model, traj, "nonexistent", tmp_path / "x.mp4")

    @needs_gl
    def test_single_trajectory_to_file(self, recording, sim_and_cam, tmp_path):
        traj, _, _ = recording
        sim, cam_name = sim_and_cam
        out = tmp_path / "video.mp4"
        render_trajectories(sim.mj_model, traj, cam_name, out, camera_res=(64, 64))
        assert iio.imread(out).shape[0] == traj.n_frames

    @needs_gl
    def test_multiple_trajectories_to_dir(self, recording, sim_and_cam, tmp_path):
        traj, _, _ = recording
        sim, cam_name = sim_and_cam
        render_trajectories(
            sim.mj_model, [traj, traj], cam_name, tmp_path, camera_res=(64, 64)
        )
        videos = sorted(tmp_path.rglob("*.mp4"))
        assert [v.parent.name for v in videos] == ["traj0000_world0", "traj0001_world0"]
