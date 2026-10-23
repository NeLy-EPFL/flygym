"""Tests for flygym.warp.recording (GPU trajectory recording and rendering)."""

import warnings

import imageio.v3 as iio
import numpy as np
import pytest

# These tests require the optional warp (GPU) extra; tag them so they can be
# excluded with ``-m "not warp"``, and skip the whole module if warp is absent.
pytestmark = pytest.mark.warp
pytest.importorskip("warp")

from flygym.recording import render_trajectories
from flygym.warp.recording import render_trajectories_gpu
from flygym.warp.rendering import modify_world_for_batch_rendering

OUTPUT_FPS = 100
N_STEPS = 200
RECORDED_WORLDS = [0, 2]


@pytest.fixture(scope="module")
def recording(gpu_sim_factory):
    """Record 2 of 4 worlds, keeping the live state at each recorded frame."""
    sim, _, cam = gpu_sim_factory(n_worlds=4, fly_name="rec_gpu_fly")
    # Make worlds diverge so that recording the wrong world would be detected
    sim.reset()
    rng = np.random.default_rng(0)
    qpos = np.tile(sim.mj_data.qpos, (4, 1))
    qpos[:, 7:] += rng.normal(scale=0.1, size=(4, qpos.shape[1] - 7))
    sim.set_state(qpos=qpos)

    recorder = sim.set_trajectory_recorder(
        output_fps=OUTPUT_FPS, worlds=RECORDED_WORLDS
    )
    expected_qpos = []
    for _ in range(N_STEPS):
        sim.step()
        if sim.record_as_needed():
            expected_qpos.append(sim.mjw_data.qpos.numpy()[RECORDED_WORLDS])
    return sim, cam, recorder.trajectories, np.array(expected_qpos)


class TestWarpTrajectoryRecorder:
    def test_one_trajectory_per_selected_world(self, recording):
        _, _, trajs, _ = recording
        assert [t.world_id for t in trajs] == RECORDED_WORLDS

    def test_records_state_of_selected_worlds(self, recording):
        _, _, trajs, expected_qpos = recording
        for i, traj in enumerate(trajs):
            assert traj.n_frames == len(expected_qpos) > 1
            assert np.array_equal(traj.qpos, expected_qpos[:, i])
        assert not np.allclose(trajs[0].qpos, trajs[1].qpos)

    def test_empty_worlds_raises(self, recording):
        sim, _, _, _ = recording
        with pytest.raises(ValueError, match="world"):
            sim.set_trajectory_recorder(worlds=[])

    def test_same_frame_times_as_renderer(self, gpu_sim_factory):
        sim, _, cam = gpu_sim_factory(n_worlds=2, fly_name="rec_render_gpu_fly")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            renderer = sim.set_renderer(
                cam, camera_res=(64, 64), output_fps=OUTPUT_FPS, worlds=[0]
            )
        recorder = sim.set_trajectory_recorder(output_fps=OUTPUT_FPS)
        sim.reset()
        for _ in range(N_STEPS):
            sim.step()
            assert sim.render_as_needed() == sim.record_as_needed()
        assert len(renderer._frames) == recorder.trajectories[0].n_frames


class TestRenderTrajectories:
    def test_gpu(self, recording, tmp_path):
        sim, cam, trajs, _ = recording
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modify_world_for_batch_rendering(sim.world)
        batch_model = sim.world.compile()[0]
        # Batch size that does not divide the total number of frames
        render_trajectories_gpu(
            batch_model, trajs, cam, tmp_path, camera_res=(64, 64), worlds_per_batch=7
        )
        videos = sorted(tmp_path.rglob("*.mp4"))
        assert len(videos) == len(trajs)
        for video, traj in zip(videos, trajs):
            assert iio.imread(video).shape[0] == traj.n_frames

    def test_gpu_recorded_on_cpu(self, recording, tmp_path):
        sim, cam, trajs, _ = recording
        out = tmp_path / "video.mp4"
        render_trajectories(sim.mj_model, trajs[0], cam, out, camera_res=(64, 64))
        assert iio.imread(out).shape[0] == trajs[0].n_frames


@pytest.mark.parametrize("use_gpu_batch_rendering", [True, False])
def test_matches_live_rendering(gpu_sim_factory, use_gpu_batch_rendering):
    sim, _, cam = gpu_sim_factory(
        n_worlds=4, fly_name=f"live_vs_rec_fly_{use_gpu_batch_rendering}"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        renderer = sim.set_renderer(
            cam,
            camera_res=(64, 64),
            output_fps=OUTPUT_FPS,
            worlds=RECORDED_WORLDS,
            use_gpu_batch_rendering=use_gpu_batch_rendering,
        )
    recorder = sim.set_trajectory_recorder(
        output_fps=OUTPUT_FPS, worlds=RECORDED_WORLDS
    )
    sim.reset()
    rng = np.random.default_rng(0)
    qpos = np.tile(sim.mj_data.qpos, (4, 1))
    qpos[:, 7:] += rng.normal(scale=0.1, size=(4, qpos.shape[1] - 7))
    sim.set_state(qpos=qpos)
    for _ in range(N_STEPS):
        sim.step()
        sim.render_as_needed()
        sim.record_as_needed()

    render_fn = (
        render_trajectories_gpu if use_gpu_batch_rendering else render_trajectories
    )
    frames = render_fn(sim.mj_model, recorder.trajectories, cam, camera_res=(64, 64))
    cam_id = renderer._cameras_names2id[cam.name]
    for world_id, traj_frames in zip(RECORDED_WORLDS, frames):
        live_frames = np.array(renderer._fetch_frames_to_cpu_oneworld(world_id, cam_id))
        assert np.array_equal(traj_frames[cam.name], live_frames)
        # Sanity check: the fly moves between the first and last frame
        assert not np.array_equal(live_frames[0], live_frames[-1])
