"""Tests for GPU trajectory recording (WarpTrajectoryRecorder) and GPU replay."""

import warnings

import numpy as np
import mujoco as mj
import pytest

# These tests require the optional warp (GPU) extra; tag them so they can be
# excluded with ``-m "not warp"``, and skip the whole module if warp is absent.
pytestmark = pytest.mark.warp
pytest.importorskip("warp")

from flygym.rendering import (
    RecordedTrajectory,
    render_trajectories,
)
from flygym.warp import (
    RendererType,
    WarpTrajectoryRecorder,
    render_trajectories_gpu,
    modify_world_for_batch_rendering,
)


# ---------------------------------------------------------------------------
# Module-scoped fixture: GPU simulation + recorded trajectories
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recorded_gpu(gpu_sim_factory):
    """Record a short trajectory from a 4-world GPU sim, sub-selecting 2 worlds."""
    sim, fly, cam = gpu_sim_factory(n_worlds=4, fly_name="rec_gpu_fly")
    rec = sim.set_renderer(
        cam,
        camera_res=(64, 64),
        output_fps=100,
        worlds=[0, 2],
        renderer_type=RendererType.RECORDED_TRAJECTORY,
    )
    sim.reset()
    for _ in range(200):
        sim.step()
        sim.render_as_needed()
    return rec, sim, cam


def _batch_render_model(sim) -> mj.MjModel:
    """Compile a batch-render-ready model from the sim's world (caller's job now)."""
    with warnings.catch_warnings():
        # modify_world_for_batch_rendering warns as it adds overhead lights/strips
        # textures; that is expected here, so silence it (matches test_rendering.py).
        warnings.simplefilter("ignore")
        modify_world_for_batch_rendering(sim.world)
    return sim.world.compile()[0]


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class TestWarpTrajectoryRecorder:
    def test_is_recorder_not_renderer(self, recorded_gpu):
        rec, _, _ = recorded_gpu
        assert isinstance(rec, WarpTrajectoryRecorder)
        # No rasterization: the recorder holds no mj.Renderer / scene_option.
        assert rec.mj_renderer is None
        assert rec.scene_option is None

    def test_one_trajectory_per_selected_world(self, recorded_gpu):
        rec, _, _ = recorded_gpu
        trajs = rec.recorded_trajectories
        assert len(trajs) == 2
        assert [t.world_id for t in trajs] == [0, 2]

    def test_trajectory_shapes(self, recorded_gpu):
        rec, sim, _ = recorded_gpu
        for t in rec.recorded_trajectories:
            assert t.n_frames > 0
            assert t.qpos.shape == (t.n_frames, sim.mj_model.nq)

    def test_save_video_raises(self, recorded_gpu):
        rec, _, _ = recorded_gpu
        with pytest.raises(RuntimeError):
            rec.save_video(0, "x.mp4")


# ---------------------------------------------------------------------------
# Serialization (backend-agnostic format)
# ---------------------------------------------------------------------------


class TestSaveLoadGPU:
    def test_roundtrip(self, recorded_gpu, tmp_path):
        rec, _, _ = recorded_gpu
        trajs = rec.recorded_trajectories
        for i, traj in enumerate(trajs):
            traj.save(tmp_path / f"traj_{i:04d}.npz")
        loaded = [
            RecordedTrajectory.from_file(p) for p in sorted(tmp_path.glob("traj_*.npz"))
        ]
        assert len(loaded) == 2
        for a, b in zip(loaded, trajs):
            assert np.array_equal(a.qpos, b.qpos)
            assert a.world_id == b.world_id


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class TestRenderGPU:
    def test_gpu_replay_writes_videos(self, recorded_gpu, tmp_path):
        rec, sim, _ = recorded_gpu
        trajs = rec.recorded_trajectories
        mj_model = _batch_render_model(sim)
        out = tmp_path / "gpu_out"
        render_trajectories_gpu(mj_model, trajs, out, worlds_per_batch=8)
        videos = sorted(out.rglob("*.mp4"))
        assert len(videos) == 2
        assert all(v.stat().st_size > 0 for v in videos)

    def test_gpu_recorded_renders_on_cpu(self, recorded_gpu, tmp_path):
        """A GPU-recorded trajectory is backend-agnostic: it replays on CPU too."""
        rec, sim, _ = recorded_gpu
        trajs = rec.recorded_trajectories
        out = tmp_path / "cpu_out"
        render_trajectories(sim.mj_model, trajs, out)
        videos = sorted(out.rglob("*.mp4"))
        assert len(videos) == 2


class TestKinematicsIdentityAcrossBackends:
    def test_cpu_and_gpu_kinematics_match_for_same_qpos(self, recorded_gpu):
        """Replay kinematics are identical across backends (not pixels, geometry).

        Pixel output differs stylistically (different rasterizers), but for the same
        recorded qpos the CPU (``mj_kinematics``) and GPU (``mjw.kinematics``) passes
        must place the geometry in the same poses. We compare ``geom_xpos`` for a few
        recorded frames.
        """
        import mujoco_warp as mjw
        import warp as wp

        rec, sim, _ = recorded_gpu
        traj = rec.recorded_trajectories[0]
        mj_model = sim.mj_model

        # Sample a few frames spread across the trajectory.
        idxs = [0, traj.n_frames // 2, traj.n_frames - 1]

        # CPU geom_xpos for each sampled qpos.
        cpu_geom = []
        d = mj.MjData(mj_model)
        for f in idxs:
            d.qpos[:] = traj.qpos[f]
            if traj.has_mocap:
                d.mocap_pos[:] = traj.mocap_pos[f]
                d.mocap_quat[:] = traj.mocap_quat[f]
            mj.mj_kinematics(mj_model, d)
            cpu_geom.append(d.geom_xpos.copy())

        # GPU geom_xpos: stage the same qpos rows into a batched mjw.Data.
        mjw_model = mjw.put_model(mj_model)
        mjw_data = mjw.put_data(mj_model, mj.MjData(mj_model), nworld=len(idxs))
        qpos_batch = np.stack([traj.qpos[f] for f in idxs]).astype(np.float32)
        mjw_data.qpos.assign(qpos_batch)
        if traj.has_mocap:
            mjw_data.mocap_pos.assign(
                np.stack([traj.mocap_pos[f] for f in idxs]).astype(np.float32)
            )
            mjw_data.mocap_quat.assign(
                np.stack([traj.mocap_quat[f] for f in idxs]).astype(np.float32)
            )
        mjw.kinematics(mjw_model, mjw_data)
        wp.synchronize()
        gpu_geom = mjw_data.geom_xpos.numpy()  # (len(idxs), ngeom, 3)

        for i in range(len(idxs)):
            assert np.allclose(cpu_geom[i], gpu_geom[i], atol=1e-5)
