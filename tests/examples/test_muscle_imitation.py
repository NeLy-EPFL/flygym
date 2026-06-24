"""Tests for the flygym_demo.muscle_imitation subpackage."""

import importlib
import platform

import numpy as np
import pytest

# The whole module needs the `rl` extra (gymnasium + Stable-Baselines3). The
# marker lets it be deselected explicitly (`pytest -m "not rl"`); the
# importorskip additionally skips it cleanly — rather than erroring at
# collection — when the optional dep simply isn't installed.
pytestmark = pytest.mark.rl
pytest.importorskip("gymnasium")

from flygym_demo.muscle_imitation import (  # noqa: E402
    DEFAULT_MOCAP_DIR,
    ImitationConfig,
    ImitationEnv,
    MoCapDataset,
    TRACKED_BODY_NAMES,
    TRACKED_JOINT_NAMES,
    tracked_joint_names_for_ncols,
)


# -----------------------------------------------------------------------------
# Asset/code location: mocap + IL stack live in flygym_demo, NOT in core
# -----------------------------------------------------------------------------


def test_mocap_dir_lives_under_the_demo_package_not_core_assets():
    # The clips moved out of flygym/assets/model into this demo's own assets/.
    assert DEFAULT_MOCAP_DIR.exists()
    parts = DEFAULT_MOCAP_DIR.parts
    assert "flygym_demo" in parts and "muscle_imitation" in parts
    assert "assets" in parts and "mocap" in parts
    # And specifically not under flygym's core model assets.
    assert "assets/model/musculoskeletal/mocap" not in DEFAULT_MOCAP_DIR.as_posix()


def test_mocap_clips_are_bundled_at_the_demo_location():
    for sub in ("qpos", "qvel", "xipos", "xivel"):
        assert (DEFAULT_MOCAP_DIR / sub / "0002.npy").exists()


def test_mocap_clips_are_gone_from_core_model_assets():
    from flygym import assets_dir

    assert not (assets_dir / "model/musculoskeletal/mocap").exists()


def test_legacy_flygym_imitation_module_is_removed():
    # The IL stack moved entirely into flygym_demo.muscle_imitation; the old
    # core import path must no longer exist.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flygym.imitation")


# -----------------------------------------------------------------------------
# data.py
# -----------------------------------------------------------------------------


def test_default_dataset_available_clips():
    ds = MoCapDataset.default()
    assert ds.available_clips() == ["0002"]


def test_load_clip_shapes_match_constants():
    ds = MoCapDataset.default()
    clip = ds.load("0002")
    assert clip.qpos.ndim == 2
    assert clip.qpos.shape[1] == len(TRACKED_JOINT_NAMES)  # 7-DoF
    assert clip.qvel.shape == clip.qpos.shape
    assert clip.xipos.shape[1] == len(TRACKED_BODY_NAMES)
    assert clip.xipos.shape[2] == 3
    assert clip.n_frames > 100


def test_tracked_joint_names_for_ncols():
    assert len(tracked_joint_names_for_ncols(7)) == 7
    assert tracked_joint_names_for_ncols(7)[3] == "joint_LFTrochanter_yaw"
    with pytest.raises(ValueError, match="No tracked-joint mapping"):
        tracked_joint_names_for_ncols(6)


def test_dataset_caches_clip_object():
    ds = MoCapDataset.default()
    assert ds.load("0002") is ds.load("0002")


# -----------------------------------------------------------------------------
# env.py (integration-style)
# -----------------------------------------------------------------------------


def _make_env(test_mode: bool = True) -> ImitationEnv:
    # Uses the default clip (0002) so tests track the recommended setup.
    from flygym_demo.muscle_imitation.fly import make_imitation_env

    return make_imitation_env(config=ImitationConfig(test=test_mode))


def test_env_resolves_7dof_clip_and_spaces():
    env = _make_env()  # default clip 0002 -> 7 DoFs
    assert env.action_space.shape == (15,)
    assert env._tracked_qposadrs.shape == (7,)
    assert env._tracked_body_ids.shape == (len(TRACKED_BODY_NAMES),)
    assert len(env.tracked_joint_names) == 7
    assert env.observation_space.shape[0] == 2 * 7 + 2 * 15 + 1


def test_env_reset_returns_finite_obs():
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert info["clip"] == "0002"
    assert info["start_idx"] == 0  # test mode locks to clip start
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))


def test_env_step_reward_in_unit_interval():
    env = _make_env()
    env.reset(seed=0)
    terminated = False
    for _ in range(5):
        obs, rew, terminated, truncated, info = env.step(
            np.full(env.n_muscles, 0.1, dtype=np.float32)
        )
        assert 0.0 <= rew <= 1.0
        assert obs.shape == env.observation_space.shape
        assert truncated is False
    assert not terminated  # test mode: no early termination in 5 steps


def test_default_clip_xpos_reward_is_high_at_mocap_pose():
    # Regression guard: the bundled clip's mocap body trajectories must match
    # the MJCF's kinematics, so xpos_rew ~ 1 at the exact recorded pose. An
    # inconsistent clip would saturate this near 0 and cap the reward.
    env = _make_env()  # default clip
    assert env.config.clip == "0002"
    env.reset(seed=0)
    clip = env._clip
    i = env._mocap_idx
    tx = clip.xipos[i]
    ax = env.sim.mj_data.xpos[env._tracked_body_ids]
    mean_dist = float(np.mean(np.linalg.norm(tx - ax, axis=-1)))
    xpos_rew = float(np.exp(-env.config.pose_rew_weight * mean_dist))
    assert xpos_rew > 0.9, f"xpos_rew={xpos_rew:.3f}, mean_dist={mean_dist:.4f} mm"


def test_reward_matches_flymimic_formula_exactly():
    # The env reward must be a bit-for-bit transcription of FlyMimic's
    # mocap_tracking after_step: L2 norm for qpos/qvel, mean-of-per-body-L2 for
    # xpos, equal weighting, clipped to [0, 1].
    def flymimic_reward(tq, aq, tv, av, tx, ax, pw=5.0, vw=3.0):
        xr = np.exp(-pw * np.mean(np.linalg.norm(tx - ax, axis=-1)))
        qr = np.exp(-pw * np.mean(np.linalg.norm(tq - aq, axis=-1)))
        vr = np.exp(-vw * np.mean(np.linalg.norm(tv - av, axis=-1)))
        return float(np.clip((qr + xr + vr) / 3, 0.0, 1.0))

    env = _make_env()
    rng = np.random.default_rng(1)
    for perturb in (0.0, 0.05, 0.4):
        env.reset(seed=0)
        for _ in range(3):
            a = (
                np.zeros(env.n_muscles, dtype=np.float32)
                if perturb == 0
                else rng.uniform(0, perturb, env.n_muscles).astype(np.float32)
            )
            env.step(a)
        i = env._mocap_idx
        tq, aq = env._clip.qpos[i], env.sim.mj_data.qpos[env._tracked_qposadrs]
        tv, av = env._clip.qvel[i], env.sim.mj_data.qvel[env._tracked_qveladrs]
        tx, ax = env._clip.xipos[i], env.sim.mj_data.xpos[env._tracked_body_ids]
        mine, _ = env._compute_reward_and_done()
        assert abs(mine - flymimic_reward(tq, aq, tv, av, tx, ax)) < 1e-9


def test_env_wrong_action_shape_raises():
    env = _make_env()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="Expected action shape"):
        env.step(np.zeros(env.n_muscles - 1, dtype=np.float32))


# -----------------------------------------------------------------------------
# rollout / video recording (record.py) — no ML deps needed
# -----------------------------------------------------------------------------


def test_scene_camera_is_named_for_rendering():
    # The muscle model ships an unnamed world camera; MusculoskeletalFly names
    # it so it can be selected via Simulation.set_renderer.
    from flygym.compose import DEFAULT_SCENE_CAMERA, build_musculoskeletal_simulation

    _, fly = build_musculoskeletal_simulation()
    assert DEFAULT_SCENE_CAMERA in fly.cameraname_to_mjcfcamera


def test_run_rollout_returns_per_step_rewards():
    from flygym_demo.muscle_imitation import random_policy, run_rollout

    env = _make_env()
    rewards = run_rollout(env, random_policy(env), n_steps=10)
    assert len(rewards) == 10
    assert all(0.0 <= r <= 1.0 for r in rewards)


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason=(
        "mujoco hardcodes CGL on macOS and GLFW on Windows; "
        "neither works headlessly in CI without a GPU"
    ),
)
def test_record_rollout_writes_video(tmp_path):
    from flygym_demo.muscle_imitation import random_policy, record_rollout

    env = _make_env()  # test mode -> full clip, no early termination
    out = tmp_path / "rollout.mp4"
    stats = record_rollout(env, random_policy(env), out, camera_res=(120, 160))
    assert out.exists() and out.stat().st_size > 0
    assert stats["n_steps"] > 0
    assert stats["video_path"] == str(out)
