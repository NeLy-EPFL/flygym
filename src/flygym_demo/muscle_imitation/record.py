"""Roll out a policy on the muscle imitation env and optionally record video.

A *policy* here is just a callable ``predict(obs) -> action`` (15 muscle
activations). Two builders are provided:

* `load_policy` — wrap a saved stable-baselines3 PPO checkpoint (deterministic).
* `random_policy` — a dependency-free baseline for smoke tests.

`record_rollout` runs one episode, rendering through FlyMimic's scene camera,
and writes an mp4. Build the env in *test* mode (``ImitationConfig(test=True)``)
so the episode starts at the clip's first frame and runs to the end without
early termination.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Callable

import numpy as np

from flygym.compose.fly.musculoskeletal import DEFAULT_SCENE_CAMERA
from flygym_demo.muscle_imitation.env import ImitationEnv

Policy = Callable[[np.ndarray], np.ndarray]


def load_policy(model_path: PathLike) -> Policy:
    """Load a saved SB3 PPO policy as a deterministic ``predict(obs)`` callable."""
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path))

    def predict(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action

    return predict


def random_policy(env: ImitationEnv, *, seed: int = 0) -> Policy:
    """A random-activation baseline policy (no ML dependencies)."""
    rng = np.random.default_rng(seed)

    def predict(_obs: np.ndarray) -> np.ndarray:
        return rng.uniform(0.05, 0.6, size=env.n_muscles).astype(np.float32)

    return predict


def run_rollout(
    env: ImitationEnv, policy: Policy, *, n_steps: int = 200
) -> list[float]:
    """Step *policy* through the env for *n_steps*, resetting on episode end.

    Args:
        env: The `ImitationEnv` to evaluate.
        policy: A ``predict(obs) -> action`` callable.
        n_steps: Total environment steps to run (across episodes).

    Returns:
        Per-step reward list of length *n_steps*. Useful as a quick numeric
        smoke test without rendering.
    """
    obs, _ = env.reset()
    rewards: list[float] = []
    for _ in range(n_steps):
        obs, rew, terminated, truncated, _ = env.step(policy(obs))
        rewards.append(float(rew))
        if terminated or truncated:
            obs, _ = env.reset()
    return rewards


def record_rollout(
    env: ImitationEnv,
    policy: Policy,
    video_path: PathLike,
    *,
    camera: str = DEFAULT_SCENE_CAMERA,
    camera_res: tuple[int, int] = (480, 640),
    playback_speed: float = 0.2,
    output_fps: int = 25,
) -> dict:
    """Render one episode of ``policy`` to an mp4 at ``video_path``.

    Args:
        env: An `ImitationEnv`, ideally built with ``ImitationConfig(test=True)``
            so the full clip is rolled out without early termination.
        policy: A ``predict(obs) -> action`` callable.
        video_path: Output mp4 path.
        camera: Scene camera name (defaults to the muscle model's world camera).
        camera_res: ``(height, width)`` in pixels.
        playback_speed: Video speed relative to real time (<1 is slow-motion).
        output_fps: Output frame rate.

    Returns:
        A stats dict: ``n_steps``, ``mean_reward``, ``video_path``.
    """
    sim = env.sim
    renderer = sim.set_renderer(
        camera,
        camera_res=camera_res,
        playback_speed=playback_speed,
        output_fps=output_fps,
    )

    obs, _ = env.reset()
    rewards: list[float] = []
    done = False
    while not done:
        obs, rew, terminated, truncated, _ = env.step(policy(obs))
        sim.render_as_needed()
        rewards.append(float(rew))
        done = terminated or truncated

    video_path = Path(video_path)
    renderer.save_video(video_path)
    return {
        "n_steps": len(rewards),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "video_path": str(video_path),
    }
