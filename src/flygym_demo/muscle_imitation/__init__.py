"""Demo: muscle-driven imitation learning for the left-front leg.

When muscles are needed, the body model is *switched* to FlyMimic's
musculoskeletal MJCF (via `flygym.muscle`) rather than overlaying muscles on
FlyGym's default rigid body. This demo wires that model into an `ImitationEnv`.

* `make_imitation_env`: build the muscle Simulation + ImitationEnv in one call.
* `__main__`: train a PPO policy (stable-baselines3 optional; falls back to a
  random-policy rollout so the example always runs).

Run as a script:

    python -m flygym_demo.muscle_imitation --total-timesteps 200000
"""

from flygym_demo.muscle_imitation.fly import make_imitation_env

__all__ = ["make_imitation_env"]
