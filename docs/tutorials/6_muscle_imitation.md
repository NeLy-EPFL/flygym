# Muscle-Based Imitation Learning

!!! warning "Experimental"

    Support for the FlyMimic musculoskeletal body model is **experimental**.
    The API may change in future releases. Only the left-front leg is
    muscle-driven in the current model; other legs are passive or locked.
    Not all features available for the default `NeuroMechFly` model are
    currently supported (e.g. per-leg ground-contact sensors are absent).

FlyGym supports **muscle-actuated** fly simulations, in which the standard joint actuators are replaced by biophysically realistic Hill-type muscles. This opens the door to studying neuromuscular control, movement biomechanics, and motor learning — areas where the mechanical properties of muscles (force–velocity relationships, passive elasticity, activation dynamics) shape how movements are generated and controlled.

**Imitation learning** is a natural application of this framework: given motion-capture recordings of real fly kinematics, a policy can be trained to activate muscles so that the simulated limb tracks the recorded trajectory. The same approach generalises to any muscle-actuated limb or behavior. What ships today is a working implementation for the *Drosophila* left-front (LF) leg, reproducing the [FlyMimic](https://github.com/gizemozd/FlyMimic) result inside FlyGym.

The musculoskeletal body model follows the same composition convention as `NeuroMechFly` and `FlyBody` — the fly and world classes live in `flygym.compose`:

| Component | What it provides |
| --- | --- |
| `flygym.compose.MusculoskeletalFly` / `MusculoskeletalWorld` | The musculoskeletal fly + world, used exactly like `NeuroMechFly`/`FlatGroundWorld`. |
| `flygym.compose.build_musculoskeletal_simulation` | One-call factory (plus the guarded GPU/MuJoCo-Warp helpers `check_mjwarp_compatibility` and `build_musculoskeletal_gpu_simulation`). |
| `flygym_demo.muscle_imitation` | The imitation-learning stack: mocap clips (bundled under `assets/mocap/`), the dataset loader (`MoCapDataset`), and a Gymnasium environment (`ImitationEnv`) with the FlyMimic tracking reward. |

FlyGym's core only ships the musculoskeletal **body model**; the mocap clips and the imitation-learning code live entirely in the `flygym_demo.muscle_imitation` demo submodule, which is also the runnable example.

!!! info "Plain flygym (CPU) — not GPU-accelerated"

    `build_musculoskeletal_simulation` and `make_imitation_env` return a plain
    `flygym.Simulation` backed by a single CPU MuJoCo world — **not**
    `flygym.warp.GPUSimulation`. Everything on this page, including training,
    runs on plain `flygym`.


## Dataset

The motion-capture clips in `flygym_demo/muscle_imitation/assets/mocap/` are recorded *Drosophila* left-front-leg kinematics, stored as NumPy arrays at a 500 Hz control rate.

Each clip `{id}` provides four arrays:

| Array | Shape | Meaning | Units |
| --- | --- | --- | --- |
| `qpos/{id}.npy` | `(T, J)` | LF-leg joint angles | rad |
| `qvel/{id}.npy` | `(T, J)` | LF-leg joint velocities | rad/s |
| `xipos/{id}.npy` | `(T, 4, 3)` | 3D positions of 4 tracked bodies | mm |
| `xivel/{id}.npy` | `(T, 4, 3)` | 3D velocities of those bodies | mm/s |

The 4 tracked bodies are `LFFemur`, `LFTibia`, `LFTarsus1`, `LFTarsus5` (claw).

One clip ships with FlyGym — **`0002`** (225 frames, 7 joint DoFs), FlyMimic's own default. Its body trajectories match the bundled model, so the full reward range is available. Its 7 qpos columns map, in order, to: 

| col | MJCF joint |
| --- | --- |
| 0 | `joint_LFCoxa_yaw` |
| 1 | `joint_LFCoxa_pitch` |
| 2 | `joint_LFCoxa_roll` |
| 3 | `joint_LFTrochanter_yaw` |
| 4 | `joint_LFTrochanter_pitch` |
| 5 | `joint_LFTrochanter_roll` |
| 6 | `joint_LFTibia_pitch` |

The mapping is keyed by qpos width (`TRACKED_JOINT_NAMES_BY_NCOLS` in `flygym_demo.muscle_imitation.data`) and `ImitationEnv` selects it from the clip's width, so observation/action shapes adapt automatically (the shipped clip → 45-dim obs).


## The musculoskeletal model

The model is `assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml` (+ STL meshes), converted from an OpenSim model with [MyoConverter](https://github.com/MyoHub/myoconverter). It has 73 bodies, **15 Hill-type muscle actuators** on the LF leg, and **15 spatial tendons**.

Each muscle is a MuJoCo `general` actuator (`dyntype/gaintype/biastype = muscle`) acting through a spatial tendon routed via attachment sites on the thorax and LF-leg segments.

How it differs from FlyGym's default rigid-body fly:

| Aspect | FlyGym default | FlyMimic muscle model |
| --- | --- | --- |
| LF-leg links | `coxa → trochanterfemur (fused) → tibia → tarsus1..5` | `LFCoxa → LFTrochanter → LFFemur → LFTibia → LFTarsus1..5` |
| Actuation | joint position/torque actuators | 15 Hill-type muscles (LF leg) via spatial tendons |
| Passive joints | spring/damper from config | `stiffness = 0.4` + per-joint spring reference angles |
| Other legs | all six actuated | LF muscle-driven; RF locked to 0; LM/LH passive |
| Base | thorax free-floating | thorax tethered (anchored to world) |
| Sensors | vision, contact, proprioception | proprioception + body kinematics; vision optional (see §3) |

`build_musculoskeletal_simulation()` loads this model and returns a standard `flygym.Simulation`, so the rest of FlyGym works against it unchanged.



## Environment, reward, and sensors

### Environment — `flygym_demo.muscle_imitation.ImitationEnv`

A Gymnasium environment wrapping the muscle simulation:

* **Action** — 15 muscle activations in `[0, 1]`.
* **Observation** — tracked joint qpos + qvel, muscle activations, muscle forces, and a time-left scalar.
* **Step** — applies the activations, advances the physics by one control step (500 Hz over a 10 kHz physics timestep), and advances the mocap frame by one.

### Reward

Per step, against the corresponding mocap frame (the FlyMimic motion-imitation
reward, with `pose_w = 5`, `vel_w = 3`):

```
qpos_rew = exp(-pose_w * ‖target_qpos - actual_qpos‖₂)
qvel_rew = exp(-vel_w  * ‖target_qvel - actual_qvel‖₂)
xpos_rew = exp(-pose_w * mean_b ‖target_xpos_b - actual_xpos_b‖₂)
reward   = clip((qpos_rew + xpos_rew + qvel_rew) / 3, 0, 1)
```

In training mode an episode ends early if the reward drops below `rew_threshold` (default `0.01`) or the clip ends.


## Results & reproducibility

We reproduced FlyMimic's imitation-learning result in FlyGym. Training a PPO
policy on clip `0002` with FlyMimic's own hyperparameters (`stable-baselines3`,
`lr = 1e-5`, `gamma = 0.99`, ReLU `[512, 512, 256]` actor/critic) drives the
mean episode reward steadily upward, and the muscle-actuated LF leg learns to
track the reference kinematics — the reward formula and weights match FlyMimic
exactly (§3), and the reward ceiling on this clip is ~1.0.

Per-step reward climbs from the random-activation baseline (~0.06) to ~0.21,
with episode length growing in step (the policy both tracks better and holds
the pose longer) and no collapse. The trained leg motion can be inspected by
rendering a rollout from a checkpoint (see the example script).

Reproduce:

```bash
# quick check: random-policy rollout (no training dependencies)
uv run python -m flygym_demo.muscle_imitation --no-train

# train a policy with logging + checkpointing, then record a video of it
uv run python -m flygym_demo.muscle_imitation \
    --clip 0002 --total-timesteps 15000000 --learning-rate 1e-5 \
    --log-dir runs/0002 --video-path runs/0002/rollout.mp4

# render a previously-saved policy without retraining
uv run python -m flygym_demo.muscle_imitation --no-train \
    --model-path runs/0002/final_model.zip --video-path rollout.mp4
```

Training writes all artifacts under `--log-dir` (default `runs/<clip>`):

| Artifact | Contents |
| --- | --- |
| `monitor.csv` | Per-episode reward + length (`stable_baselines3` `Monitor`; read with `pandas`). |
| `tb/` | TensorBoard event files — view with `tensorboard --logdir runs/<clip>/tb` (`rollout/ep_rew_mean`, `ep_len_mean`, …). TensorBoard is optional; without it the CSV is still written. |
| `checkpoints/ppo_muscle_*_steps.zip` | Periodic checkpoints (`--checkpoint-freq`, default every 50k steps; `0` disables). |
| `final_model.zip` | The policy at the end of training. |

To watch the learned behaviour, `--video-path` runs a deterministic rollout in
test mode (full clip, no early termination), rendering FlyMimic's world camera
(`--camera`, default `scene`; `--camera-res H W`) to an mp4. The same env can be
driven from Python via `flygym_demo.muscle_imitation.record_rollout` /
`load_policy`.

Training is CPU-only on most workstations (see §5 for the GPU path). At higher
learning rates, set PPO `target_kl ≈ 0.05` and keep the best checkpoint by
periodic evaluation to avoid late instability.



## API

```python
from flygym.compose import build_musculoskeletal_simulation
from flygym_demo.muscle_imitation import ImitationConfig, ImitationEnv, MoCapDataset

sim, fly = build_musculoskeletal_simulation()  # Simulation backed by the muscle model
env = ImitationEnv(
    sim, fly_name=fly.name,
    dataset=MoCapDataset.default(),
    config=ImitationConfig(clip="0002"),
)

obs, _ = env.reset()
for _ in range(200):
    action = env.action_space.sample()    # 15 muscle activations in [0, 1]
    obs, reward, terminated, truncated, info = env.step(action)
```

`build_musculoskeletal_simulation()` is a convenience wrapper over the standard
FlyGym composition flow, identical to how you'd build a `NeuroMechFly` scene:

```python
from flygym import Simulation
from flygym.compose import MusculoskeletalFly, MusculoskeletalWorld

fly = MusculoskeletalFly()
world = MusculoskeletalWorld(fly)
sim = Simulation(world)
```

Build the environment in one call:

```python
from flygym_demo.muscle_imitation import make_imitation_env
env = make_imitation_env(config=ImitationConfig(clip="0002"))
```

Train (logging + checkpointing) and record a video programmatically:

```python
from flygym_demo.muscle_imitation import (
    TrainConfig, train, make_imitation_env, load_policy, record_rollout,
)

# Writes monitor.csv, tb/, checkpoints/, final_model.zip under runs/0002
model, final_path = train("runs/0002", config=TrainConfig(clip="0002",
                                                          total_timesteps=200_000))

# Roll the trained policy out and save an mp4 (test mode = full clip)
env = make_imitation_env(config=ImitationConfig(clip="0002", test=True))
record_rollout(env, load_policy(final_path), "rollout.mp4")
```

Inspect or drive the model directly:

```python
sim, fly = build_musculoskeletal_simulation(add_vision=True)
fly.muscle_names                 # the 15 muscle actuator names
sim.get_joint_angles(fly.name)   # proprioception, body kinematics, ...
```


## Citation

If you use the musculoskeletal model in your research, please cite the
FlyMimic paper in addition to FlyGym:

> Ozdil, P. G., Ning, C., Phelps, J. S., Wang-Chen, S., Elisha, G., Blanke, A.,
> Ijspeert, A., & Ramdya, P. (2026). Musculoskeletal simulation of limb
> movement biomechanics in *Drosophila melanogaster*. *ICLR 2026*.
> [arXiv:2509.06426](https://arxiv.org/abs/2509.06426)

```bibtex
@inproceedings{Ozdil2026,
  title={Musculoskeletal simulation of limb movement biomechanics in Drosophila melanogaster},
  author={Ozdil, Pembe Gizem and Ning, Chuanfang and Phelps, Jasper S and Wang-Chen, Sibo and Elisha, Guy and Blanke, Alexander and Ijspeert, Auke and Ramdya, Pavan},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2509.06426},
}
```
