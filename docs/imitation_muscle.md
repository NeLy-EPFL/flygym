# Muscle-Based Imitation Learning

FlyGym can drive the left-front (LF) leg of the fly with biomechanical **muscles** and learn to reproduce recorded leg movements via imitation learning. When muscle actuation is selected, FlyGym uses [FlyMimic's](https://github.com/gizemozd/FlyMimic) musculoskeletal model in place of the default rigid-body model, while keeping FlyGym's `Simulation` and sensor APIs. The integration lives in two packages:

| Package | What it provides |
| --- | --- |
| `flygym.muscle` | The musculoskeletal fly/world (`MuscleFly`, `MuscleWorld`, `build_muscle_simulation`) plus optional GPU/MuJoCo-Warp helpers. |
| `flygym.imitation` | Mocap dataset loader (`MoCapDataset`) and a Gymnasium environment (`ImitationEnv`) with the FlyMimic tracking reward. |

A runnable example lives in `flygym_demo.muscle_imitation`.

---

## 1. Dataset

The motion-capture clips in `flygym/assets/mocap/` are recorded *Drosophila* left-front-leg kinematics, stored as NumPy arrays at a 500 Hz control rate.

Each clip `{id}` provides four arrays:

| Array | Shape | Meaning | Units |
| --- | --- | --- | --- |
| `qpos/{id}.npy` | `(T, J)` | LF-leg joint angles | rad |
| `qvel/{id}.npy` | `(T, J)` | LF-leg joint velocities | rad/s |
| `xipos/{id}.npy` | `(T, 4, 3)` | 3D positions of 4 tracked bodies | mm |
| `xivel/{id}.npy` | `(T, 4, 3)` | 3D velocities of those bodies | mm/s |

The 4 tracked bodies are `LFFemur`, `LFTibia`, `LFTarsus1`, `LFTarsus5` (claw).

One clip ships with FlyGym — **`0002`** (225 frames, 7 joint DoFs), FlyMimic's
own default. Its body trajectories match the bundled model, so the full reward
range is available. Its 7 qpos columns map, in order, to:

| col | MJCF joint |
| --- | --- |
| 0 | `joint_LFCoxa_yaw` |
| 1 | `joint_LFCoxa_pitch` |
| 2 | `joint_LFCoxa_roll` |
| 3 | `joint_LFTrochanter_yaw` |
| 4 | `joint_LFTrochanter_pitch` |
| 5 | `joint_LFTrochanter_roll` |
| 6 | `joint_LFTibia_pitch` |

The mapping is keyed by qpos width (`TRACKED_JOINT_NAMES_BY_NCOLS` in
`flygym.imitation.data`) and `ImitationEnv` selects it from the clip's width,
so observation/action shapes adapt automatically (the shipped clip → 45-dim
obs).

---

## 2. The musculoskeletal model

The model is `assets/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml` (+ STL meshes), converted from an OpenSim model with [MyoConverter](https://github.com/MyoHub/myoconverter). It has 73 bodies, **15 Hill-type muscle actuators** on the LF leg, and **15 spatial tendons**.

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

`build_muscle_simulation()` loads this model and returns a standard `flygym.Simulation`, so the rest of FlyGym works against it unchanged.

---

## 3. Environment, reward, and sensors

### Environment — `flygym.imitation.ImitationEnv`

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

### Sensors

| Sensor | Status |
| --- | --- |
| Proprioception (`get_joint_angles` / `get_joint_velocities`) | ✅ |
| Body kinematics (`get_body_positions` / `get_body_rotations`) | ✅ |
| Compound-eye vision (`get_ommatidia_readouts`) | ⚠️ via `MuscleFly.add_vision()`; approximate (retina calibrated for FlyGym's eyes) |
| Per-leg ground contact (`get_ground_contact_info`) | ❌ not available |
| Body contact forces (`get_bodysegment_contact_forces`) | ✅ against the floor |

(Contact is of limited use while only one leg is actuated — see §6.)

---

## 4. Results & reproducibility

On the default clip `0002`, the reward ceiling is ~1.0 (at the recorded pose
the joint and body terms are both near-perfect). A PPO policy
(`stable-baselines3`, `MlpPolicy` `[512, 512, 256]`) trained on this clip
learns to track the reference motion: the mean episode reward rises steadily
and the trained policy clearly outperforms a random-activation baseline.
Setting PPO `target_kl ≈ 0.05` keeps the long-run curve stable.

Reproduce:

```bash
# quick check: random-policy rollout (no training dependencies)
python -m flygym_demo.muscle_imitation --no-train

# train a policy (requires stable-baselines3)
python -m flygym_demo.muscle_imitation \
    --clip 0002 --total-timesteps 5000000 --learning-rate 1e-4
```

Training is CPU-only on most workstations (see §5 for the GPU path). For long
runs, prefer `target_kl` early-stopping and keep the best checkpoint by
periodic evaluation.

---

## 5. API

```python
from flygym.muscle import build_muscle_simulation
from flygym.imitation import ImitationConfig, ImitationEnv, MoCapDataset

sim, fly = build_muscle_simulation()      # Simulation backed by the muscle model
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

Build the environment in one call:

```python
from flygym_demo.muscle_imitation import make_imitation_env
env = make_imitation_env(config=ImitationConfig(clip="0002"))
```

Inspect or drive the model directly:

```python
sim, fly = build_muscle_simulation(add_vision=True)
fly.muscle_names                 # the 15 muscle actuator names
sim.get_joint_angles(fly.name)   # proprioception, body kinematics, ...
```

### GPU (MuJoCo-Warp)

`flygym.muscle` includes helpers for the GPU path, safe to import anywhere and
only requiring a CUDA GPU when used:

```python
from flygym.muscle import check_mjwarp_compatibility, build_muscle_gpu_simulation

check_mjwarp_compatibility()              # probe support (no-op without mujoco_warp)
sim, fly = build_muscle_gpu_simulation(n_worlds=4096)   # Linux + NVIDIA + [warp]
```

`GPUSimulation` runs many worlds in parallel — the main speedup for RL.
MuJoCo-Warp's support for muscle actuators, spatial tendons, and joint-equality
constraints is version-dependent, so run `check_mjwarp_compatibility()` on the
target machine first.

---

## 6. Future work

* **More legs.** Only the LF leg is muscle-driven. The middle (LM) and hind
  (LH) leg meshes are present; adding their muscle definitions would extend
  imitation to a full half-body. The right-front (RF) leg exists but is locked
  — unlocking and mirroring the LF muscles would add a second front leg.
* **Meaningful ground contact.** With the thorax tethered and one active leg,
  ground reaction forces are not yet behaviorally meaningful. Contact becomes
  useful once multiple legs are actuated and the body is freed to support and
  propel itself; per-leg contact sensors can then be added.
* **Vision calibration.** Recalibrate the fisheye retina for the FlyMimic eye
  geometry so ommatidia readouts are quantitatively correct.
* **More behaviors.** Additional mocap clips would broaden the imitation
  repertoire.
* **GPU scaling.** Vectorized PPO over many `GPUSimulation` worlds.

---

## 7. Citation

If you use the musculoskeletal model in your research, please cite our paper:

```bibtex
@inproceedings{ozdil2026musculoskeletal,
  title={Musculoskeletal simulation of limb movement biomechanics in Drosophila melanogaster},
  author={Ozdil, Pembe Gizem and Ning, Chuanfang and Phelps, Jasper S and Wang-Chen, Sibo and Elisha, Guy and Ijspeert, Auke and Ramdya, Pavan},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```
