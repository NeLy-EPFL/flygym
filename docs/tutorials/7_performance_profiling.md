# Performance profiling

This guide shows how to profile FlyGym simulations. The kinematic replay scripts, `scripts/replay_behavior_cpu.py` and `scripts/replay_behavior_gpu.py`, double as end-to-end profiling targets. Both take a `--profile PATH` flag.

## Prerequisites

Run the scripts from a development install that includes the `dev` and `examples` extras (plus `warp` for the GPU script; see the [installation guide](../installation.md)):

```sh
uv sync --extra dev --extra examples --extra warp
```

The Python profiling dependencies (`py-spy` and `nvtx`) ship with the `dev` extra. The GPU workflow additionally requires [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) (`nsys`), which is **not** a Python package and must be installed separately. It ships with the CUDA Toolkit, or you can install it standalone:

```sh
# Ubuntu/Debian, via NVIDIA's CUDA apt repository:
sudo apt-get install nsight-systems
```

Alternatively, download the installer for your platform from the [Nsight Systems page](https://developer.nvidia.com/nsight-systems/get-started). Make sure the `nsys` executable is on your `PATH`.

!!! note "Profiling output files are gitignored"

    Profiling artifacts (`*.speedscope.json`, `*.nsys-rep`) are gitignored, so you can write them anywhere in the working tree without polluting `git status`.

## CPU profiling

The CPU script records a sampling profile with [py-spy](https://github.com/benfred/py-spy) in [speedscope](https://www.speedscope.app/) format:

```sh
uv run python scripts/replay_behavior_cpu.py --profile outputs/cpu.speedscope.json
```

Open the resulting file at [speedscope.app](https://www.speedscope.app/) (or with the `speedscope` CLI).

py-spy attaches to the already-running process for the simulation loop only. Therefore, the one-time imports and model building are never sampled, and the flame graph shows the loop itself. The loop always includes rendering of the tracking camera. Adding `--save-data DIR` only writes data, plots, and video after the loop, so it does not change the profile.

Sampling uses `--native`, so native C/C++ frames, notably MuJoCo's physics step (`mj_projectConstraint`, `mju_cholFactorNumeric`, …), appear alongside the Python frames. On Linux, this needs no `sudo`: the script nominates py-spy as an allowed tracer via `prctl(PR_SET_PTRACER)`. Use a longer `--sim-duration-sec` to collect more samples. Note that native sampling slows the loop down considerably, so don't read throughput numbers from a profiled run.

For a symbolication-free view of where time goes inside the physics step (kinematics, collision broad/narrow-phase, constraint solve, integration, …), pass `--mujoco-timing` instead:

```sh
uv run python scripts/replay_behavior_cpu.py --mujoco-timing
```

This installs MuJoCo's internal timer callback (`mjcb_time`) so that the C engine fills in `mjData.timer` per phase, then prints a per-phase breakdown after the run. The callback adds some overhead to each phase, so read the output as a relative breakdown rather than an absolute throughput measurement. The two views should corroborate each other: the phases that MuJoCo's timer flags as expensive should match the hot native frames in the flame graph.

## GPU profiling

The GPU script records a timeline with [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems):

```sh
uv run python scripts/replay_behavior_gpu.py --profile outputs/gpu_profile
```

The script re-executes itself under `nsys profile` and writes `outputs/gpu_profile.nsys-rep`, which you can open in the Nsight Systems GUI. Without `--save-data`, the run is a pure physics benchmark. With `--save-data DIR`, the loop additionally records joint angles and renders the simulation, and the profile includes these as well.

The capture uses `--cuda-graph-trace=node`, so the CUDA graph replayed at every step is expanded into its individual kernels on the GPU timeline. The timeline is also annotated with NVTX ranges (via Warp's `wp.ScopedTimer(use_nvtx=True)`): `warmup`, `timed_run`, and per-step `step`. This separates JIT compilation and warm-up from the steady-state loop. Note that the `step` ranges only cover the host-side graph launch. Since the GPU executes the graph asynchronously, read per-kernel timing from the GPU rows rather than from these ranges.

For a text summary without the GUI, use `nsys stats`:

```sh
nsys stats --report cuda_gpu_kern_sum outputs/gpu_profile.nsys-rep  # per-kernel time
nsys stats --report nvtx_sum outputs/gpu_profile.nsys-rep  # NVTX ranges
```
