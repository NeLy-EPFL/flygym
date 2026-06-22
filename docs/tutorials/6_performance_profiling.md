# Performance profiling

Unlike the other tutorials, this page is not generated from a notebook — it is a
written guide to profiling FlyGym simulations.

The replay end-to-end test scripts double as end-to-end profiling targets. Both take a
`--profile PATH` flag. Run them without `--save-data` to profile the pure simulation
pipeline (no rendering/video).

## Prerequisites

The Python profiling dependencies (`py-spy` and `nvtx`) ship with the `dev` extra, so a
development install (`uv sync --extra dev`; see the
[installation guide](../installation.md)) already has them.

The **GPU** workflow additionally profiles with
[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) (`nsys`), which is
**not** a Python package and must be installed separately. It ships with the CUDA
Toolkit, or you can install it standalone:

```sh
# Ubuntu/Debian, via NVIDIA's CUDA apt repository:
apt-get install nsight-systems
```

Alternatively, download the installer for your platform from the
[Nsight Systems page](https://developer.nvidia.com/nsight-systems/get-started). Make
sure the `nsys` executable is on your `PATH`. Open the resulting `.nsys-rep` file in the
Nsight Systems GUI.

!!! note "Profiling output files are gitignored"

    Profiling artifacts (`*.speedscope.json`, `*.nsys-rep`, …) are gitignored, so you
    can write them anywhere in the working tree without polluting `git status`.

## CPU profiling (`scripts/replay_behavior_cpu.py`)

A sampling profile via [py-spy](https://github.com/benfred/py-spy), written in
[speedscope](https://www.speedscope.app/) format:

```bash
uv run python scripts/replay_behavior_cpu.py --save-data outputs/cpu_sim --profile outputs/cpu.speedscope.json
```

py-spy attaches to the **already-running process for the simulation loop only**, so the
one-time imports and model building (which otherwise dominate the flame graph as a tall
`_find_and_load` / `exec_module` tower) are never sampled — what you see is the loop
itself. Sampling uses `--native`, so native (C/C++) frames — notably MuJoCo's physics
step (`mj_projectConstraint`, `mju_cholFactorNumeric`, …) — appear alongside the Python
frames. Open the resulting file at [speedscope.app](https://www.speedscope.app/) (or
with the `speedscope` CLI). On Linux this needs no `sudo` (the script nominates py-spy
as an allowed tracer via `prctl(PR_SET_PTRACER)`); a longer `--sim-duration-sec` just
yields more samples.

For a complementary, **symbolication-free** view of where time goes *inside* the physics
step (kinematics, collision broad/narrow-phase, constraint solve, integration, …), pass
`--mujoco-timing`. This installs MuJoCo's internal timer callback (`mjcb_time`) so the C
engine fills in `mjData.timer` per phase, then prints a per-phase breakdown after the
run:

```bash
uv run python scripts/replay_behavior_cpu.py --mujoco-timing
```

It adds a per-phase callback overhead, so read it as a *relative* breakdown rather than
an absolute-throughput measurement. The two views corroborate each other — the phases
MuJoCo's timer flags as expensive should match the hot native frames
(`mj_projectConstraint`, `mju_cholFactorNumeric`, …) in the flame graph.

## GPU profiling (`scripts/replay_behavior_gpu.py`)

A timeline via [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
(`nsys`):

```bash
uv run python scripts/replay_behavior_gpu.py --save-data outputs/gpu_sim --profile outputs/gpu_profile
```

This re-executes the script under `nsys profile` and writes `profile.nsys-rep`, which
you open in the Nsight Systems GUI. The timeline is annotated with NVTX ranges (via
Warp's `wp.ScopedTimer(use_nvtx=True)`) — `warmup`, `timed_run`, and per-step `step` —
so JIT/warm-up is separated from the steady-state loop and host-side launches line up
with the captured CUDA-graph kernels. The NVTX range annotations require the `nvtx`
Python package, which is included in the `dev` extra.
