# Contributing to FlyGym

Thank you for your interest in contributing! Here's how to get started.

## Reporting issues & reaching out

Use [GitHub Issues](https://github.com/NeLy-EPFL/flygym/issues) for bugs and feature requests.
For questions and discussion, use the [Discussion Board](https://github.com/NeLy-EPFL/flygym/discussions).
We prefer these channels over emails addressed to the corresponding authors of the publications.

Feel free to reach out to us before implementing any new feature.

## Setting up a development environment

Please follow instructions [here](https://neuromechfly.org/installation/#__tabbed_1_2) to install FlyGym for development.
Be sure to follow the "Using `uv` (for development)" tab.

**Please be sure to follow the Note box on the page above regarding `nbstripout`**.
This step prevents Jupyter Notebook output blocks from being included in the commit
history. These data (e.g., images and videos) are large and can make the Git repo bloated,
thus slowing down CI workflows and Colab use cases. During development, **do not** use
IDEs to stage notebook files—they might skip the `nbstripout` filter. Always run
`git add <notebook_files>` in the terminal.


## Running tests

Run the whole suite with:

```bash
uv run pytest tests/
```

Which tests run is controlled by **markers**, not by directory. This lets a test
declare a dependency (e.g. on the GPU backend) regardless of which file it lives
in. Use pytest's `-m` flag to select or exclude groups:

| Marker     | What it covers                              | Requires                          |
| ---------- | ------------------------------------------- | --------------------------------- |
| `warp`     | GPU-accelerated (warp) backend tests        | the `warp` extra **and** a CUDA GPU |
| `tutorial` | executes each tutorial notebook end-to-end  | the `examples` extra; slow        |

```bash
# Skip the GPU tests (e.g. no CUDA GPU available):
uv run pytest tests/ -m "not warp"

# Skip both GPU and the slow notebook tests (what CI runs):
uv run pytest tests/ -m "not warp and not tutorial"

# Run only one group:
uv run pytest tests/ -m warp
uv run pytest tests/ -m tutorial
```

Notes:

- Tests marked `warp` are **automatically skipped** if the `warp` extra is not
  installed (default dev installs omit it), so you only need `-m "not warp"` to
  exclude them when warp *is* installed but you have no GPU.
- Rendering tests need a headless OpenGL context. On runners/machines where that
  is unavailable (e.g. macOS/Windows CI), set `SKIP_RENDERING_TESTS=1` to skip
  them; on Linux they run via EGL/Mesa.

## Profiling

The replay end-to-end test scripts double as end-to-end profiling targets. Both take a
`--profile PATH` flag. Run them without `--save-data` to profile the pure
simulation pipeline (no rendering/video). See
[the installation notes](https://neuromechfly.org/installation/) for the profiler
prerequisites (`py-spy` and `nvtx` come with the `dev` extra; `nsys` must be
installed separately).

**CPU (`scripts/replay_behavior_cpu.py`)** — sampling profile via
[py-spy](https://github.com/benfred/py-spy), written in
[speedscope](https://www.speedscope.app/) format:

```bash
uv run python scripts/replay_behavior_cpu.py --save-data outputs/cpu_sim --profile outputs/cpu.speedscope.json
```

py-spy attaches to the **already-running process for the simulation loop only**, so
the one-time imports and model building (which otherwise dominate the flame graph as
a tall `_find_and_load` / `exec_module` tower) are never sampled — what you see is
the loop itself. Sampling uses `--native`, so native (C/C++) frames — notably
MuJoCo's physics step (`mj_projectConstraint`, `mju_cholFactorNumeric`, …) — appear
alongside the Python frames. Open the resulting file at
[speedscope.app](https://www.speedscope.app/) (or with the `speedscope` CLI). On
Linux this needs no `sudo` (the script nominates py-spy as an allowed tracer via
`prctl(PR_SET_PTRACER)`); a longer `--sim-duration-sec` just yields more samples.

For a complementary, **symbolication-free** view of where time goes *inside* the
physics step (kinematics, collision broad/narrow-phase, constraint solve,
integration, …), pass `--mujoco-timing`. This installs MuJoCo's internal timer
callback (`mjcb_time`) so the C engine fills in `mjData.timer` per phase, then
prints a per-phase breakdown after the run:

```bash
uv run python scripts/replay_behavior_cpu.py --mujoco-timing
```

It adds a per-phase callback overhead, so read it as a *relative* breakdown rather
than an absolute-throughput measurement. The two views corroborate each other — the
phases MuJoCo's timer flags as expensive should match the hot native frames
(`mj_projectConstraint`, `mju_cholFactorNumeric`, …) in the flame graph.

**GPU (`scripts/replay_behavior_gpu.py`)** — timeline via
[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) (`nsys`):

```bash
uv run python scripts/replay_behavior_gpu.py --save-data outputs/gpu_sim --profile outputs/gpu_profile
```

This re-executes the script under `nsys profile` and writes `profile.nsys-rep`,
which you open in the Nsight Systems GUI. The timeline is annotated with NVTX
ranges (via Warp's `wp.ScopedTimer(use_nvtx=True)`) — `warmup`, `timed_run`, and
per-step `step` — so JIT/warm-up is separated from the steady-state loop and
host-side launches line up with the captured CUDA-graph kernels.

Profiling output files (`*.speedscope.json`, `*.nsys-rep`, …) are gitignored.

## Submitting changes

1. Fork the repository and create a branch **from the current `dev-vx.y.z` branch**.
2. Make your changes and add tests if applicable. Use `uv` for package management.
3. Run the test suite (see [Running tests](#running-tests) above): `uv run pytest tests/`
4. Open a pull request against the current `dev-vx.y.z` with a clear description of the change.

## Code style

- Follow [Ruff](https://docs.astral.sh/ruff/).
- Type-annotate public functions and classes.
- Keep docstrings concise.

## Adding yourself as a contributor

Feel free to add your name to [CONTRIBUTORS.md](CONTRIBUTORS.md) in your PR. It will
appear in the next release if your PR is accepted.

## License

By contributing, you agree that your contributions will be licensed under the [specified license](LICENSE).
