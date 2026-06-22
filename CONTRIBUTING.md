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

The replay smoke-test scripts double as end-to-end profiling targets. Both take a
`--profile PATH` flag. Run them without `--save-data` to profile the pure
simulation pipeline (no rendering/video). See
[the installation notes](https://neuromechfly.org/installation/) for the profiler
prerequisites (`py-spy` and `nvtx` come with the `dev` extra; `nsys` must be
installed separately).

**CPU (`scripts/replay_behavior_cpu.py`)** — sampling profile via
[py-spy](https://github.com/benfred/py-spy), written in
[speedscope](https://www.speedscope.app/) format:

```bash
uv run python scripts/replay_behavior_cpu.py --save-data outputs/cpu_smoketest --profile outputs/cpu.speedscope.json
```

This re-executes the script under `py-spy record --native`, so native (C/C++)
frames — notably MuJoCo's physics step — appear alongside the Python frames. Open
the resulting file at [speedscope.app](https://www.speedscope.app/) (or with the
`speedscope` CLI).

**GPU (`scripts/replay_behavior_gpu.py`)** — timeline via
[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) (`nsys`):

```bash
uv run python scripts/replay_behavior_gpu.py --save-data outputs/gpu_smoketest --profile outputs/gpu_profile
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
