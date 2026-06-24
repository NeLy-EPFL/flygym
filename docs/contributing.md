# Contributing to FlyGym

Thank you for your interest in contributing! Here's how to get started.

## Reporting issues & reaching out

Use [GitHub Issues](https://github.com/NeLy-EPFL/flygym/issues) for bugs and feature requests.
For questions and discussion, use the [Discussion Board](https://github.com/NeLy-EPFL/flygym/discussions).
We prefer these channels over emails addressed to the corresponding authors of the publications.

Feel free to reach out to us before implementing any new feature.

## Setting up a development environment

Follow the [installation guide](installation.md) to install FlyGym for development.
Be sure to follow the **"Using `uv` (for development)"** tab.

!!! warning "Install the `nbstripout` filter"

    Be sure to follow the note on the installation page regarding `nbstripout`.
    This step prevents Jupyter Notebook output blocks from being included in the commit
    history. These data (e.g., images and videos) are large and can make the Git repo
    bloated, thus slowing down CI workflows and Colab use cases. During development,
    **do not** use IDEs to stage notebook files—they might skip the `nbstripout`
    filter. Always run `git add <notebook_files>` in the terminal.

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
| `rl`       | imitation-learning / RL tests               | the `rl` extra (Gymnasium + SB3)  |
| `network`  | hits the live S3 asset bucket               | network access (self-skips if unreachable) |

```bash
# Skip the GPU tests -- this is what CI runs (no GPU, but notebooks included):
uv run pytest tests/ -m "not warp"

# Also skip the slow notebook tests (faster local iteration):
uv run pytest tests/ -m "not warp and not tutorial"

# Run only one group:
uv run pytest tests/ -m warp
uv run pytest tests/ -m tutorial
```

!!! note

    - Tests marked `warp` are **automatically skipped** if the `warp` extra is not
      installed (default dev installs omit it), so you only need `-m "not warp"` to
      exclude them when warp *is* installed but you have no GPU.
    - Rendering tests need a headless OpenGL context. On runners/machines where that
      is unavailable (e.g. macOS/Windows CI), set `SKIP_RENDERING_TESTS=1` to skip
      them; on Linux they run via EGL/Mesa.

## Profiling

The replay end-to-end test scripts double as performance-profiling targets. See the
[Performance profiling](tutorials/7_performance_profiling.md) guide for the CPU and GPU
workflows.

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

Feel free to add your name to the [Contributors](contributors.md) page in your PR. It
will appear in the next release if your PR is accepted.

## License

By contributing, you agree that your contributions will be licensed under the
[specified license](https://github.com/NeLy-EPFL/flygym/blob/main/LICENSE).
