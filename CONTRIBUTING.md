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


## Submitting changes

1. Fork the repository and create a branch **from the current `dev-vx.y.z` branch**.
2. Make your changes and add tests if applicable. Use `uv` for package management.
3. Run the test suite: `uv run pytest tests/`
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
