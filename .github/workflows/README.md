- `black.yaml`: Check code style formatting
- `documentation.yaml`: Check if documentation site ([neuromechfly.org](https://neuromechfly.org/)) builds successfully.
- `check_poetry_lock_*.yaml`: Check if the dependencies can be successfully resolved through the [Poetry package manager](https://python-poetry.org/) on various OS's and systems.
- `tests_full.yaml`: Test codebase on all supported Python versions using the `ubuntu-latest` image.
- `tests_docker.yaml`: Check if FlyGym installation is successful on [Docker](https://www.docker.com/)—a widely used [containerization](https://en.wikipedia.org/wiki/Containerization_(computing)) software.
- `tests_macos_apple_silicon.yaml`, `tests_macos_intel.yaml`, `tests_ubuntu22.yaml`, `tests_windows.yaml`: Test codebase on all supported systems (but only with Python 3.12).
    - Note that in `tests_macos_intel.yaml`, NumPy must be manually downgraded to v1.* (`pip install "numpy<2"`) after FlyGym is installed. See [Cross-platform compatibility](https://neuromechfly.org/installation.html#cross-platform-compatibility).


Exact OS images can be found under `os:` inside the YAML files. These runners are provided by GitHub (see [GitHub Action runners](https://docs.github.com/en/actions/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners)).