# Installation

=== "Using pip"

    !!! tip "Virtual Environments"

        We recommend that you create a virtual environment (using [venv](https://docs.python.org/3/library/venv.html), [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), [uv](https://docs.astral.sh/uv/), [Poetry](https://python-poetry.org/), etc.).

    FlyGym is published on the [Python Package Index (PyPI)](https://pypi.org/project/flygym/). You can use FlyGym using `pip` directly. If you only care about the basic functionalities:

    ```sh
    pip install flygym
    ```

    Add the `warp` optional dependency if you want to use fly.warp with GPU acceleration:
    
    ```sh
    pip install flygym[warp]
    ```

    Add the `examples` optional dependency if you want to follow the tutorials:

    ```sh
    pip install flygym[examples]
    ```

    You can combine multiple optional dependencies in one command. For example:

    ```sh
    pip install flygym[warp,examples]
    ```


=== "In editable model for development"

    FlyGym uses `uv` for package management. Install `uv` following its [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

    Clone the FlyGym repository:

    ```sh
    git clone git@github.com:NeLy-EPFL/flygym.git
    cd flygym
    ```

    Then install using `uv`. The following installs all optional dependencies, but you can

    * remove `--extra warp` if you don't care for GPU acceleration, or if you don't have a NVIDIA GPU (e.g., you're on a Mac),
    * remove `--extra dev` if you don't care for development tools (e.g., unit testing, automatic doc generation), or
    * remove `--extra examples` if you don't care for running the tutorials.
    
    ```sh
    uv sync --extra warp --extra dev --extra examples
    ```

    !!! note

        Developers should also install the `nbstripout` filter, which removes bulky Jupyter Notebook outputs upon `git add` (without touching the actual files—the outputs are only removed in the version-controled files). Note that this does not integrate well with IDEs, so you should run `git add <files>` in command line instead of using, for example, the GUI in VS Code.

        ```sh
        source .venv/bin/activate
        nbstripout --install --attributes .gitattributes
        ```

=== "Using Docker"

    Forthcoming.

=== "Online using Google Colab"

    Forthcoming.

!!! warning "Special notes for rendering on machines without a display"

    If you are using a machine without a display (e.g. a server), you will need to change the renderer to EGL (see this link for details). This requires setting the following environment variables before running FlyGym:

    ```sh
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    ```

    If you want to change this setting by default, you can add the two lines above to the end of your `.bashrc` file.

    If you are using a Conda environment, you can change the environment variables as follows (replacing my-env-name accordingly), and then re-activate the environment:

    ```sh
    conda activate my-env-name
    conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    ```

    You might need to install EGL-related dependencies on your machine. For example, on some Ubuntu/Debian systems, you might need to install the following:

    ```sh
    apt-get install libegl1-mesa-dev
    ```
