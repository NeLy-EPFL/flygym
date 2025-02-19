Installation
============


FlyGym can be installed in one of three ways:

Option 1: Installation in your separately managed Python environment
--------------------------------------------------------------------

If you are using FlyGym as a library in your own code, you might want to use FlyGym in your separately managed Python environment, you can install FlyGym via the `FlyGym package on the Python Package Index (PyPI) <https://pypi.org/project/flygym/>`_. Note that you are responsible of managing the Python environment in this case, subjected to the dependencies and version constraints of FlyGym.

In the most basic case, if you just want the bare core of FlyGym, you can install FlyGym with a single command:
   
.. code-block:: bash

   pip install "flygym"

Additionally, you can also install the following optional dependencies:

.. code-block:: bash

   # If you want to run the examples published in our paper, you need to install
   # additional dependencies for fluid dynamics simulation, network analysis,
   # deep learning, etc. You can do this by running:
   pip install "flygym[examples]"

   # If you want to contribute the FlyGym, you might want to install additional
   # tools for documentation, testing, and code formatting. You can do this by
   # running:
   pip install "flygym[dev]"
   
   # If you want both, run:
   pip install "flygym[examples,dev]"


**Importantly,** if you want to run FlyGym on a machine without a display (e.g. a server or a node on a cluster), please check the special instructions at the bottom of this page.



Option 2: Installation with strict dependency specification using Poetry
------------------------------------------------------------------------

If you want the exact dependencies and versions that we use in our development, you can install FlyGym using `Poetry <https://python-poetry.org/>`_—a dependency management and packaging tool that ensures a reproducible environment. In brief, we have generated a full recipe of the exact versions of libraries required by FlyGym. This recipe, contained in the `poetry.lock` file, is part of the FlyGym package. You can simply download the FlyGym repository and ask Poetry to install the dependencies according to this recipe.

This is the recommended method of installation for the following cases:

1. You require a strict dependency specification to ensure that things work out of the box—for example, if you are using FlyGym for educational purposes in the classroom.
2. You are a developer who wants to contribute to FlyGym and wants to ensure that your changes are compatible with the rest of the codebase.
3. You are already maintaining the rest of your codebase with Poetry.

To use this method, you first need to install Poetry (and ``pipx`` if you don't have it already, because Poetry is installed with ``pipx``). You can find instructions on how to do this on `Poetry's online documentation <https://python-poetry.org/docs/#installation>`_.

Once you have Poetry installed, you can install FlyGym by running the following commands:

.. code-block:: bash

   # Clone the FlyGym repository
   git clone https://github.com/NeLy-EPFL/flygym.git
   # Alternatively, use SSH if you have it set up already or if you want to
   # contribute to the FlyGym project:
   # git clone https://github.com/NeLy-EPFL/flygym.git

   # Change into the FlyGym directory
   cd flygym

   # Install with Poetry
   poetry install

   # If you want to run the examples published in our paper, you need to install
   # additional dependencies for fluid dynamics simulation, network analysis,
   # deep learning, etc. You can do this by running:
   poetry install -E examples

   # If you want to contribute the FlyGym, you might want to install additional
   # tools for documentation, testing, and code formatting. You can do this by
   # running:
   poetry install -E dev


**Importantly,** if you want to run FlyGym on a machine without a display (e.g. a server or a node on a cluster), please check the special instructions at the bottom of this page.


Option 3: Installation with Docker
----------------------------------
`"Containerization" <https://en.wikipedia.org/wiki/Containerization_(computing)>`_ is a way of virtualization that aims to bundle an application and its dependencies into a single portable, executable unit called a *container*. `Docker <https://docs.docker.com/guides/docker-overview/>`_ is a popular platform for developing, shipping, and running containers, making it easier to manage and deploy applications in a consistent manner.

Instead of installing FlyGym on your machine directly, you can also install Docker on your machine and run FlyGym through Docker. This might be particularly helpful if you are using container-as-a-service (CaaS) systems such as Kubernetes to train or deploy models at scale. We provide a Docker image with FlyGym and its dependencies pre-installed and publish it to `Docker Hub <https://hub.docker.com/r/nelyepfl/flygym>`_. This image is defined by the `Dockerfile <https://github.com/NeLy-EPFL/flygym/blob/main/Dockerfile>`_ at the root level of the directory.

For more information about how to interact with Docker, please refer to the `official Docker guides <https://docs.docker.com/guides/>`_. Note that you need to `install NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ if you would like to use an NVIDIA GPU from the container.


Cross-platform compatibility
----------------------------

FlyGym supports Linux, macOS, and Windows, although rendering on headless Mac and Windows servers (i.e., without displays) is not tested. To render on a headless Linux server, follow the special instructions at the bottom of this page.

FlyGym is automatically tested in the following setups:

- Linux: latest version of Ubuntu, with Python 3.10, 3.11, and 3.12.
- macOS: latest version of macOS (Apple silicon) and macOS 13 Ventura (pre Apple silicon), with Python 3.12, without rendering. We do not support the examples provided in the paper on macOS 13 Ventura. This is because Macs with Intel chips are not supported by the latest version of PyTorch.
- Windows: latest version of Windows, with Python 3.12, without rendering.


Special notes for rendering on machines without a display
---------------------------------------------------------

If you are using a machine without a display (e.g. a server), you will need to change the renderer to EGL (see `this link <https://pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html#prerequisite-for-rendering-all-mujoco-versions>`_ for details). This requires setting the following environment variables before running FlyGym:

.. code-block:: bash

   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl


If you want to change this setting by default, you can add the two lines above to the end of your ``.bashrc`` file.

If you are using a Conda environment, you can change the environment variables as follows (replacing ``my-env-name`` accordingly), and then re-activate the environment:

.. code-block:: bash

   conda activate my-env-name
   conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

You might need to install EGL-related dependencies on your machine. For example, on some Ubuntu/Debian systems, you might need to install the following:

.. code-block:: bash

   apt-get install libegl1-mesa-dev


Troubleshooting
---------------

- ``AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'?`` during ``opencv-python`` installation:
  
  - This appears to be an issue with ``opencv-python`` on certain Mac systems with Python 3.12. Please refer to `this GitHub issue <https://github.com/opencv/opencv-python/issues/988>`_. Temporary fixes (from simple to complex) include:
  
    - Use Python 3.11 instead.
    - Uninstall ``opencv-python``, clone ``opencv-python`` from GitHub, remove the line ``"setuptools==59.2.0",`` in its ``pyproject.toml``, install an up-to-date version of ``setuptools`` (e.g., 70.0.0), and install ``opencv-python`` locally from the cloned directory. Then, continue with the FlyGym installation.
    - Use Docker to run FlyGym (see above).