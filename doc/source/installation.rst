Installation
============


FlyGym can be installed as a regular Python package via `PyPI <https://pypi.org/project/flygym/>`_ or as a developer installation from the source code. In addition, we provide a Docker image with FlyGym and its dependencies pre-installed. Below, we provide instructions for each of these methods.


Installation via PyPI 
---------------------
The easiest way to install FlyGym is via PyPI. Before you start, you might want to create a Python virtual environment with virtualenv or Conda. For example, with Conda:

.. code-block:: bash

   conda create -n flygym python=3.12    # flygym is tested on Python 3.9-3.12
   conda activate flygym    # run this every time you use the environment

Then, to install the FlyGym package:

.. code-block:: bash

   pip install "flygym"

.. important::
   
   **Headless machines**: If you want to run FlyGym on a machine without a display (e.g. a server or a node on a cluster), please check the special instructions at the bottom of this page.

.. note::

   **(Optional) Additional dependencies for provided tutorials and examples:** A number of additional dependencies are required to run the provided `tutorials <https://neuromechfly.org/tutorials/index.html>`_ and `examples <https://github.com/NeLy-EPFL/flygym/tree/main/flygym/examples/>`_. To install these dependencies, you can run the following command:

   .. code-block:: bash

      pip install "flygym[examples]"
   
.. Note::

   **(Optional) Additional tools for developers:** Developers of FlyGym use a number of other tools for automated documentation generation, testing, and code linting. These tools are not required to use FlyGym as it is, but can be useful if you wish to modify the code or contribute to the project. To install these tools, you can run:

   .. code-block:: bash

      pip install "flygym[dev]"

   Note, however, that ``pip`` installation does not easily allow the user to modify the source code. If you want to modify the source code, please follow the instructions below for developer installation.


Developer installation
----------------------

First, clone this repository:

.. code-block:: bash

   git clone git@github.com:NeLy-EPFL/flygym.git

Change into the cloned directory:

.. code-block:: bash

   cd flygym

If you want to install code from a specific branch, you can checkout to the branch of your choice:

.. code-block:: bash

   git checkout <branch_name>

You might want to create a Python virtual environment with virtualenv or Conda. For example, with Conda:

.. code-block:: bash

   conda create -n flygym python=3.12    # flygym is tested on Python 3.9-3.12
   conda activate flygym    # run this every time you use the environment

Next, install the FlyGym package:

.. code-block:: bash

   pip install -e .

Note that the ``-e`` causes the package to be installed in editable mode. This means that you can modify the source code and the changes will be reflected in the installed package. This is useful if you want to modify modify the FlyGym package itself in your work — in which case we ask you to consider sharing your developments with us via a pull request (PR) to make it available to the community. Please refer to `the contribution guide <https://neuromechfly.org/contributing.html>`_ for more information.

Developers should also install the ``dev`` dependencies for testing and documentation:

.. code-block:: bash

   pip install -e ."[dev]"

.. note::

   The quotation marks around the package name are important if you are using zsh (the default shell on Macs). Without them, ``pip`` will not receive ``flygym[dev]`` as a single string.

Finally, if you want to install the additional dependencies required to run the provided examples (eg. NetworkX, PyTorch, etc.), run:

.. code-block:: bash

   pip install -e ."[examples]"


Cross-platform compatibility
----------------------------

FlyGym supports Linux, macOS, and Windows, although rendering on headless Mac and Windows servers (i.e., without displays) is not tested. To render on a headless Linux server, follow the special instructions at the bottom of this page.

FlyGym is automatically tested in the following setups:

- Linux: latest version of Ubuntu, with Python 3.9, 3.10, 3.11, and 3.12.
- macOS: latest version of macOS (Apple silicon) and macOS 13 Ventura (pre Apple silicon), wih Python 3.12, without rendering.
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


Docker image
------------

`"Containerization" <https://en.wikipedia.org/wiki/Containerization_(computing)>`_ is a way of virtualization that aims to bundle an application and its dependencies into a single portable, executable unit called a *container*. `Docker <https://docs.docker.com/guides/docker-overview/>`_ is a popular platform for developing, shipping, and running containers, making it easier to manage and deploy applications in a consistent manner.

Instead of installing FlyGym on your machine directly, you can also install Docker on your machine and run FlyGym through Docker. This might be particularly helpful if you are using container-as-a-service (CaaS) systems such as Kubernetes to train or deploy models at scale. We provide a Docker image with FlyGym and its dependencies pre-installed and publish it to `Docker Hub <https://hub.docker.com/r/nelyepfl/flygym>`_. This image is defined by the `Dockerfile <https://github.com/NeLy-EPFL/flygym/blob/main/Dockerfile>`_ at the root level of the directory.

For more information about how to interact with Docker, please refer to the `official Docker guides <https://docs.docker.com/guides/>`_. Note that you need to `install NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ if you would like to use an NVIDIA GPU from the container.


Troubleshooting
---------------

- ``AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'?`` during ``opencv-python`` installation:
  
  - This appears to be an issue with ``opencv-python`` on certain Mac systems with Python 3.12. Please refer to `this GitHub issue <https://github.com/opencv/opencv-python/issues/988>`_. Temporary fixes (from simple to complex) include:
  
    - Use Python 3.11 instead.
    - Uninstall ``opencv-python``, clone ``opencv-python`` from GitHub, remove the line ``"setuptools==59.2.0",`` in its ``pyproject.toml``, install an up-to-date version of ``setuptools`` (e.g., 70.0.0), and install ``opencv-python`` locally from the cloned directory. Then, continue with the FlyGym installation.
    - Use Docker to run FlyGym (see above).