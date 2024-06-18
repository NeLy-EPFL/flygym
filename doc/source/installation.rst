Installation
============


Installation via PyPI 
---------------------
The easiest way to install FlyGym is via PyPI. Before you start, you might want to create a Python virtual environment with virtualenv or Conda. For example, with Conda:

.. code-block:: bash

   conda create -n flygym python=3.12    # flygym is tested on Python 3.9-3.12
   conda activate flygym    # run this every time you use the environment

(Optional) If you would like to interface FlyGym with the [connectome-constrained vision model](https://github.com/TuragaLab/flyvis) from [Lappalainen et al. (2023)](https://www.biorxiv.org/content/10.1101/2023.03.11.532232), you need to install the FlyVision package separately. Since FlyVision is not published on the Python Package Index (PyPI), you must either install it manually following [its installation instructions](https://github.com/TuragaLab/flyvis?tab=readme-ov-file#install-locally-), or install it with `pip` from our fork on GitHub:

.. code-block:: bash

   pip install "flyvision @ https://github.com/Nely-EPFL/flyvis/archive/refs/heads/main.zip"

(Required) Then, to install the FlyGym package:

.. code-block:: bash

   # if you only want the basic package:
   pip install "flygym"

   # to install additional dependencies required to run the provided examples:
   pip install "flygym[examples]"

   # to install dev tools such as pytest (automated testing) and sphinx (docs building):
   pip install "flygym[dev]"
   
   # to install everything:
   pip install "flygym[examples,dev]"
   

Please note that the ``pip`` installation does not easily allow the user to modify the source code. If you want to modify the source code, please follow the instructions below for developer installation.

If you want to run FlyGym on a machine without a display (e.g. a server or a node on a cluster), please check the special instructions at the bottom of this page.


Developer installation
----------------------

First, clone this repository:

.. code-block:: bash

   git clone git@github.com:NeLy-EPFL/flygym.git

If you want to install code from a specific branch, you can checkout to the branch of your choice:

.. code-block:: bash

   git checkout <branch_name>

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