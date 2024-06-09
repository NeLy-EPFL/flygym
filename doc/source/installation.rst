Installation
============


.. note:: 

   Check special instructions at the bottom of this page if you want to run FlyGym on a machine without a display (e.g. a server).


Installation via PyPI 
---------------------
The easiest way to install FlyGym is via PyPI. Before you start, you might want to create a Python virtual environment with virtualenv or Conda. For example, with Conda:

.. code-block:: bash

   conda create -n flygym python=3.12    # flygym is tested on Python 3.9-3.12
   conda activate flygym    # run this every time you use the environment

Then, install the package:

.. code-block:: bash

   pip install "flygym"

Please note that the ``pip`` installation does not easily allow the user to modify the source code. If you want to modify the source code, please follow the instructions below for developer installation.


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

Finally, developers should also install the ``dev`` dependencies for testing and documentation:

.. code-block:: bash

   pip install -e ."[dev]"

.. note::

   The quotation marks around the package name are important if you are using zsh (the default shell on Macs). Without them, ``pip`` will not receive ``flygym[dev]`` as a single string.


Special notes for rendering on machines without a display
---------------------------------------------------------

If you are using a machine without a display (e.g. a server), you will need to change the renderer to EGL (see `this link <https://pytorch.org/rl/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html#prerequisite-for-rendering-all-mujoco-versions>`_ for details). This requires setting the following environment variables before running FlyGym:

.. code-block:: bash

   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl


If you want to change this setting by default, you can add the two lines above to the end of your ``.bashrc`` file.


If you are using a Conda environment, you can change the environment variables as follows (replacing ``my-env-name`` accordingly), and then re-activate the environment:

.. code-block:: bash

   conda activate my-env-name
   conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl