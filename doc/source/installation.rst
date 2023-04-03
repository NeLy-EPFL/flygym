Installation
============

General Information
-------------------

First, clone this repository:

.. code-block:: bash

   git clone git@github.com:NeLy-EPFL/flygym.git

Change into the cloned directory:

.. code-block:: bash

   cd flygym

You might want to create a Python virtual environment with virtualenv or Conda. For example, with Conda:

.. code-block:: bash

   conda create -n flygym python=3.8
   conda activate flygym    # run this every time you use the environment
    
.. note:: 
   The required Python depends on the requirement of underlying physics engine. The package is primarily developed and tested with Python 3.8, although a wider range of Python versions should work in principle.

Next, install the FlyGym package. There are three versions of the package, corresponding to three physics engines: PyBullet, MuJoCo, and Isaac Gym. You can install any of them, or all of them. PyBullet and MuJoCo are open source, while Isaac Gym version is a commercial product (though free to use as of April 2023).

To install the PyBullet version, run:

.. code-block:: bash

   pip install -e ."[pybullet]"

To install the MuJoCo version, run:

.. code-block:: bash

   pip install -e ."[mujoco]"

To install the Isaac Gym version, join the `NVIDIA preview program`_ and follow the instruction there to download and install Isaac Gym. Then, run:

.. code-block:: bash

   pip install -e ."[isaacgym]"

Developers of the package should also install documentation utilities:

.. code-block:: bash

   pip install -e ."[doc]"

Note that ``-e`` causes the package in editable mode. This is not necessary if you're not developing this package.

.. _NVIDIA preview program: https://developer.nvidia.com/isaac-gym


Google Colab for MuJoCo Version
-------------------------------

The MuJoCo version can be installed on `Google Colab`_ by running the following in a code block::

    #@title Install `flygym` on Colab

    # This block is modified from dm_control's tutorial notebook
    # https://github.com/deepmind/dm_control/blob/main/tutorial.ipynb

    import subprocess
    if subprocess.run('nvidia-smi').returncode:
        raise RuntimeError(
            'Cannot communicate with GPU. '
            'Make sure you are using a GPU Colab runtime. '
            'Go to the Runtime menu and select Choose runtime type.')

    print('Installing flygym')
    !pip install -q --progress-bar=off 'flygym[mujoco] @ git+https://github.com/NeLy-EPFL/flygym.git'

    # Configure dm_control to use the EGL rendering backend (requires GPU)
    %env MUJOCO_GL=egl

    print('Checking that the dm_control installation succeeded...')
    try:
        from dm_control import suite
        env = suite.load('cartpole', 'swingup')
        pixels = env.physics.render()
    except Exception as e:
        raise e from RuntimeError(
            'Something went wrong during dm_control installation. Check the shell '
            'output above for more information.\n'
            'If using a hosted Colab runtime, make sure you enable GPU acceleration '
            'by going to the Runtime menu and selecting "Choose runtime type".')
    else:
        del pixels, suite

    print('Checking that the flygym installation succeeded...')
    try:
        import flygym
        from flygym import envs
    except Exception as e:
        raise e from RuntimeError(
            'Something went wrong during flygym installation. Check the shell '
            'output above for more information.\n')
    else:
        del envs, flygym

.. _Google Colab: https://colab.research.google.com/