Installation
============

.. contents:: **Table of Contents**
   :local:
   :class: this-will-duplicate-information-and-it-is-still-useful-here
   :depth: 1


Installing MuJoCo or PyBullet versions locally
----------------------------------------------

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

If you're developing this package rather than just using it, you should also install documentation utilities:

.. code-block:: bash

   pip install -e ."[doc]"

Note that ``-e`` causes the package in editable mode. This is not necessary if you're not developing this package.


Using the MuJoCo version on Google Colab
----------------------------------------

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

.. note:: 

   In the ``pip install`` command, you can add ``@branch-name`` at the end of the GitHub URL to install a specific branch (the default branch is ``main``). For example, to install the ``dev`` branch, the line should read ``!pip install -q --progress-bar=off 'flygym[mujoco] @ git+https://github.com/NeLy-EPFL/flygym.git@dev'``

.. _Google Colab: https://colab.research.google.com/


Installing the Isaac Gym version
--------------------------------

To install the Isaac Gym version, join the `NVIDIA preview program`_ and follow the instruction there to download Isaac Gym. Note that as of the Preview 4 version, the minimum NVIDIA driver version is 470.

As of the Preview 4 version, Isaac Gym natively requires Ubuntu 18.04 or 20.04 and Python 3.6, 3.7, or 3.8. This is can be hard to satisfy, espesially on a cluster or cloud environment. In that case, the package can be installed in a Docker container.

Installing without Docker
~~~~~~~~~~~~~~~~~~~~~~~~~

First, follow NVIDIA's instruction to install the Isaac Gym package in Conda environment. Follow the instructions in ``docs/install.html`` under the downloaded ``isaacgym`` folder. Activate this environment with ``conda activate``.

Then, clone this repository and install FlyGym:

.. code:: bash

   git clone git@github.com:NeLy-EPFL/flygym.git
   cd flygym
   pip install -e ."[isaacgym]"


If you're developing this package rather than just using it, you should also install documentation utilities:

.. code-block:: bash

   pip install -e ."[doc]"

Note that ``-e`` causes the package in editable mode. This is not necessary if you're not developing this package.


Installing with Docker
~~~~~~~~~~~~~~~~~~~~~~

.. note::

   * **Docker** is a platform for deploying and managing applications using units called containers.
   * A Docker **image** is a template containing an application's dependencies and runtime environment, used for creating containers. An image ensures a consistent and reproducible execution environment across different machines. Docker images is specified by Dockerfiles.
   * A Docker **container** is a running instance of a Docker image. They can be started, stopped and destroyed as needed. They are analogous to virtual machines, but more lightweight
   
   Essentially, what we are doing here is to build a Docker image for Isaac Gym using a Dockerfile provided by NVIDIA, and then build a Docker image for FlyGym on top of it. Finally, we start a FlyGym container in which you can interact with the Isaac Gym version of the NeuroMechFly environment.

First, install the `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`_ following instructions in the link. This will install Docker with NVIDIA GPU support. You might want to follow `these instructions <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_ to run Docker as a non-root user (note that you need to restart the host machine for the changes to fully take effect). If you are using a Linux distro other than Debian or Ubuntu, you might also want to follow `these additional instructions <https://docs.docker.com/engine/install/linux-postinstall/#configure-docker-to-start-on-boot-with-systemd>`_ to start Docker on boot.

Next, follow instruction under ``docs/install.html`` in the downloaded ``isaacgym`` folder. See "Install in a Docker container" in particular. There should be a ``build.sh`` script that allows you to build an image called ``isaacgym``. Verify that the ``isaacgym`` image has been built by running ``docker images``.

Then, clone the FlyGym repository and build the FlyGym image for the Isaac Gym version:

.. code-block:: bash

   git clone git@github.com:NeLy-EPFL/flygym.git
   cd flygym
   bash docker/isaacgym/build.sh

Verify that a new ``flygym_ig``` image has been built by running ``docker images`` again.

To run the image, run:

.. code:: bash

   # With GUI:
   bash docker/isaacgym/run.sh $DISPLAY

   # or, to run it without the GUI:
   bash docker/isaacgym/run.sh

This will start a container named ``flygym_ig_container``. You will enter a bash session where

* Python 3.8, Isaac Gym, FlyGym, and PyTorch have been installed;
* You have access to the GPU (you can test this by running ``nvidia-smi``);
* You can run GUI applications (including Isaac Gym's GUI);
* You have network access through the host;
* The ``flygym`` folder on the host machine (outside the container) has been mounted to ``~/flygym`` inside the container. Any changes made to the files in the container will be reflected on the host machine, vice versa;
* Additionally, the ``~/share/flygym_ig_container_data`` folder on the host machine has been created and mounted to ``~/data`` inside the container. This is useful for saving other data generated by the container (eg. program input/output).

You can leave the bash session by running ``exit`` or pressing Control+D. The container will exit, but you can still view it by running ``docker ps --all``. To reattach to the container, first run ``docker start flygym_ig_container`` to restart it (the container should be named ``flygym_ig_container`` if you started it using the ``run.sh`` script; otherwise change it accordingly); then run ``docker attach flygym_ig_container`` to reattach to the bash session. **If you are using the GUI, you need to run ``xhost + local:`` on the host machine before restarting the container.**

To delete the container, run ``docker rm flygym_ig_container``. **Note that all data will be lost (unlesses saved under a mounted Docker volume).** To delete the image, run ``docker rmi flygym_ig``. Modify the names accordingly if you have changed them.

Since the ``flygym`` folder is mounted as a volume and files can be accessed from inside and outside the container, you can develop FlyGym in the container and manage your code and documentation using Git and Sphinx on the host machine. You can also use your favorite IDE on the host machine to edit the code.


.. _NVIDIA preview program: https://developer.nvidia.com/isaac-gym
.. _NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker