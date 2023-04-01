Installation
============

Local Installation
------------------

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

   pip install -e ."[docs]"

Note that ``-e`` causes the package in editable mode. This is not necessary if you're not developing this package.

.. _NVIDIA preview program: https://developer.nvidia.com/isaac-gym


Google Colab
------------

TODO