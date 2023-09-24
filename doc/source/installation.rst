Installation
============


Installation via PyPI 
---------------------
The easiest way to install FlyGym is via PyPI. Before you start, you might want to create a Python virtual environment with virtualenv or Conda. For example, with Conda:

.. code-block:: bash

   conda create -n flygym python=3.11    # flygym is tested on Python 3.7-3.11
   conda activate flygym    # run this every time you use the environment

Then, install the package:

.. code-block:: bash

   pip install "flygym[mujoco]"

.. note::
   
   The quotation marks around the package name are important if you are using zsh (the default shell on Macs). Without them, ``pip`` will not receive ``flygym[mujoco]`` as a single string.

Please note that the ``pip`` installation does not easily allow the user to modify the source code. If you want to modify the source code, please follow the instructions below for developer installation.


Developer installation
----------------------

First, clone this repository:

.. code-block:: bash

   git clone git@github.com:NeLy-EPFL/flygym.git

Change into the cloned directory:

.. code-block:: bash

   cd flygym

You might want to create a Python virtual environment with virtualenv or Conda. For example, with Conda:

.. code-block:: bash

   conda create -n flygym python=3.11    # flygym is tested on Python 3.7-3.11
   conda activate flygym    # run this every time you use the environment

Next, install the FlyGym package:

.. code-block:: bash

   pip install -e ."[mujoco]"

Note that the ``-e`` causes the package to be installed in editable mode. This means that you can modify the source code and the changes will be reflected in the installed package. This is useful if you want to modify modify the FlyGym package itself in your work — in which case we ask you to consider sharing your developments with us via a pull request to make it available to the community. Please refer to `the contribution guide <https://neuromechfly.org/contributing.html>`_ for more information.

Finally, developers should also intstall the ``dev`` dependencies for testing and documentation:

.. code-block:: bash

   pip install -e ."[dev]"