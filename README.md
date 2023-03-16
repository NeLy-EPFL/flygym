## FlyGym: Gymnasium environments for NeuroMechFly

This package implements [Gymnasium](https://gymnasium.farama.org) environments for NeuroMechFly ([paper](https://doi.org/10.1038/s41592-022-01466-7), [code](https://github.com/NeLy-EPFL/NeuroMechFly)), a neuromechanical model of the fruit fly _Drosophila melanogaster_. Gymnasium is "a standard API for reinforcement learning, and a diverse collection of reference environments." The environments in this package serve as wrappers to provide a unified interface to interact with fly model in different physics simulators (PyBullet, MuJoCo, Isaac Gym).

### Installation
1. Clone this repository: `git clone git@github.com:NeLy-EPFL/flygym.git`
2. Change into the cloned directory: `cd flygym`
3. Create a Python virtual environment with virtualenv or Conda.
4. To install the PyBullet or MuJoCo version, run `pip install -e ."[pybullet]"` or `pip install -e ."[mujoco]"`, or both.
5. Isaac Gym is still in preview. To install the Isaac Gym version, join the [NVIDIA preview program](https://developer.nvidia.com/isaac-gym) and follow the instruction there to download and install Isaac Gym. Then, under the `flygym` directory, simply run `pip install -e .` to install the Gym environment.

Note that `-e` causes the package in editable mode. This is not necessary if you're not developing this package.

### Usage
TODO