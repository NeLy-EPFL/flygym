MuJoCo Environment
==================

Interactiing with a Gym Environment
-----------------------------------

Overview
~~~~~~~~

Gymnasium (a continuation of the now deprecated OpenAI Gym) is a toolkit for developing and comparing control algorithms. It provides a diverse collection of environments, ranging from classic control problems, Atari games, board games, and robotics simulations. Here, we have implemented NeuroMechFly as a Gym environment for you to interact with. Gym environments are designed to offer a standardized interface for controllers, in particular RL agents, to interact with. This standardization makes it easier to develop, benchmark, and compare algorithms.

The overall steps for interacting with a Gym environment are:

#. Defining an environment (the :ref:`mujoco` will cover more details)
#. Reset the environment and get the initial observation
#. Interact with the environment with a loop:

   * Based on the last observation, the controller decides which actions to take
   * Step the simulation, applying the selected actions. The simulation will return you the new observation (and optionally some additional information)
   * Optional: render the simulation graphically
   * Break if certain conditions are met (eg. task is accomplished or failed), otherwise continue

#. Close the environment and analyze the results

This process can be shown in the following code snippet::
    
        env = MyEnvironement(...)
        obs = env.reset()
        
        for step in range(1000):    # let's simulate 1000 steps max
            action = ...    # your controller decides what to do based on obs
            obs, info = env.step(action)
            env.render()
            if is_done(obs):    # is the task already accomplished or failed?
                break
        
        env.close()

Note that the action can be selected by any means defined by the user. It could be a function consisting only of if statements, or an artificial neural network predicting the best actions based on prior observation.


Action and Observation Spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, we explain the format of the actions and observations that are passed to and from the environment.

The **observation** is a dictionary of the following format::

    {
        'joints': np.ndarray,  # NumPy array of shape (3, num_dofs)
                               # the 3 rows are the velocity, position,
                               # and force at each DoF. The order of the
                               # DoFs is the same as ``env.actuated_joints``
        'fly': np.ndarray,  # NumPy array of shape (4, 3)
                            # 0th row: x, y, z position of the fly in arena
                            # 1st row: x, y, z velocity of the fly in arena
                            # 2nd row: orientation of fly around x, y, z axes
                            # 3rd row: rate of change of fly orientation
    }

The **action** is a dictionary of the following format::

    {
        'joints': np.ndarray,  # NumPy array of shape (num_dofs,)
                               # the order of the DoFs is the same as
                               # ``env.actuated_joints``
    }

The meaning of action array depends on the controller type: if position control is used (which is the default case), the array will be interpreted as the target joint angles. If velocity or force control is used, the array will be interpreted as the target velocity or the applied force.

.. _mujoco:

MuJoCo Specifics
----------------

Example
~~~~~~~

The following code snippet executes an environment where all leg joints of the fly repeat a sinusoidal motion. The output will be saved as a video and the observation will be appended to a list. ::

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo

    # First, we initialize simulation
    run_time = 0.1
    out_dir = Path('mujoco_basic_untethered_sinewave')
    nmf = NeuroMechFlyMuJoCo(render_mode='saved', output_dir=out_dir)

    # Define the frequency, phase, and amplitude of the sinusoidal waves
    freq = 500
    phase = 2 * np.pi * np.random.rand(len(nmf.actuators))
    amp = 0.9

    obs_list = []    # keep track of the observed states
    while nmf.curr_time <= run_time:    # main loop
        joint_pos = amp * np.sin(freq * nmf.curr_time + phase)
        action = {'joints': joint_pos}
        obs, info = nmf.step(action)
        nmf.render()
        obs_list.append(obs)
    nmf.close()


API Reference
~~~~~~~~~~~~~

We provide a comprehensive API reference to the MuJoCo environment below.

.. autoclass:: flygym.envs.nmf_mujoco.NeuroMechFlyMuJoCo
   :members: __init__, reset, step, render, close