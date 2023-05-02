Interacting with an Environment
===============================

Overview
--------

Gymnasium (a continuation of the now deprecated OpenAI Gym) is a toolkit for developing and comparing control algorithms. It provides a diverse collection of environments, ranging from classic control problems, Atari games, board games, and robotics simulations. Here, we have implemented NeuroMechFly as a Gym environment for you to interact with. Gym environments are designed to offer a standardized interface for controllers, in particular reinforcement learning agents, to interact with. This standardization makes it easier to develop, benchmark, and compare algorithms.

The overall steps for interacting with a Gym environment are:

#. Defining an environment (the :ref:`specifics` section on each physics simulator will cover more details)
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
-----------------------------

In this section, we explain the format of the actions and observations that are passed to and from the environment.

The **observation** is a dictionary of the following format::

    {
        'joints': np.ndarray,  # NumPy array of shape (3, num_dofs)
                               # the 3 rows are the angle, angular velocity,
                               # and force at each DoF. The order of the
                               # DoFs is the same as ``env.actuated_joints``
        'fly': np.ndarray,  # NumPy array of shape (4, 3)
                            # 0th row: x, y, z position of the fly in arena
                            # 1st row: x, y, z velocity of the fly in arena
                            # 2nd row: orientation of fly around x, y, z axes
                            # 3rd row: rate of change of fly orientation
        'contact_forces': np.ndarray,  # readings of the touch contact sensors,
                                       # one placed for each of the
                                       # ``collision_tracked_geoms``
        'end_effectors': np.ndarray,  # x, y, z positions of the end effectors
                                      # (tarsus-5 segments)
    }

The **action** is a dictionary of the following format::

    {
        'joints': np.ndarray,  # NumPy array of shape (num_dofs,)
                               # the order of the DoFs is the same as
                               # ``env.actuated_joints``
    }

The meaning of action array depends on the controller type: if position control is used (which is the default case), the array will be interpreted as the target joint angles. If velocity or force control is used, the array will be interpreted as the target velocity or the applied force.


.. _specifics:

Physics-Engine-Specific Details
-------------------------------

.. toctree::
   :maxdepth: 2

   mujoco
   isaacgym
   pybullet
