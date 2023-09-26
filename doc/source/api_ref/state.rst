State
=====

The ``State`` is a representation of the state of the animal. It can contain the biomechanical state (ie. pose, or a collection of joint angles), the neural state (ie. the state of the neural network), or both. Its API is specified by the ``BaseState`` class, which can be extended by the user as needed.

This page provides the API reference for the ``BaseState`` and the ``KinematicPose``, which is used to define the initial pose (angles at all DoFs) of the simulated fly.

.. autoclass:: flygym.mujoco.state.BaseState
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: flygym.mujoco.state.KinematicPose
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members: