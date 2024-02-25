State
=====

The ``State`` represents the state of the animal. This can include the biomechanical state (i.e. pose, or a collection of joint angles), the neural state (i.e. the state of a neural network), or both. Its API is specified by the ``BaseState`` class, which can be extended by the user as needed.

This page provides the API reference for the ``BaseState`` and the ``KinematicPose``, which is used to define the initial pose (angles at all DoFs) of the simulated fly.

.. autoclass:: flygym.state.BaseState
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: flygym.state.KinematicPose
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members: