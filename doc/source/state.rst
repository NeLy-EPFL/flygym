State
=====

The `State` class is a representation of the state of the animal. It can contain the biomechanical state (ie. pose, or a collection of joint angles), the neural state (ie. the state of the neural network), or both. Its API is specified by the `BaseState` class:

.. autoclass:: flygym.state.BaseState
   :members: __iter__, __getitem__

The provided state implementations include:

.. autoclass:: flygym.state.KinematicPose
   :members: from_yaml