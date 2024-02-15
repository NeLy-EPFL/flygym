Simulation Parameters
=====================

Parameters of the simulation, including physics parameters (e.g. friction, gravity) and visualization parameters (e.g. render mode, render FPS), are defined in the ``Parameters`` class. This class, however, does not include parameters that are specific to the NeuroMechFly model itself (e.g. which body segments are included in collision tracking; where the model is spawned at the beginning of the simulation).

.. autoclass:: flygym.mujoco.Parameters
    :members: