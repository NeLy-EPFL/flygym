Arena
=====

Arena are the physical environments that the simulated fly is placed in. They can have rugged surfaces, visual features, odor features, or any combination thereof. The user can implement their own arenas by inheriting from the ``flygym.mujoco.arena.BaseArena`` class.

This page provides the API reference for the ``BaseArena`` abstract class as well as the preprogrammed arenas.

Base arena
----------
.. autoclass:: flygym.mujoco.arena.BaseArena
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: flygym.mujoco.arena.FlatTerrain

Preprogrammed complex terrain arenas
------------------------------------

.. autoclass:: flygym.mujoco.arena.GappedTerrain

.. autoclass:: flygym.mujoco.arena.BlocksTerrain

.. autoclass:: flygym.mujoco.arena.MixedTerrain

Preprogrammed arenas with sensory features
------------------------------------------

.. autoclass:: flygym.mujoco.arena.OdorArena