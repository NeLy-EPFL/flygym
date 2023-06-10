Arena
=====

General Information
-------------------

The FlyGym package comes with various arena for the simulated fly to behave in:

- 'flat': a simple flat floor
- 'gapped': a floor with gaps perpendicular to the fly's walking direction
- 'blocks': terrain formed by blocks at random heights
- 'ball': a spherical treadmill on which the fly is tethered (still under development)

.. figure :: ../_static/terrain.jpg
   :width: 800
   :alt: Implemented terrain types

   Implemented terrain types

These terrain types are implemented in ``flygym.terrain``. Different terrain types in different physics simulators extend the ``flygym.terrain.base.BaseTerrain`` abstract class. The general API for interacting with these implementations are therefore documented in the ``BaseTerrain`` abstract class below:

.. autoclass :: flygym.arena.BaseArena
   :members: __init__, get_spawn_position, spawn_entity


Physics-Engine-Specific Details
-------------------------------

.. toctree::
   :maxdepth: 2

   mujoco
