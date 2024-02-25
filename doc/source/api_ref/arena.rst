Arena
=====

An arena is a physical environment in which the simulated fly is placed. Arenas can have rugged surfaces, visual features, odor features, or any combination thereof. The user can implement their own arenas by inheriting from the ``flygym.arena.BaseArena`` class.

This page provides the API reference for the ``BaseArena`` abstract class as well as the preprogrammed arenas.

Base arena
----------
.. autoclass:: flygym.arena.BaseArena
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: flygym.arena.FlatTerrain

Preprogrammed complex terrain arenas
------------------------------------

.. autoclass:: flygym.arena.GappedTerrain

.. autoclass:: flygym.arena.BlocksTerrain

.. autoclass:: flygym.arena.MixedTerrain

Preprogrammed arenas with sensory features
------------------------------------------

.. autoclass:: flygym.arena.OdorArena