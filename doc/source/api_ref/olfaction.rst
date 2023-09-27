Olfaction
=========

Olfaction is simulated by calculating odor intensities at the locations of the antennae and the maxillary palps. This is accomplished by adding position sensors to the relevant body segments and calculating distance of these sensors from the odor sources. The intensities are then emulated through a difussion relationship.

To enable this calculation, the ``BaseArena`` has the following methods. The user does not have to specifically implement them if odor is not enabled.

.. automethod:: flygym.mujoco.arena.BaseArena.get_olfaction
    :noindex:

.. autoattribute:: flygym.mujoco.arena.BaseArena.odor_dimensions
    :noindex:

A useful implementation to refer to is the built-in ``OdorArena``:

.. autoclass:: flygym.mujoco.arena.OdorArena
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :noindex:
