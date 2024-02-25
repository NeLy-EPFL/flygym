Olfaction
=========

Olfaction is the sense of smell. Olfactory experience is simulated by calculating odor intensities at the locations of the antennae and maxillary palps. More precisely, this is accomplished by adding position sensors to the relevant body segments and calculating the distance between these sensors and the odor sources. Intensities are then emulated through a diffusion relationship.

To enable this calculation, the ``BaseArena`` has the following methods. The user does not have to specifically implement them if an odor is not enabled.

.. automethod:: flygym.arena.BaseArena.get_olfaction
    :noindex:

.. autoattribute:: flygym.arena.BaseArena.odor_dimensions
    :noindex:

A useful implementation to refer to is the built-in ``OdorArena``:

.. autoclass:: flygym.arena.OdorArena
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :noindex:
