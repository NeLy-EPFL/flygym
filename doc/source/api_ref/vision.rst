Vision
======

This page documents the implementation of the visual input received by the simulated fly. Note that in the typical use case, the user should **not** have to access most of the functions described here. Instead, the visual inputs are given as a part of the *observation* returned by ``NeuroMechFlyMuJoCo`` each time step. Nonetheless, the full API reference is provided here for greater transparency.

Retina simulation
-----------------

.. autoclass:: flygym.mujoco.vision.Retina
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Note that sometimes it is helpful to hide certain objects in the arena when rendering the fly's vision. For example, markers for odor sources that are meant for visualization only should not be seen by the fly. To accomplish this, we have provided two hook methods in ``BaseArena`` that allows the user to modify the arena as needed before and after we simulate the fly's vision (for example, changing the alpha value of the odor source markers here):

.. automethod:: flygym.mujoco.arena.BaseArena.pre_visual_render_hook

.. automethod:: flygym.mujoco.arena.BaseArena.post_visual_render_hook


Visualization tool
------------------

We have also provided a utility function to generate a video of the visual input during a simulation:

.. autofunction:: flygym.mujoco.vision.visualize_visual_input
