MuJoCo Specifics
================

Tutorials
---------

You might want to start from the following tutorials:

.. list-table::
   :widths: 60 30
   :header-rows: 1

   * - Tutorial
     - Colab Link
   * - Basic control
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/NeLy-EPFL/flygym/blob/main/notebooks/mujoco_sinewave.ipynb

   * - Replaying recorded behavior
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/NeLy-EPFL/flygym/blob/main/notebooks/mujoco_replay.ipynb
   
   * - Complex terrain
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/NeLy-EPFL/flygym/blob/main/notebooks/mujoco_terrain.ipynb


API Reference
-------------

We provide a comprehensive API reference to the MuJoCo environment below.

.. autoclass:: flygym.envs.nmf_mujoco.NeuroMechFlyMuJoCo
   :members: __init__, reset, step, get_observation, render, save_video, close


.. _mujoco_config:

Physics, terrain, and rendering configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A number of **physics** parameters can be set via the ``physics_config`` argument in the environment constructor. They are listed below along with their default values::

    {
      'joint_stiffness': 2500,
      'friction': (1, 0.005, 0.0001),
      'gravity': (0, 0, -9.81e5),
    }

A number of **rendering** parameters can also be set via the ``render_config`` argument in the environment constructor. Depending on the ``render_mode``, the supported options and their default values are listed beloow::

    {
      'saved': {'window_size': (640, 480), 'playspeed': 1.0, 'fps': 60},
      'headless': {}  # headless = no rendering. No options allowed
    }


Finally, a number of **terrain** parameters can be set via the ``terrain_config`` argument in the environment constructor. The exact options supported depend on the ``terrain`` type. They are listed below along with their default values::

    {
      'flat': {
        'size': (50_000, 50_000),
        'friction': (1, 0.005, 0.0001),
        'fly_pos': (0, 0, 300),
        'fly_orient': (0, 1, 0, 0.1)
      },
      'gapped': {
        'x_range': (-10_000, 10_000),
        'y_range': (-10_000, 10_000),
        'friction': (1, 0.005, 0.0001),
        'gap_width': 200,
        'block_width': 1000,
        'gap_depth': 2000,
        'fly_pos': (0, 0, 600),
        'fly_orient': (0, 1, 0, 0.1)
      },
      'blocks': {
        'x_range': (-10_000, 10_000),
        'y_range': (-10_000, 10_000),
        'friction': (1, 0.005, 0.0001),
        'block_size': 1000,
        'height_range': (300, 300),
        'rand_seed': 0,
        'fly_pos': (0, 0, 600),
        'fly_orient': (0, 1, 0, 0.1)
      },
      'ball': {
        'radius': ...,
        'fly_pos': (0, 0, ...),
        'fly_orient': (0, 1, 0, ...),
      },
    }