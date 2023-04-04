MuJoCo Specifics
================

Tutorial
--------

You might want to start with the following demo tutorials:

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
   :members: __init__, reset, step, render, close