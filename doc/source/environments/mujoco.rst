MuJoCo Specifics
================


.. warning::
   
   Information on these pages are being updated. Please refer to `the codebase <https://github.com/NeLy-EPFL/nmf2-paper>`_ accompnaying our `NeuroMechFly 2.0 paper <https://www.biorxiv.org/content/10.1101/2023.09.18.556649>`_ for the code used to generate results in the paper.
   --- 21 September 2023


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


MuJoCoParameters
----------------

.. autoclass:: flygym.envs.nmf_mujoco.MuJoCoParameters
   :members: 


NeuroMechFlyMuJoCo
------------------

.. autoclass:: flygym.envs.nmf_mujoco.NeuroMechFlyMuJoCo
   :members: __init__, reset, step, render, save_video, close, get_observation, get_reward, is_terminated, is_truncated, get_info