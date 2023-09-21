.. FlyGym documentation master file, created by
   sphinx-quickstart on Sat Apr  1 16:03:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FlyGym: Simulating embodied sensorimotor control with NeuroMechFly 2.0
======================================================================


.. toctree::
   :maxdepth: 2
   :hidden:

   neuromechfly
   installation
   environments/index
   arena/index
   state
   changelog
   contributing

`Preprint <https://www.biorxiv.org/content/10.1101/2023.09.18.556649>`_ |
`GitHub <https://github.com/NeLy-EPFL/flygym>`_

.. figure :: https://github.com/NeLy-EPFL/_media/blob/main/flygym/overview_video.gif?raw=true
   :width: 400
   :alt: Overview video

.. note::
   
   FlyGym is still in beta; the API is subject to change.
   We will add further examples and documentation in the coming weeks. Stay tuned!
   --- 21 September 2023

FlyGym is the Python library for NeuroMechFly 2.0, a digital twin of the adult fruit fly *Drosophila melanogaster* that can see, smell, walk over challenging terrain, and interact with the environment (see our `NeuroMechFly 2.0 paper <https://www.biorxiv.org/content/10.1101/2023.09.18.556649>`_).

FlyGym consists of the following components:

- **Biomechanical model:** The biomechanical model is based on a micro-CT scan of a real adult female fly (see our `original NeuroMechFly <https://doi.org/10.1038/s41592-022-01466-7>`_ publication). We have adjusted several body segments (in particular in the antennae) to better reflect the biological reality.
- **Vision:** The fly has compound eyes consisting of individual units called ommatidia arranged on a hexagonal lattice. We have simulated the visual inputs on the fly's retinas.
- **Olfaction:** The fly has odor receptors in the antennae and the maxillary palps. We have simulated the odor inputs experienced by the fly by computing the odor/chemical intensity at these locations.
- **Hierarchical control:** The fly's Central Nervous System consists of the brain and the Ventral Nerve Cord (VNC), a hierarchy analogous to our brain-spinal cord organization. The user can build a two-part model --- one handling brain-level sensory integration and decision making and one handling VNC-level motor control --- with a interface between the two consisting of descending (brain-to-VNC) and ascending (VNC-to-brain) representations.
- **Leg adhesion:** Insects have evolved specialized adhesive structures at the tips of the legs that enable locomotion vertical walls and overhanging ceilings. We have simulated these structures in our model. The mechanism by which the fly lifts the legs during locomotion despite adhesive forces is not well understood; to abstract this, adhesion can be turned on/off during leg stance/swing.
- **Mechanosensory feedback:** The user has access to the joint angles, forces, and contact forces experienced by the fly.

FlyGym formulates the control of the simulated fly as a `partially observable Markov Decision Process (MDP) <https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process>`_ and implements the `Gym interface <https://gymnasium.farama.org/>`_. This allows the user to use a wide range of reinforcement learning algorithms to train the fly to perform tasks. The standardized interface also allows the user to easily implement their own premotor computation and/or sensory preprocessing processes.

.. figure :: https://github.com/NeLy-EPFL/_media/blob/main/flygym/mdp.png?raw=true
   :width: 600
   :alt: MDP

   *The biomechanical model and its interaction with the environment are encapsulated as a MDP task. A user-defined controller interfaces with the task through actions (red) and observations (blue). The user can extend the MDP task by adding preprogrammed processing routines for sensory inputs (purple) and motor outputs (light blue), to modify the action and observation spaces handled by the controller.*

Citation
--------
If you use FlyGym or NeuroMechFly in your research, please cite the following two papers:

.. code-block:: bibtex
   
   @article{WangChen2023,
      author = {Sibo Wang-Chen and Victor Alfred Stimpfling and Pembe Gizem \"{O}zdil and Louise Genoud and Femke Hurtak and Pavan Ramdya},
      title = {NeuroMechFly 2.0, a framework for simulating embodied sensorimotor control in adult Drosophila},
      year = {2023},
      doi = {10.1101/2023.09.18.556649},
      URL = {https://www.biorxiv.org/content/early/2023/09/18/2023.09.18.556649},
      journal = {bioRxiv}
   }

   @article{LobatoRios2022,
      doi = {10.1038/s41592-022-01466-7},
      url = {https://doi.org/10.1038/s41592-022-01466-7},
      year = {2022},
      month = may,
      volume = {19},
      number = {5},
      pages = {620--627},
      author = {Victor Lobato-Rios and Shravan Tata Ramalingasetty and Pembe Gizem \"{O}zdil and Jonathan Arreguit and Auke Jan Ijspeert and Pavan Ramdya},
      title = {{NeuroMechFly}, a neuromechanical model of adult {Drosophila} melanogaster},
      journal = {Nature Methods}
   }
