.. FlyGym documentation master file, created by
   sphinx-quickstart on Sat Apr  1 16:03:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Simulating embodied sensorimotor control with NeuroMechFly 2.0
==============================================================


.. toctree::
   :maxdepth: 2
   :hidden:

   neuromechfly
   installation
   tutorials/index
   api_ref/index
   changelog
   contributing

`Preprint <https://www.biorxiv.org/content/10.1101/2023.09.18.556649>`_ |
`GitHub <https://github.com/NeLy-EPFL/flygym>`_

.. figure :: https://github.com/NeLy-EPFL/_media/blob/main/flygym/overview_video.gif?raw=true
   :width: 400
   :alt: Overview video


.. note:: 

   FlyGym is in beta. API changes may occur in future releases. See the `changelog <changelog.html>`_ for details.


FlyGym is the Python library for NeuroMechFly 2.0, a digital twin of the adult fruit fly *Drosophila melanogaster* that can see, smell, walk over challenging terrain, and interact with the environment (see our `NeuroMechFly 2.0 paper <https://www.biorxiv.org/content/10.1101/2023.09.18.556649>`_).

FlyGym consists of the following components:

- **Biomechanical model:** The biomechanical model is based on a micro-CT scan of a real adult female fly (see our `original NeuroMechFly <https://doi.org/10.1038/s41592-022-01466-7>`_ publication). In NeuroMechFly 2.0 we have updated several body segments (in particular in the antennae) to better reflect the biological reality.
- **Vision:** The fly has compound eyes consisting of individual units called ommatidia. These are arranged on a hexagonal lattice. We have simulated visual inputs on the fly's retinae.
- **Olfaction:** The fly has odor receptors in the antennae and maxillary palps. We have simulated olfactory signals experienced by the fly by computing odor/chemical intensities at the locations of the antennae and maxillary palps.
- **Hierarchical control:** The fly's central nervous system (CNS) consists of the brain and the ventral nerve cord (VNC). This hierarchy is analogous to the brain-spinal cord organization of vertebrates. The user can build a two-part model --- one handling brain-level sensory integration and decision making as well as one handling VNC-level motor control --- with a interface between the two consisting of descending (brain-to-VNC) and ascending (VNC-to-brain) neuronal pathways.
- **Leg adhesion:** Insects have evolved specialized adhesive structures at the tips of their legs. These enable locomotion on inclined and overhanging surfaces. We have simulated adhesion in our model. The mechanism by which the fly lifts the legs during locomotion despite adhesive forces is currently not well understood. Therefore, to abstract this, adhesion can be turned on or off during stance or swing of the legs, respectively.
- **Mechanosensory feedback:** The user has access to joint angles, joint torques, and collision/contact forces experienced by the simulated fly.

FlyGym formulates the control of the simulated fly as a `partially observable Markov Decision Process (MDP) <https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process>`_ and implements the `Gym interface <https://gymnasium.farama.org/>`_. This allows the user to use a wide range of reinforcement learning (RL) algorithms to train the fly to perform tasks. The standardized interface also allows the user to easily implement their own premotor computation and/or sensory preprocessing.

.. figure :: https://github.com/NeLy-EPFL/_media/blob/main/flygym/mdp.png?raw=true
   :width: 600
   :alt: MDP

   *The biomechanical model and its interaction with the environment are encapsulated as a MDP task. A user-defined controller interfaces with the task through actions (red) and observations (blue). The user can extend the MDP task by adding preprogrammed processing routines for sensory inputs (purple) and motor outputs (light blue), to modify the action and observation spaces handled by the controller.*

Citation
--------
If you use FlyGym or NeuroMechFly in your research, please cite the following two papers:

.. code-block:: bibtex
   
   % Original NeuroMechFly: Original biomechanical model, kinematic replay, CPG-based
   % neural controller, PyBullet version, DoF analysis, perturbation tests
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
   
   % NeuroMechFly 2.0: This library, MuJoCo version, leg adhesion, rule-based controller,
   % hybrid controller, complex terrain, preprogrammed steps, leg adhesion, vision,
   % olfaction, RL-based navigation, Gym environment, updated biomechanical model
   @article{WangChen2023,
      author = {Sibo Wang-Chen and Victor Alfred Stimpfling and Pembe Gizem \"{O}zdil and Louise Genoud and Femke Hurtak and Pavan Ramdya},
      title = {NeuroMechFly 2.0, a framework for simulating embodied sensorimotor control in adult Drosophila},
      year = {2023},
      doi = {10.1101/2023.09.18.556649},
      URL = {https://www.biorxiv.org/content/10.1101/2023.09.18.556649},
      journal = {bioRxiv}
   }
