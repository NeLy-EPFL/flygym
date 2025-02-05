.. FlyGym documentation master file, created by
   sphinx-quickstart on Sat Apr  1 16:03:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Simulating embodied sensorimotor control with NeuroMechFly v2
=============================================================


.. toctree::
   :maxdepth: 2
   :hidden:

   neuromechfly
   gallery/index
   installation
   tutorials/index
   api_ref/index
   workshop
   changelog
   contributing

`Paper <https://www.nature.com/articles/s41592-024-02497-y.epdf?sharing_token=jK2FbKWL99-O28WNqrpXWNRgN0jAjWel9jnR3ZoTv0MjiFZczOI3_5wYVxbEbClrTuJzjKyEfhm2kIwso489-ypEsSqlyasWAEsBCvR9WU5poT-q2bblI6hCc7Zji6wb_jZjfXl7KWLbd2pgZTmWvk_ADQ6RuzlnHwvQyipMJzg%3D>`_ |
`GitHub <https://github.com/NeLy-EPFL/flygym>`_

.. figure:: https://github.com/NeLy-EPFL/_media/blob/main/flygym/overview_video.gif?raw=true
   :width: 400
   :alt: Overview video


.. note:: 

   API changes may occur in future releases. See the `changelog <changelog.html>`_ for details.


FlyGym is the Python library for NeuroMechFly v2, a digital twin of the adult fruit fly *Drosophila melanogaster* that can see, smell, walk over challenging terrain, and interact with the environment (see our `NeuroMechFly v2 paper <https://www.nature.com/articles/s41592-024-02497-y.epdf?sharing_token=jK2FbKWL99-O28WNqrpXWNRgN0jAjWel9jnR3ZoTv0MjiFZczOI3_5wYVxbEbClrTuJzjKyEfhm2kIwso489-ypEsSqlyasWAEsBCvR9WU5poT-q2bblI6hCc7Zji6wb_jZjfXl7KWLbd2pgZTmWvk_ADQ6RuzlnHwvQyipMJzg%3D>`_).

FlyGym consists of the following components:

- **Biomechanical model:** The biomechanical model is based on a micro-CT scan of a real adult female fly (see our `original NeuroMechFly <https://doi.org/10.1038/s41592-022-01466-7>`_ publication). In NeuroMechFly v2 we have updated several body segments (in particular in the antennae) to better reflect the biological reality.
- **Vision:** The fly has compound eyes consisting of individual units called ommatidia. These are arranged on a hexagonal lattice. We have simulated visual inputs on the fly's retinas and corresponding visual neuron activities using the `FlyVision model <https://turagalab.github.io/flyvis/>`_ (see `Lappalainen et al., Nature, 2024 <https://doi.org/10.1038/s41586-024-07939-3>`_).
- **Olfaction:** The fly has odor receptors in the antennae and maxillary palps. We have simulated olfactory signals experienced by the fly by computing odor/chemical intensities at the locations of the antennae and maxillary palps.
- **Hierarchical control:** The fly's central nervous system (CNS) consists of the brain and the ventral nerve cord (VNC). This hierarchy is analogous to the brain-spinal cord organization of vertebrates. The user can build a two-part model --- one handling brain-level sensory integration and decision making as well as one handling VNC-level motor control --- with a interface between the two consisting of descending (brain-to-VNC) and ascending (VNC-to-brain) neuronal pathways.
- **Leg adhesion:** Insects have evolved specialized adhesive structures at the tips of their legs. These enable locomotion on inclined and overhanging surfaces. We have simulated adhesion in our model. The mechanism by which the fly lifts the legs during locomotion despite adhesive forces is currently not well understood. Therefore, to abstract this, adhesion can be turned on or off during stance or swing of the legs, respectively.
- **Mechanosensory feedback:** The user has access to joint angles, joint torques, and collision/contact forces experienced by the simulated fly.

FlyGym formulates the control of the simulated fly as a `partially observable Markov Decision Process (MDP) <https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process>`_ and implements the `Gym interface <https://gymnasium.farama.org/>`_. This allows the user to use a wide range of reinforcement learning (RL) algorithms to train the fly to perform tasks. The standardized interface also allows the user to easily implement their own premotor computation and/or sensory preprocessing.

.. figure:: https://github.com/NeLy-EPFL/_media/blob/main/flygym/mdp.png?raw=true
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
   
   % NeuroMechFly v2: This library, MuJoCo version, leg adhesion, rule-based controller,
   % hybrid controller, complex terrain, preprogrammed steps, leg adhesion, vision,
   % olfaction, RL-based navigation, Gym environment, updated biomechanical model
   @article{WangChen2024,
      title = {NeuroMechFly v2: simulating embodied sensorimotor control in adult Drosophila},
      volume = {21},
      ISSN = {1548-7105},
      url = {http://dx.doi.org/10.1038/s41592-024-02497-y},
      DOI = {10.1038/s41592-024-02497-y},
      number = {12},
      journal = {Nature Methods},
      publisher = {Springer Science and Business Media LLC},
      author = {Wang-Chen,  Sibo and Stimpfling,  Victor Alfred and Lam,  Thomas Ka Chung and \"{O}zdil,  Pembe Gizem and Genoud,  Louise and Hurtak,  Femke and Ramdya,  Pavan},
      year = {2024},
      month = nov,
      pages = {2353--2362}
   }

.. note:: 
   **Privacy policy:** This site uses Google Analytics to collect data about your interactions with our website. This includes information such as your IP address, browsing behavior, and device type. We use this data to improve our website and understand user preferences. Google Analytics uses Cookies, which are small text files stored on your device. See `How Google uses information from sites or apps that use our services <https://policies.google.com/technologies/partner-sites>`_. To opt-out, you can use a `browser extension <https://tools.google.com/dlpage/gaoptout>`_ to deactivate Google Analytics.
