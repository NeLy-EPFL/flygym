Change Log
==========

.. note:: 

   FlyGym uses `EffVer <https://jacobtomlinson.dev/effver/>`_ as its versioning policy. The version number will communicate how much effort we expect a user to spend to adopt the new version. 

.. note:: 
  
   See the `Release <https://github.com/NeLy-EPFL/flygym/releases>`_ page for previous releases and metadata such as release dates.

* **1.2.1:**

  * **Other changes:**
  
    * Improve vision readout computing time by avoiding redundant memory copies
    * Bump NumPy version to 2.* and end official support for macOS 13

* **1.2.0:**

  * **API-breaking changes:**

    * Enhance camera logic. See the `Camera API reference <api_ref/camera.html>`_ for details.
    * Use the `FlyVis package <https://github.com/TuragaLab/flyvis>`_ as `published on PyPI <https://pypi.org/project/flyvis/>`_. Users should now use ``flyvis`` instead of ``flyvision`` as module name.

  * **Other changes:**
  
    * Add NeuroMechFly game for outreach (used at EPFL Scientastic! Days, etc.).
    * Transition from ``setup.py`` to ``pyproject.toml``, specifically using `Poetry <https://python-poetry.org/>`_ as the build backend. Users can now install FlyGym with the exact versions of dependencies used by the developers by running ``poetry install`` from the root directory. This will create a virtual environment with the correct dependencies as specified in the included ``poetry.lock`` file, which is version-tracked as a part of the FlyGym Github repository.
    * Add `VS Code devcontainer <https://code.visualstudio.com/docs/devcontainers/containers>`_ support.
    * Remove support for Python 3.9.
   
* **1.1.0:**

  * Added cardinal direction sensing (vectors describing +x, +y, +z of the fly) to the observation space.
  * Removed legacy spawn orientation preprocessing: Previously, pi/2 was subtracted from the user-specified spawn orientation on the x-y plane. This was to make the behavior consistent with a legacy version of NeuroMechFly. This behavior is no longer desired; from this version onwards, the spawn orientation is used as is.
  * Strictly fixed the required MuJoCo version to 3.2.3, and dm_control version to 1.0.23. This is to prevent API-breaking changes in future versions of these libraries from affecting FlyGym. FlyGym maintainers will periodically check for compatibility with newer versions of these libraries.
  * Changed flip detection method: Previously, flips are reported when all legs reliably lose contact with the ground. Now, we simply check if the z component of the "up" cardinal vector is negative. Additionally, the ``detect_flip`` parameter of ``Fly`` is now deprecated; flips are always detect and reported.
  * Allowed different sets of DoFs to be monitored vs. actuated. Previously, the two sets are always the same.
  * From this version onwards, we will use `EffVer <https://jacobtomlinson.dev/effver/>`_ as the versioning policy. The version number will communicate how much effort we expect a user will need to spend to adopt the new version. While we previously tried to adhere to the stricter `SemVer <https://semver.org/>`_, we found that it was not effective because many core dependencies of FlyGym (e.g., MuJoCo, NumPy, and Python itself) do not use SemVer.

* **1.0.1:** Fixed minor bugs related to the set of DoFs in the predefined poses, and to rendering at extremely high frequencies. Fixed outdated class names and links in the docs. In addition, contact sensor placements used by the hybrid turning controller are now added to the ``preprogrammed`` module.

* **1.0.0:** In spring 2024, NeuroMechFly was used, for the second time, in a course titled "`Controlling behavior in animals and robots <https://edu.epfl.ch/coursebook/en/controlling-behavior-in-animals-and-robots-BIOENG-456>`_" at EPFL. At the same time, we revised the NeuroMechFly v2 manuscript. In these processes, we significantly improved the FlyGym package, added new functionalities, and incorporated changes as we received feedback from the students. These enhancements are released as FlyGym version 1.0.0. This release is not backward compatible; please refer to the `tutorials <https://neuromechfly.org/tutorials/index.html>`_ and `API references <https://neuromechfly.org/api_ref/index.html>`_ for more information. The main changes are:
  
  * Major API changes:
  
    * The ``NeuroMechFly`` class is split into ``Fly``, a class that represents the fly, and ``Simulation``, a class that represents the simulation, which can potentially contain multiple flies.
    * The ``Parameters`` class is deprecated. Parameters related to the fly (such as joint parameters, actuated DoFs, etc.) should be set directly on the ``Fly`` object. Parameters related to the simulation (such as the time step, the render cameras, etc.) should be set directly on the ``Simulation`` object.
    * A new ``Camera`` class is introduced. A simulation can contain multiple cameras.

  * New `examples <https://github.com/NeLy-EPFL/flygym/tree/main/flygym/examples>`_:

    * Path integration based on ascending mechanosensory feedback.
    * Head stabilization based on ascending mechanosensory feedback.
    * Navigating a complex plume, simulated separately in a fluid mechanics simulator.
    * Following another fly using a realistic, connectome-constrained neural network that processes visual inputs.

* **0.2.5:** Modify model file to make it compatible with MuJoCo 3.1.1. Disable Python 3.7 support accordingly.
* **0.2.4:** Set MuJoCo version to 2.3.7. Documentation updates.
* **0.2.3:** Various bug fixes. Improved placement of the spherical treadmill in the tethered environment.
* **0.2.2:** Changed default joint kp and adhesion forces to those used in the controller comparison task. Various minor bug fixes. Documentation updates.
* **0.2.1:** Simplified class names: ``NeuroMechFlyMuJoCo`` → ``NeuroMechFly``, ``MuJoCoParameters`` → ``Parameters``. Minor documentation updates.
* **0.2.0:** The current base version — major API change from 0.1.x.
* **0.1.x:** Versions used during the initial development of NeuroMechFly v2.
* **Unversioned:** Version used for the Spring 2023 offering of BIOENG-456 Controlling Behavior in Animals and Robots course at EPFL.
