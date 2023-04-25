The NeuroMechFly Model
======================

NeuroMechFly is a morphologically accurate model of the adult fruit fly *Drosophila melanogaster*. It is based on a micro-CT scan of the animal. See the `NeuroMechFly paper`_ for more details.

.. figure :: _static/neuromechfly.png
   :width: 700
   :alt: NeuroMechFly

   Figure from the NeuroMechFly paper (Lobato-Rios et al, 2022): a, An adult female fly encased in resin for X-ray microtomography. b, Cross-section of the resulting X-ray scan. Cuticle, muscles, nervous tissues and internal organs are visible. c, Thresholded data separating the foreground (white) from the background (black). d, 3D polygon mesh of the exoskeleton and wings. e, Articulated body parts after separation from one another. f, Body parts after reassembly into a natural resting pose and overlaid with a rigged skeleton in dark red. g, Fly model after the addition of texture.


.. _coords:

Coordinate System
-----------------

The model uses the following coordinate system: the ground plane is the x-y plane. z points upward. By default, the fly is mounted facing the positive x direction. In the initial configuration, rotation along the y axis is the *pitch*. Rotation along the x axis is the *roll*. Rotation along the z axis is the *yaw*.

.. figure :: _static/coords.png
   :width: 300
   :alt: Aircraft principal axes

   Source: https://commons.wikimedia.org/wiki/File:Yaw_Axis_Corrected.svg


.. _body:

Body Parts
----------

The biomechanical model consists of a set of rigid body parts. The body parts relevant to locomotion is shown below:

.. figure :: _static/fly_anatomy.jpg
   :width: 600
   :alt: Fly anatomy

   Source: Chyb, S., & Gompel, N. (2013). *Atlas of Drosophila Morphology*. doi:10.1016/c2009-0-61936-x

A "joint" links two body parts (see :ref:`joints`). Note that in physics simulation, "joint" refers to a single degree of freedom (DoF). Therefore, if a (biological) joint has multiple DoFs (such as the thorax-coxa joint), the (biological) joint is implemented as multiple joint links. As a result, the same (biological) body segment is simulated with multiple body segments to allow for the "virtual" links between different DoFs on the same joint. Any unlabelled link is a *pitch* DoF. Any link with a suffix of ``_roll`` is a *roll* DoF. Any link with a suffix of ``_yaw`` is a *yaw* DoF (see :ref:`coords`). 

The following is the complete list of body parts defined in the model. In general, ``L`` and ``R`` indicate the left and right side. ``F``, ``M``, ``H`` indicate the fore-, mid-, and hindlegs. ``An`` indicate the ``n``-th segement of the abdomen. For example, ``RHFemur`` means the femur of the right hindleg; ``LFTarsus1`` means the first tarsus link of the left foreleg, and ``A1A2`` means the fused first and second segments of the abdomen. ::

    ['Thorax', 'A1A2', 'A3', 'A4', 'A5', 'A6', 'Head_roll', 'Head_yaw', 
     'Head', 'LEye', 'LPedicel_roll', 'LPedicel_yaw', 'LPedicel', 
     'LFuniculus_roll', 'LFuniculus_yaw', 'LFuniculus', 'LArista_roll', 
     'LArista_yaw', 'LArista', 'REye', 'Rostrum', 'Haustellum', 
     'RPedicel_roll', 'RPedicel_yaw', 'RPedicel', 'RFuniculus_roll', 
     'RFuniculus_yaw', 'RFuniculus', 'RArista_roll', 'RArista_yaw', 
     'RArista', 'LFCoxa_roll', 'LFCoxa_yaw', 'LFCoxa', 'LFFemur', 
     'LFFemur_roll', 'LFTibia', 'LFTarsus1', 'LFTarsus2', 'LFTarsus3', 
     'LFTarsus4', 'LFTarsus5', 'LHaltere_roll', 'LHaltere_yaw', 
     'LHaltere', 'LHCoxa_roll', 'LHCoxa_yaw', 'LHCoxa', 'LHFemur', 
     'LHFemur_roll', 'LHTibia', 'LHTarsus1', 'LHTarsus2', 'LHTarsus3', 
     'LHTarsus4', 'LHTarsus5', 'LMCoxa_roll', 'LMCoxa_yaw', 'LMCoxa', 
     'LMFemur', 'LMFemur_roll', 'LMTibia', 'LMTarsus1', 'LMTarsus2', 
     'LMTarsus3', 'LMTarsus4', 'LMTarsus5', 'LWing_roll', 'LWing_yaw', 
     'LWing', 'RFCoxa_roll', 'RFCoxa_yaw', 'RFCoxa', 'RFFemur', 
     'RFFemur_roll', 'RFTibia', 'RFTarsus1', 'RFTarsus2', 'RFTarsus3', 
     'RFTarsus4', 'RFTarsus5', 'RHaltere_roll', 'RHaltere_yaw', 
     'RHaltere', 'RHCoxa_roll', 'RHCoxa_yaw', 'RHCoxa', 'RHFemur', 
     'RHFemur_roll', 'RHTibia', 'RHTarsus1', 'RHTarsus2', 'RHTarsus3', 
     'RHTarsus4', 'RHTarsus5', 'RMCoxa_roll', 'RMCoxa_yaw', 'RMCoxa', 
     'RMFemur', 'RMFemur_roll', 'RMTibia', 'RMTarsus1', 'RMTarsus2', 
     'RMTarsus3', 'RMTarsus4', 'RMTarsus5', 'RWing_roll', 'RWing_yaw', 
     'RWing']


.. _joints:

Joint Links
-----------

The following is a complete list of body parts. See the :ref:`body` section for an explanation of the DoFs. In general, the joint name only lists the child link: for example, the thorax-coxa roll DoF is listed as ``joint_XXCoxa_roll``. ::

    ['joint_Head_roll', 'joint_Head_yaw', 'joint_Head', 
    'joint_LPedicel_roll', 'joint_LPedicel_yaw', 'joint_LPedicel', 
    'joint_LFuniculus_roll', 'joint_LFuniculus_yaw', 
    'joint_LFuniculus', 'joint_LArista_roll', 'joint_LArista_yaw', 
    'joint_LArista', 'joint_RPedicel_roll', 'joint_RPedicel_yaw', 
    'joint_RPedicel', 'joint_RFuniculus_roll', 'joint_RFuniculus_yaw', 
    'joint_RFuniculus', 'joint_RArista_roll', 'joint_RArista_yaw', 
    'joint_RArista', 'joint_LFCoxa_roll', 'joint_LFCoxa_yaw', 
    'joint_LFCoxa', 'joint_LFFemur', 'joint_LFFemur_roll', 
    'joint_LFTibia', 'joint_LFTarsus1', 'joint_LFTarsus2', 
    'joint_LFTarsus3', 'joint_LFTarsus4', 'joint_LFTarsus5', 
    'joint_LHCoxa_roll', 'joint_LHCoxa_yaw', 'joint_LHCoxa', 
    'joint_LHFemur', 'joint_LHFemur_roll', 'joint_LHTibia', 
    'joint_LHTarsus1', 'joint_LHTarsus2', 'joint_LHTarsus3', 
    'joint_LHTarsus4', 'joint_LHTarsus5', 'joint_LMCoxa_roll', 
    'joint_LMCoxa_yaw', 'joint_LMCoxa', 'joint_LMFemur', 
    'joint_LMFemur_roll', 'joint_LMTibia', 'joint_LMTarsus1', 
    'joint_LMTarsus2', 'joint_LMTarsus3', 'joint_LMTarsus4', 
    'joint_LMTarsus5', 'joint_RFCoxa_roll', 'joint_RFCoxa_yaw', 
    'joint_RFCoxa', 'joint_RFFemur', 'joint_RFFemur_roll', 
    'joint_RFTibia', 'joint_RFTarsus1', 'joint_RFTarsus2', 
    'joint_RFTarsus3', 'joint_RFTarsus4', 'joint_RFTarsus5', 
    'joint_RHCoxa_roll', 'joint_RHCoxa_yaw', 'joint_RHCoxa', 
    'joint_RHFemur', 'joint_RHFemur_roll', 'joint_RHTibia', 
    'joint_RHTarsus1', 'joint_RHTarsus2', 'joint_RHTarsus3', 
    'joint_RHTarsus4', 'joint_RHTarsus5', 'joint_RMCoxa_roll', 
    'joint_RMCoxa_yaw', 'joint_RMCoxa', 'joint_RMFemur', 
    'joint_RMFemur_roll', 'joint_RMTibia', 'joint_RMTarsus1', 
    'joint_RMTarsus2', 'joint_RMTarsus3', 'joint_RMTarsus4', 
    'joint_RMTarsus5']

.. figure :: _static/dofs.png
   :width: 400
   :alt: NeuroMechFly's leg DoFs

   Figure from the NeuroMechFly paper (Lobato-Rios et al, 2022): Zero pose of NeuroMechFly from (A) front and (B) side views. Each leg is composed of 11 hinge joints. Joints with more than one DoF were modeled as a union of multiple hinge joints. The left foreleg observed from the (C) side and (D) front views. Rotational axes of joints are shown in light green. The global coordinate system’s x, y, and z axes are red, green, and blue, respectively.

For the modeling of locomotion, the leg DoFs are the most critical. In *Drosophila*, there are 7 *actuated* DoFs per leg: thorax-coxa pitch (``joint_XXCoxa``), thorax-coxa roll (``joint_XXCoxa_roll``), thorax-coxa yaw (``joint_XXCoxa_yaw``), coxa-femur pitch (``joint_XXFemur``), coxa-femur roll (``joint_XXFemur_roll``), femur-tibia pitch (``joint_XXTibia``), and tibia-tarsus pitch (``joint_XXTarsus1``). The links between tarsus segments can also move passively but are not actively actuated. To get started, one might consider using a subset of all leg DoFs: for example, the NeuroMechFly paper used 3 DoFs per leg for its optimization work: thorax-coxa pitch for the forelegs, the thorax-coxa roll for the mid- and hindlegs, coxa-femur pitch for all legs, and femur-tibia pitch for all legs.

.. note::

    FlyGym provides hardcoded shorthands for these useful lists of links::

        >>> import flygym.util.config as config

        # all actuatable leg DoFs:
        >>> config.all_leg_dofs
        ['joint_LFCoxa', 'joint_LFCoxa_roll', 'joint_LFCoxa_yaw', 'joint_LFFemur', 'joint_LFFemur_roll', 'joint_LFTibia', 'joint_LFTarsus1', 'joint_LMCoxa', 'joint_LMCoxa_roll', 'joint_LMCoxa_yaw', 'joint_LMFemur', 'joint_LMFemur_roll', 'joint_LMTibia', 'joint_LMTarsus1', 'joint_LHCoxa', 'joint_LHCoxa_roll', 'joint_LHCoxa_yaw', 'joint_LHFemur', 'joint_LHFemur_roll', 'joint_LHTibia', 'joint_LHTarsus1', 'joint_RFCoxa', 'joint_RFCoxa_roll', 'joint_RFCoxa_yaw', 'joint_RFFemur', 'joint_RFFemur_roll', 'joint_RFTibia', 'joint_RFTarsus1', 'joint_RMCoxa', 'joint_RMCoxa_roll', 'joint_RMCoxa_yaw', 'joint_RMFemur', 'joint_RMFemur_roll', 'joint_RMTibia', 'joint_RMTarsus1', 'joint_RHCoxa', 'joint_RHCoxa_roll', 'joint_RHCoxa_yaw', 'joint_RHFemur', 'joint_RHFemur_roll', 'joint_RHTibia', 'joint_RHTarsus1']

        # 3 DoFs per leg:
        >>> config.leg_dofs_3_per_leg
        ['joint_LFCoxa', 'joint_LFFemur', 'joint_LFTibia', 'joint_LMCoxa_roll', 'joint_LMFemur', 'joint_LMTibia', 'joint_LHCoxa_roll', 'joint_LHFemur', 'joint_LHTibia', 'joint_RFCoxa', 'joint_RFFemur', 'joint_RFTibia', 'joint_RMCoxa_roll', 'joint_RMFemur', 'joint_RMTibia', 'joint_RHCoxa_roll', 'joint_RHFemur', 'joint_RHTibia']


References
----------
- Lobato-Rios, V., Ramalingasetty, S. T., Özdil, P. G., Arreguit, J., Ijspeert, A. J., & Ramdya, P. (2022). NeuroMechFly, a neuromechanical model of adult *Drosophila melanogaster*. *Nature Methods*, 19(5), 620-627. https://doi.org/10.1038/s41592-022-01466-7



.. _NeuroMechFly paper: https://doi.org/10.1038/s41592-022-01466-7