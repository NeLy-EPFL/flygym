MDP Task Specifications
=======================

As discussed in the `"Interacting with NeuroMechFly" tutorial <https://neuromechfly.org/tutorials/gym_basics_and_kinematic_replay.html>`_, we formulate the control problem as a "task," or "environment," which receives an *action* from the controller and returns (i) an *observation* of the state of the fly and (ii) a reward (optional). The task also return an *info* dictionary, which can be used to provide additional information about the task, a *terminated* flag to indicate whether the task has due to certain conditions being met, and a *truncated* flag to indicate whether the task has been terminated early due to technical issues (e.g. physics errors, timeout, etc.). On this page, we will specify the content of the *action*, the *observation*, and the conditions for *termination* and *truncation* to return True.

Default ``Simulation``
----------------------

**Action:** The action space is a `Dict space <https://gymnasium.farama.org/api/spaces/composite/#dict>`_ with the following keys:

* "joints": The control signal for the actuated DoFs (e.g. if ``Fly.control == "position"``, then this is the target joint angle). This is a NumPy array of shape (num_actuated_joints,). The order of the DoFs is the same as ``Fly.actuated_joints``.
* "adhesion" (if ``Fly.enable_adhesion`` is True): The on/off signal of leg adhesion as a NumPy array of shape (6,), one for each leg. The order of the legs is: LF, LM, LH, RF, RM, RH (L/R = left/right, F/M/H = front/middle/hind).

**Observation:** The observation space is a Dict space with the following keys:

* "joints": The joint states as a NumPy array of shape (3, num_actuated_joints). The three rows are the angle, angular velocity, and force at each DoF. The order of the DoFs is the same as ``Fly.actuated_joints``
* "fly": The fly state as a NumPy array of shape (4, 3). 0th row: x, y, z position of the fly in arena. 1st row: x, y, z velocity of the fly in arena. 2nd row: orientation of fly around x, y, z axes. 3rd row: rate of change of fly orientation.
* "contact_forces": Readings of the touch contact sensors, one placed for each of the body segments specified in ``Fly.contact_sensor_placements``. This is a NumPy array of shape (num_contact_sensor_placements, 3).
* "end_effectors": The positions of the end effectors (most distal tarsus link) of the legs as a NumPy array of shape (6, 3). The order of the legs is: LF, LM, LH, RF, RM, RH (L/R = left/right, F/M/H = front/middle/hind).
* "fly_orientation": NumPy array of shape (3,). This is the vector (x, y, z) pointing toward the direction that the fly is facing.
* "vision" (if ``Fly.enable_vision`` is True): The light intensities sensed by the ommatidia on the compound eyes. This is a NumPy array of shape (2, num_ommatidia_per_eye, 2), where the zeroth dimension is the side (left, right in that order); the second dimension specifies the ommatidium, and the last column is for the spectral channel (yellow-type, pale-type in that order). Each ommatidium only has one channel with nonzero reading. The intensities are given on a [0, 1] scale.
* "odor_intensity" (if ``Fly.enable_olfaction`` is True): The odor intensities sensed by the odor sensors (by default 2 antennae and 2 maxillary palps). This is a NumPy array of shape (odor_space_dimension, num_sensors).

**Info:** The info dictionary contains the following:

* "vision_updated" (if ``Fly.enable_vision`` is True): A boolean indicating whether the vision has been updated in the current step. This is useful because the visual input is usually updated at a much lower frequency than the physics simulation.
* "flip" (if ``Fly.detect_flip`` is True): A boolean indicating whether the fly has flipped upside down.
* "flip_counter" (if ``Fly.detect_flip`` is True): The number of simulation steps during which all legs of the fly have been off the ground (detected using a threshold of ground contact forces). Useful for debugging.
* "contact_forces" (if ``Fly.detect_flip`` is True): The contact forces sensed by the legs. Useful for debugging.
* "neck_actuation" (if ``Fly.head_stabilization_model`` is specified): The neck actuation applied.

**Reward, termination, and truncation:** By default, the task always returns False for the terminated and truncated flags and 0 for the reward. The user is expected to modify this behavior by extending the ``Simulation`` or ``Fly`` classes as needed.


Examples under ``flygym/examples``
----------------------------------

Hybrid turning controller (``HybridTurningController``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Action**: The ``flygym.example.locomotion.HybridTurningController`` class expects a single NumPy array of shape (2,) as its action. The values are the descending walking drive on the left and right sides of the fly. See the `tutorial on the hybrid turning controller <https://neuromechfly.org/tutorials/turning.html>`_ for more details.

**Observation, reward, termination, and truncation:** The ``flygym.example.locomotion.HybridTurningController`` class returns the same observation, reward, "terminated" flag, and "truncated" flag as the default ``Simulation`` class.

**Info:** In addition to what is provided in the default ``Simulation``, the ``flygym.example.locomotion.HybridTurningController`` class includes the following in the "info" dictionary:

* "joints", "adhesion": The hybrid turning controller computes the appropriate joint angles and adhesion signals based on the descending inputs, CPG states, and mechanosensory feedback. These values are the computed low-level motor commands applied to the underlying base ``Simulation``.
* "net_corrections": The net correction amounts applied to the legs as a NumPy array of shape (6,). Refer to the `tutorial on the hybrid turning controller <https://neuromechfly.org/tutorials/hybrid_controller.html>`__ for more details. The order of legs is: LF, LM, LH, RF, RM, RH (L/R = left/right, F/M/H = front/middle/hind).



Simple object following (``VisualTaxis``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Action:** The ``flygym.example.vision.VisualTaxis`` class expects the same action as ``HybridTurningController``.

**Observation:** The ``flygym.example.vision.VisualTaxis`` class returns an array of shape (2, 3) as the observation. The two rows of the array specify the left vs. right eyes (in this order). The three columns are the azimuth (left-right) and elevation (top-down) positions of the object in the visual field, and the size of the object in the visual field. All values are normalized, either by the width/height or by the size of the visual field, to the range [0, 1].

**Reward, termination, truncation, and info:** The ``flygym.example.vision.VisualTaxis`` class always returns 0 for the reward, False for the "terminated" and "truncated" flags, and an empty dictionary for the "info" dictionary.



Path integration task (``PathIntegrationController``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Action, reward, termination, truncation, and info:** The ``flygym.examples.path_integration.PathIntegrationController`` class returns the same action, observation, reward, "terminated" flag, "truncated" flag, and "info" dictionary as ``HybridTurningController``.

**Observation:** In addition to what is returned by ``HybridTurningController``, ``flygym.examples.path_integration.PathIntegrationController`` also provides the following in the observation dictionary:

* "stride_diff_unmasked": The relative shift of the tips of the legs from one simulation step to another. The shift is computed in the reference from of the fly and presented as a NumPy array of shape (6, 3). The order of the legs (0th axis of the array) is: LF, LM, LH, RF, RM, RH (L/R = left/right, F/M/H = front/middle/hind). The 1st axis of the array is the x, y, z components of the shift. The shift is computed by comparing the positions of the legs in the current step with the positions in the previous step. The shift is not masked by the leg's contact with the ground.



Plume tracking task (``PlumeNavigationTask``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Action, observation, reward, termination, info:** The ``flygym.examples.olfaction.PlumeNavigationTask`` class expects the same action and returns the observation, reward, "terminated" flag, and "info" dictionary as ``HybridTurningController``.

**Truncation:** The ``flygym.examples.olfaction.PlumeNavigationTask`` class returns True for the "truncated" flag if and only if the fly has left the area on the arena where the plume is simulated.



NeuroMechFly with connectome-constrained vision network (``RealisticVisionController``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Action, reward, termination, and truncation:** The ``flygym.examples.realistic_vision.RealisticVisionController`` class expects the same action and returns the same reward, "terminated" flag, and "truncated" flags as ``HybridTurningController``.

**Observation:** In addition to what is returned by the ``HybridTurningController``, the ``flygym.examples.realistic_vision.RealisticVisionController`` class also provides the following in the observation dictionary:

* "nn_activities_arr": The activities of the visual system neurons, represented as a NumPy array of shape (2, num_cells_per_eye). The 0th dimension corresponds to the eyes in the order (left, right).

**Info:** In addition to what is returned by the ``HybridTurningController``, the ``flygym.examples.realistic_vision.RealisticVisionController`` class also provides the following in the "info" dictionary:

* "nn_activities": Activities of the visual system neurons as a ``flyvision.LayerActivity`` object. This is similar to ``obs["nn_activities_arr"]`` but in the form of a ``flyvision.LayerActivity`` object rather than a plain array.