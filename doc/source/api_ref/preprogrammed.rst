Preprogrammed
=============

We have pre programmed a number of handy constants for the user:


DoFs
----

.. code:: python

   # All actively actuatable DoFs in the legs
   >>> flygym.preprogrammed.all_leg_dofs
   ['joint_LFCoxa', 'joint_LFCoxa_roll', 'joint_LFCoxa_yaw', 'joint_LFFemur', 'joint_LFFemur_roll', 'joint_LFTibia', 'joint_LFTarsus1', 'joint_LMCoxa', 'joint_LMCoxa_roll', 'joint_LMCoxa_yaw', 'joint_LMFemur', 'joint_LMFemur_roll', 'joint_LMTibia', 'joint_LMTarsus1', 'joint_LHCoxa', 'joint_LHCoxa_roll', 'joint_LHCoxa_yaw', 'joint_LHFemur', 'joint_LHFemur_roll', 'joint_LHTibia', 'joint_LHTarsus1', 'joint_RFCoxa', 'joint_RFCoxa_roll', 'joint_RFCoxa_yaw', 'joint_RFFemur', 'joint_RFFemur_roll', 'joint_RFTibia', 'joint_RFTarsus1', 'joint_RMCoxa', 'joint_RMCoxa_roll', 'joint_RMCoxa_yaw', 'joint_RMFemur', 'joint_RMFemur_roll', 'joint_RMTibia', 'joint_RMTarsus1', 'joint_RHCoxa', 'joint_RHCoxa_roll', 'joint_RHCoxa_yaw', 'joint_RHFemur', 'joint_RHFemur_roll', 'joint_RHTibia', 'joint_RHTarsus1']
   
   # 3 DoFs per leg used in the CPG optimization task performed in Lobato-Rios et al., 2022.
   # These are the most variable DoFs during tethered locomotion
   >>> flygym.preprogrammed.leg_dofs_3_per_leg
   ['joint_LFCoxa', 'joint_LFFemur', 'joint_LFTibia', 'joint_LMCoxa_roll', 'joint_LMFemur', 'joint_LMTibia', 'joint_LHCoxa_roll', 'joint_LHFemur', 'joint_LHTibia', 'joint_RFCoxa', 'joint_RFFemur', 'joint_RFTibia', 'joint_RMCoxa_roll', 'joint_RMFemur', 'joint_RMTibia', 'joint_RHCoxa_roll', 'joint_RHFemur', 'joint_RHTibia']


Body segments
-------------

.. code:: python

   # All tarsal segments, useful for defining leg-ground contacts 
   >>> flygym.preprogrammed.all_tarsi_links
   ['LFTarsus1', 'LFTarsus2', 'LFTarsus3', 'LFTarsus4', 'LFTarsus5', 'LMTarsus1', 'LMTarsus2', 'LMTarsus3', 'LMTarsus4', 'LMTarsus5', 'LHTarsus1', 'LHTarsus2', 'LHTarsus3', 'LHTarsus4', 'LHTarsus5', 'RFTarsus1', 'RFTarsus2', 'RFTarsus3', 'RFTarsus4', 'RFTarsus5', 'RMTarsus1', 'RMTarsus2', 'RMTarsus3', 'RMTarsus4', 'RMTarsus5', 'RHTarsus1', 'RHTarsus2', 'RHTarsus3', 'RHTarsus4', 'RHTarsus5']

.. autofunction:: flygym.preprogrammed.get_collision_geometries

Pose
----

.. autofunction:: flygym.preprogrammed.get_preprogrammed_pose