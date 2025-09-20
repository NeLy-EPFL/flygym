NeuroMechFly Live and Outreach
==============================

Along with the FlyGym package meant for scientific research, we have also released a real-time version, `NeuroMechFly Live <https://github.com/NeLy-EPFL/neuromechfly-live>`_. Unlike FlyGym, NeuroMechFly Live is designed for education and outreach purposes. It uses simplified physics simulation and runs in real-time on a typical PC.


The NeuroMechFly Video Game
---------------------------

Using NeuroMechFly Live, we have developed a video game demonstrating how animals control their behaviors at different levels of abstraction. You can download this game via the link above (use a joystick for the best experience).

Level 1: High-level control using Central Pattern Generators (CPG)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When we walk, we can simply "decide" to walk forward or make a turn. This decision is made at a high level in our brain, and we do not need to think about the details of how our legs should move.

A prominent theory in neuroscience suggests that animals use neural circuits called *Central Pattern Generators (CPGs)* to generate rhythmic patterns of movement such as walking, running, or swimming. CPGs are oscillators that produce rhythmic outputs without receiving any rhythmic input—like motors that run continuously once you power them on. Multiple CPGs can be coupled together to produce coordinated rhythmic movements, and by modulating this network of CPGs, animals can adapt their movements to different speeds and directions.

In this level, you can control the fly to go forward, backward, left, or right using a joystick or four buttons on the keyboard. The CPG circuits take care of low-level motor coordination, and the fly is very easy to control.

.. raw:: html

   <video src="https://raw.githubusercontent.com/NeLy-EPFL/_media/main/flygym/outreach/cpg_small.mp4" controls autoplay muted playinline></video>
   <br><br>



Level 2: Medium-level control using a fixed gait pattern
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Experienced hikers know that walking on rough terrain requires more attention and effort than walking on a flat surface. Often, the hiker needs to pay attention to how their feet are placed in an alternating pattern to ensure stability. If the hiker uses trekking poles, they can further strategize how the poles and legs can work together to maintain balance, much like how horses use trotting or galloping gaits depending on the scenario.

As insects have six legs, their gaits are different from those of quadrupeds. A commonly used gait for insects is the "tripod gait," where three legs move together while the other three legs provide support. Each group of three legs consists of the front and hind legs on one side and the middle leg on the opposite side. This way, the three legs in stance form a stable tripod, allowing the insect to maintain balance while walking.

In this level, you can control the fly to move using the tripod gait. You can use four buttons on the joystick or keyboard to make each group of three legs move forward or backward. You will find that the fly is more challenging to control than in Level 1, but you can still manage it with some practice. Identifying neural circuits controlling these gaits is an active area of research in neuroscience. A significant portion of these circuits are thought to be located in the spinal cord of vertebrates and the ventral nerve cord of insects.

.. raw:: html

   <video src="https://raw.githubusercontent.com/NeLy-EPFL/_media/main/flygym/outreach/tripod_gait_small.mp4" controls autoplay muted playinline></video>
   <br><br>


Level 3: Low-level control of individual legs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When climbers ascend a steep cliff, they need to carefully place each foot and hand to ensure a secure grip. The control of movements happens at a very low level, meaning that all the details of limb placement are consciously managed.

Most humans have about 240 muscles in their limbs, and flies have about 84. Coordinating all these muscles to achieve smooth and purposeful movements is a daunting task. Much of the control is handled by the spinal cord in vertebrates and the ventral nerve cord in insects. They transform the reaching and grasping intentions from the brain into precise muscle activations.

If we wanted to control the contraction of each individual muscle, we'd run out of keys on our keyboard very quickly! Therefore, in this level, we simplify the task by allowing you to control each individual leg instead of muscle. You can use the joystick or keyboard to move one leg at a time. This level is very challenging, and it may take a long time to master the control. However, if you can manage it, you will have a deep appreciation of the complexity of low-level motor control in animals.

.. raw:: html

   <video src="https://raw.githubusercontent.com/NeLy-EPFL/_media/main/flygym/outreach/single_leg_small.mp4" controls autoplay muted playinline></video>


Outreach Events
---------------
We have used the NeuroMechFly Video Game as a tool to engage with the public and educate them about the neuroscience and biomechanics of behavior control. Here are some of the science outreach events:

.. figure:: https://github.com/NeLy-EPFL/_media/blob/main/flygym/outreach/epfl_scientastic_2024_00.jpg?raw=true
   :width: 800
   :alt: Outreach event

.. figure:: https://github.com/NeLy-EPFL/_media/blob/main/flygym/outreach/epfl_scientastic_2024_01.jpg?raw=true
   :width: 800
   :alt: Outreach event