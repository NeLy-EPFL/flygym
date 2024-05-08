## FlyGym examples

> [!NOTE]
> `flygym/examples` aims to provide a packaged and importable implementation of things explained in the [online tutorials](https://neuromechfly.org/tutorials/index.html). The tutorials offer a much more detailed walk-through and it is preferable to start from there.

In this subpackage, you will find some example use cases of NeuroMechFly and FlyGym.

- Locomotion:
    - Centralized walking controller based on Central Pattern Generators (CPGs)
    - Decentralized walking controller based on stepping coordination rules
    - Hybrid controller integrating CPGs with sensory feedback
    - Hybrid controller receiving a steering inputs to enable turning
- Path integration
    - Using ascending proprioceptive and tactile information to estimate the animal's spatial position
- Head stabilization
    - Using ascending proprioceptive and tactile information to actuate the neck joint and cancel out self motion in visual inputs
- Vision
    - Simple vision-guided object following
    - Integrating NeuroMechFly with a connectome-constrained visual system model and performing fly following based on neural activities
- Olfaction
    - Simple odor taxis
    - Integrating NeuroMechFly with fluid simulation and controlling the fly to follow a complex plume

Within the directory for each of these examples, you will find a README file that explains what each module or script does. 