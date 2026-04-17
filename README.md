> [!IMPORTANT]
> We introduced a new FlyGym 2.x.x API in March 2026, with a complete code rewrite and redesigned interface. This version delivers significantly improved performance:
> 
> - **~10x speed-up** for CPU-based simulations (~2x real-time throughput)
> - **~300x speed-up** for GPU-based simulation via Warp/MJWarp (~60x real-time throughput)
>
> Additional improvements include:
>
> - Improved scene composition workflow
> - Interactive viewer
> - Simplified dependency stack
>
> This version is not backward compatible, and not all features from FlyGym 1.x.x are available. Feature requests can be submitted via Issues on the Github repository. See more information about the changes [here](https://neuromechfly.org/migration/).
>
> Prefer the old version? FlyGym 1.x.x has been migrated to [`flygym-gymnasium`](https://github.com/NeLy-EPFL/flygym-gymnasium). Its documentation has been migrated to [gymnasium.neuromechfly.org](https://gymnasium.neuromechfly.org/).

## Simulating embodied sensorimotor control with NeuroMechFly v2

![](https://github.com/NeLy-EPFL/_media/blob/main/flygym/banner_large.jpg?raw=true)

[**Documentation**](https://neuromechfly.org/) | [**Paper**](https://www.nature.com/articles/s41592-024-02497-y.epdf?sharing_token=jK2FbKWL99-O28WNqrpXWNRgN0jAjWel9jnR3ZoTv0MjiFZczOI3_5wYVxbEbClrTuJzjKyEfhm2kIwso489-ypEsSqlyasWAEsBCvR9WU5poT-q2bblI6hCc7Zji6wb_jZjfXl7KWLbd2pgZTmWvk_ADQ6RuzlnHwvQyipMJzg%3D) | [**Discussion Board**](https://github.com/NeLy-EPFL/flygym/discussions)

![overview_video](https://github.com/NeLy-EPFL/_media/blob/main/flygym/overview_video.gif?raw=true)

This repository contains the source code for FlyGym, the Python library for NeuroMechFly v2, a digital twin of the adult fruit fly *Drosophila* melanogaster that can see, smell, walk over challenging terrain, and interact with the environment (see our [NeuroMechFly v2 paper](https://www.nature.com/articles/s41592-024-02497-y.epdf?sharing_token=jK2FbKWL99-O28WNqrpXWNRgN0jAjWel9jnR3ZoTv0MjiFZczOI3_5wYVxbEbClrTuJzjKyEfhm2kIwso489-ypEsSqlyasWAEsBCvR9WU5poT-q2bblI6hCc7Zji6wb_jZjfXl7KWLbd2pgZTmWvk_ADQ6RuzlnHwvQyipMJzg%3D)).

NeuroMechFly consists of the following components:
- **Biomechanical model:** The biomechanical model is based on a micro-CT scan of a real adult female fly (see our original NeuroMechFly publication). We have adjusted several body segments (in particular in the antennae) to better reflect the biological reality.
- **Vision:** The fly has compound eyes consisting of individual units called ommatidia arranged on a hexagonal lattice. We have simulated the visual inputs on the fly’s retinas.
- **Olfaction:** The fly has odor receptors in the antennae and the maxillary palps. We have simulated the odor inputs experienced by the fly by computing the odor/chemical intensity at these locations.
- **Hierarchical control:** The fly’s Central Nervous System consists of the brain and the Ventral Nerve Cord (VNC), a hierarchy analogous to our brain-spinal cord organization. The user can build a two-part model — one handling brain-level sensory integration and decision making and one handling VNC-level motor control — with an interface between the two consisting of descending (brain-to-VNC) and ascending (VNC-to-brain) representations.
- **Leg adhesion:** Insects have evolved specialized adhesive structures at the tips of the legs that enable locomotion vertical walls and overhanging ceilings. We have simulated these structures in our model. The mechanism by which the fly lifts the legs during locomotion despite adhesive forces is not well understood; to abstract this, adhesion can be turned on/off during leg stance/swing.
- **Mechanosensory feedback:** The user has access to joint angles, actuator forces, contact forces, and user-defined anatomical joint-site positions.

This package is developed at the [Neuroengineering Laboratory](https://www.epfl.ch/labs/ramdya-lab/), EPFL.

### Getting Started

For installation, see [the documentation page](https://neuromechfly.org/installation).

To get started, follow [tutorials here](https://neuromechfly.org/tutorials).