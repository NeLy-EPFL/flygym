## FlyGym examples: Vision

> [!NOTE]
> `flygym/examples` aims to provide a packaged and importable implementation of things explained in the [online tutorials](https://neuromechfly.org/tutorials/index.html). The tutorials offer a much more detailed walk-through of the concepts and it is preferable to start from there.

This directory contains to examples related to vision:
- Simple vision-guided object following
- Integrating NeuroMechFly with a connectome-constrained visual system model and performing fly following based on neural activities

### Simple visual taxis
- `arena.py`: This module contains the implementation of a `MovingObjArena`, in which a black sphere moves in a S-shaped path.
- `simple_visual_taxis.py`: In this file, we implement and test a controller that allows the fly to follow the black sphere based on visual inputs.

### Connectome-constrained visual system model
- `arena.py`: Also implemented in this module is a `MovingFlyArena`, where instead of a black sphere, a second fly moves in a preset path.
- `vision_network.py`: This module extends the [FlyVision](https://github.com/TuragaLab/flyvis) library that accompanies [Lappalainen et al. (2023)](https://www.biorxiv.org/content/10.1101/2023.03.11.532232), which constructed and optimized a connectome-based model of the _Drosophila_ visual system. In our module, we added the functionalities that allow the model to receive individual frames of visual input and simulate the network in closed loop.
- `record_baseline_response.py`: Using the model by Lappalainen et al., this script records the activities of visual neurons when then fly walks in a featureless arena in open loop.
- `realistic_vision.py`: This module implements a `NMFRealisticVision` class, which extends the `HybridTurningController`, that interfaces with the vison network simulation to provide visual neuron activities as a part of the observation in addition to the raw visual input.
- `follow_fly_closed_loop.py`: This script uses the `NMFRealisticVision` controller to perform closed-loop fly following based on the baseline activity established in `record_baseline_response.py`.
- `concatenate_fly_following_video.sh`: This shell script helps concatinate the videos of all trials into one for easier inspection. It depends on [FFmpeg](https://ffmpeg.org/).
- `save_fly_following_snapshot.py`: This script runs a very short simulation to visualize and save an example snapshot of the simulation arena as well as the visual neuron activities. It can be a helpful tool for debugging and was used to generate illustrations in the NeuroMechFly v2 paper.
- `viz.py`: This module contains various plotting functions used in other scripts.