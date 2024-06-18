## FlyGym examples: Locomotion

> [!NOTE]
> `flygym/examples` aims to provide a packaged and importable implementation of things explained in the [online tutorials](https://neuromechfly.org/tutorials/index.html). The tutorials offer a much more detailed walk-through of the concepts and it is preferable to start from there.

This subpackage includes the following files:
- `cpg_controller.py`: This file implements a `CPGNetwork` class and demonstrate how a network of Central Pattern Generators (CPGs) can coordinate walking. 
- `rule_based_controller.py`: This file implements a `RuleBasedController` and demonstrates how distributed leg coordination rules can coordinate walking.
- `hybrid_controller.py`: This script demonstrates how one can integrate CPGs with sensory feedback.
- `turning_controller.py`: This file refactors the hybrid controller into a `HybridTurningController` class. This class implements the Gymnasium API and, in addition to the hybrid controller above, receives a steering input that enables turning.
- `turning_fly.py`: This module implements a `HybridTurningFly` class, which has the same essential logic as `HybridTurningController` but functions at the level of the fly instead of the simulation. Therefore, it is useful to use this module when simulating multiple flies, each using a hybrid walking controller that be modulated to execute turns.
- `colorable_fly.py`: This module implements `ColorableFly`, a wrapper around the ``Fly`` class that facilitates the recoloring of specific segments. This is useful for, as an example, recoloring parts of the leg depending on the activation of specific correction rules and is required for `controller_comparison.py`.
- `controller_comparison.py`: This script runs a benchmark to compare the performance of the CPG, rule-based, and hybrid controller in different terrains. It runs 20 trials for each terrain-controller combination in parallel, resulting in 20x4x3=240 trials. On a machine running Intel Core i9-11900K (16 threads), it takes 30-60 minutes to execute this script. This script requires a functional FFMPEG executable to generate the controller comparison video.