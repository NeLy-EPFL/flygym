## FlyGym examples: Locomotion

> [!NOTE]
> `flygym/examples` aims to provide a packaged and importable implementation of things explained in the [online tutorials](https://neuromechfly.org/tutorials/index.html). The tutorials offer a much more detailed walk-through of the concepts and it is preferable to start from there.

This subpackage includes the following files:
- `cpg_controller`: This file implements a `CPGNetwork` class and demonstrate how a network of Central Pattern Generators (CPGs) can coordinate walking. 
- `rule_based_controller`: This file implements a `RuleBasedController` and demonstrates how distributed leg coordination rules can coordinate walking.
- `hybrid_controller`: This script demonstrates how one can integrate CPGs with sensory feedback.
- `turning_controller`: This file refactors the hybrid controller into a `HybridTurningController` class. This class implements the Gymnasium API and, in addition to the hybrid controller above, receives a steering input that enables turning.
- `turning_fly`: This module implements a `HybridTurningFly` class, which has the same essential logic as `HybridTurningController` but functions at the level of the fly instead of the simulation. Therefore, it is useful to use this module when simulating multiple flies, each using a hybrid walking controller that be modulated to execute turns.
- `colored_fly.py`: This module implements a `ColoredFly` class, which is a subclass of `Fly` that fixes the issue of segments not being colored correctly in the `Fly` class. Required for the `controller_comparison.py` script.
- `controller_comparison.py`: This script compares the performance of the CPG, rule-based, and hybrid controller in different terrains.