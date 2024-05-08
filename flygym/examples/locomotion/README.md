## FlyGym examples: Locomotion

> [!NOTE]
> `flygym/examples` aims to provide a packaged and importable implementation of things explained in the [online tutorials](https://neuromechfly.org/tutorials/index.html). The tutorials offer a much more detailed walk-through and it is preferable to start from there.

This subpackage includes the following files:
- `cpg_controller`: This file implements a `CPGNetwork` class and demonstrate how a network of Central Pattern Generators (CPGs) can coordinate walking. 
- `rule_based_controller`: This file implements a `RuleBasedSteppingController` and demonstrates how distributed leg coordination rules can coordinate walking.
- `hybrid_controller`: This script demonstrates how one can integrate CPGs with sensory feedback
- `turning_controller`: This file refactors the hybrid controller into a class implementing the Gymnasium API and adds a steering input that enables turning.