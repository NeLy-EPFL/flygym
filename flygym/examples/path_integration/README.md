## FlyGym examples: Path integration

> [!NOTE]
> `flygym/examples` aims to provide a packaged and importable implementation of things explained in the [online tutorials](https://neuromechfly.org/tutorials/index.html). The tutorials offer a much more detailed walk-through of the concepts and it is preferable to start from there.

This subpackage includes the following files:
- `arena.py`: This module defines the arena that our path integration task is performed in.
- `controller.py`: This module defines the NeuroMechFly controller that performs random exploration (i.e., bouts of straight walking separated by stochastic turning).
- `exploration.py`: With the controller mentioned above, this script runs random exploration trials using differnt gait patterns. It is somewhat computationally heavy — for reference, we ran it on a 16-core workstation for about 1.5 hours.
- `model.py`: This module defines the path integration model.
- `build_models.py`: This script trains and tests the path integrations models using the exploration trials. We ran it on a 16-core machine for about an hour.
- `util.py`: Utilities for feature extraction.
- `viz.py`: Visualization functions.