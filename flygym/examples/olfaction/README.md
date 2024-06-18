## FlyGym examples: Olfaction

> [!NOTE]
> `flygym/examples` aims to provide a packaged and importable implementation of things explained in the [online tutorials](https://neuromechfly.org/tutorials/index.html). The tutorials offer a much more detailed walk-through of the concepts and it is preferable to start from there.

This subpackage includes the following files:
- Simple odor taxis
- Integrating NeuroMechFly with fluid simulation and controlling the fly to follow a complex plume

### Simple odor taxis
- `simple_odor_taxis.py`: This file provides a simple example of a fly seeking an attractive odor source while avoiding aversive ones based on asymmetry in odor intensity.

### Tracking complex plumes
- `simulate_plume_dataset.py`: This script simulates a complex plume by solving (a simplified version of) the incompressible Navier–Stokes equations.
- `plume_tracking_arena.py`: This module implements an `OdorPlumeArena`, which allows the fly to sense the intensity of the odor plume as it moves in space.
- `plume_tracking_task.py`: This module implements `PlumeNavigationTask`, a wrapper around `HybridTurningController`. This class enables rendering with odor intensity overlayed on top of the arena floor, along with some other utilities such as detecting if the fly has reached the limit of the arena.
- `plume_tracking_controller.py`: This module implements the plume following algorithm proposed in [Demir et al., _eLife_, 2020](https://doi.org/10.7554/eLife.57524).
- `track_plume_closed_loop.py`: This script uses the controller above to perform the `PlumeNavigationTask` in closed loop. It is computationally heavy — for reference, we ran it on a 36-core node on the university cluster for a few hours.
- `visualize_plume_tracking_results.py`: This light script visualizes the results of the closed-loop plume following simulations. Please note that this script requires a manually entered string, `chosen_trial`, which is the trial for which a final video should be generated. The user must enter one of the trials printed by an earlier part of this script, and the script re-run. This is because the simulation results is only deterministic within the same MuJoCo version and chip architecture, and the set of successful trials might differ.
- `save_plume_tracking_snapshot.py`: This script runs a very short simulation to visualize and save an example snapshot of the arena. It can be a helpful tool for debugging and was used to generate illustrations in the NeuroMechFly v2 paper.