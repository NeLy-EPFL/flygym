# FlyGym v1 vs. v2 API

Until March 2026, FlyGym followed the interface defined by [Gymnasium](https://gymnasium.farama.org/), a widely used standard API for reinforcement learning environments (and the namesake of Fly*Gym*). Gymnasium provides a clean and well-established abstraction, making it easy to integrate with existing RL libraries.

However, strict compliance with the Gymnasium interface also introduces computational overhead and architectural constraints. For FlyGym 2.x.x, we therefore decided to move away from Gymnasium compliance. This redesign allows for a more direct and flexible interaction with the simulation (e.g. lazy-loading of simulated states instead of exposing all pre-configured states as "observation"). This design decision was one of the reasons why FlyGym 2.x.x achieves a ~10× speed-up on CPU-based simulations compared to FlyGym 1.x.x.

Importantly, this change does not alter the underlying modeling assumptions: the framework remains fully compatible with the principles of the [Partially Observable Markov Decision Process (POMDP)](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process). The restructuring is purely at the interface level, not at the level of the mathematical formulation.

For users who prefer the Gymnasium-style interface, it is straightforward to implement a lightweight wrapper around the new API. In most cases, such a wrapper can be written in ~10 minutes. We intentionally avoid maintaining an additional official compatibility layer, as doing so would introduce unnecessary backward-compatibility constraints for future development.

At the same time, we recognize that some users may prefer the original 1.x.x API due to its intuitiveness and the extensive tutorials built around it. We have therefore migrated the Gymnasium-compliant version of FlyGym to a separate repository: [`flygym-gymnasium`](https://github.com/NeLy-EPFL/flygym-gymnasium). We expect to occasionally make minor improvements to this API, but naturally the new 2.0.0 API will be our focus. You can install it via:

```sh
pip install flygym-gymnasium

# or, to include optional dependencies:
pip install "flygym-gymnasium[examples,dev]"
```

or by cloning the repository directly. The documentation for this version is available at [gymnasium.neuromechfly.org](https://gymnasium.neuromechfly.org/).