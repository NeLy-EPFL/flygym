# Simulating embodied sensorimotor control with NeuroMechFly v2

!!! tip "March 2026 Update"

    We released a new FlyGym v2.x.x API in February 2026, with significantly improved performance:
    
    * ~10x speed-up for CPU-based simulations (~2x real-time throughput)
    * ~300x speed-up for GPU-based simulation enabled by [MJWarp](https://mujoco.readthedocs.io/en/latest/mjwarp/) (~60x real-time throughput)

    Additional improvements include:

    * Improved scene composition workflow
    * Interactive viewer
    * Simplified dependency stack

    Prefer the old API? See [information here](migration).

![overview](https://raw.githubusercontent.com/NeLy-EPFL/_media/refs/heads/main/flygym/overview_video.gif)

FlyGym is the Python library for NeuroMechFly, a digital twin of the adult fruit fly Drosophila melanogaster that can see, smell, walk over challenging terrain, and interact with the environment.

For more information, see our [NeuroMechFly v2 paper](https://www.nature.com/articles/s41592-024-02497-y.epdf).
