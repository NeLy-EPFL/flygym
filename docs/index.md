# Simulating embodied sensorimotor control with NeuroMechFly v2

!!! tip "February 2026 Update"

    We released a new FlyGym v2.x.x API in February 2026, including:
    
    * Up to ~1000x faster simulation using GPU acceleration
    * Improved scene composition workflow
    * Interactive viewer
    * Simplified dependency stack

FlyGym is the Python library for NeuroMechFly, a digital twin of the adult fruit fly Drosophila melanogaster that can see, smell, walk over challenging terrain, and interact with the environment.

For more information, see our [NeuroMechFly v2 paper](https://www.nature.com/articles/s41592-024-02497-y.epdf).


## Choose your own flavor

FlyGym now comes in three flavors:
    
* `flygym`: for new users or users who just want something that works
* `flygym.warp`: for advanced users who want blazing fast, GPU-accelerated simulations who don't mind writing some GPU kernel code
* `flygym.guided`: for users and learners who prefer a user-friendly and guided experience who don't mind a bit of performance loss. This is the version that accompanied our [NeuroMechFly v2 paper](https://www.nature.com/articles/s41592-024-02497-y.epdf). A dozen tutorials are available with this flavor.

Note that we do not plan to backport the newer `flygym` v2.x.x API to all demos and tutorials. However, you are welcome to submit feature requests via [GitHub Issues](https://github.com/NeLy-EPFL/flygym/issues), and reasonable requests will likely be accommodated.
