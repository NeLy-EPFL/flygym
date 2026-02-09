## Simulating embodied sensorimotor control with NeuroMechFly v2

> [!CAUTION]
> FlyGym 2.x.x API is under development.
> Go to [flygym](https://github.com/NeLy-EPFL/flygym) for the stable version of 1.x.x API.

## Installation
```sh
# Clone this repository
git clone git@github.com:NeLy-EPFL/flygym-v2.git  # TODO: Move to stable flygym repo
cd flygym/

# To install withou Warp (CPU only):
# Remove 'dev' if you don't need developer tools, and 'examples' if you don't want to
# run the tutorials
uv sync --extra dev --extra examples

# To install with Warp (GPU-accelerated, requires NVIDIA GPU):
# Remove 'dev' or 'examples' as needed
uv sync --extra dev --extra examples --extra warp  # as above

# If installing with 'dev', add git pre-commit filter that strips notebook outputs
nbstripout --install --attributes .gitattributes
```