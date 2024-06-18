from .simulation import Simulation, SingleFlySimulation
from .fly import Fly
from .camera import Camera, NeckCamera
from .util import get_data_path, load_config

from os import environ

is_rendering_skipped = (
    "SKIP_RENDERING" in environ and environ["SKIP_RENDERING"] == "true"
)
