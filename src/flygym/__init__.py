from importlib.resources import files as _importlib_resources_files
from pathlib import Path as _Path

# assets_dir must be defined before any submodule is imported, because
# flygym/compose/fly.py does `from flygym import assets_dir` at module level.
assets_dir = _Path(str(_importlib_resources_files("flygym") / "assets"))

from . import anatomy
from . import compose
from .simulation import Simulation
from .rendering import Renderer, launch_interactive_viewer, preview_model

__all__ = [
    "assets_dir",
    "anatomy",
    "compose",
    "Simulation",
    "Renderer",
    "launch_interactive_viewer",
    "preview_model",
]
