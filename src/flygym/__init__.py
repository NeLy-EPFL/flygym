from importlib.resources import files as _importlib_resources_files
from pathlib import Path as _Path

# assets_dir must be defined before any submodule is imported, because
# flygym/compose/fly.py does `from flygym import assets_dir` at module level.
assets_dir = _Path(str(_importlib_resources_files("flygym") / "assets"))

from . import anatomy  # noqa: E402
from . import compose  # noqa: E402
from . import flybody  # noqa: E402
from .simulation import Simulation  # noqa: E402
from .rendering import Renderer, launch_interactive_viewer, preview_model  # noqa: E402

__all__ = [
    "assets_dir",
    "anatomy",
    "compose",
    "flybody",
    "Simulation",
    "Renderer",
    "launch_interactive_viewer",
    "preview_model",
]
