from importlib.resources import files as _importlib_resources_files
from pathlib import Path as _Path

assets_dir = _Path(str(_importlib_resources_files("flygym") / "assets"))


# from . import legacy
# from . import warp
from . import compose
