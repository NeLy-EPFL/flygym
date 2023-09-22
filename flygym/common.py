import sys
from pathlib import Path

def get_data_path(package, file):
    if sys.version_info >= (3, 9):
        import importlib.resources
        with importlib.resources.path(package, file) as path:
            return Path(path).absolute()
    else:
        import pkg_resources
        return Path(pkg_resources.resource_filename(package, file)).absolute()