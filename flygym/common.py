import sys
from pathlib import Path


def get_data_path(package: str, file: str) -> Path:
    """Given the names of the package and a file (or directory) included as
    package data, return the absolute path of it in the installed package.
    This wrapper handles the ``pkg_resources``-to-``importlib.resources``
    API change in Python."""
    if sys.version_info >= (3, 9):
        import importlib.resources

        return importlib.resources.files(package) / file
    else:
        import pkg_resources

        return Path(pkg_resources.resource_filename(package, file)).absolute()
