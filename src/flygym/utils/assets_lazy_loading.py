"""Lazy download of large asset files from a public S3 bucket.

The high-resolution `fullsize` meshes are too large to ship inside the
`flygym` package, so they are hosted on a public S3 bucket and downloaded the
first time they are needed, then cached on disk (see :func:`get_cache_root`).
The bucket is served over plain HTTP(S), so `urllib` is enough -- no boto3.

Each asset directory lives on the bucket as a `<name>.tar` archive plus a
`<name>.checksum` sidecar holding the tar's sha256 hex digest, both generated
by `scripts/dev/make_tar_for_lazy_loaded_assets.sh`. Names are versioned
(e.g. `neuromechfly_fullsize_meshes_20260623a`) so new revisions can be
uploaded without disturbing existing releases; bump the version constants in
the fly model modules to point a release at a new asset set.
"""

import hashlib
import os
import tarfile
import tempfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from loguru import logger

__all__ = ["get_cache_root", "lazy_load_asset_dir", "download_all_assets"]

#: Base HTTP(S) endpoint of the S3-compatible object store.
S3_ENDPOINT = "https://datasets.epfl.ch"
#: Name of the (public) bucket holding the FlyGym assets.
S3_BUCKET = "nely-public-share"
#: Top-level key prefix within the bucket under which all assets live.
S3_ROOT_PREFIX = "flygym_assets"

#: Read/write files in chunks of this size while streaming a download.
_CHUNK_SIZE = 1024 * 1024
#: How many times to (re)try a download before giving up. A dropped connection
#: yields a truncated tar that fails the checksum; the endpoint is flaky enough
#: that a single such failure shouldn't abort the whole run.
_MAX_ATTEMPTS = 3
#: Per-request timeout (seconds). Bounds how long a stalled connection can hang
#: before it errors out and the attempt is retried, rather than blocking forever.
_TIMEOUT = 30


def get_cache_root() -> Path:
    """Return the directory under which downloaded assets are cached:
    `$FLYGYM_ASSET_CACHE_DIR` if set (useful for CI caching), else
    `$XDG_CACHE_HOME/flygym_assets`, else `~/.cache/flygym_assets`.
    """
    env = os.environ.get("FLYGYM_ASSET_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    # Per the XDG spec, a relative XDG_CACHE_HOME is invalid and must be ignored.
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg and os.path.isabs(xdg):
        return Path(xdg) / S3_ROOT_PREFIX
    return Path.home() / ".cache" / S3_ROOT_PREFIX


def _object_url(key: str) -> str:
    # Quote each path segment but keep the slashes that delimit them.
    return f"{S3_ENDPOINT}/{S3_BUCKET}/{quote(key)}"


def _download_tar(name: str, dest: Path) -> None:
    """Download `<name>.tar` to `dest` and verify it against `<name>.checksum`,
    retrying on transient network errors and truncated (checksum-mismatched)
    downloads.
    """
    checksum_url = _object_url(f"{S3_ROOT_PREFIX}/{name}.checksum")
    with urlopen(checksum_url, timeout=_TIMEOUT) as response:
        expected = response.read().decode().split()[0]

    tar_url = _object_url(f"{S3_ROOT_PREFIX}/{name}.tar")
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        digest = hashlib.sha256()
        try:
            with (
                urlopen(tar_url, timeout=_TIMEOUT) as response,
                open(dest, "wb") as out,
            ):
                while chunk := response.read(_CHUNK_SIZE):
                    digest.update(chunk)
                    out.write(chunk)
        except OSError as e:
            reason = f"download failed ({e})"
        else:
            if digest.hexdigest() == expected:
                return
            reason = (
                f"integrity check failed "
                f"(expected sha256 {expected}, got {digest.hexdigest()})"
            )
        dest.unlink(missing_ok=True)
        if attempt == _MAX_ATTEMPTS:
            raise OSError(f"Could not download {name}.tar: {reason}")
        logger.warning(f"Retrying {name}.tar ({attempt}/{_MAX_ATTEMPTS}): {reason}")


def lazy_load_asset_dir(rel_path: os.PathLike | str) -> Path:
    """Return the absolute local path to a bucket asset directory, downloading it
    from S3 on first use.

    Args:
        rel_path: Name of the asset set on the bucket, i.e. the shared stem of
            `<rel_path>.tar` and `<rel_path>.checksum` under
            :data:`S3_ROOT_PREFIX` (as defined by each fly model's
            `*_MESH_DIR` constant).

    If the cached copy already exists it is returned as-is (no network access).
    Otherwise the tar is downloaded, verified, and extracted inside a temporary
    directory that is moved into place atomically, so an interrupted or
    concurrent download never leaves a partial cache.
    """
    name = Path(rel_path).as_posix()
    cache_dir = get_cache_root() / name
    if cache_dir.is_dir():
        return cache_dir

    logger.info(f"Downloading FlyGym asset '{name}' from S3 (one-time download)...")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=cache_dir.parent, suffix=".partial"
    ) as staging:
        staging = Path(staging)
        _download_tar(name, staging / "asset.tar")
        extracted = staging / "extracted"
        with tarfile.open(staging / "asset.tar") as tar:
            tar.extractall(extracted, filter="data")
        try:
            extracted.replace(cache_dir)
        except OSError:
            # Another process finished downloading the same asset while we were
            # working, so cache_dir is now populated and cannot be replaced.
            # Their copy is equivalent to ours: use it instead of failing.
            if not cache_dir.is_dir():
                raise
    logger.info(f"Finished downloading FlyGym asset '{name}'.")
    return cache_dir


def download_all_assets() -> list[Path]:
    """Eagerly download all remotely hosted assets into the cache.

    Useful for warming a CI cache or preparing an offline environment. Returns the
    list of local directories that now hold the assets.
    """
    # Imported lazily: each model owns its mesh-version constant, and those modules
    # import from this one, so a top-level import here would be circular.
    from flygym.compose.fly.neuromechfly import NEUROMECHFLY_FULLSIZE_MESH_DIR
    from flygym.compose.fly.flybody import FLYBODY_FULLSIZE_MESH_DIR
    from flygym.compose.fly.musculoskeletal import MUSCULOSKELETAL_MESH_DIR

    return [
        lazy_load_asset_dir(NEUROMECHFLY_FULLSIZE_MESH_DIR),
        lazy_load_asset_dir(FLYBODY_FULLSIZE_MESH_DIR),
        lazy_load_asset_dir(MUSCULOSKELETAL_MESH_DIR),
    ]
