"""Lazy download of large asset files from a public S3 bucket.

Most asset files (configs, poses, and the simplified default NeuroMechFly meshes)
are small enough to ship inside the ``flygym`` package. The high-resolution
``fullsize`` meshes -- especially the FlyBody ``.obj`` meshes, which are an order
of magnitude larger than everything else combined -- would bloat the package and
the git repository, so they are hosted on an institution-managed S3 bucket and
pulled in *the first time they are needed*, similar to how PyTorch downloads
pretrained weights.

Downloaded files are cached on disk (see :func:`get_cache_root`) so the download
happens only once per machine. The bucket is public and served over a standard
S3-compatible HTTP endpoint, so plain ``urllib`` is enough -- no extra
dependencies (boto3 etc.) are required.

The bucket stores each remotely hosted asset directory as a flat, *versioned*
sub-prefix of :data:`S3_ROOT_PREFIX`, so future revisions can be uploaded under a
new name without disturbing existing releases. Bump the version constants below to
point a release at a new version. Example:

    bucket:  flygym_assets/neuromechfly_fullsize_meshes_20260623a/<file>
    cache:   ~/.cache/flygym_assets/neuromechfly_fullsize_meshes_20260623a/<file>
"""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
from xml.etree import ElementTree

from loguru import logger

__all__ = ["get_cache_root", "lazy_load_asset_dir", "prefetch_meshes"]

#: Base HTTP(S) endpoint of the S3-compatible object store.
S3_ENDPOINT = "https://datasets.epfl.ch"
#: Name of the (public) bucket holding the FlyGym assets.
S3_BUCKET = "nely-public-share"
#: Top-level key prefix within the bucket under which all assets live.
S3_ROOT_PREFIX = "flygym_assets"


# S3 ListObjectsV2 responses are namespaced; this is the namespace MinIO/S3 use.
_S3_XML_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def get_cache_root() -> Path:
    """Return the directory under which downloaded assets are cached.

    Resolution order:

    1. ``$FLYGYM_ASSET_CACHE_DIR`` if set (useful for CI caching or shared,
       read-only installs);
    2. ``$XDG_CACHE_HOME/flygym_assets`` if ``XDG_CACHE_HOME`` is set;
    3. ``~/.cache/flygym_assets`` otherwise.

    The directory is named ``flygym_assets`` to match the bucket's top-level
    prefix (:data:`S3_ROOT_PREFIX`).
    """
    env = os.environ.get("FLYGYM_ASSET_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    # Per the XDG Base Directory spec, a relative XDG_CACHE_HOME is invalid and
    # must be ignored (as is an unset/empty value).
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg and os.path.isabs(xdg):
        return Path(xdg) / S3_ROOT_PREFIX
    return Path.home() / ".cache" / S3_ROOT_PREFIX


def _object_url(key: str) -> str:
    # Quote each path segment but keep the slashes that delimit them.
    return f"{S3_ENDPOINT}/{S3_BUCKET}/{quote(key)}"


def _list_s3_prefix(prefix: str) -> list[dict]:
    """List every object under ``prefix`` via the public ListObjectsV2 API.

    Returns a list of ``{"key", "size", "etag"}`` dicts. Handles pagination via
    continuation tokens. The bucket is public, so the request is unsigned.
    """
    if not prefix.endswith("/"):
        prefix += "/"
    objects: list[dict] = []
    continuation_token: str | None = None
    while True:
        url = f"{S3_ENDPOINT}/{S3_BUCKET}?list-type=2&prefix={quote(prefix, safe='')}"
        if continuation_token is not None:
            url += f"&continuation-token={quote(continuation_token, safe='')}"
        with urlopen(url) as response:
            tree = ElementTree.fromstring(response.read())
        for contents in tree.findall("s3:Contents", _S3_XML_NS):
            key = contents.findtext("s3:Key", namespaces=_S3_XML_NS)
            if key is None or key.endswith("/"):
                continue  # skip "directory" placeholder keys
            size = int(contents.findtext("s3:Size", default="0", namespaces=_S3_XML_NS))
            etag = contents.findtext("s3:ETag", default="", namespaces=_S3_XML_NS)
            objects.append({"key": key, "size": size, "etag": etag.strip('"')})
        is_truncated = (
            tree.findtext("s3:IsTruncated", default="false", namespaces=_S3_XML_NS)
            == "true"
        )
        if not is_truncated:
            break
        continuation_token = tree.findtext(
            "s3:NextContinuationToken", namespaces=_S3_XML_NS
        )
        if not continuation_token:
            break
    return objects


def _is_up_to_date(path: Path, size: int, etag: str) -> bool:
    """Return True if ``path`` already holds the object described by (size, etag).

    For non-multipart uploads the S3 ETag is the MD5 hex digest of the content,
    which we verify. Multipart ETags contain a ``-`` and are not plain MD5, so we
    fall back to a size check for those.
    """
    if not path.is_file():
        return False
    if path.stat().st_size != size:
        return False
    if etag and "-" not in etag:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        return digest == etag
    return True


def _download_object(key: str, dest: Path, size: int, etag: str) -> None:
    """Download a single object to ``dest`` atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urlopen(_object_url(key)) as response:
            shutil.copyfileobj(response, out)
        if not _is_up_to_date(tmp_path, size, etag):
            raise OSError(
                f"Downloaded asset failed integrity check: {key} "
                f"(expected {size} bytes, etag {etag!r})"
            )
        tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)


def _download_prefix(s3_prefix: str, dest_dir: Path) -> Path:
    """Download every object under ``s3_prefix`` into ``dest_dir`` (skipping files
    that are already present and up to date). Returns ``dest_dir``.
    """
    objects = _list_s3_prefix(s3_prefix)
    if not objects:
        raise FileNotFoundError(
            f"No assets found on S3 under prefix '{s3_prefix}'. The bucket may be "
            "unreachable or the asset may have been moved."
        )
    prefix = s3_prefix if s3_prefix.endswith("/") else s3_prefix + "/"
    pending = []
    for obj in objects:
        rel_key = obj["key"][len(prefix) :]
        dest = dest_dir / rel_key
        if not _is_up_to_date(dest, obj["size"], obj["etag"]):
            pending.append((obj, dest))

    if pending:
        total_mb = sum(obj["size"] for obj, _ in pending) / 1e6
        logger.info(
            f"Downloading {len(pending)} FlyGym asset file(s) "
            f"({total_mb:.1f} MB) from S3 to {dest_dir} (one-time download)..."
        )
        for obj, dest in pending:
            _download_object(obj["key"], dest, obj["size"], obj["etag"])
        logger.info("Finished downloading FlyGym assets.")
    return dest_dir


def lazy_load_asset_dir(rel_path: os.PathLike | str) -> Path:
    """Return the absolute local path to a bucket asset directory, downloading it
    from S3 on first use.

    Args:
        rel_path: Path of the directory within the bucket, relative to
            :data:`S3_ROOT_PREFIX` (e.g. ``"neuromechfly_fullsize_meshes_20260623a"``,
            as defined by each fly model's ``*_FULLSIZE_MESH_DIR`` constant).

    The directory is cached under :func:`get_cache_root` keyed by ``rel_path``. If
    the cached copy already exists it is returned as-is (no network access);
    otherwise the whole directory is downloaded into a temporary location and moved
    into place atomically, so an interrupted or concurrent download never leaves a
    partial cache.

    Raises:
        FileNotFoundError: If ``rel_path`` does not exist in the bucket.
    """
    rel_path = Path(rel_path)
    cache_dir = get_cache_root() / rel_path
    if cache_dir.is_dir():
        return cache_dir

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=cache_dir.parent, suffix=".partial"))
    try:
        _download_prefix(f"{S3_ROOT_PREFIX}/{rel_path.as_posix()}", staging)
        try:
            staging.replace(cache_dir)
        except OSError:
            # Another process finished downloading the same asset while we were
            # working: os.replace cannot move onto the now-populated directory.
            # Their copy is equivalent to ours, so use it instead of failing.
            if not cache_dir.is_dir():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return cache_dir


def prefetch_meshes() -> list[Path]:
    """Eagerly download all remotely hosted meshes into the cache.

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
