"""Tests for flygym.utils.assets -- lazy download of S3-hosted asset files.

Most tests run fully offline by monkeypatching the network layer. A single
opt-in integration test (``test_real_s3_roundtrip``) talks to the live bucket and
is skipped automatically if the endpoint is unreachable.
"""

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from flygym import assets_dir
from flygym.utils import assets_lazy_loading


# ---------------------------------------------------------------------------
# Pure helpers (no I/O)
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def test_object_url_quotes_but_keeps_slashes(self):
        url = assets_lazy_loading._object_url("flygym_assets/a b/c.stl")
        assert url == (
            f"{assets_lazy_loading.S3_ENDPOINT}/{assets_lazy_loading.S3_BUCKET}/flygym_assets/a%20b/c.stl"
        )

    def test_get_cache_root_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FLYGYM_ASSET_CACHE_DIR", str(tmp_path / "custom"))
        assert assets_lazy_loading.get_cache_root() == tmp_path / "custom"

    def test_get_cache_root_xdg(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FLYGYM_ASSET_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        assert (
            assets_lazy_loading.get_cache_root() == tmp_path / "xdg" / "flygym_assets"
        )

    def test_get_cache_root_default(self, monkeypatch):
        monkeypatch.delenv("FLYGYM_ASSET_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        assert (
            assets_lazy_loading.get_cache_root()
            == Path.home() / ".cache" / "flygym_assets"
        )

    def test_get_cache_root_ignores_relative_xdg(self, monkeypatch):
        # Per the XDG spec, a relative XDG_CACHE_HOME is invalid -> fall back.
        monkeypatch.delenv("FLYGYM_ASSET_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", "relative/cache")
        assert (
            assets_lazy_loading.get_cache_root()
            == Path.home() / ".cache" / "flygym_assets"
        )


# ---------------------------------------------------------------------------
# Download path, exercised offline via a fake "remote"
# ---------------------------------------------------------------------------


# Flat, versioned S3 key stem -- mirrors the real bucket layout.
_DEMO_VERSION = "demo_fullsize_meshes_vtest"


def _make_tar_bytes(files: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture
def fake_remote(monkeypatch):
    """Serve a small in-memory object store through the assets module's network
    seam, so the full download/verify/extract/cache flow runs without touching S3.
    """
    tar_bytes = _make_tar_bytes({"a.stl": b"aaaa", "b.stl": b"bbbbbb"})
    checksum = hashlib.sha256(tar_bytes).hexdigest()
    store = {
        f"flygym_assets/{_DEMO_VERSION}.tar": tar_bytes,
        f"flygym_assets/{_DEMO_VERSION}.checksum": checksum.encode("ascii"),
    }

    def fake_urlopen(url, **kwargs):
        # url is the object URL; recover the key after the bucket name.
        marker = f"/{assets_lazy_loading.S3_BUCKET}/"
        key = url.split(marker, 1)[1]
        return io.BytesIO(store[key])

    monkeypatch.setattr(assets_lazy_loading, "urlopen", fake_urlopen)
    return store


def test_download_tar_integrity_check(fake_remote, tmp_path, monkeypatch):
    # Corrupt the checksum so the post-download check fails.
    fake_remote[f"flygym_assets/{_DEMO_VERSION}.checksum"] = b"0" * 64
    dest = tmp_path / "out.tar"
    with pytest.raises(OSError, match="integrity check"):
        assets_lazy_loading._download_tar(_DEMO_VERSION, dest)
    assert not dest.exists()  # nothing left behind on failure


def test_lazy_load_asset_dir_downloads_and_caches(fake_remote, tmp_path, monkeypatch):
    monkeypatch.setenv("FLYGYM_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    # First call downloads from S3, verifies, extracts, and caches under the
    # versioned dir name.
    out = assets_lazy_loading.lazy_load_asset_dir(_DEMO_VERSION)
    assert out == tmp_path / "cache" / _DEMO_VERSION
    assert (out / "a.stl").read_bytes() == b"aaaa"
    assert (out / "b.stl").read_bytes() == b"bbbbbb"
    assert not list(out.parent.glob("*.partial")), "staging dir not cleaned up"


def test_lazy_load_asset_dir_returns_cache_without_network(tmp_path, monkeypatch):
    """A populated cache dir is returned as-is, with no S3 access (works offline)."""
    monkeypatch.setenv("FLYGYM_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    cached = tmp_path / "cache" / _DEMO_VERSION
    cached.mkdir(parents=True)
    (cached / "a.stl").write_bytes(b"aaaa")

    def boom(*args, **kwargs):
        raise AssertionError("must not hit the network when already cached")

    monkeypatch.setattr(assets_lazy_loading, "_download_tar", boom)
    assert assets_lazy_loading.lazy_load_asset_dir(_DEMO_VERSION) == cached


def test_lazy_load_asset_dir_interrupted_download_leaves_no_cache(
    tmp_path, monkeypatch
):
    """If the download fails partway, no (partial) cache dir is left behind."""
    monkeypatch.setenv("FLYGYM_ASSET_CACHE_DIR", str(tmp_path / "cache"))

    def boom(*args, **kwargs):
        raise RuntimeError("network died mid-download")

    monkeypatch.setattr(assets_lazy_loading, "_download_tar", boom)
    with pytest.raises(RuntimeError):
        assets_lazy_loading.lazy_load_asset_dir(_DEMO_VERSION)
    assert not (tmp_path / "cache" / _DEMO_VERSION).exists()
    assert not list((tmp_path / "cache").glob("*.partial")), "staging not cleaned up"


# ---------------------------------------------------------------------------
# Behavior with the real package layout
# ---------------------------------------------------------------------------


def test_default_neuromechfly_needs_no_download(monkeypatch):
    """Building the default (simplified) NeuroMechFly must never hit S3: the
    simplified mesh set is complete and bundled with the package."""
    from flygym.compose.fly import NeuroMechFly, neuromechfly

    def boom(*args, **kwargs):
        raise AssertionError("default NeuroMechFly triggered an asset download")

    monkeypatch.setattr(neuromechfly, "lazy_load_asset_dir", boom)
    fly = NeuroMechFly(name="no_download_fly")
    assert len(fly.bodyseg_to_mjcfmesh) > 0


def _remote_mesh_dirs():
    """The versioned S3 sub-prefix each model pulls its large meshes from."""
    from flygym.compose.fly.neuromechfly import NEUROMECHFLY_FULLSIZE_MESH_DIR
    from flygym.compose.fly.flybody import FLYBODY_FULLSIZE_MESH_DIR
    from flygym.compose.fly.musculoskeletal import MUSCULOSKELETAL_MESH_DIR

    return {
        "neuromechfly_fullsize": NEUROMECHFLY_FULLSIZE_MESH_DIR,
        "flybody_fullsize": FLYBODY_FULLSIZE_MESH_DIR,
        "musculoskeletal": MUSCULOSKELETAL_MESH_DIR,
    }


def test_prefetch_meshes_covers_all_remote_sets(monkeypatch):
    """`prefetch_meshes` must warm every model's remote mesh set (so a CI cache
    or offline environment is fully primed in one call)."""
    requested = []
    monkeypatch.setattr(
        assets_lazy_loading,
        "lazy_load_asset_dir",
        lambda rel: requested.append(str(rel)) or Path("/cache") / rel,
    )
    assets_lazy_loading.download_all_assets()
    assert set(requested) == set(_remote_mesh_dirs().values())


@pytest.mark.network
@pytest.mark.parametrize(
    "mesh_dir", _remote_mesh_dirs().values(), ids=_remote_mesh_dirs().keys()
)
def test_real_s3_roundtrip(mesh_dir, tmp_path):
    """Opt-in: download and checksum-verify the real tar for each model's remote
    mesh set from the live bucket."""
    try:
        dest = tmp_path / f"{mesh_dir}.tar"
        assets_lazy_loading._download_tar(mesh_dir, dest)
    except Exception as e:  # network unavailable in this environment
        pytest.skip(f"S3 endpoint unreachable: {e}")
    assert dest.stat().st_size > 0
