"""Tests for flygym.utils.assets -- lazy download of S3-hosted asset files.

Most tests run fully offline by monkeypatching the network layer. A single
opt-in integration test (``test_real_s3_roundtrip``) talks to the live bucket and
is skipped automatically if the endpoint is unreachable.
"""

import hashlib
import io
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


class TestIsUpToDate:
    def test_missing_file(self, tmp_path):
        assert not assets_lazy_loading._is_up_to_date(tmp_path / "nope", 1, "x")

    def test_size_mismatch(self, tmp_path):
        p = tmp_path / "f"
        p.write_bytes(b"abc")
        assert not assets_lazy_loading._is_up_to_date(p, 99, "")

    def test_md5_match_and_mismatch(self, tmp_path):
        p = tmp_path / "f"
        data = b"hello world"
        p.write_bytes(data)
        good = hashlib.md5(data).hexdigest()
        assert assets_lazy_loading._is_up_to_date(p, len(data), good)
        assert not assets_lazy_loading._is_up_to_date(p, len(data), "0" * 32)

    def test_multipart_etag_falls_back_to_size(self, tmp_path):
        p = tmp_path / "f"
        data = b"hello world"
        p.write_bytes(data)
        # Multipart ETags contain a dash and are not a plain MD5; size match wins.
        assert assets_lazy_loading._is_up_to_date(p, len(data), "deadbeef-2")


# ---------------------------------------------------------------------------
# Download path, exercised offline via a fake "remote"
# ---------------------------------------------------------------------------


# Flat, versioned S3 sub-prefix -- mirrors the real bucket layout.
_DEMO_VERSION = "demo_fullsize_meshes_vtest"
_DEMO_PREFIX = f"flygym_assets/{_DEMO_VERSION}/"


@pytest.fixture
def fake_remote(monkeypatch):
    """Serve a small in-memory object store through the assets module's network
    seam, so the full download/verify/cache flow runs without touching S3.
    """
    store = {
        f"{_DEMO_PREFIX}a.stl": b"aaaa",
        f"{_DEMO_PREFIX}b.stl": b"bbbbbb",
    }
    objects = [
        {"key": k, "size": len(v), "etag": hashlib.md5(v).hexdigest()}
        for k, v in store.items()
    ]

    def fake_list(prefix):
        if not prefix.endswith("/"):
            prefix += "/"
        return [o for o in objects if o["key"].startswith(prefix)]

    def fake_urlopen(url):
        # url is the object URL; recover the key after the bucket name.
        marker = f"/{assets_lazy_loading.S3_BUCKET}/"
        key = url.split(marker, 1)[1]
        return io.BytesIO(store[key])

    monkeypatch.setattr(assets_lazy_loading, "_list_s3_prefix", fake_list)
    monkeypatch.setattr(assets_lazy_loading, "urlopen", fake_urlopen)
    return store


def test_download_prefix_writes_and_is_idempotent(fake_remote, tmp_path):
    dest = tmp_path / "out"
    assets_lazy_loading._download_prefix(_DEMO_PREFIX, dest)
    assert (dest / "a.stl").read_bytes() == b"aaaa"
    assert (dest / "b.stl").read_bytes() == b"bbbbbb"

    # Re-running must not re-download (no .part temp files left behind, content
    # unchanged) since everything is already up to date.
    assets_lazy_loading._download_prefix(_DEMO_PREFIX, dest)
    assert sorted(p.name for p in dest.iterdir()) == ["a.stl", "b.stl"]


def test_download_object_integrity_check(fake_remote, tmp_path, monkeypatch):
    dest = tmp_path / "corrupt.stl"
    # Lie about the expected size so the post-download check fails.
    with pytest.raises(OSError, match="integrity check"):
        assets_lazy_loading._download_object(
            f"{_DEMO_PREFIX}a.stl", dest, size=999, etag=""
        )
    assert not dest.exists()  # nothing left behind on failure


def test_lazy_load_asset_dir_downloads_and_caches(fake_remote, tmp_path, monkeypatch):
    monkeypatch.setenv("FLYGYM_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    # First call downloads from S3 and caches under the versioned dir name.
    out = assets_lazy_loading.lazy_load_asset_dir(_DEMO_VERSION)
    assert out == tmp_path / "cache" / _DEMO_VERSION
    assert (out / "a.stl").read_bytes() == b"aaaa"
    assert not list(out.parent.glob("*.partial")), "staging dir not cleaned up"


def test_lazy_load_asset_dir_returns_cache_without_network(tmp_path, monkeypatch):
    """A populated cache dir is returned as-is, with no S3 access (works offline)."""
    monkeypatch.setenv("FLYGYM_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    cached = tmp_path / "cache" / _DEMO_VERSION
    cached.mkdir(parents=True)
    (cached / "a.stl").write_bytes(b"aaaa")

    def boom(*args, **kwargs):
        raise AssertionError("must not hit the network when already cached")

    monkeypatch.setattr(assets_lazy_loading, "_download_prefix", boom)
    assert assets_lazy_loading.lazy_load_asset_dir(_DEMO_VERSION) == cached


def test_lazy_load_asset_dir_interrupted_download_leaves_no_cache(
    tmp_path, monkeypatch
):
    """If the download fails partway, no (partial) cache dir is left behind."""
    monkeypatch.setenv("FLYGYM_ASSET_CACHE_DIR", str(tmp_path / "cache"))

    def boom(*args, **kwargs):
        raise RuntimeError("network died mid-download")

    monkeypatch.setattr(assets_lazy_loading, "_download_prefix", boom)
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


@pytest.mark.network
def test_real_s3_roundtrip():
    """Opt-in: list and download a single small object from the live bucket."""
    from flygym.compose.fly.neuromechfly import NEUROMECHFLY_FULLSIZE_MESH_DIR

    prefix = f"{assets_lazy_loading.S3_ROOT_PREFIX}/{NEUROMECHFLY_FULLSIZE_MESH_DIR}/"
    try:
        objects = assets_lazy_loading._list_s3_prefix(prefix)
    except Exception as e:  # network unavailable in this environment
        pytest.skip(f"S3 endpoint unreachable: {e}")
    assert objects, "expected objects under the fullsize prefix"
    smallest = min(objects, key=lambda o: o["size"])
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        dest = Path(d) / "obj.stl"
        assets_lazy_loading._download_object(
            smallest["key"], dest, smallest["size"], smallest["etag"]
        )
        assert dest.stat().st_size == smallest["size"]
