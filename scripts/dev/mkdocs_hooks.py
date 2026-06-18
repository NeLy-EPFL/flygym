"""MkDocs hooks: ensure vendor and generated assets are present before build/serve."""

import io
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_WASM_DIR = _REPO / "docs" / "wasm_viewer"
_VENDOR_DIR = _WASM_DIR / "vendor"
_ASSETS_DIR = _WASM_DIR / "assets"

_MUJOCO_VERSION = "3.9.0"
_THREE_VERSION = "0.169.0"


def on_startup(command, dirty):
    _ensure_vendor()
    _ensure_assets()


def _ensure_vendor():
    if (
        (_VENDOR_DIR / "mujoco" / "mujoco.wasm").exists()
        and (_VENDOR_DIR / "three" / "three.module.js").exists()
    ):
        return

    print("mkdocs: downloading WASM viewer vendor files...")
    try:
        _fetch_npm_files(
            f"https://registry.npmjs.org/@mujoco/mujoco/-/mujoco-{_MUJOCO_VERSION}.tgz",
            {
                "package/mujoco.js": _VENDOR_DIR / "mujoco" / "mujoco.js",
                "package/mujoco.wasm": _VENDOR_DIR / "mujoco" / "mujoco.wasm",
                "package/mujoco.d.ts": _VENDOR_DIR / "mujoco" / "mujoco.d.ts",
            },
        )
        three_dir = _VENDOR_DIR / "three"
        _fetch_npm_files(
            f"https://registry.npmjs.org/three/-/three-{_THREE_VERSION}.tgz",
            {
                "package/build/three.module.js": three_dir / "three.module.js",
                "package/examples/jsm/controls/OrbitControls.js": three_dir / "OrbitControls.js",
            },
        )
        (three_dir / "VERSION.txt").write_text(f"three@{_THREE_VERSION}\n")
    except Exception as exc:
        raise SystemExit(
            f"mkdocs: failed to download vendor files: {exc}\n"
            f"  @mujoco/mujoco@{_MUJOCO_VERSION} and three@{_THREE_VERSION} are required.\n"
            f"  Run: npm install @mujoco/mujoco@{_MUJOCO_VERSION} three@{_THREE_VERSION}\n"
            f"  and copy the files into {_VENDOR_DIR}/"
        ) from exc
    print("mkdocs: vendor files ready.")


def _fetch_npm_files(url, members):
    print(f"  {url}")
    with urllib.request.urlopen(url) as resp:
        data = io.BytesIO(resp.read())
    with tarfile.open(fileobj=data, mode="r:gz") as tar:
        for archive_path, dest in members.items():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(tar.getmember(archive_path)).read())


def _ensure_assets():
    if (_ASSETS_DIR / "model" / "fly.xml").exists():
        return

    print("mkdocs: building WASM viewer assets (fly.xml + STL meshes)...")
    result = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "build_wasm_viewer_assets.py")],
        cwd=_REPO,
    )
    if result.returncode != 0:
        raise SystemExit("mkdocs: build_wasm_viewer_assets.py failed.")


if __name__ == "__main__":
    import sys as _sys
    _ensure_vendor()
    if "--vendor-only" not in _sys.argv:
        _ensure_assets()
