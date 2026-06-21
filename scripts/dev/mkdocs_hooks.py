"""MkDocs hooks: ensure vendor and generated assets are present before build/serve.

The WASM apps (interactive viewer + game) live in the top-level ``wasm/`` tree,
*outside* the MkDocs ``docs/`` dir, so they are not picked up automatically.
``on_post_build`` copies ``wasm/`` into the built site (as ``<site>/wasm/``) so
the iframes in the docs (e.g. ``../wasm/viewer/viewer.html``) resolve, and
``on_serve`` watches the tree so edits trigger a rebuild during ``mkdocs serve``.
"""

import io
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_WASM_DIR = _REPO / "wasm"
_VENDOR_DIR = _WASM_DIR / "shared" / "vendor"
_VIEWER_ASSETS_DIR = _WASM_DIR / "viewer" / "assets"
_GAME_ASSETS_DIR = _WASM_DIR / "game" / "assets"
_GAME_BUILD_SCRIPT = _REPO / "scripts" / "dev" / "build_wasm_game_assets.py"

_MUJOCO_VERSION = "3.9.0"
_THREE_VERSION = "0.169.0"


def on_startup(command, dirty):
    _ensure_vendor()
    _ensure_assets()


def on_post_build(config, **kwargs):
    """Copy the top-level wasm/ tree into the built site so its apps are served."""
    dest = Path(config["site_dir"]) / "wasm"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        _WASM_DIR, dest, ignore=shutil.ignore_patterns(".gitignore", "README.md")
    )
    print(f"mkdocs: copied wasm/ -> {dest}")


def on_serve(server, config, builder, **kwargs):
    """Rebuild when anything in wasm/ changes during `mkdocs serve`."""
    server.watch(str(_WASM_DIR))
    return server


def _ensure_vendor():
    if (_VENDOR_DIR / "mujoco" / "mujoco.wasm").exists() and (
        _VENDOR_DIR / "three" / "three.module.js"
    ).exists():
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
                "package/examples/jsm/controls/OrbitControls.js": three_dir
                / "OrbitControls.js",
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
    _build_assets_if_missing(
        _VIEWER_ASSETS_DIR / "model" / "fly.xml",
        _REPO / "scripts" / "dev" / "build_wasm_viewer_assets.py",
        "viewer",
    )
    # The game build script may not exist yet (added in a later step); skip if so.
    if _GAME_BUILD_SCRIPT.exists():
        _build_assets_if_missing(
            _GAME_ASSETS_DIR / "model" / "fly.xml", _GAME_BUILD_SCRIPT, "game"
        )


def _build_assets_if_missing(sentinel: Path, script: Path, label: str):
    if sentinel.exists():
        return
    print(f"mkdocs: building WASM {label} assets (MJCF + STL meshes)...")
    result = subprocess.run([sys.executable, str(script)], cwd=_REPO)
    if result.returncode != 0:
        raise SystemExit(f"mkdocs: {script.name} failed.")


if __name__ == "__main__":
    _ensure_vendor()
    if "--vendor-only" not in sys.argv:
        _ensure_assets()
