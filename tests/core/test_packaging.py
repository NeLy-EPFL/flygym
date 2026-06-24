"""Tests that the built distribution actually ships the bundled asset files.

These are *packaging* tests, not runtime tests: they build a wheel from the
repository and inspect its contents. The motivation is a setuptools footgun ---
``include-package-data = true`` pulls in nothing on a clean tree (there is no
MANIFEST.in and setuptools-scm is not installed), so an asset that is committed
to git can still silently vanish from the shipped wheel unless it is matched by
an explicit ``[tool.setuptools.package-data]`` glob. A from-scratch build (CI,
or any tree without a stale ``SOURCES.txt``) is the only thing that catches it,
so that is exactly what these tests do.

The wheel is the artifact users ``pip install`` (flygym ships a pure-Python
``py3-none-any`` wheel), so its contents are the ground truth for "what ends up
on disk after installation".
"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

# The wheel build needs the PEP 517 frontend; skip cleanly if it is absent
# (e.g. a minimal runtime-only environment without the ``dev`` extra).
build = pytest.importorskip(
    "build", reason="the `build` package (dev extra) is required"
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

# Large meshes deliberately kept out of the package and fetched lazily from S3
# at runtime (see ``flygym.utils.assets_lazy_loading``). These mirror the
# ``[tool.setuptools.exclude-package-data]`` globs in pyproject.toml and are
# expressed as wheel-arcname prefixes (``flygym/...``).
EXCLUDED_MESH_PREFIXES = (
    "flygym/assets/model/neuromechfly/meshes/fullsize/",
    "flygym/assets/model/flybody/meshes/",
    "flygym/assets/model/musculoskeletal/meshes/",
)


def _is_excluded(arcname: str) -> bool:
    return any(arcname.startswith(p) for p in EXCLUDED_MESH_PREFIXES)


@pytest.fixture(scope="session")
def wheel_contents(tmp_path_factory) -> set[str]:
    """Build a wheel from a clean copy of the project and return its file list.

    We copy the project into a scratch dir (excluding any pre-existing
    ``*.egg-info``) rather than building in place: a stale ``SOURCES.txt`` left
    in the source tree would be reused as the manifest and could mask a missing
    ``package-data`` declaration --- the very failure these tests exist to catch.
    """
    if not (REPO_ROOT / "pyproject.toml").exists():  # pragma: no cover - sanity
        pytest.skip("cannot locate the project root (pyproject.toml)")

    work = tmp_path_factory.mktemp("flygym_pkg_build")
    proj = work / "proj"
    proj.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        src = REPO_ROOT / name
        if src.exists():
            shutil.copy2(src, proj / name)
    shutil.copytree(
        SRC_DIR,
        proj / "src",
        ignore=shutil.ignore_patterns("*.egg-info", "__pycache__"),
    )

    outdir = work / "dist"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(outdir),
            str(proj),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"wheel build failed:\n{result.stdout}\n{result.stderr}"
    )

    wheels = list(outdir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    with zipfile.ZipFile(wheels[0]) as zf:
        return {n for n in zf.namelist() if not n.endswith("/")}


def _expected_arcnames(package: str, asset_root: Path, *, exclude_meshes: bool):
    """Map every asset file under ``asset_root`` to its expected wheel arcname.

    ``asset_root`` lives under ``src/<package>/``; the wheel drops the ``src/``
    layer, so e.g. ``src/flygym/assets/x.stl`` -> ``flygym/assets/x.stl``.
    """
    for path in sorted(asset_root.rglob("*")):
        if not path.is_file():
            continue
        arcname = path.relative_to(SRC_DIR).as_posix()
        if exclude_meshes and _is_excluded(arcname):
            continue
        yield arcname


class TestFlygymAssets:
    """The core ``flygym`` package bundles meshes, poses and configs needed to
    build the default models without any network access."""

    def test_all_bundled_assets_present(self, wheel_contents):
        """Every committed asset (except the lazily-downloaded large meshes)
        must appear in the wheel. Derived from the source tree so newly added
        assets are covered automatically."""
        expected = set(
            _expected_arcnames(
                "flygym", SRC_DIR / "flygym" / "assets", exclude_meshes=True
            )
        )
        assert expected, "no source assets found -- test wiring is broken"
        missing = sorted(expected - wheel_contents)
        assert not missing, f"assets committed but missing from the wheel: {missing}"

    @pytest.mark.parametrize(
        "arcname",
        [
            # The default (simplified) NeuroMechFly mesh set -- the package is
            # unusable offline without these.
            "flygym/assets/model/neuromechfly/meshes/simplified_max2000faces/c_thorax.stl",
            # Pose / config files loaded when composing a fly.
            "flygym/assets/model/neuromechfly/pose/neutral/yaw_roll_pitch.yaml",
            "flygym/assets/model/neuromechfly/vision.yaml",
            # Vision: ommatidia layout loaded by the retina.
            "flygym/assets/model/neuromechfly/compound_eye.npz",
            # The flybody model definition.
            "flygym/assets/model/flybody/fruitfly.xml",
            # The musculoskeletal MJCF (DEFAULT_MUSCULOSKELETAL_XML) -- regression
            # guard: this was added without a matching package-data glob and so
            # was dropped from clean builds.
            "flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml",
        ],
    )
    def test_critical_asset_shipped(self, wheel_contents, arcname):
        # Only assert on files that actually exist in the source tree, so the
        # parametrization stays honest if a file is renamed.
        if not (SRC_DIR / arcname).exists():
            pytest.skip(f"{arcname} not present in source tree")
        assert arcname in wheel_contents, f"{arcname} missing from the wheel"

    def test_large_meshes_excluded(self, wheel_contents):
        """The big meshes are fetched lazily from S3 and must never be bundled,
        even if a developer has them downloaded into the source tree."""
        leaked = sorted(n for n in wheel_contents if _is_excluded(n))
        assert not leaked, f"large meshes leaked into the wheel: {leaked}"


class TestFlygymDemoAssets:
    """The ``flygym_demo`` package ships recording/mocap data used by the
    bundled demo scripts. Kept separate from the core-package checks above."""

    def test_all_demo_assets_present(self, wheel_contents):
        roots = [
            SRC_DIR / "flygym_demo" / sub / "assets"
            for sub in (
                "spotlight_data",
                "complex_terrain",
                "muscle_imitation",
                "ball_flybody_data",
            )
        ]
        expected = set()
        for root in roots:
            if root.exists():
                expected.update(
                    _expected_arcnames("flygym_demo", root, exclude_meshes=False)
                )
        assert expected, "no flygym_demo assets found -- test wiring is broken"
        missing = sorted(expected - wheel_contents)
        assert not missing, f"flygym_demo assets missing from the wheel: {missing}"
