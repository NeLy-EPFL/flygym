"""Check that the repo is ready to release the version in `pyproject.toml`.

Checks that:
- the version is not on PyPI yet,
- `docs/changelog.md` has a section for it,
- the Colab badges and pip installs in `tutorials/` point to its git tag,
- the release tag matches it (only if `--tag` is given).

Uses only the standard library so that CI can run it without installing the project.

Usage: python scripts/dev/check_release_readiness.py [--tag vX.Y.Z]
"""

import argparse
import re
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPI_URL = "https://pypi.org/pypi/flygym/{version}/json"
# Git refs in Colab badges (".../flygym/blob/<ref>/...") and pip installs
# ("...flygym.git@<ref>").
TUTORIAL_REF_PATTERN = re.compile(
    r"NeLy-EPFL/flygym(?:/blob/|\.git@)([\w.\-]+)", re.IGNORECASE
)


def is_on_pypi(version: str) -> bool:
    try:
        with urllib.request.urlopen(PYPI_URL.format(version=version), timeout=30):
            return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def has_changelog_section(version: str) -> bool:
    changelog = (REPO_ROOT / "docs/changelog.md").read_text()
    pattern = rf"^## Version {re.escape(version)}\s*$"
    return re.search(pattern, changelog, re.MULTILINE) is not None


def find_wrong_tutorial_refs(expected_ref: str) -> list[str]:
    wrong = []
    for path in sorted((REPO_ROOT / "tutorials").glob("*.ipynb")):
        for ref in TUTORIAL_REF_PATTERN.findall(path.read_text()):
            if ref != expected_ref:
                wrong.append(f"{path.relative_to(REPO_ROOT)}: {ref}")
    return wrong


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", help="Release tag to check against the version.")
    args = parser.parse_args()

    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        version = tomllib.load(f)["project"]["version"]
    tag = f"v{version}"
    print(f"Checking release readiness for version {version}")

    errors = []
    if args.tag is not None and args.tag != tag:
        errors.append(f"Release tag {args.tag} does not match version tag {tag}.")
    if is_on_pypi(version):
        errors.append(f"Version {version} already exists on PyPI.")
    if not has_changelog_section(version):
        errors.append(f"docs/changelog.md has no '## Version {version}' section.")
    for wrong_ref in find_wrong_tutorial_refs(tag):
        errors.append(
            f"Tutorial points to the wrong git ref (expected {tag}): {wrong_ref}"
        )

    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        sys.exit(1)
    print("All release readiness checks passed.")


if __name__ == "__main__":
    main()
