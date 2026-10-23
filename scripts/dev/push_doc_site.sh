#!/bin/bash
set -e

DOCS_REMOTE="flygym-docs"
DOCS_REMOTE_URL="git@github.com:NeLy-EPFL/flygym-docs.git"

read -p "Have you checked that all tutorial notebooks are properly executed and without errors? (y/n) "
if [[ $REPLY != "y" ]]; then
    echo "Stopping here."
    exit 1
fi

read -p "Have you run 'properdocs serve' and checked the site locally? (y/n) "
if [[ $REPLY != "y" ]]; then
    echo "Stopping here."
    exit 1
fi

read -p "Current directory is $(pwd). Is this the flygym root directory? (y/n) "
if [[ $REPLY != "y" ]]; then
    echo "Stopping here."
    exit 1
fi

# Only released versions are published, so deploy from main only.
if [[ $(git rev-parse --abbrev-ref HEAD) != "main" ]]; then
    echo "Docs are only deployed from main (released versions). Stopping here."
    exit 1
fi

VERSION=$(uv version --short)

# Ensure vendor files (MuJoCo-WASM + Three.js) are present, downloading them if
# needed. Also offer to regenerate the MJCF/STL assets (viewer + game) from the
# live model.
VIEWER_DIR="wasm/viewer"
GAME_DIR="wasm/game"
uv run python scripts/dev/properdocs_hooks.py --vendor-only
if [ ! -f "$VIEWER_DIR/assets/model/fly.xml" ] || [ ! -f "$GAME_DIR/assets/model/fly.xml" ]; then
    echo "WASM viewer/game assets not found; generating them now..."
    REGEN_ASSETS="y"
else
    read -p "Regenerate the interactive viewer + game assets (mesh files etc.)? (y/n) " REGEN_ASSETS
fi
if [[ $REGEN_ASSETS == "y" ]]; then
    uv run python scripts/dev/build_wasm_viewer_assets.py
    uv run python scripts/dev/build_wasm_game_assets.py
fi
uv run python scripts/dev/properdocs_hooks.py --vendor-only

# Build with mike and push the site to the gh-pages branch of the separate docs
# repo, under a subdirectory for this version. mike works on local refs only, so
# fetch the docs repo first. Then point the site root at the new version.
git remote add "$DOCS_REMOTE" "$DOCS_REMOTE_URL" 2>/dev/null || true
git fetch "$DOCS_REMOTE" gh-pages
uv run mike deploy --push --remote "$DOCS_REMOTE" "$VERSION"
uv run mike set-default --push --remote "$DOCS_REMOTE" "$VERSION"

echo "Done. Documentation deployed successfully (version '$VERSION')."
