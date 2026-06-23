#!/bin/bash
set -e

BUILD_DIR="site"
BRANCH="gh-pages"
COMMIT_MSG="Deploy $(date '+%Y-%m-%d %H:%M:%S')"

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

if [ -d "$BUILD_DIR" ]; then
    read -p "The build directory '$BUILD_DIR' already exists. Do you want to remove it and continue? (y/n) "
    if [[ $REPLY != "y" ]]; then
        echo "Stopping here."
        exit 1
    fi
    rm -rf "$BUILD_DIR"
fi

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

# Build the documentation
echo "Building documentation..."
uv run properdocs build

# Push built site to a separate branch to be served by GitHub Pages
cd "$BUILD_DIR"
git init
git checkout --orphan "$BRANCH"
git add -A
git commit -m "$COMMIT_MSG"
git remote add origin $(git -C .. remote get-url origin)
git push --force origin "$BRANCH"

# Cleanup
cd ..
rm -rf "$BUILD_DIR/.git"

echo "✅ Documentation deployed successfully to branch '$BRANCH'."