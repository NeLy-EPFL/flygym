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

read -p "Have you run 'mkdocs serve' and checked the site locally? (y/n) "
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

# The interactive (WASM) viewer's generated assets (the 39 STL meshes and
# model_meta.json) are gitignored and only ever published to the gh-pages branch,
# so make sure they exist -- and offer to regenerate them from the live model --
# before mkdocs bundles them into the site.
WASM_DIR="docs/wasm_viewer"
if [ ! -f "$WASM_DIR/assets/model/fly.xml" ]; then
    echo "Interactive viewer assets not found; generating them now..."
    REGEN_ASSETS="y"
else
    read -p "Regenerate the interactive viewer assets (mesh files etc.)? (y/n) " REGEN_ASSETS
fi
if [[ $REGEN_ASSETS == "y" ]]; then
    uv run python scripts/build_wasm_viewer_assets.py
fi
if [ ! -f "$WASM_DIR/vendor/mujoco/mujoco.wasm" ]; then
    echo "ERROR: $WASM_DIR/vendor is missing (MuJoCo-WASM + three.js)."
    echo "See $WASM_DIR/README.md for how to vendor them."
    exit 1
fi

# Build the documentation
echo "Building documentation..."
mkdocs build

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