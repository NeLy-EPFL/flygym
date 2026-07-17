#!/bin/bash
set -e

DOCS_REMOTE_NAME="flygym-docs"
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

# Read version from pyproject.toml and let the user confirm or override (e.g.
# to deploy a dev preview as "2.1.1 (dev)" before the release).
VERSION=$(uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
read -p "Version label [$VERSION]: " VERSION_INPUT
VERSION="${VERSION_INPUT:-$VERSION}"

read -p "Update 'latest' alias to '$VERSION'? (y/n) "
UPDATE_LATEST=$REPLY

read -p "Set '$VERSION' as the default version (root redirect)? (y/n) "
SET_DEFAULT=$REPLY

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

# Ensure the docs remote exists.
if ! git remote get-url "$DOCS_REMOTE_NAME" &>/dev/null; then
    echo "Adding remote '$DOCS_REMOTE_NAME' -> $DOCS_REMOTE_URL"
    git remote add "$DOCS_REMOTE_NAME" "$DOCS_REMOTE_URL"
fi

# Deploy via mike: builds with properdocs internally (mike calls the 'mkdocs'
# entry point, which properdocs registers), commits the versioned site into
# gh-pages on the docs repo, and pushes.
echo "Deploying '$VERSION' to $DOCS_REMOTE_NAME/gh-pages..."
if [[ $UPDATE_LATEST == "y" ]]; then
    uv run mike deploy --push --remote "$DOCS_REMOTE_NAME" \
        -F properdocs.yml --update-aliases "$VERSION" latest
else
    uv run mike deploy --push --remote "$DOCS_REMOTE_NAME" \
        -F properdocs.yml "$VERSION"
fi

if [[ $SET_DEFAULT == "y" ]]; then
    DEFAULT_TARGET="$VERSION"
    if [[ $UPDATE_LATEST == "y" ]]; then
        DEFAULT_TARGET="latest"
    fi
    echo "Setting '$DEFAULT_TARGET' as the default version..."
    uv run mike set-default --push --remote "$DOCS_REMOTE_NAME" \
        -F properdocs.yml "$DEFAULT_TARGET"
fi

echo "Done. Documentation deployed successfully (version '$VERSION')."
