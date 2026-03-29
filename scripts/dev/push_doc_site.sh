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