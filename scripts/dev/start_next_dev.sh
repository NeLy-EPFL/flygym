#!/bin/bash
# Starts the dev branch for the next release after the current one is published:
# creates dev-vVERSION from main, bumps the version, adds a changelog section,
# pushes the branch, and opens a draft PR into main with the release checklist as
# its first comment.
#
# Usage: scripts/dev/start_next_dev.sh VERSION (e.g. 2.1.2)
set -euo pipefail

source "$(dirname "$0")/release_common.sh"
parse_version_arg "$@"
go_to_repo_root
check_prerequisites
require_clean_tree

log "Checking that $BRANCH does not exist locally or on origin"
if git show-ref --verify --quiet "refs/heads/$BRANCH" ||
    git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null; then
    die "Branch $BRANCH already exists."
fi

log "Switching to main"
git checkout main

log "Pulling the latest main from origin"
git pull --ff-only origin main

current_version="$(uv version --short)"
log "Checking that the current version v$current_version has been released"
git ls-remote --exit-code --tags origin "refs/tags/v$current_version" >/dev/null ||
    die "Tag v$current_version not found on origin. Publish that release first."

log "Creating branch $BRANCH"
git checkout -b "$BRANCH"

log "Setting the package version to $VERSION"
uv version "$VERSION"

log "Syncing dependencies (dev, examples, and warp extras)"
uv sync --extra dev --extra examples --extra warp

log "Adding a '## Version $VERSION' section to docs/changelog.md"
perl -0pi -e "s/^# Changelog\n\n/# Changelog\n\n## Version $VERSION\n\n/" \
    docs/changelog.md
grep -qx "## Version $VERSION" docs/changelog.md ||
    die "Could not add the changelog section. Is the first line '# Changelog'?"

log "Checking that only pyproject.toml, uv.lock, and docs/changelog.md changed"
changed_files="$(git status --porcelain --untracked-files=no | awk '{print $2}' |
    sort | tr '\n' ' ')"
[[ "$changed_files" == "docs/changelog.md pyproject.toml uv.lock " ]] ||
    die "Unexpected changes: $changed_files"

log "Committing changes"
git commit -am "Start new branch for v$VERSION"

confirm "Push $BRANCH to origin?" || die "Stopped before pushing."
log "Pushing $BRANCH to origin"
git push -u origin "$BRANCH"

if confirm "Create a draft PR from $BRANCH into main?"; then
    log "Creating draft PR from $BRANCH into main"
    create_release_pr --draft
else
    log "Skipping PR creation"
fi

log "Done. Remember to delete the merged branches manually."
