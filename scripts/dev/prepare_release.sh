#!/bin/bash
# Prepares the release PR: bumps the version, points tutorials to the release tag,
# runs checks, optionally executes the tutorials and tests, pushes the dev branch,
# and opens (or marks ready) the PR into main. Run it on the dev-vVERSION branch
# after writing the changelog. It is safe to rerun, e.g. after editing the
# changelog.
#
# Usage: scripts/dev/prepare_release.sh VERSION (e.g. 2.1.2)
set -euo pipefail

source "$(dirname "$0")/release_common.sh"
parse_version_arg "$@"
go_to_repo_root
check_prerequisites

log "Checking that the current branch is $BRANCH"
[[ "$(git branch --show-current)" == "$BRANCH" ]] ||
    die "Current branch is $(git branch --show-current), not $BRANCH."
require_clean_tree

log "Setting the package version to $VERSION"
uv version "$VERSION"

log "Syncing dependencies (all extras)"
uv sync --all-extras

log "Pointing Colab badges and pip installs in tutorials/ to $TAG"
perl -pi -e "s#(NeLy-EPFL/flygym(?:/blob/|\.git\@))[\w.-]+#\${1}$TAG#gi" \
    tutorials/*.ipynb
# The working tree was clean before, so this lists exactly the notebooks changed.
changed_notebooks="$(git diff --name-only -- tutorials/)"
if [[ -z "$changed_notebooks" ]]; then
    log_detail "No notebooks changed (all already point to $TAG)"
else
    while read -r notebook; do
        log_detail "Updated $notebook"
    done <<<"$changed_notebooks"
fi

log "Checking formatting"
uv run ruff format --check .

log "Linting"
uv run ruff check .

log "Checking version, changelog, and tutorial links"
uv run python scripts/dev/check_release_readiness.py

if confirm "Execute all tutorial notebooks? (slow; some need a GPU)"; then
    log "Executing tutorial notebooks"
    uv run bash scripts/dev/run_notebooks.sh tutorials/*.ipynb ||
        die "Some notebooks failed. See the output above."
else
    log "Skipping notebook execution"
fi

if confirm "Run all tests locally, including the GPU tests that CI skips?"; then
    log "Running tests"
    uv run pytest -n auto
else
    log "Skipping local tests"
fi

# Stage first: `git status` lists executed notebooks as modified, but the
# nbstripout filter removes their outputs on staging, often leaving no change.
log "Staging changes to tracked files"
git add -u
if git diff --cached --quiet; then
    log "No changes to commit"
else
    git diff --cached --stat
    confirm "Commit the changes above?" || die "Stopped before committing."
    log "Committing changes"
    git commit -m "Prepare $TAG release"
fi

confirm "Push $BRANCH to origin?" || die "Stopped before pushing."
log "Pushing $BRANCH to origin"
git push -u origin "$BRANCH"

log "Looking for an open PR from $BRANCH into main"
read -r pr_number pr_is_draft <<<"$(gh pr list -R "$REPO" --head "$BRANCH" \
    --base main --state open --json number,isDraft \
    --jq '.[0] // empty | "\(.number) \(.isDraft)"')"
if [[ -z "$pr_number" ]]; then
    if confirm "No open PR from $BRANCH into main found. Create one?"; then
        log "Creating PR from $BRANCH into main"
        gh pr create -R "$REPO" --base main --head "$BRANCH" \
            --title "Version $VERSION" --body ""
    else
        log "Skipping PR creation"
    fi
elif [[ "$pr_is_draft" == "true" ]]; then
    log "Marking draft PR #$pr_number as ready for review"
    gh pr ready "$pr_number" -R "$REPO"
else
    log "PR #$pr_number is already open and ready for review"
fi

log "Done"
