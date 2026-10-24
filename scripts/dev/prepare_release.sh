#!/bin/bash
# Prepares a release: bumps the version, points tutorials to the release tag,
# runs checks, optionally executes the tutorials and tests, pushes the dev branch,
# opens (or marks ready) the PR into main, and creates or updates a draft GitHub
# release. Run it on the dev-vVERSION branch after writing the changelog. It is
# safe to rerun, e.g. after editing the changelog.
#
# Publishing the draft release (after merging the PR) triggers the release
# workflow, which pushes to PyPI and Docker Hub.
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

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    git status --short --untracked-files=no
    confirm "Review the changes above (incl. executed notebook outputs). Commit them?" ||
        die "Stopped before committing."
    log "Committing changes"
    git commit -am "Prepare $TAG release"
else
    log "No changes to commit"
fi

log "Pushing $BRANCH to origin"
git push -u origin "$BRANCH"

log "Looking for an open PR from $BRANCH into main"
pr_number="$(gh pr list -R "$REPO" --head "$BRANCH" --base main --state open \
    --json number --jq '.[0].number // empty')"
if [[ -z "$pr_number" ]]; then
    log "Creating PR from $BRANCH into main"
    release_checklist_pr_body |
        gh pr create -R "$REPO" --base main --head "$BRANCH" \
            --title "Version $VERSION" --body-file -
elif [[ "$(gh pr view "$pr_number" -R "$REPO" --json isDraft --jq .isDraft)" == "true" ]]; then
    log "Marking draft PR #$pr_number as ready for review"
    gh pr ready "$pr_number" -R "$REPO"
else
    log "PR #$pr_number is already open and ready for review"
fi

# Release notes: the changelog section for this version followed by GitHub's
# generated notes (list of PRs and new contributors). The notes are generated
# against the dev branch because main does not contain these changes yet.
notes_file="$(mktemp)"
trap 'rm -f "$notes_file"' EXIT
log "Extracting the changelog section for $VERSION"
awk -v heading="## Version $VERSION" '
    $0 == heading { in_section = 1; next }
    in_section && /^## / { exit }
    in_section { print }
' docs/changelog.md >"$notes_file"
log "Generating GitHub release notes for $BRANCH"
gh api "repos/$REPO/releases/generate-notes" \
    -f tag_name="$TAG" -f target_commitish="$BRANCH" --jq .body >>"$notes_file"

log "Looking for an existing release $TAG"
release_is_draft="$(gh release view "$TAG" -R "$REPO" --json isDraft --jq .isDraft \
    2>/dev/null || true)"
if [[ "$release_is_draft" == "false" ]]; then
    die "Release $TAG is already published."
elif [[ "$release_is_draft" == "true" ]]; then
    log "Updating the notes of draft release $TAG"
    gh release edit "$TAG" -R "$REPO" --notes-file "$notes_file"
else
    # The tag does not exist yet. GitHub creates it on main when the draft is
    # published, so the PR must be merged before publishing.
    prerelease_flag=""
    if [[ "$VERSION" =~ [a-z] ]]; then
        prerelease_flag="--prerelease"
    fi
    log "Creating draft release $TAG targeting main"
    gh release create "$TAG" -R "$REPO" --draft --target main \
        --title "Version $VERSION" --notes-file "$notes_file" $prerelease_flag
fi

log "Done. Next steps (see the release checklist):"
log "  1. Publish the docs site: bash scripts/dev/push_doc_site.sh $VERSION"
log "  2. Wait for CI to pass on the PR, then merge it"
log "  3. Edit and publish the draft release; this pushes to PyPI and Docker Hub"
