#!/bin/bash
# Creates (or updates) the draft GitHub release after the release PR is merged.
# Its notes are the changelog section for this version followed by GitHub's
# generated notes (list of PRs and new contributors). GitHub creates the tag on
# main when the draft is published, which triggers the release workflow. It is
# safe to rerun, e.g. after editing the changelog.
#
# Usage: scripts/dev/draft_release.sh VERSION (e.g. 2.1.2)
set -euo pipefail

source "$(dirname "$0")/release_common.sh"
parse_version_arg "$@"
go_to_repo_root
check_prerequisites

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

log "Fetching main from origin"
git fetch origin main

log "Checking that the version on main is $VERSION"
main_version="$(git show origin/main:pyproject.toml |
    awk -F'"' '/^version = / && !found { print $2; found = 1 }')"
[[ "$main_version" == "$VERSION" ]] ||
    die "The version on main is '$main_version', not $VERSION."

notes_file="$tmp_dir/notes.md"
log "Extracting the changelog section for $VERSION from main"
git show origin/main:docs/changelog.md >"$tmp_dir/changelog.md"
# Ignore trailing whitespace (incl. CR) when matching the heading.
awk -v heading="## Version $VERSION" '
    { line = $0; sub(/[[:space:]]+$/, "", line) }
    line == heading { in_section = 1; next }
    in_section && /^## / { exit }
    in_section { print }
' "$tmp_dir/changelog.md" >"$notes_file"
grep -q '[^[:space:]]' "$notes_file" ||
    die "The changelog on main has no (or an empty) '## Version $VERSION' section."

log "Generating GitHub release notes for main"
gh api "repos/$REPO/releases/generate-notes" \
    -f tag_name="$TAG" -f target_commitish=main --jq .body >>"$notes_file"

log "Looking for an existing release $TAG"
if release_is_draft="$(gh release view "$TAG" -R "$REPO" --json isDraft \
    --jq .isDraft 2>"$tmp_dir/gh_error")"; then
    :
elif grep -q "release not found" "$tmp_dir/gh_error"; then
    release_is_draft=""
else
    die "Could not look up release $TAG: $(cat "$tmp_dir/gh_error")"
fi

if [[ "$release_is_draft" == "false" ]]; then
    die "Release $TAG is already published."
elif [[ "$release_is_draft" == "true" ]]; then
    log "Updating the notes of draft release $TAG"
    gh release edit "$TAG" -R "$REPO" --notes-file "$notes_file"
elif confirm "No release $TAG found. Create a draft release?"; then
    prerelease_flag=""
    if [[ "$VERSION" =~ [a-z] ]]; then
        prerelease_flag="--prerelease"
    fi
    log "Creating draft release $TAG targeting main"
    gh release create "$TAG" -R "$REPO" --draft --target main \
        --title "Version $VERSION" --notes-file "$notes_file" $prerelease_flag
else
    log "Skipping draft release creation"
fi

log "Done"
