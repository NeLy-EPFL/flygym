# shellcheck shell=bash
# Shared helpers for the release scripts. Source this file; don't run it directly.

REPO="NeLy-EPFL/flygym"

# Main log lines are bold green so they stand out from the output of the tools
# the scripts run. Colors are used only on a terminal and if NO_COLOR is unset.
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    LOG_STYLE=$'\033[1;32m'
    RESET_STYLE=$'\033[0m'
else
    LOG_STYLE=""
    RESET_STYLE=""
fi

log() {
    echo "${LOG_STYLE}[$(basename "$0")] $*${RESET_STYLE}"
}

# For details under a main log line, in regular font.
log_detail() {
    echo "[$(basename "$0")]   $*"
}

die() {
    echo "[$(basename "$0")] ERROR: $*" >&2
    exit 1
}

confirm() {
    local reply
    read -r -p "$1 (y/n) " reply
    [[ $reply == "y" ]]
}

# Validates the VERSION argument (an optional leading "v" is stripped) and sets
# VERSION, TAG, and BRANCH accordingly.
parse_version_arg() {
    if [[ $# -ne 1 ]]; then
        echo "Usage: $0 VERSION (e.g. 2.1.2)" >&2
        exit 2
    fi
    VERSION="${1#v}"
    if [[ ! $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$ ]]; then
        die "Invalid version '$VERSION'. Expected X.Y.Z (without 'v' prefix), optionally with an a/b/rc suffix."
    fi
    # shellcheck disable=SC2034  # Used by the scripts that source this file.
    TAG="v$VERSION"
    # shellcheck disable=SC2034
    BRANCH="dev-v$VERSION"
}

go_to_repo_root() {
    local root
    root="$(git rev-parse --show-toplevel)" || die "Not inside a git repository."
    log "Changing directory to repository root: $root"
    cd "$root" || die "Cannot change directory to $root."
}

# Checks that the tools and permissions needed by the release scripts are there.
check_prerequisites() {
    log "Checking that uv is installed"
    command -v uv >/dev/null || die "uv is not installed."

    log "Checking that the GitHub CLI (gh) is installed"
    command -v gh >/dev/null || die "gh is not installed. See https://cli.github.com."

    log "Checking that gh is logged in to github.com"
    gh auth status --hostname github.com >/dev/null 2>&1 ||
        die "gh is not logged in. Run 'gh auth login'."

    log "Checking that your GitHub account has write access to $REPO"
    [[ "$(gh api "repos/$REPO" --jq .permissions.push)" == "true" ]] ||
        die "Your GitHub account cannot push to $REPO (needed for PRs and releases)."

    log "Checking that git remote 'origin' points to $REPO"
    local origin_url
    origin_url="$(git remote get-url origin)" || die "No git remote named 'origin'."
    # GitHub repo names are case-insensitive.
    local lower_url lower_repo
    lower_url="$(echo "$origin_url" | tr '[:upper:]' '[:lower:]')"
    lower_repo="$(echo "$REPO" | tr '[:upper:]' '[:lower:]')"
    [[ "$lower_url" == *"$lower_repo"* ]] ||
        die "Remote 'origin' is $origin_url, not $REPO."

    log "Checking that git can reach 'origin'"
    git ls-remote --exit-code origin HEAD >/dev/null ||
        die "Cannot reach 'origin'. Check your git credentials (e.g. SSH key)."
}

require_clean_tree() {
    log "Checking that there are no uncommitted changes"
    [[ -z "$(git status --porcelain --untracked-files=no)" ]] ||
        die "There are uncommitted changes. Commit or stash them first."
}
