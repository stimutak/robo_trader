#!/usr/bin/env bash
set -euo pipefail

ZERO_SHA="0000000000000000000000000000000000000000"
OUTPUT_PATH="${CHANGED_PYTHON_OUTPUT:-changed-python.zlist}"
DEFAULT_BRANCH="${DEFAULT_BRANCH:-main}"
HEAD_SHA="$(git rev-parse --verify "${GITHUB_SHA}^{commit}")"
EMPTY_TREE="$(git hash-object -t tree /dev/null)"
BASE_SHA=""
BASE_REASON=""

is_commit() {
    local candidate="${1:-}"
    [[ -n "$candidate" ]] &&
        [[ "$candidate" != "$ZERO_SHA" ]] &&
        git cat-file -e "${candidate}^{commit}" 2>/dev/null
}

use_merge_base() {
    local other="${1:-}"
    local reason="$2"
    local merge_base=""

    if is_commit "$other" &&
        merge_base="$(git merge-base "$HEAD_SHA" "$other" 2>/dev/null)" &&
        is_commit "$merge_base" &&
        [[ "$merge_base" != "$HEAD_SHA" ]]; then
        BASE_SHA="$merge_base"
        BASE_REASON="$reason"
        return 0
    fi
    return 1
}

fetch_branch() {
    local branch="${1:-}"
    [[ -n "$branch" ]] || return 1
    git fetch --no-tags --no-recurse-submodules origin \
        "+refs/heads/${branch}:refs/remotes/origin/${branch}" 2>/dev/null
}

case "${GITHUB_EVENT_NAME:-}" in
    pull_request | pull_request_target)
        if is_commit "${PR_BASE_SHA:-}"; then
            use_merge_base "$PR_BASE_SHA" "pull-request merge base" || true
        else
            fetch_branch "${PR_BASE_REF:-}" || true
            use_merge_base \
                "refs/remotes/origin/${PR_BASE_REF:-}" \
                "fetched pull-request merge base" || true
        fi
        ;;
    push)
        # For force-pushes, the event's before/after trees are the ref change.
        if is_commit "${PUSH_BASE_SHA:-}" &&
            [[ "$PUSH_BASE_SHA" != "$HEAD_SHA" ]]; then
            BASE_SHA="$PUSH_BASE_SHA"
            BASE_REASON="push before SHA"
        fi
        ;;
esac

if [[ -z "$BASE_SHA" ]]; then
    fetch_branch "$DEFAULT_BRANCH" || true
    use_merge_base \
        "refs/remotes/origin/${DEFAULT_BRANCH}" \
        "default-branch merge base" || true
fi

# Ambiguous history must never become an empty HEAD..HEAD success. Checking the
# complete tree is more expensive, but it is the only fail-closed fallback.
if [[ -z "$BASE_SHA" ]]; then
    BASE_SHA="$EMPTY_TREE"
    BASE_REASON="fail-closed empty-tree fallback"
fi

echo "Changed-file base: $BASE_SHA ($BASE_REASON)"

output_tmp="$(mktemp "${OUTPUT_PATH}.tmp.XXXXXX")"
trap 'rm -f "$output_tmp"' EXIT
git diff --name-only -z --diff-filter=ACMR \
    "$BASE_SHA" "$HEAD_SHA" -- '*.py' >"$output_tmp"
mv "$output_tmp" "$OUTPUT_PATH"
trap - EXIT

echo "Changed Python files:"
while IFS= read -r -d '' path; do
    printf '  %q\n' "$path"
done <"$OUTPUT_PATH"
