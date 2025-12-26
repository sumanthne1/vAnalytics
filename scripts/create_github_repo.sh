#!/usr/bin/env bash
# Create a GitHub repository via API using a Personal Access Token (PAT).
# Usage:
#   export GITHUB_USERNAME="your-username"
#   export GITHUB_TOKEN="ghp_xxx"        # PAT with "repo" scope (or "contents" + "metadata" for fine-grained)
#   export GITHUB_REPO_NAME="my-new-repo" # name only, no owner prefix
#   ./scripts/create_github_repo.sh
#
# Optional:
#   export GITHUB_PRIVATE="true"   # default: public
#   export GITHUB_DESCRIPTION="Short description"
# For org repos:
#   export GITHUB_ORG="org-name"   # set this to create under an org instead of your user

set -euo pipefail

if [[ -z "${GITHUB_USERNAME:-}" || -z "${GITHUB_TOKEN:-}" || -z "${GITHUB_REPO_NAME:-}" ]]; then
  echo "Error: Set GITHUB_USERNAME, GITHUB_TOKEN, and GITHUB_REPO_NAME env vars before running." >&2
  exit 1
fi

PRIVATE=${GITHUB_PRIVATE:-false}
DESCRIPTION=${GITHUB_DESCRIPTION:-""}

API_URL="https://api.github.com/user/repos"
if [[ -n "${GITHUB_ORG:-}" ]]; then
  API_URL="https://api.github.com/orgs/${GITHUB_ORG}/repos"
fi

payload=$(cat <<EOF
{
  "name": "${GITHUB_REPO_NAME}",
  "private": ${PRIVATE},
  "description": "${DESCRIPTION}"
}
EOF
)

response=$(curl -s -w "%{http_code}" -o /tmp/create_repo_response.json \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -d "${payload}" \
  "${API_URL}")

status="${response: -3}"
if [[ "$status" != "201" && "$status" != "200" ]]; then
  echo "Error: GitHub API returned HTTP ${status}. Details:" >&2
  cat /tmp/create_repo_response.json >&2
  exit 1
fi

html_url=$(jq -r '.html_url' /tmp/create_repo_response.json 2>/dev/null || true)
echo "✅ Repository created: ${html_url}"
