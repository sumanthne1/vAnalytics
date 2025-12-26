#!/usr/bin/env bash
# Upload the current vAnalytics workspace to GitHub using a Personal Access Token (PAT).
# Usage:
#   export GITHUB_USERNAME="your-username"
#   export GITHUB_TOKEN="ghp_xxx"        # PAT with "repo" scope
#   export GITHUB_REPO="username/repo"   # existing GitHub repo name (owner/name)
#   ./scripts/upload_to_github.sh

set -euo pipefail

if [[ -z "${GITHUB_USERNAME:-}" || -z "${GITHUB_TOKEN:-}" || -z "${GITHUB_REPO:-}" ]]; then
  echo "Error: Set GITHUB_USERNAME, GITHUB_TOKEN, and GITHUB_REPO env vars before running." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is not installed." >&2
  exit 1
fi

# Ensure we are in the repo root (script lives in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Initialize git if needed
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init
fi

# Use main as default branch
git symbolic-ref HEAD refs/heads/main >/dev/null 2>&1 || true

# Basic git config if missing
git config user.name >/dev/null 2>&1 || git config user.name "$GITHUB_USERNAME"
git config user.email >/dev/null 2>&1 || git config user.email "$GITHUB_USERNAME@users.noreply.github.com"

# Create a temporary askpass helper so the token isn't stored in git remote URL
ASKPASS="$(mktemp)"
cat >"$ASKPASS" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  Username*) echo "${GITHUB_USERNAME}";;
  Password*) echo "${GITHUB_TOKEN}";;
esac
EOF
chmod +x "$ASKPASS"
export GIT_ASKPASS="$ASKPASS"

# Stage and commit everything
git add -A
if ! git diff --cached --quiet; then
  git commit -m "Initial upload"
fi

# Set or update origin without embedding the token
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "https://github.com/${GITHUB_REPO}.git"
else
  git remote add origin "https://github.com/${GITHUB_REPO}.git"
fi

# Push to main
git push -u origin main

# Cleanup
rm -f "$ASKPASS"
unset GIT_ASKPASS
