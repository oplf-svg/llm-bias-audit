#!/usr/bin/env bash
# One-shot: get this repo onto GitHub as a private repo.
# You (the human) still authorise the browser sign-in - I never touch your credentials.

set -euo pipefail

REPO_NAME="llm-bias-audit"
REPO_DESC="Comparative evaluation of BBQ, StereoSet, CrowS-Pairs on open-source LLMs (LD7236)"

echo "==> 1/6  Checking git + gh CLI are installed"
if ! command -v git >/dev/null || ! command -v gh >/dev/null; then
  if ! command -v brew >/dev/null; then
    echo "Homebrew missing. Install it first: https://brew.sh"
    exit 1
  fi
  echo "     Installing git + gh via Homebrew..."
  brew install git gh
fi

echo ""
echo "==> 2/6  Checking GitHub auth"
if ! gh auth status >/dev/null 2>&1; then
  echo "     Not authenticated. Opening the sign-in flow now."
  echo "     Choose: GitHub.com  ->  HTTPS  ->  Login with a web browser"
  gh auth login
fi
GH_USER="$(gh api user --jq '.login')"
echo "     Signed in as: ${GH_USER}"

echo ""
echo "==> 3/6  Setting git identity if missing"
if [[ -z "$(git config --global user.name || true)" ]]; then
  git config --global user.name "${GH_USER}"
fi
if [[ -z "$(git config --global user.email || true)" ]]; then
  # Use GitHub's noreply email so your real one doesn't leak into commits
  git config --global user.email "${GH_USER}@users.noreply.github.com"
fi
echo "     Name : $(git config --global user.name)"
echo "     Email: $(git config --global user.email)"

echo ""
echo "==> 4/6  Initialising local repo"
if [[ ! -d .git ]]; then
  git init -q -b main
fi

echo ""
echo "==> 5/6  Verifying nothing sensitive is staged"
git add -A
if git diff --cached --name-only | grep -qE '(AUTHOR\.local\.md|\.env|signature.*\.(png|jpg)|.*token.*)'; then
  echo "     !! Sensitive file about to be committed. Aborting."
  git diff --cached --name-only
  exit 1
fi
echo "     Sensitive-file scan: clean"

# First commit
if [[ -z "$(git log --oneline 2>/dev/null || true)" ]]; then
  git commit -q -m "Initial repository scaffolding"
  echo "     First commit created"
else
  # Repo already had commits; add anything new
  git diff --cached --quiet || git commit -q -m "Update scaffold"
fi

echo ""
echo "==> 6/6  Creating GitHub repo and pushing"
if gh repo view "${GH_USER}/${REPO_NAME}" >/dev/null 2>&1; then
  echo "     Repo already exists: ${GH_USER}/${REPO_NAME}"
  git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/${GH_USER}/${REPO_NAME}.git"
  git push -u origin main
else
  gh repo create "${REPO_NAME}" --private --description "${REPO_DESC}" --source=. --push
fi

echo ""
echo "==> DONE"
echo ""
echo "  Repo URL:  https://github.com/${GH_USER}/${REPO_NAME}"
echo "  Clone URL: git@github.com:${GH_USER}/${REPO_NAME}.git"
echo ""
echo "  Next:"
echo "    1. Open the repo URL in your browser and verify AUTHOR.local.md is NOT listed."
echo "    2. Update the Colab badge in README.md by replacing <your-handle> with ${GH_USER}."
echo "    3. Run 'make pilot' to test the pipeline."
