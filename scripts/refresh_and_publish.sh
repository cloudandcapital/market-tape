#!/usr/bin/env bash
set -euo pipefail

target_branch="${1:?usage: refresh_and_publish.sh TARGET_BRANCH}"
max_attempts="${REFRESH_PUSH_ATTEMPTS:-3}"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

rebuild_from_latest() {
  git fetch origin "$target_branch"
  # This is an ephemeral Actions checkout. Resetting the refresh branch to the
  # remote tip ensures a stale scheduled-run SHA never overwrites newer code.
  git checkout --force -B refresh-data "origin/$target_branch"
  python scripts/build_data.py --output-dir data

  if git diff --quiet -- data; then
    echo "No data changes to commit."
    return 1
  fi

  git add -- data
  git commit -m "chore(data): refresh market snapshot [skip data refresh]"
}

rebuild_from_latest || exit 0

for ((attempt = 1; attempt <= max_attempts; attempt++)); do
  if git push origin "HEAD:refs/heads/$target_branch"; then
    exit 0
  fi

  if ((attempt == max_attempts)); then
    echo "Failed to publish refreshed data after $max_attempts attempts." >&2
    exit 1
  fi

  echo "Push raced with a newer commit; rebasing generated data (attempt $((attempt + 1))/$max_attempts)."
  git fetch origin "$target_branch"
  if git rebase "origin/$target_branch"; then
    continue
  fi

  echo "Generated data conflicted with the newer tip; rebuilding from that tip."
  git rebase --abort
  rebuild_from_latest || exit 0
done
