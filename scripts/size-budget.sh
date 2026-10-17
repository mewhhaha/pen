#!/usr/bin/env bash
# Fails when what ships outgrows its budget: each FILE (or directory, all
# that is in it) must be at most MAX megabytes (10^6 bytes).
#
#   scripts/size-budget.sh FILE MAX [FILE MAX...]
#
# The budgets are the Justfile's (`just size-budget`) and CI's; see
# docs/shipping.md, "Budgets". A build that grows past one is a decision:
# raise the budget in the same change, saying why.
set -euo pipefail

if [ $# -eq 0 ] || [ $(($# % 2)) -ne 0 ]; then
  echo "usage: $0 FILE MAX [FILE MAX...]" >&2
  exit 2
fi

say() {
  echo "size budget: $1"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then echo "- size budget: $1" >> "$GITHUB_STEP_SUMMARY"; fi
}

over=0
while [ $# -gt 0 ]; do
  path=$1
  budget=$2
  shift 2
  if [ ! -e "$path" ]; then
    say "FAILED: $path is not there" >&2
    over=1
    continue
  fi
  bytes=$(du -sb "$path" | cut -f1)
  # Megabytes to two places, and the budget's bytes, in integers.
  size=$(awk -v bytes="$bytes" 'BEGIN { printf "%.2f", bytes / 1e6 }')
  limit=$(awk -v budget="$budget" 'BEGIN { printf "%.0f", budget * 1e6 }')
  if [ "$bytes" -gt "$limit" ]; then
    say "FAILED: $path is $size MB, over its budget of $budget MB" >&2
    over=1
  else
    say "$path is $size MB of its $budget MB"
  fi
done
exit "$over"
