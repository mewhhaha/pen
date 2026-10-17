#!/usr/bin/env bash
# Builds a checkout's package twice, each with caches of its own, and fails
# unless the two are the same file for file and byte for byte, with the same
# modes and modification times: `pen build` is reproducible when the same
# checkout, runtime and options give the same package. SOURCE_DATE_EPOCH
# (the checkout's last commit time, unless set) is the time every file
# takes, so that an archive of either is the same bytes too.
#
#   scripts/reproducible.sh PEN CHECKOUT OUT [BUILD OPTION...]
#
# PEN is a full pen (with the compiler). OUT gets first/ and second/, the
# two packages (or browser exports, given --web), and their caches. Linux
# (GNU find and stat).
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "usage: $0 PEN CHECKOUT OUT [BUILD OPTION...]" >&2
  exit 2
fi
pen=$1
checkout=$2
out=$3
shift 3

if [ -z "${SOURCE_DATE_EPOCH:-}" ]; then
  SOURCE_DATE_EPOCH=$(git -C "$checkout" log -1 --format=%ct 2> /dev/null || echo 0)
fi
export SOURCE_DATE_EPOCH

rm -rf "$out"
mkdir -p "$out"
for build in first second; do
  # Cold caches: nothing compiled for one build serves the other.
  XDG_CACHE_HOME="$out/cache-$build" "$pen" build --project "$checkout" --out "$out/$build" "$@" \
    2> "$out/$build.log" || { cat "$out/$build.log" >&2; exit 1; }
done

# Every file's path, mode, size, time and SHA-256, and every directory's
# path, mode and time.
listing() {
  (
    cd "$1"
    find . -mindepth 1 -type d -printf 'directory %P %m %T@\n' | LC_ALL=C sort
    find . -type f -printf '%P\0' | LC_ALL=C sort -z | while IFS= read -r -d '' file; do
      printf 'file %s %s\n' "$(stat -c '%n %a %s %Y' "$file")" "$(sha256sum < "$file" | cut -d' ' -f1)"
    done
  )
}
listing "$out/first" > "$out/first.listing"
listing "$out/second" > "$out/second.listing"

say() {
  echo "reproducible: $1"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then echo "- reproducible: $1" >> "$GITHUB_STEP_SUMMARY"; fi
}

files=$(grep -c '^file ' "$out/first.listing")
bytes=$(du -sb "$out/first" | cut -f1)
if ! diff -u "$out/first.listing" "$out/second.listing" > "$out/differences.txt"; then
  say "FAILED: two builds of $checkout differ (SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH):" >&2
  grep '^[-+]file\|^[-+]directory' "$out/differences.txt" >&2
  exit 1
fi
summary=$(sha256sum < "$out/first.listing" | cut -c1-16)
say "$files files, $((bytes / 1000000)) MB, the same twice (listing $summary, SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH)"
