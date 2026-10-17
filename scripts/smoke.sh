#!/usr/bin/env bash
# A package's smoke test, the same on Linux, Windows (Git Bash) and macOS:
# its runtime plays a few frames of its game headless on a software adapter
# (lavapipe, WARP; PEN_SMOKE_GPU=1 takes the GPU instead) and saves the
# last one, then writes each frame's packets for tests/conformance.steps,
# which must equal EXPECTED's (another platform's, or `pen packets` of the
# checkout) bit for bit when given.
#
#   scripts/smoke.sh EXE OUT [EXPECTED]
#
# EXE is the package's runtime: dist/linux/gdev, dist/windows/gdev.exe or
# gdev.app/Contents/MacOS/gdev. OUT gets frame.png, run.log, run.trace,
# packets/ and packets.log, and the run's data directory, where a crash
# report fails the smoke. PEN_SMOKE_FRAMES sets the frames (30).
set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "usage: $0 EXE OUT [EXPECTED]" >&2
  exit 2
fi
exe=$1
out=$2
expected=${3:-}
steps="$(cd "$(dirname "$0")/.." && pwd)/tests/conformance.steps"
frames=${PEN_SMOKE_FRAMES:-30}
software=--software
if [ "${PEN_SMOKE_GPU:-0}" = 1 ]; then software=; fi

directory=$(dirname "$exe")
game=$directory/game
[ -f "$game/game.wasm" ] || game=$directory/../Resources/game
if [ ! -f "$game/game.wasm" ]; then
  echo "smoke: no game/ beside $exe (nor in its application bundle)" >&2
  exit 1
fi
rm -rf "$out"
mkdir -p "$out"

# Says how it went, in a CI run's summary too.
say() {
  echo "smoke: $1"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then echo "- $1" >> "$GITHUB_STEP_SUMMARY"; fi
}

# Says what went wrong, with the log, and fails.
fail() {
  say "FAILED: $1" >&2
  [ -f "${2:-}" ] && cat "$2" >&2
  exit 1
}

# shellcheck disable=SC2086 # $software is one word or none
"$exe" --headless $software --mute --frames "$frames" --size 960x640 \
  --data "$out/data" --screenshot "$out/frame.png" --trace "$out/run.trace" > "$out/run.log" 2>&1 ||
  fail "$exe did not play $frames frames" "$out/run.log"
cat "$out/run.log"
if [ -d "$out/data/crashes" ] && [ -n "$(ls -A "$out/data/crashes")" ]; then
  fail "it wrote a crash report" "$(find "$out/data/crashes" -type f | head -n 1)"
fi
[ -s "$out/frame.png" ] || fail "it saved no frame" "$out/run.log"
played=$(wc -l < "$out/run.trace" | tr -d ' ')
[ "$played" = "$frames" ] || fail "its trace has $played frames, not $frames" "$out/run.log"
say "$(basename "$exe") played $frames frames headless ($(sed -n 's/^GPU: //p' "$out/run.log")), the last in $out/frame.png"

"$exe" packets "$out/packets" --game "$game" --script-file "$steps" > "$out/packets.log" 2>&1 ||
  fail "$exe did not write the conformance packets" "$out/packets.log"
cat "$out/packets.log"
if [ -n "$expected" ]; then
  # Each frame's input and render packet, and the schema; diff names the
  # files that differ, -q keeps binary packets off the log.
  for part in input output schema.txt; do
    diff -rq "$expected/$part" "$out/packets/$part" || fail "its packets ($part) differ from $expected's"
  done
  count=$(find "$out/packets/output" -name '*.f32' | wc -l | tr -d ' ')
  say "its $count frames of input and render packets equal $expected's, bit for bit"
fi
