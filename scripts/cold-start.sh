#!/usr/bin/env bash
# How long a package's runtime takes to its first frame, started cold as a
# player's first run is: without pen's caches (no compiled GPU pipelines),
# headless, the frame drawn once every asset it asks for has loaded, and
# the process gone. The median of RUNS runs (5) must be at most BUDGET
# milliseconds; a warm start (the caches the first run left) is measured
# too, for comparison.
#
#   scripts/cold-start.sh EXE BUDGET [RUNS]
#
# EXE is the package's runtime (dist/linux/gdev), which plays the game/
# beside it. PEN_COLD_START_FLAGS adds runtime options (--software on a
# machine without a GPU). Each run's output is kept in
# ${TMPDIR:-/tmp}/pen-cold-start/, with the `startup:` lines of a runtime
# that marks its phases (PEN_STARTUP=1); the first cold run's are shown.
set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "usage: $0 EXE BUDGET [RUNS]" >&2
  exit 2
fi
exe=$1
budget=$2
runs=${3:-5}
read -r -a flags <<< "${PEN_COLD_START_FLAGS:-}"
out=${TMPDIR:-/tmp}/pen-cold-start
rm -rf "$out"
mkdir -p "$out"

say() {
  echo "cold start: $1"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then echo "- cold start: $1" >> "$GITHUB_STEP_SUMMARY"; fi
}

# Milliseconds from starting the runtime to its exit after one frame.
start() {
  local name=$1 cache=$2
  local begun ended
  begun=$(date +%s%N)
  if ! XDG_CACHE_HOME=$cache PEN_STARTUP=1 "$exe" --headless --frames 1 --mute --data "$out/data-$name" \
    "${flags[@]}" > "$out/$name.log" 2>&1; then
    cat "$out/$name.log" >&2
    say "FAILED: $exe did not start" >&2
    exit 1
  fi
  ended=$(date +%s%N)
  echo $(((ended - begun) / 1000000))
}

median() {
  sort -n | awk '{ times[NR] = $1 } END { print (NR % 2) ? times[(NR + 1) / 2] : int((times[NR / 2] + times[NR / 2 + 1]) / 2) }'
}

cold=()
warm=()
for run in $(seq 1 "$runs"); do
  cold+=("$(start "cold-$run" "$out/cache-$run")")
done
for run in $(seq 1 "$runs"); do
  warm+=("$(start "warm-$run" "$out/cache-1")")
done
cold_median=$(printf '%s\n' "${cold[@]}" | median)
warm_median=$(printf '%s\n' "${warm[@]}" | median)
grep -h '^startup:' "$out/cold-1.log" || true
detail="cold ${cold[*]} ms, warm ${warm[*]} ms"
if [ "$cold_median" -gt "$budget" ]; then
  say "FAILED: $exe took $cold_median ms to its first frame cold (median), over its budget of $budget ms ($detail)" >&2
  exit 1
fi
say "$exe: $cold_median ms to its first frame cold, $warm_median ms warm (medians), of a $budget ms budget ($detail)"
