#!/usr/bin/env bash
# The Linux package in the Steam Linux Runtime containers Steam runs native
# games in, soldier (Steam Runtime 2, which the default "scout" runtime runs
# in) and sniper (3.0), and on the host. In each container the runtime first
# starts with the container's own libraries only (its glibc and `--version`),
# then plays 40 frames headless with the host's graphics stack, which Steam
# imports too; every run's last frame must be the same image.
#
#   scripts/slr.sh PACKAGE OUT
#
# PACKAGE is `pen build`'s folder for Linux (`just ship`'s dist/linux, whose
# runtime `just player` built against glibc 2.28). OUT gets each run's log,
# frame and data, and pressure-vessel's copies of the runtimes, so nothing
# is written where Steam keeps them. STEAM_RUNTIMES is the directory holding
# SteamLinuxRuntime_soldier and SteamLinuxRuntime_sniper (default Steam's
# steamapps/common): Steam installs them with a game that needs them, or
# `steam steam://install/1391110` (soldier) and 1628350 (sniper). Valve
# also publishes them as archives that need no Steam: SteamLinuxRuntime_sniper.tar.xz
# under https://repo.steampowered.com/steamrt3/images/. This never starts
# Steam. The GPU runs are short; PEN_SOFTWARE=1 asks for a software adapter.
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 PACKAGE OUT" >&2
  exit 2
fi
package=$(cd "$1" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
runtimes=${STEAM_RUNTIMES:-$HOME/.local/share/Steam/steamapps/common}
name=$(sed -n 's/^ *"name": *"\(.*\)",*$/\1/p' "$package/game/game.json" | head -n 1)
exe=$package/$name
if [ -z "$name" ] || [ ! -x "$exe" ]; then
  echo "slr: $package is not a Linux package of pen build (no game/game.json naming an executable beside it)" >&2
  exit 1
fi

# The same headless run everywhere: 40 frames, into play, and the last one.
play() {
  local where=$1
  shift
  rm -rf "$out/$where-data"
  "$@" "$exe" --headless --mute --size 480x320 --frames 40 --script key=Tab \
    --data "$out/$where-data" --screenshot "$out/$where.png" > "$out/$where.log" 2>&1 || {
    cat "$out/$where.log" >&2
    echo "slr: the run on $where failed" >&2
    exit 1
  }
  grep -q '40 frames' "$out/$where.log" || {
    cat "$out/$where.log" >&2
    echo "slr: the run on $where did not play 40 frames" >&2
    exit 1
  }
}

play host
echo "host: $(getconf GNU_LIBC_VERSION), $(grep '^GPU:' "$out/host.log")"
for container in soldier sniper; do
  run=$runtimes/SteamLinuxRuntime_$container/run
  if [ ! -x "$run" ]; then
    echo "slr: no $run; install SteamLinuxRuntime_$container or set STEAM_RUNTIMES" >&2
    exit 1
  fi
  # pressure-vessel copies the runtime into its variable directory: ours.
  export PRESSURE_VESSEL_VARIABLE_DIR=$out/$container-var
  wrap=("$run" --batch --no-gc-runtimes --remove-game-overlay)
  # shellcheck disable=SC2016 # the container's shell expands $1
  "${wrap[@]}" --graphics-provider= -- sh -c 'getconf GNU_LIBC_VERSION; "$1" --version' sh "$exe" \
    > "$out/$container-libraries.log" 2>&1 || {
    cat "$out/$container-libraries.log" >&2
    echo "slr: the runtime did not start in $container with the container's own libraries" >&2
    exit 1
  }
  libraries=$(grep -E '^(glibc|pen) ' "$out/$container-libraries.log" | paste -sd ';' - | sed 's/;/; /')
  play "$container" "${wrap[@]}" --
  echo "$container: alone, $libraries; with the host's graphics, $(grep '^GPU:' "$out/$container.log")"
done
for container in soldier sniper; do
  cmp -s "$out/host.png" "$out/$container.png" || {
    echo "slr: $container's frame differs from the host's ($out/$container.png, $out/host.png)" >&2
    exit 1
  }
done
echo "slr: the host, soldier and sniper drew the same last frame ($out/host.png)"
