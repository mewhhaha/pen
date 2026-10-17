#!/usr/bin/env bash
# The game, headless, with its interface on the AT-SPI bus, read and worked
# by an AT-SPI client (probe.py) as a screen reader reads and works it:
# the tree it sees, what it would announce as the focus moves, and its
# actions reaching the game. Run: tests/atspi/run.sh [PEN [GDEV [OUT]]]
# (`just atspi`); it uses the GPU, as any headless run does.
#
# Everything runs on a D-Bus session of its own (dbus-run-session) with its
# own accessibility bus (at-spi-bus-launcher, at-spi2-registryd), its
# runtime directory in a temporary directory and GSettings in memory, with
# no display: nothing reaches the desktop's buses, settings or X server.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)

find_program() {
  for directory in /usr/lib /usr/libexec /usr/lib/at-spi2-core /usr/libexec/at-spi2-core /usr/lib/x86_64-linux-gnu/at-spi2-core; do
    if [ -x "$directory/$1" ]; then echo "$directory/$1"; return; fi
  done
}

if [ "${1:-}" = "--inside" ]; then
  shift
  pen=$1 gdev=$2 out=$3 launcher=$4 registry=$5
  "$launcher" > "$out/launcher.log" 2>&1 &
  launcher_pid=$!
  for _ in $(seq 100); do
    gdbus call --session --dest org.a11y.Bus --object-path /org/a11y/bus --method org.a11y.Bus.GetAddress > /dev/null 2>&1 && break
    sleep 0.05
  done
  # As a screen reader starting turns it on (in GSettings' memory here).
  gdbus call --session --dest org.a11y.Bus --object-path /org/a11y/bus \
    --method org.freedesktop.DBus.Properties.Set org.a11y.Status IsEnabled '<true>' > /dev/null
  "$registry" > "$out/registry.log" 2>&1 &
  registry_pid=$!
  # Six seconds of the game as its bundled scene starts it (a fresh data
  # directory), ten of the pause menu, then the game again (Escape backs out
  # of Settings, then closes the menu); the probe ends the run when done.
  PEN_LOCALE=en PEN_A11Y_ATSPI=1 PEN_A11Y_DUMP="$out/a11y.txt" "$pen" run --project "$gdev" --headless --size 960x640 --data "$out/data" \
    --script "wait=360 key=Escape wait=600 key=Escape wait=10 key=Escape wait=1800 quit" > "$out/pen.log" 2>&1 &
  pen_pid=$!
  status=0
  # libatspi warns of every AccessKit cache event it does not know; keep that out of the way.
  python3 -I "$here/probe.py" --pid "$pen_pid" --out "$out" --expected "$here/expected.txt" 2> "$out/probe.log" || status=$?
  kill "$pen_pid" 2> /dev/null || true
  wait "$pen_pid" 2> /dev/null || true
  kill "$registry_pid" "$launcher_pid" 2> /dev/null || true
  wait 2> /dev/null || true
  exit "$status"
fi

pen=$(realpath "${1:-target/release/pen}")
gdev=$(realpath "${2:-../gdev}")
out=$(realpath -m "${3:-target/atspi}")
launcher=$(find_program at-spi-bus-launcher)
registry=$(find_program at-spi2-registryd)
if [ -z "$launcher" ] || [ -z "$registry" ] || ! command -v dbus-run-session > /dev/null \
  || ! python3 -c 'import gi; gi.require_version("Atspi", "2.0"); from gi.repository import Atspi' 2> /dev/null; then
  echo "atspi: skipped: needs dbus-run-session, at-spi2-core and python-gobject with Atspi" >&2
  exit 0
fi
mkdir -p "$out"
runtime=$(mktemp -d)
trap 'rm -rf "$runtime"' EXIT
env -u DISPLAY -u WAYLAND_DISPLAY -u DBUS_SESSION_BUS_ADDRESS -u AT_SPI_BUS_ADDRESS \
  XDG_RUNTIME_DIR="$runtime" GSETTINGS_BACKEND=memory \
  dbus-run-session -- "$0" --inside "$pen" "$gdev" "$out" "$launcher" "$registry"
