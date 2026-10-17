#!/usr/bin/env bash
# The game against a live Steam client, as the owner's own app: its
# achievements, stats, overlay and Steam Cloud through pen's Steam module
# (PEN_STEAM_DEBUG steps), and the game's own requests through its Steam
# service (PEN_SERVICES_LOG), which must be answered, not `unavailable`.
#
#   scripts/steam-smoke.sh APP_ID [PACKAGE] [--achievement NAME] [--stat NAME] [--frames N] [--dry-run]
#
# APP_ID is the game's own Steamworks app, never 480 (Spacewar, whose
# achievements and cloud belong to everyone testing with it). PACKAGE is
# `pen build`'s Linux, Windows or macOS folder (default dist/linux, from
# `just ship`), copied to target/steam-smoke with steam_appid.txt beside its
# runtime, so the package itself stays as it ships. The Steam client must
# already run, logged in to an account that owns the app; this never starts
# it. It plays in a window, since the overlay hooks a window's swapchain, for
# --frames (900, 15 s), and unlocks and clears --achievement and adds 1 to
# --stat when named. --dry-run checks the arguments and the package and
# prints what it would run.
set -euo pipefail

usage() {
  echo "usage: $0 APP_ID [PACKAGE] [--achievement NAME] [--stat NAME] [--frames N] [--dry-run]" >&2
  exit 2
}
[ $# -ge 1 ] || usage
app=$1
shift
package=dist/linux
achievement=
stat=
frames=900
dry=0
while [ $# -gt 0 ]; do
  case $1 in
    --achievement) [ $# -ge 2 ] || usage; achievement=$2; shift 2 ;;
    --stat) [ $# -ge 2 ] || usage; stat=$2; shift 2 ;;
    --frames) [ $# -ge 2 ] || usage; frames=$2; shift 2 ;;
    --dry-run) dry=1; shift ;;
    -*) usage ;;
    *) package=$1; shift ;;
  esac
done

case $app in
  '' | *[!0-9]*) echo "steam-smoke: the app ID is a number, not $app" >&2; exit 2 ;;
esac
app=$((10#$app))
if [ "$app" -eq 480 ]; then
  echo "steam-smoke: refusing app 480 (Spacewar): pass the game's own app ID" >&2
  exit 2
fi
[ "$app" -gt 0 ] || { echo "steam-smoke: the app ID must be above 0" >&2; exit 2; }
case $frames in '' | *[!0-9]*) usage ;; esac
for name in "$achievement" "$stat"; do
  case $name in *[!A-Za-z0-9_]*) echo "steam-smoke: $name is not a Steamworks API name" >&2; exit 2 ;; esac
done

# The package's runtime: named after the game, beside game/ or in its app bundle.
[ -d "$package" ] || { echo "steam-smoke: no package at $package (just ship makes dist/linux)" >&2; exit 1; }
package=$(cd "$package" && pwd)
game_json=$(find "$package" -maxdepth 5 -path '*game/game.json' 2> /dev/null | head -n 1)
[ -n "$game_json" ] || { echo "steam-smoke: $package has no game/game.json: not a pen build package" >&2; exit 1; }
name=$(sed -n 's/^ *"name": *"\(.*\)",*$/\1/p' "$game_json" | head -n 1)
bin=$(dirname "$(dirname "$game_json")")
if [ -d "$bin/../MacOS" ]; then bin=$(cd "$bin/../MacOS" && pwd); fi
exe=$bin/$name
[ -f "$exe" ] || exe=$exe.exe
[ -f "$exe" ] || { echo "steam-smoke: no runtime named $name in $package" >&2; exit 1; }

steps="list overlay cloud"
if [ -n "$achievement" ]; then steps="$steps unlock=$achievement list clear=$achievement"; fi
if [ -n "$stat" ]; then steps="$steps show=$stat add=$stat=1 show=$stat"; fi
out=target/steam-smoke
copy=$out/$(basename "$package")
relative=${exe#"$package"/}
if [ "$dry" = 1 ]; then
  echo "would copy $package to $copy, write $app to steam_appid.txt beside $relative, and run:"
  echo "  PEN_STEAM=1 PEN_SERVICES_LOG=1 PEN_STEAM_DEBUG=\"$steps\" $copy/$relative --frames $frames --data $out/data"
  exit 0
fi

case $(uname -s) in
  Darwin) client=steam_osx ;;
  MINGW* | MSYS* | CYGWIN*) client=steam.exe ;;
  *) client=steam ;;
esac
if ! { pgrep -x "$client" > /dev/null 2>&1 || tasklist 2> /dev/null | grep -qi '^steam.exe'; }; then
  echo "steam-smoke: the Steam client is not running; start it and log in (this script never starts Steam)" >&2
  exit 1
fi

rm -rf "$out"
mkdir -p "$out"
cp -R "$package" "$copy"
echo "$app" > "$(dirname "$copy/$relative")/steam_appid.txt"
# PEN_STEAM_APP_ID would relaunch through Steam; the app ID file is enough.
unset PEN_STEAM_APP_ID
PEN_STEAM=1 PEN_SERVICES_LOG=1 PEN_STEAM_DEBUG=$steps "$copy/$relative" --frames "$frames" --data "$out/data" \
  > "$out/run.log" 2>&1 || { cat "$out/run.log" >&2; echo "steam-smoke: the game failed" >&2; exit 1; }
grep -E 'Steam|steam' "$out/run.log"

failed=0
check() {
  if grep -qE "$2" "$out/run.log"; then echo "ok: $1"; else echo "FAILED: $1" >&2; failed=1; fi
}
check "Steam started the game as app $app" "^Steam: app $app for "
count=$(wc -w <<< "$steps" | tr -d ' ')
check "every PEN_STEAM_DEBUG step passed" "^PEN_STEAM_DEBUG: $count steps, 0 failed$"
check "the game's Steam service answered" '^service: request [0-9]+ steam .* -> ok '
if grep -qE '^service: request [0-9]+ steam .* -> unavailable' "$out/run.log"; then
  echo "FAILED: the game's Steam service answered unavailable" >&2
  failed=1
fi
[ "$failed" = 0 ] || { echo "steam-smoke: see $out/run.log" >&2; exit 1; }
echo "steam-smoke: achievements, stats, overlay and Steam Cloud work with app $app"
