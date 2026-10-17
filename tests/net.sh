#!/bin/sh
# Net sessions and spectating, end to end through the binary, headless:
#
# 1. two peers of a net session over UDP on the loopback, each with its own
#    script: the final line of every frame of their traces must agree, and
#    the host's input recording must replay to the same trace;
# 2. a broadcast and a viewer that joins it late: every frame the viewer
#    plays must give the session's packets.
#
# Usage: tests/net.sh PEN GAME OUT, e.g.
#   tests/net.sh target/release/pen dist/gdev target/net
# GAME is a bundle that declares two players (`pen export` of gdev). It uses
# the GPU: run it alone. Ports 7461, 7462 and 7471 must be free.
set -eu
pen=$(realpath "$1")
game=$(realpath "$2")
out=$3
rm -rf "$out" && mkdir -p "$out" && cd "$out"
last() { awk '{line[$1]=$0} END {for (f in line) print line[f]}' "$1" | sort -n; }
fail() { echo "net: $*" >&2; exit 1; }

# 1. A net session.
peers=127.0.0.1:7461,127.0.0.1:7462
first="wait=10 key=Tab wait=30 click=470,150 wait=12 click=470,152 wait=12 click=471,150 wait=30 drag=300,500,600,450 wait=40"
second="wait=50 move=480,330 wait=7 click=470,151 wait=9 move=520,300 wait=5 click=469,150 wait=20 wheel=-200,480,320 wait=15 move=200,200 wait=40"
mkdir a b
"$pen" --game "$game" --headless --mute --frames 300 --net "0:$peers" --script "$first" \
  --trace a.trace --data a --shots a --screenshot a/last.png --record-input a/session.penrec > a.log 2>&1 &
host=$!
"$pen" --game "$game" --headless --mute --frames 300 --net "1:$peers" --script "$second" \
  --trace b.trace --data b --shots b --screenshot b/last.png > b.log 2>&1 &
guest=$!
wait $host || fail "the host failed: $(tail -3 a.log)"
wait $guest || fail "the guest failed: $(tail -3 b.log)"
last a.trace > a.final
last b.trace > b.final
[ "$(wc -l < a.final)" -eq 300 ] || fail "the host ran $(wc -l < a.final) frames, not 300"
cmp -s a.final b.final || fail "the peers' frames differ: $(diff a.final b.final | head -3)"
cmp -s a/last.png b/last.png || fail "the peers' last frames differ"
grep -h "^net:" a.log b.log | grep -v waiting
"$pen" --game "$game" --headless --mute --replay a/session.penrec --trace replay.trace --data r --shots r > replay.log 2>&1 \
  || fail "the replay failed: $(tail -3 replay.log)"
last replay.trace > replay.final
cmp -s replay.final a.final || fail "the session's recording replays other frames"
echo "net: two peers agree on 300 frames, and the recording replays them"

# 2. A broadcast and a late viewer.
steps="wait=10 key=Tab wait=20 console=3,6 wait=30 click=470,150 wait=15 click=470,152 wait=40 move=480,330 wait=200"
mkdir s v
"$pen" --game "$game" --headless --mute --frames 900 --script "$steps" --broadcast 127.0.0.1:7471 \
  --trace s.trace --data s --shots s > s.log 2>&1 &
session=$!
until [ -f s.trace ] && [ "$(wc -l < s.trace)" -ge 300 ]; do sleep 0.05; done
"$pen" --game "$game" --headless --mute --spectate 127.0.0.1:7471 --trace v.trace --data v --shots v > v.log 2>&1 &
viewer=$!
wait $session || fail "the session failed: $(tail -3 s.log)"
wait $viewer || fail "the viewer failed: $(tail -3 v.log)"
last s.trace > s.final
last v.trace > v.final
from=$(head -1 v.final | cut -d' ' -f1)
awk -v from="$from" '$1 >= from' s.final > s.tail
[ "$(wc -l < v.final)" -gt 100 ] || fail "the viewer played $(wc -l < v.final) frames"
cmp -s s.tail v.final || fail "the viewer's frames differ: $(diff s.tail v.final | head -3)"
echo "net: a viewer that joined at frame $from played the session's $(wc -l < v.final) frames"
