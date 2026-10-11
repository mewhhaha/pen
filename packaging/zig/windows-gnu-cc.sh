#!/bin/sh
# zig cc for x86_64-windows-gnu (see windows-gnu.cmake), minus what zig's
# driver rejects: MSVC's /guard:cf, and the cmake crate's Rust-style
# --target=x86_64-pc-windows-gnu (zig spells it x86_64-windows-gnu).
for arg; do
  shift
  case "$arg" in
    /guard:cf | --target=*) ;;
    *) set -- "$@" "$arg" ;;
  esac
done
exec "${ZIG:-zig}" cc -target x86_64-windows-gnu "$@"
