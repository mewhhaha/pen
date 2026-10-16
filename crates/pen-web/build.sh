#!/bin/sh
# Builds the browser host as an ES module and its wasm:
#   OUT/pen_web.js, OUT/pen_web_bg.wasm (and .d.ts typings)
# OUT defaults to crates/pen-web/pkg. The crate builds in the workspace with
# its `web` profile, into target/browser. The wasm-bindgen CLI must match the
# wasm-bindgen version Cargo.lock pins; the first build installs it under
# target/wasm-tools. wasm-opt (binaryen) shrinks the wasm further; without
# it on PATH the build warns and the wasm stays unoptimized.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-$here/pkg}
version=$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/version = |"/, ""); print; exit }' "$root/Cargo.lock")
[ -n "$version" ] || { echo "build.sh: Cargo.lock pins no wasm-bindgen" >&2; exit 1; }
tools="$root/target/wasm-tools"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$root/target/browser}"

rustup target list --installed | grep -qx wasm32-unknown-unknown || rustup target add wasm32-unknown-unknown
cargo build --manifest-path "$root/Cargo.toml" -p pen-web --target wasm32-unknown-unknown --profile web
bindgen="$tools/bin/wasm-bindgen"
if ! "$bindgen" --version 2>/dev/null | grep -q " $version\$"; then
  cargo install wasm-bindgen-cli --version "$version" --locked --root "$tools"
fi
mkdir -p "$out"
"$bindgen" --target web --out-dir "$out" --out-name pen_web \
  "$CARGO_TARGET_DIR/wasm32-unknown-unknown/web/pen_web.wasm"
cp "$here/guest.js" "$out/guest.js"
cp "$here/index.html" "$out/index.html"
# -O1 with the features rustc's wasm32-unknown-unknown enables: -Os and -Oz
# make a smaller module that compresses worse, so it downloads larger.
if command -v wasm-opt >/dev/null; then
  wasm-opt -O1 --enable-bulk-memory --enable-multivalue --enable-mutable-globals \
    --enable-nontrapping-float-to-int --enable-reference-types --enable-sign-ext \
    -o "$out/pen_web_bg.wasm" "$out/pen_web_bg.wasm"
  optimized="wasm-opt -O1"
else
  echo "warning: wasm-opt is not on PATH, so pen_web_bg.wasm is not optimized;" \
    "install binaryen (https://github.com/WebAssembly/binaryen/releases)" >&2
  optimized="NOT optimized: no wasm-opt"
fi
wasm=$(wc -c < "$out/pen_web_bg.wasm")
gzipped=$(gzip -9c "$out/pen_web_bg.wasm" | wc -c)
echo "built $out: pen_web_bg.wasm $wasm bytes ($gzipped gzipped, $optimized), pen_web.js $(wc -c < "$out/pen_web.js") bytes"
