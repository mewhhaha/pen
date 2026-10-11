#!/bin/sh
# Builds the browser renderer as an ES module and its wasm:
#   OUT/pen_render.js, OUT/pen_render_bg.wasm (and .d.ts typings)
# OUT defaults to crates/pen-render/web/pkg. The wasm-bindgen CLI must match
# the crate's wasm-bindgen version; the first build installs it under
# target/wasm-tools. wasm-opt shrinks the wasm further when it is on PATH.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
out=${1:-$here/pkg}
version=0.2.129
tools="$root/target/wasm-tools"
export CARGO_TARGET_DIR="$root/target"

rustup target list --installed | grep -qx wasm32-unknown-unknown || rustup target add wasm32-unknown-unknown
cargo build --manifest-path "$here/Cargo.toml" --target wasm32-unknown-unknown --release
bindgen="$tools/bin/wasm-bindgen"
if ! "$bindgen" --version 2>/dev/null | grep -q " $version\$"; then
  cargo install wasm-bindgen-cli --version "$version" --locked --root "$tools"
fi
mkdir -p "$out"
"$bindgen" --target web --out-dir "$out" --out-name pen_render \
  "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/pen_render_web.wasm"
if command -v wasm-opt >/dev/null; then
  wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int \
    -o "$out/pen_render_bg.wasm" "$out/pen_render_bg.wasm"
fi
wasm=$(wc -c < "$out/pen_render_bg.wasm")
gzipped=$(gzip -9c "$out/pen_render_bg.wasm" | wc -c)
echo "built $out: pen_render_bg.wasm $wasm bytes ($gzipped gzipped), pen_render.js $(wc -c < "$out/pen_render.js") bytes"
