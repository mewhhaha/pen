#!/bin/sh
# Builds the browser host as an ES module and its wasm:
#   OUT/pen_web.js, OUT/pen_web_bg.wasm (and .d.ts typings, plus OUT/shaders/
#   when deferred-shaders is enabled).
# OUT defaults to crates/pen-web/pkg. Pass:
#   --compact        general static single-player host
#   --3d             slim static 3D host (web-3d)
#   --2d             pure 2D / UI static host (web-2d, no 3D PBR renderer)
#   --features LIST  tailored build with only the named Cargo features
#                    (--no-default-features; pass "" for bare core)
# The crate builds in the workspace with its `web` profile, into target/browser.
# When rust-src is installed on the active toolchain (override via
# PEN_WEB_BUILD_STD=0|1), builds std with -Z build-std=std,panic_abort and
# -Cpanic=immediate-abort to strip panic formatting strings from the wasm.
# The wasm-bindgen CLI must match the wasm-bindgen version Cargo.lock pins; the
# first build installs it under target/wasm-tools. wasm-opt (binaryen) shrinks
# the wasm further; without it on PATH the build warns and the wasm stays
# unoptimized.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-$here/pkg}
ship_structured=1
ship_kernels=1
stage_features="default"
case ${2:-} in
  --compact)
    ship_structured=0
    stage_features="compact"
    set -- --no-default-features --features compact
    ;;
  --3d)
    ship_structured=0
    ship_kernels=0
    stage_features="web-3d"
    set -- --no-default-features --features web-3d
    ;;
  --2d)
    ship_structured=0
    ship_kernels=0
    stage_features="web-2d"
    set -- --no-default-features --features web-2d
    ;;
  --features)
    ship_structured=0
    features=${3:-}
    stage_features="$features"
    case ",$features," in
      *,kernels,*|*,compact,*|*,default,*) ship_kernels=1 ;;
      *) ship_kernels=0 ;;
    esac
    if [ -n "$features" ]; then
      set -- --no-default-features --features "$features"
    else
      set -- --no-default-features
    fi
    ;;
  '') set -- ;;
  *) echo "usage: $0 [OUT [--compact | --3d | --2d | --features LIST]]" >&2; exit 2 ;;
esac
version=$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/version = |"/, ""); print; exit }' "$root/Cargo.lock")
[ -n "$version" ] || { echo "build.sh: Cargo.lock pins no wasm-bindgen" >&2; exit 1; }
tools="$root/target/wasm-tools"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$root/target/browser}"

rustup target list --installed | grep -qx wasm32-unknown-unknown || rustup target add wasm32-unknown-unknown
use_build_std=0
case "${PEN_WEB_BUILD_STD:-auto}" in
  0|false|no) use_build_std=0 ;;
  1|true|yes) use_build_std=1 ;;
  *)
    if rustup component list --installed 2>/dev/null | grep -qx rust-src; then
      use_build_std=1
    fi
    ;;
esac
if [ "$use_build_std" = "1" ]; then
  std_mode="build-std immediate-abort"
  RUSTC_BOOTSTRAP=1 \
    RUSTFLAGS="${RUSTFLAGS:+${RUSTFLAGS} }-Zunstable-options -Cpanic=immediate-abort" \
    cargo build --locked --manifest-path "$root/Cargo.toml" -p pen-web \
    --target wasm32-unknown-unknown --profile web -Z build-std=std,panic_abort "$@"
else
  std_mode="prebuilt std"
  cargo build --locked --manifest-path "$root/Cargo.toml" -p pen-web \
    --target wasm32-unknown-unknown --profile web "$@"
fi
bindgen="$tools/bin/wasm-bindgen"
if ! "$bindgen" --version 2>/dev/null | grep -q " $version\$"; then
  cargo install wasm-bindgen-cli --version "$version" --locked --root "$tools"
fi
mkdir -p "$out"
"$bindgen" --target web --out-dir "$out" --out-name pen_web \
  "$CARGO_TARGET_DIR/wasm32-unknown-unknown/web/pen_web.wasm"
cp "$here/guest.js" "$out/guest.js"
cp "$here/index.html" "$out/index.html"
js_files="$out/pen_web.js $out/guest.js"
if [ "$ship_structured" = "1" ]; then
  cp "$here/structured.js" "$out/structured.js"
  js_files="$js_files $out/structured.js"
else
  rm -f "$out/structured.js"
fi
if [ "$ship_kernels" = "1" ]; then
  cp "$here/kernels.js" "$out/kernels.js"
  cp "$here/kernel_worker.js" "$out/kernel_worker.js"
  js_files="$js_files $out/kernels.js $out/kernel_worker.js"
else
  rm -f "$out/kernels.js" "$out/kernel_worker.js"
fi
if command -v node >/dev/null; then
  # shellcheck disable=SC2086
  node "$here/minify-js.mjs" $js_files
  node "$here/stage-shaders.mjs" "$root/crates/pen-render/src/shaders" "$out" "$stage_features"
else
  rm -rf "$out/shaders"
  case ",$stage_features," in
    *,default,*|*,compact,*|*,web-3d,*|*,deferred-shaders,*)
      echo "warning: node is not on PATH; copying unminified deferred shaders to $out/shaders" >&2
      mkdir -p "$out/shaders"
      cp "$root/crates/pen-render/src/shaders/"*.wgsl "$out/shaders/"
      ;;
  esac
fi
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
shader_note=""
if [ -d "$out/shaders" ]; then
  shader_count=$(find "$out/shaders" -maxdepth 1 -name '*.wgsl' | wc -l)
  shader_note=", $shader_count deferred shaders"
fi
echo "built $out: pen_web_bg.wasm $wasm bytes ($gzipped gzipped, $optimized, $std_mode), pen_web.js $(wc -c < "$out/pen_web.js") bytes$shader_note"
