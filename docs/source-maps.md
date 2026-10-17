# Guest symbols: function names and source maps

When a guest traps, pen describes where: each frame of its stack by function
and, when the module maps its code to its source, by file, line and column.
Native pen (blot-host, through wasmtime's backtrace) and the browser (pen-web,
from the exception's stack) read the same debug information the same way
(`pen_runtime::symbols`), and describe a trap alike:

```
game program failed: guest backtrace:
    0:   0x1a2b - game.update
                at src/game.blot:42:5
    1:   0x1f00 - program.run
                at ../ink/lib/pen/program.blot:77:3
    2:   0x2e10 - main
: wasm trap: wasm `unreachable` instruction executed
```

Debug information only adds words. A module without it runs the same, its
frames shown by function index; one with unreadable debug information runs
too, and pen says once what it could not read.

## Function names

pen reads the standard Wasm `name` custom section: the module's name
(subsection 0) and its functions' (subsection 1), by function index, imports
first. A function the name section does not name takes its export's name, so
`main` and Blot's `blot:*` exports are named even without one. blot2's
pending name-section change (`/var/tmp/blot2-names.patch`, "Debug names" in
`compiler/guest-abi.md`) emits exactly this section, with names such as
`game.update`, `game.update{closure}` and `blot:arena.allocate`; pen needs
nothing more from it.

`GuestModule::symbols` gives a profiler the same names
(`Symbols::label(index)`).

## Source maps: the format for Blot to adopt

A Blot module maps its code to its source with a standard Source Map
(revision 3) under the convention Wasm toolchains and browser developer tools
already share (Emscripten's `-gsource-map`, Chrome DevTools):

- **Where it lives.** The module's `sourceMappingURL` custom section, whose
  payload is a Wasm name (an unsigned LEB128 byte length, then UTF-8) holding
  the map's URL. pen reads a map the URL embeds, a `data:` URL:
  `data:application/json;base64,<the map's JSON in base64>`. Embedded, the
  map travels inside the module through the compiler plugin's interface
  (which returns one module), hot reload, `pen build` and the browser export,
  and can never fall out of step with the code it describes. A plain URL
  (`game.wasm.map`, a file beside the module) is left to browser developer
  tools; pen says it does not read one.
- **The map.** JSON with `"version": 3`, `"sources"` (the source files, as
  the checkout names them: `src/game.blot`, `../ink/lib/ecs.blot`), an
  optional `"sourceRoot"` prefixed to each, optional `"sourcesContent"` and
  `"names"` (pen reads neither), and `"mappings"`.
- **The mappings.** Wasm code is one generated line: `"mappings"` holds no
  `;`, and a segment's generated column is the byte offset of an instruction
  **from the start of the module file** (not of the code section), the same
  offset wasmtime's backtraces and V8's and Firefox's stacks
  (`wasm-function[12]:0x1a2b`) give. Segments are the usual base64 VLQ fields,
  each relative to the segment before: the offset, then the source index, the
  source line and the source column (zero-based), and optionally a name
  index. A segment of the offset alone says the code from there has no source
  position.
- **Which offsets.** A segment where an instruction's source position
  changes is enough: pen places an offset at the last segment at or before
  it. The positions that matter most are those of calls (every frame below
  the top is at one) and of instructions that can trap (`unreachable`, loads
  and stores, divisions, `call_indirect`).
- **Lines and columns** count from 1 when pen shows them (`src/game.blot:42:5`
  is line index 41, column index 4 in the map).

A minimal map for a module whose instruction at 0x40 came from line 12,
column 5 of `src/game.blot`, and whose code from 0x50 has no position:

```json
{"version": 3, "sources": ["src/game.blot"], "names": [], "mappings": "gEAWI,gB"}
```

## Limits

- **Inlining hides frames.** Every engine but `pen dev`'s inlines calls
  (blot-host's `engine` and `portable_engine`). wasmtime then shows a trap in
  an inlined function at the function it was inlined into, without an offset,
  so without a source position. `pen dev` (no inlining) shows every frame;
  `BLOT_INLINING=0` turns inlining off for any command.
- **Rust and other toolchains** name functions in their own manglings (Rust's
  `_RNv...`); pen shows names as the name section gives them.
- **The browser** reads frames from V8's and Firefox's stack formats; Safari's
  (`wasm-function[NAME]@[wasm code]`) carries no offsets and yields none.

## Tested

`crates/pen-runtime/src/symbols.rs` (names, exports, maps, VLQ, the
backtrace's text, V8 and Firefox stacks), `crates/blot-host`'s
`traps_name_their_functions_and_source_lines` (a hand-built module with a
name section and an embedded map trapping two calls deep) and
`crates/pen-web/tests/guest.mjs` (the stack V8 gives for a trapping JSPI
guest with a name section, which `js_frames`' test copies). `PEN_SYMBOLS_WASM=FILE
cargo test -p pen-runtime symbols_of_a_module -- --ignored --nocapture` lists
a real module's names.
