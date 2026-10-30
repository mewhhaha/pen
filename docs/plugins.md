# Plugins and their API

What is specific to a game's language or toolchain lives in a Wasm plugin, not
in pen: a WebAssembly component that a checkout's `pen.json` chooses and grants
what it may read and run. pen's own Blot plugin (`plugins/blot`) compiles Blot
through blot2's native compiler, keeps it open between builds so a reload
recompiles only what changed, and lays out the checkout's Blot tests. The
[plugin reference](reference.md#wasm-plugins) has the `pen.json` settings, the
grants and the limits a plugin runs under.

A plugin builds against `wit/pen-plugin.wit` in pen's repository, the WIT
package `pen:plugin`, and names its version in every interface it imports from
pen or exports to it. pen reads those names before linking a plugin and refuses
one built for a version it does not host, saying which:

```
loading plugin blot from plugins/blot.wasm: plugin blot was built for pen:plugin@0.1.0, and this pen hosts
0.2.0: rebuild it against wit/pen-plugin.wit (wit/CHANGELOG.md says what changed)
```

The changelog, below, is the WIT's record: each version, what changed in it, and
the shape of its interfaces, which a test holds the WIT to.

{{#include ../wit/CHANGELOG.md:3:}}
