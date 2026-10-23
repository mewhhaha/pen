# Ink adapter: declared save-state migrations

A game using the optional [Ink state adapter](state-adapters.md) can declare its saved types' layouts, version by version, as typed
fields, in `saved.json` beside its asset catalog. When the game restores a
state and has to reset a section because the section's saved version has no
`migrate` in the game's code, the Ink adapter migrates that section by the declaration
and restores again: each field of the build's version takes the lanes of the
saved field of the same name (or the name it `was`), a field the saved
version lacks takes its default, and a field the build dropped goes. The game
then restores the section as its own.

So adding a field to a saved type, removing one, reordering or renaming
fields needs no migration code: a new version in `saved.json` beside the
codec's new `version` keeps saves, quicksaves and a reload's running state. A change of meaning still needs the game's own
`migrate`, which runs first.

## saved.json

```json
{
  "format": "ink-persist-v1",
  "types": {
    "achievements": {
      "key": 119,
      "kind": "resource",
      "versions": {
        "1": [
          {"name": "first_brick", "type": "bool"},
          {"name": "first_building", "type": "bool"},
          {"name": "best", "lanes": 2},
          {"name": "known", "type": "bool"},
          {"name": "asking", "type": "id"},
          {"name": "presence", "type": "u32"}
        ],
        "2": [
          {"name": "first_brick", "type": "bool"},
          {"name": "first_building", "type": "bool"},
          {"name": "best", "lanes": 2},
          {"name": "known", "type": "bool"},
          {"name": "asking", "type": "id"},
          {"name": "presence", "type": "u32"},
          {"name": "streak", "type": "u32", "default": 1}
        ]
      }
    }
  }
}
```

- `format` explicitly selects the Ink adapter; it is required even for
  existing declarations. Without an adapter, Pen leaves state uninterpreted.
- `key` is the codec's key, `kind` how ink's `persist` saves the type:
  `resource`, `snapshotted resource`, `component` or `snapshotted component`.
- Each version lists its fields in lane order. A field's `type` is `f32`,
  `u32`, `bool`, `id` (one lane each), `vec2`, `vec3`, `vec4`, `quat`, `color`
  or `mat4`, or it gives `lanes` (1 to 4096). `default` (a number for every
  lane, or one per lane) is what a migration gives a field the older version
  lacks; zeros without it. `was` names the field in older versions.
- A field must keep its lanes from version to version; `pen check` refuses a
  declaration whose versions cannot all migrate forward.

## What pen does

The adapter understands Ink's `persist.blot` (gdev's README, "Saved state"): a
state is `magic, schema, section_count`, then per saved type `key, version,
width, length` and its lanes. A resource's lanes are one value; a snapshotted
resource's `current_count`, the current value and the previous; a component's
a `flag, lanes...` record per entity, `width` lanes each, absent ones zero;
a snapshotted component's current records, then as many previous ones.

1. The game restores the state, as ever, and says what it made of each
   section (the restore report).
2. If it reset a section (fate 2, no migration in its code) and `saved.json`
   declares both the saved version and the build's, the adapter rewrites that section
   (its version, width, length and every value or record), leaves every other
   section as it was, and restores the result in a new program of the same
   build. Only sections the game could not read move: its own code wins.
3. The report says so: `loaded the save; kept 36 of 37 saved sections;
   migrated section 119 from version 1 to 2 as saved.json declares`.

This runs wherever a state meets a build: starting from a saved state, loads,
undo and redo, rewinds and reloads (`GameRuntime`'s restores, in pen-runtime,
through the selected adapter, so the browser does the same). Native pen reads
`saved.json` from the asset directory at restore/tooling boundaries, so an edit
counts at the next reload; the browser reads it when the page starts. `pen
build` and `pen build --web` ship it with the assets.

## Verified

- `pen_ink::saved`'s tests: declarations, typed fields, renames,
  defaults, all four kinds of section, and that only declared resets move.
- `pen_ink::runtime_tests`: a reload through `GameRuntime::reload` from a
  build saving `points` (version 1) to one saving `points, bonus` (version 2)
  with no migration: declared, the points survive and the bonus takes its
  default, and the reload says so; undeclared, the section resets.
- gdev, compiled by blot2 (a scratch copy, not the checkout): a quicksave of
  the current build, then achievements given a `streak` field as version 2
  without a `migrate`. Loading the old save reset the section; with the
  declaration above it says `migrated section 119 from version 1 to 2 as
  saved.json declares`, and the game's next save holds the old lanes with
  `streak` at its default: `0, 0, 0, 0, 1, 5, 1` became `0, 0, 0, 0, 1, 5, 1, 1`.

## Limits

- The browser reads `saved.json` once, at start: during `pen web`, edit it
  and reload the page.
- Lanes are moved, not converted: a field that changes type (lanes) needs the
  game's `migrate`.
- Scenes (the editor's `main.scene`, scene loads) are not migrated yet: they
  use the same framing under their own magic, and the same rewrite would serve.
