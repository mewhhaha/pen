# Accessibility and localization

What the engine does for players with disabilities and for players of other
languages, for every game, and how gdev uses it. The
[accessibility reference](reference.md#accessibility) covers the tree the game
declares and how hosts show it to assistive technology; this is the rest.

## Screen readers

The game's interface reaches assistive technology through AccessKit: AT-SPI on
Linux, UI Automation on Windows, NSAccessibility on macOS
(`crates/pen/src/accessibility.rs`), and the browser's DOM mirror
(`crates/pen-web/src/accessibility.rs`), which mirrors roles (including
`listbox`/`option` promotion), `aria-label`, `aria-description`,
`aria-valuetext`, `aria-checked`, `aria-expanded`, `aria-selected`,
`aria-disabled`, `aria-modal`, `aria-live` (`polite` and `assertive`), range
values on sliders/spinbuttons/progressbars, reading direction (`dir="rtl"` /
`dir="ltr"`), focus preservation across DOM updates, and assistive-technology
action dispatch (`focus`, `blur`, `click`/`activate`/`expand`/`collapse`, and
`keydown` `increment`/`decrement`).

**Headless, on Linux, a screen reader's view is tested for real.**
`PEN_A11Y_ATSPI=1` puts a headless run's tree on the AT-SPI bus with an adapter
of its own (`accessibility/platform.rs`, accesskit_unix), as a window's is, and
keeps the run to the game's pace (a frame each 1/fps of wall time) so that what
assistive technology does reaches the game while the run lasts.
`tests/atspi/run.sh` (`just atspi`) runs the game that way on a D-Bus session of
its own, with its own accessibility bus (at-spi-bus-launcher and
at-spi2-registryd), its runtime directory in a temporary directory, GSettings in
memory and no display, so nothing reaches the desktop's buses, settings or X
server. `tests/atspi/probe.py`, an AT-SPI client (libatspi through
python-gobject, as Orca reads applications), then works the game only through
AT-SPI:

- finds the game's window and writes its tree as AT-SPI shows it
  (`tree-playing.txt`, `tree-paused.txt`, `tree-settings.txt`);
- moves the focus to the station's button (`Component.GrabFocus`), which reaches
  the game as a UiAction, and hears the focus come back;
- presses the editor's toggle twice (`Action.DoAction`) and sees it turn on and
  off;
- waits for the pause menu the script opens: a modal dialog with Continue
  focused; focuses Settings, presses it, focuses the Fog slider and sets it to
  150% (`Value.CurrentValue`), and sees the game take the value;
- writes each focus change as a screen reader announces it (name, role, states,
  value, description) to `announcements.txt`, which must match
  `tests/atspi/expected.txt` line for line.

It skips, saying so, without dbus-run-session, at-spi2-core or python-gobject.
What it cannot show: Orca itself (not installed here; the probe reads what Orca
reads, through the same library, but speaks nothing), NVDA and Narrator on
Windows, and VoiceOver on macOS, which need those systems.

**Snapshots.** `accessibility::snapshots` runs the exported game (`dist/gdev`)
without a GPU through the editor, the pause menu and the settings' Accessibility
page, and compares the tree as AccessKit gets it, and the audit's findings, with
`crates/pen/tests/a11y/*.txt`. A change to the game's interface shows as a diff;
`PEN_UPDATE_A11Y=1` writes the files.

**Testing Library-style role queries.** Both Rust and Blot tests can query and
drive interfaces strictly by ARIA role and accessible name:
- In Rust (`pen_runtime::role_testing` for `&[UiNode]` frames, and
  `pen::accessibility::testing::SessionTester` for live headless `Session`s):
  `screen.by_role("button").name("Play").get()`, `query_by_role`, `get_all_by_role`,
  `get_by_label_text`, `within(container_id)`, state filters (`checked`,
  `checked_state`, `selected`, `expanded`, `disabled`, `focused`, `value_text`),
  `listbox`/`option` promotion, active modal trapping, action builders
  (`click`, `focus`, `blur`, `increment`, `decrement`, `set_value`, `expand`,
  `collapse`), and `assert_audit_clean()`, with formatted accessible tree dumps
  on `NotFound` / `MultipleMatches`.
- In pure Blot (`ink/ui/testing`): matching `by_role`, `get_by_role`,
  `query_by_role`, `within`, `user_click_role`, `user_expand`, `user_set_value`,
  `user_type`, and `audit_tree` for host-independent `pen test-blot` suites.

## The engine's preferences

A colour filter, its strength, a text scale and reduced motion are the engine's
preferences (`pen_runtime::engine_preferences`, ENGINE_PREFERENCE in
`protocol/schema.json`): the player's, the same for every game, kept by the host
(`engine-preferences.json` in the data directory natively, localStorage in the
browser).

| Preference        | Values                                                    | Honored by                           |
| ----------------- | --------------------------------------------------------- | ------------------------------------ |
| `color_filter`    | none, protanopia, deuteranopia, tritanopia, achromatopsia | the renderer, over the whole picture |
| `filter_strength` | 0 to 1                                                    | the renderer                         |
| `text_scale`      | 1 to 2                                                    | the game, where it lays text out     |
| `reduced_motion`  | off or on                                                 | the game, where it moves things      |

They reach the game as Preference events at index `ENGINE_PREFERENCES` (64) and
up, played back after the game's own preferences at startup and after every
restore, so recordings and replays keep them. A game's settings page sets them
through the `preferences` service (`set` preference `a` to `b`); the host
clamps, keeps and applies them at once. For one run,
`PEN_COLOR_FILTER=NAME[:STRENGTH]`, `PEN_TEXT_SCALE=N` and
`PEN_REDUCED_MOTION=1` set them natively; the browser follows
`prefers-reduced-motion` until the player chooses.

**The colour filter** (`pen-render/src/filter.rs`) draws the frame, its sprites
and insets included, into a texture of the output's size, and a last pass copies
it to the output through a 3x3 matrix in linear light. The corrections are
daltonization (Fidaner, Lin and Ozguven) on Machado, Oliveira and Fernandes'
2009 simulations: what the simulation loses is shifted into the channels the
player still tells apart. Achromatopsia shows the picture's luminance.
`PEN_COLOR_FILTER=simulate-deuteranopia` (and the others) shows instead what a
player with that deficiency sees, for checking a game's colours; it is never
kept. With no filter, frames draw straight to the output as before. The GPU test
`pen-render/tests/filter.rs` checks the pass against the CPU's matrix.

**gdev** (`src/access.blot`) keeps them in its state, shows them on the
settings' Accessibility page (colour filter, text size 100 to 200%, reduced
motion), grows the menu's and the HUD's text with the scale as far as the screen
has room, and with reduced motion cuts the menu's flight to the stars (no smear,
shake or slide), stills the turning gnorp and brick, never splits the rock's
colours and never shakes the station's button.

## Languages

The game's words come from string tables (`pen_runtime::locale`): the game's
`src/text.json` is the base, and `src/locales/TAG.json` translates it
(`sv.json`, `pt-BR.json`), keyed as text.json is. A string may have a form for
each of CLDR's plural categories, chosen by the language's cardinal rules for a
count (CLDR 47; when written they were checked against ICU's, through Node's
Intl.PluralRules, on 51,414 sample counts across 82 languages), with `{count}`
filled in as the language writes numbers. The host chooses a table by BCP 47 lookup for `PEN_LOCALE` natively (or `?lang=`
and `navigator.language` in the browser), else the system's locale, and supports
runtime language switching via `LocaleService::set_tag`. Named and positional
placeholders (`{player}`, `{0}`) format safely in a single pass via
`Locale::text_with_args` and `format_args` before BiDi reordering and Arabic
shaping. `pen build --web` and `pen web` ship and serve `text.json`,
`locales/*.json`, and a generated `locales/index.json` discovery manifest (with
hot reload in `pen web`), and `pen check` lints `src/text.json` and
`src/locales/*.json` automatically via `Locales::check` under the `locale` lint.

The game's glyph drawer draws code points left to right, one after another, from
an SDF atlas: it cannot shape or reorder text. So the `locale` service answers
each string laid out to draw: right-to-left runs reordered by the Unicode
Bidirectional Algorithm (unicode-bidi), brackets mirrored, and Arabic joined
into its presentation forms (lam-alef ligatures included), with value 1 when the
text reads right to left. Such an answer also carries, after a 0, the text in
reading order and unshaped: what the game's nodes say to assistive technology,
which must not read the reversed presentation forms the font draws.
`Locales::check` lists what a translator must fix: keys the base lacks, plural
forms missing or never used by the language, placeholders lost or added,
untranslated keys, and glyphs the font lacks after shaping.

**Fonts.** gdev's atlas (`models/font/font.py`, the `font` recipe, which reruns
when a table changes) takes every character the tables need: those Fredoka has,
the rest from fallback fonts in order (subsets of Noto Sans Arabic Bold and Noto
Sans Bold, OFL), Arabic as the whole of Presentation Forms-B. Glyph metrics keep
each font's own baseline.

**gdev** (`src/locale.blot`) asks for the words of its menu, settings, HUD and
editor toolbar once (`text.locale_keys`), keeps them in its state, and shows its
own until they come; it draws the drawn text and names its nodes with the read
text; the HUD's gnorp count is said in the language's plural form, asked again
as it changes. It ships English, Swedish, Russian and Arabic. Packages, bundles,
and browser exports carry the tables beside their assets.

Not yet: speech bubbles, the build bar, the tutorial and the dev tools in gdev
are not yet translated into all locales, and the sample translations are a
developer's, unreviewed.

## The audit

`pen_runtime::audit` checks the game's interface as a frame declares it and
shows it: controls, images and dialogs without a name; focus problems (a modal
dialog with nothing to focus traps the keyboard; the focus outside the open
modal dialog, on a node the tree leaves out, or on two nodes; a focusable
control partly off the screen); text whose contrast with what lies behind it is
below WCAG 2's 4.5:1 (3:1 for large text), measured in the frame's pixels; and
controls smaller than 24 by 24 pixels.

The dev tools' `/audit` (gdev's `devtools/audit.blot`) shows the findings in a
dialog of their own, asked again every second (DEVTOOLS_OP_AUDIT); each finding
is a list item for assistive technology. Natively the host reads back the next
frame after each request, without the dev tools over it, and answers from the
last one read. `PEN_A11Y_AUDIT=FILE` writes the findings whenever the interface
changes, in a window or headless. The browser host answers `unavailable` (it
does not read frames back yet).

Contrast is measured, not declared: the background is the median luminance round
a node's edge, the text the luminance farthest from it that a twentieth of the
pixels inside reach. A node with nothing drawn on it is not measured, so text
drawn in its background's colour goes unreported; what lies over a node (the god
hand, say) counts, as the player sees it.
