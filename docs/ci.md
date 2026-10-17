# CI

`.github/workflows/ci.yml` runs on every push, pull request and manual run;
a newer push to the same branch or pull request cancels the run before it.
What each job proves, and what it needs:

| Job | Runner | Needs | Proves |
|---|---|---|---|
| `linux` | ubuntu-24.04 | nothing | Builds (with Steam, the player, the compiler component, the browser host), rustfmt, Clippy over pen's feature powerset, the workspace tests and the release renderer tests on lavapipe |
| `deny` | ubuntu-24.04 | nothing | Licenses, advisories and bans of both lockfiles |
| `player` | ubuntu-24.04 | nothing | The shipped player runtimes: Linux against glibc 2.28 and Windows (`x86_64-pc-windows-gnu`), cross-built with zig; uploads `player-runtimes` and their symbols |
| `windows` | windows-latest | nothing | The player builds with MSVC and starts; the runtime, protocol and renderer tests (GPU tests included) pass on WARP, DX12's software adapter; the packaging and app bundle tests pass on Windows |
| `android` | ubuntu-24.04 | nothing | Clippy for `aarch64-linux-android` over the portable crates: pen-protocol, pen-runtime, pen-render and blot-host with and without its compiler ([android.md](android.md)) |
| `game` | ubuntu-24.04 | the sibling checkouts | Protocol freshness, every Cargo test with its fixtures required, the full Blot suite, determinism, the native package and browser export, browser engine conformance (the browser host replays the native run's input and must give the same packets, bit for bit), FXAA witnesses, and the golden images on lavapipe; uploads the image diffs, and for the jobs below the exported game, its pen and the conformance packets |
| `package` | ubuntu-24.04 | `player`, `game` | `pen build --game` packages the game job's bundle for each player runtime without the checkouts; the Linux package plays headless on lavapipe, and its packets equal the checkout's |
| `windows-smoke` | windows-latest | `package` | The shipped Windows package plays 30 frames headless on WARP (the frame is uploaded as `smoke-windows`), and its packets for the conformance script equal Linux's, bit for bit |
| `macos` | macos-14 | `game`, when it ran | pen and the player build on Apple silicon; the runtime, protocol and renderer tests pass on Metal; with the game, an app bundle is packaged, signed (ad hoc without the owner's certificate), notarized with the owner's credentials, played on Metal, its packets compared with Linux's, and uploaded as `app-macos` |

So every run that has the game compares packets three ways: the browser's
against native Linux (`game`), and the Windows and macOS players' against
the same Linux packets (`windows-smoke`, `macos`). A pull request from a
fork gets no secrets, so its game job and everything after it skip.

## What the owner sets

Nothing below can be set from here: the commands are for the owner, with
`gh` logged in to the account that owns the repositories.

### The game job (and package, windows-smoke, macOS packaging)

The game job checks out gdev, blot2 and ink beside pen at fixed commits. Run
this in pen's checkout, with the siblings beside it at the commits CI
should test; each must be pushed:

```sh
for repo in gdev blot2 ink; do
  name=$(echo "$repo" | tr a-z A-Z)
  slug=$(git -C "../$repo" remote get-url origin | sed -E 's#^(git@github\.com:|https://github\.com/)##; s#\.git$##')
  ref=$(git -C "../$repo" rev-parse HEAD)
  [ -n "$(git -C "../$repo" branch -r --contains "$ref")" ] || echo "push ../$repo first: $ref is on no remote branch"
  gh variable set "${name}_REPOSITORY" --body "$slug"
  gh variable set "${name}_REF" --body "$ref"
done
gh secret set CHECKOUTS_TOKEN
```

`CHECKOUTS_TOKEN` is a fine-grained personal access token (GitHub →
Settings → Developer settings → Fine-grained tokens) whose repository
access is only gdev, blot2 and ink, with the permission Contents:
Read-only; `gh secret set` asks for it. On 2026-10-16 blot2's and ink's
checked-out commits were on no remote branch, so they need pushing first.

### macOS

The macOS job runs on main, on tags and on manual runs, since macOS runners
cost ten times Linux's minutes on a private repository; `gh variable set
MACOS_CI --body always` runs it on every push. `MACOS_BUNDLE_ID` names the
app bundle (`pen.gdev` when unset):

```sh
gh variable set MACOS_BUNDLE_ID --body com.example.gdev
```

Signing and notarizing need an Apple Developer Program membership. Export
the Developer ID Application certificate with its key from Keychain Access
as a .p12, and make an App Store Connect API key (Users and Access →
Integrations → Team keys, role Developer) for notarytool:

```sh
base64 -w0 DeveloperID.p12 | gh secret set MACOS_CERTIFICATE
gh secret set MACOS_CERTIFICATE_PASSWORD
gh secret set MACOS_SIGNING_IDENTITY --body "Developer ID Application: NAME (TEAMID)"
base64 -w0 AuthKey_KEYID.p8 | gh secret set MACOS_NOTARY_KEY
gh secret set MACOS_NOTARY_KEY_ID --body KEYID
gh secret set MACOS_NOTARY_ISSUER --body ISSUER-UUID
```

Without them `scripts/macos-sign.sh` signs ad hoc and says so.

## Checking it here

Hosted runs cannot be started from this machine, so the workflow is checked
locally: `actionlint` (with `shellcheck` on its `run:` scripts) must be
clean, and each new step's commands run here where Linux can run them:

| Step | Run here as |
|---|---|
| package | `pen build --game dist/gdev --runtime …/pen.exe --out …` with a `cargo zigbuild` Windows player, and the same for Linux and for `--target x86_64-apple-darwin` (the app bundle's layout) |
| smoke | `scripts/smoke.sh PACKAGE/gdev OUT EXPECTED` against `pen packets --project ../gdev --script-file tests/conformance.steps`, on lavapipe and on the GPU, and with a corrupted packet (it fails) |
| android | the same Clippy with zig standing in for the NDK's clang (`docs/android.md`) |
| game, visual | `PEN_SOFTWARE=1 just visual` with Mesa's lavapipe (`VK_DRIVER_FILES` naming its ICD) |

The Windows and macOS runs themselves, WARP, Metal, signing and
notarization can only run on their runners.
