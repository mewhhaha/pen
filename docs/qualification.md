# Local qualification

Completed on 2026-10-11 within the owner's Linux-only, headless scope.
Hardware: Ryzen 7 3800X, RTX 2070 SUPER, Vulkan, host glibc 2.44.

| Check | Result |
| --- | --- |
| Compiler component | 9 unit + 9 real compiler integration tests passed, none ignored |
| Protocol generator | 4 tests passed, including 2,930 malformed-schema mutations; generated files fresh |
| Native workspace | 113 tests passed; 4 existing exploratory audio/recording helpers ignored |
| Release renderer | 14 tests passed, including required FXAA and shadow GPU witnesses |
| Blot game and panels | 482 tests passed across 55 modules in 78.4 seconds |
| Golden images | All 6 passed; largest RMSE 0.0259%, unchanged baselines |
| Determinism | 831-frame script repeats, overlaps and recording replay match; 460 time-travel frames repeat exactly |
| Browser engine | All 367 original conformance packets match native bit-for-bit; binary transfer completes in 1.69 seconds |
| Browser platforms | IndexedDB snapshots, deterministic pathfinding, service statuses and WebAudio graph passed |
| Native panels | Pointer Pause, keyboard Step, slider drag/arrow adjustment and inspected screenshots passed |
| Browser panels | Visible Timeline/Tunables/Builds and pointer Pause/Play freeze/resume passed |
| Browser development | Transactional module/catalog/asset reload, page-reload saves/preferences and static subdirectory hosting passed |
| Bake | Actual isolated source edit and invalid-range rejection passed; original game defaults preserved |
| Packaging | Successful replacement and failed-build preservation passed; checkout source outputs rejected |
| Clippy | Development, player and browser builds passed with warnings denied |

Linux's final player is 12.63 MB and requires at most `GLIBC_2.28`. Its package
loads the portable x86-64-v2 guest and renders on the host and both installed
Steam Linux Runtime containers, soldier and sniper. All three 40-frame runs
produce identical PNGs. Container-only library probes run with glibc 2.31;
GPU runs import the host graphics stack and glibc 2.44. These checks do not
establish execution on an actual glibc 2.28 installation.

Windows cross-build and package checks passed: 9.64 MB x86-64 PE executable,
matching precompiled guest, Steam library and license notices. Windows
execution, Steam Deck hardware, a live Steam session, audible output and
manual desktop interaction were outside this qualification. No Steam app 480
was started. CI configuration was validated locally; hosted Actions was not
triggered.

[Renderer measurements](render-performance.md) record shadow preparation,
the remaining warm Tab submission cost, FXAA detail tradeoffs, GPU sample
spread and native/browser pixel agreement. The robot's Python recipe preserves
geometry and animation, reduces GPU mesh bytes by 8%, and has five inspected
headless Blender previews. The retired TypeScript/Deno hosts, tests, tools and
capture skill were removed after qualification.

Reproduce with `just test`, `just ship`, and the browser harness commands in
[`pen-web/README.md`](../crates/pen-web/README.md). Generated evidence lives in
`target/fxaa`, `target/browser-conformance`, `target/browser-shots`,
`dist/determinism`, and `~/.cache/pen/qualification`. Retired source snapshots
are archived in `~/.cache/pen/retired-typescript-20261011`.
