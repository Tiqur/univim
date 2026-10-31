# UniVim

A native Rust/OpenCV port of the hierarchical screenshot navigation detector in
`python.py`. It detects controls, text, thumbnails, icon-label compounds and
paragraphs, generates Vimium-style hints, and keeps children for instant drill-down.
The Rust executable does not invoke Python. The original Python file is unchanged.

## Development

The pinned flake provides Rust, rustfmt, Clippy, Python, NumPy and the **same
OpenCV 4.13.0** for both implementations, plus native build dependencies.

```sh
nix develop
cargo build --release
target/release/univim screenshot.png --json targets.json -o overlay.png
target/release/univim screenshot.png --turbo --interactive-only -o controls.png
target/release/univim screenshot.png --drill-down S -o spotlight.png
```

Additional Rust options: `--no-collapse-paragraphs`, `--min-score FLOAT`,
`--benchmark RUNS`, `--warmup RUNS`, and `--benchmark-json PATH`.
Unknown drill-down hints fall back to the normal overlay, like the reference.
The library exposes `detect`, `Options`, `Target`, `hints`, `hints_with_alphabet`, `drill_down`, and
`render::{overlay, drilldown_overlay}`. Detection expects a nonempty BGR `CV_8UC3`
image. The CLI sets OpenCV to one thread; library callers can choose their own
OpenCV thread policy. `drill_down` returns a `Result`; externally constructed
children must follow the detector's JSON schema (integer coordinates and numeric
scores), rather than Python's permissive string/float coercions.

```sh
nix build
./result/bin/univim screenshot.png --json targets.json
```

The package builds both `univim` and `univim-desktop`. The flake exposes only
`x86_64-linux` and `aarch64-linux`; only x86_64 Linux has been tested. Although the
desktop binary has a non-Linux error entry point, its Wayland dependencies are
unconditional in Cargo, so that guard does not establish macOS build support.

## Desktop Navigation

`univim-desktop` is a native Rust desktop service for Linux/Hyprland, not a
Python wrapper. It uses native Wayland screencopy into memory, a software-rendered
layer-shell overlay, keyboard input and virtual-pointer click injection. It does
not invoke screenshot tools or write screenshot files. While idle it maps no
surface and does not capture the screen.

Run inside your Hyprland session with `WAYLAND_DISPLAY`, `XDG_RUNTIME_DIR` and
`HYPRLAND_INSTANCE_SIGNATURE` available. The compositor must expose wlr
screencopy v3, layer shell, shared-memory buffers, a keyboard seat, wlr virtual
pointer v2, Hyprland global shortcuts and lock notification. This is not a
generic Wayland or portal backend. The dev shell includes Wayland,
libxkbcommon and `wayland-info` (from wayland-utils).

```sh
univim-desktop check
univim-desktop bindings
univim-desktop serve --no-click
# In another terminal in the same session:
univim-desktop hints
univim-desktop status
univim-desktop cancel
univim-desktop grid
univim-desktop quit
```

`check` reports capabilities and discovered outputs without capturing or injecting
input. `serve [--output NAME] [--turbo] [--interactive-only] [--no-click]
[--timeout 30]` runs the service. Output discovery is dynamic: each activation
uses the currently focused monitor unless `--output NAME` selects a connector.
Monitor position, scale and transform come from runtime metadata, not a fixed
layout. Unavailable outputs fail rather than falling back to another monitor.
`--turbo` and `--interactive-only` configure detection. `--no-click` allows
selection without injecting clicks; it still captures and takes overlay keyboard
focus during an active session. The session timeout defaults to 30 seconds
(accepted range 1-3600). `cancel` dismisses selection; `quit` stops the service.
`status` reports JSON including activity, mode, output and overlay state.

In hints mode, type a label to select its target and click its center. Labels
filter as you type without being reassigned. **Tab** toggles drill mode: selecting
a target with children expands only its immediate children and assigns them
labels; selecting a leaf still clicks. **Backspace** edits the prefix, or returns
to the parent when the prefix is empty. **Left** returns to the parent and clears
the prefix. **Escape** cancels. Grid mode first shows a **26 x 26** grid: type the
row letter then column letter (A-Z), then select a **6 x 6** subcell using
`ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789` in row-major order. The third key clicks
the subcell center. Backspace with an empty prefix or Left returns to the main
grid.

### Opt-In Hotkeys

`univim-desktop bindings` prints an `exec-once` service command and anonymous,
nonconsuming Hyprland `bindn` global bindings for left/right Shift and Ctrl.
Nothing installs these bindings or modifies your configuration automatically.
Review the printed configuration and opt in through your own Hyprland config;
do not start a second service if one is already running.

Two short taps of the **same Shift key** request hints; two taps of the **same
Ctrl key** request the grid. Each hold must be at most 250 ms and releases must
be within 300 ms. Left and right modifiers are tracked separately.
**These are best-effort gestures, not robust chord-free double-tap detection.**
Anonymous global shortcut notifications do not reveal arbitrary intervening
ordinary keys, so the service cannot reliably reject modifier use in chords;
normal typing can accidentally activate it. Nonconsuming bindings preserve
ordinary modifier events, but do not solve this limitation. The generated config
also includes commented explicit `SUPER+F` / `SUPER+G` alternatives invoking
`hints` / `grid`; use explicit activation instead of modifier bindings when
accidental activation is unacceptable.

Control IPC uses a private per-display directory under `XDG_RUNTIME_DIR`.
If a crash leaves `control.sock`, startup fails closed: verify the old service is
stopped before removing the stale socket named in the error. Screen captures stay
in memory, but an active service has screen-capture and input-injection privileges;
only run it in a trusted desktop session.

## Verification

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release
python3 tests/parity.py
python3 tests/parity.py --benchmark --runs 30 --warmup 5
```

Before files are staged, use `nix develop path:. --command COMMAND` so new files
are visible to Nix's source snapshot. After staging, ordinary `nix develop` works.
The shell and package share a source filter excluding `target`, `result`, `.git`
and Python caches. This keeps build inputs small, but a `path:.` flake may still
copy the working directory before that filter is evaluated. Prefer the ordinary
Git-backed flake once new source files are tracked; `.gitignore` alone cannot
prevent the initial `path:.` snapshot.

The full differential suite passed **465 exact detection comparisons**, **465
hint checks**, and **488 pixel-exact overlay comparisons**, including **23
drill-down cases**. It uses 18 deterministic synthetic fixtures, with light/dark
UI, paragraphs, blue links, controls, textures, noise, dense hints, and widths
640 through 2880, including resize boundaries. All boolean option combinations
and score boundaries are exercised. It checks list ordering, all coordinates,
scores, kinds, levels, hints, nested children, and missing versus empty child keys.
Timing fields are intentionally excluded. Nine detector unit tests additionally
cover convolution alignment, runs, scaling, deduplication, hints and both
drill-down modes. There are no numeric tolerances in detection/image comparisons.

These fixtures demonstrate exact agreement on the tested inputs, **not proof of
equivalence for every real-world screenshot**. No real screenshot corpus was
provided. Python remains the oracle for future differential tests.

Generated fixtures, per-case JSON/images, diagnostics and coverage are under
`target/parity/`. Raw timing samples, binary/reference hashes, machine details and
tool versions are retained locally in `target/benchmark.json`. These artifacts are
ignored by Git and can be regenerated with the commands above.

## Benchmark

Detection only, preloaded images, release Rust build, one OpenCV thread, 5 warmups
and 30 measured calls per implementation/configuration. Both implementations
reuse their process; image decoding, process startup and overlay rendering are
excluded. All runs include text and paragraph collapse at the default score.
Results below are medians in milliseconds; speedup is Python time / Rust time.
The input image is decoded once before warmup, matching an app that preloads a
screenshot before the user requests detection.

Machine: AMD Ryzen 7 7700X, Linux x86_64, Rust 1.99.0, Python 3.14.8,
OpenCV 4.13.0, NumPy 2.5.3. Results vary with system load and image content.

| Synthetic Workload | Resolution | Mode | Python | Rust | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| Light UI | 2561 x 1601 | Normal | 62.69 | 68.88 | 0.91x |
| Light UI | 2561 x 1601 | Turbo | 24.91 | 19.45 | 1.28x |
| Light UI | 1440 x 900 | Normal | 14.41 | 12.06 | 1.19x |
| Light UI | 1440 x 900 | Turbo | 14.51 | 8.15 | 1.78x |
| Dense controls | 1440 x 900 | Normal | 62.67 | 13.64 | 4.59x |
| Dense controls | 1440 x 900 | Turbo | 62.28 | 13.00 | 4.79x |

Per-stage medians use the reference's boundaries: `signal` covers scaling,
gradients and structural barriers; `frames` covers frame/thumbnail recovery and
barrier dilation; `coarse` covers coarse proposals; `recursive` covers recursive
refinement, composition, containment cleanup and final hints. Values are
Python / Rust milliseconds.

| Workload | Mode | Signal | Frames | Coarse | Recursive |
| --- | --- | ---: | ---: | ---: | ---: |
| Light UI 2561 x 1601 | Normal | 39.2 / 40.0 | 8.6 / 14.3 | 3.2 / 8.7 | 10.4 / 1.3 |
| Light UI 2561 x 1601 | Turbo | 17.0 / 16.5 | 2.7 / 1.8 | 0.8 / 0.7 | 4.2 / 0.4 |
| Light UI 1440 x 900 | Normal | 4.0 / 3.5 | 4.6 / 4.7 | 0.9 / 2.7 | 4.7 / 0.5 |
| Light UI 1440 x 900 | Turbo | 4.1 / 3.5 | 4.7 / 3.6 | 0.6 / 0.5 | 4.8 / 0.5 |
| Dense controls 1440 x 900 | Normal | 3.8 / 3.3 | 3.7 / 5.8 | 1.2 / 2.8 | 53.7 / 1.2 |
| Dense controls 1440 x 900 | Turbo | 3.8 / 3.5 | 3.6 / 5.9 | 0.9 / 1.7 | 53.6 / 1.4 |

The large normal-mode regression is localized: signal processing is essentially
equal and recursive/composition work is much faster in Rust, while `frames` and
`coarse` are slower. Those stages repeatedly allocate and inspect full-resolution
masks and contour ROIs, so large-image memory traffic, allocation and cache
behavior are the likely bottleneck rather than the core gradient operations.
The effect is not uniform across sizes; the 1440 x 900 normal case is faster in
Rust. Turbo changes the algorithm's working resolution/coarse pass, so compare
implementations within the same mode rather than normal versus turbo. Stage
medians may not sum exactly to wall-clock totals because they are measured at
internal boundaries.

See `AGENTS.md` for architecture, parity pitfalls, comment conventions and the
required workflow for future changes.
