# Project Guide

## Purpose And Layout

UniVim finds hierarchical UI navigation targets in screenshots. `python.py` is
the original behavior oracle. Rust is a native OpenCV implementation, not a
Python subprocess or embedded interpreter.

- `src/lib.rs`: public target/options types, detection, hints, drill-down, helpers.
- `src/signals.rs`: scaling, gradients, structural masks, frames and coarse regions.
- `src/refine.rs`: recursive projections, text/link classification, word children.
- `src/compose.rs`: icon-label pairing, paragraphs, rails, containment, final hints.
- `src/render.rs`: standard and spotlight overlays.
- `src/main.rs`: CLI and warmed detection-only benchmarks.
- `src/bin/univim-desktop.rs`: Linux desktop service CLI and private control IPC.
- `src/desktop/`: native Wayland capture/input/overlay runtime, Hyprland discovery,
  coordinate geometry, software drawing and pure selection/gesture state.
- `tests/parity.py`: deterministic fixtures, exact differential tests, benchmarks.

## Engineering Conventions

Keep code clean and idiomatic. Prefer the smallest correct change. Use descriptive
names, format with rustfmt, and keep helpers local unless shared. Add detailed
comments for algorithmic stages, invariants, coordinate systems, or surprising
reference behavior; do not narrate obvious assignments. Explain why an operation
or threshold exists when known, rather than inventing a rationale. Avoid unsafe
pixel access: use checked row slices for hot loops or native OpenCV operations.

Preserve the reference unless a behavior change is explicitly requested. Do not
silently fix its quirks. In particular:

- Stable sorting and first-match iteration order affect compounds and hint labels.
- Python integer `//` floors negative numbers; use `div_euclid`, not Rust `/`.
- Python coordinate `round` uses ties-to-even; casts to integer truncate.
- Child coordinates are scaled when created; top-level coordinates are scaled last.
- Word children omit `children`; composite children include it, even when empty.
- OpenCV rectangle endpoints are inclusive; NumPy ROI slices are half-open.
- NumPy convolution with an even window has asymmetric alignment.
- Drill-down flattens one child level, not arbitrary recursive descendants.
- Timing fields are nondeterministic and are not part of exact-output parity.

## Python-First Iteration

`python.py` is the authoritative behavior reference. The user may iterate on it
first, leaving Rust temporarily behind. When asked to synchronize, port, or restore
parity, inspect the current Python implementation and its changes, then update the
native Rust detector, helpers, rendering, options and desktop integration wherever
those changes apply. Do not assume the originally ported Python version is still
the specification.

Preserve the user's Python changes. Fix mismatches in Rust rather than reverting
Python, changing the reference to match Rust, hardcoding expected results, weakening
comparisons or adding numeric tolerances. If a Python change is ambiguous, ask
before choosing different behavior.

Extend differential fixtures to exercise new behavior and edge cases, including
provided screenshots when privacy/licensing permits. Compare complete ordered
targets, scores, hints, nested children, missing keys, drill-down and rendered
pixels where applicable, across all relevant options. Timing values are excluded
from exact comparisons. Run the required verification and warmed per-stage
benchmarks, report regressions, and state any untested behavior. Do not claim
parity complete until the updated differential suite passes.

## Required Verification

Desktop changes must preserve the standalone detector and benchmark updates.
Keep capture in memory, with no screenshot subprocesses/files and no idle mapped
surface. Discover outputs dynamically, default to the focused monitor, and keep
capture pixels, logical overlay coordinates and pointer coordinates distinct.
Document anonymous global shortcut gestures as best-effort: they cannot reject
arbitrary intervening chord keys. Hotkey configuration is opt-in output from
`bindings`, never an automatic edit to the user's configuration. Drill mode
expands immediate children only; retain parent navigation and cancellation.
Do not run live capture, overlay, hotkey or click tests without explicit approval.
Use CLI help and pure unit tests for noninteractive desktop verification.
The flake is Linux-only because Cargo's Wayland dependencies are unconditional;
a cfg-gated desktop entry point alone does not make macOS builds supported.

Run commands inside the pinned environment so Python and Rust share OpenCV:

```sh
nix develop
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release
python3 tests/parity.py
python3 tests/parity.py --benchmark --runs 30 --warmup 5
```

Use `nix develop path:. --command ...` if testing new files before Git staging.
Both shell and package use the flake's filtered source, excluding build artifacts.
The initial `path:.` snapshot can still copy ignored files before filtering;
prefer Git-backed flake evaluation once new files are tracked. Do not stage files
merely to speed up Nix without permission.
Run `nix build` when changing packaging. Do not claim full real-world equivalence
from synthetic fixtures alone. Add any supplied real screenshots to the
differential workflow, subject to privacy and licensing. Never weaken exact
comparisons or add tolerances to make a failed port pass.

Benchmark release builds, preloaded images, one OpenCV thread, and warmed repeated
detection calls. Report workload, resolution, mode, sample count and medians for
both implementations, including regressions. Preserve per-stage medians
(`signal`, `frames`, `coarse`, `recursive`, `total`) so regressions can be
localized instead of inferred from the total. Keep JSON samples and machine
details when publishing results. Do not compare Rust detection-only against
Python process startup or image decoding. Generated files belong under ignored
`target/`.

## Git Safety

The requested remote is `git@github.com:Tiqur/univim.git`, branch `rust`.
Do not touch `main` or `cpp`. Inspect status and diffs before staging explicit
files. Preserve unrelated user changes. Commit and push only when requested;
never force-push without explicit authorization for the affected branch.
