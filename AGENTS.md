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

## Required Verification

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
Run `nix build` when changing packaging. Do not claim full real-world equivalence
from synthetic fixtures alone. Add any supplied real screenshots to the
differential workflow, subject to privacy and licensing. Never weaken exact
comparisons or add tolerances to make a failed port pass.

Benchmark release builds, preloaded images, one OpenCV thread, and warmed repeated
detection calls. Report workload, resolution, mode, sample count and medians for
both implementations, including regressions. Keep JSON samples and machine details
when publishing results. Do not compare Rust detection-only against Python process
startup or image decoding. Generated files belong under ignored `target/`.

## Git Safety

The requested remote is `git@github.com:Tiqur/univim.git`, branch `rust`.
Do not touch `main` or `cpp`. Inspect status and diffs before staging explicit
files. Preserve unrelated user changes. Commit and push only when requested;
never force-push without explicit authorization for the affected branch.
