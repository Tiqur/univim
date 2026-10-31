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

Only `x86_64-linux` was tested; the flake also declares aarch64 Linux and macOS
development/package outputs, which have not been verified here.

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

The full differential suite passed **465 exact detection comparisons**, **465
hint checks**, and **488 pixel-exact overlay comparisons**, including **23
drill-down cases**. It uses 18 deterministic synthetic fixtures, with light/dark
UI, paragraphs, blue links, controls, textures, noise, dense hints, and widths
640 through 2880, including resize boundaries. All boolean option combinations
and score boundaries are exercised. It checks list ordering, all coordinates,
scores, kinds, levels, hints, nested children, and missing versus empty child keys.
Timing fields are intentionally excluded. Nine Rust unit tests additionally
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

Machine: AMD Ryzen 7 7700X, Linux x86_64, Rust 1.99.0, Python 3.14.8,
OpenCV 4.13.0, NumPy 2.5.3. Results vary with system load and image content.

| Synthetic Workload | Resolution | Mode | Python | Rust | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| Light UI | 2561 x 1601 | Normal | 62.49 | 83.97 | 0.74x |
| Light UI | 2561 x 1601 | Turbo | 24.94 | 19.63 | 1.27x |
| Light UI | 1440 x 900 | Normal | 14.79 | 14.64 | 1.01x |
| Light UI | 1440 x 900 | Turbo | 14.62 | 8.61 | 1.70x |
| Dense controls | 1440 x 900 | Normal | 63.21 | 13.74 | 4.60x |
| Dense controls | 1440 x 900 | Turbo | 63.65 | 12.92 | 4.93x |

Rust is about **34.4% slower** on the large normal-mode UI fixture in this run,
and large-image timings showed substantial run-to-run variation. It is not
uniformly faster. Turbo changes the algorithm's working resolution/coarse pass;
compare implementations within the same mode rather than normal versus turbo.

See `AGENTS.md` for architecture, parity pitfalls, comment conventions and the
required workflow for future changes.
