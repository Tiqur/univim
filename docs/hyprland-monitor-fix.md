# Hyprland 0.57 multi-monitor focus compatibility

Reported with keyboard-focused window on DP-3 (main) but hints always on first
Wayland output DP-1 (portrait). Pinned context parsing required numeric special
workspace IDs; live 0.57 omits the ID for an inactive special workspace:
`{"address":"","type":"","name":""}`.
That error discarded Hyprland context and Wayland capture fell back to outputs[0].

Changes in src/desktop/hyprland.rs:
- Accept numeric workspace IDs and numeric strings; empty string maps to zero
  only for special workspaces. Accept the exact ID-omitted empty-workspace shape;
  reject missing IDs on real workspaces and explicit null/malformed IDs.
- Keep numeric monitor ID/name mapping. Select focused visible client's output
  before pointer-focused output. Preserve raw focus fingerprint during our own
  layer's keyboard ownership; an empty activewindow must not cancel itself.
- Regression tests for modern/legacy IDs, invalid IDs, window/pointer focus
  disagreement and empty-desktop fallback. No detector/rendering algorithm edits.

Verification (2026-11-17, one-CPU scope/low priority while user games):
- cargo fmt --check; clippy --all-targets -- -D warnings; cargo test (36 passed).
- cargo build --release.
- Full parity: 633 detection, 633 hints, 660 renders, 27 drills; zero failures.
- Benchmark command: 30 runs / 5 warmups; full report target/parity/report.json,
  console /tmp/opencode/univim-monitor-fix-verification.log. Post-change results,
   not a paired pre/post regression claim. Single-thread BLAS/OpenMP for this run.

Live proof: headless guest with Virtual-1 (1920x1080) and VMAGENT-TEST
(1080x1920). Test Kitty PID 4849 moved between workspaces 1/2 using the real
binding; Alt placed hints on the owning output on both. Candidate ran through a
temporary owned runtime unit drop-in with --no-click. First live attempt caught
the ID-omitted case missing from the original empty-string-only fix; that failed
attempt is not counted as a pass. Reduced real JSON added as a parser fixture.
Captures: vmagent/experiments/desktop-polish-headless-20261117/ (also initially
/tmp/opencode/headless-univim-{main,portrait}-fixed.png).

User explicitly approved publishing this scoped rust-branch fix after live proof,
then pinning its revision in the desktop config. No main/cpp changes, detector
algorithm changes, nixconfig push or physical activation authorized. Restore the
test-only runtime override before ending the guest check; normal Nix packaging
must deliver the fix rather than a local source patch.
