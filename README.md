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

## Home Manager Installation

The flake exports a Home Manager module that installs UniVim, starts one
`univim-desktop serve` per Hyprland session, and can declare activation
keybindings. It does not modify your configuration files directly and does not
run any part of the desktop backend during Nix evaluation.

Requirements:

- NixOS or another Linux system using Home Manager.
- Hyprland with the protocols listed under [Desktop Navigation](#desktop-navigation).
- Hyprland configuration managed by Home Manager; this is required only for
  automatic keybindings. The service can be used with an externally managed
  Hyprland configuration by leaving `keybindings.enable` disabled.

### 1. Add the flake input

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    univim.url = "github:Tiqur/univim/rust";
  };

  outputs = { self, nixpkgs, home-manager, univim, ... }@inputs: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      specialArgs = { inherit inputs; };
      modules = [
        ./configuration.nix
        home-manager.nixosModules.home-manager
        {
          home-manager.useGlobalPkgs = true;
          home-manager.useUserPackages = true;

          home-manager.users.tiqur = {
            imports = [ univim.homeManagerModules.default ];
            services.univim = {
              enable = true;
              keybindings.enable = true;
            };
          };
        }
      ];
    };
  };
}
```

The `outputs` arguments create two equivalent names:

- `univim.homeManagerModules.default`, because `univim` is an output argument.
- `inputs.univim.homeManagerModules.default`, because `...@inputs` captures the
  named flake inputs.

The module uses the package from the same UniVim revision as the module, so the
service and control commands cannot drift apart.

### 2. Import and enable the module

The example above imports the module directly in the NixOS configuration and
enables it for user `tiqur`. If your Home Manager configuration is split into
files such as `home/tiqur/modules/hyprland.nix`, pass `inputs` to those files
and import there instead:

```nix
# In the NixOS module that configures Home Manager:
home-manager.extraSpecialArgs = { inherit inputs; };
```

```nix
# home/tiqur/modules/hyprland.nix
{ inputs, ... }:
{
  imports = [ inputs.univim.homeManagerModules.default ];

  services.univim = {
    enable = true;
    keybindings.enable = true;
  };
}
```

The `{ inputs, ... }` argument only exists after `extraSpecialArgs` is added (or
after an equivalent wrapper supplies it). Without that, Nix reports that `inputs`
is not defined.

For a standalone `home-manager.lib.homeManagerConfiguration`, add the module to
its `modules` list in the same way:

```nix
home-manager.lib.homeManagerConfiguration {
  inherit pkgs;
  modules = [
    inputs.univim.homeManagerModules.default
    ./home.nix
  ];
}
```

With the configuration above, `SUPER+F` opens hints and `SUPER+G` opens grid.
The service starts automatically with the session; do not add an
`exec-once = univim-desktop daemon` line or run `hints`/`grid` from a terminal.
Apply with your normal command, for example:

```sh
sudo nixos-rebuild switch --flake .#myhost
```

Then start it in the current session with:

```sh
systemctl --user start univim.service
```

### Session startup

The module selects `hyprland-session.target` when Home Manager's native Hyprland
systemd integration is enabled. That is the default for a normal Home Manager
Hyprland setup, and it imports `WAYLAND_DISPLAY`,
`HYPRLAND_INSTANCE_SIGNATURE`, and the other session variables before UniVim
starts.

If the Home Manager integration is disabled (for example with UWSM), the module
uses `graphical-session.target`. The external session manager must provide the
same environment and session lifetime. A custom target can be selected
explicitly:

```nix
services.univim.sessionTarget = "graphical-session.target";
```

The service has `PartOf` and `WantedBy` relationships with that target, so it
stops on logout and does not start at boot or outside a graphical session.
Logging out cleans up the socket normally. A crash can leave `control.sock`
behind; startup intentionally fails closed instead of deleting a socket that
might belong to another running service. Stop the old process and remove the
named stale socket before starting again. The module deliberately does not add
automatic restarts, because restarting cannot repair a stale socket.

### Options

```nix
services.univim = {
  enable = true;

  # Optional service settings:
  output = "DP-3";       # null follows the focused monitor
  flat = true;           # leaf-only hints, no drill-down
  turbo = true;          # alias for settings.fastHalfRes
  interactiveOnly = false;
  noClick = false;
  timeout = 30;          # seconds, 1-3600

  # Detector and gesture tuning:
  settings = {
    minScore = 0.45;
    includeText = true;
    fastHalfRes = false;
    collapseParagraphs = true;
    doubleTapMs = 260;
    cacheTtlMs = 900;
    spotlightDim = true;
  };

  # Optional declarative bindings:
  keybindings = {
    enable = true;
    hints = "SUPER, F";
    grid = "SUPER, G";
    modifierGestures = false;
  };

  # Optional global shortcuts and passive evdev gestures:
  hyprland = {
    enable = true;
    hintsBind = "SUPER, F";
    gridBind = "SUPER, G";
    doubleTapCapsLock = false;
    doubleTapF = false;
    evdev = true;
  };
};
```

Binding values use Hyprland's `"<modifiers>, <key>"` form. They are merged with
your existing `bind` and `bindn` lists; existing bindings are not replaced.
Change the chords if they conflict with another binding. The module supports
both Home Manager Hyprland configuration formats:

- `configType = "hyprlang"` generates normal `bind`/`bindn` entries.
- `configType = "lua"` generates equivalent `hl.bind` calls.

Set `modifierGestures = true` to add the four nonconsuming modifier bindings
used by double-Shift and double-Ctrl activation. These gestures are best-effort:
Hyprland's anonymous shortcut notifications cannot reveal intervening chord
keys, so ordinary typing can occasionally trigger them. Prefer explicit
bindings when accidental activation is unacceptable. See
[Opt-In Hotkeys](#opt-in-hotkeys) for the gesture timing and limitations.

If Hyprland is configured outside Home Manager, leave `keybindings.enable`
disabled and add the bindings manually. The installed command can be referenced
from `PATH`:

```ini
bind = SUPER, F, exec, univim-desktop hints
bind = SUPER, G, exec, univim-desktop grid
```

UniVim remains Hyprland-specific. The Home Manager module makes installation
declarative; it does not add backends for GNOME, KDE, Sway, X11, or generic
Wayland compositors.

### Troubleshooting

```sh
systemctl --user status univim.service
journalctl --user -u univim.service -b
univim-desktop status
```

Do not publish captured screenshots or your full environment when asking for
help. `univim-desktop status` reports the active mode and output without screen
contents.

## Desktop Navigation

The commands in this section are manual development and diagnostic controls.
For normal use, [Home Manager Installation](#home-manager-installation) starts
the service and manages the keybindings.

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
# One-shot numbered diagnostic capture; does not need the service:
univim-desktop snapshot --dir captures
```

`snapshot` captures the focused monitor unless `--output NAME` is supplied and
writes three artifacts with the same numeric stem:

- `N.png`: raw screenshot.
- `N.annotated.png`: every parent and descendant target boxed, colored by
  hierarchy depth, with `D<depth> <path> <kind>` labels.
- `N.json`: the complete ordered hierarchy in `targets`, every node in `flat`,
  and immediate/word drill-down results for every parent in `drills`, together
  with output metadata, options and detector timings.

`--turbo`, `--interactive-only` and `--min-score` apply to snapshots. Add
`--flat` to replace every composite target with its leaf descendants: only
leaf boxes are rendered and `targets`/`flat` contain leaves, with no parent
targets or `drills`. Capture files contain screen contents and are written with
private permissions. The default `captures/` directory is ignored by Git; avoid
adding it or committing captured screenshots.

`check` reports capabilities and discovered outputs without capturing or injecting
input. `daemon [--output NAME] [--fast-half-res|--turbo] [--no-text|--interactive-only]
[--no-click] [--flat] [--timeout 30]` runs the service (`serve` is an alias).
Output discovery is dynamic: each activation
uses the currently focused monitor unless `--output NAME` selects a connector.
Monitor position, scale and transform come from runtime metadata, not a fixed
layout. Unavailable outputs fail rather than falling back to another monitor.
`--fast-half-res` (`--turbo`) and `--no-text` (`--interactive-only`) configure
detection. `--no-click` allows selection without injecting clicks; it still
captures and takes overlay keyboard focus during an active session. `--flat`
flattens all composite targets to their leaf descendants before hints, so parent
hints and drill-down are absent. Target outlines and hint tags are colored by
element type.
The session timeout defaults to 30 seconds
(accepted range 1-3600). `stop` terminates the daemon.

In hints mode, type a label to select its target and click its center. Labels
filter as you type without being reassigned. Hold **Shift** or **Ctrl** on the
final hint character to drill: Shift reaches word fragments and Ctrl reaches
immediate children. On a leaf, Shift middle-clicks and Ctrl right-clicks.
**Backspace** edits the prefix, or returns to the parent when the prefix
is empty. **Escape** cancels. **Arrow keys** or **Ctrl+H/J/K/L** snap a
reticule to the nearest target; **Enter** clicks it. **;** or **Tab** cycles the
click action, **v** starts a two-point drag, **Ctrl+W** toggles focused-window
filtering, **Shift+J/K/H/L** or **Ctrl+D/U** scroll and re-capture, **F5**
refreshes, **F6** toggles continuous refresh, **F1**/`?` opens live options, and
**F2** opens the pipeline inspector. Grid mode first shows a 6x4 grid with
single-character labels; the second label selects one of its 6x4 subcells and
clicks the center.

### Opt-In Hotkeys

`univim-desktop bindings` prints an `exec-once` service command and anonymous,
nonconsuming Hyprland `bindn` global bindings for left/right Shift and Ctrl.
Nothing installs these bindings or modifies your configuration automatically.
Review the printed configuration and opt in through your own Hyprland config;
do not start a second service if one is already running. The Home Manager module
can declare the equivalent service and bindings explicitly; see
[Home Manager Installation](#home-manager-installation).

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
nix build path:. --no-link
nix flake check path:.
```

`nix flake check` evaluates the exported Home Manager module, service unit and
generated Hyprlang/Lua bindings without starting a desktop session.

Before files are staged, use `nix develop path:. --command COMMAND` so new files
are visible to Nix's source snapshot. After staging, ordinary `nix develop` works.
The shell and package share a source filter excluding `target`, `result`, `.git`
and Python caches. This keeps build inputs small, but a `path:.` flake may still
copy the working directory before that filter is evaluated. Prefer the ordinary
Git-backed flake once new source files are tracked; `.gitignore` alone cannot
prevent the initial `path:.` snapshot.

The full differential suite passed **513 exact detection comparisons**, **513
hint checks**, and **538 pixel-exact overlay comparisons**, including **25
drill-down cases**. It uses 20 deterministic synthetic fixtures, with light/dark
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
