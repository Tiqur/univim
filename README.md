# UniVim

A native Rust/OpenCV port of the hierarchical screenshot navigation detector in
`python.py`. It detects controls, text, thumbnails, icon-label compounds and
paragraphs, generates Vimium-style hints, and keeps children for instant drill-down.
The Rust executable does not invoke Python. `python.py` remains the behavior
reference; the v9 detector changes are implemented natively in Rust.

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
nix flake update univim # run from your system flake directory to fetch new UniVim
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

Here is a complete Home Manager example. Import the module and pass `inputs`
as shown above; replace the example file path and host name with yours. Every
UniVim module option is set explicitly below; comments identify defaults.
This example chooses **single-press left Alt**; double-tap left Alt is the default.

```nix
# In a Home Manager module imported by your host configuration:
{ inputs, pkgs, lib, ... }:
{
  imports = [ inputs.univim.homeManagerModules.default ];

  services.univim = {
    enable = true; # Default: false. Install package and start session service.
    package = inputs.univim.packages.${pkgs.stdenv.hostPlatform.system}.default;
      # Default: same package from this flake. Override only for a custom build.
    output = null; # Default: null = focused monitor; e.g. "DP-3" to pin one.
    sessionTarget = null; # Default: null = auto-select Hyprland or graphical target.
    timeout = 30; # Default: 30 seconds; valid 1–3600.
    noClick = false; # Default: false; true previews but never injects clicks.
    flat = false; # Default: false; true uses leaf hints without parent drill-down.
    turbo = false; # Default: false; alias for half-resolution detection.
    interactiveOnly = false; # Default: false; true omits text targets.
    colorByType = true; # Default: true; legacy setting, colors are always on.

    settings = {
      minScore = 0.45; # Default: 0.45; higher filters more detections.
      includeText = true; # Default: true; false omits text targets.
      fastHalfRes = false; # Default: false; same effect as turbo = true.
      collapseParagraphs = true; # Default: true; group multi-line prose.
      doubleTapMs = 260; # Default: 260 ms; only used in double-tap mode.
      cacheTtlMs = 900; # Default: 900 ms; speculative first-tap cache lifetime.
      spotlightDim = true; # Default: true; dim screenshot behind hints.
      pointerMotion = "instant"; # Default: "instant"; also "linear" or "ease_out".
      pointerSpeed = 1600; # Default: 1600 logical px/s; 200–6000. Ignored if instant.
      appearance = {
        hintFontScale = 0.68; # Default: 0.68 (formerly 0.36); size of hint labels.
        uiScale = 1.6; # Default: 1.6; status/keystrokes, debug HUD, options menu.
        badgeBackground = "#FFD737";
        badgeForeground = "#0A0A0A";
        typedForeground = "#5B158C"; # Bold prefix on matching hint labels.
        uiBackground = "#121214";
        uiForeground = "#FAF5F0";
        uiAccent = "#FFD000";
      };
    };

    keybindings = {
      enable = false; # Default: false; true adds SUPER+F/G exec shortcuts.
      hints = "SUPER, F"; # Default: "SUPER, F" when enabled.
      grid = "SUPER, G"; # Default: "SUPER, G" when enabled.
      modifierGestures = false; # Default: false; best-effort double Shift/Ctrl.
    };

    hyprland = {
      enable = true; # Default: false; installs the requested activation binding.
      hintsBind = null; # Default: null; e.g. "SUPER, F" for a global hint shortcut.
      gridBind = null; # Default: null; e.g. "SUPER, G" for a global grid shortcut.
      activationKey = "left-alt"; # Default: "left-alt"; also "left-ctrl", "caps-lock", "off".
      doubleTap = false; # Default: true; single left Alt also triggers on Alt chords.
      disableCapsLock = false; # Default: false; true sets XKB caps:none.
      evdev = false; # Default: false; legacy passive CapsLock gesture listener.
    };
  };

  # If you already use other XKB options, merge them with caps:none explicitly:
  # wayland.windowManager.hyprland.settings.input.kb_options =
  #   lib.mkForce "grp:alt_shift_toggle,caps:none";
}
```

For CapsLock hints instead, set `activationKey = "caps-lock"` and choose
`doubleTap = true` or `false`; set `disableCapsLock = true` if you do not
want CapsLock to toggle the lock state. Single CapsLock also binds Shift+Caps
to grid and Alt+Caps to debug. Disabling CapsLock is independent of the
activation key. An existing `input.kb_options` conflicts with `caps:none`;
merge them yourself as shown above. Keep `evdev = false` unless you specifically
need legacy passive CapsLock gestures, to avoid duplicate activations.
When upgrading from older versions, replace `hyprland.altActivation`,
`hyprland.singlePressCapsLock` and `hyprland.doubleTapCapsLock` with the
`activationKey` / `doubleTap` pair; set `disableCapsLock = true` explicitly if
you previously relied on single CapsLock to disable its normal lock action.
Avoid declaring the same
SUPER+F/G shortcuts under both `keybindings` and `hyprland.hintsBind/gridBind`.

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

`python.py` is the authoritative detector behavior reference. Always develop
and verify detector changes in Python first, then port them to native Rust and
checked with the exact differential tests. The desktop-only snapshot, input,
rendering and Home Manager features documented here do not replace that rule.

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
writes private artifacts with the same numeric stem:

- `N.png`: raw screenshot.
- `N.annotated.png`: every parent and descendant target boxed, colored by
  hierarchy depth, with `D<depth> <path> <kind>` labels.
- `N.json`: the complete ordered hierarchy in `targets`, every node in `flat`,
  and immediate/word drill-down results for every parent in `drills`, together
  with output metadata, options, detector timings, physics thresholds, stage
  timings, candidate counts and mask occupancy in `debug`.
- `N.debug-sheet.png`: all six inspector views in one labeled image.
- `N.debug-1-overlay.png` through `N.debug-6-hierarchy.png`: full-resolution
  individual visualizations from the **same** detector pass. These are not
  separate screenshots or reruns. Inspector views 1–6 are UI visualizations:
  overlay, raw capture/target outlines, gradients, barriers/frames,
  strokes/coarse regions and final hierarchy. They do not correspond one-to-one
  to the four measured phases (`signal`, `frames`, `coarse`, `recursive`) in JSON.

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
filter as you type without being reassigned. On leaves, **Shift** on the final
hint character right-clicks and **Ctrl** middle-clicks; Ctrl+Shift
double-clicks and Alt hovers. On composites,
Shift drills to words and Ctrl to immediate children instead. Press **;** or
**Tab** to cycle the default action (left/right/double/middle/hover); this also
lets you select single-letter J/K/H/L hints without triggering Shift+J/K/H/L
scroll when the prefix is empty (the modifier shortcut takes precedence for
those single-letter hints). **Backspace** edits the prefix, or returns to the
parent when empty. **Escape** cancels. **Arrow keys** or **Ctrl+H/J/K/L** snap a
reticule to the nearest target; **Enter** clicks it. **v** starts a two-point
drag, **Ctrl+W** toggles focused-window filtering, **Shift+J/K/H/L** or
**Ctrl+D/U** scroll and re-capture, **F5**
refreshes, **F6** toggles continuous refresh, **F1**/`?` opens live options, and
**F2** opens the pipeline inspector; **1–6** select its visualizations, **F3/F4**
cycle them. Its diagnostic views omit hint labels so the underlying image
remains visible. With the evdev listener enabled, double-tap CapsLock for
hints, hold Shift while double-tapping CapsLock for grid, or hold Alt while
double-tapping CapsLock for the pipeline inspector. Grid mode first shows a 6x4 grid with
single-character labels; the second label selects one of its 6x4 subcells and
clicks the center.

Scrolling moves the Wayland virtual pointer to the focused window, injects
wheel events and then re-captures; it needs no `/dev/uinput` permission.
The status pill shows the requested scroll
direction; actual application scrolling still needs live verification.
Transient popup menus may dismiss when the overlay takes keyboard focus;
those cannot be reliably selected with the current exclusive layer-shell
overlay. Avoid claiming success for menu items whose popup has closed.

With `services.univim.hyprland.enable = true`, left Alt defaults to **double-tap
for hints** (`activationKey = "left-alt"; doubleTap = true;`). Use
`activationKey = "left-ctrl"` for left Ctrl, `"caps-lock"` for CapsLock, or
`"off"` for no activation gesture. `doubleTap = false` makes one press open
hints. `disableCapsLock = true` sets `caps:none` regardless of activation key;
merge any existing Hyprland `input.kb_options` yourself. Single-press CapsLock
also binds Shift+Caps to grid and Alt+Caps to debug. These shortcuts are opt-in
and do not modify external configuration files.
The non-consuming shortcut leaves ordinary Alt events available, but a single
press also activates on the first press of ordinary Alt chords. Double-tap
recognition via compositor global shortcuts is best-effort: unrelated keys
between taps cannot always be detected. Right Alt/Ctrl are never bound by this option.
The modifier-only binding uses an empty modifier field (`bindn = , Alt_L`):
requiring `ALT` as a modifier prevents Hyprland from matching the Alt key itself.

The desktop overlay uses larger fonts by default: hint labels use scale 0.68
instead of 0.36, and the status pill, typed-keystroke display, debug HUD and
options menu use `uiScale = 1.6`. Configure both sizes and the six `#RRGGBB`
colors via `settings.appearance` (example above). As you type a partial hint,
matching labels emphasize the typed prefix in bold `typedForeground`; completed
hints select their target immediately. These are desktop-only presentation
settings; detector output and annotated snapshots keep their reference styling.

`services.univim.settings.pointerMotion` selects `"instant"` (default),
`"linear"` or `"ease_out"`; `pointerSpeed` (200–6000 logical px/s, default 1600)
controls the distance-dependent glide after hint selection (20 ms minimum,
2 seconds maximum). Both are also
adjustable for the running session in the F1 overlay options menu. With Home
Manager, edit the Nix options and rebuild to persist changes; the generated
settings file is managed by Home Manager. Instant movement remains the default.

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

The full differential suite passed **585 exact detection comparisons**, **585
hint checks**, and **613 pixel-exact overlay comparisons**, including **28
drill-down cases**. It uses 23 deterministic synthetic fixtures, with light/dark
UI, paragraphs, blue links, controls, textures, noise, dense hints, a portrait
crop, a tiny glyph chain, a multi-line terminal log with line-level drill-down,
and widths 640 through 2880, including resize boundaries. All boolean option combinations and score boundaries are exercised.
It checks list ordering, all coordinates, scores, kinds, levels, hints, nested
children, and missing versus empty child keys. Timing fields are intentionally
excluded. Library unit tests additionally cover hint generation, both drill-down
modes, flattening, and malformed externally supplied children; desktop modules
have separate unit tests. There are no numeric tolerances in detection/image
comparisons.

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
| Light UI | 2561 x 1601 | Normal | 95.21 | 108.94 | 0.87x |
| Light UI | 2561 x 1601 | Turbo | 38.20 | 33.80 | 1.13x |
| Light UI | 1440 x 900 | Normal | 30.50 | 27.13 | 1.12x |
| Light UI | 1440 x 900 | Turbo | 30.57 | 19.22 | 1.59x |
| Dense controls | 1440 x 900 | Normal | 118.86 | 17.95 | 6.62x |
| Dense controls | 1440 x 900 | Turbo | 121.84 | 21.09 | 5.78x |

Per-stage medians use the reference's boundaries: `signal` covers scaling,
gradients and structural barriers; `frames` covers frame/thumbnail recovery and
barrier dilation; `coarse` covers coarse proposals; `recursive` covers recursive
refinement, composition, containment cleanup and final hints. Values are
Python / Rust milliseconds.

| Workload | Mode | Signal | Frames | Coarse | Recursive |
| --- | --- | ---: | ---: | ---: | ---: |
| Light UI 2561 x 1601 | Normal | 50.0 / 62.9 | 21.8 / 29.6 | 3.2 / 8.2 | 19.7 / 2.9 |
| Light UI 2561 x 1601 | Turbo | 19.0 / 21.4 | 7.2 / 7.8 | 0.8 / 2.1 | 11.0 / 1.2 |
| Light UI 1440 x 900 | Normal | 6.4 / 9.5 | 11.0 / 11.8 | 1.0 / 2.7 | 11.8 / 1.5 |
| Light UI 1440 x 900 | Turbo | 6.4 / 5.5 | 11.0 / 9.1 | 1.0 / 2.7 | 11.9 / 1.5 |
| Dense controls 1440 x 900 | Normal | 6.3 / 4.4 | 7.7 / 7.4 | 1.1 / 2.8 | 103.4 / 2.7 |
| Dense controls 1440 x 900 | Turbo | 6.5 / 4.6 | 7.8 / 10.2 | 1.2 / 2.7 | 106.0 / 3.0 |

Rust is faster in five of the six configurations in this run. Large colored
screenshots are the exception in absolute medians, but they are also the noisiest:
the same release binary measured about 82 ms in standalone runs and about 106 ms
immediately after the in-suite Python benchmark. Running 30 Python detections
first reproduces that gap, so thermal or scheduling state, not detector work,
dominates this case. Controlled alternating A/B runs of the pre- and post-
optimization binaries measured 82.91 vs 82.10 ms on the large normal UI and
30.10 vs 16.91 ms on dense controls. The achromatic fast path therefore removes
the chroma pipeline for grayscale BGR screenshots without regressing colored
ones.

The remaining relative weaknesses are localized: `coarse` is slower on the large
and 1440 x 900 UI cases, and `frames` is occasionally slower on grayscale dense
controls. Those stages repeatedly allocate and inspect full-resolution masks and
contour ROIs, so memory traffic, allocation and cache behavior are likely
contributors. The largest absolute win comes from recursive refinement and
composition, which dominates Python's dense controls time. Turbo changes the
algorithm's working resolution and coarse pass, so compare implementations within
the same mode rather than normal versus turbo. Stage medians may not sum exactly
to wall-clock totals because they are measured at internal boundaries.

See `AGENTS.md` for architecture, parity pitfalls, comment conventions and the
required workflow for future changes.
