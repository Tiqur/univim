use std::{env, path::PathBuf, process};
use univim::{
    DetectorOptions,
    desktop::{
        CommandAction, RuntimeSettings, capture_snapshot, check, default_socket_path, run_daemon,
        send_command_or_autospawn,
    },
};

fn print_usage() {
    eprintln!(
        "Usage: univim-desktop <daemon|snapshot|check|bindings|hints|grid|tap-caps|debug|options|stop> [options]

Commands:
  daemon         Run the background Wayland/Hyprland overlay daemon (alias: serve)
  snapshot       Capture one monitor and save raw, annotated, and JSON artifacts
  check          Enumerate advertised protocols and outputs (no capture/input)
  bindings       Print suggested opt-in Hyprland configuration
  hints          Show full-screen Vimium hints (auto-starts daemon if not running)
  grid           Show recursive 2-step grid overlay
  tap-caps       Record a 'CapsLock' tap (1st tap pre-computes cache, 2nd tap opens hints in 0ms)
  debug          Toggle the live pipeline inspector HUD (also F2 inside overlay)
  options        Toggle the live GUI options modal (also F1 / ? inside overlay)
  stop           Stop the running daemon

Options:
  --socket PATH              Control socket path (default: $XDG_RUNTIME_DIR/univim/control.sock)
  --output NAME              Capture a specific output instead of the focused monitor
  --min-score FLOAT          Minimum target score (default: 0.45)
  --no-text                  Exclude text block targets
  --fast-half-res            Run Stage 1-4 at half resolution for ultra-low latency
  --no-collapse-paragraphs   Keep individual text lines separate
  --no-click                 Display hints but suppress synthetic pointer actions
  --flat                     Replace composite targets with their leaf descendants
  --timeout SECS             Auto-cancel the overlay after SECS (default: 30)
  --dir PATH                 Snapshot output directory (default: captures)"
    );
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(subcommand) = args.next() else {
        print_usage();
        process::exit(2);
    };
    if subcommand == "-h" || subcommand == "--help" {
        print_usage();
        return;
    }

    let mut socket: PathBuf = default_socket_path();
    let mut directory = PathBuf::from("captures");
    let mut options = DetectorOptions::default();
    let mut output: Option<String> = None;
    let mut no_click = false;
    let mut flat = false;
    let mut timeout_secs: Option<u64> = None;
    let mut min_score: Option<f64> = None;
    let mut no_text = false;
    let mut fast_half_res = false;
    let mut no_collapse = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--socket" => {
                let Some(value) = args.next() else {
                    eprintln!("--socket requires a path");
                    process::exit(2);
                };
                socket = PathBuf::from(value);
            }
            "--output" => {
                let Some(value) = args.next() else {
                    eprintln!("--output requires a name");
                    process::exit(2);
                };
                output = Some(value);
            }
            "--dir" => {
                let Some(value) = args.next() else {
                    eprintln!("--dir requires a path");
                    process::exit(2);
                };
                directory = PathBuf::from(value);
            }
            "--min-score" => {
                let Some(value) = args.next() else {
                    eprintln!("--min-score requires a float");
                    process::exit(2);
                };
                min_score = Some(value.parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("invalid --min-score value: {value}");
                    process::exit(2);
                }));
            }
            "--timeout" => {
                let Some(value) = args.next() else {
                    eprintln!("--timeout requires seconds");
                    process::exit(2);
                };
                timeout_secs = Some(value.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("invalid --timeout value: {value}");
                    process::exit(2);
                }));
            }
            "--no-text" | "--interactive-only" => no_text = true,
            "--fast-half-res" | "--turbo" => fast_half_res = true,
            "--color-by-type" => {}
            "--no-click" => no_click = true,
            "--no-collapse-paragraphs" => no_collapse = true,
            "--flat" => flat = true,
            "-h" | "--help" => {
                print_usage();
                return;
            }
            other => {
                eprintln!("unknown argument: {other}");
                print_usage();
                process::exit(2);
            }
        }
    }

    if let Some(value) = min_score {
        options.min_score = value;
    }
    if no_text {
        options.include_text = false;
    }
    if fast_half_res {
        options.fast_half_res = true;
    }
    if no_collapse {
        options.collapse_paragraphs = false;
    }

    let result = match subcommand.as_str() {
        "check" => check().map(|report| {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).unwrap_or_else(|_| report.to_string())
            );
        }),
        "bindings" => {
            println!(
                "# Start once after your Wayland environment is ready.\nexec-once = univim-desktop daemon\n# Opt-in global shortcuts.\nbind = SUPER, F, global, univim:hints\nbind = SUPER, G, global, univim:grid\nbindn = , CAPS, global, univim:caps-lock\n# Nonconsuming modifier bindings preserve ordinary modifier events.\nbindn = SHIFT, Shift_L, global, univim:shift-left\nbindn = SHIFT, Shift_R, global, univim:shift-right\nbindn = CTRL, Control_L, global, univim:control-left\nbindn = CTRL, Control_R, global, univim:control-right"
            );
            Ok(())
        }
        "snapshot" | "screenshot" | "capture" => {
            capture_snapshot(output.as_deref(), &directory, options, flat).map(|saved| {
                println!(
                    "Saved snapshot #{}: {} | {} | {}",
                    saved.number,
                    saved.screenshot.display(),
                    saved.rendered.display(),
                    saved.json.display()
                );
                println!("{} nodes, {} drill-down groups", saved.nodes, saved.drills);
            })
        }
        "daemon" | "serve" => {
            let mut settings = RuntimeSettings::load_or_init(options);
            if let Some(value) = min_score {
                settings.detector.min_score = value;
            }
            if no_text {
                settings.detector.include_text = false;
            }
            if fast_half_res {
                settings.detector.fast_half_res = true;
            }
            if no_collapse {
                settings.detector.collapse_paragraphs = false;
            }
            if output.is_some() {
                settings.output = output;
            }
            if no_click {
                settings.no_click = true;
            }
            if flat {
                settings.flat = true;
            }
            if let Some(value) = timeout_secs {
                settings.timeout_secs = value;
            }
            run_daemon(&socket, settings)
        }
        "hints" => send_command_or_autospawn(&socket, CommandAction::Hints, options)
            .map_err(|err| format!("failed to trigger hints on {}: {err}", socket.display())),
        "grid" => send_command_or_autospawn(&socket, CommandAction::Grid, options)
            .map_err(|err| format!("failed to trigger grid on {}: {err}", socket.display())),
        "tap-caps" | "caps" => send_command_or_autospawn(&socket, CommandAction::TapCaps, options)
            .map_err(|err| format!("failed to send tap-caps on {}: {err}", socket.display())),
        "debug" | "toggle-debug" => {
            send_command_or_autospawn(&socket, CommandAction::ToggleDebug, options)
                .map_err(|err| format!("failed to toggle debug on {}: {err}", socket.display()))
        }
        "options" | "toggle-options" => {
            send_command_or_autospawn(&socket, CommandAction::ToggleOptions, options)
                .map_err(|err| format!("failed to toggle options on {}: {err}", socket.display()))
        }
        "stop" => send_command_or_autospawn(&socket, CommandAction::Stop, options)
            .map_err(|err| format!("failed to stop daemon on {}: {err}", socket.display())),
        other => {
            eprintln!("unknown subcommand: {other}");
            print_usage();
            process::exit(2);
        }
    };

    if let Err(err) = result {
        eprintln!("univim-desktop: {err}");
        process::exit(1);
    }
}
