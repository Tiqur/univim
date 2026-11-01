mod draw;
pub mod evdev;
mod geometry;
mod hyprland;
mod session;
#[allow(dead_code)]
mod snapshot;
mod wayland;

use crate::DetectorOptions;
use serde::{Deserialize, Serialize};
use std::{
    env, fs, io,
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Duration,
};

pub use snapshot::Saved;
pub use wayland::{capture_snapshot, check, run_daemon};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandAction {
    Hints,
    Grid,
    DebugOpen,
    TapCaps,
    TapAlt,
    ToggleDebug,
    ToggleOptions,
    Stop,
}

impl CommandAction {
    pub fn as_wire(self) -> &'static str {
        match self {
            Self::Hints => "hints\n",
            Self::Grid => "grid\n",
            Self::DebugOpen => "debug-open\n",
            Self::TapCaps => "tap-caps\n",
            Self::TapAlt => "tap-alt\n",
            Self::ToggleDebug => "toggle-debug\n",
            Self::ToggleOptions => "toggle-options\n",
            Self::Stop => "stop\n",
        }
    }

    pub fn from_wire(text: &str) -> Option<Self> {
        match text.trim() {
            "hints" => Some(Self::Hints),
            "grid" => Some(Self::Grid),
            "debug-open" => Some(Self::DebugOpen),
            "tap-caps" | "tap_caps" | "caps" => Some(Self::TapCaps),
            "tap-alt" | "tap_alt" => Some(Self::TapAlt),
            "toggle-debug" | "debug" => Some(Self::ToggleDebug),
            "toggle-options" | "options" => Some(Self::ToggleOptions),
            "stop" => Some(Self::Stop),
            _ => None,
        }
    }
}

/// Live-editable runtime configuration persisted to `~/.config/univim/settings.json`
/// and adjustable from the in-overlay Options GUI (`F1` / `?`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeSettings {
    pub detector: DetectorOptions,
    /// Explicit output connector name; `None` follows the focused monitor.
    #[serde(default)]
    pub output: Option<String>,
    /// Display normally but suppress all synthetic pointer actions.
    #[serde(default)]
    pub no_click: bool,
    /// Replace composite targets with their leaf descendants.
    #[serde(default)]
    pub flat: bool,
    /// Auto-cancel an overlay after this many seconds.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    /// Maximum interval (ms) between CapsLock taps.
    pub double_tap_ms: u64,
    /// Maximum age (ms) of the speculative 1st-tap pre-computed detection cache.
    pub cache_ttl_ms: u64,
    /// Whether to dim the background screenshot slightly for higher hint contrast.
    pub spotlight_dim: bool,
    /// Whether to listen on `/dev/input/event*` for CapsLock gestures.
    pub use_evdev: bool,
    /// Whether to show the status & telemetry pill at the top-left of the overlay.
    pub show_status_bar: bool,
    #[serde(default)]
    pub pointer_motion: PointerMotion,
    #[serde(default = "default_pointer_speed")]
    pub pointer_speed: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointerMotion {
    #[default]
    Instant,
    Linear,
    EaseOut,
}

fn default_pointer_speed() -> u32 {
    1600
}

fn default_timeout_secs() -> u64 {
    30
}

impl Default for RuntimeSettings {
    fn default() -> Self {
        Self {
            detector: DetectorOptions::default(),
            output: None,
            no_click: false,
            flat: false,
            timeout_secs: default_timeout_secs(),
            double_tap_ms: 260,
            cache_ttl_ms: 900,
            spotlight_dim: true,
            use_evdev: true,
            show_status_bar: true,
            pointer_motion: PointerMotion::Instant,
            pointer_speed: default_pointer_speed(),
        }
    }
}

impl RuntimeSettings {
    pub const OPTION_COUNT: usize = 11;

    pub fn config_path() -> PathBuf {
        if let Some(cfg) = env::var_os("XDG_CONFIG_HOME") {
            return PathBuf::from(cfg).join("univim").join("settings.json");
        }
        if let Some(home) = env::var_os("HOME") {
            return PathBuf::from(home)
                .join(".config")
                .join("univim")
                .join("settings.json");
        }
        PathBuf::from("/tmp/univim-settings.json")
    }

    pub fn load_or_init(base_detector: DetectorOptions) -> Self {
        let path = Self::config_path();
        if let Ok(raw) = fs::read_to_string(&path)
            && let Ok(loaded) = serde_json::from_str::<Self>(&raw)
        {
            return loaded;
        }
        Self {
            detector: base_detector,
            ..Self::default()
        }
    }

    pub fn save(&self) -> io::Result<()> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let body = serde_json::to_string_pretty(self).map_err(io::Error::other)?;
        fs::write(path, body)
    }

    pub fn adjust_option(&mut self, index: usize, delta: i32) {
        match index {
            0 => {
                let stepped = (self.detector.min_score + (delta as f64) * 0.05).clamp(0.15, 0.90);
                self.detector.min_score = (stepped * 100.0).round() / 100.0;
            }
            1 => self.detector.include_text = !self.detector.include_text,
            2 => self.detector.fast_half_res = !self.detector.fast_half_res,
            3 => self.detector.collapse_paragraphs = !self.detector.collapse_paragraphs,
            4 => {
                let next = (self.double_tap_ms as i64 + (delta as i64) * 20).clamp(120, 600);
                self.double_tap_ms = next as u64;
            }
            5 => {
                let next = (self.cache_ttl_ms as i64 + (delta as i64) * 100).clamp(200, 3000);
                self.cache_ttl_ms = next as u64;
            }
            6 => self.spotlight_dim = !self.spotlight_dim,
            7 => self.show_status_bar = !self.show_status_bar,
            8 => self.use_evdev = !self.use_evdev,
            9 => {
                self.pointer_motion = match (self.pointer_motion, delta.signum()) {
                    (PointerMotion::Instant, 1) | (PointerMotion::EaseOut, -1) => {
                        PointerMotion::Linear
                    }
                    (PointerMotion::Linear, 1) => PointerMotion::EaseOut,
                    (PointerMotion::Linear, -1) => PointerMotion::Instant,
                    _ => self.pointer_motion,
                }
            }
            10 => {
                self.pointer_speed =
                    (self.pointer_speed as i64 + delta as i64 * 200).clamp(200, 6000) as u32
            }
            _ => {}
        }
    }
}

pub fn default_socket_path() -> PathBuf {
    if let Some(runtime_dir) = env::var_os("XDG_RUNTIME_DIR") {
        let directory = match env::var("WAYLAND_DISPLAY") {
            Ok(display)
                if !display.is_empty() && !display.contains('/') && !display.contains("..") =>
            {
                format!("univim-{display}")
            }
            _ => "univim".to_string(),
        };
        return PathBuf::from(runtime_dir)
            .join(directory)
            .join("control.sock");
    }
    PathBuf::from("/tmp/univim-control.sock")
}

pub fn send_command(socket: &Path, action: CommandAction) -> io::Result<()> {
    use io::Write;
    let mut stream = UnixStream::connect(socket)?;
    stream.write_all(action.as_wire().as_bytes())?;
    stream.flush()
}

/// Send a command to the running daemon, or automatically spawn the background daemon if
/// the control socket is missing or stale (`ECONNREFUSED`). This makes `univim-desktop hints`
/// or `univim-desktop tap-caps` work out-of-the-box even if the systemd service wasn't started.
pub fn send_command_or_autospawn(
    socket: &Path,
    action: CommandAction,
    options: DetectorOptions,
) -> io::Result<()> {
    match send_command(socket, action) {
        Ok(()) => Ok(()),
        Err(err)
            if matches!(
                err.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
            ) =>
        {
            if action == CommandAction::Stop {
                return Err(err);
            }
            spawn_detached_daemon(socket, options)?;
            for _ in 0..25 {
                std::thread::sleep(Duration::from_millis(20));
                if send_command(socket, action).is_ok() {
                    return Ok(());
                }
            }
            send_command(socket, action)
        }
        Err(err) => Err(err),
    }
}

fn spawn_detached_daemon(socket: &Path, options: DetectorOptions) -> io::Result<()> {
    let exe = env::current_exe()?;
    let mut cmd = Command::new(exe);
    cmd.arg("daemon")
        .arg("--socket")
        .arg(socket)
        .arg("--min-score")
        .arg(options.min_score.to_string());
    if !options.include_text {
        cmd.arg("--no-text");
    }
    if options.fast_half_res {
        cmd.arg("--fast-half-res");
    }
    if !options.collapse_paragraphs {
        cmd.arg("--no-collapse-paragraphs");
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let _child = cmd.spawn()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::RuntimeSettings;

    #[test]
    fn home_manager_settings_json_loads_without_cli_only_fields() {
        let json = r#"{
            "cache_ttl_ms": 900,
            "detector": {
                "collapse_paragraphs": true,
                "fast_half_res": true,
                "include_text": true,
                "min_score": 0.52
            },
            "double_tap_ms": 240,
            "show_status_bar": true,
            "spotlight_dim": true,
            "use_evdev": false
        }"#;
        let settings: RuntimeSettings = serde_json::from_str(json).unwrap();
        assert_eq!(settings.detector.min_score, 0.52);
        assert!(settings.detector.fast_half_res);
        assert!(!settings.detector.turbo);
        assert_eq!(settings.double_tap_ms, 240);
        assert!(!settings.use_evdev);
        assert_eq!(settings.timeout_secs, 30);
        assert!(!settings.no_click);
    }
}
