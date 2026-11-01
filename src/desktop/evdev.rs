//! Optional passive Linux `/dev/input/event*` listener for chord-aware `ff` / `gg`,
//! double-tap `CapsLock`, and rolling `fd` / `jk` detection, plus `/dev/uinput`
//! backspace cleanup and scroll wheel injection.
//!
//! Why this exists:
//! - Double-tapping `CapsLock` (Linux `KEY_CAPSLOCK = 58`) is the ideal zero-side-effect
//!   trigger: tapping `CapsLock` twice toggles CapsLock ON then immediately OFF, and emits
//!   zero printable characters into the active window.
//! - For `ff` / `gg`, tracking all keyboard keys allows a 250 ms pre-typing pause filter
//!   so typing words like *"coffee"*, *"buffer"*, or *"diff"* never triggers hints, and if
//!   `ff` is pressed intentionally from idle, `UinputEmitter::emit_backspaces(2)` can
//!   erase the two `f` characters before the overlay appears.
use super::session::{GestureKey, TapEvent, TapTracker};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::{
        fd::{AsFd, AsRawFd, BorrowedFd},
        unix::fs::OpenOptionsExt,
    },
    path::PathBuf,
    time::{Duration, Instant},
};

const EV_SYN: u16 = 0x00;
const EV_KEY: u16 = 0x01;
const EV_REL: u16 = 0x02;
const SYN_REPORT: u16 = 0x00;
const REL_HWHEEL: u16 = 0x06;
const REL_WHEEL: u16 = 0x08;

const KEY_BACKSPACE: u16 = 14;
const KEY_LEFTSHIFT: u16 = 42;
const KEY_RIGHTSHIFT: u16 = 54;
const KEY_LEFTCTRL: u16 = 29;
const KEY_RIGHTCTRL: u16 = 97;
const KEY_LEFTALT: u16 = 56;
const KEY_RIGHTALT: u16 = 100;
const KEY_LEFTMETA: u16 = 125;
const KEY_RIGHTMETA: u16 = 126;

const KEY_D: u16 = 32;
const KEY_F: u16 = 33;
const KEY_G: u16 = 34;
const KEY_J: u16 = 36;
const KEY_K: u16 = 37;
const KEY_CAPSLOCK: u16 = 58;

/// Minimum quiet period (no other alphanumeric keys pressed) required before the
/// first `f` or `g` tap can start a double-tap sequence. Prevents mid-word double
/// letters (`"coffee"`, `"office"`, `"buffer"`, `"egg"`) from triggering hints.
const PRE_TAP_TYPING_COOLDOWN: Duration = Duration::from_millis(250);

/// Maximum interval between `f` -> `d` or `j` -> `k` for rolling home-row chords.
const ROLLING_CHORD_WINDOW: Duration = Duration::from_millis(120);

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawInputEvent {
    tv_sec: libc::time_t,
    tv_usec: libc::suseconds_t,
    type_: u16,
    code: u16,
    value: i32,
}

pub struct EvdevKeyboardSet {
    devices: Vec<File>,
    shift_down: u8,
    ctrl_down: u8,
    alt_down: u8,
    meta_down: u8,
    last_other_key_at: Option<Instant>,
    pending_chord_lead: Option<(u16, Instant)>,
    uinput: Option<UinputEmitter>,
}

impl EvdevKeyboardSet {
    /// Discover and open readable keyboard devices under `/dev/input/event*` using
    /// `/proc/bus/input/devices` capability bitmasks (`EV=120013` or keyboards with `EV_KEY`).
    pub fn discover() -> Self {
        let mut devices = Vec::new();
        for path in candidate_keyboard_paths() {
            if let Ok(file) = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
                .open(&path)
            {
                devices.push(file);
            }
        }
        Self {
            devices,
            shift_down: 0,
            ctrl_down: 0,
            alt_down: 0,
            meta_down: 0,
            last_other_key_at: None,
            pending_chord_lead: None,
            uinput: UinputEmitter::open().ok(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    pub fn len(&self) -> usize {
        self.devices.len()
    }

    pub fn files(&self) -> &[File] {
        &self.devices
    }

    /// Emit synthetic `KEY_BACKSPACE` taps via `/dev/uinput` to erase characters typed
    /// during a bare `ff`, `fd`, or `jk` trigger in an unfocused text input.
    pub fn emit_cleanup_backspaces(&mut self, count: usize) {
        if let Some(uinput) = self.uinput.as_mut() {
            let _ = uinput.emit_backspaces(count);
        }
    }

    /// Emit synthetic mouse wheel scroll ticks via `/dev/uinput` (`REL_WHEEL` / `REL_HWHEEL`).
    /// Returns `true` if the event was injected via `/dev/uinput`.
    pub fn emit_scroll(&mut self, dx: i32, dy: i32) -> bool {
        if let Some(uinput) = self.uinput.as_mut() {
            uinput.emit_wheel(dx, dy).is_ok()
        } else {
            false
        }
    }

    /// Drain all pending kernel input events across opened devices and feed them into `TapTracker`.
    pub fn drain_events(
        &mut self,
        tap_tracker: &mut TapTracker,
        double_tap_ms: u64,
    ) -> Vec<(GestureKey, TapEvent)> {
        let mut out = Vec::new();
        let event_size = std::mem::size_of::<RawInputEvent>();
        let mut buf = [0u8; std::mem::size_of::<RawInputEvent>() * 16];
        let mut cleanup_backspaces = 0usize;

        for file in &mut self.devices {
            loop {
                match file.read(&mut buf) {
                    Ok(n) if n >= event_size => {
                        let count = n / event_size;
                        for idx in 0..count {
                            let start = idx * event_size;
                            let raw = unsafe {
                                std::ptr::read_unaligned(
                                    buf[start..start + event_size].as_ptr() as *const RawInputEvent
                                )
                            };
                            if raw.type_ != EV_KEY {
                                continue;
                            }
                            match (raw.code, raw.value) {
                                (KEY_LEFTSHIFT | KEY_RIGHTSHIFT, 1) => {
                                    self.shift_down = self.shift_down.saturating_add(1);
                                }
                                (KEY_LEFTSHIFT | KEY_RIGHTSHIFT, 0) => {
                                    self.shift_down = self.shift_down.saturating_sub(1);
                                }
                                (KEY_LEFTCTRL | KEY_RIGHTCTRL, 1) => {
                                    self.ctrl_down = self.ctrl_down.saturating_add(1);
                                }
                                (KEY_LEFTCTRL | KEY_RIGHTCTRL, 0) => {
                                    self.ctrl_down = self.ctrl_down.saturating_sub(1);
                                }
                                (KEY_LEFTALT | KEY_RIGHTALT, 1) => {
                                    self.alt_down = self.alt_down.saturating_add(1);
                                }
                                (KEY_LEFTALT | KEY_RIGHTALT, 0) => {
                                    self.alt_down = self.alt_down.saturating_sub(1);
                                }
                                (KEY_LEFTMETA | KEY_RIGHTMETA, 1) => {
                                    self.meta_down = self.meta_down.saturating_add(1);
                                }
                                (KEY_LEFTMETA | KEY_RIGHTMETA, 0) => {
                                    self.meta_down = self.meta_down.saturating_sub(1);
                                }
                                // Double-tap CapsLock: works immediately with zero typing cooldown
                                // and never emits characters that need backspacing!
                                (KEY_CAPSLOCK, 1)
                                    if self.ctrl_down == 0
                                        && self.alt_down == 0
                                        && self.meta_down == 0 =>
                                {
                                    self.pending_chord_lead = None;
                                    let ev = tap_tracker.on_tap(
                                        GestureKey::CapsLock,
                                        Duration::from_millis(double_tap_ms.max(40)),
                                    );
                                    out.push((GestureKey::CapsLock, ev));
                                }
                                // Bare `f` or `g` press
                                (KEY_F, 1)
                                    if self.shift_down == 0
                                        && self.ctrl_down == 0
                                        && self.alt_down == 0
                                        && self.meta_down == 0 =>
                                {
                                    let now = Instant::now();
                                    let in_typing_stream =
                                        self.last_other_key_at.is_some_and(|t| {
                                            now.saturating_duration_since(t)
                                                < PRE_TAP_TYPING_COOLDOWN
                                        });
                                    if in_typing_stream {
                                        tap_tracker.on_other_key();
                                        self.pending_chord_lead = None;
                                        self.last_other_key_at = Some(now);
                                    } else {
                                        self.pending_chord_lead = Some((KEY_F, now));
                                        let ev = tap_tracker.on_tap(
                                            GestureKey::F,
                                            Duration::from_millis(double_tap_ms.max(40)),
                                        );
                                        if ev == TapEvent::Activate {
                                            cleanup_backspaces = 2;
                                            self.pending_chord_lead = None;
                                        }
                                        out.push((GestureKey::F, ev));
                                    }
                                }
                                (KEY_G, 1)
                                    if self.shift_down == 0
                                        && self.ctrl_down == 0
                                        && self.alt_down == 0
                                        && self.meta_down == 0 =>
                                {
                                    let now = Instant::now();
                                    let in_typing_stream =
                                        self.last_other_key_at.is_some_and(|t| {
                                            now.saturating_duration_since(t)
                                                < PRE_TAP_TYPING_COOLDOWN
                                        });
                                    self.pending_chord_lead = None;
                                    if in_typing_stream {
                                        tap_tracker.on_other_key();
                                        self.last_other_key_at = Some(now);
                                    } else {
                                        let ev = tap_tracker.on_tap(
                                            GestureKey::G,
                                            Duration::from_millis(double_tap_ms.max(40)),
                                        );
                                        if ev == TapEvent::Activate {
                                            cleanup_backspaces = 2;
                                        }
                                        out.push((GestureKey::G, ev));
                                    }
                                }
                                // Rolling `fd` chord: `f` (which already triggered Warmup) followed within 120ms by `d`
                                (KEY_D, 1)
                                    if self.shift_down == 0
                                        && self.ctrl_down == 0
                                        && self.alt_down == 0
                                        && self.meta_down == 0 =>
                                {
                                    let now = Instant::now();
                                    if let Some((KEY_F, lead_at)) = self.pending_chord_lead.take()
                                        && now.saturating_duration_since(lead_at)
                                            <= ROLLING_CHORD_WINDOW
                                    {
                                        tap_tracker.clear();
                                        cleanup_backspaces = 2;
                                        out.push((GestureKey::F, TapEvent::Activate));
                                        continue;
                                    }
                                    tap_tracker.on_other_key();
                                    self.last_other_key_at = Some(now);
                                }
                                // Rolling `jk` chord: `j` followed within 120ms by `k`
                                (KEY_J, 1)
                                    if self.shift_down == 0
                                        && self.ctrl_down == 0
                                        && self.alt_down == 0
                                        && self.meta_down == 0 =>
                                {
                                    let now = Instant::now();
                                    let in_typing_stream =
                                        self.last_other_key_at.is_some_and(|t| {
                                            now.saturating_duration_since(t)
                                                < PRE_TAP_TYPING_COOLDOWN
                                        });
                                    tap_tracker.on_other_key();
                                    if !in_typing_stream {
                                        self.pending_chord_lead = Some((KEY_J, now));
                                        out.push((GestureKey::F, TapEvent::Warmup));
                                    } else {
                                        self.pending_chord_lead = None;
                                        self.last_other_key_at = Some(now);
                                    }
                                }
                                (KEY_K, 1)
                                    if self.shift_down == 0
                                        && self.ctrl_down == 0
                                        && self.alt_down == 0
                                        && self.meta_down == 0 =>
                                {
                                    let now = Instant::now();
                                    if let Some((KEY_J, lead_at)) = self.pending_chord_lead.take()
                                        && now.saturating_duration_since(lead_at)
                                            <= ROLLING_CHORD_WINDOW
                                    {
                                        tap_tracker.clear();
                                        cleanup_backspaces = 2;
                                        out.push((GestureKey::F, TapEvent::Activate));
                                        continue;
                                    }
                                    tap_tracker.on_other_key();
                                    self.last_other_key_at = Some(now);
                                }
                                // Any other non-modifier key down cancels pending double-tap state
                                // and updates the typing stream timestamp.
                                (_, 1) => {
                                    tap_tracker.on_other_key();
                                    self.pending_chord_lead = None;
                                    self.last_other_key_at = Some(Instant::now());
                                }
                                _ => {}
                            }
                        }
                    }
                    Ok(_) => break,
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(_) => break,
                }
            }
        }

        if cleanup_backspaces > 0 {
            self.emit_cleanup_backspaces(cleanup_backspaces);
        }

        out
    }

    pub fn first_fd(&self) -> Option<BorrowedFd<'_>> {
        self.devices.first().map(AsFd::as_fd)
    }
}

/// Lightweight `/dev/uinput` virtual keyboard + scroll wheel emitter.
struct UinputEmitter {
    file: File,
}

#[repr(C)]
struct UinputSetup {
    id: [u16; 4],
    name: [u8; 80],
    ff_effects_max: u32,
}

impl UinputEmitter {
    fn open() -> std::io::Result<Self> {
        let path = if PathBuf::from("/dev/uinput").exists() {
            "/dev/uinput"
        } else {
            "/dev/input/uinput"
        };
        let file = OpenOptions::new()
            .write(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open(path)?;
        let fd = file.as_raw_fd();

        // Constants from <linux/uinput.h>
        const UI_SET_EVBIT: libc::c_ulong = 0x4004_5564;
        const UI_SET_KEYBIT: libc::c_ulong = 0x4004_5565;
        const UI_SET_RELBIT: libc::c_ulong = 0x4004_5566;
        const UI_DEV_SETUP: libc::c_ulong = 0x405c_5503;
        const UI_DEV_CREATE: libc::c_ulong = 0x5501;

        unsafe {
            if libc::ioctl(fd, UI_SET_EVBIT, libc::c_int::from(EV_KEY)) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::ioctl(fd, UI_SET_KEYBIT, libc::c_int::from(KEY_BACKSPACE)) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::ioctl(fd, UI_SET_EVBIT, libc::c_int::from(EV_REL)) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::ioctl(fd, UI_SET_RELBIT, libc::c_int::from(REL_WHEEL)) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::ioctl(fd, UI_SET_RELBIT, libc::c_int::from(REL_HWHEEL)) < 0 {
                return Err(std::io::Error::last_os_error());
            }

            let mut setup = UinputSetup {
                id: [0x03, 0x1209, 0x5556, 1],
                name: [0u8; 80],
                ff_effects_max: 0,
            };
            let label = b"univim-uinput-helper";
            setup.name[..label.len()].copy_from_slice(label);

            if libc::ioctl(fd, UI_DEV_SETUP, &setup as *const UinputSetup) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::ioctl(fd, UI_DEV_CREATE) < 0 {
                return Err(std::io::Error::last_os_error());
            }
        }

        Ok(Self { file })
    }

    fn write_event(&mut self, type_: u16, code: u16, value: i32) -> std::io::Result<()> {
        let ev = RawInputEvent {
            tv_sec: 0,
            tv_usec: 0,
            type_,
            code,
            value,
        };
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (&ev as *const RawInputEvent) as *const u8,
                std::mem::size_of::<RawInputEvent>(),
            )
        };
        self.file.write_all(bytes)
    }

    fn emit_backspaces(&mut self, count: usize) -> std::io::Result<()> {
        for _ in 0..count {
            self.write_event(EV_KEY, KEY_BACKSPACE, 1)?;
            self.write_event(EV_SYN, SYN_REPORT, 0)?;
            self.write_event(EV_KEY, KEY_BACKSPACE, 0)?;
            self.write_event(EV_SYN, SYN_REPORT, 0)?;
        }
        Ok(())
    }

    fn emit_wheel(&mut self, dx: i32, dy: i32) -> std::io::Result<()> {
        if dy != 0 {
            self.write_event(EV_REL, REL_WHEEL, dy)?;
        }
        if dx != 0 {
            self.write_event(EV_REL, REL_HWHEEL, dx)?;
        }
        self.write_event(EV_SYN, SYN_REPORT, 0)
    }
}

impl Drop for UinputEmitter {
    fn drop(&mut self) {
        const UI_DEV_DESTROY: libc::c_ulong = 0x5502;
        unsafe {
            let _ = libc::ioctl(self.file.as_raw_fd(), UI_DEV_DESTROY);
        }
    }
}

fn candidate_keyboard_paths() -> Vec<PathBuf> {
    let Ok(proc_devices) = fs::read_to_string("/proc/bus/input/devices") else {
        return Vec::new();
    };
    parse_proc_bus_input_devices(&proc_devices)
}

fn parse_proc_bus_input_devices(text: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for block in text.split("\n\n") {
        let mut event_name: Option<String> = None;
        let mut is_keyboard = false;
        let mut is_self_uinput = false;
        for line in block.lines() {
            if line.starts_with("N: Name=") && line.contains("univim-uinput-helper") {
                is_self_uinput = true;
            } else if let Some(handlers) = line.strip_prefix("H: Handlers=") {
                if handlers.split_whitespace().any(|tok| tok == "kbd") {
                    for tok in handlers.split_whitespace() {
                        if tok.starts_with("event") {
                            event_name = Some(tok.to_string());
                        }
                    }
                }
            } else if let Some(ev_bits) = line.strip_prefix("B: EV=")
                && let Ok(mask) = u64::from_str_radix(ev_bits.trim(), 16)
                && (mask & 0x120013) == 0x120013
            {
                is_keyboard = true;
            }
        }
        if is_keyboard
            && !is_self_uinput
            && let Some(ev) = event_name
        {
            out.push(PathBuf::from("/dev/input").join(ev));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::parse_proc_bus_input_devices;
    use std::path::PathBuf;

    #[test]
    fn parses_keyboard_handlers_from_proc_bus_input() {
        let sample = "\
I: Bus=0019 Vendor=0000 Product=0001 Version=0000
N: Name=\"Power Button\"
H: Handlers=kbd event0
B: EV=3

I: Bus=0011 Vendor=0001 Product=0001 Version=ab83
N: Name=\"AT Translated Set 2 keyboard\"
H: Handlers=sysrq kbd leds event3
B: EV=120013

I: Bus=0003 Vendor=1209 Product=5556 Version=0001
N: Name=\"univim-uinput-helper\"
H: Handlers=sysrq kbd leds event9
B: EV=120013
";
        let paths = parse_proc_bus_input_devices(sample);
        assert_eq!(paths, vec![PathBuf::from("/dev/input/event3")]);
    }
}
