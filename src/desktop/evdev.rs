//! Optional passive Linux keyboard listener for double-CapsLock gestures.
use super::session::{GestureKey, TapEvent, TapTracker};
use std::{
    fs::{self, File, OpenOptions},
    io::Read,
    os::{
        fd::{AsFd, BorrowedFd},
        unix::fs::OpenOptionsExt,
    },
    path::PathBuf,
    time::Duration,
};

const EV_KEY: u16 = 0x01;

const KEY_LEFTSHIFT: u16 = 42;
const KEY_RIGHTSHIFT: u16 = 54;
const KEY_LEFTCTRL: u16 = 29;
const KEY_RIGHTCTRL: u16 = 97;
const KEY_LEFTALT: u16 = 56;
const KEY_RIGHTALT: u16 = 100;
const KEY_LEFTMETA: u16 = 125;
const KEY_RIGHTMETA: u16 = 126;

const KEY_CAPSLOCK: u16 = 58;

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

    /// Drain all pending kernel input events across opened devices and feed them into `TapTracker`.
    pub fn drain_events(
        &mut self,
        tap_tracker: &mut TapTracker,
        double_tap_ms: u64,
    ) -> Vec<(GestureKey, TapEvent)> {
        let mut out = Vec::new();
        let event_size = std::mem::size_of::<RawInputEvent>();
        let mut buf = [0u8; std::mem::size_of::<RawInputEvent>() * 16];

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
                                (KEY_CAPSLOCK, 1) if self.ctrl_down == 0 && self.meta_down == 0 => {
                                    let key = if self.alt_down > 0 {
                                        GestureKey::Debug
                                    } else if self.shift_down > 0 {
                                        GestureKey::Grid
                                    } else {
                                        GestureKey::CapsLock
                                    };
                                    let ev = tap_tracker
                                        .on_tap(key, Duration::from_millis(double_tap_ms.max(40)));
                                    out.push((key, ev));
                                }
                                // Ordinary typing cancels a pending CapsLock sequence.
                                (_, 1) => {
                                    tap_tracker.on_other_key();
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

        out
    }

    pub fn first_fd(&self) -> Option<BorrowedFd<'_>> {
        self.devices.first().map(AsFd::as_fd)
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
