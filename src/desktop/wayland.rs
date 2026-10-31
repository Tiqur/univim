//! Persistent Wayland runtime. No surface or keyboard focus is held while idle.
//! Selection release draining is bounded to two seconds; expiry cancels without input.
//! Shortcut timing uses compositor timestamps, independent of event delivery delays.
use std::{
    collections::HashSet,
    io,
    os::{fd::AsFd, unix::net::UnixDatagram},
    sync::{Arc, Mutex, mpsc},
    thread,
    time::{Duration, Instant},
};

use super::{
    Config,
    geometry::{FrameGeometry, Output},
    hyprland,
    session::{GestureKey, Input, Mode, Outcome, Session, TapTracker},
};
use nix::poll::{PollFd, PollFlags, poll};
use nix::sys::{
    signal::{SigSet, SigmaskHow, Signal, pthread_sigmask},
    signalfd::{SfdFlags, SignalFd},
};
use opencv::core::{self, Mat, MatTraitConst, MatTraitManual, Scalar};
use serde_json::{Value, json};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_keyboard, delegate_layer, delegate_output, delegate_pointer,
    delegate_registry, delegate_seat, delegate_shm,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    seat::{
        Capability, SeatHandler, SeatState,
        keyboard::{KeyEvent, KeyboardHandler, Keysym, Modifiers, RawModifiers},
        pointer::{PointerEvent, PointerEventKind, PointerHandler},
    },
    shell::{
        WaylandSurface,
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
            LayerSurfaceConfigure,
        },
    },
    shm::{Shm, ShmHandler, raw::RawPool, slot::SlotPool},
};
use wayland_client::{
    Connection, Dispatch, QueueHandle, WEnum,
    globals::registry_queue_init,
    protocol::{
        wl_buffer, wl_callback, wl_keyboard, wl_output, wl_pointer, wl_seat, wl_shm, wl_surface,
    },
};
use wayland_protocols_hyprland::{
    global_shortcuts::v1::client::{
        hyprland_global_shortcut_v1 as shortcut, hyprland_global_shortcuts_manager_v1 as shortcuts,
    },
    lock_notify::v1::client::{
        hyprland_lock_notification_v1 as lock, hyprland_lock_notifier_v1 as locks,
    },
};
use wayland_protocols_wlr::{
    screencopy::v1::client::{
        zwlr_screencopy_frame_v1 as frame, zwlr_screencopy_manager_v1 as capture,
    },
    virtual_pointer::v1::client::{
        zwlr_virtual_pointer_manager_v1 as pointers, zwlr_virtual_pointer_v1 as pointer,
    },
};

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn ready_to_unmap(drain_input: bool, keys: usize, buttons: usize, modifiers: bool) -> bool {
    // Capture-only cancellation has no keyboard focus and cannot receive releases.
    // A mapped overlay drains selection AND cancellation keys before losing focus.
    !drain_input || (keys == 0 && buttons == 0 && !modifiers)
}

fn validate_session(active_work: bool, finishing: bool, syncing: bool, click: bool) -> bool {
    (active_work && !finishing && !syncing) || click
}

fn release_drain_expired(deadline: &mut Option<Instant>, now: Instant) -> bool {
    if deadline.is_some_and(|end| now >= end) {
        *deadline = None;
        true
    } else {
        false
    }
}

fn gesture_event(
    tracker: &mut TapTracker,
    key: GestureKey,
    event: shortcut::Event,
) -> Option<Mode> {
    let (pressed, hi, lo, nanos) = match event {
        shortcut::Event::Pressed {
            tv_sec_hi,
            tv_sec_lo,
            tv_nsec,
        } => (true, tv_sec_hi, tv_sec_lo, tv_nsec),
        shortcut::Event::Released {
            tv_sec_hi,
            tv_sec_lo,
            tv_nsec,
        } => (false, tv_sec_hi, tv_sec_lo, tv_nsec),
        _ => return None,
    };
    if nanos >= 1_000_000_000 {
        tracker.reset();
        return None;
    }
    let time = Duration::new((u64::from(hi) << 32) | u64::from(lo), nanos);
    // TapTracker rejects backward protocol timestamps and invalidates the gesture.
    if pressed {
        tracker.press(key, time);
        None
    } else {
        tracker.release(key, time)
    }
}

/// Enumeration only: deliberately does not bind input extensions or create surfaces.
pub fn check() -> Result<Value, String> {
    let conn = Connection::connect_to_env().map_err(error)?;
    let (globals, mut queue) = registry_queue_init::<State>(&conn).map_err(error)?;
    let qh = queue.handle();
    let mut state = State::new(&globals, &qh)?;
    queue.roundtrip(&mut state).map_err(error)?;
    let protocols = globals.contents().with_list(|list| {
        list.iter()
            .map(|g| json!({"interface":g.interface,"version":g.version}))
            .collect::<Vec<_>>()
    });
    let outputs = state
        .outputs
        .outputs()
        .filter_map(|o| state.outputs.info(&o).and_then(|i| i.name))
        .collect::<Vec<_>>();
    Ok(json!({"protocols":protocols,"outputs":outputs,
        "gesture_limitation":"Anonymous compositor shortcuts cannot observe arbitrary intervening chords; configure nonconsuming modifier-only press/release bindings."}))
}

/// One-shot native capture for numbered diagnostic snapshots. No surface or
/// input object is created; the frame is decoded upright into owned BGR memory.
pub struct Captured {
    pub image: Mat,
    pub targets: Vec<crate::Target>,
    pub output: Output,
    pub timings: Value,
}

pub fn capture(output_name: Option<&str>, options: crate::Options) -> Result<Captured, String> {
    let conn = Connection::connect_to_env().map_err(error)?;
    let (globals, mut queue) = registry_queue_init::<State>(&conn).map_err(error)?;
    let qh = queue.handle();
    let mut state = State::new(&globals, &qh)?;
    state.capture_manager = Some(globals.bind(&qh, 3..=3, ()).map_err(error)?);
    queue.roundtrip(&mut state).map_err(error)?;

    let output = hyprland::selected_output(output_name)?;
    let (logical_w, logical_h) = output.logical_size()?;
    if !logical_w.is_finite()
        || !logical_h.is_finite()
        || logical_w < 1.
        || logical_h < 1.
        || logical_w > 32768.
        || logical_h > 32768.
    {
        return Err("invalid output logical size".into());
    }
    let wl_output = state
        .outputs
        .outputs()
        .find(|candidate| {
            state
                .outputs
                .info(candidate)
                .is_some_and(|info| info.name.as_deref() == Some(output.name.as_str()))
        })
        .ok_or("selected output is absent from Wayland registry")?;
    state.output = Some(wl_output.clone());
    state.geometry = Some(FrameGeometry {
        output: output.clone(),
        pixel_width: 1,
        pixel_height: 1,
    });
    state.size = (logical_w.round() as u32, logical_h.round() as u32);
    let manager = state
        .capture_manager
        .as_ref()
        .ok_or("screencopy unavailable")?;
    state.capture = Some(Capture {
        frame: manager.capture_output(0, &wl_output, &qh, ()),
        pool: None,
        buffer: None,
        format: None,
        width: 0,
        height: 0,
        stride: 0,
        inverted: false,
        started: Instant::now(),
    });

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        queue.dispatch_pending(&mut state).map_err(error)?;
        if let Some(failure) = state.failure.take() {
            return Err(failure);
        }
        if state.captured.is_some() {
            break;
        }
        if Instant::now() >= deadline {
            return Err("snapshot capture timed out".into());
        }
        let blocked = match conn.flush() {
            Ok(()) => false,
            Err(wayland_client::backend::WaylandError::Io(ref e))
                if e.kind() == io::ErrorKind::WouldBlock =>
            {
                true
            }
            Err(e) => return Err(error(e)),
        };
        let Some(guard) = queue.prepare_read() else {
            continue;
        };
        let mut flags = PollFlags::POLLIN;
        if blocked {
            flags |= PollFlags::POLLOUT;
        }
        let mut fds = [PollFd::new(conn.as_fd(), flags)];
        poll(&mut fds, Some(250u16)).map_err(error)?;
        let revents = fds[0].revents().unwrap_or(PollFlags::empty());
        if revents.intersects(PollFlags::POLLERR | PollFlags::POLLHUP | PollFlags::POLLNVAL) {
            drop(guard);
            return Err("Wayland connection closed during snapshot".into());
        }
        if revents.contains(PollFlags::POLLIN) {
            match guard.read() {
                Ok(_) => {}
                Err(wayland_client::backend::WaylandError::Io(ref e))
                    if e.kind() == io::ErrorKind::WouldBlock => {}
                Err(e) => return Err(error(e)),
            }
        } else {
            drop(guard);
        }
    }

    let image = state.captured.take().ok_or("snapshot produced no image")?;
    if let Some(geometry) = state.geometry.as_mut() {
        geometry.pixel_width = image.cols();
        geometry.pixel_height = image.rows();
        let output = geometry.output.clone();
        let (targets, timings) = crate::detect(&image, options).map_err(error)?;
        return Ok(Captured {
            image,
            targets,
            output,
            timings,
        });
    }
    Err("snapshot geometry is missing".into())
}

struct Capture {
    frame: frame::ZwlrScreencopyFrameV1,
    pool: Option<RawPool>,
    buffer: Option<wl_buffer::WlBuffer>,
    format: Option<wl_shm::Format>,
    width: u32,
    height: u32,
    stride: u32,
    inverted: bool,
    started: Instant,
}
impl Drop for Capture {
    fn drop(&mut self) {
        self.frame.destroy();
        if let Some(buffer) = self.buffer.take() {
            buffer.destroy();
        }
    }
}

struct Work {
    generation: u64,
    image: Mat,
    options: crate::Options,
}
struct Detection {
    generation: u64,
    result: Result<(Mat, Vec<crate::Target>), String>,
}

// One running job plus one replaceable pending job; cancellation never spawns threads.
struct Worker {
    pending: Arc<Mutex<Option<Work>>>,
    trigger: Option<mpsc::SyncSender<()>>,
    results: Arc<Mutex<Option<Detection>>>,
    wake: UnixDatagram,
    join: Option<thread::JoinHandle<()>>,
}
impl Worker {
    fn new() -> Result<Self, String> {
        let (wake, notify) = UnixDatagram::pair().map_err(error)?;
        wake.set_nonblocking(true).map_err(error)?;
        notify.set_nonblocking(true).map_err(error)?;
        let pending = Arc::new(Mutex::new(None::<Work>));
        let shared = pending.clone();
        let (trigger, jobs) = mpsc::sync_channel(1);
        let results = Arc::new(Mutex::new(None));
        let send = results.clone();
        let join = thread::spawn(move || {
            while jobs.recv().is_ok() {
                let work = shared.lock().ok().and_then(|mut p| p.take());
                let Some(work) = work else { continue };
                let start = Instant::now();
                let result = crate::detect(&work.image, work.options)
                    .map(|(targets, timing)| {
                        eprintln!(
                            "univim detection timing={timing} elapsed_ms={:.1}",
                            start.elapsed().as_secs_f64() * 1000.
                        );
                        (work.image, targets)
                    })
                    .map_err(error);
                // Replace stale results without ever blocking on the event loop.
                if let Ok(mut slot) = send.lock() {
                    *slot = Some(Detection {
                        generation: work.generation,
                        result,
                    });
                    let _ = notify.send(&[1]);
                }
            }
        });
        Ok(Self {
            pending,
            trigger: Some(trigger),
            results,
            wake,
            join: Some(join),
        })
    }
    fn submit(&self, work: Work) -> Result<(), String> {
        *self.pending.lock().map_err(error)? = Some(work);
        if let Some(trigger) = &self.trigger {
            let _ = trigger.try_send(());
        }
        Ok(())
    }
    fn clear(&self) {
        if let Ok(mut p) = self.pending.lock() {
            *p = None;
        }
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.clear();
        self.trigger.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

struct State {
    registry: RegistryState,
    seats: SeatState,
    outputs: OutputState,
    compositor: CompositorState,
    layers: LayerShell,
    shm: Shm,
    pool: SlotPool,
    keyboard: Option<wl_keyboard::WlKeyboard>,
    physical_pointer: Option<wl_pointer::WlPointer>,
    input_enabled: bool,
    layer: Option<LayerSurface>,
    configured: bool,
    size: (u32, u32),
    dirty: bool,
    color_by_type: bool,
    capture_manager: Option<capture::ZwlrScreencopyManagerV1>,
    pointer_manager: Option<pointers::ZwlrVirtualPointerManagerV1>,
    shortcuts: Vec<shortcut::HyprlandGlobalShortcutV1>,
    notification: Option<lock::HyprlandLockNotificationV1>,
    capture: Option<Capture>,
    captured: Option<Mat>,
    session: Option<Session>,
    background: Option<Mat>,
    geometry: Option<FrameGeometry>,
    output: Option<wl_output::WlOutput>,
    context: Option<hyprland::Context>,
    mode: Option<Mode>,
    requested: Option<Mode>,
    generation: u64,
    pressed: HashSet<u32>,
    buttons: HashSet<u32>,
    modifiers: bool,
    finishing: bool,
    release_deadline: Option<Instant>,
    click: Option<(f64, f64)>,
    sync_pending: bool,
    locked: bool,
    quit: bool,
    failure: Option<String>,
    invalidated: bool,
    tracker: TapTracker,
    clock: Instant,
    active_since: Instant,
    context_checked: Instant,
    overlay_committed: bool,
    session_count: u64,
}
impl State {
    fn new(
        globals: &wayland_client::globals::GlobalList,
        qh: &QueueHandle<Self>,
    ) -> Result<Self, String> {
        let shm = Shm::bind(globals, qh).map_err(error)?;
        let pool = SlotPool::new(4096, &shm).map_err(error)?;
        Ok(Self {
            registry: RegistryState::new(globals),
            seats: SeatState::new(globals, qh),
            outputs: OutputState::new(globals, qh),
            compositor: CompositorState::bind(globals, qh).map_err(error)?,
            layers: LayerShell::bind(globals, qh).map_err(error)?,
            shm,
            pool,
            keyboard: None,
            physical_pointer: None,
            input_enabled: false,
            layer: None,
            configured: false,
            size: (1, 1),
            dirty: false,
            color_by_type: false,
            capture_manager: None,
            pointer_manager: None,
            shortcuts: vec![],
            notification: None,
            capture: None,
            captured: None,
            session: None,
            background: None,
            geometry: None,
            output: None,
            context: None,
            mode: None,
            requested: None,
            generation: 0,
            pressed: HashSet::new(),
            buttons: HashSet::new(),
            modifiers: false,
            finishing: false,
            release_deadline: None,
            click: None,
            sync_pending: false,
            locked: false,
            quit: false,
            failure: None,
            invalidated: false,
            tracker: TapTracker::default(),
            clock: Instant::now(),
            active_since: Instant::now(),
            context_checked: Instant::now(),
            overlay_committed: false,
            session_count: 0,
        })
    }
    fn cancel(&mut self) {
        let drain_input = !self.finishing
            && self.layer.is_some()
            && (!self.pressed.is_empty() || !self.buttons.is_empty() || self.modifiers);
        self.generation = self.generation.wrapping_add(1);
        self.capture.take();
        self.captured = None;
        self.session = None;
        self.background = None;
        self.dirty = false;
        self.click = None;
        self.release_deadline = drain_input.then(|| Instant::now() + Duration::from_secs(2));
        self.finishing = self.mode.is_some();
        self.tracker.reset();
    }
    fn unmap(&mut self, conn: &Connection, qh: &QueueHandle<Self>) {
        if let Some(layer) = self.layer.take() {
            let surface = layer.wl_surface().clone();
            layer.wl_surface().attach(None, 0, 0);
            layer.commit();
            drop(layer);
            surface.destroy();
        }
        self.configured = false;
        self.release_deadline = None;
        self.dirty = false;
        self.pressed.clear();
        self.buttons.clear();
        self.modifiers = false;
        self.sync_pending = true;
        conn.display().sync(qh, ());
    }
    fn begin(&mut self, mode: Mode, config: &Config, qh: &QueueHandle<Self>) -> Result<(), String> {
        if self.locked {
            return Err("session is locked".into());
        }
        if self.keyboard.is_none() {
            return Err("no Wayland keyboard is available for exclusive cancellation".into());
        }
        let context = hyprland::context()?;
        let output = hyprland::selected_output(config.output.as_deref())?;
        let (w, h) = output.logical_size()?;
        if !w.is_finite() || !h.is_finite() || w < 1. || h < 1. || w > 32768. || h > 32768. {
            return Err("invalid output logical size".into());
        }
        let wl_output = self
            .outputs
            .outputs()
            .find(|o| {
                self.outputs
                    .info(o)
                    .is_some_and(|i| i.name.as_deref() == Some(output.name.as_str()))
            })
            .ok_or("selected output is absent from Wayland registry")?;
        if !hyprland::still_current(&context)? {
            return Err("context changed before capture".into());
        }
        self.generation = self.generation.wrapping_add(1);
        self.mode = Some(mode);
        self.context = Some(context);
        self.output = Some(wl_output.clone());
        self.geometry = Some(FrameGeometry {
            output,
            pixel_width: 1,
            pixel_height: 1,
        });
        self.size = (w.round() as u32, h.round() as u32);
        self.active_since = Instant::now();
        self.overlay_committed = false;
        self.session_count = self.session_count.wrapping_add(1);
        self.context_checked = Instant::now();
        self.invalidated = false;
        self.finishing = false;
        self.release_deadline = None;
        self.click = None;
        self.dirty = false;
        // Capture must finish before even the loading surface is created.
        let manager = self
            .capture_manager
            .as_ref()
            .ok_or("screencopy unavailable")?;
        self.capture = Some(Capture {
            frame: manager.capture_output(0, &wl_output, qh, ()),
            pool: None,
            buffer: None,
            format: None,
            width: 0,
            height: 0,
            stride: 0,
            inverted: false,
            started: Instant::now(),
        });
        Ok(())
    }
    fn show(&mut self, qh: &QueueHandle<Self>) {
        let surface = self.compositor.create_surface(qh);
        let layer = self.layers.create_layer_surface(
            qh,
            surface,
            Layer::Overlay,
            Some("univim"),
            self.output.as_ref(),
        );
        layer.set_anchor(Anchor::TOP | Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT);
        layer.set_exclusive_zone(-1);
        layer.set_keyboard_interactivity(KeyboardInteractivity::Exclusive);
        layer.set_size(self.size.0, self.size.1);
        layer.commit();
        self.layer = Some(layer);
    }
    fn draw(&mut self) -> Result<(), String> {
        if !self.configured || self.finishing {
            return Ok(());
        }
        let Some(layer) = &self.layer else {
            return Ok(());
        };
        let (w, h) = self.size;
        let loading;
        let session = if let Some(session) = &self.session {
            session
        } else {
            loading = Session::hints(w as i32, h as i32, vec![]);
            &loading
        };
        let pixels =
            super::draw::render(session, self.background.as_ref(), w, h, self.color_by_type)?;
        let expected = w as usize * h as usize * 4;
        if pixels.len() != expected {
            return Err("renderer returned wrong buffer size".into());
        }
        let (buffer, canvas) = self
            .pool
            .create_buffer(w as i32, h as i32, w as i32 * 4, wl_shm::Format::Argb8888)
            .map_err(error)?;
        canvas.copy_from_slice(&pixels);
        layer.wl_surface().damage_buffer(0, 0, w as i32, h as i32);
        buffer.attach_to(layer.wl_surface()).map_err(error)?;
        layer.commit();
        if !self.overlay_committed {
            eprintln!(
                "univim overlay_commit_ms={:.1} loading={}",
                self.active_since.elapsed().as_secs_f64() * 1000.,
                self.session.is_none()
            );
            self.overlay_committed = true;
        }
        self.dirty = false;
        Ok(())
    }
    fn key(&mut self, event: KeyEvent) {
        if self.finishing {
            return;
        }
        let input = match event.keysym {
            Keysym::Escape => Some(Input::Escape),
            Keysym::BackSpace => Some(Input::Backspace),
            Keysym::Tab => Some(Input::Tab),
            Keysym::Return | Keysym::KP_Enter => Some(Input::Enter),
            Keysym::Left => Some(Input::Parent),
            _ => event.utf8.as_deref().and_then(|s| {
                let mut c = s.chars();
                let first = c.next()?;
                (c.next().is_none() && !first.is_control()).then_some(Input::Character(first))
            }),
        };
        if event.keysym == Keysym::Escape {
            self.cancel();
            return;
        }
        if let (Some(input), Some(session)) = (input, self.session.as_mut()) {
            match session.input(input) {
                Outcome::None => {}
                Outcome::Changed => self.dirty = true,
                Outcome::Cancel => self.cancel(),
                Outcome::Click { x, y } => {
                    self.click = Some((x, y));
                    self.finishing = true;
                    self.release_deadline = Some(Instant::now() + Duration::from_secs(2));
                }
            }
        }
    }
}

pub fn run(config: Config, control: UnixDatagram) -> Result<(), String> {
    // Block before creating the worker so it inherits the mask. Consume shutdown
    // signals through poll, allowing normal surface/socket cleanup on SIGTERM.
    let mut shutdown = SigSet::empty();
    shutdown.add(Signal::SIGINT);
    shutdown.add(Signal::SIGTERM);
    let mut previous = SigSet::empty();
    pthread_sigmask(SigmaskHow::SIG_BLOCK, Some(&shutdown), Some(&mut previous)).map_err(error)?;
    struct RestoreMask(SigSet);
    impl Drop for RestoreMask {
        fn drop(&mut self) {
            let _ = pthread_sigmask(SigmaskHow::SIG_SETMASK, Some(&self.0), None);
        }
    }
    let _restore = RestoreMask(previous);
    let signals = SignalFd::with_flags(&shutdown, SfdFlags::SFD_NONBLOCK | SfdFlags::SFD_CLOEXEC)
        .map_err(error)?;
    control.set_nonblocking(true).map_err(error)?;
    let conn = Connection::connect_to_env().map_err(error)?;
    let (globals, mut queue) = registry_queue_init::<State>(&conn).map_err(error)?;
    let qh = queue.handle();
    let mut state = State::new(&globals, &qh)?;
    state.input_enabled = true;
    state.color_by_type = config.color_by_type;
    state.capture_manager = Some(globals.bind(&qh, 3..=3, ()).map_err(error)?);
    state.pointer_manager = Some(globals.bind(&qh, 2..=2, ()).map_err(error)?);
    let manager: shortcuts::HyprlandGlobalShortcutsManagerV1 =
        globals.bind(&qh, 1..=1, ()).map_err(error)?;
    for (id, key) in [
        ("shift-left", GestureKey::ShiftLeft),
        ("shift-right", GestureKey::ShiftRight),
        ("control-left", GestureKey::ControlLeft),
        ("control-right", GestureKey::ControlRight),
    ] {
        state.shortcuts.push(manager.register_shortcut(
            id.into(),
            "univim".into(),
            "UniVim modifier tap".into(),
            "".into(),
            &qh,
            key,
        ));
    }
    let notifier: locks::HyprlandLockNotifierV1 = globals.bind(&qh, 1..=1, ()).map_err(error)?;
    state.notification = Some(notifier.get_lock_notification(&qh, ()));
    queue.roundtrip(&mut state).map_err(error)?;
    eprintln!(
        "univim: idle native runtime; anonymous shortcuts cannot cancel taps on arbitrary intervening chords; use nonconsuming modifier-only compositor bindings"
    );
    let worker = Worker::new()?;
    let result = (|| -> Result<(), String> {
        loop {
            queue.dispatch_pending(&mut state).map_err(error)?;
            if signals.read_signal().map_err(error)?.is_some() {
                state.quit = true;
                state.requested = None;
                state.cancel();
            }
            let mut command = [0u8; 64];
            // Bound each drain so a local sender cannot starve Wayland or cancellation.
            for _ in 0..32 {
                match control.recv_from(&mut command) {
                    Ok((len, address)) => {
                        match std::str::from_utf8(&command[..len]).unwrap_or("").trim() {
                            "hints" => state.requested = Some(Mode::Hints),
                            "grid" => state.requested = Some(Mode::Grid),
                            "cancel" => {
                                state.requested = None;
                                state.cancel();
                                worker.clear();
                            }
                            "quit" => {
                                state.quit = true;
                                state.requested = None;
                                state.cancel();
                            }
                            "status" => {
                                if let Some(path) = address.as_pathname() {
                                    let reply = json!({"active":state.mode.is_some(),"locked":state.locked,"loading":!state.finishing && (state.capture.is_some()||(state.mode.is_some()&&state.session.is_none())),"no_click":config.no_click,"flat":config.flat,"color_by_type":config.color_by_type,
                                        "mode":state.mode.map(|m| match m { Mode::Hints => "hints", Mode::Grid => "grid" }),
                                        "output":state.geometry.as_ref().map(|g| &g.output.name),
                                        "session_count":state.session_count,"finishing":state.finishing,
                                        "overlay_mapped":state.layer.is_some() && state.configured && state.overlay_committed,
                                        "capture_pending":state.capture.is_some(),"session_ready":state.session.is_some(),
                                        "age_ms":state.active_since.elapsed().as_millis(),"sync_pending":state.sync_pending});
                                    let _ = control.send_to(reply.to_string().as_bytes(), path);
                                }
                            }
                            _ => {}
                        }
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                    Err(e) => return Err(error(e)),
                }
            }
            if release_drain_expired(&mut state.release_deadline, Instant::now()) {
                state.requested = None;
                state.cancel();
                worker.clear();
            }
            // Once the teardown sync is acknowledged, finalize below before
            // evaluating stale context/timeouts again. Otherwise an expired
            // session repeatedly schedules sync and can never return to idle.
            let active_work =
                state.layer.is_some() || state.capture.is_some() || state.captured.is_some();
            if state.invalidated
                || (state.mode.is_some()
                    && validate_session(
                        active_work,
                        state.finishing,
                        state.sync_pending,
                        state.click.is_some(),
                    )
                    && state.active_since.elapsed() >= config.timeout)
            {
                state.requested = None;
                state.cancel();
                state.invalidated = false;
                worker.clear();
            }
            if state
                .capture
                .as_ref()
                .is_some_and(|c| c.started.elapsed() > Duration::from_secs(3))
            {
                eprintln!("univim: capture timed out");
                state.cancel();
            }
            if state.mode.is_some()
                && validate_session(
                    active_work,
                    state.finishing,
                    state.sync_pending,
                    state.click.is_some(),
                )
                && state.context_checked.elapsed() >= Duration::from_millis(250)
            {
                state.context_checked = Instant::now();
                if !state
                    .context
                    .as_ref()
                    .is_some_and(|c| hyprland::still_current(c).unwrap_or(false))
                {
                    state.requested = None;
                    state.cancel();
                }
            }
            if let Some(failure) = state.failure.take() {
                eprintln!("univim: {failure}");
                state.cancel();
            }
            if state.requested.is_some() && state.mode.is_some() && !state.finishing {
                state.cancel();
                worker.clear();
            }
            if state.finishing
                && ready_to_unmap(
                    state.click.is_some() || state.release_deadline.is_some(),
                    state.pressed.len(),
                    state.buttons.len(),
                    state.modifiers,
                )
                && !state.sync_pending
            {
                state.unmap(&conn, &qh);
                state.finishing = false;
            }
            // A display sync acknowledges destruction before either capture or injection.
            if state.mode.is_some()
                && state.layer.is_none()
                && state.capture.is_none()
                && state.captured.is_none()
                && !state.sync_pending
                && !state.finishing
            {
                if let Some((x, y)) = state.click.take() {
                    let valid = !state.locked
                        && state
                            .context
                            .as_ref()
                            .is_some_and(|c| hyprland::still_current(c).unwrap_or(false));
                    if valid
                        && let (Some(geometry), Some(output)) = (&state.geometry, &state.output)
                    {
                        let (lx, ly) = geometry.pixel_to_logical(x, y)?;
                        let (w, h) = geometry.output.logical_size()?;
                        if lx.is_finite()
                            && ly.is_finite()
                            && lx >= 0.
                            && ly >= 0.
                            && lx < w
                            && ly < h
                        {
                            if config.no_click {
                                eprintln!(
                                    "univim selection {}",
                                    json!({"x":lx,"y":ly,"injected":false})
                                );
                            } else {
                                let pointer = state
                                    .pointer_manager
                                    .as_ref()
                                    .ok_or("virtual pointer unavailable")?
                                    .create_virtual_pointer_with_output(
                                        None,
                                        Some(output),
                                        &qh,
                                        (),
                                    );
                                let time = state.clock.elapsed().as_millis() as u32;
                                const EXTENT: u32 = 1_000_000;
                                pointer.motion_absolute(
                                    time,
                                    (lx / w * EXTENT as f64).round() as u32,
                                    (ly / h * EXTENT as f64).round() as u32,
                                    EXTENT,
                                    EXTENT,
                                );
                                pointer.frame();
                                pointer.button(time, 0x110, wl_pointer::ButtonState::Pressed);
                                pointer.frame();
                                pointer.button(time, 0x110, wl_pointer::ButtonState::Released);
                                pointer.frame();
                                pointer.destroy();
                            }
                        }
                    }
                }
                state.mode = None;
                state.session = None;
                state.background = None;
                state.context = None;
                state.geometry = None;
                state.output = None;
            }
            if !state.sync_pending
                && state.mode.is_none()
                && !state.quit
                && let Some(mode) = state.requested.take()
                && let Err(e) = state.begin(mode, &config, &qh)
            {
                eprintln!("univim: activation rejected: {e}");
                state.cancel();
            }
            if let Some(image) = state.captured.take()
                && !state.locked
                && !state.finishing
            {
                let (w, h) = (image.cols(), image.rows());
                if let Some(geometry) = state.geometry.as_mut() {
                    geometry.pixel_width = w;
                    geometry.pixel_height = h;
                }
                state.show(&qh);
                match state.mode {
                    Some(Mode::Hints) => worker.submit(Work {
                        generation: state.generation,
                        image,
                        options: config.options,
                    })?,
                    Some(Mode::Grid) => {
                        state.session = Some(Session::grid(w, h));
                        state.background = Some(image);
                        state.dirty = true;
                    }
                    None => {}
                }
            }
            let mut wake = [0u8; 32];
            while worker.wake.recv(&mut wake).is_ok() {}
            let result = worker.results.lock().map_err(error)?.take();
            if let Some(result) = result
                && result.generation == state.generation
                && !state.finishing
                && state.mode.is_some()
            {
                match result.result {
                    Ok((image, targets)) => {
                        let targets = if config.flat {
                            crate::flatten_leaves(&targets).map_err(error)?
                        } else {
                            targets
                        };
                        state.session = Some(Session::hints(image.cols(), image.rows(), targets));
                        state.background = Some(image);
                        state.dirty = true;
                    }
                    Err(e) => {
                        eprintln!("univim: detection failed: {e}");
                        state.cancel();
                    }
                }
            }
            if state.dirty
                && let Err(e) = state.draw()
            {
                eprintln!("univim: render failed: {e}");
                state.cancel();
            }
            // A full nonblocking Wayland socket is backpressure, not a lost
            // connection. Wait for writability before retrying the flush.
            let blocked = match conn.flush() {
                Ok(()) => false,
                Err(wayland_client::backend::WaylandError::Io(ref e))
                    if e.kind() == io::ErrorKind::WouldBlock =>
                {
                    true
                }
                Err(e) => return Err(error(e)),
            };
            if state.quit && state.mode.is_none() && !state.sync_pending {
                break;
            }
            let Some(guard) = queue.prepare_read() else {
                continue;
            };
            let mut fds = [
                PollFd::new(
                    conn.as_fd(),
                    PollFlags::POLLIN
                        | if blocked {
                            PollFlags::POLLOUT
                        } else {
                            PollFlags::empty()
                        },
                ),
                PollFd::new(control.as_fd(), PollFlags::POLLIN),
                PollFd::new(worker.wake.as_fd(), PollFlags::POLLIN),
                PollFd::new(signals.as_fd(), PollFlags::POLLIN),
            ];
            let timeout = if state.mode.is_some() {
                Some(250u16)
            } else {
                None
            };
            match poll(&mut fds, timeout) {
                Ok(_) => {}
                Err(nix::errno::Errno::EINTR) => {
                    drop(guard);
                    continue;
                }
                Err(e) => return Err(error(e)),
            }
            if fds[0]
                .revents()
                .unwrap_or(PollFlags::empty())
                .intersects(PollFlags::POLLERR | PollFlags::POLLHUP | PollFlags::POLLNVAL)
            {
                return Err("Wayland connection closed".into());
            }
            if fds[0]
                .revents()
                .unwrap_or(PollFlags::empty())
                .contains(PollFlags::POLLIN)
            {
                match guard.read() {
                    Ok(_) => {}
                    Err(wayland_client::backend::WaylandError::Io(ref e))
                        if e.kind() == io::ErrorKind::WouldBlock => {}
                    Err(e) => return Err(error(e)),
                }
            } else {
                drop(guard);
            }
        }
        Ok(())
    })();
    state.capture.take();
    if let Some(layer) = state.layer.take() {
        let surface = layer.wl_surface().clone();
        drop(layer);
        surface.destroy();
    }
    let _ = conn.flush();
    drop(worker);
    result
}

impl Dispatch<frame::ZwlrScreencopyFrameV1, ()> for State {
    fn event(
        state: &mut Self,
        proxy: &frame::ZwlrScreencopyFrameV1,
        event: frame::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        let Some(c) = state.capture.as_mut().filter(|c| c.frame == *proxy) else {
            return;
        };
        let result = (|| -> Result<(), String> {
            match event {
                frame::Event::Buffer {
                    format,
                    width,
                    height,
                    stride,
                } => {
                    let WEnum::Value(format) = format else {
                        return Err("unknown screencopy format".into());
                    };
                    if width == 0
                        || height == 0
                        || width > 32768
                        || height > 32768
                        || stride < width * 4
                        || stride > i32::MAX as u32
                        || u64::from(stride) * u64::from(height) > 512 * 1024 * 1024
                    {
                        return Err("invalid or oversized screencopy buffer".into());
                    }
                    if !supported(format) {
                        return Err(format!("unsupported screencopy format: {format:?}"));
                    }
                    c.format = Some(format);
                    c.width = width;
                    c.height = height;
                    c.stride = stride;
                }
                frame::Event::BufferDone => {
                    let format = c.format.ok_or("screencopy has no supported SHM format")?;
                    let mut pool = RawPool::new(c.stride as usize * c.height as usize, &state.shm)
                        .map_err(error)?;
                    let buffer = pool.create_buffer(
                        0,
                        c.width as i32,
                        c.height as i32,
                        c.stride as i32,
                        format,
                        (),
                        qh,
                    );
                    c.frame.copy(&buffer);
                    c.buffer = Some(buffer);
                    c.pool = Some(pool);
                }
                frame::Event::Flags { flags } => {
                    let WEnum::Value(flags) = flags else {
                        return Err("unknown capture flags".into());
                    };
                    c.inverted = flags.contains(frame::Flags::YInvert);
                }
                frame::Event::Ready { .. } => {
                    let geometry = state.geometry.as_ref().ok_or("capture has no geometry")?;
                    if c.width as i32 != geometry.output.width
                        || c.height as i32 != geometry.output.height
                    {
                        return Err("capture dimensions disagree with Hyprland native mode; refusing ambiguous orientation".into());
                    }
                    let pool = c.pool.as_mut().ok_or("capture has no SHM pool")?;
                    state.captured = Some(decode(
                        pool.mmap(),
                        c.width,
                        c.height,
                        c.stride,
                        c.format.ok_or("missing format")?,
                        c.inverted,
                        geometry.output.transform,
                    )?);
                    eprintln!(
                        "univim capture_ms={:.1}",
                        c.started.elapsed().as_secs_f64() * 1000.
                    );
                }
                frame::Event::Failed => return Err("compositor rejected screencopy".into()),
                _ => {}
            }
            Ok(())
        })();
        if let Err(e) = result {
            state.failure = Some(e);
            state.capture.take();
        } else if state.captured.is_some() {
            state.capture.take();
        }
    }
}

fn supported(format: wl_shm::Format) -> bool {
    matches!(
        format,
        wl_shm::Format::Argb8888
            | wl_shm::Format::Xrgb8888
            | wl_shm::Format::Abgr8888
            | wl_shm::Format::Xbgr8888
            | wl_shm::Format::Argb2101010
            | wl_shm::Format::Xrgb2101010
            | wl_shm::Format::Abgr2101010
            | wl_shm::Format::Xbgr2101010
    )
}

fn decode(
    bytes: &[u8],
    w: u32,
    h: u32,
    stride: u32,
    format: wl_shm::Format,
    inverted: bool,
    transform: i32,
) -> Result<Mat, String> {
    if !supported(format)
        || w == 0
        || h == 0
        || w > 32768
        || h > 32768
        || u64::from(stride) < u64::from(w) * 4
        || (bytes.len() as u64) < u64::from(stride) * u64::from(h)
    {
        return Err("invalid capture storage".into());
    }
    let mut raw =
        Mat::new_rows_cols_with_default(h as i32, w as i32, core::CV_8UC3, Scalar::all(0.))
            .map_err(error)?;
    let bgr = raw.data_bytes_mut().map_err(error)?;
    let swapped = matches!(
        format,
        wl_shm::Format::Abgr8888
            | wl_shm::Format::Xbgr8888
            | wl_shm::Format::Abgr2101010
            | wl_shm::Format::Xbgr2101010
    );
    let ten = matches!(
        format,
        wl_shm::Format::Argb2101010
            | wl_shm::Format::Xrgb2101010
            | wl_shm::Format::Abgr2101010
            | wl_shm::Format::Xbgr2101010
    );
    for y in 0..h as usize {
        let sy = if inverted { h as usize - 1 - y } else { y };
        let row = &bytes[sy * stride as usize..sy * stride as usize + w as usize * 4];
        for (x, pixel) in row.as_chunks::<4>().0.iter().enumerate() {
            let word = u32::from_ne_bytes([pixel[0], pixel[1], pixel[2], pixel[3]]);
            let (lo, g, hi) = if ten {
                (
                    ((word & 1023) * 255 + 511) / 1023,
                    (((word >> 10) & 1023) * 255 + 511) / 1023,
                    (((word >> 20) & 1023) * 255 + 511) / 1023,
                )
            } else {
                (word & 255, (word >> 8) & 255, (word >> 16) & 255)
            };
            let i = (y * w as usize + x) * 3;
            bgr[i] = if swapped { hi } else { lo } as u8;
            bgr[i + 1] = g as u8;
            bgr[i + 2] = if swapped { lo } else { hi } as u8;
        }
    }
    // Hyprland 0.55 ScreenshareSession::calculateConstraints uses m_pixelSize,
    // not m_transformedSize; ScreenshareFrame exports the raw framebuffer with
    // output transform metadata. wlr-screencopy omits that metadata. Apply it
    // here exactly once (including square/180-degree outputs, not size heuristics).
    let mut upright = Mat::default();
    let rotation = match transform {
        0 | 4 => None,
        1 | 5 => Some(core::ROTATE_90_COUNTERCLOCKWISE),
        2 | 6 => Some(core::ROTATE_180),
        3 | 7 => Some(core::ROTATE_90_CLOCKWISE),
        _ => return Err("invalid output transform".into()),
    };
    if transform >= 4 {
        core::flip(&raw, &mut upright, 1).map_err(error)?;
        raw = upright;
        upright = Mat::default();
    }
    if let Some(rotation) = rotation {
        core::rotate(&raw, &mut upright, rotation).map_err(error)?;
        Ok(upright)
    } else {
        Ok(raw)
    }
}

impl Dispatch<wl_callback::WlCallback, ()> for State {
    fn event(
        state: &mut Self,
        _: &wl_callback::WlCallback,
        _: wl_callback::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        state.sync_pending = false;
    }
}
impl Dispatch<shortcut::HyprlandGlobalShortcutV1, GestureKey> for State {
    fn event(
        state: &mut Self,
        _: &shortcut::HyprlandGlobalShortcutV1,
        event: shortcut::Event,
        key: &GestureKey,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if state.locked {
            state.tracker.reset();
            return;
        }
        if let Some(mode) = gesture_event(&mut state.tracker, *key, event) {
            state.requested = Some(mode);
        }
    }
}
impl Dispatch<lock::HyprlandLockNotificationV1, ()> for State {
    fn event(
        state: &mut Self,
        _: &lock::HyprlandLockNotificationV1,
        event: lock::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            lock::Event::Locked => {
                state.locked = true;
                state.requested = None;
                state.cancel();
            }
            lock::Event::Unlocked => {
                state.locked = false;
                state.tracker.reset();
            }
            _ => {}
        }
    }
}
wayland_client::delegate_noop!(State: ignore capture::ZwlrScreencopyManagerV1);
wayland_client::delegate_noop!(State: ignore pointers::ZwlrVirtualPointerManagerV1);
wayland_client::delegate_noop!(State: ignore pointer::ZwlrVirtualPointerV1);
wayland_client::delegate_noop!(State: ignore shortcuts::HyprlandGlobalShortcutsManagerV1);
wayland_client::delegate_noop!(State: ignore locks::HyprlandLockNotifierV1);
wayland_client::delegate_noop!(State: ignore wl_buffer::WlBuffer);

impl CompositorHandler for State {
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: i32,
    ) {
    }
    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: wl_output::Transform,
    ) {
    }
    fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: u32) {}
    fn surface_enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
    fn surface_leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
}
impl OutputHandler for State {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.outputs
    }
    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {
        self.invalidated = self.mode.is_some();
    }
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {
        self.invalidated = self.mode.is_some();
    }
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {
        self.invalidated = self.mode.is_some();
    }
}
impl LayerShellHandler for State {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, layer: &LayerSurface) {
        if self.layer.as_ref() == Some(layer) {
            self.cancel();
        }
    }
    fn configure(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        layer: &LayerSurface,
        c: LayerSurfaceConfigure,
        _: u32,
    ) {
        if self.layer.as_ref() != Some(layer) {
            return;
        }
        if (c.new_size.0 != 0 && c.new_size.0 != self.size.0)
            || (c.new_size.1 != 0 && c.new_size.1 != self.size.1)
        {
            self.cancel();
            return;
        }
        self.configured = true;
        self.dirty = true;
    }
}
impl ShmHandler for State {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}
impl SeatHandler for State {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seats
    }
    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
    fn new_capability(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        cap: Capability,
    ) {
        if !self.input_enabled {
            return;
        }
        if cap == Capability::Keyboard && self.keyboard.is_none() {
            match self.seats.get_keyboard(qh, &seat, None) {
                Ok(k) => self.keyboard = Some(k),
                Err(e) => self.failure = Some(error(e)),
            }
        }
        if cap == Capability::Pointer && self.physical_pointer.is_none() {
            match self.seats.get_pointer(qh, &seat) {
                Ok(p) => self.physical_pointer = Some(p),
                Err(e) => self.failure = Some(error(e)),
            }
        }
    }
    fn remove_capability(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wl_seat::WlSeat,
        cap: Capability,
    ) {
        if cap == Capability::Keyboard {
            if let Some(k) = self.keyboard.take() {
                k.release();
            }
            self.pressed.clear();
            self.modifiers = false;
            self.cancel();
        }
        if cap == Capability::Pointer {
            if let Some(p) = self.physical_pointer.take() {
                p.release();
            }
            self.buttons.clear();
            self.cancel();
        }
    }
    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {
        self.cancel();
    }
}
impl KeyboardHandler for State {
    fn enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        surface: &wl_surface::WlSurface,
        _: u32,
        raw: &[u32],
        _: &[Keysym],
    ) {
        if self
            .layer
            .as_ref()
            .is_some_and(|l| l.wl_surface() == surface)
        {
            self.pressed.clear();
            self.pressed.extend(raw.iter().copied());
        }
    }
    fn leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        surface: &wl_surface::WlSurface,
        _: u32,
    ) {
        if !self
            .layer
            .as_ref()
            .is_some_and(|l| l.wl_surface() == surface)
        {
            return;
        }
        if self.mode.is_some() {
            self.click = None;
            self.cancel();
        }
        self.pressed.clear();
        self.modifiers = false;
    }
    fn press_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        event: KeyEvent,
    ) {
        if self.layer.is_some() {
            self.pressed.insert(event.raw_code);
            self.key(event);
        }
    }
    fn repeat_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: KeyEvent,
    ) {
    }
    fn release_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        event: KeyEvent,
    ) {
        self.pressed.remove(&event.raw_code);
    }
    fn update_modifiers(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: Modifiers,
        raw: RawModifiers,
        _: u32,
    ) {
        // Locked modifiers (Caps/Num Lock) do not represent held keys.
        self.modifiers = raw.depressed != 0 || raw.latched != 0;
    }
}
impl PointerHandler for State {
    fn pointer_frame(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for event in events {
            if !self
                .layer
                .as_ref()
                .is_some_and(|l| l.wl_surface() == &event.surface)
            {
                continue;
            }
            match event.kind {
                PointerEventKind::Press { button, .. } => {
                    self.buttons.insert(button);
                    self.cancel();
                }
                PointerEventKind::Release { button, .. } => {
                    self.buttons.remove(&button);
                }
                PointerEventKind::Axis { .. } => self.cancel(),
                _ => {}
            }
        }
    }
}
delegate_compositor!(State);
delegate_output!(State);
delegate_shm!(State);
delegate_seat!(State);
delegate_keyboard!(State);
delegate_pointer!(State);
delegate_layer!(State);
delegate_registry!(State);
impl ProvidesRegistryState for State {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry
    }
    registry_handlers![OutputState, SeatState];
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::MatTraitConstManual;

    #[test]
    fn teardown_ack_is_finalized_before_timeout_revalidation() {
        assert!(validate_session(true, false, false, false));
        assert!(!validate_session(false, false, false, false));
        assert!(!validate_session(false, false, true, false));
        assert!(!validate_session(true, true, false, false));
        assert!(validate_session(false, true, false, true));
    }

    #[test]
    fn selection_release_deadline_expires_once_and_can_be_rearmed() {
        let now = Instant::now();
        let mut deadline = Some(now + Duration::from_secs(2));
        assert!(!release_drain_expired(&mut deadline, now));
        assert!(!release_drain_expired(
            &mut deadline,
            now + Duration::from_secs(2) - Duration::from_nanos(1)
        ));
        assert!(release_drain_expired(
            &mut deadline,
            now + Duration::from_secs(2)
        ));
        assert_eq!(deadline, None);
        assert!(!release_drain_expired(
            &mut deadline,
            now + Duration::from_secs(3)
        ));
        deadline = Some(now + Duration::from_secs(4));
        assert!(!release_drain_expired(
            &mut deadline,
            now + Duration::from_secs(3)
        ));
        // Cancellation or unmapping disarms the deadline for the next lifecycle.
        deadline = None;
        assert!(!release_drain_expired(
            &mut deadline,
            now + Duration::from_secs(5)
        ));
    }

    #[test]
    fn batched_shortcuts_use_protocol_holds_and_tap_gaps() {
        let key = GestureKey::ShiftLeft;
        // All events are processed together, with no processing-time delay.
        for (times, expected) in [
            ([0, 10, 20, 30], Some(Mode::Hints)),
            ([0, 251, 260, 270], None),
            ([0, 10, 400, 410], None),
            ([100, 110, 120, 119], None),
        ] {
            let mut tracker = TapTracker::default();
            for (i, millis) in times.into_iter().enumerate() {
                let event = if i % 2 == 0 {
                    shortcut::Event::Pressed {
                        tv_sec_hi: 1,
                        tv_sec_lo: 42,
                        tv_nsec: millis * 1_000_000,
                    }
                } else {
                    shortcut::Event::Released {
                        tv_sec_hi: 1,
                        tv_sec_lo: 42,
                        tv_nsec: millis * 1_000_000,
                    }
                };
                assert_eq!(
                    gesture_event(&mut tracker, key, event),
                    if i == 3 { expected } else { None }
                );
            }
        }
    }

    #[test]
    fn shortcut_nanoseconds_are_validated_without_normalization() {
        let mut tracker = TapTracker::default();
        for nanos in [0, 1_000_000_000, 10] {
            assert_eq!(
                gesture_event(
                    &mut tracker,
                    GestureKey::ControlLeft,
                    shortcut::Event::Pressed {
                        tv_sec_hi: 0,
                        tv_sec_lo: 0,
                        tv_nsec: nanos
                    }
                ),
                None
            );
            assert_eq!(
                gesture_event(
                    &mut tracker,
                    GestureKey::ControlLeft,
                    shortcut::Event::Released {
                        tv_sec_hi: 0,
                        tv_sec_lo: 0,
                        tv_nsec: nanos
                    }
                ),
                None
            );
        }
    }

    #[test]
    fn shortcut_timestamps_cross_seconds_and_recover_after_rejection() {
        let key = GestureKey::ControlLeft;
        for invalid in [1_000_000_000, 0] {
            let mut tracker = TapTracker::default();
            for (pressed, lo, nanos, expected) in [
                (true, u32::MAX, 980_000_000, None),
                (false, u32::MAX, 990_000_000, None),
                // Invalid nanos or a backward press must discard the first tap.
                (true, u32::MAX, invalid, None),
                (false, u32::MAX, 995_000_000, None),
                (true, 0, 0, None),
                (false, 0, 10_000_000, None),
                (true, 0, 20_000_000, None),
                (false, 0, 30_000_000, Some(Mode::Grid)),
            ] {
                let hi = u32::from(lo == 0);
                let event = if pressed {
                    shortcut::Event::Pressed {
                        tv_sec_hi: hi,
                        tv_sec_lo: lo,
                        tv_nsec: nanos,
                    }
                } else {
                    shortcut::Event::Released {
                        tv_sec_hi: hi,
                        tv_sec_lo: lo,
                        tv_nsec: nanos,
                    }
                };
                assert_eq!(gesture_event(&mut tracker, key, event), expected);
            }
        }
    }

    #[test]
    fn cancellation_does_not_wait_for_unobservable_releases() {
        for keys in [0, 1] {
            for buttons in [0, 1] {
                for modifiers in [false, true] {
                    assert!(ready_to_unmap(false, keys, buttons, modifiers));
                    assert_eq!(
                        ready_to_unmap(true, keys, buttons, modifiers),
                        keys == 0 && buttons == 0 && !modifiers
                    );
                }
            }
        }
    }

    #[test]
    fn capture_normalizes_every_output_transform_once() {
        let bytes = [1u32, 2, 3, 4, 5, 6]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>();
        let expected = [
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 4, 6, 1, 3, 5],
            vec![6, 5, 4, 3, 2, 1],
            vec![5, 3, 1, 6, 4, 2],
            vec![2, 1, 4, 3, 6, 5],
            vec![1, 3, 5, 2, 4, 6],
            vec![5, 6, 3, 4, 1, 2],
            vec![6, 4, 2, 5, 3, 1],
        ];
        for (transform, expected) in expected.iter().enumerate() {
            let image = decode(
                &bytes,
                2,
                3,
                8,
                wl_shm::Format::Xrgb8888,
                false,
                transform as i32,
            )
            .unwrap();
            assert_eq!(
                image
                    .data_bytes()
                    .unwrap()
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|p| p[0])
                    .collect::<Vec<_>>(),
                *expected
            );
            assert_eq!(
                (image.cols(), image.rows()),
                if transform % 2 == 0 { (2, 3) } else { (3, 2) }
            );
        }
    }

    #[test]
    fn capture_honors_row_stride_inversion_and_channel_order() {
        let bytes = [0x00112233u32, 0xffffffff, 0x00445566, 0xffffffff]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>();
        let image = decode(&bytes, 1, 2, 8, wl_shm::Format::Xrgb8888, true, 0).unwrap();
        assert_eq!(
            image.data_bytes().unwrap(),
            &[0x66, 0x55, 0x44, 0x33, 0x22, 0x11]
        );
        let image = decode(&bytes, 1, 2, 8, wl_shm::Format::Abgr8888, false, 0).unwrap();
        assert_eq!(
            image.data_bytes().unwrap(),
            &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66]
        );
        let word = (1023u32 << 20) | (512 << 10);
        let image = decode(
            &word.to_ne_bytes(),
            1,
            1,
            4,
            wl_shm::Format::Xbgr2101010,
            false,
            0,
        )
        .unwrap();
        assert_eq!(image.data_bytes().unwrap(), &[255, 128, 0]);
    }

    #[test]
    fn capture_inversion_precedes_every_output_transform() {
        let bytes = [1u32, 2, 3, 4, 5, 6]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>();
        let inverted = [4u32, 5, 6, 1, 2, 3]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>();
        for transform in 0..8 {
            let actual =
                decode(&bytes, 3, 2, 12, wl_shm::Format::Xrgb8888, true, transform).unwrap();
            let expected = decode(
                &inverted,
                3,
                2,
                12,
                wl_shm::Format::Xrgb8888,
                false,
                transform,
            )
            .unwrap();
            assert_eq!(actual.size().unwrap(), expected.size().unwrap());
            assert_eq!(actual.data_bytes().unwrap(), expected.data_bytes().unwrap());
        }
    }
}
