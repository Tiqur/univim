use super::{
    CommandAction, PointerMotion, RuntimeSettings,
    draw::{RenderedSurface, render_session_surface},
    evdev::EvdevKeyboardSet,
    geometry::FrameGeometry,
    hyprland::{self, Context as HyprContext},
    session::{
        Cardinal, ClickAction, GestureKey, Input, Mode, Outcome, PrecomputedFrame, Session,
        TapEvent, TapTracker,
    },
    snapshot::{self, RawCaptureFrame, Saved, decode_raw_frame_to_bgr},
};
use crate::{DebugData, DetectorOptions, detect_with_debug};
use opencv::{core::Mat, prelude::*};
use std::{
    env,
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom, Write},
    os::{
        fd::{AsFd, AsRawFd, BorrowedFd},
        unix::{
            fs::{DirBuilderExt, PermissionsExt},
            net::UnixListener,
        },
    },
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::{Duration, Instant},
};
use tempfile::tempfile;
use wayland_client::{
    Connection, Dispatch, EventQueue, Proxy, QueueHandle, WEnum, delegate_noop,
    globals::{GlobalList, GlobalListContents, registry_queue_init},
    protocol::{
        wl_buffer::{self, WlBuffer},
        wl_callback::{self, WlCallback},
        wl_compositor::WlCompositor,
        wl_display::WlDisplay,
        wl_keyboard::{self, KeyState, WlKeyboard},
        wl_output::{self, Transform, WlOutput},
        wl_pointer::{self},
        wl_registry::{self, WlRegistry},
        wl_seat::{self, WlSeat},
        wl_shm::{self, Format as ShmFormat, WlShm},
        wl_shm_pool::WlShmPool,
        wl_surface::WlSurface,
    },
};
use wayland_protocols::wp::viewporter::client::{
    wp_viewport::WpViewport, wp_viewporter::WpViewporter,
};
use wayland_protocols_hyprland::global_shortcuts::v1::client::{
    hyprland_global_shortcut_v1::{self, HyprlandGlobalShortcutV1},
    hyprland_global_shortcuts_manager_v1::{self, HyprlandGlobalShortcutsManagerV1},
};
use wayland_protocols_wlr::{
    layer_shell::v1::client::{
        zwlr_layer_shell_v1::{Layer, ZwlrLayerShellV1},
        zwlr_layer_surface_v1::{self, Anchor, KeyboardInteractivity, ZwlrLayerSurfaceV1},
    },
    screencopy::v1::client::{
        zwlr_screencopy_frame_v1::{self, Flags as ScreencopyFlags, ZwlrScreencopyFrameV1},
        zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1,
    },
    virtual_pointer::v1::client::{
        zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1,
        zwlr_virtual_pointer_v1::ZwlrVirtualPointerV1,
    },
};
use xkbcommon::xkb;

const BTN_LEFT: u32 = 0x110;
const BTN_RIGHT: u32 = 0x111;
const BTN_MIDDLE: u32 = 0x112;

/// Interval between automatic pipeline re-captures when `F6` Continuous Loop mode is active.
const AUTO_LOOP_INTERVAL: Duration = Duration::from_millis(650);

static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

extern "C" fn request_shutdown(_: libc::c_int) {
    SHUTDOWN_REQUESTED.store(true, Ordering::SeqCst);
}

/// Install graceful SIGTERM/SIGINT handling before worker threads are spawned.
/// Systemd restarts and logout otherwise terminate the process without running
/// the socket cleanup at the end of `run_daemon`.
fn install_shutdown_handlers() {
    SHUTDOWN_REQUESTED.store(false, Ordering::SeqCst);
    // SAFETY: the handler only performs an async-signal-safe atomic store.
    unsafe {
        libc::signal(
            libc::SIGTERM,
            request_shutdown as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGINT,
            request_shutdown as *const () as libc::sighandler_t,
        );
    }
}

pub fn run_daemon(socket_path: &Path, settings: RuntimeSettings) -> Result<(), String> {
    ensure_wayland_and_hyprland_env();
    install_shutdown_handlers();

    if let Some(parent) = socket_path.parent() {
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
        let metadata = fs::symlink_metadata(parent)
            .map_err(|err| format!("failed to inspect {}: {err}", parent.display()))?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(format!(
                "control socket parent {} is not a real directory",
                parent.display()
            ));
        }
    }
    if socket_path.exists() {
        return Err(format!(
            "control socket {} already exists; verify no daemon is running, then remove it",
            socket_path.display()
        ));
    }

    let listener = UnixListener::bind(socket_path)
        .map_err(|err| format!("failed to bind {}: {err}", socket_path.display()))?;
    fs::set_permissions(socket_path, fs::Permissions::from_mode(0o600))
        .map_err(|err| format!("failed to secure {}: {err}", socket_path.display()))?;
    listener
        .set_nonblocking(true)
        .map_err(|err| format!("failed to make control socket nonblocking: {err}"))?;

    let conn = Connection::connect_to_env()
        .map_err(|err| format!("failed to connect to Wayland compositor: {err}"))?;
    let (globals, mut event_queue) = registry_queue_init::<DaemonState>(&conn)
        .map_err(|err| format!("failed to initialize Wayland globals: {err}"))?;
    let qh = event_queue.handle();

    let mut state = DaemonState::bind(&globals, conn.display(), &qh, settings)?;
    event_queue
        .roundtrip(&mut state)
        .map_err(|err| format!("initial Wayland roundtrip failed: {err}"))?;
    state.finish_init(&qh)?;
    event_queue
        .roundtrip(&mut state)
        .map_err(|err| format!("post-init Wayland roundtrip failed: {err}"))?;

    let result = daemon_loop(&conn, &mut event_queue, &qh, &listener, &mut state);
    let _ = fs::remove_file(socket_path);
    result
}

/// Capture one monitor and save numbered raw, annotated, and JSON artifacts
/// without creating an overlay layer or listening for input.
pub fn capture_snapshot(
    output: Option<&str>,
    directory: &Path,
    options: DetectorOptions,
    flat: bool,
) -> Result<Saved, String> {
    ensure_wayland_and_hyprland_env();
    let conn = Connection::connect_to_env()
        .map_err(|err| format!("failed to connect to Wayland compositor: {err}"))?;
    let (globals, mut event_queue) = registry_queue_init::<DaemonState>(&conn)
        .map_err(|err| format!("failed to initialize Wayland globals: {err}"))?;
    let qh = event_queue.handle();
    let settings = RuntimeSettings {
        detector: options,
        output: output.map(str::to_owned),
        no_click: true,
        flat,
        ..RuntimeSettings::default()
    };
    let mut state = DaemonState::bind(&globals, conn.display(), &qh, settings)?;
    event_queue
        .roundtrip(&mut state)
        .map_err(|err| format!("initial Wayland roundtrip failed: {err}"))?;
    state.finish_init(&qh)?;
    event_queue
        .roundtrip(&mut state)
        .map_err(|err| format!("post-init Wayland roundtrip failed: {err}"))?;
    state.snapshot_dir = Some(directory.to_path_buf());
    state.start_capture(CapturePurpose::Snapshot, &qh)?;

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        event_queue
            .blocking_dispatch(&mut state)
            .map_err(|err| format!("snapshot event loop failed: {err}"))?;
        if let Some(err) = state.last_error.take() {
            return Err(err);
        }
        if let Some(saved) = state.snapshot_saved.take() {
            return Ok(saved);
        }
        state.step(&qh)?;
        if let Some(saved) = state.snapshot_saved.take() {
            return Ok(saved);
        }
        if Instant::now() >= deadline {
            return Err("snapshot capture timed out".into());
        }
    }
}

struct CheckState;

impl Dispatch<WlRegistry, GlobalListContents> for CheckState {
    fn event(
        _state: &mut Self,
        _registry: &WlRegistry,
        _event: wl_registry::Event,
        _data: &GlobalListContents,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

/// Enumerate advertised Wayland globals and the Hyprland focused output without
/// binding input extensions or creating a surface.
pub fn check() -> Result<serde_json::Value, String> {
    ensure_wayland_and_hyprland_env();
    let conn = Connection::connect_to_env()
        .map_err(|err| format!("failed to connect to Wayland compositor: {err}"))?;
    let (globals, mut event_queue) = registry_queue_init::<CheckState>(&conn)
        .map_err(|err| format!("failed to initialize Wayland globals: {err}"))?;
    let mut state = CheckState;
    event_queue
        .roundtrip(&mut state)
        .map_err(|err| format!("Wayland registry roundtrip failed: {err}"))?;
    let protocols = globals.contents().with_list(|list| {
        list.iter()
            .map(|global| {
                serde_json::json!({
                    "interface": global.interface,
                    "version": global.version,
                })
            })
            .collect::<Vec<_>>()
    });
    let focused_output = hyprland::selected_output(None).ok();
    Ok(serde_json::json!({
        "protocols": protocols,
        "focused_output": focused_output,
        "gesture_limitation": "Anonymous compositor shortcuts cannot observe arbitrary intervening chords; configure nonconsuming modifier-only or evdev bindings."
    }))
}

/// Auto-discover `WAYLAND_DISPLAY` and `HYPRLAND_INSTANCE_SIGNATURE` inside `$XDG_RUNTIME_DIR`
/// if systemd user service started before the environment variables were imported.
fn ensure_wayland_and_hyprland_env() {
    let Some(runtime_dir) = env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from) else {
        return;
    };
    if env::var_os("WAYLAND_DISPLAY").is_none()
        && let Ok(entries) = fs::read_dir(&runtime_dir)
    {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let s = name.to_string_lossy();
            if s.starts_with("wayland-") && !s.ends_with(".lock") {
                // SAFETY: environment mutation happens once during startup,
                // before any other threads are spawned.
                unsafe {
                    env::set_var("WAYLAND_DISPLAY", s.as_ref());
                }
                break;
            }
        }
    }
    if env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_none() {
        let hypr_dir = runtime_dir.join("hypr");
        if let Ok(entries) = fs::read_dir(hypr_dir) {
            for entry in entries.flatten() {
                if entry.path().join(".socket.sock").exists() {
                    // SAFETY: see the WAYLAND_DISPLAY discovery above.
                    unsafe {
                        env::set_var("HYPRLAND_INSTANCE_SIGNATURE", entry.file_name());
                    }
                    break;
                }
            }
        }
    }
}

fn daemon_loop(
    conn: &Connection,
    event_queue: &mut EventQueue<DaemonState>,
    qh: &QueueHandle<DaemonState>,
    listener: &UnixListener,
    state: &mut DaemonState,
) -> Result<(), String> {
    while !state.should_exit && !SHUTDOWN_REQUESTED.load(Ordering::SeqCst) {
        event_queue
            .dispatch_pending(state)
            .map_err(|err| format!("Wayland dispatch failed: {err}"))?;
        state.poll_worker_results(qh)?;
        state.step(qh)?;
        conn.flush()
            .map_err(|err| format!("Wayland flush failed: {err}"))?;

        accept_control_commands(listener, state, qh)?;
        state.poll_evdev_shortcuts(qh)?;
        state.poll_auto_loop(qh)?;
        state.poll_worker_results(qh)?;
        state.step(qh)?;
        conn.flush()
            .map_err(|err| format!("Wayland flush failed: {err}"))?;

        if state.should_exit {
            break;
        }

        if let Some(read_guard) = event_queue.prepare_read() {
            let wayland_fd = read_guard.connection_fd();
            let evdev_fd = state.evdev_fd();
            let timeout = if state.has_async_work() {
                5
            } else if state.is_auto_loop_active() {
                35
            } else {
                250
            };
            let ready = poll_fds(wayland_fd, listener.as_fd(), evdev_fd, timeout)?;
            if ready.wayland_readable {
                // poll(2) can report readability that is consumed before the
                // read guard runs, which surfaces as EAGAIN on the nonblocking
                // Wayland socket. It is not a connection failure.
                if let Err(err) = read_guard.read()
                    && !matches!(
                        &err,
                        wayland_client::backend::WaylandError::Io(io)
                            if io.kind() == io::ErrorKind::WouldBlock
                    )
                {
                    return Err(format!("failed to read Wayland events: {err}"));
                }
            } else {
                drop(read_guard);
            }
            if ready.control_readable {
                accept_control_commands(listener, state, qh)?;
            }
            if ready.evdev_readable {
                state.poll_evdev_shortcuts(qh)?;
            }
        }
        event_queue
            .dispatch_pending(state)
            .map_err(|err| format!("Wayland dispatch failed: {err}"))?;
        state.poll_worker_results(qh)?;
        state.step(qh)?;
    }
    Ok(())
}

struct PollOutcome {
    wayland_readable: bool,
    control_readable: bool,
    evdev_readable: bool,
}

fn poll_fds(
    wayland_fd: BorrowedFd<'_>,
    control_fd: BorrowedFd<'_>,
    evdev_fd: Option<BorrowedFd<'_>>,
    timeout_ms: i32,
) -> Result<PollOutcome, String> {
    let mut fds = vec![
        libc::pollfd {
            fd: wayland_fd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        },
        libc::pollfd {
            fd: control_fd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        },
    ];
    if let Some(efd) = evdev_fd {
        fds.push(libc::pollfd {
            fd: efd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        });
    }
    let rc = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, timeout_ms) };
    if rc < 0 {
        let err = io::Error::last_os_error();
        if err.kind() == io::ErrorKind::Interrupted {
            return Ok(PollOutcome {
                wayland_readable: false,
                control_readable: false,
                evdev_readable: false,
            });
        }
        return Err(format!("poll(2) failed: {err}"));
    }
    Ok(PollOutcome {
        wayland_readable: (fds[0].revents & libc::POLLIN) != 0,
        control_readable: (fds[1].revents & libc::POLLIN) != 0,
        evdev_readable: fds
            .get(2)
            .is_some_and(|pfd| (pfd.revents & libc::POLLIN) != 0),
    })
}

fn accept_control_commands(
    listener: &UnixListener,
    state: &mut DaemonState,
    qh: &QueueHandle<DaemonState>,
) -> Result<(), String> {
    loop {
        match listener.accept() {
            Ok((mut stream, _addr)) => {
                let mut text = String::new();
                stream
                    .read_to_string(&mut text)
                    .map_err(|err| format!("failed to read control command: {err}"))?;
                if let Some(action) = CommandAction::from_wire(&text) {
                    state.enqueue_command(action, qh)?;
                }
            }
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => return Ok(()),
            Err(err) => return Err(format!("failed to accept control socket client: {err}")),
        }
    }
}

struct WorkerJob {
    job_id: u64,
    purpose: CapturePurpose,
    output_name: Option<String>,
    geometry: FrameGeometry,
    upright_w: i32,
    upright_h: i32,
    bgr_bytes: Vec<u8>,
    options: DetectorOptions,
    flat: bool,
}

struct WorkerOutput {
    job_id: u64,
    purpose: CapturePurpose,
    geometry: FrameGeometry,
    frame: PrecomputedFrame,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CapturePurpose {
    /// Silent background capture triggered on the first CapsLock tap.
    SpeculativeWarmup,
    /// Immediate overlay presentation (either direct `hints` or promoted 2nd tap).
    PresentOverlay(Mode),
    /// Re-capture while an overlay session is already active (after scroll or `r` / `F5` / `F6` loop).
    RefreshOverlay,
    /// One-shot numbered snapshot with no overlay or input surface.
    Snapshot,
}

struct DaemonState {
    settings: RuntimeSettings,
    display: WlDisplay,
    compositor: WlCompositor,
    shm: WlShm,
    layer_shell: ZwlrLayerShellV1,
    screencopy_manager: ZwlrScreencopyManagerV1,
    virtual_pointer_manager: ZwlrVirtualPointerManagerV1,
    viewporter: Option<WpViewporter>,
    shortcuts_manager: Option<HyprlandGlobalShortcutsManagerV1>,
    shortcuts: Vec<HyprlandGlobalShortcutV1>,
    seat: Option<WlSeat>,
    keyboard: Option<WlKeyboard>,
    outputs: Vec<OutputInfo>,
    xkb_context: xkb::Context,
    xkb_state: Option<xkb::State>,
    pending_command: Option<CommandAction>,
    capture: Option<ActiveCapture>,
    overlay: Option<ActiveOverlay>,
    layer: Option<OverlayLayer>,
    pending_click: Option<PendingClick>,
    pointer_glide: Option<PointerGlide>,
    pending_re_capture: Option<PendingReCapture>,
    sync_pending: bool,
    should_exit: bool,
    last_error: Option<String>,
    tap_tracker: TapTracker,
    cached_frame: Option<PrecomputedFrame>,
    debug_on_present: bool,
    next_job_id: u64,
    inflight_job: Option<(u64, CapturePurpose)>,
    worker_tx: Sender<WorkerJob>,
    worker_rx: Receiver<WorkerOutput>,
    evdev: EvdevKeyboardSet,
    snapshot_dir: Option<PathBuf>,
    snapshot_saved: Option<Saved>,
}

#[derive(Clone)]
struct OutputInfo {
    wl_output: WlOutput,
    global_name: u32,
    name: Option<String>,
    logical_width: i32,
    logical_height: i32,
    mode_width: i32,
    mode_height: i32,
    x: i32,
    y: i32,
    scale: i32,
    transform: Transform,
}

impl OutputInfo {
    fn new(wl_output: WlOutput, global_name: u32) -> Self {
        Self {
            wl_output,
            global_name,
            name: None,
            logical_width: 0,
            logical_height: 0,
            mode_width: 0,
            mode_height: 0,
            x: 0,
            y: 0,
            scale: 1,
            transform: Transform::Normal,
        }
    }

    fn as_output(&self) -> crate::desktop::geometry::Output {
        crate::desktop::geometry::Output {
            name: self.name.clone().unwrap_or_default(),
            x: self.x,
            y: self.y,
            width: self.mode_width,
            height: self.mode_height,
            scale: f64::from(self.scale.max(1)),
            transform: match self.transform {
                Transform::Normal => 0,
                Transform::_90 => 1,
                Transform::_180 => 2,
                Transform::_270 => 3,
                Transform::Flipped => 4,
                Transform::Flipped90 => 5,
                Transform::Flipped180 => 6,
                Transform::Flipped270 => 7,
                _ => 0,
            },
        }
    }

    fn resolved_logical_size(&self) -> (i32, i32) {
        if self.logical_width > 0 && self.logical_height > 0 {
            return (self.logical_width, self.logical_height);
        }
        let scale = self.scale.max(1);
        let (w, h) = match self.transform {
            Transform::_90 | Transform::_270 | Transform::Flipped90 | Transform::Flipped270 => {
                (self.mode_height, self.mode_width)
            }
            _ => (self.mode_width, self.mode_height),
        };
        ((w / scale).max(1), (h / scale).max(1))
    }
}

struct ActiveCapture {
    purpose: CapturePurpose,
    output: OutputInfo,
    frame: ZwlrScreencopyFrameV1,
    spec: Option<BufferSpec>,
    shm_file: Option<File>,
    buffer: Option<WlBuffer>,
    flags: ScreencopyFlags,
    ready: bool,
    failed: bool,
}

#[derive(Clone, Copy, Debug)]
struct BufferSpec {
    format: ShmFormat,
    width: u32,
    height: u32,
    stride: u32,
}

struct ActiveOverlay {
    output: OutputInfo,
    geometry: FrameGeometry,
    base_bgr: Mat,
    debug_data: Option<DebugData>,
    session: Session,
    last_loop_refresh: Instant,
}

struct OverlayLayer {
    surface: WlSurface,
    layer_surface: ZwlrLayerSurfaceV1,
    viewport: Option<WpViewport>,
    logical_width: u32,
    logical_height: u32,
    configured: bool,
    closed: bool,
    slot: Option<ShmSlot>,
    dirty: bool,
}

struct ShmSlot {
    file: File,
    buffer: WlBuffer,
    len: usize,
    width: i32,
    height: i32,
}

#[derive(Clone, Copy)]
struct PendingClick {
    output_index: usize,
    logical_width: u32,
    logical_height: u32,
    x: u32,
    y: u32,
    action: ClickAction,
    drag_from: Option<(u32, u32)>,
}

struct PointerGlide {
    pointer: ZwlrVirtualPointerV1,
    click: PendingClick,
    from: (u32, u32),
    started: Instant,
    duration: Duration,
    curve: PointerMotion,
}

fn glide_fraction(elapsed: Duration, duration: Duration, curve: PointerMotion) -> f64 {
    let t = (elapsed.as_secs_f64() / duration.as_secs_f64()).clamp(0.0, 1.0);
    match curve {
        PointerMotion::EaseOut => 1.0 - (1.0 - t).powi(3),
        _ => t,
    }
}

#[cfg(test)]
mod glide_tests {
    use super::{PointerMotion, glide_fraction};
    use std::time::Duration;

    #[test]
    fn linear_and_ease_out_reach_target_without_overshooting() {
        let duration = Duration::from_millis(100);
        let halfway = Duration::from_millis(50);
        assert_eq!(
            glide_fraction(halfway, duration, PointerMotion::Linear),
            0.5
        );
        assert_eq!(
            glide_fraction(halfway, duration, PointerMotion::EaseOut),
            0.875
        );
        assert_eq!(
            glide_fraction(Duration::from_secs(2), duration, PointerMotion::EaseOut),
            1.0
        );
    }
}

struct PendingReCapture {
    scroll: Option<(i32, i32)>,
    settle_until: Option<Instant>,
}

impl DaemonState {
    fn bind(
        globals: &GlobalList,
        display: WlDisplay,
        qh: &QueueHandle<Self>,
        settings: RuntimeSettings,
    ) -> Result<Self, String> {
        let compositor: WlCompositor = globals
            .bind(qh, 4..=6, ())
            .map_err(|err| format!("missing wl_compositor: {err}"))?;
        let shm: WlShm = globals
            .bind(qh, 1..=1, ())
            .map_err(|err| format!("missing wl_shm: {err}"))?;
        let layer_shell: ZwlrLayerShellV1 = globals
            .bind(qh, 1..=4, ())
            .map_err(|err| format!("missing zwlr_layer_shell_v1: {err}"))?;
        let screencopy_manager: ZwlrScreencopyManagerV1 = globals
            .bind(qh, 1..=3, ())
            .map_err(|err| format!("missing zwlr_screencopy_manager_v1: {err}"))?;
        let virtual_pointer_manager: ZwlrVirtualPointerManagerV1 = globals
            .bind(qh, 1..=2, ())
            .map_err(|err| format!("missing zwlr_virtual_pointer_manager_v1: {err}"))?;
        let viewporter: Option<WpViewporter> = globals.bind(qh, 1..=1, ()).ok();
        let shortcuts_manager: Option<HyprlandGlobalShortcutsManagerV1> =
            globals.bind(qh, 1..=1, ()).ok();
        let seat: Option<WlSeat> = globals.bind(qh, 1..=8, ()).ok();

        let mut outputs = Vec::new();
        globals.contents().with_list(|list| {
            for global in list {
                if global.interface == WlOutput::interface().name {
                    let version = global.version.min(4);
                    let wl_output = globals.registry().bind::<WlOutput, _, _>(
                        global.name,
                        version,
                        qh,
                        global.name,
                    );
                    outputs.push(OutputInfo::new(wl_output, global.name));
                }
            }
        });

        let (worker_tx, job_rx) = mpsc::channel::<WorkerJob>();
        let (res_tx, worker_rx) = mpsc::channel::<WorkerOutput>();
        thread::Builder::new()
            .name("univim-detector".to_string())
            .spawn(move || detector_worker_loop(job_rx, res_tx))
            .map_err(|err| format!("failed to spawn background detector thread: {err}"))?;

        let evdev = EvdevKeyboardSet::discover();

        Ok(Self {
            settings,
            display,
            compositor,
            shm,
            layer_shell,
            screencopy_manager,
            virtual_pointer_manager,
            viewporter,
            shortcuts_manager,
            shortcuts: Vec::new(),
            seat,
            keyboard: None,
            outputs,
            xkb_context: xkb::Context::new(xkb::CONTEXT_NO_FLAGS),
            xkb_state: None,
            pending_command: None,
            capture: None,
            overlay: None,
            layer: None,
            pending_click: None,
            pointer_glide: None,
            pending_re_capture: None,
            sync_pending: false,
            should_exit: false,
            last_error: None,
            tap_tracker: TapTracker::new(),
            cached_frame: None,
            debug_on_present: false,
            next_job_id: 1,
            inflight_job: None,
            worker_tx,
            worker_rx,
            evdev,
            snapshot_dir: None,
            snapshot_saved: None,
        })
    }

    fn finish_init(&mut self, qh: &QueueHandle<Self>) -> Result<(), String> {
        if self.outputs.is_empty() {
            return Err("Wayland compositor exposed no wl_output globals".to_string());
        }
        if let Some(manager) = &self.shortcuts_manager {
            let hints = manager.register_shortcut(
                "hints".to_string(),
                "univim".to_string(),
                "Show UniVim hint labels".to_string(),
                "Capture the focused monitor and present full-screen Vimium hints".to_string(),
                qh,
                CommandAction::Hints,
            );
            let grid = manager.register_shortcut(
                "grid".to_string(),
                "univim".to_string(),
                "Show UniVim grid overlay".to_string(),
                "Capture the focused monitor and present recursive 2-step grid hints".to_string(),
                qh,
                CommandAction::Grid,
            );
            let tap_caps = manager.register_shortcut(
                "caps-lock".to_string(),
                "univim".to_string(),
                "Double-tap CapsLock trigger (speculative warmup on 1st tap, instant hints on 2nd)".to_string(),
                "Warm up detection cache on 1st CapsLock press and open hints immediately on 2nd press".to_string(),
                qh,
                CommandAction::TapCaps,
            );
            let tap_alt = manager.register_shortcut(
                "alt-left".to_string(),
                "univim".to_string(),
                "Double-tap left Alt to show hints".to_string(),
                "Warm up on the first left Alt tap and show hints on the second".to_string(),
                qh,
                CommandAction::TapAlt,
            );
            self.shortcuts.extend([hints, grid, tap_caps, tap_alt]);
        }
        Ok(())
    }

    fn evdev_fd(&self) -> Option<BorrowedFd<'_>> {
        if self.settings.use_evdev {
            self.evdev.first_fd()
        } else {
            None
        }
    }

    fn has_async_work(&self) -> bool {
        self.inflight_job.is_some()
            || self.capture.is_some()
            || self.pending_re_capture.is_some()
            || self.pointer_glide.is_some()
    }

    fn is_auto_loop_active(&self) -> bool {
        self.overlay
            .as_ref()
            .is_some_and(|ov| ov.session.auto_loop())
            && self.layer.is_some()
    }

    fn poll_evdev_shortcuts(&mut self, qh: &QueueHandle<Self>) -> Result<(), String> {
        if !self.settings.use_evdev || self.evdev.is_empty() {
            return Ok(());
        }
        let events = self
            .evdev
            .drain_events(&mut self.tap_tracker, self.settings.double_tap_ms);
        if self.overlay.is_some() {
            return Ok(());
        }
        for (key, tap_ev) in events {
            self.handle_tap_event(key, tap_ev, qh)?;
        }
        Ok(())
    }

    fn poll_auto_loop(&mut self, qh: &QueueHandle<Self>) -> Result<(), String> {
        let should_refresh = self.overlay.as_ref().is_some_and(|ov| {
            ov.session.auto_loop()
                && ov.session.typed().is_empty()
                && !ov.session.show_options_menu()
                && ov.last_loop_refresh.elapsed() >= AUTO_LOOP_INTERVAL
        }) && self.layer.is_some()
            && self.capture.is_none()
            && self.inflight_job.is_none()
            && self.pending_re_capture.is_none();

        if should_refresh {
            self.begin_overlay_re_capture(None, qh);
        }
        Ok(())
    }

    fn handle_tap_event(
        &mut self,
        key: GestureKey,
        tap_ev: TapEvent,
        qh: &QueueHandle<Self>,
    ) -> Result<(), String> {
        match (key, tap_ev) {
            (GestureKey::CapsLock | GestureKey::LeftAlt | GestureKey::Debug, TapEvent::Warmup) => {
                let ttl = Duration::from_millis(self.settings.cache_ttl_ms);
                let has_fresh_cache = self
                    .cached_frame
                    .as_ref()
                    .is_some_and(|f| f.is_fresh(ttl, None));
                if !has_fresh_cache && self.capture.is_none() && self.inflight_job.is_none() {
                    let _ = self.start_capture(CapturePurpose::SpeculativeWarmup, qh);
                }
            }
            (
                GestureKey::CapsLock | GestureKey::LeftAlt | GestureKey::Debug,
                TapEvent::Activate,
            ) => {
                self.debug_on_present = key == GestureKey::Debug;
                self.activate_hints_with_cache(qh)?;
            }
            (GestureKey::Grid, TapEvent::Warmup) => {}
            (GestureKey::Grid, TapEvent::Activate) => {
                self.enqueue_command(CommandAction::Grid, qh)?;
            }
        }
        Ok(())
    }

    fn activate_hints_with_cache(&mut self, qh: &QueueHandle<Self>) -> Result<(), String> {
        let hypr_context = hyprland::context().ok();
        let (output_idx, output) = self.select_output(hypr_context.as_ref())?;
        let ttl = Duration::from_millis(self.settings.cache_ttl_ms);

        // 1. Instant path: if the 1st tap already finished pre-computing a fresh frame, open overlay in 0ms!
        if let Some(cached) = self.cached_frame.take()
            && cached.is_fresh(ttl, output.name.as_deref())
        {
            let geometry = FrameGeometry {
                output: output.as_output(),
                pixel_width: cached.width,
                pixel_height: cached.height,
            };
            let focused_rect = hypr_context
                .as_ref()
                .and_then(|ctx| ctx.focused_pixel_rect(&geometry));
            return self.present_precomputed_frame(
                output_idx,
                output,
                geometry,
                cached,
                focused_rect,
                true,
                qh,
            );
        }

        // 2. In-flight promotion: if the 1st tap capture or worker job is currently mid-flight,
        // promote it to PresentOverlay(Mode::Hints) so it opens the instant it finishes!
        if let Some(capture) = self.capture.as_mut()
            && capture.purpose == CapturePurpose::SpeculativeWarmup
        {
            capture.purpose = CapturePurpose::PresentOverlay(Mode::Hints);
            return Ok(());
        }
        if let Some((_, purpose)) = self.inflight_job.as_mut()
            && *purpose == CapturePurpose::SpeculativeWarmup
        {
            *purpose = CapturePurpose::PresentOverlay(Mode::Hints);
            return Ok(());
        }

        // 3. Fallback: start a direct capture & detection pass.
        self.start_capture(CapturePurpose::PresentOverlay(Mode::Hints), qh)
    }

    fn enqueue_command(
        &mut self,
        action: CommandAction,
        qh: &QueueHandle<Self>,
    ) -> Result<(), String> {
        match action {
            CommandAction::Stop => {
                self.close_overlay();
                self.should_exit = true;
                return Ok(());
            }
            CommandAction::ToggleDebug => {
                if let Some(overlay) = self.overlay.as_mut() {
                    let _ = overlay.session.handle_input(Input::ToggleDebugHud);
                    if let Some(layer) = self.layer.as_mut() {
                        layer.dirty = true;
                    }
                }
                return Ok(());
            }
            CommandAction::DebugOpen => {
                if self.overlay.is_some() {
                    self.close_overlay();
                }
                self.debug_on_present = true;
                return self.activate_hints_with_cache(qh);
            }
            CommandAction::ToggleOptions => {
                if let Some(overlay) = self.overlay.as_mut() {
                    let _ = overlay.session.handle_input(Input::ToggleOptionsMenu);
                    if let Some(layer) = self.layer.as_mut() {
                        layer.dirty = true;
                    }
                }
                return Ok(());
            }
            CommandAction::TapCaps | CommandAction::TapAlt => {
                if self.overlay.is_some() {
                    self.close_overlay();
                    return Ok(());
                }
                let key = if action == CommandAction::TapAlt {
                    GestureKey::LeftAlt
                } else {
                    GestureKey::CapsLock
                };
                let ev = self.tap_tracker.on_tap(
                    key,
                    Duration::from_millis(self.settings.double_tap_ms.max(40)),
                );
                return self.handle_tap_event(key, ev, qh);
            }
            CommandAction::Hints => {
                if self.overlay.is_some() {
                    self.close_overlay();
                }
                return self.activate_hints_with_cache(qh);
            }
            CommandAction::Grid => {
                if self.overlay.is_some() || self.capture.is_some() {
                    self.close_overlay();
                }
                self.pending_command = Some(CommandAction::Grid);
            }
        }
        self.step(qh)
    }

    fn poll_worker_results(&mut self, qh: &QueueHandle<Self>) -> Result<(), String> {
        while let Ok(done) = self.worker_rx.try_recv() {
            let mut effective_purpose = done.purpose;
            if let Some((active_id, active_purpose)) = self.inflight_job
                && active_id == done.job_id
            {
                effective_purpose = active_purpose;
                self.inflight_job = None;
            }
            match effective_purpose {
                CapturePurpose::SpeculativeWarmup => {
                    self.cached_frame = Some(done.frame);
                }
                CapturePurpose::PresentOverlay(_mode) => {
                    let hypr_context = hyprland::context().ok();
                    let (output_idx, output) = self.select_output(hypr_context.as_ref())?;
                    let focused_rect = hypr_context
                        .as_ref()
                        .and_then(|ctx| ctx.focused_pixel_rect(&done.geometry));
                    self.present_precomputed_frame(
                        output_idx,
                        output,
                        done.geometry,
                        done.frame,
                        focused_rect,
                        false,
                        qh,
                    )?;
                }
                CapturePurpose::RefreshOverlay => {
                    let hypr_context = hyprland::context().ok();
                    let focused_rect = hypr_context
                        .as_ref()
                        .and_then(|ctx| ctx.focused_pixel_rect(&done.geometry));
                    self.finish_overlay_refresh(done.geometry, done.frame, focused_rect, qh)?;
                }
                CapturePurpose::Snapshot => {}
            }
        }
        Ok(())
    }

    fn step(&mut self, qh: &QueueHandle<Self>) -> Result<(), String> {
        if let Some(err) = self.last_error.take() {
            self.close_overlay();
            return Err(err);
        }

        if self.overlay.as_ref().is_some_and(|overlay| {
            overlay.last_loop_refresh.elapsed()
                >= Duration::from_secs(self.settings.timeout_secs.max(1))
        }) {
            self.close_overlay();
        }

        // Advance any pending unmap -> scroll -> settle -> re-capture pipeline loop
        if !self.sync_pending {
            let mut pending_scroll = None;
            let ready_to_capture = if let Some(pending) = self.pending_re_capture.as_mut() {
                pending_scroll = pending.scroll.take();
                pending
                    .settle_until
                    .is_none_or(|deadline| Instant::now() >= deadline)
            } else {
                false
            };
            if let Some((dx, dy)) = pending_scroll {
                self.inject_scroll(dx, dy, qh);
                if let Some(pending) = self.pending_re_capture.as_mut() {
                    pending.settle_until = Some(Instant::now() + Duration::from_millis(45));
                }
            }
            if ready_to_capture {
                self.pending_re_capture = None;
                self.start_capture(CapturePurpose::RefreshOverlay, qh)?;
            }
        }

        if let Some(action) = self.pending_command.take()
            && action == CommandAction::Grid
        {
            self.start_capture(CapturePurpose::PresentOverlay(Mode::Grid), qh)?;
        }

        let capture_ready = self.capture.as_ref().is_some_and(|c| c.ready);
        let capture_failed = self.capture.as_ref().is_some_and(|c| c.failed);
        if capture_failed {
            self.destroy_capture();
            if self.overlay.is_some()
                && self.layer.is_none()
                && let Some(ov) = self.overlay.as_ref()
            {
                let (lw, lh) = ov.output.resolved_logical_size();
                self.create_overlay_layer(&ov.output.wl_output.clone(), lw as u32, lh as u32, qh)?;
            }
        } else if capture_ready {
            self.finish_capture(qh)?;
        }

        if self
            .layer
            .as_ref()
            .is_some_and(|layer| layer.closed && self.pending_click.is_none())
        {
            self.close_overlay();
        }

        if self
            .layer
            .as_ref()
            .is_some_and(|layer| layer.configured && layer.dirty)
        {
            self.render_and_commit_overlay(qh)?;
        }

        if !self.sync_pending
            && let Some(click) = self.pending_click.take()
        {
            self.emit_virtual_action(click, qh);
        }
        self.advance_pointer_glide();

        Ok(())
    }

    fn start_capture(
        &mut self,
        purpose: CapturePurpose,
        qh: &QueueHandle<Self>,
    ) -> Result<(), String> {
        self.destroy_capture();
        let hypr_context = hyprland::context().ok();
        let (_, output) = self.select_output(hypr_context.as_ref())?;
        let frame = self
            .screencopy_manager
            .capture_output(0, &output.wl_output, qh, ());
        self.capture = Some(ActiveCapture {
            purpose,
            output,
            frame,
            spec: None,
            shm_file: None,
            buffer: None,
            flags: ScreencopyFlags::empty(),
            ready: false,
            failed: false,
        });
        Ok(())
    }

    fn select_output(
        &self,
        hypr_context: Option<&HyprContext>,
    ) -> Result<(usize, OutputInfo), String> {
        if let Some(focused_name) = hypr_context.and_then(|ctx| ctx.focused_monitor())
            && let Some((idx, info)) = self
                .outputs
                .iter()
                .enumerate()
                .find(|(_, info)| info.name.as_deref() == Some(focused_name))
        {
            return Ok((idx, info.clone()));
        }
        self.outputs
            .first()
            .cloned()
            .map(|output| (0, output))
            .ok_or_else(|| "no Wayland outputs available".to_string())
    }

    fn finish_capture(&mut self, qh: &QueueHandle<Self>) -> Result<(), String> {
        let mut capture = self
            .capture
            .take()
            .ok_or_else(|| "internal error: missing active capture".to_string())?;
        let spec = capture
            .spec
            .ok_or_else(|| "screencopy completed without a buffer spec".to_string())?;
        let mut file = capture
            .shm_file
            .take()
            .ok_or_else(|| "screencopy completed without a backing file".to_string())?;
        let byte_len = (spec.stride as usize) * (spec.height as usize);
        file.seek(SeekFrom::Start(0))
            .map_err(|err| format!("failed to seek screencopy buffer: {err}"))?;
        let mut bytes = vec![0u8; byte_len];
        file.read_exact(&mut bytes)
            .map_err(|err| format!("failed to read screencopy pixels: {err}"))?;

        if let Some(buffer) = capture.buffer.take() {
            buffer.destroy();
        }
        capture.frame.destroy();

        let raw_frame = RawCaptureFrame {
            width: spec.width,
            height: spec.height,
            stride: spec.stride as usize,
            format: spec.format,
            invert_y: capture.flags.contains(ScreencopyFlags::YInvert),
            transform: capture.output.as_output().transform,
            bytes: &bytes,
        };
        let upright_bgr = decode_raw_frame_to_bgr(raw_frame)?;
        let upright_w = upright_bgr.cols();
        let upright_h = upright_bgr.rows();
        let geometry = FrameGeometry {
            output: capture.output.as_output(),
            pixel_width: upright_w,
            pixel_height: upright_h,
        };

        if capture.purpose == CapturePurpose::Snapshot {
            let (targets, debug, _) = detect_with_debug(&upright_bgr, self.settings.detector, true)
                .map_err(|err| format!("detector failed: {err}"))?;
            let targets = if self.settings.flat {
                crate::flatten_leaves(&targets).unwrap_or(targets)
            } else {
                targets
            };
            let rendered = crate::render::hierarchy_overlay(&upright_bgr, &targets)?;
            let timings = serde_json::json!({
                "signal_ms": debug.timings.stage1_gradients_ms,
                "frames_ms": debug.timings.stage2_frames_ms,
                "coarse_ms": debug.timings.stage3_strokes_ms,
                "recursive_ms": debug.timings.stage5_post_ms,
                "total_ms": debug.timings.total_ms,
            });
            let directory = self
                .snapshot_dir
                .as_deref()
                .ok_or("snapshot directory is missing")?;
            let saved = snapshot::save(
                &upright_bgr,
                &rendered,
                &targets,
                &capture.output.as_output(),
                &timings,
                &debug,
                self.settings.detector,
                directory,
            )?;
            self.snapshot_saved = Some(saved);
            self.should_exit = true;
            return Ok(());
        }

        if capture.purpose == CapturePurpose::PresentOverlay(Mode::Grid) {
            let session = Session::grid(upright_w, upright_h);
            let (logical_w, logical_h) = capture.output.resolved_logical_size();
            self.overlay = Some(ActiveOverlay {
                output: capture.output.clone(),
                geometry,
                base_bgr: upright_bgr,
                debug_data: None,
                session,
                last_loop_refresh: Instant::now(),
            });
            self.create_overlay_layer(
                &capture.output.wl_output,
                logical_w as u32,
                logical_h as u32,
                qh,
            )?;
            return Ok(());
        }

        let bgr_bytes = mat_to_bgr_vec(&upright_bgr)?;
        let job_id = self.next_job_id;
        self.next_job_id += 1;
        self.inflight_job = Some((job_id, capture.purpose));
        let _ = self.worker_tx.send(WorkerJob {
            job_id,
            purpose: capture.purpose,
            output_name: capture.output.name.clone(),
            geometry,
            upright_w,
            upright_h,
            bgr_bytes,
            options: self.settings.detector,
            flat: self.settings.flat,
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn present_precomputed_frame(
        &mut self,
        _output_idx: usize,
        output: OutputInfo,
        geometry: FrameGeometry,
        frame: PrecomputedFrame,
        focused_rect: Option<[i32; 4]>,
        cached_hit: bool,
        qh: &QueueHandle<Self>,
    ) -> Result<(), String> {
        let base_bgr = bgr_vec_to_mat(&frame.bgr, frame.width, frame.height)?;
        let mut session =
            Session::hints_with_focus(frame.width, frame.height, frame.targets, focused_rect)
                .with_telemetry(cached_hit, frame.pipeline_ms);
        if std::mem::take(&mut self.debug_on_present) {
            let _ = session.handle_input(Input::ToggleDebugHud);
            let _ = session.handle_input(Input::CycleDebugNext);
        }

        let (logical_w, logical_h) = output.resolved_logical_size();
        self.overlay = Some(ActiveOverlay {
            output: output.clone(),
            geometry,
            base_bgr,
            debug_data: Some(frame.debug),
            session,
            last_loop_refresh: Instant::now(),
        });
        self.create_overlay_layer(&output.wl_output, logical_w as u32, logical_h as u32, qh)?;
        Ok(())
    }

    fn finish_overlay_refresh(
        &mut self,
        geometry: FrameGeometry,
        frame: PrecomputedFrame,
        focused_rect: Option<[i32; 4]>,
        qh: &QueueHandle<Self>,
    ) -> Result<(), String> {
        let base_bgr = bgr_vec_to_mat(&frame.bgr, frame.width, frame.height)?;
        if let Some(overlay) = self.overlay.as_mut() {
            overlay.geometry = geometry;
            overlay.base_bgr = base_bgr;
            overlay.debug_data = Some(frame.debug);
            overlay.session.update_targets(
                frame.width,
                frame.height,
                frame.targets,
                focused_rect,
                frame.pipeline_ms,
            );
            overlay.last_loop_refresh = Instant::now();
            let (logical_w, logical_h) = overlay.output.resolved_logical_size();
            let wl_output = overlay.output.wl_output.clone();
            if self.layer.is_none() {
                self.create_overlay_layer(&wl_output, logical_w as u32, logical_h as u32, qh)?;
            } else if let Some(layer) = self.layer.as_mut() {
                layer.dirty = true;
            }
        }
        Ok(())
    }

    /// Unmap the layer-shell surface, wait for compositor sync, optionally inject scroll,
    /// and re-capture the screen cleanly without capturing the overlay itself.
    fn begin_overlay_re_capture(&mut self, scroll: Option<(i32, i32)>, qh: &QueueHandle<Self>) {
        if let Some((dx, dy)) = scroll
            && let Some(overlay) = self.overlay.as_mut()
        {
            overlay.session.mark_scrolled(dx, dy);
        }
        self.destroy_layer_only();
        self.pending_re_capture = Some(PendingReCapture {
            scroll,
            settle_until: None,
        });
        let _callback: WlCallback = self.display.sync(qh, ());
        self.sync_pending = true;
    }

    fn inject_scroll(&mut self, dx: i32, dy: i32, qh: &QueueHandle<Self>) {
        // The virtual pointer is already required for clicking. Using it for
        // wheel events avoids both /dev/uinput permission failures and scrolling
        // the wrong window when the current pointer is outside the focused one.
        let Some(overlay) = self.overlay.as_ref() else {
            return;
        };
        let output = &overlay.output;
        let pointer = self
            .virtual_pointer_manager
            .create_virtual_pointer_with_output(
                self.seat.as_ref(),
                Some(&output.wl_output),
                qh,
                (),
            );
        let (logical_w, logical_h) = output.resolved_logical_size();
        let [fx, fy, fw, fh] = overlay.session.focused_rect().unwrap_or([
            0,
            0,
            overlay.base_bgr.cols(),
            overlay.base_bgr.rows(),
        ]);
        let (lx, ly) = overlay.geometry.upright_pixel_to_logical(
            fx + fw / 2,
            fy + fh / 2,
            logical_w,
            logical_h,
        );
        pointer.motion_absolute(
            0,
            lx.round() as u32,
            ly.round() as u32,
            logical_w as u32,
            logical_h as u32,
        );
        pointer.frame();
        if dy != 0 {
            pointer.axis_discrete(1, wl_pointer::Axis::VerticalScroll, (-dy * 10) as f64, -dy);
        }
        if dx != 0 {
            pointer.axis_discrete(1, wl_pointer::Axis::HorizontalScroll, (dx * 10) as f64, dx);
        }
        pointer.frame();
        pointer.destroy();
    }

    fn create_overlay_layer(
        &mut self,
        wl_output: &WlOutput,
        logical_width: u32,
        logical_height: u32,
        qh: &QueueHandle<Self>,
    ) -> Result<(), String> {
        self.destroy_layer_only();
        let surface = self.compositor.create_surface(qh, ());
        let layer_surface = self.layer_shell.get_layer_surface(
            &surface,
            Some(wl_output),
            Layer::Overlay,
            "univim".to_string(),
            qh,
            (),
        );
        layer_surface.set_anchor(Anchor::Top | Anchor::Bottom | Anchor::Left | Anchor::Right);
        layer_surface.set_exclusive_zone(-1);
        layer_surface.set_keyboard_interactivity(KeyboardInteractivity::Exclusive);
        layer_surface.set_size(logical_width.max(1), logical_height.max(1));
        let viewport = self
            .viewporter
            .as_ref()
            .map(|vp| vp.get_viewport(&surface, qh, ()));
        surface.commit();

        self.layer = Some(OverlayLayer {
            surface,
            layer_surface,
            viewport,
            logical_width: logical_width.max(1),
            logical_height: logical_height.max(1),
            configured: false,
            closed: false,
            slot: None,
            dirty: true,
        });
        Ok(())
    }

    fn render_and_commit_overlay(&mut self, qh: &QueueHandle<Self>) -> Result<(), String> {
        let Some(overlay) = self.overlay.as_ref() else {
            return Ok(());
        };
        let rendered: RenderedSurface = render_session_surface(
            &overlay.base_bgr,
            &overlay.session,
            &overlay.geometry,
            overlay.debug_data.as_ref(),
            &self.settings,
        )
        .map_err(|err| format!("failed to render overlay surface: {err}"))?;

        let Some(layer) = self.layer.as_mut() else {
            return Ok(());
        };
        let needs_new_slot = layer.slot.as_ref().is_none_or(|slot| {
            slot.width != rendered.width
                || slot.height != rendered.height
                || slot.len != rendered.argb.len()
        });
        if needs_new_slot {
            if let Some(old) = layer.slot.take() {
                old.buffer.destroy();
            }
            let file =
                tempfile().map_err(|err| format!("failed to allocate overlay shm file: {err}"))?;
            file.set_len(rendered.argb.len() as u64)
                .map_err(|err| format!("failed to size overlay shm file: {err}"))?;
            let pool = self
                .shm
                .create_pool(file.as_fd(), rendered.argb.len() as i32, qh, ());
            let buffer = pool.create_buffer(
                0,
                rendered.width,
                rendered.height,
                rendered.width * 4,
                ShmFormat::Argb8888,
                qh,
                (),
            );
            pool.destroy();
            layer.slot = Some(ShmSlot {
                file,
                buffer,
                len: rendered.argb.len(),
                width: rendered.width,
                height: rendered.height,
            });
        }

        let slot = layer.slot.as_mut().unwrap();
        slot.file
            .seek(SeekFrom::Start(0))
            .map_err(|err| format!("failed to seek overlay shm file: {err}"))?;
        slot.file
            .write_all(&rendered.argb)
            .map_err(|err| format!("failed to write overlay pixels: {err}"))?;
        slot.file
            .flush()
            .map_err(|err| format!("failed to flush overlay pixels: {err}"))?;

        if let Some(viewport) = &layer.viewport {
            viewport.set_destination(layer.logical_width as i32, layer.logical_height as i32);
        } else {
            layer.surface.set_buffer_scale(overlay.output.scale.max(1));
            layer.surface.set_buffer_transform(Transform::Normal);
        }
        layer.surface.attach(Some(&slot.buffer), 0, 0);
        layer
            .surface
            .damage_buffer(0, 0, rendered.width, rendered.height);
        layer.surface.commit();
        layer.dirty = false;
        Ok(())
    }

    fn handle_key_press(&mut self, raw_keycode: u32, qh: &QueueHandle<Self>) {
        let Some(xkb_state) = self.xkb_state.as_ref() else {
            return;
        };
        let keycode = xkb::Keycode::new(raw_keycode + 8);
        let sym = xkb_state.key_get_one_sym(keycode);
        let shift = xkb_state.mod_name_is_active(xkb::MOD_NAME_SHIFT, xkb::STATE_MODS_EFFECTIVE);
        let ctrl = xkb_state.mod_name_is_active(xkb::MOD_NAME_CTRL, xkb::STATE_MODS_EFFECTIVE);
        let alt = xkb_state.mod_name_is_active(xkb::MOD_NAME_ALT, xkb::STATE_MODS_EFFECTIVE);

        let input = match sym {
            xkb::Keysym::Escape => Some(Input::Escape),
            xkb::Keysym::BackSpace => Some(Input::Backspace),
            xkb::Keysym::Return | xkb::Keysym::KP_Enter => Some(Input::Enter),
            xkb::Keysym::F1 => Some(Input::ToggleOptionsMenu),
            xkb::Keysym::F2 => Some(Input::ToggleDebugHud),
            xkb::Keysym::F3 => Some(Input::CycleDebugPrev),
            xkb::Keysym::F4 => Some(Input::CycleDebugNext),
            xkb::Keysym::F5 => Some(Input::Refresh),
            xkb::Keysym::F6 => Some(Input::ToggleLoop),
            xkb::Keysym::Tab => Some(Input::CycleClickAction),
            xkb::Keysym::Page_Down => Some(Input::Scroll { dx: 0, dy: -6 }),
            xkb::Keysym::Page_Up => Some(Input::Scroll { dx: 0, dy: 6 }),
            xkb::Keysym::Up => {
                if self
                    .overlay
                    .as_ref()
                    .is_some_and(|ov| ov.session.show_options_menu())
                {
                    Some(Input::MenuUp)
                } else {
                    Some(Input::Snap(Cardinal::Up))
                }
            }
            xkb::Keysym::Down => {
                if self
                    .overlay
                    .as_ref()
                    .is_some_and(|ov| ov.session.show_options_menu())
                {
                    Some(Input::MenuDown)
                } else {
                    Some(Input::Snap(Cardinal::Down))
                }
            }
            xkb::Keysym::Left => {
                if self
                    .overlay
                    .as_ref()
                    .is_some_and(|ov| ov.session.show_options_menu())
                {
                    Some(Input::MenuLeft)
                } else {
                    Some(Input::Snap(Cardinal::Left))
                }
            }
            xkb::Keysym::Right => {
                if self
                    .overlay
                    .as_ref()
                    .is_some_and(|ov| ov.session.show_options_menu())
                {
                    Some(Input::MenuRight)
                } else {
                    Some(Input::Snap(Cardinal::Right))
                }
            }
            _ => {
                let utf8 = xkb_state.key_get_utf8(keycode);
                // Ctrl+letter can be returned as a control character by xkb;
                // prefer the printable keysym for shortcuts and hint actions.
                let ch_opt = (ctrl && (0x20..=0x7e).contains(&sym.raw()))
                    .then(|| char::from(sym.raw() as u8))
                    .or_else(|| utf8.chars().next())
                    .or_else(|| {
                        let raw = sym.raw();
                        if (0x20..=0x7e).contains(&raw) {
                            char::from_u32(raw)
                        } else {
                            None
                        }
                    });
                if let Some(ch) = ch_opt {
                    let lower = ch.to_ascii_lowercase();
                    let typed_empty = self
                        .overlay
                        .as_ref()
                        .is_some_and(|ov| ov.session.typed().is_empty());
                    if ctrl && !shift && !alt {
                        match lower {
                            'w' => Some(Input::ToggleFocusedOnly),
                            'v' => Some(Input::ToggleDragMode),
                            'r' => Some(Input::Refresh),
                            'd' => Some(Input::Scroll { dx: 0, dy: -5 }),
                            'u' => Some(Input::Scroll { dx: 0, dy: 5 }),
                            'h' => Some(Input::Snap(Cardinal::Left)),
                            'j' => Some(Input::Snap(Cardinal::Down)),
                            'k' => Some(Input::Snap(Cardinal::Up)),
                            'l' => Some(Input::Snap(Cardinal::Right)),
                            _ => Some(Input::Char {
                                ch,
                                shift,
                                ctrl,
                                alt,
                            }),
                        }
                    } else if shift && !ctrl && !alt && typed_empty {
                        match lower {
                            'j' => Some(Input::Scroll { dx: 0, dy: -3 }),
                            'k' => Some(Input::Scroll { dx: 0, dy: 3 }),
                            'h' => Some(Input::Scroll { dx: -3, dy: 0 }),
                            'l' => Some(Input::Scroll { dx: 3, dy: 0 }),
                            _ => Some(Input::Char {
                                ch,
                                shift,
                                ctrl,
                                alt,
                            }),
                        }
                    } else {
                        Some(Input::Char {
                            ch,
                            shift,
                            ctrl,
                            alt,
                        })
                    }
                } else {
                    None
                }
            }
        };

        let Some(input) = input else {
            return;
        };
        let Some(overlay) = self.overlay.as_mut() else {
            return;
        };

        match overlay
            .session
            .handle_input_with_settings(input, &mut self.settings)
        {
            Outcome::Continue => {}
            Outcome::Redraw => {
                if let Some(layer) = self.layer.as_mut() {
                    layer.dirty = true;
                }
            }
            Outcome::SettingsChanged => {
                if let Ok((targets, dbg, ms)) =
                    detect_from_mat(&overlay.base_bgr, self.settings.detector)
                {
                    overlay.debug_data = Some(dbg);
                    overlay.session.update_targets(
                        overlay.base_bgr.cols(),
                        overlay.base_bgr.rows(),
                        targets,
                        None,
                        ms,
                    );
                }
                if let Some(layer) = self.layer.as_mut() {
                    layer.dirty = true;
                }
            }
            Outcome::SaveSettings => {
                let _ = self.settings.save();
                if let Some(layer) = self.layer.as_mut() {
                    layer.dirty = true;
                }
            }
            Outcome::Refresh => {
                self.begin_overlay_re_capture(None, qh);
            }
            Outcome::Scroll { dx, dy } => {
                self.begin_overlay_re_capture(Some((dx, dy)), qh);
            }
            Outcome::Cancel => {
                self.close_overlay();
            }
            Outcome::Click { x, y } => {
                self.schedule_pointer_action(x, y, ClickAction::LeftClick, None, qh);
            }
            Outcome::Action { x, y, action } => {
                self.schedule_pointer_action(x, y, action, None, qh);
            }
            Outcome::Drag {
                from_x,
                from_y,
                to_x,
                to_y,
            } => {
                self.schedule_pointer_action(
                    to_x,
                    to_y,
                    ClickAction::LeftClick,
                    Some((from_x, from_y)),
                    qh,
                );
            }
        }
    }

    fn schedule_pointer_action(
        &mut self,
        x: i32,
        y: i32,
        action: ClickAction,
        drag_from: Option<(i32, i32)>,
        qh: &QueueHandle<Self>,
    ) {
        if self.settings.no_click {
            self.close_overlay();
            return;
        }
        let Some(overlay) = self.overlay.as_ref() else {
            return;
        };
        let (logical_w, logical_h) = overlay.output.resolved_logical_size();
        let (lx, ly) = overlay
            .geometry
            .upright_pixel_to_logical(x, y, logical_w, logical_h);
        let drag_logical = drag_from.map(|(fx, fy)| {
            let (fx, fy) = overlay
                .geometry
                .upright_pixel_to_logical(fx, fy, logical_w, logical_h);
            (
                fx.round().clamp(0.0, f64::from(u32::MAX)) as u32,
                fy.round().clamp(0.0, f64::from(u32::MAX)) as u32,
            )
        });
        let output_index = self
            .outputs
            .iter()
            .position(|candidate| candidate.global_name == overlay.output.global_name)
            .unwrap_or(0);

        self.close_overlay();
        self.pending_click = Some(PendingClick {
            output_index,
            logical_width: logical_w.max(1) as u32,
            logical_height: logical_h.max(1) as u32,
            x: lx.round().clamp(0.0, f64::from(u32::MAX)) as u32,
            y: ly.round().clamp(0.0, f64::from(u32::MAX)) as u32,
            action,
            drag_from: drag_logical,
        });
        let _callback: WlCallback = self.display.sync(qh, ());
        self.sync_pending = true;
    }

    fn emit_virtual_action(&mut self, click: PendingClick, qh: &QueueHandle<Self>) {
        let start = hyprland::cursor_position().and_then(|(x, y)| {
            let output = self.outputs.get(click.output_index)?.as_output();
            let local_x = x - f64::from(output.x);
            let local_y = y - f64::from(output.y);
            if local_x < 0.
                || local_y < 0.
                || local_x >= f64::from(click.logical_width)
                || local_y >= f64::from(click.logical_height)
            {
                None
            } else {
                Some((local_x.round() as u32, local_y.round() as u32))
            }
        });
        let pointer: ZwlrVirtualPointerV1 =
            if let Some(output) = self.outputs.get(click.output_index) {
                self.virtual_pointer_manager
                    .create_virtual_pointer_with_output(
                        self.seat.as_ref(),
                        Some(&output.wl_output),
                        qh,
                        (),
                    )
            } else {
                self.virtual_pointer_manager
                    .create_virtual_pointer(self.seat.as_ref(), qh, ())
            };

        if let Some(from) = start
            && self.settings.pointer_motion != PointerMotion::Instant
            && click.drag_from.is_none()
        {
            let dx = click.x as f64 - from.0 as f64;
            let dy = click.y as f64 - from.1 as f64;
            let distance = dx.hypot(dy);
            if distance >= 2.0 {
                self.pointer_glide = Some(PointerGlide {
                    pointer,
                    click,
                    from,
                    started: Instant::now(),
                    // Bound extreme cross-output travel so selection never stalls
                    // the cursor for an unreasonably long time at low speeds.
                    duration: Duration::from_secs_f64(
                        (distance / f64::from(self.settings.pointer_speed.max(1))).clamp(0.02, 2.0),
                    ),
                    curve: self.settings.pointer_motion,
                });
                return;
            }
        }

        if let Some((from_x, from_y)) = click.drag_from {
            pointer.motion_absolute(
                0,
                from_x,
                from_y,
                click.logical_width.max(1),
                click.logical_height.max(1),
            );
            pointer.frame();
            pointer.button(1, BTN_LEFT, wl_pointer::ButtonState::Pressed);
            pointer.frame();
            let steps = 8u32;
            for step in 1..=steps {
                let ix = ((from_x as i64 * (steps - step) as i64 + click.x as i64 * step as i64)
                    / steps as i64) as u32;
                let iy = ((from_y as i64 * (steps - step) as i64 + click.y as i64 * step as i64)
                    / steps as i64) as u32;
                pointer.motion_absolute(
                    step * 8,
                    ix,
                    iy,
                    click.logical_width.max(1),
                    click.logical_height.max(1),
                );
                pointer.frame();
            }
            pointer.button(80, BTN_LEFT, wl_pointer::ButtonState::Released);
            pointer.frame();
            pointer.destroy();
            return;
        }

        pointer.motion_absolute(
            0,
            click.x,
            click.y,
            click.logical_width.max(1),
            click.logical_height.max(1),
        );
        pointer.frame();

        Self::emit_pointer_button(&pointer, click.action);
        pointer.destroy();
    }

    fn advance_pointer_glide(&mut self) {
        let Some(glide) = self.pointer_glide.as_ref() else {
            return;
        };
        let elapsed = glide.started.elapsed();
        let t = glide_fraction(elapsed, glide.duration, glide.curve);
        let x =
            (glide.from.0 as f64 + (glide.click.x as f64 - glide.from.0 as f64) * t).round() as u32;
        let y =
            (glide.from.1 as f64 + (glide.click.y as f64 - glide.from.1 as f64) * t).round() as u32;
        glide.pointer.motion_absolute(
            0,
            x,
            y,
            glide.click.logical_width,
            glide.click.logical_height,
        );
        glide.pointer.frame();
        if t >= 1.0 {
            let finished = self.pointer_glide.take().unwrap();
            Self::emit_pointer_button(&finished.pointer, finished.click.action);
            finished.pointer.destroy();
        }
    }

    fn emit_pointer_button(pointer: &ZwlrVirtualPointerV1, action: ClickAction) {
        match action {
            ClickAction::HoverOnly => {}
            ClickAction::LeftClick => {
                pointer.button(1, BTN_LEFT, wl_pointer::ButtonState::Pressed);
                pointer.frame();
                pointer.button(2, BTN_LEFT, wl_pointer::ButtonState::Released);
                pointer.frame();
            }
            ClickAction::RightClick => {
                pointer.button(1, BTN_RIGHT, wl_pointer::ButtonState::Pressed);
                pointer.frame();
                pointer.button(2, BTN_RIGHT, wl_pointer::ButtonState::Released);
                pointer.frame();
            }
            ClickAction::MiddleClick => {
                pointer.button(1, BTN_MIDDLE, wl_pointer::ButtonState::Pressed);
                pointer.frame();
                pointer.button(2, BTN_MIDDLE, wl_pointer::ButtonState::Released);
                pointer.frame();
            }
            ClickAction::DoubleClick => {
                pointer.button(1, BTN_LEFT, wl_pointer::ButtonState::Pressed);
                pointer.frame();
                pointer.button(2, BTN_LEFT, wl_pointer::ButtonState::Released);
                pointer.frame();
                pointer.button(15, BTN_LEFT, wl_pointer::ButtonState::Pressed);
                pointer.frame();
                pointer.button(16, BTN_LEFT, wl_pointer::ButtonState::Released);
                pointer.frame();
            }
        }
    }

    fn destroy_capture(&mut self) {
        if let Some(mut capture) = self.capture.take() {
            if let Some(buffer) = capture.buffer.take() {
                buffer.destroy();
            }
            capture.frame.destroy();
        }
    }

    fn destroy_layer_only(&mut self) {
        if let Some(mut layer) = self.layer.take() {
            if let Some(slot) = layer.slot.take() {
                slot.buffer.destroy();
            }
            if let Some(viewport) = layer.viewport.take() {
                viewport.destroy();
            }
            layer.layer_surface.destroy();
            layer.surface.destroy();
        }
    }

    fn close_overlay(&mut self) {
        self.destroy_capture();
        self.pending_re_capture = None;
        self.overlay = None;
        self.destroy_layer_only();
    }
}

fn detector_worker_loop(rx: Receiver<WorkerJob>, tx: Sender<WorkerOutput>) {
    while let Ok(job) = rx.recv() {
        let Ok(bgr_mat) = bgr_vec_to_mat(&job.bgr_bytes, job.upright_w, job.upright_h) else {
            continue;
        };
        let started = Instant::now();
        let Ok((mut targets, debug, _)) = detect_with_debug(&bgr_mat, job.options, true) else {
            continue;
        };
        if job.flat {
            targets = crate::flatten_leaves(&targets).unwrap_or(targets);
        }
        let pipeline_ms = started.elapsed().as_secs_f64() * 1000.0;
        let frame = PrecomputedFrame {
            created_at: Instant::now(),
            output_name: job.output_name,
            width: job.upright_w,
            height: job.upright_h,
            bgr: job.bgr_bytes,
            targets,
            debug,
            pipeline_ms,
        };
        let _ = tx.send(WorkerOutput {
            job_id: job.job_id,
            purpose: job.purpose,
            geometry: job.geometry,
            frame,
        });
    }
}

fn detect_from_mat(
    bgr: &Mat,
    options: DetectorOptions,
) -> Result<(Vec<crate::Target>, DebugData, f64), String> {
    let t0 = Instant::now();
    let (targets, dbg, _) =
        detect_with_debug(bgr, options, true).map_err(|err| format!("detector failed: {err}"))?;
    Ok((targets, dbg, t0.elapsed().as_secs_f64() * 1000.0))
}

fn mat_to_bgr_vec(mat: &Mat) -> Result<Vec<u8>, String> {
    let rows = mat.rows();
    let cols = mat.cols();
    let mut out = vec![0u8; (rows as usize) * (cols as usize) * 3];
    for r in 0..rows {
        let row = mat
            .at_row::<opencv::core::Vec3b>(r)
            .map_err(|err| format!("mat row read failed: {err}"))?;
        let dst =
            &mut out[(r as usize) * (cols as usize) * 3..(r as usize + 1) * (cols as usize) * 3];
        for (c, px) in row.iter().enumerate() {
            dst[c * 3] = px[0];
            dst[c * 3 + 1] = px[1];
            dst[c * 3 + 2] = px[2];
        }
    }
    Ok(out)
}

fn bgr_vec_to_mat(bytes: &[u8], width: i32, height: i32) -> Result<Mat, String> {
    let mut mat = Mat::new_rows_cols_with_default(
        height,
        width,
        opencv::core::CV_8UC3,
        opencv::core::Scalar::all(0.0),
    )
    .map_err(|err| format!("failed to allocate Mat: {err}"))?;
    for r in 0..height {
        let row = mat
            .at_row_mut::<opencv::core::Vec3b>(r)
            .map_err(|err| format!("mat row write failed: {err}"))?;
        let src =
            &bytes[(r as usize) * (width as usize) * 3..(r as usize + 1) * (width as usize) * 3];
        for c in 0..width as usize {
            row[c] = opencv::core::Vec3b::from([src[c * 3], src[c * 3 + 1], src[c * 3 + 2]]);
        }
    }
    Ok(mat)
}

impl Dispatch<WlRegistry, GlobalListContents> for DaemonState {
    fn event(
        state: &mut Self,
        registry: &WlRegistry,
        event: wl_registry::Event,
        _data: &GlobalListContents,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        match event {
            wl_registry::Event::Global {
                name,
                interface,
                version,
            } => {
                if interface == WlOutput::interface().name {
                    let wl_output = registry.bind::<WlOutput, _, _>(name, version.min(4), qh, name);
                    state.outputs.push(OutputInfo::new(wl_output, name));
                }
            }
            wl_registry::Event::GlobalRemove { name } => {
                state.outputs.retain(|output| output.global_name != name);
            }
            _ => {}
        }
    }
}

impl Dispatch<WlOutput, u32> for DaemonState {
    fn event(
        state: &mut Self,
        _output: &WlOutput,
        event: wl_output::Event,
        global_name: &u32,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        let Some(info) = state
            .outputs
            .iter_mut()
            .find(|candidate| candidate.global_name == *global_name)
        else {
            return;
        };
        match event {
            wl_output::Event::Geometry {
                x, y, transform, ..
            } => {
                info.x = x;
                info.y = y;
                if let WEnum::Value(value) = transform {
                    info.transform = value;
                }
            }
            wl_output::Event::Mode {
                flags,
                width,
                height,
                ..
            } => {
                if let WEnum::Value(mode_flags) = flags
                    && mode_flags.contains(wl_output::Mode::Current)
                {
                    info.mode_width = width;
                    info.mode_height = height;
                }
            }
            wl_output::Event::Scale { factor } => {
                info.scale = factor.max(1);
            }
            wl_output::Event::Name { name } => {
                info.name = Some(name);
            }
            _ => {}
        }
    }
}

impl Dispatch<WlSeat, ()> for DaemonState {
    fn event(
        state: &mut Self,
        seat: &WlSeat,
        event: wl_seat::Event,
        _data: &(),
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities {
            capabilities: WEnum::Value(capabilities),
        } = event
            && capabilities.contains(wl_seat::Capability::Keyboard)
            && state.keyboard.is_none()
        {
            state.keyboard = Some(seat.get_keyboard(qh, ()));
        }
    }
}

impl Dispatch<WlKeyboard, ()> for DaemonState {
    fn event(
        state: &mut Self,
        _keyboard: &WlKeyboard,
        event: wl_keyboard::Event,
        _data: &(),
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        match event {
            wl_keyboard::Event::Keymap {
                format: WEnum::Value(wl_keyboard::KeymapFormat::XkbV1),
                fd,
                size,
            } => {
                let keymap = unsafe {
                    xkb::Keymap::new_from_fd(
                        &state.xkb_context,
                        fd,
                        size as usize,
                        xkb::KEYMAP_FORMAT_TEXT_V1,
                        xkb::KEYMAP_COMPILE_NO_FLAGS,
                    )
                };
                if let Ok(Some(keymap)) = keymap {
                    state.xkb_state = Some(xkb::State::new(&keymap));
                }
            }
            wl_keyboard::Event::Modifiers {
                mods_depressed,
                mods_latched,
                mods_locked,
                group,
                ..
            } => {
                if let Some(xkb_state) = state.xkb_state.as_mut() {
                    xkb_state.update_mask(mods_depressed, mods_latched, mods_locked, 0, 0, group);
                }
            }
            wl_keyboard::Event::Key {
                key,
                state: WEnum::Value(KeyState::Pressed),
                ..
            } => {
                state.handle_key_press(key, qh);
            }
            _ => {}
        }
    }
}

impl Dispatch<ZwlrScreencopyFrameV1, ()> for DaemonState {
    fn event(
        state: &mut Self,
        frame: &ZwlrScreencopyFrameV1,
        event: zwlr_screencopy_frame_v1::Event,
        _data: &(),
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        let Some(capture) = state.capture.as_mut() else {
            return;
        };
        if &capture.frame != frame {
            return;
        }
        match event {
            zwlr_screencopy_frame_v1::Event::Buffer {
                format: WEnum::Value(format),
                width,
                height,
                stride,
            } => {
                if matches!(
                    format,
                    ShmFormat::Argb8888
                        | ShmFormat::Xrgb8888
                        | ShmFormat::Abgr8888
                        | ShmFormat::Xbgr8888
                ) && capture.spec.is_none()
                {
                    let spec = BufferSpec {
                        format,
                        width,
                        height,
                        stride,
                    };
                    let byte_len = (stride as usize) * (height as usize);
                    match tempfile().and_then(|file| {
                        file.set_len(byte_len as u64)?;
                        Ok(file)
                    }) {
                        Ok(file) => {
                            let pool = state.shm.create_pool(file.as_fd(), byte_len as i32, qh, ());
                            let buffer = pool.create_buffer(
                                0,
                                width as i32,
                                height as i32,
                                stride as i32,
                                format,
                                qh,
                                (),
                            );
                            pool.destroy();
                            capture.spec = Some(spec);
                            capture.shm_file = Some(file);
                            capture.buffer = Some(buffer.clone());
                            if frame.version() < 3 {
                                frame.copy(&buffer);
                            }
                        }
                        Err(err) => {
                            state.last_error =
                                Some(format!("failed to allocate screencopy shm file: {err}"));
                            capture.failed = true;
                        }
                    }
                }
            }
            zwlr_screencopy_frame_v1::Event::BufferDone => {
                if let Some(buffer) = &capture.buffer {
                    frame.copy(buffer);
                } else {
                    state.last_error = Some(
                        "zwlr_screencopy_frame_v1 offered no supported 32-bit SHM format"
                            .to_string(),
                    );
                    capture.failed = true;
                }
            }
            zwlr_screencopy_frame_v1::Event::Flags {
                flags: WEnum::Value(flags),
            } => {
                capture.flags = flags;
            }
            zwlr_screencopy_frame_v1::Event::Ready { .. } => {
                capture.ready = true;
            }
            zwlr_screencopy_frame_v1::Event::Failed => {
                capture.failed = true;
            }
            _ => {}
        }
    }
}

impl Dispatch<ZwlrLayerSurfaceV1, ()> for DaemonState {
    fn event(
        state: &mut Self,
        layer_surface: &ZwlrLayerSurfaceV1,
        event: zwlr_layer_surface_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        let Some(layer) = state.layer.as_mut() else {
            return;
        };
        if &layer.layer_surface != layer_surface {
            return;
        }
        match event {
            zwlr_layer_surface_v1::Event::Configure {
                serial,
                width,
                height,
            } => {
                layer_surface.ack_configure(serial);
                if width > 0 {
                    layer.logical_width = width;
                }
                if height > 0 {
                    layer.logical_height = height;
                }
                if let Some(overlay) = state.overlay.as_mut() {
                    overlay.output.logical_width = layer.logical_width as i32;
                    overlay.output.logical_height = layer.logical_height as i32;
                }
                layer.configured = true;
                layer.dirty = true;
            }
            zwlr_layer_surface_v1::Event::Closed => {
                layer.closed = true;
            }
            _ => {}
        }
    }
}

impl Dispatch<HyprlandGlobalShortcutV1, CommandAction> for DaemonState {
    fn event(
        state: &mut Self,
        _shortcut: &HyprlandGlobalShortcutV1,
        event: hyprland_global_shortcut_v1::Event,
        action: &CommandAction,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if matches!(event, hyprland_global_shortcut_v1::Event::Pressed { .. }) {
            let _ = state.enqueue_command(*action, qh);
        }
    }
}

impl Dispatch<WlCallback, ()> for DaemonState {
    fn event(
        state: &mut Self,
        _callback: &WlCallback,
        event: wl_callback::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if matches!(event, wl_callback::Event::Done { .. }) {
            state.sync_pending = false;
        }
    }
}

impl Dispatch<WlShm, ()> for DaemonState {
    fn event(
        _state: &mut Self,
        _shm: &WlShm,
        _event: wl_shm::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlBuffer, ()> for DaemonState {
    fn event(
        _state: &mut Self,
        _buffer: &WlBuffer,
        _event: wl_buffer::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<HyprlandGlobalShortcutsManagerV1, ()> for DaemonState {
    fn event(
        _state: &mut Self,
        _manager: &HyprlandGlobalShortcutsManagerV1,
        _event: hyprland_global_shortcuts_manager_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

delegate_noop!(DaemonState: ignore WlCompositor);
delegate_noop!(DaemonState: ignore WlSurface);
delegate_noop!(DaemonState: ignore WlShmPool);
delegate_noop!(DaemonState: ignore ZwlrLayerShellV1);
delegate_noop!(DaemonState: ignore ZwlrScreencopyManagerV1);
delegate_noop!(DaemonState: ignore ZwlrVirtualPointerManagerV1);
delegate_noop!(DaemonState: ignore ZwlrVirtualPointerV1);
delegate_noop!(DaemonState: ignore WpViewporter);
delegate_noop!(DaemonState: ignore WpViewport);
