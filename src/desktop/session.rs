use super::RuntimeSettings;
use crate::{DebugData, Target, drill_down, generate_vimium_hints};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct Session {
    mode: Mode,
    screen_w: i32,
    screen_h: i32,
    root_targets: Vec<Target>,
    targets: Vec<Target>,
    typed: String,
    grid_step_idx: Option<usize>,
    action: ClickAction,
    debug_view: DebugView,
    show_debug_hud: bool,
    show_options_menu: bool,
    selected_option: usize,
    cached_hit: bool,
    pipeline_ms: f64,
    focused_rect: Option<[i32; 4]>,
    focused_only: bool,
    snapped_idx: Option<usize>,
    drag_mode: bool,
    drag_anchor: Option<(i32, i32)>,
    auto_loop: bool,
    scroll_feedback: Option<(i32, i32)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Hints,
    Drilldown,
    Grid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClickAction {
    LeftClick,
    RightClick,
    DoubleClick,
    MiddleClick,
    HoverOnly,
}

impl ClickAction {
    pub fn next(self) -> Self {
        match self {
            Self::LeftClick => Self::RightClick,
            Self::RightClick => Self::DoubleClick,
            Self::DoubleClick => Self::MiddleClick,
            Self::MiddleClick => Self::HoverOnly,
            Self::HoverOnly => Self::LeftClick,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::LeftClick => "L-Click",
            Self::RightClick => "R-Click",
            Self::DoubleClick => "Dbl-Click",
            Self::MiddleClick => "M-Click",
            Self::HoverOnly => "Hover",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DebugView {
    OverlayHud,
    RawPhysics,
    Gradients,
    BarriersFrames,
    StrokesCoarse,
    Hierarchy,
}

impl DebugView {
    pub fn next(self) -> Self {
        match self {
            Self::OverlayHud => Self::RawPhysics,
            Self::RawPhysics => Self::Gradients,
            Self::Gradients => Self::BarriersFrames,
            Self::BarriersFrames => Self::StrokesCoarse,
            Self::StrokesCoarse => Self::Hierarchy,
            Self::Hierarchy => Self::OverlayHud,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::OverlayHud => Self::Hierarchy,
            Self::RawPhysics => Self::OverlayHud,
            Self::Gradients => Self::RawPhysics,
            Self::BarriersFrames => Self::Gradients,
            Self::StrokesCoarse => Self::BarriersFrames,
            Self::Hierarchy => Self::StrokesCoarse,
        }
    }

    pub fn from_number(n: u8) -> Option<Self> {
        match n {
            1 => Some(Self::OverlayHud),
            2 => Some(Self::RawPhysics),
            3 => Some(Self::Gradients),
            4 => Some(Self::BarriersFrames),
            5 => Some(Self::StrokesCoarse),
            6 => Some(Self::Hierarchy),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::OverlayHud => "1:Final Overlay",
            Self::RawPhysics => "2:Stage0 Physics",
            Self::Gradients => "3:Stage1 Gradients",
            Self::BarriersFrames => "4:Stage2 Frames",
            Self::StrokesCoarse => "5:Stage3 Strokes",
            Self::Hierarchy => "6:Stage4 Hierarchy",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cardinal {
    Left,
    Down,
    Up,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    Char {
        ch: char,
        shift: bool,
        ctrl: bool,
        alt: bool,
    },
    Backspace,
    Escape,
    Enter,
    ToggleDebugHud,
    CycleDebugNext,
    CycleDebugPrev,
    #[allow(dead_code)]
    SetDebugStage(u8),
    ToggleOptionsMenu,
    CycleClickAction,
    ToggleFocusedOnly,
    ToggleDragMode,
    Refresh,
    ToggleLoop,
    Scroll {
        dx: i32,
        dy: i32,
    },
    Snap(Cardinal),
    MenuUp,
    MenuDown,
    MenuLeft,
    MenuRight,
    #[allow(dead_code)]
    MenuSave,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Continue,
    Redraw,
    Cancel,
    Click {
        x: i32,
        y: i32,
    },
    Action {
        x: i32,
        y: i32,
        action: ClickAction,
    },
    Drag {
        from_x: i32,
        from_y: i32,
        to_x: i32,
        to_y: i32,
    },
    Scroll {
        dx: i32,
        dy: i32,
    },
    Refresh,
    SettingsChanged,
    SaveSettings,
}

/// Speculative detection cache populated on the first CapsLock tap.
#[derive(Clone)]
pub struct PrecomputedFrame {
    pub created_at: Instant,
    pub output_name: Option<String>,
    pub width: i32,
    pub height: i32,
    pub bgr: Vec<u8>,
    pub targets: Vec<Target>,
    pub debug: DebugData,
    pub pipeline_ms: f64,
}

impl PrecomputedFrame {
    pub fn is_fresh(&self, ttl: Duration, expected_output: Option<&str>) -> bool {
        if self.created_at.elapsed() > ttl {
            return false;
        }
        match (self.output_name.as_deref(), expected_output) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GestureKey {
    CapsLock,
    Grid,
    Debug,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TapEvent {
    /// First tap recorded: caller should start background capture + detection warmup.
    Warmup,
    /// Second tap arrived within the chord window: caller should immediately present overlay!
    Activate,
}

/// Tracks modifier-specific CapsLock double taps.
#[derive(Clone, Debug, Default)]
pub struct TapTracker {
    last_key: Option<GestureKey>,
    last_tap_at: Option<Instant>,
}

impl TapTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn on_tap(&mut self, key: GestureKey, threshold: Duration) -> TapEvent {
        let now = Instant::now();
        if let (Some(prev_key), Some(prev_at)) = (self.last_key, self.last_tap_at)
            && prev_key == key
            && now.saturating_duration_since(prev_at) <= threshold
        {
            self.last_key = None;
            self.last_tap_at = None;
            return TapEvent::Activate;
        }
        self.last_key = Some(key);
        self.last_tap_at = Some(now);
        TapEvent::Warmup
    }

    pub fn on_other_key(&mut self) {
        self.last_key = None;
        self.last_tap_at = None;
    }

    pub fn clear(&mut self) {
        self.last_key = None;
        self.last_tap_at = None;
    }
}

/// Prioritize targets so that:
/// 1. Targets inside the focused window (`focused_rect`) receive the shortest, easiest
///    home-row hint labels (`S`, `A`, `D`, `F`, `J`, `K`, `L`, `E`...).
/// 2. Within the same focus tier, `level == "control"` targets precede `level == "paragraph"`
///    / `"line"` blocks so interactive controls always get primary labels before prose.
pub fn prioritize_targets_for_focus(
    targets: &mut [Target],
    focused_rect: Option<[i32; 4]>,
    alphabet: &str,
) {
    if targets.is_empty() {
        return;
    }
    let Some([fx, fy, fw, fh]) = focused_rect else {
        return;
    };
    if fw <= 0 || fh <= 0 {
        return;
    }

    let in_focus = |t: &Target| -> bool {
        let cx = t.x + t.w / 2;
        let cy = t.y + t.h / 2;
        cx >= fx && cx < fx + fw && cy >= fy && cy < fy + fh
    };
    let is_control = |t: &Target| -> bool { t.level == "control" };

    targets.sort_by(|a, b| {
        let af = in_focus(a);
        let bf = in_focus(b);
        let ac = is_control(a);
        let bc = is_control(b);
        bf.cmp(&af)
            .then_with(|| bc.cmp(&ac))
            .then_with(|| (a.y / 12).cmp(&(b.y / 12)))
            .then_with(|| a.x.cmp(&b.x))
    });

    let hints = generate_vimium_hints(targets.len(), alphabet);
    for (target, hint) in targets.iter_mut().zip(hints) {
        target.hint = hint;
    }
}

impl Session {
    #[allow(dead_code)]
    pub fn hints(width: i32, height: i32, targets: Vec<Target>) -> Self {
        Self::hints_with_focus(width, height, targets, None)
    }

    pub fn hints_with_focus(
        width: i32,
        height: i32,
        mut targets: Vec<Target>,
        focused_rect: Option<[i32; 4]>,
    ) -> Self {
        prioritize_targets_for_focus(&mut targets, focused_rect, "sadfjklewcmpgh");
        Self {
            mode: Mode::Hints,
            screen_w: width.max(1),
            screen_h: height.max(1),
            root_targets: targets.clone(),
            targets,
            typed: String::new(),
            grid_step_idx: None,
            action: ClickAction::LeftClick,
            debug_view: DebugView::OverlayHud,
            show_debug_hud: false,
            show_options_menu: false,
            selected_option: 0,
            cached_hit: false,
            pipeline_ms: 0.0,
            focused_rect,
            focused_only: false,
            snapped_idx: None,
            drag_mode: false,
            drag_anchor: None,
            auto_loop: false,
            scroll_feedback: None,
        }
    }

    /// Replace the live targets and screen dimensions after an in-overlay scroll or pipeline loop refresh
    /// while preserving all active modal state (debug HUD, focus filter, drag anchor, auto-loop, click action).
    pub fn update_targets(
        &mut self,
        width: i32,
        height: i32,
        mut targets: Vec<Target>,
        focused_rect: Option<[i32; 4]>,
        pipeline_ms: f64,
    ) {
        self.screen_w = width.max(1);
        self.screen_h = height.max(1);
        self.focused_rect = focused_rect.or(self.focused_rect);
        prioritize_targets_for_focus(&mut targets, self.focused_rect, "sadfjklewcmpgh");
        self.root_targets = targets;
        self.pipeline_ms = pipeline_ms;
        self.cached_hit = false;
        self.rebuild_active_targets();
    }

    fn rebuild_active_targets(&mut self) {
        if self.focused_only {
            if let Some([fx, fy, fw, fh]) = self.focused_rect {
                let mut filtered: Vec<Target> = self
                    .root_targets
                    .iter()
                    .filter(|t| {
                        let cx = t.x + t.w / 2;
                        let cy = t.y + t.h / 2;
                        cx >= fx && cx < fx + fw && cy >= fy && cy < fy + fh
                    })
                    .cloned()
                    .collect();
                let hints = generate_vimium_hints(filtered.len(), "sadfjklewcmpgh");
                for (t, h) in filtered.iter_mut().zip(hints) {
                    t.hint = h;
                }
                self.targets = filtered;
            } else {
                self.targets = self.root_targets.clone();
            }
        } else {
            self.targets = self.root_targets.clone();
        }
        if let Some(idx) = self.snapped_idx
            && idx >= self.targets.len()
        {
            self.snapped_idx = if self.targets.is_empty() {
                None
            } else {
                Some(0)
            };
        }
    }

    pub fn with_telemetry(mut self, cached_hit: bool, pipeline_ms: f64) -> Self {
        self.cached_hit = cached_hit;
        self.pipeline_ms = pipeline_ms;
        self
    }

    pub fn grid(width: i32, height: i32) -> Self {
        let mut session = Self {
            mode: Mode::Grid,
            screen_w: width.max(1),
            screen_h: height.max(1),
            root_targets: Vec::new(),
            targets: Vec::new(),
            typed: String::new(),
            grid_step_idx: Some(0),
            action: ClickAction::LeftClick,
            debug_view: DebugView::OverlayHud,
            show_debug_hud: false,
            show_options_menu: false,
            selected_option: 0,
            cached_hit: false,
            pipeline_ms: 0.0,
            focused_rect: None,
            focused_only: false,
            snapped_idx: None,
            drag_mode: false,
            drag_anchor: None,
            auto_loop: false,
            scroll_feedback: None,
        };
        session.targets = session.make_grid_Step(0, 0, session.screen_w, session.screen_h);
        session
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn targets(&self) -> &[Target] {
        &self.targets
    }

    pub fn typed(&self) -> &str {
        &self.typed
    }

    pub fn action(&self) -> ClickAction {
        self.action
    }

    pub fn debug_view(&self) -> DebugView {
        self.debug_view
    }

    pub fn show_debug_hud(&self) -> bool {
        self.show_debug_hud
    }

    pub fn show_options_menu(&self) -> bool {
        self.show_options_menu
    }

    pub fn selected_option(&self) -> usize {
        self.selected_option
    }

    pub fn cached_hit(&self) -> bool {
        self.cached_hit
    }

    pub fn pipeline_ms(&self) -> f64 {
        self.pipeline_ms
    }

    pub fn focused_rect(&self) -> Option<[i32; 4]> {
        self.focused_rect
    }

    pub fn focused_only(&self) -> bool {
        self.focused_only
    }

    pub fn snapped_target(&self) -> Option<&Target> {
        self.snapped_idx.and_then(|idx| self.targets.get(idx))
    }

    pub fn drag_mode(&self) -> bool {
        self.drag_mode
    }

    pub fn drag_anchor(&self) -> Option<(i32, i32)> {
        self.drag_anchor
    }

    pub fn auto_loop(&self) -> bool {
        self.auto_loop
    }

    pub fn scroll_feedback(&self) -> Option<(i32, i32)> {
        self.scroll_feedback
    }

    pub fn mark_scrolled(&mut self, dx: i32, dy: i32) {
        self.scroll_feedback = Some((dx, dy));
    }

    pub fn handle_input(&mut self, input: Input) -> Outcome {
        let mut dummy = RuntimeSettings::default();
        self.handle_input_with_settings(input, &mut dummy)
    }

    pub fn handle_input_with_settings(
        &mut self,
        input: Input,
        settings: &mut RuntimeSettings,
    ) -> Outcome {
        self.scroll_feedback = None;
        if self.show_options_menu {
            match input {
                Input::Escape | Input::ToggleOptionsMenu => {
                    self.show_options_menu = false;
                    return Outcome::Redraw;
                }
                Input::MenuUp => {
                    self.selected_option = (self.selected_option + RuntimeSettings::OPTION_COUNT
                        - 1)
                        % RuntimeSettings::OPTION_COUNT;
                    return Outcome::Redraw;
                }
                Input::MenuDown => {
                    self.selected_option =
                        (self.selected_option + 1) % RuntimeSettings::OPTION_COUNT;
                    return Outcome::Redraw;
                }
                Input::MenuLeft => {
                    settings.adjust_option(self.selected_option, -1);
                    return Outcome::SettingsChanged;
                }
                Input::MenuRight | Input::Enter => {
                    settings.adjust_option(self.selected_option, 1);
                    return Outcome::SettingsChanged;
                }
                Input::MenuSave => {
                    return Outcome::SaveSettings;
                }
                Input::Char { ch, .. } => match ch.to_ascii_lowercase() {
                    'j' => {
                        self.selected_option =
                            (self.selected_option + 1) % RuntimeSettings::OPTION_COUNT;
                        return Outcome::Redraw;
                    }
                    'k' => {
                        self.selected_option =
                            (self.selected_option + RuntimeSettings::OPTION_COUNT - 1)
                                % RuntimeSettings::OPTION_COUNT;
                        return Outcome::Redraw;
                    }
                    'h' => {
                        settings.adjust_option(self.selected_option, -1);
                        return Outcome::SettingsChanged;
                    }
                    'l' | ' ' => {
                        settings.adjust_option(self.selected_option, 1);
                        return Outcome::SettingsChanged;
                    }
                    's' => return Outcome::SaveSettings,
                    'q' => {
                        self.show_options_menu = false;
                        return Outcome::Redraw;
                    }
                    _ => return Outcome::Continue,
                },
                _ => return Outcome::Continue,
            }
        }

        match input {
            Input::Escape => {
                if self.drag_anchor.is_some() {
                    self.drag_anchor = None;
                    self.typed.clear();
                    return Outcome::Redraw;
                }
                if !self.typed.is_empty() {
                    self.typed.clear();
                    return Outcome::Redraw;
                }
                Outcome::Cancel
            }
            Input::Enter => {
                if let Some(target) = self.snapped_target().cloned() {
                    return self.activate_target_point(
                        target.x + target.w / 2,
                        target.y + target.h / 2,
                        self.action,
                    );
                }
                Outcome::Continue
            }
            Input::ToggleDebugHud => {
                self.show_debug_hud = !self.show_debug_hud;
                Outcome::Redraw
            }
            Input::CycleDebugNext => {
                self.show_debug_hud = true;
                self.debug_view = self.debug_view.next();
                Outcome::Redraw
            }
            Input::CycleDebugPrev => {
                self.show_debug_hud = true;
                self.debug_view = self.debug_view.prev();
                Outcome::Redraw
            }
            Input::SetDebugStage(stage) => {
                if let Some(view) = DebugView::from_number(stage) {
                    self.show_debug_hud = true;
                    self.debug_view = view;
                    Outcome::Redraw
                } else {
                    Outcome::Continue
                }
            }
            Input::ToggleOptionsMenu => {
                self.show_options_menu = !self.show_options_menu;
                Outcome::Redraw
            }
            Input::CycleClickAction => {
                self.action = self.action.next();
                Outcome::Redraw
            }
            Input::ToggleFocusedOnly => {
                if self.focused_rect.is_some() {
                    self.focused_only = !self.focused_only;
                    self.typed.clear();
                    self.rebuild_active_targets();
                    Outcome::Redraw
                } else {
                    Outcome::Continue
                }
            }
            Input::ToggleDragMode => {
                self.drag_mode = !self.drag_mode;
                if !self.drag_mode {
                    self.drag_anchor = None;
                }
                Outcome::Redraw
            }
            Input::Refresh => Outcome::Refresh,
            Input::ToggleLoop => {
                self.auto_loop = !self.auto_loop;
                Outcome::Redraw
            }
            Input::Scroll { dx, dy } => Outcome::Scroll { dx, dy },
            Input::Snap(dir) => {
                self.move_snap(dir);
                Outcome::Redraw
            }
            Input::Backspace => {
                if self.typed.pop().is_some() {
                    Outcome::Redraw
                } else if self.mode == Mode::Drilldown {
                    self.mode = Mode::Hints;
                    self.rebuild_active_targets();
                    Outcome::Redraw
                } else {
                    Outcome::Continue
                }
            }
            Input::Char {
                ch,
                shift,
                ctrl,
                alt,
            } => {
                if ch == '?' {
                    self.show_options_menu = !self.show_options_menu;
                    return Outcome::Redraw;
                }
                if ch == ';' {
                    self.action = self.action.next();
                    return Outcome::Redraw;
                }
                if ch == ']' {
                    self.show_debug_hud = true;
                    self.debug_view = self.debug_view.next();
                    return Outcome::Redraw;
                }
                if ch == '[' {
                    self.show_debug_hud = true;
                    self.debug_view = self.debug_view.prev();
                    return Outcome::Redraw;
                }
                if ch.is_ascii_digit()
                    && let Some(view) = DebugView::from_number(ch as u8 - b'0')
                {
                    self.show_debug_hud = true;
                    self.debug_view = view;
                    return Outcome::Redraw;
                }
                // If nothing is typed yet, `v` toggles 2-point drag mode and `r` triggers a live pipeline re-capture.
                if self.typed.is_empty() && !shift && !ctrl && !alt {
                    if ch.eq_ignore_ascii_case(&'v') {
                        self.drag_mode = !self.drag_mode;
                        if !self.drag_mode {
                            self.drag_anchor = None;
                        }
                        return Outcome::Redraw;
                    }
                    if ch.eq_ignore_ascii_case(&'r') {
                        return Outcome::Refresh;
                    }
                }
                if !ch.is_ascii_alphabetic() {
                    return Outcome::Continue;
                }
                self.push_char(ch.to_ascii_uppercase(), shift, ctrl, alt)
            }
            _ => Outcome::Continue,
        }
    }

    fn activate_target_point(&mut self, x: i32, y: i32, action: ClickAction) -> Outcome {
        if self.drag_mode {
            if let Some((from_x, from_y)) = self.drag_anchor.take() {
                self.drag_mode = false;
                return Outcome::Drag {
                    from_x,
                    from_y,
                    to_x: x,
                    to_y: y,
                };
            } else {
                self.drag_anchor = Some((x, y));
                self.typed.clear();
                return Outcome::Redraw;
            }
        }
        if action == ClickAction::LeftClick {
            Outcome::Click { x, y }
        } else {
            Outcome::Action { x, y, action }
        }
    }

    fn move_snap(&mut self, dir: Cardinal) {
        if self.targets.is_empty() {
            self.snapped_idx = None;
            return;
        }
        let Some(cur_idx) = self.snapped_idx.filter(|&i| i < self.targets.len()) else {
            // Pick target closest to focused window center or screen center
            let (cx0, cy0) = if let Some([fx, fy, fw, fh]) = self.focused_rect {
                (fx + fw / 2, fy + fh / 2)
            } else {
                (self.screen_w / 2, self.screen_h / 2)
            };
            let best = self
                .targets
                .iter()
                .enumerate()
                .min_by_key(|(_, t)| {
                    let dx = (t.x + t.w / 2 - cx0) as i64;
                    let dy = (t.y + t.h / 2 - cy0) as i64;
                    dx * dx + dy * dy
                })
                .map(|(i, _)| i);
            self.snapped_idx = best;
            return;
        };

        let cur = &self.targets[cur_idx];
        let cx = cur.x + cur.w / 2;
        let cy = cur.y + cur.h / 2;

        let best = self
            .targets
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != cur_idx)
            .filter_map(|(i, t)| {
                let tx = t.x + t.w / 2;
                let ty = t.y + t.h / 2;
                let dx = tx - cx;
                let dy = ty - cy;
                let (primary, ortho) = match dir {
                    Cardinal::Left if dx < -4 => (-dx, dy.abs()),
                    Cardinal::Right if dx > 4 => (dx, dy.abs()),
                    Cardinal::Up if dy < -4 => (-dy, dx.abs()),
                    Cardinal::Down if dy > 4 => (dy, dx.abs()),
                    _ => return None,
                };
                // Weight orthogonal distance by 3x so cardinal navigation stays in the row/column
                let cost = (primary as i64) + (ortho as i64) * 3;
                Some((cost, i))
            })
            .min_by_key(|&(cost, _)| cost)
            .map(|(_, i)| i);

        if let Some(next_idx) = best {
            self.snapped_idx = Some(next_idx);
        }
    }

    fn push_char(&mut self, ch: char, shift: bool, ctrl: bool, alt: bool) -> Outcome {
        let candidate = format!("{}{}", self.typed, ch);
        let matches: Vec<usize> = self
            .targets
            .iter()
            .enumerate()
            .filter_map(|(idx, target)| target.hint.starts_with(&candidate).then_some(idx))
            .collect();

        if matches.is_empty() {
            return Outcome::Continue;
        }

        self.typed = candidate;
        if let Some(&idx) = matches
            .iter()
            .find(|&&idx| self.targets[idx].hint == self.typed)
        {
            self.snapped_idx = Some(idx);
            let target = self.targets[idx].clone();
            let effective_action = if alt {
                ClickAction::HoverOnly
            } else if ctrl && shift {
                ClickAction::DoubleClick
            } else if ctrl {
                ClickAction::MiddleClick
            } else if shift && (self.mode != Mode::Hints || target.children.is_empty()) {
                ClickAction::RightClick
            } else {
                self.action
            };

            match self.mode {
                Mode::Hints => {
                    if (shift || ctrl) && !target.children.is_empty() && !alt {
                        let Ok(sub) = drill_down(&target, shift) else {
                            return Outcome::Redraw;
                        };
                        if sub.len() <= 1 {
                            let chosen = sub.first().unwrap_or(&target);
                            return self.activate_target_point(
                                chosen.x + chosen.w / 2,
                                chosen.y + chosen.h / 2,
                                effective_action,
                            );
                        }
                        self.mode = Mode::Drilldown;
                        self.targets = sub;
                        self.typed.clear();
                        return Outcome::Redraw;
                    }
                    return self.activate_target_point(
                        target.x + target.w / 2,
                        target.y + target.h / 2,
                        effective_action,
                    );
                }
                Mode::Drilldown => {
                    return self.activate_target_point(
                        target.x + target.w / 2,
                        target.y + target.h / 2,
                        effective_action,
                    );
                }
                Mode::Grid => {
                    let step_idx = self.grid_step_idx.unwrap_or(0);
                    if step_idx == 0 {
                        self.grid_step_idx = Some(1);
                        self.targets = self.make_grid_Step(target.x, target.y, target.w, target.h);
                        self.typed.clear();
                        return Outcome::Redraw;
                    }
                    return self.activate_target_point(
                        target.x + target.w / 2,
                        target.y + target.h / 2,
                        effective_action,
                    );
                }
            }
        }

        Outcome::Redraw
    }

    #[allow(non_snake_case)]
    fn make_grid_Step(&self, rx: i32, ry: i32, rw: i32, rh: i32) -> Vec<Target> {
        let cols = 6;
        let rows = 4;
        // The first grid level must fit in single-character labels so one
        // keystroke narrows to a cell and the second picks a subcell.
        let hints = generate_vimium_hints((cols * rows) as usize, "sadfjklewcmpghqwertyuiop");
        let mut cells = Vec::with_capacity((cols * rows) as usize);
        let mut idx = 0usize;
        for row in 0..rows {
            for col in 0..cols {
                let x1 = rx + (col * rw) / cols;
                let y1 = ry + (row * rh) / rows;
                let x2 = rx + ((col + 1) * rw) / cols;
                let y2 = ry + ((row + 1) * rh) / rows;
                cells.push(Target {
                    x: x1,
                    y: y1,
                    w: (x2 - x1).max(1),
                    h: (y2 - y1).max(1),
                    score: 1.0,
                    kind: "grid".to_string(),
                    hint: hints[idx].clone(),
                    level: "grid".to_string(),
                    children: Vec::new(),
                });
                idx += 1;
            }
        }
        cells
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Cardinal, ClickAction, DebugView, GestureKey, Input, Mode, Outcome, Session, TapEvent,
        TapTracker, prioritize_targets_for_focus,
    };
    use crate::Target;
    use std::time::Duration;

    fn sample_target() -> Target {
        let child = |x, y, w, kind: &str, children: Vec<serde_json::Value>| {
            serde_json::to_value(Target {
                x,
                y,
                w,
                h: 20,
                score: 0.85,
                kind: kind.to_string(),
                hint: String::new(),
                level: kind.to_string(),
                children,
            })
            .unwrap()
        };
        Target {
            x: 100,
            y: 200,
            w: 80,
            h: 40,
            score: 0.9,
            kind: "text_block".to_string(),
            hint: "S".to_string(),
            level: "paragraph".to_string(),
            children: vec![
                child(
                    100,
                    200,
                    35,
                    "line",
                    vec![
                        child(100, 200, 16, "word", Vec::new()),
                        child(120, 200, 15, "word", Vec::new()),
                    ],
                ),
                child(100, 220, 60, "line", Vec::new()),
            ],
        }
    }

    #[test]
    fn hints_session_keeps_composite_targets_as_single_overlay_labels() {
        let session = Session::hints(800, 600, vec![sample_target()]);
        assert_eq!(session.targets().len(), 1);
        assert_eq!(session.targets()[0].hint, "S");
        assert_eq!(session.targets()[0].children.len(), 2);
    }

    #[test]
    fn hints_session_clicks_matching_target() {
        let mut session = Session::hints(800, 600, vec![sample_target()]);
        let outcome = session.handle_input(Input::Char {
            ch: 's',
            shift: false,
            ctrl: false,
            alt: false,
        });
        assert_eq!(outcome, Outcome::Click { x: 140, y: 220 });
    }

    #[test]
    fn shift_drills_down_to_words_and_backspace_returns_to_root() {
        let mut session = Session::hints(800, 600, vec![sample_target()]);
        let outcome = session.handle_input(Input::Char {
            ch: 's',
            shift: true,
            ctrl: false,
            alt: false,
        });
        assert_eq!(outcome, Outcome::Redraw);
        assert_eq!(session.mode(), Mode::Drilldown);
        assert_eq!(session.targets().len(), 3);

        let outcome = session.handle_input(Input::Backspace);
        assert_eq!(outcome, Outcome::Redraw);
        assert_eq!(session.mode(), Mode::Hints);
        assert_eq!(session.targets().len(), 1);
    }

    #[test]
    fn grid_session_refines_then_clicks_subcell() {
        let mut session = Session::grid(600, 400);
        let first_hint = session.targets()[0].hint.chars().next().unwrap();
        let outcome = session.handle_input(Input::Char {
            ch: first_hint,
            shift: false,
            ctrl: false,
            alt: false,
        });
        assert_eq!(outcome, Outcome::Redraw);

        let second_hint = session.targets()[0].hint.chars().next().unwrap();
        let outcome = session.handle_input(Input::Char {
            ch: second_hint,
            shift: false,
            ctrl: false,
            alt: false,
        });
        assert_eq!(outcome, Outcome::Click { x: 8, y: 12 });
    }

    #[test]
    fn tap_tracker_warms_up_on_first_tap_and_activates_on_second_including_capslock() {
        let mut tracker = TapTracker::new();
        assert_eq!(
            tracker.on_tap(GestureKey::CapsLock, Duration::from_millis(250)),
            TapEvent::Warmup
        );
        assert_eq!(
            tracker.on_tap(GestureKey::CapsLock, Duration::from_millis(250)),
            TapEvent::Activate
        );
        // Next tap starts a fresh cycle
        assert_eq!(
            tracker.on_tap(GestureKey::Grid, Duration::from_millis(250)),
            TapEvent::Warmup
        );
        assert_eq!(
            tracker.on_tap(GestureKey::Debug, Duration::from_millis(250)),
            TapEvent::Warmup
        );
        assert_eq!(
            tracker.on_tap(GestureKey::Debug, Duration::from_millis(250)),
            TapEvent::Activate
        );
        tracker.on_other_key();
        assert_eq!(
            tracker.on_tap(GestureKey::Grid, Duration::from_millis(250)),
            TapEvent::Warmup
        );
    }

    #[test]
    fn debug_and_action_shortcuts_update_session_state() {
        let mut session = Session::hints(800, 600, vec![sample_target()]);
        assert_eq!(session.action(), ClickAction::LeftClick);
        assert_eq!(
            session.handle_input(Input::CycleClickAction),
            Outcome::Redraw
        );
        assert_eq!(session.action(), ClickAction::RightClick);

        assert!(!session.show_debug_hud());
        assert_eq!(
            session.handle_input(Input::SetDebugStage(3)),
            Outcome::Redraw
        );
        assert!(session.show_debug_hud());
        assert_eq!(session.debug_view(), DebugView::Gradients);
    }

    #[test]
    fn focused_window_and_control_targets_are_prioritized_first() {
        let bg_paragraph = Target {
            x: 20,
            y: 20,
            w: 100,
            h: 40,
            score: 0.95,
            kind: "text_block".to_string(),
            hint: "S".to_string(),
            level: "paragraph".to_string(),
            children: Vec::new(),
        };
        let focused_paragraph = Target {
            x: 500,
            y: 200,
            w: 120,
            h: 40,
            score: 0.85,
            kind: "text_block".to_string(),
            hint: "A".to_string(),
            level: "paragraph".to_string(),
            children: Vec::new(),
        };
        let focused_control = Target {
            x: 520,
            y: 300,
            w: 80,
            h: 30,
            score: 0.90,
            kind: "frame".to_string(),
            hint: "D".to_string(),
            level: "control".to_string(),
            children: Vec::new(),
        };
        let mut targets = vec![bg_paragraph, focused_paragraph, focused_control];
        prioritize_targets_for_focus(&mut targets, Some([450, 150, 300, 300]), "sadfjklewcmpgh");

        // The control inside the focused window comes first ("S"), followed by the focused paragraph ("A"),
        // followed by the unfocused background target ("D").
        assert_eq!(targets[0].level, "control");
        assert_eq!(targets[0].hint, "S");
        assert_eq!(targets[1].x, 500);
        assert_eq!(targets[1].hint, "A");
        assert_eq!(targets[2].x, 20);
        assert_eq!(targets[2].hint, "D");
    }

    #[test]
    fn snap_and_two_point_drag_mode_work() {
        let t1 = Target {
            x: 100,
            y: 100,
            w: 40,
            h: 20,
            score: 0.9,
            kind: "frame".to_string(),
            hint: "S".to_string(),
            level: "control".to_string(),
            children: Vec::new(),
        };
        let t2 = Target {
            x: 300,
            y: 100,
            w: 40,
            h: 20,
            score: 0.9,
            kind: "frame".to_string(),
            hint: "A".to_string(),
            level: "control".to_string(),
            children: Vec::new(),
        };
        let mut session = Session::hints(800, 600, vec![t1, t2]);
        assert_eq!(
            session.handle_input(Input::Snap(Cardinal::Right)),
            Outcome::Redraw
        );
        assert!(session.snapped_target().is_some());

        // Toggle drag mode with `v`
        assert_eq!(
            session.handle_input(Input::Char {
                ch: 'v',
                shift: false,
                ctrl: false,
                alt: false
            }),
            Outcome::Redraw
        );
        assert!(session.drag_mode());

        // First hint selects drag anchor
        assert_eq!(
            session.handle_input(Input::Char {
                ch: 's',
                shift: false,
                ctrl: false,
                alt: false
            }),
            Outcome::Redraw
        );
        assert_eq!(session.drag_anchor(), Some((120, 110)));

        // Second hint completes the drag
        assert_eq!(
            session.handle_input(Input::Char {
                ch: 'a',
                shift: false,
                ctrl: false,
                alt: false
            }),
            Outcome::Drag {
                from_x: 120,
                from_y: 110,
                to_x: 320,
                to_y: 110
            }
        );
    }

    #[test]
    fn modified_leaf_hints_choose_right_and_middle_click() {
        let mut target = sample_target();
        target.children.clear();
        let hint = target.hint.chars().next().unwrap();
        for (shift, ctrl, expected) in [
            (true, false, ClickAction::RightClick),
            (false, true, ClickAction::MiddleClick),
        ] {
            let mut session = Session::hints(800, 600, vec![target.clone()]);
            assert_eq!(
                session.handle_input(Input::Char {
                    ch: hint,
                    shift,
                    ctrl,
                    alt: false
                }),
                Outcome::Action {
                    x: target.x + target.w / 2,
                    y: target.y + target.h / 2,
                    action: expected
                }
            );
        }
    }
}
