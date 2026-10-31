//! Pure selection and modifier-tap state, independent of Wayland event delivery.
use crate::{Target, drill_down, hints};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Hints,
    Grid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    Character(char),
    Backspace,
    Escape,
    Tab,
    Parent,
    Enter,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Outcome {
    None,
    Changed,
    Cancel,
    Click { x: f64, y: f64 },
}

pub struct Session {
    pub mode: Mode,
    pub width: i32,
    pub height: i32,
    pub targets: Vec<Target>,
    pub prefix: String,
    /// Main-grid (row, column), each in 0..26.
    pub grid_cell: Option<(usize, usize)>,
    pub drill: bool,
    history: Vec<Vec<Target>>,
}

impl Session {
    pub fn hints(width: i32, height: i32, targets: Vec<Target>) -> Self {
        let mut session = Self::grid(width, height);
        session.mode = Mode::Hints;
        session.targets = session.label_targets(targets);
        session
    }

    pub fn grid(width: i32, height: i32) -> Self {
        Self {
            mode: Mode::Grid,
            width,
            height,
            targets: vec![],
            prefix: String::new(),
            grid_cell: None,
            drill: false,
            history: vec![],
        }
    }

    fn valid_target(&self, t: &Target) -> bool {
        t.x >= 0
            && t.y >= 0
            && t.w > 0
            && t.h > 0
            && i64::from(t.x) + i64::from(t.w) <= i64::from(self.width)
            && i64::from(t.y) + i64::from(t.h) <= i64::from(self.height)
    }

    fn label_targets(&self, targets: Vec<Target>) -> Vec<Target> {
        let mut targets: Vec<_> = targets
            .into_iter()
            .filter(|t| self.valid_target(t))
            .collect();
        let labels = hints(targets.len());
        for (target, label) in targets.iter_mut().zip(labels) {
            target.hint = label;
        }
        targets
    }

    /// Filtering never removes targets or regenerates their assigned labels.
    pub fn visible_targets(&self) -> impl Iterator<Item = &Target> {
        self.targets.iter().filter(|t| {
            self.mode == Mode::Hints && self.valid_target(t) && t.hint.starts_with(&self.prefix)
        })
    }

    fn parent(&mut self) -> Outcome {
        self.prefix.clear();
        if self.mode == Mode::Grid {
            return if self.grid_cell.take().is_some() {
                Outcome::Changed
            } else {
                Outcome::None
            };
        }
        if let Some(targets) = self.history.pop() {
            self.targets = targets;
            Outcome::Changed
        } else {
            Outcome::None
        }
    }

    fn select(&mut self) -> Outcome {
        let Some(target) = self
            .visible_targets()
            .find(|t| t.hint == self.prefix)
            .cloned()
        else {
            return Outcome::None;
        };
        if self.drill && !target.children.is_empty() {
            let Ok(children) = drill_down(&target, false) else {
                return Outcome::None;
            };
            let children = self.label_targets(children);
            if children.is_empty() {
                return Outcome::None;
            }
            self.history
                .push(std::mem::replace(&mut self.targets, children));
            self.prefix.clear();
            return Outcome::Changed;
        }
        Outcome::Click {
            x: f64::from(target.x) + f64::from(target.w) / 2.0,
            y: f64::from(target.y) + f64::from(target.h) / 2.0,
        }
    }

    pub fn input(&mut self, input: Input) -> Outcome {
        match input {
            Input::Escape => Outcome::Cancel,
            Input::Tab => {
                self.drill = !self.drill;
                Outcome::Changed
            }
            Input::Parent => {
                let had_prefix = !self.prefix.is_empty();
                let result = self.parent();
                if had_prefix { Outcome::Changed } else { result }
            }
            Input::Backspace => {
                if self.prefix.pop().is_some() {
                    Outcome::Changed
                } else {
                    self.parent()
                }
            }
            Input::Enter => {
                if self.mode == Mode::Hints {
                    self.select()
                } else {
                    Outcome::None
                }
            }
            Input::Character(c) => {
                if !c.is_ascii_alphanumeric() {
                    return Outcome::None;
                }
                let c = c.to_ascii_uppercase();
                if self.mode == Mode::Hints {
                    let candidate = format!("{}{c}", self.prefix);
                    if !self
                        .visible_targets()
                        .any(|t| t.hint.starts_with(&candidate))
                    {
                        return Outcome::None;
                    }
                    self.prefix = candidate;
                    match self.select() {
                        Outcome::None => Outcome::Changed,
                        result => result,
                    }
                } else {
                    if self.width <= 0 || self.height <= 0 {
                        return Outcome::None;
                    }
                    if let Some((row, col)) = self.grid_cell {
                        if row >= 26 || col >= 26 {
                            return Outcome::None;
                        }
                        let Some(index) = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".find(c) else {
                            return Outcome::None;
                        };
                        Outcome::Click {
                            x: ((col * 6 + index % 6) as f64 + 0.5) * f64::from(self.width) / 156.0,
                            y: ((row * 6 + index / 6) as f64 + 0.5) * f64::from(self.height)
                                / 156.0,
                        }
                    } else {
                        if !c.is_ascii_uppercase() || self.prefix.len() >= 2 {
                            return Outcome::None;
                        }
                        self.prefix.push(c);
                        if self.prefix.len() == 2 {
                            let bytes = self.prefix.as_bytes();
                            self.grid_cell =
                                Some(((bytes[0] - b'A') as usize, (bytes[1] - b'A') as usize));
                            self.prefix.clear();
                        }
                        Outcome::Changed
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GestureKey {
    ShiftLeft,
    ShiftRight,
    ControlLeft,
    ControlRight,
}

impl GestureKey {
    fn index(self) -> usize {
        self as usize
    }
}

/// Recognizes two short, non-overlapping taps of the exact same modifier key.
/// Global shortcut events do not reveal arbitrary ordinary chord keys. The
/// caller must call `reset` on observed chords, focus loss or event-stream loss;
/// this tracker alone cannot guarantee that a modifier was used without a chord.
pub struct TapTracker {
    interval: Duration,
    max_hold: Duration,
    pressed: [Option<Duration>; 4],
    invalid: [bool; 4],
    tapped: [Option<Duration>; 4],
    last_event: Option<Duration>,
}

impl Default for TapTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl TapTracker {
    pub fn new() -> Self {
        Self::with_thresholds(Duration::from_millis(300), Duration::from_millis(250))
    }

    /// Interval is measured between tap releases; hold is press to release.
    pub fn with_thresholds(interval: Duration, max_hold: Duration) -> Self {
        Self {
            interval,
            max_hold,
            pressed: [None; 4],
            invalid: [false; 4],
            tapped: [None; 4],
            last_event: None,
        }
    }

    pub fn reset(&mut self) {
        self.pressed = [None; 4];
        self.invalid = [false; 4];
        self.tapped = [None; 4];
        self.last_event = None;
    }

    fn timestamp(&mut self, now: Duration) -> bool {
        if self.last_event.is_some_and(|last| now < last) {
            self.reset();
            self.last_event = Some(now);
            return false;
        }
        self.last_event = Some(now);
        true
    }

    pub fn press(&mut self, key: GestureKey, now: Duration) {
        if !self.timestamp(now) {
            return;
        }
        let i = key.index();
        if self.pressed.iter().any(Option::is_some) {
            self.tapped = [None; 4];
            for j in 0..4 {
                self.invalid[j] |= self.pressed[j].is_some();
            }
            self.invalid[i] = true;
        }
        self.pressed[i].get_or_insert(now);
    }

    pub fn release(&mut self, key: GestureKey, now: Duration) -> Option<Mode> {
        if !self.timestamp(now) {
            return None;
        }
        let i = key.index();
        let Some(start) = self.pressed[i].take() else {
            self.tapped[i] = None;
            return None;
        };
        if std::mem::take(&mut self.invalid[i]) || now - start > self.max_hold {
            self.tapped[i] = None;
            return None;
        }
        if self.tapped[i]
            .take()
            .is_some_and(|last| now - last <= self.interval)
        {
            Some(match key {
                GestureKey::ShiftLeft | GestureKey::ShiftRight => Mode::Hints,
                GestureKey::ControlLeft | GestureKey::ControlRight => Mode::Grid,
            })
        } else {
            self.tapped[i] = Some(now);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(x: i32) -> Target {
        Target::new(x, 10, 10, 10, 1.0, "text", "text")
    }
    fn type_text(s: &mut Session, text: &str) -> Outcome {
        text.chars()
            .fold(Outcome::None, |_, c| s.input(Input::Character(c)))
    }

    #[test]
    fn immutable_labels_editing_and_exact_submit() {
        let mut s = Session::hints(100, 100, (0..15).map(|_| target(0)).collect());
        let labels: Vec<_> = s.targets.iter().map(|t| t.hint.clone()).collect();
        assert_eq!(type_text(&mut s, "s"), Outcome::Changed);
        assert_eq!(s.visible_targets().count(), 14);
        assert_eq!(type_text(&mut s, "!z"), Outcome::None);
        assert_eq!(s.prefix, "S");
        assert_eq!(s.input(Input::Backspace), Outcome::Changed);
        assert_eq!(s.visible_targets().count(), 15);
        assert_eq!(
            s.targets.iter().map(|t| t.hint.clone()).collect::<Vec<_>>(),
            labels
        );
        assert_eq!(type_text(&mut s, "ss"), Outcome::Click { x: 5.0, y: 15.0 });
        assert_eq!(s.input(Input::Escape), Outcome::Cancel);
    }

    #[test]
    fn drill_is_immediate_and_parent_restores() {
        let mut child = target(20);
        child.children = vec![serde_json::to_value(target(40)).unwrap()];
        let mut root = target(0);
        root.children = vec![serde_json::to_value(child).unwrap()];
        let mut s = Session::hints(100, 100, vec![root]);
        s.input(Input::Tab);
        assert_eq!(type_text(&mut s, "s"), Outcome::Changed);
        assert_eq!(s.targets[0].x, 20);
        assert_eq!(type_text(&mut s, "s"), Outcome::Changed);
        assert_eq!(s.targets[0].x, 40);
        assert_eq!(s.input(Input::Backspace), Outcome::Changed);
        assert_eq!(s.targets[0].x, 20);
        s.input(Input::Parent);
        assert_eq!(s.targets[0].x, 0);
        s.input(Input::Tab);
        assert_eq!(type_text(&mut s, "s"), Outcome::Click { x: 5.0, y: 15.0 });
        assert_eq!(s.input(Input::Enter), Outcome::Click { x: 5.0, y: 15.0 });
    }

    #[test]
    fn invalid_targets_and_children_are_safe() {
        let mut bad = target(i32::MAX);
        bad.w = i32::MAX;
        let mut root = target(0);
        root.children = vec![serde_json::json!({"x": "bad"})];
        let mut s = Session::hints(100, 100, vec![bad, target(-1), root]);
        assert_eq!(s.targets.len(), 1);
        s.input(Input::Tab);
        assert_eq!(type_text(&mut s, "s"), Outcome::Changed);
        assert_eq!(s.targets[0].x, 0);
        assert_eq!(s.input(Input::Enter), Outcome::None);
        assert_eq!(
            Session::hints(0, 0, vec![target(0)])
                .visible_targets()
                .count(),
            0
        );
    }

    #[test]
    fn grid_row_column_subcells_and_navigation() {
        let mut s = Session::grid(312, 312);
        assert_eq!(type_text(&mut s, "a1"), Outcome::None);
        assert_eq!(s.prefix, "A");
        s.input(Input::Backspace);
        type_text(&mut s, "ab");
        assert_eq!(s.grid_cell, Some((0, 1)));
        assert_eq!(type_text(&mut s, "a"), Outcome::Click { x: 13.0, y: 1.0 });
        assert_eq!(type_text(&mut s, "9"), Outcome::Click { x: 23.0, y: 11.0 });
        s.input(Input::Backspace);
        assert_eq!(s.grid_cell, None);
        type_text(&mut s, "zz");
        assert_eq!(
            type_text(&mut s, "9"),
            Outcome::Click { x: 311.0, y: 311.0 }
        );
        s.input(Input::Parent);
        assert_eq!(s.grid_cell, None);
        assert_eq!(type_text(&mut Session::grid(-1, 1), "aa"), Outcome::None);
    }

    fn tap(t: &mut TapTracker, key: GestureKey, start: u64, end: u64) -> Option<Mode> {
        t.press(key, Duration::from_millis(start));
        t.release(key, Duration::from_millis(end))
    }

    #[test]
    fn exact_keys_thresholds_and_modes() {
        for (key, mode) in [
            (GestureKey::ShiftLeft, Mode::Hints),
            (GestureKey::ShiftRight, Mode::Hints),
            (GestureKey::ControlLeft, Mode::Grid),
            (GestureKey::ControlRight, Mode::Grid),
        ] {
            let mut t = TapTracker::new();
            assert_eq!(tap(&mut t, key, 0, 10), None);
            assert_eq!(tap(&mut t, key, 60, 310), Some(mode));
            assert_eq!(tap(&mut t, key, 320, 330), None);
        }
        let mut t = TapTracker::default();
        assert_eq!(tap(&mut t, GestureKey::ShiftLeft, 0, 10), None);
        assert_eq!(tap(&mut t, GestureKey::ShiftRight, 20, 30), None);
        assert_eq!(tap(&mut t, GestureKey::ShiftLeft, 400, 410), None);
    }

    #[test]
    fn rejects_repeat_hold_overlap_unmatched_and_reordered_events() {
        let key = GestureKey::ShiftLeft;
        let mut t = TapTracker::new();
        tap(&mut t, key, 0, 10);
        t.press(key, Duration::from_millis(20));
        t.press(key, Duration::from_millis(21));
        assert_eq!(t.release(key, Duration::from_millis(30)), None);
        assert_eq!(tap(&mut t, key, 40, 50), None);
        assert_eq!(tap(&mut t, key, 60, 311), None);
        t.reset();
        t.press(key, Duration::ZERO);
        t.press(GestureKey::ControlLeft, Duration::from_millis(1));
        assert_eq!(t.release(key, Duration::from_millis(2)), None);
        assert_eq!(
            t.release(GestureKey::ControlLeft, Duration::from_millis(3)),
            None
        );
        assert_eq!(tap(&mut t, key, 4, 5), None);
        assert_eq!(t.release(key, Duration::from_millis(6)), None);
        assert_eq!(tap(&mut t, key, 7, 8), None);
        t.press(key, Duration::from_millis(100));
        assert_eq!(t.release(key, Duration::from_millis(90)), None);
        assert_eq!(tap(&mut t, key, 101, 102), None);
        t.reset();
        assert_eq!(tap(&mut t, key, 103, 104), None);
    }
}
