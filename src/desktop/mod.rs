//! Native desktop integration; detector behavior remains independent of Wayland.
pub mod draw;
pub mod geometry;
pub mod hyprland;
pub mod session;
pub mod snapshot;
pub mod wayland;

use std::time::Duration;

pub struct Config {
    pub output: Option<String>,
    pub options: crate::Options,
    pub no_click: bool,
    pub flat: bool,
    pub color_by_type: bool,
    pub timeout: Duration,
}
