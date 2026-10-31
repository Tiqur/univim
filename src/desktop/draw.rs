//! Logical-pixel overlays. OpenCV draws straight-alpha colors; the export step
//! premultiplies exactly once for little-endian wl_shm ARGB8888 (B, G, R, A).
use super::session::{Mode, Session};
use crate::Target;
use opencv::{core::*, imgproc};

// OpenCV colors are BGR(A). The target outline is muted so the solid neon
// labels remain the highest-contrast element on the overlay.
const NEON: Scalar = Scalar::new(20., 255., 40., 255.);
const BLACK: Scalar = Scalar::new(0., 0., 0., 255.);
const MUTED: Scalar = Scalar::new(48., 48., 48., 255.);
const GREEN: Scalar = Scalar::new(70., 220., 70., 255.);
const GREEN_FILL: Scalar = Scalar::new(70., 220., 70., 48.);
const DARK: Scalar = Scalar::new(18., 18., 18., 255.);
const OUTLINE: Scalar = Scalar::new(70., 220., 70., 220.);

const LABEL_FONT: i32 = imgproc::FONT_HERSHEY_SIMPLEX;
const LABEL_BOLD_FONT: i32 = imgproc::FONT_HERSHEY_DUPLEX;
const LABEL_SCALE: f64 = 0.48;
const LABEL_THICKNESS: i32 = 1;
const LABEL_PAD_X: i32 = 3;
const LABEL_PAD_Y: i32 = 2;

const STATUS_SCALE: f64 = 0.45;
const STATUS_THICKNESS: i32 = 1;

/// Bright, high-luminance colors so the black tag text stays readable.
fn type_color(kind: &str) -> Scalar {
    let (r, g, b) = match kind {
        "frame" => (198, 255, 64),        // lime
        "thumbnail" => (255, 72, 224),    // magenta
        "desktop_icon" => (255, 194, 64), // amber
        "inline_link" => (64, 232, 255),  // cyan
        "text_link" => (92, 176, 255),    // sky
        "icon" => (255, 136, 64),         // orange
        "text_block" => (255, 104, 176),  // rose
        "line" => (255, 236, 84),         // lemon
        "word" => (202, 128, 255),        // violet
        "text" => (255, 72, 224),         // magenta fallback
        _ => (20, 255, 40),               // neon green
    };
    Scalar::new(b as f64, g as f64, r as f64, 255.)
}

fn tinted(color: Scalar, alpha: f64) -> Scalar {
    Scalar::new(color[0], color[1], color[2], alpha)
}

fn canvas(width: u32, height: u32) -> Result<Mat, String> {
    // Bound both dimensions and total storage before handing sizes to OpenCV.
    if width == 0 || height == 0 || width > 16384 || height > 16384 {
        return Err("logical dimensions must be between 1 and 16384".into());
    }
    if u64::from(width) * u64::from(height) > 16_777_216 {
        return Err("overlay exceeds the 64 MiB buffer limit".into());
    }
    Mat::new_rows_cols_with_default(height as i32, width as i32, CV_8UC4, Scalar::all(0.))
        .map_err(|e| e.to_string())
}

// Clip before constructing Rect: target coordinates may be negative or extreme.
fn rect(m: &mut Mat, bounds: [i32; 4], color: Scalar, fill: bool) -> opencv::Result<()> {
    let [x, y, w, h] = bounds;
    if w <= 0 || h <= 0 {
        return Ok(());
    }
    let left = x.clamp(0, m.cols());
    let top = y.clamp(0, m.rows());
    let right = (i64::from(x) + i64::from(w)).clamp(0, i64::from(m.cols())) as i32;
    let bottom = (i64::from(y) + i64::from(h)).clamp(0, i64::from(m.rows())) as i32;
    if right > left && bottom > top {
        imgproc::rectangle(
            m,
            Rect::new(left, top, right - left, bottom - top),
            color,
            if fill { -1 } else { 1 },
            imgproc::LINE_8,
            0,
        )?;
    }
    Ok(())
}

fn sanitize(value: &str) -> String {
    // Hershey fonts are ASCII-only. Bound work and replace unsupported glyphs
    // rather than passing arbitrary UTF-8 bytes to OpenCV's text renderer.
    value
        .chars()
        .take(120)
        .map(|c| {
            if c.is_ascii_graphic() || c == ' ' {
                c
            } else {
                '?'
            }
        })
        .collect()
}

fn measure(value: &str) -> opencv::Result<(i32, i32, i32)> {
    let mut baseline = 0;
    let size = imgproc::get_text_size(
        &sanitize(value),
        LABEL_FONT,
        LABEL_SCALE,
        LABEL_THICKNESS,
        &mut baseline,
    )?;
    Ok((size.width, size.height, baseline))
}

fn label_size(value: &str) -> opencv::Result<(i32, i32)> {
    let (width, height, baseline) = measure(value)?;
    Ok((width + LABEL_PAD_X * 2, height + baseline + LABEL_PAD_Y * 2))
}

fn tag(
    m: &mut Mat,
    value: &str,
    x: i32,
    y: i32,
    highlight: usize,
    color: Scalar,
) -> opencv::Result<()> {
    let value = sanitize(value);
    let (width, height, baseline) = measure(&value)?;
    let w = width + LABEL_PAD_X * 2;
    let h = height + baseline + LABEL_PAD_Y * 2;
    let x = x.clamp(0, (m.cols() - w).max(0));
    let y = y.clamp(0, (m.rows() - h).max(0));
    rect(m, [x, y, w, h], color, true)?;
    let origin = Point::new(x + LABEL_PAD_X, y + height + LABEL_PAD_Y);
    let highlight = highlight.min(value.chars().count());
    let color = if highlight == 0 { BLACK } else { MUTED };
    imgproc::put_text(
        m,
        &value,
        origin,
        LABEL_FONT,
        LABEL_SCALE,
        color,
        LABEL_THICKNESS,
        imgproc::LINE_AA,
        false,
    )?;
    // The prefix is laid out at the same origin as the full label. Duplex is a
    // naturally heavier face, so typed characters stand out without the thick,
    // hand-drawn look of an extra stroke.
    if highlight > 0 {
        let prefix: String = value.chars().take(highlight).collect();
        imgproc::put_text(
            m,
            &prefix,
            origin,
            LABEL_BOLD_FONT,
            LABEL_SCALE,
            BLACK,
            LABEL_THICKNESS,
            imgproc::LINE_AA,
            false,
        )?;
    }
    Ok(())
}

fn status(m: &mut Mat, value: &str, x: i32, y: i32) -> opencv::Result<()> {
    let value = sanitize(value);
    let mut baseline = 0;
    let size = imgproc::get_text_size(
        &value,
        imgproc::FONT_HERSHEY_SIMPLEX,
        STATUS_SCALE,
        STATUS_THICKNESS,
        &mut baseline,
    )?;
    let w = size.width + 8;
    let h = size.height + baseline + 6;
    let x = x.clamp(0, (m.cols() - w).max(0));
    let y = y.clamp(0, (m.rows() - h).max(0));
    rect(m, [x, y, w, h], DARK, true)?;
    imgproc::put_text(
        m,
        &value,
        Point::new(x + 4, y + size.height + 3),
        imgproc::FONT_HERSHEY_SIMPLEX,
        STATUS_SCALE,
        NEON,
        STATUS_THICKNESS,
        imgproc::LINE_AA,
        false,
    )
}

/// Keep labels attached to their targets, but move the tag outside when it
/// would obscure most of a small target. Prefer outside above, then below,
/// then to the right and left before falling back to the top-left corner.
fn place_label(target: [i32; 4], label: (i32, i32), cols: i32, rows: i32) -> (i32, i32) {
    let [x, y, w, h] = target;
    let (label_w, label_h) = label;
    let x = x.clamp(0, (cols - label_w).max(0));
    // Require a comfortable margin so labels do not sit on small targets.
    if w >= label_w + 24 && h >= label_h + 24 {
        return (x, y);
    }
    let above = y - label_h - 2;
    if above >= 0 {
        return (x, above);
    }
    let below = y + h + 2;
    if below + label_h <= rows {
        return (x, below);
    }
    let right = x + w + 2;
    if right + label_w <= cols {
        return (right, y);
    }
    let left = x - label_w - 2;
    if left >= 0 {
        return (left, y);
    }
    (x, y)
}

fn hints(
    m: &mut Mat,
    targets: &[Target],
    prefix: &str,
    width: i32,
    height: i32,
    color_by_type: bool,
) -> opencv::Result<()> {
    let sx = f64::from(m.cols()) / f64::from(width);
    let sy = f64::from(m.rows()) / f64::from(height);
    for t in targets.iter().filter(|t| t.hint.starts_with(prefix)) {
        if t.w <= 0 || t.h <= 0 {
            continue;
        }
        let left = (f64::from(t.x) * sx).floor();
        let top = (f64::from(t.y) * sy).floor();
        let right = ((f64::from(t.x) + f64::from(t.w)) * sx).ceil();
        let bottom = ((f64::from(t.y) + f64::from(t.h)) * sy).ceil();
        if right <= 0. || bottom <= 0. || left >= f64::from(m.cols()) || top >= f64::from(m.rows())
        {
            continue;
        }
        let x = left.max(0.) as i32;
        let y = top.max(0.) as i32;
        let w = right.min(f64::from(m.cols())) as i32 - x;
        let h = bottom.min(f64::from(m.rows())) as i32 - y;
        let (tag_color, outline, fill) = if color_by_type {
            let color = type_color(&t.kind);
            (color, tinted(color, 220.), tinted(color, 48.))
        } else {
            (NEON, OUTLINE, GREEN_FILL)
        };
        rect(m, [x, y, w, h], fill, true)?;
        rect(m, [x, y, w, h], outline, false)?;
        // Only typed characters are emphasized. With no prefix the whole tag
        // is black, so the first letter is not special until it is actually used.
        let highlight = prefix.chars().count();
        let (lx, ly) = place_label([x, y, w, h], label_size(&t.hint)?, m.cols(), m.rows());
        tag(m, &t.hint, lx, ly, highlight, tag_color)?;
    }
    Ok(())
}

fn grid(m: &mut Mat, bounds: Rect, count: i32, prefix: &str) -> opencv::Result<()> {
    for row in 0..count {
        for col in 0..count {
            let label = if count == 26 {
                format!(
                    "{}{}",
                    (b'A' + row as u8) as char,
                    (b'A' + col as u8) as char
                )
            } else {
                let index = (row * 6 + col) as u8;
                char::from(if index < 26 {
                    b'A' + index
                } else {
                    b'0' + index - 26
                })
                .to_string()
            };
            let x = bounds.x + col * bounds.width / count;
            let y = bounds.y + row * bounds.height / count;
            let w = (col + 1) * bounds.width / count - col * bounds.width / count;
            let h = (row + 1) * bounds.height / count - row * bounds.height / count;
            rect(m, [x, y, w, h], OUTLINE, false)?;
            if label.starts_with(prefix) {
                let (label_w, label_h) = label_size(&label)?;
                if w >= label_w + 4 && h >= label_h + 4 {
                    tag(m, &label, x + 2, y + 2, prefix.chars().count(), NEON)?;
                }
            }
        }
    }
    Ok(())
}

fn export(m: &Mat) -> opencv::Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(m.total() * 4);
    for y in 0..m.rows() {
        for pixel in m.at_row::<Vec4b>(y)? {
            let alpha = u16::from(pixel[3]);
            for channel in &pixel.0[..3] {
                bytes.push(((u16::from(*channel) * alpha + 127) / 255) as u8);
            }
            bytes.push(pixel[3]);
        }
    }
    Ok(bytes)
}

pub fn render(
    session: &Session,
    frame: Option<&Mat>,
    logical_width: u32,
    logical_height: u32,
    color_by_type: bool,
) -> Result<Vec<u8>, String> {
    if session.width <= 0 || session.height <= 0 {
        return Err("screenshot dimensions must be positive".into());
    }
    let mut m = canvas(logical_width, logical_height)?;
    let mut draw = || -> opencv::Result<Vec<u8>> {
        match session.mode {
            Mode::Hints => hints(
                &mut m,
                &session.targets,
                &session.prefix,
                session.width,
                session.height,
                color_by_type,
            )?,
            Mode::Grid => {
                let bounds = Rect::new(0, 0, m.cols(), m.rows());
                if let Some((row, col)) = session.grid_cell {
                    if row >= 26 || col >= 26 {
                        return Err(opencv::Error::new(
                            StsBadArg,
                            "grid cell is outside 26x26 grid",
                        ));
                    }
                    rect(
                        &mut m,
                        [0, 0, bounds.width, bounds.height],
                        Scalar::new(0., 0., 0., 75.),
                        true,
                    )?;
                    let x = col as i32 * bounds.width / 26;
                    let y = row as i32 * bounds.height / 26;
                    let w = (col as i32 + 1) * bounds.width / 26 - x;
                    let h = (row as i32 + 1) * bounds.height / 26 - y;
                    rect(&mut m, [x, y, w, h], GREEN_FILL, true)?;
                    rect(&mut m, [x, y, w, h], GREEN, false)?;
                    // Integer partitions match the coarse grid, including remainder pixels.
                    let sx =
                        col as i32 * (session.width / 26) + col as i32 * (session.width % 26) / 26;
                    let sy = row as i32 * (session.height / 26)
                        + row as i32 * (session.height % 26) / 26;
                    let ex = ((col + 1) as i64 * i64::from(session.width) / 26) as i32;
                    let ey = ((row + 1) as i64 * i64::from(session.height) / 26) as i32;
                    if ex > sx && ey > sy {
                        let scale = (f64::from(bounds.width) * 0.7 / f64::from(ex - sx))
                            .min(f64::from(bounds.height) * 0.7 / f64::from(ey - sy));
                        let pw = (f64::from(ex - sx) * scale).round().max(1.) as i32;
                        let ph = (f64::from(ey - sy) * scale).round().max(1.) as i32;
                        let panel =
                            Rect::new((bounds.width - pw) / 2, (bounds.height - ph) / 2, pw, ph);
                        rect(&mut m, [panel.x, panel.y, pw, ph], DARK, true)?;
                        if let Some(frame) = frame {
                            if frame.cols() != session.width
                                || frame.rows() != session.height
                                || frame.typ() != CV_8UC3
                            {
                                return Err(opencv::Error::new(
                                    StsBadArg,
                                    "preview frame must be screenshot-sized CV_8UC3 BGR",
                                ));
                            }
                            let crop = frame.roi(Rect::new(sx, sy, ex - sx, ey - sy))?;
                            let mut resized = Mat::default();
                            imgproc::resize(
                                &crop,
                                &mut resized,
                                Size::new(pw, ph),
                                0.,
                                0.,
                                imgproc::INTER_LINEAR,
                            )?;
                            let mut bgra = Mat::default();
                            imgproc::cvt_color_def(&resized, &mut bgra, imgproc::COLOR_BGR2BGRA)?;
                            bgra.copy_to(&mut m.roi_mut(panel)?)?;
                        }
                        grid(&mut m, panel, 6, &session.prefix)?;
                    }
                } else {
                    grid(&mut m, bounds, 26, &session.prefix)?;
                }
            }
        }
        let mode = match session.mode {
            Mode::Hints => "Hints",
            Mode::Grid => "Grid",
        };
        status(
            &mut m,
            &format!(
                "{mode} [{}]  Escape cancel / Tab drill {} / Backspace",
                session.prefix,
                if session.drill { "on" } else { "off" }
            ),
            8,
            8,
        )?;
        export(&m)
    };
    draw().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_and_buffer_size() {
        for (w, h) in [(0, 1), (1, 0), (u32::MAX, 2), (16384, 16384)] {
            assert!(canvas(w, h).is_err());
        }
        for (w, h) in [(1, 1), (31, 17), (640, 480)] {
            assert_eq!(
                export(&canvas(w, h).unwrap()).unwrap().len(),
                (w * h * 4) as usize
            );
        }
    }

    #[test]
    fn hints_leave_untouched_pixels_transparent() {
        let mut m = canvas(320, 200).unwrap();
        let mut t = Target::new(40, 60, 30, 20, 1., "icon", "test");
        t.hint = "AA".into();
        hints(&mut m, &[t], "A", 320, 200, false).unwrap();
        let pixels = export(&m).unwrap();
        assert_eq!(&pixels[(150 * 320 + 250) * 4..][..4], &[0; 4]);
        assert!(pixels.iter().any(|&v| v != 0));
    }

    #[test]
    fn clipping_unicode_and_premultiplication() {
        let mut m = canvas(80, 60).unwrap();
        rect(
            &mut m,
            [i32::MIN, i32::MIN, i32::MAX, i32::MAX],
            OUTLINE,
            false,
        )
        .unwrap();
        rect(&mut m, [-5, -5, 30, 30], OUTLINE, true).unwrap();
        tag(&mut m, "é\n界", i32::MAX, i32::MIN, 0, NEON).unwrap();
        let mut t = Target::new(i32::MAX, i32::MIN, i32::MAX, i32::MAX, 1., "icon", "test");
        t.hint = "界".into();
        hints(&mut m, &[t], "", 1, 1, false).unwrap();
        let bytes = export(&m).unwrap();
        assert!(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[..3].iter().all(|&v| v <= p[3]))
        );
        assert!(bytes.as_chunks::<4>().0.iter().any(|p| p[3] == 220));
    }

    #[test]
    fn typed_hint_prefix_is_emphasized() {
        let mut plain = canvas(80, 40).unwrap();
        let mut typed = canvas(80, 40).unwrap();
        tag(&mut plain, "MGF", 2, 2, 0, NEON).unwrap();
        tag(&mut typed, "MGF", 2, 2, 1, NEON).unwrap();
        assert_ne!(export(&plain).unwrap(), export(&typed).unwrap());
    }

    #[test]
    fn labels_are_black_on_solid_neon() {
        let mut m = canvas(60, 30).unwrap();
        tag(&mut m, "M", 2, 2, 0, NEON).unwrap();
        let pixels = export(&m).unwrap();
        assert!(
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .any(|p| p[0] < 32 && p[1] < 32 && p[2] < 32)
        );
        assert!(
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .any(|p| p[3] == 255 && p[1] > p[0] && p[1] > p[2])
        );
    }

    #[test]
    fn small_targets_place_labels_outside() {
        let label = (20, 14);
        assert_eq!(place_label([40, 40, 20, 10], label, 200, 100), (40, 24));
        assert_eq!(place_label([40, 40, 100, 50], label, 200, 100), (40, 40));
        assert_eq!(place_label([40, 0, 20, 10], label, 200, 100), (40, 12));
    }

    #[test]
    fn hint_boxes_use_a_translucent_green_fill() {
        let mut m = canvas(120, 80).unwrap();
        let mut t = Target::new(20, 20, 60, 30, 1., "icon", "test");
        t.hint = "AB".into();
        hints(&mut m, &[t], "", 120, 80, false).unwrap();
        let pixels = export(&m).unwrap();
        let fill = &pixels[((47 * 120) + 70) * 4..][..4];
        assert!(fill[1] > fill[0] && fill[1] > fill[2]);
        assert_eq!(fill[3], GREEN_FILL[3] as u8);
    }

    #[test]
    fn element_type_colors_are_applied_when_requested() {
        let lime = type_color("frame");
        assert!(lime[1] > lime[0] && lime[1] > lime[2]);
        let magenta = type_color("thumbnail");
        assert!(magenta[0] > magenta[1] && magenta[2] > magenta[1]);

        let mut m = canvas(160, 80).unwrap();
        let mut t = Target::new(10, 10, 130, 60, 1., "thumbnail", "control");
        t.hint = "AB".into();
        hints(&mut m, &[t], "", 160, 80, true).unwrap();
        let pixels = export(&m).unwrap();
        assert!(
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .any(|p| { p[3] == 255 && p[0] > 200 && p[2] > 200 && p[1] < 100 })
        );
    }
}
