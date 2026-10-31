//! Logical-pixel overlays. OpenCV draws straight-alpha colors; the export step
//! premultiplies exactly once for little-endian wl_shm ARGB8888 (B, G, R, A).
use super::session::{Mode, Session};
use crate::Target;
use opencv::{core::*, imgproc};

const YELLOW: Scalar = Scalar::new(65., 222., 255., 255.);
const DARK: Scalar = Scalar::new(18., 18., 18., 255.);
const OUTLINE: Scalar = Scalar::new(65., 222., 255., 110.);

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

fn text(m: &mut Mat, value: &str, x: i32, y: i32, badge: bool) -> opencv::Result<()> {
    // Hershey fonts are ASCII-only. Bound work and replace unsupported glyphs
    // rather than passing arbitrary UTF-8 bytes to OpenCV's text renderer.
    let value: String = value
        .chars()
        .take(120)
        .map(|c| {
            if c.is_ascii_graphic() || c == ' ' {
                c
            } else {
                '?'
            }
        })
        .collect();
    let mut baseline = 0;
    let size =
        imgproc::get_text_size(&value, imgproc::FONT_HERSHEY_SIMPLEX, 0.4, 1, &mut baseline)?;
    let w = size.width + 8;
    let h = size.height + baseline + 6;
    let x = x.clamp(0, (m.cols() - w).max(0));
    let y = y.clamp(0, (m.rows() - h).max(0));
    rect(m, [x, y, w, h], if badge { YELLOW } else { DARK }, true)?;
    imgproc::put_text(
        m,
        &value,
        Point::new(x + 4, y + size.height + 3),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.4,
        if badge { DARK } else { YELLOW },
        1,
        imgproc::LINE_8,
        false,
    )
}

fn hints(
    m: &mut Mat,
    targets: &[Target],
    prefix: &str,
    width: i32,
    height: i32,
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
        rect(m, [x, y, w, h], OUTLINE, false)?;
        text(m, &t.hint, x, y, true)?;
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
            if label.starts_with(prefix) && w >= 12 && h >= 16 {
                text(m, &label, x + 2, y + 2, true)?;
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
                    rect(&mut m, [x, y, w, h], YELLOW, false)?;
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
        text(
            &mut m,
            &format!(
                "{mode} [{}]  Escape cancel / Tab drill {} / Backspace",
                session.prefix,
                if session.drill { "on" } else { "off" }
            ),
            8,
            8,
            false,
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
        hints(&mut m, &[t], "A", 320, 200).unwrap();
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
        text(&mut m, "é\n界", i32::MAX, i32::MIN, true).unwrap();
        let mut t = Target::new(i32::MAX, i32::MIN, i32::MAX, i32::MAX, 1., "icon", "test");
        t.hint = "界".into();
        hints(&mut m, &[t], "", 1, 1).unwrap();
        let bytes = export(&m).unwrap();
        assert!(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[..3].iter().all(|&v| v <= p[3]))
        );
        assert!(bytes.as_chunks::<4>().0.iter().any(|p| p[3] == 110));
    }
}
