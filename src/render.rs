//! OpenCV overlays matching the reference, including inclusive rectangle endpoints.
use crate::*;

fn color(kind: &str) -> Scalar {
    let (b, g, r) = match kind {
        "frame" => (80, 220, 100),
        "thumbnail" => (50, 180, 255),
        "desktop_icon" => (255, 210, 60),
        "inline_link" => (255, 140, 255),
        "text_link" => (60, 205, 255),
        "icon" => (255, 225, 70),
        "text_block" => (210, 185, 130),
        "line" => (120, 230, 190),
        "word" => (90, 245, 255),
        _ => (80, 220, 255),
    };
    Scalar::new(b as f64, g as f64, r as f64, 0.)
}

pub fn overlay(bgr: &Mat, targets: &[Target]) -> opencv::Result<Mat> {
    let mut canvas = bgr.try_clone()?;
    let mut fill = bgr.try_clone()?;
    for t in targets {
        rect(
            &mut fill,
            t.x,
            t.y,
            t.w,
            t.h,
            color(&t.kind),
            -1,
            imgproc::LINE_8,
        )?
    }
    opencv::core::add_weighted(&fill, 0.14, bgr, 0.86, 0., &mut canvas, -1)?;
    for t in targets {
        let near_x: Vec<i32> = targets
            .iter()
            .filter(|v| (v.x - t.x).abs() <= 3)
            .map(|v| v.x)
            .collect();
        let near_y: Vec<i32> = targets
            .iter()
            .filter(|v| (v.y - t.y).abs() <= 3)
            .map(|v| v.y)
            .collect();
        let x = if near_x.len() >= 2 {
            median(&near_x).round_ties_even() as i32
        } else {
            t.x
        };
        let y = if near_y.len() >= 2 {
            median(&near_y).round_ties_even() as i32
        } else {
            t.y
        };
        let thick =
            if matches!(t.kind.as_str(), "thumbnail" | "desktop_icon") || t.level == "paragraph" {
                2
            } else {
                1
            };
        rect(
            &mut canvas,
            t.x,
            t.y,
            t.w,
            t.h,
            color(&t.kind),
            thick,
            imgproc::LINE_AA,
        )?;
        let label = if t.hint.is_empty() { "??" } else { &t.hint };
        let mut baseline = 0;
        let size =
            imgproc::get_text_size(label, imgproc::FONT_HERSHEY_SIMPLEX, 0.36, 1, &mut baseline)?;
        let (bw, bh) = (size.width + 6, size.height + 5);
        let (bx, by) = if t.kind == "text_link" && t.w <= 54 && t.h <= 32 && x >= bw + 6 {
            ((x - bw - 3).max(0), (y + (t.h - bh).div_euclid(2)).max(0))
        } else if t.h <= 18 && y >= bh + 2 {
            ((x - 2).max(0), (y - bh).max(0))
        } else {
            ((x - 2).max(0), (y - 2).max(0))
        };
        let right = (bgr.cols() - 1).min(bx + bw);
        let bottom = (bgr.rows() - 1).min(by + bh);
        rect(
            &mut canvas,
            bx - 1,
            by - 1,
            right - bx + 2,
            bottom - by + 2,
            Scalar::all(15.),
            -1,
            imgproc::LINE_8,
        )?;
        let bg = if t.kind == "text_block" {
            Scalar::new(180., 225., 245., 0.)
        } else {
            Scalar::new(55., 215., 255., 0.)
        };
        rect(
            &mut canvas,
            bx,
            by,
            right - bx,
            bottom - by,
            bg,
            -1,
            imgproc::LINE_8,
        )?;
        imgproc::put_text(
            &mut canvas,
            label,
            Point::new(bx + 3, bottom - 3),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.36,
            Scalar::all(10.),
            1,
            imgproc::LINE_AA,
            false,
        )?;
    }
    Ok(canvas)
}

pub fn drilldown_overlay(bgr: &Mat, parent: &Target, to_words: bool) -> opencv::Result<Mat> {
    let mut dimmed = bgr.try_clone()?;
    // The reference multiplies float32 pixels and truncates, unlike OpenCV's
    // saturating conversions, which round. Preserve that distinction here.
    for pixel in dimmed.data_bytes_mut()? {
        *pixel = (*pixel as f32 * 0.35) as u8;
    }
    let x = (parent.x - 8).max(0);
    let y = (parent.y - 8).max(0);
    let right = (parent.right() + 8).min(bgr.cols());
    let bottom = (parent.bottom() + 8).min(bgr.rows());
    if right > x && bottom > y {
        let roi = Rect::new(x, y, right - x, bottom - y);
        Mat::roi(bgr, roi)?.copy_to(&mut Mat::roi_mut(&mut dimmed, roi)?)?;
    }
    rect(
        &mut dimmed,
        x,
        y,
        right - x,
        bottom - y,
        Scalar::new(80., 230., 255., 0.),
        2,
        imgproc::LINE_AA,
    )?;
    let children = drill_down(parent, to_words)
        .map_err(|error| opencv::Error::new(opencv::core::StsBadArg, error.to_string()))?;
    overlay(&dimmed, &children)
}

/// Draw every detected node, including nested drill-down children, using
/// depth-based colors and unique path labels. This is intended for visual
/// debugging and vision-model inspection, not for the interactive overlay.
pub fn hierarchy_overlay(bgr: &Mat, targets: &[Target]) -> Result<Mat, String> {
    fn color(depth: usize) -> Scalar {
        const COLORS: [(u8, u8, u8); 6] = [
            (0, 0, 255),   // depth 0: red
            (0, 200, 0),   // depth 1: green
            (255, 0, 0),   // depth 2: blue
            (0, 255, 255), // depth 3: yellow
            (255, 0, 255), // depth 4: magenta
            (255, 255, 0), // depth 5: cyan
        ];
        let (b, g, r) = COLORS[depth % COLORS.len()];
        Scalar::new(b as f64, g as f64, r as f64, 0.)
    }

    fn draw(
        canvas: &mut Mat,
        node: &Target,
        path: &mut Vec<String>,
        depth: usize,
    ) -> Result<(), String> {
        let label_path = path.join("/");
        let label = format!("D{depth} {label_path} {}", node.kind);
        let label: String = label.chars().take(96).collect();
        let mut baseline = 0;
        let size = imgproc::get_text_size(
            &label,
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.34,
            1,
            &mut baseline,
        )
        .map_err(|e| e.to_string())?;
        let (tw, th) = (size.width, size.height);
        let bx = node.x.max(0);
        let by = (node.y - th - 8).max(0);
        let right = (canvas.cols() - 1).min(bx + tw + 8);
        let bottom = (canvas.rows() - 1).min(by + th + 6);
        if right > bx && bottom > by {
            rect(
                canvas,
                bx - 1,
                by - 1,
                right - bx + 2,
                bottom - by + 2,
                Scalar::all(10.),
                -1,
                imgproc::LINE_8,
            )
            .map_err(|e| e.to_string())?;
            rect(
                canvas,
                bx,
                by,
                right - bx,
                bottom - by,
                color(depth),
                -1,
                imgproc::LINE_8,
            )
            .map_err(|e| e.to_string())?;
            let text_color = if depth == 0 {
                Scalar::all(255.)
            } else {
                Scalar::all(10.)
            };
            imgproc::put_text(
                canvas,
                &label,
                Point::new(bx + 3, bottom - 3),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.34,
                text_color,
                1,
                imgproc::LINE_AA,
                false,
            )
            .map_err(|e| e.to_string())?;
        }
        let thickness = match depth {
            0 => 3,
            1 => 2,
            _ => 1,
        };
        rect(
            canvas,
            node.x,
            node.y,
            node.w,
            node.h,
            color(depth),
            thickness,
            imgproc::LINE_AA,
        )
        .map_err(|e| e.to_string())?;

        for (index, child) in node.children.iter().enumerate() {
            let child: Target = serde_json::from_value(child.clone())
                .map_err(|e| format!("invalid child target: {e}"))?;
            path.push(if child.hint.is_empty() {
                format!("#{index}")
            } else {
                child.hint.clone()
            });
            draw(canvas, &child, path, depth + 1)?;
            path.pop();
        }
        Ok(())
    }

    let mut canvas = bgr.try_clone().map_err(|e| e.to_string())?;
    for target in targets {
        let mut path = vec![if target.hint.is_empty() {
            "?".to_string()
        } else {
            target.hint.clone()
        }];
        draw(&mut canvas, target, &mut path, 0)?;
    }
    Ok(canvas)
}
