use super::{
    Appearance, PointerMotion, RuntimeSettings,
    geometry::FrameGeometry,
    session::{Cardinal, DebugView, Mode, Session},
};
use crate::{DebugData, Target};
use opencv::{
    core::{CV_8UC3, Mat, Point, Rect, Scalar, Size},
    imgproc::{self, LINE_AA},
    prelude::*,
};

#[derive(Clone, Debug)]
pub struct RenderedSurface {
    pub width: i32,
    pub height: i32,
    pub argb: Vec<u8>,
}

pub fn render_session_surface(
    base_bgr: &Mat,
    session: &Session,
    geometry: &FrameGeometry,
    debug_data: Option<&DebugData>,
    settings: &RuntimeSettings,
) -> opencv::Result<RenderedSurface> {
    let mut canvas = render_debug_stage_canvas_styled(
        base_bgr,
        session,
        debug_data,
        settings.spotlight_dim,
        Some(&settings.appearance),
    )?;

    draw_focused_window_And_reticule(&mut canvas, session)?;
    draw_snap_neighbor_lines(&mut canvas, session)?;

    if settings.show_status_bar || !session.typed().is_empty() || session.show_debug_hud() {
        draw_status_pill(&mut canvas, session, &settings.appearance)?;
    }

    if session.show_debug_hud() {
        draw_debug_hud(&mut canvas, session, debug_data, &settings.appearance)?;
    }

    if session.show_options_menu() {
        draw_options_modal(&mut canvas, session, settings)?;
    }

    let raw = geometry.to_compositor_buffer(&canvas)?;
    Ok(RenderedSurface {
        width: raw.cols(),
        height: raw.rows(),
        argb: bgr_to_argb_bytes(&raw)?,
    })
}

pub(super) fn render_debug_stage_canvas(
    base_bgr: &Mat,
    session: &Session,
    debug_data: Option<&DebugData>,
    spotlight_dim: bool,
) -> opencv::Result<Mat> {
    render_debug_stage_canvas_styled(base_bgr, session, debug_data, spotlight_dim, None)
}

fn render_debug_stage_canvas_styled(
    base_bgr: &Mat,
    session: &Session,
    debug_data: Option<&DebugData>,
    spotlight_dim: bool,
    appearance: Option<&Appearance>,
) -> opencv::Result<Mat> {
    let Some(dbg) = debug_data else {
        let mut base = base_bgr.try_clone()?;
        if spotlight_dim {
            dim_mat_with_focus(&mut base, session.focused_rect())?;
        }
        return render_standard_overlay(&base, session, appearance);
    };

    match session.debug_view() {
        DebugView::OverlayHud => {
            let mut base = base_bgr.try_clone()?;
            if spotlight_dim {
                dim_mat_with_focus(&mut base, session.focused_rect())?;
            }
            render_standard_overlay(&base, session, appearance)
        }
        DebugView::RawPhysics => {
            let mut vis = base_bgr.try_clone()?;
            dim_mat(&mut vis, 0.72)?;
            for t in session.targets() {
                let color = match t.kind.as_str() {
                    "frame" => Scalar::new(0.0, 220.0, 80.0, 0.0),
                    "thumbnail" => Scalar::new(255.0, 170.0, 0.0, 0.0),
                    "icon" | "desktop_icon" => Scalar::new(255.0, 90.0, 220.0, 0.0),
                    _ => Scalar::new(240.0, 200.0, 40.0, 0.0),
                };
                imgproc::rectangle(
                    &mut vis,
                    Rect::new(t.x, t.y, t.w, t.h),
                    color,
                    1,
                    imgproc::LINE_8,
                    0,
                )?;
            }
            Ok(vis)
        }
        DebugView::Gradients => {
            let g_u8 = gray_slice_to_bgr(
                &dbg.grad_max,
                dbg.work_w,
                dbg.work_h,
                base_bgr.cols(),
                base_bgr.rows(),
                [40, 255, 140],
            )?;
            Ok(g_u8)
        }
        DebugView::BarriersFrames => {
            let b_u8 = gray_slice_to_bgr(
                &dbg.bar_all,
                dbg.work_w,
                dbg.work_h,
                base_bgr.cols(),
                base_bgr.rows(),
                [80, 180, 255],
            )?;
            Ok(b_u8)
        }
        DebugView::StrokesCoarse => {
            let s_u8 = dual_mask_to_bgr(
                &dbg.stroke_mask,
                &dbg.coarse_mask,
                dbg.work_w,
                dbg.work_h,
                base_bgr.cols(),
                base_bgr.rows(),
            )?;
            Ok(s_u8)
        }
        DebugView::Hierarchy => {
            let mut vis = base_bgr.try_clone()?;
            dim_mat(&mut vis, 0.45)?;
            for t in session.targets() {
                imgproc::rectangle(
                    &mut vis,
                    Rect::new(t.x, t.y, t.w, t.h),
                    Scalar::new(0.0, 210.0, 90.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    0,
                )?;
                for child in &t.children {
                    let Ok(line) = serde_json::from_value::<Target>(child.clone()) else {
                        continue;
                    };
                    imgproc::rectangle(
                        &mut vis,
                        Rect::new(line.x, line.y, line.w, line.h),
                        Scalar::new(255.0, 190.0, 0.0, 0.0),
                        1,
                        imgproc::LINE_8,
                        0,
                    )?;
                    for child in &line.children {
                        let Ok(word) = serde_json::from_value::<Target>(child.clone()) else {
                            continue;
                        };
                        imgproc::rectangle(
                            &mut vis,
                            Rect::new(word.x, word.y, word.w, word.h),
                            Scalar::new(180.0, 80.0, 255.0, 0.0),
                            1,
                            imgproc::LINE_8,
                            0,
                        )?;
                    }
                }
            }
            Ok(vis)
        }
    }
}

fn render_standard_overlay(
    base_bgr: &Mat,
    session: &Session,
    appearance: Option<&Appearance>,
) -> opencv::Result<Mat> {
    let filtered = visible_targets(session.targets(), session.typed());
    match session.mode() {
        // Drill-down sessions already replace their targets with the immediate
        // children, so the ordinary labelled overlay is the correct surface.
        Mode::Hints | Mode::Drilldown | Mode::Grid => {
            let style = appearance.map(|a| crate::render::BadgeStyle {
                font_scale: a.hint_font_scale,
                background: Appearance::color(&a.badge_background),
                foreground: Appearance::color(&a.badge_foreground),
                typed_foreground: Appearance::color(&a.typed_foreground),
            });
            crate::render::overlay_with_style(base_bgr, &filtered, session.typed(), style.as_ref())
        }
    }
}

#[allow(non_snake_case)]
fn draw_focused_window_And_reticule(canvas: &mut Mat, session: &Session) -> opencv::Result<()> {
    let w = canvas.cols();
    let h = canvas.rows();

    if let Some([fx, fy, fw, fh]) = session.focused_rect() {
        let rx = fx.clamp(0, (w - 1).max(0));
        let ry = fy.clamp(0, (h - 1).max(0));
        let rw = fw.min(w - rx).max(0);
        let rh = fh.min(h - ry).max(0);
        if rw > 4 && rh > 4 {
            let color = if session.focused_only() {
                Scalar::new(80.0, 230.0, 255.0, 0.0)
            } else {
                Scalar::new(180.0, 140.0, 60.0, 0.0)
            };
            imgproc::rectangle(
                canvas,
                Rect::new(rx, ry, rw, rh),
                color,
                if session.focused_only() { 2 } else { 1 },
                imgproc::LINE_8,
                0,
            )?;
        }
    }

    if let Some(target) = session.snapped_target() {
        let pad = 3;
        let rx = (target.x - pad).max(0);
        let ry = (target.y - pad).max(0);
        let rw = (target.w + pad * 2).min(w - rx).max(1);
        let rh = (target.h + pad * 2).min(h - ry).max(1);
        imgproc::rectangle(
            canvas,
            Rect::new(rx, ry, rw, rh),
            Scalar::new(0.0, 230.0, 255.0, 0.0),
            2,
            LINE_AA,
            0,
        )?;
        let cx = (target.x + target.w / 2).clamp(0, w - 1);
        let cy = (target.y + target.h / 2).clamp(0, h - 1);
        imgproc::circle(
            canvas,
            Point::new(cx, cy),
            4,
            Scalar::new(0.0, 230.0, 255.0, 0.0),
            -1,
            LINE_AA,
            0,
        )?;
    }

    if let Some((ax, ay)) = session.drag_anchor() {
        let cx = ax.clamp(0, w - 1);
        let cy = ay.clamp(0, h - 1);
        imgproc::circle(
            canvas,
            Point::new(cx, cy),
            7,
            Scalar::new(90.0, 90.0, 255.0, 0.0),
            2,
            LINE_AA,
            0,
        )?;
        imgproc::put_text(
            canvas,
            "DRAG START",
            Point::new((cx + 10).min((w - 90).max(0)), (cy - 6).max(14)),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.40,
            Scalar::new(110.0, 110.0, 255.0, 0.0),
            1,
            LINE_AA,
            false,
        )?;
    }

    Ok(())
}

/// Connector lines from the edges of the snapped box to the target each arrow
/// key would select. Each direction gets its own color and a small arrowhead so
/// the four destinations stay distinguishable when lines cross.
fn draw_snap_neighbor_lines(canvas: &mut Mat, session: &Session) -> opencv::Result<()> {
    let Some(current) = session.snapped_target() else {
        return Ok(());
    };
    let (w, h) = (canvas.cols(), canvas.rows());
    if w <= 1 || h <= 1 {
        return Ok(());
    }
    let cur_left = current.x.clamp(0, w - 1);
    let cur_top = current.y.clamp(0, h - 1);
    let cur_right = (current.x + current.w).clamp(0, w - 1);
    let cur_bottom = (current.y + current.h).clamp(0, h - 1);
    let (cx, cy) = ((cur_left + cur_right) / 2, (cur_top + cur_bottom) / 2);

    for (dir, [x, y, tw, th]) in session.snap_neighbors() {
        let left = x.clamp(0, w - 1);
        let top = y.clamp(0, h - 1);
        let right = (x + tw).clamp(0, w - 1);
        let bottom = (y + th).clamp(0, h - 1);
        let (start, end, color) = match dir {
            Cardinal::Up => (
                Point::new(cx, cur_top),
                Point::new((left + right) / 2, bottom),
                Scalar::new(255.0, 220.0, 80.0, 0.0),
            ),
            Cardinal::Down => (
                Point::new(cx, cur_bottom),
                Point::new((left + right) / 2, top),
                Scalar::new(80.0, 170.0, 255.0, 0.0),
            ),
            Cardinal::Left => (
                Point::new(cur_left, cy),
                Point::new(right, (top + bottom) / 2),
                Scalar::new(120.0, 255.0, 140.0, 0.0),
            ),
            Cardinal::Right => (
                Point::new(cur_right, cy),
                Point::new(left, (top + bottom) / 2),
                Scalar::new(255.0, 130.0, 220.0, 0.0),
            ),
        };
        if start.x == end.x && start.y == end.y {
            continue;
        }
        imgproc::line(canvas, start, end, color, 2, LINE_AA, 0)?;
        draw_arrow_head(canvas, start, end, color)?;
    }
    Ok(())
}

fn draw_arrow_head(
    canvas: &mut Mat,
    start: Point,
    end: Point,
    color: Scalar,
) -> opencv::Result<()> {
    let dx = (end.x - start.x) as f64;
    let dy = (end.y - start.y) as f64;
    let length = dx.hypot(dy);
    if length < 14.0 {
        return Ok(());
    }
    let (ux, uy) = (dx / length, dy / length);
    for spread in [0.5_f64, -0.5] {
        let (sin, cos) = spread.sin_cos();
        let back = Point::new(
            (end.x as f64 - 11.0 * (ux * cos - uy * sin)) as i32,
            (end.y as f64 - 11.0 * (ux * sin + uy * cos)) as i32,
        );
        imgproc::line(canvas, end, back, color, 2, LINE_AA, 0)?;
    }
    Ok(())
}

fn dim_mat_with_focus(mat: &mut Mat, focused_rect: Option<[i32; 4]>) -> opencv::Result<()> {
    let w = mat.cols();
    let h = mat.rows();
    if let Some([fx, fy, fw, fh]) = focused_rect {
        let rx = fx.clamp(0, w);
        let ry = fy.clamp(0, h);
        let rw = fw.min(w - rx).max(0);
        let rh = fh.min(h - ry).max(0);
        if rw > 0 && rh > 0 && (rw < w || rh < h) {
            for r in 0..h {
                let in_y = r >= ry && r < ry + rh;
                let row = mat.at_row_mut::<opencv::core::Vec3b>(r)?;
                for (c, px) in row.iter_mut().enumerate() {
                    let inside = in_y && c as i32 >= rx && (c as i32) < rx + rw;
                    if inside {
                        px[0] = ((px[0] as u16 * 238) >> 8) as u8;
                        px[1] = ((px[1] as u16 * 238) >> 8) as u8;
                        px[2] = ((px[2] as u16 * 238) >> 8) as u8;
                    } else {
                        px[0] = ((px[0] as u16 * 178) >> 8) as u8;
                        px[1] = ((px[1] as u16 * 178) >> 8) as u8;
                        px[2] = ((px[2] as u16 * 178) >> 8) as u8;
                    }
                }
            }
            return Ok(());
        }
    }
    dim_mat(mat, 0.84)
}

fn dim_mat(mat: &mut Mat, factor: f64) -> opencv::Result<()> {
    let scale = (factor.clamp(0.1, 1.0) * 256.0) as u16;
    let rows = mat.rows();
    for r in 0..rows {
        let row = mat.at_row_mut::<opencv::core::Vec3b>(r)?;
        for px in row.iter_mut() {
            px[0] = ((px[0] as u16 * scale) >> 8) as u8;
            px[1] = ((px[1] as u16 * scale) >> 8) as u8;
            px[2] = ((px[2] as u16 * scale) >> 8) as u8;
        }
    }
    Ok(())
}

fn gray_slice_to_bgr(
    gray: &[u8],
    work_w: i32,
    work_h: i32,
    out_w: i32,
    out_h: i32,
    tint_bgr: [u8; 3],
) -> opencv::Result<Mat> {
    let mut small = Mat::new_rows_cols_with_default(work_h, work_w, CV_8UC3, Scalar::all(0.0))?;
    for r in 0..work_h {
        let row = small.at_row_mut::<opencv::core::Vec3b>(r)?;
        let base = (r as usize) * (work_w as usize);
        for (c, px) in row[..work_w as usize].iter_mut().enumerate() {
            let v = gray.get(base + c).copied().unwrap_or(0) as u16;
            px[0] = ((v * tint_bgr[0] as u16) >> 8) as u8;
            px[1] = ((v * tint_bgr[1] as u16) >> 8) as u8;
            px[2] = ((v * tint_bgr[2] as u16) >> 8) as u8;
        }
    }
    if work_w == out_w && work_h == out_h {
        return Ok(small);
    }
    let mut out = Mat::default();
    imgproc::resize(
        &small,
        &mut out,
        Size::new(out_w, out_h),
        0.0,
        0.0,
        imgproc::INTER_NEAREST,
    )?;
    Ok(out)
}

fn dual_mask_to_bgr(
    stroke: &[u8],
    coarse: &[u8],
    work_w: i32,
    work_h: i32,
    out_w: i32,
    out_h: i32,
) -> opencv::Result<Mat> {
    let mut small = Mat::new_rows_cols_with_default(work_h, work_w, CV_8UC3, Scalar::all(12.0))?;
    for r in 0..work_h {
        let row = small.at_row_mut::<opencv::core::Vec3b>(r)?;
        let base = (r as usize) * (work_w as usize);
        for (c, px) in row[..work_w as usize].iter_mut().enumerate() {
            let s = stroke.get(base + c).copied().unwrap_or(0);
            let k = coarse.get(base + c).copied().unwrap_or(0);
            if s > 0 {
                *px = opencv::core::Vec3b::from([60, 245, 255]);
            } else if k > 0 {
                *px = opencv::core::Vec3b::from([120, 70, 35]);
            }
        }
    }
    if work_w == out_w && work_h == out_h {
        return Ok(small);
    }
    let mut out = Mat::default();
    imgproc::resize(
        &small,
        &mut out,
        Size::new(out_w, out_h),
        0.0,
        0.0,
        imgproc::INTER_NEAREST,
    )?;
    Ok(out)
}

fn visible_targets(targets: &[Target], typed: &str) -> Vec<Target> {
    if typed.is_empty() {
        return targets.to_vec();
    }
    targets
        .iter()
        .filter(|target| target.hint.starts_with(typed))
        .cloned()
        .collect()
}

fn scaled(value: i32, scale: f64) -> i32 {
    ((value as f64 * scale).round() as i32).max(1)
}

fn draw_status_pill(
    canvas: &mut Mat,
    session: &Session,
    appearance: &Appearance,
) -> opencv::Result<()> {
    let mode_text = match session.mode() {
        Mode::Hints => "HINTS",
        Mode::Drilldown => "DRILL",
        Mode::Grid => "GRID",
    };
    let cache_tag = if session.cached_hit() {
        "CACHED 0ms"
    } else if session.pipeline_ms() > 0.0 {
        "LIVE"
    } else {
        "READY"
    };
    let focus_tag = if session.focused_only() {
        " [WIN-ONLY]"
    } else if session.focused_rect().is_some() {
        " [WIN-PRIO]"
    } else {
        ""
    };
    let drag_tag = if session.drag_anchor().is_some() {
        " [DRAG->DEST]"
    } else if session.drag_mode() {
        " [DRAG:PICK]"
    } else {
        ""
    };
    let loop_tag = if session.auto_loop() { " [LOOP]" } else { "" };
    let scroll_tag = match session.scroll_feedback() {
        Some((_, dy)) if dy < 0 => " [SCROLLED DOWN]",
        Some((_, dy)) if dy > 0 => " [SCROLLED UP]",
        Some((dx, _)) if dx < 0 => " [SCROLLED LEFT]",
        Some((dx, _)) if dx > 0 => " [SCROLLED RIGHT]",
        _ => "",
    };

    let typed_part = if session.typed().is_empty() {
        String::from("-")
    } else {
        session.typed().to_string()
    };

    let status = format!(
        "{mode_text} [{typed_part}] | {} | {cache_tag}{focus_tag}{drag_tag}{loop_tag}{scroll_tag} | Shift:Right Ctrl:Middle ;:Act v:Drag F1:Opts F2:Dbg",
        session.action().label()
    );

    let font = imgproc::FONT_HERSHEY_SIMPLEX;
    let scale = 0.44 * appearance.ui_scale;
    let thickness = 1;
    let mut baseline = 0;
    let size = imgproc::get_text_size(&status, font, scale, thickness, &mut baseline)?;
    let pad_x = scaled(10, appearance.ui_scale);
    let pad_y = scaled(7, appearance.ui_scale);
    let x = scaled(14, appearance.ui_scale);
    let y = scaled(14, appearance.ui_scale);
    let rect = Rect::new(
        x,
        y,
        (size.width + pad_x * 2).min(canvas.cols() - x - 2).max(20),
        (size.height + pad_y * 2).min(canvas.rows() - y - 2).max(16),
    );

    imgproc::rectangle(
        canvas,
        rect,
        Appearance::color(&appearance.ui_background),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    let border_col = if session.cached_hit() {
        Scalar::new(80.0, 235.0, 130.0, 0.0)
    } else {
        Appearance::color(&appearance.ui_accent)
    };
    imgproc::rectangle(canvas, rect, border_col, 1, imgproc::LINE_8, 0)?;
    imgproc::put_text(
        canvas,
        &status,
        Point::new(x + pad_x, y + pad_y + size.height - 1),
        font,
        scale,
        Appearance::color(&appearance.ui_foreground),
        thickness,
        LINE_AA,
        false,
    )?;
    Ok(())
}

fn draw_debug_hud(
    canvas: &mut Mat,
    session: &Session,
    debug_data: Option<&DebugData>,
    appearance: &Appearance,
) -> opencv::Result<()> {
    let scale = appearance.ui_scale;
    let panel_w = scaled(385, scale).min(canvas.cols() - 24).max(220);
    let panel_h = scaled(250, scale).min(canvas.rows() - 24).max(140);
    let x = (canvas.cols() - panel_w - scaled(14, scale)).max(8);
    let y = scaled(14, scale);
    let rect = Rect::new(x, y, panel_w, panel_h);

    imgproc::rectangle(
        canvas,
        rect,
        Appearance::color(&appearance.ui_background),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::rectangle(
        canvas,
        rect,
        Appearance::color(&appearance.ui_accent),
        1,
        imgproc::LINE_8,
        0,
    )?;

    let font = imgproc::FONT_HERSHEY_SIMPLEX;
    let mut cy = y + scaled(22, scale);
    let title = format!("PIPELINE INSPECTOR ({})", session.debug_view().label());
    imgproc::put_text(
        canvas,
        &title,
        Point::new(x + scaled(12, scale), cy),
        font,
        0.46 * scale,
        Appearance::color(&appearance.ui_accent),
        1,
        LINE_AA,
        false,
    )?;
    cy += scaled(18, scale);

    imgproc::put_text(
        canvas,
        "Views 1..6 visualizations | F3/F4 cycle | F2 hide",
        Point::new(x + scaled(12, scale), cy),
        font,
        0.38 * scale,
        Appearance::color(&appearance.ui_foreground),
        1,
        LINE_AA,
        false,
    )?;
    cy += scaled(20, scale);

    if let Some(dbg) = debug_data {
        let p = &dbg.physics;
        let t = &dbg.timings;
        let lines = [
            format!(
                "Physics: L0={:.1}px  noise={}  flat={}  border={}  stroke={}",
                p.l0, p.noise_floor, p.flat_thr, p.border_thr, p.stroke_thr
            ),
            format!(
                "Work Res: {}x{} (scale {:.2}) | Targets: {}",
                dbg.work_w,
                dbg.work_h,
                p.scale,
                session.targets().len()
            ),
            format!(
                "Source: {} | Total: {:.2} ms",
                if session.cached_hit() {
                    "1st-Tap Speculative Cache"
                } else {
                    "Fresh Capture"
                },
                t.total_ms
            ),
        ];
        for line in &lines {
            imgproc::put_text(
                canvas,
                line,
                Point::new(x + scaled(12, scale), cy),
                font,
                0.38 * scale,
                Appearance::color(&appearance.ui_foreground),
                1,
                LINE_AA,
                false,
            )?;
            cy += scaled(17, scale);
        }
        cy += scaled(4, scale);

        let stages = [
            (
                "Signal",
                t.stage1_gradients_ms,
                Scalar::new(255.0, 190.0, 70.0, 0.0),
            ),
            (
                "Frames",
                t.stage2_frames_ms,
                Scalar::new(90.0, 230.0, 110.0, 0.0),
            ),
            (
                "Coarse",
                t.stage3_strokes_ms,
                Scalar::new(70.0, 210.0, 255.0, 0.0),
            ),
            (
                "Recursive+Post",
                t.stage5_post_ms,
                Scalar::new(110.0, 140.0, 255.0, 0.0),
            ),
        ];
        let denom = t.total_ms.max(1.0);
        let bar_max_w = (panel_w - scaled(165, scale)).max(40);
        for (name, ms, col) in stages {
            let label = format!("{name:<12} {:>5.1}ms", ms);
            imgproc::put_text(
                canvas,
                &label,
                Point::new(x + scaled(12, scale), cy),
                font,
                0.37 * scale,
                Appearance::color(&appearance.ui_foreground),
                1,
                LINE_AA,
                false,
            )?;
            let bw = (((ms / denom) * (bar_max_w as f64)).round() as i32).clamp(2, bar_max_w);
            imgproc::rectangle(
                canvas,
                Rect::new(
                    x + scaled(150, scale),
                    cy - scaled(9, scale),
                    bw,
                    scaled(9, scale),
                ),
                col,
                -1,
                imgproc::LINE_8,
                0,
            )?;
            cy += scaled(16, scale);
        }
    } else {
        imgproc::put_text(
            canvas,
            "Debug buffers unavailable in Grid mode.",
            Point::new(x + scaled(12, scale), cy),
            font,
            0.40 * scale,
            Appearance::color(&appearance.ui_foreground),
            1,
            LINE_AA,
            false,
        )?;
    }

    Ok(())
}

fn draw_options_modal(
    canvas: &mut Mat,
    session: &Session,
    settings: &RuntimeSettings,
) -> opencv::Result<()> {
    let scale = settings.appearance.ui_scale;
    let modal_w = scaled(620, scale).min(canvas.cols() - 24).max(260);
    let modal_h = scaled(370, scale).min(canvas.rows() - 24).max(200);
    let x = ((canvas.cols() - modal_w) / 2).max(8);
    let y = ((canvas.rows() - modal_h) / 2).max(8);
    let rect = Rect::new(x, y, modal_w, modal_h);

    imgproc::rectangle(
        canvas,
        rect,
        Appearance::color(&settings.appearance.ui_background),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::rectangle(
        canvas,
        rect,
        Appearance::color(&settings.appearance.ui_accent),
        2,
        imgproc::LINE_8,
        0,
    )?;

    let font = imgproc::FONT_HERSHEY_SIMPLEX;
    let mut cy = y + scaled(26, scale);
    imgproc::put_text(
        canvas,
        "UNIVIM OPTIONS MENU (j/k select, h/l adjust, s save, Esc close)",
        Point::new(x + scaled(14, scale), cy),
        font,
        0.41 * scale,
        Appearance::color(&settings.appearance.ui_accent),
        1,
        LINE_AA,
        false,
    )?;
    cy += scaled(24, scale);

    let rows = [
        (
            "Min Target Score",
            format!("{:.2}", settings.detector.min_score),
        ),
        (
            "Include Text Blocks",
            on_off(settings.detector.include_text),
        ),
        (
            "Fast Half-Resolution",
            on_off(settings.detector.fast_half_res),
        ),
        (
            "Collapse Paragraphs",
            on_off(settings.detector.collapse_paragraphs),
        ),
        (
            "Double-Tap Window",
            format!("{} ms", settings.double_tap_ms),
        ),
        ("1st-Tap Cache TTL", format!("{} ms", settings.cache_ttl_ms)),
        ("Spotlight Background Dim", on_off(settings.spotlight_dim)),
        ("Show Status Pill", on_off(settings.show_status_bar)),
        ("Evdev Listener (Caps)", on_off(settings.use_evdev)),
        (
            "Pointer Movement",
            match settings.pointer_motion {
                PointerMotion::Instant => "Instant".to_string(),
                PointerMotion::Linear => "Linear".to_string(),
                PointerMotion::EaseOut => "Ease Out".to_string(),
            },
        ),
        ("Pointer Speed", format!("{} px/s", settings.pointer_speed)),
    ];

    for (idx, (label, val)) in rows.iter().enumerate() {
        let selected = idx == session.selected_option();
        if selected {
            imgproc::rectangle(
                canvas,
                Rect::new(
                    x + scaled(10, scale),
                    cy - scaled(15, scale),
                    modal_w - scaled(20, scale),
                    scaled(21, scale),
                ),
                Scalar::new(55.0, 50.0, 68.0, 0.0),
                -1,
                imgproc::LINE_8,
                0,
            )?;
        }
        let prefix = if selected { "> " } else { "  " };
        let text = format!("{prefix}{label}");
        let color = if selected {
            Appearance::color(&settings.appearance.ui_accent)
        } else {
            Appearance::color(&settings.appearance.ui_foreground)
        };
        imgproc::put_text(
            canvas,
            &text,
            Point::new(x + scaled(14, scale), cy),
            font,
            0.42 * scale,
            color,
            1,
            LINE_AA,
            false,
        )?;
        imgproc::put_text(
            canvas,
            val,
            Point::new(x + modal_w - scaled(135, scale), cy),
            font,
            0.42 * scale,
            color,
            1,
            LINE_AA,
            false,
        )?;
        cy += scaled(24, scale);
    }

    Ok(())
}

fn on_off(v: bool) -> String {
    if v {
        "ON".to_string()
    } else {
        "OFF".to_string()
    }
}

pub fn bgr_to_argb_bytes(bgr: &Mat) -> opencv::Result<Vec<u8>> {
    let rows = bgr.rows();
    let cols = bgr.cols();
    let mut out = vec![0u8; (rows as usize) * (cols as usize) * 4];
    for r in 0..rows {
        let row = bgr.at_row::<opencv::core::Vec3b>(r)?;
        let row_out =
            &mut out[(r as usize) * (cols as usize) * 4..(r as usize + 1) * (cols as usize) * 4];
        for (c, px) in row.iter().enumerate() {
            let idx = c * 4;
            row_out[idx] = px[0];
            row_out[idx + 1] = px[1];
            row_out[idx + 2] = px[2];
            row_out[idx + 3] = 255;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{bgr_to_argb_bytes, render_session_surface};
    use crate::{
        Target,
        desktop::{
            RuntimeSettings,
            geometry::FrameGeometry,
            session::{Input, Session},
        },
    };
    use opencv::core::{CV_8UC3, Mat, Scalar};

    #[test]
    fn converts_bgr_pixels_to_little_endian_argb8888() {
        let mat =
            Mat::new_rows_cols_with_default(1, 1, CV_8UC3, Scalar::new(10.0, 20.0, 30.0, 0.0))
                .unwrap();
        let bytes = bgr_to_argb_bytes(&mat).unwrap();
        assert_eq!(bytes, vec![10, 20, 30, 255]);
    }

    #[test]
    fn filtered_session_render_keeps_surface_dimensions() {
        let base =
            Mat::new_rows_cols_with_default(60, 80, CV_8UC3, Scalar::new(240.0, 240.0, 240.0, 0.0))
                .unwrap();
        let target = Target {
            x: 12,
            y: 14,
            w: 24,
            h: 18,
            score: 0.9,
            kind: "frame".to_string(),
            hint: "SA".to_string(),
            level: "control".to_string(),
            children: Vec::new(),
        };
        let mut session = Session::hints(80, 60, vec![target]);
        let _ = session.handle_input(Input::Char {
            ch: 's',
            shift: false,
            ctrl: false,
            alt: false,
        });
        let geometry = FrameGeometry {
            output: crate::desktop::geometry::Output {
                name: "test".into(),
                x: 0,
                y: 0,
                width: 80,
                height: 60,
                scale: 1.0,
                transform: 0,
            },
            pixel_width: 80,
            pixel_height: 60,
        };
        let settings = RuntimeSettings::default();
        let surface = render_session_surface(&base, &session, &geometry, None, &settings).unwrap();
        assert_eq!((surface.width, surface.height), (80, 60));
        assert_eq!(surface.argb.len(), 80 * 60 * 4);
    }
}
