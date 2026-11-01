//! Build shared detector signals in working-image coordinates: grayscale/chroma
//! gradients -> structural barriers and closed frames -> coarse stroke proposals.
//! `State` carries these images and proposals into refinement and composition;
//! `scale_back` maps working coordinates to the original image only downstream.
//!
//! Exact reference parity depends on resize rounding/interpolation, morphology
//! anchors, threshold boundaries, and contour/stable-sort order. Most masks use
//! 0/255, but `stroke_01` and `blue_01` deliberately use 0/1 for downstream sums.
//! ROI counts use clipped, half-open bounds; drawing uses the shared rectangle
//! helper's inclusive endpoints. These conventions are not interchangeable.

use crate::*;
use opencv::core;

pub struct State {
    pub gray: Mat,
    pub dx: Mat,
    pub dy: Mat,
    pub dx_gray: Mat,
    pub dy_gray: Mat,
    pub grad_max: Mat,
    pub chroma: Mat,
    pub barrier_inv: Mat,
    pub flat_pix: Mat,
    pub stroke_01: Mat,
    pub blue_01: Mat,
    pub thumb_occ: Mat,
    pub frame_occ: Mat,
    pub frames: Vec<Target>,
    pub thumbs: Vec<Target>,
    pub tile_containers: Vec<(i32, i32, i32, i32)>,
    pub panel_corners: Vec<(i32, i32)>,
    pub coarse_boxes: Vec<(i32, i32, i32, i32)>,
    pub w: i32,
    pub h: i32,
    pub s: f64,
    pub l0: f64,
    pub stroke_thr: i32,
    pub noise_floor: i32,
    pub flat_thr: i32,
    pub border_thr: i32,
    pub scale_back: f64,
    pub signal_ms: f64,
    pub frames_ms: f64,
    pub coarse_ms: f64,
}

struct Physics {
    noise_floor: i32,
    flat_thr: i32,
    border_thr: i32,
    stroke_thr: i32,
    l0: f64,
    u: f64,
}

/// Chroma gradients are attenuated before combination so JPEG color ringing
/// ranks below gray structural edges: `(v * 166) >> 8` with truncation.
fn scale_chroma(src: &Mat) -> opencv::Result<Mat> {
    let mut out = Mat::zeros(src.rows(), src.cols(), CV_8UC1)?.to_mat()?;
    for y in 0..src.rows() {
        let row = src.at_row::<u8>(y)?;
        let dst = out.at_row_mut::<u8>(y)?;
        for (x, &v) in row[..src.cols() as usize].iter().enumerate() {
            dst[x] = ((v as u16 * 166) >> 8) as u8;
        }
    }
    Ok(out)
}

/// Ratio-orthogonal edge mask: a primary edge must clear the floor and be at
/// least twice the cross-axis response, so diagonal glyph strokes drop out.
fn pure_mask(primary: &Mat, cross: &Mat, thr: i32) -> opencv::Result<Mat> {
    let mut out = Mat::zeros(primary.rows(), primary.cols(), CV_8UC1)?.to_mat()?;
    for y in 0..primary.rows() {
        let p = primary.at_row::<u8>(y)?;
        let c = cross.at_row::<u8>(y)?;
        let dst = out.at_row_mut::<u8>(y)?;
        for x in 0..primary.cols() as usize {
            dst[x] = if p[x] as i32 >= thr && p[x] as u16 >= 2 * c[x] as u16 {
                255
            } else {
                0
            };
        }
    }
    Ok(out)
}

/// Estimate contrast cutoffs and the fundamental text line height from the
/// image itself, replacing fixed intensity/pixel magic numbers.
fn calibrate(grad_max: &Mat) -> opencv::Result<Physics> {
    let (w, h) = (grad_max.cols(), grad_max.rows());
    let mut hist = vec![0_u64; 256];
    for y in 0..h {
        for &v in grad_max.at_row::<u8>(y)?[..w as usize].iter() {
            hist[v as usize] += 1;
        }
    }
    let total = (w as u64 * h as u64).max(1) as f64;
    let quantile = |target: f64| {
        let mut cumulative = 0_u64;
        for (i, &count) in hist.iter().enumerate() {
            cumulative += count;
            if cumulative as f64 / total >= target {
                return i as i32;
            }
        }
        255
    };
    let p35 = quantile(0.35);
    let p65 = quantile(0.65);
    let p82 = quantile(0.82);
    let noise_floor = p35.clamp(1, 5);
    let flat_thr = (noise_floor + 2).clamp(3, 6);
    let border_thr = (noise_floor + 1)
        .max((p65 as f64 * 0.75) as i32)
        .clamp(3, 6);
    let stroke_thr = ((p82 as f64 * 0.85) as i32).clamp(14, 21);

    // Modal horizontal stroke-band height approximates the UI's base font size.
    let s_bin = threshold(grad_max, stroke_thr as f64, 255.0, false)?;
    let s_closed = morph(&s_bin, imgproc::MORPH_CLOSE, &kernel(12, 2)?)?;
    let mut heights = Vec::new();
    for c in contours(&s_closed, imgproc::RETR_EXTERNAL)? {
        let bb = imgproc::bounding_rect(&c)?;
        let (bw, bh) = (bb.width, bb.height);
        if (9..=32).contains(&bh) && bw as f64 >= bh as f64 * 2.2 && bw as f64 <= w as f64 * 0.45 {
            heights.push(bh);
        }
    }
    let mut l0 = if heights.len() >= 8 {
        let mut counts = vec![0_u64; 36];
        for &bh in &heights {
            counts[bh as usize] += 1;
        }
        let mut best = 0;
        let mut best_count = 0;
        for i in 0..25 {
            let smooth = counts[8 + i] + counts[7 + i] + counts[9 + i];
            if smooth > best_count {
                best_count = smooth;
                best = i;
            }
        }
        let modal_h = (8 + best) as f64;
        let med_h = median(&heights);
        0.6 * modal_h + 0.4 * med_h
    } else {
        16.0 * ((w * h) as f64 / (1440.0 * 900.0)).sqrt().clamp(0.70, 1.60)
    };
    l0 = l0.clamp(11.5, 26.0);
    let u = (l0 / 16.0).clamp(0.68, 1.75);
    Ok(Physics {
        noise_floor,
        flat_thr,
        border_thr,
        stroke_thr,
        l0,
        u,
    })
}

/// Fraction of selected columns containing a strong vertical edge.
fn edge_columns_of_rows(
    m: &Mat,
    x1: i32,
    x2: i32,
    y1: i32,
    y2: i32,
    thr: u8,
) -> opencv::Result<f64> {
    let x1 = x1.clamp(0, m.cols());
    let x2 = x2.clamp(0, m.cols());
    let y1 = y1.clamp(0, m.rows());
    let y2 = y2.clamp(0, m.rows());
    if x2 <= x1 {
        return Ok(0.0);
    }
    let mut seen = vec![false; (x2 - x1) as usize];
    for y in y1..y2 {
        let row = m.at_row::<u8>(y)?;
        for (i, &v) in row[x1 as usize..x2 as usize].iter().enumerate() {
            if v >= thr {
                seen[i] = true;
            }
        }
    }
    Ok(seen.iter().filter(|&&v| v).count() as f64 / seen.len() as f64)
}

/// Fraction of selected rows containing a strong horizontal edge.
fn edge_rows_of_columns(
    m: &Mat,
    x1: i32,
    x2: i32,
    y1: i32,
    y2: i32,
    thr: u8,
) -> opencv::Result<f64> {
    let x1 = x1.clamp(0, m.cols());
    let x2 = x2.clamp(0, m.cols());
    let y1 = y1.clamp(0, m.rows());
    let y2 = y2.clamp(0, m.rows());
    if y2 <= y1 {
        return Ok(0.0);
    }
    let mut active = 0;
    for y in y1..y2 {
        let row = m.at_row::<u8>(y)?;
        if row[x1 as usize..x2 as usize].iter().any(|&v| v >= thr) {
            active += 1;
        }
    }
    Ok(active as f64 / (y2 - y1) as f64)
}

/// Require plausible straight card borders. A dark surround also qualifies:
/// rounded thumbnails on dark panels have short arcs rather than full edges.
pub fn has_rect_card_borders(
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    dx_gray: &Mat,
    dy_gray: &Mat,
    gray: &Mat,
) -> opencv::Result<bool> {
    let (height, width) = (gray.rows(), gray.cols());
    if w < 16 || h < 16 {
        return Ok(false);
    }
    let top_e = edge_columns_of_rows(dy_gray, x, x + w, (y - 3).max(0), (y + 4).min(height), 6)?;
    let bot_e = if y + h >= height - 8 {
        1.0
    } else {
        edge_columns_of_rows(
            dy_gray,
            x,
            x + w,
            (y + h - 4).max(0),
            (y + h + 3).min(height),
            6,
        )?
    };
    let left_e = edge_rows_of_columns(dx_gray, (x - 3).max(0), (x + 4).min(width), y, y + h, 6)?;
    let right_e = edge_rows_of_columns(
        dx_gray,
        (x + w - 4).max(0),
        (x + w + 3).min(width),
        y,
        y + h,
        6,
    )?;
    let mut sides = [top_e, bot_e, left_e, right_e];
    sides.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let (s3, min_side) = (sides[2], sides[3]);
    if min_side >= 0.50 && (s3 >= 0.68 || min_side >= 0.62) {
        return Ok(true);
    }
    if s3 >= 0.72 {
        let (oy1, oy2) = ((y - 6).max(0), (y + h + 6).min(height));
        let (ox1, ox2) = ((x - 6).max(0), (x + w + 6).min(width));
        let mut ring = Vec::new();
        for yy in oy1..oy2 {
            let row = gray.at_row::<u8>(yy)?;
            let inside_y = yy >= y && yy < y + h;
            for xx in ox1..ox2 {
                if !(inside_y && xx >= x && xx < x + w) {
                    ring.push(row[xx as usize]);
                }
            }
        }
        return Ok(!ring.is_empty() && median_u8(&ring) < 36.0);
    }
    Ok(false)
}

/// Distinguish photographic/video thumbnails from bimodal flat UI cards,
/// headers, and code panes using gradient sparsity, tone dominance, chroma, and
/// local texture across a 3x3 grid.
pub fn is_real_photo_thumbnail(
    gray: &Mat,
    grad: &Mat,
    chroma: &Mat,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
) -> opencv::Result<bool> {
    if h < 14 || w < 14 {
        return Ok(false);
    }
    let area = (w * h) as f64;
    let mut flat_grad = 0_u64;
    for yy in y..y + h {
        let row = grad.at_row::<u8>(yy)?;
        flat_grad += row[x as usize..(x + w) as usize]
            .iter()
            .filter(|&&v| v <= 6)
            .count() as u64;
    }
    if flat_grad as f64 / area >= 0.65 {
        return Ok(false);
    }
    let mut hist = [0_u64; 32];
    for yy in y..y + h {
        let row = gray.at_row::<u8>(yy)?;
        for &v in &row[x as usize..(x + w) as usize] {
            hist[v as usize >> 3] += 1;
        }
    }
    let mut bins: Vec<f64> = hist.iter().map(|&c| c as f64 / area).collect();
    bins.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dom2 = bins[30] + bins[31];
    if dom2 >= 0.64 {
        return Ok(false);
    }
    let ch_mean = mean_roi(chroma, x, y, w, h)?;
    if ch_mean < 5.5 && dom2 >= 0.42 {
        return Ok(false);
    }
    let mut active_tiles = 0;
    for r_i in 0..3 {
        let y0 = r_i * h / 3;
        let y1 = (r_i + 1) * h / 3;
        for c_i in 0..3 {
            let x0 = c_i * w / 3;
            let x1 = (c_i + 1) * w / 3;
            if std_roi(gray, x + x0, y + y0, x1 - x0, y1 - y0)? >= 9.5 {
                active_tiles += 1;
            }
        }
    }
    Ok(active_tiles >= 5)
}

pub fn prepare(bgr: &Mat, options: Options) -> opencv::Result<State> {
    let t0 = Instant::now();
    let (orig_h, orig_w) = (bgr.rows(), bgr.cols());
    let mut scale_back = 1.0;
    let resized;
    let bgr = if (options.turbo || options.fast_half_res) && orig_w > 1600 {
        let work_w = 1280.min(orig_w >> 1);
        let work_h =
            ((orig_h as f64 * (work_w as f64 / orig_w as f64)).round_ties_even() as i32).max(1);
        resized = resize(bgr, work_w, work_h, imgproc::INTER_AREA)?;
        scale_back = orig_w as f64 / work_w as f64;
        &resized
    } else if !(options.turbo || options.fast_half_res) && orig_w > 2560 {
        let work_h = ((orig_h as f64 * (2560.0 / orig_w as f64)).round_ties_even() as i32).max(1);
        resized = resize(bgr, 2560, work_h, imgproc::INTER_AREA)?;
        scale_back = orig_w as f64 / 2560.0;
        &resized
    } else if orig_w < 1440 {
        let work_h = ((orig_h as f64 * (1600.0 / orig_w as f64)).round_ties_even() as i32).max(1);
        resized = resize(bgr, 1600, work_h, imgproc::INTER_CUBIC)?;
        scale_back = orig_w as f64 / 1600.0;
        &resized
    } else {
        bgr
    };
    let (h, w) = (bgr.rows(), bgr.cols());
    let mut s = ((w * h) as f64 / (1440 * 900) as f64)
        .sqrt()
        .clamp(0.65, 2.0);

    // Stage 1: gradients and JPEG-resilient structural barriers.
    let mut gray = Mat::default();
    imgproc::cvt_color_def(bgr, &mut gray, imgproc::COLOR_BGR2GRAY)?;
    let mut channels = Vector::<Mat>::new();
    core::split(bgr, &mut channels)?;
    let chroma = {
        let (b, g, r) = (channels.get(0)?, channels.get(1)?, channels.get(2)?);
        let mut bg_max = Mat::default();
        let mut bgr_max = Mat::default();
        let mut bg_min = Mat::default();
        let mut bgr_min = Mat::default();
        core::max(&b, &g, &mut bg_max)?;
        core::max(&bg_max, &r, &mut bgr_max)?;
        core::min(&b, &g, &mut bg_min)?;
        core::min(&bg_min, &r, &mut bgr_min)?;
        let mut chroma = Mat::default();
        core::subtract(&bgr_max, &bgr_min, &mut chroma, &no_array(), -1)?;
        chroma
    };
    let blue_excess = {
        let (b, r) = (channels.get(0)?, channels.get(2)?);
        let mut blue_excess = Mat::default();
        core::subtract(&b, &r, &mut blue_excess, &no_array(), -1)?;
        blue_excess
    };

    let k_dx = kernel(2, 1)?;
    let k_dy = kernel(1, 2)?;
    let dx_gray = morph(&gray, imgproc::MORPH_GRADIENT, &k_dx)?;
    let dy_gray = morph(&gray, imgproc::MORPH_GRADIENT, &k_dy)?;
    let dx_ch = scale_chroma(&morph(&chroma, imgproc::MORPH_GRADIENT, &k_dx)?)?;
    let dy_ch = scale_chroma(&morph(&chroma, imgproc::MORPH_GRADIENT, &k_dy)?)?;
    let mut dx = Mat::default();
    let mut dy = Mat::default();
    let mut grad_max = Mat::default();
    core::max(&dx_gray, &dx_ch, &mut dx)?;
    core::max(&dy_gray, &dy_ch, &mut dy)?;
    core::max(&dx, &dy, &mut grad_max)?;

    let ortho_thr = if w * h <= 1_500_000 { 3 } else { 4 };
    let k_hline = kernel(24.max((34.0 * s) as i32), 1)?;
    let k_hseg = kernel(20.max((26.0 * s) as i32), 1)?;
    let k_vline_short = kernel(1, 16.max((20.0 * s) as i32))?;
    let k_hzone = kernel(9, 9)?;

    let h_pure_raw = pure_mask(&dy, &dx, ortho_thr)?;
    let h_pure = morph(&h_pure_raw, imgproc::MORPH_CLOSE, &kernel(3, 1)?)?;
    let h_bar = morph(&h_pure, imgproc::MORPH_OPEN, &k_hline)?;
    let h_zone = dilate(&morph(&h_pure, imgproc::MORPH_OPEN, &k_hseg)?, &k_hzone)?;
    let v_pure_raw = pure_mask(&dx, &dy, ortho_thr)?;
    let v_pure = morph(&v_pure_raw, imgproc::MORPH_CLOSE, &kernel(1, 3)?)?;
    let v_raw = morph(&v_pure, imgproc::MORPH_OPEN, &k_vline_short)?;
    let mut v_bar = zeros(h, w)?;
    let min_v_long = 36.max((48.0 * s) as i32);
    for vc in contours(&v_raw, imgproc::RETR_EXTERNAL)? {
        let bb = imgproc::bounding_rect(&vc)?;
        if bb.height >= min_v_long
            || count(&h_zone, bb.x, bb.y, bb.x + bb.width, bb.y + bb.height)? > 0
        {
            let mut one = Vector::<Vector<Point>>::new();
            one.push(vc);
            imgproc::draw_contours(
                &mut v_bar,
                &one,
                -1,
                Scalar::all(255.0),
                imgproc::FILLED,
                imgproc::LINE_8,
                &no_array(),
                i32::MAX,
                Point::new(0, 0),
            )?;
        }
    }
    let t_sig = Instant::now();

    // Stage 0/2: self-calibrated thresholds and modal text height.
    let physics = calibrate(&grad_max)?;
    s = (0.55 * s + 0.45 * physics.u).clamp(0.65, 2.0);
    let flat_pix = threshold(&grad_max, physics.flat_thr as f64, 255.0, true)?;
    let stroke_18 = threshold(&grad_max, physics.stroke_thr as f64, 255.0, false)?;

    let hv_small = threshold(
        &resize(
            &or(&h_pure, &v_pure)?,
            (w >> 3).max(1),
            (h >> 3).max(1),
            imgproc::INTER_AREA,
        )?,
        1.0,
        255.0,
        false,
    )?;
    let hv_seed = resize(
        &dilate(&hv_small, &kernel(9, 5)?)?,
        w,
        h,
        imgproc::INTER_NEAREST,
    )?;
    // Multi-threshold ladder keeps subtle dark borders without JPEG gutter
    // ringing; duplicate contours are later deduplicated by frame geometry.
    let mut cnts_list = Vector::<Vector<Point>>::new();
    for b_thr in [
        (physics.border_thr - 1).max(2),
        (physics.border_thr + 1).max(4),
        (physics.border_thr + 3).max(6),
    ] {
        let bp_iter = threshold(&grad_max, b_thr as f64, 255.0, false)?;
        for c in contours(&and(&bp_iter, &hv_seed)?, imgproc::RETR_LIST)? {
            cnts_list.push(c);
        }
    }
    let strip_w = w.min((260.0 * s) as i32);
    let border_hi = threshold(
        &Mat::roi(&grad_max, Rect::new(0, 0, strip_w, h))?.try_clone()?,
        12.0,
        255.0,
        false,
    )?;
    for c in contours(&border_hi, imgproc::RETR_LIST)? {
        if c.len() >= 4 {
            let bb = imgproc::bounding_rect(&c)?;
            let (cw, ch) = (bb.width, bb.height);
            if cw >= 24
                && cw <= (72.0 * s) as i32
                && ch >= 24
                && ch <= (72.0 * s) as i32
                && (cw - ch).abs() <= 6
                && imgproc::contour_area(&c, false)? >= (cw * ch) as f64 * 0.72
            {
                cnts_list.push(c);
            }
        }
    }

    let min_frame_h = 14.max((15.0 * s) as i32);
    let max_frame_h = 46.max((58.0 * s) as i32);
    let min_frame_w = 18.max((20.0 * s) as i32);
    let max_frame_w = 440.max((w as f64 * 0.85) as i32);
    let min_thumb_h = 44.max((48.0 * s) as i32);
    let max_thumb_h = 255.max((h as f64 * 0.52) as i32);
    let min_thumb_w = 56.max((62.0 * s) as i32);
    let max_thumb_w = 420.max((w as f64 * 0.48) as i32);
    let mut raw_frames = Vec::new();
    let mut raw_thumbs = Vec::new();
    let mut tile_containers = Vec::new();
    let mut panel_corners = Vec::new();
    let l0 = physics.l0;

    for c in cnts_list {
        if c.len() < 4 {
            continue;
        }
        let bb = imgproc::bounding_rect(&c)?;
        let (x, y, cw, ch) = (bb.x, bb.y, bb.width, bb.height);
        let w_cap = if ch <= (42.0 * s) as i32 {
            max_frame_w
        } else {
            (265.0 * s) as i32
        };
        if ch >= min_frame_h && ch <= max_frame_h && cw >= min_frame_w && cw <= w_cap {
            if imgproc::contour_area(&c, false)? < (cw * ch) as f64 * 0.68 {
                continue;
            }
            let flat_r = count(&flat_pix, x + 2, y + 2, x + cw - 2, y + ch - 2)? as f64
                / ((cw - 4) * (ch - 4)).max(1) as f64;
            if flat_r < 0.26 {
                continue;
            }
            let tb_s = count(&h_pure, x + 3, y, x + cw - 3, y + 4)?
                + count(&h_pure, x + 3, (y + ch - 4).max(0), x + cw - 3, y + ch)?;
            let lr_s = count(&v_pure, x, y + 3, x + 4, y + ch - 3)?
                + count(&v_pure, (x + cw - 4).max(0), y + 3, x + cw, y + ch - 3)?;
            let tb_r = tb_s as f64 / (cw - 6).max(1) as f64;
            let lr_r = lr_s as f64 / (ch - 6).max(1) as f64;
            let is_pill = tb_s >= 10.max(((cw - ch) as f64 * 0.72) as i32) && lr_s >= 4;
            if !is_pill
                && (tb_s < 8.max(((cw - 8) as f64 * 0.45) as i32)
                    || (ch >= 18 && cw >= 48 && lr_s < 4.max(((ch - 10) as f64 * 0.35) as i32)))
            {
                continue;
            }
            if max_roi(&grad_max, x, y, cw, ch)?
                < if tb_r >= 1.15 && lr_r >= 0.75 {
                    26.0
                } else {
                    44.0
                }
            {
                continue;
            }
            let inner_strokes = count(&stroke_18, x + 4, y + 3, x + cw - 4, y + ch - 3)?;
            if inner_strokes < 5.max((((cw - 8) * (ch - 6)) as f64 * 0.010) as i32) {
                let in_mean = mean_roi(&gray, x + 3, y + 3, cw - 6, ch - 6)?;
                let (oy1, oy2) = ((y - 5).max(0), (y + ch + 5).min(h));
                let (ox1, ox2) = ((x - 5).max(0), (x + cw + 5).min(w));
                let out_mean = mean_roi(&gray, ox1, oy1, ox2 - ox1, oy2 - oy1)?;
                if (in_mean - out_mean).abs() < 10.0
                    && !(cw <= (32.0 * s) as i32 && ch <= (32.0 * s) as i32 && tb_r >= 1.35)
                {
                    continue;
                }
            }
            if cw <= 48.max((62.0 * s) as i32) && ch <= 42.max((54.0 * s) as i32) && tb_r < 1.80 {
                let mut l_ink = 0;
                let mut r_ink = 0;
                let left = (x - 6).max(0) as usize..x as usize;
                let right = (x + cw) as usize..w.min(x + cw + 6) as usize;
                for yy in y + 2..y + ch - 2 {
                    let row = grad_max.at_row::<u8>(yy)?;
                    l_ink += row[left.clone()].iter().filter(|&&v| v >= 25).count() as i32;
                    r_ink += row[right.clone()].iter().filter(|&&v| v >= 25).count() as i32;
                }
                if l_ink >= 22 || (l_ink >= 12 && r_ink >= 12) {
                    continue;
                }
                let (ox1, oy1) = ((x - 10).max(0), (y - 10).max(0));
                let (ox2, oy2) = (w.min(x + cw + 10), h.min(y + ch + 10));
                let outer_area = ((ox2 - ox1) * (oy2 - oy1) - cw * ch).max(1);
                let outer_flat = (count(&flat_pix, ox1, oy1, ox2, oy2)?
                    - count(&flat_pix, x, y, x + cw, y + ch)?)
                    as f64
                    / outer_area as f64;
                if outer_flat < 0.62 {
                    continue;
                }
            }
            if cw > (85.0 * s) as i32 {
                let ix1 = (x + (22.0 * s) as i32).clamp(0, w);
                let ix2 = (x + (23.0 * s) as i32)
                    .max(x + cw - (32.0 * s) as i32)
                    .clamp(0, w);
                let mut divided = false;
                for xx in ix1..ix2 {
                    if count(&v_bar, xx, y + 3, xx + 1, y + ch - 3)? >= ch - 6 {
                        divided = true;
                        break;
                    }
                }
                if divided {
                    continue;
                }
                let mut both = 0;
                for xx in x..x + cw {
                    let mut top = false;
                    for yy in y..y + 4 {
                        if *h_pure.at_2d::<u8>(yy, xx)? > 0 {
                            top = true;
                            break;
                        }
                    }
                    let mut bot = false;
                    for yy in (y + ch - 4).max(0)..y + ch {
                        if *h_pure.at_2d::<u8>(yy, xx)? > 0 {
                            bot = true;
                            break;
                        }
                    }
                    if top && bot {
                        both += 1;
                    }
                }
                if both as f64 / (cw as f64) < 0.78 {
                    let wall_thr = 4.max(((ch - 6) as f64 * 0.45) as i32);
                    let mut wall_cols = 0;
                    let mut inner_wall = false;
                    for xx in 0..cw {
                        let mut hits = 0;
                        for yy in y + 3..y + ch - 3 {
                            if *v_pure.at_2d::<u8>(yy, x + xx)? > 0 {
                                hits += 1;
                            }
                        }
                        if hits >= wall_thr {
                            wall_cols += 1;
                            if xx >= (24.0 * s) as i32 && xx <= cw - (24.0 * s) as i32 {
                                inner_wall = true;
                            }
                        }
                    }
                    if wall_cols >= 2 && inner_wall {
                        continue;
                    }
                }
            }
            raw_frames.push(Target::new(x, y, cw, ch, 0.94, "frame", "control"));
        } else if ch >= min_thumb_h && ch <= max_thumb_h && cw >= min_thumb_w && cw <= max_thumb_w {
            if imgproc::contour_area(&c, false)? < (cw * ch) as f64 * 0.76 {
                continue;
            }
            let tb_s = count(&h_pure, x + 6, y, x + cw - 6, y + 4)?
                + count(&h_pure, x + 6, (y + ch - 4).max(0), x + cw - 6, y + ch)?;
            let lr_s = count(&v_pure, x, y + 6, x + 4, y + ch - 6)?
                + count(&v_pure, (x + cw - 4).max(0), y + 6, x + cw, y + ch - 6)?;
            if (tb_s as f64) < (cw - 12) as f64 * 0.55 || (lr_s as f64) < (ch - 12) as f64 * 0.55 {
                continue;
            }
            let ar = cw as f64 / ch as f64;
            // Vertical stacks of buttons contain internal full-width dividers.
            let inner_y1 = (y + (0.6 * l0) as i32).clamp(0, h);
            let inner_y2 = (inner_y1 + 1).max(y + ch - (0.6 * l0) as i32).clamp(0, h);
            let inner_x1 = (x + 6).clamp(0, w);
            let inner_x2 = (x + 7).max(x + cw - 6).clamp(0, w);
            let mut max_row_frac = 0.0_f64;
            if inner_y2 > inner_y1 && inner_x2 > inner_x1 {
                for yy in inner_y1..inner_y2 {
                    let row = h_pure.at_row::<u8>(yy)?;
                    let hits = row[inner_x1 as usize..inner_x2 as usize]
                        .iter()
                        .filter(|&&v| v > 0)
                        .count();
                    max_row_frac = max_row_frac.max(hits as f64 / (inner_x2 - inner_x1) as f64);
                }
            }
            let has_internal_button_dividers =
                cw <= (7.8 * l0) as i32 && ch <= (4.8 * l0) as i32 && max_row_frac >= 0.72;
            if (0.48..=2.85).contains(&ar)
                && !has_internal_button_dividers
                && has_rect_card_borders(x, y, cw, ch, &dx_gray, &dy_gray, &gray)?
                && is_real_photo_thumbnail(&gray, &grad_max, &chroma, x + 3, y + 3, cw - 6, ch - 6)?
            {
                raw_thumbs.push(Target::new(x, y, cw, ch, 0.93, "thumbnail", "control"));
            } else {
                let fr3 = count(&flat_pix, x + 3, y + 3, x + cw - 3, y + ch - 3)? as f64
                    / ((cw - 6) * (ch - 6)).max(1) as f64;
                if fr3 >= 0.72 {
                    if (0.85..=1.18).contains(&ar)
                        && cw >= (70.0 * s) as i32
                        && cw <= (180.0 * s) as i32
                        && tb_s as f64 >= (cw - 8) as f64 * 1.1
                        && lr_s as f64 >= (ch - 8) as f64 * 1.1
                    {
                        tile_containers.push((x, y, cw, ch));
                    }
                    if cw >= (160.0 * s) as i32
                        && ch >= (120.0 * s) as i32
                        && tb_s as f64 >= (cw - 8) as f64 * 1.1
                        && lr_s as f64 >= (ch - 8) as f64 * 1.1
                    {
                        panel_corners.extend([(x, y), (x + cw, y), (x, y + ch), (x + cw, y + ch)]);
                    }
                }
            }
        }
    }

    let mut filtered_thumbs = Vec::new();
    for (i, t) in raw_thumbs.iter().enumerate() {
        if raw_thumbs.iter().enumerate().any(|(j, o)| {
            i != j
                && (t.area() as f64) < 0.38 * o.area() as f64
                && overlap(t, o) as f64 >= 0.80 * t.area() as f64
        }) {
            continue;
        }
        filtered_thumbs.push(t.clone());
    }
    filtered_thumbs.sort_by_key(Target::area);
    let mut thumbs: Vec<Target> = Vec::new();
    for t in filtered_thumbs {
        if thumbs
            .iter()
            .any(|k| overlap(&t, k) as f64 >= 0.45 * t.area().min(k.area()) as f64)
        {
            continue;
        }
        thumbs.push(t);
    }

    raw_frames.sort_by_key(|f| std::cmp::Reverse(f.area()));
    let mut frames: Vec<Target> = Vec::new();
    for f in raw_frames {
        let f_area = f.area();
        if thumbs
            .iter()
            .any(|t| overlap(&f, t) as f64 >= 0.45 * f_area as f64)
        {
            continue;
        }
        let mut absorbed = false;
        for k in &mut frames {
            let inter = overlap(&f, k);
            if inter == 0 {
                continue;
            }
            let iou = inter as f64 / (f_area + k.area() - inter) as f64;
            let cov_f = inter as f64 / f_area as f64;
            if iou >= 0.45 || cov_f >= 0.75 {
                if k.w > (320.0 * s) as i32
                    && f.w <= (220.0 * s) as i32
                    && f.x >= k.x + k.w - f.w - (16.0 * s) as i32
                    && f.x - k.x >= (120.0 * s) as i32
                {
                    k.w = f.x - k.x - 2;
                } else {
                    absorbed = true;
                    break;
                }
            }
        }
        if !absorbed {
            frames.push(f);
        }
    }

    // Inline word highlights create short pure edges inside prose; remove them
    // as barriers so the prose line can still be grouped as a whole.
    let mut h_bar_clean = h_bar.clone();
    for hc in contours(&h_bar_clean, imgproc::RETR_EXTERNAL)? {
        let bb = imgproc::bounding_rect(&hc)?;
        let (hx, hy, hw, hh) = (bb.x, bb.y, bb.width, bb.height);
        if hw < (7.5 * l0) as i32 && hh <= (0.6 * l0) as i32 {
            let by1 = (hy - (1.1 * l0) as i32).max(0);
            let by2 = (hy + hh + (1.1 * l0) as i32).min(h);
            let lx1 = (hx - (1.2 * l0) as i32).max(0);
            let lx2 = (hx - 2).max(0);
            let rx1 = (hx + hw + 2).min(w);
            let rx2 = (hx + hw + (1.2 * l0) as i32).min(w);
            let l_str = if lx2 > lx1 {
                count(&stroke_18, lx1, by1, lx2, by2)?
            } else {
                0
            };
            let r_str = if rx2 > rx1 {
                count(&stroke_18, rx1, by1, rx2, by2)?
            } else {
                0
            };
            if l_str >= (0.9 * l0) as i32 && r_str >= (0.9 * l0) as i32 {
                rect(
                    &mut h_bar_clean,
                    hx,
                    hy,
                    hw,
                    hh,
                    Scalar::all(0.0),
                    -1,
                    imgproc::LINE_8,
                )?;
            }
        }
    }
    let mut v_bar_clean = v_bar.clone();
    for vc in contours(&v_bar_clean, imgproc::RETR_EXTERNAL)? {
        let bb = imgproc::bounding_rect(&vc)?;
        let (vx, vy, vw, vh) = (bb.x, bb.y, bb.width, bb.height);
        if vh <= (2.2 * l0) as i32 && vw <= (0.6 * l0) as i32 {
            let lx1 = (vx - (1.2 * l0) as i32).max(0);
            let lx2 = (vx - 2).max(0);
            let rx1 = (vx + vw + 2).min(w);
            let rx2 = (vx + vw + (1.2 * l0) as i32).min(w);
            let l_str = if lx2 > lx1 {
                count(&stroke_18, lx1, vy, lx2, vy + vh)?
            } else {
                0
            };
            let r_str = if rx2 > rx1 {
                count(&stroke_18, rx1, vy, rx2, vy + vh)?
            } else {
                0
            };
            if l_str >= (0.6 * l0) as i32 && r_str >= (0.6 * l0) as i32 {
                rect(
                    &mut v_bar_clean,
                    vx,
                    vy,
                    vw,
                    vh,
                    Scalar::all(0.0),
                    -1,
                    imgproc::LINE_8,
                )?;
            }
        }
    }

    let mut thumb_occ = zeros(h, w)?;
    let mut frame_occ = zeros(h, w)?;
    let mut barrier = or(&h_bar_clean, &v_bar_clean)?;
    for t in &thumbs {
        rect(
            &mut thumb_occ,
            t.x,
            t.y,
            t.w,
            t.h,
            Scalar::all(255.0),
            -1,
            imgproc::LINE_8,
        )?;
        rect(
            &mut barrier,
            t.x,
            t.y,
            t.w,
            t.h,
            Scalar::all(255.0),
            2,
            imgproc::LINE_8,
        )?;
    }
    for f in &frames {
        if f.h <= (44.0 * s) as i32 && f.w <= (320.0 * s) as i32 {
            rect(
                &mut frame_occ,
                f.x,
                f.y,
                f.w,
                f.h,
                Scalar::all(255.0),
                -1,
                imgproc::LINE_8,
            )?;
        }
        rect(
            &mut barrier,
            f.x,
            f.y,
            f.w,
            f.h,
            Scalar::all(255.0),
            2,
            imgproc::LINE_8,
        )?;
    }
    let barrier_inv = invert(&dilate(&barrier, &kernel(3, 3)?)?)?;
    let t_frames = Instant::now();

    // Stage 3: coarse proposals, retaining OpenCV's contour enumeration order.
    let stroke_clean = and(&threshold(&grad_max, 18.0, 255.0, false)?, &barrier_inv)?;
    let stroke_01 = threshold(&stroke_clean, 0.0, 1.0, false)?;
    let blue_01 = threshold(&blue_excess, 55.0, 1.0, false)?;
    let mut coarse_boxes = Vec::new();
    let k_c = if s >= 1.55 {
        kernel(18, 7)?
    } else {
        kernel(14, 5)?
    };
    let coarse_mask = and(
        &morph(&stroke_clean, imgproc::MORPH_CLOSE, &k_c)?,
        &barrier_inv,
    )?;
    for c in contours(&coarse_mask, imgproc::RETR_EXTERNAL)? {
        let bb = imgproc::bounding_rect(&c)?;
        let (cx, cy, cw, ch) = (bb.x, bb.y, bb.width, bb.height);
        if (cw >= 4.max((5.0 * s) as i32) && ch >= 5.max((6.0 * s) as i32))
            || (cw >= 5.max((7.0 * s) as i32) && cw <= 28.max((34.0 * s) as i32) && ch >= 1)
            || (cw >= 3 && ch >= 10.max((12.0 * s) as i32) && ch <= 36.max((42.0 * s) as i32))
        {
            coarse_boxes.push((cx, cy, cw, ch));
        }
    }
    let t_coarse = Instant::now();

    Ok(State {
        gray,
        dx,
        dy,
        dx_gray,
        dy_gray,
        grad_max,
        chroma,
        barrier_inv,
        flat_pix,
        stroke_01,
        blue_01,
        thumb_occ,
        frame_occ,
        frames,
        thumbs,
        tile_containers,
        panel_corners,
        coarse_boxes,
        w,
        h,
        s,
        l0,
        stroke_thr: physics.stroke_thr,
        noise_floor: physics.noise_floor,
        flat_thr: physics.flat_thr,
        border_thr: physics.border_thr,
        scale_back,
        signal_ms: t_sig.duration_since(t0).as_secs_f64() * 1000.0,
        frames_ms: t_frames.duration_since(t_sig).as_secs_f64() * 1000.0,
        coarse_ms: t_coarse.duration_since(t_frames).as_secs_f64() * 1000.0,
    })
}
