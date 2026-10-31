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
    pub scale_back: f64,
    pub signal_ms: f64,
    pub frames_ms: f64,
    pub coarse_ms: f64,
}

pub fn prepare(bgr: &Mat, options: Options) -> opencv::Result<State> {
    let t0 = Instant::now();
    let (orig_h, orig_w) = (bgr.rows(), bgr.cols());
    let mut scale_back = 1.0;
    let resized;
    let bgr = if options.turbo && orig_w > 1600 {
        let work_w = 1280.min(orig_w >> 1);
        let work_h =
            ((orig_h as f64 * (work_w as f64 / orig_w as f64)).round_ties_even() as i32).max(1);
        resized = resize(bgr, work_w, work_h, imgproc::INTER_AREA)?;
        scale_back = orig_w as f64 / work_w as f64;
        &resized
    } else if !options.turbo && orig_w > 2560 {
        let work_h = ((orig_h as f64 * (2560.0 / orig_w as f64)).round_ties_even() as i32).max(1);
        resized = resize(bgr, 2560, work_h, imgproc::INTER_AREA)?;
        scale_back = orig_w as f64 / 2560.0;
        &resized
    } else if orig_w < 900 {
        let work_h = ((orig_h as f64 * (960.0 / orig_w as f64)).round_ties_even() as i32).max(1);
        resized = resize(bgr, 960, work_h, imgproc::INTER_CUBIC)?;
        scale_back = orig_w as f64 / 960.0;
        &resized
    } else {
        bgr
    };
    let (h, w) = (bgr.rows(), bgr.cols());
    let s = ((w * h) as f64 / (1440 * 900) as f64)
        .sqrt()
        .clamp(0.65, 2.0);

    // Stage 1: Gradients and scale-proportional structural barriers.
    let mut gray = Mat::default();
    imgproc::cvt_color_def(bgr, &mut gray, imgproc::COLOR_BGR2GRAY)?;
    // Release channel splits and max/min temporaries before allocating the
    // full-resolution gradient masks. Keeping them alive inflates the working
    // set substantially on large screenshots without contributing to later stages.
    let (chroma, blue_excess) = {
        let mut channels = Vector::<Mat>::new();
        core::split(bgr, &mut channels)?;
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
        let mut blue_excess = Mat::default();
        core::subtract(&bgr_max, &bgr_min, &mut chroma, &no_array(), -1)?;
        core::subtract(&b, &r, &mut blue_excess, &no_array(), -1)?;
        (chroma, blue_excess)
    };

    let k_dx = kernel(2, 1)?;
    let k_dy = kernel(1, 2)?;
    let dx_gray = morph(&gray, imgproc::MORPH_GRADIENT, &k_dx)?;
    let dy_gray = morph(&gray, imgproc::MORPH_GRADIENT, &k_dy)?;
    let mut dx = Mat::default();
    let mut dy = Mat::default();
    let mut grad_max = Mat::default();
    core::max(
        &dx_gray,
        &morph(&chroma, imgproc::MORPH_GRADIENT, &k_dx)?,
        &mut dx,
    )?;
    core::max(
        &dy_gray,
        &morph(&chroma, imgproc::MORPH_GRADIENT, &k_dy)?,
        &mut dy,
    )?;
    core::max(&dx, &dy, &mut grad_max)?;

    let ortho_thr = if w * h <= 600000 { 2.0 } else { 4.0 };
    let h_pure = and(
        &threshold(&dy, ortho_thr, 255.0, false)?,
        &threshold(&dx, 2.0, 255.0, true)?,
    )?;
    let h_bar = morph(&h_pure, imgproc::MORPH_OPEN, &kernel(38, 1)?)?;
    let h_zone = dilate(
        &morph(&h_pure, imgproc::MORPH_OPEN, &kernel(30, 1)?)?,
        &kernel(9, 9)?,
    )?;
    let v_pure = and(
        &threshold(&dx, ortho_thr, 255.0, false)?,
        &threshold(&dy, 2.0, 255.0, true)?,
    )?;
    let v_raw = morph(&v_pure, imgproc::MORPH_OPEN, &kernel(1, 24)?)?;
    let mut v_bar = zeros(h, w)?;
    let min_v_long = 44.max((58.0 * s) as i32);
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

    // Stage 2: Closed-loop frames, tile containers, and media thumbnails.
    let flat_pix0 = threshold(&grad_max, 0.0, 255.0, true)?;
    let flat_pix = threshold(&grad_max, 3.0, 255.0, true)?;
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
    let border_pix = threshold(&grad_max, ortho_thr, 255.0, false)?;
    let mut cnts_list = contours(&and(&border_pix, &hv_seed)?, imgproc::RETR_LIST)?;
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
            if cw >= 28
                && cw <= (72.0 * s) as i32
                && ch >= 28
                && ch <= (72.0 * s) as i32
                && (cw - ch).abs() <= 6
                && imgproc::contour_area(&c, false)? >= (cw * ch) as f64 * 0.72
            {
                cnts_list.push(c);
            }
        }
    }

    let min_frame_h = 16.max((16.0 * s) as i32);
    let max_frame_h = 46.max((58.0 * s) as i32);
    let min_frame_w = 22.max((24.0 * s) as i32);
    let max_frame_w = 440.max((w as f64 * 0.85) as i32);
    let min_thumb_h = 48.max((52.0 * s) as i32);
    let max_thumb_h = 255.max((h as f64 * 0.52) as i32);
    let min_thumb_w = 64.max((68.0 * s) as i32);
    let max_thumb_w = 420.max((w as f64 * 0.48) as i32);
    let mut raw_frames = Vec::new();
    let mut raw_thumbs = Vec::new();
    let mut tile_containers = Vec::new();
    let mut panel_corners = Vec::new();

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
            if imgproc::contour_area(&c, false)? >= (cw * ch) as f64 * 0.70 {
                let flat_r = count(&flat_pix, x + 2, y + 2, x + cw - 2, y + ch - 2)? as f64
                    / ((cw - 4) * (ch - 4)).max(1) as f64;
                if flat_r < 0.28 || max_roi(&grad_max, x, y, cw, ch)? < 48.0 {
                    continue;
                }
                let tb_s = count(&h_pure, x + 3, y, x + cw - 3, y + 4)?
                    + count(&h_pure, x + 3, (y + ch - 4).max(0), x + cw - 3, y + ch)?;
                let lr_s = count(&v_pure, x, y + 3, x + 4, y + ch - 3)?
                    + count(&v_pure, (x + cw - 4).max(0), y + 3, x + cw, y + ch - 3)?;
                let tb_r = tb_s as f64 / (cw - 6).max(1) as f64;
                let is_pill = tb_s >= 12.max(((cw - ch) as f64 * 0.75) as i32) && lr_s >= 4;
                if !is_pill
                    && (tb_s < 8.max(((cw - 8) as f64 * 0.45) as i32)
                        || (ch >= 20 && cw >= 55 && lr_s < 5.max(((ch - 12) as f64 * 0.35) as i32)))
                {
                    continue;
                }
                if cw <= 48.max((62.0 * s) as i32) && ch <= 42.max((54.0 * s) as i32) && tb_r < 1.80
                {
                    let mut l_ink = 0;
                    let mut r_ink = 0;
                    let left = (x - 6).max(0) as usize..x as usize;
                    let right = (x + cw) as usize..w.min(x + cw + 6) as usize;
                    // Inspect neighboring ink without per-pixel OpenCV access.
                    // Row slices respect Mat stride; >= 25 must not become > 25.
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
                    if outer_flat < 0.65 {
                        continue;
                    }
                }
                if cw > (85.0 * s) as i32 {
                    let ix1 = (x + (24.0 * s) as i32).clamp(0, w);
                    let ix2 = (x + (25.0 * s) as i32)
                        .max(x + cw - (38.0 * s) as i32)
                        .clamp(0, w);
                    let mut divided = false;
                    for xx in ix1..ix2 {
                        if count(&v_bar, xx, y + 3, xx + 1, y + ch - 3)? >= ch - 5 {
                            divided = true;
                            break;
                        }
                    }
                    if divided {
                        continue;
                    }
                }
                raw_frames.push(Target::new(x, y, cw, ch, 0.94, "frame", "control"));
            }
        } else if ch >= min_thumb_h
            && ch <= max_thumb_h
            && cw >= min_thumb_w
            && cw <= max_thumb_w
            && imgproc::contour_area(&c, false)? >= (cw * ch) as f64 * 0.78
        {
            let tb_s = count(&h_pure, x + 6, y, x + cw - 6, y + 4)?
                + count(&h_pure, x + 6, (y + ch - 4).max(0), x + cw - 6, y + ch)?;
            let lr_s = count(&v_pure, x, y + 6, x + 4, y + ch - 6)?
                + count(&v_pure, (x + cw - 4).max(0), y + 6, x + cw, y + ch - 6)?;
            if (tb_s as f64) < (cw - 12) as f64 * 0.60 || (lr_s as f64) < (ch - 12) as f64 * 0.60 {
                continue;
            }
            let inner_area = ((cw - 6) * (ch - 6)).max(1) as f64;
            let fr0 = count(&flat_pix0, x + 3, y + 3, x + cw - 3, y + ch - 3)? as f64 / inner_area;
            let fr3 = count(&flat_pix, x + 3, y + 3, x + cw - 3, y + ch - 3)? as f64 / inner_area;
            let g_std = std_roi(&gray, x + 3, y + 3, cw - 6, ch - 6)?;
            let ar = cw as f64 / ch as f64;
            if fr0 < 0.60 && fr3 < 0.78 && g_std >= 24.0 && (0.75..=2.8).contains(&ar) {
                raw_thumbs.push(Target::new(x, y, cw, ch, 0.93, "thumbnail", "control"));
            } else if fr3 >= 0.72 {
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

    // Stable descending order matters when equal-area contours overlap.
    raw_frames.sort_by_key(|f| std::cmp::Reverse(f.area()));
    let mut frames: Vec<Target> = Vec::new();
    for f in raw_frames {
        let f_area = f.area();
        if thumbs
            .iter()
            .any(|t| overlap(&f, t) as f64 >= 0.5 * f_area as f64)
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

    let mut thumb_occ = zeros(h, w)?;
    let mut frame_occ = zeros(h, w)?;
    let mut barrier = or(&h_bar, &v_bar)?;
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

    // Stage 3: Coarse proposals, retaining OpenCV's contour enumeration order.
    let stroke_clean = and(&threshold(&grad_max, 20.0, 255.0, false)?, &barrier_inv)?;
    let stroke_01 = threshold(&stroke_clean, 0.0, 1.0, false)?;
    let blue_01 = threshold(&blue_excess, 55.0, 1.0, false)?;
    let mut coarse_boxes = Vec::new();
    if options.turbo && scale_back == 1.0 {
        let (h2, w2) = (h >> 1, w >> 1);
        let stroke_h = threshold(
            &resize(&stroke_clean, w2, h2, imgproc::INTER_AREA)?,
            1.0,
            255.0,
            false,
        )?;
        let barrier_inv_h = resize(&barrier_inv, w2, h2, imgproc::INTER_NEAREST)?;
        let coarse_h = and(
            &morph(&stroke_h, imgproc::MORPH_CLOSE, &kernel(8, 3)?)?,
            &barrier_inv_h,
        )?;
        for c in contours(&coarse_h, imgproc::RETR_EXTERNAL)? {
            let bb = imgproc::bounding_rect(&c)?;
            let (cx, cy, cw, ch) = (bb.x, bb.y, bb.width, bb.height);
            if (cw >= 3 && ch >= 3) || (cw >= 4 && ch >= 1) || (cw >= 2 && ch >= 7) {
                let x = ((cx << 1) - 2).max(0);
                let y = ((cy << 1) - 2).max(0);
                coarse_boxes.push((x, y, (w - x).min((cw << 1) + 4), (h - y).min((ch << 1) + 4)));
            }
        }
    } else {
        let k_c = if s >= 1.55 {
            kernel(20, 8)?
        } else {
            kernel(16, 6)?
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
                || (cw >= 3 && ch >= 10.max((13.0 * s) as i32) && ch <= 36.max((42.0 * s) as i32))
            {
                coarse_boxes.push((cx, cy, cw, ch));
            }
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
        scale_back,
        signal_ms: t_sig.duration_since(t0).as_secs_f64() * 1000.0,
        frames_ms: t_frames.duration_since(t_sig).as_secs_f64() * 1000.0,
        coarse_ms: t_coarse.duration_since(t_frames).as_secs_f64() * 1000.0,
    })
}
