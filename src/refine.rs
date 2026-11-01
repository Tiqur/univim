//! Refine coarse stroke proposals into controls, text, and frame children.
//!
//! Each proposal is cropped to its ink, then recursively split at horizontal
//! whitespace bands, isolated separators, large column gaps, blue accent
//! boundaries, or with single-pass balanced sub-blocks. Splitting precedes
//! texture rejection and classification so unrelated controls joined by coarse
//! morphology can survive independently. The order of these decisions, their
//! thresholds, and first-match tie handling follow the Python detector; changing
//! them can alter both target geometry and overlay ordering.
//!
//! Geometry stays in signal-image coordinates except for word children, whose
//! JSON coordinates are scaled here. Frame children bypass outer-target dedup.
//! Projection scans borrow one checked Mat row at a time: this avoids per-pixel
//! OpenCV access checks without assuming adjacent rows are contiguous in memory.

use crate::signals::{State, has_rect_card_borders, is_real_photo_thumbnail};
use crate::{Options, Target, hints, median, overlap};
use opencv::{core, imgproc, prelude::*};
use serde_json::json;

// Half-open runs let projection offsets serve directly as slice and box bounds.
fn runs(mask: &[bool]) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    let mut start = None;
    for (i, &active) in mask.iter().enumerate() {
        if active && start.is_none() {
            start = Some(i);
        } else if !active && let Some(s) = start.take() {
            result.push((s, i));
        }
    }
    if let Some(s) = start {
        result.push((s, mask.len()));
    }
    result
}

fn blue_closed(columns: &[u64]) -> Vec<bool> {
    // NumPy's "same" keeps max(n, 12) samples, starting at (min(n, 12) - 1) / 2.
    // For n >= 12, output i therefore sees input i-6 through i+5 (inclusive).
    let offset = (columns.len().min(12) - 1) / 2;
    (0..columns.len().max(12))
        .map(|i| {
            let full_i = i + offset;
            let start = full_i.saturating_sub(11).min(columns.len());
            let end = (full_i + 1).min(columns.len());
            columns[start..end].iter().any(|&v| v > 0)
        })
        .collect()
}

fn median_f64(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    Some(if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    })
}

pub fn refine(state: &mut State, options: Options) -> opencv::Result<(Vec<Target>, Vec<Target>)> {
    let s = state.s;
    #[allow(clippy::too_many_arguments)]
    fn refine_region(
        state: &State,
        options: Options,
        (rx, ry, rw, rh): (i32, i32, i32, i32),
        depth: i32,
        refined: &mut Vec<Target>,
        inner: &mut Vec<Target>,
    ) -> opencv::Result<()> {
        if rw < 3 || rh < 1 || depth > 16 {
            return Ok(());
        }
        let s = state.s;
        let l0 = state.l0;
        // Proposals are image-bounded, and recursive splits only shrink them.
        // Crop rows first; columns must be measured over that surviving height.
        let mut row_proj = vec![0_u64; rh as usize];
        for (y, projection) in row_proj.iter_mut().enumerate() {
            let row = state.stroke_01.at_row::<u8>(ry + y as i32)?;
            *projection = row[rx as usize..(rx + rw) as usize]
                .iter()
                .map(|&v| v as u64)
                .sum();
        }
        let Some(y0) = row_proj.iter().position(|&v| v > 0) else {
            return Ok(());
        };
        let y1 = row_proj.iter().rposition(|&v| v > 0).unwrap() + 1;
        let row_proj = &row_proj[y0..y1];
        let mut col_proj = vec![0_u64; rw as usize];
        for y in y0..y1 {
            let row = state.stroke_01.at_row::<u8>(ry + y as i32)?;
            for (x, &v) in row[rx as usize..(rx + rw) as usize].iter().enumerate() {
                col_proj[x] += v as u64;
            }
        }
        let Some(x0) = col_proj.iter().position(|&v| v > 0) else {
            return Ok(());
        };
        let x1 = col_proj.iter().rposition(|&v| v > 0).unwrap() + 1;
        let col_proj = &col_proj[x0..x1];
        // Keep the row projection from before column cropping, as in the reference.
        let (rx, ry, rw, rh) = (
            rx + x0 as i32,
            ry + y0 as i32,
            (x1 - x0) as i32,
            (y1 - y0) as i32,
        );
        if *state
            .thumb_occ
            .at_2d::<u8>(ry + (rh >> 1), rx + (rw >> 1))?
            > 0
        {
            return Ok(());
        }
        let inside_frame = *state
            .frame_occ
            .at_2d::<u8>(ry + (rh >> 1), rx + (rw >> 1))?
            > 0;
        let is_micro_icon = (((5.max((7. * s) as i32)..=28.max((32. * s) as i32)).contains(&rw)
            && (1..=6.max((7. * s) as i32)).contains(&rh))
            || ((3..=8.max((10. * s) as i32)).contains(&rw)
                && (10.max((12. * s) as i32)..=34.max((40. * s) as i32)).contains(&rh)))
            && depth <= 3;
        let is_single_char_ui = (4..=12).contains(&rw) && (7..=22).contains(&rh) && depth <= 3;
        if !is_micro_icon && !is_single_char_ui && (rw < 6 || rh < 5 || (rw <= 9 && rh <= 7)) {
            return Ok(());
        }

        // Multi-line bands split at any active-row gap; compact icon guards need
        // a sustained blank run to avoid cutting through internal detail.
        let icon_guard_w = 36.max((38. * s) as i32);
        let icon_guard_h = 34.max((36. * s) as i32);
        if rh >= 13.max((1.05 * l0) as i32) {
            if rh >= 18.max((20. * s) as i32) {
                let v_thr = if rw >= (160. * s) as i32
                    && *row_proj.iter().max().unwrap() >= (80. * s) as i32 as u64
                {
                    2.max((rw as f64 * 0.015) as i32)
                } else {
                    0
                };
                let active: Vec<bool> = row_proj.iter().map(|&v| v > v_thr as u64).collect();
                let regions = runs(&active);
                if regions.len() >= 2 {
                    let min_v_gap = if rw <= icon_guard_w && rh <= icon_guard_h {
                        6
                    } else {
                        1
                    };
                    let split: Vec<bool> = regions
                        .windows(2)
                        .map(|pair| pair[1].0 - pair[0].1 >= min_v_gap as usize)
                        .collect();
                    if split.iter().any(|&v| v) {
                        let mut band_starts = vec![regions[0].0];
                        band_starts.extend(
                            regions[1..]
                                .iter()
                                .zip(&split)
                                .filter_map(|(region, &cut)| cut.then_some(region.0)),
                        );
                        let mut band_ends: Vec<usize> = regions[..regions.len() - 1]
                            .iter()
                            .zip(&split)
                            .filter_map(|(region, &cut)| cut.then_some(region.1))
                            .collect();
                        band_ends.push(regions.last().unwrap().1);
                        if band_starts.len() >= 2 {
                            for (&bs, &be) in band_starts.iter().zip(&band_ends) {
                                if be > bs {
                                    refine_region(
                                        state,
                                        options,
                                        (rx, ry + bs as i32, rw, (be - bs) as i32),
                                        depth + 1,
                                        refined,
                                        inner,
                                    )?;
                                }
                            }
                            return Ok(());
                        }
                    }
                }
            }
            if rw > icon_guard_w && rh <= (3.4 * l0) as i32 && rw >= (2.2 * rh as f64) as i32 {
                let bands = crate::glyph::foreground_line_bands(
                    state,
                    rx,
                    ry,
                    rw,
                    rh,
                    5.max((0.36 * l0) as i32),
                )?;
                if bands.len() >= 2 {
                    for (bs, be) in bands {
                        if be - bs >= 4 {
                            refine_region(
                                state,
                                options,
                                (rx, ry + bs, rw, be - bs),
                                depth + 1,
                                refined,
                                inner,
                            )?;
                        }
                    }
                    return Ok(());
                }
            }
        }

        let mut col_blue = vec![0_u64; rw as usize];
        for y in 0..rh {
            let row = state.blue_01.at_row::<u8>(ry + y)?;
            for (x, &v) in row[rx as usize..(rx + rw) as usize].iter().enumerate() {
                col_blue[x] += v as u64;
            }
        }
        let is_ink: Vec<bool> = col_proj.iter().map(|&v| v > 0).collect();
        let ink_runs = runs(&is_ink);
        if ink_runs.len() >= 2 && !is_micro_icon && !is_single_char_ui {
            // Discard a narrow neutral separator only when whitespace isolates
            // it on both sides; blue ink may itself be an actionable accent.
            if rh <= 20.max((22. * s) as i32) {
                let min_sep_gap = 5.max((rh as f64 * 0.35) as i32) as usize;
                for idx in 1..ink_runs.len() - 1 {
                    let (rs, re) = ink_runs[idx];
                    let run_w = re - rs;
                    let l_gap = rs - ink_runs[idx - 1].1;
                    let r_gap = ink_runs[idx + 1].0 - re;
                    if l_gap >= min_sep_gap
                        && r_gap >= min_sep_gap
                        && run_w <= 8
                        && col_blue[rs..re].iter().sum::<u64>() == 0
                    {
                        let mut first_y = None;
                        let mut last_y = 0;
                        for y in 0..rh {
                            let row = state.stroke_01.at_row::<u8>(ry + y)?;
                            let active = row[rx as usize + rs..rx as usize + re]
                                .iter()
                                .any(|&v| v > 0);
                            if active {
                                first_y.get_or_insert(y);
                                last_y = y;
                            }
                        }
                        let run_h = first_y.map_or(0, |first| last_y - first + 1);
                        if (run_w <= 4 && run_h <= 5)
                            || (run_w <= 8
                                && l_gap + r_gap >= 14.max((rh as f64 * 0.80) as i32) as usize)
                        {
                            let left_w = ink_runs[idx - 1].1 as i32;
                            let right_x = ink_runs[idx + 1].0 as i32;
                            refine_region(
                                state,
                                options,
                                (rx, ry, left_w, rh),
                                depth + 1,
                                refined,
                                inner,
                            )?;
                            refine_region(
                                state,
                                options,
                                (rx + right_x, ry, rw - right_x, rh),
                                depth + 1,
                                refined,
                                inner,
                            )?;
                            return Ok(());
                        }
                    }
                }
            }

            let run_widths: Vec<i32> = ink_runs.iter().map(|&(a, b)| (b - a) as i32).collect();
            let dot_limit = 10.max((0.82 * l0) as i32) as usize;
            let is_dot_track = rh <= (1.15 * l0) as i32
                && ink_runs.len() >= 3
                && run_widths
                    .iter()
                    .filter(|&&w| w as usize <= dot_limit)
                    .count() as f64
                    / run_widths.len() as f64
                    >= 0.70;
            let gaps: Vec<usize> = ink_runs
                .windows(2)
                .map(|pair| pair[1].0 - pair[0].1)
                .collect();
            let base_gap_thr = if rh >= (1.05 * l0) as i32 {
                ((1.22 * l0) as i32).max((rh as f64 * 1.12) as i32)
            } else {
                ((0.95 * l0) as i32).max((rh as f64 * 1.18) as i32)
            };
            let excluded =
                |i: usize| ink_runs[i].1 <= (1.6 * l0) as usize && gaps[i] <= (1.35 * l0) as usize;
            let fallback = |thr: i32| -> Vec<bool> {
                (0..gaps.len())
                    .map(|i| gaps[i] >= thr as usize && !excluded(i))
                    .collect()
            };
            // Wide multi-item bands are scored over three split regimes:
            // quiet gutters and balanced widths beat tiny word fragments.
            let h_splits = if !is_dot_track
                && rw >= (12.0 * l0) as i32
                && rh <= (2.4 * l0) as i32
                && gaps.len() >= 2
            {
                let mut best_mask: Option<Vec<bool>> = None;
                let mut best_score = -1e9;
                for mult in [0.85, 1.0, 1.22] {
                    let cand_thr = 0.max(
                        ((0.80 * l0) as i32)
                            .max((base_gap_thr as f64 * mult).round_ties_even() as i32),
                    );
                    let cand_mask: Vec<bool> = (0..gaps.len())
                        .map(|i| gaps[i] >= cand_thr as usize && !excluded(i))
                        .collect();
                    if !cand_mask.iter().any(|&v| v) {
                        continue;
                    }
                    let mut seg_starts = vec![ink_runs[0].0];
                    seg_starts.extend(
                        ink_runs[1..]
                            .iter()
                            .zip(&cand_mask)
                            .filter_map(|(run, &cut)| cut.then_some(run.0)),
                    );
                    let mut seg_ends: Vec<usize> = ink_runs[..ink_runs.len() - 1]
                        .iter()
                        .zip(&cand_mask)
                        .filter_map(|(run, &cut)| cut.then_some(run.1))
                        .collect();
                    seg_ends.push(ink_runs.last().unwrap().1);
                    let seg_ws: Vec<usize> = seg_starts
                        .iter()
                        .zip(&seg_ends)
                        .map(|(&a, &b)| b - a)
                        .collect();
                    let mut gutter_sum = 0.0;
                    let mut cut_count = 0;
                    for (i, &cut) in cand_mask.iter().enumerate() {
                        if !cut {
                            continue;
                        }
                        let x1 = rx + ink_runs[i].1 as i32;
                        let x2 = rx + ink_runs[i + 1].0 as i32;
                        let area = (rh * (x2 - x1)).max(1);
                        gutter_sum += crate::count(&state.flat_pix, x1, ry, x2, ry + rh)? as f64
                            / area as f64;
                        cut_count += 1;
                    }
                    let gutter_flat = gutter_sum / cut_count.max(1) as f64;
                    let tiny_pen = seg_ws.iter().filter(|&&w| w < (1.4 * l0) as usize).count()
                        as f64
                        / seg_ws.len().max(1) as f64
                        * 0.65;
                    let giant_pen =
                        if seg_ws.iter().copied().max().unwrap_or(0) > (16.0 * l0) as usize {
                            0.45
                        } else {
                            0.0
                        };
                    let score = gutter_flat - tiny_pen - giant_pen;
                    if score > best_score {
                        best_score = score;
                        best_mask = Some(cand_mask);
                    }
                }
                best_mask
                    .filter(|_| best_score >= 0.35)
                    .unwrap_or_else(|| fallback(base_gap_thr))
            } else if is_dot_track {
                vec![false; gaps.len()]
            } else {
                fallback(base_gap_thr)
            };
            if h_splits.iter().any(|&v| v) {
                let mut seg_starts = vec![ink_runs[0].0];
                seg_starts.extend(
                    ink_runs[1..]
                        .iter()
                        .zip(&h_splits)
                        .filter_map(|(run, &cut)| cut.then_some(run.0)),
                );
                let mut seg_ends: Vec<usize> = ink_runs[..ink_runs.len() - 1]
                    .iter()
                    .zip(&h_splits)
                    .filter_map(|(run, &cut)| cut.then_some(run.1))
                    .collect();
                seg_ends.push(ink_runs.last().unwrap().1);
                for (&ss, &se) in seg_starts.iter().zip(&seg_ends) {
                    if se - ss >= 3 {
                        refine_region(
                            state,
                            options,
                            (rx + ss as i32, ry, (se - ss) as i32, rh),
                            depth + 1,
                            refined,
                            inner,
                        )?;
                    }
                }
                return Ok(());
            }
        }

        if rh > 42.max((2.9 * l0) as i32) {
            let max_thumb_h = 255.max((state.h as f64 * 0.52) as i32);
            let max_thumb_w = 420.max((state.w as f64 * 0.48) as i32);
            if rh >= 50.max((3.5 * l0) as i32)
                && rh <= max_thumb_h
                && rw >= 68.max((4.6 * l0) as i32)
                && rw <= max_thumb_w
                && (0.48..=2.85).contains(&(rw as f64 / rh as f64))
                && has_rect_card_borders(
                    rx,
                    ry,
                    rw,
                    rh,
                    &state.dx_gray,
                    &state.dy_gray,
                    &state.gray,
                )?
                && is_real_photo_thumbnail(
                    &state.gray,
                    &state.grad_max,
                    &state.chroma,
                    rx + 3,
                    ry + 3,
                    rw - 6,
                    rh - 6,
                )?
            {
                if !inside_frame {
                    refined.push(Target::new(rx, ry, rw, rh, 0.91, "thumbnail", "control"));
                }
                return Ok(());
            }
            let col_span = col_proj
                .iter()
                .filter(|&&v| v >= (rh as f64 * 0.28) as u64)
                .count() as f64
                / rw.max(1) as f64;
            let is_giant_heading =
                rh <= (4.2 * l0) as i32 && rw <= (38.0 * l0) as i32 && col_span >= 0.42;
            if !is_giant_heading {
                if depth <= 6 {
                    // Single-pass balanced decomposition for large 2-D blocks,
                    // calibrated by the image's stroke threshold and text height.
                    let t_thr = 24.max((state.stroke_thr as f64 * 1.55) as i32);
                    let kw_sub = 8.max((0.65 * l0) as i32);
                    let kh_sub = 2;
                    let roi_gm =
                        Mat::roi(&state.grad_max, core::Rect::new(rx, ry, rw, rh))?.clone_pointee();
                    let roi_bi = Mat::roi(&state.barrier_inv, core::Rect::new(rx, ry, rw, rh))?
                        .clone_pointee();
                    let sub_hi = crate::and(
                        &crate::threshold(&roi_gm, t_thr as f64, 255.0, false)?,
                        &roi_bi,
                    )?;
                    let mut sub_closed = crate::morph(
                        &sub_hi,
                        imgproc::MORPH_CLOSE,
                        &crate::kernel(kw_sub, kh_sub)?,
                    )?;
                    let mut raw_b = crate::bounding_boxes(&sub_closed)?;
                    let tiny_dots = raw_b
                        .iter()
                        .filter(|&&(_, _, sw, sh)| {
                            (4..=(1.7 * l0) as i32).contains(&sw)
                                && (4..=(0.95 * l0) as i32).contains(&sh)
                        })
                        .count();
                    if raw_b.len() >= 18 && tiny_dots as f64 >= 0.65 * raw_b.len() as f64 {
                        sub_closed = crate::morph(
                            &sub_hi,
                            imgproc::MORPH_CLOSE,
                            &crate::kernel(24.max((2.2 * l0) as i32), 2)?,
                        )?;
                        raw_b = crate::bounding_boxes(&sub_closed)?;
                    }
                    for &(sx, sy, sw, sh) in &raw_b {
                        if sw >= 5
                            && sh >= 5
                            && ((sw * sh) as f64) < 0.82 * rw as f64 * rh as f64
                            && sh <= (4.0 * l0) as i32
                        {
                            refine_region(
                                state,
                                options,
                                (rx + sx, ry + sy, sw, sh),
                                depth + 2,
                                refined,
                                inner,
                            )?;
                        }
                    }
                }
                return Ok(());
            }
        }

        let total_blue = col_blue.iter().sum::<u64>();
        if total_blue > 0 && !is_micro_icon && !is_single_char_ui {
            // Group nearby accent columns, trim back to actual blue ink, then
            // include touching neutral strokes so a glyph is never cut in half.
            for (bs, be) in runs(&blue_closed(&col_blue)) {
                // With short inputs NumPy's convolution may extend past the ROI.
                let actual = &col_blue[bs.min(col_blue.len())..be.min(col_blue.len())];
                let Some(first) = actual.iter().position(|&v| v > 0) else {
                    continue;
                };
                let mut bs_t = bs + first;
                let mut be_t = bs + actual.iter().rposition(|&v| v > 0).unwrap() + 1;
                while bs_t > 0 && is_ink[bs_t - 1] {
                    bs_t -= 1;
                }
                while be_t < rw as usize && is_ink[be_t] {
                    be_t += 1;
                }
                let bw_t = be_t - bs_t;
                let has_left_neutral = bs_t >= 16 && col_proj[..bs_t].iter().sum::<u64>() >= 12;
                let has_right_neutral =
                    rw as usize - be_t >= 16 && col_proj[be_t..].iter().sum::<u64>() >= 12;
                if bw_t >= 26 && (has_left_neutral || has_right_neutral) {
                    if has_left_neutral {
                        refine_region(
                            state,
                            options,
                            (rx, ry, bs_t as i32, rh),
                            depth + 1,
                            refined,
                            inner,
                        )?;
                    }
                    refine_region(
                        state,
                        options,
                        (rx + bs_t as i32, ry, bw_t as i32, rh),
                        depth + 1,
                        refined,
                        inner,
                    )?;
                    if has_right_neutral {
                        refine_region(
                            state,
                            options,
                            (rx + be_t as i32, ry, rw - be_t as i32, rh),
                            depth + 1,
                            refined,
                            inner,
                        )?;
                    }
                    return Ok(());
                }
            }
        }

        if (rw <= (10. * s) as i32 && rh >= (44. * s) as i32)
            || (rh <= (4. * s) as i32 && rw >= (65. * s) as i32)
        {
            return Ok(());
        }

        // Flat surroundings support weaker UI strokes; textured regions need
        // stronger gradients to avoid treating image detail as text or icons.
        let (pad_x1, pad_y1) = ((rx - 6).max(0), (ry - 6).max(0));
        let (pad_x2, pad_y2) = ((rx + rw + 6).min(state.w), (ry + rh + 6).min(state.h));
        let local_flat = crate::count(&state.flat_pix, pad_x1, pad_y1, pad_x2, pad_y2)? as f64
            / ((pad_x2 - pad_x1) * (pad_y2 - pad_y1)) as f64;
        let min_dx_mean =
            if rw <= (110. * s) as i32 && rh <= (110. * s) as i32 && local_flat >= 0.55 {
                1.5 / 0.85_f64.max(s * 0.85)
            } else {
                4.2
            };
        let mean_dx = core::mean(
            &Mat::roi(&state.dx, core::Rect::new(rx, ry, rw, rh))?,
            &core::no_array(),
        )?[0];
        let max_g = crate::max_roi(&state.grad_max, rx, ry, rw, rh)?;
        let min_g_thr = if is_micro_icon {
            60.
        } else if is_single_char_ui {
            85.
        } else if local_flat >= 0.58 {
            42.
        } else {
            58.
        };
        let active_cols = is_ink.iter().filter(|&&v| v).count();
        let total_stroke = col_proj.iter().sum::<u64>();
        let mean_ink_col = total_stroke as f64 / active_cols.max(1) as f64;
        let std_pass = (is_micro_icon || mean_dx >= min_dx_mean)
            && max_g >= min_g_thr
            && !(rh >= (18. * s) as i32 && rw >= (80. * s) as i32 && mean_ink_col < 2.5)
            && (!is_micro_icon || local_flat >= 0.58)
            && !(is_single_char_ui
                && (rw < 6 || rh < 6 || (rw <= 9 && rh <= 8))
                && (local_flat < 0.74 || max_g < 95.))
            && !(local_flat < 0.48 && max_g < 72.)
            && !(local_flat < 0.34 && (max_g < 140. || rw < 24));
        let mut glyph_words = None;
        if !std_pass {
            if is_micro_icon
                || is_single_char_ui
                || rw < 14.max((1.6 * rh as f64) as i32)
                || rh > 34.max((2.2 * l0) as i32)
                || local_flat < 0.26
                || max_g < 36.
                || mean_dx < 2.2
            {
                return Ok(());
            }
            let (confidence, words) = crate::glyph::score_and_segment(state, rx, ry, rw, rh)?;
            if confidence < 0.62 || words.is_empty() {
                return Ok(());
            }
            glyph_words = Some(words);
        }

        let is_accent = total_blue >= 8.max((total_stroke as f64 * 0.22) as u64) && rw >= 24;
        let max_icon_dim = 36.max((44. * s) as i32);
        let (kind, score, level) = if is_micro_icon
            || is_single_char_ui
            || (rw <= max_icon_dim
                && rh <= max_icon_dim
                && (0.52..=1.80).contains(&(rw as f64 / rh.max(1) as f64)))
        {
            ("icon", 0.84, "control")
        } else if is_accent && rw <= (350. * s) as i32 {
            ("inline_link", 0.92, "control")
        } else if rw >= (240. * s) as i32 || (rh <= (9. * s) as i32 && rw >= (45. * s) as i32) {
            ("text_block", 0.78, "line")
        } else {
            ("text_link", 0.85, "control")
        };
        if !options.include_text && kind == "text_block" && !inside_frame {
            return Ok(());
        }

        let mut target = Target::new(rx, ry, rw, rh, score, kind, level);
        // Word children describe gaps within an accepted text target rather
        // than independent detections; only their serialized geometry is scaled.
        if matches!(kind, "text_block" | "text_link" | "inline_link")
            && rh <= (56. * s) as i32
            && (ink_runs.len() >= 2 || rw >= 18.max((rh as f64 * 2.2) as i32))
        {
            let mut raw_words = if let Some(words) = glyph_words {
                words
            } else {
                crate::glyph::score_and_segment(state, rx, ry, rw, rh)?.1
            };
            if raw_words.len() < 2 && ink_runs.len() >= 2 {
                let w_gap_thr = (if rh <= 13.max((0.95 * l0) as i32) {
                    2
                } else {
                    3
                })
                .max((rh as f64 * 0.15) as i32) as usize;
                let splits: Vec<_> = (0..ink_runs.len() - 1)
                    .filter(|&i| ink_runs[i + 1].0 - ink_runs[i].1 >= w_gap_thr)
                    .collect();
                if !splits.is_empty() {
                    raw_words.clear();
                    let mut start = 0;
                    for end in splits
                        .into_iter()
                        .chain(std::iter::once(ink_runs.len() - 1))
                    {
                        if ink_runs[end].1 - ink_runs[start].0 >= 3 {
                            raw_words.push((ink_runs[start].0 as i32, ink_runs[end].1 as i32));
                        }
                        start = end + 1;
                    }
                }
            }
            if raw_words.len() >= 2 {
                let word_hints = hints(raw_words.len());
                for (wi, &(wx0, wx1)) in raw_words.iter().enumerate() {
                    let ww = wx1 - wx0;
                    let is_lead_icon = wi == 0
                        && rh >= 14
                        && (0.65..=1.40).contains(&(ww as f64 / rh as f64))
                        && raw_words[1].0 - wx1 >= 5.max((rh as f64 * 0.25) as i32);
                    target.children.push(json!({
                        "x": ((rx + wx0) as f64 * state.scale_back) as i32,
                        "y": (ry as f64 * state.scale_back) as i32,
                        "w": ((ww as f64 * state.scale_back) as i32).max(1),
                        "h": ((rh as f64 * state.scale_back) as i32).max(1),
                        "score": score,
                        "kind": if is_lead_icon { "icon" } else { "word" },
                        "level": if is_lead_icon { "control" } else { "word" },
                        "hint": word_hints[wi],
                    }));
                }
            }
        }
        if inside_frame {
            inner.push(target);
        } else {
            refined.push(target);
        }
        Ok(())
    }

    let mut refined_boxes = Vec::new();
    let mut inner_frame_children = Vec::new();
    for &region in &state.coarse_boxes {
        refine_region(
            state,
            options,
            region,
            0,
            &mut refined_boxes,
            &mut inner_frame_children,
        )?;
    }

    // Stage 4 thumbnails feed back into the media shelf set. They are removed
    // from ordinary targets before peer-grid completion so they cannot pair.
    let mut thumbs = std::mem::take(&mut state.thumbs);
    let mut frames = std::mem::take(&mut state.frames);
    let s4_thumbs: Vec<Target> = refined_boxes
        .iter()
        .filter(|rb| rb.kind == "thumbnail")
        .cloned()
        .collect();
    refined_boxes.retain(|rb| rb.kind != "thumbnail");
    for st in s4_thumbs {
        if thumbs
            .iter()
            .any(|k| overlap(&st, k) as f64 >= 0.40 * st.area().min(k.area()) as f64)
        {
            continue;
        }
        thumbs.push(st);
    }

    complete_media_shelves(state, &refined_boxes, &mut thumbs)?;

    thumbs.sort_by_key(|t| std::cmp::Reverse(t.area()));
    let mut clean: Vec<Target> = Vec::new();
    for t in thumbs {
        if clean
            .iter()
            .any(|k| overlap(&t, k) as f64 >= 0.45 * t.area() as f64)
        {
            continue;
        }
        clean.push(t);
    }
    thumbs = clean;

    frames.retain(|f| {
        !thumbs
            .iter()
            .any(|t| overlap(f, t) as f64 >= 0.45 * f.area() as f64)
    });

    // Keep large outer targets first, suppressing proposals mostly covered by
    // a thumbnail or an accepted target. Frame children remain separate.
    refined_boxes.sort_by_key(|b| std::cmp::Reverse(b.area()));
    let mut dedup: Vec<Target> = Vec::new();
    for rb in refined_boxes {
        let area = rb.area();
        if thumbs
            .iter()
            .any(|t| overlap(&rb, t) as f64 >= 0.35 * area as f64)
        {
            continue;
        }
        if dedup
            .iter()
            .any(|kept| overlap(&rb, kept) as f64 >= 0.72 * area as f64)
        {
            continue;
        }
        dedup.push(rb);
    }
    let mut refined_boxes = dedup;
    // Restore approximate reading order before recognizing compact, regularly
    // spaced vertical groups as links rather than standalone icons.
    refined_boxes.sort_by_key(|b| (b.y.div_euclid(14), b.x));
    for i in 0..refined_boxes.len() {
        let rbi = &refined_boxes[i];
        if rbi.h <= 18.max((20. * s) as i32) && rbi.w <= 44.max((46. * s) as i32) {
            let mut peers: Vec<usize> = refined_boxes
                .iter()
                .enumerate()
                .filter(|(_, rbj)| {
                    (rbj.x - rbi.x).abs() <= 8.max((9. * s) as i32)
                        && (rbj.h - rbi.h).abs() <= 5
                        && rbj.w <= 52.max((52. * s) as i32)
                })
                .map(|(j, _)| j)
                .collect();
            if peers.len() >= 4 {
                peers.sort_by_key(|&j| refined_boxes[j].y);
                let pitches: Vec<i32> = peers
                    .windows(2)
                    .map(|p| refined_boxes[p[1]].y - refined_boxes[p[0]].y)
                    .collect();
                if median(&pitches) <= 42.max((46. * s) as i32) as f64 {
                    for j in peers {
                        refined_boxes[j].kind = "text_link".into();
                    }
                }
            }
        }
    }

    state.thumbs = thumbs;
    state.frames = frames;
    Ok((refined_boxes, inner_frame_children))
}

/// Complete same-row media shelves and extrapolate cross-row 16:9 grids.
/// Shelves and grid rows are discovered from already retained thumbnails;
/// candidates are only accepted when interior texture or a caption supports
/// them, which avoids inventing cards on flat panels.
fn complete_media_shelves(
    state: &State,
    refined: &[Target],
    thumbs: &mut Vec<Target>,
) -> opencv::Result<()> {
    let s = state.s;
    let l0 = state.l0;
    let (w, h) = (state.w, state.h);
    if thumbs.len() < 2 {
        return Ok(());
    }
    let mut major: Vec<usize> = (0..thumbs.len())
        .filter(|&i| thumbs[i].w >= (90. * s) as i32 && thumbs[i].h >= (75. * s) as i32)
        .collect();
    if major.len() < 2 {
        return Ok(());
    }
    major.sort_by_key(|&i| (thumbs[i].y, thumbs[i].x));
    let mut used = vec![false; major.len()];
    let mut shelves: Vec<Vec<usize>> = Vec::new();
    for a in 0..major.len() {
        if used[a] {
            continue;
        }
        let ti = major[a];
        let mut shelf = vec![ti];
        used[a] = true;
        for b in 0..major.len() {
            let tj = major[b];
            if !used[b]
                && (thumbs[tj].y - thumbs[ti].y).abs() <= (22. * s) as i32
                && (thumbs[tj].h - thumbs[ti].h).abs() <= (55. * s) as i32
                && (thumbs[tj].w - thumbs[ti].w).abs() <= (75. * s) as i32
            {
                shelf.push(tj);
                used[b] = true;
            }
        }
        shelf.sort_by_key(|&i| thumbs[i].x);
        shelves.push(shelf);
    }

    let mut known: Vec<(i32, i32, f64, Vec<i32>)> = Vec::new();
    let mut new_grid: Vec<Target> = Vec::new();
    for shelf in &shelves {
        if shelf.len() < 2 {
            continue;
        }
        let med_y = shelf.iter().map(|&i| thumbs[i].y).min().unwrap();
        let med_h = shelf.iter().map(|&i| thumbs[i].h).max().unwrap();
        let med_w = shelf.iter().map(|&i| thumbs[i].w).max().unwrap();
        let has_uniform_pair = (0..shelf.len()).any(|p1| {
            (0..shelf.len()).any(|p2| {
                p1 != p2
                    && ((thumbs[shelf[p1]].w - thumbs[shelf[p2]].w).abs() <= (18. * s) as i32
                        || (thumbs[shelf[p1]].h - thumbs[shelf[p2]].h).abs() <= (18. * s) as i32)
            })
        });
        if !has_uniform_pair || med_w < (110. * s) as i32 || med_h < (85. * s) as i32 {
            continue;
        }
        for &i in shelf {
            let t = &mut thumbs[i];
            if t.w < med_w - 3 && t.x + med_w <= w - 4 {
                t.w = med_w;
            }
            if t.h < med_h - 3 && med_y + med_h <= h - 2 {
                t.y = med_y;
                t.h = med_h;
            }
        }
        let diffs: Vec<i32> = shelf
            .windows(2)
            .map(|pair| thumbs[pair[1]].x - thumbs[pair[0]].x)
            .collect();
        let mut valid_p = Vec::new();
        for d in diffs {
            for mult in 1..=5 {
                let cp = d as f64 / mult as f64;
                if cp >= (med_w - (3. * s) as i32) as f64 && cp <= (med_w + (36. * s) as i32) as f64
                {
                    valid_p.push(cp);
                }
            }
        }
        let Some(pitch) = median_f64(&mut valid_p) else {
            continue;
        };
        let anchor_x = thumbs[shelf[0]].x;
        let min_feed_x = major.iter().map(|&i| thumbs[i].x).min().unwrap() - (18. * s) as i32;
        let mut col_xs = Vec::new();
        for step in -4..=6 {
            let gx = (anchor_x as f64 + step as f64 * pitch).round_ties_even() as i32;
            if gx < (60. * s) as i32 || gx < min_feed_x || gx + med_w > w - 4 {
                continue;
            }
            col_xs.push(gx);
            let mut main: Option<(bool, usize, i32)> = None;
            for (i, ex) in thumbs.iter().enumerate() {
                let hov = (gx + med_w).min(ex.x + ex.w) - gx.max(ex.x);
                let vov = (med_y + med_h).min(ex.y + ex.h) - med_y.max(ex.y);
                if hov.max(0) as f64 >= 0.35 * med_w.min(ex.w) as f64
                    && vov.max(0) as f64 >= 0.30 * med_h.min(ex.h) as f64
                    && main.is_none_or(|(_, _, area)| ex.area() > area)
                {
                    main = Some((false, i, ex.area()));
                }
            }
            for (i, ex) in new_grid.iter().enumerate() {
                let hov = (gx + med_w).min(ex.x + ex.w) - gx.max(ex.x);
                let vov = (med_y + med_h).min(ex.y + ex.h) - med_y.max(ex.y);
                if hov.max(0) as f64 >= 0.35 * med_w.min(ex.w) as f64
                    && vov.max(0) as f64 >= 0.30 * med_h.min(ex.h) as f64
                    && main.is_none_or(|(_, _, area)| ex.area() > area)
                {
                    main = Some((true, i, ex.area()));
                }
            }
            if let Some((is_new, i, _)) = main {
                if is_new {
                    let ex = &mut new_grid[i];
                    (ex.x, ex.y, ex.w, ex.h) = (gx, med_y, med_w, med_h);
                } else {
                    let ex = &mut thumbs[i];
                    (ex.x, ex.y, ex.w, ex.h) = (gx, med_y, med_w, med_h);
                }
                continue;
            }
            let roi_flat = crate::count(
                &state.flat_pix,
                gx + 4,
                med_y + 4,
                gx + med_w - 4,
                med_y + med_h - 4,
            )? as f64
                / ((med_w - 8) * (med_h - 8)).max(1) as f64;
            let has_caption = has_refined_caption(
                refined,
                gx,
                med_y + med_h,
                (2.8 * l0) as i32,
                (1.8 * l0) as i32,
                (3.2 * l0) as i32,
            );
            let std = crate::std_roi(&state.gray, gx + 4, med_y + 4, med_w - 8, med_h - 8)?;
            if (std >= 15.0 && roi_flat < 0.72) || (std >= 11.0 && has_caption) {
                new_grid.push(Target::new(
                    gx,
                    med_y,
                    med_w,
                    med_h,
                    0.92,
                    "thumbnail",
                    "control",
                ));
            }
        }
        if (1.45..=2.05).contains(&(med_w as f64 / med_h as f64)) && col_xs.len() >= 2 {
            known.push((med_w, med_h, pitch, col_xs));
        }
    }

    for shelf in &shelves {
        if shelf.len() != 1 || known.is_empty() {
            continue;
        }
        let tx = thumbs[shelf[0]].x;
        let ty = thumbs[shelf[0]].y;
        let tw = thumbs[shelf[0]].w;
        let th = thumbs[shelf[0]].h;
        for (gw, gh, gpitch, gcols) in &known {
            if (tw - gw).abs() > (14. * s) as i32 || (th - gh).abs() > (14. * s) as i32 {
                continue;
            }
            if !gcols.iter().any(|cx| (tx - cx).abs() <= (12. * s) as i32) {
                continue;
            }
            for step in -4..=4 {
                let gx = (tx as f64 + step as f64 * gpitch).round_ties_even() as i32;
                if gx < (60. * s) as i32 || gx + gw > w - 4 {
                    continue;
                }
                let collides = thumbs.iter().chain(new_grid.iter()).any(|ex| {
                    ((gx + gw).min(ex.x + ex.w) - gx.max(ex.x)) as f64 >= 0.35 * *gw as f64
                        && ((ty + gh).min(ex.y + ex.h) - ty.max(ex.y)) as f64 >= 0.35 * *gh as f64
                });
                if collides {
                    continue;
                }
                let has_caption = has_refined_caption(
                    refined,
                    gx,
                    ty + gh,
                    (45. * s) as i32,
                    (28. * s) as i32,
                    (50. * s) as i32,
                );
                let std = crate::std_roi(&state.gray, gx + 4, ty + 4, gw - 8, gh - 8)?;
                if std >= 18.0 || (std >= 11.0 && has_caption) {
                    new_grid.push(Target::new(gx, ty, *gw, *gh, 0.92, "thumbnail", "control"));
                }
            }
            break;
        }
    }
    thumbs.extend(new_grid);
    Ok(())
}

/// Check for a caption target under a grid cell: nearby x, a small positive
/// vertical gap, and a minimum label width.
fn has_refined_caption(
    refined: &[Target],
    gx: i32,
    bottom: i32,
    x_limit: i32,
    y_limit: i32,
    w_min: i32,
) -> bool {
    refined.iter().any(|rb| {
        (rb.x - gx).abs() <= x_limit && rb.y >= bottom && rb.y - bottom <= y_limit && rb.w >= w_min
    })
}
