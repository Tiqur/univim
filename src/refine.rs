//! Refine coarse stroke proposals into controls, text, and frame children.
//!
//! Each proposal is cropped to its ink, then recursively split at horizontal
//! whitespace, isolated separators, large column gaps, or blue accent boundaries.
//! Splitting precedes texture rejection and classification so unrelated controls
//! joined by coarse morphology can survive independently. The order of these
//! decisions, their thresholds, and first-match tie handling follow the Python
//! detector; changing them can alter both target geometry and overlay ordering.
//!
//! Geometry stays in signal-image coordinates except for word children, whose
//! JSON coordinates are scaled here. Frame children bypass outer-target dedup.
//! Projection scans borrow one checked Mat row at a time: this avoids per-pixel
//! OpenCV access checks without assuming adjacent rows are contiguous in memory.

use crate::{Options, Target, hints, median, overlap};
use opencv::{core, prelude::*};
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

pub fn refine(
    state: &crate::signals::State,
    options: Options,
) -> opencv::Result<(Vec<Target>, Vec<Target>)> {
    fn refine_region(
        state: &crate::signals::State,
        options: Options,
        (rx, ry, rw, rh): (i32, i32, i32, i32),
        depth: i32,
        refined: &mut Vec<Target>,
        inner: &mut Vec<Target>,
    ) -> opencv::Result<()> {
        if rw < 3 || rh < 1 || depth > 10 {
            return Ok(());
        }
        let s = state.s;
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
            || ((3..=7.max((9. * s) as i32)).contains(&rw)
                && (10.max((13. * s) as i32)..=34.max((40. * s) as i32)).contains(&rh)))
            && depth <= 2;
        let is_single_char_ui = (4..=12).contains(&rw) && (7..=22).contains(&rh) && depth <= 2;
        if !is_micro_icon && !is_single_char_ui && (rw < 6 || rh < 6 || (rw <= 9 && rh <= 8)) {
            return Ok(());
        }

        let icon_guard_w = 38.max((38. * s) as i32);
        let icon_guard_h = 36.max((36. * s) as i32);
        // Wide, dense lines tolerate a little ink in a separator row. Compact
        // icons need a sustained blank run to avoid splitting their own details.
        if rh >= 22 {
            let v_thr = if rw >= 200 && *row_proj.iter().max().unwrap() >= 140 {
                7
            } else {
                0
            };
            let zero_mask: Vec<bool> = row_proj[5..rh as usize - 5]
                .iter()
                .map(|&v| v <= v_thr)
                .collect();
            if zero_mask.iter().any(|&v| v) {
                let max_z_run = if rw <= icon_guard_w && rh <= icon_guard_h {
                    runs(&zero_mask)
                        .iter()
                        .map(|&(a, b)| b - a)
                        .max()
                        .unwrap_or(0)
                } else {
                    99
                };
                if max_z_run >= 6 {
                    let zero_rows: Vec<usize> = zero_mask
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &v)| v.then_some(i + 5))
                        .collect();
                    let split_y = zero_rows[zero_rows.len() >> 1] as i32;
                    let mut next_y = split_y + 1;
                    // Skip the separator band and dense rules before recursing.
                    while next_y < rh - 4
                        && (row_proj[next_y as usize] <= v_thr
                            || row_proj[next_y as usize] as f64 >= rw as f64 * 0.75)
                    {
                        next_y += 1;
                    }
                    refine_region(
                        state,
                        options,
                        (rx, ry, rw, split_y),
                        depth + 1,
                        refined,
                        inner,
                    )?;
                    refine_region(
                        state,
                        options,
                        (rx, ry + next_y, rw, rh - next_y),
                        depth + 1,
                        refined,
                        inner,
                    )?;
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

            // Strictly greater preserves np.argmax's first occurrence on ties.
            let mut max_gi = 0;
            let mut max_gap = 0;
            for i in 0..ink_runs.len() - 1 {
                let gap = ink_runs[i + 1].0 - ink_runs[i].1;
                if gap > max_gap {
                    max_gi = i;
                    max_gap = gap;
                }
            }
            let left_w = ink_runs[max_gi].1 as i32;
            let right_x = ink_runs[max_gi + 1].0 as i32;
            let gap_thresh = if rh >= 17 {
                18.max((rh as f64 * 0.85) as i32)
            } else {
                13.max((rh as f64 * 0.95) as i32)
            };
            if !(left_w <= 22 && max_gap <= 18) && max_gap >= gap_thresh as usize {
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

        // Flat surroundings support weaker UI strokes; textured regions need
        // stronger gradients to avoid treating image detail as text or icons.
        let (pad_x1, pad_y1) = ((rx - 6).max(0), (ry - 6).max(0));
        let (pad_x2, pad_y2) = ((rx + rw + 6).min(state.w), (ry + rh + 6).min(state.h));
        let local_flat = crate::count(&state.flat_pix, pad_x1, pad_y1, pad_x2, pad_y2)? as f64
            / ((pad_x2 - pad_x1) * (pad_y2 - pad_y1)) as f64;
        let min_dx_mean =
            if rw <= (110. * s) as i32 && rh <= (110. * s) as i32 && local_flat >= 0.60 {
                1.6 / 0.85_f64.max(s * 0.85)
            } else {
                5.0
            };
        if !is_micro_icon
            && core::mean(
                &Mat::roi(&state.dx, core::Rect::new(rx, ry, rw, rh))?,
                &core::no_array(),
            )?[0]
                < min_dx_mean
        {
            return Ok(());
        }
        let max_g = crate::max_roi(&state.grad_max, rx, ry, rw, rh)?;
        let min_g_thr = if is_micro_icon {
            68.
        } else if is_single_char_ui {
            95.
        } else if local_flat >= 0.62 {
            52.
        } else {
            65.
        };
        if max_g < min_g_thr {
            return Ok(());
        }
        let active_cols = is_ink.iter().filter(|&&v| v).count();
        let total_stroke = col_proj.iter().sum::<u64>();
        let mean_ink_col = total_stroke as f64 / active_cols.max(1) as f64;
        if rh >= (18. * s) as i32 && rw >= (80. * s) as i32 && mean_ink_col < 2.8 {
            return Ok(());
        }
        if is_micro_icon && local_flat < 0.62 {
            return Ok(());
        }
        if is_single_char_ui
            && (rw < 6 || rh < 6 || (rw <= 9 && rh <= 8))
            && (local_flat < 0.78 || max_g < 110.)
        {
            return Ok(());
        }
        if local_flat < 0.52 && max_g < 76. {
            return Ok(());
        }
        if local_flat < 0.38 && (max_g < 145. || rw < 24) {
            return Ok(());
        }

        let is_accent = total_blue >= 8.max((total_stroke as f64 * 0.22) as u64) && rw >= 24;
        let max_thumb_h = 255.max((state.h as f64 * 0.52) as i32);
        let max_thumb_w = 420.max((state.w as f64 * 0.48) as i32);
        if ((64. * s) as i32..=max_thumb_h).contains(&rh)
            && ((96. * s) as i32..=max_thumb_w).contains(&rw)
            && (0.75..=2.8).contains(&(rw as f64 / rh as f64))
            && crate::std_roi(&state.gray, rx, ry, rw, rh)? >= 24.
        {
            if !inside_frame {
                refined.push(Target::new(rx, ry, rw, rh, 0.91, "thumbnail", "control"));
            }
            return Ok(());
        }

        let max_icon_dim = 36.max((44. * s) as i32);
        let (kind, score, level) = if is_micro_icon
            || is_single_char_ui
            || (rw <= max_icon_dim
                && rh <= max_icon_dim
                && (0.55..=1.75).contains(&(rw as f64 / rh.max(1) as f64)))
        {
            ("icon", 0.84, "control")
        } else if is_accent && rw <= (350. * s) as i32 {
            ("inline_link", 0.92, "control")
        } else if rw >= (260. * s) as i32 || (rh <= (9. * s) as i32 && rw >= (45. * s) as i32) {
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
            && ink_runs.len() >= 2
            && rh <= (56. * s) as i32
        {
            let w_gap_thr = 3.max((rh as f64 * 0.16) as i32) as usize;
            let splits: Vec<usize> = (0..ink_runs.len() - 1)
                .filter(|&i| ink_runs[i + 1].0 - ink_runs[i].1 >= w_gap_thr)
                .collect();
            if !splits.is_empty() {
                let mut raw_words = Vec::new();
                let mut start = 0;
                for end in splits
                    .into_iter()
                    .chain(std::iter::once(ink_runs.len() - 1))
                {
                    let (wx0, wx1) = (ink_runs[start].0, ink_runs[end].1);
                    if wx1 - wx0 >= 4 {
                        raw_words.push((wx0, wx1));
                    }
                    start = end + 1;
                }
                if raw_words.len() >= 2 {
                    let word_hints = hints(raw_words.len());
                    for (wi, &(wx0, wx1)) in raw_words.iter().enumerate() {
                        let ww = wx1 - wx0;
                        let is_lead_icon = wi == 0
                            && rh >= 14
                            && (0.65..=1.40).contains(&(ww as f64 / rh as f64))
                            && raw_words[1].0 - wx1 >= 5.max((rh as f64 * 0.25) as i32) as usize;
                        target.children.push(json!({
                            "x": ((rx + wx0 as i32) as f64 * state.scale_back) as i32,
                            "y": (ry as f64 * state.scale_back) as i32,
                            "w": (ww as f64 * state.scale_back) as i32,
                            "h": (rh as f64 * state.scale_back) as i32,
                            "score": score,
                            "kind": if is_lead_icon { "icon" } else { "word" },
                            "level": if is_lead_icon { "control" } else { "word" },
                            "hint": word_hints[wi],
                        }));
                    }
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

    // Keep large outer targets first, suppressing proposals mostly covered by
    // an accepted target or known thumbnail. Frame children remain separate.
    refined_boxes.sort_by_key(|b| std::cmp::Reverse(b.area()));
    let mut dedup: Vec<Target> = Vec::new();
    for rb in refined_boxes {
        if dedup
            .iter()
            .chain(state.thumbs.iter())
            .any(|kept| overlap(&rb, kept) as f64 >= 0.70 * rb.area() as f64)
        {
            continue;
        }
        dedup.push(rb);
    }
    let mut refined_boxes = dedup;
    // Restore approximate reading order before recognizing compact, regularly
    // spaced vertical groups as links rather than standalone icons.
    refined_boxes.sort_by_key(|b| (b.y.div_euclid(14), b.x));
    let s = state.s;
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
    Ok((refined_boxes, inner_frame_children))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_include_boundary_runs() {
        assert_eq!(runs(&[true, true, false, true]), vec![(0, 2), (3, 4)]);
        assert!(runs(&[false, false]).is_empty());
    }

    #[test]
    fn convolution_matches_numpy_even_window_alignment() {
        let mut columns = vec![0; 20];
        columns[10] = 1;
        assert_eq!(runs(&blue_closed(&columns)), vec![(5, 17)]);
        assert_eq!(blue_closed(&[1]), vec![true; 12]);
        assert_eq!(runs(&blue_closed(&[0, 0, 1])), vec![(1, 12)]);
    }

    fn word_state() -> opencv::Result<crate::signals::State> {
        let (w, h) = (160, 60);
        let blank = crate::zeros(h, w)?;
        let full = Mat::new_rows_cols_with_default(h, w, core::CV_8UC1, core::Scalar::all(255.))?;
        let mut stroke = blank.clone();
        for y in 20..32 {
            for (left, right) in [(20, 30), (34, 44), (48, 58)] {
                for x in left..right {
                    *stroke.at_2d_mut::<u8>(y, x)? = 1;
                }
            }
        }
        Ok(crate::signals::State {
            gray: blank.clone(),
            dx: full.clone(),
            dy: blank.clone(),
            dx_gray: blank.clone(),
            dy_gray: blank.clone(),
            grad_max: full.clone(),
            flat_pix: full,
            stroke_01: stroke,
            blue_01: blank.clone(),
            thumb_occ: blank.clone(),
            frame_occ: blank,
            frames: vec![],
            thumbs: vec![],
            tile_containers: vec![],
            panel_corners: vec![],
            coarse_boxes: vec![(18, 18, 42, 16)],
            w,
            h,
            s: 1.,
            scale_back: 1.5,
            signal_ms: 0.,
            frames_ms: 0.,
            coarse_ms: 0.,
        })
    }

    #[test]
    fn words_are_scaled_and_omit_children() -> opencv::Result<()> {
        let state = word_state()?;
        let (boxes, inner) = refine(&state, Options::default())?;
        assert!(inner.is_empty());
        assert_eq!(boxes.len(), 1);
        let target = &boxes[0];
        assert_eq!((target.x, target.y, target.w, target.h), (20, 20, 38, 12));
        assert_eq!(target.kind, "text_link");
        assert_eq!(
            target.children,
            vec![
                json!({"x":30,"y":30,"w":15,"h":18,"score":0.85,"kind":"word","level":"word","hint":"S"}),
                json!({"x":51,"y":30,"w":15,"h":18,"score":0.85,"kind":"word","level":"word","hint":"A"}),
                json!({"x":72,"y":30,"w":15,"h":18,"score":0.85,"kind":"word","level":"word","hint":"D"}),
            ]
        );
        Ok(())
    }

    #[test]
    fn contained_boxes_deduplicate_only_outside_frames() -> opencv::Result<()> {
        let mut state = word_state()?;
        state.coarse_boxes.push(state.coarse_boxes[0]);
        let (boxes, inner) = refine(&state, Options::default())?;
        assert_eq!(boxes.len(), 1);
        assert!(inner.is_empty());
        *state.frame_occ.at_2d_mut::<u8>(26, 39)? = 255;
        let (boxes, inner) = refine(&state, Options::default())?;
        assert!(boxes.is_empty());
        assert_eq!(inner.len(), 2);
        *state.thumb_occ.at_2d_mut::<u8>(26, 39)? = 255;
        let (boxes, inner) = refine(&state, Options::default())?;
        assert!(boxes.is_empty() && inner.is_empty());
        Ok(())
    }
}
