//! v9 direct-ink text analysis: foreground line bands, glyph-chain confidence,
//! and word segmentation.
//!
//! These routines mirror NumPy's float32 pipeline in the reference detector:
//! contrast, span, aspect, and fill statistics stay in `f32`, while column
//! gradient means and Python scalars such as `int(rw * 0.95)` are evaluated in
//! `f64`. Keeping that split avoids changing clamp and valley decisions that
//! would otherwise differ in the last few bits.
//!
//! `score_and_segment` follows the reference stages in order: adaptive ink
//! threshold, per-column background bounds, local letter-likeness, one-
//! dimensional contextual propagation, relative bimodal gap separation, then
//! valley refinement of fused single primitives and badge-clearance merging.
use crate::signals::State;
use opencv::prelude::*;

fn runs(mask: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, &on) in mask.iter().chain(std::iter::once(&false)).enumerate() {
        if on && start.is_none() {
            start = Some(i);
        }
        if !on && let Some(s) = start.take() {
            out.push((s, i));
        }
    }
    out
}

fn median_f32(v: &[f32]) -> f32 {
    let mut sorted = v.to_vec();
    sorted.sort_by(f32::total_cmp);
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5
    } else {
        sorted[n / 2]
    }
}

fn percentile(v: &[f32], index: usize) -> f32 {
    let mut sorted = v.to_vec();
    *sorted.select_nth_unstable_by(index, f32::total_cmp).1
}

pub fn foreground_line_bands(
    state: &State,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    min_h: i32,
) -> opencv::Result<Vec<(i32, i32)>> {
    if h < min_h * 2 + 2 || w < h * 2 {
        return Ok(vec![]);
    }
    let width = w as usize;
    let top = &state.gray.at_row::<u8>(y)?[x as usize..(x + w) as usize];
    let bottom = &state.gray.at_row::<u8>(y + h - 1)?[x as usize..(x + w) as usize];
    // NumPy selects the upper median, not the mean of the two middle samples.
    let bg = (percentile(
        &top.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        width / 2,
    ) + percentile(
        &bottom.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        width / 2,
    )) * 0.5;
    let mut means = Vec::new();
    let mut p95 = Vec::new();
    let mut strokes = Vec::new();
    let mut left_min = f32::MAX;
    for yy in y..y + h {
        let gray = &state.gray.at_row::<u8>(yy)?[x as usize..(x + w) as usize];
        let stroke = &state.stroke_01.at_row::<u8>(yy)?[x as usize..(x + w) as usize];
        let fg: Vec<f32> = gray.iter().map(|&v| (v as f32 - bg).abs()).collect();
        left_min = left_min
            .min(fg[..width.min(h as usize)].iter().sum::<f32>() / width.min(h as usize) as f32);
        means.push(fg.iter().sum::<f32>() / width as f32);
        // `int(rw * 0.95)` uses Python's double-precision scalar arithmetic.
        p95.push(percentile(&fg, (width as f64 * 0.95) as usize));
        strokes.push(stroke.iter().map(|&v| v as i32).sum::<i32>());
    }
    if w <= h * 9 && left_min >= 12.0 {
        return Ok(vec![]);
    }
    let peak_mean = means.iter().copied().fold(0.0_f32, f32::max) as f64;
    let peak_p95 = p95.iter().copied().fold(0.0_f32, f32::max) as f64;
    if peak_mean < 12.0 || peak_p95 < 28.0 {
        return Ok(vec![]);
    }
    let mean_thr = (peak_mean * 0.12).max((peak_mean * 0.24).min(18.0));
    let p95_thr = (peak_p95 * 0.18).max((peak_p95 * 0.36).min(52.0));
    let stroke_peak = *strokes.iter().max().unwrap();
    let active: Vec<bool> = (0..h as usize)
        .map(|i| {
            !(means[i] as f64 <= mean_thr
                && p95[i] as f64 <= p95_thr
                && (strokes[i]
                    <= 2.max((w as f64 * 0.08) as i32)
                        .max((stroke_peak as f64 * 0.16) as i32)
                    || (means[i] as f64 <= peak_mean * 0.10 && p95[i] as f64 <= peak_p95 * 0.15)))
        })
        .collect();
    let bands: Vec<_> = runs(&active)
        .into_iter()
        .filter(|&(a, b)| {
            b - a >= min_h as usize && means[a..b].iter().any(|&v| v as f64 >= peak_mean * 0.20)
        })
        .map(|(a, b)| (a as i32, b as i32))
        .collect();
    Ok(if bands.len() >= 2 { bands } else { vec![] })
}

pub fn score_and_segment(
    state: &State,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
) -> opencv::Result<(f32, Vec<(i32, i32)>)> {
    if h < 4 || w < ((h as f64 * 1.2) as i32) {
        return Ok((0.0, vec![]));
    }
    let (rw, rh) = (w as usize, h as usize);
    let top = &state.gray.at_row::<u8>(y)?[x as usize..(x + w) as usize];
    let bot = &state.gray.at_row::<u8>(y + h - 1)?[x as usize..(x + w) as usize];
    let bg = if h >= 6 {
        (percentile(&top.iter().map(|&v| v as f32).collect::<Vec<_>>(), rw / 2)
            + percentile(&bot.iter().map(|&v| v as f32).collect::<Vec<_>>(), rw / 2))
            * 0.5
    } else {
        let mut all = Vec::with_capacity(rw * rh);
        for yy in y..y + h {
            all.extend_from_slice(&state.gray.at_row::<u8>(yy)?[x as usize..(x + w) as usize]);
        }
        percentile(
            &all.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            all.len() / 2,
        )
    };
    let trim = if rh >= 7 {
        ((rh as f64 * 0.12).round_ties_even() as usize).max(1)
    } else {
        0
    };
    let use_core = rh >= 2 * trim + 3;
    let core_rows = if use_core { rh - 2 * trim } else { rh };
    let mut col_max = vec![0.0_f32; rw];
    let mut col_mean = vec![0.0_f32; rw];
    let mut fg = vec![vec![0.0_f32; rw]; rh];
    for (yy, fg_row) in fg.iter_mut().enumerate() {
        let row = state.gray.at_row::<u8>(y + yy as i32)?;
        for xx in 0..rw {
            let v = (row[x as usize + xx] as f32 - bg).abs();
            fg_row[xx] = v;
            if !use_core || (yy >= trim && yy < rh - trim) {
                col_max[xx] = col_max[xx].max(v);
                col_mean[xx] += v;
            }
        }
    }
    for v in &mut col_mean {
        *v /= core_rows as f32;
    }
    let peak = if rw >= 6 {
        percentile(&col_max, ((rw as f64 * 0.85) as usize).min(rw - 1))
    } else {
        col_max.iter().copied().fold(0.0_f32, f32::max)
    };
    if peak < 11.0 {
        return Ok((0.0, vec![]));
    }
    let ink_thr = (peak as f64 * 0.27).clamp(peak as f64 * 0.18, peak as f64 * 0.45);
    let mut ink: Vec<bool> = col_max.iter().map(|&v| v as f64 > ink_thr).collect();
    let min_ink_cols = 3.max((rh as f64 * 0.4) as usize);
    if ink.iter().filter(|&&v| v).count() < min_ink_cols {
        ink = (0..rw)
            .map(|xx| {
                (0..rh).any(|yy| {
                    state
                        .stroke_01
                        .at_row::<u8>(y + yy as i32)
                        .map(|r| r[x as usize + xx] > 0)
                        .unwrap_or(false)
                })
            })
            .collect();
        if ink.iter().filter(|&&v| v).count() < min_ink_cols {
            return Ok((0.0, vec![]));
        }
    }
    let primitives = runs(&ink);
    let mut top_y = vec![rh as i32; rw];
    let mut bot_y = vec![-1; rw];
    let mut counts = vec![0_i32; rw];
    let mut dx = vec![0.0_f64; rw];
    let mut dy = vec![0.0_f64; rw];
    for (yy, fg_row) in fg.iter().enumerate() {
        let dx_row = state.dx.at_row::<u8>(y + yy as i32)?;
        let dy_row = state.dy.at_row::<u8>(y + yy as i32)?;
        for xx in 0..rw {
            if fg_row[xx] as f64 > ink_thr * 0.85 {
                top_y[xx] = top_y[xx].min(yy as i32);
                bot_y[xx] = yy as i32;
                counts[xx] += 1;
            }
            dx[xx] += dx_row[x as usize + xx] as f64;
            dy[xx] += dy_row[x as usize + xx] as f64;
        }
    }
    for xx in 0..rw {
        dx[xx] /= rh as f64;
        dy[xx] /= rh as f64;
    }
    let dx_peak = dx.iter().copied().fold(0.0_f64, f64::max);
    let dx_peak_thr = 2.5_f64.max(dx_peak * 0.20);
    let mut osc = vec![0_i32; rw + 1];
    let mut fg_prefix = vec![0_i32; rw + 1];
    let mut dx_prefix = vec![0.0_f64; rw + 1];
    let mut dy_prefix = vec![0.0_f64; rw + 1];
    for xx in 0..rw {
        osc[xx + 1] = osc[xx]
            + i32::from(
                xx > 0
                    && xx + 1 < rw
                    && dx[xx] >= dx[xx - 1]
                    && dx[xx] >= dx[xx + 1]
                    && dx[xx] > dx_peak_thr,
            );
        fg_prefix[xx + 1] = fg_prefix[xx] + counts[xx];
        dx_prefix[xx + 1] = dx_prefix[xx] + dx[xx];
        dy_prefix[xx + 1] = dy_prefix[xx] + dy[xx];
    }
    let mut local = Vec::new();
    let mut mids = Vec::new();
    let mut spans = Vec::new();
    for &(a, b) in &primitives {
        let yh0 = *top_y[a..b].iter().min().unwrap();
        let yh1 = *bot_y[a..b].iter().max().unwrap() + 1;
        let ph = (yh1 - yh0).max(1) as f32;
        mids.push((yh0 + yh1) as f32 * 0.5);
        spans.push(ph);
        let hr = ph / rh as f32;
        let height = if (0.42..=1.0).contains(&hr) {
            1.0
        } else if hr >= 0.28 {
            0.55
        } else {
            0.15
        };
        let ar = (b - a) as f32 / rh as f32;
        let aspect = if (0.15..=1.25).contains(&ar) {
            1.0
        } else if ar > 1.25 && ar <= 8.5 {
            if osc[b] - osc[a] >= 1.max((ar * 0.42) as i32) {
                0.92
            } else {
                0.55
            }
        } else {
            0.25
        };
        let fill = (fg_prefix[b] - fg_prefix[a]) as f32 / (ph * (b - a) as f32).max(1.0);
        let fill_score = if (0.16..=0.82).contains(&fill) {
            1.0
        } else if (0.10..=0.90).contains(&fill) {
            0.50
        } else {
            0.15
        };
        let dx_m = (dx_prefix[b] - dx_prefix[a]) / (b - a) as f64;
        let dy_m = (dy_prefix[b] - dy_prefix[a]) / (b - a) as f64;
        let stroke = if dx_m >= 0.52 * dy_m {
            1.0
        } else if dx_m >= 0.35 * dy_m {
            0.60
        } else {
            0.20
        };
        local.push(if yh1 > yh0 {
            (0.30 * height + 0.28 * aspect + 0.22 * fill_score + 0.20 * stroke) as f32
        } else {
            0.0
        });
    }
    let mut contextual = local.clone();
    if primitives.len() >= 2 {
        let med = median_f32(&mids);
        // Python compares float32 spans to double-precision scalar cutoffs.
        let max_dy = 0.28 * rh as f64;
        let max_dh = 0.45 * rh as f64;
        let max_gap1 = 2.4 * rh as f64;
        let max_gap2 = 2.8 * rh as f64;
        for i in 0..primitives.len() {
            let (mut num, mut den) = (0.0_f32, 0.0_f32);
            for (step, weight) in [(-2, 0.16_f32), (-1, 0.34), (1, 0.34), (2, 0.16)] {
                let j = i as i32 + step;
                if j < 0 || j >= primitives.len() as i32 {
                    continue;
                }
                let j = j as usize;
                let gap = if j < i {
                    primitives[i].0 as i32 - primitives[j].1 as i32
                } else {
                    primitives[j].0 as i32 - primitives[i].1 as i32
                };
                let dy_mid = (mids[i] - mids[j]).abs() as f64;
                let dh = (spans[i] - spans[j]).abs() as f64;
                let compat = if dy_mid <= max_dy && dh <= max_dh && (gap as f64) <= max_gap1 {
                    1.0
                } else if (mids[j] - med).abs() as f64 <= max_dy && (gap as f64) <= max_gap2 {
                    0.82
                } else {
                    0.20
                };
                num += weight * local[j] * compat;
                den += weight;
            }
            if den > 0.0 {
                contextual[i] = (0.58 * local[i] + 0.56 * (num / den)).min(1.0);
            }
        }
    }
    let valid: Vec<_> = primitives
        .iter()
        .zip(&contextual)
        .filter_map(|(&(a, b), &score)| (score >= 0.48).then_some((a, b, score)))
        .collect();
    if valid.is_empty() {
        return Ok((0.0, vec![]));
    }
    let ink_width: usize = valid.iter().map(|&(a, b, _)| b - a).sum();
    let bonus = if valid.len() >= 3 || osc[rw] >= 4 {
        0.14
    } else if valid.len() == 2 {
        0.07
    } else {
        0.0
    };
    let mut confidence = (valid.iter().map(|&(_, _, v)| v).sum::<f32>() / valid.len() as f32
        + bonus)
        .clamp(0.0, 1.0);
    if ink_width < (rw as f64 * 0.18) as usize {
        confidence *= 0.65;
    }

    // Step C: relative bimodal gap separation. Monospace strips expose every
    // character as an isolated primitive, so a space must exceed both the
    // baseline kerning and a sizeable fraction of a character cell. Proportional
    // strips fuse letters into wider blobs and can use a lower relative cutoff.
    let v_widths: Vec<f32> = valid.iter().map(|&(a, b, _)| (b - a) as f32).collect();
    let gaps: Vec<i32> = valid
        .windows(2)
        .map(|p| p[1].0 as i32 - p[0].1 as i32)
        .collect();
    let min_tok_w = 2.max((rh as f64 * 0.22).round_ties_even() as usize);
    let mut raw_words: Vec<(usize, usize, usize)> = Vec::new();
    if valid.len() >= 2 {
        let med_pw = median_f32(&v_widths) as f64;
        let gaps_f32: Vec<f32> = gaps.iter().map(|&g| g as f32).collect();
        let g_max = *gaps.iter().max().unwrap() as f64;
        let k85_w = ((v_widths.len() as f64 * 0.85) as usize).min(v_widths.len() - 1);
        let p85_pw = percentile(&v_widths, k85_w) as f64;
        let k65_g = ((gaps_f32.len() as f64 * 0.65) as usize).min(gaps_f32.len() - 1);
        let kern_base = percentile(&gaps_f32, k65_g) as f64;
        let icon_pair = valid.len() == 2
            && v_widths
                .iter()
                .all(|&width| (0.68..=1.38).contains(&(width as f64 / rh.max(1) as f64)));
        let is_monospace = p85_pw <= 0.56 * rh as f64
            && p85_pw <= 1.45 * med_pw
            && !(g_max >= 4.max((rh as f64 * 0.45).round_ties_even() as i32) as f64
                && kern_base <= 2.0
                && g_max >= 3.0 * kern_base.max(1.0))
            && !(icon_pair && gaps[0] >= 3.max((rh as f64 * 0.22).round_ties_even() as i32));
        let kern_prop = percentile(&gaps_f32, gaps.len() >> 1) as f64;
        let min_space_thr = if is_monospace {
            ((rh as f64 * 0.36).round_ties_even() as i32)
                .max((kern_base * 2.2).round_ties_even() as i32 + 1)
                .max((med_pw * 0.85).round_ties_even() as i32)
        } else {
            let base = 3.max((rh as f64 * 0.22).round_ties_even() as i32);
            if icon_pair {
                base
            } else {
                base.max((kern_prop * 1.9).round_ties_even() as i32 + 1)
            }
        };
        if g_max < min_space_thr as f64 {
            raw_words.push((valid[0].0, valid.last().unwrap().1, valid.len()));
        } else {
            let mut boundaries = vec![0];
            boundaries.extend(
                gaps.iter()
                    .enumerate()
                    .filter_map(|(i, &gap)| (gap >= min_space_thr).then_some(i + 1)),
            );
            boundaries.push(valid.len());
            for pair in boundaries.windows(2) {
                let (start, end) = (pair[0], pair[1]);
                if valid[end - 1].1 - valid[start].0 < min_tok_w {
                    continue;
                }
                if !is_monospace
                    && (valid[end - 1].1 - valid[start].0) as i32
                        > (rh as f64 * 5.0).round_ties_even() as i32
                    && end - start > 1
                {
                    let sub_thr = 4
                        .max((rh as f64 * 0.18).round_ties_even() as i32)
                        .max((kern_prop * 2.0).round_ties_even() as i32 + 1);
                    let mut sub_start = start;
                    let mut split = false;
                    for i in start..end - 1 {
                        if gaps[i] >= sub_thr {
                            if valid[i].1 - valid[sub_start].0 >= min_tok_w {
                                raw_words.push((valid[sub_start].0, valid[i].1, i - sub_start + 1));
                            }
                            sub_start = i + 1;
                            split = true;
                        }
                    }
                    if split {
                        if valid[end - 1].1 - valid[sub_start].0 >= min_tok_w {
                            raw_words.push((valid[sub_start].0, valid[end - 1].1, end - sub_start));
                        }
                        continue;
                    }
                }
                raw_words.push((valid[start].0, valid[end - 1].1, end - start));
            }
        }
    } else if valid[0].1 - valid[0].0 >= min_tok_w {
        raw_words.push((valid[0].0, valid[0].1, 1));
    }

    // Step D: valley refinement only applies to one continuous primitive, never
    // to a token already assembled from separated monospace characters.
    let max_single_prim_w = (rh as f64 * 4.2).round_ties_even() as i64;
    let min_word_step = ((rh as f64 * 1.25).round_ties_even() as usize)
        .max((state.l0 * 0.65).round_ties_even() as usize);
    let max_rh_subsplit = (state.l0 * 1.85).round_ties_even() as i32;
    let min_refined_w = 2.max((rh as f64 * 0.25).round_ties_even() as usize);
    let mut refined: Vec<(i32, i32)> = Vec::new();
    for (wx0, wx1, n_sub) in raw_words {
        let ww = wx1 - wx0;
        if n_sub == 1 && ww as i64 > max_single_prim_w && rh as i32 <= max_rh_subsplit {
            let p80 = percentile(&col_mean[wx0..wx1], (ww as f64 * 0.80) as usize) as f64;
            let mut cuts: Vec<usize> = Vec::new();
            for c in min_word_step..ww.saturating_sub(min_word_step) {
                if col_mean[wx0 + c] <= col_mean[wx0 + c - 1]
                    && col_mean[wx0 + c] <= col_mean[wx0 + c + 1]
                    && (col_mean[wx0 + c] as f64) <= p80 * 0.34
                    && (col_max[wx0 + c] as f64) <= peak as f64 * 0.52
                {
                    if cuts.last().is_none_or(|&last| c - last >= min_word_step) {
                        cuts.push(c);
                    } else if col_mean[wx0 + c] < col_mean[wx0 + *cuts.last().unwrap()] {
                        *cuts.last_mut().unwrap() = c;
                    }
                }
            }
            if !cuts.is_empty() {
                let mut pts = vec![0];
                pts.extend(cuts);
                pts.push(ww);
                for pair in pts.windows(2) {
                    let (mut s0, mut s1) = (wx0 + pair[0], wx0 + pair[1]);
                    if let Some(first) = (s0..s1).find(|&i| col_max[i] as f64 > ink_thr * 0.80) {
                        let last = (s0..s1)
                            .rfind(|&i| col_max[i] as f64 > ink_thr * 0.80)
                            .unwrap();
                        s0 = first;
                        s1 = last + 1;
                    }
                    if s1 - s0 >= min_refined_w {
                        refined.push((s0 as i32, s1 as i32));
                    }
                }
                continue;
            }
        }
        refined.push((wx0 as i32, wx1 as i32));
    }

    // Step E: relative badge-clearance merge. Tiny punctuation fragments attach
    // to a close neighbour so their badge rectangles cannot overlap.
    if refined.len() >= 2 {
        let min_standalone_w = (rh as f64 * 0.95).round_ties_even() as i32;
        let max_attach_gap = (rh as f64 * 0.75).round_ties_even() as i32;
        let mut merged: Vec<(i32, i32)> = vec![refined[0]];
        for &(wx0, wx1) in &refined[1..] {
            let (px0, px1) = *merged.last().unwrap();
            let (pw, cw) = (px1 - px0, wx1 - wx0);
            let gap = wx0 - px1;
            let icon_width =
                |width: i32| (0.68..=1.38).contains(&(width as f64 / rh.max(1) as f64));
            let icon_sep = (gap >= 4.max((rh as f64 * 0.45).round_ties_even() as i32)
                && (icon_width(pw) || icon_width(cw)))
                || (gap >= 3.max((rh as f64 * 0.22).round_ties_even() as i32)
                    && icon_width(pw)
                    && icon_width(cw));
            if !icon_sep
                && gap <= max_attach_gap
                && (pw < min_standalone_w || cw < min_standalone_w)
            {
                *merged.last_mut().unwrap() = (px0, wx1);
            } else {
                merged.push((wx0, wx1));
            }
        }
        refined = merged;
    }

    Ok((confidence, refined))
}
