//! v9 direct-ink text analysis: foreground line bands, glyph-chain confidence,
//! and word segmentation.
//!
//! These routines mirror NumPy's float32 pipeline in the reference detector:
//! contrast, span, aspect, and fill statistics stay in `f32`, while Python
//! scalars such as `int(rw * 0.95)` and the thresholds derived from them are
//! evaluated in `f64`. Keeping that split avoids changing clamp and valley
//! decisions that would otherwise differ in the last few bits.
//!
//! `score_and_segment` follows the reference stages in order: adaptive ink
//! threshold, per-column background bounds, local letter-likeness, one-
//! dimensional contextual propagation, line-local bimodal gap clustering, then
//! pitch-guided valley splitting of fused runs.
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
    if h < min_h * 2 + 2 || w < 24 {
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
    let mean_thr = (peak_mean * 0.20).clamp(4.0, 18.0);
    let p95_thr = (peak_p95 * 0.36).clamp(16.0, 52.0);
    let active: Vec<bool> = (0..h as usize)
        .map(|i| {
            !(means[i] as f64 <= mean_thr
                && p95[i] as f64 <= p95_thr
                && strokes[i] <= 4.max((w as f64 * 0.045) as i32))
        })
        .collect();
    let bands: Vec<_> = runs(&active)
        .into_iter()
        .filter(|&(a, b)| {
            b - a >= min_h as usize
                && means[a..b]
                    .iter()
                    .any(|&v| v as f64 >= 12.0_f64.max(peak_mean * 0.20))
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
    if h < 4 || w < 6 {
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
    let trim = if rh >= 7 { 1 } else { 0 };
    let mut col_max = vec![0.0_f32; rw];
    let mut col_mean = vec![0.0_f32; rw];
    let mut fg = vec![vec![0.0_f32; rw]; rh];
    for (yy, fg_row) in fg.iter_mut().enumerate() {
        let row = state.gray.at_row::<u8>(y + yy as i32)?;
        for xx in 0..rw {
            let v = (row[x as usize + xx] as f32 - bg).abs();
            fg_row[xx] = v;
            if yy >= trim && yy < rh - trim {
                col_max[xx] = col_max[xx].max(v);
                col_mean[xx] += v;
            }
        }
    }
    for v in &mut col_mean {
        *v /= (rh - 2 * trim) as f32;
    }
    let peak = percentile(&col_max, ((rw as f64 * 0.85) as usize).min(rw - 1));
    if peak < 11.0 {
        return Ok((0.0, vec![]));
    }
    let ink_thr = (peak as f64 * 0.27).clamp(10.0, 42.0);
    let mut ink: Vec<bool> = col_max.iter().map(|&v| v as f64 > ink_thr).collect();
    if ink.iter().filter(|&&v| v).count() < 3 {
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
        if ink.iter().filter(|&&v| v).count() < 3 {
            return Ok((0.0, vec![]));
        }
    }
    let primitives = runs(&ink);
    let mut top_y = vec![rh as i32; rw];
    let mut bot_y = vec![-1; rw];
    let mut counts = vec![0_i32; rw];
    let mut dx = vec![0.0_f32; rw];
    let mut dy = vec![0.0_f32; rw];
    for (yy, fg_row) in fg.iter().enumerate() {
        let dx_row = state.dx.at_row::<u8>(y + yy as i32)?;
        let dy_row = state.dy.at_row::<u8>(y + yy as i32)?;
        for xx in 0..rw {
            if fg_row[xx] as f64 > ink_thr * 0.85 {
                top_y[xx] = top_y[xx].min(yy as i32);
                bot_y[xx] = yy as i32;
                counts[xx] += 1;
            }
            // NumPy means over the uint8 gradient columns in float32.
            dx[xx] += dx_row[x as usize + xx] as f32;
            dy[xx] += dy_row[x as usize + xx] as f32;
        }
    }
    for xx in 0..rw {
        dx[xx] /= rh as f32;
        dy[xx] /= rh as f32;
    }
    let mut osc = vec![0_i32; rw + 1];
    let mut fg_prefix = vec![0_i32; rw + 1];
    let mut dx_prefix = vec![0.0_f64; rw + 1];
    let mut dy_prefix = vec![0.0_f64; rw + 1];
    // NumPy cumsum stays float32 before `concatenate` upcasts it to float64.
    let (mut dx_sum, mut dy_sum) = (0.0_f32, 0.0_f32);
    for xx in 0..rw {
        osc[xx + 1] = osc[xx]
            + i32::from(
                xx > 0
                    && xx + 1 < rw
                    && dx[xx] >= dx[xx - 1]
                    && dx[xx] >= dx[xx + 1]
                    && dx[xx] > 4.0,
            );
        fg_prefix[xx + 1] = fg_prefix[xx] + counts[xx];
        dx_sum += dx[xx];
        dy_sum += dy[xx];
        dx_prefix[xx + 1] = dx_sum as f64;
        dy_prefix[xx + 1] = dy_sum as f64;
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
        } else if ar > 1.25 && ar <= 7.5 {
            if osc[b] - osc[a] >= 1.max((ar * 0.45) as i32) {
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
        let stroke = if dx_m >= 2.2 && dx_m >= 0.52 * dy_m {
            1.0
        } else if dx_m >= 1.2 {
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
        let max_dy = 2.5_f64.max(0.28 * rh as f64);
        let max_dh = 3.5_f64.max(0.45 * rh as f64);
        let max_gap1 = 18.max((2.4 * rh as f64) as i32);
        let max_gap2 = 22.max((2.8 * rh as f64) as i32);
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
                let compat = if dy_mid <= max_dy && dh <= max_dh && gap <= max_gap1 {
                    1.0
                } else if (mids[j] - med).abs() as f64 <= max_dy && gap <= max_gap2 {
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
    if ink_width < 4.max((rw as f64 * 0.18) as usize) {
        confidence *= 0.65;
    }
    let mut words = Vec::new();
    if valid.len() >= 2 {
        let gaps: Vec<i32> = valid
            .windows(2)
            .map(|p| p[1].0 as i32 - p[0].1 as i32)
            .collect();
        let std_thr = (if h <= 13.max((0.95 * state.l0) as i32) {
            2
        } else {
            3
        })
        .max((h as f64 * 0.15).round_ties_even() as i32);
        let (g_min, g_max) = (*gaps.iter().min().unwrap(), *gaps.iter().max().unwrap());
        let mut threshold = std_thr;
        if g_min <= 1 && g_max >= 2 {
            if g_max <= 3 {
                threshold = 2;
            } else {
                let mut best = -1.0_f64;
                for candidate in g_min + 1..(g_max + 1).min(5.max(std_thr + 2)) {
                    let lo: Vec<_> = gaps.iter().copied().filter(|&v| v < candidate).collect();
                    let hi: Vec<_> = gaps.iter().copied().filter(|&v| v >= candidate).collect();
                    if lo.is_empty() || hi.is_empty() {
                        continue;
                    }
                    // NumPy means the float32 gap array, then promotes to float64.
                    let mu0 = (lo.iter().sum::<i32>() as f32 / lo.len() as f32) as f64;
                    let mu1 = (hi.iter().sum::<i32>() as f32 / hi.len() as f32) as f64;
                    if mu1 - mu0 >= 1.15 {
                        let w0 = lo.len() as f64 / gaps.len() as f64;
                        let variance = w0 * (1.0 - w0) * (mu1 - mu0).powi(2);
                        if variance > best {
                            best = variance;
                            threshold = candidate;
                        }
                    }
                }
                threshold = threshold.min(std_thr);
            }
        }
        let mut start = 0;
        for (i, &gap) in gaps.iter().enumerate() {
            if gap >= threshold {
                if valid[i].1 - valid[start].0 >= 3 {
                    words.push((valid[start].0, valid[i].1));
                }
                start = i + 1;
            }
        }
        if valid.last().unwrap().1 - valid[start].0 >= 3 {
            words.push((valid[start].0, valid.last().unwrap().1));
        }
    } else if valid[0].1 - valid[0].0 >= 3 {
        words.push((valid[0].0, valid[0].1));
    }
    let mut refined = Vec::new();
    let min_step = 7.max((h as f64 * 1.10) as usize);
    for (a, b) in words {
        let ww = b - a;
        if ww > 22.max((h as f64 * 4.4) as usize) && h <= 28.max((1.9 * state.l0) as i32) {
            let p80 = percentile(&col_mean[a..b], ((ww as f64 * 0.80) as usize).min(ww - 1)) as f64;
            let mut cuts: Vec<usize> = Vec::new();
            for c in min_step..ww.saturating_sub(min_step) {
                if col_mean[a + c] <= col_mean[a + c - 1]
                    && col_mean[a + c] <= col_mean[a + c + 1]
                    && (col_mean[a + c] as f64) <= p80 * 0.38
                    && (col_max[a + c] as f64) <= peak as f64 * 0.56
                {
                    if cuts.last().is_none_or(|&last| c - last >= min_step) {
                        cuts.push(c);
                    } else if col_mean[a + c] < col_mean[a + *cuts.last().unwrap()] {
                        *cuts.last_mut().unwrap() = c;
                    }
                }
            }
            if !cuts.is_empty() {
                let mut pts = vec![0];
                pts.extend(cuts);
                pts.push(ww);
                for pair in pts.windows(2) {
                    let (mut s0, mut s1) = (a + pair[0], a + pair[1]);
                    if let Some(first) = (s0..s1).find(|&i| col_max[i] as f64 > ink_thr * 0.80) {
                        let last = (s0..s1)
                            .rfind(|&i| col_max[i] as f64 > ink_thr * 0.80)
                            .unwrap();
                        s0 = first;
                        s1 = last + 1;
                    }
                    if s1 - s0 >= 3 {
                        refined.push((s0 as i32, s1 as i32));
                    }
                }
                continue;
            }
        }
        refined.push((a as i32, b as i32));
    }
    Ok((confidence, refined))
}
