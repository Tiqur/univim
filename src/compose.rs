//! Compose work-coordinate proposals into controls and hierarchical text targets.
//! Pairing is greedy and order-sensitive: horizontal compounds precede vertical
//! ones, and containment can append children even after a compound was formed.
use crate::{Options, Target, child, hints, median, overlap};
use opencv::prelude::*;

pub fn compose(
    state: &crate::signals::State,
    mut refined_boxes: Vec<Target>,
    inner_frame_children: Vec<Target>,
    options: Options,
) -> opencv::Result<Vec<Target>> {
    let mut frames = state.frames.clone();
    let mut thumbs = state.thumbs.clone();
    let s = state.s;
    let scale_back = state.scale_back;
    let (w, h) = (state.w, state.h);
    let scaled = |v: f64| (v * s) as i32;
    let make_child =
        |t: &Target, kind: &str, level: &str, hint: &str| child(t, scale_back, kind, level, hint);
    let union = |a: &mut Target, b: &Target| {
        let (x, y) = (a.x.min(b.x), a.y.min(b.y));
        a.w = a.right().max(b.right()) - x;
        a.h = a.bottom().max(b.bottom()) - y;
        a.x = x;
        a.y = y;
    };
    // Match NumPy slice endpoints, including negative indices and empty slices.
    let slice_end = |v: i32, len: i32| {
        if v < 0 { (len + v).max(0) } else { v.min(len) }
    };
    let barrier = |x1: i32, y1: i32, x2: i32, y2: i32, vertical: bool| -> opencv::Result<bool> {
        let x1 = slice_end(x1, state.dx_gray.cols());
        let x2 = slice_end(x2, state.dx_gray.cols());
        let y1 = slice_end(y1, state.dx_gray.rows());
        let y2 = slice_end(y2, state.dx_gray.rows());
        if x2 <= x1 || y2 <= y1 {
            return Ok(false);
        }
        if vertical {
            for x in x1..x2 {
                let mut count = 0;
                for y in y1..y2 {
                    if *state.dx_gray.at_2d::<u8>(y, x)? >= 4
                        && *state.dy_gray.at_2d::<u8>(y, x)? <= 2
                    {
                        count += 1;
                    }
                }
                if count as f64 / (y2 - y1) as f64 >= 0.55 {
                    return Ok(true);
                }
            }
        } else {
            for y in y1..y2 {
                let mut count = 0;
                for x in x1..x2 {
                    if *state.dy_gray.at_2d::<u8>(y, x)? >= 4
                        && *state.dx_gray.at_2d::<u8>(y, x)? <= 2
                    {
                        count += 1;
                    }
                }
                if count as f64 / (x2 - x1) as f64 >= 0.50 {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    };

    let mut used_rb = vec![false; refined_boxes.len()];
    let mut used_fr = vec![false; frames.len()];
    let mut desktop_icons = Vec::new();

    // 5A: frames first, then refined icons, each paired with the first eligible right label.
    let max_h_icon = 62.max(scaled(68.));
    for from_frame in [true, false] {
        let len = if from_frame {
            frames.len()
        } else {
            refined_boxes.len()
        };
        for idx in 0..len {
            if if from_frame {
                used_fr[idx]
            } else {
                used_rb[idx]
            } {
                continue;
            }
            let icon = if from_frame {
                &frames[idx]
            } else {
                &refined_boxes[idx]
            };
            let aspect = icon.w as f64 / icon.h as f64;
            let square = matches!(icon.kind.as_str(), "frame" | "icon" | "inline_link")
                || (icon.kind == "text_link"
                    && (12..=max_h_icon).contains(&icon.w)
                    && (12..=max_h_icon).contains(&icon.h)
                    && (0.50..=1.65).contains(&aspect));
            if !square
                || !(12..=max_h_icon).contains(&icon.w)
                || !(12..=max_h_icon).contains(&icon.h)
                || !(0.48..=1.75).contains(&aspect)
            {
                continue;
            }
            let mut best = None;
            for (rbi, rb) in refined_boxes.iter().enumerate() {
                if used_rb[rbi] || (!from_frame && idx == rbi) {
                    continue;
                }
                if matches!(rb.kind.as_str(), "text_link" | "inline_link" | "text_block")
                    && rb.w >= 16.max(scaled(16.))
                    && rb.w <= scaled(340.)
                    && rb.h <= 32.max(scaled(38.))
                    && rb.w as f64 > rb.h as f64 * 1.10
                    && (2..=40.max(scaled(48.))).contains(&(rb.x - icon.right()))
                    && rb.y >= icon.y - 8.max(scaled(10.))
                    && rb.bottom() <= icon.bottom() + 10.max(scaled(12.))
                {
                    let (gx1, gx2) = (icon.right() + 1, rb.x - 1);
                    if gx2 > gx1
                        && barrier(
                            gx1,
                            icon.y.min(rb.y),
                            gx2,
                            icon.bottom().max(rb.bottom()),
                            true,
                        )?
                    {
                        continue;
                    }
                    best = Some(rbi);
                    break;
                }
            }
            if let Some(rbi) = best {
                let label = &refined_boxes[rbi];
                if from_frame {
                    used_fr[idx] = true
                } else {
                    used_rb[idx] = true
                }
                used_rb[rbi] = true;
                let mut compound = Target::new(
                    icon.x,
                    icon.y,
                    icon.w,
                    icon.h,
                    0.95,
                    "desktop_icon",
                    "control",
                );
                union(&mut compound, label);
                let mut icon_child = icon.clone();
                icon_child.children.clear();
                compound.children = vec![
                    make_child(&icon_child, "icon", "control", "S"),
                    make_child(label, &label.kind, "line", "A"),
                ];
                for (i, subtitle) in refined_boxes.iter().enumerate() {
                    if used_rb[i] {
                        continue;
                    }
                    if matches!(
                        subtitle.kind.as_str(),
                        "text_block" | "text_link" | "inline_link"
                    ) && subtitle.w <= scaled(260.)
                        && subtitle.h <= scaled(24.)
                        && (subtitle.x - label.x).abs() <= scaled(12.)
                        && (0..=scaled(16.)).contains(&(subtitle.y - label.bottom()))
                        && subtitle.bottom() <= icon.bottom() + scaled(22.)
                    {
                        used_rb[i] = true;
                        union(&mut compound, subtitle);
                        compound
                            .children
                            .push(make_child(subtitle, &subtitle.kind, "line", "D"));
                    }
                }
                desktop_icons.push(compound);
            }
        }
    }

    // Frames that already form a horizontal rail are columns, not icon cards;
    // skip them during vertical caption pairing.
    let mut horiz_rail_frames = vec![false; frames.len()];
    for fi in 0..frames.len() {
        let peers = (0..frames.len())
            .filter(|&fj| {
                fi != fj
                    && (frames[fi].y - frames[fj].y).abs() <= scaled(6.)
                    && (frames[fi].h - frames[fj].h).abs() <= scaled(6.)
            })
            .count();
        if peers >= 2 {
            horiz_rail_frames[fi] = true;
        }
    }

    // 5B: vertical captions and enclosing tile expansion.
    for from_frame in [true, false] {
        let len = if from_frame {
            frames.len()
        } else {
            refined_boxes.len()
        };
        for idx in 0..len {
            if if from_frame {
                used_fr[idx]
            } else {
                used_rb[idx]
            } {
                continue;
            }
            if from_frame && horiz_rail_frames[idx] {
                continue;
            }
            let icon = if from_frame {
                &frames[idx]
            } else {
                &refined_boxes[idx]
            };
            if !(16..=64.max(scaled(82.))).contains(&icon.w)
                || !(20..=64.max(scaled(76.))).contains(&icon.h)
                || !(0.55..=1.55).contains(&(icon.w as f64 / icon.h as f64))
            {
                continue;
            }
            let caption_fits = |label: &Target| {
                icon.h >= (label.h as f64 * 1.22) as i32
                    && label.w >= 28.max((label.h as f64 * 1.45) as i32)
                    && label.w <= 110.max(scaled(145.))
                    && label.h <= 24.max(scaled(36.))
                    && (0..=18.max(scaled(20.))).contains(&(label.y - icon.bottom()))
                    && ((icon.x as f64 + icon.w as f64 * 0.5)
                        - (label.x as f64 + label.w as f64 * 0.5))
                        .abs()
                        <= 26.max(scaled(28.)) as f64
            };
            let mut best = None;
            // Python updates rsc for every examined, non-skipped refined box.
            let mut rsc = 0.88;
            for (rbi, rb) in refined_boxes.iter().enumerate() {
                if used_rb[rbi] || (!from_frame && idx == rbi) {
                    continue;
                }
                rsc = rb.score;
                if matches!(rb.kind.as_str(), "text_link" | "inline_link" | "text_block")
                    && caption_fits(rb)
                {
                    let (gx1, gx2) = (icon.x.min(rb.x), icon.right().max(rb.right()));
                    if rb.y > icon.bottom()
                        && gx2 > gx1
                        && barrier(gx1, icon.bottom(), gx2, rb.y, false)?
                    {
                        continue;
                    }
                    best = Some(rbi);
                    break;
                }
            }
            if let Some(li) = best {
                let label = &refined_boxes[li];
                used_rb[li] = true;
                if from_frame {
                    used_fr[idx] = true
                } else {
                    used_rb[idx] = true
                }
                let mut compound = Target::new(
                    icon.x,
                    icon.y,
                    icon.w,
                    icon.h,
                    0.95,
                    "desktop_icon",
                    "control",
                );
                union(&mut compound, label);
                let mut icon_child = icon.clone();
                icon_child.children.clear();
                let mut label_child = label.clone();
                label_child.score = rsc;
                compound.children = vec![
                    make_child(&icon_child, "icon", "control", "S"),
                    make_child(&label_child, "text_link", "line", "A"),
                ];
                for &(tx, ty, tw, th) in &state.tile_containers {
                    if compound.x >= tx - 4
                        && compound.y >= ty - 4
                        && compound.right() <= tx + tw + 4
                        && compound.bottom() <= ty + th + 4
                    {
                        compound.x = tx;
                        compound.y = ty;
                        compound.w = tw;
                        compound.h = th;
                        break;
                    }
                }
                desktop_icons.push(compound);
            }
        }
    }

    let mut rem_rbs: Vec<Target> = refined_boxes
        .drain(..)
        .enumerate()
        .filter_map(|(i, rb)| (!used_rb[i]).then_some(rb))
        .collect();
    let mut final_frames = Vec::new();
    for (i, frame) in frames.drain(..).enumerate() {
        if used_fr[i] {
            continue;
        }
        if frame.w <= 32
            && frame.h <= 18
            && let Some(rb) = rem_rbs.iter_mut().find(|rb| {
                let v_ov = frame.bottom().min(rb.bottom()) - frame.y.max(rb.y);
                let h_gap = frame.x - rb.right();
                v_ov as f64 >= frame.h as f64 * 0.5 && (-2..=14).contains(&h_gap)
            })
        {
            union(rb, &frame);
            continue;
        }
        final_frames.push(frame);
    }
    for ic in inner_frame_children {
        let (cx, cy) = (
            ic.x as f64 + ic.w as f64 * 0.5,
            ic.y as f64 + ic.h as f64 * 0.5,
        );
        for frame in &mut final_frames {
            if frame.x as f64 <= cx
                && cx <= frame.right() as f64
                && frame.y as f64 <= cy
                && cy <= frame.bottom() as f64
                && (ic.area() as f64) < 0.90 * frame.area() as f64
            {
                let hint = hints(frame.children.len() + 1).pop().unwrap();
                frame
                    .children
                    .push(make_child(&ic, &ic.kind, &ic.level, &hint));
                break;
            }
        }
    }
    rem_rbs.retain(|rb| {
        !(rb.kind == "icon"
            && rb.w <= 34.max(scaled(36.))
            && rb.h <= 34.max(scaled(36.))
            && (((rb.x <= 4 || rb.right() >= w - 4) && (rb.y <= 4 || rb.bottom() >= h - 4))
                || state.panel_corners.iter().any(|&(pcx, pcy)| {
                    (rb.x as f64 + rb.w as f64 * 0.5 - pcx as f64).abs()
                        <= 18.max(scaled(22.)) as f64
                        && (rb.y as f64 + rb.h as f64 * 0.5 - pcy as f64).abs()
                            <= 18.max(scaled(22.)) as f64
                })))
    });

    // 5C: sequential padding observes all previously padded refined boxes.
    let base_min_dim = 26.max(scaled(30.));
    let base_pad = 6.max(scaled(8.));
    for i in 0..rem_rbs.len() {
        if rem_rbs[i].kind != "icon" {
            continue;
        }
        for frac in [1.0, 0.65, 0.40] {
            let rb = &rem_rbs[i];
            let min_d = (base_min_dim as f64 * frac) as i32;
            let pad = (base_pad as f64 * frac) as i32;
            let mut padded = rb.clone();
            padded.w = min_d.max(rb.w + 2 * pad);
            padded.h = min_d.max(rb.h + 2 * pad);
            padded.x = (w - padded.w)
                .min(rb.x + (rb.w - padded.w).div_euclid(2))
                .max(0);
            padded.y = (h - padded.h)
                .min(rb.y + (rb.h - padded.h).div_euclid(2))
                .max(0);
            let collision = rem_rbs
                .iter()
                .enumerate()
                .any(|(j, other)| i != j && overlap(&padded, other) > 0)
                || thumbs
                    .iter()
                    .chain(&desktop_icons)
                    .chain(&final_frames)
                    .any(|other| overlap(&padded, other) > 0);
            if !collision {
                rem_rbs[i] = padded;
                break;
            }
        }
    }

    // 5D: paragraph membership is fixed before chaining, including eligible link lines.
    if options.collapse_paragraphs && options.include_text {
        let mut indices: Vec<usize> = rem_rbs
            .iter()
            .enumerate()
            .filter_map(|(i, rb)| {
                (rb.kind == "text_block"
                    || (rb.kind == "text_link"
                        && rem_rbs.iter().any(|other| {
                            other.kind == "text_block"
                                && (0..=12.max(scaled(14.))).contains(&(rb.y - other.bottom()))
                                && (rb.x - other.x).abs() <= 10.max(scaled(12.))
                                && (rb.h - other.h).abs() <= 6
                        })))
                .then_some(i)
            })
            .collect();
        indices.sort_by_key(|&i| (rem_rbs[i].y, rem_rbs[i].x));
        let mut merged = vec![false; rem_rbs.len()];
        let mut paragraphs = Vec::new();
        for &idx in &indices {
            if merged[idx] {
                continue;
            }
            let mut chain = vec![idx];
            merged[idx] = true;
            let mut curr = &rem_rbs[idx];
            for &idx2 in &indices {
                if merged[idx2] {
                    continue;
                }
                let cand = &rem_rbs[idx2];
                if (0..=12.max(scaled(14.))).contains(&(cand.y - curr.bottom()))
                    && (cand.x - curr.x).abs() <= 12.max(scaled(14.))
                {
                    chain.push(idx2);
                    merged[idx2] = true;
                    curr = cand;
                }
            }
            if chain.len() < 2 {
                paragraphs.push(rem_rbs[idx].clone());
                continue;
            }
            let mut paragraph = Target::new(
                rem_rbs[idx].x,
                rem_rbs[idx].y,
                rem_rbs[idx].w,
                rem_rbs[idx].h,
                0.80,
                "text_block",
                "paragraph",
            );
            for &i in &chain {
                union(&mut paragraph, &rem_rbs[i]);
            }
            if rem_rbs.iter().enumerate().any(|(i, rb)| {
                !chain.contains(&i) && rb.kind != "inline_link" && overlap(&paragraph, rb) > 0
            }) {
                paragraphs.extend(chain.iter().map(|&i| rem_rbs[i].clone()));
                continue;
            }
            for (&i, hint) in chain.iter().zip(hints(chain.len())) {
                let mut line = rem_rbs[i].clone();
                line.score = 0.78;
                paragraph
                    .children
                    .push(make_child(&line, "line", "line", &hint));
            }
            paragraphs.push(paragraph);
        }
        let mut is_text = vec![false; rem_rbs.len()];
        for i in indices {
            is_text[i] = true;
        }
        rem_rbs = rem_rbs
            .into_iter()
            .enumerate()
            .filter_map(|(i, rb)| (!is_text[i]).then_some(rb))
            .chain(paragraphs)
            .collect();
    }

    // 5E: rail proposals are checked together, then committed together.
    let mut di_indices: Vec<usize> = (0..desktop_icons.len()).collect();
    di_indices.sort_by_key(|&i| desktop_icons[i].y);
    let mut visited = vec![false; desktop_icons.len()];
    for &i in &di_indices {
        if visited[i] {
            continue;
        }
        let di_x = desktop_icons[i].x;
        let rail: Vec<usize> = di_indices
            .iter()
            .copied()
            .filter(|&j| !visited[j] && (desktop_icons[j].x - di_x).abs() <= 8.max(scaled(10.)))
            .collect();
        if rail.len() < 3 {
            continue;
        }
        let mut chain = vec![rail[0]];
        for &j in &rail[1..] {
            let last = &desktop_icons[*chain.last().unwrap()];
            let next = &desktop_icons[j];
            let dy = (next.y as f64 + next.h as f64 * 0.5) - (last.y as f64 + last.h as f64 * 0.5);
            if (24.0..=75.max(scaled(85.)) as f64).contains(&dy) {
                chain.push(j);
            }
        }
        if chain.len() < 3 {
            continue;
        }
        let norm_x = chain.iter().map(|&j| desktop_icons[j].x).min().unwrap();
        let norm_r = (norm_x + 160.max(scaled(185.))).min(
            chain
                .iter()
                .map(|&j| desktop_icons[j].right())
                .max()
                .unwrap(),
        );
        let heights: Vec<i32> = chain.iter().map(|&j| desktop_icons[j].h).collect();
        let norm_h = median(&heights) as i32 + scaled(6.);
        let proposals: Vec<Target> = chain
            .iter()
            .map(|&j| {
                let icon = &desktop_icons[j];
                let cy = icon.y as f64 + icon.h as f64 * 0.5;
                let ny = (h - norm_h)
                    .min((cy - norm_h as f64 * 0.5).round_ties_even() as i32)
                    .max(0);
                Target::new(
                    norm_x,
                    ny,
                    norm_r - norm_x,
                    norm_h,
                    icon.score,
                    &icon.kind,
                    &icon.level,
                )
            })
            .collect();
        let collision = proposals.iter().enumerate().any(|(j, proposed)| {
            proposals
                .iter()
                .enumerate()
                .any(|(k, other)| j != k && overlap(proposed, other) > 0)
                || thumbs
                    .iter()
                    .chain(&final_frames)
                    .chain(&rem_rbs)
                    .any(|other| overlap(proposed, other) > 0)
        });
        if !collision {
            for (&j, proposal) in chain.iter().zip(proposals) {
                visited[j] = true;
                let icon = &mut desktop_icons[j];
                icon.x = proposal.x;
                icon.y = proposal.y;
                icon.w = proposal.w;
                icon.h = proposal.h;
            }
        }
    }

    // Containment cleanup runs before score filtering, and can mutate parent children.
    final_frames.retain(|frame| {
        !thumbs
            .iter()
            .chain(&desktop_icons)
            .any(|other| overlap(frame, other) as f64 >= 0.25 * frame.area().max(1) as f64)
    });
    rem_rbs.sort_by_key(|rb| std::cmp::Reverse(rb.area()));
    let mut clean_rbs: Vec<Target> = Vec::new();
    for rb in rem_rbs {
        let area = rb.area().max(1);
        let mut absorbed = false;
        for parent in desktop_icons
            .iter_mut()
            .chain(&mut thumbs)
            .chain(&mut final_frames)
        {
            if overlap(&rb, parent) as f64 >= 0.25 * area as f64 {
                if matches!(parent.kind.as_str(), "desktop_icon" | "frame") {
                    let hint = hints(parent.children.len() + 1).pop().unwrap();
                    parent
                        .children
                        .push(make_child(&rb, &rb.kind, &rb.level, &hint));
                }
                absorbed = true;
                break;
            }
        }
        if !absorbed {
            absorbed = clean_rbs.iter().any(|kept| {
                overlap(&rb, kept) as f64 >= 0.65 * area as f64 && area <= kept.w * kept.h
            });
        }
        if !absorbed {
            clean_rbs.push(rb);
        }
    }
    let mut targets: Vec<Target> = thumbs
        .into_iter()
        .chain(desktop_icons)
        .chain(final_frames)
        .chain(clean_rbs)
        .filter(|t| t.score >= options.min_score)
        .map(|mut t| {
            t.x = (t.x as f64 * scale_back) as i32;
            t.y = (t.y as f64 * scale_back) as i32;
            t.w = (t.w as f64 * scale_back) as i32;
            t.h = (t.h as f64 * scale_back) as i32;
            t
        })
        .collect();
    targets.sort_by_key(|t| (t.y.div_euclid(14), t.x));
    let target_hints = hints(targets.len());
    for (target, hint) in targets.iter_mut().zip(target_hints) {
        target.hint = hint;
    }
    Ok(targets)
}
