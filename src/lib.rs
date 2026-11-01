//! Native port of the reference detector. Work-coordinate targets stay unscaled
//! until finalization; serialized children already use original-image coordinates.
use opencv::{core::*, imgproc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Instant;

pub mod compose;
#[cfg(target_os = "linux")]
pub mod desktop;
pub mod refine;
pub mod render;
pub mod signals;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Target {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub score: f64,
    pub kind: String,
    #[serde(default)]
    pub hint: String,
    pub level: String,
    #[serde(default)]
    pub children: Vec<Value>,
}
impl Target {
    pub fn new(x: i32, y: i32, w: i32, h: i32, score: f64, kind: &str, level: &str) -> Self {
        Self {
            x,
            y,
            w,
            h,
            score,
            kind: kind.into(),
            level: level.into(),
            hint: String::new(),
            children: vec![],
        }
    }
    pub fn area(&self) -> i32 {
        self.w * self.h
    }
    pub fn right(&self) -> i32 {
        self.x + self.w
    }
    pub fn bottom(&self) -> i32 {
        self.y + self.h
    }
}
pub fn overlap(a: &Target, b: &Target) -> i32 {
    (a.right().min(b.right()) - a.x.max(b.x)).max(0)
        * (a.bottom().min(b.bottom()) - a.y.max(b.y)).max(0)
}
pub fn hints(n: usize) -> Vec<String> {
    hints_with_alphabet(n, "sadfjklewcmpgh")
}

/// Generate labels using the reference's greedy expansion and alphabet order.
/// At least two characters are required for expansion beyond single-letter hints.
pub fn hints_with_alphabet(n: usize, alphabet: &str) -> Vec<String> {
    if n == 0 {
        return vec![];
    }
    let a: Vec<char> = alphabet.to_uppercase().chars().collect();
    let k = a.len();
    assert!(k > 0, "hint alphabet must not be empty");
    if n <= k {
        return a[..n].iter().map(char::to_string).collect();
    }
    let two: Vec<String> = (0..k * k)
        .map(|i| format!("{}{}", a[i / k], a[i % k]))
        .collect();
    if n <= k * k {
        return two[..n].to_vec();
    }
    assert!(k > 1, "hint expansion requires at least two characters");
    let expand = (n - k * k + k - 2) / (k - 1);
    // Match Python's negative slice semantics when the reference exceeds three characters.
    let split = if expand <= k * k {
        k * k - expand
    } else {
        (2 * k * k).saturating_sub(expand)
    };
    let mut out = two[..split].to_vec();
    for p in &two[split..] {
        for c in &a {
            out.push(format!("{p}{c}"));
            if out.len() >= n {
                return out;
            }
        }
    }
    out.truncate(n);
    out
}
pub fn child(t: &Target, scale: f64, kind: &str, level: &str, hint: &str) -> Value {
    // Decimal formatting rounds the represented binary value as Python round(x, 2).
    json!({"x":(t.x as f64*scale) as i32,"y":(t.y as f64*scale) as i32,
        "w":(t.w as f64*scale) as i32,"h":(t.h as f64*scale) as i32,
        "score":format!("{:.2}",t.score).parse::<f64>().unwrap(),"kind":kind,"level":level,"hint":hint,"children":t.children})
}
pub fn zeros(h: i32, w: i32) -> opencv::Result<Mat> {
    Mat::zeros(h, w, CV_8UC1)?.to_mat()
}
pub fn kernel(w: i32, h: i32) -> opencv::Result<Mat> {
    imgproc::get_structuring_element(imgproc::MORPH_RECT, Size::new(w, h), Point::new(-1, -1))
}
pub fn morph(m: &Mat, op: i32, k: &Mat) -> opencv::Result<Mat> {
    let mut out = Mat::default();
    imgproc::morphology_ex(
        m,
        &mut out,
        op,
        k,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    Ok(out)
}
pub fn dilate(m: &Mat, k: &Mat) -> opencv::Result<Mat> {
    morph(m, imgproc::MORPH_DILATE, k)
}
pub fn threshold(m: &Mat, t: f64, max: f64, inverse: bool) -> opencv::Result<Mat> {
    let mut out = Mat::default();
    imgproc::threshold(
        m,
        &mut out,
        t,
        max,
        if inverse {
            imgproc::THRESH_BINARY_INV
        } else {
            imgproc::THRESH_BINARY
        },
    )?;
    Ok(out)
}
pub fn and(a: &Mat, b: &Mat) -> opencv::Result<Mat> {
    let mut out = Mat::default();
    bitwise_and(a, b, &mut out, &no_array())?;
    Ok(out)
}
pub fn or(a: &Mat, b: &Mat) -> opencv::Result<Mat> {
    let mut out = Mat::default();
    bitwise_or(a, b, &mut out, &no_array())?;
    Ok(out)
}
pub fn invert(a: &Mat) -> opencv::Result<Mat> {
    let mut out = Mat::default();
    bitwise_not(a, &mut out, &no_array())?;
    Ok(out)
}
pub fn resize(a: &Mat, w: i32, h: i32, mode: i32) -> opencv::Result<Mat> {
    let mut out = Mat::default();
    imgproc::resize(a, &mut out, Size::new(w, h), 0., 0., mode)?;
    Ok(out)
}
pub fn contours(a: &Mat, mode: i32) -> opencv::Result<Vector<Vector<Point>>> {
    let mut out = Vector::new();
    imgproc::find_contours(
        a,
        &mut out,
        mode,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;
    Ok(out)
}
pub fn bounding_boxes(a: &Mat) -> opencv::Result<Vec<(i32, i32, i32, i32)>> {
    let mut out = Vec::new();
    for c in contours(a, imgproc::RETR_EXTERNAL)? {
        let bb = imgproc::bounding_rect(&c)?;
        out.push((bb.x, bb.y, bb.width, bb.height));
    }
    Ok(out)
}
// Keep the reference's x/y/width/height arguments while drawing inclusive endpoints.
#[allow(clippy::too_many_arguments)]
pub fn rect(
    m: &mut Mat,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    color: Scalar,
    thickness: i32,
    line: i32,
) -> opencv::Result<()> {
    imgproc::rectangle_points(
        m,
        Point::new(x, y),
        Point::new(x + w, y + h),
        color,
        thickness,
        line,
        0,
    )
}
// NumPy slices clip endpoints and allow empty regions.
pub fn count(m: &Mat, x1: i32, y1: i32, x2: i32, y2: i32) -> opencv::Result<i32> {
    let x1 = x1.clamp(0, m.cols());
    let x2 = x2.clamp(0, m.cols());
    let y1 = y1.clamp(0, m.rows());
    let y2 = y2.clamp(0, m.rows());
    if x2 <= x1 || y2 <= y1 {
        return Ok(0);
    }
    count_non_zero(&Mat::roi(m, Rect::new(x1, y1, x2 - x1, y2 - y1))?)
}
pub fn max_roi(m: &Mat, x: i32, y: i32, w: i32, h: i32) -> opencv::Result<f64> {
    let mut max = 0.;
    min_max_loc(
        &Mat::roi(m, Rect::new(x, y, w, h))?,
        None,
        Some(&mut max),
        None,
        None,
        &no_array(),
    )?;
    Ok(max)
}
pub fn std_roi(m: &Mat, x: i32, y: i32, w: i32, h: i32) -> opencv::Result<f64> {
    let mut mean = Scalar::default();
    let mut std = Scalar::default();
    mean_std_dev(
        &Mat::roi(m, Rect::new(x, y, w, h))?,
        &mut mean,
        &mut std,
        &no_array(),
    )?;
    Ok(std[0])
}
pub fn mean_roi(m: &Mat, x: i32, y: i32, w: i32, h: i32) -> opencv::Result<f64> {
    let x1 = x.clamp(0, m.cols());
    let y1 = y.clamp(0, m.rows());
    let x2 = (x + w).clamp(0, m.cols());
    let y2 = (y + h).clamp(0, m.rows());
    if x2 <= x1 || y2 <= y1 {
        return Ok(f64::NAN);
    }
    Ok(opencv::core::mean(
        &Mat::roi(m, Rect::new(x1, y1, x2 - x1, y2 - y1))?,
        &no_array(),
    )?[0])
}
pub fn median_u8(v: &[u8]) -> f64 {
    let mut v = v.to_vec();
    v.sort_unstable();
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2] as f64
    } else {
        (v[n / 2 - 1] as f64 + v[n / 2] as f64) / 2.
    }
}
pub fn median(v: &[i32]) -> f64 {
    let mut v = v.to_vec();
    v.sort();
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2] as f64
    } else {
        (v[n / 2 - 1] as f64 + v[n / 2] as f64) / 2.
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Options {
    pub min_score: f64,
    pub include_text: bool,
    pub turbo: bool,
    /// Desktop spelling for `turbo`; both request the half-resolution pass.
    pub fast_half_res: bool,
    pub collapse_paragraphs: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            min_score: 0.45,
            include_text: true,
            turbo: false,
            fast_half_res: false,
            collapse_paragraphs: true,
        }
    }
}

/// Desktop-facing name for the detector options. The standalone CLI and parity
/// suite use `Options`, while the interactive daemon historically used this name.
pub type DetectorOptions = Options;

/// Generate home-row hint labels for a target count.
pub fn generate_vimium_hints(n: usize, alphabet: &str) -> Vec<String> {
    hints_with_alphabet(n, alphabet)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DebugPhysics {
    pub l0: f64,
    pub noise_floor: i32,
    pub flat_thr: i32,
    pub border_thr: i32,
    pub stroke_thr: i32,
    pub scale: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DebugTimings {
    pub stage1_gradients_ms: f64,
    pub stage2_frames_ms: f64,
    pub stage3_strokes_ms: f64,
    pub stage4_decompose_ms: f64,
    pub stage5_post_ms: f64,
    pub total_ms: f64,
}

/// Intermediate pipeline visual artifacts and calibrated physics parameters
/// used by the interactive Debug Mode (`F2`) inside `univim-desktop`.
#[derive(Clone)]
pub struct DebugData {
    pub grad_bgr: Mat,
    pub barriers_bgr: Mat,
    pub strokes_coarse_bgr: Mat,
    pub hierarchy_bgr: Mat,
    pub grad_max: Vec<u8>,
    pub bar_all: Vec<u8>,
    pub stroke_mask: Vec<u8>,
    pub coarse_mask: Vec<u8>,
    pub physics: DebugPhysics,
    pub timings: DebugTimings,
    pub l0: f64,
    pub stroke_thr: i32,
    pub scale_back: f64,
    pub work_w: i32,
    pub work_h: i32,
    pub frames_count: usize,
    pub thumbs_count: usize,
    pub coarse_count: usize,
}

fn gray_bytes(mat: &Mat) -> opencv::Result<Vec<u8>> {
    let mut out = Vec::with_capacity((mat.cols() * mat.rows()).max(0) as usize);
    for y in 0..mat.rows() {
        out.extend_from_slice(&mat.at_row::<u8>(y)?[..mat.cols() as usize]);
    }
    Ok(out)
}

fn build_debug_data(
    bgr: &Mat,
    state: &signals::State,
    targets: &[Target],
    timings: DebugTimings,
) -> opencv::Result<DebugData> {
    let (orig_w, orig_h) = (bgr.cols(), bgr.rows());
    let (w, h) = (state.w, state.h);

    // 1. Gradient + Chroma stage visualization (BGR)
    let mut grad_work = Mat::zeros(h, w, CV_8UC3)?.to_mat()?;
    for y in 0..h {
        let g_row = state.grad_max.at_row::<u8>(y)?;
        let c_row = state.chroma.at_row::<u8>(y)?;
        let dst = grad_work.at_row_mut::<Vec3b>(y)?;
        for x in 0..w as usize {
            let g = g_row[x];
            let c = c_row[x];
            dst[x] = Vec3b::from([c.saturating_mul(2), g, g.max(c)]);
        }
    }
    let grad_bgr = if w != orig_w || h != orig_h {
        resize(&grad_work, orig_w, orig_h, imgproc::INTER_NEAREST)?
    } else {
        grad_work
    };

    // 2. Structural Barriers, Closed Frames (lime), and Thumbnails (magenta)
    let mut bar_work = Mat::zeros(h, w, CV_8UC3)?.to_mat()?;
    for y in 0..h {
        let b_inv = state.barrier_inv.at_row::<u8>(y)?;
        let f_occ = state.frame_occ.at_row::<u8>(y)?;
        let t_occ = state.thumb_occ.at_row::<u8>(y)?;
        let gray = state.gray.at_row::<u8>(y)?;
        let dst = bar_work.at_row_mut::<Vec3b>(y)?;
        for x in 0..w as usize {
            let base = gray[x] / 4;
            if b_inv[x] == 0 {
                // Active barrier wall in bright cyan/white
                dst[x] = Vec3b::from([255, 220, 60]);
            } else if t_occ[x] > 0 {
                dst[x] = Vec3b::from([180, 50, 200]);
            } else if f_occ[x] > 0 {
                dst[x] = Vec3b::from([60, 180, 120]);
            } else {
                dst[x] = Vec3b::from([base, base, base]);
            }
        }
    }
    for f in &state.frames {
        rect(
            &mut bar_work,
            f.x,
            f.y,
            f.w,
            f.h,
            Scalar::new(64., 255., 198., 0.),
            2,
            imgproc::LINE_8,
        )?;
    }
    for t in &state.thumbs {
        rect(
            &mut bar_work,
            t.x,
            t.y,
            t.w,
            t.h,
            Scalar::new(224., 72., 255., 0.),
            2,
            imgproc::LINE_8,
        )?;
    }
    let barriers_bgr = if w != orig_w || h != orig_h {
        resize(&bar_work, orig_w, orig_h, imgproc::INTER_NEAREST)?
    } else {
        bar_work
    };

    // 3. Binary Stroke Mask (green), Blue Excess Link Mask (cyan), and Coarse Proposals (amber)
    let mut stroke_work = Mat::zeros(h, w, CV_8UC3)?.to_mat()?;
    for y in 0..h {
        let s_row = state.stroke_01.at_row::<u8>(y)?;
        let bl_row = state.blue_01.at_row::<u8>(y)?;
        let gray = state.gray.at_row::<u8>(y)?;
        let dst = stroke_work.at_row_mut::<Vec3b>(y)?;
        for x in 0..w as usize {
            let base = gray[x] / 5;
            let is_s = s_row[x] > 0;
            let is_b = bl_row[x] > 0;
            dst[x] = match (is_s, is_b) {
                (true, true) => Vec3b::from([255, 255, 80]),
                (true, false) => Vec3b::from([60, 235, 60]),
                (false, true) => Vec3b::from([255, 190, 40]),
                (false, false) => Vec3b::from([base, base, base]),
            };
        }
    }
    for &(cx, cy, cw, ch) in &state.coarse_boxes {
        rect(
            &mut stroke_work,
            cx,
            cy,
            cw,
            ch,
            Scalar::new(40., 190., 255., 0.),
            1,
            imgproc::LINE_8,
        )?;
    }
    let strokes_coarse_bgr = if w != orig_w || h != orig_h {
        resize(&stroke_work, orig_w, orig_h, imgproc::INTER_NEAREST)?
    } else {
        stroke_work
    };

    // 4. Full Hierarchy Overlay
    let hierarchy_bgr = render::hierarchy_overlay(bgr, targets)
        .unwrap_or_else(|_| bgr.try_clone().unwrap_or_default());

    let bar_all = {
        let mut out = Vec::with_capacity((w * h).max(0) as usize);
        for y in 0..h {
            let row = state.barrier_inv.at_row::<u8>(y)?;
            out.extend(
                row[..w as usize]
                    .iter()
                    .map(|&v| if v == 0 { 255 } else { 0 }),
            );
        }
        out
    };
    let mut coarse_mask = vec![0_u8; (w * h).max(0) as usize];
    for &(x, y, cw, ch) in &state.coarse_boxes {
        let x0 = x.clamp(0, w);
        let y0 = y.clamp(0, h);
        let x1 = (x + cw).clamp(0, w);
        let y1 = (y + ch).clamp(0, h);
        for row in y0..y1 {
            let base = (row * w) as usize;
            for col in x0..x1 {
                coarse_mask[base + col as usize] = 255;
            }
        }
    }

    Ok(DebugData {
        grad_bgr,
        barriers_bgr,
        strokes_coarse_bgr,
        hierarchy_bgr,
        grad_max: gray_bytes(&state.grad_max)?,
        bar_all,
        stroke_mask: gray_bytes(&state.stroke_01)?,
        coarse_mask,
        physics: DebugPhysics {
            l0: state.l0,
            noise_floor: state.noise_floor,
            flat_thr: state.flat_thr,
            border_thr: state.border_thr,
            stroke_thr: state.stroke_thr,
            scale: 1.0 / state.scale_back.max(f64::EPSILON),
        },
        timings,
        l0: state.l0,
        stroke_thr: state.stroke_thr,
        scale_back: state.scale_back,
        work_w: state.w,
        work_h: state.h,
        frames_count: state.frames.len(),
        thumbs_count: state.thumbs.len(),
        coarse_count: state.coarse_boxes.len(),
    })
}

pub fn detect(bgr: &Mat, options: Options) -> opencv::Result<(Vec<Target>, Value)> {
    let start = Instant::now();
    let mut state = signals::prepare(bgr, options)?;
    let (refined, inner) = refine::refine(&mut state, options)?;
    let targets = compose::compose(&state, refined, inner, options)?;
    let total = start.elapsed().as_secs_f64() * 1000.;
    Ok((
        targets,
        json!({"signal_ms":state.signal_ms,"frames_ms":state.frames_ms,"coarse_ms":state.coarse_ms,"recursive_ms":total-state.signal_ms-state.frames_ms-state.coarse_ms,"total_ms":total}),
    ))
}

pub fn detect_with_debug(
    bgr: &Mat,
    options: Options,
    _interactive_only: bool,
) -> opencv::Result<(Vec<Target>, DebugData, f64)> {
    let start = Instant::now();
    let mut state = signals::prepare(bgr, options)?;
    let (refined, inner) = refine::refine(&mut state, options)?;
    let targets = compose::compose(&state, refined, inner, options)?;
    let total = start.elapsed().as_secs_f64() * 1000.;
    let recursive = (total - state.signal_ms - state.frames_ms - state.coarse_ms).max(0.);
    let debug = build_debug_data(
        bgr,
        &state,
        &targets,
        DebugTimings {
            stage1_gradients_ms: state.signal_ms,
            stage2_frames_ms: state.frames_ms,
            stage3_strokes_ms: state.coarse_ms,
            stage4_decompose_ms: 0.,
            stage5_post_ms: recursive,
            total_ms: total,
        },
    )?;
    Ok((targets, debug, total))
}
/// Expand detector-produced children, retaining exactly one flattening level.
/// Externally supplied children must follow the detector's JSON schema; malformed
/// children return a deserialization error rather than panicking.
pub fn drill_down(target: &Target, to_words: bool) -> Result<Vec<Target>, serde_json::Error> {
    if target.children.is_empty() {
        return Ok(vec![target.clone()]);
    }
    let mut leaves = vec![];
    for c in &target.children {
        if to_words
            && c.get("children")
                .and_then(Value::as_array)
                .is_some_and(|v| !v.is_empty())
        {
            leaves.extend(c["children"].as_array().unwrap().iter().cloned())
        } else {
            leaves.push(c.clone())
        }
    }
    let mut out = vec![];
    for mut c in leaves {
        if let Some(obj) = c.as_object_mut() {
            obj.entry("score").or_insert(json!(0.85));
            obj.entry("kind").or_insert(json!("word"));
            let kind = obj["kind"].clone();
            obj.entry("level").or_insert(kind);
        }
        out.push(serde_json::from_value::<Target>(c)?);
    }
    let labels = hints(out.len());
    for (t, h) in out.iter_mut().zip(labels) {
        t.hint = h
    }
    Ok(out)
}

/// Recursively replace composite targets with their leaf descendants. This is
/// used by `--flat` hints: parents are not shown, only clickable leaves.
pub fn flatten_leaves(targets: &[Target]) -> Result<Vec<Target>, serde_json::Error> {
    fn visit(node: &Target, leaves: &mut Vec<Target>) -> Result<(), serde_json::Error> {
        if node.children.is_empty() {
            leaves.push(node.clone());
            return Ok(());
        }
        for child in &node.children {
            let child: Target = serde_json::from_value(child.clone())?;
            visit(&child, leaves)?;
        }
        Ok(())
    }

    let mut leaves = vec![];
    for target in targets {
        visit(target, &mut leaves)?;
    }
    Ok(leaves)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hint_boundaries_preserve_reference_order() {
        assert!(hints(0).is_empty());
        assert_eq!(hints(3), ["S", "A", "D"]);
        assert_eq!(&hints(14)[13], "H");
        assert_eq!(&hints(15)[14], "AS");
        assert_eq!(&hints(196)[195], "HH");
        let expanded = hints(197);
        assert_eq!(&expanded[194..], ["HG", "HHS", "HHA"]);
        for n in [1, 14, 15, 196, 197, 300, 2744] {
            let labels = hints(n);
            assert_eq!(labels.len(), n);
            for (i, a) in labels.iter().enumerate() {
                assert!(
                    !labels
                        .iter()
                        .enumerate()
                        .any(|(j, b)| i != j && b.starts_with(a))
                );
            }
        }
    }

    #[test]
    fn drilldown_flattens_exactly_one_level_and_applies_defaults() {
        let leaf = json!({"x":1,"y":2,"w":3,"h":4});
        let mut parent = Target::new(0, 0, 100, 100, 0.9, "frame", "control");
        let mut line = leaf.clone();
        line["children"] = json!([{"x":5,"y":6,"w":7,"h":8,"children":[leaf]}]);
        parent.children = vec![line, leaf];
        let immediate = drill_down(&parent, false).unwrap();
        assert_eq!(immediate.len(), 2);
        assert_eq!(immediate[0].x, 1);
        let words = drill_down(&parent, true).unwrap();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].x, 5);
        assert_eq!(words[0].children.len(), 1);
        assert_eq!(words[1].kind, "word");
        assert_eq!(words[1].level, "word");
        assert_eq!(words[1].score, 0.85);
        assert_eq!(words[0].hint, "S");
        assert_eq!(words[1].hint, "A");
    }

    #[test]
    fn drilldown_keeps_a_leaf_unchanged() {
        let mut leaf = Target::new(1, 2, 3, 4, 0.85, "icon", "control");
        leaf.hint = "DA".into();
        assert_eq!(drill_down(&leaf, true).unwrap(), vec![leaf.clone()]);
        assert_eq!(drill_down(&leaf, false).unwrap(), vec![leaf]);
    }

    #[test]
    fn malformed_children_return_errors() {
        let mut parent = Target::new(0, 0, 10, 10, 0.9, "frame", "control");
        for value in [
            json!(null),
            json!({"x": 1.5, "y": 2, "w": 3, "h": 4}),
            json!({}),
        ] {
            parent.children = vec![value];
            assert!(drill_down(&parent, true).is_err());
        }
    }

    #[test]
    fn flatten_leaves_removes_all_composite_parents() {
        let mut parent = Target::new(0, 0, 100, 100, 0.9, "frame", "control");
        parent.children = vec![
            json!({"x":1,"y":2,"w":30,"h":20,"score":0.85,"kind":"line","level":"line","hint":"A","children":[
                {"x":2,"y":3,"w":10,"h":8,"score":0.85,"kind":"word","level":"word","hint":"B"}
            ]}),
            json!({"x":3,"y":4,"w":12,"h":9,"score":0.85,"kind":"word","level":"word","hint":"C"}),
        ];
        let leaves = flatten_leaves(&[parent]).unwrap();
        assert_eq!(leaves.len(), 2);
        assert_eq!(leaves[0].x, 2);
        assert_eq!(leaves[1].x, 3);
        assert!(leaves.iter().all(|target| target.children.is_empty()));
    }

    #[test]
    fn custom_alphabet_preserves_uppercase_and_expansion_order() {
        assert_eq!(hints_with_alphabet(0, ""), Vec::<String>::new());
        assert_eq!(hints_with_alphabet(2, "ab"), ["A", "B"]);
        assert_eq!(hints_with_alphabet(4, "ab"), ["AA", "AB", "BA", "BB"]);
        assert_eq!(
            hints_with_alphabet(5, "ab"),
            ["AA", "AB", "BA", "BBA", "BBB"]
        );
    }
}
