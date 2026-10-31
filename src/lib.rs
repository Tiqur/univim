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

#[derive(Clone, Copy)]
pub struct Options {
    pub min_score: f64,
    pub include_text: bool,
    pub turbo: bool,
    pub collapse_paragraphs: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            min_score: 0.45,
            include_text: true,
            turbo: false,
            collapse_paragraphs: true,
        }
    }
}
pub fn detect(bgr: &Mat, options: Options) -> opencv::Result<(Vec<Target>, Value)> {
    let start = Instant::now();
    let state = signals::prepare(bgr, options)?;
    let (refined, inner) = refine::refine(&state, options)?;
    let targets = compose::compose(&state, refined, inner, options)?;
    let total = start.elapsed().as_secs_f64() * 1000.;
    Ok((
        targets,
        json!({"signal_ms":state.signal_ms,"frames_ms":state.frames_ms,"coarse_ms":state.coarse_ms,"recursive_ms":total-state.signal_ms-state.frames_ms-state.coarse_ms,"total_ms":total}),
    ))
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
