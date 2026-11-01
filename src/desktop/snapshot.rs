//! Numbered snapshot artifacts for replay and vision-assisted debugging.
//! Each capture uses the same stem for the raw image, annotated image and JSON.
use super::geometry::Output;
use crate::{Options, Target, drill_down};
use opencv::{core, imgcodecs, prelude::*};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    fs::{self, DirBuilder, OpenOptions},
    io::Write,
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::{Path, PathBuf},
};
use wayland_client::protocol::wl_shm;

/// Borrowed screencopy storage plus the metadata needed to convert it into an
/// owned, upright BGR image.
pub struct RawCaptureFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub format: wl_shm::Format,
    pub invert_y: bool,
    pub transform: i32,
    pub bytes: &'a [u8],
}

fn supported(format: wl_shm::Format) -> bool {
    matches!(
        format,
        wl_shm::Format::Argb8888
            | wl_shm::Format::Xrgb8888
            | wl_shm::Format::Abgr8888
            | wl_shm::Format::Xbgr8888
            | wl_shm::Format::Argb2101010
            | wl_shm::Format::Xrgb2101010
            | wl_shm::Format::Abgr2101010
            | wl_shm::Format::Xbgr2101010
    )
}

/// Decode a screencopy frame and apply the output transform exactly once.
pub fn decode_raw_frame_to_bgr(frame: RawCaptureFrame<'_>) -> Result<Mat, String> {
    let RawCaptureFrame {
        width: w,
        height: h,
        stride,
        format,
        invert_y,
        transform,
        bytes,
    } = frame;
    if !supported(format)
        || w == 0
        || h == 0
        || w > 32768
        || h > 32768
        || stride < w as usize * 4
        || bytes.len() < stride * h as usize
    {
        return Err("invalid capture storage".into());
    }
    let mut raw =
        Mat::new_rows_cols_with_default(h as i32, w as i32, core::CV_8UC3, core::Scalar::all(0.))
            .map_err(|e| e.to_string())?;
    let bgr = raw.data_bytes_mut().map_err(|e| e.to_string())?;
    let swapped = matches!(
        format,
        wl_shm::Format::Abgr8888
            | wl_shm::Format::Xbgr8888
            | wl_shm::Format::Abgr2101010
            | wl_shm::Format::Xbgr2101010
    );
    let ten = matches!(
        format,
        wl_shm::Format::Argb2101010
            | wl_shm::Format::Xrgb2101010
            | wl_shm::Format::Abgr2101010
            | wl_shm::Format::Xbgr2101010
    );
    for y in 0..h as usize {
        let sy = if invert_y { h as usize - 1 - y } else { y };
        let row = &bytes[sy * stride..sy * stride + w as usize * 4];
        for (x, pixel) in row.as_chunks::<4>().0.iter().enumerate() {
            let word = u32::from_ne_bytes([pixel[0], pixel[1], pixel[2], pixel[3]]);
            let (lo, g, hi) = if ten {
                (
                    ((word & 1023) * 255 + 511) / 1023,
                    (((word >> 10) & 1023) * 255 + 511) / 1023,
                    (((word >> 20) & 1023) * 255 + 511) / 1023,
                )
            } else {
                (word & 255, (word >> 8) & 255, (word >> 16) & 255)
            };
            let i = (y * w as usize + x) * 3;
            bgr[i] = if swapped { hi } else { lo } as u8;
            bgr[i + 1] = g as u8;
            bgr[i + 2] = if swapped { lo } else { hi } as u8;
        }
    }
    // Hyprland exports the raw framebuffer with transform metadata; apply it
    // here exactly once, including square/180-degree outputs.
    let rotation = match transform {
        0 | 4 => None,
        1 | 5 => Some(core::ROTATE_90_COUNTERCLOCKWISE),
        2 | 6 => Some(core::ROTATE_180),
        3 | 7 => Some(core::ROTATE_90_CLOCKWISE),
        _ => return Err("invalid output transform".into()),
    };
    let mut upright = Mat::default();
    if transform >= 4 {
        core::flip(&raw, &mut upright, 1).map_err(|e| e.to_string())?;
        raw = upright;
        upright = Mat::default();
    }
    if let Some(rotation) = rotation {
        core::rotate(&raw, &mut upright, rotation).map_err(|e| e.to_string())?;
        Ok(upright)
    } else {
        Ok(raw)
    }
}

pub struct Saved {
    pub number: u32,
    pub screenshot: PathBuf,
    pub rendered: PathBuf,
    pub json: PathBuf,
    pub nodes: usize,
    pub drills: usize,
}

#[derive(Serialize)]
struct Document<'a> {
    number: u32,
    screenshot: String,
    rendered: String,
    width: i32,
    height: i32,
    output: &'a Output,
    options: Value,
    timings: &'a Value,
    /// Complete ordered hierarchy: top-level targets and all nested children.
    targets: &'a [Target],
    /// Every node, flattened in hierarchy traversal order.
    flat: Vec<Value>,
    /// Immediate children and word drill-down results for every parent node.
    drills: Vec<Value>,
}

fn collect(
    node: &Target,
    path: &mut Vec<String>,
    depth: usize,
    flat: &mut Vec<Value>,
    drills: &mut Vec<Value>,
) -> Result<(), String> {
    flat.push(json!({
        "path": path,
        "depth": depth,
        "x": node.x,
        "y": node.y,
        "w": node.w,
        "h": node.h,
        "score": node.score,
        "kind": node.kind,
        "level": node.level,
        "hint": node.hint,
    }));
    if node.children.is_empty() {
        return Ok(());
    }
    let immediate = drill_down(node, false).map_err(|e| e.to_string())?;
    let words = drill_down(node, true).map_err(|e| e.to_string())?;
    drills.push(json!({
        "path": path,
        "immediate": immediate,
        "words": words,
    }));
    for (index, child) in node.children.iter().enumerate() {
        let child: Target = serde_json::from_value(child.clone())
            .map_err(|e| format!("invalid child target: {e}"))?;
        path.push(if child.hint.is_empty() {
            format!("#{index}")
        } else {
            child.hint.clone()
        });
        collect(&child, path, depth + 1, flat, drills)?;
        path.pop();
    }
    Ok(())
}

fn document<'a>(
    number: u32,
    width: i32,
    height: i32,
    output: &'a Output,
    options: Options,
    timings: &'a Value,
    targets: &'a [Target],
) -> Result<(Document<'a>, usize, usize), String> {
    let mut flat = Vec::new();
    let mut drills = Vec::new();
    for target in targets {
        let mut path = vec![if target.hint.is_empty() {
            "?".to_string()
        } else {
            target.hint.clone()
        }];
        collect(target, &mut path, 0, &mut flat, &mut drills)?;
    }
    let screenshot = format!("{number}.png");
    let rendered = format!("{number}.annotated.png");
    let nodes = flat.len();
    let drill_count = drills.len();
    Ok((
        Document {
            number,
            screenshot,
            rendered,
            width,
            height,
            output,
            options: json!({
                "min_score": options.min_score,
                "include_text": options.include_text,
                "turbo": options.turbo,
                "collapse_paragraphs": options.collapse_paragraphs,
            }),
            timings,
            targets,
            flat,
            drills,
        },
        nodes,
        drill_count,
    ))
}

fn next_number(directory: &Path) -> Result<u32, String> {
    let entries = fs::read_dir(directory).map_err(|e| e.to_string())?;
    let mut max = 0_u32;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name();
        let stem = name.to_string_lossy();
        let stem = stem.split('.').next().unwrap_or("");
        if let Ok(number) = stem.parse::<u32>() {
            max = max.max(number);
        }
    }
    max.checked_add(1)
        .ok_or_else(|| "snapshot number overflow".to_string())
}

fn ensure_directory(directory: &Path) -> Result<(), String> {
    if directory.exists() {
        let metadata = fs::symlink_metadata(directory).map_err(|e| e.to_string())?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err("snapshot path is not a real directory".into());
        }
        return Ok(());
    }
    DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(directory)
        .map_err(|e| e.to_string())
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .map_err(|e| format!("create {}: {e}", path.display()))?;
    file.write_all(bytes)
        .map_err(|e| format!("write {}: {e}", path.display()))
}

pub fn save(
    image: &Mat,
    rendered: &Mat,
    targets: &[Target],
    output: &Output,
    timings: &Value,
    options: Options,
    directory: &Path,
) -> Result<Saved, String> {
    ensure_directory(directory)?;
    let number = next_number(directory)?;
    let png = directory.join(format!("{number}.png"));
    let annotated = directory.join(format!("{number}.annotated.png"));
    let json_path = directory.join(format!("{number}.json"));

    let (doc, nodes, drills) = document(
        number,
        image.cols(),
        image.rows(),
        output,
        options,
        timings,
        targets,
    )?;
    let mut png_bytes = core::Vector::<u8>::new();
    if !imgcodecs::imencode(".png", image, &mut png_bytes, &core::Vector::new())
        .map_err(|e| e.to_string())?
    {
        return Err("failed to encode snapshot PNG".into());
    }
    let mut annotated_bytes = core::Vector::<u8>::new();
    if !imgcodecs::imencode(".png", rendered, &mut annotated_bytes, &core::Vector::new())
        .map_err(|e| e.to_string())?
    {
        return Err("failed to encode annotated snapshot PNG".into());
    }
    let json = serde_json::to_vec_pretty(&doc).map_err(|e| e.to_string())?;
    write_private(&png, png_bytes.as_slice())?;
    if let Err(error) = write_private(&annotated, annotated_bytes.as_slice()) {
        let _ = fs::remove_file(&png);
        return Err(error);
    }
    if let Err(error) = write_private(&json_path, &json) {
        let _ = fs::remove_file(&png);
        let _ = fs::remove_file(&annotated);
        return Err(error);
    }
    Ok(Saved {
        number,
        screenshot: png,
        rendered: annotated,
        json: json_path,
        nodes,
        drills,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "univim-snapshot-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn numbers_advance_from_any_numeric_artifact() {
        let dir = temp_dir("numbers");
        fs::create_dir_all(&dir).unwrap();
        assert_eq!(next_number(&dir).unwrap(), 1);
        fs::write(dir.join("7.png"), b"x").unwrap();
        fs::write(dir.join("3.annotated.png"), b"x").unwrap();
        fs::write(dir.join("notes.json"), b"x").unwrap();
        assert_eq!(next_number(&dir).unwrap(), 8);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn document_contains_every_node_and_drill() {
        let mut root = Target::new(1, 2, 30, 20, 0.9, "frame", "control");
        root.hint = "S".into();
        root.children = vec![
            json!({"x":1,"y":2,"w":30,"h":20,"score":0.85,"kind":"line","level":"line","hint":"A","children":[
                {"x":2,"y":3,"w":10,"h":8,"score":0.85,"kind":"word","level":"word","hint":"B"}
            ]}),
            json!({"x":3,"y":4,"w":12,"h":9,"score":0.85,"kind":"word","level":"word","hint":"C"}),
        ];
        let output = Output {
            name: "fake".into(),
            x: 0,
            y: 0,
            width: 100,
            height: 100,
            scale: 1.0,
            transform: 0,
        };
        let timings = json!({});
        let targets = [root];
        let (doc, nodes, drills) =
            document(1, 100, 100, &output, Options::default(), &timings, &targets).unwrap();
        assert_eq!(nodes, 4);
        assert_eq!(drills, 2);
        assert_eq!(doc.targets.len(), 1);
        assert!(doc.screenshot.ends_with("1.png"));
        assert!(doc.rendered.ends_with("1.annotated.png"));
    }
}
