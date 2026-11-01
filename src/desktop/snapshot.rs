//! Numbered snapshot artifacts for replay and vision-assisted debugging.
//! Each capture uses the same stem for the raw image, annotated image and JSON.
use super::{
    draw::render_debug_stage_canvas,
    geometry::Output,
    session::{Input, Session},
};
use crate::{DebugData, Options, Target, drill_down};
use opencv::{core, imgcodecs, imgproc, prelude::*};
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
    pub debug_sheet: PathBuf,
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
    debug: Value,
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

#[allow(clippy::too_many_arguments)]
fn document<'a>(
    number: u32,
    width: i32,
    height: i32,
    output: &'a Output,
    options: Options,
    timings: &'a Value,
    debug: Value,
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
            debug,
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
    if let Err(error) = file.write_all(bytes) {
        drop(file);
        let _ = fs::remove_file(path);
        return Err(format!("write {}: {error}", path.display()));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn save(
    image: &Mat,
    rendered: &Mat,
    targets: &[Target],
    output: &Output,
    timings: &Value,
    debug: &DebugData,
    options: Options,
    directory: &Path,
) -> Result<Saved, String> {
    ensure_directory(directory)?;
    let number = next_number(directory)?;
    let png = directory.join(format!("{number}.png"));
    let annotated = directory.join(format!("{number}.annotated.png"));
    let json_path = directory.join(format!("{number}.json"));

    // Snapshot each inspector view from the same detector pass. The six views
    // are visualizations, not six separate detector passes.
    let mut stages = Vec::new();
    let mut tiles = Vec::new();
    for (index, name, description, related_phase) in [
        (1, "overlay", "Original capture with clickable hints", None),
        (
            2,
            "physics",
            "Capture with classified target outlines",
            None,
        ),
        (
            3,
            "gradients",
            "Maximum local gradient strength",
            Some("signal"),
        ),
        (
            4,
            "barriers",
            "Structural barriers and frame boundaries",
            Some("frames"),
        ),
        (
            5,
            "strokes",
            "Stroke and coarse candidate masks",
            Some("coarse"),
        ),
        (
            6,
            "hierarchy",
            "Final parent, child and word boxes",
            Some("recursive"),
        ),
    ] {
        let mut session = Session::hints(image.cols(), image.rows(), targets.to_vec());
        let _ = session.handle_input(Input::SetDebugStage(index));
        let visual = render_debug_stage_canvas(image, &session, Some(debug), false)
            .map_err(|e| e.to_string())?;
        let filename = format!("{number}.debug-{index}-{name}.png");
        stages.push(json!({"view": index, "name": name, "description": description, "image": filename, "related_phase": related_phase}));
        tiles.push((filename, name, visual));
    }
    let thumb_w = image.cols().clamp(1, 640);
    let thumb_h = image.rows().clamp(1, 400);
    let mut sheet = Mat::new_rows_cols_with_default(
        thumb_h * 2 + 60,
        thumb_w * 3,
        core::CV_8UC3,
        core::Scalar::new(20., 20., 20., 0.),
    )
    .map_err(|e| e.to_string())?;
    for (idx, (_, name, visual)) in tiles.iter().enumerate() {
        let mut scaled = Mat::default();
        imgproc::resize(
            visual,
            &mut scaled,
            core::Size::new(thumb_w, thumb_h),
            0.,
            0.,
            imgproc::INTER_AREA,
        )
        .map_err(|e| e.to_string())?;
        let x = (idx % 3) as i32 * thumb_w;
        let y = (idx / 3) as i32 * (thumb_h + 30);
        let mut roi = sheet
            .roi_mut(core::Rect::new(x, y + 30, thumb_w, thumb_h))
            .map_err(|e| e.to_string())?;
        scaled.copy_to(&mut roi).map_err(|e| e.to_string())?;
        imgproc::put_text(
            &mut sheet,
            &format!("{}: {}", idx + 1, name),
            core::Point::new(x + 10, y + 22),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.55,
            core::Scalar::new(255., 255., 255., 0.),
            1,
            imgproc::LINE_AA,
            false,
        )
        .map_err(|e| e.to_string())?;
    }
    let sheet_name = format!("{number}.debug-sheet.png");
    let nonzero = |mask: &[u8]| mask.iter().filter(|&&value| value != 0).count();
    let debug_json = json!({
        "sheet": sheet_name, "views": stages,
        "work_resolution": [debug.work_w, debug.work_h],
        "physics": {"l0": debug.physics.l0, "noise_floor": debug.physics.noise_floor, "flat_threshold": debug.physics.flat_thr, "border_threshold": debug.physics.border_thr, "stroke_threshold": debug.physics.stroke_thr, "scale": debug.physics.scale},
        "counts": {"frames": debug.frames_count, "thumbnails": debug.thumbs_count, "coarse_regions": debug.coarse_count, "gradient_nonzero": nonzero(&debug.grad_max), "barrier_nonzero": nonzero(&debug.bar_all), "stroke_nonzero": nonzero(&debug.stroke_mask), "coarse_nonzero": nonzero(&debug.coarse_mask)},
        "pipeline_ms": {"signal": debug.timings.stage1_gradients_ms, "frames": debug.timings.stage2_frames_ms, "coarse": debug.timings.stage3_strokes_ms, "recursive": debug.timings.stage5_post_ms, "total": debug.timings.total_ms}
    });

    let (doc, nodes, drills) = document(
        number,
        image.cols(),
        image.rows(),
        output,
        options,
        timings,
        debug_json,
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
    let mut artifacts = vec![
        (png.clone(), png_bytes.to_vec()),
        (annotated.clone(), annotated_bytes.to_vec()),
        (json_path.clone(), json),
    ];
    for (filename, visual) in tiles
        .iter()
        .map(|(filename, _, visual)| (filename, visual))
        .chain(std::iter::once((&sheet_name, &sheet)))
    {
        let mut bytes = core::Vector::<u8>::new();
        if !imgcodecs::imencode(".png", visual, &mut bytes, &core::Vector::new())
            .map_err(|e| e.to_string())?
        {
            return Err(format!("failed to encode {filename}"));
        }
        artifacts.push((directory.join(filename), bytes.to_vec()));
    }
    let mut written = Vec::new();
    for (path, bytes) in artifacts {
        if let Err(error) = write_private(&path, &bytes) {
            for created in written {
                let _ = fs::remove_file(created);
            }
            return Err(error);
        }
        written.push(path);
    }
    Ok(Saved {
        number,
        screenshot: png,
        rendered: annotated,
        debug_sheet: directory.join(sheet_name),
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
        PathBuf::from("/tmp/opencode").join(format!(
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
        let (doc, nodes, drills) = document(
            1,
            100,
            100,
            &output,
            Options::default(),
            &timings,
            json!({}),
            &targets,
        )
        .unwrap();
        assert_eq!(nodes, 4);
        assert_eq!(drills, 2);
        assert_eq!(doc.targets.len(), 1);
        assert!(doc.screenshot.ends_with("1.png"));
        assert!(doc.rendered.ends_with("1.annotated.png"));
    }

    #[test]
    fn snapshot_writes_six_debug_views_and_sheet() {
        let image =
            Mat::new_rows_cols_with_default(64, 80, core::CV_8UC3, core::Scalar::all(40.)).unwrap();
        let mut target = Target::new(4, 5, 20, 12, 0.8, "frame", "control");
        target.hint = "S".into();
        let targets = [target];
        let debug = DebugData {
            grad_bgr: Mat::default(),
            barriers_bgr: Mat::default(),
            strokes_coarse_bgr: Mat::default(),
            hierarchy_bgr: Mat::default(),
            grad_max: vec![60; 80 * 64],
            bar_all: vec![0; 80 * 64],
            stroke_mask: vec![0; 80 * 64],
            coarse_mask: vec![0; 80 * 64],
            physics: crate::DebugPhysics::default(),
            timings: crate::DebugTimings::default(),
            l0: 12.,
            stroke_thr: 10,
            scale_back: 1.,
            work_w: 80,
            work_h: 64,
            frames_count: 1,
            thumbs_count: 0,
            coarse_count: 0,
        };
        let output = Output {
            name: "synthetic".into(),
            x: 0,
            y: 0,
            width: 80,
            height: 64,
            scale: 1.,
            transform: 0,
        };
        let directory = temp_dir("debug");
        let saved = save(
            &image,
            &image,
            &targets,
            &output,
            &json!({}),
            &debug,
            Options::default(),
            &directory,
        )
        .unwrap();
        let doc: Value = serde_json::from_slice(&fs::read(saved.json).unwrap()).unwrap();
        assert_eq!(doc["debug"]["views"].as_array().unwrap().len(), 6);
        assert_eq!(doc["debug"]["counts"]["frames"], 1);
        assert_eq!(doc["debug"]["views"][2]["related_phase"], "signal");
        assert!(doc["debug"]["pipeline_ms"]["recursive"].is_number());
        assert!(saved.debug_sheet.exists());
        for view in doc["debug"]["views"].as_array().unwrap() {
            assert!(directory.join(view["image"].as_str().unwrap()).exists());
        }
        fs::remove_dir_all(directory).unwrap();
    }
}
