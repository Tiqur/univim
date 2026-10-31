use clap::Parser;
use opencv::{core, imgcodecs, prelude::*};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};
use univim::{Options, detect, render};

const STAGES: [&str; 5] = ["signal", "frames", "coarse", "recursive", "total"];

fn summarize(samples: &[f64]) -> Value {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let runs = sorted.len();
    let median = if runs % 2 == 1 {
        sorted[runs / 2]
    } else {
        (sorted[runs / 2 - 1] + sorted[runs / 2]) / 2.
    };
    json!({
        "median_ms": median,
        "mean_ms": samples.iter().sum::<f64>() / runs as f64,
        "min_ms": sorted[0],
        "max_ms": sorted[runs - 1],
        "samples_ms": samples,
    })
}

#[derive(Parser)]
#[command(about = "Structure-aware hierarchical UI detector")]
struct Args {
    input: PathBuf,
    #[arg(short, long)]
    output: Option<PathBuf>,
    #[arg(long)]
    json: Option<PathBuf>,
    #[arg(long)]
    interactive_only: bool,
    #[arg(long)]
    turbo: bool,
    #[arg(long)]
    no_collapse_paragraphs: bool,
    #[arg(long, default_value_t = 0.45, allow_hyphen_values = true)]
    min_score: f64,
    #[arg(long)]
    drill_down: Option<String>,
    /// Reuse the decoded image and detector options for detection-only timings.
    #[arg(long)]
    benchmark: Option<usize>,
    #[arg(long, default_value_t = 5)]
    warmup: usize,
    #[arg(long)]
    benchmark_json: Option<PathBuf>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    core::set_num_threads(1)?;
    if args.benchmark == Some(0) {
        return Err("--benchmark must be positive".into());
    }
    let bgr = imgcodecs::imread(
        args.input.to_str().ok_or("input path is not UTF-8")?,
        imgcodecs::IMREAD_COLOR,
    )?;
    if bgr.empty() {
        return Err(format!("Could not read image: {}", args.input.display()).into());
    }
    let options = Options {
        min_score: args.min_score,
        include_text: !args.interactive_only,
        turbo: args.turbo,
        collapse_paragraphs: !args.no_collapse_paragraphs,
    };
    let (targets, timings) = detect(&bgr, options)?;
    println!(
        "Detected {} top-level targets ({} drill-down children) in {:.2} ms (signal={:.1}ms, frames={:.1}ms, coarse={:.1}ms, recursive+rails={:.1}ms)",
        targets.len(),
        targets.iter().map(|t| t.children.len()).sum::<usize>(),
        timings["total_ms"].as_f64().unwrap(),
        timings["signal_ms"].as_f64().unwrap(),
        timings["frames_ms"].as_f64().unwrap(),
        timings["coarse_ms"].as_f64().unwrap(),
        timings["recursive_ms"].as_f64().unwrap()
    );
    if let Some(path) = args.json {
        std::fs::write(path, serde_json::to_string_pretty(&targets)?)?
    }
    if let Some(path) = args.output {
        let parent = args
            .drill_down
            .as_ref()
            .and_then(|hint| targets.iter().find(|t| t.hint == hint.to_uppercase()));
        let canvas = if let Some(parent) = parent {
            render::drilldown_overlay(&bgr, parent, true)?
        } else {
            render::overlay(&bgr, &targets)?
        };
        if !imgcodecs::imwrite(
            path.to_str().ok_or("output path is not UTF-8")?,
            &canvas,
            &core::Vector::new(),
        )? {
            return Err("Could not write overlay".into());
        }
    }
    if let Some(runs) = args.benchmark {
        for _ in 0..args.warmup {
            std::hint::black_box(detect(&bgr, options)?);
        }
        let mut samples = Vec::with_capacity(runs);
        // Stage timings come from inside detect(), so they exclude CLI setup and
        // image decoding just like the outer wall-clock samples.
        let mut stage_samples: BTreeMap<&str, Vec<f64>> = STAGES
            .iter()
            .map(|stage| (*stage, Vec::with_capacity(runs)))
            .collect();
        for _ in 0..runs {
            let start = Instant::now();
            let (_, timings) = std::hint::black_box(detect(&bgr, options)?);
            samples.push(start.elapsed().as_secs_f64() * 1000.);
            for stage in STAGES {
                stage_samples.get_mut(stage).unwrap().push(
                    timings[format!("{stage}_ms")]
                        .as_f64()
                        .expect("detector stage timing must be numeric"),
                );
            }
        }
        let mut report = summarize(&samples);
        report["targets"] = json!(targets.len());
        report["stages"] = Value::Object(
            STAGES
                .iter()
                .map(|stage| ((*stage).into(), summarize(&stage_samples[stage])))
                .collect(),
        );
        if let Some(path) = args.benchmark_json {
            std::fs::write(path, serde_json::to_string_pretty(&report)?)?
        }
        println!("{report}");
    }
    Ok(())
}
