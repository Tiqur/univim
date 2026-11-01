#!/usr/bin/env python3
"""Deterministic Python/Rust differential tests; no downloaded fixtures required.

Run from any directory: python3 tests/parity.py [--quick] [--benchmark]
Build the release CLI first. All generated inputs, expectations, diagnostics and
coverage live in target/parity; optional timings live in target/benchmark.json.
JSON object key order is irrelevant, but list order, missing keys, numeric values,
and every nested child are compared without normalization or score tolerances.
The CLI exposes word drill-down through rendering, not a standalone hints API or
immediate-child drill-down API. Those unexposed API modes are NOT claimed tested.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "target" / "parity"
SEED = 20261030
WIDTHS = (640, 899, 900, 1440, 1600, 1601, 1920, 2560, 2561, 2880)
STAGES = ("signal", "frames", "coarse", "recursive", "total")


def load_reference():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("univim_parity_reference", ROOT / "python.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cv2.setNumThreads(1)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def timing_summary(samples):
    return {"median_ms": statistics.median(samples), "mean_ms": statistics.mean(samples),
            "min_ms": min(samples), "max_ms": max(samples), "samples_ms": samples}


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


def fixtures(quick):
    """Use fixed OpenCV primitives, font, PRNG seed and lossless PNGs only."""
    rng = np.random.default_rng(SEED)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def text(im, value, x, y, scale=0.55, color=(30, 30, 30), thick=1):
        cv2.putText(im, value, (x, y), font, scale, color, thick, cv2.LINE_AA)

    def ui(dark):
        bg, fg, border = ((28, 28, 30), (232, 232, 232), (150, 150, 150)) if dark else (
            (250, 250, 250), (28, 28, 28), (80, 80, 80))
        im = np.full((900, 1440, 3), bg, np.uint8)
        cv2.line(im, (220, 0), (220, 899), border, 1)
        cv2.line(im, (0, 70), (1439, 70), border, 1)
        text(im, "Categories / Editors' Choice", 250, 48, 1.05, fg, 2)
        for i, label in enumerate(("Home files", "Saved items", "Recent work", "Settings")):
            y = 105 + i * 62
            cv2.rectangle(im, (22, y), (54, y + 32), border, 2)
            cv2.circle(im, (38, y + 16), 7, fg, 2)
            text(im, label, 68, y + 23, 0.5, fg)
        for x, label in ((260, "Open file"), (450, "Save changes"), (670, "Search here")):
            cv2.rectangle(im, (x, 95), (x + 165, 135), border, 2)
            text(im, label, x + 12, 121, 0.58, fg)
        for i, line in enumerate((
            "This reproducible paragraph contains several ordinary words.",
            "The next sentence shares its margin and a compact line pitch.",
            "Every child retains its original coordinates and exact score.",
            "Short final line ends the paragraph.",
        )):
            text(im, line, 260, 205 + i * 25, 0.62, fg)
        text(im, "Read the", 260, 350, 0.6, fg)
        text(im, "documentation", 355, 350, 0.6, (230, 80, 20))
        text(im, "for more details", 515, 350, 0.6, fg)
        cv2.line(im, (250, 380), (1360, 380), border, 1)
        for i, label in enumerate(("Projects", "Pictures", "Archive")):
            x = 290 + i * 155
            cv2.rectangle(im, (x, 420), (x + 48, 468), border, 2)
            cv2.line(im, (x + 8, 435), (x + 38, 455), fg, 3)
            text(im, label, x - 8, 492, 0.53, fg)
        for i, value in enumerate(("8", "9", "10", "10.5", "12", "72")):
            text(im, value, 970, 430 + i * 34, 0.56, fg)
        for i, value in enumerate(("#", "B", "C", "E", "?", "+", "1")):
            text(im, value, 280 + i * 75, 575, 0.58, fg)
        cv2.line(im, (270, 630), (285, 630), fg, 2)
        cv2.line(im, (350, 620), (350, 641), fg, 2)
        cv2.circle(im, (430, 630), 8, fg, 2)
        photo = rng.integers(0, 256, (142, 220, 3), dtype=np.uint8)
        im[700:842, 270:490] = photo
        cv2.rectangle(im, (268, 698), (491, 843), border, 2)
        text(im, "Synthetic photo", 270, 875, 0.55, fg)
        return im

    light, dark = ui(False), ui(True)
    widths = (640, 1440, 1601, 2561) if quick else WIDTHS
    for i, width in enumerate(widths):
        source = dark if i % 2 else light
        height = round(width * 900 / 1440)
        yield f"ui-{'dark' if i % 2 else 'light'}-{width}", cv2.resize(
            source, (width, height), interpolation=cv2.INTER_AREA)
    # Both themes at the baseline size, independently of the resolution sweep.
    yield "ui-light-native", light
    yield "ui-dark-native", dark
    yield "blank-light", np.full((480, 640, 3), 255, np.uint8)
    if not quick:
        yield "blank-dark", np.zeros((563, 900, 3), np.uint8)
        noise = rng.integers(0, 256, (480, 899, 3), dtype=np.uint8)
        yield "random-noise", noise
        media = np.full((900, 1440, 3), 245, np.uint8)
        photo = rng.integers(0, 256, (240, 380, 3), dtype=np.uint8)
        photo = cv2.GaussianBlur(photo, (5, 5), 0)
        media[100:340, 80:460] = photo
        cv2.rectangle(media, (78, 98), (461, 341), (20, 20, 20), 2)
        terrain = rng.integers(40, 220, (450, 650, 3), dtype=np.uint8)
        terrain = cv2.GaussianBlur(terrain, (9, 9), 0)
        media[100:550, 650:1300] = terrain
        for i in range(20):
            pts = rng.integers((650, 100), (1300, 550), (8, 2), dtype=np.int32)
            cv2.polylines(media, [pts], False, (230, 220, 185), 2, cv2.LINE_AA)
        text(media, "OpenStreetMap", 700, 320, 1.1, (30, 30, 30), 2)
        yield "textured-photo-map", media
        prose = np.full((900, 1440, 3), 255, np.uint8)
        for col in range(2):
            for row in range(22):
                text(prose, "A dense paragraph keeps words and lines in order.",
                     40 + col * 700, 45 + row * 27, 0.64)
            cv2.line(prose, (40 + col * 700, 670), (670 + col * 700, 670), (100, 100, 100), 1)
            text(prose, "Blue reference link", 40 + col * 700, 710, 0.7, (230, 65, 15))
        yield "dense-prose", prose
        # Terminal rows repeat a left label and a long unbroken path with a
        # one-pixel vertical skew between the fragments. This exercises
        # row-band consolidation, monospace path tokenization, and the
        # paragraph-to-line drill step for blocks with many word leaves.
        terminal = np.full((1200, 900, 3), (18, 20, 22), np.uint8)
        plain = cv2.FONT_HERSHEY_PLAIN
        for row in range(52):
            y = 24 + row * 20
            left = "[tiqur@nixos:~]$" if row == 0 else (
                "deleting" if row % 4 else "copying")
            path = "/nix/store/" + ("abcdefghijklmnopqrstuvwxyz0123456789" * (1 + row % 2))
            color = (120, 220, 130) if row == 0 else (
                (225, 225, 225) if row % 3 else (165, 165, 165))
            skew = row & 1
            cv2.putText(terminal, left, (24, y + skew), plain, 0.62, color, 1, cv2.LINE_AA)
            cv2.putText(terminal, f"'{path}'", (78, y - skew), plain, 0.62, color, 1, cv2.LINE_AA)
        cv2.putText(terminal, "52 files deleted, 3.4 GiB freed",
                    (24, 24 + 52 * 20), plain, 0.62, (120, 220, 130), 1, cv2.LINE_AA)
        yield "terminal-log", terminal
    dense = np.full((900, 1440, 3), 255, np.uint8)
    for row in range(15):
        for col in range(20):
            text(dense, "Menu", 14 + col * 71, 30 + row * 58, 0.43)
    yield "dense-hints", dense
    # Borderless rounded pills exercise contour-ladder recovery of controls
    # without rectangular card borders; the ellipse has no pure H/V seed.
    pills = np.full((900, 1440, 3), 246, np.uint8)

    def pill(x, y, w, h, color, ink, label):
        radius = 4
        cv2.rectangle(pills, (x + radius, y), (x + w - radius, y + h), color, -1)
        cv2.rectangle(pills, (x, y + radius), (x + w, y + h - radius), color, -1)
        for cx, cy in ((x + radius, y + radius), (x + w - radius, y + radius),
                       (x + radius, y + h - radius), (x + w - radius, y + h - radius)):
            cv2.circle(pills, (cx, cy), radius, color, -1)
        text(pills, label, x + 12, y + h // 2 + 5, 0.52, ink, 2)

    pill(90, 120, 190, 36, (190, 110, 40), (30, 104, 30), "Continue")
    pill(330, 124, 150, 34, (50, 140, 70), (10, 134, 10), "Apply")
    pill(540, 118, 220, 38, (180, 70, 140), (180, 64, 10), "Open all settings")
    cv2.ellipse(pills, (195, 250), (95, 18), 0, 0, 360, (60, 130, 190), -1)
    text(pills, "Ellipse action", 120, 256, 0.52, (60, 124, 20), 2)
    yield "rounded-pills", pills
    # Two media rows exercise peer-shelf width/height normalization, gap filling
    # and cross-row 16:9 completion from a single-card row. Textured candidates
    # have no card border so only shelf completion may retain them.
    shelf = np.full((900, 1440, 3), 245, np.uint8)

    def texture(x, y, w, h):
        patch = cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (5, 5), 0)
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 255, -1)
        roi = shelf[y:y + h, x:x + w]
        roi[mask > 0] = patch[mask > 0]

    def card(x, y, w, h, label):
        patch = cv2.GaussianBlur(rng.integers(0, 256, (h - 4, w - 4, 3), dtype=np.uint8), (5, 5), 0)
        shelf[y + 2:y + h - 2, x + 2:x + w - 2] = patch
        cv2.rectangle(shelf, (x, y), (x + w, y + h), (20, 20, 20), 2)
        text(shelf, label, x, y + h + 26, 0.55)

    card(80, 100, 220, 142, "First video")
    card(340, 100, 240, 142, "Second video")
    card(600, 100, 230, 142, "Third video")
    texture(860, 100, 240, 142)
    texture(1120, 100, 240, 142)
    text(shelf, "Fourth video", 860, 268, 0.55)
    text(shelf, "Fifth video", 1120, 268, 0.55)
    card(80, 400, 240, 135, "Single-card row")
    texture(340, 400, 240, 135)
    texture(600, 400, 240, 135)
    yield "media-shelf", shelf
    # Portrait input must not inflate a narrow crop to a 1600px-wide working image.
    portrait = np.full((1200, 400, 3), 246, np.uint8)
    for row in range(18):
        text(portrait, "Narrow words in a portrait view", 20, 32 + row * 46, 0.43)
    yield "portrait-small-text", portrait
    # Closely spaced small-font lines exercise direct-ink valleys and line bands.
    tiny = np.full((900, 1440, 3), 38, np.uint8)
    for row in range(12):
        text(tiny, "Small glyph chain with words and spacing", 140, 110 + row * 16,
             0.31, (158, 158, 158))
    yield "tiny-glyph-chain", tiny


@dataclass(frozen=True)
class Options:
    include_text: bool = True
    turbo: bool = False
    collapse_paragraphs: bool = True
    min_score: float = 0.45

    def flags(self):
        result = ["--min-score", repr(self.min_score)]
        if not self.include_text:
            result.append("--interactive-only")
        if self.turbo:
            result.append("--turbo")
        if not self.collapse_paragraphs:
            result.append("--no-collapse-paragraphs")
        return result

    def detector(self, reference):
        return reference.ViselDetector(min_score=self.min_score, include_text=self.include_text,
                                       fast_half_res=self.turbo,
                                       collapse_paragraphs=self.collapse_paragraphs)


def first_difference(expected, actual, path="$"):
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return (f"{path}: missing keys={sorted(expected.keys() - actual.keys())}, "
                    f"extra keys={sorted(actual.keys() - expected.keys())}")
        for key in expected:
            difference = first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
    elif isinstance(expected, list) and isinstance(actual, list):
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        if len(expected) != len(actual):
            return f"{path}: length Python={len(expected)}, Rust={len(actual)}"
    elif type(expected) is not type(actual):
        # JSON has one number category; 1 and 1.0 are the same parsed number.
        if (type(expected) in (int, float) and type(actual) in (int, float)
                and expected == actual):
            return None
        return f"{path}: Python={expected!r} ({type(expected).__name__}), Rust={actual!r} ({type(actual).__name__})"
    elif expected != actual:
        return f"{path}: Python={expected!r}, Rust={actual!r}"
    return None


class Harness:
    def __init__(self, binary, reference):
        self.binary = binary
        self.reference = reference
        self.failures = []
        self.counts = Counter()
        self.coverage = Counter()
        self.kinds = Counter()
        self.levels = Counter()
        self.hint_lengths = Counter()
        self.cases = []

    def fail(self, case, detail):
        self.failures.append({"case": case, "detail": detail})
        if len(self.failures) <= 12:
            print(f"FAIL {case}: {detail}", flush=True)

    def invoke(self, image_path, options, directory, extra=()):
        output = directory / "rust.json"
        rendering = directory / "rust.png"
        # Never accept stale outputs from a previous invocation or previous run.
        output.unlink(missing_ok=True)
        rendering.unlink(missing_ok=True)
        command = [str(self.binary), str(image_path), "--json", str(output),
                   "-o", str(rendering), *options.flags(), *extra]
        write_json(directory / "command.json", command)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   OPENCV_FOR_THREADS_NUM="1")
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, env=env)
        (directory / "stdout.txt").write_text(result.stdout)
        (directory / "stderr.txt").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"CLI exit {result.returncode}: {result.stderr[-1500:]} {result.stdout[-500:]}")
        return json.loads(output.read_text()), rendering

    def pixels(self, expected, actual_path, directory, case):
        actual = cv2.imread(str(actual_path), cv2.IMREAD_UNCHANGED)
        self.counts["render_checks"] += 1
        if actual is None or actual.shape != expected.shape or actual.dtype != expected.dtype:
            self.fail(case, f"render shape/dtype Python={expected.shape}/{expected.dtype}, "
                      f"Rust={None if actual is None else (actual.shape, str(actual.dtype))}")
            save_image(directory / "python.png", expected)
            return
        mask = np.any(expected != actual, axis=2)
        if np.any(mask):
            y, x = np.argwhere(mask)[0]
            delta = cv2.absdiff(expected, actual)
            save_image(directory / "python.png", expected)
            save_image(directory / "pixel-diff.png", delta)
            self.fail(case, f"render: {np.count_nonzero(mask)} pixels differ; first (x={x}, y={y}) "
                      f"Python={expected[y, x].tolist()}, Rust={actual[y, x].tolist()}, max delta={delta.max()}")

    def record_coverage(self, targets):
        self.coverage["targets"] += len(targets)
        self.coverage["empty_detections"] += not targets
        self.hint_lengths.update(len(t.hint) for t in targets)

        def visit(nodes, depth):
            for node in nodes:
                self.kinds[node["kind"]] += 1
                self.levels[node["level"]] += 1
                children = node.get("children", [])
                self.coverage["parents"] += bool(children)
                self.coverage["nested_parents"] += bool(children) and depth > 0
                self.coverage["children"] += len(children)
                self.coverage["missing_children_key"] += "children" not in node
                visit(children, depth + 1)
        visit([vars(t) for t in targets], 0)

    def check(self, name, image, image_path, options, index, drill=False):
        case = f"{name}/{index:03d}"
        directory = ARTIFACTS / name / f"{index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        targets, _ = options.detector(self.reference).detect(image)
        expected = [vars(t) for t in targets]
        write_json(directory / "python.json", expected)
        write_json(directory / "options.json", asdict(options))
        self.record_coverage(targets)
        self.counts["detection_checks"] += 1
        self.cases.append({"case": case, "options": asdict(options), "targets": len(targets)})
        hints = self.reference.generate_vimium_hints(len(targets))
        if [t.hint for t in targets] != hints:
            self.fail(case, "Python target hints do not match its hints API")
        try:
            actual, rendering = self.invoke(image_path, options, directory)
            difference = first_difference(expected, actual)
            if difference:
                self.fail(case, f"JSON (Python targets={len(expected)}, "
                          f"Rust targets={len(actual) if isinstance(actual, list) else 'not a list'}): {difference}")
            self.counts["hint_checks"] += 1
            self.pixels(self.reference.render_vimium_overlay(image, targets), rendering, directory, case)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            self.fail(case, str(error))
        if not drill:
            return
        # Cover each distinct hierarchy type, plus a no-children leaf and an
        # unknown hint (CLI fallback), without launching once for every target.
        selected = {}
        for target in targets:
            key = (target.kind, target.level, bool(target.children),
                   any(c.get("children") for c in target.children))
            selected.setdefault(key, target)
        parents = list(selected.values())
        for number, parent in enumerate([*parents, None]):
            subdir = directory / f"drill-{number:02d}"
            subdir.mkdir(exist_ok=True)
            hint = parent.hint.lower() if parent else "NOT-A-HINT"
            try:
                actual, rendering = self.invoke(image_path, options, subdir, ["--drill-down", hint])
                self.counts["drill_checks"] += 1
                difference = first_difference(expected, actual)
                if difference:
                    self.fail(f"{case}/drill-{number}", f"drill changed top-level JSON: {difference}")
                if parent:
                    leaves = self.reference.drill_down(parent, to_words=True)
                    write_json(subdir / "python-drill-targets.json", [vars(t) for t in leaves])
                    self.coverage["drill_parent" if parent.children else "drill_leaf"] += 1
                    self.coverage["drill_paragraph"] += parent.level == "paragraph"
                    rendered = self.reference.render_drilldown_overlay(image, parent, to_words=True)
                else:
                    self.coverage["drill_unknown_hint"] += 1
                    rendered = self.reference.render_vimium_overlay(image, targets)
                self.pixels(rendered, rendering, subdir, f"{case}/drill-{number}")
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                self.fail(f"{case}/drill-{number}", str(error))

    def guard_coverage(self, quick):
        required = {"targets", "children", "parents", "nested_parents", "empty_detections",
                    "missing_children_key", "drill_parent", "drill_leaf", "drill_paragraph", "drill_unknown_hint"}
        for name in sorted(required):
            if not self.coverage[name]:
                self.fail("coverage", f"no {name} exercised")
        for kind in ("frame", "desktop_icon", "icon", "inline_link", "text_block", "word", "line"):
            if not self.kinds[kind]:
                self.fail("coverage", f"no {kind} produced by reference fixtures")
        if not quick and not self.kinds["thumbnail"]:
            self.fail("coverage", "no thumbnail produced by reference fixtures")
        if not self.levels["paragraph"]:
            self.fail("coverage", "no paragraph produced by reference fixtures")
        for length in (1, 2, 3):
            if not self.hint_lengths[length]:
                self.fail("coverage", f"no {length}-character hints exercised through CLI")

    def benchmark(self, images, runs, warmup):
        results = []
        for name, image in images:
            for turbo in (False, True):
                options = Options(turbo=turbo)
                detector = options.detector(self.reference)
                for _ in range(warmup):
                    detector.detect(image)
                samples = []
                stage_samples = {stage: [] for stage in STAGES}
                for _ in range(runs):
                    start = time.perf_counter_ns()
                    targets, timings = detector.detect(image)
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                    for stage in STAGES:
                        stage_samples[stage].append(timings[f"{stage}_ms"])
                python = {"median_ms": statistics.median(samples), "mean_ms": statistics.mean(samples),
                          "min_ms": min(samples), "max_ms": max(samples), "samples_ms": samples,
                          "targets": len(targets),
                          "stages": {stage: timing_summary(values)
                                     for stage, values in stage_samples.items()}}
                directory = ARTIFACTS / name / f"benchmark-{'turbo' if turbo else 'normal'}"
                directory.mkdir(parents=True, exist_ok=True)
                output = directory / "benchmark.json"
                output.unlink(missing_ok=True)
                entry = {"fixture": name, "width": image.shape[1], "height": image.shape[0],
                         "options": asdict(options), "python": python}
                try:
                    self.invoke(ARTIFACTS / name / "input.png", options, directory,
                                ["--benchmark", str(runs), "--warmup", str(warmup),
                                 "--benchmark-json", str(output)])
                    rust = json.loads(output.read_text())
                    for field in ("median_ms", "mean_ms", "min_ms", "max_ms", "samples_ms", "targets"):
                        if field not in rust:
                            raise ValueError(f"benchmark missing {field}")
                    rs = rust["samples_ms"]
                    if len(rs) != runs or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in rs):
                        raise ValueError("invalid benchmark samples or sample count")
                    for field, value in (("median_ms", statistics.median(rs)),
                                         ("mean_ms", statistics.mean(rs)),
                                         ("min_ms", min(rs)), ("max_ms", max(rs))):
                        if not math.isclose(rust[field], value, rel_tol=1e-9, abs_tol=1e-9):
                            raise ValueError(f"inconsistent benchmark {field}: {rust[field]} vs {value}")
                    if rust["targets"] != len(targets):
                        raise ValueError(f"benchmark target count Python={len(targets)}, Rust={rust['targets']}")
                    stage_speedup = {}
                    for stage in STAGES:
                        try:
                            values = rust["stages"][stage]["samples_ms"]
                        except (KeyError, TypeError) as error:
                            raise ValueError(f"benchmark missing stage {stage}") from error
                        if len(values) != runs or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
                            raise ValueError(f"invalid {stage} benchmark samples or sample count")
                        summary = timing_summary(values)
                        for field in ("median_ms", "mean_ms", "min_ms", "max_ms"):
                            if not math.isclose(rust["stages"][stage][field], summary[field],
                                                rel_tol=1e-9, abs_tol=1e-9):
                                raise ValueError(f"inconsistent {stage} benchmark {field}")
                        py_median = python["stages"][stage]["median_ms"]
                        stage_speedup[stage] = py_median / summary["median_ms"] if summary["median_ms"] else None
                    entry["rust"] = rust
                    entry["median_speedup"] = python["median_ms"] / rust["median_ms"] if rust["median_ms"] else None
                    entry["stage_speedup"] = stage_speedup
                    stages = ", ".join(
                        f"{stage} {python['stages'][stage]['median_ms']:.1f}/{rust['stages'][stage]['median_ms']:.1f}"
                        for stage in STAGES)
                    print(f"BENCH {name} turbo={turbo}: Python {python['median_ms']:.2f} ms, "
                          f"Rust {rust['median_ms']:.2f} ms | py/rust stages: {stages}", flush=True)
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, TypeError, KeyError) as error:
                    entry["error"] = str(error)
                    self.fail(f"benchmark/{name}/{turbo}", str(error))
                results.append(entry)
        write_json(ROOT / "target" / "benchmark.json", {
            "runs": runs, "warmup": warmup, "opencv_threads": cv2.getNumThreads(),
            "scope": "detection only, preloaded image, repeated in one process per configuration; per-stage medians included",
            "environment": environment(self.binary), "results": results})


def environment(binary):
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return {"seed": SEED, "python": sys.version, "opencv": cv2.__version__,
            "numpy": np.__version__, "platform": platform.platform(), "cpu": cpu,
            "logical_cpus": os.cpu_count(), "binary": str(binary),
            "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256((ROOT / "python.py").read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=ROOT / "target" / "release" / "univim")
    parser.add_argument("--quick", action="store_true", help="Reduced fixtures and scores; keeps all boolean combinations")
    parser.add_argument("--benchmark", action="store_true", help="Also run same-process detection-only benchmarks")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be positive and --warmup nonnegative")
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        print(f"UNRUN: executable not found: {binary}; build the Rust CLI first", file=sys.stderr)
        return 2
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    # Generated artifacts ignore themselves even in a checkout without a root ignore file.
    (ARTIFACTS / ".gitignore").write_text("*\n")
    reference = load_reference()
    harness = Harness(binary, reference)
    started = time.perf_counter()
    benchmark_images = []
    scores = (0.45, 0.95) if args.quick else (0.45, 0.84, 0.95)
    matrix = [Options(*values) for values in itertools.product((True, False), (False, True), (True, False), scores)]
    manifest = []
    for name, image in fixtures(args.quick):
        directory = ARTIFACTS / name
        directory.mkdir(exist_ok=True)
        image_path = directory / "input.png"
        save_image(image_path, image)
        manifest.append({"name": name, "width": image.shape[1], "height": image.shape[0],
                         "sha256": hashlib.sha256(image.tobytes()).hexdigest()})
        if name in ("ui-light-native", "dense-hints") or name.endswith("-2561"):
            benchmark_images.append((name, image))
        before = len(harness.failures)
        for index, options in enumerate(matrix):
            harness.check(name, image, image_path, options, index,
                          drill=options == Options() and name in ("ui-light-native", "ui-dark-native", "dense-hints", "blank-light", "terminal-log"))
        if name == "ui-light-native" and not args.quick:
            # Inclusive filtering at every detector score and its nearest f64 neighbors.
            boundaries = (0.0, 0.78, 0.80, 0.84, 0.85, 0.91, 0.92, 0.93, 0.94, 0.95, 1.0)
            for index, score in enumerate(sorted({value for b in boundaries for value in (
                    math.nextafter(b, -math.inf), b, math.nextafter(b, math.inf))})):
                harness.check(name, image, image_path, Options(min_score=score), len(matrix) + index)
        print(f"{name}: {len(matrix)} option combinations, "
              f"{len(harness.failures) - before} failures", flush=True)
    harness.guard_coverage(args.quick)
    if args.benchmark:
        harness.benchmark(benchmark_images, args.runs, args.warmup)
    report = {"passed": not harness.failures, "quick": args.quick,
              "elapsed_s": time.perf_counter() - started, "environment": environment(binary),
              "fixtures": manifest, "counts": dict(harness.counts),
              "coverage": dict(harness.coverage), "kinds": dict(harness.kinds),
              "levels": dict(harness.levels), "hint_lengths": dict(harness.hint_lengths),
              "cases": harness.cases, "failures": harness.failures,
              "api_limits": ["arbitrary hints(n) counts and custom alphabets are not exposed by CLI",
                             "drill_down(to_words=False) is not exposed by CLI",
                             "word drill_down checked via exact overlay pixels; CLI JSON remains top-level"]}
    write_json(ARTIFACTS / "report.json", report)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {dict(harness.counts)}, "
          f"{len(harness.failures)} failures in {report['elapsed_s']:.1f}s")
    print(f"Coverage: {dict(harness.coverage)}; hints: {dict(harness.hint_lengths)}")
    print(f"Report: {ARTIFACTS / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
