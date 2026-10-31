#!/usr/bin/env python3
"""
visel_detector.py (v6) — Vision-Refined Structure-Aware, Hierarchical Drill-Down UI & Text Detector
====================================================================================================
Fast single-pass + 1D recursive refinement + container-scoped structural rail detector for
universal Vimium navigation across interactive controls, text hierarchies, media thumbnails,
cards, and desktop/sidebar compounds.

Key Capabilities & v6 Vision-Guided Upgrades:
  1. Single-Character UI Section Header & Control Recovery (`#`, `B`, `C`, `E`, `?`, `+`, `1..9`):
     - Recovers high-contrast single-character UI buttons and alphabet section headers sitting on
       clean flat UI chrome (`local_flat >= 0.78`, `max_g >= 110`) with uniform rail-aligned
       clickable hitboxes while suppressing specks on textured backgrounds.
  2. Large Display Heading Integrity (Scale-Proportional Barrier Kernels + Glyph-Loop Guard):
     - Scales structural barrier kernels (`h_bar = max(64, int(92 * s))`, `v_bar = max(48, int(68 * s))`)
       so vertical letter stems (`d`, `i`, `t`, `l`, `g`, `M`) in large `28-64px` display headings
       (`Categories`, `Editors' Choice`, `Edit pictures without leaving Word`, `OpenStreetMap`) and
       `76x88` app icons (`Things 3`, `Pixelmator Pro`) are never erased by `v_bar`/`h_bar`.
     - Prevents closed counter loops inside large bold letters from being misclassified as small
       button frames by checking adjacent left/right glyph stroke ink (`l_ink`, `r_ink`).
     - Scales horizontal word-split thresholds (`gap_thresh = max(18, int(rh * 0.85))`) and restricts
       `/` punctuation-separator splitting to standard text heights (`rh <= int(22 * s)`), keeping
       multi-word display headings unified while storing individual words in `target.children`.
  3. Complete Horizontal & Vertical `[Icon + Label]` Compound Fusion + Inner Frame Children:
     - Pairs square-ish icons/frames across sidebars, 3-column App Store category grids and app
       cards (`12_macos`), Control Center rows (`15_macos`), dark admin sidebars (`20_shop`), and
       Windows 10 Live Tiles (`03_windows`) with their adjacent labels and stacked subtitles.
     - Enforces the geometric invariant `fh >= rh * 1.22` and horizontal-barrier checks in vertical
       `[Icon + Bottom Caption]` pairing so stacked numeric dropdown rows (`10` and `10.5`, or `72`
       and `Page 1 of 1` in `11_macos`) never fuse across rows.
     - Preserves inner icons and text labels inside closed `frame` buttons/inputs as drill-down
       `children` rather than discarding them.
  4. Step 8 — Rounded-Corner / Borderless Thumbnail & Map/Photo-Interior Coherence Masking:
     - Uses block texture/flatness coherence (`outer_flat`, `flat_r`, `tb_r`, `lr_r`, and `max_g`)
       to reject false road/river contour frames and terrain text on map canvases (`26_tool`) and
       photographic backgrounds while removing rounded-corner arc artifacts on translucent modals.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import argparse
import json
import time
import cv2
import numpy as np

cv2.setNumThreads(1)


@dataclass
class UITarget:
    x: int
    y: int
    w: int
    h: int
    score: float
    kind: str  # "frame", "thumbnail", "desktop_icon", "inline_link", "text_link", "icon", "text_block"
    hint: str = ""
    level: str = "control"  # "control", "paragraph", "line", "word"
    children: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def area(self) -> int:
        return self.w * self.h


def generate_vimium_hints(n: int, alphabet: str = "sadfjklewcmpgh") -> List[str]:
    """Generate unique, prefix-free ergonomic home-row Vimium hint labels for any target count."""
    if n <= 0:
        return []
    alpha = alphabet.upper()
    k = len(alpha)
    if n <= k:
        return [alpha[i] for i in range(n)]
    if n <= k * k:
        return [alpha[i // k] + alpha[i % k] for i in range(n)]
    extra = n - k * k
    n_expand = (extra + (k - 2)) // (k - 1)
    two_char = [alpha[i // k] + alpha[i % k] for i in range(k * k)]
    hints = two_char[: k * k - n_expand]
    for prefix in two_char[k * k - n_expand :]:
        for ch in alpha:
            hints.append(prefix + ch)
            if len(hints) >= n:
                return hints
    return hints[:n]


def drill_down(target: UITarget, to_words: bool = True) -> List[UITarget]:
    """
    Instant 0.0 ms hierarchical drill-down when the user selects a target with a modifier key.
    - For a `paragraph` (or `line` / `desktop_icon` / `frame`), returns either its immediate
      `children` (`to_words=False`) or flattens down to individual `word` / leaf targets (`to_words=True`).
    """
    if not target.children:
        return [target]
    leaves: List[Dict[str, Any]] = []
    if to_words:
        for ch in target.children:
            sub = ch.get("children", [])
            if sub:
                leaves.extend(sub)
            else:
                leaves.append(ch)
    else:
        leaves = list(target.children)

    out = [
        UITarget(
            x=int(c["x"]),
            y=int(c["y"]),
            w=int(c["w"]),
            h=int(c["h"]),
            score=float(c.get("score", 0.85)),
            kind=str(c.get("kind", "word")),
            hint=str(c.get("hint", "")),
            level=str(c.get("level", c.get("kind", "word"))),
            children=list(c.get("children", [])),
        )
        for c in leaves
    ]
    for t, h_str in zip(out, generate_vimium_hints(len(out))):
        t.hint = h_str
    return out


class ViselDetector:
    def __init__(
        self,
        min_score: float = 0.45,
        include_text: bool = True,
        fast_half_res: bool = False,
        collapse_paragraphs: bool = True,
    ):
        self.min_score = min_score
        self.include_text = include_text
        self.fast_half_res = fast_half_res
        self.collapse_paragraphs = collapse_paragraphs
        self.k_dx = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        self.k_dy = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        self.k_d95 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        self.k_coarse = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 6))
        self.k_coarse_hi = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 8))
        self.k_coarse_h = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 3))
        self.k_bar_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.one_12 = np.ones(12, dtype=np.uint8)

    def detect(
        self,
        bgr: np.ndarray,
        include_text: bool = None,
        fast_half_res: bool = None,
        collapse_paragraphs: bool = None,
    ) -> Tuple[List[UITarget], Dict[str, float]]:
        if include_text is None:
            include_text = self.include_text
        if fast_half_res is None:
            fast_half_res = self.fast_half_res
        if collapse_paragraphs is None:
            collapse_paragraphs = self.collapse_paragraphs

        t0 = time.perf_counter()
        orig_H, orig_W = bgr.shape[:2]

        scale_back = 1.0
        if fast_half_res and orig_W > 1600:
            work_W = min(1280, orig_W >> 1)
            work_H = max(1, int(round(orig_H * (work_W / float(orig_W)))))
            bgr = cv2.resize(bgr, (work_W, work_H), interpolation=cv2.INTER_AREA)
            scale_back = orig_W / float(work_W)
        elif not fast_half_res and orig_W > 2560:
            work_W = 2560
            work_H = max(1, int(round(orig_H * (2560.0 / float(orig_W)))))
            bgr = cv2.resize(bgr, (work_W, work_H), interpolation=cv2.INTER_AREA)
            scale_back = orig_W / 2560.0
        elif orig_W < 900:
            work_W = 960
            work_H = max(1, int(round(orig_H * (960.0 / float(orig_W)))))
            bgr = cv2.resize(bgr, (work_W, work_H), interpolation=cv2.INTER_CUBIC)
            scale_back = orig_W / 960.0

        H, W = bgr.shape[:2]
        s = float(np.clip(np.sqrt((W * H) / float(1440 * 900)), 0.65, 2.0))

        # Stage 1: Gradients & Scale-Proportional Structural Barriers
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(bgr)
        chroma = cv2.subtract(cv2.max(cv2.max(b, g), r), cv2.min(cv2.min(b, g), r))
        blue_excess = cv2.subtract(b, r)

        dx_gray = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, self.k_dx)
        dy_gray = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, self.k_dy)
        dx = cv2.max(dx_gray, cv2.morphologyEx(chroma, cv2.MORPH_GRADIENT, self.k_dx))
        dy = cv2.max(dy_gray, cv2.morphologyEx(chroma, cv2.MORPH_GRADIENT, self.k_dy))
        grad_max = cv2.max(dx, dy)

        ortho_thr = 2 if (W * H <= 600000) else 4
        k_hline = cv2.getStructuringElement(cv2.MORPH_RECT, (38, 1))
        k_hseg = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        k_vline_short = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 24))
        k_hzone = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

        _, dy_ge5 = cv2.threshold(dy, ortho_thr, 255, cv2.THRESH_BINARY)
        _, dx_le2 = cv2.threshold(dx, 2, 255, cv2.THRESH_BINARY_INV)
        h_pure = cv2.bitwise_and(dy_ge5, dx_le2)
        h_bar = cv2.morphologyEx(h_pure, cv2.MORPH_OPEN, k_hline)
        h_zone = cv2.dilate(cv2.morphologyEx(h_pure, cv2.MORPH_OPEN, k_hseg), k_hzone)

        _, dx_ge5 = cv2.threshold(dx, ortho_thr, 255, cv2.THRESH_BINARY)
        _, dy_le2 = cv2.threshold(dy, 2, 255, cv2.THRESH_BINARY_INV)
        v_pure = cv2.bitwise_and(dx_ge5, dy_le2)
        v_raw = cv2.morphologyEx(v_pure, cv2.MORPH_OPEN, k_vline_short)
        v_cnts, _ = cv2.findContours(v_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_bar = np.zeros_like(v_raw)
        min_v_long = max(44, int(58 * s))
        for vc in v_cnts:
            vx, vy, vw, vh = cv2.boundingRect(vc)
            if vh >= min_v_long or cv2.countNonZero(h_zone[vy : vy + vh, vx : vx + vw]) > 0:
                cv2.drawContours(v_bar, [vc], -1, 255, -1)

        t_sig = time.perf_counter()

        # Stage 2: Closed-Loop Control Frame, Tile Container & Media Thumbnail Recovery (with Step 8 Coherence Filter)
        _, flat_pix0 = cv2.threshold(grad_max, 0, 255, cv2.THRESH_BINARY_INV)
        _, flat_pix = cv2.threshold(grad_max, 3, 255, cv2.THRESH_BINARY_INV)

        hv_small = cv2.resize(
            cv2.bitwise_or(h_pure, v_pure),
            (max(1, W >> 3), max(1, H >> 3)),
            interpolation=cv2.INTER_AREA,
        )
        _, hv_small = cv2.threshold(hv_small, 1, 255, cv2.THRESH_BINARY)
        hv_seed = cv2.resize(cv2.dilate(hv_small, self.k_d95), (W, H), interpolation=cv2.INTER_NEAREST)

        border_thr = 2 if (W * H <= 600000) else 4
        _, border_pix = cv2.threshold(grad_max, border_thr, 255, cv2.THRESH_BINARY)
        cnts_list = list(
            cv2.findContours(cv2.bitwise_and(border_pix, hv_seed), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        )

        strip_w = min(W, int(260 * s))
        _, border_hi = cv2.threshold(grad_max[:, :strip_w], 12, 255, cv2.THRESH_BINARY)
        cnts_hi, _ = cv2.findContours(border_hi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_hi:
            if len(c) >= 4:
                x, y, w, h = cv2.boundingRect(c)
                if 28 <= w <= int(72 * s) and 28 <= h <= int(72 * s) and abs(w - h) <= 6:
                    if cv2.contourArea(c) >= w * h * 0.72:
                        cnts_list.append(c)

        min_frame_h = max(16, int(16 * s))
        max_frame_h = max(46, int(58 * s))
        min_frame_w = max(22, int(24 * s))
        max_frame_w = max(440, int(W * 0.85))
        min_thumb_h = max(48, int(52 * s))
        max_thumb_h = max(255, int(H * 0.52))
        min_thumb_w = max(64, int(68 * s))
        max_thumb_w = max(420, int(W * 0.48))

        raw_frames = []
        raw_thumbs = []
        tile_containers = []
        panel_corners = []

        for c in cnts_list:
            if len(c) < 4:
                continue
            x, y, w, h = cv2.boundingRect(c)
            w_cap = max_frame_w if h <= int(42 * s) else int(265 * s)
            if min_frame_h <= h <= max_frame_h and min_frame_w <= w <= w_cap:
                area = cv2.contourArea(c)
                if area >= w * h * 0.70:
                    flat_r = cv2.countNonZero(
                        flat_pix[y + 2 : y + h - 2, x + 2 : x + w - 2]
                    ) / float(max(1, (w - 4) * (h - 4)))
                    if flat_r < 0.28:
                        continue
                    if float(grad_max[y : y + h, x : x + w].max()) < 48:
                        continue
                    tb_s = cv2.countNonZero(
                        h_pure[y : y + 4, x + 3 : x + w - 3]
                    ) + cv2.countNonZero(h_pure[max(0, y + h - 4) : y + h, x + 3 : x + w - 3])
                    lr_s = cv2.countNonZero(
                        v_pure[y + 3 : y + h - 3, x : x + 4]
                    ) + cv2.countNonZero(v_pure[y + 3 : y + h - 3, max(0, x + w - 4) : x + w])
                    tb_r = tb_s / float(max(1, w - 6))
                    is_pill = tb_s >= max(12, int((w - h) * 0.75)) and lr_s >= 4
                    if not is_pill and (
                        tb_s < max(8, int((w - 8) * 0.45))
                        or (h >= 20 and w >= 55 and lr_s < max(5, int((h - 12) * 0.35)))
                    ):
                        continue

                    if w <= max(48, int(62 * s)) and h <= max(42, int(54 * s)) and tb_r < 1.80:
                        l_ink = int((grad_max[y + 2 : y + h - 2, max(0, x - 6) : x] >= 25).sum())
                        r_ink = int((grad_max[y + 2 : y + h - 2, x + w : min(W, x + w + 6)] >= 25).sum())
                        if l_ink >= 22 or (l_ink >= 12 and r_ink >= 12):
                            continue
                        ox1, oy1 = max(0, x - 10), max(0, y - 10)
                        ox2, oy2 = min(W, x + w + 10), min(H, y + h + 10)
                        outer_area = max(1, (ox2 - ox1) * (oy2 - oy1) - w * h)
                        outer_flat = (
                            cv2.countNonZero(flat_pix[oy1:oy2, ox1:ox2])
                            - cv2.countNonZero(flat_pix[y : y + h, x : x + w])
                        ) / float(outer_area)
                        if outer_flat < 0.65:
                            continue

                    if w > int(85 * s):
                        iv = v_bar[y + 3 : y + h - 3, x + int(24 * s) : max(x + int(25 * s), x + w - int(38 * s))]
                        if iv.size > 0 and np.any(np.sum(iv > 0, axis=0) >= h - 5):
                            continue
                    raw_frames.append([x, y, w, h, 0.94, "frame", "control", []])
            elif min_thumb_h <= h <= max_thumb_h and min_thumb_w <= w <= max_thumb_w:
                area = cv2.contourArea(c)
                if area >= w * h * 0.78:
                    tb_s = cv2.countNonZero(
                        h_pure[y : y + 4, x + 6 : x + w - 6]
                    ) + cv2.countNonZero(h_pure[max(0, y + h - 4) : y + h, x + 6 : x + w - 6])
                    lr_s = cv2.countNonZero(
                        v_pure[y + 6 : y + h - 6, x : x + 4]
                    ) + cv2.countNonZero(v_pure[y + 6 : y + h - 6, max(0, x + w - 4) : x + w])
                    if tb_s < (w - 12) * 0.60 or lr_s < (h - 12) * 0.60:
                        continue
                    fr0 = cv2.countNonZero(
                        flat_pix0[y + 3 : y + h - 3, x + 3 : x + w - 3]
                    ) / float(max(1, (w - 6) * (h - 6)))
                    fr3 = cv2.countNonZero(
                        flat_pix[y + 3 : y + h - 3, x + 3 : x + w - 3]
                    ) / float(max(1, (w - 6) * (h - 6)))
                    g_std = float(gray[y + 3 : y + h - 3, x + 3 : x + w - 3].std())
                    ar = w / float(h)
                    if fr0 < 0.60 and fr3 < 0.78 and g_std >= 24.0 and 0.75 <= ar <= 2.8:
                        raw_thumbs.append([x, y, w, h, 0.93, "thumbnail", "control", []])
                    elif fr3 >= 0.72:
                        if 0.85 <= ar <= 1.18 and int(70 * s) <= w <= int(180 * s) and tb_s >= (w - 8) * 1.1 and lr_s >= (h - 8) * 1.1:
                            tile_containers.append((x, y, w, h))
                        if w >= int(160 * s) and h >= int(120 * s) and tb_s >= (w - 8) * 1.1 and lr_s >= (h - 8) * 1.1:
                            panel_corners.extend([(x, y), (x + w, y), (x, y + h), (x + w, y + h)])

        filtered_thumbs = []
        for i, t in enumerate(raw_thumbs):
            tx, ty, tw, th = t[:4]
            t_area = tw * th
            if any(
                i != j
                and t_area < 0.38 * (o[2] * o[3])
                and max(0, min(tx + tw, o[0] + o[2]) - max(tx, o[0]))
                * max(0, min(ty + th, o[1] + o[3]) - max(ty, o[1]))
                >= 0.80 * t_area
                for j, o in enumerate(raw_thumbs)
            ):
                continue
            filtered_thumbs.append(t)
        filtered_thumbs.sort(key=lambda t: t[2] * t[3])
        thumbs = []
        for t in filtered_thumbs:
            tx, ty, tw, th = t[:4]
            t_area = tw * th
            if any(
                max(0, min(tx + tw, kx + kw) - max(tx, kx))
                * max(0, min(ty + th, ky + kh) - max(ty, ky))
                >= 0.45 * min(t_area, kw * kh)
                for kx, ky, kw, kh, *_ in thumbs
            ):
                continue
            thumbs.append(t)

        raw_frames.sort(key=lambda f: f[2] * f[3], reverse=True)
        frames = []
        for f in raw_frames:
            fx, fy, fw, fh = f[:4]
            f_area = fw * fh
            if any(
                max(0, min(fx + fw, tx + tw) - max(fx, tx))
                * max(0, min(fy + fh, ty + th) - max(fy, ty))
                >= 0.5 * f_area
                for tx, ty, tw, th, *_ in thumbs
            ):
                continue
            absorbed = False
            for k in frames:
                kx, ky, kw, kh = k[:4]
                ix1, iy1 = max(fx, kx), max(fy, ky)
                ix2, iy2 = min(fx + fw, kx + kw), min(fy + fh, ky + kh)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter == 0:
                    continue
                iou = inter / float(f_area + kw * kh - inter)
                cov_f = inter / float(f_area)
                if iou >= 0.45 or cov_f >= 0.75:
                    if kw > int(320 * s) and fw <= int(220 * s) and fx >= kx + kw - fw - int(16 * s) and (fx - kx) >= int(120 * s):
                        k[2] = fx - kx - 2
                    else:
                        absorbed = True
                        break
            if not absorbed:
                frames.append(f)

        thumb_occ = np.zeros((H, W), dtype=np.uint8)
        frame_occ = np.zeros((H, W), dtype=np.uint8)
        barrier = cv2.bitwise_or(h_bar, v_bar)
        for t in thumbs:
            tx, ty, tw, th = t[:4]
            cv2.rectangle(thumb_occ, (tx, ty), (tx + tw, ty + th), 255, -1)
            cv2.rectangle(barrier, (tx, ty), (tx + tw, ty + th), 255, 2)
        for f in frames:
            fx, fy, fw, fh = f[:4]
            cv2.rectangle(frame_occ, (fx, fy), (fx + fw, fy + fh), 255, -1)
            cv2.rectangle(barrier, (fx, fy), (fx + fw, fy + fh), 255, 2)
        barrier_dil = cv2.dilate(barrier, self.k_bar_dil)
        barrier_inv = cv2.bitwise_not(barrier_dil)

        t_frames = time.perf_counter()

        # Stage 3: Coarse Proposal Generation
        _, stroke_bin = cv2.threshold(grad_max, 20, 255, cv2.THRESH_BINARY)
        stroke_clean = cv2.bitwise_and(stroke_bin, barrier_inv)

        _, stroke_01 = cv2.threshold(stroke_clean, 0, 1, cv2.THRESH_BINARY)
        _, blue_01 = cv2.threshold(blue_excess, 55, 1, cv2.THRESH_BINARY)

        if fast_half_res and scale_back == 1.0:
            H2, W2 = H >> 1, W >> 1
            stroke_h = cv2.resize(stroke_clean, (W2, H2), interpolation=cv2.INTER_AREA)
            _, stroke_h = cv2.threshold(stroke_h, 1, 255, cv2.THRESH_BINARY)
            barrier_inv_h = cv2.resize(barrier_inv, (W2, H2), interpolation=cv2.INTER_NEAREST)
            coarse_h = cv2.bitwise_and(
                cv2.morphologyEx(stroke_h, cv2.MORPH_CLOSE, self.k_coarse_h),
                barrier_inv_h,
            )
            c_cnts, _ = cv2.findContours(coarse_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            coarse_boxes = [
                (
                    max(0, (cx << 1) - 2),
                    max(0, (cy << 1) - 2),
                    min(W - max(0, (cx << 1) - 2), (cw << 1) + 4),
                    min(H - max(0, (cy << 1) - 2), (ch << 1) + 4),
                )
                for c in c_cnts
                for (cx, cy, cw, ch) in [cv2.boundingRect(c)]
                if (cw >= 3 and ch >= 3) or (cw >= 4 and ch >= 1) or (cw >= 2 and ch >= 7)
            ]
        else:
            k_c = self.k_coarse_hi if s >= 1.55 else self.k_coarse
            coarse_mask = cv2.bitwise_and(
                cv2.morphologyEx(stroke_clean, cv2.MORPH_CLOSE, k_c),
                barrier_inv,
            )
            c_cnts, _ = cv2.findContours(coarse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            coarse_boxes = [
                (cx, cy, cw, ch)
                for c in c_cnts
                for (cx, cy, cw, ch) in [cv2.boundingRect(c)]
                if (cw >= max(4, int(5 * s)) and ch >= max(5, int(6 * s))) or (max(5, int(7 * s)) <= cw <= max(28, int(34 * s)) and ch >= 1) or (cw >= 3 and max(10, int(13 * s)) <= ch <= max(36, int(42 * s)))
            ]

        t_coarse = time.perf_counter()

        # Stage 4: 1D Recursive Projection & Word Children
        refined_boxes = []
        inner_frame_children = []

        def refine_region(rx: int, ry: int, rw: int, rh: int, depth: int = 0):
            if rw < 3 or rh < 1 or depth > 10:
                return
            roi_s = stroke_01[ry : ry + rh, rx : rx + rw]
            row_proj = roi_s.sum(axis=1)
            r_idx = np.flatnonzero(row_proj)
            if len(r_idx) == 0:
                return
            y0, y1 = int(r_idx[0]), int(r_idx[-1]) + 1
            if y0 > 0 or y1 < rh:
                roi_s = roi_s[y0:y1, :]
                row_proj = row_proj[y0:y1]

            col_proj = roi_s.sum(axis=0)
            c_idx = np.flatnonzero(col_proj)
            if len(c_idx) == 0:
                return
            x0, x1 = int(c_idx[0]), int(c_idx[-1]) + 1
            if x0 > 0 or x1 < rw:
                roi_s = roi_s[:, x0:x1]
                col_proj = col_proj[x0:x1]

            rx, ry = rx + x0, ry + y0
            rw, rh = x1 - x0, y1 - y0
            if thumb_occ[ry + (rh >> 1), rx + (rw >> 1)] > 0:
                return
            inside_frame = bool(frame_occ[ry + (rh >> 1), rx + (rw >> 1)] > 0)

            is_micro_icon = (
                ((max(5, int(7 * s)) <= rw <= max(28, int(32 * s)) and 1 <= rh <= max(6, int(7 * s))) or (3 <= rw <= max(7, int(9 * s)) and max(10, int(13 * s)) <= rh <= max(34, int(40 * s))))
                and depth <= 2
            )
            is_single_char_ui = (4 <= rw <= 12 and 7 <= rh <= 22 and depth <= 2)
            if not is_micro_icon and not is_single_char_ui and (rw < 6 or rh < 6 or (rw <= 9 and rh <= 8)):
                return

            # Step 1: Vertical Multi-Line X-Y Split
            icon_guard_w = max(38, int(38 * s))
            icon_guard_h = max(36, int(36 * s))
            if rh >= 22:
                inner = row_proj[5 : rh - 5]
                v_thr = 7 if (rw >= 200 and int(row_proj.max()) >= 140) else 0
                if len(inner) > 0 and inner.min() <= v_thr:
                    zero_mask = inner <= v_thr
                    if rw <= icon_guard_w and rh <= icon_guard_h:
                        z_pad = np.concatenate(([False], zero_mask, [False]))
                        z_starts = np.flatnonzero(z_pad[1:] & ~z_pad[:-1])
                        z_ends = np.flatnonzero(~z_pad[1:] & z_pad[:-1])
                        max_z_run = int(np.max(z_ends - z_starts)) if len(z_starts) > 0 else 0
                    else:
                        max_z_run = 99
                    if max_z_run >= 6:
                        zero_rows = np.flatnonzero(zero_mask) + 5
                        split_y = int(zero_rows[len(zero_rows) >> 1])
                        next_y = split_y + 1
                        while next_y < rh - 4 and (
                            row_proj[next_y] <= v_thr or row_proj[next_y] >= rw * 0.75
                        ):
                            next_y += 1
                        refine_region(rx, ry, rw, split_y, depth + 1)
                        refine_region(rx, ry + next_y, rw, rh - next_y, depth + 1)
                        return

            col_blue = blue_01[ry : ry + rh, rx : rx + rw].sum(axis=0)
            is_ink = col_proj > 0
            padded = np.concatenate(([False], is_ink, [False]))
            starts = np.flatnonzero(padded[1:] & ~padded[:-1])
            ends = np.flatnonzero(~padded[1:] & padded[:-1])

            if len(starts) >= 2 and not is_micro_icon and not is_single_char_ui:
                if rh <= max(20, int(22 * s)):
                    min_sep_gap = max(5, int(rh * 0.35))
                    for idx_r in range(1, len(starts) - 1):
                        rs, re = int(starts[idx_r]), int(ends[idx_r])
                        run_w = re - rs
                        l_gap = rs - int(ends[idx_r - 1])
                        r_gap = int(starts[idx_r + 1]) - re
                        if l_gap >= min_sep_gap and r_gap >= min_sep_gap and run_w <= 8 and col_blue[rs:re].sum() == 0:
                            nz_y = np.flatnonzero(roi_s[:, rs:re].sum(axis=1))
                            run_h = int(nz_y[-1] - nz_y[0] + 1) if len(nz_y) > 0 else 0
                            if (run_w <= 4 and run_h <= 5) or (run_w <= 8 and l_gap + r_gap >= max(14, int(rh * 0.80))):
                                refine_region(rx, ry, int(ends[idx_r - 1]), rh, depth + 1)
                                refine_region(
                                    rx + int(starts[idx_r + 1]),
                                    ry,
                                    rw - int(starts[idx_r + 1]),
                                    rh,
                                    depth + 1,
                                )
                                return

                gaps = starts[1:] - ends[:-1]
                max_gi = int(np.argmax(gaps))
                max_gap = int(gaps[max_gi])
                left_w = int(ends[max_gi])
                right_x = int(starts[max_gi + 1])
                right_w = rw - right_x

                gap_thresh = max(18, int(rh * 0.85)) if rh >= 17 else max(13, int(rh * 0.95))
                if not (left_w <= 22 and max_gap <= 18) and max_gap >= gap_thresh:
                    refine_region(rx, ry, left_w, rh, depth + 1)
                    refine_region(rx + right_x, ry, right_w, rh, depth + 1)
                    return

            total_blue = int(col_blue.sum())
            if total_blue > 0 and not is_micro_icon and not is_single_char_ui:
                b_1d = (col_blue > 0).astype(np.uint8)
                b_closed = np.convolve(b_1d, self.one_12, mode="same") > 0
                b_pad = np.concatenate(([False], b_closed, [False]))
                b_starts = np.flatnonzero(b_pad[1:] & ~b_pad[:-1])
                b_ends = np.flatnonzero(~b_pad[1:] & b_pad[:-1])

                for bs, be in zip(b_starts, b_ends):
                    actual_b = np.flatnonzero(col_blue[bs:be])
                    if len(actual_b) == 0:
                        continue
                    bs_t = int(bs + actual_b[0])
                    be_t = int(bs + actual_b[-1]) + 1
                    while bs_t > 0 and is_ink[bs_t - 1]:
                        bs_t -= 1
                    while be_t < rw and is_ink[be_t]:
                        be_t += 1
                    bw_t = be_t - bs_t

                    has_left_neutral = bs_t >= 16 and col_proj[:bs_t].sum() >= 12
                    has_right_neutral = (rw - be_t) >= 16 and col_proj[be_t:].sum() >= 12
                    if bw_t >= 26 and (has_left_neutral or has_right_neutral):
                        if has_left_neutral:
                            refine_region(rx, ry, bs_t, rh, depth + 1)
                        refine_region(rx + bs_t, ry, bw_t, rh, depth + 1)
                        if has_right_neutral:
                            refine_region(rx + be_t, ry, rw - be_t, rh, depth + 1)
                        return

            pad_x1, pad_y1 = max(0, rx - 6), max(0, ry - 6)
            pad_x2, pad_y2 = min(W, rx + rw + 6), min(H, ry + rh + 6)
            local_flat = cv2.countNonZero(flat_pix[pad_y1:pad_y2, pad_x1:pad_x2]) / float(
                (pad_x2 - pad_x1) * (pad_y2 - pad_y1)
            )

            if rw <= int(110 * s) and rh <= int(110 * s) and local_flat >= 0.60:
                min_dx_mean = 1.6 / max(0.85, s * 0.85)
            else:
                min_dx_mean = 5.0
            if not is_micro_icon and float(cv2.mean(dx[ry : ry + rh, rx : rx + rw])[0]) < min_dx_mean:
                return

            _, max_g, _, _ = cv2.minMaxLoc(grad_max[ry : ry + rh, rx : rx + rw])
            min_g_thr = 68 if is_micro_icon else (95 if is_single_char_ui else (52 if local_flat >= 0.62 else 65))
            if max_g < min_g_thr:
                return

            active_cols = np.count_nonzero(is_ink)
            mean_ink_col = float(col_proj.sum()) / float(max(1, active_cols))
            if rh >= int(18 * s) and rw >= int(80 * s) and mean_ink_col < 2.8:
                return
            if is_micro_icon and local_flat < 0.62:
                return
            if is_single_char_ui and (rw < 6 or rh < 6 or (rw <= 9 and rh <= 8)):
                if local_flat < 0.78 or max_g < 110:
                    return
            if local_flat < 0.52 and max_g < 76:
                return
            if local_flat < 0.38:
                if max_g < 145 or rw < 24:
                    return

            total_stroke = int(col_proj.sum())
            is_accent = total_blue >= max(8, int(total_stroke * 0.22)) and rw >= 24

            if int(64 * s) <= rh <= max_thumb_h and int(96 * s) <= rw <= max_thumb_w and 0.75 <= rw / float(rh) <= 2.8:
                g_std = float(gray[ry : ry + rh, rx : rx + rw].std())
                if g_std >= 24.0:
                    if not inside_frame:
                        refined_boxes.append([rx, ry, rw, rh, 0.91, "thumbnail", "control", []])
                    return

            max_icon_dim = max(36, int(44 * s))
            if (
                is_micro_icon
                or is_single_char_ui
                or (rw <= max_icon_dim and rh <= max_icon_dim and 0.55 <= rw / float(max(1, rh)) <= 1.75)
            ):
                kind = "icon"
                score = 0.84
                level = "control"
            elif is_accent and rw <= int(350 * s):
                kind = "inline_link"
                score = 0.92
                level = "control"
            elif rw >= int(260 * s) or (rh <= int(9 * s) and rw >= int(45 * s)):
                kind = "text_block"
                score = 0.78
                level = "line"
            else:
                kind = "text_link"
                score = 0.85
                level = "control"

            if not include_text and kind == "text_block" and not inside_frame:
                return

            word_children = []
            if kind in ("text_block", "text_link", "inline_link") and len(starts) >= 2 and rh <= int(56 * s):
                w_gap_thr = max(3, int(rh * 0.16))
                w_splits = np.flatnonzero((starts[1:] - ends[:-1]) >= w_gap_thr)
                if len(w_splits) >= 1:
                    w_idx_starts = np.concatenate(([0], w_splits + 1))
                    w_idx_ends = np.concatenate((w_splits, [len(starts) - 1]))
                    raw_words = []
                    for is0, ie0 in zip(w_idx_starts, w_idx_ends):
                        wx0, wx1 = int(starts[is0]), int(ends[ie0])
                        if wx1 - wx0 >= 4:
                            raw_words.append((wx0, wx1))
                    if len(raw_words) >= 2:
                        w_hints = generate_vimium_hints(len(raw_words))
                        for wi, (wx0, wx1) in enumerate(raw_words):
                            ww = wx1 - wx0
                            is_lead_icon = (
                                wi == 0
                                and rh >= 14
                                and 0.65 <= ww / float(rh) <= 1.40
                                and (raw_words[1][0] - wx1) >= max(5, int(rh * 0.25))
                            )
                            word_children.append(
                                {
                                    "x": int((rx + wx0) * scale_back),
                                    "y": int(ry * scale_back),
                                    "w": int(ww * scale_back),
                                    "h": int(rh * scale_back),
                                    "score": round(score, 2),
                                    "kind": "icon" if is_lead_icon else "word",
                                    "level": "control" if is_lead_icon else "word",
                                    "hint": w_hints[wi],
                                }
                            )

            if inside_frame:
                inner_frame_children.append([rx, ry, rw, rh, score, kind, level, word_children])
            else:
                refined_boxes.append([rx, ry, rw, rh, score, kind, level, word_children])

        for (cx, cy, cw, ch) in coarse_boxes:
            refine_region(cx, cy, cw, ch, 0)

        # Deduplicate contained refined_boxes BEFORE pairing
        refined_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        dedup_rbs = []
        for rb in refined_boxes:
            rx, ry, rw, rh = rb[:4]
            r_area = rw * rh
            if any(
                max(0, min(rx + rw, kx + kw) - max(rx, kx))
                * max(0, min(ry + rh, ky + kh) - max(ry, ky))
                >= 0.70 * r_area
                for kx, ky, kw, kh, *_ in dedup_rbs + thumbs
            ):
                continue
            dedup_rbs.append(rb)
        refined_boxes = dedup_rbs
        refined_boxes.sort(key=lambda b: (b[1] // 14, b[0]))

        # v6 Vertical Numeric/Menu Column Normalization:
        for i, rbi in enumerate(refined_boxes):
            if rbi[3] <= max(18, int(20 * s)) and rbi[2] <= max(44, int(46 * s)):
                col_peers = [
                    j
                    for j, rbj in enumerate(refined_boxes)
                    if abs(rbj[0] - rbi[0]) <= max(8, int(9 * s))
                    and abs(rbj[3] - rbi[3]) <= 5
                    and rbj[2] <= max(52, int(52 * s))
                ]
                if len(col_peers) >= 4:
                    col_peers.sort(key=lambda j: refined_boxes[j][1])
                    pitches = [
                        refined_boxes[col_peers[k + 1]][1] - refined_boxes[col_peers[k]][1]
                        for k in range(len(col_peers) - 1)
                    ]
                    if float(np.median(pitches)) <= max(42, int(46 * s)):
                        for j in col_peers:
                            refined_boxes[j][5] = "text_link"

        # ------------------------------------------------------------------
        # Stage 5: Clean, Non-Overlapping Compound Pairing (Horizontal FIRST, then Vertical)
        # ------------------------------------------------------------------
        v_pure_gray = (dx_gray >= 4) & (dy_gray <= 2)
        h_pure_gray = (dy_gray >= 4) & (dx_gray <= 2)
        used_rb = [False] * len(refined_boxes)
        used_fr = [False] * len(frames)
        desktop_icons = []

        def make_child_dict(bx, by, bw, bh, bsc, bkd, blv, bhnt, bch=None):
            return {
                "x": int(bx * scale_back),
                "y": int(by * scale_back),
                "w": int(bw * scale_back),
                "h": int(bh * scale_back),
                "score": round(float(bsc), 2),
                "kind": str(bkd),
                "level": str(blv),
                "hint": str(bhnt),
                "children": list(bch) if bch else [],
            }

        # 5A. Horizontal [Square Icon/Frame] + [Right Text Label] Pairing
        max_h_icon = max(62, int(68 * s))
        for src_type, src_list, used_src in [("fr", frames, used_fr), ("rb", refined_boxes, used_rb)]:
            for idx1, item1 in enumerate(src_list):
                if used_src[idx1]:
                    continue
                fx, fy, fw, fh, fsc, fkd = item1[:6]
                is_square_icon_cand = (
                    fkd in ("frame", "icon", "inline_link")
                    or (fkd == "text_link" and 24 <= fw <= max_h_icon and 24 <= fh <= max_h_icon and 0.68 <= fw / float(fh) <= 1.38)
                )
                if (
                    (fy >= int(58 * s) or fx <= int(140 * s))
                    and is_square_icon_cand
                    and 16 <= fw <= max_h_icon
                    and 16 <= fh <= max_h_icon
                    and 0.58 <= fw / float(fh) <= 1.58
                ):
                    best_rbi = None
                    for rbi, rb2 in enumerate(refined_boxes):
                        if used_rb[rbi] or (src_type == "rb" and idx1 == rbi):
                            continue
                        rx, ry, rw, rh, rsc, rkd = rb2[:6]
                        if (
                            rkd in ("text_link", "inline_link")
                            and rw >= max(18, int(18 * s))
                            and rh <= max(30, int(36 * s))
                            and rw > rh * 1.15
                        ):
                            h_gap = rx - (fx + fw)
                            if (
                                3 <= h_gap <= max(36, int(44 * s))
                                and ry >= fy - max(8, int(10 * s))
                                and (ry + rh) <= (fy + fh + max(10, int(12 * s)))
                            ):
                                gy1, gy2 = min(fy, ry), max(fy + fh, ry + rh)
                                gx1, gx2 = fx + fw + 1, rx - 1
                                if gx2 > gx1 and np.any(v_pure_gray[gy1:gy2, gx1:gx2].mean(axis=0) >= 0.55):
                                    continue
                                best_rbi = rbi
                                break
                    if best_rbi is not None:
                        rb2 = refined_boxes[best_rbi]
                        rx, ry, rw, rh, rsc, rkd, rlv, rch = rb2
                        used_src[idx1] = True
                        used_rb[best_rbi] = True
                        nx = min(fx, rx)
                        ny = min(fy, ry)
                        nw = max(fx + fw, rx + rw) - nx
                        nh = max(fy + fh, ry + rh) - ny
                        comp_ch = [
                            make_child_dict(fx, fy, fw, fh, fsc, "icon", "control", "S"),
                            make_child_dict(rx, ry, rw, rh, rsc, rkd, "line", "A", rch),
                        ]
                        # Also absorb any stacked subtitle line directly under the label
                        for rbi3, rb3 in enumerate(refined_boxes):
                            if used_rb[rbi3]:
                                continue
                            sx, sy, sw, sh, ssc, skd, slv, sch = rb3
                            if skd in ("text_block", "text_link", "inline_link") and sw <= int(260 * s) and sh <= int(24 * s):
                                if abs(sx - rx) <= int(12 * s) and 0 <= sy - (ry + rh) <= int(16 * s) and (sy + sh) <= (fy + fh + int(22 * s)):
                                    used_rb[rbi3] = True
                                    x0, y0 = min(nx, sx), min(ny, sy)
                                    nw = max(nx + nw, sx + sw) - x0
                                    nh = max(ny + nh, sy + sh) - y0
                                    nx, ny = x0, y0
                                    comp_ch.append(make_child_dict(sx, sy, sw, sh, ssc, skd, "line", "D", sch))
                        desktop_icons.append([nx, ny, nw, nh, 0.95, "desktop_icon", "control", comp_ch])

        # 5B. Vertical [Icon/Frame] + [Bottom Horizontal Text Caption] Desktop/Tile/Ribbon Pairing
        for src_type, src_list, used_src in [("fr", frames, used_fr), ("rb", refined_boxes, used_rb)]:
            for idx1, item1 in enumerate(src_list):
                if used_src[idx1]:
                    continue
                fx, fy, fw, fh, fsc, fkd = item1[:6]
                if 16 <= fw <= max(64, int(82 * s)) and 20 <= fh <= max(64, int(76 * s)) and 0.55 <= fw / float(fh) <= 1.55:
                    fcx = fx + fw * 0.5
                    best_lbl = None
                    for rbi, rb2 in enumerate(refined_boxes):
                        if used_rb[rbi] or (src_type == "rb" and idx1 == rbi):
                            continue
                        rx, ry, rw, rh, rsc, rkd = rb2[:6]
                        v_gap = ry - (fy + fh)
                        rcx = rx + rw * 0.5
                        if (
                            rkd in ("text_link", "inline_link", "text_block")
                            and fh >= int(rh * 1.22)
                            and rw >= max(28, int(rh * 1.45))
                            and rw <= max(110, int(145 * s))
                            and rh <= max(24, int(36 * s))
                            and 0 <= v_gap <= max(22, int(24 * s))
                            and abs(fcx - rcx) <= max(26, int(28 * s))
                        ):
                            gy1, gy2 = fy + fh, ry
                            gx1, gx2 = min(fx, rx), max(fx + fw, rx + rw)
                            if gy2 > gy1 and gx2 > gx1 and np.any(h_pure_gray[gy1:gy2, gx1:gx2].mean(axis=1) >= 0.50):
                                continue
                            best_lbl = ("rb", rbi, rx, ry, rw, rh, rb2[7])
                            break
                    if best_lbl is None and src_type == "fr":
                        for fj, f2 in enumerate(frames):
                            if idx1 == fj or used_fr[fj]:
                                continue
                            rx, ry, rw, rh = f2[:4]
                            v_gap = ry - (fy + fh)
                            rcx = rx + rw * 0.5
                            if (
                                fh >= int(rh * 1.22)
                                and rw >= max(28, int(rh * 1.45))
                                and rw <= max(110, int(145 * s))
                                and rh <= max(24, int(36 * s))
                                and 0 <= v_gap <= max(22, int(24 * s))
                                and abs(fcx - rcx) <= max(26, int(28 * s))
                            ):
                                best_lbl = ("fr", fj, rx, ry, rw, rh, [])
                                break
                    if best_lbl is not None:
                        kind_l, idx_l, rx, ry, rw, rh, lch = best_lbl
                        if kind_l == "rb":
                            used_rb[idx_l] = True
                        else:
                            used_fr[idx_l] = True
                        used_src[idx1] = True
                        nx = min(fx, rx)
                        ny = min(fy, ry)
                        nw = max(fx + fw, rx + rw) - nx
                        nh = max(fy + fh, ry + rh) - ny
                        comp_ch = [
                            make_child_dict(fx, fy, fw, fh, fsc, "icon", "control", "S"),
                            make_child_dict(rx, ry, rw, rh, rsc if kind_l == "rb" else 0.88, "text_link", "line", "A", lch),
                        ]
                        for rbi3, rb3 in enumerate(refined_boxes):
                            if used_rb[rbi3]:
                                continue
                            sx, sy, sw, sh, ssc, skd, slv, sch = rb3
                            if skd in ("text_block", "text_link", "inline_link") and sw <= max(100, int(130 * s)) and sh <= max(18, int(22 * s)):
                                if abs(sx - rx) <= max(10, int(12 * s)) and 0 <= sy - (ry + rh) <= max(8, int(10 * s)):
                                    used_rb[rbi3] = True
                                    x0, y0 = min(nx, sx), min(ny, sy)
                                    nw = max(nx + nw, sx + sw) - x0
                                    nh = max(ny + nh, sy + sh) - y0
                                    nx, ny = x0, y0
                                    comp_ch.append(make_child_dict(sx, sy, sw, sh, ssc, skd, "line", "D", sch))
                        for (tx, ty, tw, th) in tile_containers:
                            if nx >= tx - 4 and ny >= ty - 4 and (nx + nw) <= (tx + tw + 4) and (ny + nh) <= (ty + th + 4):
                                nx, ny, nw, nh = tx, ty, tw, th
                                break
                        desktop_icons.append([nx, ny, nw, nh, 0.95, "desktop_icon", "control", comp_ch])

        rem_frames = [f for fi, f in enumerate(frames) if not used_fr[fi]]
        rem_rbs = [rb for rbi, rb in enumerate(refined_boxes) if not used_rb[rbi]]

        final_frames = []
        for f in rem_frames:
            fx, fy, fw, fh = f[:4]
            if fw <= 32 and fh <= 18:
                attached = False
                for rb in rem_rbs:
                    v_ov = min(fy + fh, rb[1] + rb[3]) - max(fy, rb[1])
                    h_gap = fx - (rb[0] + rb[2])
                    if v_ov >= fh * 0.5 and -2 <= h_gap <= 14:
                        nx = min(rb[0], fx)
                        ny = min(rb[1], fy)
                        rb[2] = max(rb[0] + rb[2], fx + fw) - nx
                        rb[3] = max(rb[1] + rb[3], fy + fh) - ny
                        rb[0], rb[1] = nx, ny
                        attached = True
                        break
                if attached:
                    continue
            final_frames.append(f)

        # Attach inner_frame_children (icons/text/words found inside closed frames) into their enclosing final_frames
        for ic in inner_frame_children:
            ix, iy, iw, ih, isc, ikd, ilv, ich = ic
            icx, icy = ix + iw * 0.5, iy + ih * 0.5
            for ff in final_frames:
                fx, fy, fw, fh = ff[:4]
                if fx <= icx <= fx + fw and fy <= icy <= fy + fh and (iw * ih) < 0.90 * (fw * fh):
                    ff[7].append(
                        make_child_dict(
                            ix,
                            iy,
                            iw,
                            ih,
                            isc,
                            ikd,
                            ilv,
                            generate_vimium_hints(len(ff[7]) + 1)[-1],
                            ich,
                        )
                    )
                    break

        # Remove outer window-corner and panel-corner rounding artifacts
        rem_rbs = [
            rb
            for rb in rem_rbs
            if not (
                rb[5] == "icon"
                and rb[2] <= max(34, int(36 * s))
                and rb[3] <= max(34, int(36 * s))
                and (
                    ((rb[0] <= 4 or rb[0] + rb[2] >= W - 4) and (rb[1] <= 4 or rb[1] + rb[3] >= H - 4))
                    or any(
                        abs((rb[0] + rb[2] * 0.5) - pcx) <= max(18, int(22 * s))
                        and abs((rb[1] + rb[3] * 0.5) - pcy) <= max(18, int(22 * s))
                        for (pcx, pcy) in panel_corners
                    )
                )
            )
        ]

        # 5C. Adaptive Fitts's Law Clickable Hitbox Padding for standalone compact icons (strictly non-overlapping)
        all_other = thumbs + desktop_icons + final_frames
        base_min_dim = max(30, int(36 * s))
        base_pad = max(8, int(11 * s))
        for i, rb in enumerate(rem_rbs):
            rx, ry, rw, rh, rsc, rkd = rb[:6]
            if rkd == "icon":
                for frac in (1.0, 0.65, 0.40):
                    min_d = int(base_min_dim * frac)
                    pad_p = int(base_pad * frac)
                    nw = max(min_d, rw + 2 * pad_p)
                    nh = max(min_d, rh + 2 * pad_p)
                    nx = int(max(0, min(W - nw, rx + (rw - nw) // 2)))
                    ny = int(max(0, min(H - nh, ry + (rh - nh) // 2)))
                    collision = False
                    for j, rb2 in enumerate(rem_rbs):
                        if i == j:
                            continue
                        if max(0, min(nx + nw, rb2[0] + rb2[2]) - max(nx, rb2[0])) * max(0, min(ny + nh, rb2[1] + rb2[3]) - max(ny, rb2[1])) > 0:
                            collision = True
                            break
                    if not collision:
                        for ob in all_other:
                            if max(0, min(nx + nw, ob[0] + ob[2]) - max(nx, ob[0])) * max(0, min(ny + nh, ob[1] + ob[3]) - max(ny, ob[1])) > 0:
                                collision = True
                                break
                    if not collision:
                        rb[0], rb[1], rb[2], rb[3] = nx, ny, nw, nh
                        break

        # 5D. Hierarchical Paragraph Grouping: merge vertically stacked static prose lines sharing a left margin
        if collapse_paragraphs and include_text:
            tb_indices = [
                i
                for i, rb in enumerate(rem_rbs)
                if rb[5] == "text_block"
                or (
                    rb[5] == "text_link"
                    and any(
                        rb2[5] == "text_block"
                        and 0 <= rb[1] - (rb2[1] + rb2[3]) <= max(12, int(14 * s))
                        and abs(rb[0] - rb2[0]) <= max(10, int(12 * s))
                        and abs(rb[3] - rb2[3]) <= 6
                        for rb2 in rem_rbs
                    )
                )
            ]
            tb_indices.sort(key=lambda idx: (rem_rbs[idx][1], rem_rbs[idx][0]))
            merged_tb = set()
            para_rbs = []
            for idx in tb_indices:
                if idx in merged_tb:
                    continue
                chain_tb = [idx]
                merged_tb.add(idx)
                curr = rem_rbs[idx]
                for idx2 in tb_indices:
                    if idx2 in merged_tb:
                        continue
                    cand = rem_rbs[idx2]
                    v_gap = cand[1] - (curr[1] + curr[3])
                    if 0 <= v_gap <= max(12, int(14 * s)) and abs(cand[0] - curr[0]) <= max(12, int(14 * s)):
                        chain_tb.append(idx2)
                        merged_tb.add(idx2)
                        curr = cand
                if len(chain_tb) >= 2:
                    px = min(rem_rbs[k][0] for k in chain_tb)
                    py = min(rem_rbs[k][1] for k in chain_tb)
                    pw = max(rem_rbs[k][0] + rem_rbs[k][2] for k in chain_tb) - px
                    ph = max(rem_rbs[k][1] + rem_rbs[k][3] for k in chain_tb) - py
                    chain_set = set(chain_tb)
                    has_intruder = any(
                        m not in chain_set
                        and max(0, min(px + pw, rem_rbs[m][0] + rem_rbs[m][2]) - max(px, rem_rbs[m][0]))
                        * max(0, min(py + ph, rem_rbs[m][1] + rem_rbs[m][3]) - max(py, rem_rbs[m][1]))
                        > 0
                        for m in range(len(rem_rbs))
                    )
                    if has_intruder:
                        for k in chain_tb:
                            para_rbs.append(rem_rbs[k])
                        continue
                    l_hints = generate_vimium_hints(len(chain_tb))
                    line_children = [
                        make_child_dict(
                            rem_rbs[k][0],
                            rem_rbs[k][1],
                            rem_rbs[k][2],
                            rem_rbs[k][3],
                            0.78,
                            "line",
                            "line",
                            l_hints[li],
                            rem_rbs[k][7],
                        )
                        for li, k in enumerate(chain_tb)
                    ]
                    para_rbs.append([px, py, pw, ph, 0.80, "text_block", "paragraph", line_children])
                else:
                    para_rbs.append(rem_rbs[idx])
            tb_set = set(tb_indices)
            rem_rbs = [rb for i, rb in enumerate(rem_rbs) if i not in tb_set] + para_rbs

        # 5E. Sibling Rail Normalization (for aligned desktop_icon sidebar/menu stacks, capped to group content width, zero-overlap!)
        di_indices = list(range(len(desktop_icons)))
        di_indices.sort(key=lambda i: desktop_icons[i][1])
        visited_di = set()
        for i in di_indices:
            if i in visited_di:
                continue
            di_x = desktop_icons[i][0]
            rail = [
                j
                for j in di_indices
                if j not in visited_di and abs(desktop_icons[j][0] - di_x) <= max(8, int(10 * s))
            ]
            if len(rail) >= 3:
                chain = [rail[0]]
                for j in rail[1:]:
                    dy_c = (desktop_icons[j][1] + desktop_icons[j][3] * 0.5) - (
                        desktop_icons[chain[-1]][1] + desktop_icons[chain[-1]][3] * 0.5
                    )
                    if 24 <= dy_c <= max(75, int(85 * s)):
                        chain.append(j)
                if len(chain) >= 3:
                    norm_x = min(desktop_icons[j][0] for j in chain)
                    norm_r = min(norm_x + max(160, int(185 * s)), max(desktop_icons[j][0] + desktop_icons[j][2] for j in chain))
                    norm_w = norm_r - norm_x
                    norm_h = int(np.median([desktop_icons[j][3] for j in chain])) + int(6 * s)
                    can_normalize = True
                    new_boxes = {}
                    for j in chain:
                        cy = desktop_icons[j][1] + desktop_icons[j][3] * 0.5
                        ny = int(max(0, min(H - norm_h, round(cy - norm_h * 0.5))))
                        new_boxes[j] = (norm_x, ny, norm_w, norm_h)
                    for j, (nx, ny, nw, nh) in new_boxes.items():
                        for k, (ox, oy, ow, oh) in new_boxes.items():
                            if j != k and max(0, min(nx + nw, ox + ow) - max(nx, ox)) * max(0, min(ny + nh, oy + oh) - max(ny, oy)) > 0:
                                can_normalize = False
                                break
                        if not can_normalize:
                            break
                        for ob in thumbs + final_frames + rem_rbs:
                            if max(0, min(nx + nw, ob[0] + ob[2]) - max(nx, ob[0])) * max(0, min(ny + nh, ob[1] + ob[3]) - max(ny, ob[1])) > 0:
                                can_normalize = False
                                break
                    if can_normalize:
                        for j, (nx, ny, nw, nh) in new_boxes.items():
                            visited_di.add(j)
                            desktop_icons[j][0] = nx
                            desktop_icons[j][1] = ny
                            desktop_icons[j][2] = nw
                            desktop_icons[j][3] = nh

        # Final containment cleanup: absorb any sub-element that overlaps >= 25% of its area with a desktop_icon, thumbnail, or larger item
        stage4_thumbs = [rb for rb in rem_rbs if rb[5] == "thumbnail"]
        rem_rbs = [rb for rb in rem_rbs if rb[5] != "thumbnail"]
        thumbs.extend(stage4_thumbs)

        clean_frames = []
        for f in final_frames:
            fx, fy, fw, fh = f[:4]
            f_area = max(1, fw * fh)
            if any(
                max(0, min(fx + fw, ox + ow) - max(fx, ox)) * max(0, min(fy + fh, oy + oh) - max(fy, oy)) >= 0.25 * f_area
                for ox, oy, ow, oh, *_ in thumbs + desktop_icons
            ):
                continue
            clean_frames.append(f)
        final_frames = clean_frames

        rem_rbs.sort(key=lambda b: b[2] * b[3], reverse=True)
        clean_rbs = []
        for rb in rem_rbs:
            rx, ry, rw, rh = rb[:4]
            r_area = max(1, rw * rh)
            absorbed_into_parent = False
            for parent in desktop_icons + thumbs + final_frames:
                px, py, pw, ph = parent[:4]
                inter = max(0, min(rx + rw, px + pw) - max(rx, px)) * max(0, min(ry + rh, py + ph) - max(ry, py))
                if inter >= 0.25 * r_area:
                    if parent[5] in ("desktop_icon", "frame"):
                        parent[7].append(
                            make_child_dict(
                                rx,
                                ry,
                                rw,
                                rh,
                                rb[4],
                                rb[5],
                                rb[6],
                                generate_vimium_hints(len(parent[7]) + 1)[-1],
                                rb[7],
                            )
                        )
                    absorbed_into_parent = True
                    break
            if not absorbed_into_parent:
                for k_rb in clean_rbs:
                    kx, ky, kw, kh = k_rb[:4]
                    inter = max(0, min(rx + rw, kx + kw) - max(rx, kx)) * max(0, min(ry + rh, ky + kh) - max(ry, ky))
                    if inter >= 0.25 * min(r_area, kw * kh):
                        if r_area <= kw * kh:
                            absorbed_into_parent = True
                            break
            if not absorbed_into_parent:
                clean_rbs.append(rb)
        rem_rbs = clean_rbs

        all_items = thumbs + desktop_icons + final_frames + rem_rbs
        targets = [
            UITarget(
                int(b[0] * scale_back),
                int(b[1] * scale_back),
                int(b[2] * scale_back),
                int(b[3] * scale_back),
                float(b[4]),
                str(b[5]),
                level=str(b[6]),
                children=list(b[7]),
            )
            for b in all_items
            if b[4] >= self.min_score
        ]
        targets.sort(key=lambda t: (t.y // 14, t.x))
        for t, h_str in zip(targets, generate_vimium_hints(len(targets))):
            t.hint = h_str

        t_end = time.perf_counter()
        timings = {
            "signal_ms": (t_sig - t0) * 1000.0,
            "frames_ms": (t_frames - t_sig) * 1000.0,
            "coarse_ms": (t_coarse - t_frames) * 1000.0,
            "recursive_ms": (t_end - t_coarse) * 1000.0,
            "total_ms": (t_end - t0) * 1000.0,
        }
        return targets, timings


def render_vimium_overlay(bgr: np.ndarray, targets: List[UITarget]) -> np.ndarray:
    """Render rail-aligned Vimium hint badges and color-coded target bounding boxes."""
    canvas = bgr.copy()
    overlay = bgr.copy()

    palette = {
        "frame": (80, 220, 100),         # Emerald green: framed buttons, inputs, toggles, pills, cards
        "thumbnail": (50, 180, 255),     # Vibrant orange-gold: bounded photo/video thumbnails
        "desktop_icon": (255, 210, 60),  # Sky-cyan: unified desktop/sidebar/tile icon + caption compounds
        "inline_link": (255, 140, 255),  # Magenta-pink: accent/inline colored links
        "text_link": (60, 205, 255),     # Warm amber-gold: nav/tree/table text links & headings
        "icon": (255, 225, 70),          # Cyan: standalone compact icons with clickable hitbox
        "text_block": (210, 185, 130),   # Soft slate-blue: hierarchical paragraphs/lines of prose
        "line": (120, 230, 190),         # Mint green: drill-down child line
        "word": (90, 245, 255),          # Bright yellow-cyan: drill-down leaf word
    }

    for t in targets:
        color = palette.get(t.kind, (80, 220, 255))
        cv2.rectangle(overlay, (t.x, t.y), (t.x + t.w, t.y + t.h), color, -1)

    cv2.addWeighted(overlay, 0.14, canvas, 0.86, 0, canvas)

    xs = [t.x for t in targets]
    ys = [t.y for t in targets]
    snapped_x = []
    snapped_y = []
    for t in targets:
        near_x = [x for x in xs if abs(x - t.x) <= 3]
        near_y = [y for y in ys if abs(y - t.y) <= 3]
        snapped_x.append(int(round(np.median(near_x))) if len(near_x) >= 2 else t.x)
        snapped_y.append(int(round(np.median(near_y))) if len(near_y) >= 2 else t.y)

    H, W = bgr.shape[:2]
    for idx, t in enumerate(targets):
        color = palette.get(t.kind, (80, 220, 255))
        thick = 2 if t.kind in ("thumbnail", "desktop_icon") or t.level == "paragraph" else 1
        cv2.rectangle(canvas, (t.x, t.y), (t.x + t.w, t.y + t.h), color, thick, cv2.LINE_AA)

        label = t.hint or "??"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.36
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        badge_w = tw + 6
        badge_h = th + 5

        # For narrow text items (e.g. font-size numbers in a dropdown), place the badge to the left so the digits remain 100% uncovered
        if t.kind == "text_link" and t.w <= 54 and t.h <= 32 and snapped_x[idx] >= badge_w + 6:
            bx1 = max(0, snapped_x[idx] - badge_w - 3)
            by1 = max(0, snapped_y[idx] + (t.h - badge_h) // 2)
        elif t.h <= 18 and snapped_y[idx] >= badge_h + 2:
            bx1 = max(0, snapped_x[idx] - 2)
            by1 = max(0, snapped_y[idx] - badge_h)
        else:
            bx1 = max(0, snapped_x[idx] - 2)
            by1 = max(0, snapped_y[idx] - 2)
        bx2 = min(W - 1, bx1 + badge_w)
        by2 = min(H - 1, by1 + badge_h)

        cv2.rectangle(canvas, (bx1 - 1, by1 - 1), (bx2 + 1, by2 + 1), (15, 15, 15), -1)
        badge_bg = (55, 215, 255) if t.kind != "text_block" else (180, 225, 245)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), badge_bg, -1)
        cv2.putText(
            canvas,
            label,
            (bx1 + 3, by2 - 3),
            font,
            scale,
            (10, 10, 10),
            thickness,
            cv2.LINE_AA,
        )

    return canvas


def render_drilldown_overlay(bgr: np.ndarray, parent: UITarget, to_words: bool = True) -> np.ndarray:
    """Render a dimmed screen spotlighting `parent` and displaying its pre-computed child/word hints."""
    dimmed = (bgr.astype(np.float32) * 0.35).astype(np.uint8)
    H, W = bgr.shape[:2]
    px1, py1 = max(0, parent.x - 8), max(0, parent.y - 8)
    px2, py2 = min(W, parent.x + parent.w + 8), min(H, parent.y + parent.h + 8)
    dimmed[py1:py2, px1:px2] = bgr[py1:py2, px1:px2]
    cv2.rectangle(dimmed, (px1, py1), (px2, py2), (80, 230, 255), 2, cv2.LINE_AA)
    sub_targets = drill_down(parent, to_words=to_words)
    return render_vimium_overlay(dimmed, sub_targets)


def main():
    parser = argparse.ArgumentParser(description="Visel v6 Vision-Refined Structure-Aware Hierarchical UI Detector")
    parser.add_argument("input", nargs="?", help="Path to input screenshot PNG/JPG")
    parser.add_argument("-o", "--output", help="Path to save annotated Vimium overlay PNG")
    parser.add_argument("--json", help="Path to save detected hint targets (with children) as JSON")
    parser.add_argument("--interactive-only", action="store_true", help="Exclude prose/static text blocks")
    parser.add_argument("--turbo", action="store_true", help="Use half-resolution pass for faster CPU execution")
    parser.add_argument("--drill-down", help="Hint code of a target to spotlight and expand into child/word hints")
    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        return

    bgr = cv2.imread(args.input)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    det = ViselDetector(include_text=not args.interactive_only, fast_half_res=args.turbo)
    targets, timings = det.detect(bgr)
    total_children = sum(len(t.children) for t in targets)
    print(
        f"Detected {len(targets)} top-level targets ({total_children} drill-down children) in {timings['total_ms']:.2f} ms "
        f"(signal={timings['signal_ms']:.1f}ms, frames={timings['frames_ms']:.1f}ms, "
        f"coarse={timings['coarse_ms']:.1f}ms, recursive+rails={timings['recursive_ms']:.1f}ms)"
    )

    if args.output:
        if args.drill_down:
            key = args.drill_down.upper()
            match = [t for t in targets if t.hint == key]
            if match:
                vis = render_drilldown_overlay(bgr, match[0], to_words=True)
            else:
                vis = render_vimium_overlay(bgr, targets)
        else:
            vis = render_vimium_overlay(bgr, targets)
        cv2.imwrite(args.output, vis)
        print(f"Saved annotated overlay to {args.output}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump([t.__dict__ for t in targets], f, indent=2)


if __name__ == "__main__":
    main()
