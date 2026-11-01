#!/usr/bin/env python3
"""
visel_detector.py (v9-GlyphChain) — Self-Calibrating, Contour-Ladder UI & Contextual Glyph-Chain Detector
=========================================================================================================
Streamlined, fast, zero-dependency computer-vision UI detector for Vimium navigation across any
OS, theme, DPI scale, zoom level, or compression level.

Key Features & Optimizations:
  1. Stage 0 Self-Calibrating Image Physics (`_calibrate_image_physics`):
     - Automatically estimates the screenshot's background noise floor (`noise_floor`), gradient
       contrast quantiles (`stroke_thr`, `border_thr`, `flat_thr`), and fundamental modal text line
       height `L_0` (down to 8.0 px for zoomed-out / small-font displays) in ~1 ms, with aspect-aware
       upscaling so narrow/portrait crops never over-inflate `L_0`.
  2. Multi-Threshold Stage 2 Contour Ladder:
     - Extracts closed frames and thumbnails across a 3-threshold ladder (`border_thr - 1`, `+ 1`, `+ 3`)
       capturing subtle dark-mode buttons (`OBS`, `Blender`) and media cards simultaneously without
       needing heavy color-delta pill peeling.
  3. Stage 4 Single-Pass Balanced 2D Decomposition + Foreground-Contrast Multi-Line Splitting:
     - Slices complex 1D text rows via quiet gutter flatness scoring, separates tightly-spaced small-font
       lines via direct foreground luminance valleys (bypassing vertical morphological gradient dilation),
       and decomposes 2D sub-blocks in a single fast balanced pass.
  4. 1D Contextual Glyph-Chain Sequence Scorer (`[L, L, ?, L, L]`) + Bimodal Valley Word Splitter:
     - Scores candidate character primitives along horizontal strips and propagates sequence confidence
       across collinear neighbors (`[Letter, Letter, Maybe-Not, Letter, Letter]` -> boosts weak/small
       glyphs while rejecting isolated noise), rescuing zoomed-out / low-contrast text lines.
     - Segments individual words using direct foreground ink contrast valleys (eliminating +1 px horizontal
       `MORPH_GRADIENT` edge bleed), line-local bimodal gap clustering (separating 1 px kerning from 2 px
       word spaces), and pitch-guided sub-pixel valley splitting for lightly bridged words.
  5. Post-Stage-4 Peer-Grid & Cross-Row Media Shelf Completion + Flattened Paragraph Word Hierarchy:
     - Completes dark-bordered 16:9 video cards and 9:16 Shorts shelves while preserving flat 2-level
       paragraph -> line -> word hierarchies across consolidated collinear line fragments.
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


def _collect_word_leaves(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    leaves: List[Dict[str, Any]] = []
    for ch in nodes:
        sub = ch.get("children", [])
        if sub:
            leaves.extend(_collect_word_leaves(sub))
        else:
            leaves.append(ch)
    return leaves


def drill_down(target: UITarget, to_words: bool = True) -> List[UITarget]:
    """
    Instant 0.0 ms hierarchical drill-down when the user selects a target with a modifier key.
    For massive multi-line blocks (e.g. a 50-line terminal paragraph where total leaf words > 75),
    stepping to lines first prevents hundreds of 3-character hint badges from overlapping; selecting
    any line then drills down to the words on that line.
    """
    if not target.children:
        return [target]
    if to_words:
        leaves = _collect_word_leaves(target.children)
        if target.level == "paragraph" and len(target.children) >= 8 and len(leaves) > 75:
            leaves = list(target.children)
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
        self.k_coarse = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 5))
        self.k_coarse_hi = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 7))
        self.k_bar_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.k_h_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        self.k_v_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        self.one_12 = np.ones(12, dtype=np.uint8)

    @staticmethod
    def _calibrate_image_physics(
        grad_max: np.ndarray, W: int, H: int
    ) -> Tuple[int, int, int, float, float]:
        """
        Self-calibrate gradient thresholds (noise_floor, flat_thr, border_thr, stroke_thr)
        and fundamental text line height L_0 from the image's own statistics.
        Supports zoomed-out / small-font displays down to L_0 = 8.0 px.
        """
        hist = cv2.calcHist([grad_max], [0], None, [256], [0, 256]).flatten()
        cdf = np.cumsum(hist) / float(max(1, W * H))
        p35 = int(np.searchsorted(cdf, 0.35))
        p65 = int(np.searchsorted(cdf, 0.65))
        p82 = int(np.searchsorted(cdf, 0.82))

        noise_floor = int(np.clip(p35, 1, 5))
        flat_thr = int(np.clip(noise_floor + 2, 3, 6))
        border_thr = int(np.clip(max(noise_floor + 1, int(p65 * 0.75)), 3, 6))
        stroke_thr = int(np.clip(int(p82 * 0.85), 14, 21))

        # Estimate fundamental modal text line height L_0 from horizontal stroke bands
        _, s_bin = cv2.threshold(grad_max, stroke_thr, 255, cv2.THRESH_BINARY)
        k_line = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 2))
        s_closed = cv2.morphologyEx(s_bin, cv2.MORPH_CLOSE, k_line)
        cnts, _ = cv2.findContours(s_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        heights = []
        small_heights = []
        for c in cnts:
            _, _, bw, bh = cv2.boundingRect(c)
            if 9 <= bh <= 32 and bw >= bh * 2.2 and bw <= W * 0.45:
                heights.append(bh)
            elif 6 <= bh < 9 and bw >= bh * 2.5 and bw <= W * 0.45:
                small_heights.append(bh)
        if len(heights) >= 8:
            h_arr = np.array(heights, dtype=np.int32)
            counts = np.bincount(h_arr, minlength=36)
            # Smooth 3-bin window to find robust modal text height L_0
            smooth_counts = counts[8:33] + counts[7:32] + counts[9:34]
            modal_h = float(8 + int(np.argmax(smooth_counts)))
            med_h = float(np.median(h_arr))
            L_0 = 0.6 * modal_h + 0.4 * med_h
            L_0 = float(np.clip(L_0, 11.5, 26.0))
        elif len(heights) + len(small_heights) >= 8:
            h_arr = np.array(heights + small_heights, dtype=np.int32)
            counts = np.bincount(h_arr, minlength=36)
            smooth_counts = counts[6:33] + counts[5:32] + counts[7:34]
            modal_h = float(6 + int(np.argmax(smooth_counts)))
            med_h = float(np.median(h_arr))
            L_0 = float(np.clip(0.6 * modal_h + 0.4 * med_h, 8.5, 26.0))
        else:
            L_0 = 16.0 * float(np.clip(np.sqrt((W * H) / float(1440 * 900)), 0.70, 1.60))
            L_0 = float(np.clip(L_0, 11.5, 26.0))

        u = float(np.clip(L_0 / 16.0, 0.68, 1.75))
        return flat_thr, border_thr, stroke_thr, L_0, u

    @staticmethod
    def _score_and_segment_glyph_chain(
        roi_gray: np.ndarray,
        roi_dx: np.ndarray,
        roi_dy: np.ndarray,
        roi_stroke: np.ndarray,
        L_0: float,
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Vectorized 1D Contextual Glyph-Chain Scorer (`[L, L, ?, L, L]`) + Relative Bimodal Word Splitter.

        All spatial and contrast rules are dimensionless ratios of the strip's own height `rh`,
        85th-percentile ink contrast `peak`, median primitive width `med_pw`, and baseline intra-word
        kerning `kern_base` — eliminating hardcoded pixel constants across any zoom level or font:
          1. Direct foreground contrast `|roi_gray - bg_med|` in the core x-height band avoids +1 px
             horizontal `MORPH_GRADIENT` edge dilation.
          2. 1D neighborhood belief propagation (`[L, L, ?, L, L]`) rescues weak/small glyphs aligned
             on the same baseline while rejecting isolated noise specks.
          3. Relative bimodal gap separation distinguishes intra-word character kerning (including
             2-4 px monospace letter spacing) from true inter-word spaces, and merges ultra-narrow
             adjacent fragments (`< 0.95 * rh`) so hint badges never overlap.
        """
        rh, rw = roi_gray.shape[:2]
        if rh < 4 or rw < int(rh * 1.2):
            return 0.0, []

        # Background luminance from top and bottom border rows
        if rh >= 6:
            b_top = roi_gray[0, :]
            b_bot = roi_gray[-1, :]
            bg_med = 0.5 * float(
                int(np.partition(b_top, rw >> 1)[rw >> 1])
                + int(np.partition(b_bot, rw >> 1)[rw >> 1])
            )
        else:
            flat_g = roi_gray.ravel()
            bg_med = float(np.partition(flat_g, len(flat_g) >> 1)[len(flat_g) >> 1])

        fg = np.abs(roi_gray.astype(np.float32) - bg_med)

        # Core vertical x-height band (trim ~12% top/bottom anti-alias halo on lines >= 7px tall)
        y_trim = max(1, int(round(rh * 0.12))) if rh >= 7 else 0
        fg_core = fg[y_trim : rh - y_trim, :] if (rh - 2 * y_trim) >= 3 else fg

        col_max = fg_core.max(axis=0)
        col_mean = fg_core.mean(axis=0)
        idx85 = min(rw - 1, int(rw * 0.85))
        peak = float(np.partition(col_max, idx85)[idx85]) if rw >= 6 else float(col_max.max())
        if peak < 11.0:
            return 0.0, []

        # Relative column valley threshold (27% of 85th-percentile ink contrast)
        ink_thr = float(np.clip(peak * 0.27, peak * 0.18, peak * 0.45))
        is_ink = col_max > ink_thr

        if np.count_nonzero(is_ink) < max(3, int(rh * 0.4)):
            col_s = roi_stroke.sum(axis=0)
            is_ink = col_s > 0
            if np.count_nonzero(is_ink) < max(3, int(rh * 0.4)):
                return 0.0, []

        pad = np.concatenate(([False], is_ink, [False]))
        starts = np.flatnonzero(pad[1:] & ~pad[:-1])
        ends = np.flatnonzero(~pad[1:] & pad[:-1])
        n_prim = len(starts)
        if n_prim == 0:
            return 0.0, []

        # Vectorized per-primitive structural features (all dimensionless ratios of rh)
        fg_bin = fg > (ink_thr * 0.85)
        row_indices = np.arange(rh, dtype=np.int32)[:, None]
        y_top_col = np.where(fg_bin, row_indices, rh).min(axis=0)
        y_bot_col = np.where(fg_bin, row_indices, -1).max(axis=0)
        fg_count_col = fg_bin.sum(axis=0)
        dx_col = roi_dx.mean(axis=0)
        dy_col = roi_dy.mean(axis=0)
        dx_peak_thr = max(2.5, float(dx_col.max()) * 0.20)
        osc_col = np.zeros(rw, dtype=np.int32)
        if rw >= 3:
            osc_col[1:-1] = (
                (dx_col[1:-1] >= dx_col[:-2])
                & (dx_col[1:-1] >= dx_col[2:])
                & (dx_col[1:-1] > dx_peak_thr)
            ).astype(np.int32)
        osc_prefix = np.concatenate(([0], np.cumsum(osc_col)))
        fg_prefix = np.concatenate(([0], np.cumsum(fg_count_col)))
        dx_prefix = np.concatenate(([0.0], np.cumsum(dx_col)))
        dy_prefix = np.concatenate(([0.0], np.cumsum(dy_col)))

        p_widths = (ends - starts).astype(np.float32)
        yh0 = np.minimum.reduceat(y_top_col, starts)
        yh1 = np.maximum.reduceat(y_bot_col, starts) + 1
        ph = np.maximum(1.0, (yh1 - yh0).astype(np.float32))
        y_mids = 0.5 * (yh0 + yh1).astype(np.float32)
        h_spans = ph

        h_ratio = ph / float(max(1, rh))
        f_height = np.where(
            (h_ratio >= 0.42) & (h_ratio <= 1.0),
            1.0,
            np.where(h_ratio >= 0.28, 0.55, 0.15),
        )

        ar = p_widths / float(max(1, rh))
        osc_counts = osc_prefix[ends] - osc_prefix[starts]
        f_aspect = np.where(
            (ar >= 0.15) & (ar <= 1.25),
            1.0,
            np.where(
                (ar > 1.25) & (ar <= 8.5),
                np.where(osc_counts >= np.maximum(1, (ar * 0.42).astype(np.int32)), 0.92, 0.55),
                0.25,
            ),
        )

        ink_pixels = (fg_prefix[ends] - fg_prefix[starts]).astype(np.float32)
        fill = ink_pixels / np.maximum(1.0, ph * p_widths)
        f_fill = np.where(
            (fill >= 0.16) & (fill <= 0.82),
            1.0,
            np.where((fill >= 0.10) & (fill <= 0.90), 0.50, 0.15),
        )

        dx_m = (dx_prefix[ends] - dx_prefix[starts]) / p_widths
        dy_m = (dy_prefix[ends] - dy_prefix[starts]) / p_widths
        f_stroke = np.where(
            dx_m >= 0.52 * dy_m,
            1.0,
            np.where(dx_m >= 0.35 * dy_m, 0.60, 0.20),
        )

        s_local = (
            0.30 * f_height + 0.28 * f_aspect + 0.22 * f_fill + 0.20 * f_stroke
        ).astype(np.float32)
        s_local = np.where(yh1 > yh0, s_local, 0.0)

        # Step B: 1D Contextual Belief Propagation (`[L, L, ?, L, L]`)
        s_ctx = s_local.copy()
        if n_prim >= 2:
            med_ymid = float(np.median(y_mids))
            ctx_num = np.zeros(n_prim, dtype=np.float32)
            ctx_den = np.zeros(n_prim, dtype=np.float32)
            max_dy = 0.28 * rh
            max_dh = 0.45 * rh
            max_gap1 = 2.4 * rh
            max_gap2 = 2.8 * rh

            for step, w_step in ((-2, 0.16), (-1, 0.34), (1, 0.34), (2, 0.16)):
                if step < 0:
                    src = slice(0, n_prim + step)
                    dst = slice(-step, n_prim)
                    gap_arr = starts[dst] - ends[src]
                else:
                    src = slice(step, n_prim)
                    dst = slice(0, n_prim - step)
                    gap_arr = starts[src] - ends[dst]
                if gap_arr.size == 0:
                    continue
                dy_mid = np.abs(y_mids[dst] - y_mids[src])
                dh = np.abs(h_spans[dst] - h_spans[src])
                compat = np.where(
                    (dy_mid <= max_dy) & (dh <= max_dh) & (gap_arr <= max_gap1),
                    1.0,
                    np.where(
                        (np.abs(y_mids[src] - med_ymid) <= max_dy) & (gap_arr <= max_gap2),
                        0.82,
                        0.20,
                    ),
                ).astype(np.float32)
                ctx_num[dst] += w_step * s_local[src] * compat
                ctx_den[dst] += w_step

            has_w = ctx_den > 0
            s_ctx[has_w] = np.minimum(
                1.0, 0.58 * s_local[has_w] + 0.56 * (ctx_num[has_w] / ctx_den[has_w])
            )

        valid_mask = s_ctx >= 0.48
        if not np.any(valid_mask):
            return 0.0, []

        v_starts = starts[valid_mask]
        v_ends = ends[valid_mask]
        v_scores = s_ctx[valid_mask]
        v_widths = v_ends - v_starts

        total_ink_w = int(np.sum(v_widths))
        dx_osc_total = int(osc_prefix[-1])
        chain_bonus = (
            0.14
            if (len(v_starts) >= 3 or dx_osc_total >= 4)
            else (0.07 if len(v_starts) == 2 else 0.0)
        )
        chain_conf = float(np.clip(np.mean(v_scores) + chain_bonus, 0.0, 1.0))
        if total_ink_w < int(rw * 0.18):
            chain_conf *= 0.65

        # Step C: Relative Bimodal Gap Separation (Distinguishing Monospace/Proportional Kerning from Word Spaces)
        raw_words: List[Tuple[int, int, int]] = []  # (x0, x1, num_primitives)
        if len(v_starts) >= 2:
            gaps = v_starts[1:] - v_ends[:-1]
            med_pw = float(np.median(v_widths))
            g_min = float(gaps.min())
            g_max = float(gaps.max())

            # Distinguish discrete-character/monospace strips (where virtually all primitives are single
            # isolated glyphs: p85_pw <= 0.56 * rh and p85_pw <= 1.45 * med_pw) from proportional text strips
            # (where adjacent letters frequently fuse into multi-letter sub-word blobs so p85_pw > 0.56 * rh).
            k85_w = min(len(v_widths) - 1, int(len(v_widths) * 0.85))
            p85_pw = float(np.partition(v_widths, k85_w)[k85_w])
            k65_g = min(len(gaps) - 1, int(len(gaps) * 0.65))
            kern_base = float(np.partition(gaps, k65_g)[k65_g])

            is_monospace = bool(
                p85_pw <= 0.56 * rh
                and p85_pw <= 1.45 * med_pw
                and not (g_max >= max(4, int(round(rh * 0.45))) and kern_base <= 2.0 and g_max >= 3.0 * max(1.0, kern_base))
            )
            if len(v_starts) == 2 and (
                0.68 <= v_widths[0] / float(max(1, rh)) <= 1.38
                and 0.68 <= v_widths[1] / float(max(1, rh)) <= 1.38
                and gaps[0] >= max(3, int(round(rh * 0.22)))
            ):
                is_monospace = False
            if is_monospace:
                # Discrete-character / monospace font: a space character (' ') spans at least one full
                # character cell plus kerning (>= 0.36 * rh, >= 2.2 * kern_base + 1, and >= 0.85 * med_pw).
                kern_prop = kern_base
                min_space_thr = max(
                    int(round(rh * 0.36)),
                    int(round(kern_base * 2.2)) + 1,
                    int(round(med_pw * 0.85)),
                )
            else:
                # Proportional font (mix of single characters and multi-letter fused blobs):
                # intra-word kerning is the 50th-percentile gap (`kern_prop`, typically 1 px), while
                # inter-word spaces are >= max(3, round(rh * 0.22), round(kern_prop * 1.9) + 1).
                k50_g = len(gaps) >> 1
                kern_prop = float(np.partition(gaps, k50_g)[k50_g])
                if len(v_starts) == 2 and (
                    0.68 <= v_widths[0] / float(max(1, rh)) <= 1.38
                    and 0.68 <= v_widths[1] / float(max(1, rh)) <= 1.38
                ):
                    min_space_thr = max(3, int(round(rh * 0.22)))
                else:
                    min_space_thr = max(
                        3,
                        int(round(rh * 0.22)),
                        int(round(kern_prop * 1.9)) + 1,
                    )

            if g_max < min_space_thr:
                # All gaps on this strip are intra-word letter kerning (e.g. a single word, '[tiqur@nixos:~]$',
                # or a single contiguous path): do NOT split inside it!
                raw_words = [(int(v_starts[0]), int(v_ends[-1]), len(v_starts))]
            else:
                w_splits = np.flatnonzero(gaps >= min_space_thr)
                w_idx_s = np.concatenate(([0], w_splits + 1))
                w_idx_e = np.concatenate((w_splits, [len(v_starts) - 1]))
                min_tok_w = max(2, int(round(rh * 0.22)))
                raw_words = []
                for i0, i1 in zip(w_idx_s, w_idx_e):
                    ww = int(v_ends[i1] - v_starts[i0])
                    if ww < min_tok_w:
                        continue
                    if not is_monospace and ww > int(round(rh * 5.0)) and i1 > i0:
                        sub_g = gaps[i0:i1]
                        sub_thr = max(4, int(round(rh * 0.18)), int(round(kern_prop * 2.0)) + 1)
                        sub_sp = np.flatnonzero(sub_g >= sub_thr)
                        if len(sub_sp) > 0:
                            si0 = [int(i0)] + [int(i0 + sp + 1) for sp in sub_sp]
                            si1 = [int(i0 + sp) for sp in sub_sp] + [int(i1)]
                            for a, b_idx in zip(si0, si1):
                                if int(v_ends[b_idx] - v_starts[a]) >= min_tok_w:
                                    raw_words.append((int(v_starts[a]), int(v_ends[b_idx]), int(b_idx - a + 1)))
                            continue
                    raw_words.append((int(v_starts[i0]), int(v_ends[i1]), int(i1 - i0 + 1)))
        else:
            min_tok_w = max(2, int(round(rh * 0.22)))
            if int(v_ends[0] - v_starts[0]) >= min_tok_w:
                raw_words = [(int(v_starts[0]), int(v_ends[0]), 1)]

        # Step D: Pitch-Guided Sub-Pixel Valley Refinement ONLY for Single Un-Split Fused Primitives
        # Runs ONLY when a single continuous ink primitive (`n_sub == 1`) is wider than `4.2 * rh`
        # because anti-aliasing bridged across inter-word spaces in small proportional text.
        # Never re-splits a token formed by grouping separated monospace characters (`n_sub > 1`).
        refined_words: List[Tuple[int, int]] = []
        max_single_prim_w = int(round(rh * 4.2))
        min_word_step = max(int(round(rh * 1.25)), int(round(L_0 * 0.65)))
        for (wx0, wx1, n_sub) in raw_words:
            ww = wx1 - wx0
            if n_sub == 1 and ww > max_single_prim_w and rh <= int(round(L_0 * 1.85)):
                sub_e = col_mean[wx0:wx1]
                sub_m = col_max[wx0:wx1]
                idx80 = min(ww - 1, int(ww * 0.80))
                local_p80 = float(np.partition(sub_e, idx80)[idx80])
                cand_cuts: List[int] = []
                for c in range(min_word_step, ww - min_word_step):
                    if sub_e[c] <= sub_e[c - 1] and sub_e[c] <= sub_e[c + 1]:
                        if sub_e[c] <= local_p80 * 0.34 and sub_m[c] <= peak * 0.52:
                            if not cand_cuts or (c - cand_cuts[-1]) >= min_word_step:
                                cand_cuts.append(c)
                            elif sub_e[c] < sub_e[cand_cuts[-1]]:
                                cand_cuts[-1] = c
                if cand_cuts:
                    pts = [0] + cand_cuts + [ww]
                    for k in range(len(pts) - 1):
                        s0, s1 = wx0 + pts[k], wx0 + pts[k + 1]
                        nz = np.flatnonzero(col_max[s0:s1] > ink_thr * 0.80)
                        if len(nz) > 0:
                            s0, s1 = s0 + int(nz[0]), s0 + int(nz[-1]) + 1
                        if s1 - s0 >= max(2, int(round(rh * 0.25))):
                            refined_words.append((s0, s1))
                    continue
            refined_words.append((wx0, wx1))

        # Step E: Relative Badge-Clearance Merge
        # Merge ultra-narrow punctuation/1-char fragments (`w < 0.95 * rh`) with an adjacent word if the
        # gap between them is small (`< 0.75 * rh`), preventing adjacent hint badges from overlapping.
        if len(refined_words) >= 2:
            min_standalone_w = int(round(rh * 0.95))
            max_attach_gap = int(round(rh * 0.75))
            merged_words: List[Tuple[int, int]] = [refined_words[0]]
            for wx0, wx1 in refined_words[1:]:
                px0, px1 = merged_words[-1]
                pw = px1 - px0
                cw = wx1 - wx0
                gap = wx0 - px1
                is_icon_sep = (
                    (gap >= max(4, int(round(rh * 0.45))) and (0.68 <= pw / float(max(1, rh)) <= 1.38 or 0.68 <= cw / float(max(1, rh)) <= 1.38))
                    or (gap >= max(3, int(round(rh * 0.22))) and 0.68 <= pw / float(max(1, rh)) <= 1.38 and 0.68 <= cw / float(max(1, rh)) <= 1.38)
                )
                if not is_icon_sep and gap <= max_attach_gap and (pw < min_standalone_w or cw < min_standalone_w):
                    merged_words[-1] = (px0, wx1)
                else:
                    merged_words.append((wx0, wx1))
            refined_words = merged_words

        return chain_conf, refined_words

    @staticmethod
    def _find_foreground_line_bands(
        roi_gray: np.ndarray, roi_s: np.ndarray, min_line_h: int = 5
    ) -> List[Tuple[int, int]]:
        """
        Detect horizontal text line bands using direct foreground luminance contrast `|gray - bg_med|`
        so that 1-3 px vertical gaps between small/zoomed-out lines are not bridged by `MORPH_GRADIENT`
        vertical edge dilation, while never slicing through a left-side avatar or tall icon.
        """
        rh, rw = roi_gray.shape[:2]
        if rh < min_line_h * 2 + 2 or rw < rh * 2:
            return []
        b_top = roi_gray[0, :]
        b_bot = roi_gray[-1, :]
        bg_med = 0.5 * float(
            int(np.partition(b_top, rw >> 1)[rw >> 1])
            + int(np.partition(b_bot, rw >> 1)[rw >> 1])
        )
        fg = np.abs(roi_gray.astype(np.float32) - bg_med)
        if rw <= rh * 9 and float(fg[:, : min(rw, rh)].mean(axis=1).min()) >= 12.0:
            return []
        k95 = min(rw - 1, int(rw * 0.95))
        row_p95 = np.partition(fg, k95, axis=1)[:, k95]
        row_mean = fg.mean(axis=1)
        peak_mean = float(row_mean.max())
        peak_p95 = float(row_p95.max())
        if peak_mean < 12.0 or peak_p95 < 28.0:
            return []
        valley_mean_thr = max(peak_mean * 0.12, min(peak_mean * 0.24, 18.0))
        valley_p95_thr = max(peak_p95 * 0.18, min(peak_p95 * 0.36, 52.0))
        row_s = roi_s.sum(axis=1)
        row_s_peak = int(row_s.max())
        is_valley = (
            (row_mean <= valley_mean_thr)
            & (row_p95 <= valley_p95_thr)
            & (
                (row_s <= max(2, int(rw * 0.08), int(row_s_peak * 0.16)))
                | ((row_mean <= peak_mean * 0.10) & (row_p95 <= peak_p95 * 0.15))
            )
        )
        is_line_row = ~is_valley
        r_pad = np.concatenate(([False], is_line_row, [False]))
        r_starts = np.flatnonzero(r_pad[1:] & ~r_pad[:-1])
        r_ends = np.flatnonzero(~r_pad[1:] & r_pad[:-1])
        if len(r_starts) < 2:
            return []
        bands = [
            (int(bs), int(be))
            for bs, be in zip(r_starts, r_ends)
            if (be - bs) >= min_line_h
            and float(row_mean[bs:be].max()) >= peak_mean * 0.20
        ]
        return bands if len(bands) >= 2 else []

    @staticmethod
    def _has_rect_card_borders(
        x: int, y: int, w: int, h: int,
        dx_gray: np.ndarray, dy_gray: np.ndarray, gray: np.ndarray
    ) -> bool:
        H, W = gray.shape[:2]
        if w < 16 or h < 16:
            return False
        top_e = float((dy_gray[max(0, y - 3) : min(H, y + 4), x : x + w] >= 6).any(axis=0).mean())
        bot_e = 1.0 if (y + h >= H - 8) else float((dy_gray[max(0, y + h - 4) : min(H, y + h + 3), x : x + w] >= 6).any(axis=0).mean())
        left_e = float((dx_gray[y : y + h, max(0, x - 3) : min(W, x + 4)] >= 6).any(axis=1).mean())
        right_e = float((dx_gray[y : y + h, max(0, x + w - 4) : min(W, x + w + 3)] >= 6).any(axis=1).mean())
        sides = sorted([top_e, bot_e, left_e, right_e], reverse=True)
        min_side = sides[3]
        s3 = sides[2]
        if min_side >= 0.50 and (s3 >= 0.68 or min_side >= 0.62):
            return True
        if s3 >= 0.72:
            oy1, oy2 = max(0, y - 6), min(H, y + h + 6)
            ox1, ox2 = max(0, x - 6), min(W, x + w + 6)
            ring_mask = np.ones((oy2 - oy1, ox2 - ox1), dtype=bool)
            ring_mask[y - oy1 : y - oy1 + h, x - ox1 : x - ox1 + w] = False
            ring_vals = gray[oy1:oy2, ox1:ox2][ring_mask]
            if ring_vals.size > 0 and float(np.median(ring_vals)) < 36.0:
                return True
        return False

    @staticmethod
    def _is_real_photo_thumbnail(
        roi_gray: np.ndarray, roi_grad: np.ndarray, roi_chroma: np.ndarray
    ) -> bool:
        """
        Distinguish photographic/video thumbnails from bimodal flat UI cards/headers/code panes.
        """
        rh, rw = roi_gray.shape[:2]
        if rh < 14 or rw < 14:
            return False
        area = float(rh * rw)
        fr6 = float(np.count_nonzero(roi_grad <= 6)) / area
        if fr6 >= 0.65:
            return False
        hist = cv2.calcHist([roi_gray], [0], None, [32], [0, 256]).flatten() / area
        dom2 = float(np.sort(hist)[-2:].sum())
        if dom2 >= 0.64:
            return False
        ch_mean = float(roi_chroma.mean())
        if ch_mean < 5.5 and dom2 >= 0.42:
            return False
        active_tiles = 0
        for r_i in range(3):
            y0, y1 = (r_i * rh) // 3, ((r_i + 1) * rh) // 3
            for c_i in range(3):
                x0, x1 = (c_i * rw) // 3, ((c_i + 1) * rw) // 3
                sub = roi_gray[y0:y1, x0:x1]
                if sub.size > 0 and float(sub.std()) >= 9.5:
                    active_tiles += 1
        return active_tiles >= 5

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
        elif orig_W < 1440:
            # Aspect-safe upscale: standard landscape screenshots (<1440w) upscale to 1600w,
            # while tall portrait crops cap work_H at 1800px so L_0 and runtime never blow up.
            target_w = 1600
            if orig_H > orig_W * 1.25 and int(round(orig_H * (1600.0 / float(orig_W)))) > 1800:
                target_w = max(orig_W, int(round(orig_W * (1800.0 / float(orig_H)))))
            if target_w != orig_W:
                work_W = target_w
                work_H = max(1, int(round(orig_H * (float(work_W) / float(orig_W)))))
                bgr = cv2.resize(bgr, (work_W, work_H), interpolation=cv2.INTER_CUBIC)
                scale_back = orig_W / float(work_W)

        H, W = bgr.shape[:2]
        s = float(np.clip(np.sqrt((W * H) / float(1440 * 900)), 0.65, 2.0))

        # Stage 1: Gradients & JPEG-Resilient Ratio-Orthogonal Structural Barriers
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(bgr)
        chroma = cv2.subtract(cv2.max(cv2.max(b, g), r), cv2.min(cv2.min(b, g), r))
        blue_excess = cv2.subtract(b, r)

        dx_gray = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, self.k_dx)
        dy_gray = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, self.k_dy)
        dx_ch = (cv2.morphologyEx(chroma, cv2.MORPH_GRADIENT, self.k_dx).astype(np.uint16) * 166 >> 8).astype(np.uint8)
        dy_ch = (cv2.morphologyEx(chroma, cv2.MORPH_GRADIENT, self.k_dy).astype(np.uint16) * 166 >> 8).astype(np.uint8)
        dx = cv2.max(dx_gray, dx_ch)
        dy = cv2.max(dy_gray, dy_ch)
        grad_max = cv2.max(dx, dy)

        ortho_thr = 3 if (W * H <= 1500000) else 4
        k_hline = cv2.getStructuringElement(cv2.MORPH_RECT, (max(24, int(34 * s)), 1))
        k_hseg = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(26 * s)), 1))
        k_vline_short = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(16, int(20 * s))))
        k_hzone = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

        h_pure_raw = ((dy >= ortho_thr) & (dy.astype(np.uint16) >= (dx.astype(np.uint16) << 1))).astype(np.uint8) * 255
        h_pure = cv2.morphologyEx(h_pure_raw, cv2.MORPH_CLOSE, self.k_h_bridge)
        h_bar = cv2.morphologyEx(h_pure, cv2.MORPH_OPEN, k_hline)
        h_zone = cv2.dilate(cv2.morphologyEx(h_pure, cv2.MORPH_OPEN, k_hseg), k_hzone)

        v_pure_raw = ((dx >= ortho_thr) & (dx.astype(np.uint16) >= (dy.astype(np.uint16) << 1))).astype(np.uint8) * 255
        v_pure = cv2.morphologyEx(v_pure_raw, cv2.MORPH_CLOSE, self.k_v_bridge)
        v_raw = cv2.morphologyEx(v_pure, cv2.MORPH_OPEN, k_vline_short)
        v_cnts, _ = cv2.findContours(v_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_bar = np.zeros_like(v_raw)
        min_v_long = max(36, int(48 * s))
        for vc in v_cnts:
            vx, vy, vw, vh = cv2.boundingRect(vc)
            if vh >= min_v_long or cv2.countNonZero(h_zone[vy : vy + vh, vx : vx + vw]) > 0:
                cv2.drawContours(v_bar, [vc], -1, 255, -1)

        t_sig = time.perf_counter()

        # Stage 0/2: Self-Calibrating Image Physics (Noise Floor, Contrast Quantiles & Modal Font Height L_0)
        flat_thr, border_thr, stroke_thr, L_0, u = self._calibrate_image_physics(grad_max, W, H)
        s = float(np.clip(0.55 * s + 0.45 * u, 0.65, 2.0))

        _, flat_pix = cv2.threshold(grad_max, flat_thr, 255, cv2.THRESH_BINARY_INV)
        _, stroke_18 = cv2.threshold(grad_max, stroke_thr, 255, cv2.THRESH_BINARY)

        hv_small = cv2.resize(
            cv2.bitwise_or(h_pure, v_pure),
            (max(1, W >> 3), max(1, H >> 3)),
            interpolation=cv2.INTER_AREA,
        )
        _, hv_small = cv2.threshold(hv_small, 1, 255, cv2.THRESH_BINARY)
        hv_seed = cv2.resize(cv2.dilate(hv_small, self.k_d95), (W, H), interpolation=cv2.INTER_NEAREST)

        # Multi-threshold contour ladder: extracts closed frames and thumbnails across 3 gradient cutoffs
        cnts_list = []
        for b_thr in (max(2, border_thr - 1), max(4, border_thr + 1), max(6, border_thr + 3)):
            _, bp_iter = cv2.threshold(grad_max, b_thr, 255, cv2.THRESH_BINARY)
            cnts_list.extend(
                cv2.findContours(cv2.bitwise_and(bp_iter, hv_seed), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            )
        _, border_pix = cv2.threshold(grad_max, max(4, border_thr + 1), 255, cv2.THRESH_BINARY)

        strip_w = min(W, int(260 * s))
        _, border_hi = cv2.threshold(grad_max[:, :strip_w], 12, 255, cv2.THRESH_BINARY)
        cnts_hi, _ = cv2.findContours(border_hi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_hi:
            if len(c) >= 4:
                x, y, w, h = cv2.boundingRect(c)
                if 24 <= w <= int(72 * s) and 24 <= h <= int(72 * s) and abs(w - h) <= 6:
                    if cv2.contourArea(c) >= w * h * 0.72:
                        cnts_list.append(c)

        min_frame_h = max(14, int(15 * s))
        max_frame_h = max(46, int(58 * s))
        min_frame_w = max(18, int(20 * s))
        max_frame_w = max(440, int(W * 0.85))
        min_thumb_h = max(44, int(48 * s))
        max_thumb_h = max(255, int(H * 0.52))
        min_thumb_w = max(56, int(62 * s))
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
                if area < w * h * 0.68:
                    continue
                flat_r = cv2.countNonZero(
                    flat_pix[y + 2 : y + h - 2, x + 2 : x + w - 2]
                ) / float(max(1, (w - 4) * (h - 4)))
                if flat_r < 0.26:
                    continue
                tb_s = cv2.countNonZero(
                    h_pure[y : y + 4, x + 3 : x + w - 3]
                ) + cv2.countNonZero(h_pure[max(0, y + h - 4) : y + h, x + 3 : x + w - 3])
                lr_s = cv2.countNonZero(
                    v_pure[y + 3 : y + h - 3, x : x + 4]
                ) + cv2.countNonZero(v_pure[y + 3 : y + h - 3, max(0, x + w - 4) : x + w])
                tb_r = tb_s / float(max(1, w - 6))
                lr_r = lr_s / float(max(1, h - 6))
                is_pill = tb_s >= max(10, int((w - h) * 0.72)) and lr_s >= 4
                if not is_pill and (
                    tb_s < max(8, int((w - 8) * 0.45))
                    or (h >= 18 and w >= 48 and lr_s < max(4, int((h - 10) * 0.35)))
                ):
                    continue

                max_g_val = float(grad_max[y : y + h, x : x + w].max())
                if max_g_val < (26 if (tb_r >= 1.15 and lr_r >= 0.75) else 44):
                    continue

                inner_strokes = cv2.countNonZero(stroke_18[y + 3 : y + h - 3, x + 4 : x + w - 4])
                if inner_strokes < max(5, int((w - 8) * (h - 6) * 0.010)):
                    in_mean = float(gray[y + 3 : y + h - 3, x + 3 : x + w - 3].mean())
                    oy1, oy2 = max(0, y - 5), min(H, y + h + 5)
                    ox1, ox2 = max(0, x - 5), min(W, x + w + 5)
                    out_mean = float(gray[oy1:oy2, ox1:ox2].mean())
                    if abs(in_mean - out_mean) < 10.0 and not (w <= int(32 * s) and h <= int(32 * s) and tb_r >= 1.35):
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
                    if outer_flat < 0.62:
                        continue

                if w > int(85 * s):
                    iv = v_bar[y + 3 : y + h - 3, x + int(22 * s) : max(x + int(23 * s), x + w - int(32 * s))]
                    if iv.size > 0 and np.any(np.sum(iv > 0, axis=0) >= h - 6):
                        continue
                    top_b = (h_pure[y : y + 4, x : x + w] > 0).any(axis=0)
                    bot_b = (h_pure[max(0, y + h - 4) : y + h, x : x + w] > 0).any(axis=0)
                    tb_both = top_b & bot_b
                    if tb_both.mean() < 0.78:
                        vp_col = (v_pure[y + 3 : y + h - 3, x : x + w] > 0).sum(axis=0)
                        wall_cols = np.flatnonzero(vp_col >= max(4, int((h - 6) * 0.45)))
                        if len(wall_cols) >= 2:
                            if np.any((wall_cols >= int(24 * s)) & (wall_cols <= w - int(24 * s))):
                                continue

                raw_frames.append([x, y, w, h, 0.94, "frame", "control", []])

            elif min_thumb_h <= h <= max_thumb_h and min_thumb_w <= w <= max_thumb_w:
                area = cv2.contourArea(c)
                if area >= w * h * 0.76:
                    tb_s = cv2.countNonZero(
                        h_pure[y : y + 4, x + 6 : x + w - 6]
                    ) + cv2.countNonZero(h_pure[max(0, y + h - 4) : y + h, x + 6 : x + w - 6])
                    lr_s = cv2.countNonZero(
                        v_pure[y + 6 : y + h - 6, x : x + 4]
                    ) + cv2.countNonZero(v_pure[y + 6 : y + h - 6, max(0, x + w - 4) : x + w])
                    if tb_s < (w - 12) * 0.55 or lr_s < (h - 12) * 0.55:
                        continue
                    ar = w / float(h)
                    inner_hp = h_pure[y + int(0.6 * L_0) : max(y + int(0.6 * L_0) + 1, y + h - int(0.6 * L_0)), x + 6 : max(x + 7, x + w - 6)]
                    has_internal_button_dividers = (w <= int(7.8 * L_0) and h <= int(4.8 * L_0)) and inner_hp.size > 0 and float((inner_hp > 0).mean(axis=1).max()) >= 0.72
                    if (
                        0.48 <= ar <= 2.85
                        and not has_internal_button_dividers
                        and self._has_rect_card_borders(x, y, w, h, dx_gray, dy_gray, gray)
                        and self._is_real_photo_thumbnail(
                            gray[y + 3 : y + h - 3, x + 3 : x + w - 3],
                            grad_max[y + 3 : y + h - 3, x + 3 : x + w - 3],
                            chroma[y + 3 : y + h - 3, x + 3 : x + w - 3],
                        )
                    ):
                        raw_thumbs.append([x, y, w, h, 0.93, "thumbnail", "control", []])
                    else:
                        fr3 = cv2.countNonZero(
                            flat_pix[y + 3 : y + h - 3, x + 3 : x + w - 3]
                        ) / float(max(1, (w - 6) * (h - 6)))
                        if fr3 >= 0.72:
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
                >= 0.45 * f_area
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
        # Remove short inline word-highlight edges (<mark>search</mark>) embedded inside prose lines from h_bar & v_bar
        h_bar_clean = h_bar.copy()
        hb_cnts, _ = cv2.findContours(h_bar_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for hc in hb_cnts:
            hx, hy, hw, hh = cv2.boundingRect(hc)
            if hw < int(7.5 * L_0) and hh <= int(0.6 * L_0):
                by1, by2 = max(0, hy - int(1.1 * L_0)), min(H, hy + hh + int(1.1 * L_0))
                lx1, lx2 = max(0, hx - int(1.2 * L_0)), max(0, hx - 2)
                rx1, rx2 = min(W, hx + hw + 2), min(W, hx + hw + int(1.2 * L_0))
                l_str = cv2.countNonZero(stroke_18[by1:by2, lx1:lx2]) if lx2 > lx1 else 0
                r_str = cv2.countNonZero(stroke_18[by1:by2, rx1:rx2]) if rx2 > rx1 else 0
                if l_str >= int(0.9 * L_0) and r_str >= int(0.9 * L_0):
                    cv2.rectangle(h_bar_clean, (hx, hy), (hx + hw, hy + hh), 0, -1)

        v_bar_clean = v_bar.copy()
        vb_cnts, _ = cv2.findContours(v_bar_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for vc in vb_cnts:
            vx, vy, vw, vh = cv2.boundingRect(vc)
            if vh <= int(2.2 * L_0) and vw <= int(0.6 * L_0):
                lx1, lx2 = max(0, vx - int(1.2 * L_0)), max(0, vx - 2)
                rx1, rx2 = min(W, vx + vw + 2), min(W, vx + vw + int(1.2 * L_0))
                l_str = cv2.countNonZero(stroke_18[vy : vy + vh, lx1:lx2]) if lx2 > lx1 else 0
                r_str = cv2.countNonZero(stroke_18[vy : vy + vh, rx1:rx2]) if rx2 > rx1 else 0
                if l_str >= int(0.6 * L_0) and r_str >= int(0.6 * L_0):
                    cv2.rectangle(v_bar_clean, (vx, vy), (vx + vw, vy + vh), 0, -1)

        barrier = cv2.bitwise_or(h_bar_clean, v_bar_clean)
        for t in thumbs:
            tx, ty, tw, th = t[:4]
            cv2.rectangle(thumb_occ, (tx, ty), (tx + tw, ty + th), 255, -1)
            cv2.rectangle(barrier, (tx, ty), (tx + tw, ty + th), 255, 2)
        for f in frames:
            fx, fy, fw, fh = f[:4]
            if fh <= int(44 * s) and fw <= int(320 * s):
                cv2.rectangle(frame_occ, (fx, fy), (fx + fw, fy + fh), 255, -1)
            cv2.rectangle(barrier, (fx, fy), (fx + fw, fy + fh), 255, 2)
        barrier_dil = cv2.dilate(barrier, self.k_bar_dil)
        barrier_inv = cv2.bitwise_not(barrier_dil)

        t_frames = time.perf_counter()

        # Stage 3: Coarse Proposal Generation (Self-Calibrated to L_0 & stroke_thr)
        _, stroke_bin = cv2.threshold(grad_max, min(18, stroke_thr + 2), 255, cv2.THRESH_BINARY)
        stroke_clean = cv2.bitwise_and(stroke_bin, barrier_inv)

        _, stroke_01 = cv2.threshold(stroke_clean, 0, 1, cv2.THRESH_BINARY)
        _, blue_01 = cv2.threshold(blue_excess, 55, 1, cv2.THRESH_BINARY)

        kc_w = max(12, int(round(0.88 * L_0)))
        kc_h = max(4, int(round(0.32 * L_0)))
        k_c = cv2.getStructuringElement(cv2.MORPH_RECT, (kc_w, kc_h))
        coarse_mask = cv2.bitwise_and(
            cv2.morphologyEx(stroke_clean, cv2.MORPH_CLOSE, k_c),
            barrier_inv,
        )
        c_cnts, _ = cv2.findContours(coarse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coarse_boxes = [
            (cx, cy, cw, ch)
            for c in c_cnts
            for (cx, cy, cw, ch) in [cv2.boundingRect(c)]
            if (cw >= max(4, int(5 * s)) and ch >= max(5, int(6 * s)))
            or (max(5, int(7 * s)) <= cw <= max(28, int(34 * s)) and ch >= 1)
            or (cw >= 3 and max(10, int(12 * s)) <= ch <= max(36, int(42 * s)))
        ]

        t_coarse = time.perf_counter()

        # Stage 4: Multi-Way 1D X-Y Projection, Foreground Line Valley Split & Contextual Glyph-Chain Engine
        refined_boxes = []
        inner_frame_children = []

        def refine_region(rx: int, ry: int, rw: int, rh: int, depth: int = 0):
            if rw < 3 or rh < 1 or depth > 16:
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
                ((max(5, int(7 * s)) <= rw <= max(28, int(32 * s)) and 1 <= rh <= max(6, int(7 * s)))
                 or (3 <= rw <= max(8, int(10 * s)) and max(10, int(12 * s)) <= rh <= max(34, int(40 * s))))
                and depth <= 3
            )
            is_single_char_ui = (
                max(3, int(round(0.24 * L_0))) <= rw <= max(12, int(round(0.80 * L_0)))
                and max(6, int(round(0.42 * L_0))) <= rh <= max(22, int(round(1.40 * L_0)))
                and depth <= 3
            )
            min_box_w = max(5, int(round(0.35 * L_0)))
            min_box_h = max(4, int(round(0.30 * L_0)))
            if not is_micro_icon and not is_single_char_ui and (rw < min_box_w or rh < min_box_h or (rw <= int(round(0.55 * L_0)) and rh <= int(round(0.45 * L_0)))):
                return

            icon_guard_w = max(36, int(38 * s))
            icon_guard_h = max(34, int(36 * s))
            if rh >= max(13, int(1.05 * L_0)):
                if rh >= max(18, int(20 * s)):
                    v_thr = max(2, int(rw * 0.015)) if (rw >= int(160 * s) and int(row_proj.max()) >= int(80 * s)) else 0
                    is_active_row = row_proj > v_thr
                    r_pad = np.concatenate(([False], is_active_row, [False]))
                    r_starts = np.flatnonzero(r_pad[1:] & ~r_pad[:-1])
                    r_ends = np.flatnonzero(~r_pad[1:] & r_pad[:-1])
                    if len(r_starts) >= 2:
                        v_gaps = r_starts[1:] - r_ends[:-1]
                        min_v_gap = 6 if (rw <= icon_guard_w and rh <= icon_guard_h) else 1
                        split_mask = v_gaps >= min_v_gap
                        if np.any(split_mask):
                            band_starts = np.concatenate(([r_starts[0]], r_starts[1:][split_mask]))
                            band_ends = np.concatenate((r_ends[:-1][split_mask], [r_ends[-1]]))
                            if len(band_starts) >= 2:
                                for bs, be in zip(band_starts, band_ends):
                                    if be - bs >= 1:
                                        refine_region(rx, ry + int(bs), rw, int(be - bs), depth + 1)
                                return

                # Foreground-Contrast Multi-Line Valley Split:
                # When 2-3 small text lines have a 1-3px vertical gap that `dy` morphological gradient
                # dilated across (e.g., 2-line YouTube titles, 2-line Discord messages, zoomed-out paragraphs),
                # split along quiet horizontal luminance valleys.
                if rw > icon_guard_w and rh <= int(3.4 * L_0) and rw >= int(2.2 * rh):
                    fg_bands = self._find_foreground_line_bands(
                        gray[ry : ry + rh, rx : rx + rw],
                        roi_s,
                        min_line_h=max(5, int(0.36 * L_0)),
                    )
                    if len(fg_bands) >= 2:
                        for bs, be in fg_bands:
                            if be - bs >= 4:
                                refine_region(rx, ry + bs, rw, be - bs, depth + 1)
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
                                refine_region(rx + int(starts[idx_r + 1]), ry, rw - int(starts[idx_r + 1]), rh, depth + 1)
                                return

                run_widths = ends - starts
                is_dot_track = (
                    rh <= int(1.15 * L_0)
                    and len(starts) >= 3
                    and np.mean(run_widths <= max(10, int(0.82 * L_0))) >= 0.70
                )
                gaps = starts[1:] - ends[:-1]
                base_gap_thr = max(int(1.22 * L_0), int(rh * 1.12)) if rh >= int(1.05 * L_0) else max(int(0.95 * L_0), int(rh * 1.18))
                if not is_dot_track and rw >= int(12.0 * L_0) and rh <= int(2.4 * L_0) and len(gaps) >= 2:
                    best_mask = None
                    best_score = -1e9
                    roi_fl_band = flat_pix[ry : ry + rh, rx : rx + rw]
                    for mult in (0.85, 1.0, 1.22):
                        cand_thr = max(int(0.80 * L_0), int(round(base_gap_thr * mult)))
                        cand_mask = (gaps >= cand_thr) & ~((ends[:-1] <= int(1.6 * L_0)) & (gaps <= int(1.35 * L_0)))
                        if not np.any(cand_mask):
                            continue
                        c_starts = np.concatenate(([starts[0]], starts[1:][cand_mask]))
                        c_ends = np.concatenate((ends[:-1][cand_mask], [ends[-1]]))
                        seg_ws = c_ends - c_starts
                        cut_indices = np.flatnonzero(cand_mask)
                        gutter_flat = float(np.mean([
                            cv2.countNonZero(roi_fl_band[:, int(ends[ci]) : int(starts[ci + 1])])
                            / float(max(1, rh * (int(starts[ci + 1]) - int(ends[ci]))))
                            for ci in cut_indices
                        ]))
                        tiny_pen = float(np.mean(seg_ws < int(1.4 * L_0))) * 0.65
                        giant_pen = 0.45 if np.max(seg_ws) > int(16.0 * L_0) else 0.0
                        sc_val = gutter_flat - tiny_pen - giant_pen
                        if sc_val > best_score:
                            best_score = sc_val
                            best_mask = cand_mask
                    h_splits = best_mask if (best_mask is not None and best_score >= 0.35) else (
                        (gaps >= base_gap_thr) & ~((ends[:-1] <= int(1.6 * L_0)) & (gaps <= int(1.35 * L_0)))
                    )
                else:
                    h_splits = (
                        np.zeros_like(gaps, dtype=bool)
                        if is_dot_track
                        else ((gaps >= base_gap_thr) & ~((ends[:-1] <= int(1.6 * L_0)) & (gaps <= int(1.35 * L_0))))
                    )
                if np.any(h_splits):
                    seg_starts = np.concatenate(([starts[0]], starts[1:][h_splits]))
                    seg_ends = np.concatenate((ends[:-1][h_splits], [ends[-1]]))
                    if len(seg_starts) >= 2:
                        for ss, se in zip(seg_starts, seg_ends):
                            if se - ss >= 3:
                                refine_region(rx + int(ss), ry, int(se - ss), rh, depth + 1)
                        return

            if rh > max(42, int(2.9 * L_0)):
                if (
                    max(50, int(3.5 * L_0)) <= rh <= max_thumb_h
                    and max(68, int(4.6 * L_0)) <= rw <= max_thumb_w
                    and 0.48 <= rw / float(rh) <= 2.85
                    and self._has_rect_card_borders(rx, ry, rw, rh, dx_gray, dy_gray, gray)
                    and self._is_real_photo_thumbnail(
                        gray[ry + 3 : ry + rh - 3, rx + 3 : rx + rw - 3],
                        grad_max[ry + 3 : ry + rh - 3, rx + 3 : rx + rw - 3],
                        chroma[ry + 3 : ry + rh - 3, rx + 3 : rx + rw - 3],
                    )
                ):
                    if not inside_frame:
                        refined_boxes.append([rx, ry, rw, rh, 0.91, "thumbnail", "control", []])
                    return

                col_span = np.count_nonzero(roi_s.sum(axis=0) >= int(rh * 0.28)) / float(max(1, rw))
                is_giant_heading = (rh <= int(4.2 * L_0) and rw <= int(38.0 * L_0) and col_span >= 0.42)
                if not is_giant_heading:
                    if depth <= 6:
                        roi_gm = grad_max[ry : ry + rh, rx : rx + rw]
                        roi_bi = barrier_inv[ry : ry + rh, rx : rx + rw]
                        t_thr = max(24, int(stroke_thr * 1.55))
                        kw_sub = max(8, int(0.65 * L_0))
                        kh_sub = 2
                        _, sub_hi = cv2.threshold(roi_gm, t_thr, 255, cv2.THRESH_BINARY)
                        sub_hi = cv2.bitwise_and(sub_hi, roi_bi)
                        k_tr = cv2.getStructuringElement(cv2.MORPH_RECT, (kw_sub, kh_sub))
                        sub_closed = cv2.morphologyEx(sub_hi, cv2.MORPH_CLOSE, k_tr)
                        s_cnts, _ = cv2.findContours(sub_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        raw_b = [cv2.boundingRect(sc) for sc in s_cnts]
                        tiny_dots = sum(1 for (_, _, sw, sh) in raw_b if 4 <= sw <= int(1.7 * L_0) and 4 <= sh <= int(0.95 * L_0))
                        if len(raw_b) >= 18 and tiny_dots >= 0.65 * len(raw_b):
                            k_track = cv2.getStructuringElement(cv2.MORPH_RECT, (max(24, int(2.2 * L_0)), 2))
                            sub_closed = cv2.morphologyEx(sub_hi, cv2.MORPH_CLOSE, k_track)
                            s_cnts, _ = cv2.findContours(sub_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            raw_b = [cv2.boundingRect(sc) for sc in s_cnts]
                        for (sx, sy, sw, sh) in raw_b:
                            if (sw >= 5 and sh >= 5) and (sw * sh < 0.82 * rw * rh) and sh <= int(4.0 * L_0):
                                refine_region(rx + sx, ry + sy, sw, sh, depth + 2)
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

                    min_side_w = max(18, int(round(rh * 1.35)))
                    left_gap_ok = bs_t == 0 or (bs_t >= min_side_w and int(is_ink[max(0, bs_t - 3) : min(rw, bs_t + 2)].sum()) <= 2)
                    right_gap_ok = be_t == rw or ((rw - be_t) >= min_side_w and int(is_ink[max(0, be_t - 2) : min(rw, be_t + 3)].sum()) <= 2)
                    has_left_part = bs_t >= min_side_w and col_proj[:bs_t].sum() >= 12
                    has_right_part = (rw - be_t) >= min_side_w and col_proj[be_t:].sum() >= 12
                    has_left_neutral = has_left_part and col_blue[:bs_t].sum() <= int(col_proj[:bs_t].sum() * 0.12)
                    has_right_neutral = has_right_part and col_blue[be_t:].sum() <= int(col_proj[be_t:].sum() * 0.12)
                    if bw_t >= max(28, int(round(rh * 1.6))) and left_gap_ok and right_gap_ok and (has_left_neutral or has_right_neutral):
                        if has_left_part:
                            refine_region(rx, ry, bs_t, rh, depth + 1)
                        refine_region(rx + bs_t, ry, bw_t, rh, depth + 1)
                        if has_right_part:
                            refine_region(rx + be_t, ry, rw - be_t, rh, depth + 1)
                        return

            if (rw <= int(10 * s) and rh >= int(44 * s)) or (rh <= int(4 * s) and rw >= int(65 * s)):
                return

            pad_x1, pad_y1 = max(0, rx - 6), max(0, ry - 6)
            pad_x2, pad_y2 = min(W, rx + rw + 6), min(H, ry + rh + 6)
            local_flat = cv2.countNonZero(flat_pix[pad_y1:pad_y2, pad_x1:pad_x2]) / float(
                (pad_x2 - pad_x1) * (pad_y2 - pad_y1)
            )

            if rw <= int(110 * s) and rh <= int(110 * s) and local_flat >= 0.55:
                min_dx_mean = 1.5 / max(0.85, s * 0.85)
            else:
                min_dx_mean = 4.2
            mean_dx_val = float(cv2.mean(dx[ry : ry + rh, rx : rx + rw])[0])
            _, max_g, _, _ = cv2.minMaxLoc(grad_max[ry : ry + rh, rx : rx + rw])
            # Smooth contrast threshold scaled to calibrated `stroke_thr` and local background flatness `local_flat`
            base_g_thr = max(36.0, float(stroke_thr) * (2.35 if local_flat >= 0.58 else 3.20))
            if is_micro_icon:
                min_g_thr = max(52.0, float(stroke_thr) * 3.30)
            elif is_single_char_ui:
                is_sym_icon = (
                    local_flat >= 0.62
                    and rw >= max(6, int(round(0.40 * L_0)))
                    and rh >= max(6, int(round(0.40 * L_0)))
                    and 0.62 <= rw / float(max(1, rh)) <= 1.55
                )
                min_g_thr = base_g_thr if is_sym_icon else max(78.0, float(stroke_thr) * 4.70)
            else:
                min_g_thr = base_g_thr

            active_cols = np.count_nonzero(is_ink)
            mean_ink_col = float(col_proj.sum()) / float(max(1, active_cols))

            # Check standard v8 filters first (0.0 ms overhead for normal boxes)
            std_pass = True
            if not is_micro_icon and mean_dx_val < min_dx_mean:
                std_pass = False
            elif max_g < min_g_thr:
                std_pass = False
            elif rh >= int(18 * s) and rw >= int(80 * s) and mean_ink_col < 2.5:
                std_pass = False
            elif is_micro_icon and local_flat < 0.58:
                std_pass = False
            elif is_single_char_ui and (rw < 6 or rh < 6 or (rw <= 9 and rh <= 8)) and (local_flat < 0.74 or (max_g < 95 and not (rw >= 6 and rh >= 6 and 0.70 <= rw / float(max(1, rh)) <= 1.40 and local_flat >= 0.74))):
                std_pass = False
            elif local_flat < 0.48 and max_g < 72:
                std_pass = False
            elif local_flat < 0.34 and (max_g < 140 or rw < 24):
                std_pass = False

            glyph_words: Optional[List[Tuple[int, int]]] = None
            if not std_pass:
                # 1D Contextual Glyph-Chain Rescue (`[L, L, ?, L, L]`):
                # If a small/zoomed-out/muted text strip failed the hard gradient/flatness cutoffs,
                # check if it forms a high-confidence collinear glyph chain!
                if (
                    not is_micro_icon
                    and not is_single_char_ui
                    and rw >= max(14, int(1.6 * rh))
                    and rh <= max(34, int(2.2 * L_0))
                    and local_flat >= 0.26
                    and max_g >= 36
                    and mean_dx_val >= 2.2
                ):
                    chain_conf, gw = self._score_and_segment_glyph_chain(
                        gray[ry : ry + rh, rx : rx + rw],
                        dx[ry : ry + rh, rx : rx + rw],
                        dy[ry : ry + rh, rx : rx + rw],
                        roi_s,
                        L_0,
                    )
                    if chain_conf >= 0.62 and len(gw) >= 1:
                        glyph_words = gw
                    else:
                        return
                else:
                    return

            total_stroke = int(col_proj.sum())
            is_accent = total_blue >= max(8, int(total_stroke * 0.22)) and rw >= 24

            max_icon_dim = max(36, int(44 * s))
            if (
                is_micro_icon
                or is_single_char_ui
                or (rw <= max_icon_dim and rh <= max_icon_dim and 0.52 <= rw / float(max(1, rh)) <= 1.80)
            ):
                kind = "icon"
                score = 0.84
                level = "control"
            elif is_accent and rw <= int(350 * s):
                kind = "inline_link"
                score = 0.92
                level = "control"
            elif rw >= int(240 * s) or (rh <= int(9 * s) and rw >= int(45 * s)):
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
            if kind in ("text_block", "text_link", "inline_link") and rh <= int(56 * s) and (len(starts) >= 2 or rw >= max(18, int(rh * 2.2))):
                if glyph_words is None:
                    _, glyph_words = self._score_and_segment_glyph_chain(
                        gray[ry : ry + rh, rx : rx + rw],
                        dx[ry : ry + rh, rx : rx + rw],
                        dy[ry : ry + rh, rx : rx + rw],
                        roi_s,
                        L_0,
                    )
                raw_words = glyph_words
                if len(raw_words) >= 1:
                    w_hints = generate_vimium_hints(len(raw_words))
                    for wi, (wx0, wx1) in enumerate(raw_words):
                        ww = wx1 - wx0
                        is_lead_icon = (
                            wi == 0
                            and len(raw_words) >= 2
                            and rh >= 14
                            and 0.65 <= ww / float(rh) <= 1.40
                            and (raw_words[1][0] - wx1) >= max(5, int(rh * 0.25))
                        )
                        word_children.append(
                            {
                                "x": int((rx + wx0) * scale_back),
                                "y": int(ry * scale_back),
                                "w": max(1, int(ww * scale_back)),
                                "h": max(1, int(rh * scale_back)),
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

        # Merge Stage 4 thumbnails into `thumbs` and run Peer-Grid + Cross-Row Shelf Completion
        s4_thumbs = [rb for rb in refined_boxes if rb[5] == "thumbnail"]
        refined_boxes = [rb for rb in refined_boxes if rb[5] != "thumbnail"]
        for st in s4_thumbs:
            sx, sy, sw, sh = st[:4]
            s_area = sw * sh
            if any(
                max(0, min(sx + sw, kx + kw) - max(sx, kx))
                * max(0, min(sy + sh, ky + kh) - max(sy, ky))
                >= 0.40 * min(s_area, kw * kh)
                for kx, ky, kw, kh, *_ in thumbs
            ):
                continue
            thumbs.append(st)

        # Complete same-row shelves and cross-row 16:9 grids (e.g. YouTube Dark Mode videos & Shorts)
        major_thumbs = [t for t in thumbs if t[2] >= int(90 * s) and t[3] >= int(75 * s)]
        if len(major_thumbs) >= 2:
            major_thumbs.sort(key=lambda t: (t[1], t[0]))
            used_sh = set()
            shelves = []
            for i, ti in enumerate(major_thumbs):
                if i in used_sh:
                    continue
                sh_list = [ti]
                used_sh.add(i)
                for j, tj in enumerate(major_thumbs):
                    if (
                        j not in used_sh
                        and abs(tj[1] - ti[1]) <= int(22 * s)
                        and abs(tj[3] - ti[3]) <= int(55 * s)
                        and abs(tj[2] - ti[2]) <= int(75 * s)
                    ):
                        sh_list.append(tj)
                        used_sh.add(j)
                sh_list.sort(key=lambda t: t[0])
                shelves.append(sh_list)

            known_16_9_grids = []
            new_grid_thumbs = []
            for sh_list in shelves:
                if len(sh_list) >= 2:
                    med_y = int(round(np.min([t[1] for t in sh_list])))
                    med_h = int(round(np.max([t[3] for t in sh_list])))
                    med_w = int(round(np.max([t[2] for t in sh_list])))
                    has_uniform_pair = any(
                        p1 != p2 and (abs(sh_list[p1][2] - sh_list[p2][2]) <= int(18 * s) or abs(sh_list[p1][3] - sh_list[p2][3]) <= int(18 * s))
                        for p1 in range(len(sh_list)) for p2 in range(p1 + 1, len(sh_list))
                    )
                    if has_uniform_pair and med_w >= int(110 * s) and med_h >= int(85 * s):
                        for t in sh_list:
                            if t[2] < med_w - 3 and t[0] + med_w <= W - 4:
                                t[2] = med_w
                            if t[3] < med_h - 3 and med_y + med_h <= H - 2:
                                t[1] = med_y
                                t[3] = med_h
                        diffs = [sh_list[k + 1][0] - sh_list[k][0] for k in range(len(sh_list) - 1)]
                        valid_p = []
                        for d in diffs:
                            for mult in (1, 2, 3, 4, 5):
                                cp = d / float(mult)
                                if med_w - int(3 * s) <= cp <= med_w + int(36 * s):
                                    valid_p.append(cp)
                        if valid_p:
                            pitch = float(np.median(valid_p))
                            anchor_x = sh_list[0][0]
                            min_feed_x = min([t[0] for t in major_thumbs]) - int(18 * s)
                            col_xs = []
                            for step in range(-4, 7):
                                gx = int(round(anchor_x + step * pitch))
                                if gx < max(int(60 * s), min_feed_x) or gx + med_w > W - 4:
                                    continue
                                col_xs.append(gx)
                                overlapping = [
                                    ex for ex in thumbs + new_grid_thumbs
                                    if max(0, min(gx + med_w, ex[0] + ex[2]) - max(gx, ex[0])) >= 0.35 * min(med_w, ex[2])
                                    and max(0, min(med_y + med_h, ex[1] + ex[3]) - max(med_y, ex[1])) >= 0.30 * min(med_h, ex[3])
                                ]
                                if overlapping:
                                    main_ex = max(overlapping, key=lambda e: e[2] * e[3])
                                    main_ex[0], main_ex[1], main_ex[2], main_ex[3] = gx, med_y, med_w, med_h
                                else:
                                    roi_g = gray[med_y + 4 : med_y + med_h - 4, gx + 4 : gx + med_w - 4]
                                    roi_fl = flat_pix[med_y + 4 : med_y + med_h - 4, gx + 4 : gx + med_w - 4]
                                    fl_r = cv2.countNonZero(roi_fl) / float(max(1, roi_fl.size))
                                    has_cap = any(
                                        abs(rb[0] - gx) <= int(2.8 * L_0) and 0 <= rb[1] - (med_y + med_h) <= int(1.8 * L_0) and rb[2] >= int(3.2 * L_0)
                                        for rb in refined_boxes
                                    )
                                    if roi_g.size > 0 and (
                                        (float(roi_g.std()) >= 15.0 and fl_r < 0.72)
                                        or (float(roi_g.std()) >= 11.0 and has_cap)
                                    ):
                                        new_grid_thumbs.append([gx, med_y, med_w, med_h, 0.92, "thumbnail", "control", []])
                            if 1.45 <= med_w / float(med_h) <= 2.05 and len(col_xs) >= 2:
                                known_16_9_grids.append((med_w, med_h, pitch, col_xs))

            for sh_list in shelves:
                if len(sh_list) == 1 and known_16_9_grids:
                    t_first = sh_list[0]
                    tx, ty, tw, th = t_first[:4]
                    for (gw, gh, gpitch, gcols) in known_16_9_grids:
                        if abs(tw - gw) <= int(14 * s) and abs(th - gh) <= int(14 * s):
                            if any(abs(tx - cx) <= int(12 * s) for cx in gcols):
                                for step in range(-4, 5):
                                    gx = int(round(tx + step * gpitch))
                                    if gx < int(60 * s) or gx + gw > W - 4:
                                        continue
                                    if any(
                                        max(0, min(gx + gw, ex[0] + ex[2]) - max(gx, ex[0])) >= 0.35 * gw
                                        and max(0, min(ty + gh, ex[1] + ex[3]) - max(ty, ex[1])) >= 0.35 * gh
                                        for ex in thumbs + new_grid_thumbs
                                    ):
                                        continue
                                    roi_g = gray[ty + 4 : ty + gh - 4, gx + 4 : gx + gw - 4]
                                    has_caption_below = any(
                                        abs(rb[0] - gx) <= int(45 * s) and 0 <= rb[1] - (ty + gh) <= int(28 * s) and rb[2] >= int(50 * s)
                                        for rb in refined_boxes
                                    )
                                    if roi_g.size > 0 and (float(roi_g.std()) >= 18.0 or (float(roi_g.std()) >= 11.0 and has_caption_below)):
                                        new_grid_thumbs.append([gx, ty, gw, gh, 0.92, "thumbnail", "control", []])
                                break

            thumbs.extend(new_grid_thumbs)

        thumbs.sort(key=lambda t: t[2] * t[3], reverse=True)
        clean_thumbs = []
        for t in thumbs:
            tx, ty, tw, th = t[:4]
            t_area = tw * th
            if any(
                max(0, min(tx + tw, kx + kw) - max(tx, kx))
                * max(0, min(ty + th, ky + kh) - max(ty, ky))
                >= 0.45 * t_area
                for kx, ky, kw, kh, *_ in clean_thumbs
            ):
                continue
            clean_thumbs.append(t)
        thumbs = clean_thumbs

        frames = [
            f for f in frames
            if not any(
                max(0, min(f[0] + f[2], tx + tw) - max(f[0], tx))
                * max(0, min(f[1] + f[3], ty + th) - max(f[1], ty))
                >= 0.45 * (f[2] * f[3])
                for tx, ty, tw, th, *_ in thumbs
            )
        ]

        refined_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        dedup_rbs = []
        for rb in refined_boxes:
            rx, ry, rw, rh = rb[:4]
            r_area = rw * rh
            if any(
                max(0, min(rx + rw, tx + tw) - max(rx, tx))
                * max(0, min(ry + rh, ty + th) - max(ry, ty))
                >= 0.35 * r_area
                for tx, ty, tw, th, *_ in thumbs
            ):
                continue
            if any(
                max(0, min(rx + rw, kx + kw) - max(rx, kx))
                * max(0, min(ry + rh, ky + kh) - max(ry, ky))
                >= 0.72 * r_area
                for kx, ky, kw, kh, *_ in dedup_rbs
            ):
                continue
            dedup_rbs.append(rb)
        refined_boxes = dedup_rbs
        refined_boxes.sort(key=lambda b: (b[1] // 14, b[0]))

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

        # Stage 5: Clean, Non-Overlapping Compound Pairing
        v_pure_gray = (dx_gray >= 4) & (dy_gray <= 2)
        h_pure_gray = (dy_gray >= 4) & (dx_gray <= 2)
        used_rb = [False] * len(refined_boxes)
        used_fr = [False] * len(frames)
        desktop_icons = []

        def make_child_dict(bx, by, bw, bh, bsc, bkd, blv, bhnt, bch=None):
            return {
                "x": int(bx * scale_back),
                "y": int(by * scale_back),
                "w": max(1, int(bw * scale_back)),
                "h": max(1, int(bh * scale_back)),
                "score": round(float(bsc), 2),
                "kind": str(bkd),
                "level": str(blv),
                "hint": str(bhnt),
                "children": list(bch) if bch else [],
            }

        max_h_icon = max(62, int(68 * s))
        for src_type, src_list, used_src in [("fr", frames, used_fr), ("rb", refined_boxes, used_rb)]:
            for idx1, item1 in enumerate(src_list):
                if used_src[idx1]:
                    continue
                fx, fy, fw, fh, fsc, fkd = item1[:6]
                is_square_icon_cand = (
                    fkd in ("frame", "icon", "inline_link")
                    or (fkd == "text_link" and 12 <= fw <= max_h_icon and 12 <= fh <= max_h_icon and 0.50 <= fw / float(fh) <= 1.65)
                )
                if (
                    is_square_icon_cand
                    and 12 <= fw <= max_h_icon
                    and 12 <= fh <= max_h_icon
                    and 0.48 <= fw / float(fh) <= 1.75
                ):
                    best_rbi = None
                    for rbi, rb2 in enumerate(refined_boxes):
                        if used_rb[rbi] or (src_type == "rb" and idx1 == rbi):
                            continue
                        rx, ry, rw, rh, rsc, rkd = rb2[:6]
                        if (
                            rkd in ("text_link", "inline_link", "text_block")
                            and max(16, int(16 * s)) <= rw <= int(340 * s)
                            and rh <= max(32, int(38 * s))
                            and rw > rh * 1.10
                        ):
                            h_gap = rx - (fx + fw)
                            if (
                                2 <= h_gap <= max(40, int(48 * s))
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

        horiz_rail_frames = set()
        for fi, f1 in enumerate(frames):
            peers = [
                fj for fj, f2 in enumerate(frames)
                if fi != fj and abs(f1[1] - f2[1]) <= int(6 * s) and abs(f1[3] - f2[3]) <= int(6 * s)
            ]
            if len(peers) >= 2:
                horiz_rail_frames.add(fi)

        for src_type, src_list, used_src in [("fr", frames, used_fr), ("rb", refined_boxes, used_rb)]:
            for idx1, item1 in enumerate(src_list):
                if used_src[idx1]:
                    continue
                if src_type == "fr" and idx1 in horiz_rail_frames:
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
                            and 0 <= v_gap <= max(18, int(20 * s))
                            and abs(fcx - rcx) <= max(26, int(28 * s))
                        ):
                            gy1, gy2 = fy + fh, ry
                            gx1, gx2 = min(fx, rx), max(fx + fw, rx + rw)
                            if gy2 > gy1 and gx2 > gx1 and np.any(h_pure_gray[gy1:gy2, gx1:gx2].mean(axis=1) >= 0.50):
                                continue
                            best_lbl = ("rb", rbi, rx, ry, rw, rh, rb2[7])
                            break
                    if best_lbl is not None:
                        kind_l, idx_l, rx, ry, rw, rh, lch = best_lbl
                        used_rb[idx_l] = True
                        used_src[idx1] = True
                        nx = min(fx, rx)
                        ny = min(fy, ry)
                        nw = max(fx + fw, rx + rw) - nx
                        nh = max(fy + fh, ry + rh) - ny
                        comp_ch = [
                            make_child_dict(fx, fy, fw, fh, fsc, "icon", "control", "S"),
                            make_child_dict(rx, ry, rw, rh, rsc, "text_link", "line", "A", lch),
                        ]
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

        for ic in inner_frame_children:
            ix, iy, iw, ih, isc, ikd, ilv, ich = ic
            icx, icy = ix + iw * 0.5, iy + ih * 0.5
            for ff in final_frames:
                fx, fy, fw, fh = ff[:4]
                if fx <= icx <= fx + fw and fy <= icy <= fy + fh and (iw * ih) < 0.90 * (fw * fh):
                    ff[7].append(
                        make_child_dict(
                            ix, iy, iw, ih, isc, ikd, ilv,
                            generate_vimium_hints(len(ff[7]) + 1)[-1],
                            ich,
                        )
                    )
                    break

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

        all_other = thumbs + desktop_icons + final_frames
        base_min_dim = max(26, int(30 * s))
        base_pad = max(6, int(8 * s))
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

        if collapse_paragraphs and include_text:
            # User-Designed Area & Structure-Aware Line & Paragraph Engine:
            # Detect vertical column rails (left-edges shared by >= 4 stacked items) so a left-column
            # table/infobox label ('Original language', 'Written by') never merges across a column gutter.
            def is_column_rail(x_val: int, y_val: int) -> bool:
                peers = [
                    rb for rb in rem_rbs
                    if rb[5] in ("text_block", "text_link", "inline_link")
                    and abs(rb[0] - x_val) <= max(4, int(round(0.28 * L_0)))
                    and abs(rb[1] - y_val) <= int(round(12.0 * L_0))
                ]
                return len(peers) >= 4

            # 1. Consolidate collinear same-line fragments (inline links, highlighted terms, badges on same row)
            #    Partition each row band left-to-right across all `start_k in row_cands` so non-merged right-hand
            #    segments are NEVER dropped, and enforce per-word/segment monotonic area growth & fill conservation.
            line_consolidated = []
            rem_rbs.sort(key=lambda b: (b[1] + (b[3] >> 1), b[0]))
            used_cons = set()
            for i, rbi in enumerate(rem_rbs):
                if i in used_cons:
                    continue
                if rbi[5] not in ("text_block", "text_link", "inline_link"):
                    line_consolidated.append(rbi)
                    continue
                row_cands = [i]
                ry0, ry1 = rbi[1], rbi[1] + rbi[3]
                for j in range(len(rem_rbs)):
                    if j == i or j in used_cons:
                        continue
                    rbj = rem_rbs[j]
                    if rbj[5] not in ("text_block", "text_link", "inline_link"):
                        continue
                    v_ov = min(ry1, rbj[1] + rbj[3]) - max(ry0, rbj[1])
                    if v_ov >= 0.62 * min(ry1 - ry0, rbj[3]) and abs((ry1 - ry0) - rbj[3]) <= int(round(0.45 * max(ry1 - ry0, L_0))):
                        row_cands.append(j)
                        ry0 = min(ry0, rbj[1])
                        ry1 = max(ry1, rbj[1] + rbj[3])
                row_cands.sort(key=lambda k: rem_rbs[k][0])
                for start_k in row_cands:
                    if start_k in used_cons:
                        continue
                    rbf = rem_rbs[start_k]
                    same_row = [start_k]
                    used_cons.add(start_k)
                    curr_x0, curr_y0 = rbf[0], rbf[1]
                    curr_right, curr_y1 = rbf[0] + rbf[2], rbf[1] + rbf[3]
                    curr_h = rbf[3]
                    curr_env_area = float(rbf[2] * rbf[3])
                    curr_word_area = float(rbf[2] * rbf[3])
                    for k in row_cands:
                        if k in used_cons or rem_rbs[k][0] < curr_x0:
                            continue
                        rbj = rem_rbs[k]
                        h_gap = rbj[0] - curr_right
                        max_merge_gap = int(round(2.45 * max(curr_h, rbj[3], L_0)))
                        if -int(round(0.25 * curr_h)) <= h_gap <= max_merge_gap:
                            if h_gap >= max(3, int(round(0.22 * curr_h))) and (curr_right - curr_x0) <= int(round(12.0 * L_0)) and is_column_rail(rbj[0], rbj[1]):
                                continue
                            gy1, gy2 = max(curr_y0, rbj[1]), min(curr_y1, rbj[1] + rbj[3])
                            gx1, gx2 = curr_right, rbj[0]
                            if gx2 > gx1 and gy2 > gy1 and np.any(v_pure_gray[gy1:gy2, gx1:gx2].mean(axis=0) >= 0.55):
                                continue
                            nx0 = min(curr_x0, rbj[0])
                            ny0 = min(curr_y0, rbj[1])
                            nx1 = max(curr_right, rbj[0] + rbj[2])
                            ny1 = max(curr_y1, rbj[1] + rbj[3])
                            new_env_area = float((nx1 - nx0) * (ny1 - ny0))
                            seg_area = float(rbj[2] * rbj[3])
                            new_word_area = curr_word_area + seg_area
                            if new_env_area < curr_env_area + 0.50 * seg_area:
                                continue
                            if new_word_area / max(1.0, new_env_area) < 0.58:
                                continue
                            same_row.append(k)
                            used_cons.add(k)
                            curr_x0, curr_y0 = nx0, ny0
                            curr_right, curr_y1 = nx1, ny1
                            curr_h = ny1 - ny0
                            curr_env_area = new_env_area
                            curr_word_area = new_word_area
                    if len(same_row) == 1:
                        line_consolidated.append(rem_rbs[same_row[0]])
                    else:
                        lx = min(rem_rbs[k][0] for k in same_row)
                        ly = min(rem_rbs[k][1] for k in same_row)
                        lw = max(rem_rbs[k][0] + rem_rbs[k][2] for k in same_row) - lx
                        lh = max(rem_rbs[k][1] + rem_rbs[k][3] for k in same_row) - ly
                        flat_words: List[Dict[str, Any]] = []
                        for k in same_row:
                            sub_words = rem_rbs[k][7]
                            if sub_words:
                                for sw in sub_words:
                                    flat_words.append(dict(sw))
                            else:
                                flat_words.append(
                                    make_child_dict(
                                        rem_rbs[k][0],
                                        rem_rbs[k][1],
                                        rem_rbs[k][2],
                                        rem_rbs[k][3],
                                        rem_rbs[k][4],
                                        "word" if rem_rbs[k][5] != "inline_link" else "inline_link",
                                        "word" if rem_rbs[k][5] != "inline_link" else "control",
                                        "",
                                        [],
                                    )
                                )
                        for fw_item, h_str in zip(flat_words, generate_vimium_hints(len(flat_words))):
                            fw_item["hint"] = h_str
                        line_consolidated.append([lx, ly, lw, lh, 0.85, "text_block", "line", flat_words])

            rem_rbs = line_consolidated

            # 2. Structural & Area-Conservation Paragraph Chaining
            #    Enforces:
            #      - Straight left edge, uniform line height, and uniform vertical pitch
            #      - Monotonic per-line area growth (`new_para_area >= prev_para_area + 0.85 * cand_area`)
            #      - Cumulative line-to-paragraph fill density (`sum_line_area / new_para_area >= 0.56`)
            #      - Prose width continuity: non-final lines in a paragraph must span >= 58% of the paragraph's max line width
            text_line_indices = [
                i for i, rb in enumerate(rem_rbs)
                if rb[5] in ("text_block", "text_link")
                and rb[2] >= int(3.5 * L_0)
                and rb[3] <= int(2.4 * L_0)
                and not (rb[7] and rb[7][0].get("kind") == "icon")
            ]
            text_line_indices.sort(key=lambda idx: (rem_rbs[idx][1], rem_rbs[idx][0]))
            merged_tb = set()
            para_rbs = []

            for idx in text_line_indices:
                if idx in merged_tb:
                    continue
                chain_tb = [idx]
                merged_tb.add(idx)
                curr = rem_rbs[idx]
                curr_left = curr[0]
                curr_h = curr[3]
                prev_pitch = None
                px0, py0 = curr[0], curr[1]
                px1, py1 = curr[0] + curr[2], curr[1] + curr[3]
                para_env_area = float(curr[2] * curr[3])
                sum_line_area = float(curr[2] * curr[3])
                max_w_chain = curr[2]

                for idx2 in text_line_indices:
                    if idx2 in merged_tb:
                        continue
                    cand = rem_rbs[idx2]
                    left_diff = abs(cand[0] - curr_left)
                    h_diff = abs(cand[3] - curr_h)
                    v_gap = cand[1] - (curr[1] + curr[3])

                    max_v_gap = min(max(14, int(0.78 * L_0)), max(5, int(curr_h * 0.85)))
                    if left_diff <= max(9, int(0.48 * L_0)) and h_diff <= max(5, int(0.25 * L_0)):
                        if 1 <= v_gap <= max_v_gap:
                            pitch = v_gap
                            if prev_pitch is None or abs(pitch - prev_pitch) <= max(4, int(0.24 * L_0)):
                                if np.any(h_pure_gray[curr[1]+curr[3]:cand[1], min(curr[0], cand[0]):min(curr[0]+curr[2], cand[0]+cand[2])].mean(axis=1) >= 0.55):
                                    continue
                                # Do not chain narrow two-column table/infobox labels into a paragraph when they have a right-hand column neighbor on the same row
                                if curr[2] <= int(round(10.5 * L_0)) and any(
                                    other[0] >= curr[0] + curr[2]
                                    and min(curr[1] + curr[3], other[1] + other[3]) - max(curr[1], other[1]) >= 0.5 * min(curr[3], other[3])
                                    for other in rem_rbs
                                ):
                                    continue
                                new_max_w = max(max_w_chain, cand[2])
                                if curr[2] < 0.65 * new_max_w:
                                    continue
                                npx0 = min(px0, cand[0])
                                npy0 = min(py0, cand[1])
                                npx1 = max(px1, cand[0] + cand[2])
                                npy1 = max(py1, cand[1] + cand[3])
                                new_env_area = float((npx1 - npx0) * (npy1 - npy0))
                                cand_area = float(cand[2] * cand[3])
                                new_sum_line_area = sum_line_area + cand_area
                                if new_env_area < para_env_area + 0.85 * cand_area:
                                    continue
                                if new_sum_line_area / max(1.0, new_env_area) < 0.56:
                                    continue
                                chain_tb.append(idx2)
                                merged_tb.add(idx2)
                                prev_pitch = pitch
                                curr = cand
                                px0, py0, px1, py1 = npx0, npy0, npx1, npy1
                                para_env_area = new_env_area
                                sum_line_area = new_sum_line_area
                                max_w_chain = new_max_w
                                curr_left = int(round(np.median([rem_rbs[k][0] for k in chain_tb])))

                if len(chain_tb) >= 2:
                    px = min(rem_rbs[k][0] for k in chain_tb)
                    py = min(rem_rbs[k][1] for k in chain_tb)
                    pw = max(rem_rbs[k][0] + rem_rbs[k][2] for k in chain_tb) - px
                    ph = max(rem_rbs[k][1] + rem_rbs[k][3] for k in chain_tb) - py
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
                    para_rbs.append([px, py, pw, ph, 0.82, "text_block", "paragraph", line_children])
                else:
                    para_rbs.append(rem_rbs[idx])

            tb_set = set(text_line_indices)
            rem_rbs = [rb for i, rb in enumerate(rem_rbs) if i not in tb_set] + para_rbs

        # Sibling Rail Normalization
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

        # Final containment cleanup
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
                    if inter >= 0.65 * r_area and r_area <= kw * kh:
                        if k_rb[6] == "paragraph" and k_rb[7]:
                            child_inter = sum(
                                max(0, min(int(rx * scale_back + rw * scale_back), ch["x"] + ch["w"]) - max(int(rx * scale_back), ch["x"]))
                                * max(0, min(int(ry * scale_back + rh * scale_back), ch["y"] + ch["h"]) - max(int(ry * scale_back), ch["y"]))
                                for ch in k_rb[7]
                            )
                            if child_inter < 0.45 * max(1.0, r_area * scale_back * scale_back):
                                continue
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
                max(1, int(b[2] * scale_back)),
                max(1, int(b[3] * scale_back)),
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
    parser = argparse.ArgumentParser(description="Visel v9 Contextual Glyph-Chain & Structure-Aware UI Detector")
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
