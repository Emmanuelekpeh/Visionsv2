#!/usr/bin/env python3
"""
continuity.py -- the cross-cut measurements, in one place.

Three tools grew their own copy of the same two numbers: `pacing_study` (cut
detection), `sequence_ab` (finished cuts) and `previs` (stills). They drifted:
previs needed an attention-slicing fix the renderer did not, and the histogram
implementations took different colour orders. This module is the single source:

  * `hist_signature(frame)` / `hist_distance(a, b)` -- HSV histogram distance
  * `luma_jump(a, b)` -- mean absolute channel-mean difference across a cut
  * `pair_similarity(a, b, scorer)` -- CLIP cosine of two frames

Frames may be PIL images or uint8 numpy arrays. `rgb=True` (default) assumes the
array is RGB; pass `rgb=False` for OpenCV's BGR frames.
"""

from __future__ import annotations

from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

Frame = Union[Image.Image, np.ndarray]

# The one set of cut thresholds. previs (stills), its gate and output_audit (the
# finished film) each carried their own copies (luma 8 / 10, CLIP 0.45 / 0.55).
# A boundary INSIDE one scene is a defect when it exceeds these; at a scene cut the
# same numbers are only informational (the edit, with per-scene palettes, is style).
HOLD_LUMA_MAX = 10.0        # mean channel-mean jump, 0-255  -> "flash across the cut"
HOLD_HIST_MAX = 0.80        # HSV Bhattacharyya distance      -> "reads as a jump cut"
HOLD_CLIP_MIN = 0.45        # CLIP cosine below this          -> "images don't belong together"


def _rgb(frame: Frame, rgb: bool = True) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr.astype(np.uint8)
    if not rgb:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


def hist_signature(frame: Frame, rgb: bool = True) -> np.ndarray:
    """A normalised HSV histogram. V is included: a hue/saturation-only
    histogram is identical for a monochrome film (it gave distance 0.0 on
    `project-9` and missed 11 cuts)."""
    hsv = cv2.cvtColor(_rgb(frame, rgb), cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [32, 16, 16],
                     [0, 180, 0, 256, 0, 256])
    cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
    return h


def hist_distance(a: Frame, b: Frame, rgb: bool = True) -> float:
    """Bhattacharyya distance, 0-1. Higher = more different."""
    return float(cv2.compareHist(hist_signature(a, rgb), hist_signature(b, rgb),
                                 cv2.HISTCMP_BHATTACHARYYA))


def luma_jump(a: Frame, b: Frame, rgb: bool = True) -> float:
    """Mean absolute difference of the channel means: the exposure/lighting
    jump across a cut, 0-255."""
    la = _rgb(a, rgb).astype(np.float32)
    lb = _rgb(b, rgb).astype(np.float32)
    return float(np.abs(la.mean(axis=(0, 1)) - lb.mean(axis=(0, 1))).mean())


def pair_similarity(a: Frame, b: Frame, scorer) -> Optional[float]:
    """CLIP cosine of two frames; None if CLIP is unavailable."""
    if scorer is None:
        return None
    try:
        e = scorer.embed_images([Image.fromarray(_rgb(a)),
                                 Image.fromarray(_rgb(b))])
        return round(float((e[0] @ e[1]).item()), 3)
    except Exception:                              # noqa: BLE001
        return None
