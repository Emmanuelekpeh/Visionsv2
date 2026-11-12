"""The shared continuity primitives: one definition used by previs, sequence_ab
and the cut detector. This pins the parts that used to drift between copies."""

from __future__ import annotations

import numpy as np
from PIL import Image

import continuity


def _img(colour):
    return Image.new("RGB", (48, 48), colour)


def test_hist_distance_separates_places_and_ignores_near_identical():
    a = _img((40, 50, 60))
    b = _img((42, 52, 62))
    c = _img((200, 120, 40))
    assert continuity.hist_distance(a, b) < 0.02
    assert continuity.hist_distance(a, c) > 0.8


def test_luma_jump_is_channel_mean_difference():
    a = _img((10, 10, 10))
    b = _img((110, 110, 110))
    assert continuity.luma_jump(a, b) == 100.0


def test_accepts_bgr_numpy_frames():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200            # red in RGB
    bgr = rgb[:, :, ::-1].copy()  # the same image expressed in BGR
    assert np.allclose(continuity.hist_signature(rgb, rgb=True),
                       continuity.hist_signature(bgr, rgb=False))


def test_pair_similarity_without_scorer_is_none():
    assert continuity.pair_similarity(_img((1, 2, 3)), _img((1, 2, 3)), None) is None


def test_sequence_ab_does_not_shadow_the_continuity_module():
    """`def continuity(...)` hid `import continuity`, so every call to
    continuity.luma_jump inside it raised AttributeError."""
    import types
    import sequence_ab

    assert isinstance(sequence_ab.continuity, types.ModuleType)
    assert callable(sequence_ab.cut_continuity)
