"""Tests for review.py -- the take judgement that has to work on any production.

These are pure: no store, no video, no VLM. The point they protect is that the
cheap measurement never turns itself into a pass/fail verdict (that is the vision
pass's job), because the whole failure being fixed is a take that looked fine to
every automatic check.
"""
from __future__ import annotations

import numpy as np

import review as rv


def _frame(w=160, h=120, square=None, bg=200):
    """A bright background with an optional DARK subject square."""
    f = np.full((h, w, 3), bg, np.uint8)
    if square is not None:
        x, y, s = square
        f[y:y + s, x:x + s] = 30
    return f


def _moving(n=8, w=160, h=120, bg=200, s=26, skip=()):
    frames = []
    for i in range(n):
        if i in skip:
            frames.append(_frame(w, h, None, bg))
        else:
            frames.append(_frame(w, h, (4 + i * 12, 40, s), bg))
    return frames


def test_a_moving_dark_subject_is_found_and_stable():
    m = rv.dark_foreground(_moving())
    assert m["status"] == "ok"
    assert m["typical_area"] > 0.01
    areas = [r["area_frac"] for r in m["rows"]]
    assert min(areas) > 0.5 * max(areas)


def test_a_missing_subject_collapses_that_frames_area():
    m = rv.dark_foreground(_moving(skip=(3, 4)))
    areas = [r["area_frac"] for r in m["rows"]]
    assert areas[3] < 0.25 * areas[2]
    assert areas[4] < 0.25 * areas[2]


def test_a_static_take_is_reported_static_not_failed():
    frames = [_frame(square=(40, 40, 26)) for _ in range(6)]
    m = rv.dark_foreground(frames)
    assert m["status"] == "static"
    sig = rv.signals(m)
    assert any("barely moves" in f or "very little motion" in f for f in sig["flags"])


def test_a_take_with_no_dark_subject_says_so_without_failing():
    # a slowly shifting bright gradient: lots of motion, nothing darker than its
    # own median to find.
    frames = []
    for i in range(6):
        g = np.tile(np.linspace(150 + i * 4, 210, 160), (120, 1)).astype(np.uint8)
        frames.append(np.stack([g] * 3, axis=-1))
    m = rv.dark_foreground(frames)
    assert m["status"] in ("no_subject", "ok")
    sig = rv.signals(m)
    assert "suspicious" in sig           # diagnostic only


def test_signals_flag_a_swinging_foreground():
    frames = _moving(n=8)
    # one frame where the subject is much bigger
    frames[5] = _frame(square=(10, 10, 70))
    m = rv.dark_foreground(frames)
    sig = rv.signals(m)
    assert any("swings" in f for f in sig["flags"])


def test_a_verdict_is_never_invented_from_the_numbers():
    row = {"expect_figure": True, "signals": {"suspicious": True, "flags": ["x"]}}
    assert rv._verdict_of(row) == "unreviewed"
    row["vision"] = {"trusted": True, "verdict": "fail"}
    assert rv._verdict_of(row) == "fail"
    row["verdict"] = "pass"          # a person's recorded verdict wins
    assert rv._verdict_of(row) == "pass"


def test_an_untrusted_vision_answer_is_not_a_verdict():
    # measured: qwen2.5vl saw a figure in empty water, so its answer is evidence
    row = {"expect_figure": True, "signals": {"flags": []},
           "vision": {"trusted": False, "verdict": "fail"}}
    assert rv._verdict_of(row) == "unreviewed"


def test_a_trusted_vision_answer_is_used():
    row = {"expect_figure": True,
           "vision": {"trusted": True, "verdict": "fail"}}
    assert rv._verdict_of(row) == "fail"


def test_non_figure_shots_are_skipped_not_unreviewed():
    assert rv._verdict_of({"expect_figure": False}) == "skip"


def test_figure_expected_from_subject_then_prompt():
    assert rv.figure_expected({"subject": "scuba diver", "prompt": "deep water"})
    assert rv.figure_expected({"subject": "", "prompt": "a woman under a bridge"})
    assert not rv.figure_expected({"subject": "", "prompt": "empty fog over a river"})
    assert not rv.figure_expected({})


def test_sample_indices_spread_and_cover_the_ends():
    assert rv._sample_indices(16, 3) == [0, 8, 15]
    assert rv._sample_indices(2, 3) == [0, 1]
    assert len(rv._sample_indices(16, 5)) == 5


def test_parse_json_survives_model_noise():
    assert rv._parse_json('{"figure": true, "hands": 3}')["hands"] == 3
    assert rv._parse_json('here you go:\n```json\n{"figure": false}\n```')["figure"] is False
    assert rv._parse_json("no json here") is None
    assert rv._parse_json("") is None


def test_crop_for_frame_is_tight_around_the_subject():
    frame = _frame(w=400, h=300, square=(160, 120, 40), bg=200)
    crop = rv.crop_for_frame(frame, [160, 120, 40, 40], margin=0.5)
    assert crop["image"].shape[0] < 300 and crop["image"].shape[1] < 400
    x0, y0, x1, y1 = crop["box"]
    assert x0 <= 160 and y0 <= 120 and x1 >= 200 and y1 >= 160


def test_crop_for_frame_without_a_box_is_the_whole_frame():
    frame = np.full((200, 200, 3), 200, np.uint8)
    crop = rv.crop_for_frame(frame, None)
    assert crop["box"] == [0, 0, 200, 200]


def test_crop_for_frame_downsizes_for_the_model():
    frame = np.full((1600, 1600, 3), 200, np.uint8)
    crop = rv.crop_for_frame(frame, [400, 400, 800, 800], max_side=640)
    assert max(crop["image"].shape[:2]) <= 640


def test_a_take_with_no_frames_is_reported_not_crashed():
    m = rv.dark_foreground([])
    assert m["frames"] == 0
    assert rv.signals(m)["status"] == "not_measurable"
    assert rv._verdict_of({"expect_figure": True, "signals": m}) == "unreviewed"


def test_signals_on_one_or_two_frames_return_a_dict_not_a_tuple():
    """A trailing comma after the returned dict made it a 1-tuple, so signals() raised
    AttributeError for any take with fewer than 3 frames."""
    import numpy as np
    import review

    frames = [np.full((32, 32, 3), 40 + 20 * i, np.uint8) for i in range(2)]
    out = review.dark_foreground(frames)
    assert isinstance(out, dict)
    assert out["status"] == "too_few_frames" and out["frames"] == 2
