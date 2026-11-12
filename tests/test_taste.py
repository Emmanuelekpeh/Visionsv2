"""Taste capture: an A/B choice becomes a labelled row, not a lost conversation.

The studio's value in the end is editorial taste. These tests pin the storage
contract and the summary's honest limits (it reports what separated the pick from
the reject; it does not claim to explain taste).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import taste

SCRATCH = Path(__file__).resolve().parent.parent / "output" / "_test_taste"


def test_record_and_summary(monkeypatch):
    labels = SCRATCH / "labels.jsonl"
    if labels.exists():
        labels.unlink()
    monkeypatch.setattr(taste, "LABELS", labels)
    monkeypatch.setattr(taste, "film_metrics", lambda p: {
        "cuts_per_min": 22.0 if str(p).endswith("a.mp4") else 12.0,
        "asl_s": 2.0 if str(p).endswith("a.mp4") else 3.5,
        "shot_cv": 0.2 if str(p).endswith("a.mp4") else 0.7,
        "duration_s": 8.0,
    })
    taste.record("a.mp4", "b.mp4", "a", note="coverage read as one scene",
                 tags=["story", "coverage"])
    taste.record("a.mp4", "b.mp4", "b", note="", tags=["story", "montage"])
    taste.record("a.mp4", "b.mp4", "tie", tags=["story"])
    s = taste.summary()
    assert s["labels"] == 3
    assert s["picks"] == {"a": 1, "b": 1, "tie": 1}
    assert s["by_tag"]["story"]["tie"] == 1
    # winner minus reject is computed only on non-tie rows; the fake metrics are
    # symmetric across the two picks, so the mean difference is zero. The point
    # of the field is that it is computed from recorded labels, honestly.
    assert s["winner_minus_reject"]["cuts_per_min"] == 0.0
    text = taste.format_summary(s)
    assert "3 label(s)" in text and "coverage read as one scene" in text
    labels.unlink()


def test_pick_must_be_valid(monkeypatch):
    monkeypatch.setattr(taste, "LABELS", SCRATCH / "x.jsonl")
    with pytest.raises(ValueError):
        taste.record("a.mp4", "b.mp4", "c")


def test_missing_metrics_do_not_lose_the_label(monkeypatch):
    labels = SCRATCH / "missing.jsonl"
    if labels.exists():
        labels.unlink()
    monkeypatch.setattr(taste, "LABELS", labels)
    real = taste.film_metrics

    def fake(p):
        return {"error": "missing"} if str(p).endswith("gone.mp4") else real(p)

    monkeypatch.setattr(taste, "film_metrics", fake)
    row = taste.record("gone.mp4", "also.mp4", "tie", tags=["test"])
    assert row["metrics"]["a"].get("error") == "missing"
    assert taste.load()[-1]["pick"] == "tie"
    labels.unlink()
