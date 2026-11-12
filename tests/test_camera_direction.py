"""The camera-direction policy: four modes, one decision per shot.

The store path used to freeze every shot at `camera="none"`. These tests pin the
replacement policy without loading CLIP: `director.choose` is stubbed, so what is
tested is the DECISION LOGIC (off / suggestions / director / hybrid + fallback),
not the fit model.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import camera_direction as cd
from edit_planner import Shot


def crew(cameras) -> list:
    beats = ["establish", "develop", "turn", "resolve"]
    return [Shot(f"a concrete image {i + 1}", "medium", c, 2.0,
                 beat=beats[min(i, 3)])
            for i, c in enumerate(cameras)]


class FakeReport:
    choices = []


def _stub_director(monkeypatch, cameras):
    """Make `director.choose` return `cameras` and record the call."""
    import director

    calls = []

    def fake_choose(shots, **kwargs):
        calls.append(kwargs)
        return [replace(s, camera=c) for s, c in zip(shots, cameras)], FakeReport()

    monkeypatch.setattr(director, "choose", fake_choose)
    monkeypatch.setattr(director, "FitScorer", lambda *a, **k: SimpleNamespace(ok=True))
    return calls


def test_mode_off_is_all_still():
    cams, info = cd.decide(crew(["zoom-in", "pan-left", "none", "tilt-up"]),
                           mode="off")
    assert cams == ["none"] * 4
    assert info["mode"] == "off"


def test_suggestions_mode_uses_the_writer_and_rejects_junk():
    cams, _ = cd.decide(crew(["zoom-in", "not-a-move", "", "tilt-up"]),
                        mode="suggestions")
    assert cams == ["zoom-in", "none", "none", "tilt-up"]


def test_director_mode_uses_the_director(monkeypatch):
    calls = _stub_director(monkeypatch, ["pan-left", "none", "zoom-in", "none"])
    cams, info = cd.decide(crew(["none"] * 4), mode="director")
    assert cams == ["pan-left", "none", "zoom-in", "none"]
    assert info["mode"] == "director"
    assert calls, "the director must actually be called"
    # only camera LoRAs (plus none) are offered by default
    assert "walking-frontview" not in calls[0]["allow"]


def test_hybrid_keeps_the_writers_move_where_the_director_declines(monkeypatch):
    _stub_director(monkeypatch, ["pan-left", "none", "none", "none"])
    cams, info = cd.decide(crew(["none", "tilt-up", "none", "zoom-out"]),
                           mode="hybrid")
    # a 4-shot film has a mover budget of 2: the director's pan-left plus ONE
    # writer suggestion (earliest wins), not both
    assert cams == ["pan-left", "tilt-up", "none", "none"]
    assert info["overlay_kept"] == 1


def test_hybrid_never_exceeds_the_mover_budget():
    """Measured without the cap: 3 moves in a 6-shot film against a brief of 2."""
    cap_cams = cd.overlay_suggestions(
        ["none", "none", "none", "none", "none", "none"],
        ["pan-left", "tilt-up", "zoom-in", "none", "none", "none"],
        max_movers=2)
    assert sum(1 for c in cap_cams if c != "none") == 2
    # director moves are kept before writer suggestions
    kept = cd.overlay_suggestions(
        ["none", "zoom-in", "none", "none"], ["tilt-up", "pan-left", "none", "none"],
        max_movers=2)
    assert kept[1] == "zoom-in" and kept[0] == "tilt-up"
    assert sum(1 for c in kept if c != "none") == 2


def test_hybrid_budget_is_not_spent_on_empty_suggestions():
    """Regression: `none` is a valid camera name, so counting it as a kept
    suggestion consumed the budget and dropped a real move later in the list."""
    cams = cd.overlay_suggestions(
        ["none", "none", "none", "none"],
        ["none", "none", "none", "pan-left"], max_movers=2)
    assert cams == ["none", "none", "none", "pan-left"]


def test_hybrid_does_not_override_a_director_move(monkeypatch):
    _stub_director(monkeypatch, ["pan-left", "none", "none", "none"])
    cams, _ = cd.decide(crew(["tilt-up", "none", "none", "none"]), mode="hybrid")
    assert cams[0] == "pan-left"       # the director owns taste/continuity


def test_a_failed_director_falls_back_to_suggestions(monkeypatch):
    import director

    def boom(*a, **k):
        raise RuntimeError("CLIP unavailable")

    monkeypatch.setattr(director, "choose", boom)
    monkeypatch.setattr(director, "FitScorer", lambda *a, **k: SimpleNamespace(ok=True))
    cams, info = cd.decide(crew(["zoom-in", "none"]), mode="hybrid")
    assert cams == ["zoom-in", "none"]
    assert "CLIP unavailable" in info["fallback"]


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown camera mode"):
        cd.decide(crew(["none"]), mode="magic")


def test_summary_counts_moves_reversals_and_runs():
    cams = ["pan-left", "pan-right", "none", "zoom-in", "zoom-in"]
    s = cd.summary(crew(cams), cams)
    assert s["movers"] == 4
    assert s["reversals"] == [2]          # pan-left -> pan-right across a cut
    assert s["longest_mover_run"] == 2
    assert s["mover_share"] == 0.8
