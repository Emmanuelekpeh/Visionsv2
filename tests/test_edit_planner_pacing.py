"""The planner's holds must be deliverable by the renderer.

The bug this file pins: `_rhythm_curve_beats` assigned 4.5-6.5 s holds to shots
whose source clips contain 1.94 s of motion (16 raw frames -> RAFT x2 -> 31
delivered frames at 16 fps). `make_film.render` fills a longer hold by repeating
frames, so those holds play as slow-motion judder. The planner now clamps to
`handheld.max_hold_seconds` and reports a target it cannot reach instead of
faking it.
"""

from __future__ import annotations

import pytest

from edit_planner import Shot, _rhythm_curve_beats, plan
from handheld import (fit_hold_seconds, max_hold_seconds, natural_seconds,
                      stretch_ratio)

FPS = 16
BEATS = ["establish", "develop", "turn", "resolve"]


def crew(n: int = 6, camera: str = "none") -> list:
    return [Shot(f"a concrete image {i + 1}", "medium" if i % 2 else "wide",
                 camera, 2.0, beat=BEATS[min(i, 3)]) for i in range(n)]


@pytest.mark.parametrize("frames", [12, 16, 24, 32])
def test_no_planned_hold_exceeds_the_generated_motion(frames):
    planned, _ = plan(crew(), frames=frames, fps=FPS)
    for s in planned:
        cap = max_hold_seconds(frames, FPS, moving=s.moving)
        assert s.seconds <= cap + 1e-6, (
            f"{s.seconds}s hold is more than the {cap:.2f}s budget at "
            f"{frames} raw frames")


def test_moving_shots_get_the_tighter_ceiling():
    # a single camera move, so R6/R3 cannot rewrite it to 'none'
    one = [Shot("a wide street at night", "wide", "zoom-in", 2.0, beat="establish")]
    planned, _ = plan(one, frames=16, fps=FPS)
    assert planned[0].camera == "zoom-in"
    assert planned[0].seconds <= max_hold_seconds(16, FPS, moving=True) + 1e-6
    assert max_hold_seconds(16, FPS, moving=True) < max_hold_seconds(16, FPS)


def test_target_duration_that_cannot_be_reached_is_reported_not_stretched():
    planned, report = plan(crew(6), total=30.0, frames=16, fps=FPS)
    achieved = sum(s.seconds for s in planned)
    assert achieved < 30.0
    assert report.notes, "an unreachable target must be a note, not silence"
    assert "NOT reachable" in report.notes[0]
    for s in planned:
        assert s.seconds <= max_hold_seconds(16, FPS, moving=s.moving) + 1e-6


def test_reachable_target_has_no_shortfall_note():
    # six 16-frame shots supply ~11.6s of motion; ask for what fits
    planned, report = plan(crew(6), total=11.0, frames=16, fps=FPS)
    assert sum(s.seconds for s in planned) >= 11.0 * 0.9
    assert not report.notes


def test_keeping_director_timing_still_warns_about_judder():
    too_long = [Shot("a concrete image", "medium", "none", 6.0, beat="establish")]
    planned, report = plan(too_long, keep_director_timing=True, frames=16, fps=FPS)
    assert planned[0].seconds == 6.0           # not rewritten
    assert report.notes and "judder" in report.notes[0]


def test_curve_scales_with_the_frame_setting():
    low = _rhythm_curve_beats(crew(6), frames=12, fps=FPS)
    high = _rhythm_curve_beats(crew(6), frames=32, fps=FPS)
    assert sum(high) > 2.0 * sum(low)


def test_establish_and_resolve_are_longer_than_the_turn():
    # mean over several shapes: the turn should be the shortest beat
    shots = [Shot("image", "medium", "none", 2.0, beat=b) for b in BEATS]
    values = _rhythm_curve_beats(shots, frames=32, fps=FPS)
    by_beat = dict(zip(BEATS, values))
    assert by_beat["turn"] < by_beat["establish"]
    assert by_beat["resolve"] > by_beat["turn"]


def test_stretch_ratio_is_the_renderers_own_math():
    # 2.0s at 16fps = 32 delivered frames against 31 generated
    assert stretch_ratio(2.0, 16, 16) == pytest.approx(32 / 31, abs=0.01)
    assert natural_seconds(16, 16) == pytest.approx(31 / 16)


def test_fit_hold_seconds_clamps_only_what_would_judder():
    assert fit_hold_seconds(2.0, "none", 16) == pytest.approx(2.0)
    assert fit_hold_seconds(3.6, "none", 16) == pytest.approx(
        max_hold_seconds(16, 16))
    # a camera move is capped tighter than a static shot
    assert fit_hold_seconds(3.6, "zoom-in", 16) == pytest.approx(
        max_hold_seconds(16, 16, moving=True))
    assert (fit_hold_seconds(3.6, "none", 16)
            > fit_hold_seconds(3.6, "zoom-in", 16))


def test_unreachable_target_keeps_the_shaped_rhythm(monkeypatch):
    """Pinning every hold to the ceiling would satisfy no beat shape at all."""
    planned, report = plan(crew(6), total=30.0, frames=16, fps=FPS)
    values = [s.seconds for s in planned]
    assert len(set(values)) > 1, "holds went uniform at the ceiling"
    assert report.notes


def test_adoption_paces_the_writers_holds_and_framing(monkeypatch):
    """The store path never runs the planner, so adoption is the last guarantee.

    Adoption now runs `edit_planner.plan`: R4 re-times the writer's guesses into a
    beat-shaped rhythm of real generated motion, and R1 repairs adjacent framing
    scales. The writer's text survives; only timing/framing are paced.
    """
    import film_store as fs
    from types import SimpleNamespace
    project = {"slug": "pacing-test", "settings": {"frames": 16, "fps": 16},
               "timeline": [], "scenes": [], "problems": []}
    monkeypatch.setattr(fs, "save_shot", lambda *a, **k: None)
    proposal = SimpleNamespace(
        shots=[Shot("a concrete image", "wide", "none", 3.6, beat="establish"),
               Shot("a second image", "wide", "none", 2.0, beat="develop"),
               Shot("a third image", "close", "none", 3.0, beat="resolve")],
        raw={}, logline="a test", problems=[])
    made = fs.shots_from_proposal(project, proposal)

    # never untrimmed: every hold is inside the renderer's budget
    for m in made:
        assert m["seconds"] <= max_hold_seconds(16, 16) + 1e-6
    # re-timed away from the writer's guesses (3.6, 2.0, 3.0)
    assert [m["seconds"] for m in made] != [3.6, 2.0, 3.0]
    # R1: two adjacent 'wide' cannot survive adoption
    assert made[0]["scale"] != made[1]["scale"]
    # the pacing decision is recorded for the UI / a later reader
    pace = project["pacing"]
    assert pace["writer_holds"] == [3.6, 2.0, 3.0]
    assert len(pace["planned_holds"]) == 3
    assert len(pace["changes"]) > 0
    assert pace["natural_s"] == round(natural_seconds(16, 16), 2)
    # pacing is not a writer problem -- it is the machine's job
    assert not any("trimmed to the renderer" in p for p in project["problems"])


def test_adoption_stores_the_director_decision(monkeypatch):
    """With a direction decision, the camera is stored on the shot -- that is what
    makes R2/R3/R6 reach a render at all."""
    import film_store as fs
    from types import SimpleNamespace
    project = {"slug": "camera-test", "settings": {"frames": 16, "fps": 16},
               "timeline": [], "scenes": [], "problems": []}
    monkeypatch.setattr(fs, "save_shot", lambda *a, **k: None)
    proposal = SimpleNamespace(
        shots=[Shot("a concrete image one", "wide", "pan-left", 2.0, beat="establish",
                    purpose="establish", link="open"),
               Shot("a concrete image two", "medium", "none", 2.0, beat="develop",
                    purpose="master", link="match-action"),
               Shot("a concrete image three", "close", "tilt-up", 2.0,
                    beat="resolve", purpose="resolve", link="eyeline")],
        raw={}, logline="a test", problems=[])
    made = fs.shots_from_proposal(project, proposal,
                                  cameras=["pan-left", "none", "tilt-up"])
    assert [m["camera"] for m in made] == ["pan-left", "none", "tilt-up"]
    # the writer's own suggestion is kept as the caption/audit value
    assert [m["camera_suggestion"] for m in made] == ["pan-left", None, "tilt-up"]
    # purpose/link survive adoption: the cut is reviewable as a sequence
    assert [m["purpose"] for m in made] == ["establish", "master", "resolve"]
    assert [m["link"] for m in made] == ["open", "match-action", "eyeline"]
    assert project["pacing"]["movers"] == 2
    assert project["pacing"]["cameras"] == ["pan-left", "none", "tilt-up"]


def test_adoption_lets_the_planner_repair_a_reversed_move(monkeypatch):
    """A direction decision is not above R6: pan-left then pan-right is repaired."""
    import film_store as fs
    from types import SimpleNamespace
    project = {"slug": "camera-test", "settings": {"frames": 16, "fps": 16},
               "timeline": [], "scenes": [], "problems": []}
    monkeypatch.setattr(fs, "save_shot", lambda *a, **k: None)
    proposal = SimpleNamespace(
        shots=[Shot("a concrete image one", "wide", "none", 2.0, beat="establish"),
               Shot("a concrete image two", "medium", "none", 2.0, beat="develop")],
        raw={}, logline="a test", problems=[])
    made = fs.shots_from_proposal(project, proposal,
                                  cameras=["pan-left", "pan-right"])
    assert made[0]["camera"] == "pan-left"
    assert made[1]["camera"] == "none", "R6 must repair the reversal"


def test_adoption_rejects_a_misaligned_camera_decision(monkeypatch):
    import film_store as fs
    from types import SimpleNamespace
    project = {"slug": "camera-test", "settings": {"frames": 16, "fps": 16},
               "timeline": [], "scenes": [], "problems": []}
    monkeypatch.setattr(fs, "save_shot", lambda *a, **k: None)
    proposal = SimpleNamespace(
        shots=[Shot("a concrete image one", "wide", "none", 2.0, beat="establish")],
        raw={}, logline="a test", problems=[])
    with pytest.raises(ValueError, match="must line up"):
        fs.shots_from_proposal(project, proposal, cameras=["none", "none"])
