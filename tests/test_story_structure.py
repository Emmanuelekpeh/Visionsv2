"""The sequence grammar: R7 scene structure, R8 coverage, R9 rhythm, R10 handoff.

The failure these exist for, measured on the productions: `seasoning` has 13 shots
over 3 scene labels, but 0/10 adjacent within-scene pairs share a subject and its
within-scene prompt overlap (0.05) is no higher than across scenes (0.08). The
labels were cosmetic. A film is 2-3 sequences, each covering one place/action
with several shots; "two seconds, hard stop, new scenery, repeat" is a slideshow.

The fixtures are the smallest shot lists that exhibit each failure, so a fix to a
check cannot silently pass a new one.
"""

from __future__ import annotations

import pytest

from edit_planner import Shot, plan
from story import structure_problems, validate


def shot(prompt, scale="medium", camera="none", seconds=2.0, subject="",
         scene="s1", beat="develop", purpose="master", link="match-action") -> dict:
    return {"prompt": prompt, "scale": scale, "camera": camera,
            "seconds": seconds, "subject": subject, "scene": scene,
            "beat": beat, "purpose": purpose, "link": link, "energy": 0.2}


# A clean 5-shot, 2-scene coverage sequence: master/insert on the action, then a
# second place opened long and closed on a reaction. This is the reference shape.
GOOD = {"logline": "a woman crosses the bridge and is followed",
        "shots": [
            shot("a rain-soaked footbridge at night, a woman in a dark coat "
                 "stepping onto the planks, streetlights, fog", "wide", seconds=2.5,
                 subject="woman", scene="s1", beat="establish",
                 purpose="establish", link="open"),
            shot("the same woman walking away along the bridge, seen from behind, "
                 "rain on the planks, fog", "medium", seconds=1.5,
                 subject="woman", scene="s1", beat="develop",
                 purpose="master", link="match-action"),
            shot("a woman's hand tightening on the wet iron railing, rain beading",
                 "close", seconds=1.5, subject="woman", scene="s1", beat="develop",
                 purpose="insert", link="match-action"),
            shot("a second figure under the far streetlight at the end of the same "
                 "bridge, fog, facing her", "wide", seconds=2.0,
                 subject="figure", scene="s2", beat="turn",
                 purpose="reaction", link="scene-cut"),
            shot("the two figures facing each other on the rain-soaked bridge, "
                 "streetlights between them", "close", seconds=1.5,
                 subject="figure", scene="s2", beat="resolve",
                 purpose="resolve", link="eyeline"),
        ]}


def test_good_coverage_sequence_has_no_structure_or_rhythm_problems():
    _shots, problems = validate(GOOD)
    structure = [p for p in problems if p.startswith(("structure:", "rhythm:"))]
    assert structure == [], structure
    assert not [p for p in problems if p.startswith("story:")]

def test_postcard_montage_is_caught():
    """Four shots, four places, one action each. The 'hard stop, new scenery'."""
    raw = {"logline": "x", "shots": [
        shot("a rain-soaked footbridge at night, empty", "wide", seconds=2.0,
             scene="s1", beat="establish", purpose="establish", link="open"),
        shot("wet iron railing, raindrops beading", "close", seconds=2.0,
             scene="s2", beat="develop", link="scene-cut"),
        shot("a woman walking under a streetlight in an alley", "medium",
             seconds=2.0, subject="woman", scene="s3", beat="turn",
             link="scene-cut"),
        shot("a distant figure in fog on another bridge", "wide", seconds=2.0,
             subject="figure", scene="s4", beat="resolve", link="scene-cut"),
    ]}
    _shots, problems = validate(raw)
    structure = [p for p in problems if p.startswith("structure:")]
    joined = " ".join(structure)
    assert "R7" in joined and "scene" in joined
    assert any("scene cut" in p for p in structure)


def test_interleaved_scenes_are_caught():
    raw = {"logline": "x", "shots": [
        shot("a wet bridge at night", "wide", seconds=2.5, scene="s1",
             beat="establish", purpose="establish", link="open"),
        shot("a woman's face in the rain", "close", seconds=1.8, subject="woman",
             scene="s2", beat="develop", link="scene-cut"),
        shot("the bridge railing, rain", "medium", seconds=2.0, scene="s1",
             beat="turn", link="scene-cut"),
        shot("the woman looking back", "medium", seconds=2.2, subject="woman",
             scene="s2", beat="resolve", link="scene-cut"),
    ]}
    _shots, problems = validate(raw)
    assert any("is revisited" in p for p in problems), problems


def test_scene_with_one_scale_has_no_coverage():
    raw = {"logline": "x", "shots": [
        shot("a wet bridge at night", "wide", seconds=2.5, scene="s1",
             beat="establish", purpose="establish", link="open"),
        shot("a woman stepping onto the bridge", "wide", seconds=2.0,
             subject="woman", scene="s1", beat="develop", link="match-action"),
        shot("the far streetlight in fog", "close", seconds=2.2, scene="s2",
             beat="turn", link="scene-cut"),
        shot("a figure under the far light", "medium", seconds=2.0,
             subject="figure", scene="s2", beat="resolve", link="match-action"),
    ]}
    _shots, problems = validate(raw)
    assert any("R8" in p and "framing scale" in p for p in problems), problems


def test_link_must_match_the_scene_change():
    raw = {"logline": "x", "shots": [
        shot("a wet bridge at night", "wide", seconds=2.5, scene="s1",
             beat="establish", purpose="establish", link="open"),
        shot("a woman on the bridge", "medium", seconds=2.0, subject="woman",
             scene="s1", beat="develop", link="match-action"),
        shot("a new place, an alley at night", "wide", seconds=2.2, scene="s2",
             beat="turn", link="match-action"),     # wrong: this is a scene cut
        shot("a figure in the alley", "close", seconds=2.0, subject="figure",
             scene="s2", beat="resolve", link="match-action"),
    ]}
    _shots, problems = validate(raw)
    assert any("across a scene change" in p or "scene-cut" in p
               for p in problems), problems


def test_camera_move_before_a_scene_cut_is_flagged():
    raw = {"logline": "x", "shots": [
        shot("a wet bridge at night", "wide", camera="pan-left", seconds=2.0,
             scene="s1", beat="establish", purpose="establish", link="open"),
        shot("a woman on the bridge", "close", seconds=2.0, subject="woman",
             scene="s2", beat="turn", link="scene-cut"),
    ]}
    _shots, problems = validate(raw)
    assert any("R10" in p for p in problems), problems


def test_planner_rewrites_a_move_that_hands_off_to_new_scenery():
    shots = [Shot("a wet bridge at night", "wide", "pan-left", 2.0, scene="s1",
                  beat="establish"),
             Shot("a woman on the bridge", "close", "none", 2.0, subject="woman",
                  scene="s2", beat="turn")]
    planned, report = plan(shots, frames=16, fps=16)
    assert planned[0].camera == "none"
    assert any("R10" in c for c in report.changes)


def test_planner_keeps_a_move_within_its_scene():
    shots = [Shot("a wet bridge at night", "wide", "pan-left", 2.0, scene="s1",
                  beat="establish"),
             Shot("a woman on the bridge", "close", "none", 2.0, subject="woman",
                  scene="s1", beat="develop")]
    planned, report = plan(shots, frames=16, fps=16)
    assert planned[0].camera == "pan-left"
    assert not any("R10" in c for c in report.changes)


def test_uniform_holds_and_short_scene_opening_are_rhythm_problems():
    raw = {"logline": "x", "shots": [
        shot("a wet bridge at night", "wide", seconds=2.0, scene="s1",
             beat="establish", purpose="establish", link="open"),
        shot("a woman on the bridge", "medium", seconds=2.0, subject="woman",
             scene="s1", beat="develop", link="match-action"),
        shot("her hand on the railing", "close", seconds=2.0, subject="woman",
             scene="s1", beat="turn", link="match-action"),
        shot("a figure at the far light", "wide", seconds=2.0, scene="s2",
             beat="resolve", link="scene-cut"),
    ]}
    _shots, problems = validate(raw)
    assert any("R9" in p and "0.1s" in p for p in problems), problems
    # ...and the whole-film metronome check: equal 2.0s holds have CV 0.0
    assert any("metronomic" in p for p in problems), problems


def test_missing_purpose_and_link_are_reported_per_shot():
    raw = {"logline": "x", "shots": [
        {"prompt": "a wet bridge at night, rain", "scale": "wide", "camera": "none",
         "seconds": 2.0, "scene": "s1", "subject": "", "energy": 0.2,
         "beat": "establish"},
        {"prompt": "a woman on the bridge, rain", "scale": "close", "camera": "none",
         "seconds": 2.0, "scene": "s1", "subject": "woman", "energy": 0.2,
         "beat": "resolve"},
    ]}
    _shots, problems = validate(raw)
    assert sum(1 for p in problems if "purpose" in p and "structure:" in p) == 2
    assert sum(1 for p in problems if "link" in p and "structure:" in p) == 2
