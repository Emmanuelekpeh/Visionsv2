"""The visual organisation spine: derive holds, diagnose misalignment, rearrange.

The failure these tests pin is the one measured on the archive: a "scene" label
that changes place, light and figure on every cut (within-scene prompt overlap
0.05 vs 0.08 across, 0/10 adjacent within-scene pairs sharing a subject). The
fix is not a new rule -- it is a first-class hold/change object that the prompt,
palette, planner and model stages can all read.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from edit_planner import Shot
from visual_org import (apply_anchors, as_shots, diagnose, effective_prompt,
                        organise, organisation_stats, reference_plan, strip_png,
                        strip_text, suggestion)

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "output" / "_test_visual_org"


def shot(prompt, scene="s1", subject="", link="", purpose="", seconds=2.0,
         scale="medium") -> Shot:
    return Shot(prompt=prompt, scale=scale, camera="none", seconds=seconds,
                subject=subject, scene=scene, beat="develop", purpose=purpose,
                link=link)


def test_insert_does_not_break_a_hold():
    shots = [shot("a woman on the wet bridge, rain", "s1", "woman", "open"),
             shot("a hand on the iron railing, rain", "s1", "", "match-action"),
             shot("the same woman walking away on the bridge", "s1", "woman",
                  "match-action")]
    runs = organise(shots)
    assert len(runs) == 1
    assert runs[0].hold == 3 and runs[0].figure == "woman" and runs[0].kind == "hold"


def test_figure_change_splits_the_run_even_in_one_place():
    shots = [shot("a woman on the bridge", "s1", "woman", "open"),
             shot("a second figure on the bridge", "s1", "figure", "match-action")]
    runs = organise(shots)
    assert len(runs) == 2
    assert runs[1].change_before == "figure"
    rep = diagnose(shots, runs, mode="story")
    assert any(i["code"] == "figure_drift" for i in rep["issues"])


def test_place_change_must_be_declared_as_a_scene_cut():
    shots = [shot("a red barn in a field", "s1", "", "open"),
             shot("a neon sign in rain", "s2", "", "match-action")]
    rep = diagnose(shots, mode="story")
    assert any(i["code"] == "undeclared_place_change" and i["level"] == "break"
               for i in rep["issues"]), rep["issues"]
    shots[1] = shot("a neon sign in rain", "s2", "", "scene-cut")
    rep = diagnose(shots, mode="story")
    assert not any(i["code"] == "undeclared_place_change" for i in rep["issues"])


def test_cosmetic_hold_is_a_break_not_a_note():
    """Two 'scenes', each claiming a hold, with unrelated images inside them."""
    shots = [shot("a red barn in a green field", "s1", "", "open"),
             shot("a neon sign in the rain", "s1", "", "match-action"),
             shot("a neon sign at night", "s2", "", "scene-cut"),
             shot("a marble statue in a hall", "s2", "", "match-action")]
    runs = organise(shots)
    rep = diagnose(shots, runs, mode="story")
    assert any(i["code"] == "cosmetic_hold" and i["level"] == "break"
               for i in rep["issues"]), rep["issues"]


def test_a_declared_montage_is_intent_not_a_failure():
    shots = [shot("a red barn in a field", "m1", "", "open"),
             shot("a neon sign in the rain", "m2", "", "scene-cut"),
             shot("a marble statue in a hall", "m3", "", "scene-cut"),
             shot("a chrome diner at noon", "m4", "", "scene-cut")]
    rep = diagnose(shots, mode="montage")
    assert rep["breaks"] == 0
    assert any(i["code"] == "rapid_change" and i["level"] == "intent"
               for i in rep["issues"])


def test_effective_prompt_appends_only_missing_anchor_clauses():
    s = shot("a hand tightening on the iron railing, rain beading")
    out = effective_prompt(s, "rain-soaked footbridge at night, foggy streetlights")
    assert out.startswith(s.prompt)
    assert "footbridge" in out
    # the place words already present are not duplicated as a clause
    s2 = shot("a rain-soaked footbridge at night, empty planks, fog, streetlights")
    out2 = effective_prompt(s2, "rain-soaked footbridge at night, foggy streetlights")
    assert out2 == s2.prompt


def test_apply_anchors_only_touches_shots_in_the_run():
    shots = [shot("a woman on the bridge, rain", "s1", "woman", "open"),
             shot("her hand on the railing, rain", "s1", "", "match-action")]
    runs = organise(shots)
    out = apply_anchors(shots, runs,
                        places={"s1": "wet iron footbridge, night"},
                        figures={"woman": "woman in a dark coat"})
    assert "footbridge" in out[0].prompt and "dark coat" in out[1].prompt
    assert out[0].scale == shots[0].scale


def test_suggestion_montage_and_continuous():
    shots = [shot("a red barn in a field", "s1", "", "open"),
             shot("a neon sign in rain", "s1", "", "match-action")]
    mont, notes = suggestion(shots, "montage")
    assert [s.scene for s in mont] == ["m1", "m2"]
    assert mont[1].link == "scene-cut"
    cont, notes = suggestion(shots, "continuous")
    assert {s.scene for s in cont} == {"s1"}
    assert cont[1].link == "match-action"


def test_reference_plan_conditions_only_multi_shot_runs():
    """The renderer needs one anchor frame per hold; a rapid change gets none."""
    shots = [shot("a woman on the bridge", "s1", "woman", "open"),
             shot("her hand on the railing", "s1", "", "match-action"),
             shot("the same bridge, rain", "s1", "woman", "match-action"),
             shot("a new alley at night", "s2", "", "scene-cut")]
    plan = reference_plan(shots)
    assert plan == {1: 0, 2: 0}          # shots 2,3 reference shot 1
    assert 3 not in plan                 # the rapid change is not conditioned


def test_strip_and_png_are_produced():
    shots = [shot("a woman on the wet bridge", "s1", "woman"),
             shot("the bridge railing in rain", "s1", ""),
             shot("a neon sign in the rain", "s2", "", "scene-cut")]
    runs = organise(shots)
    text = strip_text(shots, runs)
    assert "s1" in text and "s2" in text
    out = strip_png(shots, runs, "story", SCRATCH / "strip.png", title="t")
    assert out.exists() and out.stat().st_size > 1000


def test_suggestion_does_not_merge_places_on_boilerplate():
    """Cinematic/filmic-grain/rain are the film's look, not its places.

    An early version merged a bridge railing into a distant figure because both
    prompts contained the same atmosphere words. Distinctive-word matching is
    what keeps a rapid-change run a rapid-change run.
    """
    shots = [shot("a bridge in rain, cinematic, filmic grain", "s1"),
             shot("a railing in rain, cinematic, filmic grain", "s2"),
             shot("an alley in rain, cinematic, filmic grain", "s3"),
             shot("a distant figure in rain, cinematic, filmic grain", "s4")]
    new, notes = suggestion(shots, "hybrid")
    assert len({s.scene for s in new}) == 4, notes
    assert any("kept as a change" in n for n in notes)


def test_film_project_organisation_stage(monkeypatch):
    import film_project as fp
    monkeypatch.setattr(fp, "WORKSPACE", SCRATCH / "workspace")
    proj = fp.new("orgtest", "a woman follows a figure across a bridge")
    raw = {"logline": "x", "shots": [
        {"prompt": "a rain-soaked footbridge at night, a woman in a dark coat "
                   "stepping onto the planks, streetlights, fog", "scale": "wide",
         "camera": "none", "seconds": 2.5, "subject": "woman", "scene": "s1",
         "energy": 0.2, "beat": "establish", "purpose": "establish", "link": "open"},
        {"prompt": "the same woman walking along the footbridge in the rain, "
                   "streetlights, fog", "scale": "medium", "camera": "none",
         "seconds": 1.7, "subject": "woman", "scene": "s1", "energy": 0.2,
         "beat": "develop", "purpose": "master", "link": "match-action"},
        {"prompt": "a second figure under the far streetlight on the same "
                   "footbridge, fog and rain", "scale": "medium", "camera": "none",
         "seconds": 2.2, "subject": "figure", "scene": "s2", "energy": 0.3,
         "beat": "turn", "purpose": "reaction", "link": "scene-cut"},
        {"prompt": "the two figures facing each other on the rain-soaked "
                   "footbridge, fog", "scale": "close", "camera": "none",
         "seconds": 1.5, "subject": "figure", "scene": "s2", "energy": 0.1,
         "beat": "resolve", "purpose": "resolve", "link": "eyeline"},
    ]}
    fp.shots_from_raw(proj, raw)
    assert proj["status"]["organise"] == "stale"
    fp.organise_project(proj, mode="story",
                        places={"s1": "rain-soaked footbridge at night, fog",
                                "s2": "rain-soaked footbridge at night, fog"},
                        figures={"woman": "a woman in a dark hooded coat"})
    assert proj["status"]["organise"] in ("organised", "breaks")
    assert proj["organisation"]["runs"]
    fp.apply_organisation(proj, "montage")
    assert proj["organisation"]["mode"] == "montage"
    assert [d["scene"] for d in proj["shots"]] == ["m1", "m2", "m3", "m4"]
    assert proj["plan"]["frozen"] == []
    # a montage has no breaks; approving it must work
    if proj["organisation"]["breaks"] == 0:
        fp.approve_organisation(proj)
        assert proj["status"]["organise"] == "approved"


def test_store_shots_are_accepted():
    """The spine must work on the older production store too, not just the new process."""
    shots = as_shots([{"prompt": "a wet bridge, rain", "scene": "sc1",
                       "subject": "woman", "camera": "none", "camera_suggestion": None},
                      {"prompt": "the same bridge railing, rain", "scene": "sc1",
                       "subject": "", "camera": "none"}])
    assert len(shots) == 2
    runs = organise(shots)
    assert len(runs) == 1 and runs[0].hold == 2
