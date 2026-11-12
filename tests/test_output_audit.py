"""output_audit: the delivered film measured against the writing that made it.

The first rendered film was audited by a one-off script tied to that production.
This test file pins the parts that make the tool general: the pure verdict logic
(so the report, the exit code and the flags cannot drift), the resolve's
"new imagery" warning that names the car, and an end-to-end audit of a real store
without CLIP (no model load, no render).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import output_audit as oa

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "output" / "_test_output_audit"


def test_hold_cv_sees_a_metronome_and_a_rhythm():
    assert oa.hold_cv([]) == 0.0
    assert oa.hold_cv([2.0, 2.0, 2.0]) < 0.01
    assert oa.hold_cv([1.0, 2.0, 4.0, 2.0]) > 0.4


def test_beat_bar_marks_the_film_and_an_unknown_beat():
    assert oa.beat_bar(["establish", "develop", "turn", "fall", "resolve"]
                       ) == "EdTfR"
    assert oa.beat_bar(["establish", "weird"]) == "E?"


def test_cut_by_mode_groups_by_the_direction_it_was_cut_under():
    cuts = {"s01->s02": {"clip": 0.9, "hist": 0.3, "luma": 1.0},
            "s02->s03": {"clip": 0.6, "hist": 0.9, "luma": 20.0},
            "s03->s04": {"clip": 0.7, "hist": 0.8, "luma": 3.0}}
    modes = {"s01->s02": "hold", "s02->s03": "montage",
             "s03->s04": "montage"}
    out = oa.cut_by_mode(cuts, modes)
    assert out["hold"]["cuts"] == 1
    assert out["montage"]["cuts"] == 2
    assert out["montage"]["mean_clip"] == 0.65
    # a cut with no known mode is not silently dropped
    assert oa.cut_by_mode(cuts, {})["?"]["cuts"] == 3


def test_the_resolve_new_imagery_warning_names_the_car():
    """The measured failure: the resolve introduced a car the film never had."""
    before = ["a man standing alone on a wet parking lot, long shadow",
              "a rain-soaked bridge at night, fog and streetlights"]
    after = ("the protagonist's shadow on the slick reflective surface of an old "
             "car; symbolic of ego's acceptance")
    words = oa.new_words(before, after)
    assert "car" in words
    # a word the film already had is not "new"
    assert "shadow" not in words


def test_flags_name_the_measured_failures_including_the_good_cuts():
    metrics = {
        "alignment": {"mean_margin": 0.011,
                      "margin_by_shot": {"s01": 0.05, "s02": -0.04}},
        "cut": {"per_cut": {"s07->s08": {"clip": 0.74, "hist": 0.99,
                                         "luma": 20.7}}},
        "flow": {"holds_s": {"s01": 2.0}, "hold_cv": 0.084},
        "writing": {"flagged_prompts": ["1", "2"],
                    "coverage": ["coverage: scene 's1' holds a LABEL but not a "
                                 "place"]},
        "resolve": {"new_words": ["car"]},
    }
    codes = {f["code"] for f in oa.flags_from(metrics)}
    assert {"weak-alignment", "shot-mismatch", "exposure-jump", "metronomic",
            "prompt-shape", "no-coverage", "resolve-new-imagery"} <= codes
    # a hard cut is information, not a defect: a montage is legitimate
    info = [f for f in oa.flags_from(metrics) if f["code"] == "hard-cut"]
    assert info == []


def test_a_palette_change_at_a_scene_cut_is_style_not_an_exposure_flash():
    """Same principle as previs: inside a hold a luma jump is a defect; at a scene
    cut the palette/light change is the edit. The render normalises exposure first,
    so the audit must not call the deliberate grade a flash."""
    metrics = {
        "cut": {"per_cut": {
            "s04->s05": {"clip": 0.8, "hist": 0.9, "luma": 18.0,
                         "same_scene": False},
            "s02->s03": {"clip": 0.8, "hist": 0.2, "luma": 18.0,
                         "same_scene": True},
        }},
    }
    flags = oa.flags_from(metrics)
    codes = {(f["code"], f["level"]) for f in flags}
    assert ("style-cut", "info") in codes
    assert ("exposure-jump", "warn") in codes


def test_anchored_prompts_make_the_alignment_margin_informational():
    metrics = {"anchors": True,
               "alignment": {"mean_margin": -0.013,
                             "margin_by_shot": {"s02": -0.058}}}
    flags = oa.flags_from(metrics)
    codes = {f["code"] for f in flags}
    assert "weak-alignment" not in codes
    assert "shot-mismatch" not in codes
    assert "anchored-prompts" in codes


def test_anchored_prompts_are_not_reported_as_writer_faults(monkeypatch):
    """Anchors intentionally make every prompt longer than one image.

    The writer's shape was measured BEFORE the planned place/figure was appended,
    so a store that records anchors must not be flagged for the append.
    """
    import film_store as fs
    import story

    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    prod = SCRATCH / "productions"
    prod.mkdir(parents=True)
    monkeypatch.setattr(fs, "PRODUCTIONS", prod)
    try:
        slug = "anchored"
        project = fs.new_project(slug, concept="a woman on a bridge")
        long_prompt = ("a rain-soaked footbridge at night, a woman in a dark coat "
                       "walking along the wet planks, streetlights in fog, rain "
                       "beading on the iron railing, distant city lights")
        raw = {"logline": "x", "shots": [
            {"prompt": long_prompt, "scale": "wide", "camera": "none",
             "seconds": 2.5, "subject": "woman", "scene": "s1", "energy": 0.1,
             "beat": "establish", "purpose": "establish", "link": "open"},
            {"prompt": long_prompt + ", seen from behind", "scale": "close",
             "camera": "none", "seconds": 1.8, "subject": "woman", "scene": "s1",
             "energy": 0.4, "beat": "turn", "purpose": "reaction",
             "link": "match-action"},
            {"prompt": long_prompt + ", the bridge empty at dawn", "scale": "medium",
             "camera": "none", "seconds": 2.2, "subject": "woman", "scene": "s1",
             "energy": 0.1, "beat": "resolve", "purpose": "resolve",
             "link": "eyeline"},
        ]}
        shots, _ = story.validate(raw)
        proposal = story.Proposal(concept="x", model="stub", logline="x",
                                  shots=shots, raw=raw, seconds=0.0,
                                  decode_tok_s=0.0, problems=[])
        fs.shots_from_proposal(project, proposal)
        fs.save(project)

        flagged_before = [f for f in oa.audit(slug, with_clip=False)["flags"]
                          if f["code"] == "prompt-shape"]
        assert flagged_before, "a 30-word composite prompt is a writer fault"

        project = fs.load(slug)
        project["anchors"] = {"bound": 3, "applied": True}   # as adoption records it
        fs.save(project)
        flagged_after = [f for f in oa.audit(slug, with_clip=False)["flags"]
                         if f["code"] == "prompt-shape"]
        assert flagged_after == []
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)


def test_audit_runs_on_a_real_store_without_clip(monkeypatch):
    """The tool must work on any store with no CLIP, no render, no network."""
    import film_store as fs
    import story

    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    prod = SCRATCH / "productions"
    prod.mkdir(parents=True)
    monkeypatch.setattr(fs, "PRODUCTIONS", prod)
    try:
        slug = "audited"
        project = fs.new_project(slug, concept="a woman on a bridge")
        raw = {"logline": "x", "shots": [
            {"prompt": "a rain-soaked footbridge at night, a woman in a dark "
                       "coat on the planks, streetlights and fog", "scale": "wide",
             "camera": "none", "seconds": 2.5, "subject": "woman", "scene": "s1",
             "energy": 0.1, "beat": "establish", "purpose": "establish",
             "link": "open"},
            {"prompt": "the same woman walking along the footbridge, rain and fog",
             "scale": "medium", "camera": "none", "seconds": 2.0,
             "subject": "woman", "scene": "s1", "energy": 0.2, "beat": "develop",
             "purpose": "master", "link": "match-action"},
            {"prompt": "a second figure under the far streetlight on the bridge, "
                       "fog, facing her", "scale": "medium", "camera": "none",
             "seconds": 1.8, "subject": "figure", "scene": "s2", "energy": 0.4,
             "beat": "turn", "purpose": "reaction", "link": "scene-cut"},
            {"prompt": "the two figures on the rain-soaked bridge, fog between "
                       "them", "scale": "close", "camera": "none", "seconds": 1.5,
             "subject": "figure", "scene": "s2", "energy": 0.1, "beat": "resolve",
             "purpose": "resolve", "link": "eyeline"},
        ]}
        shots, _ = story.validate(raw)
        proposal = story.Proposal(concept="x", model="stub", logline="x",
                                  shots=shots, raw=raw, seconds=0.0,
                                  decode_tok_s=0.0, problems=[])
        fs.shots_from_proposal(project, proposal)
        fs.save(project)

        metrics = oa.audit(slug, with_clip=False)
        assert metrics["render"] is None
        assert metrics["cut"]["beat_shape"] == "EdTR"
        assert len(metrics["writing"]["prompt_words"]) == 4
        assert sum(metrics["writing"]["prompt_words"]) > 30
        assert metrics["flow"]["hold_cv"] > 0
        assert isinstance(metrics["flags"], list)
        text = oa.report_text(metrics)
        assert "beat shape: EdTR" in text
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)


def test_take_mid_images_use_the_take_the_film_cuts_to(tmp_path, monkeypatch):
    """It used takes[-1] (the newest), so an older CHOSEN take was audited as if the
    newer one had been delivered; and a shot with no take crashed with IndexError."""
    import pytest
    from PIL import Image
    import film_store as fs
    import output_audit as oa

    monkeypatch.setattr(fs, "PRODUCTIONS", tmp_path)
    project = fs.new_project("auditcase")
    fs.save(project)
    shot = fs.add_shot(project, prompt="a lighthouse", scene="sc1")
    for name, colour in (("t01", (200, 0, 0)), ("t02", (0, 0, 200))):
        d = fs.take_dir(project, shot, name) / "frames"
        d.mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (8, 8), colour).save(d / f"f{i:03d}.png")
        fs._write_json(d.parent / "take.json", {"take": name, "shot": shot["id"]})
    fs.choose_take(project, shot, "t01")          # chosen is the OLDER take
    img = oa.take_mid_images(project)[0]
    assert img.getpixel((0, 0)) == (200, 0, 0)

    empty = fs.add_shot(project, prompt="no take yet", scene="sc1")
    with pytest.raises(SystemExit):
        oa.take_mid_images(project)
