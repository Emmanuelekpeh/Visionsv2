"""The render seam: a frozen workbench plan becomes a real production.

This is the integration test the audit said was missing. It does not monkeypatch
`shot_state` or `shots_in_order`: it writes a real store under a temporary
PRODUCTIONS root, then asks the same functions the renderer uses whether the
second shot of a hold is conditioned, what its anchor is, and whether the place
palettes survived an organisation rewrite.
"""

from __future__ import annotations

from pathlib import Path

import film_store as fs
import film_project as fp

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "output" / "_test_seam"


def _raw():
    return {"logline": "x", "shots": [
        {"prompt": "a rain-soaked footbridge at night, a woman stepping on, "
                   "streetlights and fog", "scale": "wide", "camera": "none",
         "seconds": 2.5, "subject": "woman", "scene": "s1", "energy": 0.2,
         "beat": "establish", "purpose": "establish", "link": "open"},
        {"prompt": "the same woman walking along the footbridge, rain and fog",
         "scale": "medium", "camera": "none", "seconds": 1.7, "subject": "woman",
         "scene": "s1", "energy": 0.2, "beat": "develop", "purpose": "master",
         "link": "match-action"},
        {"prompt": "a second figure under the streetlight on the same bridge, fog",
         "scale": "medium", "camera": "none", "seconds": 2.0, "subject": "figure",
         "scene": "s2", "energy": 0.3, "beat": "turn", "purpose": "reaction",
         "link": "scene-cut"},
        {"prompt": "the two figures on the rain-soaked bridge, fog",
         "scale": "close", "camera": "none", "seconds": 1.5, "subject": "figure",
         "scene": "s2", "energy": 0.1, "beat": "resolve", "purpose": "resolve",
         "link": "eyeline"},
    ]}


def _project(name: str = "seamtest"):
    proj = fp.new(name, "a woman crosses a rainy bridge")
    fp.shots_from_raw(proj, _raw())
    shots = fp.shot_objects(proj)
    proj["plan"]["frozen"] = [fp._shot_to_dict(s, i) for i, s in enumerate(shots)]
    proj["plan"]["approved"] = True
    proj["plan"]["approved_at"] = "test"
    return proj


def _workspace(monkeypatch):
    prod = SCRATCH / "productions"
    if prod.exists():
        import shutil
        shutil.rmtree(prod)
    prod.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(fs, "PRODUCTIONS", prod)
    return prod


def test_materialise_is_a_real_store_and_the_hold_conditions(monkeypatch):
    _workspace(monkeypatch)
    proj = _project()
    conv = fp.materialize_store(proj, name="seamtest")
    assert conv["slug"] == "fp_seamtest"
    store = fs.load("fp_seamtest")
    assert store["timeline"] == ["s01", "s02", "s03", "s04"]
    assert [s.get("name") for s in store["scenes"]] == ["s1", "s2"]
    shots = fs.shots_in_order(store)          # the REAL store path, no patching
    assert len(shots) == 4
    refs = [fs.generation_inputs(store, s, seed=7).get("hold_reference")
            for s in shots]
    assert refs[0] == ""                      # the anchor is not conditioned
    assert refs[1].startswith("s01:")         # shot 2 is anchored on shot 1
    assert refs[2] == ""                      # new place: not conditioned
    assert refs[3].startswith("s03:")         # shot 4 anchored on shot 3
    # re-materialising the same frozen plan is idempotent and keeps take slots
    conv2 = fp.materialize_store(proj, name="seamtest")
    assert conv2["created"] is False
    assert fs.load("fp_seamtest")["timeline"] == store["timeline"]


def test_store_can_be_rearranged_and_anchored_in_the_studio(monkeypatch):
    """The studio's own model: apply an arrangement + bake anchors, no workbench."""
    _workspace(monkeypatch)
    import visual_org as vo
    conv = fp.materialize_store(_project(), name="storeorg")
    p = fs.load(conv["slug"])
    fs.set_palette(p, p["scenes"][0]["id"], "warm-gold")
    fs.save(p)

    new, notes = vo.apply_to_store(fs.load(conv["slug"]), "montage")
    p2 = fs.load(conv["slug"])
    shots = fs.shots_in_order(p2)
    assert len({s["scene"] for s in shots}) == 4
    assert all(s.get("link") == "scene-cut" for s in shots[1:])
    palettes = {s.get("name"): s.get("palette") for s in p2["scenes"]}
    assert palettes.get("m1") == "warm-gold"      # palette followed the place

    p2["organisation"] = {"places": {
        "m1": "brass lamp posts, black iron railing, wet cobbles"},
        "figures": {}}
    fs.save(p2)
    changed = vo.apply_anchors_to_store(fs.load(conv["slug"]))
    assert changed >= 1
    prompts = [s["prompt"] for s in fs.shots_in_order(fs.load(conv["slug"]))]
    assert any("brass lamp posts" in pr for pr in prompts)
    # a prompt that already carries the anchor words is not duplicated
    assert sum(1 for pr in prompts if pr.count("brass lamp posts") > 1) == 0


def test_place_palette_survives_an_organisation_rewrite(monkeypatch):
    _workspace(monkeypatch)
    proj = _project()
    fp.set_place_palette(proj, "s1", "warm-gold")
    fp.apply_organisation(proj, "montage")
    # an organisation change invalidates the frozen plan (correctly); re-freeze
    proj["plan"]["frozen"] = [fp._shot_to_dict(s, i)
                              for i, s in enumerate(fp.shot_objects(proj))]
    conv = fp.materialize_store(proj, name="palettetest")
    store = fs.load(conv["slug"])
    palettes = {s.get("name"): s.get("palette") for s in store["scenes"]}
    # s1's shots became m1 (first shot); the place palette travelled with them
    assert palettes.get("m1") == "warm-gold"
    assert len(store["timeline"]) == 4


def test_adoption_keeps_the_planned_palette_when_scenes_are_relabelled(monkeypatch):
    """The director rewrites scene labels (planN -> movement ids) BEFORE adoption.

    Measured on the squid draft: a scene-label-keyed palette map then missed every
    planN key, so all three store scenes fell back to neutral. The palette must
    travel on the SHOT.
    """
    import story
    from edit_planner import Shot

    _workspace(monkeypatch)
    shots = [
        Shot("a diver's helmet in bioluminescent coral light", "wide", "none",
             2.5, scene="mv1", beat="establish", purpose="establish",
             link="open", palette="emerald"),
        Shot("the diver over a silty floor, jellyfish drifting", "medium", "none",
             2.0, scene="mv1", beat="develop", purpose="master",
             link="match-action", palette="emerald"),
        Shot("a glowing squid eye in the dark", "close", "none", 1.8,
             scene="mv2", beat="resolve", purpose="resolve", link="scene-cut",
             palette="magenta"),
    ]
    raw = {"logline": "x", "scene_plan": {"scenes": [
        {"place": "a", "figure": "the diver", "action": "x",
         "light": "green light", "palette": "cold-steel", "shots": 2},
        {"place": "b", "figure": "the diver", "action": "y",
         "light": "blue light", "palette": "cyan-night", "shots": 1},
    ]}}
    proposal = story.Proposal(concept="x", model="stub", logline="x", shots=shots,
                              raw=raw, seconds=0.0, decode_tok_s=0.0,
                              problems=[])
    project = fs.new_project("relabel", concept="x")
    fs.shots_from_proposal(project, proposal)
    palettes = {sc["id"]: sc.get("palette") for sc in project["scenes"]}
    assert "emerald" in palettes.values()
    assert "magenta" in palettes.values()


def test_adoption_applies_the_plans_scene_palettes(monkeypatch):
    """The plan's palettes are the grade make_film applies. Adoption must carry
    them onto the store scenes, not default everything to neutral."""
    import story
    from edit_planner import Shot

    _workspace(monkeypatch)

    def _shot(prompt, scene, beat, purpose, link, seconds=2.4):
        return Shot(prompt=prompt, scale="wide", camera="none", seconds=seconds,
                    subject="woman", scene=scene, energy=0.2, beat=beat,
                    purpose=purpose, link=link)

    shots = [
        _shot("a rain-soaked footbridge at night, a woman on the wet planks, fog",
              "plan1", "establish", "establish", "open"),
        _shot("the same woman walking along the footbridge, rain on the planks",
              "plan1", "develop", "master", "match-action"),
        _shot("a dim room with a cracked mirror, the woman facing it", "plan2",
              "turn", "reaction", "scene-cut"),
        _shot("the cracked mirror, the woman's reflection, cold light", "plan2",
              "resolve", "resolve", "eyeline"),
    ]
    raw = {"logline": "x", "scene_plan": {"scenes": [
        {"place": "a footbridge", "figure": "the woman", "action": "crosses",
         "light": "hard amber streetlight", "palette": "warm-gold", "shots": 2},
        {"place": "a room", "figure": "the woman", "action": "faces it",
         "light": "cold silver light", "palette": "cold-steel", "shots": 2},
    ]}}
    proposal = story.Proposal(concept="x", model="stub", logline="x", shots=shots,
                              raw=raw, seconds=0.0, decode_tok_s=0.0,
                              problems=[])
    project = fs.new_project("paletteplan", concept="x")
    fs.shots_from_proposal(project, proposal)
    palettes = [sc.get("palette") for sc in project.get("scenes") or []]
    assert "warm-gold" in palettes
    assert "cold-steel" in palettes
