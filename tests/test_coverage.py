"""Coverage: a film of 2-3 held scenes, not one new place per shot.

Measured motivation. The writer has always been told "2-3 scenes, one place and
one figure each, 2-4 shots". Nothing measured it. The first rendered `ego` film
came back as 13 shots across ~12 places, so `sequence_direction` -- correctly --
montaged them: 32 cuts/min, no recurring figure, and the film read as an anthology
of ego images rather than a story.

So: a cheap SCENE PLAN pass before the shots (2-3 places, a recurring figure, shot
budgets summing to the request), `coverage:` measurement using the sequence
director's OWN affinity (`visual_org.diagnose`), and ONE coverage repair re-ask
when the story holds but no scene does. Coverage is ADVISORY at the gate -- a
montage is a legitimate arrangement -- but it is a demand of the writing path.
"""

from __future__ import annotations

import json

import story
from story import (SCENE_PLAN_SCHEMA, SHOT_SCHEMA, _normalise_scene_plan,
                   _showrunner_user, coverage_problems, write_scene_plan,
                   write_shots_ensemble)


def _shot(prompt, scale, beat, scene, purpose, link, subject="woman",
          seconds=2.0, energy=0.2) -> dict:
    return {"prompt": prompt, "scale": scale, "camera": "none",
            "seconds": seconds, "subject": subject, "scene": scene,
            "energy": energy, "beat": beat, "purpose": purpose, "link": link}


# one new place per shot: a montage of postcards
MONTAGE = {"logline": "a man meets his shadow", "shots": [
    _shot("a rain-soaked footbridge at night, fog and streetlights", "wide",
          "establish", "s1", "establish", "open", subject=""),
    _shot("a bare room with an empty chair, cold window light", "medium",
          "develop", "s2", "master", "scene-cut", subject=""),
    _shot("a dark forest path, mist between the trees at night", "close",
          "develop", "s3", "insert", "scene-cut", subject=""),
    _shot("a campfire in a rocky canyon at night, orange flames", "wide",
          "turn", "s4", "reaction", "scene-cut", subject=""),
    _shot("a flooded parking lot at dawn, grey sky and wet asphalt", "medium",
          "resolve", "s5", "master", "scene-cut", subject=""),
    _shot("a blue car in a showroom, hard overhead lights", "close",
          "resolve", "s6", "resolve", "scene-cut", subject=""),
]}

# two held scenes: one place + one figure, covered from three angles each
COVERAGE = {"logline": "a woman crosses a footbridge and faces her reflection",
            "shots": [
    _shot("a rain-soaked footbridge at night, a woman in a dark coat on the "
          "wet planks, fog", "wide", "establish", "s1", "establish", "open"),
    _shot("the same woman walking along the footbridge, rain on the planks, fog",
          "medium", "develop", "s1", "master", "match-action"),
    _shot("the woman's hand on the wet iron railing of the footbridge, rain",
          "close", "develop", "s1", "insert", "match-action"),
    _shot("a dim room with a cracked mirror, the woman's face reflected, cold "
          "light", "wide", "turn", "s2", "reaction", "scene-cut"),
    _shot("the same woman facing the cracked mirror in the dim room, cold light",
          "medium", "resolve", "s2", "resolve", "match-action"),
    _shot("the cracked mirror in the dim room, the woman's reflection, cold "
          "light", "close", "resolve", "s2", "reaction", "match-action"),
]}

PLAN = {"scenes": [
    {"place": "a rain-soaked footbridge at night", "figure": "a woman in a dark "
     "coat", "action": "she crosses the bridge and is followed", "shots": 3},
    {"place": "a dim room with a cracked mirror", "figure": "the same woman",
     "action": "she faces the reflection", "shots": 3},
]}


def _shots(raw):
    shots, _problems = story.validate(raw)
    return shots


# --------------------------------------------------------------------------- #
# the measurement
# --------------------------------------------------------------------------- #

def test_a_montage_of_places_is_flagged():
    probs = coverage_problems(_shots(MONTAGE))
    assert probs, "a one-place-per-shot film must not pass the coverage check"
    assert all(p.startswith("coverage:") for p in probs)
    assert any("no scene holds" in p or "scene" in p for p in probs)


def test_a_film_of_two_held_scenes_passes():
    shots = _shots(COVERAGE)
    assert coverage_problems(shots) == []
    # and it is the same film the writer was asked for: 2 scenes, one figure
    assert len({s.scene for s in shots}) == 2
    assert len({s.subject for s in shots}) == 1


def test_a_scene_label_that_does_not_hold_is_flagged():
    """The archive failure: one scene id, several unrelated pictures."""
    cosmetic = {"logline": "x", "shots": [
        _shot("a cracked concrete floor under moonlight", "wide", "establish",
              "s1", "establish", "open"),
        _shot("a snow-covered mountain ridge at noon", "medium", "develop", "s1",
              "master", "match-action"),
        _shot("a crowded subway platform, fluorescent light", "close", "develop",
              "s1", "insert", "match-action"),
        _shot("a swimming pool at night, blue tiles", "wide", "turn", "s2",
              "reaction", "scene-cut"),
        _shot("the empty swimming pool at dawn, blue tiles", "medium", "resolve",
              "s2", "resolve", "match-action"),
    ]}
    probs = coverage_problems(_shots(cosmetic))
    assert probs
    assert any("cosmetic" in p or "hold" in p or "coverage" in p for p in probs)


def test_validate_carries_coverage_and_the_gate_treats_it_as_advisory():
    from draft import story_gate

    _shots, problems = story.validate(MONTAGE)
    coverage = [p for p in problems if p.startswith("coverage:")]
    assert coverage
    allowed, breaks, warns = story_gate(problems)
    assert allowed is True
    assert breaks == []
    assert coverage[0] in warns


# --------------------------------------------------------------------------- #
# the scene plan
# --------------------------------------------------------------------------- #

def test_plan_is_normalised_to_the_requested_shot_count():
    plan = _normalise_scene_plan({"scenes": [
        {"place": "a", "figure": "the man", "action": "walks", "shots": 9},
        {"place": "b", "figure": "the man", "action": "stops", "shots": 1},
    ]}, 6)
    assert plan is not None
    assert len(plan["scenes"]) == 2
    assert sum(sc["shots"] for sc in plan["scenes"]) == 6
    # too few/many scenes, or missing fields, are rejected rather than fudged
    assert _normalise_scene_plan({"scenes": [
        {"place": "a", "figure": "m", "action": "x", "shots": 6}]}, 6) is None
    assert _normalise_scene_plan({"scenes": [
        {"place": "a", "figure": "", "action": "x", "shots": 3},
        {"place": "b", "figure": "m", "action": "y", "shots": 3}]}, 6) is None


def test_anchor_clauses_do_not_end_on_a_dangling_word():
    from story import _clip_clause

    assert _clip_clause("a man sitting cross-legged in a dimly lit room", 6) == (
        "a man sitting cross-legged")
    assert _clip_clause("a man sitting cross-legged in a", 8) == (
        "a man sitting cross-legged")
    assert _clip_clause("the keeper", 6) == "the keeper"


def test_anchor_clauses_drop_dangling_adverbs_and_prepositions():
    """Measured on the lighthouse draft: shots 3-10 started 'a rocky coastline with
    only,' and the light clause ended 'silhouetted against,'."""
    from story import _clip_clause

    assert _clip_clause("a rocky coastline with only a distant glowing signal", 6) == (
        "a rocky coastline")
    assert _clip_clause("a lone keeper silhouetted against amber dusk", 4) == (
        "a lone keeper silhouetted")
    assert _clip_clause("a boat drifting near", 8) == "a boat drifting"


def test_a_shot_that_does_not_match_its_scenes_place_gets_no_place_anchor():
    """Shots after the last strong match all land in the final scene (lighthouse:
    1/1/8). 'calm bay at dawn' must not be prefixed 'a rocky coastline'; it keeps
    its scene label, figure and light."""
    from story import anchor_plan

    plan = {"scenes": [
        {"place": "a lone lighthouse in a storm", "figure": "a keeper",
         "action": "watches", "light": "cold daylight", "palette": "cold-steel",
         "shots": 1},
        {"place": "a rocky coastline", "figure": "a keeper", "action": "lands",
         "light": "amber dusk", "palette": "amber-dusk", "shots": 1},
    ]}
    raw = {"logline": "x", "shots": [
        _shot("a lone lighthouse in a storm, waves below, a keeper at the rail",
              "wide", "establish", "s1", "establish", "open", subject="keeper"),
        _shot("the keeper steps onto a rocky coastline at night, wet stones",
              "medium", "develop", "s2", "master", "scene-cut", subject="keeper"),
        _shot("a calm bay at dawn with soft glow on still water, the keeper small",
              "wide", "resolve", "s2", "reaction", "match-action", subject="keeper"),
    ]}
    out = anchor_plan(_shots(raw), plan)
    assert out[1].prompt.startswith("a rocky coastline")
    assert "rocky coastline" not in out[2].prompt.lower()
    assert out[2].scene == "plan2" and out[2].palette == "amber-dusk"


def test_the_planned_palette_survives_the_revalidate_after_anchoring():
    """`anchor_plan` stamps the scene's grade on each shot, then the pipeline
    round-trips shots through shot_dict -> validate. That used to drop `palette`:
    the lighthouse plan chose three grades and the store got all-neutral."""
    from story import anchor_plan

    plan = {"scenes": [
        {"place": "a rain-soaked footbridge", "figure": "a woman", "action": "crosses",
         "light": "hard amber streetlight", "palette": "warm-gold", "shots": 3},
        {"place": "a dim room with a cracked mirror", "figure": "a woman",
         "action": "faces it", "light": "cold window light", "palette": "cyan-night",
         "shots": 3},
    ]}
    anchored = anchor_plan(_shots(COVERAGE), plan)
    assert {s.palette for s in anchored} == {"warm-gold", "cyan-night"}
    raw = {"logline": "x", "shots": [story.shot_dict(s) for s in anchored]}
    revalidated, _problems = story.validate(raw)
    assert [s.palette for s in revalidated] == [s.palette for s in anchored]


def test_a_plan_keeps_ONE_figure_across_its_scenes():
    """The 13-shot plan wrote a different figure description per scene, and the
    anchors then bound two different figures into the film."""
    plan = _normalise_scene_plan({"scenes": [
        {"place": "a courtyard", "figure": "man in suit, tied to chair",
         "action": "faces the shadow", "shots": 2},
        {"place": "a room", "figure": "man in suit, tied to chair",
         "action": "faces the shadow", "shots": 2},
        {"place": "a square", "figure": "person in dim light",
         "action": "walks free", "shots": 2},
    ]}, 6)
    assert plan is not None
    assert len({sc["figure"] for sc in plan["scenes"]}) == 1
    assert plan["scenes"][0]["figure"] == "man in suit, tied to chair"


def test_a_plan_carries_one_light_and_palette_per_scene():
    """The 13-shot films were uniformly foggy because every scene shared the same
    atmosphere words. The plan now commits each scene to its own light/palette, and
    the palette is the grade `make_film` will apply."""
    plan = _normalise_scene_plan({"scenes": [
        {"place": "a courtyard", "figure": "the man", "action": "walks",
         "light": "hard low sun, warm dust", "palette": "warm-gold", "shots": 3},
        {"place": "a room", "figure": "the man", "action": "stops",
         "light": "cold blue night, one lamp", "palette": "cyan-night", "shots": 3},
    ]}, 6)
    assert plan is not None
    assert plan["scenes"][0]["light"] == "hard low sun, warm dust"
    assert plan["scenes"][1]["palette"] == "cyan-night"
    # an unknown palette is not trusted into the grade
    bad = _normalise_scene_plan({"scenes": [
        {"place": "a", "figure": "m", "action": "x", "light": "l",
         "palette": "hot-pink", "shots": 3},
        {"place": "b", "figure": "m", "action": "y", "light": "l",
         "palette": "bleach", "shots": 3},
    ]}, 6)
    assert bad["scenes"][0]["palette"] == "neutral"
    assert bad["scenes"][1]["palette"] == "bleach"


def test_the_scenes_light_lands_at_the_end_and_the_place_leads():
    """Measured twice: full anchor blocks at the FRONT dominated the image and
    every shot in a scene drew the same thing (squid). Place leads SHORT, the
    shot's action stays early, and the light colours from the END."""
    from story import anchor_plan

    lit = {"scenes": [
        {"place": "a rain-soaked footbridge", "figure": "a woman",
         "action": "crosses", "light": "hard amber streetlight", "shots": 3},
        {"place": "a dim room with a cracked mirror", "figure": "a woman",
         "action": "faces it", "light": "cold silver window light", "shots": 3},
    ]}
    anchored = anchor_plan(_shots(COVERAGE), lit)
    assert "hard amber streetlight" in anchored[0].prompt[-60:]
    # scene 2's prompt already says "cold light", so the light clause is not
    # duplicated (only absent clauses are appended)
    assert "cold silver window light" not in anchored[3].prompt
    assert anchored[0].prompt.startswith("a rain-soaked footbridge")
    assert anchored[3].prompt.startswith("a dim room")
    # the action sentence survives early, not buried under a repeated block
    assert "walking along the footbridge" in anchored[1].prompt[:120]


def test_anchors_are_assigned_by_content_not_by_the_plan_budget():
    """The 13-shot run's plan said 4/4/5; its images ran 4 courtyard, 4 courtyard/
    stained, 5 square. Assigning by the budget anchored courtyard images with
    stained-glass words. Content assignment (order preserved, budgets a tie-break)
    puts each shot where its own nouns belong."""
    from story import anchor_plan

    raw = {"logline": "x", "shots": []}
    for i in range(1, 5):
        raw["shots"].append(_shot(f"courtyard at dusk, man in suit, long shadow "
                                  f"{i}", "medium", "establish" if i == 1
                                  else "develop", "s1", "establish" if i == 1
                                  else "master", "open" if i == 1
                                  else "match-action", subject="man"))
    for i in range(2):
        raw["shots"].append(_shot(f"stained glass window, moonlit tiled floor "
                                  f"{i}", "wide", "turn", "s2", "reaction",
                                  "scene-cut", subject="man"))
    plan = {"scenes": [
        {"place": "an old courtyard, overcast sky", "figure": "man in suit",
         "action": "faces the shadow", "shots": 3},
        {"place": "moonlight through a cracked stained glass window",
         "figure": "man in suit", "action": "the shadow dissolves", "shots": 3},
    ]}
    anchored = anchor_plan(_shots(raw), plan)
    sizes = {}
    for s in anchored:
        sizes[s.scene] = sizes.get(s.scene, 0) + 1
    # content beats the 3/3 budget: the courtyard is four shots, not three
    assert sizes == {"plan1": 4, "plan2": 2}
    assert all("courtyard" in s.prompt for s in anchored[:4])
    assert all("stained glass" in s.prompt for s in anchored[4:])


def test_the_brief_puts_the_scene_plan_above_the_scouts_ideas():
    msg = _showrunner_user("a woman on a bridge", 6, "- [farm] an idea",
                           scene_plan=PLAN)
    assert "THE SCENES" in msg
    assert "a rain-soaked footbridge at night" in msg
    assert msg.index("THE SCENES") < msg.index("scouts' raw ideas")
    assert "DIFFERENT ANGLES" in msg


def test_anchoring_binds_the_prompts_and_scenes_to_the_plan():
    """The measured real-run failure this closes: the plan's places were followed
    in the scene labels, but every prompt described a different aspect with
    different nouns (within-scene affinity 0.06). The pipeline already had the
    fix -- canonical place/figure anchors -- used at freeze by the workbench and
    never by the store draft path."""
    from story import anchor_plan

    shots = _shots(COVERAGE)                    # two held scenes, content matches
    anchored = anchor_plan(shots, PLAN)
    assert len(anchored) == 6
    # the plan's two scenes are the film's scenes
    assert {s.scene for s in anchored} == {"plan1", "plan2"}
    # every prompt carries its scene's place words, and has a figure
    assert all("rain-soaked footbridge" in s.prompt
               for s in anchored[:3])
    assert all("cracked mirror" in s.prompt for s in anchored[3:])
    # the place leads but the figure is not duplicated at the front, and the
    # light is a tail (squid failure: a repeated front block drew the same image)
    assert anchored[0].prompt.startswith("a rain-soaked footbridge")
    assert "a woman in a dark coat" in anchored[0].prompt
    assert all(s.subject for s in anchored)
    # and the measured coverage now holds
    raw = {"logline": "x", "shots": [story.shot_dict(s) for s in anchored]}
    _checked, problems = story.validate(raw)
    assert [p for p in problems if p.startswith("coverage:")] == []


def _fake_planner(replies):
    def chat(**kwargs):
        reply = replies.pop(0)
        return {"message": {"content": json.dumps(reply)}, "eval_count": 40}
    return chat


def test_write_scene_plan_retries_an_unusable_plan(monkeypatch):
    import ollama

    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        if kwargs.get("format") is SCENE_PLAN_SCHEMA:
            if len(calls) == 1:
                return {"message": {"content": json.dumps({"scenes": []})},
                        "eval_count": 10}
            return {"message": {"content": json.dumps(PLAN)}, "eval_count": 40}
        return {"message": {"content": "{}"}, "eval_count": 1}

    monkeypatch.setattr(ollama, "chat", chat)
    plan, info = write_scene_plan("x", 6, "- an idea", showrunner="stub",
                                  verbose=False)
    assert plan is not None and len(plan["scenes"]) == 2
    assert info["attempts"] == 2 and info["plan"] == plan


def test_no_usable_plan_means_the_writer_proceeds_without_one(monkeypatch):
    import ollama

    monkeypatch.setattr(ollama, "chat",
                        _fake_planner([{"scenes": []}, {"scenes": []}]))
    plan, info = write_scene_plan("x", 6, "- an idea", showrunner="stub",
                                  verbose=False)
    assert plan is None and info["error"]


# --------------------------------------------------------------------------- #
# the loop: one coverage repair, then adopt the best
# --------------------------------------------------------------------------- #

def _pool() -> list:
    return [{"model": "stub-farm", "angle": "places",
             "ideas": ["a wet footbridge railing", "a cracked mirror in a room"],
             "seconds": 0.0}]


def _writer(calls, first, second):
    def chat(**kwargs):
        calls.append(kwargs)
        fmt = kwargs.get("format")
        if fmt is SCENE_PLAN_SCHEMA:
            return {"message": {"content": json.dumps(PLAN)}, "eval_count": 40}
        return {"message": {"content": json.dumps(first if len(
            [c for c in calls if c.get("format") is SHOT_SCHEMA]) == 1
            else second)}, "eval_count": 60}
    return chat


def test_the_loop_re_asks_once_when_the_story_holds_but_no_scene_does(monkeypatch):
    import ollama

    calls: list = []
    monkeypatch.setattr(ollama, "chat", _writer(calls, MONTAGE, COVERAGE))
    p = write_shots_ensemble(
        "a woman on a bridge", farms=[], farm_out=_pool(), showrunner="stub",
        n_shots=6, story_retries=0, coverage_retries=1, scene_plan=True,
        fresh=True, use_cache=False, save_cache=False, verbose=False,
        num_predict=900)

    assert p.coverage_repairs == 1
    assert p.attempts == 2
    assert p.scene_plan is not None and len(p.scene_plan["scenes"]) == 2
    # the plan is bound into the final list: two planned scenes, anchored prompts
    assert p.anchored is True
    assert {s.scene for s in p.shots} == {"plan1", "plan2"}
    assert all("rain-soaked footbridge" in s.prompt for s in p.shots[:3])
    # coverage is fixed by the anchors; the writer's raw weakness is still recorded
    assert not [x for x in p.problems if x.startswith("coverage:")]
    assert p.raw["loop"]["coverage_seen"] >= 1
    # prompts that already opened with the planned place are not rewritten; the
    # anchor still binds the rest (place in front, light at the tail)
    assert p.raw["loop"]["anchors"]["bound"] >= 4
    # both shot-writing messages carried the planned scenes, and the second
    # carried the coverage complaint + the previous list
    shot_msgs = [c["messages"][1]["content"] for c in calls
                 if c.get("format") is SHOT_SCHEMA]
    assert len(shot_msgs) == 2
    assert "THE SCENES" in shot_msgs[0]
    assert "coverage:" in shot_msgs[1] and "MONTAGE" in shot_msgs[1]
    assert p.raw["loop"]["coverage_repairs"] == 1


def test_without_the_coverage_demand_the_montage_is_kept(monkeypatch):
    import ollama

    calls: list = []
    monkeypatch.setattr(ollama, "chat", _writer(calls, MONTAGE, COVERAGE))
    p = write_shots_ensemble(
        "a woman on a bridge", farms=[], farm_out=_pool(), showrunner="stub",
        n_shots=6, story_retries=0, coverage_retries=0, scene_plan=False,
        fresh=True, use_cache=False, save_cache=False, verbose=False,
        num_predict=900)
    assert p.coverage_repairs == 0
    assert p.attempts == 1
    assert [x for x in p.problems if x.startswith("coverage:")]
    # only one SHOT_SCHEMA call: no plan and no coverage repair
    assert len([c for c in calls if c.get("format") is SHOT_SCHEMA]) == 1
