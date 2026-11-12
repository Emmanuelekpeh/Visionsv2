"""The prompt pass: one image per shot, without touching the film.

Measured motivation -- the first real rendered film (`ego`, 13 shots). The story
gate accepted it, the cut was assembled, and then the audit found that 12 of 13
prompts were two-image composites:

    A barefoot man walks across a dusty city floor, a cracked concrete patch
    reflecting his unsettled gaze; the shadow of his ego looms large against a
    moonlit forest, leaves rustling around him as the confrontation begins.

The renderer drew one clause (or the concept prior), and the film ended on a blue
car because the resolve prompt named "an old car" and not the protagonist. CLIP
prompt->frame margin over a different shot's prompt was +0.018, and four frames
matched another shot's prompt better than their own.

So: `prompt:` is an advisory (reported, fed to the repair brief), and
`rewrite_prompts` is an explicit TEXT pass on the approved list -- same shots,
same beats, same order, only the words changed, and a per-shot best-of keeps the
original when the rewrite is not measurably better.
"""

from __future__ import annotations

import json

import story
from story import (PROMPT_SCHEMA, _one_image_text, _prompt_issues, _repair_note,
                   prompt_problems, rewrite_prompts, shots_from_dicts,
                   story_shape, validate, write_shots_ensemble)


def _shot(prompt, scale, beat, scene, purpose, link, subject="woman",
          seconds=2.0, energy=0.2) -> dict:
    return {"prompt": prompt, "scale": scale, "camera": "none",
            "seconds": seconds, "subject": subject, "scene": scene,
            "energy": energy, "beat": beat, "purpose": purpose, "link": link}


CLEAN_PROMPTS = [
    "a rain-soaked footbridge at night, a woman in a dark coat on the planks, "
    "streetlights and fog",
    "the same woman walking along the footbridge seen from behind, rain and fog",
    "a hand tightening on the wet iron railing, rain beading on the metal",
    "a second figure under the far streetlight at the end of the bridge, fog",
    "the figure stepping out of the fog toward the woman, streetlight behind it",
    "the two figures facing each other on the rain-soaked bridge, fog between them",
]


def _clean() -> dict:
    return {"logline": "a woman on a bridge meets her own shadow", "shots": [
        _shot(CLEAN_PROMPTS[0], "wide", "establish", "s1", "establish", "open"),
        _shot(CLEAN_PROMPTS[1], "medium", "develop", "s1", "master",
              "match-action"),
        _shot(CLEAN_PROMPTS[2], "close", "develop", "s1", "insert",
              "match-action"),
        _shot(CLEAN_PROMPTS[3], "wide", "develop", "s2", "establish",
              "scene-cut", subject="figure"),
        _shot(CLEAN_PROMPTS[4], "medium", "turn", "s2", "reaction",
              "match-action", subject="figure", energy=0.45),
        _shot(CLEAN_PROMPTS[5], "close", "resolve", "s2", "resolve", "eyeline",
              subject="figure", energy=0.15),
    ]}


def _messy() -> dict:
    """The same film, written the way the real writer wrote it: image; meaning."""
    raw = _clean()
    tails = [
        "; the shadow of her ego looms large as the confrontation begins",
        "; the confrontation intensifies under the symbolic moonlight",
        "; the ego's hold tightens, the aftermath of awareness settling",
        "; the confrontation reaches its climax, tension intensifying",
        "; the ego surrenders, the shadow fully embodying her inner turmoil",
        "; symbolic of the ego's acceptance and the new self-awareness",
    ]
    for shot, tail in zip(raw["shots"], tails):
        shot["prompt"] = shot["prompt"] + tail
    return raw


# --------------------------------------------------------------------------- #
# the check: a prompt is an image, not a sentence about an image
# --------------------------------------------------------------------------- #

def test_a_single_concrete_image_passes():
    for p in CLEAN_PROMPTS:
        assert _prompt_issues(p) == [], p


def test_a_palette_name_in_the_prompt_is_a_grade_leak():
    """Measured on the squid draft: the plan brief showed 'palette: warm-gold' and
    the prompt pass copied 'warm-gold palette' into 10/10 prompts. The grade is
    applied AFTER generation, so the name must never appear in the prompt."""
    issues = _prompt_issues("a diver's helmet in glowing coral light, "
                            "rich-green palette, faint moonlight")
    assert any("grade palette" in i for i in issues)
    assert _prompt_issues("a diver's helmet in rich green coral light") == []


def test_a_fused_or_interpreted_prompt_is_flagged():
    fused = ("a barefoot man walks across a dusty city floor; the shadow of his "
             "ego looms large against a moonlit forest as the confrontation "
             "begins")
    issues = _prompt_issues(fused)
    assert any("more than one image" in i for i in issues)
    assert any("interprets" in i for i in issues)
    # the report names the shot and the reason, never a silent rewrite
    probs = prompt_problems(shots_from_dicts(_messy()["shots"]))
    assert len(probs) == 6
    assert all(p.startswith("prompt: shot ") for p in probs)


def test_validate_carries_the_prompt_problem():
    shots, problems = validate(_messy())
    assert len(shots) == 6
    assert len([p for p in problems if p.startswith("prompt:")]) == 6
    # and it is a text problem, not a story problem: the gate must still see none
    assert [p for p in problems if p.startswith("story:")] == []


def test_prompt_problems_do_not_block_adoption_but_are_reported():
    from draft import story_gate

    allowed, breaks, warns = story_gate(
        ["prompt: shot 13 fuses more than one image into one prompt"])
    assert allowed is True
    assert breaks == []
    assert warns == ["prompt: shot 13 fuses more than one image into one prompt"]


def test_the_repair_brief_carries_prompt_complaints():
    note = _repair_note(["prompt: shot 4 fuses more than one image into one prompt"],
                        shots_from_dicts(_messy()["shots"]))
    assert "prompt: shot 4" in note
    assert "ONE concrete image" in note


# --------------------------------------------------------------------------- #
# the pass: same film, better text
# --------------------------------------------------------------------------- #

def _fake_chat(replies, calls=None):
    def chat(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"message": {"content": json.dumps(reply)}, "eval_count": 40}

    return chat


def test_rewrite_prompts_replaces_composites_and_keeps_the_film(monkeypatch):
    import ollama

    calls: list = []
    monkeypatch.setattr(ollama, "chat",
                        _fake_chat([{"prompts": CLEAN_PROMPTS}], calls))
    shots = shots_from_dicts(_messy()["shots"])
    out, info = rewrite_prompts(shots, showrunner="stub", verbose=False)

    assert info["rewritten"] == 6 and info["kept"] == 0
    assert [s.prompt for s in out] == CLEAN_PROMPTS
    # everything that is NOT the text is the approved value
    for before, after in zip(shots, out):
        assert (after.beat, after.scene, after.scale, after.purpose, after.link,
                after.seconds) == (before.beat, before.scene, before.scale,
                                   before.purpose, before.link, before.seconds)
    assert story_shape(out) == story_shape(shots)
    assert calls and calls[0]["format"] is PROMPT_SCHEMA


def test_rewrite_is_only_accepted_when_it_is_measurably_better(monkeypatch):
    """A garbage rewrite leaves the approved prompt alone (per-shot best-of)."""
    import ollama

    worse = ["a man"] * 6                       # terse, not an improvement
    monkeypatch.setattr(ollama, "chat", _fake_chat([{"prompts": worse}]))
    shots = shots_from_dicts(_messy()["shots"])
    out, info = rewrite_prompts(shots, showrunner="stub", verbose=False)
    assert info["rewritten"] == 0 and info["kept"] == 6
    assert [s.prompt for s in out] == [s.prompt for s in shots]


def test_a_count_mismatch_is_retried_then_kept_untouched(monkeypatch):
    import ollama

    monkeypatch.setattr(ollama, "chat",
                        _fake_chat([{"prompts": CLEAN_PROMPTS[:3]},
                                    {"prompts": CLEAN_PROMPTS[:3]}]))
    shots = shots_from_dicts(_messy()["shots"])
    out, info = rewrite_prompts(shots, showrunner="stub", verbose=False)
    assert out == list(shots)
    assert info["rewritten"] == 0
    assert info["attempts"] == 2
    assert "must match" in info["error"]


def test_a_count_mismatch_on_the_first_reply_is_recovered(monkeypatch):
    """Measured on a real 13-shot draft: mistral returned 12 prompts for 13 shots
    and the whole pass was silently skipped. One retry recovers it."""
    import ollama

    monkeypatch.setattr(ollama, "chat",
                        _fake_chat([{"prompts": CLEAN_PROMPTS[:3]},
                                    {"prompts": CLEAN_PROMPTS}]))
    shots = shots_from_dicts(_messy()["shots"])
    out, info = rewrite_prompts(shots, showrunner="stub", verbose=False)
    assert info["attempts"] == 2
    assert info["rewritten"] == 6
    assert [s.prompt for s in out] == CLEAN_PROMPTS


def test_a_failed_pass_never_kills_the_draft(monkeypatch):
    import ollama

    monkeypatch.setattr(ollama, "chat",
                        _fake_chat([RuntimeError("ollama died"),
                                    RuntimeError("ollama died")]))
    shots = shots_from_dicts(_messy()["shots"])
    out, info = rewrite_prompts(shots, showrunner="stub", verbose=False)
    assert out == list(shots)
    assert info["attempts"] == 2
    assert "ollama died" in info["error"]


def test_the_image_clause_is_kept_when_the_model_only_cleans_words():
    """Measured: told 'no semicolon', granite still returns 'image; meaning'.

    The cut is on the REWRITE only, and it is what makes the pass useful even
    when the model only trims adjectives: the first clause is the frame.
    """
    assert _one_image_text("a man on a bridge; the ego surrenders") == (
        "a man on a bridge")
    assert _one_image_text("one image here. a second meaning follows") == (
        "one image here.")
    assert _one_image_text("no separator at all, just an image, honestly") == (
        "no separator at all, just an image, honestly")


def test_an_echoed_composite_still_gets_its_meaning_clause_dropped(monkeypatch):
    import ollama

    shots = shots_from_dicts(_messy()["shots"])
    echo = [s.prompt for s in shots]            # a useless model: no change
    monkeypatch.setattr(ollama, "chat", _fake_chat([{"prompts": echo}]))
    out, info = rewrite_prompts(shots, showrunner="stub", verbose=False)
    assert info["rewritten"] == 6
    before_semis = sum(1 for s in shots if ";" in s.prompt)
    after_semis = sum(1 for s in out if ";" in s.prompt)
    assert after_semis < before_semis
    # and every accepted prompt is still a real image, not a fragment
    for s in out:
        if _prompt_issues(s.prompt) == []:
            assert len(s.prompt.split()) >= 8


def test_a_rewrite_that_leaks_field_values_is_rejected():
    """A model that fuses 'camera: none' or '2.0 seconds' into the prompt has
    made the image worse, whatever the sentence shape says."""
    from story import _better_prompt

    old = CLEAN_PROMPTS[0]
    leaky = old + ", 2.0 seconds, camera: none"
    text, changed = _better_prompt(old, leaky)
    assert changed is False
    assert text == old


# --------------------------------------------------------------------------- #
# the ceiling: a hold is the same place from DIFFERENT angles
# --------------------------------------------------------------------------- #

def _vshot(prompt, scene, beat="develop", purpose="master",
           link="match-action", subject="woman") -> dict:
    return {"prompt": prompt, "scale": "medium", "camera": "none",
            "seconds": 2.0, "subject": subject, "scene": scene, "energy": 0.2,
            "beat": beat, "purpose": purpose, "link": link}


def test_the_same_sentence_twice_inside_a_scene_is_flagged():
    from story import prompt_variation_problems

    repeated = [
        _vshot("a rain-soaked footbridge at night, a woman in a dark coat, fog",
               "s1", beat="establish", purpose="establish", link="open"),
        _vshot("a rain-soaked footbridge at night, a woman in a dark coat, fog",
               "s1"),
    ]
    shots = shots_from_dicts(repeated)
    probs = prompt_variation_problems(shots)
    assert probs and probs[0].startswith("prompt:")
    assert "repeats" in probs[0]


def test_a_hold_from_different_angles_passes():
    from story import prompt_variation_problems

    varied = [
        _vshot("a rain-soaked footbridge at night, a woman in a dark coat, fog",
               "s1", beat="establish", purpose="establish", link="open"),
        _vshot("the woman's hand on the wet iron railing of the footbridge, rain",
               "s1", purpose="insert"),
        _vshot("the same woman walking away along the footbridge planks, fog",
               "s1", purpose="reaction"),
    ]
    assert prompt_variation_problems(shots_from_dicts(varied)) == []


def test_validate_carries_the_variation_ceiling():
    repeated = [
        _vshot("a rain-soaked footbridge at night, a woman in a dark coat, fog",
               "s1", beat="establish", purpose="establish", link="open"),
        _vshot("a rain-soaked footbridge at night, a woman in a dark coat, fog",
               "s1"),
        _vshot("a second figure on the far side of the bridge, fog", "s1",
               beat="resolve", purpose="resolve", link="eyeline",
               subject="figure"),
    ]
    _shots, problems = validate({"logline": "x", "shots": repeated})
    assert any(p.startswith("prompt:") and "repeats" in p for p in problems)


# --------------------------------------------------------------------------- #
# wired into the writing loop, off by default
# --------------------------------------------------------------------------- #

def _writer_with_prompt_pass(monkeypatch, calls, prompt_reply):
    import ollama

    def chat(**kwargs):
        calls.append(kwargs)
        if kwargs.get("format") is PROMPT_SCHEMA:
            return {"message": {"content": json.dumps(prompt_reply)},
                    "eval_count": 40}
        return {"message": {"content": json.dumps(_messy())}, "eval_count": 60}

    monkeypatch.setattr(ollama, "chat", chat)


def _pool() -> list:
    return [{"model": "stub-farm", "angle": "places",
             "ideas": ["a rain-soaked footbridge railing", "a streetlight in fog"],
             "seconds": 0.0}]


def test_the_loop_runs_the_prompt_pass_on_an_approved_list(monkeypatch):
    calls: list = []
    _writer_with_prompt_pass(monkeypatch, calls, {"prompts": CLEAN_PROMPTS})
    p = write_shots_ensemble(
        "a woman on a bridge", farms=[], farm_out=_pool(), showrunner="stub",
        n_shots=6, story_retries=0, fresh=True, use_cache=False,
        save_cache=False, verbose=False, num_predict=900,
        prompt_pass=True)

    assert p.prompts_rewritten == 6
    assert [s.prompt for s in p.shots] == CLEAN_PROMPTS
    assert not [x for x in p.problems if x.startswith("prompt:")]
    assert p.raw["loop"]["prompt_pass"]["rewritten"] == 6
    assert any(c.get("format") is PROMPT_SCHEMA for c in calls)


def test_the_prompt_pass_is_off_by_default_so_measurements_stay_honest(monkeypatch):
    calls: list = []
    _writer_with_prompt_pass(monkeypatch, calls, {"prompts": CLEAN_PROMPTS})
    p = write_shots_ensemble(
        "a woman on a bridge", farms=[], farm_out=_pool(), showrunner="stub",
        n_shots=6, story_retries=0, fresh=True, use_cache=False,
        save_cache=False, verbose=False, num_predict=900)
    assert p.prompts_rewritten == 0
    assert len([x for x in p.problems if x.startswith("prompt:")]) == 6
    assert not any(c.get("format") is PROMPT_SCHEMA for c in calls)


# --------------------------------------------------------------------------- #
# the job runner can show the pass
# --------------------------------------------------------------------------- #

def test_the_job_runner_reports_the_prompt_pass():
    import job_runner as jr

    prog: dict = {}
    jr._draft_progress("  [story] prompt pass: rewriting 13 prompt(s) as single "
                       "images (<= 3000 tokens)", prog)
    assert prog["phase"] == "prompt pass"

    prog2: dict = {}
    jr._draft_progress("  [story] prompt pass: rewrote 12/13 prompt(s) as single "
                       "images; 1 still flagged", prog2)
    assert (prog2["prompts_rewritten"], prog2["prompts_total"]) == (12, 13)
