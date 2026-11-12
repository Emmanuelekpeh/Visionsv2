"""The story gate at adoption: a broken beat order must be a DECISION, not a default.

Measured motivation: 9 of the 11 productions in `productions/` turned out to have
non-monotonic beats, and several shipped with `story:` problems in their own
record. The validator caught the failure and `draft.py` adopted it anyway. The
gate is the last point where the pipeline can refuse.
"""

from __future__ import annotations

import json

from draft import story_gate


def _shot(prompt: str, scale: str, beat: str, scene: str = "s1") -> dict:
    return {"prompt": prompt, "scale": scale, "camera": "none", "seconds": 2.0,
            "scene": scene, "energy": 0.2, "beat": beat}


BAD = {"logline": "a mood board", "shots": [
    _shot("a concrete image one", "wide", "establish"),
    _shot("a concrete image two", "medium", "establish"),
    _shot("a concrete image three", "close", "resolve", "s2"),
]}
GOOD = {"logline": "a film that develops", "shots": [
    _shot("a concrete image one", "wide", "establish"),
    _shot("a concrete image two", "medium", "develop"),
    _shot("a concrete image three", "close", "turn", "s2"),
    _shot("a concrete image four", "wide", "resolve", "s2"),
]}


def test_breaks_block_adoption_by_default():
    allowed, breaks, warns = story_gate(["story: no 'turn'"])
    assert allowed is False
    assert breaks == ["story: no 'turn'"]


def test_an_explicit_override_adopts_anyway():
    allowed, breaks, warns = story_gate(["story: no 'turn'"],
                                        allow_broken_story=True)
    assert allowed is True
    assert breaks == ["story: no 'turn'"]


def test_warnings_never_block():
    """A lexical proxy cannot tell 'linked but worded differently' from
    'unrelated', so the warning stays advisory -- the by-eye read is the arbiter.
    Rhythm advisories are advisory too: the planner owns hold timing."""
    link = "story-link: the resolve shares almost no imagery with the turn"
    rhythm = "rhythm: R9 shots 1-3 hold within 0.1s of each other"
    allowed, breaks, warns = story_gate([link, rhythm])
    assert allowed is True
    assert breaks == []
    assert warns == [link, rhythm]


def test_structure_is_advisory_not_blocking():
    """A montage is a legitimate arrangement; the gate must not refuse it.

    The A/B (SEQUENCE_AB.md) and the organisation layer both say a rapid-change
    montage and a held coverage scene are valid. Structure problems are surfaced
    for the ORGANISE stage; only a broken STORY blocks adoption.
    """
    structure = ("structure: R7 4 scene cuts -- at most 2 hard stops to a new "
                 "place")
    allowed, breaks, warns = story_gate([structure])
    assert allowed is True
    assert breaks == []
    assert warns == [structure]


def test_clean_proposals_pass():
    allowed, breaks, warns = story_gate([])
    assert (allowed, breaks, warns) == (True, [], [])


def test_ensemble_retry_sends_the_repair_note_not_a_blind_reroll(monkeypatch):
    """The retry must carry the validator's complaint and the previous beats.

    End-to-end through `write_shots_ensemble` with a stubbed model: the first
    reply is a mood board (two establishes, no turn), the second is a real arc.
    The assertion that matters is the SECOND message: it has to name the failure.
    """
    import ollama
    import story

    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        reply = BAD if len(calls) == 1 else GOOD
        return {"message": {"content": json.dumps(reply)}, "eval_count": 50}

    monkeypatch.setattr(ollama, "chat", fake_chat)
    pool = [{"model": "stub-farm", "angle": "places",
             "ideas": ["a wet lighthouse stair", "a light out at sea"],
             "seconds": 0.0}]
    p = story.write_shots_ensemble(
        "a lighthouse keeper sees a second light", farms=[], farm_out=pool,
        showrunner="stub-model", n_shots=4, story_retries=1, fresh=True,
        use_cache=False, save_cache=False, verbose=False, num_predict=900,
        repair_prompt=True)

    assert len(calls) == 2, "a broken story must trigger exactly one retry"
    first = calls[0]["messages"][1]["content"]
    second = calls[1]["messages"][1]["content"]
    assert "REJECTED BY THE STORY/STRUCTURE CHECK" not in first
    assert "REJECTED BY THE STORY/STRUCTURE CHECK" in second
    assert "2 establish" in second          # the validator's exact words
    assert "[establish]" in second          # the arrangement it actually produced
    assert story.story_shape(p.shots) == "EdTR"
    assert p.attempts == 2
    assert not any(x.startswith("story:") for x in p.problems)
