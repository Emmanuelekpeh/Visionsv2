"""The reply parser is the safety net under every local model. These tests pin
what it may and may not rewrite.

Why this file exists: the repair rules were added because one model was reported
to emit malformed JSON, and a repair path that silently corrupts a valid reply is
worse than the malformed replies were. So the contract is:

  * a VALID reply is parsed verbatim, never "repaired";
  * a reply missing only separators can be recovered (`repaired`);
  * a TRUNCATED reply is reported as truncated, never patched into a plausible
    but wrong object -- truncation is a token-budget problem with a different fix.
"""

from __future__ import annotations

import json

import pytest

from edit_planner import Shot
from story import (_repair_note, _showrunner_user, json_failure, json_from_text,
                   parse_reply, predict_budget)

VALID = {
    "logline": "a woman follows a figure and finds it is her reflection",
    "shots": [
        {"prompt": "moonlit alley, a lone woman, wet cobbles, fog, filmic grain",
         "scale": "wide", "camera": "none", "seconds": 2.0, "scene": "s1",
         "energy": 0.1, "beat": "establish"},
        {"prompt": "a mirror in a dim room, two identical silhouettes, one eye lit",
         "scale": "close", "camera": "zoom-in", "seconds": 1.6, "scene": "s1",
         "energy": 0.4, "beat": "turn"},
    ],
}
VALID_TEXT = json.dumps(VALID)


def test_valid_json_is_parsed_raw_and_unchanged():
    value, mode = parse_reply(VALID_TEXT)
    assert mode == "raw"
    assert value == VALID
    assert json_from_text(VALID_TEXT) == VALID


def test_fenced_json_is_accepted():
    text = "```json\n" + VALID_TEXT + "\n```"
    value, mode = parse_reply(text)
    assert mode == "fenced"
    assert value == VALID


def test_prose_around_the_object_is_accepted():
    value, mode = parse_reply("Sure, here is the shot list:\n" + VALID_TEXT + "\n")
    assert mode == "braces"
    assert value == VALID


def test_missing_comma_between_string_value_and_key_is_repaired():
    broken = VALID_TEXT.replace('grain", "scale"', 'grain" "scale"')
    value, mode = parse_reply(broken)
    assert mode == "repaired"
    assert value == VALID


def test_missing_comma_after_number_is_repaired():
    broken = VALID_TEXT.replace('"seconds": 2.0,', '"seconds": 2.0')
    value, mode = parse_reply(broken)
    assert mode == "repaired"
    assert value == VALID


def test_missing_comma_after_nested_object_is_repaired():
    broken = VALID_TEXT.replace('}, {"prompt"', '} {"prompt"')
    value, mode = parse_reply(broken)
    assert mode == "repaired"
    assert value == VALID


def test_trailing_comma_is_removed():
    broken = VALID_TEXT.replace('"beat": "turn"}', '"beat": "turn"},')
    value, mode = parse_reply(broken)
    assert mode == "repaired"
    assert value == VALID


def test_truncated_reply_is_reported_not_patched():
    text = VALID_TEXT[: len(VALID_TEXT) // 2]
    value, mode = parse_reply(text)
    assert value is None and mode == ""
    why = json_failure(text)
    assert "TRUNCATED" in why
    with pytest.raises(ValueError, match="no parseable JSON"):
        json_from_text(text)


def test_empty_reply_raises_the_original_error():
    with pytest.raises(ValueError, match="empty response"):
        json_from_text("   ")


def test_repair_only_fires_on_invalid_input():
    """A valid reply must be returned byte-identical, not normalised.

    The repair regexes rewrite whitespace and separators; if they ran before the
    strict parse, a reply could be changed even though it was already valid.
    """
    spaced = '{\n  "logline": "x",\n  "shots": []\n}'
    value, mode = parse_reply(spaced)
    assert mode == "raw"
    assert list(value.keys()) == ["logline", "shots"]


def test_predict_budget_scales_with_the_shot_count():
    """The truncation that was reported as 'malformed JSON' must not recur.

    Measured: 16 shots needs ~1790 tokens; a flat 1200 cut granite off mid-object.
    The budget is a cap, so being generous costs nothing when the model finishes.
    """
    assert predict_budget(None) == 2000            # ~6 shots + logline
    assert predict_budget(6) == 2000
    assert predict_budget(16) > 2000               # must cover 16 shots
    assert predict_budget(16) >= 2800
    assert predict_budget(24) > predict_budget(16)
    assert predict_budget(1000) == 6000            # bounded


def _beats() -> list:
    return [Shot("a moonlit alley with one figure", beat="establish"),
            Shot("a cracked mirror reflecting a second figure", beat="turn"),
            Shot("two identical silhouettes in the fog", beat="resolve")]


def test_repair_note_names_the_failure_and_the_previous_beats():
    """A repair retry must carry the validator's words, not re-roll blind."""
    problems = ["story: beats go backwards (3:resolve->develop)",
                "structure: R8 scene 's1' uses one framing scale (wide)",
                "story-link: the resolve shares almost no imagery with the turn"]
    note = _repair_note(problems, _beats())
    for p in problems:
        assert p in note
    assert "[establish]" in note            # the beats it actually produced
    assert "[turn]" in note
    assert "never backwards" in note
    # it can keep imagery because it is shown it
    assert "cracked mirror" in note


def test_repair_note_is_empty_without_a_story_failure():
    """A re-roll for a jump cut must not be told a story problem exists."""
    assert _repair_note(["jump cut: shots 2,3 both medium"], _beats()) == ""


def test_showrunner_user_carries_the_repair_context():
    base = _showrunner_user("a concept", 6, "- [farm] an idea")
    assert "an idea" in base and "Write exactly 6 shots." in base
    assert "REJECTED" not in base
    repaired = _showrunner_user("a concept", 6, "- [farm] an idea",
                               _repair_note(["story: no turn"], _beats()))
    assert "REJECTED BY THE STORY/STRUCTURE CHECK" in repaired
