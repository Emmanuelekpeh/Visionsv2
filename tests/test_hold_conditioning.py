"""Hold conditioning is part of the take's identity.

A shot rendered from a reference frame is not the same pixels as the same prompt
rendered alone, so the store's staleness hash must include which anchor (and which
anchor take) the conditioning came from. Without this, re-rendering a hold was
silently skipped because the recorded inputs said "current".
"""

from __future__ import annotations

import copy

import film_store as fs


def _project() -> dict:
    shots = [
        {"id": "s01", "prompt": "a woman on a wet bridge, rain", "scale": "wide",
         "camera": "none", "subject": "woman", "scene": "sc1", "take": "t01",
         "seconds": 2.5, "energy": 0.2, "beat": "establish"},
        {"id": "s02", "prompt": "the same woman walking away, rain", "scale": "medium",
         "camera": "none", "subject": "woman", "scene": "sc1", "take": "t01",
         "seconds": 1.7, "energy": 0.2, "beat": "develop"},
        {"id": "s03", "prompt": "a new alley at night", "scale": "wide",
         "camera": "none", "subject": "", "scene": "sc2", "take": "t01",
         "seconds": 2.0, "energy": 0.2, "beat": "resolve"},
    ]
    return {"slug": "holdcond", "settings": {"frames": 16, "fps": 16},
            "timeline": shots, "scenes": [], "problems": []}


def test_hold_reference_names_the_anchor_shot_and_take(monkeypatch):
    p = _project()
    shots = p["timeline"]
    monkeypatch.setattr(fs, "shots_in_order", lambda project: shots)
    first = fs.generation_inputs(p, shots[0], seed=7)
    second = fs.generation_inputs(p, shots[1], seed=7)
    change = fs.generation_inputs(p, shots[2], seed=7)
    assert first["hold_reference"] == ""          # the anchor itself is not conditioned
    assert second["hold_reference"] == "s01:t01"  # anchored on shot 1's chosen take
    assert change["hold_reference"] == ""         # a rapid change is never conditioned


def test_generation_hash_changes_when_the_anchor_take_changes(monkeypatch):
    p = _project()
    shots = p["timeline"]
    monkeypatch.setattr(fs, "shots_in_order", lambda project: project["timeline"])
    h1 = fs.input_hash(fs.generation_inputs(p, shots[1], seed=7))
    p["timeline"][0]["take"] = "t02"              # anchor re-shot
    after = fs.generation_inputs(p, p["timeline"][1], seed=7)
    assert after["hold_reference"] == "s01:t02"
    assert fs.input_hash(after) != h1


def test_hold_reference_is_tolerant_for_legacy_takes():
    """A take recorded before this key existed must not all become stale."""
    assert "hold_reference" in fs.ADDED_GEN_KEYS
