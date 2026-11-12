"""Audit #11: the ensemble proposal cache must know what it was written for, and a
cache hit must keep the loop's evidence."""
import json
import re

import pytest

import story

FARMS = ["farm-a", "farm-b"]
SHOWRUNNER = "runner:1b"
CONCEPT = "a keeper rows to a second light"


def _raw(**extra):
    shots = [
        {"prompt": "a lone lighthouse stands in a stormy night with waves crashing",
         "scale": "wide", "camera": "none", "seconds": 2.5, "subject": "keeper",
         "scene": "plan1", "energy": 0.2, "beat": "establish", "purpose": "establish",
         "link": "open", "movement": "m1", "palette": "cold-steel"},
        {"prompt": "the keeper rows a small boat across dark water under one lamp",
         "scale": "medium", "camera": "none", "seconds": 2.5, "subject": "keeper",
         "scene": "plan2", "energy": 0.3, "beat": "develop", "purpose": "master",
         "link": "scene-cut", "movement": "m2", "palette": "warm-gold"},
        {"prompt": "the keeper steps onto a rocky shore and lifts the lantern high",
         "scale": "close", "camera": "none", "seconds": 2.5, "subject": "keeper",
         "scene": "plan2", "energy": 0.3, "beat": "turn", "purpose": "insert",
         "link": "match-action", "movement": "m2", "palette": "warm-gold"},
        {"prompt": "dawn light on still water, the lighthouse small and quiet beyond",
         "scale": "wide", "camera": "none", "seconds": 2.5, "subject": "keeper",
         "scene": "plan1", "energy": 0.1, "beat": "resolve", "purpose": "resolve",
         "link": "scene-cut", "movement": "m3", "palette": "cold-steel"},
    ]
    raw = {"logline": "a keeper meets his own light", "shots": shots,
           "ideas": [{"model": m, "ideas": []} for m in FARMS],
           "scene_plan": {"scenes": [{"place": "a", "figure": "k", "action": "x",
                                      "light": "l", "palette": "neutral", "shots": 4}]},
           "loop": {"attempts": 2, "requested": 4,
                    "extension": {"calls": [1, 2]},
                    "prompt_pass": {"rewritten": 3},
                    "coverage_repairs": 1, "prior_fed_back": False,
                    "anchors": {"bound": 4}}}
    raw.update(extra)
    return raw


class _FarmCalled(Exception):
    pass


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(story, "ROOT", tmp_path)
    monkeypatch.setattr(story, "farm_ideas",
                        lambda *a, **k: (_ for _ in ()).throw(_FarmCalled()))
    d = tmp_path / "output" / "film" / "proposals"
    d.mkdir(parents=True)
    slug = re.sub(r"\W+", "-", CONCEPT.lower()).strip("-")[:60]
    path = d / (f"{slug}__ensemble-{SHOWRUNNER.replace(':', '-')}"
                f"-{len(FARMS)}farms__n4.json")
    return path


def _ask(**kw):
    args = dict(concept=CONCEPT, farms=FARMS, showrunner=SHOWRUNNER, n_shots=4,
                verbose=False, scene_plan=True, prompt_pass=True, extend=True)
    args.update(kw)
    return story.write_shots_ensemble(**args)


def _config(**over):
    cfg = {"farms": sorted(FARMS), "extend": True, "prompt_pass": True,
           "scene_plan": True}
    cfg.update(over)
    return cfg


def test_a_matching_hit_keeps_the_loops_evidence(cache):
    cache.write_text(json.dumps(_raw(_cache_config=_config())), encoding="utf-8")
    prop = _ask()
    assert prop.cached is True
    assert prop.attempts == 2 and prop.requested == 4
    assert prop.extended is True and prop.extend_calls == 2
    assert prop.prompts_rewritten == 3 and prop.coverage_repairs == 1
    assert prop.anchored is True and prop.scene_plan


def test_a_different_configuration_does_not_replay_the_entry(cache):
    cache.write_text(json.dumps(_raw(_cache_config=_config(scene_plan=False))),
                     encoding="utf-8")
    with pytest.raises(_FarmCalled):             # cache skipped -> it went to the farms
        _ask()


def test_a_legacy_entry_with_a_different_farm_set_is_not_replayed(cache):
    raw = _raw()
    raw["ideas"] = [{"model": "someone-else", "ideas": []},
                    {"model": "farm-b", "ideas": []}]
    cache.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(_FarmCalled):
        _ask()


def test_a_legacy_entry_that_matches_is_still_served(cache):
    """Discarding every pre-existing proposal would re-run 10-minute drafts."""
    cache.write_text(json.dumps(_raw()), encoding="utf-8")
    assert _ask().cached is True


def test_a_legacy_entry_without_the_demanded_scene_plan_is_not_replayed(cache):
    raw = _raw()
    raw["scene_plan"] = None
    cache.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(_FarmCalled):
        _ask()


def test_draft_reports_a_showrunner_failure_instead_of_a_traceback(
        tmp_path, monkeypatch, capsys):
    import draft
    import film_store as fs
    monkeypatch.setattr(fs, "PRODUCTIONS", tmp_path)
    fs.save(fs.new_project("failcase", "x"))

    def boom(*a, **k):
        raise story.ShowrunnerFailure(
            "the showrunner produced nothing usable (attempt 1: bad json)",
            [{"attempt": 1, "failure": "bad json"}])

    monkeypatch.setattr(story, "write_shots_ensemble", boom)
    monkeypatch.setattr("sys.argv", ["draft.py", "--store", "failcase",
                                     "--concept", "a test film"])
    assert draft.main() == 1
    out = capsys.readouterr().out
    assert "REFUSED" in out and "unchanged" in out and "bad json" in out


def test_draft_progress_hooks_match_the_lines_the_pipeline_really_prints():
    """Two hooks were dead: the pattern skipped a field draft.py prints, and the
    other named text story.py never emits. These lines are copied from real job logs
    (output/jobs/*-draft/log.txt)."""
    import job_runner as jr

    prog = {}
    jr._draft_progress("[draft] proposal validated: 10 shots, 0 story break(s), "
                       "6 structure advisory(ies), 11 warning(s)", prog)
    assert prog["phase"] == "validated"
    assert prog["shots"] == 10 and prog["story_breaks"] == 0

    prog = {}
    jr._draft_progress("[draft] proposal validated: 6 shots, 1 story break(s), "
                       "0 structure advisory(ies), 3 warning(s)", prog)
    assert prog["shots"] == 6 and prog["story_breaks"] == 1

    prog = {}
    jr._draft_progress("  [story] story and coverage complete -- stopping the "
                       "re-roll (advisories kept: 6 structure, 2 link)", prog)
    assert prog["phase"] == "validated"
    assert "adopting" in prog["last"]

