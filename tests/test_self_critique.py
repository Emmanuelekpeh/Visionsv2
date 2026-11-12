"""The film checks its own story order, then repairs only the shots that break it.

The prompts are the measured ones from the "skydiving" drafts. The reader model
only LOCATES events (first/last shot); plain code decides what is wrong.
"""
from __future__ import annotations

import json

import ollama

import story
from edit_planner import Shot


def _s(prompt, beat="develop", scale="medium", subject="a skydiver"):
    return Shot(prompt=prompt, scale=scale, camera="none", seconds=1.5,
                subject=subject, scene="sc1", energy=0.2, beat=beat,
                purpose="develop", link="match-action")


SKY = [
    _s("a skydiver in a black flight suit leaps from the open plane door"),
    _s("a skydiver in a black flight suit free-falls through bright cloud"),
    _s("a skydiver's parachute opens above wispy clouds, golden light"),
    _s("a skydiver gliding toward a green field, soft light"),
    _s("a green field with soft grass, skydiver's silhouette vanished, warm light"),
    _s("a green field with scattered gear, no trace of the figure, golden light"),
]
EVENTS = ["leaps from the plane", "free-falls", "parachute opens", "glides down",
          "lands"]


def _scripted(spans, repairs, events=None):
    """A fake ollama.chat. The reader returns `spans` ({event no: (first, last)}) on
    its first call and "nothing located" afterwards; the batch repair returns
    `repairs` ({shot: prompt})."""
    calls = {"plan": 0, "read": 0, "repair": 0, "repair_user": []}
    names = EVENTS if events is None else events

    def fake(model=None, messages=None, format=None, options=None, **kw):
        props = (format or {}).get("properties", {})
        if "events" in props and props["events"]["items"].get("type") == "string":
            calls["plan"] += 1
            body = {"events": names}
        elif "events" in props:
            calls["read"] += 1
            sp = spans if calls["read"] == 1 else {}
            body = {"events": [{"event": n, "first_shot": sp.get(i, (0, 0))[0],
                                "last_shot": sp.get(i, (0, 0))[1]}
                               for i, n in enumerate(names, 1)]}
        else:
            calls["repair"] += 1
            calls["repair_user"].append(messages[-1]["content"])
            body = {"fixes": [{"shot": k, "prompt": v} for k, v in repairs.items()]}
        return {"message": {"content": json.dumps(body)}}
    return fake, calls


# ---- the pure order check --------------------------------------------------- #

def test_missing_event_is_placed_in_its_slot():
    # leap 1, (nothing at 2), parachute 3, glide 4, land 5: free-fall belongs at 2
    spans = {1: (1, 1), 2: (0, 0), 3: (3, 3), 4: (4, 4), 5: (5, 5)}
    flaws = story.order_flaws(EVENTS, spans, 5)
    assert list(flaws) == [2] and "MUST SHOW event 2" in flaws[2][0]


def test_an_event_with_no_free_slot_is_reported_not_guessed():
    unplaced: list = []
    spans = {1: (2, 2), 2: (0, 0), 3: (3, 3), 4: (4, 4), 5: (5, 5)}
    assert story.order_flaws(EVENTS, spans, 5, unplaced) == {} and unplaced == [2]


def test_the_plane_door_after_the_landing_goes_backwards():
    # the real draft: the leap is shown at 2 AND again at 8, after the descent began
    spans = {1: (2, 8), 2: (4, 5), 3: (3, 9), 4: (6, 6), 5: (10, 10)}
    flaws = story.order_flaws(EVENTS, spans, 15)
    assert "goes backwards" in flaws[8][0] and "goes backwards" in flaws[9][0]


def test_a_clean_film_has_no_flags():
    spans = {1: (1, 1), 2: (2, 2), 3: (3, 3), 4: (4, 4), 5: (5, 6)}
    assert story.order_flaws(EVENTS, spans, 6) == {}


def test_an_event_that_first_appears_late_is_flagged():
    # parachute (3) first shown at 2, free-fall (2) first shown at 4: backwards
    spans = {1: (1, 1), 2: (4, 4), 3: (2, 2), 4: (5, 5), 5: (6, 6)}
    flaws = story.order_flaws(EVENTS, spans, 6)
    assert any("goes backwards" in r for rs in flaws.values() for r in rs)


# ---- the rules -------------------------------------------------------------- #

def test_absence_wording_is_flagged():
    flaws = story.prompt_flaws(SKY)
    assert set(flaws) == {5, 6}
    assert "absence" in flaws[5][0]


def test_figure_missing_from_closing_run_is_flagged_but_midfilm_cutaway_is_not():
    shots = [_s("a skydiver stands at the plane door"),
             _s("a green field of soft grass under warm light"),   # cutaway
             _s("a skydiver free-falls through the clouds"),
             _s("a skydiver lands softly in the grass"),
             _s("a wide green field under golden light"),
             _s("distant horizon over the quiet grass")]
    flaws = story.prompt_flaws(shots)
    assert set(flaws) == {5, 6}
    assert 2 not in flaws


def test_figureless_film_is_not_nagged():
    shots = [_s("rain on a tin roof", subject="roof"),
             _s("a gutter overflowing in grey light", subject="roof"),
             _s("puddles spreading across cobbles", subject="roof"),
             _s("a street lamp flickering in fog", subject="roof")]
    assert story.prompt_flaws(shots) == {}


def test_strip_absence_only_removes_the_clause():
    assert story._strip_absence("a field, no trace of the figure, golden light") \
        == "a field, golden light"
    assert story._strip_absence("a calm field, golden light") == \
        "a calm field, golden light"


# ---- the reader ------------------------------------------------------------- #

def test_out_of_range_shot_numbers_become_not_shown(monkeypatch):
    fake, _ = _scripted({1: (1, 1), 2: (2, 2), 3: (99, 99), 4: (4, 4), 5: (0, 0)}, {})
    monkeypatch.setattr(ollama, "chat", fake)
    events, spans, err = story.read_events("c", SKY, "m", EVENTS)
    assert err is None and spans[3] == (0, 0) and spans[4] == (4, 4)


def test_an_empty_reading_flags_nothing(monkeypatch):
    # nothing located must NOT mean "no event is shown" (it flagged 5 good shots)
    fake, _ = _scripted({}, {})
    monkeypatch.setattr(ollama, "chat", fake)
    events, spans, err = story.read_events("c", SKY, "m", EVENTS)
    assert spans == {} and "unreadable" in err
    out, info = story.critique_and_repair("c", SKY[:4], rounds=1, verbose=False)
    assert info["flagged"] == [] and [s.prompt for s in out] == [s.prompt for s in SKY[:4]]


# ---- the loop --------------------------------------------------------------- #

def _five():
    return [_s("a skydiver leaps from the plane door into bright sky"),
            _s("white clouds drift under a bright blue sky"),
            _s("a skydiver's parachute opens above the clouds"),
            _s("a skydiver glides over a green field, soft light"),
            _s("a skydiver touches down in the green grass")]


def test_missing_event_is_repaired_in_one_call_that_sees_the_whole_list(monkeypatch):
    shots = _five()
    fake, calls = _scripted({1: (1, 1), 2: (0, 0), 3: (3, 3), 4: (4, 4), 5: (5, 5)},
                            {2: "a skydiver in free-fall spreads limbs against white clouds"})
    monkeypatch.setattr(ollama, "chat", fake)
    out, info = story.critique_and_repair("a skydiver jumps and lands", shots,
                                          rounds=1, verbose=False)
    assert calls["plan"] == 1 and calls["read"] == 1 and calls["repair"] == 1
    sent = calls["repair_user"][0]
    assert all(s.prompt in sent for s in shots)           # the whole film
    assert "MUST SHOW event 2: free-falls" in sent and "STORY EVENTS" in sent
    assert out[1].prompt.startswith("a skydiver in free-fall")
    assert [s.prompt for s in out if s is not out[1]] == \
        [s.prompt for s in shots if s is not shots[1]]
    assert info["fixed"] == 1 and info["flagged"][0]["source"] == "order"
    assert [(s.beat, s.scene, s.seconds) for s in out] == \
        [(s.beat, s.scene, s.seconds) for s in shots]


def test_a_rewrite_that_repeats_the_absence_falls_back_to_the_strip(monkeypatch):
    fake, _ = _scripted({1: (1, 1), 2: (2, 2), 3: (3, 3), 4: (4, 4), 5: (5, 6)},
                        {5: "the green field, skydiver gone from view, warm light now",
                         6: "a green field with scattered gear, no trace of anyone here"})
    monkeypatch.setattr(ollama, "chat", fake)
    out, info = story.critique_and_repair("c", SKY, rounds=1, verbose=False)
    assert out[4].prompt == "a green field with soft grass, warm light"
    assert out[5].prompt == "a green field with scattered gear, golden light"
    assert set(info["remaining"]) == {"5", "6"}       # and the ending still lacks the figure


def test_figure_complaint_needs_the_figure_in_the_rewrite(monkeypatch):
    shots = [_s("a skydiver stands at the plane door"),
             _s("a skydiver free-falls through the clouds"),
             _s("a skydiver lands softly in the grass"),
             _s("a wide green field under golden light"),
             _s("distant horizon over the quiet grass")]
    fake, _ = _scripted({}, {4: "a wide green field under pale morning light",
                             5: "a skydiver walks away across the green field, parachute trailing"})
    monkeypatch.setattr(ollama, "chat", fake)
    out, info = story.critique_and_repair("c", shots, rounds=1, verbose=False)
    assert out[3].prompt == shots[3].prompt          # no figure -> refused
    assert "skydiver" in out[4].prompt               # figure named -> accepted
    assert info["fixed"] == 1


def test_critic_failure_never_kills_the_draft(monkeypatch):
    def boom(**kw):
        raise ConnectionError("ollama is down")
    monkeypatch.setattr(ollama, "chat", boom)
    out, info = story.critique_and_repair("c", SKY, rounds=1, verbose=False)
    # with every model call failing, the rules still strip the absence wording
    assert [s.prompt for s in out[:4]] == [s.prompt for s in SKY[:4]]
    assert not any(story.ABSENCE_RE.search(s.prompt) for s in out)
    assert "ollama is down" in info["critic_error"]
    assert info["fixed"] == 2


def test_a_round_that_makes_it_worse_is_reverted(monkeypatch):
    shots = _five()
    calls = {"read": 0}

    def fake(model=None, messages=None, format=None, options=None, **kw):
        props = (format or {}).get("properties", {})
        if "events" in props and props["events"]["items"].get("type") == "string":
            return {"message": {"content": json.dumps({"events": EVENTS})}}
        if "events" in props:
            calls["read"] += 1
            # round 1: free-fall missing (1 flag). After the rewrite the reader
            # sees the leap running to the end and two events unplaceable -- worse
            sp = ({1: (1, 1), 2: (0, 0), 3: (3, 3), 4: (4, 4), 5: (5, 5)}
                  if calls["read"] == 1 else
                  {1: (1, 5), 2: (2, 2), 3: (3, 3), 4: (0, 0), 5: (0, 0)})
            return {"message": {"content": json.dumps({"events": [
                {"event": n, "first_shot": sp.get(i, (0, 0))[0],
                 "last_shot": sp.get(i, (0, 0))[1]}
                for i, n in enumerate(EVENTS, 1)]})}}
        return {"message": {"content": json.dumps({"fixes": [
            {"shot": 2, "prompt": "a skydiver in free-fall spreads limbs against clouds"}]})}}
    monkeypatch.setattr(ollama, "chat", fake)
    out, info = story.critique_and_repair("c", shots, rounds=2, verbose=False)
    assert info["reverted"] is True and info["fixed"] == 0
    assert [s.prompt for s in out] == [s.prompt for s in shots]


def test_progress_lines_are_parsed_by_the_job_runner(capsys, monkeypatch):
    import job_runner
    fake, _ = _scripted({1: (1, 1), 2: (0, 0), 3: (3, 3), 4: (4, 4), 5: (5, 5)},
                        {2: "a skydiver in free-fall spreads limbs against clouds"})
    monkeypatch.setattr(ollama, "chat", fake)
    story.critique_and_repair("c", _five(), rounds=1, verbose=True)
    prog: dict = {}
    for line in capsys.readouterr().out.splitlines():
        job_runner._draft_progress(line, prog)
    assert prog["phase"] == "critique" and prog["total"] == 1 and "2" in prog["last"]


def test_the_event_order_comes_from_the_concept_not_from_the_draft(monkeypatch):
    # the model that reads the shots is HANDED the order; it cannot adopt the draft's
    seen = {}

    def fake(model=None, messages=None, format=None, options=None, **kw):
        props = (format or {}).get("properties", {})
        if "events" in props and props["events"]["items"].get("type") == "string":
            seen["plan_user"] = messages[-1]["content"]
            return {"message": {"content": json.dumps({"events": EVENTS})}}
        if "events" in props:
            seen["read_user"] = messages[-1]["content"]
            return {"message": {"content": json.dumps({"events": [
                {"event": e, "first_shot": 0, "last_shot": 0} for e in EVENTS]})}}
        return {"message": {"content": "{}"}}
    monkeypatch.setattr(ollama, "chat", fake)
    story.critique_and_repair("a skydiver jumps and lands", _five(), rounds=1,
                              verbose=False)
    assert "leaps from the plane" not in seen["plan_user"] and \
        "SHOT LIST" not in seen["plan_user"]               # concept only, no shots
    r = seen["read_user"]
    assert r.index("1. leaps from the plane") < r.index("2. free-falls") \
        < r.index("SHOT LIST")


def test_wrong_number_of_event_entries_is_unreadable(monkeypatch):
    def fake(model=None, messages=None, format=None, options=None, **kw):
        return {"message": {"content": json.dumps({"events": [
            {"event": "x", "first_shot": 1, "last_shot": 1}]})}}
    monkeypatch.setattr(ollama, "chat", fake)
    ev, spans, err = story.read_events("c", SKY, "m", EVENTS)
    assert spans == {} and "unreadable" in err
