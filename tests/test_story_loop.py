"""The writing loop: enough shots, a story that holds, and the writing fed back.

Measured motivation, from one real draft (13 shots requested):

    [story] showrunner attempt 1: 6 shots, 1 story break(s) ...
    [story] showrunner attempt 2: 6 shots, 1 story break(s) ...
    [story] showrunner attempt 3: 6 shots, 1 story break(s) ...
    - story: beats go backwards (4:turn->develop)

The retry already attached the validator's complaint and the previous prompts, and
the same backwards beat survived all three attempts. Two causes, and this file pins
the fixes:

  * the beat order was never given a ROSTER. "beats never backwards" is advice; "shot
    12 is the turn" is a position to obey. `beat_layout` also makes the requested
    count concrete, which is the half of the brief the model was ignoring.
  * a clean-but-short list had no path to the requested count. Re-rolling throws a
    good story away, so the count is closed by EXTENSION: coverage inserted inside
    the approved scenes, with beat/scene/movement pinned to the insertion point so
    an insert cannot reorder a story that already passed.
  * a refusal across runs was blind: the rejected writing was thrown away with the
    process. `draft._pending*` keeps it beside the production (never in it) and
    `write_shots_ensemble(prior=...)` hands it to the first attempt.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import story
from story import (_merge_extension, _repair_note, _showrunner_user, beat_layout,
                   extend_shot_list, shots_from_dicts, story_shape,
                   write_shots_ensemble)


# --------------------------------------------------------------------------- #
# the fixtures: a clean 6-shot film, and two coverage inserts for it
# --------------------------------------------------------------------------- #

def _shot(prompt, scale, beat, scene, purpose, link, subject="woman",
          seconds=2.0, energy=0.2, camera="none") -> dict:
    return {"prompt": prompt, "scale": scale, "camera": camera,
            "seconds": seconds, "subject": subject, "scene": scene,
            "energy": energy, "beat": beat, "purpose": purpose, "link": link}


CLEAN = {"logline": "a woman on a bridge realises the figure is her own shadow",
         "shots": [
             _shot("a rain-soaked footbridge at night, a woman in a dark coat "
                   "stepping onto the wet planks, streetlights, fog", "wide",
                   "establish", "s1", "establish", "open", seconds=2.5,
                   energy=0.05),
             _shot("the same woman walking away along the footbridge, seen from "
                   "behind, rain on the planks, fog", "medium", "develop", "s1",
                   "master", "match-action", seconds=1.7),
             _shot("the woman's hand tightening on the wet iron railing, rain "
                   "beading on the metal", "close", "develop", "s1", "insert",
                   "match-action", seconds=1.6),
             _shot("a second figure under the far streetlight at the end of the "
                   "bridge, fog, facing her", "wide", "develop", "s2",
                   "establish", "scene-cut", subject="figure", seconds=2.0),
             _shot("the figure stepping out of the fog toward the woman on the "
                   "bridge, streetlight behind it", "medium", "turn", "s2",
                   "reaction", "match-action", subject="figure", seconds=1.8,
                   energy=0.45),
             _shot("the two figures facing each other on the rain-soaked bridge, "
                   "fog between them, one streetlight", "close", "resolve", "s2",
                   "resolve", "eyeline", subject="figure", seconds=2.2,
                   energy=0.15),
         ]}

INSERTS = {"shots": [
    {"prompt": "the woman's hand on the soaked railing of the footbridge, rain "
               "beading, fog behind", "scale": "close", "seconds": 1.6,
     "energy": 0.2, "purpose": "insert", "after": 2},
    {"prompt": "the second figure's boots on the wet footbridge planks, fog "
               "around them, streetlight", "scale": "medium", "seconds": 1.7,
     "energy": 0.3, "purpose": "reacton", "after": 4},
]}


def _clean_shots() -> list:
    shots, _problems = story.validate(CLEAN)
    return shots


def test_the_clean_fixture_is_actually_clean():
    """The loop tests are only meaningful on a list the gate would adopt."""
    shots, problems = story.validate(CLEAN)
    assert [p for p in problems if p.startswith("story:")] == []
    assert story_shape(shots) == "EdddTR"


# --------------------------------------------------------------------------- #
# the roster: a beaten count, not a beaten order
# --------------------------------------------------------------------------- #

def test_layout_names_where_every_beat_goes_for_the_requested_count():
    layout = beat_layout(13)
    assert "exactly 13 shots" in layout
    assert "shot 1 'establish'" in layout
    assert "shot 13 'resolve'" in layout
    assert "'fall'" in layout
    # the falling action after the turn is the measured failure: a good 13-shot
    # film came back as EddddTTddRR because the shots after the turn were labelled
    # develop again, and the model then reached for a beat the schema did not have.
    # The roster must make returning to develop illegal and name 'fall'.
    assert "no shot may be 'develop' again" in layout


def test_the_falling_action_has_its_own_beat_and_the_gate_accepts_it():
    """The 13-shot failure was a missing beat, not a missing image.

    A model writes establish -> develop -> turn -> (the change settling) ->
    resolve. With no beat for the settling it labelled that shot develop, the
    order check called it backwards, and three re-rolls were wasted refusing a
    coherent film. 'fall' is the word for it and the order allows it.
    """
    assert story.BEATS == ["establish", "develop", "turn", "fall", "resolve"]
    falls = {"logline": "the change settles", "shots": [
        _shot("a rain-soaked footbridge at night, a woman in a dark coat",
              "wide", "establish", "s1", "establish", "open", seconds=2.5),
        _shot("the same woman walking along the footbridge, rain and fog",
              "medium", "develop", "s1", "master", "match-action"),
        _shot("a second figure stepping out of the fog toward her, streetlight",
              "close", "turn", "s2", "reaction", "scene-cut", seconds=1.8),
        _shot("the streetlight swinging over the empty bridge after they pass",
              "wide", "fall", "s2", "insert", "match-action", seconds=2.3),
        _shot("the rain-soaked bridge empty at dawn, the fog lifting",
              "medium", "resolve", "s2", "resolve", "eyeline", seconds=2.6),
    ]}
    shots, problems = story.validate(falls)
    assert story_shape(shots) == "EdTfR"
    assert [p for p in problems if p.startswith("story:")] == []
    # and the old backwards shape is still refused
    broken = json.loads(json.dumps(falls))
    broken["shots"][3]["beat"] = "develop"
    _shots, problems = story.validate(broken)
    assert any(p.startswith("story: beats go backwards") for p in problems)


def test_repair_note_does_not_leave_the_falling_action_unlabelled():
    """The turnaround must be allowed: turn shots may repeat, develop may not."""
    note = _repair_note(["story: beats go backwards (9:turn->develop)"],
                        _clean_shots(), n_shots=13, returned=6)
    assert "LABEL PROBLEM, NOT AN IMAGE PROBLEM" in note
    assert "Keep every shot's image exactly as it is" in note
    assert "never 'develop' again" in note


def test_layout_handles_the_smallest_films_the_gate_can_pass():
    assert "shot 2 'turn'" in beat_layout(3)
    # fewer than 3 cannot hold establish + turn + resolve at all; say so instead
    # of inventing a roster that cannot pass
    assert "at least 3 shots" in beat_layout(2)


def test_the_showrunner_brief_carries_the_count_and_its_roster():
    msg = _showrunner_user("a woman on a bridge", 13, "- [farm] an idea")
    assert "Write exactly 13 shots." in msg
    assert "shot 1 'establish'" in msg
    assert "'fall'" in msg


def test_a_short_list_is_told_the_count_it_missed():
    """A clean-but-short reply has no story complaint, so the count must be named."""
    note = _repair_note([], _clean_shots(), n_shots=13, returned=6)
    assert "TOO SHORT" in note
    assert "returned 6 shots; the film is exactly 13 shots" in note
    assert "shot 13 'resolve'" in note
    # no story failure means no invented story complaint
    assert "REJECTED BY THE STORY/STRUCTURE CHECK" not in note


def test_a_refused_draft_from_another_run_reads_as_a_continuation():
    prior = _clean_shots()
    note = _repair_note(["story: no 'turn'"], prior, n_shots=8, returned=6,
                        prefix="A PREVIOUS DRAFT OF THIS FILM WAS REFUSED.")
    assert note.startswith("A PREVIOUS DRAFT OF THIS FILM WAS REFUSED.")
    assert "REJECTED BY THE STORY/STRUCTURE CHECK" in note


# --------------------------------------------------------------------------- #
# the extension: coverage added, story untouched
# --------------------------------------------------------------------------- #

def test_extension_inserts_where_it_is_told_and_pins_the_structure():
    merged, dropped = _merge_extension(_clean_shots(), INSERTS["shots"])
    assert dropped == []
    assert len(merged) == 8
    # position: after shot 2 and after shot 4
    assert "hand on the soaked railing" in merged[2].prompt
    assert "boots on the wet footbridge planks" in merged[5].prompt
    # the inserted shots cover the scene and beat they were inserted into
    assert merged[2].scene == "s1" and merged[2].beat == "develop"
    assert merged[5].scene == "s2" and merged[5].beat == "develop"
    assert merged[2].camera == "none"
    assert merged[2].link == "match-action"
    # the story order is unchanged and still monotone
    beats = [story.BEATS.index(s.beat) for s in merged]
    assert beats == sorted(beats)
    assert story_shape(merged) == "EdddddTR"


def test_an_insert_cannot_add_a_new_ending_or_a_new_establish():
    """The model's own beat tag is IGNORED: a coverage shot covers its anchor."""
    shots = _clean_shots()
    evil = [
        {"prompt": "a totally new final image of the woman alone at dawn", "after": 6,
         "scale": "wide", "seconds": 2.0, "energy": 0.2, "purpose": "resolve"},
        {"prompt": "a second establishing shot of an empty room, cold light",
         "after": 1, "scale": "medium", "seconds": 2.0, "energy": 0.2,
         "purpose": "establish"},
    ]
    merged, _dropped = _merge_extension(shots, evil)
    # the ending was not displaced: the original last shot is still last
    assert merged[-1].prompt == shots[-1].prompt
    # the insert after the establish is a develop, not a second establish
    assert merged[1].beat == "develop"
    assert [p for p in story.story_problems(merged)
            if p.startswith("story:")] == []


def test_junk_inserts_are_dropped_not_merged():
    shots = _clean_shots()
    junk = [
        {"prompt": "too terse here", "after": 2, "scale": "close", "seconds": 2.0,
         "energy": 0.2, "purpose": "insert"},                    # < 8 words
        {"prompt": shots[1].prompt, "after": 2, "scale": "close", "seconds": 2.0,
         "energy": 0.2, "purpose": "insert"},                    # duplicate image
        {"prompt": "a completely unrelated image of a chrome diner at noon, hard "
                   "shadows on the counter", "scale": "close", "seconds": 2.0,
         "energy": 0.2, "purpose": "insert"},                    # no 'after'
    ]
    merged, dropped = _merge_extension(shots, junk)
    assert len(merged) == len(shots)
    assert len(dropped) == 3


def test_extension_scale_never_jump_cuts_the_shot_before_it():
    shots = _clean_shots()
    inserts = [{"prompt": "a close detail of the wet bridge planks and rain",
                "scale": "medium", "seconds": 1.6, "energy": 0.2,
                "purpose": "insert", "after": 2}]
    merged, _dropped = _merge_extension(shots, inserts)
    # shot 2 is medium; the insert asked for medium and must be re-scaled
    assert merged[2].scale != merged[1].scale


# --------------------------------------------------------------------------- #
# the count loop, end to end, with a stubbed model
# --------------------------------------------------------------------------- #

def _stub_chat(calls, first, second):
    """Distinguish the main writer from the extension by the schema OBJECT."""
    def fake_chat(**kwargs):
        is_extension = kwargs.get("format") is story.EXTEND_SCHEMA
        calls.append({"extension": is_extension, "messages": kwargs["messages"]})
        reply = second if is_extension else first
        return {"message": {"content": json.dumps(reply)}, "eval_count": 50}

    return fake_chat


def _pool() -> list:
    return [{"model": "stub-farm", "angle": "places",
             "ideas": ["a wet footbridge railing", "a streetlight in fog"],
             "seconds": 0.0}]


def test_a_clean_short_film_is_extended_to_the_requested_count(monkeypatch):
    calls: list = []
    monkeypatch.setattr(__import__("ollama"), "chat",
                        _stub_chat(calls, CLEAN, INSERTS))
    p = write_shots_ensemble(
        "a woman on a bridge", farms=[], farm_out=_pool(), showrunner="stub",
        n_shots=8, story_retries=0, fresh=True, use_cache=False,
        save_cache=False, verbose=False, num_predict=900, extend=True,
        extend_attempts=1)

    assert len(p.shots) == 8
    assert p.extended is True and p.extend_calls == 1
    assert p.requested == 8
    assert story_shape(p.shots) == "EdddddTR"
    assert not [x for x in p.problems if x.startswith("story:")]
    # the model was asked for the count as a roster, then for coverage
    assert "Write exactly 8 shots." in calls[0]["messages"][1]["content"]
    assert calls[1]["extension"] is True
    assert "must become 8 shots" in calls[1]["messages"][1]["content"]
    # the loop's evidence is in the cached raw object too
    assert p.raw["loop"]["returned"] == 8
    assert p.raw["loop"]["extension"]["calls"][0]["added"] == 2


def test_extension_is_not_forced_on_the_comparison_harness(monkeypatch):
    """`extend` defaults OFF so measuring the writer still measures the writer."""
    calls: list = []
    monkeypatch.setattr(__import__("ollama"), "chat",
                        _stub_chat(calls, CLEAN, INSERTS))
    p = write_shots_ensemble(
        "a woman on a bridge", farms=[], farm_out=_pool(), showrunner="stub",
        n_shots=8, story_retries=0, fresh=True, use_cache=False,
        save_cache=False, verbose=False, num_predict=900)
    assert len(p.shots) == 6            # the writer's own count, unextended
    assert p.extended is False and p.extend_calls == 0


def test_a_broken_story_is_not_extended(monkeypatch):
    """Adding coverage cannot fix a backwards beat, so it is not attempted."""
    broken = {"logline": "a mood board", "shots": CLEAN["shots"][:2] + [
        _shot(CLEAN["shots"][4]["prompt"], "close", "develop", "s2", "master",
              "match-action", subject="figure", seconds=1.8, energy=0.4),
        CLEAN["shots"][5],
    ]}
    calls: list = []
    monkeypatch.setattr(__import__("ollama"), "chat",
                        _stub_chat(calls, broken, INSERTS))
    p = write_shots_ensemble(
        "a woman on a bridge", farms=[], farm_out=_pool(), showrunner="stub",
        n_shots=8, story_retries=0, fresh=True, use_cache=False,
        save_cache=False, verbose=False, num_predict=900, extend=True)
    assert p.extended is False
    assert any(x.startswith("story:") for x in p.problems)
    assert not any(c["extension"] for c in calls)


def test_a_refused_draft_is_fed_back_into_the_first_attempt(monkeypatch):
    calls: list = []
    monkeypatch.setattr(__import__("ollama"), "chat",
                        _stub_chat(calls, CLEAN, INSERTS))
    prior = {"shots": _clean_shots()[:4],
             "problems": ["story: beats go backwards (3:turn->develop)"]}
    write_shots_ensemble(
        "a woman on a bridge", farms=[], farm_out=_pool(), showrunner="stub",
        n_shots=6, story_retries=0, fresh=True, use_cache=False,
        save_cache=False, verbose=False, num_predict=900, prior=prior)
    first_user = calls[0]["messages"][1]["content"]
    assert "A PREVIOUS DRAFT OF THIS FILM WAS REFUSED" in first_user
    assert "beats go backwards" in first_user
    # and it is attached, not just described: its imagery is shown to the model
    assert "rain-soaked footbridge" in first_user


def test_the_extension_loop_asks_again_when_a_pass_is_partial(monkeypatch):
    """A model that returns 1 of 3 asked shots gets asked for the rest."""
    only_one = {"shots": [INSERTS["shots"][0]]}
    replies = [only_one, {"shots": [
        {"prompt": "the second figure's boots on the wet bridge planks, fog",
         "scale": "medium", "seconds": 1.7, "energy": 0.3, "purpose": "insert",
         "after": 5},
        {"prompt": "rain beading on the iron railing of the same footbridge, fog",
         "scale": "wide", "seconds": 1.9, "energy": 0.2, "purpose": "master",
         "after": 3},
    ]}]
    calls: list = []

    import ollama

    def fake_chat(**kwargs):
        calls.append(kwargs.get("format") is story.EXTEND_SCHEMA)
        return {"message": {"content": json.dumps(replies.pop(0))},
                "eval_count": 20}

    monkeypatch.setattr(ollama, "chat", fake_chat)
    shots, info = extend_shot_list("concept", _clean_shots(), 9, showrunner="stub",
                                   attempts=3, verbose=False)
    assert len(shots) == 9
    assert len(info["calls"]) == 2
    assert info["calls"][0]["added"] == 1
    assert info["calls"][1]["added"] == 2


# --------------------------------------------------------------------------- #
# a refusal survives the process (it is kept beside, never inside, the production)
# --------------------------------------------------------------------------- #

class _Proposalish:
    def __init__(self, shots, problems):
        self.shots = shots
        self.problems = problems


def test_a_refused_draft_is_kept_beside_the_production():
    """The refusal must survive the process, so the next run can repair it.

    Uses a workspace scratch directory rather than pytest's tmp_path: the DSH
    sandbox denies `os.listdir` on the platform temp tree, which makes tmp_path
    fixtures fail at SETUP (a harness failure, not a code failure -- the trap the
    handoff warns about).
    """
    root = Path(__file__).resolve().parent.parent
    scratch = root / "output" / "_test_story_loop"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    try:
        import film_store as fs
        old = fs.PRODUCTIONS
        fs.PRODUCTIONS = scratch
        try:
            import draft

            draft.save_pending_draft("ego", "ego death shadow", _Proposalish(
                _clean_shots()[:3],
                ["story: beats go backwards (3:resolve->develop)"]), 6)

            data = draft.load_pending_draft("ego", "ego death shadow")
            assert data and len(data["shots"]) == 3
            assert data["story_shape"] == "Edd"
            assert data["problems"] == [
                "story: beats go backwards (3:resolve->develop)"]
            # the sidecar is beside project.json, never in it
            assert (scratch / "ego" / "draft_attempt.json").exists()
            assert not (scratch / "ego" / "project.json").exists()

            # a different concept has nothing to continue from
            assert draft.load_pending_draft("ego",
                                            "a lighthouse keeper") is None

            draft.clear_pending_draft("ego")
            assert draft.load_pending_draft("ego", "ego death shadow") is None
        finally:
            fs.PRODUCTIONS = old
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_shots_from_dicts_round_trips_the_writer_shape():
    shots = shots_from_dicts([story.shot_dict(s) for s in _clean_shots()])
    assert story_shape(shots) == "EdddTR"
    gap = [p for p in story.story_problems(shots) if p.startswith("story:")]
    assert gap == []


# --------------------------------------------------------------------------- #
# the job runner can tell a long extension pass from a hung job
# --------------------------------------------------------------------------- #

def test_the_job_runner_shows_the_extension_phase():
    import job_runner as jr

    prog: dict = {}
    jr._draft_progress("  [story] extension 1/2: 6/9 shot(s) -- asking for 3 more "
                       "coverage", prog)
    assert prog["phase"] == "extension"
    assert (prog["shots"], prog["requested"]) == (6, 9)
    assert prog["extension"] == 1 and prog["extension_total"] == 2


def test_the_job_runner_reports_the_loop_result():
    import job_runner as jr

    prog: dict = {}
    jr._draft_progress("[draft] writing loop: 2 attempt(s), 1 extension call(s) "
                       "-> 9/9 shot(s), story shape EdddddddTR", prog)
    assert prog["loop_attempts"] == 2


# --------------------------------------------------------------------------- #
# the draft JOB: a refusal is kept, and the next run is handed it
# --------------------------------------------------------------------------- #

def test_the_draft_job_carries_a_refusal_into_the_next_run(monkeypatch):
    """The whole point of the loop across runs, at the entry point that does it.

    First run: the writer returns a broken story -> exit 2, the production's
    timeline is untouched, and the refused writing is on disk. Second run: the
    writer is handed that refused writing as `prior`, adopts a clean list, and the
    sidecar is cleared. A fake writer is used -- no LLM, no render, one second.
    """
    import shutil
    import sys

    import draft
    import film_store as fs
    import story
    from story import Proposal

    slug = "_test_story_loop_draft"
    concept = "a woman on a bridge"
    shutil.rmtree(fs.dir_for(slug), ignore_errors=True)
    try:
        fs.save(fs.new_project(slug, concept=concept))
        broken = Proposal(
            concept=concept, model="stub", logline="a mood board",
            shots=_clean_shots()[:2],                       # establish, then...
            raw={"logline": "a mood board"}, seconds=0.0, decode_tok_s=0.0,
            problems=["story: no 'turn' -- nothing changes"])
        clean = Proposal(
            concept=concept, model="stub", logline="a film that develops",
            shots=_clean_shots(), raw={"logline": CLEAN["logline"]},
            seconds=0.0, decode_tok_s=0.0, problems=[])
        seen: list = []

        def fake_writer(concept_arg, **kwargs):
            seen.append(kwargs)
            return broken if len(seen) == 1 else clean

        monkeypatch.setattr(story, "write_shots_ensemble", fake_writer)
        monkeypatch.setattr(sys, "argv", [
            "draft.py", "--store", slug, "--concept", concept,
            "--n-shots", "6", "--fresh"])

        first = draft.main()
        after_refusal = fs.load(slug)
        pending = draft.load_pending_draft(slug, concept)
        second = draft.main()
        adopted = fs.load(slug)

        # 1. the broken story was refused, not adopted
        assert first == 2
        assert after_refusal["timeline"] == []
        assert pending and len(pending["shots"]) == 2
        assert pending["story_shape"] == "Ed"
        # 2. the next run was handed that refusal on its FIRST attempt
        assert seen[0].get("prior") is None
        assert seen[1].get("prior") is not None
        assert len(seen[1]["prior"]["shots"]) == 2
        assert any("no 'turn'" in p for p in seen[1]["prior"]["problems"])
        # 3. every run used the count loop and the clean list was adopted
        assert all(k.get("extend") is True for k in seen)
        assert second == 0
        assert len(adopted["timeline"]) == 6
        assert draft.load_pending_draft(slug, concept) is None
    finally:
        shutil.rmtree(fs.dir_for(slug), ignore_errors=True)


# --------------------------------------------------------------------------- #
# a cut-off reply keeps its complete shots; a long reply reports real progress
# --------------------------------------------------------------------------- #

def _cut_inside_shot(n_complete: int) -> str:
    """CLEAN serialised, then cut in the MIDDLE of shot `n_complete + 1`."""
    text = json.dumps(CLEAN, indent=2)
    marks = [m.start() for m in __import__("re").finditer(r'\n    \{', text)]
    return text[:marks[n_complete] + 60]


def test_salvage_keeps_every_complete_shot_and_drops_the_unfinished_one():
    cut = _cut_inside_shot(4)
    got = story.salvage_shots(cut)
    assert got is not None and len(got["shots"]) == 4
    assert got["logline"] == CLEAN["logline"]
    assert got["shots"][3]["prompt"] == CLEAN["shots"][3]["prompt"]
    # it is a truncation, not a syntax habit: the strict parsers still refuse it
    assert story.parse_reply(cut) == (None, "")
    assert story.json_failure(cut).endswith("TRUNCATED")


def test_salvage_is_opt_in_so_farms_and_plans_still_fail_when_cut_off():
    cut = _cut_inside_shot(3)
    assert story.parse_reply(cut)[0] is None
    data, mode = story.parse_reply(cut, salvage=True)
    assert mode == "salvaged" and len(data["shots"]) == 3


def test_salvage_is_not_fooled_by_braces_and_quotes_inside_a_prompt():
    tricky = ('{"logline": "x", "shots": [{"prompt": "a sign reading \\"}{\\" in '
              'fog", "scale": "wide"}, {"prompt": "unfinished')
    got = story.salvage_shots(tricky)
    assert got is not None and len(got["shots"]) == 1
    assert got["shots"][0]["prompt"] == 'a sign reading "}{" in fog'


def test_salvage_returns_nothing_when_no_shot_completed():
    assert story.salvage_shots('{"logline": "x", "shots": [{"prompt": "cut') is None
    assert story.salvage_shots("") is None
    assert story.salvage_shots("no json here") is None


def test_a_truncated_showrunner_reply_becomes_a_short_draft_not_a_lost_attempt(
        monkeypatch):
    cut = _cut_inside_shot(4)
    monkeypatch.setattr(__import__("ollama"), "chat",
                        lambda **kw: {"message": {"content": cut}, "eval_count": 900})
    p = write_shots_ensemble(
        "a woman on a bridge", farms=[], farm_out=_pool(), showrunner="stub",
        n_shots=6, story_retries=0, fresh=True, use_cache=False,
        save_cache=False, verbose=False, num_predict=900)
    assert len(p.shots) == 4                    # the four that were whole
    assert p.shots[0].prompt.startswith("a rain-soaked footbridge")


def test_chat_streamed_reports_tokens_in_the_format_the_runner_parses(monkeypatch,
                                                                      capsys):
    import job_runner
    pieces = ["alpha ", "beta ", "gamma ", "delta"]

    def fake_chat(**kw):
        assert kw.get("stream") is True
        def gen():
            for p in pieces:
                yield {"message": {"content": p}, "done": False}
            yield {"message": {"content": ""}, "done": True, "eval_count": 4}
        return gen()

    monkeypatch.setattr(__import__("ollama"), "chat", fake_chat)
    monkeypatch.setattr(story, "PROGRESS_EVERY_S", 0.0)
    text, tokens = story.chat_streamed("showrunner attempt 2/4", 2800, model="m",
                                       messages=[])
    assert text == "alpha beta gamma delta" and tokens == 4
    lines = [l for l in capsys.readouterr().out.splitlines() if "tokens," in l]
    assert lines, "a streamed reply must print progress"
    prog: dict = {}
    job_runner._story_progress(lines[-1], prog)       # the REAL printed line
    assert prog["phase"] == "showrunner" and prog["attempt"] == 2
    assert prog["token_budget"] == 2800 and 0 < prog["tokens"] <= 4
    assert prog["tok_per_s"] >= 0


def test_a_non_streaming_client_still_works(monkeypatch):
    monkeypatch.setattr(__import__("ollama"), "chat",
                        lambda **kw: {"message": {"content": "hi"}, "eval_count": 7})
    assert story.chat_streamed("x", 100, model="m", messages=[]) == ("hi", 7)


def test_the_generating_line_resets_tokens_and_sets_the_cap():
    import job_runner
    prog = {"tokens": 2800, "token_budget": 2800}
    job_runner._draft_progress(
        "  [story] showrunner attempt 2/4: generating (<= 2800 tokens, ctx 6144, "
        "~3 tok/s on CPU -- this is the silent stretch)", prog)
    assert prog["tokens"] == 0 and prog["token_budget"] == 2800
    assert prog["attempt"] == 2 and prog["attempt_total"] == 4


def test_the_studio_bar_shows_a_real_token_figure_for_the_showrunner():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ui"))
    import studio
    import job_runner as jr
    job = {"status": jr.RUNNING, "child_started": 0,
           "progress": {"phase": "showrunner", "tokens": 700, "token_budget": 2800,
                        "tok_per_s": 3.1, "minutes_left_max": 11, "attempt": 2,
                        "attempt_total": 4}}
    f = studio.progress_facts(job)
    assert f["outer"] is not None and abs(f["outer"][0] - 0.25) < 1e-9
    assert "700/2800 tokens" in f["outer"][1] and "3.1 tok/s" in f["outer"][1]
    assert "attempt 2/4" in f["outer"][1]
    assert f["modelled"] is False          # so "no progress figure" is not shown
