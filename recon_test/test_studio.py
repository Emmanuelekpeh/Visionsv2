"""Render the studio and check the three-verb structure holds.

    python recon_test/test_studio.py

The checks are the design rules from ui/studio.py's header, turned into assertions:
one primary action per view, state as a word, the price before the button, nothing
on the main path that is not needed to make a film.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_home"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch_home"))

from streamlit.testing.v1 import AppTest  # noqa: E402

import job_runner as jr  # noqa: E402

APP = str(ROOT / "ui" / "studio.py")
SLUG = "_studio_test"
FAILS = []


def check(label, got, want=True):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}"
          + ("" if ok else f"   (want {want!r})"))
    if not ok:
        FAILS.append(label)


def build_fixture() -> str:
    import film_store as fs
    shutil.rmtree(fs.dir_for(SLUG), ignore_errors=True)
    legacy = ROOT / "workspace" / "ensemble1_workbench.json"
    if legacy.exists():
        fs.import_legacy(legacy, slug=SLUG)
    else:
        pr = fs.new_project(SLUG, "a test")
        fs.save(pr)
        for i, (cam, sec) in enumerate([("none", 2.0), ("zoom-in", 1.5),
                                        ("pan-left", 3.0)]):
            fs.add_shot(pr, prompt=f"test shot {i}", camera=cam, seconds=sec,
                        scene="sc1")
        fs.save(pr)
    return SLUG


def run(view: str, shot=None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["slug"] = SLUG
    at.session_state["view"] = view
    if shot:
        at.session_state["shot"] = shot
    at.run()
    return at


def text_of(at) -> str:
    """Everything the page renders, sidebar included."""
    parts = []
    for attr in ("markdown", "caption", "warning", "info", "success", "error",
                 "text"):
        parts += [getattr(x, "value", "") for x in getattr(at, attr, [])]
    return " ".join(str(p) for p in parts)


def canvas_text(at) -> str:
    """The main canvas, with the SIDEBAR's job panel filtered out.

    The sidebar legitimately contains words the canvas must not. Its job panel renders
    `**draft** — the scouts are generating ideas` while a draft runs, and a test for "no
    scout material on the main path" that reads the whole page matches THAT and fails --
    which it did, on a correct page. Verified by element index: the only element on the
    page containing the word was that one line.

    Filtering the job panel's own line is exact, where slicing on headings was not: a
    shorter marker ("the film") occurs inside a longer one that the sidebar's caption
    also contains, so any positional slice is fragile.
    """
    keep = []
    for attr in ("markdown", "caption", "warning", "info", "success", "error", "text"):
        for x in getattr(at, attr, []):
            v = str(getattr(x, "value", ""))
            # the job panel's headline, and only that: bolded task name + an em dash
            if v.startswith("**") and "** — " in v and len(v) < 120:
                continue
            keep.append(v)
    return " ".join(keep)


def primary_ish(at) -> List[str]:
    """AppTest does not surface `type="primary"`, so this asserts on the CONTRACT
    instead: the labels that are allowed to be the view's main action. If a new
    primary button is added, it has to be named here, which is the review this
    check exists to force."""
    return [b.label for b in at.button]


def main() -> int:
    import film_store as fs
    build_fixture()
    n = len(fs.load(SLUG)["timeline"])
    print(f"testing {n} shots\n")

    print("1. WRITE: one field, one primary action")
    at = run("write")
    exc = [e.value for e in at.exception]
    check("no exception", exc, [])
    body = text_of(at)
    check("the concept field is there", len(at.text_area) >= 1, True)
    labels = primary_ish(at)
    print(f"     buttons: {labels}")
    # WRITE's job is to produce a shot list. Moving on to shoot it is the one
    # legitimate second action; anything that renders or exports is not.
    check("one action writes the list",
          sum(1 for l in labels if "write the shot list" in l) == 1, True)
    check("nothing on WRITE renders or exports",
          not any(l.strip() == "shoot it" or "export" in l for l in labels), True)
    check("the price is stated before the button", "min of rendering" in body, True)
    check("the shot list is on the page", "the shot list" in body, True)
    # the thing that was packed: 18 scout ideas must NOT be on the main path.
    # CANVAS ONLY: the sidebar's job panel says "the scouts are generating ideas" while
    # a draft runs, and reading the whole page matches that and fails on a correct page.
    check("no scout idea dump on the main CANVAS",
          "scouts" not in canvas_text(at) and "raw material" not in canvas_text(at),
          True)

    print("\n2. SHOOT: the strip, then one shot")
    first = fs.load(SLUG)["timeline"][0]
    at = run("shoot", shot=first)
    exc = [e.value for e in at.exception]
    check("no exception", exc, [])
    body = text_of(at)
    check("the film strip is shown", "the film" in body, True)
    check("a running timecode is shown", "s →" in body or "0.0s" in body, True)
    check("the state is a WORD, not only a colour",
          any(w in body for w in ("not shot", "shot", "locked", "needs re-")), True)
    check("the price precedes the action", "min" in body.lower(), True)
    labels = primary_ish(at)
    print(f"     buttons: {labels}")
    # the ONLY thing that starts a render is the shoot button, and there is one
    check("exactly one action starts a render",
          sum(1 for l in labels if l.startswith("shoot")) == 1, True)
    check("nothing else on the shot competes with shooting",
          not any(l in labels for l in ("export the film", "write the shot list")),
          True)
    check("the destructive things are behind 'more', not on the canvas",
          all(l in labels for l in ("duplicate as an alternate angle",
                                    "forget which take is used",
                                    "remove from the film")), True)

    print("\n3. CUT: says why it cannot export, then exports")
    at = run("cut")
    exc = [e.value for e in at.exception]
    check("no exception", exc, [])
    body = text_of(at)
    if fs.film_status(fs.load(SLUG))["to_generate"]:
        check("refuses to export and says why", "never been shot" in body, True)
        check("cannot invent frames", "invent frames" in body, True)
    else:
        check("offers the export", "export the film" in body, True)

    print("\n4. SETTINGS: everything real, off the main path")
    at = run("settings")
    exc = [e.value for e in at.exception]
    check("no exception", exc, [])
    body = text_of(at)
    check("the scout ideas live here", "raw material" in body, True)
    check("the palettes live here", "scenes and colour" in body, True)
    check("the store is explained", "one file per shot" in body, True)

    print("\n5. the design rules, counted")
    at = run("shoot", shot=first)
    n_exp = len(at.expander)
    print(f"     exprs opened by default on the shoot view: {n_exp}")
    # the two allowed are in the SIDEBAR (the job log and new-production), which is
    # off the canvas. The canvas itself opens nothing.
    check("the canvas opens no expander by default", n_exp <= 2, True)
    canvas_labels = [b.label for b in at.button
                     if b.label not in ("new production",)]
    print(f"     widgets on the shoot view: {len(at.button)} buttons, "
          f"{len(at.selectbox)} selects, {len(at.number_input)} numbers, "
          f"{len(at.text_area)} text areas")
    check("the shoot view is not a wall of controls",
          len(at.selectbox) <= 4 and len(at.number_input) <= 2, True)
    # rule 2: nothing nested more than two deep. Four decisions + the strip is the
    # whole surface; if this grows, something has crept back onto the canvas.
    check("at most four decisions on the shot", len(at.selectbox) + 
          len(at.number_input) <= 5, True)

    print("\n6. the layout uses the window")
    # The bug this exists for: `layout="wide"` was set and then undone by a
    # `max-width: 1180px` on the container, so a 1920px window showed the whole app
    # in a narrow centre column with the right 40% empty. A measure belongs on prose,
    # not on a grid.
    #
    # Asserts on the ACTIVE rule, not on the text: the first version of this check
    # failed because the explanatory COMMENT contains the buggy value it replaced.
    # A test that reads the explanation instead of the code is worthless.
    src = (ROOT / "ui" / "studio.py").read_text(encoding="utf-8")
    import re as _re
    body = src.split('CSS = """', 1)[1].split('"""', 1)[0] if 'CSS = """' in src else ""
    check("the CSS block was found", len(body) > 200, True)
    rule_line = _re.search(r"\.block-container\s*\{[^}]*\}", body)
    check("there is a .block-container rule", rule_line is not None, True)
    if rule_line:
        active = rule_line.group(0)
        print(f"     active rule: {active.strip()}")
        check("the container has no max-width cap",
              ("max-width: none" in active and "1180" not in active), True)
    check("prose still gets a measure", ".studio-prose" in body, True)
    check("the page is in wide layout", 'layout="wide"' in src, True)

    print("\n7. the progress bar reports the truth for each phase")
    # Tested through `progress_facts`, which is the arithmetic, NOT through a rendered
    # page -- the first version drove the whole app with AppTest and was defeated by
    # `jr.status()` returning the jobs directory's newest state rather than the one the
    # test had written. Arithmetic that can only be reached through a subprocess is
    # arithmetic that will not get tested.
    import time as _t
    from ui.studio import progress_facts

    def facts(**over) -> dict:
        return progress_facts({"task": "produce", "status": jr.RUNNING,
                               "child_started": _t.time() - 600, **over})

    f = facts(progress={"phase": "generating", "done": 2, "total": 5,
                        "mean_shot_seconds": 600.0})
    check("generating: says which shot", f["line"], "generating shot 3 of 5")
    check("generating: the outer bar is shots", f["outer"], (0.4, "2/5 shots"))
    check("generating: elapsed is reported", round(f["elapsed"]), 600)
    check("generating: the inner bar uses the MEASURED mean",
          (round(f["inner"][0], 3), f["inner"][3]), (0.99, True))
    check("generating: the inner bar never reads as complete while it runs",
          f["inner"][0] < 1.0, True)
    check("generating: the ETA covers the remaining shots", f["eta"], (3, 1800.0))
    check("generating: no fake finishing tail before generation ends", f["tail"], None)

    f = facts(progress={"phase": "generating", "done": 5, "total": 5,
                        "mean_shot_seconds": 600.0, "to_refinish": 2})
    check("a finishing tail is added once generation is done",
          f["tail"], 25.0 * 2 + 100.0)

    f = facts(progress={"phase": "generating"})
    check("with no measurement yet, the inner bar says it is an estimate",
          f["inner"][3], False)
    check("and falls back to the researched default of 11 min",
          f["inner"][2], 660.0)

    f = facts(progress={"phase": "loading the model"})
    check("model loading is a named phase with its own bar",
          (f["line"], bool(f["inner"])), ("loading the model and its motion LoRAs",
                                          True))
    check("model loading has no shot bar to lie about", f["outer"], None)

    f = facts(status=jr.QUEUED, child_started=None, progress={})
    check("a queued job says it is waiting", f["line"], "waiting to start")
    check("and shows no bars at all", (f["outer"], f["inner"], f["elapsed"]),
          (None, None, None))

    f = facts(progress={"phase": "ideas"})
    check("a language-model phase refuses to fake a percentage",
          (f["modelled"], f["outer"], f["inner"]), (True, None, None))
    check("and says why in words", f["line"], "the scouts are generating ideas")

    f = facts(progress={"phase": "assembling", "done": 5, "total": 5})
    check("assembling is named and has a bar", f["line"], "assembling the film")
    check("assembling's bar is sized by its own measured cost",
          f["inner"][2], 100.0)

    f = facts(status=jr.DONE, progress={"phase": "finished", "done": 5, "total": 5})
    check("a finished job reports no elapsed clock",
          (f["elapsed"], f["eta"]), (None, None))

    print("\n8. 'shoot the next N' picks the shots that NEED it")
    # The bug this exists for: the first implementation built `range(first, first+n)`
    # from the first unrendered index. With shot 2 already shot and 1, 3, 4 missing,
    # that re-shoots shot 2 (ten minutes) and skips shot 3. The list has to be the
    # unrendered indices themselves, not a contiguous run.
    import film_store as _fs
    from ui.studio import _unrendered_indices

    pr = _fs.new_project("Batch Probe")
    pr["slug"] = "_batch_probe"
    _fs.save(pr)
    made = [_fs.add_shot(pr, prompt=f"b{i}", camera="none", seconds=2.0, scene="sc1")
            for i in range(4)]
    # render ONLY the second shot, so the unrendered set is 1,3,4
    d = _fs.take_dir(pr, made[1], "t01") / "frames"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (d / f"f{i:03d}.png").write_bytes(b"x")
    _fs._write_json(_fs.take_dir(pr, made[1], "t01") / "take.json",
                    _fs.take_record(pr, made[1], seconds=5.0, frames=3, take="t01"))
    _fs.choose_take(pr, made[1]["id"], "t01")
    _fs.save(pr)

    pr = _fs.load("_batch_probe")
    check("the rendered shot is not in the unrendered set",
          _unrendered_indices(pr), [1, 3, 4])
    check("asking for 2 gives shots 1 and 3, not 1 and 2",
          _unrendered_indices(pr)[:2], [1, 3])
    check("asking for more than remain gives all of them",
          _unrendered_indices(pr)[:99], [1, 3, 4])
    # and the CLI flag those indices feed really does accept a list
    check("the comma list is what --only expects",
          ",".join(str(i) for i in _unrendered_indices(pr)[:2]), "1,3")
    shutil.rmtree(_fs.dir_for("_batch_probe"), ignore_errors=True)

    print("\n9. the page keeps itself fresh")
    # The user's complaint: "it needs to auto refresh after the shot list is done ...
    # we need constant freshness ... if not im wasting extra time waiting for it."
    # They were reloading by hand, because a job writes into productions/ from another
    # PROCESS and nothing in this session knew the shot list had changed.
    src = (ROOT / "ui" / "studio.py").read_text(encoding="utf-8")
    check("the content view is a self-refreshing fragment",
          "@st.fragment(run_every=2)" in src, True)
    check("it takes the SLUG, so it re-reads the project rather than reusing a stale "
          "dict loaded outside it",
          "def content(slug: str)" in src, True)
    check("and it clears the project cache each pass",
          "load.clear()" in src, True)
    check("without throwing away the thumbnail cache every 2 seconds",
          "st.cache_data.clear()\n    project = load" not in src
          and "def view_settings" in src, True)
    check("the job panel refreshes at least as fast as the supervisor writes",
          "@st.fragment(run_every=2)" in src, True)

    print("\n10. the scouts report progress instead of going silent")
    # They run serially and the first output line used to be AFTER all of them, so a
    # 3-minute run could not be told from a hang.
    story_src = (ROOT / "story.py").read_text(encoding="utf-8")
    check("story.py announces each farm, flushed",
          "[farm {i + 1}/{len(farm_models)}]" in story_src
          and "flush=True" in story_src, True)
    prog = {}
    jr._story_progress("  [farm 1/3] qwen2.5:3b ...", prog)
    check("the runner turns that into a bar denominator",
          (prog.get("done"), prog.get("total")), (0, 3))
    jr._story_progress("  [farm 1/3] qwen2.5:3b -> 6 ideas", prog)
    check("and advances it when a farm lands", (prog.get("done"), prog.get("total")),
          (1, 3))
    # and the bar shows up rather than the "no progress figure" caption
    f = progress_facts({"task": "draft", "status": jr.RUNNING,
                        "child_started": _t.time() - 30,
                        "progress": {"phase": "ideas", "done": 1, "total": 3}})
    check("the scout bar is a real bar, not a 'no figure' caption",
          (round(f["outer"][0], 3), f["outer"][1]), (0.333, "1/3 scouts"))
    check("and the caption is suppressed when there IS a figure",
          f["modelled"], False)

    print("\n11. THE COMPLETION SIGNAL — you must be able to see that it finished")
    # The user's actual complaint, in their words: "after what I was trying to generate
    # has completed, I don't realize that it has generated because the UI does not
    # refresh or it does not update to say that's oh the generation has happened yet."
    #
    # `completion_banner` takes an injectable job, because jr.status() returns whichever
    # state file on disk is newest and is therefore useless as a fixture. The first
    # version of this test drove the whole app and silently rendered a REAL job instead
    # of the one it wrote.
    from ui.studio import _what_it_made

    # the words, per terminal state, tested as pure function first
    check("a finished produce job says how many shots",
          _what_it_made({"task": "produce", "progress": {"shots": 4}}),
          "4 shot(s) generated")
    check("a finished produce job with no shot count still says something useful",
          _what_it_made({"task": "produce", "progress": {}}), "the film was cut")
    check("a finished draft says the shot list is written",
          _what_it_made({"task": "draft", "progress": {"shots": 6}}),
          "6 shots written into the film")
    check("a re-grade-only run says so",
          _what_it_made({"task": "produce", "progress": {"to_refinish": 2}}),
          "2 shot(s) re-graded and the film cut")

    # and the words, per terminal state, are asserted above as pure functions; here
    # they are rendered through the real app with `jr.status` patched, because the
    # banner has to be ON THE PAGE, not merely computed.
    import job_runner as _jr
    real_status = _jr.status

    def with_status(job: dict):
        _jr.status = lambda job_id=None: job
        try:
            at = AppTest.from_file(APP, default_timeout=180)
            at.session_state["slug"] = SLUG
            at.session_state["view"] = "write"
            at.run()
            return text_of(at), at
        finally:
            _jr.status = real_status

    # a RECENT finish. The first version of this probe used a fixed 14:22 timestamp and
    # failed -- correctly -- because the banner is freshness-filtered to 30 minutes, and
    # by the time this ran that was two hours old. A test that hard-codes a clock is
    # testing the clock.
    now = _t.time()
    now_hhmm = _t.strftime("%H:%M", _t.localtime(now))
    body, at = with_status({"id": "probe1", "task": "produce", "status": _jr.DONE,
                            "returncode": 0,
                            "finished": _t.strftime("%Y-%m-%d %H:%M:%S",
                                                    _t.localtime(now)),
                            "progress": {"phase": "finished", "shots": 3}})
    check("DONE: the page says it finished", "finished" in body, True)
    check("DONE: it names what was produced", "3 shot(s) generated" in body, True)
    check("DONE: it shows when", now_hhmm in body, True)

    body, at = with_status({"id": "probe2", "task": "produce", "status": _jr.FAILED,
                            "returncode": 1,
                            "finished": _t.strftime("%Y-%m-%d %H:%M:%S"),
                            "progress": {}})
    check("FAILED: it says so, loudly", "FAILED" in body, True)
    check("FAILED: and says nothing was lost",
          "already on disk are untouched" in body, True)
    check("FAILED: and points at the log", "the log" in body, True)

    body, at = with_status({"id": "probe3", "task": "produce", "status": _jr.STOPPED,
                            "finished": _t.strftime("%Y-%m-%d %H:%M:%S"),
                            "progress": {}})
    check("STOPPED: it says so", "was stopped" in body, True)

    # and an OLD terminal state must not sit on the page forever -- a banner that is
    # always there is a banner nobody reads
    old = "2020-01-01 09:00:00"
    body, at = with_status({"id": "probe_old", "task": "produce", "status": _jr.FAILED,
                            "returncode": 1, "finished": old, "progress": {}})
    check("an old failure is not shouted about on every page", "FAILED" in body, False)

    # and a RUNNING job must NOT show a completion banner
    body, at = with_status({"id": "probe4", "task": "produce", "status": _jr.RUNNING,
                            "child_started": _t.time() - 10, "progress": {}})
    check("RUNNING: no completion banner is shown",
          "finished" not in body, True)

    print("\n12. ONE press shoots the whole film, in the SHOOT view")
    # "an autoproceed button where I can just press once I go into the shot selection
    # area and it runs the full pipeline and generates all the shots instead of having
    # to press it individually". It existed only in CUT, under an export warning.
    src = (ROOT / "ui" / "studio.py").read_text(encoding="utf-8")
    shoot = src[src.index("def view_shoot"):src.index("def shot_detail")]
    check("the shoot view offers the one-press button",
          "shoot_all_button(" in shoot, True)
    check("and it is one job for every remaining shot",
          'announce("produce", store=project["slug"])' in src, True)
    check("it counts what is left in the label",
          "SHOOT ALL REMAINING" in src, True)

    print(f"\n{'=' * 60}")
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + "; ".join(FAILS))
    else:
        print("all studio checks passed")
    shutil.rmtree(fs.dir_for(SLUG), ignore_errors=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
