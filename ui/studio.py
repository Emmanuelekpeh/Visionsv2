"""
ui/studio.py -- the studio.

    streamlit run ui/studio.py

DESIGN NOTES (read this before adding a control)
------------------------------------------------
The previous revisions failed the same way twice, and it is worth naming because it
is not a styling problem.

Revision 1 (`ui/app.py`) was SEVEN STAGES IN A LINE with five gates: STORY, PLAN,
CAMERA, GRADE, RENDER, TIMELINE, AUDIT. Every stage was a page, so you could never
see a change and its consequence at once, and all the creative decisions were asked
for before a single pixel existed.

Revision 2 (`ui/workbench.py`) fixed the data model and then did the same thing
again in a different shape: three always-visible columns, every control on screen at
once, a timeline AND an exposure sheet AND an asset tree AND a job log. Fixed, but
packed. Nine bordered boxes compete for the eye and none of them is the point.

So this file is organised by what a person is DOING, not by what the pipeline
contains. Three verbs:

    WRITE   say what the film is. One field. Then read the shot list it produced.
    SHOOT   the shots, one at a time, in a strip you can see the whole of. This is
            where ~95% of the time goes, so it is where the space goes.
    CUT     look at the film and export it.

Everything else -- scenes, palettes, seeds, the shake, provenance, the writer audit,
the job log, which takes exist -- is real and is reachable, but it is NOT on the main
path. It lives in Settings and in the sidebar. A control that is always visible is a
control that is always making a demand.

RULES for this file. They are not style preferences; each one is a specific failure
from revision 1 or 2.

  1. ONE PRIMARY ACTION PER VIEW. If a view has two equally weighted buttons, the
     view has no point of view. "Generate" outranks "delete", and delete is not a
     button on the canvas at all.
  2. NOTHING NESTED MORE THAN TWO DEEP. `st.columns` inside `st.columns` is how you
     get a 90px-wide selectbox. Two, maximum.
  3. NO BORDERED BOX UNLESS IT HOLDS A THING. Expanders are for things you opened
     on purpose, never for structure. Structure is whitespace and a rule.
  4. STATE IS A WORD AND A MARK, NOT A COLOUR. "needs the model" is legible to
     everyone; a red dot is legible to nobody and shouts at everybody. Colour is
     reserved for the ONE thing on screen that is actually wrong.
  5. THE PRICE COMES BEFORE THE BUTTON. Generating a shot costs 9-14 minutes of this
     machine's CPU. That number is on screen above the button, always, measured from
     the takes already on disk -- never after the fact in a log.
  6. SAY WHAT HAPPENED. Every job that changes the production names what it wrote,
     in the production, and it is visible without opening a log.

Conventions taken from the tools that do this well (see REDESIGN.md): the timeline is
the project and renders are views of it (every NLE); a take is never overwritten, it
is an audition you can flip between (Final Cut auditions, Resolve take selector);
cost shown on the button before you commit (Kling); generation runs detached while
you keep working (Runway; nobody else in the AI family). One thing is deliberately
NOT taken: in an NLE a render is a disposable cache, but here it is 10 minutes, so
a take is the most valuable thing in the project and the UI says so.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_home"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch_home"))

import film_store as fs                                          # noqa: E402
import job_runner as jr                                          # noqa: E402
import keyframe as kf                                            # noqa: E402

FILMS = ROOT / "output" / "film"
_V = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
FULL = {"width": "stretch"} if _V >= (1, 49) else {"use_container_width": True}

# One word per state. This is the whole status vocabulary of the studio.
STATE_WORD = {
    "unrendered": "not shot",
    "stale": "needs re-shooting",
    "restale": "needs re-grading",
    "rendered": "shot",
    "approved": "locked",
}
# A mark, so the strip is scannable without colour. A filled cell is done, a hollow
# one is not, a half is a cheap fix. Same information as the word, one glyph.
STATE_MARK = {"unrendered": "·", "stale": "!", "restale": "~",
              "rendered": "o", "approved": "x"}


# --------------------------------------------------------------------------- #
# a little typography. Streamlit ships one look; these are the two rules that make
# a dense page readable, and they are the only CSS in the file.
# --------------------------------------------------------------------------- #

CSS = """
<style>
  /* THE LAYOUT IS A GRID, NOT A COLUMN OF PROSE. `layout="wide"` was set and then
     immediately undone by a `max-width: 1180px` on the container, which left the
     right 40% of a 1920px window empty while the shot list and the film strip
     squeezed into the left. A measure protects a paragraph; it strangles a grid.
     So: no cap on the container, and a `.studio-prose` class for the few places
     that are actually read as text. */
  .block-container { max-width: none; padding-top: 2.2rem; }
  .studio-prose { max-width: 68ch; }
  /* The section label: small, spaced, quiet. Structure with a rule and a label
     instead of a box, so nothing looks clickable that is not. */
  .studio-label {
    font-size: .68rem; letter-spacing: .16em; text-transform: uppercase;
    opacity: .45; margin: 1.5rem 0 .4rem; font-weight: 600;
  }
  .studio-label:first-child { margin-top: 0; }
  /* The one number that matters: what the next render costs. */
  .studio-price { font-size: .82rem; opacity: .72; margin: .1rem 0 .5rem;
                  max-width: 82ch; }
  /* The film strip cell. A hairline, a word, no fill, no shadow. */
  .studio-cell {
    border: 1px solid rgba(230,230,226,.13); border-radius: 3px;
    padding: .35rem .45rem; font-size: .74rem; line-height: 1.35;
  }
  .studio-cell b { font-weight: 600; }
  .studio-cell span { opacity: .55; }
  /* Reduce Streamlit's own chrome: it is the loudest thing on a dense page. */
  div[data-testid="stExpander"] details { border: none; }
  div[data-testid="stExpander"] summary { font-size: .78rem; opacity: .65; }
  div[data-testid="stMetricValue"] { font-size: 1.35rem; }
  hr { margin: 1.4rem 0; opacity: .18; }
  button[kind="secondary"] { border-color: rgba(230,230,226,.18); }
</style>
"""


def rule(label: str = "") -> None:
    """A section. The label is the structure; there is no box."""
    st.markdown(f'<div class="studio-label">{label}</div>', unsafe_allow_html=True)


def prose(text: str, caption: bool = True) -> None:
    """Prose, at a readable measure. Everything else uses the full width."""
    (st.caption if caption else st.markdown)(
        f'<div class="studio-prose">{text}</div>', unsafe_allow_html=True)


def price(text: str) -> None:
    st.markdown(f'<div class="studio-price">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# reads. Everything is a file on disk, so everything is cheap and cacheable.
# --------------------------------------------------------------------------- #

def _ttl_cache(ttl: float):
    """A thread-safe TTL cache with the `.clear()` the studio already calls.

    WHY NOT `st.cache_data(ttl=...)`: the page is a 2-second fragment AND the main
    script, running on different threads. Streamlit's TTL store (cachetools.TTLCache)
    is not thread-safe, and an entry expiring between one thread's membership test
    and its read raises `KeyError: '<hash>'` out of `thumb()` -- which crashed the
    shoot view. One lock around lookup and store removes the race.
    """
    import functools
    import threading

    def deco(fn):
        lock = threading.Lock()
        store: Dict = {}

        @functools.wraps(fn)
        def wrapped(*args, **kw):
            key = (args, tuple(sorted(kw.items())))
            now = time.time()
            with lock:
                hit = store.get(key)
                if hit is not None and now - hit[0] <= ttl:
                    return hit[1]
            value = fn(*args, **kw)          # compute outside the lock
            with lock:
                store[key] = (time.time(), value)
                if len(store) > 512:         # bound memory: drop the oldest half
                    for k in sorted(store, key=lambda k: store[k][0])[:256]:
                        store.pop(k, None)
            return value

        def clear() -> None:
            with lock:
                store.clear()

        wrapped.clear = clear
        return wrapped
    return deco


def _snap(project: Dict) -> str:
    """The project as it would be written, for 'did this pass change anything'.

    The page re-runs every 2 s. A view that saves unconditionally rewrites
    project.json from a snapshot loaded moments earlier, which can overwrite what a
    running job (draft / produce) saved in between."""
    return json.dumps({k: v for k, v in project.items() if not k.startswith("_")},
                      sort_keys=True, default=str)


def load(slug: str) -> Dict:
    """The production, read fresh. A small JSON file: not worth a cache, and a cache
    here was one more thing the fragment and the main script raced on."""
    return fs.load(slug)


load.clear = lambda: None          # kept: callers invalidate it after a write


@_ttl_cache(ttl=4)
def thumb(take_dir: Optional[str], size: int = 320):
    """One frame, for a thumbnail. Its own function with its own signature -- a
    cached list-returning function shared with a single-frame call is how a list
    turns up where a frame was expected."""
    if not take_dir:
        return None
    from PIL import Image
    d = Path(take_dir) / "frames"
    files = sorted(d.glob("f*.png")) if d.exists() else []
    if not files:
        return None
    try:
        return Image.open(files[len(files) // 2]).convert("RGB").resize((size, size))
    except Exception:
        return None


@_ttl_cache(ttl=4)
def strip_frames(take_dir: str, n: int = 4, size: int = 210) -> List:
    """A few frames of a take, for the viewer. Four is enough to see whether the
    motion is real; more is a filmstrip nobody reads."""
    from PIL import Image
    d = Path(take_dir) / "frames"
    files = sorted(d.glob("f*.png")) if d.exists() else []
    if not files:
        return []
    picks = (files if len(files) <= n else
             [files[round(i * (len(files) - 1) / (n - 1))] for i in range(n)])
    out = []
    for p in picks:
        try:
            out.append(Image.open(p).convert("RGB").resize((size, size)))
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #
# the job runner, folded into the studio's own voice
# --------------------------------------------------------------------------- #

def announce(task: str, **params) -> bool:
    # NO PREVIS GATE. It used to block every render until a previs passed, and it
    # auto-started one. Previs draws its stills with the 4-step sampler, so it does
    # not show what the render will be (the full sampler and the video are different
    # pixels), and the clip does not start from its stills. A check that cannot
    # predict the result must not block it. Studio has no previs at all now (the
    # user does not want it); `previs.py` stays on disk, unused by Studio.
    try:
        jr.start(task, **params)
        return True
    except RuntimeError as exc:
        st.warning(str(exc))
        return False


def busy() -> bool:
    return jr.status().get("status") in (jr.QUEUED, jr.RUNNING)


def _hms(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60} min"
    return f"{s / 3600:.1f} h"


# Rough cost of the stages that are not generation. Measured from the pipeline's own
# logs on this machine: RAFT + shake + grade is seconds per shot, assembly ~97 s. Used
# only for an estimate, and labelled as one.
FINISH_S = 25.0
ASSEMBLE_S = 100.0


def progress_facts(job: Dict) -> Dict:
    """Everything the bar shows, as DATA. No Streamlit calls.

    Separated from the rendering on purpose. A progress indicator is arithmetic --
    fraction done, seconds elapsed, seconds left -- and arithmetic that is only
    reachable through a rendered page cannot be tested without a browser and a
    subprocess. This is the part that has to be RIGHT; `progress_view` is the part
    that has to look right, and it does nothing but print these numbers.
    """
    pr = job.get("progress") or {}
    phase = pr.get("phase") or "starting"
    state = job.get("status")
    done, total = pr.get("done"), pr.get("total")
    # An elapsed clock exists only while the job RUNS. A finished job keeps its
    # `child_started` in state.json, and reporting "running 40 min" for something that
    # stopped is exactly the kind of stale claim this UI is meant to avoid.
    t0 = job.get("child_started")
    elapsed = (time.time() - float(t0)) if (t0 and state == jr.RUNNING) else None

    phase_word = {
        "planning": "reading the production",
        "drafting": "writing the shot list",
        "ideas": "the scouts are generating ideas",
        "showrunner": "the showrunner is writing the shot list",
        "loading the model": "loading the model and its motion LoRAs",
        "generating": "generating",
        "finishing": "re-finishing takes",
        "grading": "grading",
        "assembling": "assembling the film",
        "finished": "done",
    }.get(phase, phase)
    line = phase_word
    if state == jr.QUEUED:
        line = "waiting to start"
    if phase == "generating" and done and total:
        line = f"generating shot {min(done + 1, total)} of {total}"

    # the OUTER bar: shots. Only where a shot loop actually exists.
    outer = None
    if total and phase in ("generating", "finishing", "grading", "assembling"):
        outer = (min(1.0, (done or 0) / total), f"{done or 0}/{total} shots")
    elif total and phase in ("ideas", "drafting"):
        # The scouts run SERIALLY and nothing used to be printed until all of them
        # finished, so a 3-minute run showed nothing and could not be told from a hang.
        # `story.py` now announces each farm; this turns that into a real bar.
        outer = (min(1.0, (done or 0) / total),
                 f"{(done or 0)}/{total} scouts")
    elif phase == "showrunner" and pr.get("token_budget") and state == jr.RUNNING:
        # The showrunner's reply is STREAMED, so this is a real count: tokens written
        # against the cap. The cap is a ceiling (a model that finishes earlier stops
        # earlier), so the label says "up to", not "left".
        tok, cap = int(pr.get("tokens") or 0), int(pr["token_budget"])
        attempt = (f" · attempt {pr['attempt']}/{pr['attempt_total']}"
                   if pr.get("attempt") and pr.get("attempt_total") else "")
        rate = f" · {pr['tok_per_s']:.1f} tok/s" if pr.get("tok_per_s") else ""
        left = (f" · up to {pr['minutes_left_max']} min left"
                if tok and pr.get("minutes_left_max") is not None else "")
        outer = (min(1.0, tok / cap) if cap else 0.0,
                 f"{tok}/{cap} tokens{rate}{left}{attempt}")

    # the INNER bar: this phase against a cost -- measured when the run has produced
    # a completed shot, a labelled estimate when it has not.
    measured = pr.get("mean_shot_seconds")
    est = None
    if phase == "generating":
        # the CURRENT shot's own tier price when the run knows it (a fast shot is
        # ~1/4 of a full one), else the run's mean, else 11 min
        est = pr.get("current_est_s") or measured or 660.0
    elif phase == "loading the model":
        est = 240.0
    elif phase == "finishing":
        est = FINISH_S * max(1, total or 1)
    elif phase == "grading":
        est = 20.0
    elif phase == "assembling":
        est = ASSEMBLE_S
    inner = None
    if est and elapsed is not None and state == jr.RUNNING:
        inner = (min(0.99, elapsed / est), elapsed, est, bool(measured))

    eta = tail = None
    if state == jr.RUNNING and phase == "generating" and total \
            and done is not None and pr.get("eta_seconds") is not None:
        eta = (int(pr.get("eta_shots") or 0), float(pr["eta_seconds"]))
    elif state == jr.RUNNING and phase == "generating" and total and measured \
            and done is not None:
        left = max(0, total - done)
        eta = (left, left * float(measured))
    if (total and done is not None and done >= total and pr.get("to_refinish")):
        tail = FINISH_S * pr["to_refinish"] + ASSEMBLE_S

    return {"line": line, "outer": outer, "inner": inner, "eta": eta, "tail": tail,
            "elapsed": elapsed, "state": state, "last": pr.get("last"),
            "measured": bool(measured),
            "modelled": phase in ("drafting", "ideas", "showrunner") and not outer}


def progress_view(job: Dict) -> None:
    """The bar. Two levels, because that is the honest shape of the work.

    ONE bar can only ever be one of two lies here. Per-shot progress between prints is
    invisible, because the pipeline prints nothing for the 9-14 minutes a shot takes.
    And a bar that only moves on shot boundaries sits at 0% for a quarter of an hour,
    which reads as a hang -- the single worst thing a progress indicator can do.

    So: the OUTER bar is shots (discrete and truthful, it only moves when a shot
    lands), and the INNER bar is the current phase, advancing on wall clock against a
    mean measured from this run's own completed shots. The inner bar is an estimate
    and is worded as one; it never claims to know something it does not.
    """
    f = progress_facts(job)
    st.markdown(f"**{job.get('task')}** \u2014 {f['line']}")
    if f["outer"]:
        st.progress(f["outer"][0], text=f["outer"][1])
    elif f["modelled"]:
        # no denominator exists for a language model writing; words instead of a
        # number, because a fake percentage is worse than none
        st.caption("no progress figure for this \u2014 it is a model writing, "
                   "not a loop")
    if f["inner"]:
        frac, elapsed, est, measured = f["inner"]
        st.progress(frac, text=f"this step: ~{_hms(elapsed)} of ~{_hms(est)}"
                               + ("" if measured else " (estimate)"))
    if f["elapsed"] is not None and f["state"] == jr.RUNNING:
        bits = [f"running {_hms(f['elapsed'])}"]
        if f["eta"] and f["eta"][0]:
            bits.append(f"~{_hms(f['eta'][1])} left for {f['eta'][0]} shot(s)")
        if f["tail"]:
            bits.append(f"~{_hms(f['tail'])} of finishing and assembly")
        st.caption(" \u00b7 ".join(bits))
    if f["last"]:
        st.caption(f["last"])
    if f["state"] in (jr.QUEUED, jr.RUNNING):
        if st.button("stop it", key="stop_job"):
            jr.stop()
            st.rerun()


def activity() -> None:
    """The job, in the sidebar, where it is not in the way.

    It polls on a 2-second fragment so the bars actually move: the supervisor writes
    state.json every 1.5 s, so anything slower than that makes the UI look stalled
    when it is not.
    """
    job = jr.status()
    state = job.get("status")
    if state == "idle":
        st.caption("nothing running")
        return
    progress_view(job)
    with st.expander("the log"):
        st.text(jr.tail(job["id"], 40))


@st.fragment(run_every=2)
def activity_panel() -> None:
    """Re-render the bar every 2 s.

    The supervisor writes `state.json` every POLL_S (1.5 s), so polling slower than
    that makes a moving bar look stalled. This is a read of one small JSON file.
    """
    activity()


def say_what_happened(project: Dict) -> None:
    """The last thing this production did, named. Not a log -- a sentence."""
    hist = project.get("history") or []
    if not hist:
        return
    last = hist[-1]
    act = last.get("act")
    if act == "draft":
        st.caption(f"shot list written {last.get('at', '')} — "
                   f"{last.get('shots')} shots, nothing rendered yet")
    elif act == "generate":
        st.caption(f"shot {last.get('shot')} shot as {last.get('take')} "
                   f"{last.get('at', '')} — {last.get('seconds')}s")
    elif act == "render":
        st.caption(f"cut and exported as {last.get('name')} {last.get('at', '')} — "
                   f"{last.get('seconds')}s")


# --------------------------------------------------------------------------- #
# WRITE -- say what the film is, then read the list it produced
# --------------------------------------------------------------------------- #

def _cached_proposal_for(concept: str) -> Optional[Path]:
    """Is there already a proposal on disk for this concept?

    `story.write_shots_ensemble` caches by concept slug, so this is the difference
    between "the writers ran again" and "you got the same file back". The UI has to
    be able to say which.
    """
    import re
    d = FILMS / "proposals"
    if not d.exists() or not concept.strip():
        return None
    slug = re.sub(r"\W+", "-", concept.lower()).strip("-")[:60]
    hits = [p for p in d.glob(f"{slug}__*.json")
            if "__keep" not in p.name]          # a kept-set variant is a different
    if not hits:                                # proposal, not this one
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def _pending_draft_for(project: Dict, concept: str) -> Optional[Dict]:
    """A refused draft from the last run, if it belongs to this concept.

    The write view has to say the loop has somewhere to continue from BEFORE the
    button is pressed. Otherwise "press write new ideas" reads as "start over",
    and the whole point of keeping the refused writing is invisible.
    """
    try:
        p = fs.dir_for(project["slug"]) / "draft_attempt.json"
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (data.get("concept") or "").strip().lower() != (concept or "").strip().lower():
        return None
    if not data.get("shots"):
        return None
    return data


def _ago_text(at: str) -> str:
    """`_ago` for a timestamp string instead of a file's mtime."""
    try:
        secs = max(0.0, time.time() - time.mktime(
            time.strptime(str(at), "%Y-%m-%d %H:%M:%S")))
    except Exception:
        return ""
    if secs < 90:
        return "just now"
    if secs < 5400:
        return f"{secs / 60:.0f} min ago"
    if secs < 172800:
        return f"{secs / 3600:.0f} h ago"
    return f"{secs / 86400:.0f} days ago"


def _ago(p: Path) -> str:
    secs = max(0, time.time() - p.stat().st_mtime)
    if secs < 90:
        return "just now"
    if secs < 5400:
        return f"{secs / 60:.0f} min ago"
    if secs < 172800:
        return f"{secs / 3600:.0f} h ago"
    return f"{secs / 86400:.0f} days ago"


def _job_fresh(job: Dict, within_s: float = 1800) -> bool:
    """Did this job finish RECENTLY?

    A completion banner is news; an old one is clutter. Without this, a job that failed
    last week would leave a red panel at the top of every view forever -- and a banner
    that is always there is a banner nobody reads. 30 minutes is long enough to come
    back from a coffee and still see it.
    """
    jid = job.get("id")
    if not jid:
        return False
    if job.get("finished"):
        try:
            done = time.mktime(time.strptime(str(job["finished"]),
                                             "%Y-%m-%d %H:%M:%S"))
            return (time.time() - done) <= within_s
        except Exception:
            pass
    try:                                    # the state file's own mtime is the fallback
        p = jr.state_path(jid)
        return p.exists() and (time.time() - p.stat().st_mtime) <= within_s
    except Exception:
        return False


def completion_banner(project: Dict, job: Optional[Dict] = None) -> None:
    """Say, unmistakably, that a job HAS FINISHED and what it produced.

    The complaint this answers, verbatim: *"after what I was trying to generate has
    completed, I don't realize that it has generated because the UI does not refresh or
    it does not update to say that's oh the generation has happened yet."*

    The UI already refreshed (the content fragment re-reads every 2 s) and it was still
    not enough, because nothing on the page SAID SO. A button quietly changing from
    "shoot it" to "shoot another take" is not an announcement. Three things fix that:

      * a banner that names the outcome and the artifact,
      * a TOAST the instant a running job becomes a finished one, which is noticed even
        if you are looking at the shot strip rather than the sidebar,
      * and it is driven by the job's own terminal state, so it is correct after a page
        reload instead of depending on anything this session happened to see.

    It reports the last job the RUNNER knows about, not "the last job you started", so a
    job started from another tab or from the terminal is announced here too.

    `job` is injectable so the completion path can be tested without rendering a real
    job: `jr.status()` returns whichever state file on disk is newest, which makes it
    useless as a fixture.
    """
    if job is None:
        job = jr.status()
    state = job.get("status")
    jid = job.get("id")

    # --- the moment it lands ------------------------------------------------- #
    if state in (jr.RUNNING, jr.QUEUED):
        st.session_state["_busy_job"] = jid          # remember what we are watching
        return
    if jid and st.session_state.get("_busy_job") == jid:
        # it was running in this session a moment ago and now it is not: announce it
        st.session_state["_busy_job"] = None
        if state == jr.DONE:
            st.toast(f"{job.get('task')} finished", icon="✅")
        elif state == jr.FAILED:
            st.toast(f"{job.get('task')} FAILED — see the log", icon="⚠️")

    if state == jr.DONE and _job_fresh(job):
        did = _what_it_made(job)
        st.success(f"**{job.get('task')} finished** — {did}"
                   + (f"  ·  {job.get('finished', '')[11:16]}"
                      if job.get("finished") else ""))
    elif state == jr.FAILED and _job_fresh(job):
        # A refusal is a DECISION, not a crash: the story gate exits 2 on purpose,
        # and the reason is in the log's last lines. Show them here so the user
        # does not have to hunt through a file to learn what to do next.
        tail = _log_tail(job)
        st.error(f"**{job.get('task')} FAILED** (exit {job.get('returncode')}). "
                 f"Nothing was lost — the takes already on disk are untouched."
                 + (f"\n\n```\n{tail}\n```" if tail else
                    " Open *the log* in the sidebar to see why."))
    elif state == jr.STOPPED and _job_fresh(job):
        st.warning(f"**{job.get('task')} was stopped.** Kept whatever had already "
                   f"landed on disk.")
    elif state == jr.LOST and _job_fresh(job):
        st.warning(f"**{job.get('task')} was lost** — the supervisor went away without "
                   f"finishing. {job.get('message') or ''}")


def _log_tail(job: Dict, n: int = 6) -> str:
    """The last few meaningful lines of a failed job's log, for the banner.

    A story-gate refusal exits 2 and prints what failed and what to do next; the
    log is the only place that lives, and "open the log" was a scavenger hunt.
    Each line is trimmed so a traceback does not paste 40 lines into the page.
    """
    jid = job.get("id")
    if not jid:
        return ""
    try:
        text = jr.log_path(jid).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    return "\n".join(l[-160:] for l in lines[-n:])


def _what_it_made(job: Dict) -> str:
    """What a finished job actually produced, in words, from its own progress."""
    pr = job.get("progress") or {}
    task = job.get("task")
    shots = pr.get("shots")
    if task == "produce":
        if shots:
            return f"{shots} shot(s) generated"
        gen = pr.get("done")
        if gen:
            return f"{gen} shot(s) generated"
        if pr.get("to_refinish"):
            return f"{pr['to_refinish']} shot(s) re-graded and the film cut"
        return "the film was cut"
    if task == "draft":
        n = shots or pr.get("shots")
        return (f"{n} shots written into the film" if n
                else "the shot list was written")
    if task == "story":
        return "the shot list was written"
    if task == "review":
        if pr.get("fail") is not None:
            return (f"{pr['fail']} fail · {pr.get('review', 0)} to look at · "
                    f"{pr.get('passed', 0)} pass · {pr.get('unreviewed', 0)} "
                    f"unreviewed — open *review the takes* in the shoot view")
        return "the takes were reviewed"
    if task == "film_audit":
        return (f"{pr.get('warnings', 0)} warning(s), {pr.get('notes', 0)} note(s) — "
                f"open *audit the delivered film* in the cut view")
    if task == "style":
        return (f"light and palette set for {pr['scenes']} scene(s); the prompts "
                f"changed, so those takes are stale" if pr.get("scenes")
                else "the style pass ran")
    if task == "retime":
        return (f"rhythm CV {pr['cv_before']} → {pr['cv_after']}; long holds will "
                f"re-finish on the next export" if pr.get("cv_after") is not None
                else "the cut was retimed")
    return "finished"


def shoot_all_button(project: Dict, key: str = "shoot_all") -> None:
    """ONE press, whole film. The user asked for exactly this, in the SHOOT view.

    It was in `cut`, under a warning about exporting, which is not where anyone looks
    when they want to make the shots. One job renders every shot that needs it, shares
    a single model load, and then grades and cuts the film -- so there is nothing to
    press again afterwards.
    """
    stt = fs.film_status(project)
    if not stt["to_generate"] and not stt["to_refinish"]:
        st.success("Every shot has a take. Nothing to shoot — the film is ready to cut.")
        return
    if stt["to_generate"]:
        tiers = stt.get("est_by_tier") or {}
        est = (" about " + _hms(stt["est_generate_s"])
               + " (" + " + ".join(
                   f"{v['shots']} {k} × ~{v['each_s'] / 60:.0f} min"
                   + ("" if v["measured"] else " est.") for k, v in tiers.items())
               + ")") if stt["est_generate_s"] else " roughly 45 min for 4 shots"
        price(f"{stt['to_generate']} shot(s) need generating{est}, plus seconds each "
              f"to grade. One job: the model loads once instead of {stt['to_generate']} "
              f"times.")
    else:
        price(f"Nothing needs generating. {stt['to_refinish']} shot(s) need a re-grade "
              f"— seconds each.")
    if st.button(f"▶  SHOOT ALL REMAINING ({stt['to_generate']}"
                 + (f" + {stt['to_refinish']} re-grade" if stt["to_refinish"] else "")
                 + ")", type="primary", key=key, disabled=busy(), **FULL):
        if announce("produce", store=project["slug"]):
            st.rerun()


def character_panel(project: Dict) -> None:
    """Who the figure is, once, for every shot (also used by new drafts)."""
    st.caption("Every shot's prompt is written separately, so without this each shot "
               "draws a NEW person: the skydiver started as a man and ended as a "
               "woman. One word here goes in front of the figure in every shot "
               "(\"male\" → \"a male skydiver\"), and the words to avoid go in the "
               "full-quality sampler's negative prompt. New drafts use it too.")
    ch = project.get("character") or {}
    with st.form("character_form"):
        cc = st.columns([1, 3])
        desc = cc[0].text_input("who", value=ch.get("descriptor", ""),
                                placeholder="male",
                                help="one adjective or short phrase, put in front of "
                                     "the figure's noun in every shot")
        avoid = cc[1].text_input("never draw", value=ch.get("avoid", ""),
                                 placeholder="woman, female, long hair",
                                 help="goes in the negative prompt. Only full-quality "
                                      "shots obey it (fast shots ignore negatives)")
        if st.form_submit_button("apply to every shot"):
            if desc.strip():
                n = fs.set_character(project, desc, avoid)
                st.cache_data.clear()
                st.session_state["character_msg"] = (
                    f"{n} shot prompt(s) rewritten. Those shots will be re-shot; "
                    f"shots that never name the figure were left alone.")
            else:
                project.pop("character", None)
                fs.drop_stale_cache(project)
                fs.save(project)
                st.cache_data.clear()
                st.session_state["character_msg"] = (
                    "character cleared. Prompts already rewritten stay as they are.")
            st.rerun()
    if st.session_state.get("character_msg"):
        st.info(st.session_state.pop("character_msg"))


def view_write(project: Dict) -> None:
    rows = fs.shots_in_order(project)
    # the draft job lands HERE, so this is where being told it finished matters most
    completion_banner(project)
    st.markdown("## what is the film?")
    st.caption("One sentence. The writers work from this and nothing else.")

    concept = st.text_area("concept", value=project.get("concept", ""), height=84,
                           key="concept_box", label_visibility="collapsed",
                           placeholder="what it is about, in the plainest words you "
                                       "have")

    # LENGTH FIRST, SHOTS DERIVED. The renderer supplies (2*frames-1)/fps seconds
    # of real motion per shot, so the only honest way to make a longer film is
    # more shots -- never longer holds. The studio asks for the target and does
    # the arithmetic, including the render price.
    import sequence_direction as sd
    _stg = project.get("settings") or {}
    _frames = int(_stg.get("frames") or 16)
    _fps = int(_stg.get("fps") or 16)
    _natural = (2 * _frames - 1) / _fps
    _cur_len = (sum(fs.hold_frames(s) for s in rows) / _fps) if rows else 30.0
    c = st.columns([1, 1, 2, 2])
    target_s = c[0].number_input("target length (s)", 4, 120,
                                 int(round(max(4, min(120, _cur_len)))), 2,
                                 help="filmed seconds. The shot count is derived "
                                      "from the renderer's real motion budget.")
    est = sd.estimate(target_s, _frames, _fps)
    n_shots = est["shots"]
    price(f"target {target_s}s → {n_shots} shots ≈ "
          f"{est['planned_seconds']:.0f}s planned · "
          f"{est['render_minutes']:.0f} min of rendering if every shot is fast, "
          f"up to ~{n_shots * fs.TIER_DEFAULT_S['full'] / 60:.0f} min if every "
          f"shot is full quality (medium and close shots are) "
          f"({n_shots} × {_natural:.2f}s of generated motion). "
          f"Longer needs more shots, not longer holds.")
    fresh = c[1].checkbox("write new ideas", value=True,
                          help="ON: the writers run again. They plan 2-3 held "
                               "scenes first (one place + one figure each), repair "
                               "a refused draft, extend a short list to the "
                               "requested count, re-ask once if the story holds "
                               "but no scene does, and rewrite every prompt as ONE "
                               "concrete image (a few minutes to ~40 min on CPU). "
                               "OFF: if a proposal for this concept is already on "
                               "disk you get that one back, unchanged, in seconds.")
    adopt_broken = c[2].checkbox("adopt a broken story", value=False,
                                 help="OFF by default. A draft whose beats go "
                                      "backwards is REFUSED, so the production is "
                                      "left untouched and the refused writing is "
                                      "kept for the next run to repair. Tick this to "
                                      "keep the list anyway.")

    # THE THING THAT MADE A RE-RUN LOOK LIKE IT DID NOTHING. The writers' output is
    # cached by concept, so pressing the button twice with the same concept returns
    # the SAME proposal, word for word, in a second or two. Nothing on the old page
    # said so, so a cached hit was indistinguishable from a fresh run. If a proposal
    # for this concept is already on disk, say so before the button, not after.
    pending = _pending_draft_for(project, concept) if concept.strip() else None
    prior = _cached_proposal_for(concept) if concept.strip() else None
    if pending:
        price(f"the last draft of this concept was <b>refused</b> "
              f"({len(pending.get('shots') or [])} shots, shape "
              f"{pending.get('story_shape') or '?'}, "
              f"{_ago_text(pending.get('at'))}). Pressing the button feeds that "
              f"writing back to the loop and repairs it, instead of starting from "
              f"nothing.")
    elif prior and not fresh:
        price(f"a proposal for this concept is already on disk "
              f"({prior.stat().st_size // 1024} KB, written {_ago(prior)}). With "
              f"<b>write new ideas</b> off you will get that one back — the same "
              f"shots, the same ideas — in seconds.")
    elif prior:
        price("a proposal for this concept is already on disk; <b>write new "
              "ideas</b> is on, so the writers run again and you get a different "
              "list (the loop can take a few minutes to ~35 min on CPU: writing "
              "passes, coverage, then a prompt pass).")
    else:
        price("no proposal for this concept yet, so the writers will run either "
              "way.")

    selfcheck = st.checkbox(
        "let it re-read its own shot list and fix it", value=True,
        help="After writing, the draft is checked for a story that goes backwards, "
             "absence wording (\"no people\") and the figure missing from the "
             "ending, and the flagged shots are repaired in ONE call that sees the "
             "whole list. Off = faster, no repair.")
    with st.expander("who is the figure? (keeps one person across every shot)",
                     expanded=bool(project.get("character"))):
        character_panel(project)

    if st.button("write the shot list", type="primary", key="draft_btn",
                 disabled=busy() or not concept.strip()):
        if announce("draft", store=project["slug"], concept=concept.strip(),
                    n_shots=int(n_shots), fresh=bool(fresh),
                    allow_broken_story=bool(adopt_broken),
                    no_self_critique=not selfcheck):
            project["concept"] = concept.strip()
            fs.save(project)
            st.cache_data.clear()
            st.rerun()
    if busy():
        price("a job is already running — one at a time, this machine cannot do two")

    if not rows:
        st.info("No shot list yet. Write the concept above and press the button; it "
                "runs in the background (a few minutes, longer if the story needs "
                "repair passes), so you can leave this page open and it will fill "
                "in.")
        return

    st.divider()
    rule(f"the shot list · {len(rows)} shots · "
         f"{fs.duration(project):.1f}s")
    if project.get("logline"):
        prose(project["logline"])
    if project.get("problems"):
        # the writer's own complaint about what it produced. Worth reading, and it is
        # one line of prose, so it gets the measure.
        prose("the writer's own note: " + "; ".join(project["problems"]))
        st.markdown('<div class="studio-prose"></div>', unsafe_allow_html=True)
    # what adoption did to the writer's timing. The store keeps the writer's text
    # and re-times the cut to the renderer, so this line is the difference between
    # what the model asked for and what will be shot.
    pace = project.get("pacing") or {}
    if pace.get("planned_holds"):
        writer_h = " ".join(f"{float(x):g}" for x in pace.get("writer_holds") or [])
        planned_h = " ".join(f"{float(x):g}" for x in pace["planned_holds"])
        price(f"the cut was auto-paced at adoption: "
              f"{len(pace.get('changes') or [])} timing/framing repair(s), built "
              f"from {float(pace.get('natural_s') or 0):.2f}s of generated motion "
              f"per shot. Holds {writer_h}s → {planned_h}s. "
              f"{int(pace.get('movers') or 0)} moving shot(s).")
    if pace.get("notes"):
        prose("pacing notes: " + "; ".join(pace["notes"]))

    # --- visual organisation: the hold/change spine ------------------------- #
    # Where does this film hold one place and figure, and where does it change?
    # This is the one picture that answers it: place bars, figure, and the cuts
    # that mean a change. It is read-only here; the rearrangement lives in
    # `visual_org.py --suggest` and the workbench's ORGANISE stage.
    try:
        import visual_org as vo
        runs = vo.organise(rows)
        org = vo.diagnose(rows, runs, mode="auto")
        png = Path("output") / "visual_org" / f"{project['slug']}.png"
        vo.strip_png(rows, runs, org["mode"], png, title=project["slug"])
        st.markdown("**visual organisation** · hold or change")
        st.image(str(png))
        st.caption(
            f"{org['stats']['holds']} held place(s), "
            f"{org['stats']['singles']} single-shot change(s), "
            f"{org['stats']['changes']} change(s) "
            f"({org['stats']['changes_per_min']}/min) · within-run prompt "
            f"overlap {org['inside_overlap']} vs {org['across_overlap']} across "
            f"runs · mode **{org['mode']}**")
        for it in org["diagnostics"][:6]:
            line = f"{it['code']}: {it['message']}"
            if it["level"] == "break":
                st.error(line)
            elif it["level"] == "warn":
                st.warning(line)
            else:
                st.info(line)
    except Exception as exc:                      # noqa: BLE001
        st.caption(f"visual organisation unavailable: {type(exc).__name__}: {exc}")

    # --- organisation controls: rearrange and anchor, in the studio ---------- #
    try:
        import visual_org as vo2
        import sequence_direction as sd2
        with st.expander("organisation · what the director chose"):
            st.caption("Automatic. The film is split into movements and each one "
                       "gets the grammar it needs: hold a place/figure for a beat, "
                       "or change fast through a montage run. One film can use "
                       "several. Camera moves are scoped to the movements that can "
                       "carry them.")
            direction = project.get("direction") or {}
            movements = direction.get("movements") or []
            if movements:
                for m in movements:
                    shots_m = m.get("shots") or []
                    st.markdown(
                        f"`shots {shots_m[0]}-{shots_m[-1]}` **{m.get('mode')}** · "
                        f"beat {m.get('beat')} · camera ≤{m.get('camera_budget')} · "
                        f"{m.get('why')}")
                for note in direction.get("notes") or []:
                    st.caption(note)
            else:
                st.caption("No direction stored yet. The next draft creates one, "
                           "or press re-direct to plan this shot list now.")
            if st.button("re-direct automatically", disabled=busy(),
                         key=f"redirect_{project['slug']}"):
                d = sd2.direct_project(project)
                summary = " · ".join(f"{len(m.indices)}x{m.mode}"
                                     for m in d.movements[:6])
                st.success(f"re-directed: {summary}")
                st.cache_data.clear()
                st.rerun()
            st.divider()
            st.caption("Anchors are the words every shot in a hold gets told, so the "
                       "renderer hears the same place/figure on every cut. Baking "
                       "them rewrites the prompt (and correctly stales the take).")
            runs2 = vo2.organise(rows)
            orgp = project.setdefault("organisation", {})
            saved_places = dict(orgp.get("places") or {})
            saved_figures = dict(orgp.get("figures") or {})
            places, figures = {}, {}
            seen_places, seen_figures = set(), set()
            for r in runs2:
                if r.place not in seen_places:
                    seen_places.add(r.place)
                    places[r.place] = st.text_input(
                        f"place {r.place} · visual description",
                        value=saved_places.get(r.place, ""),
                        key=f"anchor_place_{project['slug']}_{r.place}")
                if r.figure and r.figure not in seen_figures:
                    seen_figures.add(r.figure)
                    figures[r.figure] = st.text_input(
                        f"figure {r.figure} · appearance",
                        value=saved_figures.get(r.figure, ""),
                        key=f"anchor_fig_{project['slug']}_{r.figure}")
            if st.button("bake anchors into prompts", disabled=busy(),
                         key=f"anchor_apply_{project['slug']}"):
                orgp["places"] = {k: v.strip() for k, v in places.items() if v.strip()}
                orgp["figures"] = {k: v.strip() for k, v in figures.items() if v.strip()}
                fs.save(project)
                n = vo2.apply_anchors_to_store(project)
                st.success(f"anchors written into {n} prompt(s) · those takes are "
                           f"now stale, which is correct")
                st.rerun()
    except Exception as exc:                      # noqa: BLE001
        st.caption(f"organisation controls unavailable: {type(exc).__name__}: {exc}")

    try:
        finish_panel(project)
    except Exception as exc:                      # noqa: BLE001
        st.caption(f"style/rhythm unavailable: {type(exc).__name__}: {exc}")

    # The list, and nothing else. This is the artifact; everything that produced it
    # is machinery. The scouts' 18 ideas are in Settings for the one time in ten you
    # want to steer the writer by hand.
    #
    # The prompt gets the measure (68ch) while its row uses the full width, because
    # a prompt is READ and a row is a GRID. That is the difference the old layout
    # could not express, and it is what left 40% of a 1920px window empty.
    for i, s in enumerate(rows, 1):
        state = fs.shot_state(project, s)
        take = fs.best_take(project, s)
        c = st.columns([1, 13])
        img = thumb(take["_dir"], 104) if take else None
        if img is not None:
            c[0].image(img)
        else:
            c[0].markdown(f'<div class="studio-cell" style="text-align:center">'
                          f'{STATE_MARK[state]}</div>', unsafe_allow_html=True)
        c[1].markdown(
            f"<span class='studio-prose'>"
            f"<b>{i}. {s.get('beat', '')}</b> · {s.get('scale', '')} · "
            f"{s.get('camera') or 'still'} · {s.get('seconds')}s · "
            f"<span style='opacity:.5'>{STATE_WORD[state]}</span><br>"
            f"<span style='opacity:.8'>{s.get('prompt', '')}</span></span>",
            unsafe_allow_html=True)

    st.divider()
    if st.button("go and shoot it →", type="primary", key="to_shoot"):
        st.session_state["view"] = "shoot"
        st.session_state["shot"] = rows[0]["id"]
        st.rerun()


# --------------------------------------------------------------------------- #
# SHOOT -- the strip, then one shot at a time
# --------------------------------------------------------------------------- #

def film_strip(project: Dict, sel: str) -> None:
    """The whole film as cells you can see at once.

    This is the timeline. It is not an editable track and it does not pretend to be:
    the cut here is 4-12 shots and the decisions are order and length, which the
    strip shows as a running timecode. Reordering lives on the shot, where the shot
    is.
    """
    rows = fs.shots_in_order(project)
    if not rows:
        return
    rule("the film")
    cols = st.columns(len(rows), gap="small")
    t = 0.0
    for i, (col, s) in enumerate(zip(cols, rows)):
        state = fs.shot_state(project, s)
        hold = fs.hold_frames(s)
        here = s["id"] == sel
        with col:
            img = thumb((fs.best_take(project, s) or {}).get("_dir"), 200)
            if img is not None:
                st.image(img)
            # A number, not an icon. Glyph buttons read as decoration and are the
            # first thing to break in a terminal, a log, or a screenshot.
            if st.button(("* " if here else "") + str(i + 1),
                         key=f"strip_{s['id']}",
                         help=f"{s['id']} · {STATE_WORD[state]} · renders "
                              f"{fs.render_tier(s)}", **FULL):
                st.session_state["shot"] = s["id"]
                st.rerun()
            st.markdown(
                f'<div class="studio-cell" style="border:none;padding:0;opacity:'
                f'{"1" if here else ".62"}">{STATE_MARK[state]} {t:.1f}s → '
                f'{t + hold / _fps(project):.1f}s<br>'
                f'<span>{STATE_WORD[state]} · {fs.render_tier(s)}</span></div>',
                unsafe_allow_html=True)
        t += hold / _fps(project)
    film_stt = fs.film_status(project)       # once: it scans every shot's takes
    price(f"{len(rows)} shots · {t:.2f}s total · "
          f"{film_stt['to_generate']} to shoot · "
          f"{film_stt['to_refinish']} to re-grade")


def _fps(project: Dict) -> int:
    return int((project.get("settings") or {}).get("fps") or 16)


# --------------------------------------------------------------------------- #
# the quality tools, read back: take review, film audit, style, rhythm
# --------------------------------------------------------------------------- #
#
# `review`, `output_audit`, `style_pass` and `retime` were standalone CLIs whose
# results nothing read. They are jobs now (job_runner.TASKS), so the one-job-at-a-time
# lock also keeps them from writing a store while a render does. The functions below
# are the READ side and hold no Streamlit calls, so the numbers they report are testable.

def review_facts(project: Dict) -> Dict:
    """The take review as data: gate status, reasons, and the per-shot rows.

    `status` is review.gate's own word (missing / unreadable / stale / failed / ok);
    the rows come from the same review.json, so the page cannot disagree with the gate.
    """
    import json as _json
    import review
    gate = review.gate(project)
    rows: List[Dict] = []
    path = fs.dir_for(project["slug"]) / "review.json"
    if path.exists():
        try:
            rows = _json.loads(path.read_text(encoding="utf-8")).get("rows") or []
        except Exception:                              # noqa: BLE001
            rows = []
    return {"status": gate.get("status"), "ok": bool(gate.get("ok")),
            "reasons": list(gate.get("reasons") or []), "rows": rows}


def audit_facts(project: Dict) -> Dict:
    """The delivered-film audit as data, and whether it describes the CURRENT film.

    output_audit writes metrics.json and has no fingerprint, so staleness is the file
    clock: an audit older than the film it audited is reported as such, not trusted.
    """
    import json as _json
    slug = project["slug"]
    out = Path("output") / "audit" / slug
    path = out / "metrics.json"
    if not path.exists():
        return {"exists": False}
    try:
        metrics = _json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                           # noqa: BLE001
        return {"exists": False, "error": f"{type(exc).__name__}"}
    rec = project.get("deliverable") or {}
    mp4 = (fs.renders_dir(slug) / rec["render"] / "film.mp4") if rec.get("render") \
        else None
    stale = bool(mp4 and mp4.exists() and mp4.stat().st_mtime > path.stat().st_mtime)
    return {"exists": True, "stale": stale, "flags": metrics.get("flags") or [],
            "render": metrics.get("render"),
            "hold_cv": (metrics.get("flow") or {}).get("hold_cv"),
            "total_s": (metrics.get("flow") or {}).get("total_s"),
            "mean_margin": (metrics.get("alignment") or {}).get("mean_margin"),
            "contact": (out / "contact.png") if (out / "contact.png").exists() else None,
            "strip": (out / "flow_strip.png") if (out / "flow_strip.png").exists()
            else None}


def review_panel(project: Dict) -> None:
    """Take review: run it, read the verdicts, record a person's verdict."""
    import review
    slug = project["slug"]
    facts = review_facts(project)
    with st.expander(f"review the takes · {facts['status']}"):
        st.caption("Judges the TAKES, not the shot list: motion, sharpness and a "
                   "boxed-subject evidence sheet per shot. The numbers flag where to "
                   "look; they are not a verdict. A verdict is the model's (if you "
                   "ask for it) or yours, recorded below.")
        c = st.columns([1, 1, 3])
        use_vision = c[1].checkbox("ask the vision model", value=False,
                                   key=f"rv_vision_{slug}",
                                   help=f"{review.DEFAULT_VISION} on the CPU: about a "
                                        "minute per shot, and it is wrong often enough "
                                        "that its answer is recorded but not trusted.")
        if c[0].button("review takes", disabled=busy(), key=f"rv_run_{slug}"):
            if announce("review", store=slug, vision=bool(use_vision)):
                st.rerun()
        if facts["status"] == "ok":
            st.success("review gate: ok")
        elif facts["status"] in ("missing", "stale", "unreadable"):
            st.info(f"review gate: {facts['status']} — "
                    + "; ".join(facts["reasons"]))
        else:
            for r in facts["reasons"]:
                st.warning(r)
        rows = [r for r in facts["rows"] if r.get("take")]
        if not rows:
            return
        ids = [r["shot"] for r in rows]
        pick = st.selectbox(
            "shot", ids, key=f"rv_shot_{slug}",
            format_func=lambda sid: next(
                f"{sid} · {r.get('take')} · {r.get('verdict')}"
                for r in rows if r["shot"] == sid))
        row = next(r for r in rows if r["shot"] == pick)
        sig = (row.get("signals") or {}).get("flags") or []
        st.caption(f"sharpness {row.get('sharpness')} · motion {row.get('mean_diff')}"
                   + (" · " + "; ".join(sig) if sig else "")
                   + (f" · recorded by {row['verdict_by']}"
                      if row.get("verdict_by") else ""))
        ev = row.get("evidence")
        if ev and Path(ev).exists():
            st.image(str(ev))
        why = st.text_input("why (optional)", key=f"rv_why_{slug}_{pick}")
        v = st.columns(3)
        for col, verdict in zip(v, ("pass", "review", "fail")):
            if col.button(f"mark {verdict}", key=f"rv_{verdict}_{slug}_{pick}"):
                review.set_verdict(slug, pick, row.get("take"), verdict, why)
                st.rerun()


def audit_panel(project: Dict) -> None:
    """Delivered-film audit: run it, read the flags and the contact sheet."""
    slug = project["slug"]
    facts = audit_facts(project)
    with st.expander("audit the delivered film"):
        st.caption("Measures the FINISHED pictures against the prompts that made "
                   "them: does each frame match its own shot's prompt, how big are "
                   "the cuts, is the rhythm a metronome. Read-only.")
        c = st.columns([1, 1, 3])
        no_clip = c[1].checkbox("skip CLIP (faster)", value=False,
                                key=f"au_noclip_{slug}")
        if c[0].button("audit the film", disabled=busy() or not (
                project.get("deliverable") or {}).get("render"), key=f"au_run_{slug}",
                help="needs an exported film"):
            if announce("film_audit", store=slug, no_clip=bool(no_clip)):
                st.rerun()
        if not facts["exists"]:
            st.caption("no audit yet" + ("" if (project.get("deliverable") or {})
                                         .get("render") else " — export the film first"))
            return
        if facts["stale"]:
            st.warning("this audit is OLDER than the film on disk — run it again "
                       "before trusting it")
        st.caption(f"render {facts['render']} · {facts['total_s']}s · shot-length CV "
                   f"{facts['hold_cv']}"
                   + (f" · prompt→frame margin {facts['mean_margin']:+.3f}"
                      if facts["mean_margin"] is not None else ""))
        for f in facts["flags"]:
            line = f"{f['code']}: {f['message']}"
            (st.warning if f["level"] == "warn" else st.info)(line)
        if not facts["flags"]:
            st.success("no flags")
        for key in ("contact", "strip"):
            if facts[key]:
                st.image(str(facts[key]))


def finish_panel(project: Dict) -> None:
    """Style (light + palette per scene) and rhythm (retime) -- both rewrite the plan."""
    slug = project["slug"]
    with st.expander("light, palette and rhythm"):
        st.caption("Two passes over an adopted shot list. Both change what is "
                   "rendered, so the takes they touch show as stale — which is "
                   "correct.")
        st.markdown("**style** · one light and palette per scene, fog words out")
        keep_fog = st.checkbox("keep fog/mist/haze words", value=False,
                               key=f"st_fog_{slug}")
        if st.button("apply the style pass", disabled=busy(), key=f"st_run_{slug}"):
            if announce("style", store=slug, keep_fog=bool(keep_fog)):
                st.rerun()
        st.divider()
        st.markdown("**rhythm** · the measured human cut: a fast band, a long band")
        try:
            import retime
            pv = retime.apply(slug, dry_run=True)
            st.caption(f"shot-length CV {pv['before_cv']} → {pv['after_cv']} · "
                       f"{pv['before_total']}s → {pv['after_total']}s · long holds "
                       f"re-finish with RAFT x4-x6 (no re-shoot)")
            st.code("\n".join(f"{sid:<6} {b:>5.2f}s → {a:>5.2f}s  RAFT x{u}"
                              for sid, b, a, u in pv["holds"]), language=None)
        except Exception as exc:                       # noqa: BLE001
            st.caption(f"rhythm preview unavailable: {type(exc).__name__}: {exc}")
        if st.button("apply the rhythm", disabled=busy(), key=f"rt_run_{slug}"):
            if announce("retime", store=slug):
                st.rerun()


def view_shoot(project: Dict) -> None:
    rows = fs.shots_in_order(project)
    if not rows:
        st.markdown("## nothing to shoot yet")
        st.caption("Write the shot list first.")
        if st.button("← go and write it", key="back_write"):
            st.session_state["view"] = "write"
            st.rerun()
        return

    # FIRST THING ON THE PAGE, because this is where you are standing while you wait.
    # A finished job is announced here, above the strip, every time the page renders --
    # including after a reload, since it reads the job's own terminal state.
    completion_banner(project)
    shoot_all_button(project, key="shoot_all_shoot")
    st.divider()

    sel = st.session_state.get("shot")
    if sel not in [s["id"] for s in rows]:
        sel = rows[0]["id"]
        st.session_state["shot"] = sel

    film_strip(project, sel)
    st.divider()
    shot_detail(project, sel)
    st.divider()
    try:
        review_panel(project)
    except Exception as exc:                      # noqa: BLE001
        st.caption(f"take review unavailable: {type(exc).__name__}: {exc}")


def first_frame_panel(project: Dict, shot_id: str, shot: Dict) -> None:
    """The shot's first frame: make candidates, LOOK at them, pick one.

    A chosen still becomes frame 0 of the clip (full sampler, video-to-video from the
    still), so choosing re-shoots the shot -- said on the button, not discovered later.
    Making candidates writes only into keyframes/ and touches no take."""
    cands = kf.candidate_info(project, shot_id)
    cur = shot.get("keyframe")
    with st.expander("first frame" + (" · one chosen" if cur else ""),
                     expanded=bool(cur)):
        st.caption(
            "A bad first frame is in every frame of the clip, so it is worth "
            "choosing it before the ~50 min render. The aMUSEd look draws a 256 px "
            "still with a different composition and texture, then DreamShaper "
            "re-draws it at three strengths: low keeps more of aMUSEd (and its "
            "soft detail), high keeps its layout and colour only.")
        c = st.columns([2, 1, 1])
        look = c[0].radio("look", list(kf.LOOKS), horizontal=True,
                          key=f"kf_look_{shot_id}",
                          format_func=lambda v: {"plain": "DreamShaper",
                                                 "amused": "aMUSEd, then DreamShaper"}[v])
        n = int(c[1].number_input("stills to draw", 1, 4, 1, key=f"kf_n_{shot_id}"))
        per = len(kf.LOOK_STRENGTHS) if look == "amused" else 1
        c[2].caption(f"makes {n * per} candidate(s), ≈{n * per * 2}–{n * per * 4} min")
        if st.button("make candidates", key=f"kf_make_{shot_id}", disabled=busy()):
            if announce("keyframes", store=project["slug"], shot=shot_id,
                        candidates=n, look=look,
                        strengths=",".join(str(s) for s in kf.LOOK_STRENGTHS)):
                st.rerun()
        if cur:
            kp = kf.approved(project, shot)
            if kp:
                st.image(str(kp), caption=f"chosen first frame · {cur}", width=256)
            if st.button("go back to no first frame", key=f"kf_clear_{shot_id}",
                         help="the stills stay on disk; the shot renders from text "
                              "again (and re-shoots, since its inputs change)"):
                fs.update_shot(project, shot_id, keyframe=None)
                fs.save(project)
                st.cache_data.clear()
                st.rerun()
        if not cands:
            st.caption("no candidates yet")
            return
        show = cands[-6:]
        cols = st.columns(len(show))
        for col, ci in zip(cols, show):
            col.image(str(ci["path"]), width="stretch" if _V >= (1, 49) else None)
            what = ("DreamShaper" if ci.get("look") in (None, "plain") else
                    f"aMUSEd → DS {ci.get('strength')}")
            col.caption(f"{ci['candidate']} · {what}")
            chosen_this = cur and cur.endswith(f"{ci['candidate']}.png")
            if col.button("chosen" if chosen_this else "use as first frame",
                          key=f"kf_use_{shot_id}_{ci['candidate']}",
                          disabled=bool(chosen_this),
                          help="the shot will render FULL from this still and "
                               "re-shoot"):
                kf.choose(project, shot_id, int(ci["candidate"][1:]))
                fs.save(project)
                st.cache_data.clear()
                st.rerun()
        raws = sorted(kf.keyframe_dir(project, shot_id).glob("m*.png"))[-3:]
        if raws:
            with st.expander("what aMUSEd drew by itself (256 px, before DreamShaper)"):
                rc = st.columns(len(raws))
                for col, rp in zip(rc, raws):
                    col.image(str(rp), caption=rp.stem)


def shot_detail(project: Dict, shot_id: str) -> None:
    shot = fs.get_shot(project, shot_id)
    if shot is None:
        return
    i = _index_of(project, shot_id)
    state = fs.shot_state(project, shot)
    takes = fs.list_takes(project, shot_id)
    chosen = fs.current_take(project, shot)

    head = st.columns([6, 1, 1])
    head[0].markdown(f"## shot {i} of {len(project['timeline'])}")
    # reordering, where the shot is. Not a separate timeline editor.
    if head[1].button("previous shot", key="shot_up", disabled=(i == 1), **FULL):
        fs.move_shot(project, shot_id, -1)
        fs.save(project)
        st.rerun()
    if head[2].button("next shot", key="shot_dn",
                      disabled=(i == len(project["timeline"])), **FULL):
        fs.move_shot(project, shot_id, +1)
        fs.save(project)
        st.rerun()

    # --- the three decisions, on one line, two deep at most ----------------- #
    c = st.columns(4)
    camera = c[0].selectbox("movement", fs.CAMERAS,
                            index=fs.CAMERAS.index(shot["camera"])
                            if shot.get("camera") in fs.CAMERAS else 0,
                            help="still is a real choice, not an absence")
    scale = c[1].selectbox("framing", fs.SCALES,
                           index=fs.SCALES.index(shot["scale"])
                           if shot.get("scale") in fs.SCALES else 2)
    # The slider is the DELIVERED hold; the source supplies only
    # (2*frames-1) frames. Show both, and say when the difference is visible:
    # make_film.render duplicates frames to fill a longer hold.
    _fps_here = _fps(project)
    _frames_here = int((project.get("settings") or {}).get("frames") or 16)
    _source_frames = 2 * _frames_here - 1
    _natural = _source_frames / _fps_here
    _hold_now = round(float(shot.get("seconds") or 2.5) * _fps_here)
    _stretch_now = _hold_now / max(1, _source_frames)
    _warn = (" — WILL JUDDER: duplicated frames" if _stretch_now > 1.30 else
             f" — smooth (≤{1.30 * _natural:.1f}s)")
    # The auto-pacer can write a hold under 1 s (skydiving s06 = 0.94). A hard 1.0
    # minimum made Streamlit RAISE on those shots and the whole page died, so the
    # floor follows the shot: you can still lengthen it, and a short one still opens.
    _len_now = float(shot.get("seconds") or 2.5)
    seconds = c[2].number_input("length", min(1.0, _len_now), max(8.0, _len_now),
                                _len_now, 0.1,
                                help=f"holds {_hold_now} of the {_source_frames} "
                                     f"generated frames = {_stretch_now:.2f}×"
                                     f"{_warn}. Source motion is {_natural:.2f}s.")
    beat = c[3].selectbox("beat", ["establish", "develop", "turn", "resolve"],
                          index=["establish", "develop", "turn", "resolve"]
                          .index(shot.get("beat"))
                          if shot.get("beat") in
                          ["establish", "develop", "turn", "resolve"] else 1)
    if (camera, scale, round(seconds, 2), beat) != (
            shot.get("camera"), shot.get("scale"), round(float(shot.get("seconds")
                                                               or 2.5), 2),
            shot.get("beat")):
        fs.update_shot(project, shot_id, camera=camera, scale=scale,
                       seconds=float(seconds), beat=beat)
        fs.save(project)
        st.cache_data.clear()
        st.rerun()

    if shot.get("camera_suggestion"):
        st.caption(f"the writer suggested **{shot['camera_suggestion']}** here. "
                   f"{camera} is what will be shot.")
    if shot.get("purpose") or shot.get("link"):
        # the sequence grammar, visible in the cut: what this shot does in its
        # scene and how it is justified from the previous one.
        st.caption(f"coverage: **{shot.get('purpose') or '?'}** · link to previous: "
                   f"**{shot.get('link') or '?'}** · scene "
                   f"**{shot.get('scene') or '?'}**")

    # --- render quality: fast (Lightning) or full ---------------------------- #
    # The tier decides the sampler, so it decides whether the figure comes out right:
    # fast is ~10 min and draws extra limbs on a close figure; full is ~50 min and
    # does not. `auto` picks by framing (wide = fast, closer = full).
    tier_now = fs.render_tier(shot)
    q_opts = ["auto", "fast", "full"]
    q_cur = str(shot.get("quality") or "auto").lower()
    q_cur = q_cur if q_cur in q_opts else "auto"
    qc = st.columns([1, 3])
    q_pick = qc[0].selectbox(
        "render quality", q_opts, index=q_opts.index(q_cur), key=f"quality_{shot_id}",
        help="auto = fast for wide / extreme-wide framing, full for medium and "
             "closer. Override it when a wide shot draws a big figure, or a close "
             "shot is only scenery.")
    qc[1].caption(
        f"this shot renders **{tier_now}**"
        + (" (chosen by its framing)" if q_cur == "auto" else " (set by you)")
        + (" · ≈10 min on this CPU, 4 steps; a figure here can get extra limbs"
           if tier_now == "fast" else
           f" · ≈50 min on this CPU, {fs.FULL_STEPS} steps")
        + " · changing it re-shoots the shot")
    if q_pick != q_cur:
        fs.update_shot(project, shot_id, quality=q_pick)
        fs.save(project)
        st.cache_data.clear()
        st.rerun()

    # --- the prompt, which is the real work -------------------------------- #
    with st.form(f"prompt_{shot_id}"):
        prompt = st.text_area("what the camera sees", value=shot.get("prompt", ""),
                              height=110, label_visibility="visible")
        if st.form_submit_button("save the prompt"):
            fs.update_shot(project, shot_id, prompt=prompt)
            fs.save(project)
            st.cache_data.clear()
            st.rerun()

    first_frame_panel(project, shot_id, shot)

    # --- the take ---------------------------------------------------------- #
    st.divider()
    if chosen:
        frames = strip_frames(chosen["_dir"])
        if frames:
            cols = st.columns(len(frames), gap="small")
            for col, img in zip(cols, frames):
                col.image(img)
        _made = fs.take_tier(chosen)
        st.caption(f"take {chosen['take']} · made {_made}"
                   + (" (4-step; a close figure can have extra limbs)"
                      if _made == "fast" else "") + f" · "
                   f"{chosen['frames_on_disk']} frames · "
                   f"shot in {chosen.get('seconds')}s on "
                   f"{str(chosen.get('at', ''))[:16]}"
                   + (" · locked" if chosen.get("approved") else ""))
        _note = fs.hold_note(project, shot, chosen)
        if _note:
            st.caption("note: this take was " + _note)
        _drift = fs.take_drift(project, shot, chosen) if state == "stale" else []
        if _drift:
            st.warning(
                "This take was made from an older **" + "**, **".join(_drift) +
                "** than the shot says now, so Studio wants to re-shoot it.")
            if st.button("use this take as-is", key=f"adopt_{shot_id}",
                         help="keeps the take you chose and treats it as current "
                              "for the shot's present text. Nothing re-shoots; what "
                              "it was really made from stays recorded on the take."):
                fs.adopt_take(project, shot_id)
                fs.save(project)
                st.cache_data.clear()
                st.rerun()
    else:
        st.markdown(
            '<div class="studio-cell" style="padding:2.2rem;text-align:center">'
            'nothing shot yet</div>', unsafe_allow_html=True)

    # --- the price, then the button. Never the other way round. ------------ #
    stt = fs.film_status(project)
    needs_model = state in ("unrendered", "stale")
    if needs_model:
        frames_setting = int((project.get("settings") or {}).get("frames") or 16)
        mins = (stt["est_generate_s"] / 60) if stt["est_generate_s"] else 11
        price(f"shooting this renders {frames_setting} frames at 512², RAFT-upsamples and shakes "
              f"them: <b>≈{mins:.0f} min</b> on this CPU"
              + (f" (measured from {len(takes)} take(s))" if takes else
                 " (no measurement yet — the estimate is the researched default)"))
    else:
        price("this shot is current. Changing the prompt, movement or framing "
              "re-shoots it; changing the length or the palette only re-grades it, "
              "which takes seconds.")

    b = st.columns([1, 1, 1, 3])
    label = "shoot it" if needs_model else "shoot another take"
    if b[0].button(label, type="primary", key=f"gen_{shot_id}",
                   disabled=busy(), **FULL):
        if announce("produce", store=project["slug"], only=i):
            st.rerun()
    # Locking is a NOTE, not a gate. Say so, and make it reversible -- a control whose
    # effect you have to ask about is a control that is lying about something.
    if chosen:
        locked = bool(chosen.get("approved"))
        if b[1].button("unlock it" if locked else "lock this take",
                       key=f"lock_{shot_id}", **FULL):
            if locked:
                fs.unlock_take(project, shot_id)
            else:
                fs.approve_take(project, shot_id)
            fs.save(project)
            st.cache_data.clear()
            st.rerun()
        st.caption("Locking marks a take as reviewed. It does not change what the "
                   "film cuts to, does not stop you shooting another take, and does "
                   "not hide a warning if the grade or length later drifts — it is a "
                   "note to yourself. Nothing needs locking before an export.")
    with b[2].popover("more", **FULL):
        # Everything destructive or rare, behind one word. Not on the canvas.
        if st.button("duplicate as an alternate angle", key=f"dup_{shot_id}"):
            fs.duplicate_shot(project, shot_id)
            fs.save(project)
            st.cache_data.clear()
            st.rerun()
        if st.button("forget which take is used", key=f"forget_{shot_id}",
                     help="the takes stay on disk"):
            fs.update_shot(project, shot_id, take=None)
            fs.save(project)
            st.cache_data.clear()
            st.rerun()
        if st.button("remove from the film", key=f"rm_{shot_id}",
                     help="its takes stay on disk"):
            fs.remove_shot(project, shot_id)
            fs.save(project)
            st.session_state["shot"] = None
            st.cache_data.clear()
            st.rerun()

    # --- takes, only when there is a choice to make ------------------------ #
    if len(takes) > 1:
        rule(f"{len(takes)} takes — the film uses one")
        for t in takes:
            used = t["take"] == shot.get("take")
            c = st.columns([1, 3, 1])
            c[0].markdown(f"{'**' if used else ''}{t['take']}{'**' if used else ''}")
            c[1].caption(f"{t['frames_on_disk']} frames · {t.get('seconds')}s · "
                         f"{str(t.get('at', ''))[:16]}"
                         + (" · locked" if t.get("approved") else ""))
            if not used and c[2].button("use", key=f"use_{shot_id}_{t['take']}",
                                        **FULL):
                fs.choose_take(project, shot_id, t["take"])
                fs.save(project)
                st.cache_data.clear()
                st.rerun()
        st.caption("Nothing is ever overwritten. Every take stays on disk under "
                   "productions/" + project["slug"] + "/takes/.")


def _index_of(project: Dict, shot_id: str) -> int:
    tl = list(project.get("timeline") or [])
    return (tl.index(shot_id) + 1) if shot_id in tl else 0


# --------------------------------------------------------------------------- #
# CUT -- look at it, then export
# --------------------------------------------------------------------------- #

def _unrendered_indices(project: Dict) -> List[int]:
    """The 1-based timeline indices of the shots that need the model.

    NOT `range(first, first+n)`: a film can have shot 2 already shot and shots 1, 3, 4
    missing, and "the next 2 to shoot" means 1 and 3. Picking a contiguous range would
    re-shoot a finished shot (ten minutes) and silently skip a missing one.
    """
    return [i for i, s in enumerate(fs.shots_in_order(project), 1)
            if fs.needs_generation(fs.shot_state(project, s))]


def view_cut(project: Dict) -> None:
    rows = fs.shots_in_order(project)
    completion_banner(project)
    st.markdown("## the film")
    stt = fs.film_status(project)

    if not rows:
        st.caption("Nothing to cut yet.")
        if st.button("← write a shot list", key="cut_to_write"):
            st.session_state["view"] = "write"
            st.rerun()
        return

    c = st.columns(4)
    c[0].metric("length", f"{stt['duration_s']:.1f}s")
    c[1].metric("shots", stt["shots"])
    c[2].metric("frames", stt["frames"])
    c[3].metric("to shoot", stt["to_generate"])

    if stt["to_generate"]:
        est = (f" About {stt['est_generate_s'] / 60:.0f} min of rendering."
               if stt["est_generate_s"] else "")
        st.warning(f"{stt['to_generate']} shot(s) have never been shot. A cut cannot "
                   f"invent frames.{est}")
        # BATCHING IS THE OTHER LEVER, and it is worth as much as the preflight fix.
        # Every job that needs the model pays the pipeline load (~57s, measured)
        # before its first frame. Started one shot at a time, that is paid per shot;
        # started as one job, it is paid once. The preflight is cached against the
        # LoRA weights, so it is now ~0 after the first ever run.
        b = st.columns([1, 1, 2])
        if b[0].button("shoot everything missing", type="primary", key="shoot_all",
                       disabled=busy(), **FULL):
            if announce("produce", store=project["slug"]):
                st.rerun()
        n_left = stt["to_generate"]
        if n_left > 1:
            n = b[1].number_input("just the next", 1, n_left, min(2, n_left), 1,
                                  key="batch_n", label_visibility="visible")
            if b[2].button(f"shoot {int(n)} as ONE job", key="shoot_n",
                           disabled=busy(), **FULL,
                           help="the model loads once for all of them, instead of "
                                "once each"):
                only = ",".join(str(i) for i in _unrendered_indices(project)[:int(n)])
                if announce("produce", store=project["slug"], only=only):
                    st.rerun()
            price("one job loads the model once (~57s). Started shot by shot, that "
                  "is paid again for every shot.")
    else:
        st.success("Every shot has a take. Cutting regenerates nothing.")
        if stt["to_refinish"]:
            st.caption(f"{stt['to_refinish']} shot(s) will be re-graded during the "
                       f"cut — seconds each, no model.")

    rec = project.get("deliverable") or {}
    if rec.get("render"):
        d = fs.renders_dir(project["slug"]) / rec["render"]
        mp4 = d / "film.mp4"
        st.divider()
        st.markdown(f"## {rec['render']}")
        if mp4.exists():
            mb = mp4.stat().st_size / 1e6
            # THE PLAYER, AND A WAY PAST IT. `st.video` streams the file to the browser,
            # and a media transfer is the one thing on this page that a self-refreshing
            # fragment could interrupt -- the component is re-sent every 2 seconds. If
            # the player will not play, the film is still right there: a link opens it,
            # and a download button always works.
            st.video(str(mp4))
            st.caption(f"{rec.get('duration_s')}s · {rec.get('frames')} frames · "
                       f"{mb:.1f} MB · h264, silent · exported {rec.get('at')}")
            st.markdown(f"[open the file directly]({mp4.resolve().as_uri()})")
            try:
                with open(mp4, "rb") as fh:
                    st.download_button("download the mp4", fh.read(),
                                       file_name=f"{rec['render']}.mp4",
                                       mime="video/mp4",
                                       key=f"dl_{rec['render']}")
            except Exception as exc:
                st.caption(f"(download unavailable: {exc})")
            st.caption(f"on disk at `{mp4}`")
        else:
            st.warning(f"the record points at `{mp4}`, which is not on disk. The store "
                       f"still has every take, so a re-export regenerates nothing.")

    # ADVISORY, NOT A GATE: a failed take review is shown here because this is where the
    # film is about to be cut, but it never blocks an export (a verdict is a person's
    # or a model's opinion of a take; the export only cuts what is on disk).
    try:
        rv = review_facts(project)
        if rv["status"] == "failed":
            st.warning("take review flagged: " + " | ".join(rv["reasons"][:3])
                       + " — see *review the takes* in the shoot view.")
    except Exception:                                  # noqa: BLE001
        pass

    st.divider()
    can = fs.ready_to_export(project)
    price("Export reads the takes from disk, grades them, and cuts them. It does not "
          "re-shoot anything." + (" The film is silent — narration is a separate "
                                  "step and is not wired in yet." if can else ""))
    if st.button("export the film", type="primary", key="export_btn",
                 disabled=busy() or not can):
        if announce("produce", store=project["slug"],
                    name=f"{project['slug']}_render"):
            st.rerun()

    with st.expander("the cut, shot by shot"):
        t = 0.0
        for i, s in enumerate(rows, 1):
            hold = fs.hold_frames(s)
            st.markdown(
                f"`{t:5.2f}s` **{i}.** {s.get('scale')} · "
                f"{s.get('camera') or 'still'} · {hold}f · "
                f"{STATE_WORD[fs.shot_state(project, s)]}<br>"
                f"<span style='opacity:.6;font-size:.86em'>{s.get('prompt','')}</span>",
                unsafe_allow_html=True)
            t += hold / _fps(project)

    try:
        audit_panel(project)
    except Exception as exc:                      # noqa: BLE001
        st.caption(f"film audit unavailable: {type(exc).__name__}: {exc}")

    # --- taste capture ------------------------------------------------------ #
    # Every A/B in this project ended with "watch it and decide", and the decision
    # was thrown away. This is the one-click version: pick two delivered cuts,
    # say which read better, and the row is stored with its measured properties.
    st.divider()
    st.markdown("## taste · which cut read better?")
    try:
        import taste
        films = taste.available_films()
    except Exception as exc:                      # noqa: BLE001
        films = []
        st.caption(f"taste capture unavailable: {type(exc).__name__}: {exc}")
    if len(films) < 2:
        st.caption("Shoot two cuts (or two productions) and this records which one "
                   "you preferred, with the measured difference.")
    else:
        labels = [f["label"] for f in films]
        sel = st.columns(2)
        ia = sel[0].selectbox("cut A", range(len(films)),
                              format_func=lambda i: labels[i],
                              key=f"taste_a_{project['slug']}")
        ib = sel[1].selectbox("cut B", range(len(films)), index=1,
                              format_func=lambda i: labels[i],
                              key=f"taste_b_{project['slug']}")
        if ia == ib:
            st.caption("choose two different cuts")
        else:
            v = st.columns(2)
            v[0].video(films[ia]["path"])
            v[1].video(films[ib]["path"])
            note = st.text_input("why? (optional, one line)",
                                 key=f"taste_note_{project['slug']}")
            pick = None
            b = st.columns(3)
            if b[0].button("A was better", key=f"taste_pa_{project['slug']}"):
                pick = "a"
            if b[1].button("B was better", key=f"taste_pb_{project['slug']}"):
                pick = "b"
            if b[2].button("too close to call", key=f"taste_pt_{project['slug']}"):
                pick = "tie"
            if pick:
                taste.record(films[ia]["path"], films[ib]["path"], pick, note,
                             tags=[project.get("concept", "")[:24]])
                st.success("recorded — the studio now has one more labelled cut.")
        st.caption(taste.format_summary(taste.summary()))


# --------------------------------------------------------------------------- #
# SETTINGS -- everything real that is not on the main path
# --------------------------------------------------------------------------- #

def view_effects(project: Dict) -> None:
    """The visual effects view -- its own menu, not a section of settings.

    It started inside settings and settings became unreadable: that view already owns
    the frame count, the shake, the baseline, the palettes, the scouts' raw material,
    provenance and history, and dropping a chain builder with per-step popovers into the
    middle of it made a wall. The user's words: *"the setting became too crowded when you
    added it there"*. Effects deserve their own surface because they are their own job --
    you sit in them for a while, iterating, where settings are visited and left.
    """
    completion_banner(project)
    rows = fs.shots_in_order(project)
    st.markdown("## effects")
    if not rows:
        st.caption("No shots yet. Effects change how an image is represented, so there "
                   "has to be an image first.")
        return
    from ui.effects_panel import effects_panel
    before = _snap(project)
    effects_panel(project)
    if _snap(project) != before:           # the panel saves its own edits; this is a
        fs.save(project)                   # net, not a 2-second rewrite of the file


def view_settings(project: Dict) -> None:
    st.markdown("## settings")
    st.caption("Nothing here is needed to make a film. It is here because it is real "
               "and occasionally you want it — not because it deserves the space.")

    rule("how the film is rendered")
    c = st.columns(4)
    stg = project.get("settings") or {}
    # Raw frames per shot. AnimateDiff was trained on 16-frame windows, but higher
    # counts work for longer durations. Each raw frame becomes ~2 after RAFT x2.
    # At 16 FPS delivery: 16 frames = ~1.9s, 24 frames = ~2.9s, 32 frames = ~3.9s
    frames = c[0].selectbox(
        "raw frames per shot", [8, 12, 16, 24, 32], 
        index=[8, 12, 16, 24, 32].index(
            int(stg.get("frames") or 16)) if int(stg.get("frames") or 16) in
        (8, 12, 16, 24, 32) else 2,  # default to 16 (index 2)
        help="Raw frames before RAFT x2. 16 is the researched default. Higher = longer "
             "shots (24 frames ≈ 3s, 32 frames ≈ 4s per shot after interpolation). "
             "Lower = faster but less motion.")
    shake = c[1].slider("camera shake", 0.0, 3.0, float(stg.get("shake_scale", 1.0)),
                        0.05, help="1.0 is the researched default. Whole film.")
    baseline = c[2].checkbox("match exposure across shots",
                             value=bool(stg.get("technical_baseline", True)),
                             help="one photographic baseline so no cut reads as a "
                                  "technical error")
    seed = c[3].number_input("seed base", 1, 99999, int(stg.get("seed_base", 1103)),
                             1, help="each shot's seed is frozen from this once and "
                                     "then belongs to the shot, so reordering never "
                                     "re-rolls it")
    if (int(frames) != int(stg.get("frames") or 16)
            or abs(shake - float(stg.get("shake_scale", 1.0))) > 1e-9
            or baseline != bool(stg.get("technical_baseline", True))
            or int(seed) != int(stg.get("seed_base", 1103))):
        fs.set_settings(project, frames=int(frames), shake_scale=float(shake),
                        technical_baseline=bool(baseline), seed_base=int(seed))
        fs.save(project)
        st.cache_data.clear()
        st.rerun()

    rule("how the cut is directed")
    # The store used to freeze every shot at camera 'none', so the writer's
    # suggestion and the continuity rules never reached a render. This chooses the
    # policy for the NEXT shot list; existing shots keep their stored camera.
    from camera_direction import DEFAULT_MODE, MODES
    cur_mode = str(stg.get("camera_mode") or DEFAULT_MODE)
    if cur_mode not in MODES:
        cur_mode = DEFAULT_MODE
    c = st.columns([2, 2])
    cam_mode = c[0].selectbox(
        "camera moves for new drafts", list(MODES), index=MODES.index(cur_mode),
        help="off = every shot still (the old behaviour, kept for A/B). "
             "suggestions = the writer's own suggestion, continuity-repaired. "
             "director = director.choose() from CLIP fit, hints and continuity. "
             "hybrid = director first, the writer's suggestion where the director "
             "declines. A broken beat order is still refused separately.")
    c[1].caption("affects the next shot list you write; re-draft to apply. "
                 "Editing a shot's movement by hand always wins.")
    if cam_mode != cur_mode:
        project.setdefault("settings", {})["camera_mode"] = cam_mode
        fs.save(project)
        st.cache_data.clear()
        st.rerun()

    rule("the character")
    character_panel(project)

    rule("scenes and colour")
    st.caption("One palette per scene. Changing a palette re-grades that scene's "
               "shots during the cut — seconds, not minutes.")
    from looks import ARC
    before_scenes = _snap(project)
    for sc in fs.sync_scenes(project):
        c = st.columns([2, 2])
        n = sum(1 for s in fs.shots_in_order(project)
                if s.get("scene") == sc["id"])
        c[0].markdown(f"**{sc.get('name', sc['id'])}** · {n} shot(s)")
        cur = sc.get("palette") or "neutral"
        pick = c[1].selectbox(f"palette {sc['id']}", ARC,
                              index=ARC.index(cur) if cur in ARC else 0,
                              key=f"pal_{sc['id']}", label_visibility="collapsed")
        if pick != cur:
            fs.set_palette(project, sc["id"], pick)
            fs.save(project)
            st.cache_data.clear()
            st.rerun()
    if _snap(project) != before_scenes:    # sync_scenes normalised something
        fs.save(project)

    # ---- effects: the surface, after the grade ----------------------------- #
    # MOVED to its own view (`view_effects`). It lives here still as a one-line pointer
    # rather than a cross-reference nobody finds.
    st.caption("Looking for effects — posterize, halftone, dither, film damage, "
               "pixel sort and the rest? They have their own menu now, under "
               "**effects** in the sidebar.")

    # The scouts. This is where the 18 ideas live: one time in ten you want to steer
    # the writer by hand, and the other nine times they are noise on the main path.
    rule("the writers' raw material")
    ideas = project.get("ideas") or []
    total = sum(len(f.get("ideas") or []) for f in ideas)
    if not ideas:
        st.caption("No scout ideas recorded for this shot list. Rewriting it with "
                   "`start over` records them.")
    else:
        st.caption(f"{total} ideas from {len(ideas)} scouts. They are the machinery "
                   f"behind the shot list, not the shot list. Tick the ones you want "
                   f"the writer to build on, then rewrite.")
        for f in ideas:
            st.markdown(f"*{f.get('model')}* — {f.get('angle', '')}")
            for j, idea in enumerate(f.get("ideas") or []):
                st.checkbox(idea, key=f"idea_{f.get('model')}_{j}")

    rule("provenance")
    prov = project.get("provenance") or {}
    if prov:
        st.caption(f"{prov.get('_final_prompts_traced', '?')} of "
                   f"{prov.get('_final_prompts', '?')} shots trace back to a scout's "
                   f"idea. Without this, four scouts paying for one looks identical "
                   f"to an ensemble working.")
        for k, v in prov.items():
            if not k.startswith("_"):
                st.caption(f"  {k}: {v.get('used')}/{v.get('ideas')} ideas landed "
                           f"(mean overlap {v.get('mean_best')})")
    else:
        st.caption("none recorded")

    if project.get("history"):
        rule("what this production has done")
        for h in reversed(project["history"][-12:]):
            st.caption(f"{h.get('at', '')} · {h.get('act', '')} · "
                       + ", ".join(f"{k}={v}" for k, v in h.items()
                                   if k not in ("at", "act")))

    rule("the store")
    st.caption(f"`productions/{project['slug']}/` — one file per shot, immutable "
               f"takes, and every render. Nothing is ever overwritten, so nothing "
               f"you make is lost by making something else.")
    if st.button("forget the cached pages (reload from disk)"):
        st.cache_data.clear()
        st.rerun()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def new_production_flow() -> None:
    st.markdown("## a new film")
    name = st.text_input("what to call it", "film1")
    concept = st.text_input("what it is about", "",
                            placeholder="one sentence, in plain words")
    if st.button("begin", type="primary", disabled=not name.strip()):
        pr = fs.new_project(name.strip(), concept.strip())
        fs.save(pr)
        st.session_state["slug"] = pr["slug"]
        st.session_state["view"] = "write"
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="studio", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    projects = fs.list_projects()
    if not projects and not st.session_state.get("new"):
        new_production_flow()
        return

    with st.sidebar:
        st.markdown("### studio")
        if projects:
            by = {p["slug"]: p for p in projects}
            slug = st.selectbox(
                "production", [p["slug"] for p in projects],
                index=([p["slug"] for p in projects].index(st.session_state["slug"])
                       if st.session_state.get("slug") in by else 0),
                format_func=lambda s: by[s]["name"], label_visibility="collapsed")
            st.session_state["slug"] = slug
        else:
            slug = st.session_state.get("slug")
            if not slug:
                new_production_flow()
                return

        project = load(slug)
        rows = fs.shots_in_order(project)
        stt = fs.film_status(project)

        st.divider()
        view = st.radio("view", ["write", "shoot", "effects", "cut", "settings"],
                        index=["write", "shoot", "effects", "cut", "settings"].index(
                            st.session_state.get("view", "write"))
                        if st.session_state.get("view") in
                        ["write", "shoot", "effects", "cut", "settings"] else 0,
                        format_func=lambda v: {"write": "write",
                                               "shoot": "shoot",
                                               "effects": "effects",
                                               "cut": "cut",
                                               "settings": "settings"}[v],
                        label_visibility="collapsed")
        st.session_state["view"] = view

        st.divider()
        if rows:
            st.caption(f"{len(rows)} shots · {stt['duration_s']:.1f}s")
            st.caption(f"{stt['to_generate']} to shoot · "
                       f"{stt['to_refinish']} to re-grade")
        else:
            st.caption("no shot list yet")
        say_what_happened(project)

        st.divider()
        activity_panel()

        with st.expander("start a new production"):
            if st.button("new production"):
                st.session_state["new"] = True
                st.rerun()
            if st.session_state.get("new"):
                nm = st.text_input("name", "film2", key="new_name")
                cn = st.text_input("concept", "", key="new_concept")
                # `new_project` + `save` REPLACE project.json. Creating "film2" while a
                # film2 existed silently wiped its timeline and concept (its shots and
                # takes stayed on disk, orphaned), so an existing name is refused.
                _taken = fs.dir_for(fs.slugify(nm.strip())) / "project.json"
                if nm.strip() and _taken.exists():
                    st.error(f"'{fs.slugify(nm.strip())}' already exists — pick it "
                             f"in the production list, or choose another name.")
                if st.button("create it", key="create_new",
                             disabled=(not nm.strip()) or _taken.exists()):
                    pr = fs.new_project(nm.strip(), cn.strip())
                    fs.save(pr)
                    st.session_state["slug"] = pr["slug"]
                    st.session_state["new"] = False
                    st.session_state["view"] = "write"
                    st.cache_data.clear()
                    st.rerun()

    content(st.session_state["slug"])


@st.fragment(run_every=2)
def content(slug: str) -> None:
    """The view, re-read from disk and re-rendered every 2 seconds.

    WHY THIS EXISTS, in the user's words: *"it needs to auto refresh after the shot list
    is done ... we need constant freshness ... if not im wasting extra time waiting for
    it."* They were doing the reload by hand.

    Three things make it necessary and none of them is cosmetic:

      * A job writes into `productions/<slug>/` from ANOTHER PROCESS, so nothing in this
        session knows the shot list changed. A fragment re-runs on its own timer and
        picks the change up whether or not this tab started the job.
      * `load()` is cached for 4 seconds, which is right for a page render and wrong for
        a page you are watching. The cache is cleared here so every pass reads the store
        from disk -- and `load` reads the project AND its shots, which is why the slug is
        the parameter and not a project dict loaded outside the fragment. Passing a dict
        in would have handed the fragment a 4-second-old project and defeated the point.
      * The alternative -- `st.rerun()` in a polling loop -- replaces the whole page and
        makes the sidebar and every widget jump. A fragment re-runs only this subtree, so
        typing in the concept box is not interrupted and the job panel's bar holds its
        place.

    Only `load` is cleared, not `st.cache_data.clear()`: the latter also throws away the
    thumbnails and frame strips, so the page would re-decode every image twice a second
    for nothing. Two seconds is chosen against the supervisor's 1.5-second `state.json`
    write, because polling slower than the writer is what makes a moving bar look stalled.
    """
    load.clear()
    project = load(slug)
    {
        "write": view_write,
        "shoot": view_shoot,
        "effects": view_effects,
        "cut": view_cut,
        "settings": view_settings,
    }.get(st.session_state.get("view", "write"), view_write)(project)


if __name__ == "__main__":
    main()
