"""
ui/workbench.py -- the production workbench, built on the store.

    streamlit run ui/workbench.py

WHY THIS SHAPE
--------------
The old workbench was seven pages in a line (`ui/app.py`), and the data flow was
inconsistent because three things claimed to be the source of truth and were joined
by ARRAY POSITION. `AUDIT.md` measures that; `film_store.py` removes the class of
bug. This file is the UI on top of the fixed model.

Three conventions are taken from how real editing tools work, and one from how they
do not:

  1. THREE PANES AND A TIMELINE. Every NLE converges on assets on the left, the
     viewer in the middle, the inspector on the right, the timeline across the
     bottom (Premiere, Resolve, CapCut, FCP). The old workbench put each of those on
     its own page, so you could never see a change and its consequence at once.

  2. THE TIMELINE IS THE PROJECT. Order lives in the timeline and nowhere else. A
     shot is addressed by ID, so moving it moves nothing else -- which is exactly
     what the old model could not do.

  3. TAKES, NOT OVERWRITES. Re-rendering writes a NEW take (FCP calls this an
     audition; Resolve a take selector). At 540-788 s per shot, the ability to
     compare two and pick one is the difference between iterating and gambling.

  4. ...AND ONE THING THEY GET WRONG FOR THIS MACHINE. In an NLE a render is a
     disposable cache and the export is the deliverable. Here generation costs
     ten minutes a shot on CPU, so a take is NOT disposable: it is the most
     expensive artifact in the project. The UI therefore shows, always and before
     you commit, the two prices -- what needs GENERATING (minutes) and what needs
     only a RE-FINISH (seconds) -- measured from the takes already on disk.

Everything expensive runs through `job_runner` and is polled, never awaited. The
machine fits one render at a time, so the queue says so rather than pretending.
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

FILMS = ROOT / "output" / "film"
_V = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
FULL = {"width": "stretch"} if _V >= (1, 49) else {"use_container_width": True}

STATE_LABEL = {
    "unrendered": "not rendered",
    "stale": "needs generating",
    "restale": "needs a re-finish",
    "rendered": "ready",
    "approved": "approved",
}
STATE_ICON = {"unrendered": "[ ]", "stale": "[!]", "restale": "[~]",
              "rendered": "[x]", "approved": "[*]"}


# --------------------------------------------------------------------------- #
# cheap, cached reads -- the store is on disk, so everything is a file read
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False, ttl=5)
def load_project(slug: str) -> Dict:
    return fs.load(slug)


@st.cache_data(show_spinner=False, ttl=5)
def frame_strip(take_dir: str, n: int = 5, size: int = 240) -> List:
    """Up to `n` evenly-spaced frames from a take, downsampled for the UI.

    Deliberately NOT the whole clip: the point on screen is to recognise the shot
    and see that the frames differ, which five frames answer.
    """
    from PIL import Image
    d = Path(take_dir) / "frames"
    files = sorted(d.glob("f*.png")) if d.exists() else []
    if not files:
        return []
    if len(files) <= n:
        picks = files
    else:
        picks = [files[round(i * (len(files) - 1) / (n - 1))] for i in range(n)]
    out = []
    for p in picks:
        try:
            out.append(Image.open(p).convert("RGB").resize((size, size)))
        except Exception:
            continue
    return out


def single_frame(take_dir: str, size: int = 320):
    """One frame for a thumbnail.

    Deliberately NOT `frame_strip(take_dir, n=1)`: a cached function shared between
    a list call and a single call is how a list turns up where a frame was expected
    -- and it did, with `n=1` reading the size as the frame count. Its own function,
    its own signature.
    """
    from PIL import Image
    d = Path(take_dir) / "frames"
    files = sorted(d.glob("f*.png")) if d.exists() else []
    if not files:
        return None
    try:
        return Image.open(files[len(files) // 2]).convert("RGB").resize((size, size))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# queue: start work, never wait for it
# --------------------------------------------------------------------------- #

def start(task: str, **params) -> None:
    """Start a job, or say why it will not start. Never blocks the page."""
    try:
        jr.start(task, **params)
        st.toast(f"{task} started")
    except RuntimeError as exc:
        st.warning(str(exc))


def job_is_busy() -> bool:
    return jr.status().get("status") in (jr.QUEUED, jr.RUNNING)


@st.fragment(run_every=3)
def queue_panel() -> None:
    """The render queue. Polls; never blocks.

    `run_every=3` unconditionally: the poll is a read of one small JSON file, and
    making the interval depend on the job state would mean the page stops noticing
    a job that started in another tab.
    """
    job = jr.status()
    state = job.get("status")
    if state == "idle":
        st.caption("nothing running. One render at a time -- this machine fits one.")
        return
    pr = job.get("progress") or {}
    c = st.columns([2, 3, 1, 1])
    c[0].markdown(f"**{job.get('task')}** · {state} · {job.get('cost', '')}")
    c[1].write(pr.get("last") or job.get("message") or "starting...")
    done, total = pr.get("done"), pr.get("total")
    if done and total:
        c[2].progress(min(1.0, done / total))
    else:
        c[2].write("")
    if state in (jr.QUEUED, jr.RUNNING):
        if c[3].button("stop", key="q_stop"):
            jr.stop()
            st.rerun()
    with st.expander(f"log tail ({job.get('log_lines') or 0} lines)"):
        st.text(jr.tail(job["id"], 30))


# --------------------------------------------------------------------------- #
# pane 1: assets
# --------------------------------------------------------------------------- #

def assets_pane(project: Dict, sel: Dict) -> None:
    st.markdown("#### assets")
    rows = fs.shots_in_order(project)
    if not rows:
        st.info("No shots yet. Write a shot list in the panel on the right.")
        return
    st.caption(f"{len(rows)} shots · {len(project.get('scenes') or [])} scene(s)")

    scenes = {s["id"]: s for s in project.get("scenes") or []}
    by_scene: Dict[str, List[Dict]] = {}
    for s in rows:
        by_scene.setdefault(s.get("scene") or "sc1", []).append(s)

    for sc_id, group in by_scene.items():
        sc = scenes.get(sc_id) or {"name": sc_id, "palette": "neutral"}
        st.markdown(f"**{sc.get('name', sc_id)}** · `{sc.get('palette')}`")
        for shot in group:
            state = fs.shot_state(project, shot)
            take = fs.best_take(project, shot)
            chosen = "selected" if shot["id"] == sel.get("shot_id") else "shot"
            label = (f"{STATE_ICON[state]} {shot['id']}  "
                     f"{shot.get('camera') or 'none'} · "
                     f"{fs.hold_frames(shot)}f")
            c = st.columns([1, 4])
            if take:
                img = single_frame(take["_dir"], 72)
                if img is not None:
                    c[0].image(img)
            if c[1].button(label, key=f"pick_{shot['id']}", help=STATE_LABEL[state]):
                st.session_state["sel"] = {"kind": "shot", "shot_id": shot["id"]}
                st.rerun()


# --------------------------------------------------------------------------- #
# pane 2: the viewer
# --------------------------------------------------------------------------- #

def viewer_pane(project: Dict, sel: Dict) -> None:
    st.markdown("#### preview")
    rows = fs.shots_in_order(project)
    if not rows:
        return

    if sel.get("kind") == "film" or not sel.get("shot_id"):
        _program_monitor(project)
        return

    shot = fs.get_shot(project, sel["shot_id"])
    if shot is None:
        st.info("That shot is no longer in the project.")
        return
    state = fs.shot_state(project, shot)
    take = fs.best_take(project, shot)
    st.caption(f"{shot['id']} · {STATE_LABEL[state]} · "
               f"{shot.get('scale')} · {shot.get('camera') or 'none'} · "
               f"{fs.hold_frames(shot)} frames @ "
               f"{(project.get('settings') or {}).get('fps', 16)}fps")
    if take is None:
        st.info("Nothing rendered for this shot yet. **Generate** it in the panel on "
                "the right; it costs about 9-14 minutes.")
        return
    strip = frame_strip(take["_dir"])
    if strip:
        cols = st.columns(len(strip))
        for c, img in zip(cols, strip):
            c.image(img)
        st.caption(f"take **{take['take']}**, {take['frames_on_disk']} frames on "
                   f"disk, generated in {take.get('seconds')}s"
                   + (f", approved {take.get('approved_at')}" if take.get("approved")
                      else ""))
    takes = fs.list_takes(project, shot["id"])
    if len(takes) > 1:
        st.markdown("**takes** — pick the one the film uses")
        for t in takes:
            same = t["take"] == shot.get("take")
            c = st.columns([1, 3, 1])
            c[0].markdown(f"{'**' if same else ''}{t['take']}{'**' if same else ''}")
            c[1].caption(f"{t['frames_on_disk']} frames · {t.get('seconds')}s"
                         + (" · approved" if t.get("approved") else ""))
            if not same and c[2].button("use", key=f"use_{shot['id']}_{t['take']}"):
                fs.choose_take(project, shot["id"], t["take"])
                fs.save(project)
                st.cache_data.clear()
                st.rerun()
    else:
        st.caption("one take. Render again to get a second one and compare — must "
                   "run from the UI so it survives.")


def _program_monitor(project: Dict) -> None:
    """The film as it stands: which shots are in the cut and what state they are in."""
    st.caption("the film as it stands")
    stt = fs.film_status(project)
    m = st.columns(4)
    m[0].metric("duration", f"{stt['duration_s']:.2f}s")
    m[1].metric("frames", stt["frames"])
    m[2].metric("shots", stt["shots"])
    m[3].metric("to generate", stt["to_generate"])
    if stt["to_generate"] == 0:
        st.success("Every shot has a current take. The export regenerates nothing.")
    else:
        est = (f" about {stt['est_generate_s'] / 60:.0f} min"
               if stt["est_generate_s"] else "")
        st.warning(f"{stt['to_generate']} shot(s) need generating{est}, measured "
                   f"from the takes already on disk.")
    if stt["to_refinish"]:
        st.info(f"{stt['to_refinish']} shot(s) need only a re-finish — seconds each, "
                f"no model.")
    name = (project.get("deliverable") or {}).get("render")
    if name and (FILMS / name / "film.mp4").exists():
        st.video(str(FILMS / name / "film.mp4"))

# --------------------------------------------------------------------------- #
# pane 3: the inspector
# --------------------------------------------------------------------------- #

def inspector_pane(project: Dict, sel: Dict) -> None:
    st.markdown("#### inspector")
    if sel.get("kind") == "shot" and sel.get("shot_id"):
        shot_inspector(project, sel["shot_id"])
    else:
        create_inspector(project)


def create_inspector(project: Dict) -> None:
    with st.form("create"):
        concept = st.text_area("concept", value=project.get("concept", ""), height=90,
                               help="one line. The writers work from this and "
                                    "nothing else.")
        shots = st.slider("shots wanted", 2, 24,
                          max(2, min(24, len(project.get("timeline") or []) or 4)),
                          help="SHOT COUNT IS THE BUDGET: at 9-14 min per shot on "
                               "this CPU, 4 shots is about an hour.")
        c = st.columns(2)
        farms = c[0].text_input("idea farms",
                                "qwen2.5:3b,phi3:latest,mistral:latest",
                                help="several cheap models supply raw ideas; one "
                                     "showrunner authors the film")
        showrunner = c[1].text_input("showrunner", "granite4.1:8b")
        if st.form_submit_button("write a shot list (~4-7 min)"):
            fs.set_settings(project)
            project["concept"] = concept
            fs.save(project)
            start("story", concept=concept, n_shots=shots,
                  farms=farms, showrunner=showrunner)
            st.rerun()

    st.divider()
    st.markdown("**or adopt a proposal already on disk**")
    props = ROOT / "output" / "film" / "proposals"
    cands = sorted(props.glob("*.json"), key=lambda p: p.stat().st_mtime,
                   reverse=True)[:12] if props.exists() else []
    if not cands:
        st.caption("none cached")
    else:
        pick = st.selectbox(
            "cached proposals", cands,
            format_func=lambda p: f"{p.stem[:70]} ({p.stat().st_size // 1024} KB)")
        if st.button("adopt it"):
            try:
                raw = json.loads(pick.read_text(encoding="utf-8"))
                from story import Proposal, validate
                shots, problems = validate(raw)
                if not shots:
                    st.error(f"that proposal does not validate: {problems}")
                else:
                    prop = Proposal(concept=project.get("concept", ""), model="cached",
                                    logline=raw.get("logline", ""), shots=shots,
                                    raw=raw, seconds=0.0, decode_tok_s=0.0,
                                    problems=list(problems))
                    fs.shots_from_proposal(project, prop)
                    fs.save(project)
                    st.cache_data.clear()
                    st.rerun()
            except Exception as exc:
                st.exception(exc)


def shot_inspector(project: Dict, shot_id: str) -> None:
    shot = fs.get_shot(project, shot_id)
    if shot is None:
        return
    state = fs.shot_state(project, shot)
    st.markdown(f"**{shot_id}** · {STATE_LABEL[state]}")

    # the cost, before the commit. This is the control the old UI did not have.
    gen = fs.generation_hash(project, shot)
    takes = fs.list_takes(project, shot_id)
    chosen = fs.current_take(project, shot)
    if state in ("unrendered", "stale"):
        stt = fs.film_status(project)
        if stt["est_generate_s"]:
            st.warning(f"Changing this needs the model: about "
                       f"{stt['est_generate_s'] / 60:.0f} min for one shot "
                       f"(measured from {stt['shots'] - stt['to_generate']} take(s)).")
        else:
            st.warning("Needs the model: about 9-14 minutes on this CPU. Scale 512, "
                       "4-step Lightning, no GPU.")
    elif state == "restale":
        st.info("Only the finish moved. This re-runs RAFT + shake + grade from the "
                "frames already on disk — seconds, no model.")

    with st.form(f"shot_{shot_id}"):
        prompt = st.text_area("prompt", value=shot.get("prompt", ""), height=110)
        r1 = st.columns(3)
        purpose = r1[0].text_input("purpose", value=shot.get("purpose") or "",
                                   help="what this shot is FOR. Makes an edit "
                                        "reviewable: you can see whether it still "
                                        "does its job.")
        subject = r1[1].text_input("subject", value=shot.get("subject") or "")
        beat = r1[2].selectbox(
            "beat", ["establish", "develop", "turn", "resolve"],
            index=["establish", "develop", "turn", "resolve"].index(shot["beat"])
            if shot.get("beat") in ["establish", "develop", "turn", "resolve"] else 1)
        r2 = st.columns(4)
        scale = r2[0].selectbox(
            "scale", fs.SCALES,
            index=fs.SCALES.index(shot["scale"]) if shot.get("scale") in fs.SCALES else 2)
        camera = r2[1].selectbox(
            "camera", fs.CAMERAS,
            index=fs.CAMERAS.index(shot["camera"]) if shot.get("camera") in fs.CAMERAS
            else 0,
            help="the DECISION that renders. The writer's suggestion is kept "
                 "separately and never overwrites this.")
        seconds = r2[2].number_input("seconds", 1.0, 8.0,
                                     float(shot.get("seconds") or 2.5), 0.1,
                                     help=f"the exposure sheet holds it for "
                                          f"{round(float(shot.get('seconds') or 2.5) * 16)} "
                                          f"frames at 16fps")
        scene = r2[3].selectbox(
            "scene", [s["id"] for s in project.get("scenes") or []] or ["sc1"],
            index=([s["id"] for s in project.get("scenes") or []] or ["sc1"]).index(
                shot.get("scene") or "sc1")
            if (shot.get("scene") or "sc1") in
            [s["id"] for s in project.get("scenes") or []] else 0)
        energy = st.slider("energy", 0.0, 0.5, float(shot.get("energy") or 0.2), 0.05)
        if st.form_submit_button("save this shot"):
            fs.update_shot(project, shot_id, prompt=prompt, purpose=purpose,
                           subject=subject, beat=beat, scale=scale, camera=camera,
                           seconds=float(seconds), scene=scene, energy=energy)
            fs.sync_scenes(project)
            fs.save(project)
            st.cache_data.clear()
            st.rerun()

    st.divider()
    b = st.columns(2)
    if b[0].button("generate a new take" if takes else "generate the first take",
                   key=f"gen_{shot_id}"):
        start("produce", store=project["slug"], only=_index_of(project, shot_id))
        st.rerun()
    if chosen and b[1].button("approve the chosen take", key=f"app_{shot_id}"):
        fs.approve_take(project, shot_id)
        fs.save(project)
        st.cache_data.clear()
        st.rerun()

    b2 = st.columns(3)
    if b2[0].button("duplicate", key=f"dup_{shot_id}",
                    help="a second angle on the same moment — the cheapest variety"):
        fs.duplicate_shot(project, shot_id)
        fs.save(project)
        st.cache_data.clear()
        st.rerun()
    if b2[1].button("delete", key=f"rm_{shot_id}"):
        fs.remove_shot(project, shot_id, delete_files=False)
        fs.save(project)
        st.session_state["sel"] = {"kind": "film"}
        st.cache_data.clear()
        st.rerun()
    if b2[2].button("clear the takes", key=f"clr_{shot_id}",
                    help="the takes stay on disk under productions/<slug>/takes/; "
                         "this only forgets which one the film uses"):
        fs.update_shot(project, shot_id, take=None)
        fs.save(project)
        st.cache_data.clear()
        st.rerun()

    if shot.get("camera_suggestion"):
        st.caption(f"the writer suggested `{shot['camera_suggestion']}` for this shot. "
                   f"The decision above is what renders.")


def _index_of(project: Dict, shot_id: str) -> int:
    tl = list(project.get("timeline") or [])
    return (tl.index(shot_id) + 1) if shot_id in tl else 1


# --------------------------------------------------------------------------- #
# the timeline strip + the exposure sheet
# --------------------------------------------------------------------------- #

def timeline_pane(project: Dict, sel: Dict) -> None:
    rows = fs.shots_in_order(project)
    if not rows:
        return
    st.markdown("#### timeline")
    st.caption("Order lives here and nowhere else. A shot is addressed by ID, so "
               "moving it moves no setting with it.")

    for i, shot in enumerate(rows):
        state = fs.shot_state(project, shot)
        take = fs.best_take(project, shot)
        c = st.columns([1, 3, 1, 1, 1, 1])
        img = single_frame(take["_dir"], 96) if take else None
        if img is not None:
            c[0].image(img)
        in_s = _in_point(project, i)
        mark = "**" if shot["id"] == sel.get("shot_id") else ""
        c[1].markdown(
            f"{mark}{i+1}. {shot['id']} · {shot['beat']} · {shot['scale']} · "
            f"{shot.get('camera') or 'none'}{mark}  \n"
            f"<span style='opacity:.65'>{in_s:6.2f}s → "
            f"{in_s + fs.hold_frames(shot) / _fps(project):6.2f}s · "
            f"{fs.hold_frames(shot)}f · {STATE_LABEL[state]}</span>",
            unsafe_allow_html=True)
        if c[2].button("↑", key=f"up_{shot['id']}", disabled=(i == 0)):
            fs.move_shot(project, shot["id"], -1)
            fs.save(project)
            st.rerun()
        if c[3].button("↓", key=f"dn_{shot['id']}", disabled=(i == len(rows) - 1)):
            fs.move_shot(project, shot["id"], +1)
            fs.save(project)
            st.rerun()
        if c[4].button("select", key=f"sel_{shot['id']}"):
            st.session_state["sel"] = {"kind": "shot", "shot_id": shot["id"]}
            st.rerun()
        if c[5].button("✕", key=f"x_{shot['id']}", help="remove from the cut"):
            fs.remove_shot(project, shot["id"])
            fs.save(project)
            st.rerun()

    with st.expander("exposure sheet"):
        st.caption("The holds come from `handheld.hold_frames` — the same function "
                   "the renderer uses, so a timecode here is not a second opinion.")
        t = 0.0
        fps = _fps(project)
        rows_out = []
        for shot in rows:
            hold = fs.hold_frames(shot)
            rows_out.append({"#": shot["id"], "in": f"{t:.2f}s",
                             "out": f"{t + hold / fps:.2f}s", "hold (f)": hold,
                             "beat": shot["beat"], "scale": shot["scale"],
                             "camera": shot.get("camera") or "none",
                             "palette": fs.palette_for(project, shot),
                             "state": STATE_LABEL[fs.shot_state(project, shot)]})
            t += hold / fps
        st.dataframe(rows_out, hide_index=True, **FULL)
        st.caption(f"{t:.2f}s total, {sum(r['hold (f)'] for r in rows_out)} frames")


def _fps(project: Dict) -> int:
    return int((project.get("settings") or {}).get("fps") or 16)


def _in_point(project: Dict, index: int) -> float:
    fps = _fps(project)
    return sum(fs.hold_frames(s) for s in fs.shots_in_order(project)[:index]) / fps


# --------------------------------------------------------------------------- #
# scenes and delivery
# --------------------------------------------------------------------------- #

def scenes_panel(project: Dict) -> None:
    from looks import ARC
    st.markdown("#### scenes and palettes")
    scenes = fs.sync_scenes(project)
    if not scenes:
        st.caption("no scenes yet")
        return
    st.caption("One palette per SCENE, looked up by the scene's id. Changing one "
               "costs a re-finish of that scene's shots — seconds, not minutes.")
    for sc in scenes:
        c = st.columns([2, 3])
        c[0].write(f"**{sc.get('name', sc['id'])}** · "
                   f"{sum(1 for s in fs.shots_in_order(project) if s.get('scene') == sc['id'])} shots")
        cur = sc.get("palette") or "neutral"
        pick = c[1].selectbox(f"palette for {sc['id']}", ARC,
                              index=ARC.index(cur) if cur in ARC else 0,
                              key=f"pal_{sc['id']}", label_visibility="collapsed")
        if pick != cur:
            fs.set_palette(project, sc["id"], pick)
            fs.save(project)
            st.cache_data.clear()
            st.rerun()
    fs.save(project)


def delivery_panel(project: Dict) -> None:
    st.markdown("#### delivery")
    stt = fs.film_status(project)
    if not fs.ready_to_export(project):
        st.error(f"{stt['to_generate']} shot(s) still need generating. Nothing to "
                 f"export — an export cannot invent frames.")
        return
    st.success(f"{stt['duration_s']:.2f}s, {stt['frames']} frames, "
               f"{stt['shots']} shots. Nothing needs the model.")
    if stt["to_refinish"]:
        st.caption(f"{stt['to_refinish']} shot(s) will be re-finished during the "
                   f"export (seconds each).")
    name = f"{project['slug']}_render"
    if st.button("EXPORT (regenerates nothing)"):
        start("produce", store=project["slug"], name=name)
        st.rerun()
    # the deliverable as the PRODUCTION records it, so the UI never has to guess a
    # directory name -- the workaround the old TIMELINE stage needed
    rec = project.get("deliverable") or {}
    if rec.get("render"):
        d = fs.renders_dir(project["slug"]) / rec["render"]
        mp4 = d / "film.mp4"
        st.markdown(f"**{rec['render']}** · {rec.get('duration_s')}s · "
                    f"{rec.get('frames')} frames · exported {rec.get('at')}")
        if mp4.exists():
            st.video(str(mp4))
            st.caption(f"{mp4.stat().st_size / 1e6:.1f} MB · silent. Narration is a "
                       f"separate step (`add_narration.py`) and is not wired into "
                       f"the export yet — that is the largest honest gap.")
        else:
            st.warning(f"the record points at {mp4}, which is not on disk. The "
                       f"render was moved or deleted.")
    renders = fs.renders_dir(project["slug"])
    if renders.exists():
        past = sorted((p for p in renders.iterdir() if p.is_dir()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if past:
            with st.expander(f"{len(past)} render(s) in this production"):
                for p in past[:10]:
                    f = p / "film.mp4"
                    st.write(f"- `{p.name}` — "
                             f"{f.stat().st_size / 1e6:.1f} MB" if f.exists()
                             else f"- `{p.name}` — no film.mp4")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(page_title="workbench", layout="wide")
    st.title("workbench")

    projects = fs.list_projects()
    with st.sidebar:
        st.header("production")
        if not projects:
            st.info("No productions yet.")
            name = st.text_input("new production", "film1")
            concept = st.text_input("concept", "")
            if st.button("create"):
                pr = fs.new_project(name, concept)
                fs.save(pr)
                st.rerun()
            return
        slugs = [p["slug"] for p in projects]
        by_slug = {p["slug"]: p for p in projects}
        slug = st.selectbox("open", slugs,
                            format_func=lambda s: f"{by_slug[s]['name']} "
                                                  f"({by_slug[s]['shots']} shots)")
        if st.button("new production"):
            st.session_state["new"] = True
        if st.session_state.get("new"):
            name = st.text_input("name", "film1")
            concept = st.text_input("concept", "")
            if st.button("create it"):
                fs.save(fs.new_project(name, concept))
                st.session_state["new"] = False
                st.rerun()
        st.divider()
        page = st.radio("panel", ["make", "scenes", "deliver"],
                        captions=["shot list, preview, inspector, timeline",
                                  "scenes and palettes",
                                  "export and past renders"])
        st.divider()
        if st.button("refresh the page"):
            st.cache_data.clear()
            st.rerun()
        st.caption("jobs run detached. This page is safe to close and reopen; the "
                   "store is on disk at productions/<slug>/.")

        st.divider()
        st.subheader("jobs")
        queue_panel()

    project = load_project(slug)
    st.session_state.setdefault("sel", {"kind": "film"})
    sel = st.session_state["sel"]

    if page == "scenes":
        scenes_panel(project)
        return
    if page == "deliver":
        delivery_panel(project)
        return

    # the four-zone layout every editor converges on
    left, mid, right = st.columns([2.4, 4.2, 3.4], gap="medium")
    with left:
        assets_pane(project, sel)
    with mid:
        viewer_pane(project, sel)
    with right:
        inspector_pane(project, sel)

    st.divider()
    c = st.columns([6, 1, 1])
    c[0].markdown("")
    if c[1].button("show the film"):
        st.session_state["sel"] = {"kind": "film"}
        st.rerun()
    if c[2].button("add a shot"):
        shot = fs.add_shot(project, prompt="a new shot: describe one single image",
                           scene=(project.get("scenes") or [{"id": "sc1"}])[0]["id"])
        fs.save(project)
        st.session_state["sel"] = {"kind": "shot", "shot_id": shot["id"]}
        st.cache_data.clear()
        st.rerun()
    timeline_pane(project, sel)


if __name__ == "__main__":
    main()
