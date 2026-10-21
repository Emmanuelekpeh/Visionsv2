"""
ui/app.py -- the production workbench.

A system for MAKING films with this pipeline, editable at every stage. Not a
report: you can decide nothing on a report. What it takes from the old pygame
studio is the interaction model -- a stage per artifact, everything editable in
place, a human gate before the next stage, and work that never blocks the
interface -- not its design, and not its framework.

Three things hold it up:

  film_project.py  workspace/<film>.json is the source of truth. `make_film
                   --from-project` renders the FROZEN plan, so what you approve
                   is what comes out.
  job_runner.py    long steps run detached with structured progress; the board
                   polls. One job at a time, enforced.
  the pipeline     every number comes from calling the pipeline's own functions.
                   NO measurement is reimplemented here -- two definitions of one
                   measure is how this project got its normaliser wrong five times.

A stage is selected in the sidebar rather than as a tab, because streamlit runs
EVERY tab body on every rerun and the PLAN stage loads CLIP.

    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import film_project as fp                                     # noqa: E402
import handheld                                               # noqa: E402
import job_runner as jr                                       # noqa: E402
from director import (ENERGY_ARC_PEAK, SCALE_ENERGY, FitScorer,  # noqa: E402
                      _mover_cap, compare_modes, energy_arc_target,
                      energy_profile)
from edit_planner import MAX_MOVING_RUN                        # noqa: E402
from story import BEATS, story_shape, validate                 # noqa: E402

FPS = 16
FILMS = ROOT / "output" / "film"
PROPOSALS = FILMS / "proposals"
STAGES = ["STORY", "PLAN", "CAMERA", "GRADE", "RENDER", "TIMELINE", "AUDIT"]

_V = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
FULL = {"width": "stretch"} if _V >= (1, 49) else {"use_container_width": True}


@st.cache_resource(show_spinner="loading the CLIP fit scorer (once per session)")
def scorer():
    return FitScorer()


# --------------------------------------------------------------------------- #
# the project, through the pipeline's own validator
# --------------------------------------------------------------------------- #

def revalidate(proj: dict):
    """Check the project's CURRENT shots with `story.validate`.

    The same validator that judges the writer judges your edits, so a hand-edited
    shot cannot slip past a rule the writer was held to. Returns (problems, shape).
    The shots themselves stay as the project's records -- ids and render state are
    not the validator's business.
    """
    raw = {
        "logline": proj["story"].get("logline", ""),
        "shots": [{"prompt": d["prompt"], "scale": d["scale"],
                   "camera": d.get("camera") or "none",
                   "seconds": float(d.get("seconds") or 2.5),
                   "scene": d.get("scene", "scene1"),
                   "energy": d.get("energy"), "beat": d.get("beat", ""),
                   "subject": d.get("subject", "")} for d in proj["shots"]],
    }
    shots, problems = validate(raw)
    return problems, story_shape(shots)


def save(proj: dict):
    p = fp.save(proj)
    st.session_state["saved_at"] = time.strftime("%H:%M:%S")
    return p


def touch_story(proj: dict):
    """Any story edit invalidates the plan. Never let a stale plan render."""
    proj["plan"]["approved"] = False
    proj["plan"]["frozen"] = []
    proj["story"]["approved"] = False
    proj["status"]["plan"] = "stale"


def problems_html(problems, shape):
    story = [p for p in problems if p.startswith(("story:", "story-link:"))]
    other = [p for p in problems if p not in story]
    st.markdown(f"**shape** `{shape}`")
    if not story:
        st.success("story shape and the turn→resolve link are clean")
    for p in story:
        (st.error if p.startswith("story:") else st.warning)(p)
    if other:
        with st.expander(f"{len(other)} other rule break(s)"):
            for p in other:
                st.write("- " + p)


# --------------------------------------------------------------------------- #
# STORY
# --------------------------------------------------------------------------- #

def kept_path(proj: dict) -> Path:
    return fp.WORKSPACE / f"{proj['name']}.kept.json"


def write_kept(proj: dict):
    """Write the kept idea set for a seeded re-roll. None if nothing is kept.

    `story._seed_lines` matches on the idea TEXT, so that is what is stored -- not
    the label, which changes when the farms are re-run.
    """
    ideas = proj["story"].get("kept_ideas") or []
    if not ideas:
        return None
    p = kept_path(proj)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ideas, indent=2), encoding="utf-8")
    return p


def stage_story(proj: dict):
    st.header("STORY")
    st.caption("The writer's shot list, as a thing you own. Every field below is "
               "yours to change, and `story.validate` checks your edits with the "
               "same rules it held the writer to.")

    problems, shape = revalidate(proj)
    left, right = st.columns([3, 2])
    with left:
        st.text_input("logline", value=proj["story"].get("logline", ""),
                      key="logline",
                      help="stating the CHANGE, not the premise")
        if st.button("save logline"):
            proj["story"]["logline"] = st.session_state["logline"]
            save(proj)
            st.rerun()
    with right:
        problems_html(problems, shape)

    prov = proj["story"].get("provenance") or {}
    if prov:
        with st.expander(f"provenance — {prov.get('_final_prompts_traced', '?')} of "
                         f"{prov.get('_final_prompts', '?')} shots trace to a scout"):
            st.dataframe([{"farm": k, "ideas": v["ideas"], "landed": v["used"],
                           "mean best overlap": v["mean_best"]}
                          for k, v in prov.items() if not k.startswith("_")],
                         hide_index=True, **FULL)
    groups = proj["story"].get("ideas") or []
    flat = [(f.get("model", "?"), f.get("angle", ""), i)
            for f in groups for i in f.get("ideas", [])]
    if flat:
        kept = set(proj["story"].get("kept_ideas") or [])
        st.subheader(f"the scouts' ideas ({len(flat)})")
        st.caption("Tick the ideas worth keeping, then re-roll. The showrunner is "
                   "shown ONLY the kept ones and told they were chosen by hand, so "
                   "a re-roll builds on your selection instead of re-rolling the "
                   "whole pool from scratch.")
        choose = []
        for n, (model, angle, idea) in enumerate(flat):
            c = st.columns([1, 11])
            on = c[0].checkbox("keep", value=idea in kept, key=f"keep_{n}",
                               label_visibility="collapsed")
            if on:
                choose.append(idea)
            c[1].markdown(f"`{model}` · {angle}  \n{idea}")
        b = st.columns([1, 1, 3])
        if b[0].button("save the kept set"):
            proj["story"]["kept_ideas"] = choose
            proj["story"]["ideas_source"] = proj["story"].get("proposal_path")
            fp.save(proj)
            st.success(f"{len(choose)} of {len(flat)} kept")
            st.rerun()
        if b[1].button("RE-ROLL with the kept ideas"):
            proj["story"]["kept_ideas"] = choose
            fp.save(proj)
            kp = write_kept(proj)
            if kp is None:
                st.warning("Nothing kept, so there is nothing to seed. Tick at "
                           "least one idea — a re-roll with no selection is just "
                           "the ordinary re-roll.")
            else:
                try:
                    jr.start("story", concept=st.session_state.get("concept", ""),
                             fresh=True, keep_file=str(kp), keep_only=True)
                    st.success(f"started, seeded with {len(choose)} kept idea(s) "
                               f"from {kp.name}. ~4-7 min.")
                except RuntimeError as exc:
                    st.error(str(exc))

    st.subheader("shots")
    for i, d in enumerate(proj["shots"]):
        label = (f"{i+1}. {d.get('beat','?')} · {d.get('scale','?')} · "
                 f"{d.get('camera') or 'none'} · {d.get('seconds', 0):.2f}s")
        with st.expander(label, expanded=False):
            with st.form(f"shot_{proj['name']}_{d['id']}"):
                prompt = st.text_area("prompt", value=d["prompt"], height=80)
                c = st.columns(4)
                beat = c[0].selectbox("beat", BEATS,
                                      index=BEATS.index(d["beat"])
                                      if d.get("beat") in BEATS else 1)
                scale = c[1].selectbox(
                    "scale", ["macro", "close", "medium", "wide", "extreme-wide"],
                    index=["macro", "close", "medium", "wide",
                           "extreme-wide"].index(d["scale"])
                    if d.get("scale") in ("macro", "close", "medium", "wide",
                                          "extreme-wide") else 2)
                camera = c[2].text_input("camera", value=d.get("camera") or "none",
                                         help="the writer's suggestion. The PLAN "
                                              "stage decides what is allowed.")
                seconds = c[3].number_input("seconds", 1.5, 5.0,
                                            float(d.get("seconds") or 2.5), 0.1)
                c2 = st.columns(3)
                scene = c2[0].text_input("scene", value=d.get("scene", ""))
                subject = c2[1].text_input("subject", value=d.get("subject", ""))
                energy = c2[2].number_input("energy", 0.0, 0.5,
                                            float(d.get("energy") or 0.2), 0.05,
                                            help="0-0.5: the renderer's measured "
                                                 "ceiling")
                if st.form_submit_button("save shot"):
                    d.update(prompt=prompt, beat=beat, scale=scale, camera=camera,
                             seconds=seconds, scene=scene, subject=subject,
                             energy=energy)
                    touch_story(proj)
                    fp.save(proj)
                    st.rerun()
            b = st.columns([1, 1, 1, 5])
            if b[0].button("↑", key=f"up_{d['id']}") and i > 0:
                proj["shots"][i - 1], proj["shots"][i] = \
                    proj["shots"][i], proj["shots"][i - 1]
                touch_story(proj)
                fp.save(proj)
                st.rerun()
            if b[1].button("↓", key=f"dn_{d['id']}") and i < len(proj["shots"]) - 1:
                proj["shots"][i + 1], proj["shots"][i] = \
                    proj["shots"][i], proj["shots"][i + 1]
                touch_story(proj)
                fp.save(proj)
                st.rerun()
            if b[2].button("delete", key=f"del_{d['id']}"):
                proj["shots"].pop(i)
                touch_story(proj)
                fp.save(proj)
                st.rerun()

    if st.button("+ add a shot"):
        nid = (max((d["id"] for d in proj["shots"]), default=-1) + 1)
        proj["shots"].append({
            "id": nid, "prompt": "a new shot: describe one single image",
            "scale": "medium", "camera": "none", "seconds": 2.5, "scene": "scene1",
            "subject": "", "energy": 0.2, "beat": "develop", "motion_pin": None,
            "render": {"status": "pending", "seconds": None, "at": None}})
        touch_story(proj)
        fp.save(proj)
        st.rerun()

    st.subheader("gate: approve the story")
    blocked = [p for p in problems if p.startswith("story:")]
    if blocked:
        st.error(f"{len(blocked)} structural story problem(s) — fix or re-roll "
                 f"before planning. A mood board is a failure however good the "
                 f"individual images are.")
    elif st.button("APPROVE THE STORY and move to PLAN"):
        proj["story"]["approved"] = True
        proj["status"]["story"] = "approved"
        save(proj)
        st.session_state["stage"] = "PLAN"
        st.rerun()


# --------------------------------------------------------------------------- #
# PLAN
# --------------------------------------------------------------------------- #

def stage_plan(proj: dict):
    st.header("PLAN")
    st.caption("The whole-film director's assignment, and the planner's repairs. "
               "Nothing renders from this until you freeze it — and once frozen, "
               "`make_film --from-project` renders exactly this list and does NOT "
               "re-run either stage.")

    if not proj["story"].get("approved"):
        st.warning("The story is not approved yet. You can plan anyway, but the "
                   "plan will be invalidated by any story edit.")

    total = len(proj["shots"])
    c = st.columns(4)
    mode = c[0].radio("director", ["global", "greedy"],
                      index=0 if proj["plan"].get("mode", "global") == "global" else 1)
    cap_default = proj["plan"].get("max_movers")
    cap = c[1].slider("mover cap", 0, 4,
                      int(cap_default if cap_default is not None
                          else min(4, _mover_cap(total))),
                      help="the writer is told 'at most two shots may move'. The "
                           "greedy director had NO cap and emitted three in a row.")
    arc = c[2].checkbox("energy arc", value=bool(proj["plan"].get("energy_arc", True)),
                        help="spends the budget where the film's energy shape "
                             "wants it, not only where each shot alone benefits.")
    c[3].metric("shots", total)

    if st.button("COMPUTE THE PLAN (director + planner, ~16 s)"):
        with st.spinner("director and planner..."):
            try:
                fp.plan_project(proj, mode=mode, max_movers=cap, energy_arc=arc,
                                scorer=scorer())
                save(proj)
                st.rerun()
            except Exception as exc:
                st.exception(exc)

    frozen = proj["plan"].get("frozen") or []
    if not frozen:
        st.info("No plan yet. Compute it above.")
        return

    st.subheader("the frozen plan")
    rows = []
    for i, d in enumerate(frozen):
        pin = proj["shots"][i].get("motion_pin") if i < len(proj["shots"]) else None
        rows.append({"#": i + 1, "scale": d["scale"], "camera": d["camera"],
                     "seconds": d["seconds"], "beat": d.get("beat", ""),
                     "pinned": pin or "", "prompt": d["prompt"][:70]})
    st.dataframe(rows, hide_index=True, **FULL)

    changes = proj["plan"].get("planner_changes") or []
    hard = [x for x in changes if "R4" not in x]
    m = st.columns(3)
    m[0].metric("repairs that changed a CHOICE", len(hard),
                help="R4 timing rewrites are the planner's job by design and are "
                     "not counted. Anything here is a shot that stopped being what "
                     "the writer intended.")
    m[1].metric("R4 timing rewrites", len(changes) - len(hard))
    m[2].metric("moving shots", sum(1 for d in frozen
                                    if d["camera"] not in ("", "none")))
    for x in hard:
        st.write("- " + x)
    st.caption(f"no more than {MAX_MOVING_RUN} moving shots in a row")
    with st.expander("energy profile and the arc"):
        st.text(energy_profile([(d.get("energy") if d.get("energy") is not None
                                 else SCALE_ENERGY.get(d["scale"], 0.5))
                                for d in frozen]))
        st.text(energy_profile([energy_arc_target(i, len(frozen))
                                for i in range(len(frozen))]))
        st.caption("top: the film's declared energies. bottom: the target arc "
                   f"(calm -> peak at {int(ENERGY_ARC_PEAK*100)}% -> resolve).")

    st.subheader("per-shot motion pin")
    st.caption("A pin OVERRIDES the director for that shot. It is not a constraint "
               "the optimisation respects: the director is not told about it, so "
               "its budget and its 180° rule were decided against the motion the "
               "pin replaces. That limitation is real and is written in the code.")
    for i, d in enumerate(frozen):
        pin = proj["shots"][i].get("motion_pin") if i < len(proj["shots"]) else None
        cc = st.columns([2, 2, 1])
        cc[0].write(f"shot {i+1}: {d['camera']}")
        options = ["(director decides)", "none", "zoom-in", "zoom-out", "pan-left",
                   "pan-right", "tilt-up", "tilt-down", "rolling-clockwise",
                   "rolling-anticlockwise"]
        cur = options.index(pin) if pin in options else 0
        sel = cc[1].selectbox("pin", options, index=cur, key=f"pin_{i}",
                              label_visibility="collapsed")
        if cc[2].button("set", key=f"setpin_{i}"):
            proj["shots"][i]["motion_pin"] = None if sel.startswith("(") else sel
            proj["plan"]["approved"] = False
            proj["status"]["plan"] = "stale"
            fp.save(proj)
            st.rerun()

    st.subheader("gate: freeze the plan")
    if proj["plan"].get("approved"):
        st.success(f"FROZEN and approved at {proj['plan'].get('approved_at')}. "
                   f"RENDER will use exactly this list.")
    elif st.button("APPROVE AND FREEZE THIS PLAN"):
        fp.approve_plan(proj)
        save(proj)
        st.rerun()


# --------------------------------------------------------------------------- #
# CAMERA
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="applying the shake to this shot's raw frames")
def camera_preview(film: str, shot_no: int, hold: int, scale: str, kind: str,
                   seconds: float, amount_scale: float, seed: int):
    """A/B GIF of this shot's RAW frames, steady against shaken.

    Uses `handheld.stretch_to` -> `handheld.spec_for_shot` ->
    `handheld.apply_handheld` in the same order `make_film` does, so what you
    preview is what renders. The numbers come from `handheld.shake_path`, not from
    arithmetic here.
    """
    from PIL import Image, ImageDraw

    d = FILMS / film / "raw" / f"shot{shot_no:02d}"
    files = sorted(d.glob("f*.png"))
    if not files:
        return None, None
    raw = [Image.open(f).convert("RGB") for f in files]
    base = handheld.stretch_to(raw, hold)
    spec = handheld.spec_for_shot(scale, kind, seconds, seed=int(seed), fps=FPS,
                                  amount_scale=float(amount_scale))
    shaken = handheld.apply_handheld(base, spec, fps=FPS)
    path = handheld.shake_path(len(base), spec, FPS)
    numbers = {"peak translation (px)": round(float(abs(path["dx"]).max()), 2),
               "roll p-p (deg)": round(float(path["roll_deg"].max()
                                             - path["roll_deg"].min()), 2),
               "envelope floor": round(spec.floor, 2),
               "frames at the delivered length": len(base)}
    T = 236
    frames = []
    for i in range(len(base)):
        c = Image.new("RGB", (2 * T + 8, T + 18), (16, 16, 20))
        dr = ImageDraw.Draw(c)
        dr.text((4, 3), f"steady  |  shaken x{amount_scale:.2f}   "
                        f"frame {i+1}/{len(base)}", fill=(228, 228, 232))
        c.paste(base[i].resize((T, T), Image.LANCZOS), (2, 16))
        c.paste(shaken[i].resize((T, T), Image.LANCZOS), (T + 6, 16))
        frames.append(c)
    out = FILMS / film / f"preview_camera_shot{shot_no:02d}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0, optimize=True)
    return str(out), numbers


def stage_camera(proj: dict):
    st.header("CAMERA")
    st.caption("The shake, per shot, with units. `handheld` produces the spec and "
               "applies it — these are its knobs, and the render reads them from "
               "the project, so an edit here reaches the film.")

    cam = proj.setdefault("camera", {"amount_scale": 1.0, "overrides": {}})
    c = st.columns([2, 3])
    scale = c[0].slider("global shake scale", 0.0, 3.0,
                        float(cam.get("amount_scale", 1.0)), 0.05,
                        help="1.0 is the researched default hold. This multiplies "
                             "BOTH translation and roll for every shot.")
    if abs(scale - float(cam.get("amount_scale", 1.0))) > 1e-9:
        cam["amount_scale"] = round(float(scale), 3)
        fp.save(proj)

    raws = [d.name for d in sorted(FILMS.iterdir())
            if d.is_dir() and (d / "raw").is_dir()] if FILMS.exists() else []
    if not raws:
        c[1].info("No rendered film has raw frames on disk, so there is nothing "
                  "to preview against. Render one from the RENDER stage first — "
                  "previews compare STEADY against SHAKEN on the same frames.")
    else:
        film = c[1].selectbox("preview against the raw frames of", raws,
                              index=raws.index(proj.get("film", {}).get("name"))
                              if proj.get("film", {}).get("name") in raws else 0)

    frozen = proj["plan"].get("frozen") or proj["shots"]
    st.subheader("per shot")
    for i, d in enumerate(frozen):
        shot_no = i + 1
        secs = float(d.get("seconds") or 2.5)
        kind = "none"
        if d.get("camera") not in (None, "", "none"):
            import motion_vocab as _mv  # the vocabulary owns kind, not this file
            kind = _mv.get(d["camera"]).kind
        p_ = fp.shake_params(proj, i)
        with st.expander(f"{shot_no}. {d.get('scale')} · {d.get('camera') or 'none'}"
                         f" · {secs:.2f}s", expanded=False):
            cc = st.columns([1, 1, 1, 1])
            enabled = cc[0].checkbox("shake on", value=bool(p_["enabled"]),
                                     key=f"cam_on_{i}")
            amt = cc[1].number_input("scale", 0.0, 3.0,
                                     float(p_["amount_scale"]), 0.05,
                                     key=f"cam_amt_{i}")
            seed = cc[2].number_input("seed", 0, 999, int(p_["seed"]), 1,
                                      key=f"cam_seed_{i}")
            save_it = cc[3].button("save", key=f"cam_save_{i}")
            if save_it:
                fp.set_shake_param(
                    proj, i,
                    enabled=None if enabled else False,
                    amount_scale=None if abs(amt - float(cam.get("amount_scale", 1.0))) < 1e-9 else float(amt),
                    seed=None if int(seed) == i + 3 else int(seed))
                fp.save(proj)
                st.rerun()
            hold = handheld.hold_frames(type("S", (), {"seconds": secs})(), FPS)
            if raws and st.button("preview this shot", key=f"cam_prev_{i}"):
                gif, numbers = camera_preview(film, shot_no, hold,
                                              d.get("scale", "medium"), kind, secs,
                                              float(p_["amount_scale"]),
                                              int(p_["seed"]))
                if gif is None:
                    st.warning(f"no raw frames for shot {shot_no} in {film} — "
                               f"nothing to preview against")
                else:
                    st.image(gif)
                    nn = st.columns(len(numbers))
                    for k, (lab, v) in enumerate(numbers.items()):
                        nn[k].metric(lab, v)
                    st.caption("This is what RENDER will do, because it calls the "
                               "same three functions in the same order.")


# --------------------------------------------------------------------------- #
# GRADE
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="applying the palette to this shot's raw frames")
def look_preview(film: str, shot_no: int, look: str):
    """This shot's first raw frame, before and after the palette.

    `looks.apply_look` does the grading. The separation is measured by
    `looks.look_distances` on the shot's own frames, not asserted here.
    """
    from PIL import Image, ImageDraw
    import looks as looks_mod

    d = FILMS / film / "raw" / f"shot{shot_no:02d}"
    files = sorted(d.glob("f*.png"))
    if not files:
        return None, None
    raw = [Image.open(f).convert("RGB") for f in files]
    looked = looks_mod.apply_look(raw, look)
    floor = looks_mod.palette_floor(raw)
    T = 300
    c = Image.new("RGB", (2 * T + 8, T + 18), (16, 16, 20))
    dr = ImageDraw.Draw(c)
    dr.text((4, 3), f"as generated  |  look: {look}   "
                    f"(palette floor {floor:.2f})", fill=(228, 228, 232))
    c.paste(raw[0].resize((T, T), Image.LANCZOS), (2, 16))
    c.paste(looked[0].resize((T, T), Image.LANCZOS), (T + 6, 16))
    out = FILMS / film / f"preview_look_shot{shot_no:02d}_{look}.png"
    c.save(out)
    return str(out), {"palette floor": round(floor, 2)}


@st.cache_data(show_spinner="measuring the palettes on held content")
def arc_legibility(film: str, shot_no: int, names: tuple):
    """`looks.look_distances` on one shot's held frames.

    Held source, so the number is about the palette and not about content -- the
    confound that invalidated the seam metric in handoff 9.9 cannot apply.
    """
    from PIL import Image
    import looks as looks_mod
    files = sorted((FILMS / film / "raw" / f"shot{shot_no:02d}").glob("f*.png"))
    if not files or len(names) < 2:
        return None
    src = [Image.open(f).convert("RGB") for f in files]
    return looks_mod.look_distances(src, list(names))


def stage_grade(proj: dict):
    st.header("GRADE")
    st.caption("The shared photographic baseline, and one palette per scene. Both "
               "are read by the render from the project. Legibility is measured by "
               "`looks.look_distances` on held frames — the one place a palette "
               "number means anything.")

    import looks as looks_mod
    grade = proj.setdefault("grade", {"technical": True, "looks": None})
    c = st.columns([2, 3])
    tech = c[0].checkbox(
        "shared technical baseline (exposure / contrast / grain)",
        value=bool(grade.get("technical", True)),
        help="every shot is matched to one photographic baseline so no cut reads "
             "as an exposure error. Reported absolutely, in grey levels.")
    if bool(tech) != bool(grade.get("technical", True)):
        grade["technical"] = bool(tech)
        fp.save(proj)

    frozen = proj["plan"].get("frozen") or proj["shots"]
    order, seen = [], set()
    for d in frozen:
        sc = d.get("scene") or "scene1"
        if sc not in seen:
            seen.add(sc)
            order.append(sc)
    existing = grade.get("looks") or []
    per_scene = {}
    for i, sc in enumerate(order):
        idx = [j for j, d in enumerate(frozen)
               if (d.get("scene") or "scene1") == sc]
        cur = existing[idx[0]] if idx and idx[0] < len(existing) else \
            looks_mod.ARC[min(i, len(looks_mod.ARC) - 1)]
        per_scene[sc] = c[1].selectbox(f"palette for scene {sc!r}", looks_mod.ARC,
                                       index=looks_mod.ARC.index(cur)
                                       if cur in looks_mod.ARC else 0,
                                       key=f"look_{sc}")

    new_looks = [per_scene.get(d.get("scene") or "scene1") for d in frozen]
    if new_looks != existing:
        grade["looks"] = new_looks
        proj["status"]["grade"] = "set"
        fp.save(proj)

    st.subheader("the film's palette path")
    st.dataframe([{"#": i + 1, "scene": d.get("scene") or "scene1",
                   "palette": (new_looks[i] if i < len(new_looks) else "-")}
                  for i, d in enumerate(frozen)], hide_index=True, **FULL)

    raws = [d.name for d in sorted(FILMS.iterdir())
            if d.is_dir() and (d / "raw").is_dir()] if FILMS.exists() else []
    if not raws:
        st.info("No raw frames on disk yet, so there is nothing to preview or to "
                "measure against. Render once, then both work.")
        return
    film = st.selectbox("against the raw frames of", raws,
                        index=raws.index(proj.get("film", {}).get("name"))
                        if proj.get("film", {}).get("name") in raws else 0)

    used, order_used = set(), []
    for lk in new_looks:
        if lk and lk not in used:
            used.add(lk)
            order_used.append(lk)
    if len(order_used) >= 2:
        st.subheader("legibility of the transitions this film will make")
        rows = arc_legibility(film, 1, tuple(order_used))
        if rows:
            st.dataframe([{"from": r["from"], "to": r["to"],
                           "technical": r["technical"],
                           "stylistic": r["stylistic"],
                           "floor": r["floor"], "verdict": r["verdict"]}
                          for r in rows], hide_index=True, **FULL)
            leg = sum(1 for r in rows if "LEGIBLE" in str(r["verdict"]))
            (st.success if leg == len(rows) else st.warning)(
                f"{leg}/{len(rows)} adjacent transitions land in the legible band")
    else:
        st.info("One palette across the whole film, so there is no transition to "
                "measure. That is a valid choice, not a gap.")

    st.subheader("preview a palette on a shot")
    cc = st.columns([1, 2])
    shot_no = cc[0].number_input("shot", 1, max(1, len(frozen)), 1, 1)
    lk = cc[1].selectbox("palette", looks_mod.ARC,
                         index=looks_mod.ARC.index(new_looks[shot_no - 1])
                         if shot_no - 1 < len(new_looks)
                         and new_looks[shot_no - 1] in looks_mod.ARC else 0,
                         key="preview_look")
    if st.button("preview this palette"):
        img, nums = look_preview(film, int(shot_no), lk)
        if img is None:
            st.warning(f"no raw frames for shot {shot_no} in {film}")
        else:
            st.image(img)
            st.caption("left: as generated. right: with the palette applied by "
                       "`looks.apply_look`.")


# --------------------------------------------------------------------------- #
# RENDER
# --------------------------------------------------------------------------- #

def shot_stages(film: str, i: int) -> dict:
    """Which stages of shot i exist on disk. Filesystem only -- cheap."""
    d = FILMS / film
    return {st: len(sorted((d / st / f"shot{i+1:02d}").glob("f*.png")))
            if (d / st / f"shot{i+1:02d}").exists() else 0
            for st in ("raw", "smooth", "shaken")}


def sync_render_state(proj: dict, film: str) -> bool:
    """Bring the project's per-shot render state in line with the disk.

    The project is the source of truth for what you APPROVED; the disk is the
    truth for what EXISTS. An approved shot never gets downgraded by a rescan.
    """
    changed = False
    for i, d in enumerate(proj["shots"]):
        st = shot_stages(film, i)
        have = st["shaken"] > 0
        cur = (d.get("render") or {}).get("status", "pending")
        want = "rendered" if have else "pending"
        if cur == "approved" or cur == want:
            continue
        d.setdefault("render", {})["status"] = want
        d["render"]["frames"] = st["shaken"]
        changed = True
    return changed


def stage_render(proj: dict):
    st.header("RENDER")
    st.caption("One shot at a time, and the per-shot stages stay on disk. "
               "Generate a shot, look at it, approve it or render just that one "
               "again — then assemble, which regenerates nothing.")

    frozen = proj["plan"].get("frozen") or []
    if not frozen:
        st.warning("No frozen plan. The PLAN stage has to freeze one first: the "
                   "render must use the plan you approved, not a fresh one.")
        return

    c = st.columns([2, 1, 1, 2])
    films = [d.name for d in sorted(FILMS.iterdir()) if d.is_dir()] \
        if FILMS.exists() else []
    default_name = proj.get("film", {}).get("name") or proj["name"]
    film = c[0].text_input("render into (a film name)", value=default_name,
                           help="its per-shot stages land in "
                                "output/film/<name>/{raw,smooth,shaken}/shotNN/")
    c[1].metric("shots", len(frozen))
    done = sum(1 for i in range(len(frozen)) if shot_stages(film, i)["shaken"])
    c[2].metric("rendered", f"{done}/{len(frozen)}")
    if sync_render_state(proj, film):
        fp.save(proj)
    if proj["film"].get("name") != film:
        proj["film"]["name"] = film
        fp.save(proj)

    st.dataframe(
        [{"#": i + 1, "beat": d.get("beat", ""), "scale": d["scale"],
          "camera": d["camera"], "seconds": d["seconds"],
          "raw": shot_stages(film, i)["raw"],
          "RAFT": shot_stages(film, i)["smooth"],
          "shaken": shot_stages(film, i)["shaken"],
          "status": (proj["shots"][i].get("render") or {}).get("status", "pending")
          if i < len(proj["shots"]) else "-"}
         for i, d in enumerate(frozen)], hide_index=True, **FULL)

    st.subheader("look at a shot")
    cc = st.columns([1, 3])
    i = int(cc[0].number_input("shot", 1, len(frozen), 1, 1)) - 1
    if cc[1].button("show its stages"):
        st.session_state["render_shot"] = i
    i = st.session_state.get("render_shot", 0)
    stg = shot_stages(film, i)
    if not any(stg.values()):
        st.info(f"Shot {i+1} has nothing on disk yet. Render it below — "
                f"`make_film --only {i+1}` does not touch the other shots.")
    else:
        cols = st.columns(3)
        for k, (label, dirname) in enumerate([("raw (16 frames)", "raw"),
                                              ("RAFT x2", "smooth"),
                                              ("shaken", "shaken")]):
            d = FILMS / film / dirname / f"shot{i+1:02d}"
            fs = sorted(d.glob("f*.png"))
            if fs:
                cols[k].image(str(fs[len(fs) // 2]), caption=f"{label} — "
                                                             f"{len(fs)} frames")
            else:
                cols[k].caption(f"{label}: not on disk")

    st.subheader("act on a shot")
    a = st.columns([1, 1, 1, 3])
    if a[0].button(f"render shot {i+1}"):
        try:
            jr.start("render", project=proj["name"], name=film, models="off",
                     looks="auto", only=i + 1)
            st.success(f"started — shot {i+1} only. The other shots are untouched.")
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))
    if a[1].button(f"approve shot {i+1}"):
        proj["shots"][i].setdefault("render", {})["status"] = "approved"
        proj["shots"][i]["render"]["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        fp.save(proj)
        st.rerun()
    if a[2].button("render ALL remaining"):
        try:
            jr.start("render", project=proj["name"], name=film, models="off",
                     looks="auto")
            st.success("started — the full plan. ~9-14 min per shot, one at a "
                       "time.")
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))

    st.subheader("assemble")
    missing = [i + 1 for i in range(len(frozen))
               if not shot_stages(film, i)["shaken"]]
    if missing:
        st.warning(f"Cannot assemble yet: shots {missing} have no shaken frames. "
                   f"Assembly regenerates NOTHING, so it needs them all.")
    else:
        st.success("every shot is on disk. Assembly will grade and cut them "
                   "without generating anything.")
    if st.button("ASSEMBLE THE FILM (no generation)"):
        try:
            jr.start("render", project=proj["name"], name=film, models="off",
                     looks="auto", assemble=True)
            st.success("started — reading the shots from disk")
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))

    if jr.status().get("status") in ("queued", "running"):
        jb = jr.status()
        pr = jb.get("progress") or {}
        st.info(f"{jb.get('task')} is {jb.get('status')}"
                + (f" — {pr['last']}" if pr.get("last") else "")
                + "   (the sidebar has the log)")


# --------------------------------------------------------------------------- #
# TIMELINE
# --------------------------------------------------------------------------- #

def stage_timeline(proj: dict):
    st.header("TIMELINE")
    st.caption("The cut: order, the exposure sheet, and the assembled film. The "
               "holds come from `handheld.hold_frames` — the same function the "
               "render uses — so a timecode here is not a second opinion.")

    shots = fp.frozen_shots(proj)
    if not shots:
        st.warning("No frozen plan, so there is no cut to show. Freeze one in "
                   "PLAN first.")
        return
    film = proj["film"].get("name") or proj["name"]

    rows, t = [], 0.0
    for i, s in enumerate(shots):
        hold = handheld.hold_frames(s, FPS)
        rows.append({"#": i + 1, "in": f"{t:6.2f}s", "out": f"{t + hold/FPS:6.2f}s",
                     "hold (frames)": hold, "beat": s.beat, "scale": s.scale,
                     "camera": s.camera or "none", "packshot at": f"{s.seconds:.2f}s"})
        t += hold / FPS
    c = st.columns(4)
    c[0].metric("duration", f"{t:.2f} s")
    c[1].metric("shots", len(shots))
    c[2].metric("frames", sum(r["hold (frames)"] for r in rows))
    c[3].metric("moving", sum(1 for s in shots if s.camera not in ("", "none")))
    st.dataframe(rows, hide_index=True, **FULL)
    st.caption("`in`/`out` are cumulative; `packshot at` is the shot's own "
               "declared length before the exposure sheet stretches it to the hold.")

    st.subheader("order")
    st.caption("Reordering the FROZEN plan changes what assembly produces. Any "
               "story edit invalidates the plan entirely, so this is the only "
               "place the cut's order can be changed without re-planning.")
    frozen = proj["plan"]["frozen"]
    for i in range(len(frozen)):
        b = st.columns([6, 1, 1])
        b[0].write(f"{i+1}. {frozen[i]['beat']} · {frozen[i]['scale']} · "
                   f"{frozen[i]['camera'] or 'none'} — "
                   f"{str(frozen[i]['prompt'])[:60]}")
        if b[1].button("↑", key=f"tl_up_{i}") and i > 0:
            frozen[i - 1], frozen[i] = frozen[i], frozen[i - 1]
            proj["plan"]["approved"] = False
            proj["status"]["plan"] = "reordered"
            fp.save(proj)
            st.rerun()
        if b[2].button("↓", key=f"tl_dn_{i}") and i < len(frozen) - 1:
            frozen[i + 1], frozen[i] = frozen[i], frozen[i + 1]
            proj["plan"]["approved"] = False
            proj["status"]["plan"] = "reordered"
            fp.save(proj)
            st.rerun()

    st.subheader("the assembled film")
    d = FILMS / film
    stats_p = d / "stats.json"
    if not stats_p.exists():
        # A project's default film name is the project's own name, but a render
        # that already exists may live under another one (ensemble1, say). Without
        # this the TIMELINE simply cannot see it and the workbench looks emptier
        # than the disk is.
        have = [x.name for x in sorted(FILMS.iterdir())
                if x.is_dir() and (x / "stats.json").exists()] if FILMS.exists() else []
        if have:
            cc = st.columns([3, 1])
            pick = cc[0].selectbox("an assembled film already exists — attach it",
                                   have, key="tl_attach")
            if cc[1].button("attach"):
                proj["film"]["name"] = pick
                fp.save(proj)
                st.rerun()
        st.info(f"Nothing assembled at output/film/{film}/ yet. RENDER's ASSEMBLE "
                f"writes it, and it regenerates nothing — it needs every shot's "
                f"shaken frames on disk first.")
        return
    stats = json.loads(stats_p.read_text(encoding="utf-8"))
    srows = stats.get("shots", [])
    gen = [r.get("gen_seconds", 0) for r in srows if r.get("gen_seconds")]
    m = st.columns(3)
    m[0].metric("duration", f"{stats.get('duration_s', '?')} s")
    m[1].metric("shots in the cut", len(srows))
    m[2].metric("generation", f"{sum(gen)/60:.0f} min" if gen else "reused")
    # the film record, written back to the project: this is the artifact's identity
    rec = {"name": film, "duration_s": stats.get("duration_s"),
           "frames": stats.get("frames") or sum(
               r.get("smooth_frames", 0) for r in srows)}
    if proj["film"] != rec:
        proj["film"] = rec
        proj["status"]["timeline"] = "assembled"
        fp.save(proj)
    if (d / "film.mp4").exists():
        st.video(str(d / "film.mp4"))
        st.caption(f"[film.mp4]({str((d / 'film.mp4')).replace(chr(92), '/')}) — "
                   f"the deliverable.")
    if (d / "shots.png").exists():
        st.image(str(d / "shots.png"), caption="the contact sheet")
    st.dataframe([{"#": r.get("shot"), "scale": r.get("scale"),
                   "camera": r.get("camera"), "model": r.get("model"),
                   "look": r.get("look"), "hold (s)": r.get("hold_s"),
                   "frames": r.get("smooth_frames"),
                   "shake": r.get("shake_spec")} for r in srows],
                 hide_index=True, **FULL)
    if gen:
        st.caption(f"measured {min(gen):.0f}-{max(gen):.0f} s per shot, one shot at "
                   f"a time. State the price before spending it.")


# --------------------------------------------------------------------------- #
# the stages still to come, stated honestly
# --------------------------------------------------------------------------- #

TODO = {}


def stage_todo(name: str):
    st.header(name)
    st.info("Not built yet. " + TODO.get(name, ""))
    st.caption("This stage is deliberately a stub rather than a fake: a control "
               "that does nothing is worse than an honest gap.")


@st.cache_data(show_spinner="recomputing the grading from the assembled frames")
def grade_units(film: str, holds: tuple):
    """Technical offset from the SHARED baseline, in decomp units AND grey levels.

    Both, because five separate wrong numbers in this project came from a unit
    confusion and this is the cheapest place to stop the sixth. Absolute, via
    `grade_match.technical_offset` -- the normalised version printed 1e5 on a film
    like this, which is how the normaliser claim was caught.
    """
    from PIL import Image
    import grade_match as gm
    frames = sorted((FILMS / film).glob("f*.png"))
    if not frames:
        return None
    imgs = [Image.open(f).convert("RGB") for f in frames]
    sigs, at = [], 0
    for h in holds:
        seg = imgs[at:at + int(h)]
        at += int(h)
        if len(seg) < 2:
            continue
        sigs.append(gm.signature(seg))
    if len(sigs) < 2:
        return None
    tgt = gm.technical_target(sigs)
    grey = gm.SCALES[1]
    return [{"#": i + 1,
             "decomp units": round(gm.technical_offset(sg, tgt), 3),
             "grey levels of contrast": round(gm.technical_offset(sg, tgt) * grey, 2)}
            for i, sg in enumerate(sigs)]


def stage_audit(proj: dict):
    st.header("AUDIT")
    st.caption("The evidence, beside the decisions it justifies. Every number is "
               "produced by calling the pipeline's own functions; nothing here is "
               "a second opinion.")

    problems, shape = revalidate(proj)
    m = st.columns(4)
    m[0].metric("story shape", shape)
    m[1].metric("story breaks", sum(1 for p in problems if p.startswith("story:")))
    m[2].metric("story warnings",
                sum(1 for p in problems if p.startswith("story-link:")))
    m[3].metric("shots", len(proj["shots"]))
    for p in problems:
        (st.error if p.startswith("story:") else st.warning)(p)

    st.subheader("provenance — which scout the film actually came from")
    prov = proj["story"].get("provenance") or {}
    if not prov:
        st.info("No provenance stored. The ensemble writer records it on the "
                "proposal; adopt a cached ensemble proposal or re-roll.")
    else:
        st.dataframe([{"farm": k, "ideas": v["ideas"], "landed": v["used"],
                       "mean best overlap": v["mean_best"]}
                      for k, v in prov.items() if not k.startswith("_")],
                     hide_index=True, **FULL)
        st.caption(f"{prov.get('_final_prompts_traced', '?')} of "
                   f"{prov.get('_final_prompts', '?')} final prompts trace back to "
                   f"a scout's idea. Without this, four farms paying for one looks "
                   f"identical to an ensemble working.")

    st.subheader("the camera, in units")
    shots = fp.frozen_shots(proj) or fp.shot_objects(proj)
    rows = []
    for i, sh in enumerate(shots):
        params = fp.shake_params(proj, i)
        hold = handheld.hold_frames(sh, FPS)
        spec = handheld.spec_for_shot(sh.scale, sh.kind, sh.seconds,
                                      seed=int(params["seed"]), fps=FPS,
                                      amount_scale=float(params["amount_scale"]))
        path = handheld.shake_path(hold, spec, FPS)
        rows.append({"#": i + 1, "scale": sh.scale, "camera": sh.camera or "none",
                     "hold": hold,
                     "peak (px)": round(float(abs(path["dx"]).max()), 2),
                     "roll p-p (deg)": round(float(path["roll_deg"].max()
                                                   - path["roll_deg"].min()), 2),
                     "shake": "off" if not params["enabled"] else
                              f"x{params['amount_scale']:.2f}"})
    st.dataframe(rows, hide_index=True, **FULL)

    st.subheader("the grading, in units")
    film = proj["film"].get("name")
    if not film or not (FILMS / film / "stats.json").exists():
        st.info("No assembled film attached, so there are no graded frames to "
                "measure. TIMELINE can attach one.")
    else:
        stats = json.loads((FILMS / film / "stats.json").read_text(encoding="utf-8"))
        holds = tuple(r.get("smooth_frames", 0) for r in stats.get("shots", []))
        rows = grade_units(film, holds)
        if rows:
            st.dataframe(rows, hide_index=True, **FULL)
            st.caption("Technical offset from the shared baseline. Absolute, since "
                       "the distance already has physical units.")

    st.subheader("the writer audit — leaks and variance")
    ad = FILMS / "writer_audit"
    txt = ad / "audit.txt"
    if not txt.exists():
        st.info("No audit on disk. 'run the writer audit' in the job panel "
                "produces one; it costs LLM calls, so it is not run on page load.")
    else:
        body = txt.read_text(encoding="utf-8", errors="replace")
        st.caption(f"the pipeline's own report — {len(body.splitlines())} lines "
                   f"from {txt.relative_to(ROOT)}")
        st.text("\n".join(body.splitlines()[:60]))
    notes = ad / "LEAK_NOTES.md"
    if notes.exists():
        with st.expander("LEAK_NOTES.md — the by-hand reading of the leak result"):
            st.markdown(notes.read_text(encoding="utf-8", errors="replace"))

    st.subheader("director A/B — does the film go somewhere?")
    st.caption("Free, renders nothing. Greedy vs global on the same list, scored "
               "with the same objective, plus what the planner would have to undo.")
    if st.button("run the A/B"):
        with st.spinner("comparing..."):
            st.code(compare_modes(_planned_shots(proj), scorer=scorer()),
                    language="text")


# --------------------------------------------------------------------------- #
# the job panel
# --------------------------------------------------------------------------- #

def job_panel():
    """The runner, on screen. Polls -- never blocks.

    NOTE: the local is `job`, not `st`. Naming it `st` shadows the streamlit
    module inside this function and every `st.sidebar` call below it dies.
    """
    st.sidebar.divider()
    st.sidebar.subheader("jobs")
    job = jr.status()
    state = job.get("status")
    pr = job.get("progress") or {}
    if state == "idle":
        st.sidebar.caption("nothing has run yet this session")
    else:
        st.sidebar.caption(f"{job.get('task')} · {state} · {job.get('cost', '')}")
        if pr.get("last"):
            st.sidebar.write(pr["last"])
        done, total = pr.get("done"), pr.get("total")
        if done and total:
            st.sidebar.progress(min(1.0, done / total))
        if job.get("message"):
            st.sidebar.caption(job["message"])
        if state in (jr.RUNNING, jr.QUEUED) and st.sidebar.button("stop it"):
            jr.stop()
            st.rerun()
        with st.sidebar.expander(f"log tail ({job.get('log_lines') or 0} lines)"):
            st.text(jr.tail(job["id"], 25))
    if st.sidebar.button("refresh"):
        st.rerun()
    if st.sidebar.checkbox("auto-refresh while a job runs", value=False):
        if jr.status().get("status") in (jr.RUNNING, jr.QUEUED):
            time.sleep(3)
            st.rerun()

    with st.sidebar.expander("start a job"):
        st.caption("One at a time. The runner refuses a second.")
        for t in jr.describe_tasks():
            if t["task"] == "selftest":
                continue
            st.caption(f"**{t['task']}** - {t['cost']}  \n{t['desc']}")
        if st.button("re-roll the story"):
            try:
                jr.start("story", concept=st.session_state.get("concept", ""),
                         fresh=True)
                st.success("started")
            except RuntimeError as exc:
                st.error(str(exc))
        if st.button("run the writer audit"):
            try:
                jr.start("audit", runs=0, quiet=True)
                st.success("started")
            except RuntimeError as exc:
                st.error(str(exc))


def main():
    st.set_page_config(page_title="workbench", layout="wide")
    st.title("workbench")

    paths = fp.list_projects()
    st.sidebar.header("project")
    if not paths:
        st.warning("No projects in workspace/. Create one below, then adopt a "
                   "cached proposal in STORY.")
        name = st.text_input("new project name", "film1")
        if st.button("create"):
            fp.save(fp.new(name, st.session_state.get("concept", "")))
            st.rerun()
        job_panel()
        return
    names = [p.stem for p in paths]
    pick = st.sidebar.selectbox("open", names, index=0)
    proj = fp.load(pick)

    st.sidebar.text_input("concept (for a re-roll)", value=proj.get("concept", ""),
                          key="concept")
    done = ", ".join(f"{k}={v}" for k, v in proj["status"].items()
                     if v != "empty")
    st.sidebar.caption("stage status: " + (done or "nothing yet"))
    stage = st.sidebar.radio("stage", STAGES,
                             index=STAGES.index(st.session_state.get("stage", "STORY"))
                             if st.session_state.get("stage") in STAGES else 0)
    st.session_state["stage"] = stage

    # adopting a proposal is the one thing STORY needs beyond editing
    if stage == "STORY" and not proj["shots"]:
        st.header("STORY")
        st.info("This project has no shots yet. Adopt a cached proposal, or "
                "re-roll one from the job panel (that costs ~4-7 min).")
        cand = []
        if PROPOSALS.exists():
            for f in sorted(PROPOSALS.glob("*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    raw = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                shots, _ = validate(raw)
                if shots:
                    cand.append((f, raw, f"{f.stem.split('__')[0][:30]} · "
                                          f"{f.stem.split('__')[1] if '__' in f.stem else '?'}"
                                          f" · {len(shots)} shots {story_shape(shots)}"))
        if cand:
            i = st.selectbox("cached proposals", range(len(cand)),
                             format_func=lambda i: cand[i][2])
            if st.button("adopt this proposal"):
                fp.shots_from_raw(proj, cand[i][1], proposal_path=str(cand[i][0]))
                save(proj)
                st.rerun()
    else:
        {"STORY": stage_story, "PLAN": stage_plan, "CAMERA": stage_camera,
         "GRADE": stage_grade, "RENDER": stage_render,
         "TIMELINE": stage_timeline, "AUDIT": stage_audit,
         }.get(stage, lambda p: stage_todo(stage))(proj)

    job_panel()


if __name__ == "__main__":
    main()
