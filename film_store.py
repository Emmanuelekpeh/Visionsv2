"""
film_store.py -- the production store: stable ids, immutable takes, derived status.

WHY THIS EXISTS
---------------
The workbench's data flow was inconsistent because three things claimed to be the
source of truth for the same facts, and the joins between them were ARRAY
POSITIONS:

    proj["shots"]              <- what STORY edits
    proj["plan"]["frozen"]     <- what RENDERS
    output/film/<name>/*.png   <- what EXISTS

`camera.overrides` was keyed `str(index)` over one array and read with the index of
the other; `grade.looks` was a flat list built from one order and consumed in
another; `render.status` was written to `shots[i]` and displayed from `frozen[i]`.
Reordering the timeline moved every one of those onto a different shot, silently.
(Measured: see AUDIT.md sections 1-3.)

This module removes the class of bug rather than the instances:

  * A shot has a STABLE STRING ID and lives in its OWN FILE. Order lives only in
    the timeline. Reordering, deleting or inserting cannot move a setting onto
    another shot, because no setting is addressed by position.
  * A take is an IMMUTABLE directory. Re-rendering never overwrites; it writes the
    next take. That is what makes "try another camera move and keep the better
    one" possible at 540-788 s per shot.
  * Staleness is DERIVED, never stored. A take is stale iff hashing the shot's
    current render inputs differs from the hash the take was rendered from. There
    is no status string to fall out of date, and no hand-maintained stage state.
  * Cost is DERIVED from the takes on disk, so the price of the next render is on
    screen before it is spent, not in a stats file afterwards.

The pipeline's own functions are still the only measurement. This module stores
DECISIONS; it reimplements nothing.

LAYOUT
------
    productions/<slug>/
        project.json                 the film: scenes, timeline order, settings
        shots/<shot_id>.json         one file per shot -- so a shot is a thing
        takes/<shot_id>/<take>/      one immutable render
            take.json                the decision, the cost, the ladder shape
            frames/*.png             delivered frames (input to grading)
        renders/<render_id>/         an assembled film + its receipts
            film.mp4  stats.json  shots.png
        audio/*.mp3                  narration lines, music
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from edit_planner import Shot, plan

ROOT = Path(__file__).resolve().parent
PRODUCTIONS = ROOT / "productions"
PROPOSALS = ROOT / "output" / "film" / "proposals"

SCHEMA = 2
CAMERAS = ["none", "zoom-in", "zoom-out", "pan-left", "pan-right", "tilt-up",
           "tilt-down", "rolling-clockwise", "rolling-anticlockwise"]
SCALES = ["macro", "close", "medium", "wide", "extreme-wide"]


# --------------------------------------------------------------------------- #
# ids
# --------------------------------------------------------------------------- #

def slugify(text: str, fallback: str = "film") -> str:
    s = re.sub(r"\W+", "-", (text or "").lower()).strip("-")[:60]
    return s or fallback


def next_id(existing: Iterable[str], prefix: str = "s") -> str:
    """The next free short id. Never an index -- that is the whole point."""
    used = set()
    for x in existing:
        m = re.fullmatch(re.escape(prefix) + r"(\d+)", str(x))
        if m:
            used.add(int(m.group(1)))
    n = 1
    while n in used:
        n += 1
    return f"{prefix}{n:02d}"


# --------------------------------------------------------------------------- #
# the project
# --------------------------------------------------------------------------- #

def dir_for(slug: str) -> Path:
    return PRODUCTIONS / slug


def project_path(slug: str) -> Path:
    return dir_for(slug) / "project.json"


def shot_path(slug: str, shot_id: str) -> Path:
    return dir_for(slug) / "shots" / f"{shot_id}.json"


def takes_dir(slug: str, shot_id: str) -> Path:
    return dir_for(slug) / "takes" / shot_id


def renders_dir(slug: str) -> Path:
    return dir_for(slug) / "renders"


def new_project(name: str, concept: str = "") -> Dict:
    slug = slugify(name)
    return {
        "schema": SCHEMA,
        "slug": slug,
        "name": name or slug,
        "concept": concept,
        "created": _now(),
        "saved": None,
        # the creative layer
        "logline": "",
        "characters": "",
        "scenes": [{"id": "sc1", "name": "scene 1", "palette": "neutral"}],
        # the timeline: ORDER CHOICE ONLY. Everything else lives on the shot.
        "timeline": [],
        # settings that genuinely apply to the whole film
        "settings": {
            "fps": 16,
            "shake_scale": 1.0,
            "seed_base": 1103,
            "technical_baseline": True,
            "models": "off",
            # RAW FRAMES THE MODEL GENERATES PER SHOT, BEFORE RAFT x2.
            #
            # 16, and do NOT change this default without re-measuring MOTION rather than
            # cost. It was briefly 12 on this evidence: 510s against 695s for 16 (27%
            # faster), alignment and coherence equal or better, per-frame cost identical.
            # What that measurement did NOT capture is how the clip MOVES --
            # `bench_frames.py` scores alignment and coherence, not perceived motion.
            #
            # AnimateDiff's motion module was TRAINED on 16-frame windows. Every take in
            # `film2` was rendered at 12, and its owner's verdict was "the videos used to
            # actually move, now it's still". `flowcheck`, at 16, moves. There is an
            # unused 27% saving available and it is not worth a still film.
            "frames": 16,
        },
        # where the writer's material is kept, for provenance and re-rolls
        "provenance": {},
        "ideas": [],
        "kept_ideas": [],
        "proposal_path": None,
        "problems": [],
        # place/figure anchors the writing path bound into the prompts, if any
        "anchors": {},
        # the record of what has been exported
        "deliverable": None,
        "history": [],
    }


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def save(project: Dict) -> Path:
    project["saved"] = _now()
    d = dir_for(project["slug"])
    (d / "shots").mkdir(parents=True, exist_ok=True)
    p = project_path(project["slug"])
    # the shot cache is a live object graph, not project data: strip it so the file
    # stays readable and a hand-edit of project.json stays meaningful
    _write_json(p, {k: v for k, v in project.items() if not k.startswith("_")})
    return p


def load(slug: str) -> Dict:
    p = project_path(slug)
    if not p.exists():
        raise FileNotFoundError(f"no production at {p}")
    project = json.loads(p.read_text(encoding="utf-8"))
    base = new_project(project.get("name", slug), project.get("concept", ""))
    for k, v in base.items():
        project.setdefault(k, v)
    return project


def list_projects() -> List[Dict]:
    """Every production on disk, newest first. Cheap: reads one small file each."""
    if not PRODUCTIONS.exists():
        return []
    out = []
    for d in PRODUCTIONS.iterdir():
        p = d / "project.json"
        if not p.exists():
            continue
        try:
            pr = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({"slug": pr.get("slug", d.name), "name": pr.get("name", d.name),
                    "concept": pr.get("concept", ""),
                    "shots": len(pr.get("timeline") or []),
                    "saved": pr.get("saved") or pr.get("created"),
                    "deliverable": (pr.get("deliverable") or {}).get("video")})
    return sorted(out, key=lambda r: r.get("saved") or "", reverse=True)


def _write_json(p: Path, obj) -> None:
    """Atomic where Windows allows it, and never lost to a reader.

    `tmp.replace(p)` raises PermissionError (WinError 5) if ANY process has `p` open
    at that instant -- and the studio polls these files every 2 s while a job
    writes them. (`job_runner.write_state` fixed the same class of bug after it
    killed a supervisor.) So: a temp name unique to this writer (two writers used
    to share one `.tmp` and could interleave), a few retries because the reader
    lets go in milliseconds, then a direct write as the last resort -- a torn read
    for one poll is far cheaper than a lost edit or a dead job.
    """
    import os
    import threading
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    last: Optional[Exception] = None
    try:
        for attempt in range(5):
            try:
                tmp.write_text(payload, encoding="utf-8")
                tmp.replace(p)
                return
            except PermissionError as exc:
                last = exc
                time.sleep(0.05 * (attempt + 1))
        try:
            p.write_text(payload, encoding="utf-8")     # no rename needed
        except OSError as exc:
            raise (last or exc)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# shots -- one file each, addressed by id, never by position
# --------------------------------------------------------------------------- #

def new_shot(shot_id: str, prompt: str = "", **over) -> Dict:
    shot = {
        "id": shot_id,
        "prompt": prompt,
        # what the shot is FOR. The writer is asked for this now, and it is what
        # makes an edit reviewable: you can see whether the shot still does its job.
        "purpose": "",
        "scale": "medium",
        "camera": "none",          # the DECISION that renders
        "camera_suggestion": None,  # what the writer suggested; kept, not obeyed
        "seconds": 2.5,
        "scene": "sc1",
        "subject": "",
        "energy": 0.2,
        "beat": "develop",
        # how this shot connects to the previous one (open / match-action /
        # eyeline / movement / graphic / scene-cut). Stored so the edit can be
        # reviewed as a sequence rather than a list of images.
        "link": "",
        # FINISHING camera: {"move","amount","settle"} applied to the delivered
        # frames by camera_path. A generated camera LoRA cannot settle, so a hold
        # that needs a move records it here and generation stays still.
        "camera_move": None,
        # the writer's movement id, kept for the direction record
        "movement": "",
        "take": None,              # the chosen take directory name
        "created": _now(),
        # what the RENDER will pass to the generator. Stored so a change is
        # detectable before it costs 10 minutes.
        "render_inputs": None,
    }
    shot.update(over)
    return shot


def save_shot(slug: str, shot: Dict, project: Optional[Dict] = None) -> Path:
    p = shot_path(slug, shot["id"])
    shot["updated"] = _now()
    _write_json(p, shot)
    if project is not None:                     # keep the live copy authoritative
        _cache(project)[shot["id"]] = shot
    return p


def load_shot(slug: str, shot_or_id: Union[Dict, str]) -> Dict:
    p = shot_path(slug, _sid(shot_or_id))
    if not p.exists():
        raise FileNotFoundError(f"no shot {_sid(shot_or_id)} in {slug}")
    return json.loads(p.read_text(encoding="utf-8"))


def all_shots(slug: str) -> Dict[str, Dict]:
    d = dir_for(slug) / "shots"
    out: Dict[str, Dict] = {}
    if not d.exists():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(s, dict) and s.get("id"):
            out[s["id"]] = s
    return out


def _cache(project: Dict) -> Dict[str, Dict]:
    """The in-memory shot cache for a loaded project.

    WHY THIS EXISTS, and it is not an optimisation. Every mutator writes the shot
    to disk; without this, the copy already held in `project` (or in the caller's
    hands) would go stale, and `shot_state` -- which is DERIVED from the shot and
    the project together -- would report on a state that no longer exists. The
    store's own tests caught exactly that: change a prompt, then ask whether the
    take is stale, and get the answer for the old prompt. So every mutator updates
    the cache as well as the file, and `shots_in_order` prefers a cached shot over
    re-reading. One cache, filled once, corrected on every write.
    """
    c = project.get("_shots")
    if not isinstance(c, dict):
        c = all_shots(project["slug"])
        project["_shots"] = c
    return c


def get_shot(project: Dict, shot_or_id: Union[Dict, str]) -> Optional[Dict]:
    """The shot as the project currently has it. Fresh, cheap, single-sourced.

    Accepts a record or an id: a caller holding a record should be able to pass it
    to anything that wants "a shot", and `_sid` is where that is decided.
    """
    sid = _sid(shot_or_id)
    c = _cache(project)
    if sid not in c:
        try:
            c[sid] = load_shot(project["slug"], sid)
        except FileNotFoundError:
            return None
    return c[sid]


def invalidate(project: Dict) -> None:
    project.pop("_shots", None)


def shots_in_order(project: Dict) -> List[Dict]:
    """The timeline's shots, in timeline order, as shot records.

    An id in the timeline with no file is reported as a PLACEHOLDER rather than
    dropped: a shot that vanished from disk must be visible in the cut, not
    silently shorten the film.
    """
    out = []
    for sid in project.get("timeline") or []:
        s = get_shot(project, sid)
        if s is not None:
            out.append(s)
        else:
            out.append(new_shot(sid, prompt=f"(missing shot file: {sid})"))
    return out


def _ordered_ids(project: Dict) -> List[str]:
    return list(project.get("timeline") or [])


def add_shot(project: Dict, prompt: str = "", after: Optional[str] = None,
             **over) -> Dict:
    sid = next_id(_ordered_ids(project))
    shot = new_shot(sid, prompt, **over)
    tl = _ordered_ids(project)
    if after and after in tl:
        tl.insert(tl.index(after) + 1, sid)
    else:
        tl.append(sid)
    project["timeline"] = tl
    save_shot(project["slug"], shot, project)
    return shot


def add_shots(project: Dict, shots: Sequence[Dict]) -> List[Dict]:
    """Append several shots at once (a story pass). Returns them in order."""
    out = []
    for s in shots:
        out.append(add_shot(project, **s))
    return out


def update_shot(project: Dict, shot_or_id: Union[Dict, str], **fields) -> Dict:
    """Edit fields on a shot. Updates disk AND the project's live copy.

    Accepts a shot record or an id: a caller holding the record it just rendered is
    the natural caller of this, and making it unwrap the id itself is how the
    record/id confusion got in.

    The live update is the load-bearing part: `shot_state` is derived from the shot
    plus the project, so a write that left the in-memory copy behind would make the
    very next staleness question answer about the previous values.
    """
    shot = get_shot(project, shot_or_id)
    if shot is None:
        shot = load_shot(project["slug"], _sid(shot_or_id))
    shot.update(fields)
    save_shot(project["slug"], shot, project)
    return shot


def move_shot(project: Dict, shot_or_id: Union[Dict, str], delta: int) -> List[str]:
    """Move one shot in the timeline. Nothing else changes -- that is the fix.

    Under the old model this is where camera overrides, palettes, pins and
    approvals all silently moved to a neighbouring shot.
    """
    shot_id = _sid(shot_or_id)
    tl = _ordered_ids(project)
    if shot_id not in tl:
        return tl
    i = tl.index(shot_id)
    j = max(0, min(len(tl) - 1, i + delta))
    if i != j:
        tl.insert(j, tl.pop(i))
    project["timeline"] = tl
    return tl


def move_to(project: Dict, shot_id: str, index: int) -> List[str]:
    tl = _ordered_ids(project)
    if shot_id not in tl:
        return tl
    tl.insert(max(0, min(len(tl) - 1, index)), tl.pop(tl.index(shot_id)))
    project["timeline"] = tl
    return tl


def remove_shot(project: Dict, shot_or_id: Union[Dict, str],
                delete_files: bool = False) -> List[str]:
    sid = _sid(shot_or_id)
    tl = [x for x in _ordered_ids(project) if x != sid]
    project["timeline"] = tl
    _cache(project).pop(sid, None)
    if delete_files:
        shutil.rmtree(takes_dir(project["slug"], sid), ignore_errors=True)
        shot_path(project["slug"], sid).unlink(missing_ok=True)
    return tl


def archive_takes(project: Dict, shot_ids: Iterable[str]) -> Optional[Path]:
    """Move the takes (and shot records) of shots that are being replaced out of
    `takes/<id>/`, WITHOUT deleting them: `takes/_archive/<stamp>/<id>/`.

    WHY: take folders are keyed by shot id, and a re-draft reuses `s01..sN`. Left in
    place, the new shot 1 inherited the OLD film's takes -- `best_take` showed the
    previous film's frames, `shot_state` compared the old take's hash with the new
    shot, and the next take was numbered t02. Moving keeps the 10-minute renders
    (nothing is thrown away) but gives every new shot an empty take folder.

    Returns the archive directory, or None when there was nothing to move. Raises
    OSError if a move fails, so the caller can refuse to adopt rather than alias.
    """
    slug = project["slug"]
    moves = [(sid, takes_dir(slug, sid), shot_path(slug, sid)) for sid in shot_ids]
    moves = [m for m in moves if m[1].exists() or m[2].exists()]
    if not moves:
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = dir_for(slug) / "takes" / "_archive" / stamp
    n = 1
    while root.exists():                      # two drafts inside one second
        n += 1
        root = dir_for(slug) / "takes" / "_archive" / f"{stamp}-{n}"
    for sid, tdir, spath in moves:
        dest = root / sid
        dest.mkdir(parents=True, exist_ok=True)
        if tdir.exists():
            shutil.move(str(tdir), str(dest / "takes"))
        if spath.exists():
            shutil.copy2(str(spath), str(dest / "shot.json"))   # new shot overwrites it
    return root


def duplicate_shot(project: Dict, shot_or_id: Union[Dict, str]) -> Dict:
    """A second angle on the same moment -- the cheapest way to get variety."""
    sid_src = _sid(shot_or_id)
    src = dict(get_shot(project, sid_src) or load_shot(project["slug"], sid_src))
    sid = next_id(_ordered_ids(project))
    copy = dict(src)
    copy.update(id=sid, take=None, render_inputs=None, seed=None, created=_now(),
                prompt=src.get("prompt", "") + " (alternate)")
    tl = _ordered_ids(project)
    tl.insert(tl.index(sid_src) + 1, sid)
    project["timeline"] = tl
    save_shot(project["slug"], copy, project)
    return copy


# --------------------------------------------------------------------------- #
# scenes
# --------------------------------------------------------------------------- #

def scene_ids_in_timeline(project: Dict) -> List[str]:
    out = []
    for s in shots_in_order(project):
        sc = s.get("scene") or "sc1"
        if sc not in out:
            out.append(sc)
    return out


def palette_for(project: Dict, shot: Dict) -> str:
    """One palette per SCENE, looked up by the scene's id -- not by position.

    `grade.looks` used to be a flat list indexed by the shot's position in the
    frozen plan, rebuilt whenever the code passed over the list. Reorder the
    timeline and a shot's palette came from another scene.
    """
    sc = shot.get("scene") or "sc1"
    for s in project.get("scenes") or []:
        if s.get("id") == sc:
            return s.get("palette") or "neutral"
    return "neutral"


def set_palette(project: Dict, scene_id: str, palette: str) -> None:
    for s in project.get("scenes") or []:
        if s.get("id") == scene_id:
            s["palette"] = palette
            break
    else:
        project.setdefault("scenes", []).append(
            {"id": scene_id, "name": scene_id, "palette": palette})
    drop_stale_cache(project)


def drop_stale_cache(project: Dict) -> None:
    """Drop the cached shot copies after a change that is NOT a shot edit.

    Only two things do this: a scene's palette and the film-level settings. Both are
    read by `finish_inputs`, so a cached shot copy would answer the next staleness
    question with the previous palette. Shot edits do not need this -- `save_shot`
    corrects the cache directly.
    """
    project.pop("_shots", None)


def sync_scenes(project: Dict) -> List[Dict]:
    used = scene_ids_in_timeline(project)
    have = {s.get("id"): s for s in project.get("scenes") or []}
    named = {}
    for i, sid in enumerate(used, 1):
        named[sid] = have.get(sid) or {"id": sid, "name": f"scene {i}",
                                       "palette": "neutral"}
        named[sid].setdefault("name", f"scene {i}")
        named[sid].setdefault("palette", "neutral")
    project["scenes"] = [named[s] for s in used]
    drop_stale_cache(project)
    return project["scenes"]


def set_settings(project: Dict, **fields) -> Dict:
    """Film-level settings (fps, shake scale, baseline, model). All are read by
    `finish_inputs` or `generation_inputs`, so the cache has to go."""
    project.setdefault("settings", {}).update(fields)
    drop_stale_cache(project)
    return project["settings"]


# --------------------------------------------------------------------------- #
# render inputs, hashes, and DERIVED staleness
# --------------------------------------------------------------------------- #

def generation_inputs(project: Dict, shot: Dict, seed: Optional[int] = None,
                      frames: Optional[int] = None) -> Dict:
    """The inputs that CHANGE THE PIXELS THE MODEL PRODUCES.

    Only a change here justifies spending 540-788 s of generation. Deliberately
    explicit: if a knob is not in here, changing it will not mark the take
    generation-stale, so it belongs in `finish_inputs` instead -- never nowhere.

    `frames` may be overridden by the caller. The RENDERER does that, because the
    value it is about to generate with has to be the value the hash is taken against,
    and reaching that by mutating the project's settings is what let a default reach
    backwards into finished work.

    The seed is DERIVED ONCE and then frozen onto the shot. Deriving it from the
    timeline position on every call would mean a reorder silently re-rolls every
    shot -- the store's own test caught exactly that, which is why `ensure_seed`
    exists.
    """
    st = project.get("settings") or {}
    if seed is None:
        seed = ensure_seed(project, shot)
    if frames is None:
        frames = st.get("frames")
    # HOLD CONDITIONING IS A GENERATION INPUT. A shot rendered from a reference
    # frame is different pixels from the same prompt rendered alone, so turning
    # the organisation's holds on/off must make the take stale. The value names
    # the ANCHOR shot and its chosen take: if the anchor is re-shot, the shots
    # conditioned on it are correctly stale too.
    hold_ref = ""
    try:
        import visual_org as _vo
        shots = shots_in_order(project)
        idx = next((i for i, s in enumerate(shots)
                    if s.get("id") == shot.get("id")), None)
        if idx is not None:
            plan = _vo.reference_plan(_vo.as_shots(shots))
            anchor = plan.get(idx)
            if anchor is not None:
                a = shots[anchor]
                hold_ref = f"{a.get('id')}:{a.get('take') or ''}"
    except Exception:                                # noqa: BLE001
        hold_ref = ""
    return {
        "prompt": shot.get("prompt") or "",
        "scale": shot.get("scale") or "medium",
        "camera": shot.get("camera") or "none",
        "subject": shot.get("subject") or "",
        "seed": int(seed),
        "model": st.get("models") or "off",
        "hold_reference": hold_ref,
        # THE RAW FRAME COUNT IS A RENDER INPUT, because it changes the pixels and the
        # cost. Measured on this machine: 12 raw frames = 510s total, 16 = 695s -- 27%
        # faster -- with alignment and coherence equal or better (align 0.348 vs 0.346,
        # coherence 0.0155 vs 0.0240) and per-frame diffusion cost identical (41.8s vs
        # 42.8s), so the saving is purely from doing fewer frames.
        #
        # `16` when a production has never recorded one: every take on disk from before
        # this key existed was rendered at 16, so that is what they ARE.
        "frames": int(16 if frames is None else frames),
        # THE SAMPLER TIER IS A RENDER INPUT. "fast" (AnimateDiff-Lightning, 4 steps,
        # guidance 1.0) and "full" (the undistilled sampler) draw different pixels
        # from the same prompt and seed -- measured on skydiving s05: fast gave a
        # four-armed blur, full gave a correct figure. Every take on disk was
        # rendered "fast", which is why a missing key counts as "fast".
        "sampler": render_tier_tag(shot),
        # the approved first frame the clip is animated FROM (None = text-to-video).
        # Its pixels and the strength decide the clip, so both are inputs.
        "keyframe": keyframe_digest(project, shot),
        # words the FULL sampler is told to avoid (guidance > 1 only), from the
        # project's character -- it changes the pixels, so it is an input.
        "avoid": character_avoid(project),
    }


# ---- sampler tiers ---------------------------------------------------------- #
# FAST costs ~10 min/shot on this CPU, FULL ~3-5x that. `FULL_STEPS` is part of the
# recorded tag, so changing it re-renders only shots that were rendered FULL.
FULL_STEPS = 12

# a figure framed at these scales is small in the image; anything closer is large
_SMALL_FIGURE_SCALES = ("wide", "extreme-wide")


def render_tier(shot: Dict) -> str:
    """"fast" or "full" for a shot: the shot's own `quality` wins, else its SCALE.

    wide / extreme-wide -> fast (the figure is small, the 4-step sampler holds up);
    medium / close / macro -> full (the figure fills the frame and the 4-step
    sampler's extra limbs show). Chosen by the user over a text heuristic: the
    prompts all name the same outfit, so text cannot separate easy from hard. A
    "wide" label can still draw a big figure (skydiving s06 did) -- that shot gets
    `quality: "full"`.
    """
    if shot.get("keyframe"):
        return "full"            # a keyframe is only ever animated by the full tier
    q = str(shot.get("quality") or "auto").lower()
    if q in ("fast", "full"):
        return q
    return "fast" if (shot.get("scale") or "medium") in _SMALL_FIGURE_SCALES else "full"


# ---- the film's ONE character ---------------------------------------------- #
# MEASURED NEED (skydiving): every prompt said "a skydiver", so every shot drew a
# NEW person -- the figure started as a man and ended as a woman, and the suit went
# orange / green / yellow-green. `anchor_plan` pulls the figure from each SCENE's own
# wording, so scenes disagree. A film-level descriptor fixes who it is.

def character_avoid(project: Dict) -> str:
    return str(((project or {}).get("character") or {}).get("avoid") or "").strip()


def _subject_noun(shot: Dict) -> str:
    """The head noun of the shot's subject ("a skydiver" -> "skydiver")."""
    ws = [w for w in re.findall(r"[a-z][a-z\-]+", (shot.get("subject") or "").lower())
          if w not in {"a", "an", "the"}]
    return ws[-1] if ws else ""


def character_prompt(prompt: str, noun: str, descriptor: str) -> str:
    """Put `descriptor` in front of the FIRST mention of `noun` ("A skydiver in ..."
    -> "A male skydiver in ..."). Idempotent, and a prompt that never names the
    subject is left alone (a shot with no figure must not grow one)."""
    if not (prompt and noun and descriptor):
        return prompt
    m = re.search(r"\b" + re.escape(noun) + r"\b", prompt, flags=re.I)
    if not m:
        return prompt
    before = prompt[:m.start()].lower().rstrip()
    if before.endswith(descriptor.lower()):
        return prompt
    return prompt[:m.start()] + descriptor + " " + prompt[m.start():]


def apply_character(project: Dict) -> int:
    """Rewrite every shot's prompt to the project's character. Returns how many
    shots changed. Re-running it changes nothing."""
    ch = project.get("character") or {}
    desc = str(ch.get("descriptor") or "").strip()
    n = 0
    for sid in _ordered_ids(project):
        shot = get_shot(project, sid)
        new = character_prompt(shot.get("prompt") or "", _subject_noun(shot), desc)
        if new != shot.get("prompt"):
            update_shot(project, sid, prompt=new)
            n += 1
    drop_stale_cache(project)
    return n


def set_character(project: Dict, descriptor: str, avoid: str = "") -> int:
    """The film's one character: `descriptor` goes in front of the subject noun in
    every shot ("male"), `avoid` goes in the full sampler's negative prompt."""
    project["character"] = {"descriptor": descriptor.strip(), "avoid": avoid.strip()}
    n = apply_character(project)
    save(project)
    return n


# How far the sampler may move away from the approved keyframe: 0 = frozen on it,
# 1 = ignore it. UNTUNED starting value -- `keyframe.py` / `clip_from_still` are
# tested on one shot before this is trusted (HANDOFF part 5).
KEYFRAME_STRENGTH = 0.7


def keyframe_digest(project: Dict, shot: Dict) -> Optional[str]:
    """Content hash of the shot's approved keyframe (+ strength), or None."""
    rel = shot.get("keyframe")
    if not rel:
        return None
    p = dir_for(project["slug"]) / rel
    try:
        data = p.read_bytes()            # ~400 KB: hashing beats a stale cache
    except OSError:
        return "missing"                 # approved but gone: must not look current
    import hashlib
    return f"{hashlib.sha1(data).hexdigest()[:12]}@{KEYFRAME_STRENGTH}"


def render_tier_tag(shot: Dict) -> str:
    """The value recorded in the generation inputs: "fast" or "full<steps>"."""
    return "fast" if render_tier(shot) == "fast" else f"full{FULL_STEPS}"


def generation_hash(project: Dict, shot: Dict, frames: Optional[int] = None,
                    stored_generation: Optional[Dict] = None,
                    stored_hash: Optional[str] = None) -> str:
    """The generation hash, tolerant of a take recorded BEFORE A KEY EXISTED.

    Adding a key to `generation_inputs` is not free, and this cost a real scare: the
    moment `frames` was added, every take ever recorded -- all of which stored a
    generation dict WITHOUT that key -- hashed differently. Six finished shots read as
    `stale` and the UI offered to re-render the whole film. Nothing about the pixels
    had changed; only the key set had.

    So a stored hash counts as current if it matches EITHER
      * the current inputs, or
      * the current inputs MINUS the keys that particular take predates.

    That is deliberately narrow: if any key the take DID record has since changed, the
    subset cannot match and the take is correctly stale -- proven in
    `test_film_store.py` §10, which edits the prompt of a legacy record and asserts it
    still goes stale.
    """
    cur = generation_inputs(project, shot, frames=frames)
    h = input_hash(cur)
    if stored_hash is None or h == stored_hash:
        return h
    if not isinstance(stored_generation, dict) or not stored_generation:
        return h                                # nothing to compare against
    # A key the take predates is tolerated ONLY while the shot's current value is
    # the neutral one the old renderer implicitly used. `sampler` is the case that
    # matters: a take recorded before it existed was rendered "fast", so it stays
    # current for a fast shot -- but a shot that is NOW full must go stale, or the
    # tier change would never re-render anything.
    # WHICH TAKE OF THE ANCHOR A SHOT WAS CONDITIONED ON IS PROVENANCE, NOT A RENDER
    # INPUT THE USER CAN BE ASKED TO REPAY. Choosing a different take for the anchor
    # shot used to mark every shot held on it as needing a re-shoot (squid: 4 of 10
    # shots, ~40 min each) even though nothing about THEIR shot had changed and the
    # user had picked their takes. The value stays recorded on the take (and shown as
    # a note, `hold_note`); it just no longer decides staleness.
    cmp_ = dict(cur)
    if "hold_reference" in stored_generation:
        cmp_["hold_reference"] = stored_generation["hold_reference"]
    if input_hash(cmp_) == stored_hash:
        return stored_hash
    predates = {k for k in ADDED_GEN_KEYS if k not in stored_generation
                and cur.get(k) == GEN_NEUTRAL.get(k, cur.get(k))}
    if not predates:
        return h                                # it recorded every key, so it differs
    legacy = {k: v for k, v in cmp_.items() if k not in predates}
    return stored_hash if input_hash(legacy) == stored_hash else h


# Keys ADDED to `generation_inputs` after the first takes were recorded. Each entry
# means "a take from before this key existed is still comparable". Append to this list
# whenever a key is added, or every existing film will offer to re-render itself.
ADDED_GEN_KEYS = ("frames", "hold_reference", "sampler", "keyframe", "avoid")
# The value an OLD renderer implicitly used, for keys where "missing" is only the
# same thing as the current value when it equals this (see generation_hash).
GEN_NEUTRAL = {"sampler": "fast", "keyframe": None, "avoid": ""}


def finish_inputs(project: Dict, shot: Dict) -> Dict:
    """The inputs applied AFTER generation: grading, the exposure sheet, the shake.

    These are the cheap ones. Separating them from `generation_inputs` is the whole
    point of keeping the per-shot ladder on disk: retiming a shot, re-grading a
    scene or turning the shake down costs the seconds it takes to re-run
    RAFT + shake + grade, NOT another ten minutes of AnimateDiff. The old model
    treated one palette list as a render input and would have re-generated every
    shot in the scene to change a colour.

    NOTE: `in_s`/`out_s` are deliberately NOT in here. They are DERIVED from the
    shot's neighbours, so putting them in a per-shot hash would make one retime
    stale every later shot. The counter on the shot (`edit_seq`) carries the
    cascade instead, and only where the output really changes.
    """
    st = project.get("settings") or {}
    cam = shot.get("camera_move")
    if isinstance(cam, dict) and cam.get("move"):
        cam = {"move": str(cam["move"]), "amount": int(cam.get("amount") or 38),
               "settle": float(cam.get("settle") or 0.6)}
    else:
        cam = None
    return {
        "seconds": float(shot.get("seconds") or 2.5),
        "fps": int(st.get("fps") or 16),
        "shake_scale": float(st.get("shake_scale") or 1.0),
        "palette": palette_for(project, shot),
        "technical_baseline": bool(st.get("technical_baseline", True)),
        # FINISHING RAFT factor. Raising it interpolates a longer, slower clip
        # from the SAME take -- the pacing study's "procedural hold" -- so the
        # 4-8 s long band is a finishing decision, not another render.
        "upsample": int(shot.get("upsample") or 2),
        # finishing move: changing it re-finishes only, never re-generates
        "camera_move": cam,
    }


# Fields on a shot that do NOT affect the finished frames. `take` is the chosen
# take's NAME: choosing one cannot change the pixels, so it must not make the take
# you just chose look stale. `approved` is the same idea one level up.
NON_PIXEL_FIELDS = {
    "take", "approved", "approved_at", "created", "updated", "render_inputs",
    "seed", "purpose", "camera_suggestion", "beat", "scene", "subject", "energy",
}


def pixel_deps(shot: Dict) -> Dict:
    """The per-shot part of `rebuild_hash` -- only the fields that reach the pixels.

    Takes a resolved shot record, not an id: `rebuild_hash` resolves once and passes
    the record down, so this cannot disagree with what it hashed.

    `seconds` is in here because it sets the hold; a renderer that reads a
    `finishing` dict must read `seconds` from it, never from a raw field, or the
    hold it applies will not be the hold the staleness check measured.
    """
    return {"seconds": float(shot.get("seconds") or 2.5)}


def render_inputs(project: Dict, shot: Dict, seed: Optional[int] = None) -> Dict:
    """Both halves, for the take record and for a single hash. Prefer the halves."""
    out = generation_inputs(project, shot, seed)
    out.update(finish_inputs(project, shot))
    return out


# NOTE: `generation_hash` is defined ABOVE, next to `generation_inputs`, because it
# carries the added-key tolerance and must not be shadowed by a second definition.
# There was one here; it silently overrode the tolerant version and made the
# legacy-take comparison a no-op.


def in_points(project: Dict) -> Dict[str, float]:
    """Where every shot starts in the cut, keyed by id -- walked ONCE.

    The exposure sheet, computed from the timeline. Every consumer takes its in-point
    from here rather than re-walking, because two walks that disagree is how this
    project got its normaliser wrong five times.
    """
    out: Dict[str, float] = {}
    t = 0.0
    for s in shots_in_order(project):
        out[s["id"]] = round(t, 4)
        t += float(s.get("seconds") or 2.5)
    return out


def rebuild_hash(project: Dict, shot_or_id: Union[Dict, str],
                 in_s: Optional[float] = None,
                 stored_finish: Optional[Dict] = None,
                 stored_hash: Optional[str] = None) -> str:
    """What the FINISHING pass depends on, for this shot.

    Three parts, and they are the reason this is a hash rather than a counter:

      * the shot's own finishing inputs -- its hold, shake, scene, palette;
      * the palette of EVERY scene, because changing a scene's palette changes the
        grade of every shot in it, and reverting the palette must clear it again;
      * this shot's IN-POINT, passed in by the caller, so a retime upstream of it
        marks it for a re-finish (its frames move) while a retime downstream does not.

    THE IN-POINT IS AN ARGUMENT, NOT A LOOKUP, and that is a bug fix. It used to walk
    the timeline itself: to find the in-point it summed the durations of the shots
    before it, so it read each shot's stored `seconds`. Editing a shot's length then
    made its own dependency hash differ from the one recorded on the take, and the
    only way to clear that was to edit the value back AND re-derive. Worse, the walk
    made the dependency asymmetric -- a retime marked the shots after it, but the
    walking itself changed what "after" meant. Passing the in-point keeps the check
    and the exposure sheet on one definition: `in_points(project)`.

    A counter can express none of this -- it only goes up, so setting a value back
    would report stale forever. Hashing the actual dependency self-cleans, provided
    the dependency is computed the same way every time.
    """
    shot = get_shot(project, shot_or_id)
    if shot is None:
        raise KeyError(f"no shot {_sid(shot_or_id)} in {project.get('slug')!r}")
    if in_s is None:
        in_s = in_points(project).get(_sid(shot_or_id), 0.0)
    scenes = {s.get("id"): s.get("palette") for s in project.get("scenes") or []}
    fin = finish_inputs(project, shot)

    def hashed(finish: Dict) -> str:
        return input_hash({"finish": finish, "pixel": pixel_deps(shot),
                           "in_s": round(float(in_s), 4), "scenes": scenes})

    h = hashed(fin)
    if stored_hash is None or h == stored_hash:
        return h
    if not isinstance(stored_finish, dict) or not stored_finish:
        return h
    # A take recorded BEFORE a finishing key existed is still comparable -- but only
    # while that key sits at the value the old renderer implicitly used. Measured:
    # 120 takes on disk (ego, film2, ego-death, _seq_* ...) lack `upsample`, and 93
    # lack `camera_move`, so every one read `restale` ("needs re-grading") with no
    # input changed. If the shot's CURRENT value is not that neutral default, the
    # finish really did change and the take correctly stays restale.
    legacy = dict(fin)
    for key, neutral in ADDED_FINISH_KEYS.items():
        if key not in stored_finish and legacy.get(key) == neutral:
            legacy.pop(key, None)
    return stored_hash if hashed(legacy) == stored_hash else h


# Finishing keys ADDED to `finish_inputs` after the first takes were recorded, with
# the value the pre-existing renderer behaved as. Append here when adding a key.
ADDED_FINISH_KEYS = {"upsample": 2, "camera_move": None}


def ensure_seed(project: Dict, shot: Dict) -> int:
    """The shot's seed, minted on first use and then a permanent property of the shot.

    Minted from the shot's own id and the film's seed base, so it is stable across
    reorders and reopens, and a duplicate shot gets a different one.
    """
    if shot.get("seed") is not None:
        return int(shot["seed"])
    base = int((project.get("settings") or {}).get("seed_base") or 1103)
    sid = str(shot.get("id") or "s00")
    m = re.fullmatch(r"(\D*)(\d+)", sid)
    # crc32, not hash(): str hashing is randomised per process, so an id without
    # digits would get a different seed in the studio and in the render job
    import zlib
    n = int(m.group(2)) if m else zlib.crc32(sid.encode("utf-8")) % 1000
    seed = base + 101 * n
    shot["seed"] = seed
    try:
        save_shot(project["slug"], shot, project)
    except Exception:
        pass
    return seed


def input_hash(inputs: Dict) -> str:
    return hashlib.sha1(json.dumps(inputs, sort_keys=True).encode("utf-8")) \
        .hexdigest()[:12]


def _sid(shot_or_id: Union[Dict, str]) -> str:
    """A shot id, from either a shot record or the id itself.

    This exists because the two were confused: `run_store` passed a shot RECORD to
    `next_take_name`, which wanted an id, so the path became `.../takes/{'id': 's01'}`
    and blew up with `unsupported operand type(s) for /: 'WindowsPath' and 'dict'` --
    AFTER the diffusion model had loaded, which is 3-5 minutes of work thrown away.
    The unit tests all passed strings and missed it.

    The lesson is not "be careful": it is that a function whose argument can be either
    thing must accept either thing, because one of the callers will get it wrong and
    the failure will land in the most expensive place available.
    """
    if isinstance(shot_or_id, dict):
        sid = shot_or_id.get("id")
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"shot record has no usable id: {shot_or_id!r}")
        return sid
    if not isinstance(shot_or_id, str) or not shot_or_id:
        raise ValueError(f"expected a shot id or record, got {shot_or_id!r}")
    return shot_or_id


def take_dir(project: Dict, shot_or_id: Union[Dict, str], take_name: str) -> Path:
    return takes_dir(project["slug"], _sid(shot_or_id)) / take_name


def list_takes(project: Dict, shot_or_id: Union[Dict, str]) -> List[Dict]:
    """Every take on disk for a shot, oldest first, with its metadata.

    Accepts a shot record or a shot id -- see `_sid`.
    """
    d = takes_dir(project["slug"], _sid(shot_or_id))
    out = []
    if not d.exists():
        return out
    for t in sorted(d.iterdir()):
        if not t.is_dir():
            continue
        meta = t / "take.json"
        if not meta.exists():
            continue
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        m["_dir"] = str(t)
        m["frames_on_disk"] = len(list((t / "frames").glob("f*.png")))
        out.append(m)
    return out


def take_record(project: Dict, shot: Dict, seconds: float, frames: int,
                ladder: Optional[Dict] = None, **extra) -> Dict:
    """Build the `take.json` a renderer should write. One definition, so the UI and
    the renderer cannot disagree about what makes a take current."""
    return {
        "take": extra.pop("take", None),
        "shot": shot["id"],
        "at": _now(),
        "generation": generation_inputs(project, shot),
        "input_hash": generation_hash(project, shot),
        "finish": finish_inputs(project, shot),
        "rebuild_hash": rebuild_hash(project, shot),
        "seconds": round(float(seconds), 1),
        "frames": int(frames),
        "ladder": ladder or {},
        "approved": False,
        **extra,
    }


def next_take_name(project: Dict, shot_or_id: Union[Dict, str]) -> str:
    """The next free take name for a shot. Accepts a record or an id."""
    return next_id([t.get("take", "") for t in list_takes(project, shot_or_id)],
                   prefix="t")


def shot_state(project: Dict, shot_or_id: Union[Dict, str],
               takes: Optional[List[Dict]] = None) -> str:
    """DERIVED. One of:

        unrendered  nothing on disk
        restale     the take exists and the MODEL inputs are unchanged, but the
                    finishing inputs moved (grade / hold / shake) -- seconds to fix,
                    no generation
        stale       the model inputs moved -- needs another 540-788 s
        rendered    current, not yet locked
        approved    current, and marked as reviewed ("locked" in the UI)

    Never stored, so it cannot contradict the disk. The old model's `status` dict
    said `grade: empty, render: empty` while an assembled film sat on disk.

    Accepts a shot record OR an id. Both forms resolved to the same record above, so
    the two give identical answers -- which is what test 8 asserts.

    DRIFT IS REPORTED THROUGH APPROVAL, NOT MASKED BY IT. `approved` used to return
    before the `rebuild_hash` check, so locking a shot made the UI stop mentioning
    that its grade, hold or shake had since moved. Locking a shot you were happy with
    and then quietly changing the scene's palette produced no warning anywhere --
    the exact class of silent drift this store exists to remove. A lock is a note
    that you reviewed something, never a claim that it is still current.
    """
    shot = get_shot(project, shot_or_id)
    if shot is None:
        return "unrendered"                   # no such shot: nothing on disk
    if takes is None:
        takes = list_takes(project, shot)
    chosen = shot.get("take")
    if not takes:
        return "unrendered"
    if not chosen:
        # SOMETHING EXISTS, NOTHING IS PICKED. Return `rendered` only if the newest take
        # is still current -- otherwise a shot whose every take has gone stale (because
        # the frame count changed under it, say) reports `rendered` and the UI says
        # "shot" for frames it will not cut. Checked against the newest take, which is
        # what `best_take` would use.
        newest = takes[-1]
        stale = newest.get("input_hash") != generation_hash(
            project, shot,
            stored_generation=newest.get("generation"),
            stored_hash=newest.get("input_hash"))
        return "stale" if stale else "rendered"
    cur = next((t for t in takes if t.get("take") == chosen), None)
    if cur is None:
        return "stale"                        # the chosen take is gone
    # the take's own record is what makes the comparison tolerant of keys added after
    # it was written -- see `generation_hash`
    if cur.get("input_hash") != generation_hash(
            project, shot,
            stored_generation=cur.get("generation"),
            stored_hash=cur.get("input_hash")):
        return "stale"
    if cur.get("rebuild_hash") != rebuild_hash(
            project, shot, stored_finish=cur.get("finish"),
            stored_hash=cur.get("rebuild_hash")):
        return "restale"
    return "approved" if cur.get("approved") else "rendered"


_DRIFT_LABEL = {"prompt": "prompt", "scale": "framing", "camera": "movement",
                "subject": "subject", "seed": "seed", "model": "model",
                "frames": "frame count", "sampler": "render quality",
                "keyframe": "keyframe", "avoid": "avoid list"}


def take_drift(project: Dict, shot: Dict, take: Optional[Dict]) -> List[str]:
    """What the shot says NOW that its chosen take was not made from -- in words.

    The keys that actually make the take stale: `hold_reference` is excluded (it is
    a note) and so are keys the take predates while the shot is at the neutral value.
    Empty when the take is current.
    """
    gen = (take or {}).get("generation")
    if not isinstance(gen, dict) or not gen:
        return []
    cur = generation_inputs(project, shot)
    out = []
    for k, v in cur.items():
        if k == "hold_reference":
            continue
        if k not in gen:
            if k in ADDED_GEN_KEYS and v == GEN_NEUTRAL.get(k, v):
                continue
            out.append(_DRIFT_LABEL.get(k, k))
        elif gen[k] != v:
            out.append(_DRIFT_LABEL.get(k, k))
    return out


def hold_note(project: Dict, shot: Dict, take: Optional[Dict]) -> Optional[str]:
    """A one-line note when the shot was held on a different take of its anchor than
    the anchor now uses. Information only: it never makes a shot stale."""
    old = ((take or {}).get("generation") or {}).get("hold_reference")
    new = generation_inputs(project, shot).get("hold_reference")
    if old and new and old != new:
        return (f"made while conditioned on {old}; its anchor now uses {new}. "
                f"Nothing needs re-shooting")
    return None


def adopt_take(project: Dict, shot_or_id: Union[Dict, str],
               take_name: Optional[str] = None) -> Dict:
    """Treat the take the user CHOSE as current for the shot's present text.

    The pixels are untouched. The take's recorded inputs are replaced with the
    shot's current ones so it stops reading as stale, and what it was really made
    from is KEPT under `adopted` (with the time), so the record never claims the
    take was rendered from text it was not. Nothing re-shoots.
    """
    shot = get_shot(project, shot_or_id) or load_shot(project["slug"],
                                                      _sid(shot_or_id))
    name = take_name or shot.get("take")
    if not name:
        raise ValueError(f"shot {_sid(shot_or_id)} has no take chosen to adopt")
    p = take_dir(project, shot, name) / "take.json"
    if not p.exists():
        raise FileNotFoundError(f"no take {name} on shot {_sid(shot_or_id)}")
    m = json.loads(p.read_text(encoding="utf-8"))
    was = {"at": _now(), "generation": m.get("generation"),
           "input_hash": m.get("input_hash")}
    m.setdefault("adopted", []).append(was)
    m["generation"] = generation_inputs(project, shot)
    m["input_hash"] = input_hash(m["generation"])
    _write_json(p, m)
    log_history(project, act="adopt take", shot=_sid(shot_or_id), take=name)
    return m


def needs_generation(state: str) -> bool:
    """Does fixing this shot cost a diffusion run, or only a re-finish?"""
    return state in ("unrendered", "stale")


def current_take(project: Dict, shot_or_id: Union[Dict, str],
                 takes: Optional[List[Dict]] = None) -> Optional[Dict]:
    """The take the film currently cuts to, or None. Record or id, both fine.

    `takes` is an optional PRE-FETCHED `list_takes` result for this shot: a caller
    that already scanned the folder passes it instead of paying for another scan."""
    shot = get_shot(project, shot_or_id)
    if shot is None:
        return None
    chosen = shot.get("take")
    for t in (takes if takes is not None else list_takes(project, shot)):
        if t.get("take") == chosen:
            return t
    return None


def best_take(project: Dict, shot_or_id: Union[Dict, str],
              takes: Optional[List[Dict]] = None) -> Optional[Dict]:
    """What to show for a shot: the chosen take, else the newest one."""
    shot = get_shot(project, shot_or_id)
    if shot is None:
        return None
    if takes is None:
        takes = list_takes(project, shot)
    if not takes:
        return None
    return current_take(project, shot, takes=takes) or takes[-1]


def choose_take(project: Dict, shot_or_id: Union[Dict, str], take_name: str) -> Dict:
    return update_shot(project, shot_or_id, take=take_name)


def approve_take(project: Dict, shot_or_id: Union[Dict, str],
                 take_name: Optional[str] = None) -> Dict:
    shot = get_shot(project, shot_or_id) or load_shot(project["slug"],
                                                      _sid(shot_or_id))
    name = take_name or shot.get("take")
    if not name:
        raise ValueError(f"shot {_sid(shot_or_id)} has no take chosen to approve")
    p = take_dir(project, shot, name) / "take.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["approved"] = True
    m["approved_at"] = _now()
    _write_json(p, m)
    return update_shot(project, shot_or_id, take=name)


def unlock_take(project: Dict, shot_or_id: Union[Dict, str],
                take_name: Optional[str] = None) -> Dict:
    """Clear the reviewed mark on a take. The inverse of `approve_take`.

    It does NOT change what the film cuts to, and it does NOT delete anything: the
    frames and the record stay exactly as they are. It removes a note you made about
    a take.
    """
    shot = get_shot(project, shot_or_id) or load_shot(project["slug"],
                                                      _sid(shot_or_id))
    name = take_name or shot.get("take")
    if not name:
        raise ValueError(f"shot {_sid(shot_or_id)} has no take to unlock")
    p = take_dir(project, shot, name) / "take.json"
    if not p.exists():
        raise FileNotFoundError(f"no take {name} on shot {_sid(shot_or_id)}")
    m = json.loads(p.read_text(encoding="utf-8"))
    m["approved"] = False
    m.pop("approved_at", None)
    _write_json(p, m)
    return shot


# --------------------------------------------------------------------------- #
# the film as a whole: derived, always
# --------------------------------------------------------------------------- #

def hold_frames(shot: Dict, fps: Optional[int] = None) -> int:
    """The exposure sheet's hold. Delegates to `handheld.hold_frames`.

    Imported lazily and called with the same function the renderer uses, so a
    timecode in the UI is not a second opinion. `fps` is threaded through rather
    than defaulted here, because a duration computed at one fps and rendered at
    another is exactly the class of unit error this project keeps finding.
    """
    from handheld import hold_frames as _hf
    return _hf(Shot(prompt=shot.get("prompt", ""), scale=shot.get("scale", "medium"),
                    camera=shot.get("camera", "none"),
                    seconds=float(shot.get("seconds") or 2.5)),
               int(fps or 16))


def duration(project: Dict) -> float:
    fps = int((project.get("settings") or {}).get("fps") or 16)
    return sum(hold_frames(s, fps) for s in shots_in_order(project)) / fps


# What one shot costs on this CPU, per tier, when no take of that tier has been
# timed yet (measured: skydiving fast 677 s, full 2570-3410 s).
TIER_DEFAULT_S = {"fast": 660.0, "full": 2900.0}


def take_tier(take: Optional[Dict]) -> str:
    """The tier a take was rendered at. A take recorded before tiers is fast."""
    tag = str(((take or {}).get("generation") or {}).get("sampler") or "fast")
    return "fast" if tag == "fast" else "full"


def estimate_generate_s(rows: List[Dict], need_gen: List[Dict]):
    """(seconds to generate `need_gen`, {tier: {"shots", "each_s", "measured"}}).

    Priced PER TIER from the takes already on disk. One mean over every take mixed
    10-minute fast shots with 50-minute full ones, so the price said ~25 min for a
    shot that costs 50 and ~25 for one that costs 10.
    """
    seen: Dict[str, List[float]] = {"fast": [], "full": []}
    for r in rows:
        t = r.get("take")
        if t and t.get("seconds"):
            seen[take_tier(t)].append(float(t["seconds"]))
    out: Dict[str, Dict] = {}
    total = 0.0
    for tier in ("fast", "full"):
        n = sum(1 for r in need_gen if render_tier(r["shot"]) == tier)
        if not n:
            continue
        each = (sum(seen[tier]) / len(seen[tier])) if seen[tier] \
            else TIER_DEFAULT_S[tier]
        out[tier] = {"shots": n, "each_s": each, "measured": bool(seen[tier])}
        total += each * n
    return total, out


def film_status(project: Dict) -> Dict:
    """The whole film's state, derived from the shots. This is what the UI shows."""
    shots = shots_in_order(project)
    # ONE folder scan per shot. shot_state and best_take (which calls current_take)
    # each used to list the same take folder -- 25 scans for 10 shots, a directory
    # glob plus a JSON read per take, every 2 s while a render is using the CPU.
    rows = []
    for s in shots:
        takes = list_takes(project, s)
        rows.append({"shot": s, "state": shot_state(project, s, takes=takes),
                     "take": best_take(project, s, takes=takes)})
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    # price of the next pass: measured from the takes already on disk, so this is
    # the pipeline's own number rather than an estimate.
    per_shot = [t.get("seconds") for r in rows for t in
                ([r["take"]] if r["take"] else []) if t.get("seconds")]
    need_gen = [r for r in rows if needs_generation(r["state"])]
    need_fin = [r for r in rows if r["state"] == "restale"]
    mean = (sum(per_shot) / len(per_shot)) if per_shot else None
    est_gen, by_tier = estimate_generate_s(rows, need_gen)
    fps = int((project.get("settings") or {}).get("fps") or 16)
    return {
        "shots": len(shots),
        "frames": sum(hold_frames(s, fps) for s in shots),
        "duration_s": duration(project),
        "states": counts,
        # two prices, because they are three orders of magnitude apart
        "to_generate": len(need_gen),
        "to_refinish": len(need_fin),
        "est_generate_s": est_gen if need_gen else None,
        "est_by_tier": by_tier,
        "measured_per_shot_s": mean,
        "rows": rows,
    }


def ready_to_export(project: Dict) -> bool:
    """Nothing needs the model, and every shot has a take chosen.

    A `restale` shot does NOT block export: the finishing pass is part of the
    export and regenerates nothing.
    """
    return (bool(project.get("timeline"))
            and all(shot_state(project, s) in ("rendered", "approved", "restale")
                    for s in shots_in_order(project)))


def export_plan(project: Dict) -> Dict:
    """The exact render plan for the current timeline -- the renderer's input.

    This replaces `plan.frozen`. `--from-project` used to read a *copy* of the shot
    list made at freeze time; this reads the shots themselves, in timeline order, so
    an edit is never lost between planning and rendering.
    """
    shots = shots_in_order(project)
    fps = int((project.get("settings") or {}).get("fps") or 16)
    order = []
    t = 0.0
    for s in shots:
        hold = hold_frames(s, fps)
        order.append({
            "id": s["id"],
            "shot": Shot(prompt=s.get("prompt", ""), scale=s.get("scale", "medium"),
                         camera=s.get("camera", "none"),
                         seconds=float(s.get("seconds") or 2.5),
                         subject=s.get("subject", ""), scene=s.get("scene", ""),
                         energy=s.get("energy"), beat=s.get("beat", ""),
                         upsample=int(s.get("upsample") or 2)),
            "record": s,
            "generation": generation_inputs(project, s),
            "finish": finish_inputs(project, s),
            "take": s.get("take"),
            "take_dir": (str(take_dir(project, s, s["take"])) if s.get("take") else None),
            "state": shot_state(project, s),
            "hold_frames": hold,
            "upsample": int(s.get("upsample") or 2),
            "in_s": t,
            "out_s": t + hold / fps,
            "palette": palette_for(project, s),
        })
        t += hold / fps
    return {
        "slug": project["slug"],
        "fps": fps,
        "duration_s": t,
        "frames": sum(o["hold_frames"] for o in order),
        "order": order,
        "technical_baseline": bool((project.get("settings") or {})
                                   .get("technical_baseline", True)),
        "shake_scale": float((project.get("settings") or {}).get("shake_scale", 1.0)),
    }


# --------------------------------------------------------------------------- #
# render history -- one row per act, appended only
# --------------------------------------------------------------------------- #

def log_history(project: Dict, **row) -> None:
    row.setdefault("at", _now())
    project.setdefault("history", []).append(row)
    project["history"] = project["history"][-200:]


# --------------------------------------------------------------------------- #
# the story pass: the writer's output -> shot records with STABLE ids
# --------------------------------------------------------------------------- #

def shots_from_proposal(project: Dict, proposal,
                        cameras: Optional[Sequence[str]] = None) -> List[Dict]:
    """Adopt a `story.Proposal` as the timeline.

    The writer's per-shot `purpose`/`beat` and its camera SUGGESTION are kept as
    themselves. The old model wrote the suggestion into the same field the director
    then overwrote, so the field meant two different things depending on when you
    read it -- and the suggestion was fed back into the next planning run.

    THE CUT IS PACED HERE. The store path never re-runs the planner, and before
    this existed the writer's raw `seconds` became the delivered timeline: at the
    default 16 raw frames a shot contains 1.94 s of motion, so a writer guess of
    3.6 s played 1.86x slow (duplicated frames, judder). Adoption is the natural
    planning point: `edit_planner.plan(..., frames=, fps=)` gives the film a
    beat-shaped rhythm built from the motion the renderer supplies, repairs
    adjacent framing scales (R1), and clamps every hold to the budget. The
    writer's TEXT is kept; only timing and framing are paced.

    `cameras` is the DIRECTING decision, one camera name per shot, from
    `camera_direction.py`. It is applied before planning so R2/R3/R6 can repair
    it, and the result is stored as the shot's camera. When it is None the old
    behaviour is kept: every shot still, and the writer's suggestion is only a
    caption. Manual edits after adoption are NOT re-planned (the UI warns
    instead): a proposal must obey the machine, a person may choose.
    """
    raw = getattr(proposal, "raw", None) or {}
    project["logline"] = getattr(proposal, "logline", "") or raw.get("logline", "")
    project["provenance"] = raw.get("provenance", {}) or {}
    project["ideas"] = raw.get("ideas", []) or []
    project["problems"] = list(getattr(proposal, "problems", None) or [])
    # Place/figure anchors are the renderer's continuity mechanism: when the writing
    # path bound them, the PROMPTS are intentionally longer than one image (the
    # writer's own shape was measured BEFORE the anchors were appended). Record that
    # fact so tools re-reading the store do not report the anchors as writer faults.
    project["anchors"] = raw.get("anchors") or {}

    import handheld

    st = project.get("settings") or {}
    frames = int(st.get("frames") or 16)
    fps = int(st.get("fps") or 16)
    writer_shots = list(getattr(proposal, "shots", []) or [])
    if cameras is not None and len(cameras) != len(writer_shots):
        raise ValueError(f"cameras has {len(cameras)} entries for "
                         f"{len(writer_shots)} shots -- the direction decision "
                         f"must line up with the shot list")
    basis = ([replace(s, camera=str(c)) for s, c in zip(writer_shots, cameras)]
             if cameras is not None else writer_shots)
    planned, report = plan(basis, frames=frames, fps=fps)

    # one scene per distinct scene label the writer used, so the palette is a
    # per-scene decision from the start
    plan_palettes: Dict[str, str] = {}
    for i, sc in enumerate((raw.get("scene_plan") or {}).get("scenes") or [], 1):
        pal = str(sc.get("palette") or "")
        if pal:
            plan_palettes[f"plan{i}"] = pal
    scenes: Dict[str, Dict] = {}
    made = []
    scene_palettes: list = []
    for source, paced in zip(writer_shots, planned):
        sc_label = paced.scene or "sc1"
        # THE GRADE TRAVELS WITH THE SHOT. The writer's scene labels are often
        # rewritten by the director before adoption (planN -> movement ids), so a
        # scene-label-keyed map loses the plan's palette and every scene falls back
        # to neutral -- measured on the squid draft. A palette carried on the shot
        # survives any relabeling; the label map is only a fallback.
        shot_palette = (str(getattr(source, "palette", "") or "")
                        or plan_palettes.get(sc_label, "") or "neutral")
        if sc_label not in scenes:
            scenes[sc_label] = {"id": f"sc{len(scenes) + 1}",
                                "name": f"scene {len(scenes) + 1}",
                                "palette": shot_palette}
        sid = next_id(list(scenes.keys()) + [m["id"] for m in made])
        # belt and braces: plan() already respects the budget, and this keeps the
        # one definition of the cap if the planner ever stops doing so
        seconds = handheld.fit_hold_seconds(
            float(paced.seconds or 2.5), paced.camera or "none", frames, fps)
        if cameras is None:
            # legacy/no-direction: still, and the writer's suggestion is a caption
            stored_camera = "none"
            suggestion = paced.camera
        else:
            stored_camera = paced.camera
            mention = source.camera
            suggestion = mention if mention not in ("", "none") else None
        shot = new_shot(
            sid, prompt=paced.prompt, scale=paced.scale,
            camera=stored_camera,
            camera_suggestion=(suggestion if suggestion and suggestion != "none"
                               else None),
            seconds=seconds, scene=scenes[sc_label]["id"],
            subject=paced.subject, energy=paced.energy, beat=paced.beat,
            purpose=getattr(source, "purpose", "") or "",
            link=getattr(source, "link", "") or "",
            movement=getattr(source, "movement", "") or "",
            camera_move=getattr(source, "camera_move", None),
        )
        save_shot(project["slug"], shot, project)
        made.append(shot)
        scene_palettes.append(shot_palette)

    # one grade per scene, by majority of its shots' planned palettes: a shot may
    # move between writer labels, but the grade it was planned under is explicit
    from collections import Counter
    votes: Dict[str, Counter] = {}
    for m, pal in zip(made, scene_palettes):
        votes.setdefault(m["scene"], Counter())[pal] += 1
    for sc in scenes.values():
        counted = votes.get(sc["id"])
        if counted:
            sc["palette"] = counted.most_common(1)[0][0]

    # what the pacing pass did, kept on the project so the UI can say it and a
    # later reader can compare the writer's guess with the delivered cut
    project["pacing"] = {
        "frames": frames, "fps": fps,
        "natural_s": round(handheld.natural_seconds(frames, fps), 2),
        "writer_holds": [round(float(s.seconds or 2.5), 2) for s in writer_shots],
        "planned_holds": [round(float(s.seconds), 2) for s in planned],
        "changes": list(report.changes),
        "notes": list(report.notes),
        "cameras": [m["camera"] for m in made],
        "movers": sum(1 for m in made if m.get("camera") not in (None, "", "none")),
    }
    project["scenes"] = list(scenes.values())
    project["timeline"] = [m["id"] for m in made]
    return made


# --------------------------------------------------------------------------- #
# importing the old `workspace/<film>.json` projects
# --------------------------------------------------------------------------- #

def import_legacy(legacy_path: Union[Path, str], slug: Optional[str] = None) -> Dict:
    """Bring an old workbench project into the store, keeping what it knows.

    What survives: the concept, the logline, the shot prompts, the palettes per
    scene, the provenance and the ideas. What does NOT: `plan.frozen` (the store
    has no frozen copy -- order is the timeline and the shots are the shots) and the
    derived `status` strings (the store derives state from the disk).

    The director's per-shot CAMERA is imported as the decision, and the writer's
    original suggestion is imported as `camera_suggestion`, so the two stop being
    the same field -- which is what made the old workbench show two cameras for one
    shot (AUDIT.md section 1).
    """
    import json as _json
    legacy = _json.loads(Path(legacy_path).read_text(encoding="utf-8"))
    name = slug or legacy.get("name") or Path(legacy_path).stem
    project = new_project(name, legacy.get("concept", ""))
    project["slug"] = slugify(name)
    project["logline"] = (legacy.get("story") or {}).get("logline", "")
    project["provenance"] = (legacy.get("story") or {}).get("provenance") or {}
    project["ideas"] = (legacy.get("story") or {}).get("ideas") or []
    project["kept_ideas"] = (legacy.get("story") or {}).get("kept_ideas") or []
    project["proposal_path"] = (legacy.get("story") or {}).get("proposal_path")
    project["problems"] = list((legacy.get("story") or {}).get("problems") or [])
    project["settings"].update({
        "shake_scale": float((legacy.get("camera") or {}).get("amount_scale", 1.0)),
        "technical_baseline": bool((legacy.get("grade") or {}).get("technical", True)),
    })
    # the frozen plan is the director's/planner's output: the DECISION
    frozen = legacy.get("plan", {}).get("frozen") or []
    decided = {d["id"]: d for d in frozen if "id" in d}
    decided_by_index = {i: d for i, d in enumerate(frozen)}

    scenes: Dict[str, Dict] = {}
    looks = (legacy.get("grade") or {}).get("looks") or []
    made = []
    for i, s in enumerate(legacy.get("shots") or []):
        dec = decided.get(s.get("id"), decided_by_index.get(i, s))
        sc_label = s.get("scene") or dec.get("scene") or "sc1"
        if sc_label not in scenes:
            scenes[sc_label] = {"id": f"sc{len(scenes) + 1}",
                                "name": f"scene {len(scenes) + 1}",
                                "palette": (looks[i] if i < len(looks) and looks[i]
                                            else "neutral")}
        shot = new_shot(
            next_id([m["id"] for m in made]), prompt=s.get("prompt", ""),
            scale=dec.get("scale", s.get("scale", "medium")),
            camera=dec.get("camera", "none"),
            camera_suggestion=(s.get("camera")
                               if s.get("camera") not in (None, "", "none") else None),
            seconds=float(dec.get("seconds") or s.get("seconds") or 2.5),
            scene=scenes[sc_label]["id"], subject=s.get("subject", ""),
            energy=s.get("energy"), beat=s.get("beat", ""),
        )
        shot["legacy_render"] = s.get("render")
        save_shot(project["slug"], shot, project)
        made.append(shot)
    project["scenes"] = list(scenes.values())
    project["timeline"] = [m["id"] for m in made]
    project["film"] = legacy.get("film") or {}
    project["concept"] = legacy.get("concept", "")
    log_history(project, act="import", source=str(legacy_path),
                shots=len(made))
    save(project)
    return project
