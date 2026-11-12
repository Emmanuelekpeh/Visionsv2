"""
film_project.py -- the workbench's durable project file.

The problem this solves is not storage, it is *authorship*. `make_film.py`
recomputes the director and the planner on every run, so a plan you looked at and
approved is not necessarily the plan that renders, and any edit made in a UI would
be silently discarded. This module makes the plan a thing you own:

    workspace/<film>.json

holds the concept, the proposal as validated, the per-shot records, the FROZEN
plan (the exact shot list that will render, post-R1/R4), the planner's changes and
its report, and per-stage state. Every edit writes it. `make_film --from-project`
renders those frozen shots and skips both stages, so what you approved is what
comes out.

It calls the pipeline's own functions -- `story.validate`, `director.choose`,
`edit_planner.plan` -- and reimplements no measurement.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from edit_planner import Shot, plan
from story import validate

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"

# The stages a film moves through, in order. Each one produces an artifact the
# user can edit before the next; `status` is per stage so the board can show where
# a project actually is rather than guessing.
STAGES = ["story", "organise", "previs", "plan", "camera", "grade", "render",
          "timeline"]


@dataclass
class FrozenReport:
    """Just enough of an edit_planner report to satisfy the render path."""
    changes: List[str] = field(default_factory=list)
    text: str = ""

    def show(self) -> str:
        return self.text or "(frozen plan: no planner report stored)"


def path_for(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in "-_") or "film"
    return WORKSPACE / f"{safe}.json"


def new(name: str, concept: str = "") -> Dict:
    return {
        "name": name,
        "concept": concept,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "story": {"logline": "", "approved": False, "proposal_path": None,
                  "ideas": [], "provenance": {}, "problems": []},
        # THE HOLD/CHANGE SPINE. `visual_org` derives a run list from the shots
        # (one place + one figure per run), stores the diagnostics, and owns the
        # place/figure anchors. The PLAN stage freezes prompts WITH those anchors,
        # so the renderer is told the same place/figure on every shot of a run.
        "organisation": {"mode": "auto", "inferred_mode": "auto",
                         "places": {}, "figures": {}, "runs": [],
                         "diagnostics": [], "inside_overlap": None,
                         "across_overlap": None, "approved": False},
        # PREVIS: one still per shot, measured at the cuts, before the render.
        # Optional (it costs minutes): `previs_project` fills this in.
        "previs": {"done": False, "candidate": None, "report": None, "at": None},
        "shots": [],
        "plan": {"mode": "global", "max_movers": None, "energy_arc": True,
                 "approved": False, "approved_at": None, "frozen": [],
                 "planner_changes": [], "report": ""},
        "camera": {"amount_scale": 1.0, "overrides": {}},
        "grade": {"technical": True, "looks": None},
        "film": {"name": None, "duration_s": None, "frames": None},
        "status": {s: "empty" for s in STAGES},
    }


def load(name: str) -> Dict:
    p = path_for(name)
    if not p.exists():
        raise FileNotFoundError(f"no project at {p}")
    proj = json.loads(p.read_text(encoding="utf-8"))
    # a project written by an older version must not crash the board
    base = new(proj.get("name", name))
    for k, v in base.items():
        if k not in proj:
            proj[k] = v
        elif isinstance(v, dict) and isinstance(proj[k], dict):
            for kk, vv in v.items():
                proj[k].setdefault(kk, vv)
    return proj


def save(proj: Dict) -> Path:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    p = path_for(proj["name"])
    proj["saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p.write_text(json.dumps(proj, indent=2), encoding="utf-8")
    return p


def list_projects() -> List[Path]:
    """Only files that ARE film projects.

    `workspace/` also holds `current_project.json` from the retired pygame studio,
    and its schema is different. Offering it as a project would put the old app's
    data model in front of the user as if it were a film.
    """
    if not WORKSPACE.exists():
        return []
    out = []
    for p in WORKSPACE.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and "shots" in d and "plan" in d:
            out.append(p)
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


# --------------------------------------------------------------------------- #
# STORY
# --------------------------------------------------------------------------- #

def shots_from_raw(proj: Dict, raw: Dict, proposal_path: Optional[str] = None,
                   provenance: Optional[Dict] = None, ideas: Optional[list] = None):
    """Adopt a writer's proposal as the project's shot list, via `story.validate`.

    validate is the measurement -- adopting the raw JSON directly would let the
    board show a shot list the pipeline would never produce.
    """
    shots, problems = validate(raw)
    if not shots:
        raise ValueError(f"proposal did not validate: {problems}")
    proj["story"]["logline"] = str(raw.get("logline", ""))
    proj["story"]["proposal_path"] = proposal_path
    proj["story"]["provenance"] = provenance or raw.get("provenance", {})
    proj["story"]["ideas"] = ideas if ideas is not None else raw.get("ideas", [])
    proj["story"]["problems"] = list(problems)
    proj["story"]["approved"] = False
    proj["shots"] = [_shot_to_dict(s, i) for i, s in enumerate(shots)]
    proj["plan"]["frozen"] = []          # a new story invalidates the plan
    proj["plan"]["approved"] = False
    # a new story invalidates the organisation too: it belongs to these shots
    proj["organisation"] = {
        "mode": "auto", "inferred_mode": "auto", "places": {}, "figures": {},
        "runs": [], "diagnostics": [], "inside_overlap": None,
        "across_overlap": None, "approved": False}
    proj["previs"] = {"done": False, "candidate": None, "report": None, "at": None}
    proj["status"]["story"] = "written"
    proj["status"]["organise"] = "stale"
    proj["status"]["previs"] = "stale"
    proj["status"]["plan"] = "stale"
    return proj


def _shot_to_dict(s: Shot, i: int) -> Dict:
    return {"id": i, "prompt": s.prompt, "scale": s.scale, "camera": s.camera,
            "seconds": s.seconds, "scene": s.scene, "subject": s.subject,
            "energy": s.energy, "beat": s.beat,
            "purpose": getattr(s, "purpose", "") or "",
            "link": getattr(s, "link", "") or "",
            "movement": getattr(s, "movement", "") or "",
            "camera_move": getattr(s, "camera_move", None),
            # what the PLAN stage writes; None means "let the director decide"
            "motion_pin": None,
            # what the RENDER stage tracks
            "render": {"status": "pending", "seconds": None, "at": None}}


def shot_objects(proj: Dict) -> List[Shot]:
    """The project's shots as edit_planner Shots, as the WRITER declared them.

    Motion pins are deliberately NOT applied here: `director.choose` decides every
    shot's camera and overwrites whatever is on the Shot, so setting it here would
    look like a constraint while being ignored. Pins are applied after the
    director, in `plan_project`, where the override is visible for what it is.
    """
    return [Shot(prompt=d["prompt"], scale=d["scale"],
                 camera=d.get("camera") or "none",
                 seconds=float(d.get("seconds") or 2.5),
                 subject=d.get("subject", ""), scene=d.get("scene", ""),
                 energy=d.get("energy"), beat=d.get("beat", ""),
                 purpose=d.get("purpose", "") or "",
                 link=d.get("link", "") or "",
                 movement=d.get("movement", "") or "",
                 camera_move=d.get("camera_move"))
            for d in proj["shots"]]


# --------------------------------------------------------------------------- #
# ORGANISE -- the hold/change spine (visual organisation)
# --------------------------------------------------------------------------- #

def organise_project(proj: Dict, mode: str = "auto",
                     places: Optional[Dict[str, str]] = None,
                     figures: Optional[Dict[str, str]] = None) -> Dict:
    """Derive the hold/change runs and diagnose them. Writes nothing to shots.

    This is the stage the pipeline was missing: before directing camera moves or
    choosing models, decide where the film holds a place and figure and where it
    deliberately changes them. `places`/`figures` are anchors -- the description
    the renderer must be told on EVERY shot of a run; `effective_prompt` appends
    them at freeze so a shot that only says "her hand on the rail" still says
    which rail, in which rain.
    """
    import visual_org as vo

    shots = shot_objects(proj)
    org = proj.setdefault("organisation", {})
    if places:
        org["places"] = dict(places)
    if figures:
        org["figures"] = dict(figures)
    runs = vo.organise(shots)
    rep = vo.diagnose(shots, runs, mode=mode,
                      places=org.get("places") or {},
                      figures=org.get("figures") or {})
    org.update({
        "mode": rep["mode"], "inferred_mode": rep["inferred_mode"],
        "runs": [{"place": r.place, "figure": r.figure, "shots": r.indices,
                  "hold": r.hold, "kind": r.kind,
                  "change_before": r.change_before} for r in runs],
        "diagnostics": rep["issues"], "breaks": rep["breaks"],
        "warns": rep["warns"],
        "inside_overlap": rep["inside_overlap"],
        "across_overlap": rep["across_overlap"],
        "approved": False,
    })
    proj["status"]["organise"] = "breaks" if rep["breaks"] else "organised"
    proj["status"]["plan"] = "stale"
    return proj


def set_place_palette(proj: Dict, place: str, palette: str) -> Dict:
    """A place owns its palette, so re-arranging the organisation keeps it.

    The store keys palettes by scene id; if organisation rewrites the scene ids
    (montage <-> coverage), a scene-id-keyed palette is lost. The place is the
    stable identity, so the palette lives on the place here and is written into
    the materialised scene's `palette` at render time.
    """
    org = proj.setdefault("organisation", {})
    org.setdefault("place_palettes", {})[place] = palette
    proj["status"]["previs"] = "stale"
    proj["status"]["plan"] = "stale"
    return proj


def apply_organisation(proj: Dict, mode: str) -> Dict:
    """Rearrange scene ids/links to express `mode` (story/hybrid/montage/...)."""
    import visual_org as vo

    shots = shot_objects(proj)
    old_places = [d.get("scene") for d in proj["shots"]]
    old_palettes = dict((proj.get("organisation") or {}).get("place_palettes") or {})
    new, notes = vo.suggestion(shots, mode)
    for d, s in zip(proj["shots"], new):
        d["scene"] = s.scene
        d["link"] = s.link
    # carry each place's palette into whatever place its shots became
    new_palettes: Dict[str, str] = {}
    for old_place, s in zip(old_places, new):
        new_palettes.setdefault(s.scene,
                                old_palettes.get(old_place, "neutral"))
    proj.setdefault("organisation", {})["notes"] = notes
    proj["organisation"]["place_palettes"] = new_palettes
    proj["plan"]["frozen"] = []
    proj["plan"]["approved"] = False
    proj["status"]["previs"] = "stale"
    organise_project(proj, mode)
    return proj


def approve_organisation(proj: Dict, force: bool = False) -> Dict:
    org = proj.get("organisation") or {}
    if not org.get("runs"):
        raise ValueError("nothing to approve; organise the project first")
    if org.get("breaks") and not force:
        raise ValueError(
            f"{org['breaks']} organisation break(s) -- release the place/figure "
            f"anchors or pass force=True to shoot it as a deliberate montage")
    org["approved"] = True
    proj["status"]["organise"] = "approved"
    proj["status"]["plan"] = "stale"
    return proj


def previs_project(proj: Dict, candidate: str = "as_written",
                   name: Optional[str] = None) -> Dict:
    """One still per shot, measured at the cuts, before the render hour.

    Optional and minutes-not-hours: it exists to kill an arrangement whose images
    do not hold together, and to rank two candidates before committing to motion.
    The REPORT is stored on the project; the stills live under output/previs/<name>.
    """
    import previs as pv
    import visual_org as vo

    shots = shot_objects(proj)
    if not shots:
        raise ValueError("no shots to previs; adopt a story first")
    if candidate != "as_written":
        shots, _notes = vo.suggestion(shots, candidate)
    report = pv.run(shots, name or f"{proj['name']}_{candidate}")
    proj.setdefault("previs", {})
    proj["previs"].update({"done": True, "candidate": candidate,
                           "report": report,
                           "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    proj["status"]["previs"] = "done"
    proj["status"]["plan"] = "stale"
    return proj


def materialize_store(proj: Dict, name: Optional[str] = None) -> Dict:
    """Write the frozen plan into a production so it renders through the STORE path.

    WHY THIS EXISTS. The workbench had its own render path (`--from-project`), and
    the store path had the hold conditioning, the take ladder, the stats and the
    staleness hash. The process that decided holds could not condition them, and
    the path that conditioned them did not have the ORGANISE/PREVIS stages. This
    converts a frozen film_project into a real production under
    `productions/fp_<name>/`, preserving any takes from a previous materialisation
    (so re-rendering only re-shoots what actually changed), and `make_film
    --from-project` then simply calls `run_store`.

    Palette identity: scenes are the frozen plan's place labels, and the palette is
    looked up on the PLACE (`organisation.place_palettes`), not inferred from a
    scene index -- so re-arranging the organisation cannot silently drop a look.
    """
    import film_store as fs

    frozen = proj.get("plan", {}).get("frozen") or []
    if not frozen:
        raise ValueError("no frozen plan; plan and approve the project first")
    pname = (name or proj.get("name") or "film").strip()
    slug = f"fp_{fs.slugify(pname)}"
    existed = (fs.PRODUCTIONS / slug / "project.json").exists()
    store = fs.load(slug) if existed else fs.new_project(pname, proj.get("concept", ""))
    store["slug"] = slug
    store["name"] = f"{pname} (workbench)"

    film = proj.get("film") or {}
    frames = int(film.get("frames") or 16)
    fps = int(film.get("fps") or 16)
    store.setdefault("settings", {}).update({
        "frames": frames, "fps": fps, "models": "off",
        "technical_baseline": bool((proj.get("grade") or {}).get("technical", True)),
        "shake_scale": float((proj.get("camera") or {}).get("amount_scale", 1.0)),
    })
    store["concept"] = proj.get("concept", "") or store.get("concept", "")
    store["logline"] = ((proj.get("story") or {}).get("logline") or ""
                        or store.get("logline", ""))
    store["problems"] = list((proj.get("story") or {}).get("problems") or [])

    # places -> scenes, palette owned by the place
    labels = [str(d.get("scene") or "s1") for d in frozen]
    label_map: Dict[str, str] = {}
    for lab in labels:
        if lab not in label_map:
            label_map[lab] = f"sc{len(label_map) + 1}"
    old_scenes = {s.get("name"): s for s in store.get("scenes") or []}
    place_palettes = (proj.get("organisation") or {}).get("place_palettes") or {}
    store["scenes"] = [
        {"id": label_map[lab], "name": lab,
         "palette": (place_palettes.get(lab)
                     or (old_scenes.get(lab) or {}).get("palette")
                     or "neutral")}
        for lab in label_map]

    ids: List[str] = []
    for i, d in enumerate(frozen):
        sid = f"s{i + 1:02d}"
        ids.append(sid)
        fields = {
            "prompt": d.get("prompt", ""), "scale": d.get("scale", "medium"),
            "camera": d.get("camera", "none"),
            "seconds": float(d.get("seconds") or 2.5),
            "scene": label_map[labels[i]],
            "subject": d.get("subject", ""),
            "energy": float(d.get("energy") or 0.2),
            "beat": d.get("beat", "develop"),
            "purpose": d.get("purpose", "") or "",
            "link": d.get("link", "") or "",
            "movement": d.get("movement", "") or "",
            "camera_move": d.get("camera_move"),
            "camera_suggestion": None,
        }
        if sid in (store.get("timeline") or []):
            shot = fs.get_shot(store, sid) or fs.new_shot(sid)
            shot.update(fields)          # take + render_inputs survive
            fs.save_shot(slug, shot, store)
        else:
            fs.save_shot(slug, fs.new_shot(sid, **fields), store)
    store["timeline"] = ids
    fs.save(store)
    return {"slug": slug, "name": pname, "frames": frames,
            "shots": len(ids), "created": not existed}


# --------------------------------------------------------------------------- #
# PLAN
# --------------------------------------------------------------------------- #

def plan_project(proj: Dict, mode: str = "global", max_movers=None,
                 energy_arc: bool = True, scorer=None, verbose: bool = False):
    """Direct and then plan, and STORE the result. Nothing renders from this.

    A motion pin in a shot overrides the director for that shot: the pin is
    applied to the shot object handed IN, so the director sees the shot as already
    decided and its own optimisation runs around it.
    """
    if not proj["shots"]:
        raise ValueError("no shots to plan; adopt a story first")
    from director import FitScorer, choose

    if not (proj.get("organisation") or {}).get("runs"):
        organise_project(proj)       # a plan without the spine is the old guesswork

    shots = shot_objects(proj)
    directed, _rep = choose(shots, scorer=scorer or FitScorer(), mode=mode,
                            max_movers=max_movers, energy_arc=energy_arc,
                            verbose=verbose)

    pins = [d.get("motion_pin") for d in proj["shots"]]
    if any(pins):
        from dataclasses import replace
        # HONEST LIMITATION: the director was not told about the pins, so its
        # moving-shot budget and its 180-degree gate were evaluated against the
        # motions the pins replace. A pin is a deliberate override of the
        # optimisation, not a constraint it respects. Making it a constraint means
        # passing the pinned shots in as already-decided options, which is the
        # PLAN stage's job, not this function's.
        directed = [replace(s_, camera=p) if p else s_
                    for s_, p in zip(directed, pins)]

    planned, report = plan(directed)
    # FREEZE WITH ANCHORS + AUTO DIRECTION. The planner fixed camera/hold; the
    # organisation owns the place/figure identity, and the sequence director owns
    # WHERE the film holds and where it changes. Camera moves are finishing specs
    # (the generated clip cannot settle), so generation is frozen still.
    import visual_org as vo
    import sequence_direction as sd
    org = proj.get("organisation") or {}
    planned = vo.apply_anchors(planned, vo.organise(planned),
                               org.get("places") or {}, org.get("figures") or {})
    _film = proj.get("film") or {}
    direction = sd.plan(planned, frames=int(_film.get("frames") or 16),
                        fps=int(_film.get("fps") or 16))
    planned = sd.apply(planned, direction, camera_finish=True)
    proj["direction"] = direction.to_dict()
    proj["plan"].update({
        "mode": mode, "max_movers": max_movers, "energy_arc": bool(energy_arc),
        "approved": False, "approved_at": None,
        "planner_changes": list(report.changes),
        "report": report.show(),
        "frozen": [_shot_to_dict(s, i) for i, s in enumerate(planned)],
    })
    proj["status"]["plan"] = "planned"
    return proj


def approve_plan(proj: Dict):
    if not proj["plan"].get("frozen"):
        raise ValueError("nothing to approve; plan the project first")
    proj["plan"]["approved"] = True
    proj["plan"]["approved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    proj["status"]["plan"] = "approved"
    return proj


def shake_params(proj: Dict, i: int) -> Dict:
    """The per-shot shake knobs the CAMERA stage edits, for shot index i.

    Kept as PARAMETERS, not as a spec: the spec is derived from the shot by
    `handheld.spec_for_shot`, and storing a derived spec in the project would let
    the two drift. `make_film` calls this so a camera edit actually reaches the
    render instead of being quietly ignored.
    """
    cam = proj.get("camera") or {}
    out = {"amount_scale": float(cam.get("amount_scale", 1.0)),
           "seed": i + 3, "enabled": True}
    ov = (cam.get("overrides") or {}).get(str(i)) or {}
    for k in ("amount_scale", "seed", "enabled"):
        if ov.get(k) is not None:
            out[k] = ov[k]
    return out


def set_shake_param(proj: Dict, i: int, **values) -> None:
    """Write a per-shot override, dropping keys set back to the global default."""
    cam = proj.setdefault("camera", {"amount_scale": 1.0, "overrides": {}})
    ov = cam.setdefault("overrides", {}).setdefault(str(i), {})
    for k, v in values.items():
        if v is None:
            ov.pop(k, None)
        else:
            ov[k] = v
    if not ov:
        cam["overrides"].pop(str(i), None)
    proj["status"]["camera"] = "edited"


def frozen_shots(proj: Dict) -> List[Shot]:
    """The exact shot list that will render. Empty until a plan is frozen."""
    out = []
    for i, d in enumerate(proj["plan"].get("frozen") or []):
        out.append(Shot(prompt=d["prompt"], scale=d["scale"], camera=d["camera"],
                        seconds=float(d["seconds"]), subject=d.get("subject", ""),
                        scene=d.get("scene", ""), energy=d.get("energy"),
                        beat=d.get("beat", ""),
                        purpose=d.get("purpose", "") or "",
                        link=d.get("link", "") or "",
                        movement=d.get("movement", "") or "",
                        camera_move=d.get("camera_move")))
    return out


def frozen_report(proj: Dict) -> FrozenReport:
    return FrozenReport(changes=list(proj["plan"].get("planner_changes") or []),
                        text=proj["plan"].get("report", ""))
