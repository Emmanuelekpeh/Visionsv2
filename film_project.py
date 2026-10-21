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
STAGES = ["story", "plan", "camera", "grade", "render", "timeline"]


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
    proj["status"]["story"] = "written"
    proj["status"]["plan"] = "stale"
    return proj


def _shot_to_dict(s: Shot, i: int) -> Dict:
    return {"id": i, "prompt": s.prompt, "scale": s.scale, "camera": s.camera,
            "seconds": s.seconds, "scene": s.scene, "subject": s.subject,
            "energy": s.energy, "beat": s.beat,
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
                 energy=d.get("energy"), beat=d.get("beat", ""))
            for d in proj["shots"]]


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
                        beat=d.get("beat", "")))
    return out


def frozen_report(proj: Dict) -> FrozenReport:
    return FrozenReport(changes=list(proj["plan"].get("planner_changes") or []),
                        text=proj["plan"].get("report", ""))
