#!/usr/bin/env python3
"""
sequence_direction.py -- the automatic director: several movements, several
grammars, one film.

THE COMPLAINT THIS ANSWERS. The organisation layer was correct but manual: the
studio asked a person to choose story/hybrid/montage/continuous and press apply.
That is the machine pushing its job onto the user. A real piece is not one
grammar: it HOLDS a place and figure for the emotional/turn beats, and it CHANGES
fast for a time-passing or escalation stretch, inside the same film. The director
should read the material and decide where each happens.

WHAT THIS DOES. `plan(shots)` partitions the shot list into MOVEMENTS (contiguous
stretches), and for each movement chooses:

  * `hold`       -- 2-4 shots sharing a place/figure (coverage). Camera: at most
                    one move, and only if there is energy for it; the rest settle.
  * `montage`    -- a rapid-change run: every shot a new image. Camera: no moves;
                    the cuts are the motion.
  * `hybrid`     -- a movement that genuinely contains both (high-affinity pairs
                    kept as coverage, the rest single cuts).
  * `continuous` -- the whole piece is one place/figure; one scene throughout.

The choice is derived from evidence the shot list already carries: distinctive
short-text affinity (same place?), the recurring `subject`, beats, and energy.
It is not a mode picker and there is no global setting. A 12-shot film can easily
come back as hold -> montage -> hybrid -> hold, and that is the point.

SCALING. `shots_for_duration(target)` converts a target length into a shot count
using the renderer's measured motion budget: at 16 raw frames a shot supplies
(2*16-1)/16 = 1.94 s, so a longer piece needs MORE SHOTS, not longer holds. Holds
beyond that need more frames per shot, which is the next render-side budget
decision, not something to fake by stretching.

Camera scoping. `scope_cameras(decided, direction)` takes the camera decisions
from `camera_direction` and enforces the movement policy: a montage movement
becomes still (cuts carry it), a hold keeps at most one move, a turn/hold can
keep its move. Moves are never invented here -- this module only decides WHERE a
move is allowed, which is exactly the seam between sequencing and camera.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from edit_planner import Shot  # noqa: E402
from visual_org import affinity, as_shots  # noqa: E402

HOLD = "hold"
MONTAGE = "montage"
HYBRID = "hybrid"
CONTINUOUS = "continuous"
MODES = (HOLD, MONTAGE, HYBRID, CONTINUOUS)

MAX_MOVEMENT = 4          # a movement longer than this stops being one idea
HOLD_AFFINITY = 0.15      # adjacent shots at/above this read as the same place
PAIR_AFFINITY = 0.18      # in a hybrid, pairs at/above this become coverage
ENERGY_MOVE = 0.30        # below this a hold gets no camera move at all


@dataclass
class Movement:
    indices: List[int]
    mode: str
    beat: str
    energy: float
    camera_budget: int
    why: str
    camera_plan: Optional[dict] = None

    @property
    def first(self) -> int:
        return self.indices[0] + 1

    @property
    def last(self) -> int:
        return self.indices[-1] + 1


@dataclass
class Direction:
    movements: List[Movement]
    frames: int = 16
    fps: int = 16
    target_seconds: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def shot_modes(self) -> List[str]:
        out = [""] * (self.movements[-1].indices[-1] + 1 if self.movements else 0)
        for m in self.movements:
            for i in m.indices:
                out[i] = m.mode
        return out

    def to_dict(self) -> Dict:
        return {
            "frames": self.frames, "fps": self.fps,
            "target_seconds": self.target_seconds,
            "movements": [
                {"shots": [i + 1 for i in m.indices], "mode": m.mode,
                 "beat": m.beat, "energy": m.energy,
                 "camera_budget": m.camera_budget,
                 "camera_plan": m.camera_plan, "why": m.why}
                for m in self.movements],
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# Scaling: length -> shot count
# --------------------------------------------------------------------------- #

def natural_seconds(frames: int = 16, fps: int = 16) -> float:
    return (2 * frames - 1) / fps


def shots_for_duration(target_seconds: float, frames: int = 16,
                       fps: int = 16) -> int:
    """How many shots a target length needs at this frame budget.

    This is the scaling answer: fit the length with shot COUNT, never by
    stretching holds. At 16 frames the motion budget is 1.94 s per shot, so a
    30 s piece is ~16 shots whatever the writer would like.

    The floor is 3, not 2. A 2-shot film cannot satisfy the story gate at all:
    it needs an establish and a resolve at the ends and a turn somewhere between
    them, and two slots cannot hold all three -- so a 4 s target used to derive
    exactly 2 shots and guarantee a REFUSED draft. Three is the smallest list the
    gate can pass, so that is the smallest the length is allowed to ask for.
    """
    n = round(max(2.0, target_seconds) / natural_seconds(frames, fps))
    return max(3, min(24, n))


def estimate(target_seconds: float, frames: int = 16, fps: int = 16,
             seconds_per_frame: float = 43.0) -> Dict:
    n = shots_for_duration(target_seconds, frames, fps)
    return {"shots": n,
            "planned_seconds": round(n * natural_seconds(frames, fps), 2),
            "render_minutes": round(n * frames * seconds_per_frame / 60.0, 1)}


# --------------------------------------------------------------------------- #
# Camera: one motivated move, in FINISHING, with a settle
# --------------------------------------------------------------------------- #

CAMERA_MOVES = ("zoom-in", "zoom-out", "pan-left", "pan-right", "tilt-up",
                "tilt-down")
_AMOUNT_BY_SCALE = {"extreme-wide": 50, "wide": 45, "medium": 38, "close": 32,
                    "extreme-close": 28, "macro": 28}


def camera_for(shots: Sequence[Shot], idx: Sequence[int]) -> Optional[Dict]:
    """The one camera move a movement can carry, as a FINISHING spec.

    A generated LoRA move cannot settle (measured: still travelling 2.79 px/f at
    the last frame), so a held shot that needs a camera move gets it applied to
    the delivered frames by `camera_path.apply_path` -- travel, then hold. This is
    the missing render-side half the A/B identified.

    The move is chosen from the shot's purpose/scale and the writer's suggestion,
    not invented randomly: establish/wide reveals, insert/close pushes, resolve
    pulls back.
    """
    if not idx:
        return None
    preferred = [i for i in idx if shots[i].purpose in ("establish", "reaction",
                                                        "resolve")]
    i = preferred[0] if preferred else idx[0]
    s = shots[i]
    suggested = s.camera if s.camera in CAMERA_MOVES else ""
    if suggested:
        move = suggested
    elif s.purpose == "resolve":
        move = "zoom-out"
    elif s.purpose == "insert" or s.scale in ("close", "extreme-close", "macro"):
        move = "zoom-in"
    elif s.scale in ("wide", "extreme-wide"):
        move = "tilt-down"
    else:
        move = "pan-left"
    amount = _AMOUNT_BY_SCALE.get(s.scale, 38)
    settle = 0.62 if move.startswith(("pan", "tilt")) else 0.58
    return {"shot": i, "move": move, "amount": amount, "settle": settle,
            "why": f"one motivated {move} on shot {i + 1} ({s.scale}, "
                   f"{s.purpose or s.beat}): travels then settles"}


def _writer_groups(shots: Sequence[Shot]) -> Optional[List[tuple]]:
    """The writer's own movement ids, if they form contiguous, unambiguous runs.

    The writer is now asked to plan movements before shots. When it does, the
    director uses that structure instead of re-deriving it; when the ids are
    missing or interleaved, it falls back to content segmentation.
    """
    ids = [str(s.movement or "").strip() for s in shots]
    if not shots or not all(ids):
        return None
    groups: List[list] = []
    for i, mid in enumerate(ids):
        if groups and groups[-1][0] == mid:
            groups[-1][1].append(i)
        else:
            groups.append([mid, [i]])
    seen = set()
    for mid, _ in groups:
        if mid in seen:
            return None                  # interleaved A-B-A: not a usable plan
        seen.add(mid)
    return [(mid, idx) for mid, idx in groups]


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #

def _boundaries(shots: Sequence[Shot]) -> List[int]:
    n = len(shots)
    if n <= 1:
        return [0, n]
    cuts = [0]
    start = 0
    for i in range(1, n):
        aff = affinity(shots, i - 1, i)
        e_shift = abs((shots[i].energy or 0) - (shots[i - 1].energy or 0))
        too_long = i - start >= MAX_MOVEMENT
        # PLACE/FIGURE CONTINUITY IS THE PRIMARY BOUNDARY, not the beat label:
        # a hold can carry a beat change (the turn happens in the same place),
        # and splitting on beats was what forced every 4-shot piece into a
        # four-way montage. A movement still gets at least two shots unless the
        # energy changes hard.
        if aff >= HOLD_AFFINITY and not too_long:
            continue
        if not too_long and e_shift < 0.30 and i - start == 1:
            continue
        cuts.append(i)
        start = i
    cuts.append(n)
    return cuts


def _mode_for(shots: Sequence[Shot], idx: List[int], whole_film: bool) -> tuple:
    if len(idx) == 1:
        return "single", "one shot: a change or accent, not a hold"
    pairs = [affinity(shots, a, b) for a, b in zip(idx, idx[1:])]
    mean_aff = sum(pairs) / len(pairs) if pairs else 0.0
    strong = sum(1 for p in pairs if p >= HOLD_AFFINITY)
    max_e = max((shots[i].energy or 0) for i in idx)
    if mean_aff >= HOLD_AFFINITY:
        if whole_film and len(idx) >= 4:
            return CONTINUOUS, (f"one place/figure across the whole piece "
                                f"(affinity {mean_aff:.2f}); one continuous scene")
        return HOLD, (f"{strong}/{len(pairs)} adjacent pairs share place/figure "
                      f"(affinity {mean_aff:.2f}); hold it")
    if max_e >= 0.35 or any(shots[i].beat == "develop" for i in idx):
        return MONTAGE, (f"different places/figures and energy up to {max_e:.2f}; "
                         f"a rapid-change run")
    return HYBRID, (f"mixed continuity (affinity {mean_aff:.2f}); keep the pairs "
                    f"and cut the rest")


def _merge_montage(movements: List[Movement]) -> List[Movement]:
    """Consecutive rapid-change movements are ONE run, not a row of accidents."""
    merged: List[Movement] = []
    for m in movements:
        if (m.mode == MONTAGE and merged and merged[-1].mode == MONTAGE
                and merged[-1].indices[-1] + 1 == m.indices[0]
                and len(merged[-1].indices) + len(m.indices) <= MAX_MOVEMENT):
            prev = merged[-1]
            prev.indices.extend(m.indices)
            prev.energy = max(prev.energy, m.energy)
            prev.why = "rapid-change run: consecutive single-image cuts"
        else:
            merged.append(m)
    # a single-image movement is a montage run of one
    for m in merged:
        if m.mode == "single":
            m.mode = MONTAGE
            m.why = "rapid-change run: single-image cuts"
            m.camera_budget = 0
    return merged


def plan(shots: Sequence, target_seconds: Optional[float] = None,
         frames: int = 16, fps: int = 16) -> Direction:
    shots = as_shots(shots)
    n = len(shots)
    if not n:
        return Direction([], frames, fps, target_seconds)
    notes: List[str] = []
    groups = _writer_groups(shots)
    if groups:
        spans = [(idx[0], idx[-1] + 1) for _mid, idx in groups]
        notes.append("writer's movement ids respected: "
                     + ", ".join(f"{mid} ({len(idx)} shots)"
                                 for mid, idx in groups))
    else:
        spans = list(zip(_boundaries(shots), _boundaries(shots)[1:]))
    movements: List[Movement] = []
    for a, b in spans:
        idx = list(range(a, b))
        mode, why = _mode_for(shots, idx, whole_film=(n <= 4))
        movements.append(Movement(idx, mode, shots[a].beat,
                                  max((shots[i].energy or 0) for i in idx),
                                  1 if mode in (HOLD, CONTINUOUS, HYBRID) else 0,
                                  why))
    if not groups:
        movements = _merge_montage(movements)
    # camera policy: a hold only moves if it has energy for it, and never more
    # than one; a montage is still (the cuts are the motion); a single accent may
    # carry one move at a turn.
    for m in movements:
        if m.mode == MONTAGE:
            m.camera_budget = 0
        elif m.mode == HOLD and m.energy < ENERGY_MOVE:
            m.camera_budget = 0
            m.why += " · no move: low energy, let the hold settle"
        if m.camera_budget > 0:
            plan_cam = camera_for(shots, m.indices)
            if plan_cam:
                m.camera_plan = plan_cam
                m.why += f" · {plan_cam['why']}"
            else:
                m.camera_budget = 0
    if target_seconds:
        need = shots_for_duration(target_seconds, frames, fps)
        notes.append(f"target {target_seconds:.0f}s needs ~{need} shots at "
                     f"{frames} frames; got {n}. Longer needs more shots, not "
                     f"longer holds.")
    return Direction(movements, frames, fps, target_seconds, notes)


# --------------------------------------------------------------------------- #
# Applying a direction to a shot list
# --------------------------------------------------------------------------- #

def apply(shots: Sequence, direction: Direction,
          camera_finish: bool = True) -> List[Shot]:
    """Write the direction into the shot list.

    With `camera_finish=True` (the default path) generation is STATIC for every
    shot and the movement's one move is recorded as a finishing spec
    (`camera_move`): travel, then settle, applied to the delivered frames by
    `camera_path`. A generated camera LoRA cannot settle, which is why the move
    belongs here. With `camera_finish=False` the caller is using the explicit
    `--cameras` override and camera LoRAs, so the plan leaves camera fields alone.
    """
    shots = as_shots(shots)
    out = [replace(s) for s in shots]
    scene_id = 0
    prev_scene = None
    for m in direction.movements:
        if m.mode == MONTAGE:
            for i in m.indices:
                scene_id += 1
                out[i] = replace(out[i], scene=f"mv{scene_id}")
        elif m.mode in (HOLD, CONTINUOUS):
            scene_id += 1
            sid = f"mv{scene_id}"
            for i in m.indices:
                out[i] = replace(out[i], scene=sid)
        else:  # hybrid: pair the shots that actually continue each other
            i = 0
            while i < len(m.indices):
                idx = m.indices[i]
                scene_id += 1
                sid = f"mv{scene_id}"
                out[idx] = replace(out[idx], scene=sid)
                count = 1
                j = i + 1
                while (j < len(m.indices) and count < MAX_MOVEMENT
                       and affinity(shots, m.indices[j - 1], m.indices[j])
                       >= PAIR_AFFINITY):
                    out[m.indices[j]] = replace(out[m.indices[j]], scene=sid)
                    count += 1
                    j += 1
                i = j
    # links: the first shot of the film opens; a new scene is a scene-cut;
    # inside a scene, use the same-entity vocabulary
    prev_scene = None
    for i in range(len(out)):
        if i == 0:
            out[i] = replace(out[i], link="open")
        elif out[i].scene != prev_scene:
            out[i] = replace(out[i], link="scene-cut")
        else:
            out[i] = replace(out[i], link=("eyeline"
                                           if out[i - 1].scale != out[i].scale
                                           else "match-action"))
        prev_scene = out[i].scene
    if camera_finish:
        for m in direction.movements:
            for i in m.indices:
                out[i] = replace(out[i], camera="none", camera_move=None)
            plan_cam = m.camera_plan
            if plan_cam:
                j = int(plan_cam["shot"])
                out[j] = replace(out[j], camera="none", camera_move={
                    "move": plan_cam["move"],
                    "amount": plan_cam["amount"],
                    "settle": plan_cam["settle"],
                })
    return out


def scope_cameras(cameras: Sequence[str], direction: Direction) -> List[str]:
    """Enforce the movement policy on camera decisions (never invents a move)."""
    out = ["none"] * len(cameras)
    for m in direction.movements:
        allowed = [i for i in m.indices if cameras[i] != "none"]
        for i in allowed[:m.camera_budget]:
            out[i] = cameras[i]
    return out


def describe(direction: Direction) -> str:
    lines = []
    for m in direction.movements:
        lines.append(f"  shots {m.first}-{m.last} · {m.mode:<10} · beat "
                     f"{m.beat:<9} · camera <={m.camera_budget} · {m.why}")
    for n in direction.notes:
        lines.append(f"  note: {n}")
    return "\n".join(lines)


def direct_project(project: Dict, target_seconds: Optional[float] = None,
                   frames: Optional[int] = None,
                   fps: Optional[int] = None) -> Direction:
    """Re-direct a store production automatically; returns the direction.

    Rewrites scene ids/links and scopes whatever camera decisions are already on
    the shots. Palettes travel with each place (first shot of the new group keeps
    its old scene's palette). No mode argument: the plan is the decision.
    """
    import film_store as fs

    st = project.get("settings") or {}
    frames = int(frames or st.get("frames") or 16)
    fps = int(fps or st.get("fps") or 16)
    dicts = fs.shots_in_order(project)
    shots = as_shots(dicts)
    direction = plan(shots, target_seconds, frames, fps)
    # SAME CAMERA POLICY AS `draft.py`: unless the production stores an explicit
    # `camera_mode`, generation is static and each movement's one motivated move is
    # a FINISHING spec (`camera_move`). Re-direct used to compute that move and then
    # throw it away (only scene/link/camera were written), so a re-directed film
    # kept the previous plan's moves on shots the new plan no longer moves.
    auto_camera = not (project.get("settings") or {}).get("camera_mode")
    arranged = apply(shots, direction, camera_finish=auto_camera)
    old_scene = [s.scene for s in shots]
    old_pal = {s.get("id"): s.get("palette") for s in project.get("scenes") or []}
    pal_for_new: Dict[str, str] = {}
    for old_s, ns in zip(old_scene, arranged):
        pal_for_new.setdefault(ns.scene, old_pal.get(old_s, "neutral"))
    if auto_camera:
        scoped = ["none"] * len(dicts)
    else:
        scoped = scope_cameras([str(d.get("camera") or "none") for d in dicts],
                               direction)
    for d, ns, cam in zip(dicts, arranged, scoped):
        fields = dict(scene=ns.scene, link=ns.link, camera=cam)
        if auto_camera:
            fields["camera_move"] = ns.camera_move
        fs.update_shot(project, d, **fields)
    fs.sync_scenes(project)
    for sc in project.get("scenes") or []:
        sc["name"] = str(sc.get("id") or "?")
    for sid, pal in pal_for_new.items():
        fs.set_palette(project, sid, pal)
    project["direction"] = direction.to_dict()
    fs.save(project)
    return direction
