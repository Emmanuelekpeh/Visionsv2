"""
camera_direction.py -- decide which shots MOVE, for the store path.

THE GAP THIS CLOSES. `film_store.shots_from_proposal` used to freeze every shot
at `camera="none"` and keep the writer's suggestion as a caption. The director --
the module whose whole job is picking motion from fit and continuity -- was only
called by the non-store CLI path. Measured on the eleven productions in
`productions/`: 0-1 camera moves per film, i.e. R2/R3/R6 had nothing to act on
and every shot was still except for AnimateDiff's own drift and the handheld
shake.

MODES, because the honest answer is not one policy:

  off          every shot still. The old behaviour, kept as the A/B control.
  suggestions  the WRITER's camera suggestion, continuity-repaired downstream by
               `edit_planner.plan` (R2/R6/R3). The writer sees the scene and was
               explicitly asked for at most two moves, so this is a real signal;
               the pipeline just never used it.
  director     `director.choose()` over CAMERA LoRAs only: CLIP fit, keyword
               hints, continuity and the mover cap. Measured conservative: on a
               real 13-shot film it chose 1 move, because `none` starts at fit
               0.62 and a softmax over nine camera candidates rarely beats it
               without a hint word. Safe, but nearly inert on its own.
  hybrid       director first; where the director declines, keep the writer's
               suggestion if it is a valid camera move. The director owns
               continuity/taste, the writer owns "this shot wants to move".
               This is the default because it is the only mode that measured a
               useful number of moves without exceeding the brief's budget.

Subject-motion LoRAs are NOT in the default candidate set. They are semantically
sharp-edged (`ballerina-spin` got chosen for a hunched figure in testing) and the
one honest A/B on this machine is a camera-move A/B. `include_subject=True` opts
in, and `edit_planner` still enforces the reversal rules afterwards.

Everything here returns plain camera names; `film_store.shots_from_proposal`
stores them and `plan()` applies the final continuity repair, so there is still
one place that writes a shot.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import motion_vocab as mv
from edit_planner import CAMERAS, Shot

MODES = ("off", "suggestions", "director", "hybrid")
DEFAULT_MODE = "hybrid"


def camera_candidates(include_subject: bool = False) -> List[str]:
    """The moves this module is allowed to choose. Camera LoRAs by default."""
    out = ["none"]
    for name in mv.available():
        kind = mv.get(name).kind
        if kind == "camera" or (include_subject and kind == "subject"):
            out.append(name)
    return out


def _valid(name: Optional[str]) -> bool:
    return bool(name) and name in CAMERAS


def suggestion_cameras(shots: Sequence[Shot]) -> List[str]:
    """The writer's own per-shot camera, as valid camera names."""
    return [str(s.camera) if _valid(getattr(s, "camera", None)) else "none"
            for s in shots]


def overlay_suggestions(director_cameras: Sequence[str],
                        suggestions: Sequence[str],
                        max_movers: Optional[int] = None) -> List[str]:
    """Keep the writer's move where the director declined and it is valid.

    Pure function: the hybrid policy in one place, testable without CLIP. The
    overlay must LIVE INSIDE THE DIRECTOR'S MOVER BUDGET -- measured without this,
    hybrid produced 3 moves in a 6-shot film (brief: at most 2). Director moves
    are kept first because the director is the continuity/taste authority; writer
    suggestions then fill the remaining budget in shot order.
    """
    capped: List[str] = list(director_cameras)
    if max_movers is not None:
        budget = max(0, int(max_movers)
                     - sum(1 for c in capped if c != "none"))
        for i, (d, s) in enumerate(zip(director_cameras, suggestions)):
            if budget <= 0:
                break
            # `s != "none"`: "none" is itself a valid camera name, and counting
            # it as a kept suggestion burned the whole budget on silent shots
            # (measured: a move at shot 4 was dropped even when the budget was 2)
            if d == "none" and s != "none" and _valid(s):
                capped[i] = s
                budget -= 1
        return capped
    return [s if d == "none" and s != "none" and _valid(s) else d
            for d, s in zip(director_cameras, suggestions)]


def _director_choices(shots: Sequence[Shot], include_subject: bool,
                      scorer, verbose: bool) -> Tuple[List[str], Dict]:
    from director import FitScorer, choose

    scorer = scorer or FitScorer()
    directed, report = choose(
        list(shots), allow=camera_candidates(include_subject),
        scorer=scorer, mode="global", verbose=verbose)
    cams = [str(s.camera) for s in directed]
    info = {
        "fit_ok": bool(getattr(scorer, "ok", False)),
        "choices": [
            {"index": c.index, "chosen": c.chosen, "runner_up": c.runner_up,
             "fit": round(c.fit, 3), "continuity": round(c.continuity, 3),
             "total": round(c.total, 3), "notes": list(c.notes)[:3]}
            for c in report.choices],
    }
    return cams, info


def decide(shots: Sequence[Shot], mode: str = DEFAULT_MODE,
           include_subject: bool = False, scorer=None,
           verbose: bool = False) -> Tuple[List[str], Dict]:
    """Choose one camera name per shot. Returns `(cameras, info)`.

    Never raises for a missing CLIP/torch: the director is an optimisation, not a
    requirement, and a draft must not fail because the scorer could not load. On
    failure the writer's suggestions are used and `info["fallback"]` says so.
    """
    mode = (mode or DEFAULT_MODE).lower()
    if mode not in MODES:
        raise ValueError(f"unknown camera mode {mode!r}; expected one of {MODES}")
    shots = list(shots)
    suggestions = suggestion_cameras(shots)
    info: Dict = {"mode": mode, "shots": len(shots)}

    if mode == "off":
        info["cameras"] = ["none"] * len(shots)
        return info["cameras"], info
    if mode == "suggestions":
        info["cameras"] = suggestions
        return suggestions, info

    try:
        directed, dinfo = _director_choices(shots, include_subject, scorer, verbose)
        info.update(dinfo)
    except Exception as exc:  # noqa: BLE001 -- a missing scorer must not kill a draft
        info["fallback"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        info["cameras"] = suggestions
        return suggestions, info

    if mode == "director":
        cams = directed
    else:                                     # hybrid
        from director import mover_cap
        cams = overlay_suggestions(directed, suggestions, mover_cap(len(shots)))
        info["overlay_kept"] = sum(
            1 for d, c in zip(directed, cams) if d != c)
    info["cameras"] = cams
    return cams, info


def summary(shots: Sequence[Shot], cameras: Sequence[str]) -> Dict:
    """What a camera assignment does, without rendering it.

    Counts the things the editing rules exist to control: moves, movement inside
    the brief's budget, adjacent opposite moves (R2), and same-axis sign flips
    (R6). `plan()` is the final authority; this is the measurement used by the
    A/B before anything is stored.
    """
    cams = list(cameras)
    movers = sum(1 for c in cams if c != "none")
    reversals = []
    for i in range(1, len(cams)):
        a, b = mv.get(cams[i - 1]), mv.get(cams[i])
        if b.reverses(a):
            reversals.append(i + 1)
    runs, run = [], 0
    for c in cams:
        run = run + 1 if c != "none" else 0
        runs.append(run)
    return {
        "shots": len(cams),
        "movers": movers,
        "mover_share": round(movers / max(1, len(cams)), 2),
        "reversals": reversals,
        "longest_mover_run": max(runs, default=0),
        "cameras": cams,
        "per_shot": [{"index": i + 1, "scale": s.scale, "camera": c}
                     for i, (s, c) in enumerate(zip(shots, cams))],
    }
