"""
edit_planner.py -- continuity rules for a sequence of AnimateDiff shots.

The models are now good enough that the limiting factor is the EDIT. These are
the rules that decide whether a sequence of clips reads as a film or as a pile
of clips, expressed as code so they are enforced rather than remembered.

Rules
  R1  no two consecutive shots at the same framing scale  (that is a jump cut)
  R2  no opposite camera moves back to back               (reads as a mistake)
  R3  no more than MAX_RUN consecutive moving shots       (fatigue)
  R4  shot lengths follow a rhythm curve, never uniform
  R5  hard cuts only                                      (no blending)
  R6  the 180-degree rule: two shots travelling along the same degree of freedom
      must not flip sign across a cut. R2 is the special case of this where the
      move is a camera pan/zoom/tilt/roll; R6 also covers subject motion, so a
      walk toward camera cannot be followed by a walk away without a beat.

The planner rewrites a proposed sequence to satisfy these and REPORTS every
change, so the director can see what it lost. Which motion each shot *should*
use is a separate question, answered in `director.py`; this module only judges
whether the resulting sequence holds together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import motion_vocab as mv

SCALES = ["macro", "close", "medium", "wide", "extreme-wide"]

# the motion vocabulary: camera LoRAs, subject-motion LoRAs, and `none`.
# "none" is a first-class choice, not an absence.
try:
    CAMERAS = mv.available(downloaded_only=False)
except Exception:                                    # pragma: no cover
    CAMERAS = ["none", "zoom-in", "zoom-out", "pan-left", "pan-right",
               "tilt-up", "tilt-down", "rolling-clockwise", "rolling-anticlockwise"]

# which moves read as opposites and should never sit adjacent (kept for R2's
# message; R6 below is the general rule)
OPPOSITE = {
    "zoom-in": "zoom-out", "zoom-out": "zoom-in",
    "pan-left": "pan-right", "pan-right": "pan-left",
    "tilt-up": "tilt-down", "tilt-down": "tilt-up",
    "rolling-clockwise": "rolling-anticlockwise",
    "rolling-anticlockwise": "rolling-clockwise",
}

MAX_MOVING_RUN = 2      # R3
# MIN_SHOT is the writer's floor. MAX_SHOT is only an ABSOLUTE safety cap: the
# real ceiling is what the renderer generated (`handheld.max_hold_seconds`),
# because a hold longer than the source clip is duplicated frames. The earlier
# change to 8.0 s was disconnected from that: 8 s of hold needs a ~63-frame
# source, which is 32 raw frames, not the 16 the pipeline defaults to.
MIN_SHOT, MAX_SHOT = 1.5, 5.0

# ---- the measured human distribution (PACING_MEASURED_STUDY.md) ------------- #
# 700 MSR-VTT clips, 3,185 shots, our own hard-cut detector (F1 0.992):
#   <0.5s 14.0% | 0.5-1s 22.6% | 1-2s 17.8% | 2-3s 12.0% | 3-5s 14.6% | >5s 19.0%
# median 1.73s (p10 0.33, p90 8.37), within-clip CV 0.75.
#
# The policy from that document: a FAST band (~0.8-1.5 s: inserts, reactions,
# action accents) and a LONG band (~4-8 s: holds, establishing, emotional
# landing), with the 2 s band used sparingly. The renderer currently caps a still
# at 1.30x its generated motion and the takes are 16 raw frames (1.94 s), so the
# long band is bounded at ~2.5 s until a shot is generated with more frames;
# accents can go below the old 1.5 s floor by trimming a take (motion gets
# faster, which the study explicitly calls acceptable for accents).
ACCENT_SHARE, MID_SHARE, HOLD_SHARE = 0.40, 0.30, 0.30
ACCENT_BAND = (0.90, 1.45)      # s, trimming the take (faster motion, acceptable)
# The long band is bounded by what was generated, so it scales with the cap:
# 16 raw frames -> ~2.1-2.5 s; 32 raw frames -> ~4.1-5.1 s (the study's hold band).
HOLD_FILL_FRAC = (0.82, 0.99)
MID_FILL_FRAC = (0.60, 0.74)


@dataclass(frozen=True)
class Shot:
    prompt: str
    scale: str = "medium"
    camera: str = "none"          # any motion name: camera LoRA, subject LoRA, none
    seconds: float = 2.5
    subject: str = ""             # who/what the shot is about, for continuity
    scene: str = ""               # structural unit. A model switch or a palette
                                  # change is only legible at a scene boundary.
    energy: Optional[float] = None  # 0-1 intended motion energy; None = infer from scale
    beat: str = ""                 # story function: establish/develop/turn/resolve.
                                   # The film's PROGRESSION lives here, not in the
                                   # motion choice -- see story.py's beat checks.
    purpose: str = ""              # coverage function inside a scene:
                                   # establish/master/insert/reaction/resolve
    link: str = ""                 # how this shot connects to the previous one:
                                   # open/match-action/eyeline/movement/graphic/scene-cut
    movement: str = ""             # the writer's own movement id (m1, m2...): a
                                   # stretch with one job. The sequence director
                                   # respects these when they are consistent.
    camera_move: Optional[dict] = None
    # ^ FINISHING camera: {"move","amount","settle"} applied to the delivered
    # frames by camera_path. Mutually exclusive with a generation-time `camera`
    # LoRA: a generated move cannot settle (measured), so holds move in finishing.
    upsample: int = 2
    # ^ FINISHING RAFT factor: 2 = natural (31 delivered frames for 16 raw),
    # 4 = 61 frames (3.81 s) of interpolated slow motion. Long holds use this
    # instead of another AnimateDiff render ("procedural hold", per the pacing
    # study). It is part of finish_inputs, so raising it re-finishes, never
    # re-generates.
    palette: str = ""
    # ^ The planned GRADE for this shot's scene. Carried on the shot so a director
    # pass that rewrites scene LABELS cannot lose it (measured on the squid draft:
    # planN ids became movement ids and every store scene fell back to neutral).

    @property
    def moving(self) -> bool:
        return self.camera != "none"

    @property
    def motion(self) -> str:
        return self.camera

    @property
    def kind(self) -> str:
        return mv.get(self.camera).kind


@dataclass
class EditReport:
    changes: List[str]
    rhythm: List[float]
    # `notes` are not repairs: they are things the planner could NOT fix, e.g. a
    # target duration the renderer's motion budget cannot reach. Keep them
    # separate from `changes` so "what the planner altered" stays honest.
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    def show(self) -> str:
        lines = ["  planner changes:"]
        lines += [f"    - {c}" for c in self.changes] or ["    (none)"]
        if self.notes:
            lines.append("  planner notes:")
            lines += [f"    ! {n}" for n in self.notes]
        return "\n".join(lines)


# Shot-duration factors RELATIVE TO THE MOTION THE RENDERER GENERATED. This is
# the one timing opinion in the file -- the position-inferred curve below
# delegates here so there cannot be two.
#
# At the production defaults (16 raw frames, 16 fps) the source is 1.94 s. The
# ranges are DELIBERATELY WIDE and bimodal, because the previous narrow bands
# (develop 0.88-1.05, turn 0.78-0.95) produced a metronome: measured on the
# delivered 13-shot film, CV 0.074 against R9's own 0.20 floor. Human short-form
# is bimodal (sub-second accents + multi-second holds); the renderer's motion cap
# bounds the long end at 1.30x (static) / 1.15x (moving), so the rhythm has to be
# made by alternating near-cap holds with near-floor accents, not by stretching.
BEAT_HOLD_FACTORS = {
    "establish": (0.95, 1.22),   # orient the eye: long
    "develop":   (0.72, 1.08),   # advance: mid, alternates long/short
    "turn":      (0.62, 0.92),   # change: a short accent
    "fall":      (0.70, 1.06),   # consequence: mid, alternates
    "resolve":   (1.05, 1.30),   # let it land: the longest shot of the film
    "":          (0.80, 1.10),   # no beat: mid
}

# Holds land here as a fraction of the renderer's cap; accents at the union floor.
HOLD_FILL = (0.93, 1.00)     # near-cap, varied
ACCENT_SPREAD = 0.18         # seconds above MIN_SHOT for accents


def _infer_beat(i: int, n: int) -> str:
    """Position fallback for callers with no beat data. Never overrides a beat."""
    if n <= 1:
        return "establish"
    t = i / (n - 1)
    if t < 0.3:
        return "establish"
    if t < 0.7:
        return "develop"
    if t < 0.9:
        return "turn"
    return "resolve"


def _floor2(value: float) -> float:
    """Round DOWN to 0.01 s, so a capped hold can never sit a hundredth above
    the renderer's ceiling (2.51875 rounds to 2.52 and fails the budget test)."""
    return math.floor(max(0.0, float(value)) * 100.0) / 100.0


def _curve_runs(shots: List[Shot]) -> List[List[int]]:
    """Contiguous scene runs (by the writer's scene label), as index lists."""
    runs: List[List[int]] = []
    for i, shot in enumerate(shots):
        scene = getattr(shot, "scene", "") or ""
        if runs and (getattr(shots[runs[-1][0]], "scene", "") or "") == scene:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def _role_for(shot: Shot, i: int, scene_first: bool) -> str:
    """Hold (near the motion cap) or accent (near the floor) inside a scene."""
    if scene_first:
        return "hold"                     # the opening shot orients the eye
    beat = getattr(shot, "beat", "") or ""
    purpose = getattr(shot, "purpose", "") or ""
    scale = getattr(shot, "scale", "") or ""
    if beat in ("establish", "resolve") or purpose in ("establish", "resolve"):
        return "hold"
    if purpose in ("insert", "reaction") or scale in ("close", "macro"):
        return "accent"
    if beat == "turn":
        return "accent"
    # develop/fall: alternate, which is what makes the film breathe
    return "hold" if i % 2 == 0 else "accent"


def _role_score(shot: Shot, scene_first: bool) -> int:
    """3 = should be a long hold, 1 = mid, 0 = fast accent."""
    if scene_first:
        return 3
    beat = getattr(shot, "beat", "") or ""
    purpose = getattr(shot, "purpose", "") or ""
    scale = getattr(shot, "scale", "") or ""
    if beat in ("establish", "resolve") or purpose in ("establish", "resolve"):
        return 3
    if beat == "turn" or purpose in ("insert", "reaction") or scale in ("close",
                                                                       "macro"):
        return 0
    if purpose == "master" or scale in ("wide", "extreme-wide"):
        return 2
    return 1


def _study_curve(shots: List[Shot], caps: List[float],
                 frames: int = 16) -> List[float]:
    """Assign durations from the measured HUMAN distribution, not a pattern.

    The first rhythm pass alternated 2.5/1.5/2.5 -- better than uniform but still
    a metronome, and it ignored the study's actual shape. This assigns each shot
    one of the study's bands by its ROLE (accents where the film needs them,
    holds where it breathes), samples within the band deterministically, then
    repairs R9 (scene opening longest, no triple/adjacent near-equal).
    """
    n = len(shots)
    scene_first = set(r[0] for r in _curve_runs(shots))
    scores = [_role_score(s, i in scene_first) for i, s in enumerate(shots)]
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    n_hold = max(1, round(n * HOLD_SHARE)) if n else 0
    n_mid = max(1, round(n * MID_SHARE)) if n >= 3 else 0
    n_hold = min(n, n_hold)
    n_mid = min(n - n_hold, n_mid)
    roles = ["accent"] * n
    for k, i in enumerate(order[:n_hold]):
        roles[i] = "hold"
    for i in order[n_hold:n_hold + n_mid]:
        roles[i] = "mid"
    # a scene opening is always a hold (R9); steal a hold from the lowest scorer
    for i in sorted(scene_first):
        if roles[i] != "hold":
            donors = [j for j in range(n) if roles[j] == "hold"
                      and j not in scene_first]
            if donors:
                roles[max(donors, key=lambda j: scores[j] * -1)] = "accent"
                roles[i] = "hold"
    out: List[float] = []
    for i, shot in enumerate(shots):
        cap = caps[i]
        variation = (i * 0.6180339887) % 1.0
        role = roles[i]
        if role == "hold":
            lo, hi = HOLD_FILL_FRAC
            v = cap * (lo + (hi - lo) * variation)
        elif role == "mid":
            lo, hi = MID_FILL_FRAC
            v = cap * (lo + (hi - lo) * variation)
        else:
            lo, hi = ACCENT_BAND
            v = lo + (hi - lo) * variation
        # the accent floor is 0.90s by trimming; the writer's 1.5s floor applies
        # to WRITTEN holds, not to a deliberately fast cut
        out.append(_floor2(min(cap, max(ACCENT_BAND[0], v))))
    return out


def _enforce_rhythm(values: List[float], shots: List[Shot],
                    caps: List[float], floor: float = MIN_SHOT) -> List[float]:
    """Make the curve obey R9: scene opening longest, no triple near-equal,
    no adjacent near-equal. The planner must not write a metronome and then let
    the story gate complain about it. `floor` is the accent floor: 1.5 s for a
    written hold, 0.90 s for a deliberately fast trim.
    """
    out = [round(v, 2) for v in values]

    def openings_longest() -> None:
        for run in _curve_runs(shots):
            if len(run) < 2:
                continue
            a, rest = run[0], run[1:]
            if out[a] < max(out[k] for k in rest) - 0.05:
                out[a] = round(min(caps[a], max(out[k] for k in rest) + 0.12), 2)
            if out[a] < max(out[k] for k in rest) - 0.05:
                for k in rest:
                    if out[k] > out[a] - 0.06:
                        out[k] = max(floor, round(out[a] - 0.08, 2))

    openings_longest()
    for _ in range(4):
        changed = False
        for i in range(1, len(out) - 1):
            if (abs(out[i] - out[i - 1]) <= 0.1
                    and abs(out[i] - out[i + 1]) <= 0.1):
                up = min(caps[i], round(out[i] + 0.35, 2))
                down = max(floor, round(out[i] - 0.35, 2))
                out[i] = (up if abs(up - out[i - 1]) > abs(down - out[i - 1])
                          else down)
                changed = True
        if not changed:
            break
    for i in range(1, len(out)):
        if abs(out[i] - out[i - 1]) < 0.12:
            mid = (floor + caps[i]) / 2.0
            out[i] = (round(min(caps[i], out[i - 1] + 0.40), 2)
                      if out[i - 1] < mid
                      else round(max(floor, out[i - 1] - 0.40), 2))
    # the adjacent pass can lower a scene opening that the first pass fixed
    openings_longest()
    return [_floor2(min(caps[i], max(floor, v))) for i, v in enumerate(out)]


def _rhythm_curve_beats(shots: List[Shot], total: Optional[float] = None,
                        frames: int = 16, fps: int = 16) -> List[float]:
    """R4: bimodal, beat-shaped holds, clamped to the motion the renderer made.

    `frames` is the raw-frame setting (`settings.frames`), not a constant: the
    generated motion is `2*frames-1` delivered frames. A static shot may be held
    up to 1.30x that; a camera move only 1.15x, because duplicating motion reads
    as judder where duplicating a still does not.

    MEASURED FIX. The delivered 13-shot film had CV 0.074 -- the old narrow bands
    made every hold ~1.5-2.4 s in a smooth curve, which is a metronome with a
    bend in it. A film needs ACCENTS: shots near the floor (inserts, reactions,
    turns) alternating with shots near the renderer's ceiling (scene openings,
    establishes, resolves). This assigns by role, alternates develop/fall, and
    repairs R9 itself (scene opening longest, no triple/adjacent near-equal)
    instead of leaving the story gate to complain.

    `total` scales toward a target duration but is then RE-CLAMPED: the target is
    a wish, the motion budget is a fact. `plan()` reports the shortfall in
    `EditReport.notes` rather than silently duplicating frames to fake it.
    """
    from handheld import max_hold_seconds

    caps: List[float] = []
    for shot in shots:
        caps.append(max_hold_seconds(
            frames, fps,
            moving=bool(getattr(shot, "camera", "none") not in ("none", "", None)
                       or getattr(shot, "camera_move", None)),
            upsample=int(getattr(shot, "upsample", 2) or 2)))
    out = _study_curve(shots, caps, frames)
    if total:
        for _ in range(6):
            current = sum(out)
            if abs(current - total) <= 0.04 * total:
                break
            k = total / max(current, 1e-9)
            scaled = []
            for i, v in enumerate(out):
                if v <= ACCENT_BAND[1]:
                    # accents are a deliberate fast band: do not stretch them
                    scaled.append(v)
                else:
                    scaled.append(_floor2(min(caps[i],
                                              max(ACCENT_BAND[0], v * k))))
            if abs(sum(scaled) - total) >= abs(current - total):
                break
            out = scaled
    return _enforce_rhythm(out, shots, caps, floor=ACCENT_BAND[0])


def rhythm_curve(shots: List[Shot], total: Optional[float] = None,
                 frames: int = 16, fps: int = 16) -> List[float]:
    """Public R4 curve for retiming an already-adopted production."""
    return _rhythm_curve_beats(shots, total=total, frames=frames, fps=fps)


def _rhythm_curve(n: int, total: Optional[float] = None,
                  frames: int = 16, fps: int = 16) -> List[float]:
    """R4 for callers that only know a count. Delegates -- one definition."""
    demo = [Shot(f"shot{i + 1}", beat=_infer_beat(i, n)) for i in range(n)]
    return _rhythm_curve_beats(demo, total, frames, fps)


def plan(shots: List[Shot], keep_director_timing: bool = False,
         total: Optional[float] = None, frames: int = 16,
         fps: int = 16) -> Tuple[List[Shot], EditReport]:
    """Apply R1-R4, reporting each change and what could not be fixed.

    `total` scales the rhythm curve so the film lands on a target duration --
    which is how a platform length (TikTok 30s, Reels 25s) is imposed without
    making every shot the same length, which R4 forbids.

    `frames` / `fps` are the RENDERER'S settings (`settings.frames`,
    `settings.fps`), not constants. R4's holds are clamped to the motion those
    frames actually generated (`handheld.max_hold_seconds`): a longer hold is
    duplicated frames, which plays as slow-motion judder. When `total` asks for
    more duration than that budget can supply, the shortfall is reported in
    `EditReport.notes` instead of being faked by stretching.

    R4 uses the writer's beat (establish/develop/turn/resolve) as the shape.
    """
    changes: List[str] = []
    notes: List[str] = []
    out: List[Shot] = []

    for i, s in enumerate(shots):
        if s.scale not in SCALES:
            changes.append(f"shot {i+1}: unknown scale {s.scale!r} -> medium")
            s = replace(s, scale="medium")
        if s.camera not in CAMERAS:
            changes.append(f"shot {i+1}: unknown camera {s.camera!r} -> none")
            s = replace(s, camera="none")

        # R1: same framing as the previous shot
        if out and s.scale == out[-1].scale:
            j = SCALES.index(s.scale)
            # step outward if we can, else inward
            new = SCALES[j + 1] if j + 1 < len(SCALES) else SCALES[j - 1]
            changes.append(f"shot {i+1}: R1 same scale as shot {i} ({s.scale}) "
                           f"-> {new}")
            s = replace(s, scale=new)

        # R2/R6: opposite move back to back. R6 generalises R2 across every
        # degree of freedom, including subject motion.
        if out:
            prev = mv.get(out[-1].camera)
            cur = mv.get(s.camera)
            if cur.reverses(prev):
                changes.append(
                    f"shot {i+1}: R6 {cur.name} reverses shot {i}'s {prev.name} "
                    f"along the {cur.axis} axis -> none")
                s = replace(s, camera="none")

        # R3: too many moving shots in a row
        run = 0
        for p in reversed(out):
            if p.moving:
                run += 1
            else:
                break
        if s.moving and run >= MAX_MOVING_RUN:
            changes.append(f"shot {i+1}: R3 {run} moving shots in a row -> none")
            s = replace(s, camera="none")

        out.append(s)

    # R10: a camera move must hand off to the SAME scene. A move that is the last
    # thing seen of a place before a hard cut to new scenery has no continuation
    # and no hold -- it reads as a stray pan. Rewriting it to 'none' is the safe
    # repair; the alternative (requiring the next shot to continue the move) is a
    # writer decision, not something the planner can invent.
    for i in range(len(out) - 1):
        cur, nxt = out[i], out[i + 1]
        if (cur.moving and cur.scene and nxt.scene
                and nxt.scene != cur.scene):
            changes.append(
                f"shot {i+1}: R10 {cur.camera} hands off to a new scene "
                f"({cur.scene}->{nxt.scene}) -> none")
            out[i] = replace(out[i], camera="none")

    # R4: rhythm -- beat-shaped, and bounded by the generated motion
    from handheld import max_hold_seconds, natural_seconds, stretch_ratio

    if not keep_director_timing:
        curve = _rhythm_curve_beats(out, total, frames, fps)
        for i, (s, d) in enumerate(zip(out, curve)):
            if abs(s.seconds - d) > 0.05:
                why = f"beat: {s.beat or 'inferred'}"
                cap = max_hold_seconds(frames, fps, moving=s.moving)
                if abs(d - cap) < 0.005:
                    why += f"; renderer cap {cap:.2f}s at {frames} frames"
                changes.append(f"shot {i+1}: R4 hold {s.seconds}s -> {d}s ({why})")
            out[i] = replace(s, seconds=d)
        if total:
            achieved = sum(s.seconds for s in out)
            if achieved < total * 0.9:
                cap = max_hold_seconds(frames, fps)
                need = math.ceil(total / cap)
                notes.append(
                    f"target {total:.1f}s is NOT reachable with {len(out)} shots "
                    f"at {frames} raw frames: each supplies "
                    f"{natural_seconds(frames, fps):.2f}s of motion and the hold "
                    f"ceiling is {cap:.2f}s, so the most this list can deliver is "
                    f"{len(out) * cap:.1f}s (planned {achieved:.1f}s). Ask for "
                    f"~{need} shots, raise the raw-frame setting, or accept the "
                    f"shorter piece.")
    else:
        # the director already set the holds; measure, do not rewrite
        for i, s in enumerate(out):
            cap = max_hold_seconds(frames, fps, moving=s.moving)
            if s.seconds > cap + 0.01:
                notes.append(
                    f"shot {i+1}: {s.seconds}s hold is "
                    f"{stretch_ratio(s.seconds, frames, fps):.2f}x the "
                    f"{frames}-frame source (ceiling {cap:.2f}s) -- it will judder")

    # R5 is structural: there is no transition field, so cuts are the only option
    return out, EditReport(changes, [s.seconds for s in out], notes)


def describe(shots: List[Shot]) -> str:
    lines = [f"  {'#':>2} {'scale':<13} {'camera':<22} {'hold':>5}  prompt"]
    for i, s in enumerate(shots, 1):
        mark = "MOVE" if s.moving else "    "
        lines.append(f"  {i:>2} {s.scale:<13} {s.camera:<22} {s.seconds:>5.2f}  "
                     f"{mark} {s.prompt[:46]}")
    return "\n".join(lines)


def energy_profile(shots: List[Shot]) -> str:
    """A tiny visual of the motion rhythm -- is it exhausting or varied?"""
    marks = {"none": ".", "camera": "C", "subject": "S"}
    bar = "".join(marks.get(s.kind, "?") for s in shots)
    n_move = sum(s.moving for s in shots)
    n_subj = sum(s.kind == "subject" for s in shots)
    return (f"  motion energy: {bar}   "
            f"({n_move}/{len(shots)} move, {n_subj} carry a subject; "
            f"C=camera  S=subject  .=still)")


if __name__ == "__main__":
    demo = [
        Shot("dense forest, light shafts", "wide", "zoom-in", 2.5),
        Shot("dew on a leaf", "macro", "zoom-in", 2.5),          # R1 same? no
        Shot("waterfall", "wide", "zoom-out", 2.5),              # R2 vs zoom-in
        Shot("mossy rocks", "close", "pan-left", 2.5),
        Shot("the river", "medium", "pan-right", 2.5),           # R2 vs pan-left
        Shot("trees", "wide", "tilt-up", 2.5),
    ]
    planned, report = plan(demo)
    print("PROPOSED:")
    print(describe(demo))
    print("\nPLANNED:")
    print(describe(planned))
    print("\n" + report.show())
    print(energy_profile(planned))
