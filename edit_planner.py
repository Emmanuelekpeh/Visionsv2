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
MIN_SHOT, MAX_SHOT = 1.5, 5.0


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

    def show(self) -> str:
        lines = ["  planner changes:"]
        lines += [f"    - {c}" for c in self.changes] or ["    (none)"]
        return "\n".join(lines)


def _rhythm_curve(n: int, total: Optional[float] = None) -> List[float]:
    """R4: long -> shortening -> hold. Never uniform."""
    if n == 1:
        return [2.5]
    if n == 2:
        return [2.5, 3.5]
    # build toward a peak around 70% in, then let the last shot breathe
    out = []
    for i in range(n):
        t = i / (n - 1)
        if t < 0.7:
            # 3.2s down to 1.6s
            v = 3.2 + (1.6 - 3.2) * (t / 0.7)
        else:
            v = 2.6 + (3.6 - 2.6) * ((t - 0.7) / 0.3)
        out.append(round(min(MAX_SHOT, max(MIN_SHOT, v)), 2))
    if total:
        k = total / sum(out)
        out = [round(min(MAX_SHOT, max(MIN_SHOT, v * k)), 2) for v in out]
    return out


def plan(shots: List[Shot], keep_director_timing: bool = False,
         total: Optional[float] = None) -> Tuple[List[Shot], EditReport]:
    """Apply R1-R4, reporting each change.

    `total` scales the rhythm curve so the film lands on a target duration --
    which is how a platform length (TikTok 30s, Reels 25s) is imposed without
    making every shot the same length, which R4 forbids.
    """
    changes: List[str] = []
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

    # R4: rhythm
    if not keep_director_timing:
        curve = _rhythm_curve(len(out), total)
        for i, (s, d) in enumerate(zip(out, curve)):
            if abs(s.seconds - d) > 0.05:
                changes.append(f"shot {i+1}: R4 hold {s.seconds}s -> {d}s")
            out[i] = replace(s, seconds=d)

    # R5 is structural: there is no transition field, so cuts are the only option
    return out, EditReport(changes, [s.seconds for s in out])


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
