"""
motion_vocab.py -- the vocabulary the director chooses from.

`edit_planner.py` enforces continuity on a sequence that a human proposed.
This module is the layer above: what a motion *is*, what it is *for*, and where
it points -- so a machine can pick one per shot instead of a person guessing.

A motion is described on four axes:

  kind        none | camera | subject
              camera moves move the frame; subject LoRAs move something inside it
  axis        the degree of freedom it consumes
  sign        which way along that axis. This is what the 180-degree rule is
              about: two shots that both travel along `lateral` must not flip
              sign across a cut, or the film reads as a mistake.
  descriptor  plain English, embedded with CLIP and matched against the shot's
              prompt. This is how `fit` is computed rather than hand-assigned.

`subject_required` is the hard gate: a walking LoRA on a shot with no person in
it is how you get a figure wandering through a leaf macro. That failure mode is
measured, not assumed -- see the `leaf_walk_512` clip in subject_motion_test.py.

Measured fields (`energy`, `loc10`, `coherence`) are filled in from the
evidence file produced by subject_motion_test.py. Until they are, the director
treats a motion as unproven and will not give it a subject-carrying shot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
EVIDENCE_FILE = ROOT / "output" / "subject_test" / "metrics.json"


@dataclass
class Motion:
    name: str
    kind: str                       # none | camera | subject
    axis: str                       # none | depth | lateral | vertical | rotate | inplace
    sign: int                       # -1, 0, +1 along the axis
    descriptor: str                 # CLIP text used for fit
    subject_required: bool = False
    # which framing scales this motion can live in. A full-body walking LoRA has
    # no business in a macro or a face close-up -- it would try to draw a whole
    # person into a shot that has no room for one.
    scales: tuple = ("macro", "close", "medium", "wide", "extreme-wide")
    energy: Optional[float] = None      # measured mean frame-to-frame change
    loc10: Optional[float] = None       # measured top-10% variance share
    # ^ CONFOUNDED BY MAGNITUDE. A near-static clip scores high simply because its
    #   small change is concentrated. Kept for the record; do NOT gate on it.
    global_share: Optional[float] = None  # spatial uniformity of change. The
    #   metric that actually answers "is the background moving too". Lower =
    #   more concentrated = a separable subject.
    separable: Optional[bool] = None    # global_share vs the baseline. Scene
    #   dependent, so treat as information, never as a blanket gate.
    coherence: Optional[float] = None   # measured CLIP drift (lower = steadier)
    align: Optional[float] = None
    measured: bool = False

    @property
    def moving(self) -> bool:
        return self.kind != "none"

    @property
    def full_body(self) -> bool:
        return self.kind == "subject"

    def fits_scale(self, scale: str) -> bool:
        return scale in self.scales

    def same_dof_as(self, other: "Motion") -> bool:
        """Same degree of freedom AND same kind -- a subject walking toward the
        camera and the camera pulling back are not the same move, and combining
        them is legitimate rather than a reversed cut."""
        return (self.kind == other.kind and self.axis != "none"
                and self.axis == other.axis)

    def reverses(self, other: "Motion") -> bool:
        """The 180-degree violation: same degree of freedom, opposite sign."""
        return self.same_dof_as(other) and self.sign * other.sign < 0


# --------------------------------------------------------------------------- #
# the vocabulary
# --------------------------------------------------------------------------- #

def _camera(name, axis, sign, desc):
    return Motion(name, "camera", axis, sign, desc)


MOTIONS: Dict[str, Motion] = {
    "none": Motion("none", "none", "none", 0,
                   "a completely static camera, nothing moves in the frame"),

    # ---- official camera LoRAs (downloaded) -------------------------------- #
    "zoom-in":  _camera("zoom-in", "depth", +1,
                        "the camera pushes in toward the subject, which grows larger"),
    "zoom-out": _camera("zoom-out", "depth", -1,
                        "the camera pulls back away from the subject, revealing more"),
    "pan-left": _camera("pan-left", "lateral", -1,
                        "the camera pans left, sweeping across the scene"),
    "pan-right": _camera("pan-right", "lateral", +1,
                         "the camera pans right, sweeping across the scene"),
    "tilt-up":  _camera("tilt-up", "vertical", +1,
                        "the camera tilts up toward the sky"),
    "tilt-down": _camera("tilt-down", "vertical", -1,
                         "the camera tilts down toward the ground"),
    "rolling-clockwise": _camera("rolling-clockwise", "rotate", +1,
                                 "the camera rolls clockwise, the horizon tilting"),
    "rolling-anticlockwise": _camera("rolling-anticlockwise", "rotate", -1,
                                     "the camera rolls anticlockwise, the horizon tilting"),

    # ---- translation axes, NOT downloaded yet ------------------------------ #
    # official `pan` rotates; these translate. Different physical move, and the
    # handoff listed them as the next vocabulary gap.
    "camera-lateral_left": _camera("camera-lateral_left", "lateral", -1,
                                   "the camera trucks sideways to the left"),
    "camera-lateral_right": _camera("camera-lateral_right", "lateral", +1,
                                    "the camera trucks sideways to the right"),
    "camera-crane_up": _camera("camera-crane_up", "vertical", +1,
                               "the camera booms upward, rising through the scene"),
    "camera-crane_down": _camera("camera-crane_down", "vertical", -1,
                                 "the camera booms downward, descending"),

    # ---- subject-motion LoRAs (downloaded, under test) --------------------- #
    # all of these draw a full standing body, so they are restricted to the
    # framings where a full body fits. This is the rule that stops the director
    # putting a walking LoRA on a close-up of a face.
    "walking-frontview": Motion(
        "walking-frontview", "subject", "depth", +1,
        "a person walking toward the camera, full body, legs stepping, "
        "arms swinging", subject_required=True,
        scales=("medium", "wide", "extreme-wide")),
    "running-frontview": Motion(
        "running-frontview", "subject", "depth", +1,
        "a person sprinting toward the camera, full body, athletic running "
        "stride, arms pumping", subject_required=True,
        scales=("medium", "wide", "extreme-wide")),
    "ballerina-spin": Motion(
        "ballerina-spin", "subject", "inplace", 0,
        "a dancer spinning in place, body rotating, fabric swirling",
        subject_required=True, scales=("medium", "wide", "extreme-wide")),
    "casualmove-frontview": Motion(
        "casualmove-frontview", "subject", "depth", +1,
        "a person moving casually toward the camera at an easy walking pace",
        subject_required=True, scales=("medium", "wide", "extreme-wide")),
    "walk-backview": Motion(
        "walk-backview", "subject", "depth", -1,
        "a person walking away from the camera, seen from behind, receding",
        subject_required=True, scales=("medium", "wide", "extreme-wide")),
    "drone-forward": Motion(
        # an aerial move is a camera move that happens to fly, not a subject
        # performance -- it needs no person in frame
        "drone-forward", "camera", "depth", +1,
        "an aerial drone flying forward over the landscape, rising terrain",
        scales=("wide", "extreme-wide")),
}


def get(name: str) -> Motion:
    return MOTIONS.get(name, MOTIONS["none"])


# --------------------------------------------------------------------------- #
# evidence: measured behaviour overrides belief
# --------------------------------------------------------------------------- #

# which measured clip informs which motion
EVIDENCE_MAP = {
    "walking-frontview": ["fig_walk_512", "fig_walk_576x384"],
    "running-frontview": ["fig_run_512"],
    "ballerina-spin": ["fig_spin_512"],
    "none": ["fig_none_512"],
}

# localization floor for a motion to be trusted with a subject-carrying shot.
#
# LOC10_TRUST IS DEPRECATED and must not be gated on -- loc10 is confounded by
# magnitude (see the Motion dataclass). Kept only so old code does not silently
# change meaning. Use global_share, which is magnitude-independent.
LOC10_TRUST = 0.30

# Separate when the change field is this much more concentrated than the
# baseline's. 0.7 = a 30% reduction. Nothing measured on this machine has come
# below 0.86; the author's reference render sat at 0.18/0.36 = 0.5.
LOC_GLOBAL_RATIO = 0.7

# the no-LoRA baseline's measured global_share, filled in by load_evidence
BASELINE: Dict[str, float] = {}


def load_evidence(path: Path = EVIDENCE_FILE) -> Dict[str, dict]:
    """Fold measured clip metrics back into the vocabulary."""
    if not Path(path).exists():
        return {}
    try:
        store = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    clips = store.get("clips", {})
    applied = {}
    base_clip = clips.get("fig_none_512")
    if base_clip and base_clip.get("global_share"):
        BASELINE["global_share"] = float(base_clip["global_share"])
        BASELINE["clip"] = base_clip.get("tag", "fig_none_512")
    for motion, ids in EVIDENCE_MAP.items():
        found = [clips[i] for i in ids if i in clips]
        if not found:
            continue
        m = MOTIONS[motion]
        m.energy = round(sum(f["mag"] for f in found) / len(found), 3)
        m.loc10 = round(sum(f["loc10"] for f in found) / len(found), 4)
        m.coherence = round(sum(f.get("coherence", 0) for f in found) / len(found), 4)
        m.align = round(sum(f.get("align", 0) for f in found) / len(found), 4)
        m.global_share = round(
            sum(f.get("global_share", 0) for f in found) / len(found), 4)
        m.measured = True
        # Separability is measured against the no-LoRA baseline's global_share.
        # NOTE: it is SCENE-DEPENDENT -- every measurement here is on a textured
        # scene, where the background moves. A shot framed against fog or flat sky
        # may well separate where these did not, so this is information, not a
        # verdict, and it must not become a blanket refusal. That would repeat the
        # watermark-gate mistake (handoff 9.7.2).
        if m.global_share and BASELINE.get("global_share"):
            m.separable = bool(
                m.global_share / BASELINE["global_share"] <= LOC_GLOBAL_RATIO)
        applied[motion] = {"clips": [f["tag"] for f in found],
                           "energy": m.energy, "loc10": m.loc10,
                           "global_share": m.global_share,
                           "separable": m.separable}
    return applied


def subject_motions() -> List[str]:
    return [n for n, m in MOTIONS.items() if m.kind == "subject"]


def camera_motions() -> List[str]:
    return [n for n, m in MOTIONS.items() if m.kind == "camera"]


def available(downloaded_only: bool = True) -> List[str]:
    """Filter to what is actually on disk."""
    from motion_loras import CAMERA_DIR, SUBJECT_DIR
    out = ["none"]
    for n, m in MOTIONS.items():
        if n == "none":
            continue
        if m.kind == "camera" and (CAMERA_DIR / n).exists():
            out.append(n)
        elif m.kind == "subject" and (SUBJECT_DIR / f"{n}.safetensors").exists():
            out.append(n)
        elif not downloaded_only:
            out.append(n)
    return out


def describe_motion(name: str) -> str:
    m = get(name)
    ev = (f"energy={m.energy} loc10={m.loc10} coher={m.coherence}"
          if m.measured else "unmeasured")
    return (f"{m.name:<22} {m.kind:<8} {m.axis:<8} {m.sign:+d}  {ev}")


if __name__ == "__main__":
    applied = load_evidence()
    print(f"available (on disk): {available()}\n")
    for n in available():
        print("  " + describe_motion(n))
    if applied:
        print("\nevidence applied:")
        for k, v in applied.items():
            print(f"  {k:<22} from {v['clips']}  energy={v['energy']} loc10={v['loc10']}")
    else:
        print("\nno measured evidence yet -- subject motions are UNPROVEN")
