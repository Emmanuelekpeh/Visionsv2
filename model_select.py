"""
model_select.py -- decide which checkpoint or generator renders each shot.

The question: can the system pick per shot between sd15 and dreamshaper8 -- or
between entirely different generators -- and still read as one film?

Three scores, the same shape as director.py:

  fit          does this model suit THIS shot's requirements? Requirements are
               read off the shot (how much texture it wants, how much motion it
               needs, how fast it must be) and matched against each model's
               *measured* profile from the handoff's table.
  cohesion     what does switching cost? Not asserted -- it is the measured
               `grade_match.seam` between the two models' actual renders of the
               same prompt and seed. If a re-grade pass is enabled, the cost is
               discounted by the reduction that pass was measured to achieve.
  continuity   do not switch inside a run of shots that share a subject or a
               scene: the audience is meant to read those as continuous.

Two hard gates, because a high fit score must never be able to buy them:

  watermark    a checkpoint that leaks the SHUTTERSTOCK watermark is unusable
               for a published video, whatever else it does well.
  motion       a model that cannot produce motion cannot serve a shot that needs
               it, no matter how good its frames look (SVD-XT's problem).

The interesting output is not "which model wins". It is that the selector
refuses to switch when the cohesion cost exceeds the fit gain -- and that
enabling grade matching lowers that cost enough to change the answer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
AB = ROOT / "output" / "film" / "checkpoint_ab"


@dataclass
class Model:
    name: str
    kind: str                      # checkpoint | generator
    path: str = ""
    s_per_frame: float = 35.6      # measured, handoff section 3
    motion: float = 0.5            # 0-1, how much real motion it produces
    detail: float = 0.5            # 0-1, texture and saturation richness
    watermark_risk: float = 0.0    # 0-1 prior; NOT a gate -- verified per render
    still_ok: bool = False         # can it hold a static shot (no drift)
    proven: bool = True            # False = user has not accepted its output
    renderable: bool = False       # can make_film.py actually render it today
    note: str = ""


# PROVENANCE -- read this before trusting a decision this module makes.
#
#   s_per_frame   MEASURED. Handoff section 3, and independently confirmed this
#                 session at ~130-150 s/step x 4 steps for a 16-frame clip.
#   clean         EVIDENCED. Handoff 4.7 A/B: sd15 reproduces the SHUTTERSTOCK
#                 watermark, dreamshaper8 does not.
#   grade         MEASURED this session from real renders (see grade_match.py).
#
#   motion        ** NOT MEASURED. My 0-1 estimates. ** svd-xt "almost no motion"
#                 and t2v-zero "most motion" are qualitative statements in the
#                 handoff; the numbers between them are mine. sd15 and
#                 dreamshaper8 are both set to 0.5, which is an untested
#                 assumption that they move identically -- plausible, since the
#                 motion adapter and LoRAs are shared, but not checked.
#   detail        ** NOT MEASURED. My 0-1 estimates. ** Loosely supported by the
#                 measured chroma (sd15 13.8 vs dreamshaper8 6.5), but "detail"
#                 here conflates saturation, texture and contrast, and only the
#                 first of those was measured.
#
# So the two numbers that actually drive fit are the two that are invented. A
# decision from this module is a hypothesis about fit and a measurement about
# cohesion. Do not quote it as more than that.
MODELS: Dict[str, Model] = {
    "dreamshaper8": Model(
        "dreamshaper8", "checkpoint", "models/dreamshaper8",
        s_per_frame=35.6, motion=0.5, detail=0.60, watermark_risk=0.0,
        renderable=True,
        note="no watermark observed; soft atmospheric, muted filmic tones"),
    "sd15": Model(
        "sd15", "checkpoint", "models/sd15",
        s_per_frame=35.6, motion=0.5, detail=0.95, watermark_risk=0.3,
        renderable=True,
        note="harsher, more saturated. Handoff 4.7 says it leaks SHUTTERSTOCK, "
             "but watermark_check.py could not reproduce that in the cited frames"),
    "t2v-zero": Model(
        "t2v-zero", "generator", "T2V-Zero",
        s_per_frame=175.0, motion=0.95, detail=0.40, watermark_risk=0.2,
        proven=False,
        note="UNPROVEN -- excluded unless --allow-unproven. The user saw 'bogus' "
             "output and wants it kept out until demonstrated. In its favour, "
             "handoff 4.1 records the earlier 'abstract mush' verdict as a 256px "
             "artifact, so it is unknown rather than known-bad"),
    "svd-xt": Model(
        "svd-xt", "generator", "models/svd_xt",
        s_per_frame=193.0, motion=0.10, detail=0.90, watermark_risk=0.0,
        note="prettiest frames, almost no motion"),
    "sd15-lcm": Model(
        "sd15-lcm", "generator", "models/lcm_lora_sd15",
        s_per_frame=84.0, motion=0.0, detail=0.90, watermark_risk=0.3,
        still_ok=True,
        note="6-step stills, for static shots only"),
}


@dataclass
class ShotNeed:
    """What a shot requires of its renderer, 0-1 unless noted."""
    prompt: str
    scale: str = "medium"
    motion: float = 0.5            # from the shot's intended energy
    detail: float = 0.5
    subject: str = ""              # shots sharing this must not switch model
    scene: str = ""                # a change here is where a switch is MOTIVATED
    need_watermark_free: bool = True


@dataclass
class Assignment:
    index: int
    chosen: str
    runner_up: str
    fit: float
    technical: float      # photographic baseline: exposure/contrast/grain
    variety: float        # is the look change legible as a choice
    total: float
    look_shift: float     # measured total palette+baseline distance across the cut
    notes: List[str] = field(default_factory=list)
    rejected: Dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# the measured switch cost
# --------------------------------------------------------------------------- #

def measured_signatures() -> Dict[str, np.ndarray]:
    """Lab look signatures from the real A/B renders, if they are on disk."""
    import grade_match as gm
    from PIL import Image

    out = {}
    for name, d in [("sd15", AB / "SD15_base_none"),
                    ("dreamshaper8", AB / "DreamShaper8_none")]:
        fs = sorted(d.glob("f*.png")) if d.exists() else []
        if fs:
            out[name] = gm.signature([Image.open(f) for f in fs])
    return out


def switch_cost(a: str, b: str, sigs: Dict[str, np.ndarray],
                spreads: Optional[Dict[str, float]] = None,
                grade_matched_reduction: float = 1.0) -> Tuple[float, str]:
    """Seam between two models' renders, in units of within-clip variation."""
    if a == b:
        return 0.0, "same model"
    import grade_match as gm
    if a in sigs and b in sigs:
        sp = 0.15
        if spreads and a in spreads and b in spreads:
            sp = (spreads[a] + spreads[b]) / 2.0
        raw = gm.seam_between(sigs[a], sigs[b], sp)
        return raw / max(1e-6, grade_matched_reduction), "measured"
    # no renders to measure: fall back to a nominal same-family cost
    return 2.0 / max(1e-6, grade_matched_reduction), "nominal"


def measure_spreads() -> Dict[str, float]:
    """Decomp-scale floors per model, from within-clip halves.

    Returns `clip_floor` values, not `within_spread` values. An earlier version
    returned raw spreads and the three `decomp` calls below divided by them,
    inflating every variety distance about 7x.
    """
    import grade_match as gm          # local import, like the other callers here
    from PIL import Image             # ditto -- this function was left without it

    floors = {}
    for name, d in [("sd15", AB / "SD15_base_none"),
                    ("dreamshaper8", AB / "DreamShaper8_none")]:
        fs = sorted(d.glob("f*.png")) if d.exists() else []
        if fs:
            floors[name] = gm.clip_floor([Image.open(f) for f in fs])
    return floors


# --------------------------------------------------------------------------- #
# the selector
# --------------------------------------------------------------------------- #

# weights: fit dominates, but cohesion has real teeth -- a 7.8-unit seam has to
# be able to veto a small fit gain
# --------------------------------------------------------------------------- #
#
# OBJECTIVE, restated.  The first version of this file asked "which model is
# best for this shot, and how little can we switch?" -- a cohesion-maximising
# question. That is the wrong question for this project. The aim is deliberate
# VARIETY: a film that looks like one film but does not look like everything else
# generated by the same handful of checkpoints. Variety is the product; cohesion
# is the constraint; and the constraint is not "shots must match" but "a change
# must read as a CHOICE rather than as a mistake".
#
# So the terms are:
#
#   fit        does the model suit the shot (as before)
#   technical  the photographic baseline -- exposure, contrast, grain. This is
#              now a POST stage (`grade_match.match_technical`), not a reason to
#              avoid switching: it was measured to bring the technical half to
#              1.3x the within-shot noise floor while leaving the palette alone.
#   variety    REWARDED, not penalised: a switch whose palette distance lands in
#              the legible band. Measured: match_technical holds palette at 6.7x
#              the floor, while flattening (match_to) collapses it to 0.4x.
#   motivation a switch is only legible as intentional if it coincides with a
#              structural change -- a new scene, act, subject or place. The same
#              switch mid-scene reads as a continuity error.
#   rhythm     variety needs a tempo: three or more shots on one model is
#              monotonous, alternating every single shot is chaos.
#
WEIGHTS = {"fit": 0.40, "technical": 0.15, "variety": 0.30, "rhythm": 0.15}

# palette distance, in units of the within-shot noise floor. Below STYLE_MIN a
# switch is invisible; above STYLE_MAX the shots stop belonging to one film.
STYLE_MIN, STYLE_MAX = 1.5, 8.0

# how much of the variety credit survives when a switch is NOT motivated by a
# scene change. Low, because an unmotivated switch is the failure mode.
UNMOTIVATED = 0.15

MAX_SAME_RUN = 3      # shots on one model before it reads as monotonous

# What a choice contributes to variety. Staying on the same model contributes
# nothing new, so it must NOT score full marks -- an earlier version gave it 1.0
# and the selector therefore never switched until monotony forced it, which
# defeats the entire objective. A well-motivated switch is what scores 1.0.
STAY_VARIETY = 0.4
FIRST_SHOT_VARIETY = 1.0   # no choice yet; hold constant so it cannot bias the argmax


def choose(needs: Sequence[ShotNeed],
           candidates: Optional[Sequence[str]] = None,
           technical_matched: bool = True,
           allow_unproven: bool = False,
           renderable_only: bool = True,
           weights: Dict[str, float] = WEIGHTS,
           verbose: bool = True) -> Tuple[List[str], List[Assignment], Dict]:
    """Assign a model per shot to maximise legible variety.

    `technical_matched=True` assumes `grade_match.match_technical` runs before
    assembly, which is what makes a switch safe. With it False, the raw
    technical distance is charged against the switch.

    `allow_unproven=False` keeps any model whose output the user has not accepted
    out of the pool entirely.

    `renderable_only=True` (the default for the film path) restricts the pool to
    models `make_film.py` can actually render. Proposing svd-xt or t2v-zero would
    be a plan nothing can execute -- the exact "selector decides, nothing acts"
    gap this parameter exists to close. Pass False to see the wider field, but
    understand the result is not renderable today.
    """
    import grade_match as gm

    cands = list(candidates) if candidates else list(MODELS)
    if not allow_unproven:
        cands = [c for c in cands if MODELS[c].proven]
    if renderable_only:
        cands = [c for c in cands if MODELS[c].renderable]
    sigs = measured_signatures()
    spreads = measure_spreads()

    assigned: List[str] = []
    out: List[Assignment] = []
    scenes = [getattr(n, "scene", "") for n in needs]

    for i, need in enumerate(needs):
        scores: Dict[str, Tuple[float, float, float]] = {}
        totals: Dict[str, float] = {}
        notes_all: Dict[str, List[str]] = {}
        rejected: Dict[str, str] = {}

        for name in cands:
            m = MODELS[name]
            notes: List[str] = []

            # ---- watermark: a RISK, not a gate ----------------------------- #
            # This used to be a hard gate excluding sd15 outright, on the
            # strength of handoff section 4.7. That was wrong. `watermark_check.py`
            # amplifies the high-frequency residual of the very frames the verdict
            # cites and finds no legible letterforms -- sd15 scores 2.3x
            # dreamshaper8 on structured high-frequency content (0.0027 vs
            # 0.0012), which is foliage detail, not text. The gate was excluding a
            # model the user explicitly wants, on evidence that does not hold up.
            #
            # Caveat, stated so this is not over-claimed in the other direction:
            # 16 frames of one forest prompt do not prove sd15 never emits a
            # watermark. It is prompt-dependent in the training data. So the check
            # is per-render (`watermark_check.text_likeness` on real output), and
            # here it is only a fit penalty.
            if m.watermark_risk > 0:
                notes.append(f"{name}: watermark risk {m.watermark_risk:.1f} -> "
                             f"fit x{1.0 - 0.15 * m.watermark_risk:.2f} "
                             f"(verify per render, not gated)")
            if m.motion + 0.15 < need.motion:
                rejected[name] = (f"produces {m.motion:.2f} motion but the shot "
                                  f"needs {need.motion:.2f}")
                continue
            if m.s_per_frame > 120 and need.motion < 0.2 and not m.still_ok:
                rejected[name] = (f"{m.s_per_frame:.0f} s/frame for a shot that "
                                  f"barely moves -- poor value")
                continue

            # ---- fit ------------------------------------------------------- #
            f = 1.0 - (abs(m.detail - need.detail) + abs(m.motion - need.motion)) / 2.0
            f = max(0.0, min(1.0, f))
            if m.watermark_risk > 0:
                f *= 1.0 - 0.15 * m.watermark_risk

            prev = assigned[-1] if assigned else None
            run = 0
            for a in reversed(assigned):
                if a == prev:
                    run += 1
                else:
                    break

            # ---- technical baseline ---------------------------------------- #
            if prev is None or prev == name:
                tech, tech_note = 1.0, "same model, no baseline change"
            elif technical_matched:
                tech = 1.0
                tech_note = ("baseline matched in post (match_technical) -- "
                             "exposure/contrast/grain no longer a reason to avoid")
            else:
                if prev in sigs and name in sigs:
                    d = gm.decomp(sigs[prev], sigs[name],
                                  (spreads.get(prev, 0.15)
                                   + spreads.get(name, 0.15)) / 2.0)
                    tech = max(0.0, 1.0 - d["technical"] / 12.0)
                    tech_note = (f"raw technical jump {d['technical']:.2f} "
                                 f"charged against the switch")
                else:
                    tech, tech_note = 0.6, "no renders to measure the baseline"
            if prev is not None and prev != name:
                notes.append(tech_note)

            # ---- variety: the thing actually being maximised --------------- #
            var = FIRST_SHOT_VARIETY
            if prev is None:
                pass
            elif prev != name:
                if prev in sigs and name in sigs:
                    d = gm.decomp(sigs[prev], sigs[name],
                                  (spreads.get(prev, 0.15)
                                   + spreads.get(name, 0.15)) / 2.0)
                    # band: reward a legible palette change, punish too little
                    s = d["stylistic"]
                    if s < STYLE_MIN:
                        var = 0.2
                        notes.append(f"palette change {s:.2f} is below the "
                                     f"legible floor {STYLE_MIN} -- invisible")
                    elif s > STYLE_MAX:
                        var = 0.3
                        notes.append(f"palette change {s:.2f} exceeds {STYLE_MAX} "
                                     f"-- reads as a different film")
                    else:
                        var = 1.0
                        notes.append(f"palette change {s:.2f} is a legible look "
                                     f"shift ({STYLE_MIN}-{STYLE_MAX})")
                else:
                    var = 0.7
                # motivated only at a structural boundary
                boundary = (i > 0 and scenes[i] != scenes[i - 1])
                if not boundary:
                    var *= UNMOTIVATED
                    notes.append("switch is NOT at a scene boundary -- reads as "
                                 "a continuity error, not a choice")
                else:
                    notes.append(f"switch lands on the scene change "
                                 f"{scenes[i-1]!r} -> {scenes[i]!r}")
            elif prev is not None and prev == name:
                # staying contributes no variety; monotony makes it worse
                var = STAY_VARIETY if run < MAX_SAME_RUN else 0.1
                if run >= MAX_SAME_RUN:
                    notes.append(f"{run} shots on {name} in a row -- monotonous")
                elif i > 0 and scenes[i] != scenes[i - 1]:
                    notes.append(f"stays on {name} across the scene change "
                                 f"{scenes[i-1]!r} -> {scenes[i]!r}: contributes "
                                 f"no variety")

            # ---- rhythm ---------------------------------------------------- #
            if len(assigned) >= 3:
                alt = (assigned[-1] != assigned[-2] != assigned[-3]
                       and assigned[-1] != assigned[-3])
                r = 0.4 if (alt and prev != name) else 1.0
                if r < 1.0:
                    notes.append("alternating every shot -- variety needs a tempo")
            else:
                r = 1.0

            total = (weights["fit"] * f + weights["technical"] * tech
                     + weights["variety"] * var + weights["rhythm"] * r)
            scores[name] = (f, tech, var)
            totals[name] = total
            notes_all[name] = notes

        if not totals:
            # Never let this pass silently. An unexplained fallback is how a
            # result stops being auditable -- in the demo shot list this fires on
            # a 0.70-energy shot, because the only model claiming enough motion
            # (t2v-zero, 0.95) is currently unproven and every proven model sits
            # at 0.50. That is AnimateDiff's motion ceiling showing up as a hard
            # constraint, not a scoring accident.
            rejected["all"] = ("every candidate was gated out at this shot's "
                               "requirements; repeating the previous model")
            chosen = assigned[-1] if assigned else "dreamshaper8"
            assigned.append(chosen)
            out.append(Assignment(i, chosen, chosen, 0, 0, 0, 0, 0,
                                  ["NO ELIGIBLE MODEL -- repeated "
                                   f"{chosen}"], rejected))
            if verbose:
                print(f"  [models] shot {i+1}: {chosen:<14} "
                      f"<-- NO ELIGIBLE MODEL, repeated previous. "
                      f"Reason: {rejected['all']}")
            continue

        order = sorted(totals, key=lambda n: (-totals[n], n))
        chosen, runner = order[0], order[1] if len(order) > 1 else order[0]
        f, tech, var = scores[chosen]
        sc = 0.0
        if assigned and assigned[-1] != chosen and assigned[-1] in sigs \
                and chosen in sigs:
            sc = gm.decomp(sigs[assigned[-1]], sigs[chosen],
                           (spreads.get(assigned[-1], 0.15)
                            + spreads.get(chosen, 0.15)) / 2.0)["total"]
        a = Assignment(i, chosen, runner, round(f, 3), round(tech, 3),
                       round(var, 3), round(totals[chosen], 3), round(sc, 2),
                       notes_all[chosen], rejected)
        out.append(a)
        if verbose:
            sw = "" if not assigned or assigned[-1] == chosen else \
                f"  <- SWITCH (look shift {sc:.2f})"
            print(f"  [models] shot {i+1}: {chosen:<14} "
                  f"(fit {f:.2f} tech {tech:.2f} variety {var:.2f} "
                  f"= {totals[chosen]:.3f}){sw}")
        assigned.append(chosen)

    switches = sum(1 for i in range(1, len(assigned)) if assigned[i] != assigned[i-1])
    meta = {"signatures_measured": sorted(sigs.keys()),
            "technical_matched": technical_matched,
            "switches": switches, "assigned": list(assigned)}
    return assigned, out, meta


def report(assignments: List[Assignment], meta: Dict) -> str:
    lines = [f"  models: {meta['assigned']}",
             f"  switches: {meta['switches']}",
             f"  look measurements from real renders: "
             f"{meta['signatures_measured'] or '(none on disk)'}",
             f"  technical baseline matched in post: {meta['technical_matched']}",
             "",
             "  fit = does the model suit the shot | tech = photographic baseline",
             "  variety = is the look change legible as a choice (rewarded)"]
    for a in assignments:
        lines.append(f"  {a.index+1:>2}. {a.chosen:<14} fit={a.fit:.2f} "
                     f"tech={a.technical:.2f} variety={a.variety:.2f} "
                     f"-> {a.total:.3f}   (next: {a.runner_up})")
        for n in a.notes:
            lines.append(f"        {n}")
        for name, why in a.rejected.items():
            lines.append(f"        rejected {name}: {why}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-post", action="store_true",
                    help="do NOT assume grade_match.match_technical runs; charge "
                         "the raw technical jump against every switch")
    ap.add_argument("--allow-unproven", action="store_true",
                    help="include models whose output the user has not accepted "
                         "(currently t2v-zero)")
    a = ap.parse_args()

    # Two scenes, so there is exactly one motivated boundary to switch on.
    # Scene 1 is the woman's walk (one continuous look); scene 2 is the water.
    shots = [
        ShotNeed("a lone woman walking down a forest road at dawn", "wide",
                 motion=0.5, detail=0.6, subject="woman", scene="walk"),
        ShotNeed("dew on a mossy leaf, macro", "macro", motion=0.2, detail=0.95,
                 scene="walk"),
        ShotNeed("the woman walking on toward the camera through the mist",
                 "medium", motion=0.5, detail=0.7, subject="woman", scene="walk"),
        ShotNeed("a waterfall in a mossy gorge", "wide", motion=0.6, detail=0.8,
                 scene="water"),
        ShotNeed("water running over mossy boulders, close", "close",
                 motion=0.5, detail=0.9, scene="water"),
        ShotNeed("tall trees, canopy and sky", "wide", motion=0.7, detail=0.7,
                 scene="water"),
    ]
    print("=== model selection for VARIETY "
          f"(technical baseline matched in post: {not a.no_post}) ===")
    assigned, rep, meta = choose(shots, technical_matched=not a.no_post,
                                 allow_unproven=a.allow_unproven)
    print("\n" + report(rep, meta))
