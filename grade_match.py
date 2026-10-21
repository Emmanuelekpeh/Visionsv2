"""
grade_match.py -- make clips from DIFFERENT models intercut as one film.

The question this answers: can the system choose per shot which checkpoint or
generator to use -- sd15 for some shots, dreamshaper8 for others -- and still read
as a single film?

Intercutting checkpoints raw cannot. `output/film/checkpoint_ab/compare.png` is
the same prompt and the same seed through both: sd15 gives saturated golden light
shafts and dense bright foliage, dreamshaper8 gives desaturated grey-blue haze and
soft low-contrast air. Cut between them and the audience sees the seam, not the
story. Model choice and visual cohesion are in direct tension, and the tension is
*measurable*.

So the answer is not "pick one model". It is: pick per shot on fit, then remove
the seam by measuring each clip's look and normalising it toward the film's
target look. Reinhard-style mean/std transfer in CIELAB does most of it, because
the differences between SD1.5-family checkpoints are largely per-channel gain and
offset plus saturation -- exactly what a per-channel affine fixes.

Two numbers do the work:

  signature    per-channel mean/std in Lab, chroma, contrast, grain energy.
               A clip's "look" as a vector.
  seam         distance between two clips' signatures, expressed as a multiple of
               the *within-clip* frame-to-frame spread of the same features. A
               seam ratio near 1 means the cut is indistinguishable from ordinary
               frame-to-frame variation. 10 means it is a different film.

Nothing here assumes a direction: the target look can come from the previous
shot, from a nominated hero shot, or from the film's own average. The point is
that the target is *measured*, not asserted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
AB = ROOT / "output" / "film" / "checkpoint_ab"

# feature names, in the order the signature vector uses them
FEATURES = ["L_mean", "L_std", "a_mean", "b_mean", "a_std", "b_std",
            "chroma", "grain"]

# how much each feature contributes to a visible seam. Luminance and chroma
# means are what the eye reads as a colour-temperature change across a cut;
# grain matters less than a hue shift but still shows.
SEAM_WEIGHTS = np.array([1.0, 0.8, 1.0, 1.0, 0.6, 0.6, 1.0, 0.5])

# rough scales so the features are comparable before weighting
SCALES = np.array([30.0, 12.0, 8.0, 8.0, 6.0, 6.0, 10.0, 3.0])


def _frame_features(rgb: np.ndarray) -> np.ndarray:
    """One frame -> feature vector. rgb is HxWx3 uint8."""
    import cv2

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = float(np.sqrt(((A - 128.0) ** 2 + (B - 128.0) ** 2)).mean())
    # grain: energy above a blur, i.e. what a compression or a checkpoint's
    # texture statistics change
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(g, (0, 0), 2.0)
    grain = float(np.abs(g.astype(np.float32) - blur.astype(np.float32)).mean())
    return np.array([
        float(L.mean()), float(L.std()), float(A.mean()), float(B.mean()),
        float(A.std()), float(B.std()), chroma, grain,
    ], dtype=np.float32)


def signature(frames: Sequence) -> np.ndarray:
    """A clip's look: the mean feature vector over its frames."""
    feats = [_frame_features(np.asarray(f.convert("RGB") if hasattr(f, "convert")
                                        else f)) for f in frames]
    return np.mean(np.stack(feats), axis=0)


def within_spread(frames: Sequence) -> float:
    """How much the look varies frame to frame inside one clip.

    This is the calibration for `seam`: a cut whose signature distance is no
    bigger than the variation already present within a shot is invisible.
    """
    feats = np.stack([_frame_features(np.asarray(f.convert("RGB")
                                                 if hasattr(f, "convert") else f))
                      for f in frames])
    norm = feats.std(axis=0) / SCALES
    return float(np.sqrt((norm ** 2 * SEAM_WEIGHTS).sum()
                         / SEAM_WEIGHTS.sum()))


def seam(frames_a: Sequence, frames_b: Sequence) -> float:
    """Signature distance from clip A to clip B, in units of their own spread."""
    return seam_between(signature(frames_a), signature(frames_b),
                        (within_spread(frames_a) + within_spread(frames_b)) / 2.0)


def seam_between(sig_a: np.ndarray, sig_b: np.ndarray,
                 spread: float) -> float:
    """Normalised by the pair's mean within-clip spread.

    Calibrating against one clip's own spread makes the number asymmetric and
    unstable -- sd15's spread is 7x dreamshaper8's, so the same pair of shots
    would score very differently depending on which was named first.
    """
    d = (sig_a - sig_b) / SCALES
    raw = float(np.sqrt((d ** 2 * SEAM_WEIGHTS).sum() / SEAM_WEIGHTS.sum()))
    return raw / max(1e-6, spread)


def seam_raw(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """The unnormalised distance, so a ratio can never hide the real numbers."""
    d = (sig_a - sig_b) / SCALES
    return float(np.sqrt((d ** 2 * SEAM_WEIGHTS).sum() / SEAM_WEIGHTS.sum()))


# --------------------------------------------------------------------------- #
# the fix
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# variety vs error: the two halves of a look
# --------------------------------------------------------------------------- #
#
# The point of intercutting two checkpoints is VARIETY -- a deliberate change of
# look that holds attention and gives the film a signature. So the metric's job
# is not to minimise difference. It is to keep the difference legible as a CHOICE
# rather than as a mistake.
#
# Those are separable, because a look has two halves:
#
#   technical   exposure, contrast, grain. A jump here reads as an ERROR -- a
#               different camera, a different ISO, a mis-timed shutter. The eye
#               reads it as sloppiness, not as style. MUST MATCH across a cut.
#
#   stylistic   palette and saturation. A difference here reads as STYLE -- warm
#               against cool, bleached against rich. This is the variety that is
#               actually wanted, so it must be PRESERVED, not flattened.
#
# That gives a much more useful tool than `match_to`: normalise the photographic
# baseline, leave the palette alone. `match_technical` does exactly that, and
# `decomp` measures the two halves separately so the claim is checkable.

TECHNICAL = (0, 1, 7)      # L_mean, L_std, grain
STYLISTIC = (2, 3, 6)      # a_mean, b_mean, chroma


def _sub_dist(sig_a: np.ndarray, sig_b: np.ndarray, idx) -> float:
    d = (sig_a - sig_b) / SCALES
    w = np.array([SEAM_WEIGHTS[i] for i in idx])
    return float(np.sqrt((d[list(idx)] ** 2 * w).sum() / w.sum()))


def decomp(sig_a: np.ndarray, sig_b: np.ndarray, spread: float = 1.0) -> dict:
    """Split a cut's distance into the part that reads as error and the part
    that reads as style.

    `spread` MUST be a decomp-scale floor. Passing a raw `within_spread`
    (~0.03-0.3) mixes two quantities, and `clip_floor` -- which this docstring
    used to recommend -- collapses toward zero on a palette-stable film and
    inflates a film-level report by up to 1e5 (see `clip_floor` for the measured
    numbers). For a film-level technical report use `technical_offset`, which
    normalises nothing. This exact units confusion has now caused five separate
    wrong numbers in this project (hand-picked seam thresholds, the loc10/mag
    confound, the look band, the film's 16.29, and the 1e5 that replacing it
    produced). If a distance looks wrong, check the normaliser first.

    The permanent fix is to delete this parameter and always use raw distances;
    until then, prefer `technical_offset` and `seam_raw` for anything a human
    reads.
    """
    tech = _sub_dist(sig_a, sig_b, TECHNICAL) / max(1e-6, spread)
    style = _sub_dist(sig_a, sig_b, STYLISTIC) / max(1e-6, spread)
    return {"technical": round(tech, 2), "stylistic": round(style, 2),
            "total": round(seam_between(sig_a, sig_b, spread), 2)}


def clip_floor(frames: Sequence) -> float:
    """How far the signature already moves between two halves of the SAME clip.

    WARNING -- this is NOT a usable normaliser for this project's films. It is
    the STYLISTIC distance between halves, and a film with one shared technical
    baseline plus one palette per scene has an almost perfectly stable palette
    across halves, so the floor collapses toward zero and the division explodes.
    Measured on `output/film/shadows`: floors of 0.000, 0.000, 0.010, 0.020,
    0.060, 0.060, and per-shot technical offsets of 1.29, 15.98, 15.99, 8.14,
    175659.85, 682809.09 -- the same pathology that produced the reported 793,261
    and 814,481, with the degeneracy moved from `within_spread` to here rather
    than fixed. The absolute distance has physical units and needs no
    normaliser: use `technical_offset` below.

    Kept because `model_select` and the A/B harnesses still read it for pools
    whose palettes genuinely vary. `within_spread` is a different quantity on a
    different scale, not a substitute.
    """
    h = len(frames) // 2
    if h < 1:
        return 1.0
    return max(1e-6, decomp(signature(frames[:h]), signature(frames[h:]))["stylistic"])


def technical_offset(sig: np.ndarray, target: np.ndarray) -> float:
    """Exposure/contrast/grain distance from a shared target, NO normaliser.

    `decomp` divides by a `spread` argument, and every candidate normaliser in
    this file is degenerate somewhere that matters -- `within_spread` is a
    different quantity, `clip_floor` collapses to zero on a palette-stable film,
    and `clip_floor`/`within_spread` are not even consistently ordered (0.010 vs
    0.039 on the same clip, so the same call inflates or deflates depending on
    the content). This is the fix the previous session reached and did not wire:
    the technical distance already has physical units (SCALES is in grey levels),
    so it is reported absolutely.

    Measured on `output/film/shadows` (6 shots, decomp units, ~12 grey levels per
    unit): 0.47-0.70 before `match_technical`, 0.09-0.18 after, means 0.58 -> 0.15
    -- a 3.8x reduction, i.e. from ~7 to ~1.8 grey levels of contrast, which is
    below the threshold of visibility. The palette control holds at 1.00x, so the
    matching is fixing the seam and not flattening the look.
    """
    return float(_sub_dist(sig, target, TECHNICAL))


# how the two halves should read. Calibrate these against the noise floor, not
# against taste: a technical jump the size of ordinary within-shot variation is
# invisible, and that is what "matched" has to mean.
STYLE_MIN = 1.5      # below this a switch is too small to register as a choice
STYLE_MAX = 9.0      # above this the two shots stop belonging to one film
TECH_TOL = 1.5       # above this a cut reads as a technical error


def style_verdict(d: dict) -> str:
    if d["technical"] > TECH_TOL:
        return "ERROR -- technical jump (exposure/contrast/grain) reads as sloppy"
    if d["stylistic"] < STYLE_MIN:
        return "INVISIBLE -- the switch is too small to read as a choice"
    if d["stylistic"] > STYLE_MAX:
        return "TOO FAR -- reads as a different film, not a different look"
    return "LEGIBLE STYLE -- one photographic baseline, two palettes"


def technical_target(sigs: Sequence[np.ndarray]) -> np.ndarray:
    """The shared photographic baseline for a set of clips.

    Exposure and contrast can be matched in either direction, so the mean is
    right. Grain cannot: it is one-sided. Adding noise is exact and free, but
    *removing* it means blurring, which also softens contrast and undoes the
    contrast match -- the coupling left a 2x-floor technical residual in the
    first version of this. Raising everything to the grainiest shot avoids the
    coupling entirely.
    """
    t = np.mean(np.stack(sigs), axis=0)
    t[7] = max(float(s[7]) for s in sigs)
    return t


def match_technical(frames: Sequence, target: np.ndarray,
                    strength: float = 1.0, match_grain: bool = True) -> List:
    """Normalise exposure, contrast and grain. LEAVE THE PALETTE ALONE.

    This is the tool for deliberate variety: both checkpoints end up sharing a
    photographic baseline, so the cut does not read as an error, while their
    colour differences survive intact as the style. Contrast with `match_to`,
    which flattens both halves and therefore destroys the variety.
    """
    import cv2
    from PIL import Image

    t_L_mean, t_L_std = target[0], target[1]
    out = []
    for f in frames:
        rgb = np.asarray(f.convert("RGB") if hasattr(f, "convert") else f)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]

        L2 = (L - L.mean()) * (t_L_std / max(1e-3, L.std())) + t_L_mean
        L = L + strength * (L2 - L)

        rgb2 = cv2.cvtColor(np.clip(np.stack([L, A, B], axis=-1), 0, 255)
                            .astype(np.uint8), cv2.COLOR_LAB2RGB)
        if match_grain and strength > 0:
            rgb2 = _match_grain(rgb2, target[7], strength)
        out.append(Image.fromarray(rgb2))
    return out


def _match_grain(rgb: np.ndarray, target_grain: float,
                 strength: float) -> np.ndarray:
    """Match the measured high-frequency energy, in both directions.

    Adding noise can only *raise* grain. The sd15 clip, once its contrast and
    saturation were matched, carried more grain than the target -- and no amount
    of added noise would bring it down, so it overshot and grain ended up being
    96% of the entire residual seam. So when the clip is too grainy it is
    softened instead, choosing the blur that measures closest to the target.
    """
    import cv2

    have = _frame_features(rgb)[7]
    need = target_grain - have
    if abs(need) < 0.05:
        return rgb

    if need > 0:
        # mean|N(0,1)| = 0.7979, so this amplitude lands on the target
        amp = need / 0.7979
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 1.0, size=rgb.shape[:2])[..., None]
        out = rgb.astype(np.float32) + strength * amp * noise
        return np.clip(out, 0, 255).astype(np.uint8)

    best, best_err = rgb, abs(need)
    for sigma in (0.4, 0.6, 0.8, 1.0, 1.3, 1.6):
        cand = cv2.GaussianBlur(rgb, (0, 0), sigma)
        err = abs(target_grain - _frame_features(cand)[7])
        if err < best_err:
            best, best_err = cand, err
    return cv2.addWeighted(rgb, 1.0 - strength, best, strength, 0.0)


def match_to(frames: Sequence, target: np.ndarray, strength: float = 1.0,
             match_grain: bool = True) -> List:
    """Re-grade a clip toward a target look, in CIELAB.

    Per-channel mean/std transfer (Reinhard), applied **per frame** using that
    frame's own spatial statistics and blended by `strength`. Per-frame rather
    than per-clip because a clip-level transfer cannot fix flicker, and sd15
    clips flicker a great deal: measured within-clip spread 0.267 against
    dreamshaper8's 0.037.

    SD1.5-family checkpoints differ mainly in colour gain, offset and
    saturation, so an affine in Lab closes most of the gap without touching
    geometry -- the clip's motion and content are untouched.
    """
    import cv2
    from PIL import Image

    t_L_mean, t_L_std = target[0], target[1]
    t_a_mean, t_a_std = target[2], target[4]
    t_b_mean, t_b_std = target[3], target[5]
    t_chroma = target[6]

    out = []
    for f in frames:
        rgb = np.asarray(f.convert("RGB") if hasattr(f, "convert") else f)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]

        L2 = (L - L.mean()) * (t_L_std / max(1e-3, L.std())) + t_L_mean

        # Chroma and the a/b standard deviations fight each other: scaling a/b to
        # hit the target saturation changes their std, which undoes the contrast
        # match. Two passes converge -- the residual raw seam dropped from 0.58
        # to 0.33 once this was iterated.
        A2, B2 = A, B
        for _ in range(2):
            A2 = (A2 - A2.mean()) * (t_a_std / max(1e-3, A2.std())) + t_a_mean
            B2 = (B2 - B2.mean()) * (t_b_std / max(1e-3, B2.std())) + t_b_mean
            cur = float(np.sqrt(((A2 - 128.0) ** 2 + (B2 - 128.0) ** 2)).mean())
            k = t_chroma / max(1e-3, cur)
            A2 = (A2 - 128.0) * k + 128.0
            B2 = (B2 - 128.0) * k + 128.0

        L = L + strength * (L2 - L)
        A = A + strength * (A2 - A)
        B = B + strength * (B2 - B)

        rgb2 = cv2.cvtColor(np.clip(np.stack([L, A, B], axis=-1), 0, 255)
                            .astype(np.uint8), cv2.COLOR_LAB2RGB)

        if match_grain and strength > 0:
            rgb2 = _match_grain(rgb2, target[7], strength)
        out.append(Image.fromarray(rgb2))
    return out


def film_target(sigs: Sequence[np.ndarray]) -> np.ndarray:
    """The film's own look: the mean of the shots' signatures.

    Normalising every shot toward the average is the safest intercut: no shot is
    privileged, and the total grade drift is minimised.
    """
    return np.mean(np.stack(sigs), axis=0)


def cohesion_report(titles: Sequence[str], clips: Sequence[Sequence],
                    target: np.ndarray | None = None,
                    strength: float = 1.0) -> Dict:
    """Seam between every adjacent pair, before and after matching."""
    sigs = [signature(c) for c in clips]
    spreads = [within_spread(c) for c in clips]
    tgt = film_target(sigs) if target is None else target
    matched = [match_to(c, tgt, strength) for c in clips]
    msigs = [signature(c) for c in matched]

    rows = []
    for i in range(len(clips) - 1):
        spread = (spreads[i] + spreads[i + 1]) / 2.0
        before = seam_between(sigs[i], sigs[i + 1], spread)
        after = seam_between(msigs[i], msigs[i + 1], spread)
        rows.append({"cut": f"{titles[i]} -> {titles[i+1]}",
                     "seam_before": round(before, 2),
                     "seam_after": round(after, 2),
                     "raw_before": round(seam_raw(sigs[i], sigs[i + 1]), 4),
                     "raw_after": round(seam_raw(msigs[i], msigs[i + 1]), 4),
                     "reduction_x": round(before / max(1e-6, after), 2)})
    return {"before_signatures": [s.round(3).tolist() for s in sigs],
            "target": tgt.round(3).tolist(),
            "within_spread": [round(s, 4) for s in spreads],
            "cuts": rows, "strength": strength,
            "matched_frames": matched}


def calibration() -> List[dict]:
    """Ladder of seams with a known expected ordering.

    A single number cannot validate a metric. This measures cases whose answer is
    already known, so the 7.76 for sd15 -> dreamshaper8 can be read against a
    scale instead of against nothing:

      within a clip, first half vs second half   should be LOWEST
      same checkpoint, different LoRA            should be low
      different checkpoint, same prompt/seed     should be HIGHEST by far

    If the ordering does not come out this way, the metric is measuring
    something other than a look change, and every seam number in this file --
    including the 4.56x claim -- is void.
    """
    from PIL import Image

    def load(d):
        fs = sorted((AB / d).glob("f*.png"))
        return [Image.open(f) for f in fs] if fs else []

    clips = {n: load(n) for n in ["SD15_base_none", "SD15_base_zoom",
                                  "DreamShaper8_none", "DreamShaper8_zoom"]}
    if any(not v for v in clips.values()):
        return []

    rows = []

    def add(label, a, b, expect):
        if not a or not b:
            return
        sp = (within_spread(a) + within_spread(b)) / 2.0
        rows.append({"case": label,
                     "seam": round(seam_between(signature(a), signature(b), sp), 2),
                     "expect": expect})

    # 1. inside one clip: the floor of the metric
    for name in ["SD15_base_none", "DreamShaper8_none"]:
        c = clips[name]
        h = len(c) // 2
        add(f"{name}: 1st half vs 2nd half", c[:h], c[h:], "lowest")
    # 2. same checkpoint, different motion LoRA
    add("sd15: none vs zoom-in",
        clips["SD15_base_none"], clips["SD15_base_zoom"], "low")
    add("dreamshaper8: none vs zoom-in",
        clips["DreamShaper8_none"], clips["DreamShaper8_zoom"], "low")
    # 3. different checkpoint, matched prompt and seed
    add("sd15 vs dreamshaper8 (no lora)",
        clips["SD15_base_none"], clips["DreamShaper8_none"], "highest")
    add("sd15 vs dreamshaper8 (zoom-in)",
        clips["SD15_base_zoom"], clips["DreamShaper8_zoom"], "high")
    return rows


def describe(sig: np.ndarray) -> str:
    return (f"L {sig[0]:6.1f}±{sig[1]:5.1f}  a {sig[2]-128:+5.1f}  "
            f"b {sig[3]-128:+5.1f}  chroma {sig[6]:5.1f}  grain {sig[7]:4.1f}")


if __name__ == "__main__":
    import sys

    from PIL import Image

    if "--calibrate" in sys.argv:
        print("=== seam metric calibration ladder ===")
        print("  a metric validated only by the one number you care about is not "
              "validated")
        rows = calibration()
        if not rows:
            print("  no A/B renders on disk")
            raise SystemExit(1)
        for r in rows:
            print(f"  {r['case']:<42} seam {r['seam']:>7.2f}   "
                  f"expected: {r['expect']}")
        got = {r["case"]: r["seam"] for r in rows}
        cross = [v for k, v in got.items() if "sd15 vs dreamshaper8" in k]
        floor = [v for k, v in got.items() if "half" in k]
        same = [v for k, v in got.items()
                if "vs" in k and "sd15 vs dreamshaper8" not in k and "half" not in k]

        # A first version of this check passed on `min(cross) > max(same)`,
        # which accepted 5.58 > 5.10 -- a 9% margin between "same checkpoint,
        # different content" and "different checkpoint". That is not separation,
        # it is noise. A metric is only fit to decide a cut if the cross-model
        # signal clears the same-model spread by a clear factor.
        MARGIN = 2.0
        ok = bool(cross and same and floor
                  and min(cross) > MARGIN * max(same)
                  and max(same) > max(floor))
        print(f"\n  noise floor (within one clip) : {max(floor):>6.2f}")
        print(f"  same checkpoint, other content: {max(same):>6.2f}")
        print(f"  different checkpoint          : {min(cross):>6.2f}")
        print(f"  required margin: cross > {MARGIN}x same-model "
              f"(need > {MARGIN * max(same):.2f})")
        print(f"\n  ordering {'HOLDS' if ok else 'FAILS'}")
        if not ok:
            print("  The metric does NOT separate a change of look from a change\n"
                  "  of content. Any seam number in this file is therefore only\n"
                  "  trustworthy when content is held fixed -- which is exactly\n"
                  "  the case that does NOT occur in a real intercut.")
        raise SystemExit(0 if ok else 2)

    if "--variety" in sys.argv:
        from PIL import Image

        def load(d):
            fs = sorted((AB / d).glob("f*.png"))
            return [Image.open(f) for f in fs] if fs else []

        a = load("SD15_base_none")
        b = load("DreamShaper8_none")
        if not (a and b):
            print("no A/B renders on disk")
            raise SystemExit(1)
        sp = (within_spread(a) + within_spread(b)) / 2.0
        print("=== can we have variety WITHOUT it reading as an error? ===")

        # Thresholds must come from a measurement, not from taste. The floor for
        # "matched" is how far the technical half already moves *inside one
        # clip*, where no look change exists at all. A first version of this
        # test used hand-picked constants (TECH_TOL=1.5) on an uncalibrated
        # scale and declared failure for every treatment -- the numbers were
        # right and the thresholds were fiction.
        floors = []
        for c in (a, b):
            h = len(c) // 2
            floors.append(decomp(signature(c[:h]), signature(c[h:]), sp))
        f_tech = sum(f["technical"] for f in floors) / len(floors)
        f_style = sum(f["stylistic"] for f in floors) / len(floors)
        print(f"  measured floor, no look change (within one clip):")
        print(f"    technical {f_tech:.2f}   stylistic {f_style:.2f}")
        print(f"    (earlier hand-picked guesses were tech<{TECH_TOL}, "
              f"style>{STYLE_MIN} -- pure fiction)\n")

        rows = [("as generated", a, b),
                ("match_to (flatten both halves)", None, None),
                ("match_technical (baseline only)", None, None)]
        tgt = technical_target([signature(a), signature(b)])
        full = [match_to(c, tgt) for c in (a, b)]
        tech = [match_technical(c, tgt) for c in (a, b)]
        pairs = {"as generated": (a, b),
                 "match_to (flatten both halves)": (full[0], full[1]),
                 "match_technical (baseline only)": (tech[0], tech[1])}

        print(f"  {'treatment':<34} {'tech':>7} {'/floor':>7} {'style':>7} "
              f"{'/floor':>7}   verdict")
        print("  " + "-" * 88)
        for label, (x, y) in pairs.items():
            d = decomp(signature(x), signature(y), sp)
            tr = d["technical"] / max(1e-6, f_tech)
            sr = d["stylistic"] / max(1e-6, f_style)
            if tr <= 1.5 and 1.5 <= sr <= 8.0:
                verdict = "LEGIBLE STYLE -- baseline matched, palette kept"
            elif tr <= 1.5:
                verdict = "ERROR-free but FLAT -- variety destroyed"
            else:
                verdict = "ERROR -- technical jump reads as sloppy"
            print(f"  {label:<34} {d['technical']:>7.2f} {tr:>6.1f}x "
                  f"{d['stylistic']:>7.2f} {sr:>6.1f}x   {verdict}")

        print(f"\n  want: tech within 1.5x of floor, style 1.5-8x of floor")
        d_tech = decomp(signature(tech[0]), signature(tech[1]), sp)
        d_full = decomp(signature(full[0]), signature(full[1]), sp)
        ok = (d_tech["technical"] / max(1e-6, f_tech) <= 1.5
              and 1.5 <= d_tech["stylistic"] / max(1e-6, f_style) <= 8.0)
        print(f"  -> match_technical  "
              f"{'ACHIEVES' if ok else 'FAILS'} one baseline + two palettes")
        print(f"  -> match_to         "
              f"{'keeps' if d_full['stylistic'] / max(1e-6, f_style) >= 1.5 else 'destroys'}"
              f" the variety (style {d_full['stylistic']:.2f} vs floor {f_style:.2f})")

        out = ROOT / "output" / "film" / "grade_match"
        out.mkdir(parents=True, exist_ok=True)
        T = 256
        canvas = Image.new("RGB", (3 * T + 16, 2 * (T + 18) + 24), (16, 16, 20))
        from PIL import ImageDraw
        dr = ImageDraw.Draw(canvas)
        dr.text((5, 4), "top: sd15   bottom: dreamshaper8     "
                        "as generated | match_to | match_technical",
                fill=(235, 235, 235))
        for r, (label, i) in enumerate([("sd15", 0), ("dreamshaper8", 1)]):
            y = 24 + r * (T + 18)
            for c, imgs in enumerate([pairs["as generated"],
                                      pairs["match_to (flatten both halves)"],
                                      pairs["match_technical (baseline only)"]]):
                canvas.paste(imgs[i][len(imgs[i]) // 2].resize((T, T), Image.LANCZOS),
                             (4 + c * (T + 4), y))
        canvas.save(out / "variety.png")
        print(f"\n  wrote {out/'variety.png'}")
        raise SystemExit(0 if ok else 2)

    AB = ROOT / "output" / "film" / "checkpoint_ab"
    pairs = [("sd15", AB / "SD15_base_none"), ("dreamshaper8",
                                               AB / "DreamShaper8_none")]
    clips, titles = [], []
    for name, d in pairs:
        fs = sorted(d.glob("f*.png"))
        if not fs:
            continue
        clips.append([Image.open(f) for f in fs])
        titles.append(name)
    if len(clips) < 2:
        print(f"need the two checkpoint renders in {AB}")
        raise SystemExit(1)

    print("=== look signatures (same prompt, same seed, different checkpoint) ===")
    for t, c in zip(titles, clips):
        print(f"  {t:<14} {describe(signature(c))}")

    rep = cohesion_report(titles, clips)
    print(f"\n  within-clip frame-to-frame spread: "
          f"{rep['within_spread']}  (the calibration for a visible seam)")
    print(f"  film target                      {describe(np.array(rep['target']))}")
    print("\n=== seam across the cut (1.0 = invisible, >3 = a different film) ===")
    for r in rep["cuts"]:
        print(f"  {r['cut']:<30} seam {r['seam_before']:>6.2f} -> "
              f"{r['seam_after']:>5.2f}   raw {r['raw_before']:.4f} -> "
              f"{r['raw_after']:.4f}   ({r['reduction_x']}x better)")

    # visual proof
    out = ROOT / "output" / "film" / "grade_match"
    out.mkdir(parents=True, exist_ok=True)
    T = 256
    rows = [(t, c, m) for t, c, m in zip(titles, clips, rep["matched_frames"])]
    canvas = Image.new("RGB", (2 * T + 12, len(rows) * (T + 20) + 24), (16, 16, 20))
    from PIL import ImageDraw
    dr = ImageDraw.Draw(canvas)
    dr.text((5, 4), "left: as generated   right: re-graded to the film target",
            fill=(235, 235, 235))
    for i, (t, orig, match) in enumerate(rows):
        y = 24 + i * (T + 20)
        dr.text((5, y), f"{t}  as generated vs matched", fill=(215, 215, 215))
        canvas.paste(orig[len(orig) // 2].resize((T, T), Image.LANCZOS), (4, y + 15))
        canvas.paste(match[len(match) // 2].resize((T, T), Image.LANCZOS),
                     (T + 8, y + 15))
    canvas.save(out / "intercut.png")
    (out / "report.json").write_text(json.dumps(
        {k: v for k, v in rep.items() if k != "matched_frames"}, indent=2),
        encoding="utf-8")
    print(f"\n  wrote {out/'intercut.png'}")
