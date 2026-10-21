"""
looks.py -- N deliberate palettes from ONE checkpoint.

This is the route to a signature look without depending on two checkpoints
happening to differ pleasingly. `grade_match.match_technical` gives every shot a
shared photographic baseline (exposure, contrast, grain). On top of that baseline
each *scene* gets a chosen palette, so the film has variety that is deliberate,
repeatable and free -- no extra rendering, no download, no model roulette.

Why this and not model swapping:

  deterministic   the same look spec produces the same grade every run. A
                  checkpoint's palette is whatever it happens to be; a look is
                  something you chose.
  free            grading existing frames costs seconds, not ten minutes a clip.
  controllable    "act one warm, act two cold" is a directable decision. Hoping
                  sd15 and dreamshaper8 differ well is not.
  composable      it works on top of ANY model, including a future clean
                  checkpoint, so it does not compete with the model work.

The rule that keeps it from looking broken is section 10.2 of the handoff: only
touch the STYLISTIC half. `apply_look` never alters global exposure or contrast
shape -- it shifts palette and saturation -- so it cannot create the
technical-error seam that reads as sloppiness. `look_distances` measures the
resulting separation in units of the within-shot noise floor, so a palette set is
validated rather than assumed.

    python looks.py            # apply every look to one frame and measure it
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import grade_match as gm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "film" / "looks"


@dataclass(frozen=True)
class Look:
    """A palette spec. Only the stylistic half of the signature is touched.

    chroma      saturation multiplier (1.0 = unchanged)
    tint        (a, b) shift in Lab, i.e. green-magenta and blue-yellow
    lift        small luminance lift, applied as a curve not a level shift so
                that exposure stays with the technical stage
    """
    name: str
    chroma: float = 1.0
    tint: tuple = (0.0, 0.0)
    lift: float = 0.0
    note: str = ""


# A starting set. The names are the point: these are directable choices, and a
# film picks a progression through them rather than a different checkpoint per
# shot. Warm-to-cold across a film is a legible arc; alternating at random is not.
LOOKS: Dict[str, Look] = {
    "neutral": Look("neutral", note="as generated, no palette shift"),
    "warm-gold": Look("warm-gold", chroma=1.14, tint=(2.0, 6.0), lift=1.5,
                      note="golden, late-afternoon; the 'memory' look"),
    "cold-steel": Look("cold-steel", chroma=0.86, tint=(-2.0, -7.0), lift=-1.0,
                       note="desaturated blue; the 'distance' look"),
    "bleach": Look("bleach", chroma=0.62, tint=(0.0, -2.0), lift=3.0,
                   note="near-monochrome, high-key; the 'shock' look"),
    "rich-green": Look("rich-green", chroma=1.18, tint=(-4.0, 2.0), lift=-1.0,
                       note="deep forest green; the 'alive' look"),
    "amber-dusk": Look("amber-dusk", chroma=1.07, tint=(3.0, 8.0), lift=-2.0,
                       note="heavy amber, low key; the 'dread' look"),
    "cyan-night": Look("cyan-night", chroma=0.86, tint=(-3.0, -10.0), lift=-3.0,
                       note="cold cyan cast; the 'night' look"),
}

# Which looks may follow which, if a film wants a legible arc rather than a
# random walk. Ordered so consecutive entries are visibly but not absurdly
# different. A film can also just name a look per scene.
ARC = ["neutral", "warm-gold", "rich-green", "cold-steel", "amber-dusk",
       "bleach", "cyan-night"]


def apply_look(frames: Sequence, look: str | Look,
               strength: float = 1.0) -> List:
    """Grade a clip to a palette. Exposure and contrast are left alone."""
    import cv2
    from PIL import Image

    if isinstance(look, str):
        look = LOOKS.get(look, LOOKS["neutral"])
    if look.name == "neutral" and strength >= 1.0:
        return [f.copy() if hasattr(f, "copy") else f for f in frames]

    da, db = look.tint
    out = []
    for f in frames:
        rgb = np.asarray(f.convert("RGB") if hasattr(f, "convert") else f)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]

        # saturation: scale chroma about the neutral axis
        k = 1.0 + strength * (look.chroma - 1.0)
        A = (A - 128.0) * k + 128.0
        B = (B - 128.0) * k + 128.0

        # tint
        A = A + strength * da
        B = B + strength * db

        # lift as a gentle curve, not a level shift: a level shift would change
        # L_mean and register as a technical error
        if look.lift:
            x = L / 255.0
            L = 255.0 * (x + strength * look.lift / 255.0 * (x * (1.0 - x) * 4.0))

        rgb2 = cv2.cvtColor(np.clip(np.stack([L, A, B], axis=-1), 0, 255)
                            .astype(np.uint8), cv2.COLOR_LAB2RGB)
        out.append(Image.fromarray(rgb2))
    return out


def palette_floor(source: Sequence) -> float:
    """The floor that a `decomp` value must be compared against.

    This has to be a `decomp` value, not a raw `within_spread`. Passing the raw
    spread mixed two different quantities and inflated every distance several-
    fold -- the same units confusion that produced the hand-picked thresholds in
    handoff 10.3. The floor is how far the stylistic half already moves between
    two halves of the *same* clip, where no palette change exists at all.
    """
    h = len(source) // 2
    if h < 1:
        return 1.0
    return max(1e-6, gm.decomp(gm.signature(source[:h]),
                               gm.signature(source[h:]),
                               gm.within_spread(source))["stylistic"])


def look_distances(source: Sequence, names: Sequence[str] | None = None,
                   floor: float | None = None) -> List[dict]:
    """Stylistic separation between each pair of looks on the SAME source frame.

    Held-source measurement, so the number is purely about the palette and not
    about content -- the confound that invalidated the seam metric in handoff
    9.9 cannot apply here.
    """
    names = list(names or ARC)
    if floor is None:
        floor = palette_floor(source)
    sigs = {n: gm.signature(apply_look(source, n)) for n in names}
    rows = []
    for i in range(len(names) - 1):
        d = gm.decomp(sigs[names[i]], sigs[names[i + 1]], floor)
        rows.append({"from": names[i], "to": names[i + 1],
                     "technical": d["technical"], "stylistic": d["stylistic"],
                     "verdict": gm.style_verdict(d), "floor": round(floor, 2)})
    return rows


def sheet(source: Sequence, names: Sequence[str] | None = None,
          path: Path | None = None) -> Path:
    from PIL import Image, ImageDraw

    names = list(names or LOOKS)
    OUT.mkdir(parents=True, exist_ok=True)
    path = path or OUT / "looks.png"
    T = 224
    cols = min(4, len(names))
    rows = (len(names) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (T + 4) + 4, rows * (T + 20) + 24),
                       (16, 16, 20))
    dr = ImageDraw.Draw(canvas)
    dr.text((5, 5), "one checkpoint, N deliberate palettes "
                    "(technical baseline held)", fill=(235, 235, 235))
    for i, n in enumerate(names):
        r, c = divmod(i, cols)
        y = 24 + r * (T + 20)
        dr.text((4 + c * (T + 4), y), n, fill=(215, 215, 215))
        im = apply_look(source, n)[len(source) // 2]
        canvas.paste(im.resize((T, T), Image.LANCZOS), (4 + c * (T + 4), y + 14))
    canvas.save(path)
    return path


if __name__ == "__main__":
    from PIL import Image

    src_dir = ROOT / "output" / "film" / "checkpoint_ab" / "DreamShaper8_none"
    fs = sorted(src_dir.glob("f*.png"))
    if not fs:
        print(f"no source frames in {src_dir}")
        raise SystemExit(1)
    source = [Image.open(f) for f in fs]
    floor = palette_floor(source)

    print("=== one source clip, shared technical baseline ===")
    print(f"  source: {src_dir.name}  ({len(source)} frames)")
    print(f"  palette floor (same clip, no look change): {floor:.2f}\n")

    rows = look_distances(source, floor=floor)
    print(f"  {'transition':<28} {'tech':>7} {'style':>7}   verdict")
    print("  " + "-" * 74)
    for r in rows:
        print(f"  {r['from'] + ' -> ' + r['to']:<28} {r['technical']:>7.2f} "
              f"{r['stylistic']:>7.2f}   {r['verdict']}")

    ok = all(r["technical"] <= gm.TECH_TOL for r in rows)
    legible = sum(1 for r in rows if "LEGIBLE" in r["verdict"])
    print(f"\n  {legible}/{len(rows)} adjacent transitions land in the legible "
          f"band; technical held on {len(rows) if ok else 0}/{len(rows)}")
    print(f"  -> a look set that holds the baseline is "
          f"{'VALID' if ok else 'INVALID'}")
    print(f"\n  wrote {sheet(source)}")
