"""
watermark_check.py -- is the SHUTTERSTOCK watermark actually there?

The handoff records the watermark as solved by swapping the base checkpoint, and
`model_select.py` therefore hard-gates sd15 out of the candidate pool. But the
frame the verdict rests on (`output/film/watermark_ab/none/f003.png`, byte-
identical to `checkpoint_ab/SD15_base_none/f003.png`) does not show a watermark at
full size, and neither does the contact sheet built to demonstrate it.

A semi-transparent watermark is exactly the kind of thing that survives downscaling
into a contact sheet as nothing, so "I cannot see it" is not evidence either way.
This amplifies the high-frequency residual, where a faint overlay becomes obvious,
and pairs each checkpoint side by side so the comparison is like for like.

    python watermark_check.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "film" / "watermark_probe"

SOURCES = {
    "sd15": ROOT / "output" / "film" / "checkpoint_ab" / "SD15_base_none",
    "dreamshaper8": ROOT / "output" / "film" / "checkpoint_ab" / "DreamShaper8_none",
}


def amplify(img: Image.Image, sigma: float = 3.0, gain: float = 9.0) -> Image.Image:
    """High-pass the frame and blow up the residual.

    Faint text edges are a few grey levels deep and vanish in a thumbnail; after
    a 9x gain on the high-frequency band they are unmistakable.
    """
    import cv2

    a = np.asarray(img.convert("L"), dtype=np.float32)
    lo = cv2.GaussianBlur(a, (0, 0), sigma)
    hp = (a - lo) * gain
    return Image.fromarray(np.clip(128 + hp, 0, 255).astype(np.uint8))


def text_likeness(img: Image.Image) -> float:
    """Crude 'is there structured high-frequency content' score.

    A watermark is thin, high-contrast strokes; foliage is broadband. Measured as
    the fraction of high-pass energy sitting in the extreme tail -- strokes
    concentrate energy, texture does not.
    """
    import cv2

    a = np.asarray(img.convert("L"), dtype=np.float32)
    hp = np.abs(a - cv2.GaussianBlur(a, (0, 0), 2.0))
    if hp.max() <= 0:
        return 0.0
    n = hp / hp.max()
    return float((n > 0.55).mean())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sets = {}
    for name, d in SOURCES.items():
        fs = sorted(d.glob("f*.png"))
        sets[name] = [Image.open(f).convert("RGB") for f in fs]
        if not sets[name]:
            print(f"  {name}: no frames at {d}")
    if not all(sets.values()):
        print("need frames from both checkpoints")
        return 1

    print("=== high-frequency tail fraction (structured strokes) ===")
    for name, frames in sets.items():
        scores = [text_likeness(f) for f in frames]
        print(f"  {name:<14} mean {np.mean(scores):.4f}  "
              f"max {np.max(scores):.4f}  over {len(frames)} frames")

    names = list(sets)
    n = min(len(sets[k]) for k in names)
    T = 256
    cols = min(4, n)
    canvas = Image.new("RGB", (cols * (T + 4) + 4,
                               len(names) * 2 * (T + 22) + 26), (16, 16, 20))
    dr = ImageDraw.Draw(canvas)
    dr.text((5, 6), "each pair: as rendered  /  high-frequency residual x9 "
                    "(a watermark appears as legible strokes)", fill=(235, 235, 235))
    y = 26
    for name in names:
        for row, (label, fn) in enumerate([("as rendered", None),
                                           ("residual x9", amplify)]):
            dr.text((5, y), f"{name} -- {label}", fill=(215, 215, 215))
            for c in range(cols):
                im = sets[name][c]
                im = im if fn is None else fn(im)
                canvas.paste(im.resize((T, T), Image.LANCZOS),
                             (4 + c * (T + 4), y + 14))
            y += T + 22
    p = OUT / "residual.png"
    canvas.save(p)
    print(f"\n  wrote {p}")
    print("  Look for legible letterforms. Nothing legible = no watermark in "
          "these frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
