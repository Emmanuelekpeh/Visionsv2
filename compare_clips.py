"""
compare_clips.py -- put two configurations side by side so a human can judge.

No metric in this project has ever settled a quality question. `align` said
4-step and 12-step were identical (0.3683 vs 0.3680); `global_share` and `loc10`
disagreed in direction on the same pair; `heat.png` rendered a well-separated clip
as a blob and a badly-separated one as a blob. Every time the decision mattered,
someone had to look.

So this makes looking cheap. For each pair it writes:

  <pair>.gif    both clips side by side, looping, at a size that shows motion.
                Motion is the thing strips cannot show and the thing that matters.
  <pair>.png    four matched frames from each, plus the metrics.

Animated, because "is this watchable" is a question about movement over time, and
that is the question metrics keep failing to answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "compare"

# where a clip id lives
def clip_dir(cid: str) -> Path:
    p = ROOT / "output" / "subject_test" / cid
    if p.exists():
        return p
    return ROOT / "output" / "film" / cid


def load_frames(cid: str) -> List[Image.Image]:
    d = clip_dir(cid)
    fs = sorted(d.glob("f*.png"))
    return [Image.open(f).convert("RGB") for f in fs]


def metrics(cid: str) -> Dict:
    p = ROOT / "output" / "subject_test" / "metrics.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("clips", {}).get(cid, {})
        except Exception:
            return {}
    return {}


def label_for(cid: str, m: Dict) -> str:
    cfg = ""
    if m:
        lightning = m.get("lightning")
        steps = m.get("steps")
        if steps is not None:
            cfg = f"  {steps}-step{'' if lightning else ' no-lightning'}"
    bits = []
    if m:
        bits.append(f"align {m.get('align', 0):.3f}")
        bits.append(f"global {m.get('global_share', 0):.2f}")
        bits.append(f"mag {m.get('mag', 0):.1f}")
        bits.append(f"loc10 {m.get('loc10', 0):.3f}")
    return f"{cid}{cfg}" + ("   " + "  ".join(bits) if bits else "")


def pair_artifacts(a: str, b: str, size: int = 240, every: int = 1) -> Tuple[Path, Path]:
    fa, fb = load_frames(a), load_frames(b)
    if not fa or not fb:
        raise FileNotFoundError(f"missing frames for {a} or {b}")
    n = min(len(fa), len(fb))
    ma, mb = metrics(a), metrics(b)

    # side-by-side animation: motion is the thing that decides
    frames = []
    for i in range(0, n, every):
        canvas = Image.new("RGB", (2 * size + 8, size + 26), (16, 16, 20))
        dr = ImageDraw.Draw(canvas)
        dr.text((4, 4), f"A: {a}", fill=(235, 235, 235))
        dr.text((size + 8, 4), f"B: {b}", fill=(235, 235, 235))
        canvas.paste(fa[i].resize((size, size), Image.LANCZOS), (2, 22))
        canvas.paste(fb[i].resize((size, size), Image.LANCZOS), (size + 6, 22))
        frames.append(canvas)
    OUT.mkdir(parents=True, exist_ok=True)
    gif = OUT / f"{a}__VS__{b}.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=140, loop=0, optimize=True)

    # strip + metrics panel
    picks = [round(i * (n - 1) / 3) for i in range(4)]
    T = size
    canvas = Image.new("RGB", (4 * T + 14, 2 * (T + 24) + 34), (16, 16, 20))
    dr = ImageDraw.Draw(canvas)
    dr.text((4, 5), f"{a}   vs   {b}", fill=(240, 240, 240))
    y = 24
    for cid, fr, m in ((a, fa, ma), (b, fb, mb)):
        dr.text((5, y), label_for(cid, m), fill=(220, 220, 220))
        for k, pi in enumerate(picks):
            canvas.paste(fr[pi].resize((T, T), Image.LANCZOS),
                         (4 + k * (T + 2), y + 20))
        y += T + 24
    png = OUT / f"{a}__VS__{b}.png"
    canvas.save(png)
    return gif, png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True,
                    help="a:b,c:d  -- clip ids from output/subject_test/")
    ap.add_argument("--size", type=int, default=240)
    ap.add_argument("--every", type=int, default=1,
                    help="use every Nth frame in the gif (2 halves the size)")
    a = ap.parse_args()

    for spec in a.pairs.split(","):
        if ":" not in spec:
            print(f"  skipping {spec!r}: expected id_a:id_b")
            continue
        a_id, b_id = (s.strip() for s in spec.split(":", 1))
        try:
            gif, png = pair_artifacts(a_id, b_id, a.size, a.every)
            print(f"  wrote {png.name}")
            print(f"        {gif.name}  <-- watch this one, motion is the point")
        except FileNotFoundError as exc:
            print(f"  {exc}")


if __name__ == "__main__":
    main()
