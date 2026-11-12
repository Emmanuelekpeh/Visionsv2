#!/usr/bin/env python3
"""
camera_ab_visual.py -- the pixel-level half of the camera-policy A/B.

`camera_ab.py` compares DECISIONS (which shots move, reversals, repairs). This
tool compares the two DELIVERED takes of the same shot, differing only in the
camera decision, and answers the only question that cannot be answered from text:
did the move actually happen, and did it cost image quality?

It does three things:

1. **Measures the camera motion that reached the pixels.** Phase correlation
   between consecutive delivered frames gives a luma-shift estimate; summed over
   the clip it is the net travel. A tilt-down should show a consistent vertical
   path; `none` should show near-zero net travel (AnimateDiff drift + handheld
   shake are small and zero-mean by construction).
2. **Measures how much the image changes** (mean absolute frame difference), so
   a "camera move" cannot be confused with a scene that simply regenerated worse.
3. **Writes the side-by-side a person must watch** (mp4 + gif) and contact
   sheets for still comparison.

    python camera_ab_visual.py \
        --a productions/_camera_ab/takes/s01/t01 \
        --b productions/_camera_ab/takes/s01/t02 \
        --label-a none --label-b tilt-down \
        --out output/camera_ab/visual

Read it as: phase travel is the MOVE, frame difference is the CHANGE. A good
camera move raises travel without raising frame difference much (the same scene
sliding), and the contact sheets should read as the same shot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
FPS = 16


def load_frames(d: Path) -> List[Image.Image]:
    d = Path(d)
    if (d / "frames").is_dir():            # a take directory, not its frames/ folder
        d = d / "frames"
    files = sorted(d.glob("f*.png")) or sorted(d.glob("*.png"))
    if not files:
        raise SystemExit(f"no frames under {d}")
    return [Image.open(f).convert("RGB") for f in files]


def _phase_travel(frames: Sequence[Image.Image]) -> Dict:
    """Sum of consecutive-frame luma shifts, in output pixels.

    `phaseCorrelate` returns the shift (dx, dy) that maps the first window onto
    the second. Sub-pixel, robust to small exposure changes, and cheap. Handheld
    shake is zero-mean by design, so it cancels in the sum; a designed camera
    move does not.
    """
    dxs, dys = [], []
    prev = cv2.cvtColor(np.array(frames[0]), cv2.COLOR_RGB2GRAY).astype(np.float32)
    for f in frames[1:]:
        cur = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2GRAY).astype(np.float32)
        (dx, dy), response = cv2.phaseCorrelate(prev, cur)
        dxs.append(float(dx))
        dys.append(float(dy))
        prev = cur
    return {
        "dx_per_frame": [round(x, 2) for x in dxs],
        "dy_per_frame": [round(x, 2) for x in dys],
        "net_dx_px": round(sum(dxs), 2),
        "net_dy_px": round(sum(dys), 2),
        "path_px": round(float(sum(abs(x) for x in dxs) + sum(abs(y) for y in dys)), 2),
        "max_abs_dy_px": round(max((abs(y) for y in dys), default=0.0), 2),
        "mean_abs_dy_px": round(float(np.mean(np.abs(dys))) if dys else 0.0, 2),
    }


def _mean_abs_diff(frames: Sequence[Image.Image]) -> float:
    diffs = []
    for a, b in zip(frames, frames[1:]):
        da = np.asarray(a, dtype=np.float32)
        db = np.asarray(b, dtype=np.float32)
        diffs.append(float(np.abs(da - db).mean()))
    return round(float(np.mean(diffs)), 3) if diffs else 0.0


def _image_stats(frames: Sequence[Image.Image]) -> Dict:
    """Luma and sharpness, so a quality change is not read as a camera effect.

    Sharpness is the mean variance of the Laplacian -- crude, but the right
    question for "did the LoRA blur or steady the image", and identical to how
    the project measured stability elsewhere.
    """
    lumas, sharp = [], []
    for f in frames:
        g = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2GRAY)
        lumas.append(float(g.mean()))
        sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    return {"mean_luma": round(float(np.mean(lumas)), 2),
            "sharpness_lap_var": round(float(np.mean(sharp)), 1)}


def _contact(frames: Sequence[Image.Image], path: Path, picks=(0, None, -1),
             size=384) -> None:
    picks = [picks[0], len(frames) // 2 if picks[1] is None else picks[1], picks[2]]
    cells = []
    for i, p in enumerate(picks):
        im = frames[p].resize((size, size), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.text((6, 4), f"frame {p + 1}/{len(frames)}", fill=(255, 255, 0))
        cells.append(im)
    sheet = Image.new("RGB", (size * len(cells), size), (10, 10, 10))
    for i, c in enumerate(cells):
        sheet.paste(c, (i * size, 0))
    sheet.save(path)


def side_by_side(a: Sequence[Image.Image], b: Sequence[Image.Image],
                 path_gif: Path, path_mp4: Path,
                 label_a: str, label_b: str, size=384) -> int:
    n = min(len(a), len(b))
    panels = []
    for i in range(n):
        row = []
        for im, label in ((a[i], label_a), (b[i], label_b)):
            small = im.resize((size, size), Image.LANCZOS)
            d = ImageDraw.Draw(small)
            d.rectangle((0, 0, 220, 22), fill=(0, 0, 0))
            d.text((6, 5), f"{label}  ({i + 1}/{n})", fill=(255, 255, 0))
            row.append(small)
        canvas = Image.new("RGB", (size * 2 + 4, size), (0, 0, 0))
        canvas.paste(row[0], (0, 0))
        canvas.paste(row[1], (size + 4, 0))
        panels.append(canvas)

    # gif at half rate, like make_film's contact gif
    halves = panels[::2]
    halves[0].save(path_gif, save_all=True, append_images=halves[1:],
                   duration=int(1000 / (FPS / 2)), loop=0)

    try:
        from diffusers.utils import export_to_video
        export_to_video(panels, str(path_mp4), fps=FPS)
        return len(panels)
    except Exception as exc:  # noqa: BLE001 -- gif already exists
        print(f"  mp4 failed: {exc}")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Pixel-level A/B of one camera decision")
    ap.add_argument("--a", required=True, help="take dir A (frames/*.png)")
    ap.add_argument("--b", required=True, help="take dir B (frames/*.png)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", default="output/camera_ab/visual")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fa, fb = load_frames(Path(a.a)), load_frames(Path(a.b))
    if len(fa) != len(fb):
        print(f"  !! frame counts differ ({len(fa)} vs {len(fb)}); using the shorter")
    n = min(len(fa), len(fb))
    fa, fb = fa[:n], fb[:n]

    ma, mb = _phase_travel(fa), _phase_travel(fb)
    report = {
        "label_a": a.label_a, "label_b": a.label_b,
        "frames": n, "fps": FPS, "seconds": round(n / FPS, 2),
        "a": {**ma, "mean_abs_frame_diff": _mean_abs_diff(fa), **_image_stats(fa)},
        "b": {**mb, "mean_abs_frame_diff": _mean_abs_diff(fb), **_image_stats(fb)},
    }
    print(f"\n=== camera A/B pixels: {a.label_a} vs {a.label_b} ===")
    print(f"  {n} frames · {n / FPS:.2f}s at {FPS}fps")
    for k in ("net_dy_px", "max_abs_dy_px", "mean_abs_dy_px", "net_dx_px",
              "mean_abs_frame_diff", "mean_luma", "sharpness_lap_var"):
        print(f"  {k:<24} {a.label_a}: {report['a'][k]:>8}   "
              f"{a.label_b}: {report['b'][k]:>8}")

    _contact(fa, out / "contact_a.png")
    _contact(fb, out / "contact_b.png")
    rendered = side_by_side(fa, fb, out / "side_by_side.gif",
                            out / "side_by_side.mp4", a.label_a, a.label_b)
    print(f"  contact sheets: {out / 'contact_a.png'}, {out / 'contact_b.png'}")
    print(f"  gif: {out / 'side_by_side.gif'}"
          + (f"  · mp4: {out / 'side_by_side.mp4'}" if rendered else ""))
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  metrics: {out / 'metrics.json'}")

    dy_a, dy_b = abs(report["a"]["net_dy_px"]), abs(report["b"]["net_dy_px"])
    travel_a = abs(report["a"]["net_dy_px"]) + abs(report["a"]["net_dx_px"])
    travel_b = abs(report["b"]["net_dy_px"]) + abs(report["b"]["net_dx_px"])
    print("\n=== reading ===")
    hi, lo, hi_label = ((travel_a, travel_b, a.label_a)
                        if travel_a >= travel_b else
                        (travel_b, travel_a, a.label_b))
    if hi > 2 * max(1.0, lo):
        print(f"  the camera decision MOVED the frame: '{hi_label}' travelled "
              f"{hi:.1f}px net vs {lo:.1f}px. Watch whether it helps or jerks.")
    else:
        print(f"  no clear move reached the pixels ({hi:.1f}px vs {lo:.1f}px); "
              f"check the render log before trusting the policy.")
    print("  sharpness is NOT directly comparable between two generated takes: "
          "the move brings different content into frame, and Laplacian variance "
          "follows content. Read the contact sheets, and compare a second pair "
          "before calling a quality difference causal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
