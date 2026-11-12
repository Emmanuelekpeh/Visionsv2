#!/usr/bin/env python3
"""frame_strip.py -- watch the film as frames, not stills.

WHY. Contact sheets pick one flattering frame per shot. They cannot show the
things that make a delivered film look wrong: duplicated/interpolated frames,
slow-motion mush in long holds, frozen takes, or the actual amount of motion
between cuts. This reads the mp4 frame by frame and builds:

  * <name>_film_strip.png  -- every Nth frame of the whole film, labelled
  * <name>_holds.png       -- EVERY frame of the longest holds, in a grid
  * per-shot motion table  -- consecutive-frame difference, duplicate frames

    python frame_strip.py --store squid --render film
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2                                # noqa: E402
import numpy as np                        # noqa: E402
from PIL import Image, ImageDraw          # noqa: E402


def sheet(thumbs, out, cols, w, label):
    h = int(w * thumbs[0].size[1] / thumbs[0].size[0])
    rows = (len(thumbs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (w + 3) + 3, rows * (h + 18) + 3),
                       (18, 18, 20))
    d = ImageDraw.Draw(canvas)
    for i, im in enumerate(thumbs):
        r, c = divmod(i, cols)
        x, y = 3 + c * (w + 3), 3 + r * (h + 18)
        canvas.paste(im.resize((w, h), Image.LANCZOS), (x, y))
        d.text((x + 2, y + h + 2), label(i), fill=(220, 220, 220))
    canvas.save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--render", default="film")
    ap.add_argument("--out", default=None)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--top-holds", type=int, default=3)
    ap.add_argument("--shots", default=None,
                    help="comma list of 1-based shot numbers to grid frame-by-frame")
    a = ap.parse_args()

    import film_store as fs
    project = fs.load(a.store)
    plan = fs.export_plan(project)
    fps = plan["fps"]
    order = plan["order"]
    mp4 = fs.renders_dir(a.store) / a.render / "film.mp4"
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # shot -> frame range
    ranges, start = [], 0
    for o in order:
        n = int(o["hold_frames"])
        ranges.append((o["id"], start, start + n))
        start += n
    holds = sorted(ranges, key=lambda r: r[2] - r[1], reverse=True)
    if a.shots:
        wanted = {int(x) for x in a.shots.split(",") if x.strip()}
        watch = {sid for i, (sid, _, _) in enumerate(ranges, 1) if i in wanted}
    else:
        watch = {sid for sid, s, e in holds[:a.top_holds]}

    full, picks = [], {sid: [] for sid in watch}
    diffs = {sid: [] for sid, _, _ in ranges}
    prev = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        if idx % a.step == 0:
            full.append(im.copy())
        sid = next((s for s, a0, b0 in ranges if a0 <= idx < b0), None)
        if sid in watch:
            picks[sid].append(im.copy())
        small = cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_BGR2GRAY)
        if prev is not None and sid is not None:
            diffs[sid].append(float(np.abs(small.astype(np.int16)
                                           - prev.astype(np.int16)).mean()))
        prev = small
        idx += 1
    cap.release()

    out = Path(a.out) if a.out else ROOT / "output" / "inspect" / \
        f"{a.store}_{a.render}"
    out.mkdir(parents=True, exist_ok=True)
    base = f"{a.store}_{a.render}"
    film_strip = sheet(full, out / f"{base}_film_strip.png", cols=13, w=140,
                       label=lambda i: f"f{i * a.step}")
    hold_files = []
    for sid in sorted(watch, key=lambda s: dict((x[0], x[1]) for x in ranges)[s]):
        s0 = next(n for n, b, _ in ranges if n == sid)
        hold_files.append(sheet(
            picks[sid], out / f"{base}_hold_{sid}.png", cols=10, w=110,
            label=lambda i, sid=sid: f"{sid} f{i}"))

    print(f"{mp4}: {total} frames @ {fps}fps")
    print(f"full strip: {film_strip}")
    for f in hold_files:
        print(f"hold grid:  {f}")
    print("\nper-shot motion (consecutive frame diff, 128px grey):")
    print(f"{'shot':<6} {'frames':>6} {'mean':>7} {'max':>7} {'dups':>6}")
    for sid, s0, e0 in ranges:
        d = diffs[sid]
        if not d:
            continue
        dups = sum(1 for x in d if x < 0.35)
        print(f"{sid:<6} {len(d) + 1:>6} {np.mean(d):7.2f} "
              f"{np.max(d):7.2f} {dups:6d}  ({100 * dups / max(1, len(d)):.0f}% frozen)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
