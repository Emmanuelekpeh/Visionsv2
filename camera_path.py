#!/usr/bin/env python3
"""
camera_path.py -- a motivated camera move that ends in a HOLD, applied in finishing.

WHY THIS EXISTS. A motion LoRA bakes movement into generation: it runs for the
whole 31-frame clip, so the shot is "all move" and there is no time to settle.
That is why the rendered tilt-down could not do what a film tilt does -- travel,
then hold on the information. A camera path applied to the DELIVERED frames can
do that: ease in, travel, settle for the last part of the shot. It also avoids
re-conditioning the diffusion model, so it costs no image detail (measured: the
LoRA tilt take came back at 38 Laplacian-var against 77 for `none`).

It is the same projective-warp family `handheld.py` uses: pre-scale for headroom,
then transform each frame. No optical flow, no new pixels.

    python camera_path.py --frames output/film/ab_none --out output/film/ab_procedural \
        --move tilt-down --amount 69 --settle 0.6

Moves: tilt-down/up, pan-left/right, zoom-in/out, push-in (zoom with a small
vertical bias is not implemented -- keep one variable at a time).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from PIL import Image

MOVES = ("tilt-down", "tilt-up", "pan-left", "pan-right", "zoom-in", "zoom-out")


def _ease(t: float) -> float:
    """Smoothstep: zero velocity at both ends. A move should not start or stop
    with a visible jerk; the settle at the end is what makes it read as a camera."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _affine_for(move: str, progress: float, amount: float, size: int,
                frames: int) -> tuple:
    """(matrix, needed) for PIL.Image.AFFINE at one frame.

    The image is pre-scaled by `margin` about its centre so a translation has
    pixels to reveal rather than a black edge. `amount` is in output pixels of
    travel at 512; the margin is derived from it, not guessed.
    """
    w = h = size
    # HEADROOM, derived not guessed: after scaling by m about the centre, the
    # content extends (m-1)*size/2 beyond each edge, so a translation of
    # `amount` needs (m-1)*size/2 >= amount. Anything less reveals a black band.
    m = 1.0 + 2.0 * (abs(amount) + 8.0) / min(w, h)
    e = _ease(progress)
    dx = dy = 0.0
    scale = m
    if move == "tilt-down":
        dy = -amount * e                     # scene moves up = camera tilts down
    elif move == "tilt-up":
        dy = amount * e
    elif move == "pan-left":
        dx = amount * e                      # scene moves right = camera pans left
    elif move == "pan-right":
        dx = -amount * e
    elif move == "zoom-in":
        scale = m + 0.10 * e
    elif move == "zoom-out":
        scale = m + 0.10 * (1.0 - e)
    else:
        raise ValueError(f"unknown move {move!r}; expected {MOVES}")
    # scale about the centre, then translate; PIL maps output -> input
    tx = (w / 2.0) * (1.0 - scale) + dx
    ty = (h / 2.0) * (1.0 - scale) + dy
    # AFFINE inverse map: output pixel -> source pixel
    inv = 1.0 / scale
    return (inv, 0.0, -tx * inv, 0.0, inv, -ty * inv), scale


def apply_path(frames: Sequence[Image.Image], move: str, amount: float = 69.0,
               settle: float = 0.6) -> List[Image.Image]:
    """Apply one motivated move with a settle/hold to a delivered clip.

    `settle` is the fraction of the clip used to travel; the remainder holds.
    At settle=0.6 on a 2 s shot the move takes 1.2 s and settles for 0.8 s --
    which is what makes the cut that follows readable as "after the move" rather
    than "mid-move".
    """
    frames = list(frames)
    n = len(frames)
    if n < 2:
        return frames
    w, h = frames[0].size
    assert w == h, "camera_path expects square delivered frames"
    out: List[Image.Image] = []
    for i, f in enumerate(frames):
        progress = 1.0 if n == 1 else i / (n - 1)
        travel = min(1.0, progress / max(1e-6, settle))
        coeffs, _scale = _affine_for(move, travel, amount, w, n)
        out.append(f.transform((w, h), Image.AFFINE, coeffs,
                              resample=Image.BICUBIC))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply a motivated camera path with settle")
    ap.add_argument("--frames", required=True, help="dir of f*.png (delivered frames)")
    ap.add_argument("--out", required=True, help="dir to write f*.png")
    ap.add_argument("--move", default="tilt-down", choices=MOVES)
    ap.add_argument("--amount", type=float, default=69.0,
                    help="travel in output pixels over the move (512-wide frame)")
    ap.add_argument("--settle", type=float, default=0.6,
                    help="fraction of the clip used to travel; rest holds")
    a = ap.parse_args()

    src = sorted(Path(a.frames).glob("f*.png"))
    if not src:
        raise SystemExit(f"no f*.png under {a.frames}")
    frames = [Image.open(p).convert("RGB") for p in src]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    moved = apply_path(frames, a.move, a.amount, a.settle)
    for i, f in enumerate(moved):
        f.save(out / f"f{i:04d}.png")
    hold_frames = int(round(len(frames) * (1.0 - a.settle)))
    print(f"{a.move} {a.amount:.0f}px over {a.settle:.0%} of {len(frames)} frames, "
          f"then {hold_frames} held -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
