"""
handheld_ab.py -- the shake comparison a human has to look at.

Handoff 11.1: no metric in this project has ever settled a quality question, so
this builds the side-by-side instead. Motion is the question and a strip cannot
show motion, so the artifact is an animated GIF.

Three panes, one shared base:

    steady    the RAFT-upsampled clip with no shake at all
    before    the generator as it shipped (HANDOFF section 6: binary envelope,
              +-5 px, integer crop). FROZEN COPY below, so this stays
              reproducible after handheld.py is rewritten.
    after     the current handheld.py

Deliberately on the PRE-grade frames. Handoff 10.9 Trap 1: `decomp` is only valid
between renderings of the SAME content, and grading is a per-frame affine that
differs shot to shot. Comparing shake on ungraded frames keeps every pane on
identical content, so any difference you see is the shake and nothing else.

The two shaken panes are built at the DELIVERED length (the exposure sheet's hold
for that shot, 29-58 frames at 16fps), because that is the timeline the audience
sees. The shipped pipeline instead shook the 31-frame clip and then let
`render()` stretch it by up to 1.86x, which shifts every designed frequency down;
`before` reproduces that faithfully, so the timing fix is visible rather than
merely asserted.

    python handheld_ab.py --shots shot01,shot02
    python handheld_ab.py --shots all --every 2      # smaller gifs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
FILM = ROOT / "output" / "film" / "shadows"
RAW = FILM / "raw"
OUT = ROOT / "output" / "film" / "handheld_ab"
FPS = 16

CANVAS = (16, 16, 20)
INK = (235, 235, 235)
DIM = (150, 150, 158)
ACCENT = {"steady": (120, 200, 255), "before": (240, 150, 90),
          "after": (140, 235, 150)}


# --------------------------------------------------------------------------- #
# RAFT x2 -- same operation make_film performs, without its file coupling
# --------------------------------------------------------------------------- #

def raft_x2(frames: Sequence[Image.Image], cache: Path,
            verbose: bool = True) -> List[Image.Image]:
    fs = sorted(cache.glob("f*.png")) if cache.exists() else []
    if len(fs) >= 2 * len(frames) - 1:
        return [Image.open(f).convert("RGB") for f in fs]

    from supplement import to_image, to_tensor, warp
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
    from torchvision.transforms.functional import resize
    import torch
    import torch.nn.functional as F

    H, W = frames[0].size[1], frames[0].size[0]
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=False).eval()
    transforms = weights.transforms()

    def flow(a, b):
        ia = resize((a * 255).to(torch.uint8), [256, 256])
        ib = resize((b * 255).to(torch.uint8), [256, 256])
        ba, bb = transforms(ia, ib)
        with torch.no_grad():
            f = model(ba, bb)[-1]
        if 256 != W:
            f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            f = f * (W / 256)
        return f

    t0 = time.time()
    tens = [to_tensor(f) for f in frames]
    out = [frames[0]]
    for i in range(len(frames) - 1):
        a, b = tens[i], tens[i + 1]
        f_ab, f_ba = flow(a, b), flow(b, a)
        mix = warp(a, f_ab * 0.5) * 0.5 + warp(b, f_ba * 0.5) * 0.5
        out.append(to_image(mix))
        out.append(frames[i + 1])

    cache.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(out):
        f.save(cache / f"f{i:03d}.png")
    if verbose:
        print(f"    raft x2: {len(frames)} -> {len(out)} frames in "
              f"{time.time()-t0:.1f}s (cached)")
    return out


# --------------------------------------------------------------------------- #
# what shipped. FROZEN COPY -- do not "fix" this, it is the control.
# --------------------------------------------------------------------------- #

def _legacy_value_noise(n: int, octaves: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=np.float64)
    amp, total, freq = 1.0, 0.0, 1.0
    for _ in range(octaves):
        k = max(2, int(np.ceil(freq)) + 1)
        knots = rng.normal(0, 1, k)
        x = np.linspace(0, k - 1, n)
        i = np.floor(x).astype(int).clip(0, k - 2)
        t = x - i
        s = t * t * (3 - 2 * t)
        out += amp * ((1 - s) * knots[i] + s * knots[i + 1])
        total += amp
        amp *= 0.5
        freq *= 2.3
    return np.clip(out / (total + 1e-9), -1, 1)


def _legacy_offsets(n, amount, drift, jitter, seed):
    slow = _legacy_value_noise(n, 3, seed)
    fast = _legacy_value_noise(n, 5, seed + 101)
    vert = _legacy_value_noise(n, 3, seed + 202)
    rot = _legacy_value_noise(n, 2, seed + 303)
    return (amount * (drift * slow + jitter * fast),
            amount * (drift * vert + jitter * fast * 0.8),
            amount * 0.12 * rot)


def _legacy_envelope(n, seed=0, active_frac=0.35, peak=1.0):
    rng = np.random.default_rng(seed)
    env = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        i += int(rng.integers(4, 14))
        run = max(3, int(rng.integers(4, max(5, int(n * active_frac)))))
        ramp = np.sin(np.linspace(0, np.pi, run)) ** 1.5 * peak
        env[i:i + run] = ramp[: max(0, min(run, n - i))]
        i += run
    return env


def legacy_shake(frames: Sequence[Image.Image], amount=5.0, drift=0.8,
                 jitter=0.3, seed=3, margin=1.04) -> List[Image.Image]:
    """Verbatim behaviour of the pre-rework apply_handheld, incl. int() crop."""
    n = len(frames)
    W, H = frames[0].size
    dx, dy, dth = _legacy_offsets(n, amount, drift, jitter, seed)
    env = _legacy_envelope(n, seed=seed, active_frac=0.35)
    dx, dy, dth = dx * env, dy * env, dth * env
    w2, h2 = int(W * margin), int(H * margin)
    out = []
    for i, f in enumerate(frames):
        big = f.convert("RGB").resize((w2, h2), Image.LANCZOS)
        big = big.rotate(dth[i], resample=Image.BILINEAR, center=(w2 / 2, h2 / 2))
        cx = float(np.clip((w2 - W) / 2 + dx[i], 0, w2 - W))
        cy = float(np.clip((h2 - H) / 2 + dy[i], 0, h2 - H))
        out.append(big.crop((int(cx), int(cy), int(cx) + W, int(cy) + H)))
    return out


# --------------------------------------------------------------------------- #
# the shared timeline
# --------------------------------------------------------------------------- #

def shot_meta() -> Dict[str, Dict]:
    """Per shot: delivered hold, framing scale, motion kind, camera name.

    Read from the film's own stats.json rather than guessed, so the A/B shakes
    each shot with the character that shot would actually have been given.
    """
    p = FILM / "stats.json"
    if not p.exists():
        return {}
    meta = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for i, s in enumerate(meta.get("shots", [])):
        out[f"shot{i+1:02d}"] = {
            "hold": max(1, round(s["hold_s"] * FPS)),
            "scale": s.get("scale", ""),
            "kind": s.get("kind", "none"),
            "camera": s.get("camera", "none"),
            "seconds": float(s.get("hold_s", 0.0)),
        }
    return out


def shots_of(spec: str) -> List[str]:
    have = sorted(d.name for d in RAW.glob("shot*") if d.is_dir())
    if spec == "all":
        return have
    return [s.strip() for s in spec.split(",") if s.strip()]


def stretch(src: Sequence[Image.Image], length: int) -> List[Image.Image]:
    """The delivered timeline, from the one definition of it (handheld.stretch_to).

    `before` must reproduce the SHIPPED timeline, so it shakes the 31-frame clip
    and only then stretches -- which is exactly the ordering bug the rework fixes.
    """
    import handheld
    return handheld.stretch_to(src, length)


# --------------------------------------------------------------------------- #
# the panes
# --------------------------------------------------------------------------- #

def build_panes(shot: str, frames16: Sequence[Image.Image], meta: Dict,
                verbose: bool = True):
    """steady / before / after, all from the same RAFT base frames."""
    hold = int(meta["hold"])
    base = raft_x2(frames16, OUT / "raft" / shot, verbose)
    panes = {"steady": stretch(base, hold)}

    i = int(shot.replace("shot", "")) - 1
    panes["before"] = stretch(
        legacy_shake(base, amount=5.0, drift=0.8, jitter=0.3, seed=i + 3), hold)

    spec = None
    try:
        import handheld
        spec = handheld.spec_for_shot(meta.get("scale", ""), meta.get("kind", "none"),
                                      meta.get("seconds", hold / FPS),
                                      seed=i + 3, fps=FPS)
        panes["after"] = handheld.apply_handheld(
            stretch(base, hold), spec, fps=FPS)
    except (ImportError, AttributeError) as exc:
        if verbose:
            print(f"    (no `after` pane yet: {exc})")
    return panes, spec


def gif(panes: Dict[str, List[Image.Image]], path: Path, size: int,
        every: int, labels: Dict[str, str]) -> Path:
    names = [k for k in ("steady", "before", "after") if k in panes]
    n = min(len(panes[k]) for k in names)
    frames = []
    head = 30
    for i in range(0, n, every):
        canvas = Image.new("RGB", (len(names) * size + 4, size + head), CANVAS)
        dr = ImageDraw.Draw(canvas)
        dr.text((4, 4), f"frame {i:02d}/{n-1}   @{FPS}fps", fill=DIM)
        dr.text((4, 16), labels.get("title", ""), fill=INK)
        for c, k in enumerate(names):
            dr.text((4 + c * size, head - 12), labels.get(k, k),
                    fill=ACCENT.get(k, INK))
            canvas.paste(panes[k][i].resize((size, size), Image.LANCZOS),
                         (2 + c * size, head))
        frames.append(canvas)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=max(20, int(1000 / FPS)), loop=0, optimize=True)
    return path


def _smooth(v: np.ndarray, k: int = 4) -> np.ndarray:
    ker = np.ones(k) / k
    return np.convolve(np.pad(v, k // 2, mode="edge"), ker, mode="valid")


def plot_paths(panes: Dict[str, List[Image.Image]], shot: str, spec,
               title: str) -> Path:
    """The camera's own path -- where the jerk is legible at a glance.

    Row 4 is the amplitude envelope: the binary envelope steps 0 -> full in one
    frame (0.81 in a single step), which is the jerk. Row 5 is the second
    difference of the LOW-PASS path, so the deliberate tremor is not mistaken
    for a discontinuity.
    """
    names = [k for k in ("steady", "before", "after") if k in panes]
    W, Ph = 980, 128
    keys = [("dx", "px"), ("dy", "px"), ("roll_deg", "deg"),
            ("env", ""), ("jerk", "px")]
    canvas = Image.new("RGB", (W, len(keys) * Ph + 40), CANVAS)
    dr = ImageDraw.Draw(canvas)
    dr.text((6, 4), title, fill=INK)
    dr.text((6, 18), "camera path: dx, dy, roll   |   amplitude envelope "
                     "(the step is the jerk)   |   |d2| of the low-pass path",
            fill=DIM)

    import handheld
    i = int(shot.replace("shot", "")) - 1
    series: Dict[str, Dict[str, np.ndarray]] = {}
    for k in names:
        n = len(panes[k])
        if k == "before":
            dx, dy, r = _legacy_offsets(n, 5.0, 0.8, 0.3, i + 3)
            e = _legacy_envelope(n, seed=i + 3, active_frac=0.35)
            dx, dy, r = dx * e, dy * e, r * e
        elif k == "steady":
            dx = dy = r = np.zeros(n)
            e = np.ones(n)
        else:
            p = handheld.shake_path(n, spec, FPS)
            dx, dy, r, e = p["dx"], p["dy"], p["roll_deg"], p["env"]
        series[k] = {"dx": dx, "dy": dy, "roll_deg": r, "env": e,
                     "jerk": np.abs(np.diff(_smooth(dx), 2))}

    for row, (key, unit) in enumerate(keys):
        y0 = 40 + row * Ph
        v_all = np.concatenate([series[k][key] for k in names])
        if key == "env":
            lim = 1.0
            base = y0 + Ph - 18
            v_all = v_all
        elif key == "jerk":
            lim = max(0.15, float(np.abs(v_all).max()))
            base = y0 + Ph - 18
        else:
            lim = max(0.5, float(np.abs(v_all).max()))
            base = y0 + Ph / 2
        if key in ("dx", "dy", "roll_deg"):
            dr.line((44, base, W - 10, base), fill=(60, 60, 68))
        for k in names:
            v = series[k][key]
            span = (Ph - 46) * (0.5 if key in ("dx", "dy", "roll_deg") else 1.0)
            y = base - v / lim * span
            x = np.linspace(48, W - 12, len(v))
            dr.line(list(zip(x.tolist(), y.tolist())),
                    fill=ACCENT.get(k, INK), width=2)
        dr.text((4, y0 + 2), key, fill=DIM)
        dr.text((4, y0 + Ph - 14), f"+-{lim:.2f}{unit}", fill=DIM)
        if key == "env":
            dr.text((70, y0 + 2), "0 = shake switched off (the old failure)",
                    fill=(200, 140, 120))
    return _save(canvas, OUT / f"path_{shot}.png")


def _save(img: Image.Image, p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="shot01,shot02")
    ap.add_argument("--size", type=int, default=340)
    ap.add_argument("--every", type=int, default=1,
                    help="use every Nth frame (2 halves the gif size)")
    a = ap.parse_args()

    hs = shot_meta()
    made = []
    for shot in shots_of(a.shots):
        d = RAW / shot
        fs = sorted(d.glob("f*.png"))
        if not fs:
            print(f"  no frames in {d}")
            continue
        meta = hs.get(shot) or {"hold": 2 * len(fs) - 1, "scale": "",
                                "kind": "none", "seconds": 0.0}
        frames16 = [Image.open(f).convert("RGB") for f in fs]
        hold = int(meta["hold"])
        print(f"\n=== {shot}: {len(frames16)} raw frames -> hold {hold} frames "
              f"({hold/FPS:.2f}s delivered), scale={meta['scale']} "
              f"kind={meta['kind']} camera={meta['camera']} ===")
        panes, spec = build_panes(shot, frames16, meta)
        print(f"    panes {dict((k, len(v)) for k, v in panes.items())}")
        if spec is not None:
            import handheld
            print("    after: " + handheld.report(spec) + f"  zoom {_zoom(panes, spec):.3f}")

        cap = (f"shadows/{shot}   hold {hold}f ({hold/FPS:.2f}s)   "
               f"same RAFT x2 base, no grading   {meta['scale']} / {meta['kind']}")
        labels = {
            "title": cap,
            "steady": "steady -- no shake",
            "before": "BEFORE: binary envelope, int() crop",
            "after": "AFTER: continuous envelope, sub-pixel, roll",
        }
        g = gif(panes, OUT / f"ab_{shot}.gif", a.size, a.every, labels)
        p = plot_paths(panes, shot, spec, f"shadows/{shot} camera path   "
                                          f"{meta['scale']} / {meta['kind']}")
        print(f"    wrote {g.relative_to(ROOT)}")
        print(f"    wrote {p.relative_to(ROOT)}")
        made.append((shot, g, p))

    if made:
        print("\n  watch the gifs; the plots say where the motion lives")
    print("DONE")


def _zoom(panes, spec) -> float:
    import handheld
    n = len(panes.get("after", panes.get("steady")))
    W, H = panes["after"][0].size
    return handheld.zoom_needed(W, H, handheld.shake_path(n, spec, FPS))


if __name__ == "__main__":
    main()
