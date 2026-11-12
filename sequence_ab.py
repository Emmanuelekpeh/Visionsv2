#!/usr/bin/env python3
"""
sequence_ab.py -- compare two cuts of the same 4-shot action, after they are rendered.

The pacing corpus study says human short video holds a place and varies shot
length; the montage arrangement does neither. But only watching the two cuts back
to back can say whether the coverage cut actually reads as one continuous action.
This tool measures the parts that can be measured, and builds the thing to watch.

Measured per cut:
  * structure: shots, scenes, scene cuts, average shot length, shot-length CV,
    share of holds under 1 s / over 5 s
  * cross-cut continuity (the point of coverage):
      - CLIP cosine between the last frame of shot i and the first frame of
        shot i+1 (higher = the next shot looks like the same place/thing)
      - mean luma jump and colour-histogram jump across the cut (lower = smoother)
      - motion-direction agreement: the direction of travel just before the cut
        vs just after it (a match-on-action cut keeps direction; a jump to new
        scenery does not)
  * a side-by-side mp4/gif with shot boundaries labelled, plus a per-shot grid

    python sequence_ab.py --current _seq_current --coverage _seq_coverage

Reads `productions/<slug>/renders/<name>/stats.json` for the true cut times and
`output/film/<name>/f*.png` for the delivered frames.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import continuity  # noqa: E402
import film_store as fs  # noqa: E402


def _render_for(slug: str) -> Tuple[Dict, List[Path], Path]:
    project = fs.load(slug)
    rec = project.get("deliverable") or {}
    name = rec.get("render")
    renders = ROOT / "productions" / slug / "renders"
    if not name or not (renders / name).exists():
        cands = [p for p in renders.iterdir() if p.is_dir()] if renders.exists() else []
        if not cands:
            raise SystemExit(f"{slug}: no render found")
        name = sorted(cands)[-1].name
    stats_path = renders / name / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    frame_dir = ROOT / "output" / "film" / name
    frames = sorted(frame_dir.glob("f*.png"))
    if not frames:
        raise SystemExit(f"{name}: no delivered frames in {frame_dir}")
    return project, frames, stats_path


def _structure(project: Dict, stats: Dict) -> Dict:
    shots = fs.shots_in_order(project)
    holds = [float(s.get("seconds") or 2.5) for s in shots]
    mean = sum(holds) / len(holds)
    sd = (sum((x - mean) ** 2 for x in holds) / len(holds)) ** 0.5
    scenes = [str(s.get("scene") or "?") for s in shots]
    return {
        "shots": len(shots),
        "scenes": len(set(scenes)),
        "scene_cuts": sum(1 for i in range(1, len(scenes))
                          if scenes[i] != scenes[i - 1]),
        "duration_s": round(sum(holds), 2),
        "asl_s": round(mean, 2),
        "hold_cv": round(sd / max(1e-9, mean), 2),
        "under_1s_pct": round(100 * sum(1 for h in holds if h < 1) / len(holds), 1),
        "over_5s_pct": round(100 * sum(1 for h in holds if h > 5) / len(holds), 1),
        "holds": [round(h, 2) for h in holds],
        "purposes": [s.get("purpose") or "?" for s in shots],
        "links": [s.get("link") or "?" for s in shots],
        "cameras": [s.get("camera") or "none" for s in shots],
    }


def _clip_scorer():
    try:
        from motion_metrics import ClipScorer
        return ClipScorer()
    except Exception as exc:  # noqa: BLE001
        print(f"  !! CLIP unavailable ({type(exc).__name__}); "
              f"cross-cut colour continuity still measured")
        return None


def _motion_vector(frames: Sequence[Image.Image]) -> Tuple[float, float]:
    """Mean per-frame shift over a short run of delivered frames (phase correlation)."""
    if len(frames) < 2:
        return 0.0, 0.0
    dxs, dys = [], []
    prev = cv2.cvtColor(np.array(frames[0]), cv2.COLOR_RGB2GRAY).astype(np.float32)
    for f in frames[1:]:
        cur = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2GRAY).astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(prev, cur)
        dxs.append(float(dx))
        dys.append(float(dy))
        prev = cur
    return float(np.mean(dxs)), float(np.mean(dys))


def cut_continuity(project: Dict, frames: List[Path], stats: Dict,
                   scorer=None) -> Dict:
    # NOT named `continuity`: that shadowed the `continuity` module imported above, so
    # the three calls below raised AttributeError on the function object.
    shots = stats.get("shots") or []
    by_shot: Dict[str, List[Path]] = {}
    for s in shots:
        by_shot[s["shot"]] = []
    order = [s["shot"] for s in shots]
    fps = int(stats.get("fps") or 16)
    for i, p in enumerate(frames):
        t = i / fps
        for s in shots:
            start = float(s.get("in_s") or 0)
            end = start + int(s.get("hold_frames") or 0) / fps
            if start <= t < end:
                by_shot[s["shot"]].append(p)
                break
    rows = []
    clip_sims, luma_jumps, hist_jumps, angles = [], [], [], []
    for i in range(1, len(order)):
        prev, nxt = order[i - 1], order[i]
        if not by_shot[prev] or not by_shot[nxt]:
            continue
        last = Image.open(by_shot[prev][-1]).convert("RGB")
        first = Image.open(by_shot[nxt][0]).convert("RGB")
        # motion direction just before vs just after the cut
        tail = [Image.open(p).convert("RGB") for p in by_shot[prev][-5:]]
        head = [Image.open(p).convert("RGB") for p in by_shot[nxt][:5]]
        v_prev = _motion_vector(tail)
        v_next = _motion_vector(head)
        mag_p = (v_prev[0] ** 2 + v_prev[1] ** 2) ** 0.5
        mag_n = (v_next[0] ** 2 + v_next[1] ** 2) ** 0.5
        angle = None
        if mag_p > 0.3 and mag_n > 0.3:
            cosang = (v_prev[0] * v_next[0] + v_prev[1] * v_next[1]) / (mag_p * mag_n)
            angle = round(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), 1)
            angles.append(angle)
        # image continuity across the cut -- shared primitives, so this cannot
        # drift from previs or the cut detector
        luma = continuity.luma_jump(last, first)
        histd = continuity.hist_distance(last, first)
        sim = continuity.pair_similarity(last, first, scorer)
        if sim is not None:
            clip_sims.append(sim)
        luma_jumps.append(luma)
        hist_jumps.append(histd)
        rows.append({"cut": f"{prev}->{nxt}", "clip_sim": sim,
                     "luma_jump": round(luma, 1), "hist_jump": round(histd, 3),
                     "motion_angle_deg": angle,
                     "motion_before_px": round(mag_p, 2),
                     "motion_after_px": round(mag_n, 2)})
    return {
        "cuts": rows,
        "clip_sim_mean": round(statistics.fmean(clip_sims), 3) if clip_sims else None,
        "luma_jump_mean": round(statistics.fmean(luma_jumps), 1) if luma_jumps else None,
        "hist_jump_mean": round(statistics.fmean(hist_jumps), 3) if hist_jumps else None,
        "motion_angle_mean": (round(statistics.fmean(angles), 1) if angles else None),
    }


def _side_by_side(a: List[Path], b: List[Path], label_a: str, label_b: str,
                  cuts_a: List[float], cuts_b: List[float], out: Path,
                  size: int = 384) -> int:
    n = max(len(a), len(b))
    panels = []
    for i in range(n):
        row = []
        for frames, label, cuts in ((a, label_a, cuts_a), (b, label_b, cuts_b)):
            p = frames[min(i, len(frames) - 1)]
            im = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, 250, 22), fill=(0, 0, 0))
            d.text((5, 5), f"{label}  {i + 1}/{n}", fill=(255, 255, 0))
            t = i / 16.0
            if any(abs(t - c) < 1.0 / 16 for c in cuts):
                d.rectangle((0, 0, size - 1, size - 1), outline=(255, 80, 80), width=4)
            row.append(im)
        canvas = Image.new("RGB", (size * 2 + 4, size), (0, 0, 0))
        canvas.paste(row[0], (0, 0))
        canvas.paste(row[1], (size + 4, 0))
        panels.append(canvas)
    halves = panels[::2]
    halves[0].save(out / "sequence_ab.gif", save_all=True,
                   append_images=halves[1:], duration=125, loop=0)
    try:
        from diffusers.utils import export_to_video
        export_to_video(panels, str(out / "sequence_ab.mp4"), fps=16)
        return len(panels)
    except Exception as exc:  # noqa: BLE001
        print(f"  mp4 failed: {exc}")
        return 0


def _shot_grid(project: Dict, frames_a: List[Path], frames_b: List[Path],
               stats_a: Dict, stats_b: Dict, label_a: str, label_b: str,
               out: Path, size: int = 300) -> None:
    def first_frames(stats, frames):
        out = []
        fps = int(stats.get("fps") or 16)
        for s in stats.get("shots") or []:
            i = int(round(float(s.get("in_s") or 0) * fps))
            out.append(frames[min(i, len(frames) - 1)])
        return out
    fa, fb = first_frames(stats_a, frames_a), first_frames(stats_b, frames_b)
    n = max(len(fa), len(fb))
    grid = Image.new("RGB", (size * n + 4 * (n - 1), size * 2 + 4), (0, 0, 0))
    for row, (frames, label) in enumerate([(fa, label_a), (fb, label_b)]):
        for col, p in enumerate(frames):
            im = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, 240, 20), fill=(0, 0, 0))
            d.text((5, 4), f"{label} shot {col + 1}", fill=(255, 255, 0))
            grid.paste(im, (col * (size + 4), row * (size + 4)))
    grid.save(out / "per_shot_grid.png")


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B two rendered cuts of one action")
    ap.add_argument("--current", default="_seq_current")
    ap.add_argument("--coverage", default="_seq_coverage")
    ap.add_argument("--out", default="output/sequence_ab")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("loading CLIP for cross-cut similarity...", flush=True)
    scorer = _clip_scorer()

    pj_c, files_c, stats_c = _render_for(a.current)
    pj_v, files_v, stats_v = _render_for(a.coverage)
    st_c = _structure(pj_c, json.loads(stats_c.read_text(encoding="utf-8")))
    st_v = _structure(pj_v, json.loads(stats_v.read_text(encoding="utf-8")))

    print(f"\n=== sequence A/B: {a.current} vs {a.coverage} ===")
    print(f"  {st_c['shots']} shots / {st_v['shots']} shots")
    print(f"  structure:")
    for k in ("scenes", "scene_cuts", "duration_s", "asl_s", "hold_cv",
              "under_1s_pct", "over_5s_pct"):
        print(f"    {k:<14} {st_c[k]:>7}   {st_v[k]:>7}")
    print(f"    holds  {st_c['holds']}  vs  {st_v['holds']}")

    cont_c = cut_continuity(pj_c, files_c, json.loads(stats_c.read_text()), scorer)
    cont_v = cut_continuity(pj_v, files_v, json.loads(stats_v.read_text()), scorer)
    print(f"\n  cross-cut continuity (higher clip_sim / lower jumps = more continuous):")
    for k, better in (("clip_sim_mean", "higher"), ("luma_jump_mean", "lower"),
                      ("hist_jump_mean", "lower"), ("motion_angle_mean", "lower")):
        print(f"    {k:<18} {str(cont_c[k]):>8}   {str(cont_v[k]):>8}   ({better} is better)")
    for row_c, row_v in zip(cont_c["cuts"], cont_v["cuts"]):
        print(f"    {row_c['cut']:<10} clip {row_c['clip_sim']} vs {row_v['clip_sim']} · "
              f"luma {row_c['luma_jump']} vs {row_v['luma_jump']} · "
              f"angle {row_c['motion_angle_deg']} vs {row_v['motion_angle_deg']}")

    clip_cuts = [s["in_s"] for s in json.loads(stats_c.read_text())["shots"][1:]]
    clip_cuts_v = [s["in_s"] for s in json.loads(stats_v.read_text())["shots"][1:]]
    n = _side_by_side(files_c, files_v, "MONTAGE", "COVERAGE", clip_cuts, clip_cuts_v,
                      out)
    _shot_grid(pj_c, files_c, files_v, json.loads(stats_c.read_text()),
               json.loads(stats_v.read_text()), "MONTAGE", "COVERAGE", out)
    report = {"current": {"slug": a.current, "structure": st_c, "continuity": cont_c},
              "coverage": {"slug": a.coverage, "structure": st_v, "continuity": cont_v}}
    (out / "sequence_ab.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  gif: {out / 'sequence_ab.gif'}"
          + (f" · mp4: {out / 'sequence_ab.mp4'}" if n else ""))
    print(f"  per-shot grid: {out / 'per_shot_grid.png'}")
    print(f"  json: {out / 'sequence_ab.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
