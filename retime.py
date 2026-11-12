#!/usr/bin/env python3
"""retime.py -- give an already-adopted production the measured HUMAN rhythm.

WHY. The delivered 13-shot film had shot-length CV 0.074. A first pass fixed the
planner's own narrow bands but could only alternate 2.5/1.5 -- still a metronome,
and it ignored PACING_MEASURED_STUDY.md, which measured human short-form cutting:
<0.5s 14.0% | 0.5-1s 22.6% | 1-2s 17.8% | 2-3s 12.0% | 3-5s 14.6% | >5s 19.0%,
within-clip CV 0.75. Human cutting has a FAST band and a LONG band, with the 2 s
band used sparingly.

Two budgets make that reachable WITHOUT re-generating anything:
  * accents: trim a 16-frame take to 0.9-1.45 s (motion gets faster, which the
    study calls acceptable for accents);
  * long holds: raise the shot's FINISHING RAFT factor. 16 raw frames at x4 =
    61 interpolated frames = 3.8 s, x6 = 91 = 5.7 s -- the study's "procedural
    hold". `upsample` lives in `finish_inputs`, so this re-finishes from the take
    on disk (RAFT + shake + grade), never re-generates.

    python retime.py --store ego_v5             # apply the study-shaped rhythm
    python retime.py --store ego_v5 --dry-run   # show it, change nothing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import edit_planner as ep                     # noqa: E402


def cv(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    return sd / max(1e-9, mean)


def choose_upsamples(rows: List[Dict], max_up: int = 6) -> List[int]:
    """Long takes for the long band; default x2 elsewhere.

    Scene openings and establishes/resolves get x4 (3.8 s of smooth motion, hold
    ceiling 4.95 s); the final resolve gets x6 (5.7 s, ceiling 7.4 s) so the film
    can have the study's >5 s band. Everything else stays at the take's native x2.
    """
    scenes = [r.get("scene") or "" for r in rows]
    firsts = set()
    prev = object()
    for i, sc in enumerate(scenes):
        if sc != prev:
            firsts.add(i)
            prev = sc
    out: List[int] = []
    for i, r in enumerate(rows):
        beat = r.get("beat") or ""
        if i == len(rows) - 1 and beat == "resolve":
            up = max_up
        elif i in firsts or beat in ("establish", "resolve"):
            up = min(4, max_up)
        else:
            up = 2
        out.append(max(2, int(up)))
    return out


def shots_for_curve(rows: List[Dict], upsamples: List[int]) -> List[ep.Shot]:
    return [ep.Shot(
        prompt=r.get("prompt") or "", scale=r.get("scale") or "medium",
        camera=r.get("camera") or "none",
        seconds=float(r.get("seconds") or 2.5),
        subject=r.get("subject") or "", scene=r.get("scene") or "",
        energy=r.get("energy"), beat=r.get("beat") or "",
        purpose=r.get("purpose") or "", link=r.get("link") or "",
        camera_move=r.get("camera_move"),
        upsample=u) for r, u in zip(rows, upsamples)]


def propose(rows: List[Dict], frames: int, fps: int,
            total: Optional[float] = None,
            max_up: int = 6) -> Tuple[List[float], List[int]]:
    ups = choose_upsamples(rows, max_up)
    shots = shots_for_curve(rows, ups)
    return ep.rhythm_curve(shots, total=total, frames=frames, fps=fps), ups


def apply(slug: str, dry_run: bool = False, total: Optional[float] = None,
          max_up: int = 6) -> Dict:
    import film_store as fs
    project = fs.load(slug)
    st = project.get("settings") or {}
    frames, fps = int(st.get("frames") or 16), int(st.get("fps") or 16)
    rows = [fs.load_shot(slug, sid) for sid in project["timeline"]]
    before = [float(s.get("seconds") or 2.5) for s in rows]
    seconds, ups = propose(rows, frames, fps, total=total, max_up=max_up)
    summary = {
        "store": slug, "frames": frames, "fps": fps,
        "before_cv": round(cv(before), 3), "after_cv": round(cv(seconds), 3),
        "before_total": round(sum(before), 2),
        "after_total": round(sum(seconds), 2),
        "holds": [(s["id"], round(b, 2), round(a, 2), u)
                  for s, b, a, u in zip(rows, before, seconds, ups)],
    }
    if not dry_run:
        for shot, secs, up in zip(rows, seconds, ups):
            fs.update_shot(project, shot, seconds=secs, upsample=up)
        fs.save(project)
        summary["applied"] = True
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Study-shaped rhythm pass")
    ap.add_argument("--store", required=True)
    ap.add_argument("--total", type=float, default=None,
                    help="target duration (default: let the measured rhythm set it)")
    ap.add_argument("--max-upsample", type=int, default=6,
                    help="largest RAFT factor for long holds (default 6 = 5.7 s "
                         "of smooth motion from a 16-frame take)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    summary = apply(a.store, dry_run=a.dry_run, total=a.total,
                    max_up=a.max_upsample)
    print(f"{summary['store']}: CV {summary['before_cv']} -> "
          f"{summary['after_cv']}  |  {summary['before_total']}s -> "
          f"{summary['after_total']}s")
    print(f"  {'shot':<6} {'before':>7} {'after':>7} {'raft':>5}")
    for sid, before, after, up in summary["holds"]:
        print(f"  {sid:<6} {before:>7.2f} {after:>7.2f} x{up:<4}")
    if a.dry_run:
        print("  (dry run -- nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
