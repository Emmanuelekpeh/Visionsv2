#!/usr/bin/env python3
"""
camera_ab.py -- compare camera-direction policies on ONE shot list, before rendering.

The question this answers: the store used to render every shot with `camera="none"`.
Now adoption can direct the cut four ways (`camera_direction.MODES`). Which one
should ship? Rendering both films would take hours and conflate the camera choice
with the sampler, so this measures the DECISION first:

  * how many shots move (the brief allows at most two in a six-shot film),
  * what the planner had to repair (R6 reversals, R3 runs, framing),
  * which shots move and whether it reads as intentional.

The last one is a judgement: the tool prints the per-shot cameras so you can read
them against the prompts, and `--render N` is deliberately NOT implemented --
a real A/B needs watching, not a metric.

    python camera_ab.py --store seasoning
    python camera_ab.py --store longing --modes off,suggestions,director,hybrid
    python camera_ab.py --proposal output/showrunner_test/<stamp>/replies/granite4.1-8b__run2.txt
    python camera_ab.py --store seasoning --json-out output/camera_ab_seasoning.json

A `--proposal` is a raw ensemble reply (the JSON with `shots`, whose `camera`
field is the writer's suggestion); this is how a NEW shot list is compared
without adopting it into a production first.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import camera_direction as cd  # noqa: E402
import film_store as fs  # noqa: E402
from edit_planner import Shot, plan  # noqa: E402
from story import parse_reply, validate  # noqa: E402

ALL_MODES = list(cd.MODES)


def shots_from_store(slug: str) -> Tuple[List[Shot], int, int, Dict]:
    project = fs.load(slug)
    st = project.get("settings") or {}
    frames = int(st.get("frames") or 16)
    fps = int(st.get("fps") or 16)
    rows = fs.shots_in_order(project)
    shots = [Shot(prompt=s.get("prompt", ""), scale=s.get("scale", "medium"),
                  camera=s.get("camera_suggestion") or "none",
                  seconds=float(s.get("seconds") or 2.5),
                  subject=s.get("subject", ""), scene=s.get("scene", ""),
                  energy=s.get("energy"), beat=s.get("beat", ""))
             for s in rows]
    meta = {"source": f"store:{slug}", "shot_count": len(shots),
            "current_cameras": [s.get("camera") or "none" for s in rows],
            "writer_suggestions": [s.camera for s in shots]}
    return shots, frames, fps, meta


def shots_from_proposal(path: str) -> Tuple[List[Shot], int, int, Dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    value, mode = parse_reply(json.dumps(raw))
    if value is None:
        raise SystemExit(f"{path} is not parseable JSON")
    shots, problems = validate(value)
    if not shots:
        raise SystemExit(f"{path} has no usable shots: {problems[:3]}")
    meta = {"source": f"proposal:{Path(path).name}", "shot_count": len(shots),
            "current_cameras": ["none"] * len(shots),
            "writer_suggestions": [s.camera for s in shots],
            "proposal_problems": problems}
    return shots, 16, 16, meta


def run_mode(shots: Sequence[Shot], mode: str, frames: int, fps: int,
             scorer=None) -> Dict:
    cams, info = cd.decide(shots, mode=mode, scorer=scorer)
    basis = [replace(s, camera=c) for s, c in zip(shots, cams)]
    planned, report = plan(basis, frames=frames, fps=fps)
    final = [s.camera for s in planned]
    before = cd.summary(shots, cams)
    after = cd.summary(planned, final)
    repairs = {"r1_framing": sum(1 for c in report.changes if "R1" in c),
               "r6_reversal": sum(1 for c in report.changes if "R6" in c),
               "r3_runs": sum(1 for c in report.changes if "R3" in c),
               "r4_holds": sum(1 for c in report.changes if "R4" in c),
               "total": len(report.changes)}
    return {"mode": mode, "cameras": final, "before_plan": before,
            "after_plan": after, "repairs": repairs, "info": info,
            "holds": [round(s.seconds, 2) for s in planned]}


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B camera-direction policies")
    ap.add_argument("--store", default=None, help="production slug")
    ap.add_argument("--proposal", default=None,
                    help="raw ensemble reply JSON (a new shot list)")
    ap.add_argument("--modes", default=",".join(ALL_MODES))
    ap.add_argument("--frames", type=int, default=None,
                    help="override the raw-frame setting (for a proposal)")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    if not a.store and not a.proposal:
        ap.error("give --store or --proposal")

    if a.store:
        shots, frames, fps, meta = shots_from_store(a.store)
    else:
        shots, frames, fps, meta = shots_from_proposal(a.proposal)
    if a.frames:
        frames = int(a.frames)
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in cd.MODES]
    if bad:
        raise SystemExit(f"unknown mode(s) {bad}; expected {cd.MODES}")

    print(f"\n=== camera A/B: {meta['source']} ===")
    print(f"  {len(shots)} shots · {frames} raw frames ({fps} fps) · "
          f"motion per shot {(2 * frames - 1) / fps:.2f}s")
    print(f"  writer suggested: {sum(1 for c in meta['writer_suggestions'] if c != 'none')}"
          f" move(s) · currently stored: "
          f"{sum(1 for c in meta['current_cameras'] if c != 'none')} move(s)")

    scorer = None
    if any(m in ("director", "hybrid") for m in modes):
        from director import FitScorer
        print("  loading CLIP fit scorer (the director's fit)...", flush=True)
        scorer = FitScorer()

    results = []
    for m in modes:
        r = run_mode(shots, m, frames, fps, scorer=scorer)
        results.append(r)
        print(f"\n--- {m} ---")
        print(f"  moves before plan {r['before_plan']['movers']}/{len(shots)} "
              f"· after plan {r['after_plan']['movers']}/{len(shots)} "
              f"· reversals repaired {r['repairs']['r6_reversal']} "
              f"· runs repaired {r['repairs']['r3_runs']} "
              f"· total planner changes {r['repairs']['total']}")
        print(f"  cameras: {', '.join(r['cameras'])}")
        if r["info"].get("fallback"):
            print(f"  !! director fallback: {r['info']['fallback']}")

    # per-shot decision matrix -- the part a person reads
    print(f"\n{'=' * 78}\n  PER-SHOT CAMERAS (writer suggestion vs each policy)\n"
          f"{'=' * 78}")
    header = f"  {'#':>2} {'scale':<12} {'suggest':<14}" + "".join(
        f"{m:<13}" for m in modes)
    print(header)
    by_mode = {r["mode"]: r["cameras"] for r in results}
    for i, s in enumerate(shots):
        row = (f"  {i + 1:>2} {s.scale:<12} "
               f"{(s.camera if s.camera else 'none'):<14}")
        row += "".join(f"{by_mode[m][i]:<13}" for m in modes)
        print(row)
        if len(s.prompt) > 0:
            print(f"     {s.prompt[:100]}")

    # what moved and on which policy
    print("\n=== which policy adds movement inside the brief? ===")
    for r in results:
        movers = r["after_plan"]["movers"]
        flag = "OK" if movers <= max(2, round(len(shots) / 3)) else "OVER BUDGET"
        print(f"  {r['mode']:<12} {movers} move(s) after planning  [{flag}]  "
              f"reversals after plan: {r['after_plan']['reversals']}")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(
            json.dumps({"meta": meta, "frames": frames, "fps": fps,
                        "results": results}, indent=2), encoding="utf-8")
        print(f"\n  json: {a.json_out}")
    print("\n  reading it: 'mover cap' in the writer's brief is 2 for a 6-shot film; "
          "the planner scales it for longer films. Anything above that is fatigue, "
          "not style.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
