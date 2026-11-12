#!/usr/bin/env python3
"""
sequence_audit.py -- replay the sequence/coverage grammar over work that already exists.

The grammar rules (R7-R10) are new. Before spending a render on them, the useful
question is: what would they have changed on the shot lists we already have? This
reads cached proposals and/or productions, runs the same `story.validate` and
`edit_planner.plan` the pipeline would, and reports:

  * STORY breaks      (beat order) -- the old gate
  * STRUCTURE breaks  (R7 scene runs/cuts, R8 coverage, R10 move->scene handoff)
  * RHYTHM breaks     (R9 uniform holds / scene-opening not longest; the planner
                       owns hold timing, so these are advisory)
  * what R10 would REWRITE in the planner (a move before a scene cut -> none)
  * the organisation numbers: shots/scene, scene cuts, within-scene overlap

No LLM calls. No renders. This is the "what would the grammar do to our archive"
answer, and it is how a rule earns its place before it gates new work.

    python sequence_audit.py                 # cached proposals
    python sequence_audit.py --store seasoning
    python sequence_audit.py --file output/film/proposals/<name>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from edit_planner import plan  # noqa: E402
from pacing_audit import load_production, organisation  # noqa: E402
from story import CONCEPT_DEFAULT, validate  # noqa: E402

CACHE = ROOT / "output" / "film" / "proposals"


def _row(name: str, raw: Dict) -> Dict:
    shots, problems = validate(raw)
    if not shots:
        return {"name": name, "ok": False,
                "error": "; ".join(problems[:2]) or "no shots"}
    frames = 16
    planned, report = plan(shots, frames=frames, fps=16)
    r10 = [c for c in report.changes if "R10" in c]
    structure = [p for p in problems if p.startswith("structure:")]
    # Cached proposals predate the purpose/link fields, so "missing metadata" is
    # a baseline, not a verdict on the writer. R7/R8 are the real structure
    # violations; keep them separate or the archive looks worse than it is.
    meta = sum(1 for p in structure if "purpose" in p or " link " in p
               or "no link" in p)
    return {
        "name": name, "ok": True,
        "shots": len(shots),
        "scenes": len({s.scene for s in shots}),
        "story": sum(1 for p in problems if p.startswith("story:")),
        "structure": len(structure),
        "meta": meta,
        "r7": sum(1 for p in structure if "R7" in p),
        "r8": sum(1 for p in structure if "R8" in p),
        "r10_problems": sum(1 for p in structure if "R10" in p),
        "rhythm": sum(1 for p in problems if p.startswith("rhythm:")),
        "r10_rewrites": len(r10),
        **{k: v for k, v in organisation(
            [{"prompt": s.prompt, "scene": s.scene, "seconds": s.seconds,
              "subject": s.subject, "camera": s.camera} for s in planned]).items()
           if k in ("scene_runs", "scene_cuts", "single_shot_scenes",
                    "shots_per_scene", "within_scene_overlap",
                    "across_scene_overlap", "same_subject_within")},
    }


def _print(rows: List[Dict]) -> None:
    print(f"\n{'name':<46} {'shot':>4} {'scn':>4} {'runs':>4} {'cuts':>4} "
          f"{'story':>5} {'meta':>4} {'R7':>3} {'R8':>3} {'R10':>4} {'rhy':>4} "
          f"{'in/out ovl':>11}")
    print("-" * 118)
    for r in rows:
        if not r.get("ok"):
            print(f"{r['name']:<46}  !! {r['error']}")
            continue
        print(f"{r['name'][-46:]:<46} {r['shots']:>4} {r['scenes']:>4} "
              f"{r['scene_runs']:>4} {r['scene_cuts']:>4} {r['story']:>5} "
              f"{r['meta']:>4} {r['r7']:>3} {r['r8']:>3} {r['r10_problems']:>4} "
              f"{r['rhythm']:>4} "
              f"{str(r['within_scene_overlap']):>5}/{str(r['across_scene_overlap']):<5}")
    ok = [r for r in rows if r.get("ok")]
    if ok:
        print(f"\n  {len(ok)} shot lists replayed. Cached proposals predate "
              f"purpose/link, so 'meta' misses are a baseline, not a writer verdict.")
        print(f"  real structure violations: R7 {sum(r['r7'] for r in ok)}, "
              f"R8 {sum(r['r8'] for r in ok)}, R10 {sum(r['r10_problems'] for r in ok)}")
        print(f"  the planner would rewrite {sum(r['r10_rewrites'] for r in ok)} "
              f"camera move(s) that hand off to new scenery.")
        print(f"  rhythm advisories: {sum(r['rhythm'] for r in ok)}.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay the grammar over existing shot lists")
    ap.add_argument("--cached", action="store_true",
                    help="cached ensemble/single proposals (default if nothing else)")
    ap.add_argument("--store", default=None)
    ap.add_argument("--file", default=None)
    a = ap.parse_args()
    rows: List[Dict] = []
    if a.file:
        raw = json.loads(Path(a.file).read_text(encoding="utf-8"))
        rows.append(_row(Path(a.file).name, raw))
    elif a.store:
        p = load_production(a.store)
        if p is None:
            raise SystemExit(f"no production {a.store!r}")
        rows.append(_row(f"store:{a.store}", {"logline": "", "shots": [
            {"prompt": s.get("prompt"), "scale": s.get("scale"),
             "camera": s.get("camera_suggestion") or "none",
             "seconds": s.get("seconds"), "scene": s.get("scene"),
             "subject": s.get("subject"), "energy": s.get("energy"),
             "beat": s.get("beat"), "purpose": s.get("purpose", ""),
             "link": s.get("link", "")} for s in p["_shots"]]}))
    else:
        for f in sorted(CACHE.glob("*.json")):
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append(_row(f.stem, raw))
    if not rows:
        print("nothing to audit -- no cached proposals found")
        return 1
    _print(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
