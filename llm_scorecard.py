"""
llm_scorecard.py -- pick the writer on the metric that actually matters.

`llm_bench.py` measured tokens per second, which is the wrong criterion. Decode
speed decides how long you wait; it does not decide whether the shot list is any
good. The cost that matters is how much work the writer leaves for the rest of
the system, because every repair is a shot that stopped being what the writer
intended.

This runs several models over the same concept and reports, per model:

  rule breaks    how many of the constraints it was explicitly given it broke
  repairs        how many changes edit_planner had to make
  preserved      the fraction of the proposal that survived to the render
  scenes         whether it produced boundaries for a look/checkpoint change
  terse          prompts too short to render as an image

Read it as: a model with 0 repairs at 6 tok/s beats one with 6 repairs at 20,
because the repairs are the thing that silently rewrites your film.

    python llm_scorecard.py
    python llm_scorecard.py --models llama3:latest,qwen2.5:3b --fresh
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from edit_planner import plan
from story import CONCEPT_DEFAULT, scorecard, write_shots

DEFAULT = ["llama3:latest", "granite4.1:8b", "mistral:latest", "qwen2.5:3b"]


def run(model: str, concept: str, fresh: bool) -> Dict:
    try:
        p = write_shots(concept, model=model, fresh=fresh, verbose=False)
    except Exception as exc:
        return {"model": model, "error": f"{type(exc).__name__}: {str(exc)[:90]}"}

    planned, report = plan(p.shots)
    sc = scorecard(p, planned, report.changes, verbose=False)
    sc["terse"] = sum(1 for pr in p.problems if "terse" in pr or "heading" in pr)
    sc["fallback"] = p.used_fallback
    sc["cached"] = p.cached
    sc["logline_echoes_concept"] = (
        p.logline.strip().lower().strip(".") == concept.strip().lower().strip("."))
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default=CONCEPT_DEFAULT)
    ap.add_argument("--models", default=",".join(DEFAULT))
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the proposal cache and re-roll")
    ap.add_argument("--from-cache", action="store_true",
                    help="re-score the cached proposals only, no LLM calls. Use "
                         "this after changing the metric -- otherwise you pay "
                         "for generation again just to re-read the same output")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat each model N times: the writers are "
                         "non-deterministic, and with runs=1 you are comparing "
                         "one sample against another")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    if a.from_cache:
        a.runs = 1

    print(f'concept: "{a.concept}"')
    print(f"runs per model: {a.runs}"
          + ("   (from cache, no LLM calls)" if a.from_cache else "") + "\n")
    print(f"  {'model':<18} {'run':>3} {'shots':>5} {'breaks':>6} {'struct':>6} "
          f"{'timing':>6} {'pres':>5} {'scenes':>6} {'terse':>5} {'leak':>5} "
          f"{'wall_s':>6} {'tok/s':>6}")
    print("  " + "-" * 102)

    rows: List[Dict] = []
    for m in models:
        for r in range(1, a.runs + 1):
            sc = run(m, a.concept,
                     fresh=(a.fresh or r > 1) and not a.from_cache)
            sc["run"] = r
            rows.append(sc)
            if "error" in sc:
                print(f"  {m:<18} {r:>3}  {sc['error']}")
                continue
            tag = " (cached)" if sc.get("cached") else ""
            print(f"  {m:<18} {r:>3} {sc['shots']:>5} {sc['rule_breaks']:>6} "
                  f"{sc['structural_repairs']:>6} {sc['timing_repairs']:>6} "
                  f"{sc['preserved']:>5.2f} {sc['scenes']:>6} "
                  f"{sc.get('terse', 0):>5} {sc.get('prompt_leaks', 0):>5} "
                  f"{sc['seconds']:>6.1f} "
                  f"{sc['decode_tok_s']:>6.1f}{tag}")

    ok = [r for r in rows if "error" not in r and not r.get("fallback")]
    if ok:
        # rank on STRUCTURAL repairs -- the writer's actual mistakes -- and only
        # then on speed. Timing rewrites are the planner's job by design.
        best = sorted(ok, key=lambda r: (r["structural_repairs"],
                                         -r["decode_tok_s"]))[0]
        print(f"\n  fewest structural repairs: {best['model']} "
              f"({best['structural_repairs']}, {best['decode_tok_s']} tok/s, "
              f"{best['shots']} shots, {best['scenes']} scene(s))")
        print("  remember a writer must produce >=2 scenes or the look and "
              "checkpoint stages have no boundary to work on.")
        by_model: Dict[str, List[Dict]] = {}
        for r in ok:
            by_model.setdefault(r["model"], []).append(r)
        if a.runs > 1:
            print("\n  run-to-run stability (structural repairs):")
            for m, rs in by_model.items():
                v = [x["structural_repairs"] for x in rs]
                s = [x["shots"] for x in rs]
                print(f"    {m:<18} repairs {v}  shots {s}")
            print("  a writer whose repair count swings between runs is a "
                  "liability when each render costs an hour.")


if __name__ == "__main__":
    main()
