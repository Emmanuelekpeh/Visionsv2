"""
writer_audit.py -- is the writer the ceiling? Answer it for free before paying.

The S6 question, and why it exists: a proposal takes 80-260 s while a 6-shot
render takes ~60 min, so the text layer is 2-5% of runtime and "don't optimise
the writer" was the correct call. The override was that shot quality had to be
proven not to be the writer's fault FIRST, because otherwise you cannot tell a
bad writer from a bad sampler when the film comes out weak.

The design that costs an hour: 3 writers, same concept, fresh proposals, render
2 shots each at identical seed/scale/camera, measure CLIP align and the composite
score, then look. Two problems with starting there. `align`'s seed sd is 0.018
(handoff 12), so a real difference must exceed ~0.04 and n=2 per writer will only
show a LARGE effect; and align is not quality -- it measures text-image agreement,
so the test only works if you look at the pictures.

So this runs the cheap half first. It generates the fresh proposals and diffs the
PROMPTS, which is ~5 minutes and no rendering. If the three writers are saying
substantially the same thing, the text layer cannot be the ceiling and the hour
does not need spending. If one writer is visibly vaguer or more repetitive, that
is the answer and it is also free.

Two text measures do most of the work:

  self-repetition   mean Jaccard overlap between a writer's OWN prompts. High
                    means it re-describes the same shot -- visible as a film that
                    repeats itself, and R1 forbids a repeated SCALE, not repeated
                    content.
  cross-writer      Jaccard between writers' combined content words. This is the
                    one that decides the spend: if all three say the same thing,
                    the writer is not a lever.

    python writer_audit.py --runs 1                 # fresh, ~3-13 min
    python writer_audit.py --runs 0 --cached-only    # re-read what exists, free
"""

from __future__ import annotations

import argparse
import json
import re
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "film" / "writer_audit"
CACHE = ROOT / "output" / "film" / "proposals"

DEFAULT_MODELS = ["granite4.1:8b", "llama3:latest", "mistral:latest"]

# One definition of the text measures, shared with the ensemble's provenance
# check in story.py. Duplicating them here would let the audit and the pipeline
# disagree about what "the same idea" means.
from story import content_words, jaccard  # noqa: E402


def plan_and_score(shots: Sequence):
    from edit_planner import plan
    from story import scorecard
    planned, report = plan(list(shots))
    return planned, report


def collect(model: str, concept: str, runs: int, cached_extra: bool,
            verbose: bool = True) -> List[Dict]:
    """Every sample for one writer: fresh runs first, then free older caches."""
    from story import write_shots

    samples: List[Dict] = []
    for r in range(runs):
        t0 = time.time()
        try:
            p = write_shots(concept, model=model, fresh=True, verbose=verbose)
        except Exception as exc:
            print(f"    {model}: {type(exc).__name__}: {str(exc)[:110]}")
            continue
        samples.append(sample_of(p, model, f"fresh{r+1}", time.time() - t0))

    if cached_extra:
        # The pre-`__n` cache files are from the section 11.2 scorecard run.
        slug = re.sub(r"\W+", "-", concept.lower()).strip("-")[:60]
        for f in sorted(CACHE.glob(f"{slug}__{model.replace(':', '-')}*.json")):
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            from story import validate
            shots, problems = validate(raw)
            if not shots:
                continue
            if any(s["prompts"] == [sh.prompt for sh in shots] for s in samples):
                continue                      # already have this exact proposal
            samples.append({
                "model": model, "tag": f"cached:{f.stem.split('__')[-1]}",
                "logline": str(raw.get("logline", ""))[:200],
                "prompts": [sh.prompt for sh in shots],
                "scales": [sh.scale for sh in shots],
                "seconds": [sh.seconds for sh in shots],
                "scenes": [getattr(sh, "scene", "") for sh in shots],
                "problems": problems, "planned": None,
                "report": None, "score": None, "wall_s": 0.0, "tok_s": 0.0,
            })
    return samples


def sample_of(p, model: str, tag: str, wall: float) -> Dict:
    planned, report = plan_and_score(p.shots)
    from story import scorecard
    sc = scorecard(p, planned, report.changes, verbose=False)
    return {
        "model": model, "tag": tag, "logline": p.logline,
        "prompts": [s.prompt for s in planned],
        "scales": [s.scale for s in planned],
        "seconds": [round(s.seconds, 2) for s in planned],
        "scenes": [s.scene for s in planned],
        "problems": list(p.problems), "score": sc,
        "changes": list(report.changes), "wall_s": round(wall, 1),
        "tok_s": p.decode_tok_s, "planned": None, "report": None,
    }


def leaks(text: str) -> List[str]:
    """Field values fused into the prompt string.

    One definition, not two: this is `story.prompt_leaks`, the same detector
    `validate` records as a problem. Duplicating the regex here would let the
    audit and the pipeline disagree about what a leak is.

    It matters more than any style difference: `prompt` is what CLIP and SD
    receive, so "2.1 seconds, energy 0.2" or "camera: none" appended to it is
    noise the image model will try to draw. A render test would show this as worse
    alignment and blame the sampler -- the exact mis-attribution S6 exists to
    prevent.
    """
    from story import prompt_leaks
    return prompt_leaks(text)


def annotate(s: Dict) -> Dict:
    """Text-side measures for one sample."""
    sets = [content_words(pr) for pr in s["prompts"]]
    self_j = [jaccard(a, b) for a, b in combinations(sets, 2)]
    words = [len(content_words(pr)) for pr in s["prompts"]]
    raw_words = [len(pr.split()) for pr in s["prompts"]]
    dups = [(i, j) for i, j in combinations(range(len(sets)), 2)
            if jaccard(sets[i], sets[j]) > 0.6]
    s["text"] = {
        "content_words": words,
        "raw_words": raw_words,
        "self_jaccard_mean": round(sum(self_j) / len(self_j), 3) if self_j else 0.0,
        "self_jaccard_max": round(max(self_j), 3) if self_j else 0.0,
        "near_duplicate_pairs": dups,
        "leaks": [{"shot": i + 1, "fields": leaks(pr), "prompt": pr[:70]}
                  for i, pr in enumerate(s["prompts"]) if leaks(pr)],
        "union": set().union(*sets) if sets else set(),
        "scales": list(s["scales"]),
    }
    return s


def table(samples: List[Dict], per_model: Dict[str, List[Dict]]) -> str:
    L = []
    L.append("=" * 78)
    L.append("PER WRITER")
    L.append("=" * 78)
    L.append(f"  {'writer':<18} {'run':<14} {'shots':>5} {'scenes':>6} {'struct':>6} "
             f"{'timing':>6} {'breaks':>6} {'terse':>5} {'wall_s':>7} {'tok/s':>6}")
    for s in samples:
        sc = s["score"] or {}
        L.append(f"  {s['model']:<18} {s['tag']:<14} {len(s['prompts']):>5} "
                 f"{sc.get('scenes', '-'):>6} {sc.get('structural_repairs', '-'):>6} "
                 f"{sc.get('timing_repairs', '-'):>6} {sc.get('rule_breaks', '-'):>6} "
                 f"{sc.get('terse', '-'):>5} {s['wall_s']:>7.1f} {s['tok_s']:>6.1f}")
    L.append("")
    L.append(f"  {'writer':<18} {'content words/shot':>19} {'self-jaccard':>13} "
             f"{'worst pair':>11} {'near-dupes':>11} {'leaks':>6}")
    for m, ss in per_model.items():
        for s in ss:
            t = s["text"]
            L.append(f"  {m:<18} {str(t['content_words']):>19} "
                     f"{t['self_jaccard_mean']:>13.3f} {t['self_jaccard_max']:>11.3f} "
                     f"{len(t['near_duplicate_pairs']):>11} {len(t['leaks']):>6}")
    L.append("")
    L.append("PROMPT TEXT WITH NON-VISUAL FIELD VALUES FUSED INTO IT")
    L.append("(`prompt` is what CLIP and SD receive -- this is drawn, not ignored)")
    any_leak = False
    for m, ss in per_model.items():
        for s in ss:
            for lk in s["text"]["leaks"]:
                any_leak = True
                L.append(f"  {m:<17} {s['tag']:<22} shot {lk['shot']:<3} "
                         f"[{'+'.join(lk['fields'])}]  {lk['prompt']}...")
    if not any_leak:
        L.append("  none")
    return "\n".join(L)


def side_by_side(per_model: Dict[str, List[Dict]]) -> str:
    """Row = shot index, column = writer. Vagueness and repetition show here."""
    models = [m for m, ss in per_model.items() if ss]
    if not models:
        return ""
    # first sample per writer: the like-for-like comparison
    first = {m: per_model[m][0] for m in models}
    n = max(len(first[m]["prompts"]) for m in models)
    L = []
    L.append("")
    L.append("=" * 78)
    L.append("SIDE BY SIDE -- same concept, fresh proposals, prompt text only")
    L.append("=" * 78)
    for i in range(n):
        L.append("")
        for m in models:
            s = first[m]
            if i < len(s["prompts"]):
                L.append(f"  shot {i+1}  {m:<17} {s['scales'][i]:<8} "
                         f"{s['seconds'][i]:>4.2f}s  {s['prompts'][i]}")
            else:
                L.append(f"  shot {i+1}  {m:<17} (no shot {i+1} -- writer produced "
                         f"{len(s['prompts'])})")
    return "\n".join(L)


def cross_writer(per_model: Dict[str, List[Dict]]) -> Tuple[str, float]:
    """The measure that decides the spend.

    If the writers' combined vocabularies overlap heavily, they are describing the
    same film and the text layer cannot be the ceiling.
    """
    first = {m: per_model[m][0] for m in per_model if per_model[m]}
    L = ["", "=" * 78, "CROSS-WRITER OVERLAP (does the writer change the film?)",
         "=" * 78]
    vals = []
    for a, b in combinations(first, 2):
        j = jaccard(first[a]["text"]["union"], first[b]["text"]["union"])
        shared = first[a]["text"]["union"] & first[b]["text"]["union"]
        vals.append(j)
        only_a = first[a]["text"]["union"] - first[b]["text"]["union"]
        only_b = first[b]["text"]["union"] - first[a]["text"]["union"]
        L.append(f"  {a} vs {b}")
        L.append(f"    content-word Jaccard {j:.3f}   shared {len(shared)}  "
                 f"only {a.split(':')[0]} {len(only_a)}  only {b.split(':')[0]} {len(only_b)}")
        L.append(f"    shared: {', '.join(sorted(shared)[:22])}")
        if only_a:
            L.append(f"    only {a.split(':')[0]}: {', '.join(sorted(only_a)[:14])}")
        if only_b:
            L.append(f"    only {b.split(':')[0]}: {', '.join(sorted(only_b)[:14])}")
    mean = sum(vals) / len(vals) if vals else 0.0
    return "\n".join(L), mean


def verdict(per_model: Dict[str, List[Dict]], mean_cross: float) -> str:
    L = ["", "=" * 78, "READ THIS BEFORE SPENDING AN HOUR", "=" * 78]
    all_s = [s for ss in per_model.values() for s in ss]
    short = min((min(s["text"]["content_words"]) for s in all_s), default=0)
    worst_self = max((s["text"]["self_jaccard_max"] for s in all_s), default=0.0)
    dupes = sum(len(s["text"]["near_duplicate_pairs"]) for s in all_s)
    L.append(f"  mean cross-writer overlap      {mean_cross:.3f}")
    L.append(f"  shortest prompt (content words) {short}")
    L.append(f"  worst within-writer overlap     {worst_self:.3f}")
    L.append(f"  near-duplicate shot pairs       {dupes}")
    leaked = sum(len(s["text"]["leaks"]) for s in all_s)
    by_model = {}
    for s in all_s:
        by_model.setdefault(s["model"], [0, 0])
        by_model[s["model"]][0] += len(s["text"]["leaks"])
        by_model[s["model"]][1] += len(s["prompts"])
    L.append(f"  prompts carrying leaked fields  {leaked}")
    L.append("")
    if leaked:
        L.append("  -> FIX THIS FIRST, BEFORE ANY RENDER TEST. `prompt` is the text")
        L.append("     CLIP and SD receive, so a fused 'camera: none' or '2.1 seconds,")
        L.append("     energy 0.2' is noise the image model tries to draw. It would")
        L.append("     show up as worse `align` and the render test would blame the")
        L.append("     sampler or the model -- which is the mis-attribution this test")
        L.append("     exists to prevent. It is free to fix (a validator on the")
        L.append("     proposal, since `story.validate` does not catch it).")
        L.append("     Leaks per writer (leaked/shot): "
                 + ", ".join(f"{m} {v[0]}/{v[1]}" for m, v in by_model.items()))
    if mean_cross >= 0.45 and short >= 6 and worst_self < 0.6 and dupes == 0:
        L.append("")
        L.append("  -> The writers are describing substantially the same film and none")
        L.append("     is vague or repetitive. The text layer is NOT the ceiling and the")
        L.append("     ~1 hour render is not worth spending to confirm it.")
    elif mean_cross < 0.25:
        L.append("")
        L.append("  -> The writers produce markedly DIFFERENT content, so the writer is a")
        L.append("     live variable. But check structure and leaks above first: if every")
        L.append("     writer scores 0 structural repairs, the difference is subject")
        L.append("     matter, not quality, and a render test whose metric cannot see")
        L.append("     quality will not settle it.")
    else:
        L.append("")
        L.append("  -> Mixed. Read the side-by-side and judge the prompts yourself; that")
        L.append("     judgement is cheaper and more reliable than a 6-clip render whose")
        L.append("     metric cannot see quality anyway.")
    if short < 6:
        L.append("")
        L.append("  !! at least one prompt is very short -- vague prompts are the")
        L.append("     failure mode a render would blame on the sampler.")
    if dupes:
        L.append("")
        L.append("  !! near-duplicate shots within a writer: the film would repeat")
        L.append("     itself, and R1 only forbids a repeated SCALE.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default=None)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--runs", type=int, default=1,
                    help="fresh proposals per writer (0 = reuse caches only)")
    ap.add_argument("--cached-only", action="store_true")
    ap.add_argument("--no-cached", action="store_true",
                    help="fresh runs only; do not fold in older cached proposals")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    from story import CONCEPT_DEFAULT
    concept = a.concept or CONCEPT_DEFAULT
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    runs = 0 if a.cached_only else a.runs

    print(f'concept: "{concept}"')
    print(f"writers: {', '.join(models)}")
    print(f"fresh proposals: {runs} each   cached folded in: {not a.no_cached}")
    print("(no rendering: this is the cheap half of the writer test)")

    per_model: Dict[str, List[Dict]] = {}
    t0 = time.time()
    for m in models:
        print(f"\n=== {m} ===", flush=True)
        ss = collect(m, concept, runs, cached_extra=not a.no_cached,
                     verbose=not a.quiet)
        per_model[m] = [annotate(s) for s in ss]
        for s in per_model[m]:
            print(f"  {s['tag']:<14} {len(s['prompts'])} shots, "
                  f"{s['text']['self_jaccard_mean']:.2f} self-overlap, "
                  f"{'cached' if s['wall_s'] == 0 else f'{s["wall_s"]:.0f}s'}")
    print(f"\ngeneration wall time: {time.time()-t0:.0f}s")

    all_s = [s for ss in per_model.values() for s in ss]
    body = [table(all_s, per_model), side_by_side(per_model)]
    cross, mean_cross = cross_writer(per_model)
    report = "\n".join(body) + "\n" + cross + "\n" + verdict(per_model, mean_cross)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.txt").write_text(report + "\n", encoding="utf-8")
    # samples.json is the latest view, but it is OVERWRITTEN each run -- and
    # `write_shots`'s own cache key has no run id, so a fresh re-roll of the same
    # model+concept overwrites the proposal it replaced. That destroyed the first
    # round of this audit (12 leaked prompts) before it could be re-analysed. A
    # run-to-run comparison needs samples that cannot be clobbered, so every round
    # is also written to its own timestamped file and appended to a jsonl.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    clean = [{k: v for k, v in s.items() if k != "text"} for s in all_s]
    (OUT / f"samples_{stamp}.json").write_text(
        json.dumps(clean, indent=2), encoding="utf-8")
    with (OUT / "samples.jsonl").open("a", encoding="utf-8") as fh:
        for s in clean:
            fh.write(json.dumps({"taken": stamp, **s}) + "\n")
    (OUT / "samples.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    print(report)
    print(f"\nwrote {(OUT/'audit.txt').relative_to(ROOT)}")
    print(f"wrote {(OUT/f'samples_{stamp}.json').relative_to(ROOT)}  (immutable)")
    print(f"appended {(OUT/'samples.jsonl').relative_to(ROOT)}")
    print("NOTE: output/film/proposals/<slug>__<model>__nauto.json is keyed with no")
    print("      run id, so a fresh proposal OVERWRITES the previous one. Keep")
    print("      copies here; do not treat that cache as a sample history.")


if __name__ == "__main__":
    main()
