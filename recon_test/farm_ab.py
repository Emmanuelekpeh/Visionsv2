"""Does a "different place each time" rule actually reduce repetition?

Run properly: three farm models x both briefs x two farm angles, one draw each.

WHY THE METRIC CHANGED. The first attempt scored "distinct leading terms" by word
length, which is not a measure of anything -- it reported 6/6 distinct for both
briefs because the longest word in each idea happened to differ. A metric that
cannot fail is not a metric.

What the rule actually asks for is stated in the rule: DIFFERENT PLACES. So parse
the locative phrase out of each idea ("in a dimly lit alleyway", "under a moonlit
sky", "on a cobblestone street"), reduce it to its head noun, and count how many
ideas share one. That is `story.content_words`-based, deterministic, and it can fail.

Two readings are reported per cell, because they answer different questions:
  places      distinct head nouns / ideas      -> the rule's own terms
  overlap     max pairwise Jaccard between ideas -> did the WORDING diverge

    python recon_test/farm_ab.py [--concept TEXT] [--draws N]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import story  # noqa: E402
from story import FARM_ANGLES, IDEA_SYSTEM, content_words, farm_ideas, jaccard  # noqa: E402

MODELS = ["qwen2.5:3b", "phi3:latest", "mistral:latest"]
N_IDEAS = 6

# The smallest change that could plausibly matter: keep every existing rule (no
# plot, no camera moves, one line, under 20 words) and add ONE, in the rule's own
# terms. Nothing else about the brief moves.
SPREAD_SYSTEM = IDEA_SYSTEM.replace(
    "Each idea is ONE line of visual description, under 20 words.",
    "Each idea is ONE line of visual description, under 20 words.\n"
    "\n"
    "Each idea must be in a DIFFERENT PLACE from the others. Six ideas means six "
    "different locations, not six versions of one location with different light. If "
    "two ideas share a place, replace one.")

# Locative markers. An idea names a place after one of these; the head noun of that
# phrase is the place. Deliberately shallow -- it has to be predictable, not clever.
LOC = re.compile(
    r"\b(?:in|inside|on|at|under|over|beneath|behind|beside|near|by|through|"
    r"across|along|within|around|outside|above|below|down|up)\s+"
    r"(?:a|an|the|its|his|her|their|this|that)?\s*"
    r"([a-z][a-z\-]*(?:\s+[a-z][a-z\-]*){0,2})", re.I)

# Words that are never the place itself, so the parser does not stop on them.
STOP = set("""dim dark lit empty old new small large tall wide open closed broken
one two three four five six several many some few lone single still faint pale
deep damp wet cold warm hot grey gray blue red green yellow black white soft hard
long short thin thick heavy light bright shadowy shadowed misty foggy rainy
half bare rusted rusting worn faded cracked dirty clean quiet silent slow fast
night nighttime day daytime morning evening dusk dawn noon afternoon
""".split())


def place_of(idea: str) -> str:
    """The head noun of the first locative phrase, or '' if the idea names none."""
    for m in LOC.finditer(idea):
        words = [w for w in re.findall(r"[a-z][a-z\-]*", m.group(1).lower())
                 if w not in STOP]
        if words:
            return words[-1]
    return ""


def score(ideas) -> dict:
    if not ideas:
        return {"n": 0, "places": 0, "named": 0, "max_overlap": 0.0}
    heads = [place_of(i) for i in ideas]
    named = [h for h in heads if h]
    ws = [content_words(i) for i in ideas]
    pairs = [jaccard(ws[a], ws[b]) for a in range(len(ws)) for b in range(a + 1, len(ws))]
    return {"n": len(ideas), "places": len(set(named)), "named": len(named),
            "max_overlap": round(max(pairs), 2) if pairs else 0.0,
            "heads": heads}


def run(model: str, angle: str, brief: str, system: str) -> dict:
    original = story.IDEA_SYSTEM
    try:
        story.IDEA_SYSTEM = system
        out = farm_ideas(CONCEPT, model, angle, n_ideas=N_IDEAS, verbose=False)
    finally:
        story.IDEA_SYSTEM = original
    ideas = out.get("ideas") or []
    s = score(ideas)
    s.update(model=model, angle=angle.split(":")[0], brief=brief, ideas=ideas)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="within the shadows, friends and foes "
                                        "appear the same")
    ap.add_argument("--draws", type=int, default=1)
    a = ap.parse_args()

    global CONCEPT
    CONCEPT = a.concept
    print(f"concept: {CONCEPT}")
    print(f"{len(MODELS)} models x 2 briefs x 2 angles x {a.draws} draw(s)\n")

    angles = FARM_ANGLES[:2]
    rows = []
    for model in MODELS:
        for angle in angles:
            for brief, system in (("shipped", IDEA_SYSTEM),
                                  ("spread", SPREAD_SYSTEM)):
                for d in range(a.draws):
                    r = run(model, angle, brief, system)
                    rows.append(r)
                    print(f"  {model:<15} {angle[:14]:<14} {brief:<7} "
                          f"ideas={r['n']} named_a_place={r['named']} "
                          f"distinct_places={r['places']} "
                          f"max_word_overlap={r['max_overlap']}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    for brief in ("shipped", "spread"):
        sel = [r for r in rows if r["brief"] == brief]
        places = sum(r["places"] for r in sel)
        named = sum(r["named"] for r in sel)
        ideas = sum(r["n"] for r in sel)
        mo = sum(r["max_overlap"] for r in sel) / max(1, len(sel))
        print(f"  {brief:<8} ideas={ideas}  named a place={named}  "
              f"distinct places={places}  "
              f"({places / max(1, named):.0%} of named places are distinct)  "
              f"mean max word overlap={mo:.2f}")

    # the honest verdict rule, stated before looking at which way it went
    ship = [r for r in rows if r["brief"] == "shipped"]
    spr = [r for r in rows if r["brief"] == "spread"]
    sp = sum(r["places"] for r in ship) / max(1, sum(r["named"] for r in ship))
    pp = sum(r["places"] for r in spr) / max(1, sum(r["named"] for r in spr))
    print(f"\n  distinct-place rate: shipped {sp:.0%} vs spread {pp:.0%}")
    if pp - sp < 0.10:
        print("  VERDICT: the rule does NOT meaningfully change location diversity "
              "on this evidence. Do not add it to the writer.")
    else:
        print("  VERDICT: the rule helps. Candidate for story.py -- but re-run with "
              "more draws before changing a working writer.")
    print("\n  One draw per cell: this is a signal, not a verdict. Raise --draws for"
          "\n  a decision you intend to act on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
