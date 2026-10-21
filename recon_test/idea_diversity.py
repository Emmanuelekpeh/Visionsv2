"""Dedup and diversity report on a proposal's scout ideas.

The complaint this answers: "all the scout ideas are repetitions of the same thing".
`story.py` gets 18 ideas from 3 farms and hands ALL of them to the showrunner. That
is the right thing for the WRITER -- more material is more to choose from -- and the
wrong thing for a HUMAN, who is shown 18 near-identical lines and asked to care.

So this reports, for one proposal:
  * how many ideas there are, and how many are DISTINCT by content overlap
  * the overlap distribution, so "repetitive" is a number and not an impression
  * which ideas survive a dedup pass

    python recon_test/idea_diversity.py <proposal.json> [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from story import content_words, jaccard  # noqa: E402


def ideas_of(raw: dict):
    out = []
    for f in raw.get("ideas") or []:
        for idea in f.get("ideas") or []:
            out.append((f.get("model", "?"), f.get("angle", ""), idea))
    return out


def dedup(items, threshold: float = 0.42):
    """Keep an idea only if it does not overlap an already-kept one.

    Jaccard on content-word sets -- the same measure `story._provenance` uses to
    decide whether a final prompt traces to an idea, so the two cannot disagree
    about what "the same" means.
    """
    kept, dropped = [], []
    for model, angle, idea in items:
        w = content_words(idea)
        hit = None
        for k in kept:
            if jaccard(w, k[3]) >= threshold:
                hit = k
                break
        if hit is None:
            kept.append((model, angle, idea, w))
        else:
            dropped.append((model, angle, idea, hit[2],
                            round(jaccard(w, hit[3]), 3)))
    return kept, dropped


def motif_saturation(items, top: int = 3):
    """The repetition that word-overlap CANNOT see, and the one people actually mean.

    Measured on the `shadows` concept: six qwen ideas with pairwise overlap 0.19 or
    less -- each one lexically distinct -- that are five ALLEYS. The words differ;
    the shot does not. A Jaccard measure reports that as perfectly diverse, which is
    why the "18 ideas" never looked repetitive to any automated check and looked
    like the same idea six times to a person.

    So: count how many ideas share each content word, and report the words that
    dominate. A term in more than a third of the ideas is a saturated motif.
    """
    from collections import Counter
    counts: Counter = Counter()
    for _, _, idea in items:
        # count a word once per idea, so one rambling line cannot dominate
        for w in set(content_words(idea)):
            counts[w] += 1
    n = len(items)
    hot = [(w, c) for w, c in counts.most_common(40)
           if c >= max(3, n / 3) and len(w) > 3]
    return n, hot[:top]


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        d = ROOT / "output" / "film" / "proposals"
        paths = [str(p) for p in sorted(d.glob("*.json"),
                                        key=lambda p: p.stat().st_mtime,
                                        reverse=True)[:3]]
    for p in paths:
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        items = ideas_of(raw)
        if not items:
            continue
        kept, dropped = dedup(items)
        # pairwise overlap, for the distribution
        ws = [content_words(i[2]) for i in items]
        pairs = []
        for a in range(len(ws)):
            for b in range(a + 1, len(ws)):
                pairs.append(jaccard(ws[a], ws[b]))
        pairs.sort(reverse=True)
        over = [x for x in pairs if x >= 0.30]

        print("=" * 78)
        print(Path(p).name[:74])
        print(f"  ideas: {len(items)}   same-words-at-0.42: {len(dropped)} dropped")
        print(f"  word overlap: max {pairs[0]:.2f}, median "
              f"{pairs[len(pairs)//2]:.2f} — "
              f"{'LEXICALLY DIVERSE' if pairs[0] < 0.30 else 'repetitive'}")
        n, hot = motif_saturation(items)
        if hot:
            print(f"  SATURATED MOTIFS (present in >=1/3 of the {n} ideas):")
            for w, c in hot:
                which = [i[2][:52] for i in items if w in content_words(i[2])]
                print(f"    '{w}' in {c}/{n} ideas:")
                for ww in which[:6]:
                    print(f"        {ww}")
        else:
            print("  no saturated motif at the word level")
        # per-farm internal overlap: is one scout repeating itself?
        for f in raw.get("ideas") or []:
            iset = [content_words(i) for i in (f.get("ideas") or [])]
            ip = [jaccard(iset[a], iset[b]) for a in range(len(iset))
                  for b in range(a + 1, len(iset))]
            if ip:
                print(f"    within {f.get('model','?'):<16} "
                      f"max pair overlap {max(ip):.2f}  "
                      f"({sum(1 for x in ip if x >= 0.30)}/{len(ip)} pairs >=0.30)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
