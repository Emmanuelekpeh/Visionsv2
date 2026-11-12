#!/usr/bin/env python3
"""
taste.py -- record which cut a person preferred, and learn what that implies.

WHY. Every A/B in this project ended with "watch them back to back and decide".
That decision was thrown away. The studio cannot learn a per-format organisation
policy from metrics alone: continuity is not quality, and the corpus study says
montage and held coverage are both legitimate. What is missing is LABELS.

This module records one label per comparison:

    {at, a, b, pick, note, tags, metrics: {a: {...}, b: {...}}}

and summarises them: win counts, wins by tag/format, and the measured difference
between the pick and the reject (cut rate, shot-length CV, duration, hold stats).
It is deliberately simple: no model, no training. The point is to start a
dataset and to make the A/B choice a recorded decision instead of a conversation.

    python taste.py add --a <cutA.mp4> --b <cutB.mp4> --pick a --tags montage,coverage
    python taste.py list
    python taste.py summary
    python taste.py films          # what is available to compare

The studio's CUT view calls `record()` from a two-video picker, so the one-click
version stores the same row.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

LABELS = ROOT / "output" / "taste" / "labels.jsonl"
PRODUCTIONS = ROOT / "productions"
FILM_DIR = ROOT / "output" / "film"
PICKS = ("a", "b", "tie")


# --------------------------------------------------------------------------- #
# Available material
# --------------------------------------------------------------------------- #

def available_films() -> List[Dict]:
    """Every rendered film the studio can offer for comparison."""
    out: List[Dict] = []
    for p in sorted(PRODUCTIONS.glob("*/renders/*/film.mp4")):
        slug = p.parent.parent.parent.name
        name = p.parent.name
        out.append({"label": f"{slug}/{name}", "path": str(p),
                    "slug": slug, "kind": "production"})
    for p in sorted(FILM_DIR.glob("*/film.mp4")):
        label = p.parent.name
        if not any(f["path"] == str(p) for f in out):
            out.append({"label": label, "path": str(p), "slug": None,
                        "kind": "film"})
    return out


# --------------------------------------------------------------------------- #
# Metrics (cheap, film-level)
# --------------------------------------------------------------------------- #

def film_metrics(path: str | Path) -> Dict:
    """Duration, cut rate and shot-length rhythm of one delivered film.

    Uses the validated hard-cut detector from `pacing_study` (F1 0.992 on our
    renders), so a taste label also carries the measurable difference between the
    two cuts. Failure to measure is not a failure to record: the pick still counts.
    """
    p = Path(path)
    if not p.exists():
        return {"error": "missing"}
    try:
        from pacing_study import detect_cuts
        det = detect_cuts(p)
    except Exception as exc:                       # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:120]}
    if det.get("error"):
        return {"error": det["error"]}
    dur = float(det["duration"] or 0)
    cuts = det["cuts"]
    bounds = [0.0] + cuts + [dur]
    lens = [b - a for a, b in zip(bounds, bounds[1:]) if b - a > 0.05]
    mean = sum(lens) / len(lens) if lens else 0.0
    sd = ((sum((x - mean) ** 2 for x in lens) / len(lens)) ** 0.5
          if lens else 0.0)
    return {
        "duration_s": round(dur, 2), "shots": len(lens),
        "cuts": len(cuts),
        "cuts_per_min": round(60.0 * len(cuts) / dur, 1) if dur else None,
        "asl_s": round(mean, 2),
        "shot_cv": round(sd / mean, 2) if mean else None,
        "shortest_s": round(min(lens), 2) if lens else None,
        "longest_s": round(max(lens), 2) if lens else None,
    }


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

def record(a: str | Path, b: str | Path, pick: str, note: str = "",
           tags: Optional[Sequence[str]] = None,
           labels_file: Optional[Path] = None) -> Dict:
    if pick not in PICKS:
        raise ValueError(f"pick must be one of {PICKS}, got {pick!r}")
    row = {
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "a": str(a), "b": str(b), "pick": pick,
        "note": note.strip(), "tags": list(tags or []),
        "metrics": {"a": film_metrics(a), "b": film_metrics(b)},
    }
    path = Path(labels_file or LABELS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=True) + "\n")
    return row


def load(labels_file: Optional[Path] = None) -> List[Dict]:
    path = Path(labels_file or LABELS)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def summary(labels_file: Optional[Path] = None) -> Dict:
    rows = load(labels_file)
    if not rows:
        return {"labels": 0, "picks": {}, "by_tag": {}, "notes": []}
    picks = {p: sum(1 for r in rows if r["pick"] == p) for p in PICKS}
    by_tag: Dict[str, Dict[str, int]] = {}
    for r in rows:
        for tag in r.get("tags") or []:
            d = by_tag.setdefault(tag, {p: 0 for p in PICKS})
            d[r["pick"]] = d.get(r["pick"], 0) + 1
    # Do the measured properties separate the pick from the reject? This is the
    # beginning of a preference model: it reports a difference, it does not
    # pretend to explain taste.
    diffs: Dict[str, List[float]] = {}
    for r in rows:
        if r["pick"] == "tie":
            continue
        win = r["metrics"]["a" if r["pick"] == "a" else "b"]
        lose = r["metrics"]["b" if r["pick"] == "a" else "a"]
        for key in ("cuts_per_min", "asl_s", "shot_cv", "duration_s"):
            if win.get(key) is not None and lose.get(key) is not None:
                diffs.setdefault(key, []).append(float(win[key]) - float(lose[key]))
    return {
        "labels": len(rows), "picks": picks, "by_tag": by_tag,
        "winner_minus_reject": {k: round(statistics.fmean(v), 2)
                                for k, v in diffs.items() if v},
        "notes": [r["note"] for r in rows if r.get("note")],
    }


def format_summary(s: Dict) -> str:
    if not s.get("labels"):
        return "no taste labels yet -- compare two cuts and record the pick"
    lines = [f"{s['labels']} label(s): " + " · ".join(
        f"{p} {n}" for p, n in s["picks"].items() if n)]
    for tag, counts in sorted(s["by_tag"].items()):
        lines.append(f"  {tag:<12} " + " · ".join(
            f"{p} {n}" for p, n in counts.items() if n))
    if s.get("winner_minus_reject"):
        lines.append("  the pick, minus the reject: " + " · ".join(
            f"{k} {v:+.2f}" for k, v in s["winner_minus_reject"].items()))
    for n in s.get("notes", [])[-5:]:
        lines.append(f"  note: {n}")
    return "\n".join(lines)


def _print_films() -> None:
    films = available_films()
    if not films:
        print("no rendered films found")
        return
    for i, f in enumerate(films, 1):
        print(f"{i:>2}. {f['label']:<42} {f['path']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Record and summarise cut preferences")
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add")
    add.add_argument("--a", required=True)
    add.add_argument("--b", required=True)
    add.add_argument("--pick", required=True, choices=PICKS)
    add.add_argument("--note", default="")
    add.add_argument("--tags", default="", help="comma list, e.g. montage,coverage")
    sub.add_parser("list")
    sub.add_parser("summary")
    sub.add_parser("films")
    a = ap.parse_args()
    if a.cmd == "films":
        _print_films()
        return 0
    if a.cmd == "add":
        tags = [t.strip() for t in a.tags.split(",") if t.strip()]
        row = record(a.a, a.b, a.pick, a.note, tags)
        print(f"recorded pick={row['pick']} ({row['metrics']['a'].get('cuts')}, "
              f"{row['metrics']['b'].get('cuts')}) cuts · tags {row['tags']}")
        return 0
    if a.cmd == "summary":
        print(format_summary(summary()))
        return 0
    if a.cmd == "list":
        for r in load():
            print(f"{r['at']} · {r['pick']} · {Path(r['a']).name} vs "
                  f"{Path(r['b']).name}" + (f" · {r['note']}" if r.get("note") else ""))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
