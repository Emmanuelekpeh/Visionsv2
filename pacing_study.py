#!/usr/bin/env python3
"""
pacing_study.py -- measure how real short videos are cut, with our own detector.

WHY THIS EXISTS. The pacing section of `PACING_AND_FLOW_RESEARCH.md` originally
quoted two 2026 commercial blogs (551 shots from 9 AI shorts; 1,400 scripts from
one vendor's tool). Those are not a basis for a rule: tiny, self-interested
samples. This tool replaces them with a measurement over a corpus that is already
on this machine:

  * MSR-VTT (`~/Videos/msr_vtt/TrainValVideo`, 7,009 human-made web videos,
    10-30 s each) -- the same target duration band this pipeline shoots for.
  * Our own renders (`productions/*/renders/*/film.mp4`), where the cut times are
    KNOWN from `stats.json`, so the detector can be validated exactly.

It detects hard cuts with a colour-histogram + luma change at 8 samples/s, then
reports the numbers editing theory actually depends on: shots per video, average
and median shot length, the shot-length distribution, cuts per minute, how often
a video is a single held shot, and how much the shot lengths vary inside a video
(uniform 2 s cuts have a low coefficient of variation; films do not).

    python pacing_study.py --validate                     # detector vs known cuts
    python pacing_study.py --corpus msrvtt --limit 400    # own pacing survey
    python pacing_study.py --corpus renders               # our films, compared
    python pacing_study.py --corpus msrvtt --limit 400 --json-out out.json

The detector is deliberately simple and its limitations are reported: it finds
HARD cuts, not dissolves, and a flash can fool it. That is enough for the
question "is a cut every 2 s normal in human short video?", which is the one
this pipeline got wrong.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import continuity

DEFAULT_MSRVTT = Path.home() / "Videos" / "msr_vtt" / "TrainValVideo"
PRODUCTIONS = Path(__file__).resolve().parent / "productions"

SAMPLE_FPS = 8.0          # cut detection works on ~8 samples/s, not every frame
MIN_SHOT_S = 0.5          # shortest plausible shot; below this it is a flash
FRAME_W, FRAME_H = 160, 90

# Cut decision. The two signals fail in different films: a colour histogram
# misses cuts between two grey, similarly-lit scenes (`project-9`), while raw
# luma difference misses cuts between scenes of similar brightness. A cut is
# also a ONE-FRAME SPIKE -- fast motion is sustained, a cut is not -- so the
# decision compares each sample with its neighbours, not against a global mean.
HIST_CUT = 0.40           # strong histogram change, any luma
HIST_CUT_STRONG = 0.55
LUMA_SPIKE_FACTOR = 2.2   # luma change vs the neighbouring samples
HIST_SPIKE_FACTOR = 1.8
LUMA_CUT = 0.05


def _pick_cuts(scores: Sequence[Tuple[float, float, float]]) -> List[float]:
    """Turn (t, hist_dist, luma_diff) samples into cut times, with a min gap."""
    cuts: List[float] = []
    last = -1e9
    n = len(scores)
    for i, (t, hd, pd) in enumerate(scores):
        lo, hi = max(0, i - 2), min(n, i + 3)
        neigh = [s for j, s in enumerate(scores[lo:hi], start=lo) if j != i]
        med_hd = statistics.median([s[1] for s in neigh]) if neigh else 0.0
        med_pd = statistics.median([s[2] for s in neigh]) if neigh else 0.0
        hist_spike = hd > HIST_SPIKE_FACTOR * med_hd + 0.02
        luma_spike = pd > LUMA_SPIKE_FACTOR * med_pd + 0.02
        is_cut = (
            (hd > HIST_CUT and pd > LUMA_CUT)          # both signals agree
            or hd > HIST_CUT_STRONG                     # unmistakable colour cut
            or (luma_spike and hist_spike and pd > LUMA_CUT)  # both spike locally
            or (luma_spike and pd > 0.09 and hd > 0.02)  # grey scene, structure change
        )
        if is_cut and (t - last) >= MIN_SHOT_S:
            cuts.append(round(t, 2))
            last = t
    return cuts


def _hist(frame: np.ndarray) -> np.ndarray:
    """HSV signature from the shared `continuity` module (BGR input here)."""
    return continuity.hist_signature(frame, rgb=False)


def _small(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)


def detect_cuts(path: Path, verbose: bool = False) -> Dict:
    """Hard cuts in one video, with the duration it needs to normalise them."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "error": "cannot open"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = (n_frames / fps) if n_frames and fps else 0.0
    step = max(1, int(round(fps / SAMPLE_FPS)))
    sample_fps = fps / step

    prev_hist = prev_gray = None
    scores: List[Tuple[float, float, float]] = []   # (t, hist_dist, luma_diff)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            t = i / fps
            small = _small(frame)
            h = _hist(small)
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            if prev_hist is not None:
                hd = float(cv2.compareHist(prev_hist, h,
                                           cv2.HISTCMP_BHATTACHARYYA))
                pd = float(np.abs(g - prev_gray).mean())
                scores.append((t, hd, pd))
            prev_hist, prev_gray = h, g
        i += 1
    cap.release()
    if not duration and scores:
        duration = scores[-1][0] + 1.0 / max(1e-6, sample_fps)
    cuts = _pick_cuts(scores)
    if verbose:
        print(f"  {path.name}: {duration:.1f}s, {len(cuts)} cut(s) at {cuts[:8]}")
    return {"path": str(path), "duration": round(duration, 2),
            "cuts": cuts, "n_shots": len(cuts) + 1,
            "error": None}


# --------------------------------------------------------------------------- #
# distribution statistics
# --------------------------------------------------------------------------- #

def _pct(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 2)


def summarise_clips(clips: List[Dict]) -> Dict:
    """Per-clip and pooled shot statistics for one corpus."""
    good = [c for c in clips if not c.get("error")]
    shots: List[float] = []          # every shot's length, pooled over the corpus
    per_clip_asl: List[float] = []
    per_clip_cv: List[float] = []
    single = 0
    durations = []
    for c in good:
        dur = c["duration"]
        cuts = c["cuts"]
        durations.append(dur)
        bounds = [0.0] + cuts + [dur]
        lens = [round(b - a, 2) for a, b in zip(bounds, bounds[1:]) if b - a > 0.05]
        if not lens:
            continue
        shots.extend(lens)
        per_clip_asl.append(round(dur / len(lens), 2))
        if len(lens) == 1:
            single += 1
        if len(lens) >= 3:
            m = statistics.fmean(lens)
            sd = statistics.pstdev(lens)
            if m > 0:
                per_clip_cv.append(round(sd / m, 3))

    def share(pred) -> float:
        return round(100.0 * sum(1 for s in shots if pred(s)) / max(1, len(shots)), 1)

    buckets = {"<0.5s": share(lambda s: s < 0.5),
               "0.5-1s": share(lambda s: 0.5 <= s < 1),
               "1-2s": share(lambda s: 1 <= s < 2),
               "2-3s": share(lambda s: 2 <= s < 3),
               "3-5s": share(lambda s: 3 <= s < 5),
               ">5s": share(lambda s: s >= 5)}
    total_time = sum(durations)
    return {
        "clips": len(good),
        "errors": len(clips) - len(good),
        "total_seconds": round(total_time, 1),
        "total_shots": len(shots),
        "cuts_per_minute": round(60.0 * (len(shots) - len(good)) / max(1e-9, total_time), 2),
        "asl_pooled": round(total_time / max(1, len(shots)), 2),
        "shot_median": _pct(shots, 0.5),
        "shot_p10": _pct(shots, 0.10),
        "shot_p25": _pct(shots, 0.25),
        "shot_p75": _pct(shots, 0.75),
        "shot_p90": _pct(shots, 0.90),
        "shot_max": round(max(shots), 2) if shots else None,
        "clip_asl_median": _pct(per_clip_asl, 0.5),
        "clip_asl_p10": _pct(per_clip_asl, 0.10),
        "clip_asl_p90": _pct(per_clip_asl, 0.90),
        "single_shot_share_pct": round(100.0 * single / max(1, len(good)), 1),
        "shot_length_cv_median": _pct(per_clip_cv, 0.5),
        "uniform_cv_share_pct": round(100.0 * sum(1 for c in per_clip_cv if c < 0.15)
                                      / max(1, len(per_clip_cv)), 1),
        "buckets_pct": buckets,
        "clip_duration_median": _pct(durations, 0.5),
    }


def _print_summary(name: str, s: Dict) -> None:
    print(f"\n=== {name} ===")
    print(f"  clips {s['clips']} ({s['errors']} unreadable) · "
          f"{s['total_seconds'] / 60:.1f} min · {s['total_shots']} shots")
    print(f"  cuts/min {s['cuts_per_minute']} · ASL pooled {s['asl_pooled']}s · "
          f"shot median {s['shot_median']}s (p10 {s['shot_p10']}, p90 {s['shot_p90']})")
    print(f"  clip ASL median {s['clip_asl_median']}s "
          f"(p10 {s['clip_asl_p10']}, p90 {s['clip_asl_p90']})")
    print(f"  single-shot clips {s['single_shot_share_pct']}% · "
          f"shot-length CV median {s['shot_length_cv_median']} · "
          f"near-uniform clips (CV<0.15) {s['uniform_cv_share_pct']}%")
    print("  shot-length distribution: " + " · ".join(
        f"{k} {v}%" for k, v in s["buckets_pct"].items()))


# --------------------------------------------------------------------------- #
# corpora
# --------------------------------------------------------------------------- #

def corpus_msrvtt(directory: Path, limit: int, seed: int) -> Dict:
    files = sorted(directory.glob("*.mp4"))
    if limit and limit < len(files):
        rng = random.Random(seed)
        files = rng.sample(files, limit)
    clips = []
    for i, f in enumerate(files, 1):
        clips.append(detect_cuts(f))
        if i % 50 == 0:
            print(f"  {i}/{len(files)} clips", flush=True)
    return {"dataset": "msr_vtt", "dir": str(directory),
            "sample_limit": limit, "seed": seed,
            "summary": summarise_clips(clips), "clips": clips}


def render_ground_truth(stats_path: Path) -> List[float]:
    meta = json.loads(stats_path.read_text(encoding="utf-8"))
    return [round(float(s["in_s"]), 2) for s in meta.get("shots") or []
            if float(s.get("in_s") or 0) > 0]


def corpus_renders() -> Dict:
    """Our own films, where stats.json gives the true cut times."""
    items = []
    for stats in sorted(PRODUCTIONS.glob("*/renders/*/stats.json")):
        d = stats.parent
        mp4 = d / "film.mp4"
        if not mp4.exists():
            continue
        truth = render_ground_truth(stats)
        det = detect_cuts(mp4)
        items.append({"name": f"{d.parent.parent.name}/{d.name}",
                      "duration": det["duration"], "truth_cuts": truth,
                      "detected_cuts": det["cuts"], "error": det["error"]})
    clips = [{"duration": it["duration"], "cuts": it["detected_cuts"],
              "error": it["error"]} for it in items]
    return {"dataset": "renders", "summary": summarise_clips(clips),
            "items": items}


def validate() -> int:
    """Detector accuracy against the known cut times of our own renders."""
    run = corpus_renders()
    tp = fp = fn = 0
    print(f"\n=== detector validation on our renders (known cuts) ===")
    print(f"  {'render':<38} {'shots':>5} {'true':>5} {'det':>5} {'TP':>3} "
          f"{'FP':>3} {'FN':>3}")
    for it in run["items"]:
        truth = it["truth_cuts"]
        det = it["detected_cuts"]
        matched = [t for t in truth if any(abs(t - d) <= 0.35 for d in det)]
        extra = [d for d in det if not any(abs(d - t) <= 0.35 for t in truth)]
        tp += len(matched)
        fp += len(extra)
        fn += len(truth) - len(matched)
        print(f"  {it['name']:<38} {len(truth) + 1:>5} {len(truth):>5} "
              f"{len(det):>5} {len(matched):>3} {len(extra):>3} "
              f"{len(truth) - len(matched):>3}")
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    print(f"\n  precision {prec:.3f} · recall {rec:.3f} · F1 "
          f"{2 * prec * rec / max(1e-9, prec + rec):.3f}  (tolerance 0.35 s)")
    print("  The detector is validated on hard cuts; it will miss dissolves and "
          "can be fooled by flashes. MSR-VTT is short web video, so hard cuts "
          "dominate.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure editing pacing, our own corpus")
    ap.add_argument("--corpus", choices=["msrvtt", "renders"], default=None)
    ap.add_argument("--dir", default=str(DEFAULT_MSRVTT))
    ap.add_argument("--limit", type=int, default=300,
                    help="random clip sample (0 = all)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    if a.validate:
        return validate()
    if not a.corpus:
        ap.error("choose --corpus or --validate")

    if a.corpus == "msrvtt":
        report = corpus_msrvtt(Path(a.dir), a.limit, a.seed)
    else:
        report = corpus_renders()
    _print_summary(a.corpus, report["summary"])

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  json: {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
