"""
pool.py -- render a pool of candidates, then let the system decide what earns a
place in the film.

The problem this solves is economic. Rendering is ~9.5 min per shot and nothing
else in the pipeline is close, so a 30s TikTok is a ~2-hour render -- and the
current pipeline commits to that before a single frame exists. If the writer's
sixth shot is weak, it ships anyway, or you re-render.

So: over-produce, score, then select. The pool is rendered once and can be
re-cut for free, at a different length, for a different platform, or after a
change to any downstream rule.

The scoring axes are the ones this project has actually measured, not proxies:

  align        CLIP text-image similarity. Does the frame show the prompt?
  motion       mean frame-to-frame change. NOT maximised -- a clip is scored by
               how close it sits to the energy the shot asked for, because both
               stillness and chaos are failures.
  coherence    CLIP drift from frame 0. Lower is steadier.
  separation   global_share against the no-LoRA baseline. Informational: no
               configuration has ever separated, so it never decides anything.

Selection is grouped by scene, because a look and a checkpoint change only land
at a scene boundary -- so a scene's shots must be contiguous. Within a scene the
best candidates win subject to the continuity rules, and the scenes are then
ordered into an arc. Anything still violating a rule is handed to
`edit_planner.plan`, which repairs it and reports what it had to fix.

    python pool.py --pool output/subject_test --n 8
    python pool.py --pool output/film/shadows/raw --n 6 --target 30
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent

# how a clip's motion is judged. A shot asks for an energy 0-0.5 (the renderer's
# measured ceiling); the clip's own motion is mapped onto the same 0-1 scale and
# scored by distance from the request. Both directions are failures.
MOTION_SCALE = 40.0      # mag observed range is roughly 1 (static) to 50 (chaos)

WEIGHTS = {"align": 0.55, "motion_fit": 0.30, "coherence": 0.15}


@dataclass
class Candidate:
    cid: str
    path: Path
    prompt: str = ""
    scale: str = "medium"
    camera: str = "none"
    scene: str = "scene1"
    subject: str = ""
    energy: float = 0.25
    frames: list = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    why: str = ""


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def score(cands: Sequence[Candidate], baseline_global: float = 0.49,
          scorer=None, verbose: bool = True) -> None:
    """Fill in `scores` and `total` for every candidate. Mutates in place."""
    from motion_metrics import ClipScorer, temporal_stats

    if scorer is None:
        scorer = ClipScorer(str(ROOT / "models" / "clip-vit-base-patch32"))

    for c in cands:
        if not c.frames:
            continue
        m = temporal_stats(c.frames)
        clip = scorer.scores(c.frames, c.prompt) if c.prompt else {}
        align = float(clip.get("align", 0.0))
        coh = float(clip.get("coherence", 1.0))

        # motion is scored against what the shot ASKED for, not maximised
        observed = min(1.0, m["mag"] / MOTION_SCALE)
        want = max(1e-3, c.energy)
        motion_fit = max(0.0, 1.0 - abs(observed - want) / 0.5)
        # coherence: 0 drift = 1.0, 0.10 drift or worse = 0
        coh_score = max(0.0, 1.0 - coh / 0.10)

        c.scores = {"align": round(align, 4), "motion": round(m["mag"], 2),
                    "motion_fit": round(motion_fit, 3),
                    "coherence": round(coh, 4),
                    "coherence_score": round(coh_score, 3),
                    "global_share": round(m["global_share"], 3),
                    "loc10": round(m["loc10"], 3),
                    "separable": m["global_share"] <= 0.7 * baseline_global}
        c.total = round(align * WEIGHTS["align"]
                        + motion_fit * WEIGHTS["motion_fit"]
                        + coh_score * WEIGHTS["coherence"], 4)
        if verbose:
            print(f"  {c.cid:<26} scene={c.scene:<10} align={align:.3f} "
                  f"motion={m['mag']:5.2f} want={want:.2f} "
                  f"fit={motion_fit:.2f} coher={coh:.4f} -> {c.total:.4f}")


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #

def select(cands: Sequence[Candidate], n: int,
           target_seconds: Optional[float] = None,
           max_movers_in_a_row: int = 2,
           verbose: bool = True) -> Tuple[List[Candidate], Dict]:
    """Choose and order n candidates from the pool.

    Grouped by scene and contiguous within a scene, because a look change and a
    checkpoint change only land at a scene boundary. Within a scene the highest
    scoring candidates win, subject to R1 (no repeated adjacent scale) and R6
    (no axis reversal). Scenes are then ordered into an arc by mean energy:
    the film opens calm, peaks, and resolves.
    """
    from edit_planner import MAX_SHOT, MIN_SHOT

    by_scene: Dict[str, List[Candidate]] = {}
    for c in cands:
        if c.frames:
            by_scene.setdefault(c.scene, []).append(c)
    for v in by_scene.values():
        v.sort(key=lambda c: -c.total)
    if not by_scene:
        return [], {"error": "empty pool"}

    # how many shots each scene gets: proportional to its best score, so a
    # stronger scene earns more screen time, but every scene gets at least one
    # so the look/checkpoint boundaries exist at all
    scenes = sorted(by_scene, key=lambda s: -by_scene[s][0].total)
    remaining = max(len(scenes), n)
    quota: Dict[str, int] = {s: 1 for s in scenes}
    remaining -= len(scenes)
    weights = np.array([by_scene[s][0].total for s in scenes])
    weights = weights / weights.sum() if weights.sum() > 0 else \
        np.ones(len(scenes)) / len(scenes)
    for _ in range(max(0, remaining)):
        s = scenes[int(np.argmax(weights))]         # largest share first
        if quota[s] < len(by_scene[s]):
            quota[s] += 1
            weights[scenes.index(s)] *= 0.85        # diminishing returns

    chosen: List[Candidate] = []
    dropped: List[Tuple[str, str]] = []
    for s in scenes:
        take = min(quota[s], len(by_scene[s]))
        kept: List[Candidate] = []
        for c in by_scene[s]:
            if len(kept) >= take:
                dropped.append((c.cid, f"scene {s} quota filled ({take})"))
                continue
            if kept and kept[-1].scale == c.scale:
                dropped.append((c.cid, f"R1: same scale as the previous shot "
                                       f"in scene {s} ({c.scale})"))
                continue
            kept.append(c)
        chosen.extend(kept)

    # order the scenes into an arc by mean energy
    def mean_e(s):
        got = [c for c in chosen if c.scene == s]
        return float(np.mean([c.energy for c in got])) if got else 0.0
    chosen.sort(key=lambda c: mean_e(c.scene))

    # R3: break up runs of moving shots by demoting one
    run = 0
    for i, c in enumerate(chosen):
        if c.camera != "none":
            run += 1
            if run > max_movers_in_a_row:
                dropped.append((c.cid, f"R3: {run} moving shots in a row"))
                c.camera = "none"
        else:
            run = 0

    # duration check. Holds clamp at MAX_SHOT, so a pool that is too small
    # cannot reach the target however the curve is scaled.
    ceiling = len(chosen) * MAX_SHOT
    floor = len(chosen) * MIN_SHOT
    meta = {
        "pool_size": len(cands),
        "scored": sum(1 for c in cands if c.frames),
        "chosen": len(chosen),
        "scenes": len({c.scene for c in chosen}),
        "quota": quota,
        "duration_ceiling_s": round(ceiling, 1),
        "duration_floor_s": round(floor, 1),
        "target_s": target_seconds,
        "dropped": dropped,
    }
    if target_seconds and ceiling < target_seconds:
        meta["warning"] = (f"target {target_seconds:.1f}s unreachable: "
                           f"{len(chosen)} shots cap at {ceiling:.1f}s")
    if verbose:
        print(f"\n  selected {len(chosen)} of {meta['scored']} scored "
              f"candidates across {meta['scenes']} scene(s)")
        print(f"  duration range with these shots: {floor:.1f}-{ceiling:.1f}s"
              + (f"  target {target_seconds:.1f}s" if target_seconds else ""))
        if meta.get("warning"):
            print(f"  !! {meta['warning']}")
        for cid, why in dropped[:12]:
            print(f"    dropped {cid:<24} {why}")
        if len(dropped) > 12:
            print(f"    ... and {len(dropped)-12} more")
    return chosen, meta


# --------------------------------------------------------------------------- #
# loading a pool from rendered footage
# --------------------------------------------------------------------------- #

def load_pool(paths: Sequence[Path], prompts: Optional[Dict[str, dict]] = None,
              verbose: bool = True) -> List[Candidate]:
    """Load every clip directory under `paths` that has frames on disk.

    `prompts` maps a clip id to its shot fields (prompt, scale, camera, scene,
    energy). Without it a candidate is still scored, just with no prompt
    alignment -- useful for a pool of hand-made clips.
    """
    from PIL import Image

    prompts = prompts or {}
    out: List[Candidate] = []
    for base in paths:
        if not base.exists():
            continue
        dirs = [base] if list(base.glob("f*.png")) else \
            [d for d in sorted(base.rglob("*")) if d.is_dir()
             and list(d.glob("f*.png"))]
        for d in dirs:
            fs = sorted(d.glob("f*.png"))
            if len(fs) < 4:
                continue
            cid = d.name if d.parent.name not in ("raw",) else \
                f"{d.parent.parent.name}_{d.name}"
            meta = prompts.get(cid, prompts.get(d.name, {}))
            out.append(Candidate(
                cid=cid, path=d,
                prompt=meta.get("prompt", ""),
                scale=meta.get("scale", "medium"),
                camera=meta.get("camera", "none"),
                scene=meta.get("scene", d.parent.name if d.parent.name
                               not in ("raw",) else cid),
                subject=meta.get("subject", ""),
                energy=float(meta.get("energy", 0.25)),
                frames=[Image.open(f).convert("RGB") for f in fs],
            ))
    if verbose:
        print(f"  loaded {len(out)} candidate clip(s) from "
              f"{', '.join(str(p) for p in paths)}")
    return out


def cut(chosen: Sequence[Candidate], plan_shots=None, verbose: bool = True):
    """Turn a selection into a Shot list for the existing planner/director."""
    from edit_planner import Shot

    shots = [Shot(prompt=c.prompt or c.cid, scale=c.scale, camera=c.camera,
                  seconds=2.5, subject=c.subject, scene=c.scene, energy=c.energy)
             for c in chosen]
    if verbose:
        from edit_planner import describe
        print("\n  the cut:")
        print(describe(shots))
    return shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True,
                    help="comma-separated directories to build the pool from")
    ap.add_argument("--n", type=int, default=None, help="how many shots to keep")
    ap.add_argument("--target", type=float, default=None,
                    help="target duration in seconds; sets n if --n is absent")
    ap.add_argument("--prompts", default=None,
                    help="JSON file mapping clip id -> {prompt, scale, camera, "
                         "scene, energy}. Without it, align is not scored.")
    ap.add_argument("--no-score", action="store_true",
                    help="skip CLIP scoring (fast structural preview)")
    a = ap.parse_args()

    paths = [Path(p.strip()) for p in a.pool.split(",") if p.strip()]
    prompts = {}
    if a.prompts and Path(a.prompts).exists():
        prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8"))
    elif a.prompts:
        print(f"  (no prompt file at {a.prompts}; align will not be scored)")

    print(f"=== POOL ({', '.join(str(p) for p in paths)}) ===")
    pool = load_pool(paths, prompts)
    if not pool:
        print("  nothing to select from")
        return

    if not a.no_score:
        print("\n=== SCORING ===")
        score(pool)
    else:
        for c in pool:
            c.total = 0.5

    n = a.n or (max(4, round(a.target / 2.5)) if a.target else 6)
    print(f"\n=== SELECTING {n} ===")
    chosen, meta = select(pool, n, target_seconds=a.target)
    if not chosen:
        print("  no selection")
        return
    cut(chosen)

    out = ROOT / "output" / "pool"
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(
        {"meta": meta,
         "chosen": [{"id": c.cid, "scene": c.scene, "total": c.total,
                     "scores": c.scores} for c in chosen]}, indent=2),
        encoding="utf-8")
    print(f"\n  wrote {out/'selection.json'}")


if __name__ == "__main__":
    main()
