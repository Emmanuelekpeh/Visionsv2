#!/usr/bin/env python3
"""
pacing_audit.py -- how long are our shots REALLY, and what does that cost?

The pipeline has two independent notions of shot length and nothing compares
them:

  GENERATED motion   `settings.frames` raw frames go through RAFT x2, giving
                     `2*frames - 1` delivered frames at `settings.fps`. That is
                     the amount of real motion a shot contains. At the default
                     16 frames / 16 fps it is 31 frames = **1.94 s**.
  DELIVERED hold     `hold_frames()` holds the shot for `round(seconds*fps)`
                     frames. `make_film.render` resamples the source clip to the
                     hold by index, so any hold longer than 1.94 s duplicates
                     frames: motion plays slower with judder.

The ratio is `stretch`. At 1.0 the film plays the motion the renderer actually
generated. Above ~1.3 the duplication is visible. The shipped UI path takes
`seconds` straight from the WRITER (film_store.shots_from_proposal keeps them;
the store never re-runs edit_planner), so the writer's guess is what decides
whether a shot is smooth. This tool makes that visible.

    python pacing_audit.py                     # audit every production
    python pacing_audit.py --store longing     # one production
    python pacing_audit.py --render PATH/stats.json
    python pacing_audit.py --budget            # length -> shots -> hours
    python pacing_audit.py --curve 6           # what edit_planner would assign

Read-only. It changes nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
PRODUCTIONS = ROOT / "productions"

# Measured on this machine (make_film.py MIN_PER_SHOT / README): 4-step
# Lightning at 512x512, ~43 s per raw frame, ~57 s model load per job (amortised).
SEC_PER_RAW_FRAME = 43.0
DEFAULT_FPS = 16
DEFAULT_FRAMES = 16
JUDDER_AT = 1.3          # duplication becomes visible around here
SEVERE_AT = 2.0


def natural_seconds(frames: int = DEFAULT_FRAMES, fps: int = DEFAULT_FPS) -> float:
    """Delivered duration of the motion the model actually generated.

    RAFT x2 of n frames gives 2n-1 frames (the first and last are originals).
    """
    return (2 * frames - 1) / fps


def hold_frames(seconds: float, fps: int = DEFAULT_FPS) -> int:
    """MUST match handheld.hold_frames / make_film.render."""
    return max(1, round(float(seconds) * fps))


def judge(seconds: float, frames: int = DEFAULT_FRAMES,
          fps: int = DEFAULT_FPS, camera: str = "none") -> Dict:
    nat = natural_seconds(frames, fps)
    hold = hold_frames(seconds, fps)
    smooth = max(1, 2 * frames - 1)
    stretch = hold / smooth
    if stretch <= 1.0 + 1e-9:
        verdict = "trim" if hold < smooth else "ok"
    elif stretch <= JUDDER_AT:
        verdict = "ok"
    elif stretch <= SEVERE_AT:
        verdict = "judder"
    else:
        verdict = "severe"
    # a static shot stretched duplicates a still image, which the eye accepts;
    # a camera move stretched duplicates MOTION, which reads as stutter
    if verdict in ("judder", "severe") and camera not in ("none", "", None):
        verdict += "+move"
    return {"hold_frames": hold, "smooth_frames": smooth, "stretch": round(stretch, 2),
            "natural_s": round(nat, 2), "unique_fps": round(fps * min(1.0,
            smooth / hold), 1), "verdict": verdict}


# --------------------------------------------------------------------------- #
# productions
# --------------------------------------------------------------------------- #

def load_production(slug: str) -> Optional[Dict]:
    d = PRODUCTIONS / slug
    pj = d / "project.json"
    if not pj.exists():
        return None
    project = json.loads(pj.read_text(encoding="utf-8"))
    shots = []
    for sid in project.get("timeline") or []:
        f = d / "shots" / f"{sid}.json"
        if f.exists():
            shots.append(json.loads(f.read_text(encoding="utf-8")))
    project["_shots"] = shots
    project["_dir"] = str(d)
    return project


def audit_project(project: Dict, verbose: bool = True) -> Dict:
    st = project.get("settings") or {}
    fps = int(st.get("fps") or DEFAULT_FPS)
    frames = int(st.get("frames") or DEFAULT_FRAMES)
    shots = project["_shots"]
    rows = []
    for s in shots:
        j = judge(float(s.get("seconds") or 2.5), frames, fps,
                  s.get("camera") or "none")
        rows.append({"id": s.get("id"), "beat": s.get("beat") or "?",
                     "scale": s.get("scale") or "?", "camera": s.get("camera") or "none",
                     "seconds": float(s.get("seconds") or 2.5), **j})
    total = sum(r["seconds"] for r in rows)
    stretches = [r["stretch"] for r in rows] or [0]
    out = {
        "slug": project["slug"], "fps": fps, "frames": frames,
        "shots": len(rows), "rows": rows,
        "total_s": round(total, 2),
        "mean_s": round(total / max(1, len(rows)), 2),
        "min_s": round(min((r["seconds"] for r in rows), default=0), 2),
        "max_s": round(max((r["seconds"] for r in rows), default=0), 2),
        "natural_s": round(natural_seconds(frames, fps), 2),
        "stretch_mean": round(sum(stretches) / len(stretches), 2),
        "stretch_max": max(stretches),
        "judder_shots": sum(1 for r in rows if r["verdict"] in
                            ("judder", "severe", "judder+move", "severe+move")),
        "movers": sum(1 for r in rows if r["camera"] not in ("none", "", None)),
        "scenes": len({s.get("scene") for s in shots}),
        "beats": "".join({"establish": "E", "develop": "d", "turn": "T",
                          "resolve": "R"}.get(s.get("beat") or "", "?")
                         for s in shots),
        "beats_monotonic": _beats_monotonic(shots),
        **organisation(shots),
        "problems": project.get("problems") or [],
        "deliverable": project.get("deliverable"),
        "render_hours_measured": _render_hours(project),
    }
    if verbose:
        _print_project(out)
    return out


def organisation(shots: List[Dict]) -> Dict:
    """How the film is ORGANISED, not just how long its shots are.

    The failure this measures, in one sentence: "two seconds, hard stop, new
    scenery, repeat". A film is grouped into scenes/sequences, and the shots
    inside a scene cover one action or place. This project's writer is asked for
    2-3 scene labels, but the labels are cosmetic unless the shots inside a scene
    actually relate -- so both the STRUCTURE (runs, cuts, single-shot scenes) and
    the CONTENT CONTINUITY (do adjacent same-scene shots share subject/imagery?)
    are measured. `story.content_words`/`jaccard` are reused deliberately: one
    definition of "the same thing" across the audit and the writer validation.
    """
    from story import content_words, jaccard

    scenes = [str(s.get("scene") or "?") for s in shots]
    if not shots:
        return {}
    # contiguous scene runs (a revisited scene is a NEW run, and interleaving is
    # visible as more runs than distinct scenes)
    runs: List[str] = []
    lens: List[int] = []
    for sc in scenes:
        if runs and runs[-1] == sc:
            lens[-1] += 1
        else:
            runs.append(sc)
            lens.append(1)
    scene_cuts = sum(1 for i in range(1, len(scenes))
                     if scenes[i] != scenes[i - 1])

    within, across, same_subject = [], [], []
    for i in range(1, len(shots)):
        s0, s1 = shots[i - 1], shots[i]
        ov = jaccard(content_words(s0.get("prompt", "")),
                     content_words(s1.get("prompt", "")))
        if scenes[i] == scenes[i - 1]:
            within.append(ov)
            if (s0.get("subject") and s0.get("subject") == s1.get("subject")):
                same_subject.append(1)
        else:
            across.append(ov)

    holds = [round(float(s.get("seconds") or 2.5), 2) for s in shots]
    uniform = longest = 1
    for i in range(1, len(holds)):
        uniform = uniform + 1 if holds[i] == holds[i - 1] else 1
        longest = max(longest, uniform)
    mean_hold = sum(holds) / len(holds)
    hold_sd = (sum((x - mean_hold) ** 2 for x in holds) / len(holds)) ** 0.5
    hold_cv = hold_sd / max(1e-9, mean_hold)

    # a camera move that hands off to new scenery: the move has no continuation
    move_to_scene_cut = sum(
        1 for i in range(len(shots) - 1)
        if (shots[i].get("camera") or "none") != "none"
        and scenes[i + 1] != scenes[i])

    return {
        "scene_runs": len(runs),
        "scene_cuts": scene_cuts,
        "single_shot_scenes": sum(1 for n in lens if n == 1),
        "max_scene_len": max(lens, default=0),
        "shots_per_scene": round(len(shots) / max(1, len(set(scenes))), 2),
        "within_scene_overlap": round(sum(within) / len(within), 3) if within else None,
        "across_scene_overlap": round(sum(across) / len(across), 3) if across else None,
        "same_subject_within": (f"{len(same_subject)}/{len(within)}"
                                if within else "0/0"),
        "hold_sd": round(hold_sd, 2),
        "hold_cv": round(hold_cv, 2),
        "uniform_run": longest,
        "move_before_scene_cut": move_to_scene_cut,
    }


def _beats_monotonic(shots: List[Dict]) -> bool:
    order = {"establish": 0, "develop": 1, "turn": 2, "resolve": 3}
    got = [order[s.get("beat")] for s in shots if (s.get("beat") in order)]
    return all(got[i + 1] >= got[i] for i in range(len(got) - 1))


def _render_hours(project: Dict) -> Optional[float]:
    secs = [h.get("seconds") for h in project.get("history") or []
            if h.get("act") == "generate" and h.get("seconds")]
    if not secs:
        return None
    return round(sum(secs) / 3600, 2)


def _print_project(o: Dict) -> None:
    print(f"\n=== {o['slug']}  ({o['shots']} shots, {o['total_s']:.1f}s, "
          f"{o['fps']}fps, {o['frames']} raw frames) ===")
    print(f"  natural motion per shot: {o['natural_s']:.2f}s  "
          f"(holds above this duplicate frames)")
    print(f"  {'id':<5} {'beat':<9} {'scale':<11} {'camera':<18} {'sec':>5} "
          f"{'hold':>5} {'stretch':>7}  verdict")
    for r in o["rows"]:
        print(f"  {r['id']:<5} {r['beat']:<9} {r['scale']:<11} {r['camera']:<18} "
              f"{r['seconds']:>5.2f} {r['hold_frames']:>5} {r['stretch']:>7.2f}  "
              f"{r['verdict']}")
    print(f"  mean {o['mean_s']:.2f}s - min {o['min_s']:.2f} - max {o['max_s']:.2f} "
          f"- stretch mean {o['stretch_mean']:.2f} max {o['stretch_max']:.2f} - "
          f"{o['judder_shots']} shot(s) over {JUDDER_AT}x")
    print(f"  beats {o['beats']} "
          f"({'monotonic' if o['beats_monotonic'] else 'NON-MONOTONIC'}) - "
          f"{o['scenes']} scene(s) - {o['movers']} camera move(s)")
    print(f"  organisation: {o['scene_runs']} scene run(s) - "
          f"{o['scene_cuts']} scene cut(s) - {o['single_shot_scenes']} single-shot "
          f"scene(s) - {o['shots_per_scene']} shots/scene")
    print(f"  content: within-scene overlap {o['within_scene_overlap']} vs "
          f"across-scene {o['across_scene_overlap']} - same subject within "
          f"{o['same_subject_within']} adjacent pairs - hold sd {o['hold_sd']}s "
          f"(CV {o['hold_cv']}; human short-form ~0.75) - "
          f"longest uniform run {o['uniform_run']} - camera move -> scene cut "
          f"{o['move_before_scene_cut']}")
    if o["render_hours_measured"]:
        print(f"  measured generation: {o['render_hours_measured']:.1f} h")
    if o["problems"]:
        print(f"  writer problems: {'; '.join(o['problems'])[:220]}")


def audit_render(stats_path: str, verbose: bool = True) -> Dict:
    """Audit a FINISHED render from its stats.json (holds are already decided)."""
    d = Path(stats_path)
    if d.is_dir():
        d = d / "stats.json"
    meta = json.loads(d.read_text(encoding="utf-8"))
    fps = int(meta.get("fps") or DEFAULT_FPS)
    rows = []
    for s in meta.get("shots") or []:
        hold = int(s.get("hold_frames") or 0)
        rows.append({"id": s.get("shot"), "scale": s.get("scale"),
                     "camera": s.get("camera") or "none", "hold_frames": hold,
                     "seconds": round(hold / fps, 2)})
    total = sum(r["seconds"] for r in rows)
    out = {"name": meta.get("name"), "fps": fps, "shots": len(rows),
           "total_s": round(total, 2), "frames": meta.get("total_frames"),
           "rows": rows}
    if verbose:
        print(f"\n=== render {out['name']}  ({out['shots']} shots, "
              f"{out['total_s']}s, {out['fps']}fps) ===")
        print(f"  {'id':<5} {'scale':<12} {'camera':<18} {'hold':>5} {'sec':>5}")
        for r in rows:
            print(f"  {str(r['id']):<5} {str(r['scale']):<12} "
                  f"{r['camera']:<18} {r['hold_frames']:>5} {r['seconds']:>5.2f}")
    return out


# --------------------------------------------------------------------------- #
# budget: length -> shots -> hours
# --------------------------------------------------------------------------- #

def budget_table() -> None:
    """What a target length actually costs, at the renderer's real motion budget.

    The useful result: for a fixed length, render time is ALMOST INDEPENDENT of
    the raw-frame setting. More frames per shot is more generation per shot, but
    fewer shots are needed, and the two cancel. What the setting changes is the
    CREATIVE split -- many short cuts vs fewer long shots -- not the bill.
    """
    lengths = {"reels": 25.0, "tiktok": 30.0, "shorts": 45.0}
    print("\n=== shot count / render cost by length and raw-frame setting ===")
    print("  (renderer: 43 s per raw frame, RAFT x2, delivered at 16 fps)")
    print(f"  {'length':<8} {'frames':>6} {'natural':>8} {'shots':>6} "
          f"{'hours':>6} {'stretch at 6 shots':>19}")
    for name, target in lengths.items():
        for frames in (12, 16, 24, 32):
            nat = natural_seconds(frames)
            shots = max(1, math.ceil(target / nat))
            hours = shots * frames * SEC_PER_RAW_FRAME / 3600
            stretch6 = (target / 6) / nat
            print(f"  {name:<8} {frames:>6} {nat:>7.2f}s {shots:>6} "
                  f"{hours:>6.1f} {stretch6:>18.2f}x")
        print()

    print("  How to read it:")
    print("  - stretch at 6 shots = how much the renderer would have to duplicate")
    print("    frames to hit the target with only six shots. 1.0 = no duplication;")
    print(f"    above {JUDDER_AT} the motion judders.")
    print("  - hours is nearly constant down a column: a longer film costs render")
    print("    time, and buying length by raising `frames` trades shots for")
    print("    smoothness at the same price. It cannot be bought for free.")
    print("  - the current UI stores the WRITER's `seconds` (1.5-3.6 s), so a")
    print("    film's length is whatever the writer guessed -- not the target.")


def curve_table(n: int = 6) -> None:
    """What edit_planner assigns now, and what it costs in duplication."""
    from edit_planner import _rhythm_curve_beats, Shot
    demo = [Shot(f"shot{i + 1}", "medium", "none", 2.5,
                 beat=["establish", "develop", "turn", "resolve"][
                     min(i, 3)]) for i in range(n)]
    planned = _rhythm_curve_beats(demo)
    print(f"\n=== edit_planner rhythm curve, {n} shots (beat-inferred) ===")
    print(f"  {'#':>2} {'beat':<10} {'seconds':>8} {'hold':>5} {'stretch':>8} verdict")
    for i, (s, d) in enumerate(zip(demo, planned), 1):
        j = judge(d, DEFAULT_FRAMES, DEFAULT_FPS, "none")
        print(f"  {i:>2} {s.beat:<10} {d:>7.2f}s {j['hold_frames']:>5} "
              f"{j['stretch']:>8.2f} {j['verdict']}")
    print(f"  total {sum(planned):.1f}s - natural total {n * natural_seconds():.1f}s "
          f"- mean stretch {sum(judge(d)['stretch'] for d in planned) / n:.2f}")
    print("  The writer is told 'planner imposes a rhythm curve and will overwrite")
    print("  your seconds' -- TRUE on the CLI path, FALSE on the UI/store path,")
    print("  which keeps the writer's seconds. The two paths disagree by design.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=None,
                    help="production slug (default: all)")
    ap.add_argument("--render", default=None, help="a stats.json or its directory")
    ap.add_argument("--budget", action="store_true")
    ap.add_argument("--curve", type=int, default=0, metavar="N")
    ap.add_argument("--json", action="store_true", help="dump audits as JSON")
    a = ap.parse_args()

    audits: List[Dict] = []
    if a.render:
        audits.append(audit_render(a.render))
    elif a.store:
        p = load_production(a.store)
        if p is None:
            print(f"no production {a.store!r} under {PRODUCTIONS}")
            return 1
        audits.append(audit_project(p))
    else:
        for d in sorted(PRODUCTIONS.iterdir()):
            p = load_production(d.name) if d.is_dir() else None
            if p and p["_shots"]:
                audits.append(audit_project(p))

    if audits:
        print("\n=== summary ===")
        print(f"  {'production':<16} {'shots':>5} {'sec':>6} {'mean':>5} "
              f"{'natural':>7} {'stretch':>7} {'judder':>6} {'moves':>5} "
              f"{'beats':>7} {'render h':>8}")
        for o in audits:
            print(f"  {o.get('slug') or o.get('name') or '?':<16} "
                  f"{o.get('shots', 0):>5} {o.get('total_s', 0):>6.1f} "
                  f"{o.get('mean_s', 0):>5.2f} {o.get('natural_s', 0):>7.2f} "
                  f"{o.get('stretch_mean', 0):>7.2f} {o.get('judder_shots', 0):>6} "
                  f"{o.get('movers', 0):>5} {o.get('beats', '?'):>7} "
                  f"{o.get('render_hours_measured') or 0:>8.1f}")
        if a.json:
            print(json.dumps(audits, indent=2))

        # ORGANISATION, separately. Shot-length health can look fine while the
        # film is still "2 s, hard stop, new scenery, repeat".
        org = [o for o in audits if o.get("scene_runs") is not None]
        if org:
            print("\n=== organisation (scene / coverage / rhythm) ===")
            print(f"  {'production':<16} {'runs':>5} {'cuts':>5} {'1-shot':>7} "
                  f"{'sh/scn':>7} {'within':>7} {'across':>7} {'same subj':>10} "
                  f"{'hold cv':>8} {'uni run':>8} {'mv->cut':>7}")
            for o in org:
                print(f"  {o.get('slug') or '?':<16} {o['scene_runs']:>5} "
                      f"{o['scene_cuts']:>5} {o['single_shot_scenes']:>7} "
                      f"{o['shots_per_scene']:>7.2f} "
                      f"{str(o['within_scene_overlap']):>7} "
                      f"{str(o['across_scene_overlap']):>7} "
                      f"{o['same_subject_within']:>10} "
                      f"{o['hold_cv']:>8.2f} {o['uniform_run']:>8} "
                      f"{o['move_before_scene_cut']:>7}")
            print("  reading: within >> across = the scene labels mean something. "
                  "within ~ across = cosmetic labels. hold cv: human short-form "
                  "median 0.75 (PACING_MEASURED_STUDY.md); ours ~0.17. "
                  "mv->cut = a camera move that hands off to new scenery.")

    if a.budget:
        budget_table()
    if a.curve:
        curve_table(a.curve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
