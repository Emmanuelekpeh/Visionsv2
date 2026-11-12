#!/usr/bin/env python3
"""
compare_showrunners.py -- which model should run the ensemble?

The ensemble has one model writing the final shot list: the SHOWRUNNER. That
model decides what survives from the scouts' idea pool and whether the shots add
up to a film. It is the only text that can reach a render, so it is the only text
worth comparing.

WHY THIS EXISTS, and what was wrong with the first version
----------------------------------------------------------
The pipeline was changed to a different showrunner and quality visibly dropped;
the old choice was blamed for malformed JSON. Both claims are testable, but not
the way the first version of this script did it. Three flaws made its numbers
meaningless:

  1. The farms were re-run for every model. Farms are non-deterministic, so each
     model was handed DIFFERENT ideas and the comparison measured farm luck.
     Fixed: the idea pool is collected ONCE and injected (`farm_out`).
  2. It read scorecard keys that do not exist (`writer_story_breaks`,
     `total_repairs`, `planner_r1_scale`), so every model scored zero.
     Fixed: the metric names are taken from `story.scorecard`'s actual output.
  3. It ran each model once. HANDOFF §11.3 already measured that a single
     showrunner run says nothing: structural repairs were 0, 0 and 2 across three
     runs of the same model. Fixed: `--runs` repeats each model and the report
     shows the spread, not just the mean.

The run also keeps every raw reply. When a model emits malformed JSON, the
evidence (the text, the parse error, whether it was TRUNCATED) is saved under
`replies/`, because "the model did something weird" is not actionable and the
raw reply is.

    # the actual question: granite (shipped) vs qwen2.5:7b (the change)
    python compare_showrunners.py --models "granite4.1:8b,qwen2.5:7b" --runs 3

    # add a low baseline and a JSON stress test at 16 shots
    python compare_showrunners.py --models "granite4.1:8b,qwen2.5:7b,mistral:latest,llama3:latest" \
        --runs 2 --stress-shots 16

    # reuse the idea pool from a previous run (no farm cost)
    python compare_showrunners.py --ideas output/showrunner_test/<stamp>/ideas.json

Read the output as: a model with 0 story breaks in 3/3 runs and 3 minutes per
attempt beats one with 1 break in 3/3 and 9 minutes, because the break is a film
that does not develop and the minutes are yours.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import story  # noqa: E402
from edit_planner import Shot, plan  # noqa: E402

OUT = ROOT / "output" / "showrunner_test"
DEFAULT_MODELS = ["granite4.1:8b", "qwen2.5:7b", "mistral:latest"]
FPS = 16                 # delivered frame rate (make_film.FPS)
RAW_FRAMES = 16          # default settings.frames


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def collect_ideas(concept: str, farms: Sequence[str], n_ideas: int = 6,
                  verbose: bool = True) -> List[Dict]:
    """One farm pass. The SAME pool is then shown to every showrunner."""
    pool: List[Dict] = []
    for i, m in enumerate(farms):
        angle = story.FARM_ANGLES[i % len(story.FARM_ANGLES)]
        if verbose:
            print(f"  [farm {i + 1}/{len(farms)}] {m} ({angle.split(':')[0]}) ...",
                  flush=True)
        f = story.farm_ideas(concept, m, angle, n_ideas=n_ideas, verbose=verbose)
        pool.append(f)
        if verbose:
            print(f"    -> {len(f.get('ideas') or [])} ideas"
                  + (f" ({f.get('error')})" if f.get("error") else ""), flush=True)
    return pool


# --------------------------------------------------------------------------- #
# measuring one proposal
# --------------------------------------------------------------------------- #

def _stretch(seconds: float, frames: int = RAW_FRAMES, fps: int = FPS) -> Dict:
    """How much of this hold is real generated motion (2*frames-1 delivered frames).

    Above 1.0 the renderer duplicates frames: `render()` resamples the source
    clip to the hold by index, so 2x means every generated frame appears twice and
    motion plays at half speed with judder. This is the number the pacing report
    is about, so the comparison reports it rather than ignoring it.
    """
    smooth = max(1, 2 * frames - 1)
    hold = max(1, round(seconds * fps))
    return {"hold_frames": hold, "smooth_frames": smooth,
            "stretch": round(hold / smooth, 2),
            "unique_fps": round(fps * min(1.0, smooth / hold), 1)}


def measure(p: story.Proposal, requested: int, frames: int, fps: int,
            parse_mode: str = "", wall_s: float = 0.0,
            attempts: int = 0) -> Dict:
    """Everything one showrunner run can be scored on, without a render.

    Two timing readings are kept because the pipeline has TWO timing paths:
      * `shipped` is what the UI/store renders: the writer's own `seconds`
        (shots_from_proposal keeps them; the store never re-runs the planner).
      * `planned` is what the CLI --shots llm path renders: edit_planner's
        rhythm curve, which overwrites the writer's seconds.
    Reporting one and not the other is how a pacing change can look harmless.
    """
    raw = p.raw or {}
    prov = raw.get("provenance") or {}
    problems = list(p.problems or [])
    shots = list(p.shots)
    planned, report = plan(shots, frames=frames, fps=fps)

    writer_seconds = [round(s.seconds, 2) for s in shots]
    planned_seconds = [round(s.seconds, 2) for s in planned]
    stresses = [_stretch(s, frames, fps) for s in writer_seconds]
    planned_stresses = [_stretch(s, frames, fps) for s in planned_seconds]

    scales = [s.scale for s in shots]
    adjacent = [i for i in range(len(scales) - 1) if scales[i] == scales[i + 1]]

    # self-repetition: does the film restate one image? (mean pairwise Jaccard)
    words = [story.content_words(s.prompt) for s in shots]
    pairs = [(len(a & b) / max(1, len(a | b)))
             for i, a in enumerate(words) for b in words[i + 1:]]
    self_rep = round(sum(pairs) / len(pairs), 3) if pairs else 0.0

    beats = [getattr(s, "beat", "") for s in shots]
    order = [story.BEATS.index(b) for b in beats if b in story.BEATS]
    beats_monotonic = all(order[i + 1] >= order[i] for i in range(len(order) - 1))

    return {
        "model": p.model,
        "shots": len(shots),
        "requested": requested,
        "shot_count_delta": (len(shots) - requested) if requested else 0,
        "logline": p.logline,
        "story_shape": story.story_shape(shots),
        "beats": beats,
        "beats_monotonic": beats_monotonic,
        "story_breaks": sum(1 for x in problems if x.startswith("story:")),
        "story_warnings": sum(1 for x in problems if x.startswith("story-link:")),
        "rule_breaks": len(problems),
        "prompt_leaks": sum(1 for x in problems if "carries field values" in x),
        "jump_cuts": len(adjacent),
        "scenes": len({getattr(s, "scene", "") for s in shots}),
        "single_scene": len({getattr(s, "scene", "") for s in shots}) < 2,
        "self_repetition": self_rep,
        "parse_mode": parse_mode,
        "wall_s": round(wall_s, 1),
        "attempts": attempts,
        "eval_count": p.eval_count,
        "writer_seconds": writer_seconds,
        "writer_total_s": round(sum(writer_seconds), 2),
        "writer_mean_s": round(sum(writer_seconds) / max(1, len(writer_seconds)), 2),
        "planned_total_s": round(sum(planned_seconds), 2),
        "planned_mean_s": round(sum(planned_seconds) / max(1, len(planned_seconds)), 2),
        "planner_changes": len(report.changes),
        "shipped_stretch_mean": round(
            sum(x["stretch"] for x in stresses) / max(1, len(stresses)), 2),
        "shipped_stretch_max": max((x["stretch"] for x in stresses), default=0.0),
        "shipped_over_1_3": sum(1 for x in stresses if x["stretch"] > 1.3),
        "planned_stretch_mean": round(
            sum(x["stretch"] for x in planned_stresses) / max(1, len(planned_stresses)), 2),
        "planned_stretch_max": max((x["stretch"] for x in planned_stresses),
                                   default=0.0),
        "provenance": {k: v for k, v in prov.items() if not k.startswith("_")},
        "provenance_traced": prov.get("_final_prompts_traced", 0),
        "provenance_total": prov.get("_final_prompts", 0),
        "provenance_ratio": round(
            (prov.get("_final_prompts_traced", 0) /
             max(1, prov.get("_final_prompts", 0))), 2),
        "shot_list": [{"beat": getattr(s, "beat", ""), "scale": s.scale,
                       "camera": s.camera, "seconds": s.seconds,
                       "prompt": s.prompt} for s in shots],
    }


# --------------------------------------------------------------------------- #
# the runs
# --------------------------------------------------------------------------- #

def run_model(model: str, pool: List[Dict], concept: str, n_shots: Optional[int],
              temperature: float, story_retries: int, frames: int, fps: int,
              run: int, replies_dir: Path, verbose: bool,
              num_predict: int = 3000) -> Dict:
    """One showrunner sample. Never raises: a failure IS a result."""
    print(f"\n{'=' * 78}\n  {model}  run {run}\n{'=' * 78}", flush=True)
    t0 = time.time()
    try:
        p = story.write_shots_ensemble(
            concept, showrunner=model, farms=[],  # farms unused: pool injected
            farm_out=pool, n_shots=n_shots, temperature=temperature,
            story_retries=story_retries, fresh=True, verbose=verbose,
            use_cache=False, save_cache=False, num_predict=num_predict)
    except story.ShowrunnerFailure as exc:
        wall = time.time() - t0
        name = f"{model.replace(':', '-')}__run{run}__FAILED.txt"
        (replies_dir / name).write_text(
            json.dumps(exc.replies, indent=2), encoding="utf-8")
        print(f"  !! no usable reply after {len(exc.replies)} attempt(s) "
              f"({wall:.0f}s) -> {name}", flush=True)
        return {"model": model, "run": run, "ok": False, "wall_s": round(wall, 1),
                "error": str(exc)[:400], "attempts": len(exc.replies),
                "replies": exc.replies}
    except Exception as exc:  # noqa: BLE001 -- a crash is a result here
        wall = time.time() - t0
        (replies_dir / f"{model.replace(':', '-')}__run{run}__CRASH.txt"
         ).write_text(traceback.format_exc(), encoding="utf-8")
        print(f"  !! CRASH {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        return {"model": model, "run": run, "ok": False, "wall_s": round(wall, 1),
                "error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    wall = time.time() - t0
    m = measure(p, n_shots or 0, frames, fps, p.parse_mode, wall, p.attempts)
    m["model"] = model          # the candidate, not "ensemble(model+Nfarms)"
    m["proposal_model"] = p.model
    m["run"] = run
    m["ok"] = True
    m["raw_reply_chars"] = len(p.reply_text or "")
    # keep the full reply on disk; the JSON report stays readable
    (replies_dir / f"{model.replace(':', '-')}__run{run}.txt").write_text(
        p.reply_text or "", encoding="utf-8")
    print(f"  {m['shots']} shots, shape {m['story_shape']}, "
          f"{m['story_breaks']} story break(s), {m['story_warnings']} warning(s), "
          f"{m['rule_breaks']} problem(s), parse={m['parse_mode']}, "
          f"{m['writer_total_s']}s shipped, {wall:.0f}s wall", flush=True)
    print(f"  logline: {p.logline[:140]}", flush=True)
    return m


def stress_json(model: str, pool: List[Dict], concept: str, n_shots: int,
                temperature: float, out_dir: Path, verbose: bool,
                num_predict: int = 3000) -> Dict:
    """Ask for a long shot list and record EXACTLY how the reply fails.

    This is the malformed-JSON claim under controlled conditions. Truncation is
    reported separately from a syntax error because they have different fixes:
    truncation is a token budget, a missing comma is a model habit.
    """
    print(f"\n--- JSON stress: {model} ({n_shots} shots requested, "
          f"num_predict={num_predict}) ---", flush=True)
    t0 = time.time()
    try:
        p = story.write_shots_ensemble(
            concept, showrunner=model, farms=[], farm_out=pool,
            n_shots=n_shots, temperature=temperature, story_retries=0,
            fresh=True, verbose=verbose, use_cache=False, save_cache=False,
            num_predict=num_predict)
        wall = time.time() - t0
        (out_dir / f"{model.replace(':', '-')}__stress.txt").write_text(
            p.reply_text or "", encoding="utf-8")
        r = {"model": model, "ok": True, "shots": len(p.shots),
             "requested": n_shots, "parse_mode": p.parse_mode,
             "num_predict": num_predict,
             "story_breaks": sum(1 for x in (p.problems or [])
                                 if x.startswith("story:")),
             "rule_breaks": len(p.problems or []),
             "wall_s": round(wall, 1), "chars": len(p.reply_text or ""),
             "eval_count": p.eval_count}
        print(f"  OK: {len(p.shots)}/{n_shots} shots, parse={p.parse_mode}, "
              f"{r['chars']} chars, {r['eval_count']} tok, {wall:.0f}s", flush=True)
        return r
    except story.ShowrunnerFailure as exc:
        wall = time.time() - t0
        (out_dir / f"{model.replace(':', '-')}__stress_FAILED.txt").write_text(
            json.dumps(exc.replies, indent=2), encoding="utf-8")
        r = {"model": model, "ok": False, "requested": n_shots,
             "num_predict": num_predict,
             "wall_s": round(wall, 1), "attempts": len(exc.replies),
             "failure": str(exc)[:400],
             "replies": [{"failure": x.get("failure"),
                          "chars": x.get("chars"),
                          "eval_count": x.get("eval_count"),
                          "tail": (x.get("text") or "")[-200:]}
                         for x in exc.replies]}
        print(f"  FAILED after {len(exc.replies)} attempt(s): "
              f"{r['failure'][:220]}", flush=True)
        return r
    except Exception as exc:  # noqa: BLE001
        wall = time.time() - t0
        (out_dir / f"{model.replace(':', '-')}__stress_CRASH.txt").write_text(
            traceback.format_exc(), encoding="utf-8")
        print(f"  CRASH {type(exc).__name__}: {exc}", flush=True)
        return {"model": model, "ok": False, "requested": n_shots,
                "wall_s": round(wall, 1),
                "failure": f"{type(exc).__name__}: {str(exc)[:300]}"}


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #

# A JUDGEMENT, not a measurement: what makes a showrunner's output shootable.
#
# The first version of this file ranked purely on story breaks and printed a
# "winner" that contradicted every other column -- one model was cleaner on
# warnings while producing three adjacent medium shots and a single scene (i.e.
# a static mood board). These weights say, explicitly, that a broken story is
# the worst outcome, a jump cut is next, and "one scene" disables the
# palette/model stages that make a film look like one film.
FILMCRAFT_PENALTIES = {
    "story_break": 20.0,       # the film does not develop: the hard failure
    "story_warning": 6.0,      # turn not paid off (lexical proxy; a warning)
    "jump_cut": 5.0,           # R1: same framing scale across a cut
    "single_scene": 10.0,      # no boundary for a palette/model change
    "other_rule": 2.0,         # movers, terse prompts, leaks (not double-counted)
    "self_repetition": 10.0,   # x mean pairwise prompt overlap (0-1)
    "provenance_gap": 15.0,    # x (1 - fraction traced to the idea pool)
}


def filmcraft_score(story_breaks: float, story_warnings: float,
                    jump_cuts: float, single_scene_rate: float,
                    rule_breaks: float, self_repetition: float,
                    provenance_ratio: float) -> float:
    """100 minus the weighted failures. See FILMCRAFT_PENALTIES for why."""
    other = max(0.0, (rule_breaks or 0) - (jump_cuts or 0)
                - (story_breaks or 0) - (story_warnings or 0))
    penalty = (
        FILMCRAFT_PENALTIES["story_break"] * (story_breaks or 0)
        + FILMCRAFT_PENALTIES["story_warning"] * (story_warnings or 0)
        + FILMCRAFT_PENALTIES["jump_cut"] * (jump_cuts or 0)
        + FILMCRAFT_PENALTIES["single_scene"] * (single_scene_rate or 0)
        + FILMCRAFT_PENALTIES["other_rule"] * other
        + FILMCRAFT_PENALTIES["self_repetition"] * (self_repetition or 0)
        + FILMCRAFT_PENALTIES["provenance_gap"] * (1 - (provenance_ratio or 0))
    )
    return round(max(0.0, min(100.0, 100.0 - penalty)), 1)


def summarise(results: List[Dict]) -> List[Dict]:
    """Per-model aggregates. Mean AND spread, because n=1 is how §11.3 misled."""
    by: Dict[str, List[Dict]] = {}
    for r in results:
        by.setdefault(r["model"], []).append(r)
    rows = []
    for model, rs in by.items():
        ok = [r for r in rs if r.get("ok")]
        fails = [r for r in rs if not r.get("ok")]
        def mean(key):
            v = [r[key] for r in ok if r.get(key) is not None]
            return round(sum(v) / len(v), 2) if v else None
        def spread(key):
            v = sorted(r[key] for r in ok if r.get(key) is not None)
            return v
        single_scene_rate = (round(sum(1 for r in ok if r.get("single_scene"))
                                   / len(ok), 2) if ok else None)
        row = {
            "model": model,
            "ok": len(ok), "failed": len(fails),
            "parse_failures": sum(1 for f in fails
                                  if "parse" in (f.get("error") or "").lower()),
            "story_breaks": mean("story_breaks"),
            "story_breaks_runs": spread("story_breaks"),
            "story_warnings": mean("story_warnings"),
            "rule_breaks": mean("rule_breaks"),
            "jump_cuts": mean("jump_cuts"),
            "shots": mean("shots"),
            "shots_runs": spread("shots"),
            "scenes": mean("scenes"),
            "single_scene_rate": single_scene_rate,
            "self_repetition": mean("self_repetition"),
            "provenance_ratio": mean("provenance_ratio"),
            "writer_total_s": mean("writer_total_s"),
            "shipped_stretch_mean": mean("shipped_stretch_mean"),
            "shipped_stretch_max": max((r["shipped_stretch_max"] for r in ok),
                                       default=None),
            "planned_total_s": mean("planned_total_s"),
            "wall_s": mean("wall_s"),
            "wall_runs": spread("wall_s"),
        }
        row["filmcraft"] = filmcraft_score(
            row["story_breaks"], row["story_warnings"], row["jump_cuts"],
            single_scene_rate, row["rule_breaks"], row["self_repetition"],
            row["provenance_ratio"]) if ok else 0.0
        rows.append(row)

    rows.sort(key=lambda r: (r["failed"] + r["parse_failures"],
                             -r["filmcraft"]))
    return rows


def _table(rows: List[Dict]) -> str:
    heads = ["model", "runs", "fail", "craft", "breaks", "warn", "rules",
             "jump", "shots", "scenes", "1scn", "self-rep", "prov", "ship-s",
             "stretch", "wall"]
    lines = ["  " + "".join(f"{h:>11}" for h in heads),
             "  " + "-" * (11 * len(heads))]
    for r in rows:
        vals = [r["model"], f"{r['ok']}/{r['ok'] + r['failed']}", r["failed"],
                r["filmcraft"], r["story_breaks"], r["story_warnings"],
                r["rule_breaks"], r["jump_cuts"], f"{r['shots']}",
                f"{r['scenes']}", r["single_scene_rate"],
                f"{r['self_repetition']}", f"{r['provenance_ratio']}",
                f"{r['writer_total_s']}", f"{r['shipped_stretch_mean']}",
                f"{r['wall_s']}"]
        lines.append("  " + "".join(f"{str(v):>11}" for v in vals))
    return "\n".join(lines)


def _markdown(rows, stress, cfg) -> str:
    out = [f"# Showrunner comparison — {cfg['stamp']}", "",
           f"concept: `{cfg['concept']}`  ",
           f"n_shots: {cfg['n_shots']}  ·  runs/model: {cfg['runs']}  ·  "
           f"temperature: {cfg['temperature']}  ·  shared idea pool: "
           f"{cfg['ideas']} ideas from {len(cfg['farms'])} farms", "",
           "| model | craft | runs | fail | story breaks (runs) | warnings | "
           "rule breaks | jump cuts | shots | scenes | 1-scene rate | self-rep | "
           "provenance | writer total s | stretch mean/max | wall s (runs) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['model']} | **{r['filmcraft']}** | "
            f"{r['ok']}/{r['ok'] + r['failed']} | {r['failed']} | "
            f"{r['story_breaks']} ({r['story_breaks_runs']}) | "
            f"{r['story_warnings']} | {r['rule_breaks']} | {r['jump_cuts']} | "
            f"{r['shots']} ({r['shots_runs']}) | {r['scenes']} | "
            f"{r['single_scene_rate']} | {r['self_repetition']} | "
            f"{r['provenance_ratio']} | {r['writer_total_s']} | "
            f"{r['shipped_stretch_mean']}/{r['shipped_stretch_max']} | "
            f"{r['wall_s']} ({r['wall_runs']}) |")
    if stress:
        out += ["", "## JSON stress", "",
                "| model | requested | ok | shots | parse mode | broke | chars | "
                "tok | failure |", "|---|---|---|---|---|---|---|---|---|"]
        for s in stress:
            out.append(
                f"| {s['model']} | {s.get('requested')} | {s.get('ok')} | "
                f"{s.get('shots', '-')} | {s.get('parse_mode', '-')} | "
                f"{s.get('rule_breaks', '-')} | {s.get('chars', '-')} | "
                f"{s.get('eval_count', '-')} | {s.get('failure', '')[:160]} |")
    out += ["", "## How to read this", "",
            "* **craft** — 0-100, a JUDGEMENT with explicit weights (see "
            "`FILMCRAFT_PENALTIES`): a story break costs 20, a jump cut 5, a "
            "one-scene film 10, and so on. It exists because the first version "
            "of this script ranked on warning order and picked a model whose "
            "lists were three adjacent medium shots.",
            "* **story breaks** — `story:` problems (no turn, beats backwards, "
            "ends unresolved). Mean over runs, with the per-run list: a model "
            "that is 0,0,0 is reliable; 0,2,1 is a lottery.",
            "* **jump cuts** — adjacent shots at the same framing scale. The "
            "writer was told never to do this; R1 repairs it in the planner, but "
            "in the shipped store path the writer's scale IS the cut.",
            "* **1-scene rate** — fraction of runs that used one scene. One "
            "scene disables the per-scene palette and checkpoint stages.",
            "* **self-rep** — mean pairwise Jaccard between a run's own prompts. "
            "Higher = the film restates one image (a mood board).",
            "* **prov** — fraction of final prompts that trace back to a scout "
            "idea. Low means the showrunner ignored the pool and wrote from "
            "scratch; the farms were then decoration.",
            "* **stretch** — delivered hold ÷ generated motion ("
            f"{(2 * cfg['frames'] - 1)} frames at {cfg['fps']}fps = "
            f"{(2 * cfg['frames'] - 1) / cfg['fps']:.2f}s of real motion). "
            "Above ~1.3 the renderer duplicates frames; see "
            "`PACING_AND_FLOW_RESEARCH.md`.",
            "* **wall** — seconds per attempt. Text time is real but small "
            "against a ~10 min/shot render; do not over-weight it.",
            "", "Read the shot lists in `results.json` and the raw replies in "
            "`replies/`. The table can be won by a model whose films you would "
            "not shoot; that is what the eye is for."]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def compare(concept: str = story.CONCEPT_DEFAULT,
            models: Optional[Sequence[str]] = None,
            farms: Optional[Sequence[str]] = None,
            n_shots: Optional[int] = 6, runs: int = 1,
            temperature: float = 0.8, story_retries: int = 0,
            frames: int = RAW_FRAMES, fps: int = FPS,
            verbose: bool = True) -> Dict:
    """Run the fair comparison. Returns the serialisable report dict."""
    models = list(models) if models else list(story.SHOWRUNNER_CANDIDATES[:3])
    farms = list(farms) if farms else list(story.FARM_MODELS)
    stamp = _stamp()
    out_dir = OUT / stamp
    replies_dir = out_dir / "replies"
    stress_dir = out_dir / "stress"
    for d in (out_dir, replies_dir, stress_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 78}\n# SHOWRUNNER COMPARISON  {stamp}\n{'#' * 78}", flush=True)
    print(f"concept: {concept!r}", flush=True)
    print(f"models: {', '.join(models)}", flush=True)
    print(f"farms: {', '.join(farms)}  (one pool, shared by every model)", flush=True)
    print(f"runs/model: {runs}  n_shots: {n_shots}  temp: {temperature}  "
          f"story_retries: {story_retries}", flush=True)

    print(f"\n=== collecting the idea pool (once) ===", flush=True)
    pool = collect_ideas(concept, farms, verbose=verbose)
    (out_dir / "ideas.json").write_text(json.dumps(pool, indent=2), encoding="utf-8")
    total_ideas = sum(len(f.get("ideas") or []) for f in pool)
    print(f"  pool: {total_ideas} ideas from {len(pool)} farms -> {out_dir.name}",
          flush=True)

    results: List[Dict] = []
    for model in models:
        for run in range(1, runs + 1):
            results.append(run_model(model, pool, concept, n_shots, temperature,
                                     story_retries, frames, fps, run,
                                     replies_dir, verbose))
    rows = summarise(results)
    report = {
        "stamp": stamp, "concept": concept, "models": models, "farms": farms,
        "runs": runs, "n_shots": n_shots, "temperature": temperature,
        "frames": frames, "fps": fps,
        "idea_pool": {"total": total_ideas,
                      "per_farm": {f["model"]: len(f.get("ideas") or [])
                                   for f in pool}},
        "results": results, "summary": rows,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2),
                                          encoding="utf-8")
    print(f"\n{'=' * 78}\n  SUMMARY\n{'=' * 78}", flush=True)
    print(_table(rows), flush=True)
    best = rows[0] if rows else None
    if best:
        print(f"\n  best filmcraft score: {best['model']}  "
              f"({best['filmcraft']}/100 -- a judgement; read the shot lists)",
              flush=True)
    print(f"  full results: {(out_dir / 'results.json')}", flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare ensemble showrunner models")
    ap.add_argument("--concept", default=story.CONCEPT_DEFAULT)
    ap.add_argument("--models", default=None,
                    help="comma list. Default: " + ",".join(DEFAULT_MODELS))
    ap.add_argument("--farms", default=",".join(story.FARM_MODELS))
    ap.add_argument("--runs", type=int, default=1,
                    help="samples per model. n=1 cannot separate a model from "
                         "its own variance (HANDOFF §11.3)")
    ap.add_argument("--n-shots", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--story-retries", type=int, default=0,
                    help="0 = one attempt per run, i.e. the raw model. The "
                         "pipeline re-rolls; that is a separate question")
    ap.add_argument("--stress-shots", type=int, default=0,
                    help="after the main run, ask each model for this many shots "
                         "and record exactly how JSON fails (0 = off)")
    ap.add_argument("--stress-models", default=None,
                    help="comma list for the stress test. Default: the first "
                         "two models (the ones the question is about)")
    ap.add_argument("--stress-budgets", default="1200,3000",
                    help="num_predict budgets to stress. 1200 is the old value "
                         "that was reported as malformed JSON; 3000 is the fix")
    ap.add_argument("--frames", type=int, default=RAW_FRAMES,
                    help="raw frames per shot, for the stretch report")
    args = ap.parse_args()
    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              if args.models else list(DEFAULT_MODELS))
    farms = [m.strip() for m in args.farms.split(",") if m.strip()]

    report = compare(concept=args.concept, models=models, farms=farms,
                     n_shots=args.n_shots, runs=args.runs,
                     temperature=args.temperature,
                     story_retries=args.story_retries, frames=args.frames)

    if args.stress_shots:
        out_dir = OUT / report["stamp"]
        pool = json.loads((out_dir / "ideas.json").read_text(encoding="utf-8"))
        stress_models = ([m.strip() for m in args.stress_models.split(",")
                          if m.strip()] if args.stress_models else models[:2])
        budgets = [int(b) for b in args.stress_budgets.split(",") if b.strip()]
        stress = []
        for m in stress_models:
            for budget in budgets:
                stress.append(stress_json(
                    m, pool, args.concept, args.stress_shots,
                    args.temperature, out_dir / "stress", verbose=True,
                    num_predict=budget))
        report["stress"] = stress
        (out_dir / "results.json").write_text(json.dumps(report, indent=2),
                                              encoding="utf-8")
        (out_dir / "summary.md").write_text(_markdown(
            report["summary"], stress,
            {"stamp": report["stamp"], "concept": args.concept,
             "n_shots": args.n_shots, "runs": args.runs,
             "temperature": args.temperature, "frames": args.frames,
             "fps": FPS, "ideas": report["idea_pool"]["total"],
             "farms": farms}), encoding="utf-8")
        print(f"\n  summary: {out_dir / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
