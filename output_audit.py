#!/usr/bin/env python3
"""
output_audit.py -- audit a DELIVERED film against the writing that made it.

WHY THIS EXISTS. The first rendered film (`ego`, 13 takes, 24.38 s) passed the
story gate, the structure checks and the render. Only after it existed did anyone
measure whether the pictures answered the prompts:

  * 12 of 13 prompts were two-image composites ("...; the shadow of his ego looms
    large..."), so the renderer drew one clause or the concept prior;
  * prompt->frame CLIP had a mean margin of only +0.018 over a DIFFERENT shot's
    prompt, and four frames matched another shot's prompt better than their own;
  * the film ended on a blue car because the resolve prompt named an "old car" --
    and that shot's alignment was *good*, because the prompt itself was wrong.

That evidence came from a one-off script tied to that production. This is the same
measurement as a reusable tool: any store, any render, and the same numbers
`pacing_audit.py` and `sequence_audit.py` already report -- on the finished
pixels. Read-only; it writes only into output/audit/<slug>/.

    python output_audit.py --store ego
    python output_audit.py --store ego --render ego_render
    python output_audit.py --store ego --no-clip        # fast: hist/luma/pacing
    python output_audit.py --store ego --stills         # takes only, no render

Writes: metrics.json, report.txt, contact.png (one frame per shot with its prompt),
flow_strip.png (the cut left to right). Exit 1 if a `warn` flag is raised, else 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

import continuity                                # noqa: E402
import film_store as fs                          # noqa: E402
import story                                     # noqa: E402

WEAK_MARGIN = 0.02          # mean prompt->frame margin over the whole film
GOOD_CUT_CLIP = continuity.HOLD_CLIP_MIN   # below this an adjacent pair is a hard change
LUMA_SHOCK = continuity.HOLD_LUMA_MAX      # exposure jump that reads as a flash
METRONOMIC_CV = 0.20        # PACING_MEASURED_STUDY's cut for "no rhythm"


# --------------------------------------------------------------------------- #
# pure helpers (tested without CLIP, video or a store)
# --------------------------------------------------------------------------- #

def hold_cv(holds: Sequence[float]) -> float:
    """Coefficient of variation of the holds. Human short-form sits ~0.75."""
    if not holds:
        return 0.0
    mean = float(np.mean(holds))
    return float(np.std(holds) / mean) if mean else 0.0


def beat_bar(beats: Sequence[str]) -> str:
    return "".join(story.BEAT_MARKS.get(b, "?") for b in beats)


def cut_by_mode(cuts: Dict[str, Dict], pair_mode: Dict[str, str]) -> Dict:
    """Mean continuity per movement grammar: did the pixels deliver the grammar?

    A montage run should cut bigger than a hold. If it does not, the direction
    chose a grammar the material does not support.
    """
    grouped: Dict[str, List[Dict]] = {}
    for pair, row in cuts.items():
        grouped.setdefault(pair_mode.get(pair, "?"), []).append(row)
    out = {}
    for mode, rows in grouped.items():
        clips = [r["clip"] for r in rows if r.get("clip") is not None]
        out[mode] = {
            "cuts": len(rows),
            "mean_clip": round(float(np.mean(clips)), 3) if clips else None,
            "mean_hist": round(float(np.mean([r["hist"] for r in rows])), 3),
            "mean_luma": round(float(np.mean([r["luma"] for r in rows])), 2),
        }
    return out


def new_words(before: Sequence[str], after: str) -> List[str]:
    """Content words in the RESOLVE that never appear earlier in the film.

    A resolve may introduce a new world. The first rendered film's resolve
    introduced "car" -- a concrete object the film never had -- and the renderer
    drew the car instead of the man. This names it without pretending to know
    whether the new world was intended.
    """
    seen = set()
    for p in before:
        seen |= story.content_words(p)
    return sorted(w for w in story.content_words(after) if w not in seen)


def flags_from(metrics: Dict) -> List[Dict]:
    """The audit's verdicts, in one place so the report and the exit code agree.

    Like previs, the audit must not average two different kinds of cut together:
    a jump INSIDE a hold is a defect; a jump at a SCENE CUT is the edit -- and with
    deliberate per-scene palettes the colour difference there is style, not an
    exposure error (the render normalises the technical baseline before palettes).
    Anchored prompts also make the prompt->frame margin structurally ambiguous,
    because every shot in a scene shares the front anchor by design.
    """
    flags: List[Dict] = []
    anchored = bool(metrics.get("anchors"))
    al = metrics.get("alignment") or {}
    if al.get("mean_margin") is not None and al["mean_margin"] < WEAK_MARGIN:
        if anchored:
            flags.append({"level": "info", "code": "anchored-prompts",
                          "message": "prompt->frame margin is "
                                     f"{al['mean_margin']:+.3f}; prompts share "
                                     "their scene's front anchor by design, so "
                                     "this margin is informational, not a defect"})
        else:
            flags.append({"level": "warn", "code": "weak-alignment",
                          "message": "prompt->frame margin is only "
                                     f"{al['mean_margin']:+.3f} over a different "
                                     "shot's prompt: the pictures are generic to the "
                                     "film, not specific to each shot"})
    if not anchored:
        for shot, m in (al.get("margin_by_shot") or {}).items():
            if m is not None and m < 0:
                flags.append({"level": "warn", "code": "shot-mismatch",
                              "message": f"{shot}: its frame matches ANOTHER "
                                         f"shot's prompt better than its own "
                                         f"(margin {m:+.3f})"})
    for pair, row in (metrics.get("cut") or {}).get("per_cut", {}).items():
        if row.get("clip") is not None and row["clip"] < GOOD_CUT_CLIP:
            flags.append({"level": "info", "code": "hard-cut",
                          "message": f"cut {pair}: CLIP {row['clip']:.2f} -- a "
                                     "large change (legitimate in a montage run)"})
        if row.get("luma") is not None and row["luma"] > LUMA_SHOCK:
            if row.get("same_scene", True):
                flags.append({"level": "warn", "code": "exposure-jump",
                              "message": f"cut {pair}: luma jump "
                                         f"{row['luma']:.1f} inside a scene -- "
                                         "reads as a flash across the cut"})
            else:
                flags.append({"level": "info", "code": "style-cut",
                              "message": f"cut {pair}: luma change "
                                         f"{row['luma']:.1f} at a scene cut -- "
                                         "the palette/light change is the edit, "
                                         "not an exposure error"})
    cv = (metrics.get("flow") or {}).get("hold_cv") or 0.0
    if metrics.get("flow", {}).get("holds_s") and cv < METRONOMIC_CV:
        flags.append({"level": "warn", "code": "metronomic",
                      "message": f"shot-length CV {cv:.3f} -- the holds are all "
                                 "one length; human short-form mixes accents "
                                 "with holds (CV~0.75)"})
    resolve = metrics.get("resolve") or {}
    if resolve.get("new_words"):
        flags.append({"level": "info", "code": "resolve-new-imagery",
                      "message": "the resolve introduces imagery no earlier shot "
                                 "has: " + ", ".join(resolve["new_words"])})
    writing = metrics.get("writing") or {}
    if writing.get("flagged_prompts"):
        flags.append({"level": "warn", "code": "prompt-shape",
                      "message": f"{len(writing['flagged_prompts'])} prompt(s) "
                                 "describe more than one image or interpret the "
                                 "shot: " + ", ".join(writing["flagged_prompts"])})
    coverage = writing.get("coverage") or []
    if coverage:
        flags.append({"level": "warn", "code": "no-coverage",
                      "message": f"{len(coverage)} coverage problem(s): "
                                 + "; ".join(coverage[:3])})
    return flags


# --------------------------------------------------------------------------- #
# reading the store and the render
# --------------------------------------------------------------------------- #

def latest_render(project: Dict) -> Tuple[Optional[str], Optional[Path]]:
    """The render to audit: the project's deliverable, else the newest on disk."""
    base = fs.renders_dir(project["slug"])
    deliverable = (project.get("deliverable") or {}).get("render")
    if deliverable and (base / deliverable / "film.mp4").exists():
        return deliverable, base / deliverable
    if base.exists():
        dirs = [d for d in base.iterdir()
                if d.is_dir() and (d / "film.mp4").exists()]
        if dirs:
            d = max(dirs, key=lambda p: (p / "film.mp4").stat().st_mtime)
            return d.name, d
    return None, None


def shot_frame_indices(stats: Dict) -> Dict[Tuple[int, str], int]:
    """The frame numbers the audit needs from the finished film."""
    fps = float(stats.get("fps") or 16)
    total = int(stats.get("total_frames") or 0)
    cursor, want = 0.0, {}
    for i, st in enumerate(stats.get("shots") or []):
        start = int(round(cursor * fps))
        hold = int(st.get("hold_frames") or 0)
        want[(i, "mid")] = min(start + hold // 2, max(total - 1, 0))
        if i > 0:
            want[(i, "before")] = max(start - 1, 0)
            want[(i, "after")] = min(start, max(total - 1, 0))
        cursor += hold / fps
    return want


def read_render_frames(mp4: Path, want: Dict[Tuple[int, str], int]) -> Dict:
    """Stream the video once, keeping only the requested frames.

    A 390-frame 512² film is ~300 MB as a list; the audit needs ~40 frames.
    """
    wanted = sorted(set(want.values()))
    picked: Dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(mp4))
    idx = -1
    k = 0
    while k < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx == wanted[k]:
            picked[idx] = frame
            k += 1
    cap.release()
    return {key: picked[idx] for key, idx in want.items() if idx in picked}


def take_mid_images(project: Dict) -> List[Image.Image]:
    out = []
    for sid in project["timeline"]:
        # the take the film CUTS TO (chosen, else newest) -- not just the newest: an
        # older chosen take is what was delivered
        take = fs.best_take(project, sid)
        files = (sorted((Path(take["_dir"]) / "frames").glob("f*.png"))
                 if take else [])
        if not files:
            raise SystemExit(f"shot {sid} has no take with frames; render it before "
                             f"auditing stills (or audit the delivered film)")
        out.append(Image.open(files[len(files) // 2]).convert("RGB"))
    return out


# --------------------------------------------------------------------------- #
# the CLIP half (optional: --no-clip skips the model load)
# --------------------------------------------------------------------------- #

def clip_metrics(mids: Sequence[Image.Image], prompts: Sequence[str],
                 cut_pairs: Sequence[Tuple[Image.Image, Image.Image]]) -> Dict:
    """One CLIP load: prompt<->frame alignment, margins, and cut similarity."""
    import torch
    from motion_metrics import ClipScorer

    sc = ClipScorer(str(ROOT / "models" / "clip-vit-base-patch32"))
    T = torch.stack([sc.embed_text(p) for p in prompts])
    I = sc.embed_images(list(mids))
    sim = (I @ T.T).cpu()
    n = len(mids)
    align = [round(float(sim[i, i]), 4) for i in range(n)]
    other_max, best_other = {}, {}
    for i in range(n):
        j = max((j for j in range(n) if j != i),
                key=lambda j: float(sim[i, j]))
        best_other[i] = j
        other_max[i] = float(sim[i, j])
    margins = {i: round(align[i] - other_max[i], 3) for i in range(n)}
    img_sim = (I @ I.T).cpu()
    cut_clip = []
    if cut_pairs:
        flat = [x for pair in cut_pairs for x in pair]
        E = sc.embed_images(flat)
        for k in range(len(cut_pairs)):
            e0, e1 = E[2 * k], E[2 * k + 1]
            cut_clip.append(round(float((e0 @ e1).item()), 3))
    return {
        "align": align,
        "mean_align": round(float(np.mean(align)), 4),
        "mean_off_diagonal": round(float(np.mean(
            [float(sim[i, j]) for i in range(n) for j in range(n) if i != j])), 4),
        "margins": margins,
        "mean_margin": round(float(np.mean(list(margins.values()))), 4),
        "best_other": best_other,
        "image_redundancy": {
            i: {"looks_most_like": best_other[i],
                "clip": round(float(img_sim[i, best_other[i]]), 3)}
            for i in range(n)},
        "cut_clip": cut_clip,
    }


# --------------------------------------------------------------------------- #
# the audit
# --------------------------------------------------------------------------- #

def audit(store: str, render: Optional[str] = None, with_clip: bool = True,
          stills_only: bool = False) -> Dict:
    project = fs.load(store)
    shots = [fs.load_shot(store, sid) for sid in project["timeline"]]
    names = list(project["timeline"])
    prompts = [s.get("prompt", "") for s in shots]
    beats = [s.get("beat", "") for s in shots]

    render_name, render_dir = (None, None) if stills_only else latest_render(project)
    if render and not stills_only:
        render_name = render
        render_dir = fs.renders_dir(store) / render
        if not (render_dir / "film.mp4").exists():
            raise SystemExit(f"no render {render!r} for {store!r}")
    if render_dir is not None:
        stats = json.loads((render_dir / "stats.json").read_text(encoding="utf-8"))
    else:
        stats = {}
    frames = {} if stills_only else read_render_frames(
        render_dir / "film.mp4", shot_frame_indices(stats)) if render_dir else {}

    from story import prompt_problems, shots_from_dicts
    # ANCHORED PROMPTS ARE LONGER ON PURPOSE. When the writing path bound the
    # planned place/figure anchors, the writer's prompt shape was measured BEFORE
    # the append; re-reporting the anchors as writer faults would be measuring the
    # pipeline, not the writer.
    validation_shots = shots_from_dicts([
        {k: s.get(k) for k in ("prompt", "scale", "camera", "seconds", "subject",
                               "scene", "energy", "beat", "purpose", "link")}
        for s in shots])
    shape = ([] if project.get("anchors")
             else prompt_problems(validation_shots))
    flagged = [p.split()[2] for p in shape]          # "prompt: shot N ..."
    writing = {"flagged_prompts": flagged, "problems": shape,
               "prompt_words": [len(p.split()) for p in prompts],
               "coverage": story.coverage_problems(shots_from_dicts([
                   {k: s.get(k) for k in ("prompt", "scale", "camera", "seconds",
                                          "subject", "scene", "energy", "beat",
                                          "purpose", "link")}
                   for s in shots]))}

    holds = ([int(st["hold_frames"]) / float(stats.get("fps") or 16)
              for st in stats.get("shots") or []]
             or [float(s.get("seconds") or 2.5) for s in shots])
    pair_mode: Dict[str, str] = {}
    direction = (project.get("direction") or {}).get("movements", [])
    for m in direction:
        ids = [project["timeline"][n - 1] for n in m["shots"]]
        for a, b in zip(ids, ids[1:]):
            pair_mode[f"{a}->{b}"] = m["mode"]

    import continuity
    cuts: Dict[str, Dict] = {}
    for i in range(1, len(shots)):
        before = frames.get((i, "before"))
        after = frames.get((i, "after"))
        if before is None or after is None:
            continue
        a = Image.fromarray(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
        b = Image.fromarray(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
        cuts[f"{names[i-1]}->{names[i]}"] = {
            "clip": None, "hist": round(continuity.hist_distance(a, b), 3),
            "luma": round(continuity.luma_jump(a, b), 2),
            "same_scene": shots[i].get("scene") == shots[i - 1].get("scene")}

    alignment: Dict = {}
    mids: List[Image.Image] = []
    if with_clip:
        try:
            if frames:
                mids = [Image.fromarray(cv2.cvtColor(frames[(i, "mid")],
                                                     cv2.COLOR_BGR2RGB))
                        for i in range(len(shots)) if (i, "mid") in frames]
            else:
                mids = take_mid_images(project)      # --stills, or no render yet
        except (IndexError, FileNotFoundError, OSError):
            mids = []
    if len(mids) == len(shots) and mids:
        pairs, keys = [], []
        for i in range(1, len(shots)):
            if frames and (i, "before") in frames and (i, "after") in frames:
                pairs.append((
                    Image.fromarray(cv2.cvtColor(frames[(i, "before")],
                                                 cv2.COLOR_BGR2RGB)),
                    Image.fromarray(cv2.cvtColor(frames[(i, "after")],
                                                 cv2.COLOR_BGR2RGB))))
            else:
                pairs.append((mids[i - 1], mids[i]))
            keys.append(f"{names[i-1]}->{names[i]}")
        m = clip_metrics(mids, prompts, pairs)
        for key, c in zip(keys, m["cut_clip"]):
            cuts.setdefault(key, {"clip": None, "hist": None, "luma": None})
            cuts[key]["clip"] = c
        alignment = {
            "clip_prompt_to_frame": dict(zip(names, m["align"])),
            "mean_align": m["mean_align"],
            "mean_off_diagonal": m["mean_off_diagonal"],
            "mean_margin": m["mean_margin"],
            "margin_by_shot": {names[i]: m["margins"][i]
                               for i in range(len(shots))},
            "best_other": {names[i]: names[m["best_other"][i]]
                           for i in range(len(shots))},
            "image_redundancy": {
                names[i]: {"looks_most_like":
                           names[m["image_redundancy"][i]["looks_most_like"]],
                           "clip": m["image_redundancy"][i]["clip"]}
                for i in range(len(shots))},
        }

    turn_i = next((i for i, b in enumerate(beats) if b == "turn"), None)
    resolve = ({"turn_shot": names[turn_i], "resolve_shot": names[-1],
                "new_words": new_words(prompts[:-1], prompts[-1])}
               if turn_i is not None else {})

    metrics = {
        "store": store, "render": render_name,
        "render_dir": str(render_dir) if render_dir else None,
        "anchors": bool(project.get("anchors")),
        "shots": [{"id": s["id"], "beat": s.get("beat"),
                   "prompt": s.get("prompt")} for s in shots],
        "writing": writing,
        "alignment": alignment,
        "cut": {"beat_shape": beat_bar(beats), "per_cut": cuts,
                "by_movement_mode": cut_by_mode(cuts, pair_mode)},
        "flow": {"holds_s": dict(zip(names, [round(h, 3) for h in holds])),
                 "hold_cv": round(hold_cv(holds), 3),
                 "total_s": round(sum(holds), 2),
                 "direction": [(m["shots"], m["mode"]) for m in direction]},
        "resolve": resolve,
    }
    metrics["flags"] = flags_from(metrics)
    return metrics


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #

def _grid(frames: Dict, names: Sequence[str], beats: Sequence[str],
          prompts: Sequence[str], out: Path, cols: int = 4,
          width: int = 400) -> Path:
    imgs = [Image.fromarray(cv2.cvtColor(frames[(i, "mid")], cv2.COLOR_BGR2RGB))
            for i in range(len(names))]
    w = width
    h = int(w * imgs[0].size[1] / imgs[0].size[0])
    pad, bar = 8, 50
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (w + pad) + pad,
                              rows * (h + bar + pad) + pad), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        x, y = pad + c * (w + pad), pad + r * (h + bar + pad)
        sheet.paste(im.resize((w, h), Image.LANCZOS), (x, y))
        d.rectangle([x, y, x + w, y + h], outline=(90, 90, 95))
        d.text((x + 4, y + h + 4), f"{names[i]} {beats[i]}: {prompts[i][:48]}",
               fill=(235, 235, 235))
        d.text((x + 4, y + h + 18), " " * 8 + prompts[i][48:96],
               fill=(180, 180, 185))
    sheet.save(out)
    return out


def _strip(frames: Dict, names: Sequence[str], beats: Sequence[str],
           holds: Dict[str, float], out: Path, width: int = 176) -> Path:
    imgs = [Image.fromarray(cv2.cvtColor(frames[(i, "mid")], cv2.COLOR_BGR2RGB))
            for i in range(len(names))]
    w = width
    h = int(w * imgs[0].size[1] / imgs[0].size[0])
    pad = 6
    strip = Image.new("RGB", (len(imgs) * (w + pad) + pad, h + 30 + pad),
                      (18, 18, 20))
    d = ImageDraw.Draw(strip)
    for i, im in enumerate(imgs):
        x = pad + i * (w + pad)
        strip.paste(im.resize((w, h), Image.LANCZOS), (x, pad))
        d.rectangle([x, pad, x + w, pad + h], outline=(90, 90, 95))
        d.text((x + 3, pad + h + 3),
               f"{names[i]} {beats[i][:4]} {holds[names[i]]:.1f}s",
               fill=(235, 235, 235))
    strip.save(out)
    return out


def report_text(metrics: Dict) -> str:
    lines = [f"output audit: {metrics['store']}  render={metrics['render']}",
             f"beat shape: {metrics['cut']['beat_shape']}"]
    al = metrics.get("alignment") or {}
    if al:
        lines.append(f"prompt->frame: mean {al['mean_align']:.3f}, margin "
                     f"{al['mean_margin']:+.3f} over a different shot's prompt")
    lines.append(f"shot-length CV: {metrics['flow']['hold_cv']:.3f}  "
                 f"({metrics['flow']['total_s']:.2f}s total)")
    for f in metrics["flags"]:
        lines.append(f"  [{f['level']}] {f['code']}: {f['message']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="audit a delivered film against its prompts")
    ap.add_argument("--store", required=True)
    ap.add_argument("--render", default=None,
                    help="render name under productions/<store>/renders/ "
                         "(default: the project's deliverable)")
    ap.add_argument("--out", default=None,
                    help="output dir (default output/audit/<store>)")
    ap.add_argument("--no-clip", action="store_true",
                    help="skip CLIP (fast: histogram/luma/pacing only)")
    ap.add_argument("--stills", action="store_true",
                    help="audit the takes only, no assembled render")
    ap.add_argument("--report-only", action="store_true",
                    help="exit 0 even when a warn flag is raised (the studio's job "
                         "runner treats a non-zero exit as a failed job, and a "
                         "warning about the film is a finding, not a failure)")
    a = ap.parse_args()

    out = Path(a.out) if a.out else (ROOT / "output" / "audit" / a.store)
    out.mkdir(parents=True, exist_ok=True)
    metrics = audit(a.store, render=a.render, with_clip=not a.no_clip,
                    stills_only=a.stills)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                      encoding="utf-8")
    text = report_text(metrics)
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)

    render_dir = Path(metrics["render_dir"]) if metrics.get("render_dir") else None
    if render_dir is not None:
        stats = json.loads((render_dir / "stats.json").read_text(encoding="utf-8"))
        frames = read_render_frames(render_dir / "film.mp4", shot_frame_indices(stats))
        if frames:
            names = [s["id"] for s in metrics["shots"]]
            beats = [s["beat"] for s in metrics["shots"]]
            prompts = [s["prompt"] for s in metrics["shots"]]
            _grid(frames, names, beats, prompts, out / "contact.png")
            _strip(frames, names, beats, metrics["flow"]["holds_s"],
                   out / "flow_strip.png")
            print(f"artifacts: {out}/contact.png, {out}/flow_strip.png, "
                  f"{out}/metrics.json")
    if a.report_only:
        return 0
    return 1 if any(f["level"] == "warn" for f in metrics["flags"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
