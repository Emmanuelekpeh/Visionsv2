#!/usr/bin/env python3
"""
previs.py -- one still per shot, before the 45-90 minute render.

WHY. `SEQUENCE_AB.md` showed that the coverage cut reads better than the montage,
and that the difference is visible at the CUTS: a new place vs the same bridge.
Those are exactly the properties a single still per shot can be measured on --
CLIP similarity and colour/luma continuity across the boundary frames. So a
previs pass is not a guess: it is the same measurement the finished cut gets,
made before the expensive part.

What it does, per candidate shot list:
  * renders ONE keyframe per shot with the local SD15 checkpoint (4-step Lightning),
    seeded consistently, so two candidate arrangements are compared on equal terms;
  * measures cross-cut continuity on the stills (CLIP, colour-histogram jump,
    luma jump);
  * reports structure and pacing (scenes, cuts, ASL, shot-length CV);
  * writes a contact sheet + report, and logs the wall time and the extrapolated
    full-render cost saved.

It does NOT pretend to replace watching: motion, performance and the cut's
rhythm are not in a still. It is a preflight that kills the arrangements whose
images do not even hold together, before a render hour is spent.

    python previs.py --store _seq_coverage --name coverage
    python previs.py --store _seq_current  --name montage
    python previs.py --proposal output/film/proposals/<name>.json

Cost on this machine (CPU, 4-step 512²): ~20-40 s per still, plus ~60-90 s of
pipeline load -- about 3-4 minutes for 4 shots, against ~45-60 minutes to render
them as motion. The numbers are recorded in the report.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import conditioning  # noqa: E402
import continuity  # noqa: E402
from edit_planner import Shot, plan  # noqa: E402
from visual_org import as_shots  # noqa: E402

OUT = ROOT / "output" / "previs"
NEG = ("blurry, low quality, deformed, watermark, text, extra limbs, "
       "haze, glow, bloom, soft focus")
SIZE = 512
STEPS = 4
SEED = 7
# A previs must show the SUBJECT the film is about, not just a well-graded place.
# CLIP text-image similarity to the expected subject phrase; measured on the failed
# squid cut the stills were glowing helmet-creatures, so a legibility floor stops a
# render before the 2-hour spend. Calibrate per checkpoint; 0.20 is deliberately low.
SUBJECT_MIN = 0.20


# --------------------------------------------------------------------------- #
# Rendering one still per shot
# --------------------------------------------------------------------------- #

def _pipeline():
    import torch
    from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
    from safetensors.torch import load_file
    pipe = StableDiffusionPipeline.from_pretrained(
        str(ROOT / "models" / "dreamshaper8"), torch_dtype=torch.float32)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)
    # NO attention slicing here: in diffusers 0.32 it installs
    # SlicedAttnProcessor, and IP-Adapter's processor swap then raises
    # `SlicedAttnProcessor.__init__() missing 1 required positional argument:
    # 'slice_size'`. Measured on the first conditioned previs run. VAE slicing is
    # safe and enough for memory at 512.
    pipe.enable_vae_slicing()
    sd = load_file(str(ROOT / "models" / "animatediff_lightning" /
                       "animatediff_lightning_4step_diffusers.safetensors"))
    pipe.unet.load_state_dict(sd, strict=False)
    return pipe, torch


def render_stills(shots: Sequence[Shot], outdir: Path,
                  condition: bool = True, verbose: bool = True,
                  palettes: Optional[Sequence[str]] = None,
                  baseline: bool = True) -> Dict:
    """One keyframe per shot, through the RENDER's own look pipeline.

    Two measured lessons are baked in:

    1. The previs must mirror the render POLICY, not just the prompts: shots in a
       hold are IP-Adapter conditioned on the run's anchor still, exactly as
       `make_film` conditions them on the anchor take.
    2. It must also mirror the render's LOOK stage: shared technical baseline
       (exposure/contrast/grain normalised across every shot) FIRST, then the
       per-scene palette. Without the baseline, a previs reports every palette
       change as a technical jump and invites the wrong fix (weaker palettes).
       Baseline is what makes a bold palette safe: the cut cannot read as an
       exposure error, so the palette difference stays stylistic.

    `palettes` is the per-shot grade; `baseline` can be turned off to measure the
    raw model output (`--no-baseline`).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    pipe, torch = _pipeline()
    refs: Dict[int, Path] = {}
    if condition:
        try:
            from visual_org import reference_plan
            refs = reference_plan(as_shots(shots))
        except Exception:                            # noqa: BLE001
            refs = {}
    ip_ready = False
    raws, times, conditioned = [], [], []
    for i, s in enumerate(shots):
        anchor = refs.get(i)
        t0 = time.time()
        kwargs = dict(prompt=s.prompt, negative_prompt=NEG,
                      num_inference_steps=STEPS, guidance_scale=1.0,
                      width=SIZE, height=SIZE,
                      generator=torch.Generator("cpu").manual_seed(SEED + i))
        used = False
        if anchor is not None:
            if not ip_ready:
                ip_ready = _load_ip(pipe)
            if ip_ready:
                # condition on the anchor's RAW still: the render conditions on
                # the anchor take before any grading is applied
                kwargs["ip_adapter_image"] = raws[anchor]
                used = True
        elif ip_ready:
            # same diffusers quirk as the renderer: loaded + no image crashes
            conditioning.unload(pipe)
            ip_ready = False
        with torch.no_grad():
            im = pipe(**kwargs).images[0]
        raws.append(im)
        conditioned.append(used)
        times.append(round(time.time() - t0, 1))
        if verbose:
            print(f"  still {i + 1}/{len(shots)} {times[-1]:.0f}s"
                  f"{' (hold-conditioned)' if used else ''}", flush=True)
    if ip_ready:
        conditioning.unload(pipe)

    # ---- the look stage, in the render's order ---------------------------- #
    graded = list(raws)
    baseline_applied = False
    if baseline and len(raws) > 1:
        try:
            import grade_match as gm
            sigs = [gm.signature([im]) for im in raws]
            tgt = gm.technical_target(sigs)
            graded = [gm.match_technical([im], tgt)[0] for im in raws]
            baseline_applied = True
        except Exception as exc:                     # noqa: BLE001
            print(f"  !! previs baseline failed: {type(exc).__name__}: {exc}")
    looks_used = []
    if palettes:
        import looks
        for i, pal in enumerate(palettes[:len(graded)]):
            pal = str(pal or "")
            looks_used.append(pal)
            if pal and pal != "neutral":
                graded[i] = looks.apply_look([graded[i]], pal)[0]
    while len(looks_used) < len(graded):
        looks_used.append("")

    files = []
    for i, im in enumerate(graded):
        p = outdir / f"shot{i + 1:02d}.png"
        im.save(p)
        files.append(p)
        if verbose and looks_used[i] and looks_used[i] != "neutral":
            print(f"    shot{i + 1:02d} graded {looks_used[i]}", flush=True)
    return {"files": files, "seconds": times,
            "total_s": round(sum(times), 1),
            "conditioned": conditioned, "palettes": looks_used,
            "baseline": baseline_applied}


def _load_ip(pipe) -> bool:
    try:
        return conditioning.load(pipe)
    except Exception as exc:                         # noqa: BLE001
        print(f"  !! previs IP-Adapter failed: {type(exc).__name__}: {exc}")
        return False


# --------------------------------------------------------------------------- #
# Measurement: exactly what the finished-cut tool measures, on stills
# --------------------------------------------------------------------------- #

def split_continuity(rows: Sequence[Dict]) -> Dict:
    """Continuity WITHIN a scene vs ACROSS a scene cut.

    THE POINT. A single mean over every cut conflates two different things: a
    jump inside a hold is a defect (the same place/figure must hold), while a jump
    at a scene cut is the edit working -- and once a film carries deliberate
    per-scene palettes, the colour difference at a scene cut is STYLE, not error.
    The render already normalises the technical half of the look (shared exposure/
    contrast baseline) before applying palettes, so a bold palette cannot read as
    an exposure mistake. Measuring the two kinds separately is the fix for
    "stronger palettes -> bigger colour jumps -> just lower the strength".
    """

    def agg(subset: Sequence[Dict], key: str):
        vals = [r[key] for r in subset if r.get(key) is not None]
        return round(statistics.fmean(vals), 3) if vals else None

    within = [r for r in rows if r.get("same_scene")]
    across = [r for r in rows if not r.get("same_scene")]
    return {
        "within_scene": {"cuts": len(within),
                         "luma_jump": agg(within, "luma_jump"),
                         "hist_jump": agg(within, "hist_jump"),
                         "clip_sim": agg(within, "clip_sim")},
        "across_scenes": {"cuts": len(across),
                          "luma_jump": agg(across, "luma_jump"),
                          "hist_jump": agg(across, "hist_jump"),
                          "clip_sim": agg(across, "clip_sim")},
    }


def measure(files: Sequence[Path], shots: Sequence[Shot],
            scorer=None, expect: Optional[Sequence[str]] = None) -> Dict:
    imgs = [Image.open(f).convert("RGB") for f in files]
    rows = []
    clip_sims, hist_jumps, luma_jumps = [], [], []
    for i in range(1, len(imgs)):
        a, b = imgs[i - 1], imgs[i]
        luma = continuity.luma_jump(a, b)
        histd = continuity.hist_distance(a, b)
        sim = continuity.pair_similarity(a, b, scorer)
        if sim is not None:
            clip_sims.append(sim)
        hist_jumps.append(histd)
        luma_jumps.append(luma)
        rows.append({"cut": f"{i}->{i + 1}", "clip_sim": sim,
                     "hist_jump": round(histd, 3),
                     "luma_jump": round(luma, 1),
                     "same_scene": shots[i].scene == shots[i - 1].scene})
    holds = [s.seconds for s in shots]
    mean = sum(holds) / len(holds)
    sd = (sum((h - mean) ** 2 for h in holds) / len(holds)) ** 0.5
    split = split_continuity(rows)

    # DOES EACH STILL ANSWER ITS OWN PROMPT? The rendered `ego` film was made from
    # 12 two-image prompts; the stiller's version of that failure is visible here,
    # before the render: a still whose image matches a DIFFERENT shot's prompt
    # better than its own (negative margin) or a film whose margin is near zero
    # (the prompts are generic to the film, not specific to each shot). Reported,
    # never gated -- CLIP is a proxy, the contact sheet is the arbiter.
    prompt_align, prompt_margin, warnings = [], [], []
    if scorer is not None and len(imgs) > 1:
        try:
            import torch
            T = torch.stack([scorer.embed_text(getattr(s, "prompt", "") or "x")
                             for s in shots])
            I = scorer.embed_images(imgs)
            sim = (I @ T.T).cpu()
            prompt_align = [round(float(sim[i, i]), 3) for i in range(len(imgs))]
            for i in range(len(imgs)):
                other = max(float(sim[i, j]) for j in range(len(imgs))
                            if j != i)
                prompt_margin.append(round(prompt_align[i] - other, 3))
            mean_margin = round(statistics.fmean(prompt_margin), 3)
            if mean_margin < 0.02:
                warnings.append(
                    f"prompts barely beat a DIFFERENT shot's prompt (mean margin "
                    f"{mean_margin:+.3f}): the stills are generic to the film, "
                    f"not specific to each shot")
            for i, m in enumerate(prompt_margin):
                if m < 0:
                    warnings.append(f"shot {i + 1}: its still matches another "
                                    f"shot's prompt better (margin {m:+.3f})")
        except Exception as exc:                       # noqa: BLE001
            warnings.append(f"prompt alignment unavailable "
                            f"({type(exc).__name__})")
    # WITHIN-SCENE jumps are defects (the hold must hold). ACROSS-SCENE jumps are
    # the edit doing its job -- with deliberate per-scene palettes they are style,
    # and the render normalises exposure/contrast first so they are not technical
    # errors. Only the within-scene ones gate/warn.
    within = split["within_scene"]
    if within["cuts"]:
        if within["luma_jump"] is not None and within["luma_jump"] > continuity.HOLD_LUMA_MAX:
            warnings.append(
                f"hold exposure jump: within-scene luma {within['luma_jump']:.1f} "
                f"over {within['cuts']} cut(s) -- the same scene should not "
                f"brighten/darken between angles")
        if within["hist_jump"] is not None and within["hist_jump"] > continuity.HOLD_HIST_MAX:
            warnings.append(
                f"hold colour jump: within-scene hist {within['hist_jump']:.2f} "
                f"-- reads as a jump cut inside the hold")
        if within["clip_sim"] is not None and within["clip_sim"] < continuity.HOLD_CLIP_MIN:
            warnings.append(
                f"hold CLIP {within['clip_sim']:.2f} -- the stills inside a hold "
                f"do not belong together")

    # DOES THE FILM SHOW WHAT IT IS ABOUT? The failed squid cut passed continuity
    # and palettes and still rendered helmet-creatures instead of a diver and a
    # squid: every prompt was anchored on a prop. This is the legibility floor.
    subject_sim: Dict[str, float] = {}
    if scorer is not None and expect:
        try:
            import torch
            I = scorer.embed_images(imgs)
            for term in expect:
                t = scorer.embed_text(term)
                sims = (I @ t).cpu().tolist()
                subject_sim[term] = round(float(statistics.fmean(sims)), 4)
            missing = [t for t, v in subject_sim.items() if v < SUBJECT_MIN]
            for term in missing:
                warnings.append(
                    f"expected subject not legible: '{term}' (CLIP mean "
                    f"{subject_sim[term]:.3f} < {SUBJECT_MIN}) -- the stills do not "
                    f"show it; do not spend the render")
        except Exception as exc:                       # noqa: BLE001
            warnings.append(f"subject legibility unavailable "
                            f"({type(exc).__name__})")
    return {
        "shots": len(shots),
        "scenes": len({s.scene for s in shots}),
        "cuts": len(shots) - 1,
        "asl_s": round(mean, 2),
        "shot_cv": round(sd / mean, 2) if mean else None,
        "clip_sim_mean": round(statistics.fmean(clip_sims), 3) if clip_sims else None,
        "hist_jump_mean": round(statistics.fmean(hist_jumps), 3) if hist_jumps else None,
        "luma_jump_mean": round(statistics.fmean(luma_jumps), 1) if luma_jumps else None,
        "within_scene": within,
        "across_scenes": split["across_scenes"],
        "subject_sim": subject_sim,
        "prompt_align": prompt_align,
        "prompt_margin": prompt_margin,
        "prompt_margin_mean": (round(statistics.fmean(prompt_margin), 3)
                               if prompt_margin else None),
        "warnings": warnings,
        "cuts_detail": rows,
    }


def _contact_sheet(files: Sequence[Path], out: Path,
                   shots: Sequence[Shot] = None) -> Path:
    """One still per shot, with the prompt UNDER it.

    The prompt label is the systemic part: the first rendered film ended on a car,
    and the same failure in a previs still would have been visible here for 30 s
    of render instead of an hour. The sheet is for the eye; the metrics are not
    trusted to replace it.
    """
    n = len(files)
    W = 300
    band = 34 if shots else 0
    sheet = Image.new("RGB", (W * n + 4 * (n - 1), W + band), (0, 0, 0))
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB").resize((W, W), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 90, 20), fill=(0, 0, 0))
        d.text((4, 4), f"shot {i + 1}", fill=(255, 255, 0))
        sheet.paste(im, (i * (W + 4), 0))
        if shots and i < len(shots):
            prompt = getattr(shots[i], "prompt", "") or ""
            beat = getattr(shots[i], "beat", "") or ""
            d2 = ImageDraw.Draw(sheet)
            d2.text((i * (W + 4) + 3, W + 3),
                    f"{beat[:5]}  {prompt[:40]}", fill=(235, 235, 235))
            d2.text((i * (W + 4) + 3, W + 18), f"      {prompt[40:76]}",
                    fill=(175, 175, 180))
    sheet.save(out)
    return out


# --------------------------------------------------------------------------- #
# The gate: previs is part of the render decision, not an optional extra
# --------------------------------------------------------------------------- #

def fingerprint(project: Dict) -> str:
    """A hash of exactly what was measured: prompt/scene/link/seconds/camera.

    If the cut changes, the previs report is stale and the gate closes again. This
    is the seam that makes "previs" a gate rather than a screenshot someone
    maybe looked at.
    """
    import film_store as fs
    shots = fs.shots_in_order(project)
    st = project.get("settings") or {}
    payload = {
        "frames": int(st.get("frames") or 16), "fps": int(st.get("fps") or 16),
        "shots": [{
            "prompt": s.get("prompt"), "scale": s.get("scale"),
            "scene": s.get("scene"), "link": s.get("link"),
            "seconds": s.get("seconds"), "camera": s.get("camera"),
            "camera_move": s.get("camera_move"), "subject": s.get("subject"),
            "beat": s.get("beat"), "purpose": s.get("purpose"),
        } for s in shots],
    }
    import hashlib
    import json as _json
    return hashlib.sha1(_json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def name_for_store(slug: str) -> str:
    """The folder a store's previs must live in for `gate()` to find it."""
    return f"{slug}_previs"


def report_path(slug: str) -> Path:
    return OUT / name_for_store(slug) / "report.json"


def gate(project: Dict) -> Dict:
    """May this cut be rendered? Previs must be current and clean.

    Conservative, because a false block costs a click while a false pass costs an
    hour of CPU. It only fails a cut where a SAME-SCENE boundary measures like a
    jump (histogram > 0.80 or CLIP < 0.45) -- the exact failure the A/B showed.
    Missing/stale reports are their own status so the studio can run previs
    automatically instead of asking the user to remember.
    """
    slug = project.get("slug") or ""
    path = report_path(slug)
    fp = fingerprint(project)
    if not path.exists():
        return {"ok": False, "status": "missing", "stale": True,
                "reasons": ["previs has not been run for this shot list"],
                "report": None, "path": str(path)}
    try:
        rep = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                         # noqa: BLE001
        return {"ok": False, "status": "stale", "stale": True,
                "reasons": [f"previs report unreadable: {type(exc).__name__}"],
                "report": None, "path": str(path)}
    if rep.get("fingerprint") != fp:
        return {"ok": False, "status": "stale", "stale": True,
                "reasons": ["the cut changed since previs was run"],
                "report": rep, "path": str(path)}
    reasons: List[str] = []
    for row in rep.get("cuts_detail") or []:
        if not row.get("same_scene"):
            continue
        hist = row.get("hist_jump")
        clip = row.get("clip_sim")
        if hist is not None and hist > continuity.HOLD_HIST_MAX:
            reasons.append(f"cut {row.get('cut')}: same scene but colour jump "
                           f"{hist:.2f} (reads as a jump cut)")
        if clip is not None and clip < continuity.HOLD_CLIP_MIN:
            reasons.append(f"cut {row.get('cut')}: same scene but CLIP "
                           f"{clip:.2f} (images do not belong together)")
    # the subject must be on screen; continuity + grading alone let the failed
    # squid cut (helmet-creatures) through
    for term, sim in (rep.get("subject_sim") or {}).items():
        if sim < SUBJECT_MIN:
            reasons.append(f"expected subject '{term}' is not legible "
                           f"(CLIP {sim:.3f} < {SUBJECT_MIN})")
    return {"ok": not reasons, "status": "ok" if not reasons else "failed",
            "stale": False, "reasons": reasons[:5], "report": rep,
            "path": str(path)}


# --------------------------------------------------------------------------- #
# Shot-list sources
# --------------------------------------------------------------------------- #

def shots_from_store(slug: str) -> List[Shot]:
    import film_store as fs
    project = fs.load(slug)
    shots = []
    for d in fs.shots_in_order(project):
        cam = d.get("camera_suggestion") or d.get("camera") or "none"
        shots.append(Shot(prompt=d.get("prompt") or "",
                          scale=d.get("scale") or "medium",
                          camera=cam if cam != "none" else "none",
                          seconds=float(d.get("seconds") or 2.5),
                          subject=d.get("subject") or "",
                          scene=d.get("scene") or "s1",
                          energy=float(d.get("energy") or 0.2),
                          beat=d.get("beat") or "develop",
                          purpose=d.get("purpose") or "",
                          link=d.get("link") or ""))
    return shots


def shots_from_proposal(path: str) -> List[Shot]:
    from story import validate
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    shots, _ = validate(raw)
    return shots


def run(shots: List[Shot], name: str, scorer=None,
        condition: bool = True, fingerprint: Optional[str] = None,
        palettes: Optional[Sequence[str]] = None,
        baseline: bool = True,
        expect: Optional[Sequence[str]] = None) -> Dict:
    out = OUT / name
    if out.exists():
        for f in out.glob("*.png"):
            f.unlink()
    t0 = time.time()
    print(f"\n=== previs: {name} ({len(shots)} stills) ===")
    rendered = render_stills(shots, out, condition=condition, palettes=palettes,
                             baseline=baseline)
    print("loading CLIP...", flush=True)
    if scorer is None:
        from motion_metrics import ClipScorer
        scorer = ClipScorer()
    rep = measure(rendered["files"], shots, scorer, expect=expect)
    rep["render_seconds"] = rendered["total_s"]
    rep["conditioned"] = rendered["conditioned"]
    rep["palettes"] = rendered.get("palettes") or []
    rep["baseline"] = bool(rendered.get("baseline"))
    rep["fingerprint"] = fingerprint
    rep["render_wall_s"] = round(time.time() - t0, 1)
    rep["seed"] = SEED
    rep["files"] = [str(f) for f in rendered["files"]]
    # the render this previs is meant to protect against (measured: ~11-15 min
    # per 16-frame shot on this CPU; see HANDOFF)
    full = len(shots) * 12 * 60
    rep["full_render_estimate_s"] = full
    rep["saved_ratio"] = round(full / max(1.0, rep["render_wall_s"]), 1)
    _contact_sheet(rendered["files"], out / "contact.png", shots=shots)
    (out / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"  continuity (stills): clip {rep['clip_sim_mean']} · "
          f"hist jump {rep['hist_jump_mean']} · luma jump {rep['luma_jump_mean']}")
    w, a = rep.get("within_scene") or {}, rep.get("across_scenes") or {}
    print(f"    within a scene: luma {w.get('luma_jump')} · "
          f"hist {w.get('hist_jump')} · clip {w.get('clip_sim')} "
          f"({w.get('cuts')} cut(s) -- holds must stay continuous)")
    print(f"    scene cuts:     luma {a.get('luma_jump')} · "
          f"hist {a.get('hist_jump')} · clip {a.get('clip_sim')} "
          f"({a.get('cuts')} cut(s) -- style difference, not an error)")
    if rep.get("subject_sim"):
        print("  subject legibility: "
              + ", ".join(f"{t}={v:.3f}" for t, v in rep["subject_sim"].items()))
    if rep.get("baseline"):
        print("  technical baseline applied before palettes (as the render does)")
    if rep.get("prompt_margin_mean") is not None:
        print(f"  prompt->still: align {statistics.fmean(rep['prompt_align']):.3f}"
              f" · margin {rep['prompt_margin_mean']:+.3f} over a different "
              f"shot's prompt")
    for w in rep.get("warnings") or []:
        print(f"  !! {w}")
    print(f"  structure: {rep['scenes']} scene(s), {rep['cuts']} cut(s), "
          f"ASL {rep['asl_s']}s, shot CV {rep['shot_cv']}")
    print(f"  cost: {rep['render_wall_s']}s previs vs ~{full / 60:.0f} min full "
          f"render ({rep['saved_ratio']}x)")
    print(f"  files: {out / 'contact.png'} · {out / 'report.json'}")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="One keyframe per shot, before rendering")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--store")
    src.add_argument("--proposal")
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-condition", action="store_true",
                    help="do NOT condition held stills on their anchor (the "
                         "naive previs; measured NOT to predict cut continuity)")
    ap.add_argument("--no-palette", action="store_true",
                    help="render ungraded stills instead of applying each "
                         "scene's palette the way make_film will")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the shared technical baseline the render applies "
                         "before palettes (raw model output for comparison)")
    ap.add_argument("--expect", default=None,
                    help="comma-separated subject phrases the stills must show, "
                         "e.g. 'a scuba diver in a wetsuit,a giant squid'. CLIP "
                         "legibility below 0.20 fails the gate and blocks the render")
    a = ap.parse_args()
    palettes: Optional[List[str]] = None
    if a.store:
        import film_store as fs
        project = fs.load(a.store)
        shots = shots_from_store(a.store)
        # MUST match gate()/report_path(): a CLI previs written to previs/<slug>/
        # left the gate reading previs/<slug>_previs/ and reporting "missing" for
        # every CLI-run previs, so the render gate could never be satisfied.
        name = a.name or name_for_store(a.store)
        fp = fingerprint(project)
        palettes = [fs.palette_for(project, d) for d in fs.shots_in_order(project)]
        if a.no_palette:
            palettes = None
    else:
        shots = shots_from_proposal(a.proposal)
        name = a.name or Path(a.proposal).stem
        fp = None
    if not shots:
        print("no shots")
        return 1
    expect = ([x.strip() for x in a.expect.split(",") if x.strip()]
              if a.expect else None)
    run(shots, name, condition=not a.no_condition, fingerprint=fp,
        palettes=palettes, baseline=not a.no_baseline, expect=expect)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
