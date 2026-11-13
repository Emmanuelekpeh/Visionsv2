#!/usr/bin/env python3
"""
keyframe.py -- an approved FIRST FRAME per shot, before the clip is animated.

WHY. Measured on skydiving s05: the four-armed figure in frame 0 is in all 16 frames.
AnimateDiff denoises the frames together and the clip barely moves, so whatever the
sampler drew at the start IS the shot. A bad start costs the whole ~47 min render.
A still costs a couple of minutes, so the still is made, LOOKED AT, and only an
approved one is animated (`make_film.Generator.clip_from_still`).

This is not previs. `previs.py` renders its stills with the 4-step Lightning sampler
(the one that draws the extra limbs) and the clip does not start from them. A
keyframe is rendered with the FULL sampler and IS the clip's first frame.

    python keyframe.py --store skydiving --shot s05 --candidates 3
    python keyframe.py --store skydiving                  # every shot, 1 candidate
    python keyframe.py --store squid --shot s05 --look amused   # aMUSEd, then DreamShaper

Stills go to  productions/<slug>/keyframes/<shot>/c<k>.png  with a contact sheet
`sheet.png` per shot; choosing one is `fstore`'s `keyframe` field (see
`choose`). Nothing here is judged by a metric: the sheet is for the eye.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import film_store as fs  # noqa: E402

# a still is ONE frame, so it can afford more steps than a clip at the same cost
KEYFRAME_STEPS = 20
SIZE = 512

# THE aMUSEd LOOK. aMUSEd-256 (a masked-token image model, not a diffusion UNet)
# draws a 256 px still with its own composition, palette and texture; that still is
# upscaled to 512 and re-drawn by DreamShaper at `strength`, so the first frame the
# clip grows from carries a look DreamShaper alone does not produce. Low strength
# keeps more of aMUSEd (and its soft, blobby detail); high strength keeps only its
# layout and colour. Which one is good is a question for the eye -- every strength is
# saved as its own candidate and nothing is chosen for you.
AMUSED_REPO = "amused/amused-256"
AMUSED_SIZE = 256
AMUSED_STEPS = 12
AMUSED_CFG = 10.0
# Measured on five skydiving shots (same seeds, aMUSEd only): the character's avoid list
# as the NEGATIVE is what keeps the figure's suit and anatomy right (without it s10
# became a long snail-like body and s12 a green suit), and subject-first ordering
# (`amused_prompt`) is what put a figure in s01. Both are on by default.
LOOK_STRENGTHS = (0.5, 0.65, 0.8)
LOOKS = ("plain", "amused")


def amused_prompt(prompt: str, subject: str = "") -> str:
    """The shot prompt re-ordered for aMUSEd: the clause that NAMES THE SUBJECT first.

    Story prompts often open with scene tags ("mountain ridge mid-afternoon wind gusts
    up, A male skydiver ..."). aMUSEd draws a 16x16 grid of tokens from one CLIP
    embedding, and CLIP weights early words most, so a figure named late gets few of
    them. The tags are moved to the END, not dropped. A prompt that already starts
    with its subject, or never names it, is returned unchanged.
    """
    parts = [x.strip() for x in prompt.split(",") if x.strip()]
    words = (subject or "").lower().replace("-", " ").split()
    key = words[-1] if words else ""
    if not key or len(parts) < 2:
        return prompt.strip()
    at = next((i for i, x in enumerate(parts) if key in x.lower()), None)
    if not at:                                   # None (never named) or 0 (already first)
        return prompt.strip()
    return ", ".join(parts[at:] + parts[:at])


def next_candidate(project: Dict, shot_id: str) -> int:
    """The first free candidate number, so a new batch never overwrites an old one."""
    d = keyframe_dir(project, shot_id)
    taken = [int(p.stem[1:]) for p in d.glob("c*.png") if p.stem[1:].isdigit()] \
        if d.exists() else []
    return max(taken) + 1 if taken else 0


def candidate_info(project: Dict, shot_id: str) -> List[Dict]:
    """Every candidate still for a shot with what made it, oldest first."""
    d = keyframe_dir(project, shot_id)
    out = []
    if not d.exists():
        return out
    for p in sorted(d.glob("c*.png"), key=lambda q: int(q.stem[1:] or 0)
                    if q.stem[1:].isdigit() else 0):
        meta = {}
        j = p.with_suffix(".json")
        if j.exists():
            try:
                meta = json.loads(j.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        out.append({"path": p, "candidate": p.stem, **meta})
    return out


def _amused_stills(prompts: Dict[str, List[tuple]], threads: Optional[int]):
    """aMUSEd text-to-image for {shot: [(seed, prompt, negative)]} -> {shot: [PIL]}.

    Loaded from the local HuggingFace cache first (this machine is often offline);
    freed before the SD stage so both never sit in RAM together."""
    import gc
    import torch
    from diffusers import AmusedPipeline
    if threads:
        torch.set_num_threads(threads)
    try:
        pipe = AmusedPipeline.from_pretrained(AMUSED_REPO, local_files_only=True)
    except Exception:
        pipe = AmusedPipeline.from_pretrained(AMUSED_REPO)
    pipe.set_progress_bar_config(disable=True)
    out: Dict[str, List[Image.Image]] = {}
    for sid, jobs in prompts.items():
        out[sid] = []
        for seed, prompt, neg in jobs:
            t0 = time.time()
            with torch.no_grad():
                im = pipe(prompt=prompt, negative_prompt=neg,
                          num_inference_steps=AMUSED_STEPS,
                          guidance_scale=AMUSED_CFG, height=AMUSED_SIZE,
                          width=AMUSED_SIZE,
                          generator=torch.Generator("cpu").manual_seed(seed)
                          ).images[0]
            out[sid].append(im.convert("RGB"))
            print(f"  [keyframe] {sid} aMUSEd seed {seed}: {time.time() - t0:.0f}s",
                  flush=True)
    del pipe
    gc.collect()
    return out


def keyframe_dir(project: Dict, shot_id: str) -> Path:
    return fs.dir_for(project["slug"]) / "keyframes" / shot_id


def candidate_seed(project: Dict, shot: Dict, k: int) -> int:
    """Candidate 0 uses the shot's own seed; later ones step from it."""
    return int(fs.generation_inputs(project, shot)["seed"]) + int(k)


def keyframe_plan() -> Dict:
    """Sampler settings for a still: the FULL tier's guidance/negative, more steps."""
    import make_film
    plan = dict(make_film.sampling_plan("full"))
    plan["steps"] = KEYFRAME_STEPS
    return plan


def _pipeline():
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        str(ROOT / "models" / "dreamshaper8"), torch_dtype=torch.float32)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_vae_slicing()
    return pipe, torch


def render(project: Dict, shot_ids: List[str], candidates: int = 1,
           threads: Optional[int] = None, look: str = "plain",
           strengths=LOOK_STRENGTHS) -> Dict[str, List[Path]]:
    """Render stills per shot. Returns {shot_id: [new paths]}.

    look="plain":  `candidates` DreamShaper stills (full sampler).
    look="amused": `candidates` aMUSEd stills, EACH re-drawn by DreamShaper at every
                   strength in `strengths` -> candidates x len(strengths) files.
    New files continue the numbering; old candidates are never overwritten.
    """
    if look not in LOOKS:
        raise ValueError(f"look must be one of {LOOKS}, not {look!r}")
    plan = keyframe_plan()
    avoid = fs.character_avoid(project)
    neg = f"{avoid}, {plan['neg']}" if avoid else plan["neg"]
    shots = {sid: fs.get_shot(project, sid) for sid in shot_ids}
    firsts = {sid: next_candidate(project, sid) for sid in shot_ids}

    raw: Dict[str, List[Image.Image]] = {}
    if look == "amused":
        raw = _amused_stills(
            {sid: [(candidate_seed(project, shots[sid], k),
                    amused_prompt(shots[sid]["prompt"], shots[sid].get("subject", "")),
                    avoid) for k in range(candidates)] for sid in shot_ids}, threads)

    pipe, torch = _pipeline()
    if threads:
        torch.set_num_threads(threads)
    img2img = None
    if look == "amused":
        from diffusers import StableDiffusionImg2ImgPipeline
        img2img = StableDiffusionImg2ImgPipeline.from_pipe(pipe)
        img2img.set_progress_bar_config(disable=True)
    out: Dict[str, List[Path]] = {}
    for sid in shot_ids:
        shot = shots[sid]
        d = keyframe_dir(project, sid)
        d.mkdir(parents=True, exist_ok=True)
        out[sid] = []
        n = firsts[sid]
        for k in range(candidates):
            seed = candidate_seed(project, shot, k)
            if look == "plain":
                variants = [(None, None)]
            else:
                m = raw[sid][k]
                m.save(d / f"m{n}.png")                 # what aMUSEd itself drew
                base = m.resize((SIZE, SIZE), Image.LANCZOS)
                variants = [(base, float(s)) for s in strengths]
            for base, strength in variants:
                t0 = time.time()
                gen = torch.Generator("cpu").manual_seed(seed)
                with torch.no_grad():
                    if base is None:
                        im = pipe(prompt=shot["prompt"], negative_prompt=neg,
                                  num_inference_steps=plan["steps"],
                                  guidance_scale=plan["cfg"], width=SIZE, height=SIZE,
                                  generator=gen).images[0]
                    else:
                        im = img2img(prompt=shot["prompt"], negative_prompt=neg,
                                     image=base, strength=strength,
                                     num_inference_steps=plan["steps"],
                                     guidance_scale=plan["cfg"], generator=gen
                                     ).images[0]
                p = d / f"c{n}.png"
                im.save(p)
                (d / f"c{n}.json").write_text(json.dumps({
                    "shot": sid, "candidate": n, "seed": seed,
                    "prompt": shot["prompt"], "steps": plan["steps"],
                    "cfg": plan["cfg"], "neg": neg, "look": look,
                    "strength": strength,
                    "amused": f"m{n}.png" if base is not None else None,
                    "amused_prompt": (amused_prompt(shot["prompt"],
                                                    shot.get("subject", ""))
                                      if base is not None else None),
                    "seconds": round(time.time() - t0, 1)}, indent=2),
                    encoding="utf-8")
                out[sid].append(p)
                print(f"  [keyframe] {sid} c{n} seed {seed} {look}"
                      + (f" strength {strength}" if strength else "")
                      + f": {time.time() - t0:.0f}s", flush=True)
                n += 1
        sheet([c["path"] for c in candidate_info(project, sid)],
              d / "sheet.png", sid)
    return out


def sheet(files: List[Path], out: Path, label: str = "") -> Path:
    """Candidates side by side, for the eye."""
    W = 384
    s = Image.new("RGB", (W * len(files), W + 16), "black")
    d = ImageDraw.Draw(s)
    for i, f in enumerate(files):
        s.paste(Image.open(f).convert("RGB").resize((W, W)), (i * W, 16))
        d.text((i * W + 4, 2), f"{label} {f.stem}", fill="white")
    s.save(out)
    return out


def choose(project: Dict, shot_id: str, candidate: int) -> Path:
    """Approve one candidate: it becomes the shot's keyframe."""
    p = keyframe_dir(project, shot_id) / f"c{int(candidate)}.png"
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist -- render it first")
    fs.update_shot(project, shot_id, keyframe=str(p.relative_to(
        fs.dir_for(project["slug"]))).replace("\\", "/"))
    return p


def approved(project: Dict, shot: Dict) -> Optional[Path]:
    """The shot's approved keyframe file, or None."""
    rel = shot.get("keyframe")
    if not rel:
        return None
    p = fs.dir_for(project["slug"]) / rel
    return p if p.exists() else None


ANIMATE_MODES = ("v2v", "move", "ip", "ramp", "ip+ramp", "svd")


SVD_SIZE = (512, 288)          # width, height: what was measured on this machine
SVD_FRAMES = 14


def _svd_pipeline():
    import torch
    from diffusers import StableVideoDiffusionPipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        str(ROOT / "models" / "svd_xt"), torch_dtype=torch.float32, variant="fp16")
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    return pipe


def svd_animate(project: Dict, shot_id: str, candidate: int, motion_bucket: int = 200,
                noise_aug: float = 0.1, crop_y: int = 150, steps: int = 25) -> int:
    """Stable Video Diffusion XT on the chosen still (a real image-to-video model).

    The earlier SVD measurement (HANDOFF section 3) used the defaults, motion bucket
    127 and noise augmentation 0.02, and found almost no motion. Both knobs trade
    fidelity for motion: a higher bucket asks for more movement, a higher noise
    augmentation loosens how tightly the still is followed. The 512x512 still is
    cropped to 512x288 at `crop_y` (SVD wants a wide frame); no take is written.
    """
    import torch
    shot = fs.get_shot(project, shot_id)
    still = keyframe_dir(project, shot_id) / f"c{int(candidate)}.png"
    if not still.exists():
        print(f"{still} does not exist")
        return 2
    w, h = SVD_SIZE
    img = Image.open(still).convert("RGB")
    crop_y = max(0, min(int(crop_y), img.height - int(img.width * h / w)))
    img = img.crop((0, crop_y, img.width, crop_y + round(img.width * h / w))).resize(
        (w, h), Image.LANCZOS)
    pipe = _svd_pipeline()
    t0 = time.time()
    with torch.no_grad():
        out = pipe(img, num_frames=SVD_FRAMES, decode_chunk_size=2, height=h, width=w,
                   num_inference_steps=int(steps), min_guidance_scale=1.0,
                   max_guidance_scale=3.0, motion_bucket_id=int(motion_bucket),
                   noise_aug_strength=float(noise_aug), fps=7,
                   generator=torch.Generator("cpu").manual_seed(
                       candidate_seed(project, shot, candidate)))
    dt = time.time() - t0
    d = (ROOT / "output" / "audit" / project["slug"]
         / f"kf_{shot_id}_c{candidate}_svd_m{int(motion_bucket)}")
    d.mkdir(parents=True, exist_ok=True)
    img.save(d / "input.png")
    for i, fr in enumerate(out.frames[0]):
        fr.convert("RGB").save(d / f"f{i:03d}.png")
    print(f"[keyframe] animated {shot_id} c{candidate} svd bucket {motion_bucket} "
          f"noise_aug {noise_aug}: {len(out.frames[0])} frames in {dt:.0f}s -> {d}",
          flush=True)
    return 0


def animate_test(project: Dict, shot_id: str, candidate: int,
                 strength: Optional[float], mode: str = "v2v",
                 ip_scale: Optional[float] = None,
                 ramp: Optional[tuple] = None, move: str = "zoom-in",
                 move_amount: float = 0.0, svd: Optional[Dict] = None,
                 avoid_extra: str = "") -> int:
    """Animate one candidate through the render's own generator. No take is written.

    v2v      `clip_from_still`: the still repeated, noised to `strength`. Measured on
             skydiving s01: the figure held, the clip barely moved (it starts after the
             high-noise steps that decide motion).
    ip       generate FROM NOISE (full schedule) with the still as an IP-Adapter
             reference, which carries look, suit and figure; pose may drift.
    move     video-to-video on an input that ALREADY MOVES: the still given a slow
             camera path (`camera_path.apply_path`, no hold), so the temporal layers
             are handed real motion to keep instead of 16 identical frames. The
             motion is the camera's; the subject's own motion is up to the model.
    ramp     generate from noise but re-impose the still on the first frames after
             every step (`make_film.anchor_callback`): frame 0 is the still, later
             frames are freer.
    ip+ramp  both.
    The IP modes refuse to run if the adapter did not load: an unconditioned clip
    would pass for a result it is not.
    """
    if mode not in ANIMATE_MODES:
        raise ValueError(f"mode must be one of {ANIMATE_MODES}, not {mode!r}")
    if mode == "svd":
        return svd_animate(project, shot_id, candidate, **(svd or {}))
    import make_film
    from previs import shots_from_store
    shot = fs.get_shot(project, shot_id)
    still = keyframe_dir(project, shot_id) / f"c{int(candidate)}.png"
    if not still.exists():
        print(f"{still} does not exist")
        return 2
    ids = fs._ordered_ids(project)
    s = shots_from_store(project["slug"])[ids.index(shot_id)]
    gen = make_film.Generator("models/dreamshaper8", loras=[])
    seed = candidate_seed(project, shot, candidate)
    avoid = fs.character_avoid(project)
    if avoid_extra:                       # extra negatives for THIS test only
        avoid = f"{avoid}, {avoid_extra}" if avoid else avoid_extra
    use_ip, use_ramp = "ip" in mode, "ramp" in mode
    if mode in ("v2v", "move"):
        src = still
        if mode == "move":
            from camera_path import apply_path
            img = Image.open(still).convert("RGB").resize((make_film.SIZE,) * 2)
            src = apply_path([img] * make_film.FRAMES, move, float(move_amount),
                             settle=1.0)
        frames, dt = gen.clip_from_still(s, seed, src, strength=strength,
                                         avoid=avoid)
        label = ((f"move {move} {move_amount} + " if mode == "move" else "") +
                 "v2v strength "
                 f"{strength if strength is not None else fs.KEYFRAME_STRENGTH}")
        suffix = ("_move" if mode == "move" else "") + (
            "" if strength is None else f"_s{int(round(strength * 100))}")
    else:
        if ip_scale is not None:
            make_film.IP_ADAPTER_SCALE = float(ip_scale)
        img = Image.open(still).convert("RGB")
        frames, dt = gen.clip(s, seed, reference=img if use_ip else None,
                              tier="full", avoid=avoid,
                              anchor=img if use_ramp else None, ramp=ramp)
        if use_ip and not gen.conditioned:
            print("[keyframe] the IP-Adapter did not load, so this clip is NOT "
                  "conditioned on the still; discarding it")
            return 3
        label = " + ".join(
            ([f"ip-adapter scale {make_film.IP_ADAPTER_SCALE}"] if use_ip else [])
            + ([f"ramp {tuple(ramp or make_film.ANCHOR_RAMP)}"] if use_ramp else []))
        suffix = "_" + mode.replace("+", "") + ("_x" if avoid_extra else "")
    out = (ROOT / "output" / "audit" / project["slug"]
           / f"kf_{shot_id}_c{candidate}{suffix}")
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(out / f"f{i:03d}.png")
    print(f"[keyframe] animated {shot_id} c{candidate} {label}: "
          f"{len(frames)} frames in {dt:.0f}s -> {out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Full-sampler first frame per shot")
    ap.add_argument("--store", required=True)
    ap.add_argument("--shot", help="one shot id (default: every shot)")
    ap.add_argument("--candidates", type=int, default=1)
    ap.add_argument("--look", choices=LOOKS, default="plain",
                    help="plain = DreamShaper; amused = aMUSEd-256 still, then "
                         "DreamShaper img2img at each --looks-strengths value")
    ap.add_argument("--looks-strengths", default=",".join(str(s) for s in LOOK_STRENGTHS),
                    help="img2img strengths for --look amused, comma separated")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads (lower it while a render is running)")
    ap.add_argument("--choose", type=int, default=None,
                    help="approve this candidate for --shot instead of rendering")
    ap.add_argument("--animate", type=int, default=None,
                    help="animate this candidate of --shot (a test: frames go to "
                         "output/audit/<store>/kf_<shot>_c<N>/, no take is written)")
    ap.add_argument("--mode", choices=ANIMATE_MODES, default="v2v",
                    help="--animate: v2v = start from the still; ip = from noise with "
                         "the still as an IP-Adapter reference; ramp = from noise with "
                         "the still pinned on the first frames; ip+ramp = both")
    ap.add_argument("--motion-bucket", type=int, default=200,
                    help="--mode svd: 0..255, higher asks for more movement (default "
                         "SVD value is 127)")
    ap.add_argument("--noise-aug", type=float, default=0.1,
                    help="--mode svd: higher loosens how tightly the still is "
                         "followed (SVD default is 0.02)")
    ap.add_argument("--crop-y", type=int, default=150,
                    help="--mode svd: top of the 512x288 window cut from the still")
    ap.add_argument("--avoid-extra", default="",
                    help="--animate: extra negative-prompt words for this test only, "
                         "e.g. 'two people, duplicate'")
    ap.add_argument("--move", default="zoom-in",
                    help="--mode move: a camera_path move (zoom-in, zoom-out, pan-left, "
                         "pan-right, tilt-up, tilt-down)")
    ap.add_argument("--move-amount", type=float, default=0.0,
                    help="--mode move: travel in px for pans/tilts (0 for a zoom)")
    ap.add_argument("--ramp", default=None,
                    help="--mode ramp: per-frame still weights, e.g. 1,0.6,0.3,0.1")
    ap.add_argument("--ip-scale", type=float, default=None,
                    help="--mode ip: reference strength (default: the film's 0.7)")
    ap.add_argument("--strength", type=float, default=None,
                    help="how far the clip may move from the still (0..1)")
    a = ap.parse_args()
    project = fs.load(a.store)
    if a.animate is not None:
        if not a.shot:
            print("--animate needs --shot")
            return 2
        return animate_test(project, a.shot, a.animate, a.strength, mode=a.mode,
                            ip_scale=a.ip_scale, move=a.move,
                            move_amount=a.move_amount,
                            avoid_extra=a.avoid_extra,
                            svd={"motion_bucket": a.motion_bucket,
                                 "noise_aug": a.noise_aug, "crop_y": a.crop_y},
                            ramp=(tuple(float(x) for x in a.ramp.split(","))
                                  if a.ramp else None))
    if a.choose is not None:
        if not a.shot:
            print("--choose needs --shot")
            return 2
        print(f"approved {choose(project, a.shot, a.choose)}")
        return 0
    ids = [a.shot] if a.shot else fs._ordered_ids(project)
    render(project, ids, candidates=a.candidates, threads=a.threads, look=a.look,
           strengths=tuple(float(x) for x in a.looks_strengths.split(",") if x.strip()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
