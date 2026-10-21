"""
make_montage.py -- 60 images, montage density, authored rhythm.

The previous film was 8 images held 2.5s each: a slideshow. This one attacks
both deficits found by measurement:

  image count  SD1.5 + LCM-LoRA at 6 steps = ~43 images/hour, vs 17 for tiny-sd.
               60 images instead of 8.
  rhythm       holds are authored, not uniform: a slow open, accelerating cuts,
               a fast burst at the turn, then a long close. Mixed exposure is
               exactly what the exposure sheet exists for.

Still nothing blends: every frame is one image, or a crop of one image.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from animation_spec import (
    AnimationState,
    Drawing,
    DrawingSource,
    Easing,
    ExposureSheet,
    ExposureTrack,
    KeyPose,
    Layer,
    MotionPlan,
)

FPS = 24
SRC = 512
OUT_W, OUT_H = 960, 540
STEPS = 6
CFG = 1.5

AESTHETIC = ("filmic grain, muted restrained color, natural texture, soft detail, "
             "imperfect exposure, 35mm photograph")

DIRECTOR_PROMPT = """You are directing a short poetic film from this concept:

"{concept}"

This is a MONTAGE, not a slideshow. It must have a shape:
  ACT 1 (settling): the world before. Calm, distant, observational.
  ACT 2 (descent): going deeper in. Things get closer, stranger, more intimate.
  ACT 3 (the turn): the moment it accepts. This is the emotional peak.
  ACT 4 (after): changed. Quiet, resolved, wide again.

Write exactly 20 beats, 5 per act, in order. Each beat is ONE static image.
Four rules:
- Describe physical contents FIRST (what is literally in frame), then light/mood.
- Every beat must ADVANCE the film. No two beats may be interchangeable.
- Vary scale hard: wide landscape, macro detail, a figure, an object, texture, light.
- No text, no logos, no fantasy cliches.

Return ONLY JSON:
{{"title": "short title",
  "narration": "four short poetic lines separated by |",
  "beats": [{{"act": 1, "prompt": "visual description"}}]}}"""

# 20 beats x 3 framings = 60 images
FRAMINGS = [
    "Wide establishing view: ",
    "Intimate close detail: ",
    "Distant, atmospheric: ",
]

# Authored montage rhythm: (images in this phase, frames held each)
RHYTHM = [
    ("open",       8, 24),   # 1.00s  settling
    ("descend",   12, 12),   # 0.50s  accelerating
    ("turn",       8,  8),   # 0.33s  tightening
    ("burst",     20,  4),   # 0.17s  the peak
    ("resolve",    8, 10),   # 0.42s  easing out
    ("close",      4, 34),   # 1.42s  long final hold
]

CAMERA_FOR_SLOW = True  # camera only on the long holds; bursts are pure cuts


def direct(concept: str, model: str) -> dict:
    import ollama
    print(f"  asking {model} ...", flush=True)
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": DIRECTOR_PROMPT.format(concept=concept)}],
        format="json",
        options={"temperature": 0.85, "num_predict": 2600},
    )
    raw = resp["message"]["content"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
    beats = [b for b in (data.get("beats") or []) if (b.get("prompt") or "").strip()]
    return {"title": data.get("title", concept[:40]),
            "narration": data.get("narration", ""),
            "beats": beats}


def build_prompts(beats) -> list:
    """20 beats x 3 framings, interleaved so each act's three scales sit together."""
    prompts = []
    for b in beats:
        base = b["prompt"].strip()
        for f in FRAMINGS:
            prompts.append(f"{f}{base}. {AESTHETIC}")
    return prompts


def build_rhythm(n_images: int) -> list:
    """Expand RHYTHM into one (phase, hold_frames) per image, scaled to n_images."""
    total_units = sum(n for _, n, _ in RHYTHM)
    out = []
    for phase, n, hold in RHYTHM:
        count = max(1, round(n * n_images / total_units))
        out.extend([(phase, hold)] * count)
    # trim or pad to exactly n_images
    while len(out) < n_images:
        out.append(out[-1])
    return out[:n_images]


def generate(prompts, out: Path, device="cpu"):
    from diffusers import LCMScheduler, StableDiffusionPipeline

    out.mkdir(parents=True, exist_ok=True)
    print("  loading SD1.5 + LCM-LoRA ...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        "models/sd15", torch_dtype=torch.float32, safety_checker=None, variant="fp16")
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("models/lcm_lora_sd15")
    pipe.fuse_lora()
    pipe.enable_attention_slicing()

    paths, done, t_start = [], 0, time.time()
    for i, prompt in enumerate(prompts):
        path = out / f"img_{i:03d}.png"
        paths.append(str(path))
        if path.exists():
            done += 1
            continue
        g = torch.Generator(device=device).manual_seed(500 + i)
        with torch.no_grad():
            img = pipe(prompt, num_inference_steps=STEPS, guidance_scale=CFG,
                       height=SRC, width=SRC, generator=g).images[0]
        img.save(path)
        done += 1
        if done % 5 == 0 or done == len(prompts):
            el = time.time() - t_start
            rate = done / max(el, 1e-9) * 3600
            left = (len(prompts) - done) / max(done / max(el, 1e-9), 1e-9)
            print(f"  [{done}/{len(prompts)}] {rate:5.1f} img/hr, "
                  f"~{left/60:4.1f} min left", flush=True)
    return paths


CAMERAS = {"push_in": 0.80, "pull_out": 1.22, "hold": 1.0}


def camera_window(kind, t):
    zoom = CAMERAS.get(kind, 1.0)
    e = Easing.SMOOTH.sample(t) if kind != "hold" else 0.0
    z = 1.0 + (zoom - 1.0) * e
    bh = 1.0 / z
    bw = bh * (16.0 / 9.0)
    if bw > 1.0:
        bw, bh = 1.0, 9.0 / 16.0
    x0 = (1.0 - bw) / 2.0
    y0 = (1.0 - bh) / 2.0
    return x0, y0, bw, bh


def render_frame(src, kind, t):
    W, H = src.size
    x0, y0, bw, bh = camera_window(kind, t)
    box = (int(x0 * W), int(y0 * H), int((x0 + bw) * W), int((y0 + bh) * H))
    box = (box[0], box[1], max(box[2], box[0] + 8), max(box[3], box[1] + 8))
    return src.crop(box).resize((OUT_W, OUT_H), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="I dove into nature and it accepted me")
    ap.add_argument("--out", default="output/montage")
    ap.add_argument("--model", default="qwen2.5:3b")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(12)

    print(f"\nCONCEPT: {args.concept}\n")
    print("=== DIRECTION (ollama) ===", flush=True)
    data = direct(args.concept, args.model)
    beats = data["beats"]
    print(f"  title    : {data['title']}")
    print(f"  narration: {data['narration']}")
    for i, b in enumerate(beats):
        print(f"    act{b.get('act','?')} {i+1:2d}. {b['prompt'][:72]}")

    prompts = build_prompts(beats)
    print(f"\n  {len(beats)} beats x {len(FRAMINGS)} framings = {len(prompts)} images")

    print("\n=== IMAGES (SD1.5 + LCM, 6 steps) ===", flush=True)
    t0 = time.time()
    paths = generate(prompts, out / "images")
    elapsed = time.time() - t0
    print(f"  {len(paths)} images in {elapsed/60:.1f} min "
          f"({len(paths)/max(elapsed,1e-9)*3600:.1f} img/hr)", flush=True)

    # ---- authored rhythm -> exposure sheet ----
    rhythm = build_rhythm(len(paths))
    poses = tuple(
        KeyPose(f"img{i:03d}", AnimationState.of(index=i),
                ease_into=Easing.SNAP, hold=hold)
        for i, (phase, hold) in enumerate(rhythm)
    )
    motion = MotionPlan(poses)
    sheet = ExposureSheet(sheet_id="montage", fps=FPS)
    for i, p in enumerate(paths):
        sheet.add_drawing(Drawing(id=f"img{i:03d}", source=DrawingSource.AUTHORED, note=p))
    sheet.add_layer(Layer(name="scene", track=motion.to_track()))
    sheet.validate()

    print("\n=== EXPOSURE SHEET ===")
    print(f"  {len(paths)} images -> {sheet.total_frames} frames @ {FPS}fps "
          f"= {sheet.duration_seconds:.1f}s")
    print(f"  image changes/sec: {len(paths)/sheet.duration_seconds:.2f} "
          f"(was 0.40 in the 8-image version)")
    phase_counts = {}
    for (phase, _), p in zip(rhythm, paths):
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    for phase, n, hold in RHYTHM:
        print(f"    {phase:8s} ~{phase_counts.get(phase,0):2d} images "
              f"held {hold:2d} frames ({hold/FPS:.2f}s)")

    # ---- render ----
    print("\n=== RENDER (cuts only; camera on long holds) ===", flush=True)
    kinds = []
    for (phase, hold) in rhythm:
        if CAMERA_FOR_SLOW and hold >= 10:
            kinds.append("push_in" if phase in ("open", "close") else "hold")
        else:
            kinds.append("hold")

    frames = []
    for i, (path, (phase, hold)) in enumerate(zip(paths, rhythm)):
        src = Image.open(path).convert("RGB")
        kind = kinds[i]
        for k in range(hold):
            t = k / max(1, hold - 1)
            frames.append(np.array(render_frame(src, kind, t)))
    print(f"  {len(frames)} frames rendered", flush=True)

    mp4 = out / "montage.mp4"
    try:
        from moviepy.editor import ImageSequenceClip
        ImageSequenceClip(frames, fps=FPS).write_videofile(
            str(mp4), fps=FPS, codec="libx264", audio=False, logger=None)
        print(f"wrote {mp4}", flush=True)
    except Exception as exc:
        print(f"mp4 failed: {exc}", flush=True)

    # contact sheet of every image, in order
    T = 150
    cols = 10
    rows = (len(paths) + cols - 1) // cols
    cs = Image.new("RGB", (cols * (T + 4) + 4, rows * (T + 4) + 4), (24, 24, 28))
    for i, p in enumerate(paths):
        r, c = divmod(i, cols)
        cs.paste(Image.open(p).resize((T, T), Image.LANCZOS), (4 + c * (T + 4), 4 + r * (T + 4)))
    cs.save(out / "all_images.png")
    print(f"wrote {out/'all_images.png'}", flush=True)

    (out / "direction.json").write_text(
        json.dumps({"concept": args.concept, "rhythm": RHYTHM, **data}, indent=2),
        encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
