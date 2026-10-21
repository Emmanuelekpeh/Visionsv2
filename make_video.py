"""
make_video.py -- concept -> story -> images -> a video worth watching.

This is the last stage of the pipeline, rebuilt so that NOTHING blends:

  * every frame is a crop of ONE generated image -- no crossfade, no morph,
    no interpolation between two different pictures
  * a "hold" shows the identical pixels for N frames
  * movement comes from a simulated camera over a single image

Timing comes from the locked exposure sheet, so holds and shot lengths are
authored data, not arithmetic.

    Ollama  -> narration + shot list (visual prompts, durations, camera moves)
    tiny-sd -> one image per shot
    sheet   -> how long each image is held
    render  -> 16:9 frames, camera applied, concatenated
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
    ExposureSheet,
    Easing,
    KeyPose,
    Layer,
    MotionPlan,
)

FPS = 24
SRC = 512                  # tiny-sd generation size (square)
OUT_W, OUT_H = 960, 540    # 16:9 delivery
STEPS = 25

AESTHETIC = ("Favor imperfect, physical imagery over pristine digital perfection. "
             "Soft detail, natural texture, filmic grain, subtle optical imperfections, "
             "restrained color, imperfect exposure.")

DIRECTOR_PROMPT = """You are directing a short poetic film from this concept:

"{concept}"

Write a shot list. Rules:
- 8 shots exactly.
- Each shot is a single striking static image. No camera instructions inside the prompt.
- Describe physical contents FIRST (what is actually in frame), then mood/light.
- Vary distance and subject: wide landscape, close detail, a figure, texture, sky.
- Restrained, observational, no fantasy cliches, no text.

Return ONLY JSON:
{{"title": "short title",
  "narration": "3 or 4 short poetic lines separated by |",
  "shots": [{{"prompt": "visual description", "seconds": 2.5,
             "camera": "push_in"}}]}}

camera must be one of: hold, push_in, pull_out, pan_left, pan_right.
seconds between 1.5 and 4.0."""


# ---------------------------------------------------------------------------
# 1. Ollama: story + shot list
# ---------------------------------------------------------------------------

def direct(concept: str, model: str = "qwen2.5:3b") -> dict:
    import ollama

    print(f"  asking {model} for a shot list ...", flush=True)
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": DIRECTOR_PROMPT.format(concept=concept)}],
        format="json",
        options={"temperature": 0.9, "num_predict": 1400},
    )
    raw = resp["message"]["content"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}

    shots = data.get("shots") or []
    clean = []
    for s in shots[:8]:
        p = (s.get("prompt") or "").strip()
        if not p:
            continue
        try:
            secs = float(s.get("seconds", 2.5))
        except (TypeError, ValueError):
            secs = 2.5
        cam = s.get("camera", "hold")
        if cam not in ("hold", "push_in", "pull_out", "pan_left", "pan_right"):
            cam = "hold"
        clean.append({"prompt": p, "seconds": min(4.0, max(1.5, secs)), "camera": cam})

    return {"title": data.get("title", concept[:40]),
            "narration": data.get("narration", ""), "shots": clean}


# ---------------------------------------------------------------------------
# 2. tiny-sd: one image per shot
# ---------------------------------------------------------------------------

def generate_images(shots, out: Path, device="cpu"):
    from diffusers import StableDiffusionPipeline

    out.mkdir(parents=True, exist_ok=True)
    print("  loading tiny-sd ...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        "models/tiny-sd", torch_dtype=torch.float32, safety_checker=None).to(device)
    pipe.set_progress_bar_config(disable=True)

    paths = []
    for i, shot in enumerate(shots):
        path = out / f"shot_{i:02d}.png"
        if path.exists():
            print(f"  [{i+1}/{len(shots)}] cached {path.name}", flush=True)
            paths.append(str(path))
            continue
        prompt = f"{shot['prompt']}. {AESTHETIC}"
        t0 = time.time()
        g = torch.Generator(device=device).manual_seed(100 + i)
        with torch.no_grad():
            img = pipe(prompt, num_inference_steps=STEPS, guidance_scale=7.5,
                       height=SRC, width=SRC, generator=g).images[0]
        img.save(path)
        paths.append(str(path))
        print(f"  [{i+1}/{len(shots)}] {time.time()-t0:5.0f}s  {shot['prompt'][:58]}",
              flush=True)
    return paths


# ---------------------------------------------------------------------------
# 3. Camera: a 16:9 window over one image, animated. No blending of any kind.
# ---------------------------------------------------------------------------

CAMERAS = {
    "hold":      ((0.00, 0.00), (0.00, 0.00), 1.00),
    "push_in":   ((0.00, 0.00), (0.00, 0.00), 0.78),
    "pull_out":  ((0.00, 0.00), (0.00, 0.00), 1.28),
    "pan_left":  ((0.10, 0.00), (-0.10, 0.00), 0.88),
    "pan_right": ((-0.10, 0.00), (0.10, 0.00), 0.88),
}


def camera_window(kind: str, t: float):
    """Normalized 16:9 crop rect at progress t, inside the square source."""
    start, end, zoom = CAMERAS.get(kind, CAMERAS["hold"])
    e = Easing.SMOOTH.sample(t) if kind != "hold" else 0.0
    dx = start[0] + (end[0] - start[0]) * e
    dy = start[1] + (end[1] - start[1]) * e
    z = 1.0 + (zoom - 1.0) * e if kind in ("push_in", "pull_out") else zoom

    base_h = 1.0 / z
    base_w = base_h * (16.0 / 9.0)
    if base_w > 1.0:
        base_w = 1.0
        base_h = base_w * 9.0 / 16.0
    x0 = (1.0 - base_w) / 2.0 + dx * base_w
    y0 = (1.0 - base_h) / 2.0 + dy * base_h
    x0 = min(max(x0, 0.0), 1.0 - base_w)
    y0 = min(max(y0, 0.0), 1.0 - base_h)
    return x0, y0, base_w, base_h


def render_frame(src: Image.Image, kind: str, t: float) -> Image.Image:
    W, H = src.size
    x0, y0, bw, bh = camera_window(kind, t)
    box = (int(x0 * W), int(y0 * H), int((x0 + bw) * W), int((y0 + bh) * H))
    box = (box[0], box[1], max(box[2], box[0] + 8), max(box[3], box[1] + 8))
    return src.crop(box).resize((OUT_W, OUT_H), Image.LANCZOS)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="I dove into nature and it accepted me")
    ap.add_argument("--out", default="output/film")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(12)

    print(f"\nCONCEPT: {args.concept}\n")
    print("=== STORY (ollama) ===", flush=True)
    plan_data = direct(args.concept, args.model)
    shots = plan_data["shots"][: args.shots]
    print(f"  title    : {plan_data['title']}")
    print(f"  narration: {plan_data['narration']}")
    for i, s in enumerate(shots):
        print(f"    {i+1}. {s['seconds']:.1f}s {s['camera']:9s} {s['prompt'][:70]}")
    if not shots:
        print("no shots returned; aborting")
        return

    print("\n=== IMAGES (tiny-sd) ===", flush=True)
    paths = generate_images(shots, out / "shots")

    # ---- exposure sheet: one drawing per shot, held for its duration ----
    poses = []
    for i, (shot, path) in enumerate(zip(shots, paths)):
        poses.append(KeyPose(f"shot{i}", AnimationState.of(index=i),
                             ease_into=Easing.SNAP,
                             hold=max(1, round(shot["seconds"] * FPS))))

    motion = MotionPlan(tuple(poses))
    sheet = ExposureSheet(sheet_id="film", fps=FPS)
    for i, path in enumerate(paths):
        sheet.add_drawing(Drawing(id=f"shot{i}", source=DrawingSource.AUTHORED, note=path))
    sheet.add_layer(Layer(name="scene", track=motion.to_track()))
    sheet.validate()

    print("\n=== EXPOSURE SHEET ===")
    print(f"  {sheet.total_frames} frames @ {FPS}fps = "
          f"{sheet.duration_seconds:.1f}s from {len(paths)} images")
    print(f"  on-screen images: {len(paths) / sheet.duration_seconds:.2f}/sec "
          f"(vs {FPS}/sec if every frame were new)")
    print(sheet.to_ascii(max_frames=min(sheet.total_frames, 26)))

    # ---- render: every frame is a crop of exactly one image ----
    print("\n=== RENDER (no blending) ===", flush=True)
    sources = {f"shot{i}": Image.open(p).convert("RGB") for i, p in enumerate(paths)}
    cam_by_drawing = {f"shot{i}": shots[i]["camera"] for i in range(len(shots))}

    segments = sheet.layer("scene").track.segments()
    frames = []
    for drawing_id, _start, length in segments:
        src = sources[drawing_id]
        kind = cam_by_drawing[drawing_id]
        for k in range(length):
            t = k / max(1, length - 1)
            frames.append(np.array(render_frame(src, kind, t)))
        print(f"  {drawing_id}: held {length} frames, camera={kind}", flush=True)

    mp4 = out / "film.mp4"
    try:
        from moviepy.editor import ImageSequenceClip
        ImageSequenceClip(frames, fps=FPS).write_videofile(
            str(mp4), fps=FPS, codec="libx264", audio=False, logger=None)
        print(f"\nwrote {mp4}", flush=True)
    except Exception as exc:
        print(f"mp4 failed: {exc}", flush=True)

    gif_frames = [Image.fromarray(f) for f in frames[::2]]
    gif_frames[0].save(out / "film.gif", save_all=True,
                       append_images=gif_frames[1:], duration=int(1000 / (FPS / 2)), loop=0)
    print(f"wrote {out/'film.gif'}", flush=True)

    (out / "shotlist.json").write_text(json.dumps(plan_data, indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
