"""
make_composite.py -- AnimateDiff foreground over a held background, composed and timed.

The idea (from Scene Dynamics, expressed as compositing rather than training):
generate the moving shot, then HOLD the parts that should not move.

Implementation detail that matters: the held background is the clip's OWN first
frame. Compositing a separately generated still would not align with the clip.
Freezing the clip's own low-activity regions kills AnimateDiff's background
drift while guaranteeing pixel alignment.

Metrics, with a number rather than an opinion:
  frame_diff        motion magnitude (did anything move?)
  bg_drift          how much the held regions change (should go to ~0)
  clip_drift        MoCoGAN-style identity drift: mean pairwise cosine distance
                    between per-frame CLIP features. Lower = more consistent.
  clip_prompt       mean CLIP similarity of frames to the prompt

    python make_composite.py            # all prompts
    python make_composite.py river      # one
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".ltx_env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

torch.set_num_threads(12)

OUT = ROOT / "output" / "composite"
OUT.mkdir(parents=True, exist_ok=True)

FRAMES = 16
SIZE = 512
STEPS = 4
SEED = 7

PROMPTS = {
    "river": ("a river flowing through a mossy forest, water moving over rocks, "
              "light shifting through the leaves, cinematic, filmic grain"),
    "city": ("a busy city street at night in the rain, neon signs reflecting on "
             "wet asphalt, people with umbrellas walking, cinematic, filmic grain"),
    "grass": ("a field of tall grass swaying in the wind at golden hour, clouds "
              "drifting across the sky, cinematic, filmic grain"),
    "smoke": ("incense smoke curling upward in a dark room, a single shaft of "
              "window light, dust motes drifting, cinematic, filmic grain"),
}
NEG = "blurry, static, still image, watermark, text, jitter, worst quality"


# ---------------------------------------------------------------------------

def generate(prompt: str):
    from diffusers import AnimateDiffPipeline, EulerDiscreteScheduler, MotionAdapter
    from safetensors.torch import load_file

    adapter = MotionAdapter.from_pretrained(
        "models/animatediff_motion_v3", torch_dtype=torch.float32, variant="fp16")
    pipe = AnimateDiffPipeline.from_pretrained(
        "models/sd15", motion_adapter=adapter, torch_dtype=torch.float32,
        variant="fp16")
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    sd = load_file("models/animatediff_lightning/"
                   "animatediff_lightning_4step_diffusers.safetensors")
    pipe.unet.load_state_dict(sd, strict=False)

    t0 = time.time()
    with torch.no_grad():
        out = pipe(prompt=prompt, negative_prompt=NEG, num_frames=FRAMES,
                   guidance_scale=1.0, num_inference_steps=STEPS,
                   width=SIZE, height=SIZE,
                   generator=torch.Generator("cpu").manual_seed(SEED))
    dt = time.time() - t0
    del pipe, adapter
    return [f.convert("RGB") for f in out.frames[0]], dt


# ---------------------------------------------------------------------------

def activity_mask(frames, blur=9, percentile=55):
    """Where does the picture actually move? Temporal std, feathered."""
    arr = np.stack([np.asarray(f).astype(np.float32) for f in frames])  # N,H,W,3
    var = arr.std(axis=0).mean(axis=2)                                  # H,W
    v = var / (var.max() + 1e-6)
    # feather so compositing does not leave hard seams
    m = Image.fromarray((v * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(blur))
    m = np.asarray(m).astype(np.float32) / 255.0
    thr = np.percentile(m, percentile)
    m = np.clip((m - thr) / (m.max() - thr + 1e-6), 0.0, 1.0)
    m = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                   .filter(ImageFilter.GaussianBlur(blur // 2))).astype(np.float32) / 255.0
    return m[..., None]  # H,W,1


def composite(frames, mask):
    """Hold frame 0 wherever the picture is not moving."""
    base = np.asarray(frames[0]).astype(np.float32)
    out = []
    for f in frames:
        a = np.asarray(f).astype(np.float32)
        blend = base * (1.0 - mask) + a * mask
        out.append(Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8)))
    return out


# ---------------------------------------------------------------------------

_CLIP = {}


def clip_metrics(frames, prompt):
    """MoCoGAN-style identity drift + prompt alignment, using the project's CLIP."""
    if not _CLIP:
        from transformers import CLIPModel, CLIPProcessor
        _CLIP["m"] = CLIPModel.from_pretrained("models/clip-vit-base-patch32")
        _CLIP["p"] = CLIPProcessor.from_pretrained("models/clip-vit-base-patch32")
        _CLIP["m"].eval()
    model, proc = _CLIP["m"], _CLIP["p"]

    def feats(imgs):
        with torch.no_grad():
            inp = proc(images=imgs, return_tensors="pt")
            f = model.get_image_features(**inp)
            return f / f.norm(dim=-1, keepdim=True)

    F = feats(frames)
    sim = (F @ F.T).cpu().numpy()
    n = len(frames)
    off = sim[~np.eye(n, dtype=bool)]
    drift = float(1.0 - off.mean())

    with torch.no_grad():
        t = proc(text=[prompt], return_tensors="pt", padding=True, truncation=True)
        tf = model.get_text_features(**t)
        tf = tf / tf.norm(dim=-1, keepdim=True)
    align = float((F @ tf.T).mean().item())
    return round(drift, 4), round(align, 4)


def motion_stats(frames, mask=None):
    arr = [np.asarray(f).astype(np.float32) for f in frames]
    d = [float(np.abs(arr[i] - arr[i - 1]).mean()) for i in range(1, len(arr))]
    out = {"frame_diff": round(float(np.mean(d)), 2)}
    if mask is not None:
        m = mask[..., 0] < 0.15          # the "held" region
        base = arr[0]
        bg = [float(np.abs(a[m] - base[m]).mean()) for a in arr[1:]]
        out["bg_drift"] = round(float(np.mean(bg)), 3)
        out["held_pct"] = round(float(m.mean() * 100), 1)
    return out


def save(name, frames, meta):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(d / f"f{i:03d}.png")
    frames[0].save(d / "clip.gif", save_all=True, append_images=frames[1:],
                   duration=125, loop=0)
    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, str(d / "clip.mp4"), fps=8)
    except Exception as exc:
        print(f"    mp4 skipped: {exc}")
    (d / "stats.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run(name):
    prompt = PROMPTS[name]
    print(f"\n=== {name} ===")
    print(f"  prompt: {prompt}")
    frames, gen_s = generate(prompt)
    print(f"  {FRAMES} frames in {gen_s:.0f}s ({gen_s/FRAMES:.1f}s/frame)")

    mask = activity_mask(frames)
    comp = composite(frames, mask)

    raw_m = motion_stats(frames)
    comp_m = motion_stats(comp, mask)
    drift_r, align_r = clip_metrics(frames, prompt)
    drift_c, align_c = clip_metrics(comp, prompt)

    meta = {
        "prompt": prompt, "frames": FRAMES, "size": SIZE, "steps": STEPS,
        "generation_seconds": round(gen_s, 1),
        "raw": {**raw_m, "clip_drift": drift_r, "clip_prompt": align_r},
        "composited": {**comp_m, "clip_drift": drift_c, "clip_prompt": align_c},
    }
    save(name, frames, {**meta, "variant": "raw"})
    save(name + "_composed", comp, {**meta, "variant": "composited"})

    print(f"  RAW        motion {raw_m['frame_diff']:6.2f}  clip_drift {drift_r:.4f}  "
          f"prompt {align_r:.4f}")
    print(f"  COMPOSITED motion {comp_m['frame_diff']:6.2f}  clip_drift {drift_c:.4f}  "
          f"prompt {align_c:.4f}  bg_drift {comp_m.get('bg_drift','-')} "
          f"(held {comp_m.get('held_pct','-')}%)")
    return meta


def sheet():
    from PIL import Image as I
    names = [d.name for d in sorted(OUT.iterdir()) if d.is_dir()]
    if not names:
        return
    T = 130
    rows = len(names)
    canvas = I.new("RGB", (FRAMES * (T + 3) + 3, rows * (T + 20) + 3), (20, 20, 24))
    dr = ImageDraw.Draw(canvas)
    for r, n in enumerate(names):
        s = json.loads((OUT / n / "stats.json").read_text())
        y = r * (T + 20) + 3
        v = s["raw"] if s["variant"] == "raw" else s["composited"]
        dr.text((5, y), f"{n:<26} motion {v['frame_diff']:6.2f}   "
                        f"clip_drift {v['clip_drift']:.4f}   prompt {v['clip_prompt']:.4f}"
                        + (f"   bg_drift {v.get('bg_drift')}" if "bg_drift" in v else ""),
                fill=(235, 235, 235))
        files = sorted((OUT / n).glob("f*.png"))
        for c, f in enumerate(files[:FRAMES]):
            canvas.paste(I.open(f).resize((T, T), I.LANCZOS), (3 + c * (T + 3), y + 16))
    canvas.save(OUT / "all.png")
    print(f"\nwrote {OUT/'all.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="all")
    a = ap.parse_args()
    todo = list(PROMPTS) if a.name == "all" else [a.name]
    results = {}
    for n in todo:
        results[n] = run(n)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    sheet()
    print("\nDONE")
