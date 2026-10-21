"""
compare_video_models.py -- run several small video generators on the same prompt
and compare them on the same terms.

Every method gets: same prompt, same frame count, same seed.
Reported per method: wall time, s/frame, and frame-to-frame difference, which
distinguishes "something moved" from "the image flickered" or "nothing happened".

    python compare_video_models.py t2vzero
    python compare_video_models.py animatediff
    python compare_video_models.py svd
    python compare_video_models.py sheet
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
from PIL import Image, ImageDraw  # noqa: E402

torch.set_num_threads(12)

OUT = ROOT / "output" / "compare"
OUT.mkdir(parents=True, exist_ok=True)

PROMPT = ("a river flowing through a mossy forest, water moving over rocks, "
          "light shifting through the leaves, cinematic, filmic grain")
NEG = "blurry, static, still image, watermark, text, jitter"
FRAMES = 16
SIZE = 256
SEED = 7


def stats(frames):
    arr = [np.asarray(f.convert("RGB")).astype(np.float32) for f in frames]
    d = [float(np.abs(arr[i] - arr[i - 1]).mean()) for i in range(1, len(arr))]
    # temporal noise: how much does each frame differ from the MEAN frame?
    mean = np.mean(np.stack(arr), axis=0)
    dev = [float(np.abs(a - mean).mean()) for a in arr]
    return {
        "frame_diff_mean": round(float(np.mean(d)), 2),
        "frame_diff_min": round(float(np.min(d)), 2),
        "frame_diff_max": round(float(np.max(d)), 2),
        "temporal_dev_mean": round(float(np.mean(dev)), 2),
        "frames": len(frames),
        "size": list(frames[0].size),
    }


def save(name, frames, seconds, extra=None):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        f.convert("RGB").save(d / f"f{i:03d}.png")
    gif = d / "clip.gif"
    frames[0].convert("RGB").save(
        gif, save_all=True, append_images=[f.convert("RGB") for f in frames[1:]],
        duration=125, loop=0)
    s = stats(frames)
    s["seconds"] = round(seconds, 1)
    s["seconds_per_frame"] = round(seconds / len(frames), 2)
    s["prompt"] = PROMPT
    if extra:
        s.update(extra)
    (d / "stats.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(f"\n  --- {name} ---")
    for k, v in s.items():
        print(f"    {k}: {v}")
    return s


# ---------------------------------------------------------------------------


def _to_pil(f):
    """decode_latents returns NHWC float in [0,1]; PIL needs uint8."""
    if isinstance(f, Image.Image):
        return f
    a = np.asarray(f)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return Image.fromarray(a)


def run_t2vzero():
    """SD1.5 + CLIP with cross-frame attention. No new weights.

    NOTE on what its motion IS: the pipeline denoises frame 1, then translates
    that latent by a motion field and denoises all frames together. So the motion
    is a generated camera pan, not subject motion. t0/t1 are indices into the
    timestep schedule (defaults 44/47 of 50), so they must be scaled to the step
    count or the pipeline indexes out of bounds.
    """
    from diffusers import TextToVideoZeroPipeline

    # 512, not 256: SD1.5 produces tiled green banding at 256 (verified directly),
    # which is what made the first T2V-Zero run look like mush.
    size = 512
    frames = 8
    steps = 20
    t0 = int(44 / 50 * steps)
    t1 = int(47 / 50 * steps)
    print(f"=== Text2Video-Zero (SD1.5 + CLIP, no extra weights) "
          f"{frames}f @ {size}px steps={steps} t0={t0} t1={t1} motion_field=6 ===")
    t0_time = time.time()
    pipe = TextToVideoZeroPipeline.from_pretrained(
        "models/sd15", torch_dtype=torch.float32, safety_checker=None, variant="fp16")
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    print(f"  loaded in {time.time()-t0_time:.0f}s")

    t0_time = time.time()
    with torch.no_grad():
        result = pipe(
            prompt=PROMPT, negative_prompt=NEG,
            video_length=frames, height=size, width=size,
            num_inference_steps=steps, guidance_scale=7.5,
            motion_field_strength_x=6, motion_field_strength_y=6,
            t0=t0, t1=t1,
            generator=torch.Generator("cpu").manual_seed(SEED),
        )
    dt = time.time() - t0_time
    # 0.32 returns TextToVideoPipelineOutput(images=...), older builds used frames=
    for attr in ("images", "frames"):
        if hasattr(result, attr):
            raw = getattr(result, attr)
            break
    else:
        raw = result[0] if isinstance(result, (list, tuple)) else result["images"]
    frames = [_to_pil(f) for f in raw]
    return save("t2vzero", frames, dt, extra={"steps": steps, "t0": t0, "t1": t1})


# ---------------------------------------------------------------------------


def run_animatediff():
    """AnimateDiff v3 + Lightning. Run at 512 because v3 is trained there."""
    from diffusers import AnimateDiffPipeline, EulerDiscreteScheduler, MotionAdapter
    from safetensors.torch import load_file

    size = 512
    steps = 4  # must match the Lightning checkpoint: 4step weights -> 4 steps
    print(f"=== AnimateDiff v3 + Lightning ({size}px, {steps} steps, "
          f"v3 native resolution) ===")
    t0 = time.time()
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

    # Lightning is a replacement state dict for the motion modules, not a LoRA
    sd = load_file("models/animatediff_lightning/"
                   "animatediff_lightning_4step_diffusers.safetensors")
    motion_keys = [k for k in sd if "motion_modules" in k]
    unet_keys = set(pipe.unet.state_dict().keys())
    overlap = [k for k in motion_keys if k in unet_keys]
    print(f"  lightning tensors: {len(sd)} ({len(motion_keys)} motion), "
          f"{len(overlap)} match the unet")
    missing, unexpected = pipe.unet.load_state_dict(sd, strict=False)
    print(f"  loaded; {len(unexpected)} unexpected, {len(missing)} unmatched "
          f"(base unet weights are expected to be unmatched)")
    print(f"  loaded in {time.time()-t0:.0f}s")

    t0 = time.time()
    with torch.no_grad():
        out = pipe(prompt=PROMPT, negative_prompt=NEG, num_frames=FRAMES,
                   guidance_scale=1.0, num_inference_steps=steps,
                   width=size, height=size,
                   generator=torch.Generator("cpu").manual_seed(SEED))
    dt = time.time() - t0
    return save("animatediff", out.frames[0], dt,
                extra={"size_px": size, "steps": steps})


# ---------------------------------------------------------------------------


def run_svd():
    """Image-to-video: animate one still. CLIP image encoder, no T5."""
    from diffusers import StableVideoDiffusionPipeline

    assert (ROOT / "models" / "svd_xt" / "unet").exists(), \
        "SVD not downloaded yet"

    key = ROOT / "output" / "compare" / "key_frame.png"
    if not key.exists():
        from diffusers import StableDiffusionPipeline
        print("  generating a shared key frame with SD1.5 ...")
        sp = StableDiffusionPipeline.from_pretrained(
            "models/sd15", torch_dtype=torch.float32, safety_checker=None,
            variant="fp16")
        sp.set_progress_bar_config(disable=True)
        with torch.no_grad():
            img = sp(PROMPT, num_inference_steps=25, guidance_scale=7.5,
                     height=512, width=512,
                     generator=torch.Generator("cpu").manual_seed(SEED)).images[0]
        img.save(key)
        del sp

    img = Image.open(key).convert("RGB").resize((512, 288))
    print("=== Stable Video Diffusion XT (image-to-video, no T5) ===")
    t0 = time.time()
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "models/svd_xt", torch_dtype=torch.float32, variant="fp16")
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    print(f"  loaded in {time.time()-t0:.0f}s")

    t0 = time.time()
    with torch.no_grad():
        out = pipe(img, num_frames=14, decode_chunk_size=2, height=288, width=512,
                   num_inference_steps=25, min_guidance_scale=1.0,
                   max_guidance_scale=3.0,
                   generator=torch.Generator("cpu").manual_seed(SEED))
    dt = time.time() - t0
    return save("svd", out.frames[0], dt, extra={"size_px": 512})


# ---------------------------------------------------------------------------


def build_sheet():
    dirs = [d for d in sorted(OUT.iterdir()) if d.is_dir() and (d / "stats.json").exists()]
    if not dirs:
        print("nothing to compare yet")
        return
    T = 150
    cols = FRAMES
    rows = len(dirs)
    canvas = Image.new("RGB", (cols * (T + 3) + 3, rows * (T + 22) + 3), (22, 22, 26))
    dr = ImageDraw.Draw(canvas)
    print(f"{'method':<14} {'sec':>7} {'s/frame':>8} {'framediff':>10} {'tempdev':>8}")
    for r, d in enumerate(dirs):
        s = json.loads((d / "stats.json").read_text())
        y = r * (T + 22) + 3
        dr.text((5, y), f"{d.name}   {s['seconds']}s   "
                        f"{s['seconds_per_frame']}s/frame   "
                        f"framediff {s['frame_diff_mean']}   "
                        f"tempdev {s.get('temporal_dev_mean','-')}",
                fill=(235, 235, 235))
        files = sorted(d.glob("f*.png"))
        for c, f in enumerate(files[:cols]):
            canvas.paste(Image.open(f).resize((T, T), Image.LANCZOS),
                         (3 + c * (T + 3), y + 18))
        print(f"{d.name:<14} {s['seconds']:>7} {s['seconds_per_frame']:>8} "
              f"{s['frame_diff_mean']:>10} {s.get('temporal_dev_mean','-'):>8}")
    canvas.save(OUT / "comparison.png")
    print(f"\nwrote {OUT/'comparison.png'}")


METHODS = {"t2vzero": run_t2vzero, "animatediff": run_animatediff,
           "svd": run_svd, "sheet": build_sheet}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=list(METHODS))
    a = ap.parse_args()
    METHODS[a.method]()
    print("\nDONE")
