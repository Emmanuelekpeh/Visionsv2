"""
ltx_generate.py -- LTX-Video 2B distilled, quantized, on CPU.

    transformer  ltxv-2b-0.9.8-distilled-q8_0.gguf   2.17 GB
    text encoder t5xxl_fp16-q4_0.gguf                 2.90 GB
    vae          ltxv_0.9_vae_fp16.safetensors        0.84 GB

Needs a newer diffusers than the project's 0.28 (no LTX, no GGUF), so we prepend
an isolated install dir rather than upgrading in place.

    python ltx_generate.py --prompt "..." --seconds 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# isolated newer diffusers/transformers take precedence over the project's
sys.path.insert(0, str(ROOT / ".ltx_env"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_home"))

import torch  # noqa: E402

torch.set_num_threads(12)

GGUF_DIR = ROOT / "models" / "ltx_gguf"
CFG_DIR = ROOT / "models" / "ltx_video"
OUT = ROOT / "output" / "ltx"
OUT.mkdir(parents=True, exist_ok=True)


def probe():
    """Report what the environment can actually do before we try anything."""
    import diffusers
    import transformers

    print(f"diffusers    {diffusers.__version__}")
    print(f"transformers {transformers.__version__}")
    for name in ["LTXPipeline", "LTXVideoTransformer3DModel", "GGUFQuantizationConfig"]:
        print(f"  {name}: {hasattr(diffusers, name)}")
    try:
        from diffusers.models import AutoencoderKLLTXVideo  # noqa: F401
        print("  AutoencoderKLLTXVideo: True")
    except Exception as exc:
        print(f"  AutoencoderKLLTXVideo: False ({exc})")


def load_text_encoder():
    """Quantized T5-XXL. transformers GGUF support for T5 is not guaranteed."""
    from transformers import T5EncoderModel, T5Tokenizer

    tok = T5Tokenizer.from_pretrained(str(CFG_DIR / "tokenizer"))
    gguf = GGUF_DIR / "t5xxl_fp16-q4_0.gguf"

    # Try GGUF first (2.9 GB). If transformers cannot parse it, say so plainly.
    try:
        te = T5EncoderModel.from_pretrained(
            str(gguf), gguf_file=gguf.name, torch_dtype=torch.float32)
        print("  text encoder: GGUF T5 loaded")
        return te, tok
    except Exception as exc:
        print(f"  text encoder: GGUF path failed -> {type(exc).__name__}: "
              f"{str(exc)[:180]}")

    raise SystemExit(
        "\nCannot load a quantized T5-XXL with this transformers build, and the\n"
        "unquantized one is 19 GB (larger than this machine's RAM).\n"
        "Options: newer transformers with T5 GGUF support, or precompute the\n"
        "prompt embedding elsewhere and cache it."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a river flowing through a mossy forest, "
                                        "light shifting through leaves, filmic grain")
    ap.add_argument("--negative", default="worst quality, blurry, jitter, watermark")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q", default="q8_0", choices=["q8_0", "q4_k_m"])
    args = ap.parse_args()

    print("=== environment ===")
    probe()

    from diffusers import GGUFQuantizationConfig, LTXPipeline, LTXVideoTransformer3DModel
    from diffusers.utils import export_to_video

    n_frames = max(9, int(args.seconds * args.fps))
    # LTX wants (n-1) % 8 == 0
    n_frames = ((n_frames - 1) // 8) * 8 + 1
    print(f"\n=== plan: {n_frames} frames @ {args.fps}fps = "
          f"{n_frames/args.fps:.1f}s, {args.size}px, {args.steps} steps ===")

    print("\n=== loading transformer (GGUF) ===", flush=True)
    t0 = time.time()
    transformer = LTXVideoTransformer3DModel.from_single_file(
        str(GGUF_DIR / f"ltxv-2b-0.9.8-distilled-{args.q}.gguf"),
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float32),
        torch_dtype=torch.float32,
    )
    print(f"  transformer in {time.time()-t0:.0f}s", flush=True)

    print("=== loading text encoder ===", flush=True)
    t0 = time.time()
    text_encoder, tokenizer = load_text_encoder()
    print(f"  text encoder in {time.time()-t0:.0f}s", flush=True)

    print("=== loading vae ===", flush=True)
    from diffusers.models import AutoencoderKLLTXVideo

    vae = AutoencoderKLLTXVideo.from_single_file(
        str(GGUF_DIR / "ltxv_0.9_vae_fp16.safetensors"), torch_dtype=torch.float32)

    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(CFG_DIR / "scheduler"))

    pipe = LTXPipeline(
        transformer=transformer, text_encoder=text_encoder, tokenizer=tokenizer,
        vae=vae, scheduler=scheduler)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

    print(f"\n=== generating ===", flush=True)
    t0 = time.time()
    with torch.no_grad():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative,
            width=args.size,
            height=args.size,
            num_frames=n_frames,
            num_inference_steps=args.steps,
            guidance_scale=1.0,
            generator=torch.Generator("cpu").manual_seed(args.seed),
        )
    dt = time.time() - t0
    frames = result.frames[0]
    print(f"  {len(frames)} frames in {dt:.0f}s  ({dt/len(frames):.1f}s/frame)")
    print(f"  => {len(frames)/(dt/3600):.0f} frames/hour")

    for i, f in enumerate(frames):
        f.save(OUT / f"f{i:02d}.png")
    try:
        export_to_video(frames, str(OUT / "clip.mp4"), fps=args.fps)
        print(f"  wrote {OUT/'clip.mp4'}")
    except Exception as exc:
        print(f"  mp4 failed: {exc}")

    gif = OUT / "clip.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0)
    print(f"  wrote {gif}")

    import numpy as np

    arr = [np.asarray(f.convert("RGB")).astype(np.float32) for f in frames]
    diffs = [float(np.abs(arr[i] - arr[i - 1]).mean()) for i in range(1, len(arr))]
    print(f"\n  mean frame-to-frame difference: {np.mean(diffs):.2f}/255")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
