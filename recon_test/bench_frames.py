"""12 raw frames vs 16: the only remaining compute lever.

`make_film.FRAMES = 16` and the AnimateDiff motion adapter was trained at 16. The
question is not "is 12 faster" -- it is 25% less diffusion work, so of course it is --
but whether the motion survives. HANDOFF 12.1 rejected 8 frames on exactly this
ground (422s vs 552s, only 24% faster, and align fell 0.368 -> 0.313), which is why
12 is the untested point rather than 8.

Same prompt, same seed, same camera LoRA, ONE loaded pipeline for both runs -- so the
difference is the frame count and nothing else, and no model load is paid twice.

    python recon_test/bench_frames.py [--frames 16,12] [--prompt "..."]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch_home"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_home"))

DEFAULT_PROMPT = ("a woman in a long coat walking through a rain-slick alley at "
                  "night, neon signs reflected in puddles, heavy rain")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="16,12")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=1103)
    ap.add_argument("--lora", default="zoom-in",
                    help="a camera LoRA, so the motion module is actually exercised")
    ap.add_argument("--out", default="output/frames_ab")
    a = ap.parse_args()

    counts = [int(x) for x in a.frames.split(",")]
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    import make_film as mf
    from edit_planner import Shot
    from handheld import apply_handheld, hold_frames, spec_for_shot, stretch_to
    from motion_metrics import ClipScorer, temporal_stats

    shot = Shot(prompt=a.prompt, scale="medium", camera=a.lora, seconds=3.2)
    print(f"prompt: {a.prompt[:70]}...")
    print(f"camera LoRA: {a.lora}   seed {a.seed}   shot length {shot.seconds}s")
    print(f"the exposure sheet would hold this for {hold_frames(shot, mf.FPS)} "
          f"frames at {mf.FPS}fps\n")

    # ONE pipeline, loaded once, reused for every frame count. Two loads would put a
    # 57s constant into one arm of the comparison for no reason.
    gen = mf.Generator("models/dreamshaper8", loras=None, verify=True)
    gen.set_motion(a.lora)

    print("\nloading CLIP once for the quality metrics")
    scorer = ClipScorer()

    results = []
    original_frames = mf.FRAMES
    for n in counts:
        mf.FRAMES = n
        print(f"\n=== {n} raw frames ===", flush=True)
        t0 = time.time()
        frames, reported = gen.clip(shot, a.seed)
        t_gen = time.time() - t0
        print(f"  diffusion: {t_gen:.1f}s for {len(frames)} frames "
              f"({t_gen / len(frames):.1f}s/frame)")

        t0 = time.time()
        smooth = mf.raft_upsample(frames)
        t_raft = time.time() - t0
        print(f"  RAFT x2:   {t_raft:.1f}s -> {len(smooth)} frames")

        t0 = time.time()
        delivered = stretch_to(smooth, hold_frames(shot, mf.FPS))
        spec = spec_for_shot(shot.scale, "none", shot.seconds, seed=a.seed,
                             fps=mf.FPS)
        delivered = apply_handheld(delivered, spec, fps=mf.FPS)
        t_finish = time.time() - t0
        print(f"  finish:    {t_finish:.1f}s -> {len(delivered)} delivered frames")

        # ---- SAVE FIRST, MEASURE SECOND ------------------------------------- #
        # The first version of this script computed the metrics and printed them
        # before saving anything, then died on a wrong dict key -- and threw away
        # 653 seconds of generated frames. Generated frames are the expensive thing;
        # anything that can fail goes after they are on disk.
        d = out / f"raw{n:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            f.save(d / f"f{i:03d}.png")
        print(f"  saved {len(frames)} raw frames to {d}")

        # ---- the quality measurements ---------------------------------------- #
        try:
            motion = temporal_stats(frames)
            sc = scorer.scores(frames, a.prompt)
        except Exception as exc:
            print(f"  MEASUREMENT FAILED ({type(exc).__name__}: {exc}) -- the frames "
                  f"are on disk, so this is recoverable without re-generating")
            raise
        # ClipScorer returns {"coherence", "align"}: `coherence` IS the drift
        # (1 - cos to the first frame), which is why the handoff reports "drift".
        drift = sc.get("coherence")
        align = sc.get("align")
        print(f"  motion: mag {motion['mag']}  centroid_travel "
              f"{motion['centroid_travel']}px  loc10 {motion['loc10']:.3f}  "
              f"global_share {motion['global_share']}")
        print(f"  CLIP:   alignment {align:.3f}   coherence/drift {drift:.4f}")

        # a contact sheet: 4 frames evenly spaced, so the motion is visible in a still
        from PIL import Image, ImageDraw
        picks = [frames[round(i * (len(frames) - 1) / 3)] for i in range(4)]
        T = 256
        sheet = Image.new("RGB", (4 * (T + 4) + 4, T + 30), (18, 18, 22))
        dr = ImageDraw.Draw(sheet)
        dr.text((6, 5), f"{len(frames)} raw frames -> {len(delivered)} delivered  |  "
                        f"diffusion {t_gen:.0f}s  |  align {align:.3f}  "
                        f"coherence {drift:.4f}  |  mag {motion['mag']}",
                fill=(235, 235, 235))
        for k, f in enumerate(picks):
            sheet.paste(f.resize((T, T), Image.LANCZOS), (4 + k * (T + 4), 24))
        sheet.save(out / f"sheet_{n:02d}.png")

        results.append({
            "frames": n, "diffusion_s": round(t_gen, 1),
            "raft_s": round(t_raft, 1), "finish_s": round(t_finish, 1),
            "total_s": round(t_gen + t_raft + t_finish, 1),
            "per_frame_s": round(t_gen / len(frames), 1),
            "raft_frames": len(smooth), "delivered": len(delivered),
            **{k: motion[k] for k in ("mag", "global_share", "centroid_travel",
                                      "loc10", "loc25")},
            "align": align, "coherence": drift,
        })
        # write results after EVERY arm, so a crash in the second does not discard
        # the first arm's numbers
        (out / "results.json").write_text(json.dumps(results, indent=2),
                                          encoding="utf-8")
    print("\n" + "=" * 78)
    hdr = (f"{'raw':>4} {'diff s':>8} {'total s':>8} {'s/frame':>8} "
           f"{'raft':>5} {'mag':>7} {'travel':>7} {'align':>7} {'coher':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['frames']:>4} {r['diffusion_s']:>8} {r['total_s']:>8} "
              f"{r['per_frame_s']:>8} {r['raft_frames']:>5} {r['mag']:>7} "
              f"{r['centroid_travel']:>7} {r['align']:>7} {r['coherence']:>8}")
    if len(results) >= 2:
        base, alt = results[0], results[1]
        print(f"\n  {alt['frames']} vs {base['frames']} raw frames:")
        print(f"    diffusion {base['diffusion_s']:.0f}s -> "
              f"{alt['diffusion_s']:.0f}s  "
              f"({(1 - alt['diffusion_s'] / base['diffusion_s']) * 100:.0f}% faster)")
        print(f"    total     {base['total_s']:.0f}s -> {alt['total_s']:.0f}s  "
              f"({(1 - alt['total_s'] / base['total_s']) * 100:.0f}% faster)")
        print(f"    motion mag {base['mag']} -> {alt['mag']}   "
              f"travel {base['centroid_travel']} -> {alt['centroid_travel']}")
        print(f"    alignment {base['align']:.3f} -> {alt['align']:.3f}   "
              f"coherence {base['coherence']:.4f} -> {alt['coherence']:.4f}")
        print(f"\n  contact sheets: {out}/sheet_{base['frames']:02d}.png and "
              f"sheet_{alt['frames']:02d}.png -- LOOK AT THEM. A metric cannot tell "
              f"you whether the motion reads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
