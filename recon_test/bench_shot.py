"""Where does a shot's time actually go?

Splits one shot's wall clock into the parts, because "it takes forever" is not
actionable and "the fixed per-job cost is N minutes of the M" is.

  fixed part   pipeline load + motion-LoRA install + the preflight verification
               (28 UNet passes: 2 per LoRA x 14 LoRAs)
  variable     the 4-step diffusion itself, the VAE decode, RAFT, shake, grading

Only the variable part scales with shot count, so only the fixed part is worth
engineering away. HANDOFF 12.1 already measured the frame-count dial (8 frames = 422s
vs 16 = 552s, rejected: the fixed cost dominates), so this measures the OTHER half.

    python recon_test/bench_shot.py [--no-load] [--frames N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch_home"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_home"))


def mb(n: int) -> str:
    return f"{n / 1e6:.0f} MB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--no-load", action="store_true",
                    help="skip the load measurement (it is the slow one)")
    ap.add_argument("--generate", action="store_true",
                    help="also run one real clip, to get the variable half")
    ap.add_argument("--base", default="models/dreamshaper8")
    a = ap.parse_args()

    if a.frames:
        import make_film as mf
        mf.FRAMES = a.frames
        print(f"frames overridden to {a.frames}")

    import make_film as mf
    from edit_planner import Shot

    # ---- the fixed cost: load + LoRA install, then the preflight on top -------- #
    if not a.no_load:
        print("\n=== FIXED COST: pipeline load + 14 motion LoRAs ===", flush=True)
        t0 = time.time()
        gen = mf.Generator(a.base, loras=None, verify=False)
        t_load = time.time() - t0
        print(f"  load without preflight: {t_load:.1f}s   "
              f"adapters available: {len(gen.adapters)}")

        print(f"\n=== FIXED COST: the preflight ({len(gen.adapters)} LoRAs x 2 "
              f"UNet passes) ===", flush=True)
        t0 = time.time()
        effects = {}
        for m in gen.adapters:
            import motion_loras as ml
            effects[m] = ml.verify_effect(gen.pipe, gen.ff, m)
        t_verify = time.time() - t0
        print(f"  preflight: {t_verify:.1f}s   "
              f"({t_verify / max(1, len(gen.adapters)):.1f}s per LoRA)")
        dead = [k for k, v in effects.items() if v == 0.0]
        print(f"  dead adapters (would refuse to generate): {dead or 'none'}")
        gen.release()
        del gen
        print(f"\n  FIXED TOTAL PER JOB: {t_load + t_verify:.1f}s "
              f"({(t_load + t_verify) / 60:.1f} min)")
        print(f"  of which the model load is {t_load:.0f}s and the preflight "
              f"{t_verify:.0f}s")
        print(f"  -> across 6 separately-started shots that is "
              f"{(t_load + t_verify) * 6 / 60:.0f} min of pure overhead")

    # ---- the variable cost: one real clip ------------------------------------- #
    if a.generate:
        print("\n=== VARIABLE COST: one real clip ===", flush=True)
        gen = mf.Generator(a.base, loras=None, verify=False)
        shot = Shot(prompt="a quiet street at dusk, wet pavement, one lit window",
                    scale="wide", camera="none", seconds=2.5)
        t0 = time.time()
        frames, dt = gen.clip(shot, 1103)
        t_clip = time.time() - t0
        print(f"  clip(): {t_clip:.1f}s for {len(frames)} frames "
              f"({t_clip / max(1, len(frames)):.1f}s/frame)  [pipeline reported "
              f"{dt:.1f}s]")

        t0 = time.time()
        smooth = mf.raft_upsample(frames)
        t_raft = time.time() - t0
        print(f"  raft_upsample: {t_raft:.1f}s -> {len(smooth)} frames")

        from handheld import apply_handheld, hold_frames, spec_for_shot, stretch_to
        t0 = time.time()
        base = stretch_to(smooth, hold_frames(shot, mf.FPS))
        spec = spec_for_shot(shot.scale, "none", shot.seconds, seed=1103, fps=mf.FPS)
        base = apply_handheld(base, spec, fps=mf.FPS)
        t_shake = time.time() - t0
        print(f"  stretch + shake: {t_shake:.1f}s -> {len(base)} frames")

        print(f"\n  VARIABLE TOTAL PER SHOT: {t_clip + t_raft + t_shake:.1f}s")
        print(f"    diffusion {t_clip:.0f}s + RAFT {t_raft:.0f}s + "
              f"shake {t_shake:.0f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
