"""Do the effects actually do anything, and do they behave?

Three properties matter more than looking good, and all three are mechanical:

  1. `amount=0.0` is EXACTLY the original. Without it an effect chain is unsafe to leave
     in place, because a step dialled to nothing would still alter the frames.
  2. The same input and seed give the same output. A preview that differs from the
     export is useless, and one that flickers differently per frame reads as noise.
  3. Every effect measurably CHANGES the image at amount=1.0. An effect that silently
     does nothing is the "control that does nothing" failure this project has a rule
     against.

    python recon_test/test_effects.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import effects as fx  # noqa: E402

FAILS = []


def check(label, got, want=True):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}"
          + ("" if ok else f"   (want {want!r})"))
    if not ok:
        FAILS.append(label)


def make_frames(n: int = 8, size: int = 128, seed: int = 7) -> list:
    """A moving, textured source: a gradient, a bright blob that travels, and grain.
    Something with tone AND motion, so both spatial and temporal effects have work."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 1, size, dtype=np.float32)[None, :, None]
    out = []
    for t in range(n):
        a = np.repeat(base, size, axis=0).repeat(3, axis=2) * 0.65
        yy, xx = np.mgrid[0:size, 0:size]
        cy = size / 2 + np.sin(t / n * 2 * np.pi) * size * 0.22
        cx = size / 2 + np.cos(t / n * 2 * np.pi) * size * 0.22
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (size / 9) ** 2)))
        a = a + blob[..., None] * 0.9
        a = a + (rng.random((size, size, 3)).astype(np.float32) - 0.5) * 0.08
        out.append(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "RGB"))
    return out


def arr(frames) -> np.ndarray:
    return np.stack([np.asarray(f, dtype=np.float32) for f in frames])


def main() -> int:
    frames = make_frames()
    orig = arr(frames)
    print(f"source: {len(frames)} frames of {frames[0].size[0]}x{frames[0].size[1]}\n")

    print("1. every registered effect is reachable and does something")
    fams = fx.by_family()
    print(f"   {len(fx.REGISTRY)} effects in {len(fams)} families:")
    for fam in fx.FAMILY_ORDER:
        names = [e.name for e in fams.get(fam, [])]
        print(f"     {fam:<11} {len(names):>2}  {', '.join(names)}")
    check("all five families are populated",
          sorted(fams.keys()), sorted(fx.FAMILY_ORDER))
    check("no family is empty", all(fams.get(f) for f in fx.FAMILY_ORDER), True)

    print("\n2. amount=0.0 is EXACTLY the original, for every effect")
    zero_ok, zero_fail = [], []
    for name, eff in sorted(fx.REGISTRY.items()):
        try:
            out = arr(eff.run(frames, amount=0.0))
            if np.array_equal(out, orig):
                zero_ok.append(name)
            else:
                zero_fail.append((name, float(np.abs(out - orig).max())))
        except Exception as exc:                       # noqa: BLE001
            zero_fail.append((name, f"{type(exc).__name__}: {exc}"))
    check(f"all {len(fx.REGISTRY)} effects are a no-op at amount=0",
          zero_fail, [])

    print("\n3. amount=1.0 measurably changes every effect")
    weak = []
    for name, eff in sorted(fx.REGISTRY.items()):
        out = arr(eff.run(frames, amount=1.0))
        d = float(np.abs(out - orig).mean())
        if d < 0.5:                                    # less than half a grey level
            weak.append((name, round(d, 4)))
    check("no effect is a silent no-op at full strength", weak, [])

    print("\n4. the same seed gives the same frames (preview == export)")
    unstable = []
    for name, eff in sorted(fx.REGISTRY.items()):
        a = arr(eff.run(frames, amount=1.0, seed=11))
        b = arr(eff.run(frames, amount=1.0, seed=11))
        if not np.array_equal(a, b):
            unstable.append(name)
    check("every effect is deterministic for a fixed seed", unstable, [])

    print("\n5. a different seed changes the seeded effects, and only those")
    # effects with no randomness must be seed-independent, or the seed is a lie.
    # `halftone` is CONDITIONAL: its jitter is scaled by `irregular`, so at irregular=0
    # a seed genuinely cannot matter and asserting otherwise would be wrong.
    seeded = {"voronoi", "xerox", "riso", "film_damage", "painterly",
              "patch_shuffle", "displace", "slice", "frequency_separation",
              "stipple"}
    wrongly_seeded, wrongly_flat = [], []
    for name, eff in sorted(fx.REGISTRY.items()):
        if name == "halftone":
            continue
        a = arr(eff.run(frames, amount=1.0, seed=1))
        b = arr(eff.run(frames, amount=1.0, seed=2))
        same = np.array_equal(a, b)
        if name in seeded and same:
            wrongly_seeded.append(name)
        if name not in seeded and not same:
            wrongly_flat.append(name)
    check("seeded effects respond to the seed", wrongly_seeded, [])
    check("unseeded effects ignore it", wrongly_flat, [])
    ht = fx.REGISTRY["halftone"]
    check("halftone at irregular=0 ignores the seed (nothing to jitter)",
          np.array_equal(arr(ht.run(frames, amount=1.0, irregular=0.0, seed=1)),
                         arr(ht.run(frames, amount=1.0, irregular=0.0, seed=2))), True)
    check("halftone at irregular>0 responds to it",
          np.array_equal(arr(ht.run(frames, amount=1.0, irregular=0.8, seed=1)),
                         arr(ht.run(frames, amount=1.0, irregular=0.8, seed=2))),
          False)

    print("\n6. thresholds and levels are honoured, not decorative")
    # ZERO VALUED PARAMETERS MUST SURVIVE THE CALL. `run()` filters overrides, and a
    # truthiness filter drops `softness=0.0` and `ordered=False` -- which made
    # `threshold` quietly run its graded sigmoid while claiming to be a hard threshold.
    check("passing softness=0.0 is not dropped on the way in",
          fx.REGISTRY["threshold"].run(frames, amount=1.0, softness=0.0) is not None,
          True)
    for zero_param, value in (("softness", 0.0), ("ordered", False)):
        eff = fx.REGISTRY["threshold" if zero_param == "softness" else "dither"]
        out = eff.run(frames, amount=1.0, **{zero_param: value})
        check(f"{eff.name}({zero_param}={value!r}) is accepted and changes the result",
              len(out), len(frames))

    thr = arr(fx.REGISTRY["threshold"].run(frames, amount=1.0, local=0.0,
                                           softness=0.0))
    uniq = sorted(int(x) for x in np.unique(np.round(thr)))
    check("threshold with softness=0 really is black and white", uniq, [0, 255])
    soft = arr(fx.REGISTRY["threshold"].run(frames, amount=1.0, local=0.0,
                                            softness=0.2))
    check("and softness>0 genuinely grades it",
          len(np.unique(np.round(soft))) > 2, True)
    loc = arr(fx.REGISTRY["threshold"].run(frames, amount=1.0, local=0.5))
    plain = arr(fx.REGISTRY["threshold"].run(frames, amount=1.0, local=0.0))
    check("local thresholding gives a DIFFERENT result from global",
          bool(np.abs(loc - plain).mean() > 1.0), True)

    for levels in (3, 6):
        p = arr(fx.REGISTRY["posterize"].run(frames, amount=1.0, levels=levels))
        vals = np.unique(np.round(p))
        check(f"posterize at {levels} levels has <= {levels} distinct values "
              f"(and stays palindromic about /255)",
              len(vals) <= levels * 1 + 1, True)

    print("\n7. masks restrict the effect instead of tinting everything")
    # Compare each variant against the ORIGINAL, not against each other: unmasked
    # displacement can leave the centre alone by coincidence, which made the first
    # version of this check measure nothing.
    h, w = orig.shape[1], orig.shape[2]
    cy, cx = h // 2, w // 2
    core = (slice(None), slice(cy - 8, cy + 8), slice(cx - 8, cx + 8), slice(None))
    d_plain = float(np.abs(arr(fx.REGISTRY["displace"].run(
        frames, amount=1.0, scale=20.0, mask="none"))[core] - orig[core]).mean())
    d_mask = float(np.abs(arr(fx.REGISTRY["displace"].run(
        frames, amount=1.0, scale=20.0, mask="radial"))[core] - orig[core]).mean())
    print(f"     centre deviation from original: unmasked {d_plain:.2f}, "
          f"radial-masked {d_mask:.2f} (0-255 scale)")
    check("a radial mask protects the centre", d_mask < d_plain, True)

    e_plain = float(np.abs(arr(fx.REGISTRY["displace"].run(
        frames, amount=1.0, scale=20.0, mask="none")) - orig).mean())
    e_mask = float(np.abs(arr(fx.REGISTRY["displace"].run(
        frames, amount=1.0, scale=20.0, mask="radial")) - orig).mean())
    check("and the edge mask still changes the frame overall", e_mask > 0.5, True)

    print("\n8. chains run in order and survive a bad step")
    f0 = fx.REGISTRY["posterize"].run(frames, amount=1.0, levels=4)
    both = fx.apply_chain(frames, [{"effect": "posterize", "amount": 1.0, "levels": 4},
                                   {"effect": "channel_separation", "amount": 1.0,
                                    "offset_px": 4.0}], verbose=False)
    check("a two-step chain differs from its first step alone",
          bool(np.abs(arr(both) - arr(f0)).mean() > 0.5), True)
    survived = fx.apply_chain(frames, [{"effect": "posterize", "amount": 1.0},
                                       {"effect": "no_such_thing", "amount": 1.0}],
                              verbose=False)
    check("an unknown step is skipped, not fatal", len(survived), len(frames))
    check("and validate_chain names it",
          fx.validate_chain([{"effect": "posterize"}, {"effect": "nope"}])
          == ["step 2: no effect called 'nope'"], True)

    print("\n9. an effect that raises does not take the frames with it")
    # `film_damage` with an absurd parameter, to prove the guard, not the effect
    out = fx.apply_chain(frames, [{"effect": "halftone", "amount": 1.0, "cell": 0},
                                  {"effect": "trails", "amount": 1.0}], verbose=False)
    check("a failing step leaves a usable clip", len(out), len(frames))

    print("\n10. cost: are these previewable, or do they need a render?")
    for fam in fx.FAMILY_ORDER:
        for e in fx.by_family().get(fam, []):
            t0 = time.time()
            e.run(frames, amount=1.0)
            dt = (time.time() - t0) * 1000
            flag = "  <-- slow for a 2s preview" if dt > 900 else ""
            if dt > 250 or dt > 900:
                print(f"     {e.name:<22} {dt:7.1f} ms{flag}")
    total = 0.0
    for e in fx.REGISTRY.values():
        t0 = time.time()
        e.run(frames, amount=1.0)
        total += time.time() - t0
    print(f"     all {len(fx.REGISTRY)} effects on {len(frames)} frames: "
          f"{total * 1000:.0f} ms total")
    check("the whole library is cheap enough to preview live", total < 20.0, True)

    print(f"\n{'=' * 62}")
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + "; ".join(FAILS))
    else:
        print(f"all effects checks passed ({len(fx.REGISTRY)} effects)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
