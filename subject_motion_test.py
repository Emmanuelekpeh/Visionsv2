"""
subject_motion_test.py -- do the subject-motion LoRAs give AnimateDiff a moving
*subject*, or just more global drift?

This is the one untested lead in the handoff. AnimateDiff's measured weakness:

    top 5% of pixels hold 7.7% of the temporal variance, top 10% hold 14.2%

i.e. near-uniform. That is what disproved fg/background compositing. If a
`walking-frontview` LoRA concentrates the motion on a figure, the compositing
idea comes back and the film gains a real subject.

Three stages, in increasing cost:

  --refs     measure the LoRA author's own candidate renders. Free, and it shows
             what the LoRA is *capable* of before this machine is involved.
  --run      generate the A/B set here. ~9.5 min per clip, resumable; metrics are
             flushed to disk after every clip so a session can die safely.
  --report   build the comparison sheet from whatever is on disk.

Every clip is generated with the SAME prompt/seed/steps except for the LoRA, so
the difference is the LoRA and nothing else -- per the handoff, four false
verdicts in one session came from comparing misconfigured runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".ltx_env"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch_home"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_home"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import motion_loras as ml  # noqa: E402
from motion_metrics import ClipScorer, localisation_verdict, temporal_stats  # noqa: E402

torch.set_num_threads(12)

OUT = ROOT / "output" / "subject_test"
OUT.mkdir(parents=True, exist_ok=True)
METRICS = OUT / "metrics.json"
REFDIR = ROOT / "models" / "subject_lora" / "reference"

FRAMES = 16
STEPS = 4
GUIDANCE = 1.0        # Lightning is distilled for guidance-free sampling
LIGHTNING = True      # load the 4-step distillation weights into the UNet
SEED = 11
BASE = "models/dreamshaper8"
NEG = "blurry, static, still image, watermark, text, jitter, worst quality"

FIGURE = ("a lone woman in a long coat walking toward the camera down an empty "
          "road at dusk, cinematic lighting, filmic grain")
RUNNER = ("a runner sprinting toward the camera along a rain-soaked street at "
          "night, cinematic lighting, filmic grain")
DANCER = ("a ballerina spinning on a bare stage under a single spotlight, "
          "cinematic, filmic grain")
LEAF = ("dew on a mossy leaf, tiny water droplets, macro, cinematic, filmic grain")

# The fog experiment: does a LOW-DETAIL background let the walking LoRA produce
# a separable subject, where a textured street did not?
#
# Two clips, not one. Comparing fog+walk against the street baseline would change
# the scene and the LoRA at once, which is uninterpretable -- the confound this
# session has hit repeatedly. The control is the SAME fog prompt with no LoRA, so
# the pair isolates the LoRA inside one scene, and the existing street pair
# isolates the scene.
FOG_FIGURE = ("a lone woman in a long coat walking toward the camera on a foggy "
              "forest road, thick mist, no buildings, cinematic, filmic grain")

# the A/B set. same prompt+seed within each group, only the LoRA changes.
CLIPS = [
    {"id": "fig_none_512",     "lora": None,                 "size": 512, "prompt": FIGURE,  "group": "figure",  "why": "baseline"},
    {"id": "fig_walk_512",     "lora": "walking-frontview",  "size": 512, "prompt": FIGURE,  "group": "figure",  "why": "the A/B"},
    {"id": "fig_walk_576x384", "lora": "walking-frontview",  "size": "576x384", "prompt": FIGURE, "group": "figure", "why": "native train aspect"},
    {"id": "fig_run_512",      "lora": "running-frontview",  "size": 512, "prompt": RUNNER,  "group": "runner",  "why": "second subject LoRA"},
    {"id": "fig_spin_512",     "lora": "ballerina-spin",     "size": 512, "prompt": DANCER,  "group": "dancer",  "why": "in-place subject motion"},
    {"id": "leaf_walk_512",    "lora": "walking-frontview",  "size": 512, "prompt": LEAF,    "group": "misfit",  "why": "does a walking LoRA inject a person into a leaf macro?"},
    {"id": "fog_none_512",     "lora": None,                 "size": 512, "prompt": FOG_FIGURE, "group": "fog", "why": "fog control: low-detail background, no LoRA"},
    {"id": "fog_walk_512",     "lora": "walking-frontview",  "size": 512, "prompt": FOG_FIGURE, "group": "fog", "why": "fog + walking LoRA: is the subject separable here?"},
    # The 4-step Lightning confound. Everything above ran at 4 steps with the
    # distillation weights and guidance 1.0, which may under-express an adapter
    # trained on a full-step schedule. These two repeat the fog pair without
    # Lightning. BOTH are needed: full-step sampling changes the no-LoRA baseline
    # too (more coherent sampling drifts less), so the LoRA effect has to be read
    # as a within-configuration contrast, not against the 4-step numbers.
    {"id": "fog_none_full_512", "lora": None,                "size": 512, "prompt": FOG_FIGURE, "group": "fog-full", "why": "full-step control (no Lightning)"},
    {"id": "fog_walk_full_512", "lora": "walking-frontview", "size": 512, "prompt": FOG_FIGURE, "group": "fog-full", "why": "full-step + walking LoRA: does more sampling express the adapter?"},

    # --- the tests that decide what to build next --------------------------- #
    # SEED VARIANCE. Every number in this project is n=1. Without an error bar
    # there is no way to know whether a difference of 0.02 in global_share, or
    # 0.003 in align, means anything at all. Seeds 12 and 13 complete a 3-seed
    # sample at the winning configuration.
    {"id": "fog_none_s12", "lora": None,                "size": 512, "prompt": FOG_FIGURE, "group": "seedvar", "seed": 12, "why": "seed variance (no LoRA)"},
    {"id": "fog_none_s13", "lora": None,                "size": 512, "prompt": FOG_FIGURE, "group": "seedvar", "seed": 13, "why": "seed variance (no LoRA)"},
    {"id": "fog_walk_s12", "lora": "walking-frontview", "size": 512, "prompt": FOG_FIGURE, "group": "seedvar", "seed": 12, "why": "seed variance (walking)"},
    {"id": "fog_walk_s13", "lora": "walking-frontview", "size": 512, "prompt": FOG_FIGURE, "group": "seedvar", "seed": 13, "why": "seed variance (walking)"},
    # FRAME COUNT. AnimateDiff's motion module is trained on a 16-frame context,
    # so 8 frames may break the motion. If it does not, render time halves --
    # and with RAFT x2 doubling the frames afterwards, the delivered clip is
    # still 16 frames long.
    {"id": "fog_walk_8frame", "lora": "walking-frontview", "size": 512, "prompt": FOG_FIGURE, "group": "frames", "frames": 8, "why": "8 raw frames: does motion survive? halves render time"},
    # STEP SWEEP. We have 4 (Lightning) and 12 (no Lightning). This is the
    # middle point that says whether the 4-step result is the optimum or just
    # the fastest of two bad options.
    {"id": "fog_walk_s8", "lora": "walking-frontview", "size": 512, "prompt": FOG_FIGURE, "group": "steps", "steps": 8, "guidance": 1.0, "why": "8-step Lightning: the knee of the quality/cost curve"},
]


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #

class Generator:
    """AnimateDiff + DreamShaper 8 + Lightning 4-step, with a swappable LoRA.

    Subject LoRAs are loaded through diffusers after key conversion; their
    feed-forward halves are attached by hand because diffusers has no slot for
    them (see motion_loras.py). Both halves are counted and reported, so a null
    result can be told apart from a failed load.
    """

    def __init__(self, base: str = BASE, loras=("walking-frontview",
                                                "running-frontview",
                                                "ballerina-spin", "zoom-in")):
        from diffusers import (AnimateDiffPipeline, EulerDiscreteScheduler,
                               MotionAdapter)
        from safetensors.torch import load_file

        adapter = MotionAdapter.from_pretrained(
            "models/animatediff_motion_v3", torch_dtype=torch.float32, variant="fp16")
        kw = {"variant": "fp16"} if base == "models/sd15" else {}
        self.pipe = AnimateDiffPipeline.from_pretrained(
            base, motion_adapter=adapter, torch_dtype=torch.float32, **kw)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing")
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_slicing()
        sd = load_file("models/animatediff_lightning/"
                       "animatediff_lightning_4step_diffusers.safetensors")
        if LIGHTNING:
            self.pipe.unet.load_state_dict(sd, strict=False)
        else:
            print("  [sampling] Lightning weights NOT loaded: "
                  f"{STEPS} steps at guidance {GUIDANCE}")

        self.base_params = ml.count_unet_lora_params(self.pipe.unet)
        self.ff = ml.FFBank()
        self.adapters: list[str] = []
        self.effects: dict[str, float] = {}
        for name in loras:
            entry = ml.catalog().get(name)
            if not entry or not entry["ready"]:
                print(f"    lora {name}: MISSING")
                continue
            try:
                if entry["kind"] == "subject":
                    p = ml.prepare(name)
                    ff_file = entry["ff"]
                    if ff_file.exists():
                        n_ff, missing = self.ff.install(
                            self.pipe.unet, name, load_file(str(ff_file)))
                        print(f"    {name}: {n_ff} feed-forward blocks attached"
                              + (f", {len(missing)} unresolved" if missing else ""))
                else:
                    p = entry["path"]
                self.pipe.load_lora_weights(str(p), adapter_name=name)
                self.adapters.append(name)
            except Exception as exc:
                print(f"    lora {name} FAILED: {type(exc).__name__}: {str(exc)[:150]}")

        self.attn_params = ml.count_unet_lora_params(self.pipe.unet)
        print(f"  attention LoRA params in unet: {self.base_params} "
              f"-> {self.attn_params}  (+{self.attn_params - self.base_params})")
        print(f"  feed-forward LoRA tensors attached: {self.ff.num_tensors()}")
        print(f"  adapters: {self.adapters}")

        # ---- preflight: measure the configuration before trusting any result -- #
        print("  preflight -- effect of each adapter on a UNet forward pass:")
        dead = []
        for name in self.adapters:
            d = ml.verify_effect(self.pipe, self.ff, name)
            self.effects[name] = d
            print(f"    {name:<22} max|delta| = {d:.6f}"
                  + ("   <-- NOT IN THE GRAPH" if d == 0.0 else "")
                  + ("   (check failed)" if d < 0 else ""))
            if d == 0.0:
                dead.append(name)
        if dead:
            raise RuntimeError(
                f"these adapters had NO effect on the UNet: {dead}. "
                f"Generating with them would produce a false negative.")
        self.set_motion(None)

    def set_motion(self, name: str | None):
        if name and name in self.adapters:
            self.pipe.enable_lora()
            self.pipe.set_adapters([name], adapter_weights=[1.0])
        else:
            self.pipe.disable_lora()
        self.ff.set_active(name if name in self.ff.installed else None, 1.0)

    def clip(self, spec):
        self.set_motion(spec["lora"])
        w, h = (spec["size"], spec["size"]) if isinstance(spec["size"], int) \
            else map(int, spec["size"].split("x"))
        # per-clip sampling overrides, so a sweep runs in one process
        steps = spec.get("steps", STEPS)
        guidance = spec.get("guidance", GUIDANCE)
        frames = spec.get("frames", FRAMES)
        t0 = time.time()

        # per-step visibility. A blocked run previously burned 100 minutes at
        # 10% CPU with nothing in the log to show for it; with this on, a stall
        # is visible within one step instead of one clip.
        kwargs = {}
        if os.environ.get("SM_TEST_PROGRESS"):
            def _cb(pipe, step, timestep, cbkw):
                print(f"      step {step+1}/{steps}  {time.time()-t0:6.1f}s",
                      flush=True)
                return cbkw
            kwargs["callback_on_step_end"] = _cb

        with torch.no_grad():
            out = self.pipe(prompt=spec["prompt"], negative_prompt=NEG,
                            num_frames=frames, guidance_scale=guidance,
                            num_inference_steps=steps, width=w, height=h,
                            generator=torch.Generator("cpu").manual_seed(
                                spec.get("seed", SEED)), **kwargs)
        frames_out = [f.convert("RGB") for f in out.frames[0]]
        return frames_out, time.time() - t0


# --------------------------------------------------------------------------- #
# visuals
# --------------------------------------------------------------------------- #

def variance_heat(frames) -> Image.Image:
    """Where the clip's motion lives. A blob = a subject; edges everywhere = drift."""
    import cv2
    a = np.stack([np.asarray(f, np.float32) for f in frames])
    v = a.mean(axis=3).var(axis=0)
    v = (v / (v.max() + 1e-6) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(v, cv2.COLORMAP_INFERNO)
    return Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))


def strip(frames, n=4, w=200) -> Image.Image:
    picks = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    tiles = [frames[i].resize((w, w), Image.LANCZOS) for i in picks]
    out = Image.new("RGB", (n * w + (n - 1) * 3, w), (16, 16, 20))
    for i, t in enumerate(tiles):
        out.paste(t, (i * (w + 3), 0))
    return out


def sheet(rows: list[tuple[str, Image.Image, Image.Image]], path: Path, title: str):
    w = 800
    rowh = 200 + 22
    canvas = Image.new("RGB", (w + 8, 22 + len(rows) * rowh), (18, 18, 22))
    dr = ImageDraw.Draw(canvas)
    dr.text((6, 4), title, fill=(240, 240, 240))
    for r, (label, a, b) in enumerate(rows):
        y = 22 + r * rowh
        dr.text((6, y + 2), label, fill=(220, 220, 220))
        canvas.paste(a.resize((400, 200), Image.LANCZOS), (4, y + 18))
        canvas.paste(b.resize((400, 200), Image.LANCZOS), (404, y + 18))
    canvas.save(path)


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #

def load_metrics() -> dict:
    return json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}


def save_metrics(m: dict):
    METRICS.write_text(json.dumps(m, indent=2), encoding="utf-8")


def analyse(frames, prompt, scorer, tag: str) -> dict:
    d = dict(temporal_stats(frames))
    d["tag"] = tag
    if scorer is not None:
        d.update(scorer.scores(frames, prompt))
    return d


def stage_refs():
    """Measure the author's own candidate renders -- free evidence."""
    import cv2
    scorer = ClipScorer(str(ROOT / "models" / "clip-vit-base-patch32"))
    got = {}
    for f in sorted(REFDIR.glob("*.mp4")):
        cap = cv2.VideoCapture(str(f))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        cap.release()
        if not frames:
            continue
        m = analyse(frames, "", scorer, f"ref:{f.stem}")
        m["source"] = "author render"
        got[f.stem] = m
        d = OUT / "reference"
        d.mkdir(parents=True, exist_ok=True)
        variance_heat(frames).resize((400, 400)).save(d / f"{f.stem}_heat.png")
        strip(frames).save(d / f"{f.stem}_strip.png")
        print(f"  {f.stem:<28} loc5={m['loc5']:.3f} loc10={m['loc10']:.3f} "
              f"loc25={m['loc25']:.3f} mag={m['mag']:.2f} "
              f"spread={m['mask_spread']:.3f} travel={m['centroid_travel']:.1f}px "
              f"global={m['global_share']:.2f} {frames[0].size}")
    m = load_metrics()
    m["refs"] = got
    save_metrics(m)
    return got


def stage_run(only=None):
    specs = [c for c in CLIPS if not only or c["id"] in only]
    store = load_metrics()
    done = store.setdefault("clips", {})

    todo = [c for c in specs if c["id"] not in done]
    if not todo:
        print("  nothing to do -- all requested clips already measured")
        return done

    print(f"=== GENERATING {len(todo)} clip(s) ===")
    # Load only the adapters these clips actually need. Loading all four plus
    # 480 feed-forward wrappers for a `none` baseline wastes several hundred MB
    # on a machine with 15 GB, and memory pressure on this box has already cost
    # one session (see the handoff).
    needed = sorted({c["lora"] for c in todo if c["lora"]})
    print(f"  adapters required: {needed or ['(none)']}")
    gen = Generator(loras=needed)
    scorer = ClipScorer(str(ROOT / "models" / "clip-vit-base-patch32"))

    for i, spec in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {spec['id']}  lora={spec['lora']} "
              f"size={spec['size']}  ({spec['why']})", flush=True)
        d = OUT / spec["id"]
        d.mkdir(parents=True, exist_ok=True)
        try:
            frames, dt = gen.clip(spec)
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        for j, f in enumerate(frames):
            f.save(d / f"f{j:03d}.png")
        m = analyse(frames, spec["prompt"], scorer, spec["id"])
        m.update({"lora": spec["lora"], "size": spec["size"],
                  "prompt": spec["prompt"], "group": spec["group"],
                  "seconds": round(dt, 1), "why": spec["why"],
                  "steps": spec.get("steps", STEPS),
                  "guidance": spec.get("guidance", GUIDANCE),
                  "frames": spec.get("frames", FRAMES),
                  "seed": spec.get("seed", SEED),
                  "lightning": LIGHTNING})
        done[spec["id"]] = m
        save_metrics(store)                      # flush after EVERY clip
        variance_heat(frames).save(d / "heat.png")
        strip(frames).save(d / "strip.png")
        print(f"    {dt:.0f}s   loc5={m['loc5']:.3f} loc10={m['loc10']:.3f} "
              f"loc25={m['loc25']:.3f}  mag={m['mag']:.2f} "
              f"spread={m['mask_spread']:.3f} travel={m['centroid_travel']:.1f}px "
              f"global={m['global_share']:.2f} "
              f"coher={m.get('coherence','-')} align={m.get('align','-')}", flush=True)
    return done


def stage_report():
    store = load_metrics()
    clips = store.get("clips", {})
    refs = store.get("refs", {})
    if not clips and not refs:
        print("  no metrics yet")
        return

    print("\n" + "=" * 108)
    print("  LOC 5/10/25 = share of temporal variance in the top-k% of pixels "
          "(uniform would be 0.05/0.10/0.25)")
    print("  spread = radius of the busiest region / diagonal  (0.5 = whole frame)")
    print("=" * 108)
    print(f"  {'clip':<20} {'loc5':>6} {'loc10':>6} {'loc25':>6} {'mag':>6} "
          f"{'spread':>7} {'travel':>7} {'global':>7} {'coher':>7} {'align':>7}")
    print("  " + "-" * 104)

    base = clips.get("fig_none_512")
    for k in ["refs", "clips"]:
        src = refs if k == "refs" else clips
        for cid, m in src.items():
            print(f"  {cid[:20]:<20} {m['loc5']:>6.3f} {m['loc10']:>6.3f} "
                  f"{m['loc25']:>6.3f} {m['mag']:>6.2f} {m['mask_spread']:>7.3f} "
                  f"{m['centroid_travel']:>7.1f} {m['global_share']:>7.2f} "
                  f"{m.get('coherence', float('nan')):>7.4f} "
                  f"{m.get('align', float('nan')):>7.4f}")
        if k == "refs" and refs:
            print("  " + "-" * 104 + "   ^ author's own renders")

    if base:
        print("\n  --- verdicts vs this run's own no-LoRA baseline ---")
        for cid, m in clips.items():
            if cid == "fig_none_512":
                continue
            v = localisation_verdict(m, base)
            print(f"  {cid:<20} global x{v['global_ratio_vs_baseline']:<5} "
                  f"(loc10 x{v['loc10_ratio_vs_baseline']}, confounded by mag)  "
                  f"{v['verdict']}")
    else:
        print("\n  (baseline clip fig_none_512 not generated yet -- verdicts deferred)")

    # visual sheet
    rows = []
    for cid in clips:
        d = OUT / cid
        if (d / "strip.png").exists() and (d / "heat.png").exists():
            rows.append((f"{cid}  lora={clips[cid]['lora']}  "
                         f"loc10={clips[cid]['loc10']:.3f}",
                         Image.open(d / "strip.png"), Image.open(d / "heat.png")))
    if rows:
        sheet(rows, OUT / "comparison.png",
              "left: 4 frames across the clip   right: temporal-variance heat "
              "(bright = where motion lives)")
        print(f"\n  wrote {OUT/'comparison.png'}")
    print(f"  wrote {METRICS}")


def stage_recompute():
    """Re-derive metrics from frames already on disk. No generation.

    Metric definitions change; renders are expensive. This keeps the two
    decoupled so a fix to the measurement never costs ten minutes of CPU.
    """
    store = load_metrics()
    done = store.setdefault("clips", {})
    scorer = ClipScorer(str(ROOT / "models" / "clip-vit-base-patch32"))
    n = 0
    for spec in CLIPS:
        d = OUT / spec["id"]
        fs = sorted(d.glob("f*.png"))
        if not fs:
            continue
        frames = [Image.open(f).convert("RGB") for f in fs]
        m = analyse(frames, spec["prompt"], scorer, spec["id"])
        m.update({"lora": spec["lora"], "size": spec["size"],
                  "prompt": spec["prompt"], "group": spec["group"],
                  "why": spec["why"], "seconds": done.get(spec["id"], {}).get("seconds")})
        done[spec["id"]] = m
        n += 1
        print(f"  recomputed {spec['id']:<18} loc10={m['loc10']:.3f} "
              f"global={m['global_share']:.2f} spread={m['mask_spread']:.3f}")
    save_metrics(store)
    print(f"  {n} clip(s) recomputed")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", action="store_true", help="measure author renders")
    ap.add_argument("--preflight", action="store_true",
                    help="load everything and prove the LoRAs are live, then stop")
    ap.add_argument("--run", action="store_true", help="generate the A/B set")
    ap.add_argument("--recompute", action="store_true",
                    help="re-derive metrics from saved frames, no generation")
    ap.add_argument("--only", default="", help="comma-separated clip ids")
    ap.add_argument("--steps", type=int, default=None,
                    help="override the sampler step count (default 4)")
    ap.add_argument("--guidance", type=float, default=None,
                    help="override CFG scale (default 1.0, which is what the "
                         "Lightning distillation expects)")
    ap.add_argument("--no-lightning", action="store_true",
                    help="do NOT load the 4-step distillation weights. Pair this "
                         "with --steps and --guidance or you get an "
                         "un-distilled 4-step image, which is the 256px class of "
                         "misconfiguration this project keeps getting burned by")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    global STEPS, GUIDANCE, LIGHTNING
    if a.steps is not None:
        STEPS = a.steps
    if a.guidance is not None:
        GUIDANCE = a.guidance
    if a.no_lightning:
        LIGHTNING = False
        if a.steps is None or a.guidance is None:
            print("refusing to run: --no-lightning needs BOTH --steps and "
                  "--guidance, or this is a misconfigured test")
            return
    if not (a.refs or a.run or a.report or a.preflight or a.recompute):
        a.refs = a.report = True
    if a.preflight:
        print("=== PREFLIGHT: load the pipeline and verify every LoRA is live ===")
        Generator()
        print("\npreflight OK")
        return
    if a.refs:
        print("=== AUTHOR REFERENCE RENDERS (free evidence) ===")
        stage_refs()
    if a.recompute:
        print("=== RECOMPUTE metrics from saved frames ===")
        stage_recompute()
    if a.run:
        from keep_awake import keep_awake
        with keep_awake("subject-motion A/B"):
            stage_run([s for s in a.only.split(",") if s] or None)
    if a.report:
        stage_report()


if __name__ == "__main__":
    main()
