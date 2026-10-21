"""
make_film.py -- planner + AnimateDiff + camera LoRAs + RAFT -> an assembled film.

Pipeline
  1. shots are proposed (prompt, scale, camera, hold)
  2. edit_planner enforces continuity rules R1-R4 and assigns a rhythm
  3. each shot is generated with its camera motion LoRA (or none)
  4. each clip is RAFT-upsampled x2 for smoothness
  5. hard cuts, timed by the exposure sheet

    python make_film.py --test        # 2 shots, verifies the LoRA mechanism
    python make_film.py               # the full sequence
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".ltx_env"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch_home"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_home"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from edit_planner import Shot, describe, energy_profile, plan  # noqa: E402
from supplement import run_raft, to_image, to_tensor, warp  # noqa: E402
import motion_loras as ml  # noqa: E402
from handheld import apply_handheld, hold_frames, spec_for_shot, stretch_to  # noqa: E402

torch.set_num_threads(12)

OUT = ROOT / "output" / "film"
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 512
FRAMES = 16
STEPS = 4
SEED = 11
FPS = 16          # after RAFT x2 of an 8fps base
NEG = "blurry, static, still image, watermark, text, jitter, worst quality"

# Platform lengths, from 2026 short-form guidance.
#
# The completion curve is the thing to design against: a 15s video is watched to
# ~80-90% completion, a 45s one to ~50-65%, a 60s one to ~40-55%. Total watch
# time is completion x length, and the product peaks in the range below, so
# "long enough to be worth watching, short enough to finish" is the target.
#   TikTok   narrative 35-50s, most-liked cluster 21-34s  -> 30s
#   Reels    narrative 20-30s, shortest consistently wins -> 25s
#   Shorts   narrative 35-55s, longer pays off if earned  -> 45s
#
# SHOT COUNT IS THE BUDGET. Rendering is ~9-10 min per shot at 4-step Lightning
# and nothing else in the pipeline is close to that cost, so duration converts
# almost linearly into hours. `_rhythm_curve` averages ~2.5s per shot (it runs
# 3.2s down to 1.6s and back up to 3.6s), which gives the estimate below.
LENGTHS = {
    "tiktok": 30.0,
    "reels": 25.0,
    "shorts": 45.0,
    "custom": None,
}
MIN_PER_SHOT = 9.5      # measured, 4-step Lightning at 512x512, 16 frames


def budget_report(target: float) -> str:
    """Duration -> shots -> hours, before anyone commits to a render."""
    avg = 2.5
    n = max(1, round(target / avg))
    hrs = n * MIN_PER_SHOT / 60.0
    return (f"  target {target:.0f}s -> about {n} shots "
            f"(at the rhythm curve's ~{avg}s average)\n"
            f"  render budget: {n} x {MIN_PER_SHOT} min = "
            f"{n * MIN_PER_SHOT:.0f} min ({hrs:.1f} h)")

# A small arc: settle -> detail -> the subject -> through it -> resolve
SHOTS = [
    Shot("a vast dense forest at dawn, mist between the tall trees, "
         "light shafts, cinematic, filmic grain", "wide", "zoom-in", 3.0),
    Shot("dew on a mossy leaf, tiny water droplets, macro, "
         "cinematic, filmic grain", "macro", "none", 2.0),
    Shot("a waterfall in a mossy gorge, water falling, "
         "cinematic, filmic grain", "wide", "zoom-out", 2.0),
    Shot("mossy boulders with water running over them, close, "
         "cinematic, filmic grain", "close", "pan-left", 2.0),
    Shot("a wide river pool reflecting the sky, "
         "cinematic, filmic grain", "medium", "pan-right", 2.0),
    Shot("tall trees, canopy and sky above, "
         "cinematic, filmic grain", "wide", "tilt-up", 3.0),
]

# A short film built to exercise everything at once, and short enough to render
# in one sitting. Two scenes, so there is exactly one motivated boundary for the
# model switch and the palette change to land on.
#
# The subject shot is framed against FOG on purpose. The subject-motion A/B
# (handoff 9.7.1) showed the walking LoRA produces a coherent articulated figure,
# but that on a textured street the background does not hold, so the motion is not
# separable. Background stability is the precondition, so the test puts the figure
# in front of the one thing AnimateDiff holds well: fog and flat sky.
TEST_SHOTS = [
    Shot("a lone woman in a long coat walking toward the camera on a foggy "
         "forest road, thick mist, no buildings, cinematic, filmic grain",
         "wide", "none", 3.0, subject="woman", scene="walk", energy=0.55),
    Shot("fog drifting between tall bare trees, empty road, "
         "cinematic, filmic grain", "wide", "none", 2.5, scene="walk",
         energy=0.4),
    Shot("a waterfall in a mossy gorge, water falling, close, "
         "cinematic, filmic grain", "close", "none", 2.0, scene="water",
         energy=0.55),
    Shot("water running over mossy boulders, macro, "
         "cinematic, filmic grain", "macro", "none", 2.0, scene="water",
         energy=0.35),
]
FIGURE_SHOTS = [
    Shot("a lone woman in a long coat walking down an empty forest road at "
         "dawn, mist between the trees, wide shot, cinematic, filmic grain",
         "wide", "none", 3.0, subject="woman", energy=0.6),
    Shot("dew on a mossy leaf, tiny water droplets, macro, "
         "cinematic, filmic grain", "macro", "none", 2.0, energy=0.2),
    Shot("the woman walking on toward the camera through the mist, "
         "medium shot, cinematic, filmic grain",
         "medium", "none", 2.5, subject="woman", energy=0.55),
    Shot("she runs now, full body, urgent, medium shot, "
         "cinematic, filmic grain", "medium", "none", 2.0,
         subject="woman", energy=0.9),
    Shot("a waterfall in a mossy gorge, water falling, "
         "cinematic, filmic grain", "wide", "none", 2.0, energy=0.6),
    Shot("tall trees, canopy and sky above, "
         "cinematic, filmic grain", "wide", "none", 3.0, energy=0.7),
]


# The preflight result, keyed by adapter name and the sha1 of the weights it measured.
# In `output/film/` with the rest of the derived state -- it is a measurement, not a
# setting, and deleting it costs nothing but time.
VERIFY_CACHE = OUT / "preflight_cache.json"


def _sha1_path(path) -> str:
    """A content hash of a LoRA, whether it is a file or a directory.

    Most of the camera LoRAs are DIRECTORIES (`models/motion_lora/zoom-in/`, holding a
    diffusers adapter), and the subject ones are single `.safetensors` files. Hashing a
    directory path the naive way returns '' -- and then every directory-backed adapter
    would share one cache key, so one verification would silently stand in for
    thirteen others. The whole point is that the key is the exact bytes measured, so
    the walk has to be real.

    Returns '' only when the path cannot be read at all, and callers must treat '' as
    "uncacheable" rather than as a key.
    """
    import hashlib
    from pathlib import Path as _P
    try:
        p = _P(path)
        h = hashlib.sha1()
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if not child.is_file():
                    continue
                # the relative name matters: a renamed file with identical bytes is a
                # different adapter as far as loading is concerned
                h.update(str(child.relative_to(p)).encode("utf-8"))
                h.update(str(child.stat().st_size).encode("utf-8"))
                with open(child, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
        elif p.is_file():
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        else:
            return ""
        return h.hexdigest()
    except Exception:
        return ""

def _load_verify_cache() -> Dict:
    try:
        return json.loads(VERIFY_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_verify_cache(cache: Dict) -> None:
    try:
        VERIFY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        VERIFY_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


class Generator:
    """AnimateDiff renderer that can hot-swap the base checkpoint per shot.

    Two SD1.5-family checkpoints are supported. They are NOT held in memory at
    once: this box has 15.3 GB total and typically 3-8 GB free, and one fp32
    SD1.5 UNet plus its motion adapter is already several GB. So `ensure_base`
    builds on demand and releases the previous one -- about a minute per switch,
    which is nothing next to ten minutes of sampling, and a film only switches at
    scene boundaries.

    Two families of LoRA are handled (see motion_loras.py): the official camera
    moves, and the subject-motion LoRAs whose feed-forward half diffusers has no
    slot for. Everything loaded is *proven* live by a forward-pass difference
    before any generation happens -- four false verdicts in the handoff session
    came from trusting a configuration that had silently half-loaded.
    """

    # model_select.py names -> checkpoint directories on disk
    BASES = {"dreamshaper8": "models/dreamshaper8", "sd15": "models/sd15"}

    def __init__(self, base: str = "models/dreamshaper8", loras=None,
                 verify: bool = True):
        self.loras = loras            # None = everything ready on disk
        self.verify = verify
        self.base = None
        self.pipe = None
        self.ff = ml.FFBank()
        self.adapters = []
        self.effects = {}
        self.ensure_base(base)

    # -- checkpoint switching ------------------------------------------------ #
    def ensure_base(self, base: str):
        if base == self.base and self.pipe is not None:
            return
        if self.pipe is not None:
            print(f"  [base] releasing {self.base}, loading {base} (~1 min)")
            self.release()
        self._build(base)
        self.base = base

    def ensure_model(self, name: str):
        """Take a model_select name ('sd15') or a path ('models/sd15')."""
        path = self.BASES.get(name, name)
        if path not in self.BASES.values():
            raise ValueError(
                f"{name!r} is not renderable by this pipeline. Only the "
                f"AnimateDiff-family checkpoints work here: "
                f"{sorted(self.BASES)}. A different generator (t2v-zero, svd-xt, "
                f"sd15-lcm) needs its own pipeline and has not been wired.")
        self.ensure_base(path)

    def release(self):
        self.pipe = None
        self.ff = ml.FFBank()
        self.adapters = []
        import gc
        gc.collect()

    def _build(self, base: str):
        from diffusers import AnimateDiffPipeline, EulerDiscreteScheduler, MotionAdapter
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
        self.pipe.unet.load_state_dict(sd, strict=False)

        cat = ml.catalog()
        wanted = self.loras if self.loras is not None else [
            n for n in cat if cat[n]["kind"] != "none" and cat[n]["ready"]]

        for m in wanted:
            entry = cat.get(m)
            if not entry or not entry["ready"]:
                continue
            try:
                if entry["kind"] == "subject":
                    ml.prepare(m)
                    ff_file = entry["ff"]
                    if ff_file.exists():
                        self.ff.install(self.pipe.unet, m,
                                        load_file(str(ff_file)))
                self.pipe.load_lora_weights(str(entry["path"] if
                                                entry["kind"] == "camera"
                                                else ml.prepare(m)),
                                            adapter_name=m)
                self.adapters.append(m)
            except Exception as exc:
                print(f"    lora {m} failed: {type(exc).__name__}: {str(exc)[:90]}")
        print(f"  [base] {base}: motion LoRAs available: {self.adapters}")

        if self.verify:
            # THE PREFLIGHT IS CACHED AGAINST THE WEIGHTS, and that is most of the
            # speed-up available on this machine. Measured: the preflight is 179s of a
            # 236s fixed cost per job (28 UNet passes, ~12.8s each) -- three times the
            # model load, and 24 minutes of pure overhead across six separately-started
            # shots. It is worth paying ONCE per set of LoRA weights. The weights are
            # deterministic files on disk, so their sha1 is the whole condition: if it
            # has not changed, neither has the answer.
            #
            # The check is NOT dropped, because the thing it protects against is real
            # (HANDOFF 9.4: an adapter that silently fails to attach produces a clip
            # that is merely AnimateDiff, and the null result gets blamed on the
            # method). It is only skipped when the exact bytes it measured are
            # unchanged.
            cat = ml.catalog()
            cache = _load_verify_cache()
            dead, checked, reused = [], [], []
            for m in self.adapters:
                path = cat.get(m, {}).get("path")
                digest = _sha1_path(path) if path else ""
                if digest and cache.get(m, {}).get("sha1") == digest:
                    self.effects[m] = cache[m]["effect"]
                    reused.append(m)
                    if self.effects[m] == 0.0:
                        dead.append(m)
                    continue
                d = ml.verify_effect(self.pipe, self.ff, m)
                self.effects[m] = d
                checked.append(m)
                if digest:
                    cache[m] = {"sha1": digest, "effect": d,
                                "at": time.strftime("%Y-%m-%d %H:%M:%S")}
                if d == 0.0:
                    dead.append(m)
            _save_verify_cache(cache)
            if reused:
                print(f"  preflight: {len(reused)} adapter(s) already verified "
                      f"against these exact weights -- skipped "
                      f"({', '.join(reused[:4])}"
                      f"{', ...' if len(reused) > 4 else ''}). Delete "
                      f"{VERIFY_CACHE.name} in output/film/ to force a re-check.")
            if checked:
                print("  preflight (verified now): " + ", ".join(
                    f"{m}={self.effects[m]:.3f}" for m in checked))
                if dead:
                    raise RuntimeError(
                        f"adapters with no effect on the UNet: {dead}. Refusing to "
                        f"generate -- the result would be a false negative.")
        self.set_motion("none")

    def set_motion(self, name):
        try:
            if name in self.adapters:
                self.pipe.enable_lora()
                self.pipe.set_adapters([name], adapter_weights=[1.0])
            else:
                self.pipe.disable_lora()
            self.ff.set_active(name if name in self.ff.installed else None, 1.0)
        except Exception as exc:
            print(f"    set_motion({name}) failed: {str(exc)[:90]}")

    # kept for compatibility with the earlier camera-only code path
    _set = set_motion

    def clip(self, shot, seed, model: str | None = None):
        if model:
            self.ensure_model(model)
        self.set_motion(shot.camera)
        t0 = time.time()
        with torch.no_grad():
            out = self.pipe(prompt=shot.prompt, negative_prompt=NEG,
                            num_frames=FRAMES, guidance_scale=1.0,
                            num_inference_steps=STEPS, width=SIZE, height=SIZE,
                            generator=torch.Generator("cpu").manual_seed(seed))
        return [f.convert("RGB") for f in out.frames[0]], time.time() - t0


def stage_dir(name: str, i: int, stage: str) -> Path:
    return OUT / name / stage / f"shot{i+1:02d}"


def _st(frames, name: str, i: int, stage: str) -> None:
    """Write one shot's stage frames. `raw` is saved separately at the top."""
    d = stage_dir(name, i, stage)
    d.mkdir(parents=True, exist_ok=True)
    for j, f in enumerate(frames):
        f.save(d / f"f{j:03d}.png")


def _ld(name: str, i: int):
    """Load a shot's shaken frames, or None if that shot was never rendered."""
    d = stage_dir(name, i, "shaken")
    fs = sorted(d.glob("f*.png")) if d.exists() else []
    if not fs:
        return None
    from PIL import Image
    return [Image.open(f).convert("RGB") for f in fs]


def raft_upsample(frames, factor=2, flow_size=256):
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
    from torchvision.transforms.functional import resize
    import torch.nn.functional as F

    H, W = frames[0].size[1], frames[0].size[0]
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=False).eval()
    transforms = weights.transforms()

    def flow(a, b):
        ia = resize((a * 255).to(torch.uint8), [flow_size, flow_size])
        ib = resize((b * 255).to(torch.uint8), [flow_size, flow_size])
        ba, bb = transforms(ia, ib)
        with torch.no_grad():
            f = model(ba, bb)[-1]
        if flow_size != W:
            f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            f = f * (W / flow_size)
        return f

    tens = [to_tensor(f) for f in frames]
    out = [frames[0]]
    for i in range(len(frames) - 1):
        a, b = tens[i], tens[i + 1]
        f_ab, f_ba = flow(a, b), flow(b, a)
        for k in range(1, factor):
            t = k / factor
            mix = warp(a, f_ab * t) * (1 - t) + warp(b, f_ba * (1 - t)) * t
            out.append(to_image(mix))
        out.append(frames[i + 1])
    return out


def render(planned, clips, name="film"):
    """Exposure sheet timing -> hard-cut frame list."""
    from animation_spec import AnimationState, KeyPose, MotionPlan, Easing

    poses = tuple(
        KeyPose(f"shot{i}", AnimationState.of(i=i), ease_into=Easing.SNAP,
                hold=hold_frames(s, FPS))
        for i, s in enumerate(planned))
    plan_obj = MotionPlan(poses)
    segs = plan_obj.to_track().segments()

    frames = []
    for drawing_id, _start, length in segs:
        i = int(drawing_id.replace("shot", ""))
        src = clips[i]
        # each source clip already has its own motion; stretch or trim to the hold
        if len(src) >= length:
            idx = [round(k * (len(src) - 1) / max(1, length - 1)) for k in range(length)]
        else:
            idx = [round(k * (len(src) - 1) / max(1, length - 1)) for k in range(length)]
        frames.extend(src[j] for j in idx)

    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(d / f"f{i:04d}.png")
    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, str(d / "film.mp4"), fps=FPS)
    except Exception as exc:
        print(f"    mp4 failed: {exc}")
    small = [f.resize((256, 256)) for f in frames[::2]]
    small[0].save(d / "film.gif", save_all=True, append_images=small[1:],
                  duration=int(1000 / (FPS / 2)), loop=0)
    return frames, d


def run_store(a) -> int:
    """Render a production from `productions/<slug>/` -- the store path.

    This is `--from-project` done properly. Three differences that matter:

      * The plan is read from the SHOTS THEMSELVES, in timeline order. There is no
        frozen copy to drift out of date, so an edit made one second before the
        render is the edit that renders.
      * A shot that needs generating writes a NEW TAKE and never overwrites the
        old one. `--takes` then lets the UI choose between them.
      * A shot whose MODEL inputs are unchanged but whose finishing inputs moved
        (grade, hold, shake) is re-finished from its existing frames. That is
        seconds of RAFT + shake + grade instead of 9-14 minutes of AnimateDiff,
        which is the whole reason the per-shot ladder is kept on disk.

    Only shots that need it are touched. Everything else is loaded from disk.
    """
    import film_store as fstore
    import effects
    from looks import apply_look

    project = fstore.load(a.store)
    plan = fstore.export_plan(project)
    order = plan["order"]
    if not order:
        raise SystemExit(f"production {a.store!r} has no shots in its timeline")
    fps = plan["fps"]

    # The raw frame count is a per-production SETTING, not a constant, because it is
    # the one dial that measurably buys speed: 12 frames = 510s against 16 = 695s on
    # this machine, with alignment and coherence equal or better.
    #
    # A PRODUCTION THAT HAS NEVER RECORDED A FRAME COUNT IS TREATED AS 16, not as the
    # new default. Every take already on disk was rendered at 16, and the frame count
    # is part of the generation hash -- so applying the new default to an existing film
    # would mark every shot stale and silently re-render a finished film at a different
    # frame count. New productions get 12 explicitly (film_store.new_project), so the
    # speed-up applies where it is free and nowhere else.
    #
    # NOTE: this does NOT write the value into the project here. Nothing is mutated
    # before the render succeeds -- the frame count is passed to the store explicitly
    # (`frames=FRAMES`) wherever a hash is taken, so there is no window in which a
    # half-applied setting could be read by something else.
    global FRAMES
    st_settings = project.get("settings") or {}
    recorded = st_settings.get("frames")
    if recorded is None:
        print(f"=== raw frames: {FRAMES} (this production has not recorded one; "
              f"keeping the value its takes were rendered at) ===")
    want_frames = int(recorded) if recorded is not None else int(FRAMES)
    if getattr(a, "frames", None):
        want_frames = int(a.frames)
    if want_frames != FRAMES:
        print(f"=== raw frames: {want_frames} (pipeline default {FRAMES}) ===")
        FRAMES = want_frames
    # WARN, DO NOT SILENTLY OBEY. AnimateDiff's motion module was trained on 16-frame
    # windows, so a lower count is a real risk to the thing the film is FOR. A film whose
    # takes were all rendered at 12 came back reading as still frames, and nothing in the
    # run said why. Say it, every time, before spending an hour generating.
    if FRAMES < 16:
        print(f"  !! {FRAMES} raw frames, below the 16 the motion module was trained on."
              f" Motion may read as a still frame. If a clip comes back static, this is"
              f" the first thing to change (settings -> raw frames per shot).")

    only = ({int(x) for x in str(a.only).split(",") if x.strip()} if a.only else None)
    todo = [o for o in order if only is None or (order.index(o) + 1) in only]
    needing_model = [o for o in todo if fstore.needs_generation(o["state"])]
    refinishing = [o for o in todo if o["state"] == "restale"]

    print(f"=== FROM STORE {a.store} ===")
    print(f"  {len(order)} shots, {plan['frames']} frames, "
          f"{plan['duration_s']:.2f}s at {fps}fps")
    print(f"  this run: {len(needing_model)} to generate, "
          f"{len(refinishing)} to re-finish, "
          f"{len(todo) - len(needing_model) - len(refinishing)} already current")
    if a.dry_run:
        for i, o in enumerate(order, 1):
            print(f"   {i:>2}. {o['id']:<5} {o['state']:<10} "
                  f"{o['shot'].camera:<20} {o['hold_frames']:>3}f  "
                  f"{o['palette']:<12} {o['shot'].prompt[:58]}")
        print("\n(dry run -- nothing generated)")
        return 0

    # ---- generate the shots whose MODEL inputs moved ---------------------- #
    clips: Dict[str, list] = {}
    gen = None
    if needing_model:
        gen = Generator(a.base, loras=None)
        print(f"\n=== GENERATING {len(needing_model)} shot(s) ===")
        for n, o in enumerate(needing_model, 1):
            shot = o["shot"]
            t0 = time.time()
            frames, dt = gen.clip(shot, o["generation"]["seed"],
                                  model=(o["generation"].get("model")
                                         if o["generation"].get("model") not in
                                         (None, "", "off") else None))
            # every store call below takes the shot RECORD, which `film_store._sid`
            # accepts as well as an id -- one of these used to pass a record to a
            # function that wanted an id, and it failed after the model had loaded
            take = fstore.next_take_name(project, o["record"])
            tdir = fstore.take_dir(project, o["record"], take)
            frames_dir = tdir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for j, f in enumerate(frames):
                f.save(frames_dir / f"f{j:03d}.png")
            smooth = raft_upsample(frames)
            rec = fstore.take_record(
                project, o["record"], seconds=round(time.time() - t0, 1),
                frames=len(smooth),
                ladder={"raw": len(frames), "smooth": len(smooth),
                        "delivered": o["hold_frames"]},
                take=take, model=gen.base, camera=shot.camera)
            fstore._write_json(tdir / "take.json", rec)
            fstore.choose_take(project, o["record"], take)
            fstore.log_history(project, act="generate", shot=o["record"]["id"],
                               take=take, seconds=rec["seconds"])
            fstore.save(project)
            project = fstore.load(a.store)          # re-derive against the new take
            print(f"  {n}/{len(needing_model)} {o['record']['id']} "
                  f"{shot.camera:<20} {dt:5.0f}s -> take {take} "
                  f"({len(frames)} raw, {len(smooth)} smooth)", flush=True)

    # ---- PARTIAL RUN: generate, land the takes, and STOP ------------------- #
    # `--only` means "these shots, and nothing else". Without this, a partial run fell
    # through to the finishing loop, which walks the WHOLE timeline and raises on the
    # first shot that has no take yet ("shot s02 has no take after this run and is
    # needed for assembly"). A user batching shots 1 and 3 while shot 2 was unrendered
    # would have paid ten minutes of generation and then lost the run.
    #
    # The error message was correct for its own purpose -- an assembled film needs
    # every shot -- and wrong as the outcome of asking to shoot two of them. Stopping
    # and saying so is the honest behaviour: takes are written, nothing is graded, the
    # takes survive, and the next run picks up where this one left off.
    if only is not None:
        done_ids = [o["record"]["id"] for o in todo]
        print(f"\n=== PER-SHOT RUN: shots {sorted(only)} ===")
        print(f"  generated and kept: {done_ids or '(nothing needed generating)'}")
        print(f"  their takes are on disk under productions/{a.store}/takes/. "
              f"Grading and assembly need EVERY shot, so they were skipped.")
        still = fstore.film_status(fstore.load(a.store))
        print(f"  the film now needs: {still['to_generate']} more shot(s) generated, "
              f"{still['to_refinish']} re-finished")
        return 0

    # ---- load every shot's frames: fresh take, or the ladder on disk ------- #
    print("\n=== FINISHING ===")
    stretched, planned, looks_used = [], [], []
    for i, o in enumerate(order, 1):
        shot = o["shot"]
        state = fstore.shot_state(project, o["record"])
        if only is not None and i not in only and o["take"]:
            # untouched shots still have to be in the film, so load what is there
            pass
        take = fstore.best_take(project, o["record"])
        if take is None:
            raise SystemExit(f"shot {o['record']['id']} has no take after this "
                             f"run and is needed for assembly. Render it first.")
        src = [Image.open(p).convert("RGB")
               for p in sorted((Path(take["_dir"]) / "frames").glob("f*.png"))]
        hold = o["hold_frames"]
        base = stretch_to(src, hold)
        sp = fstore.finish_inputs(project, o["record"])
        if plan["shake_scale"] > 0:
            spec = spec_for_shot(shot.scale, shot.kind, shot.seconds,
                                 seed=int(o["generation"]["seed"]), fps=fps,
                                 amount_scale=float(sp["shake_scale"]))
            base = apply_handheld(base, spec, fps=fps)
        stretched.append(base)
        planned.append(shot)
        looks_used.append(sp["palette"])
        print(f"  {i:>2}. {o['record']['id']:<5} take {take['take']}  "
              f"{len(src)} -> {len(base)} frames  palette {sp['palette']}")

    # ---- grade, exactly as the legacy path does and in the same order ------ #
    if plan["technical_baseline"] or any(lk and lk != "neutral"
                                         for lk in looks_used):
        import grade_match as gm
        print("\n=== GRADING ===")
        if plan["technical_baseline"]:
            sigs = [gm.signature(c) for c in stretched]
            tgt = gm.technical_target(sigs)
            stretched = [gm.match_technical(c, tgt) for c in stretched]
            print("  shared technical baseline applied")
        if any(lk and lk != "neutral" for lk in looks_used):
            stretched = [apply_look(c, lk) for c, lk in zip(stretched, looks_used)]
            print("  palettes applied: "
                  + ", ".join(f"{i}={lk}" for i, lk in enumerate(looks_used, 1)))

    # ---- effects: AFTER grading, BEFORE assembly --------------------------- #
    # Order matters and this is the right one. The technical baseline exists to make
    # exposure and colour consistent across shots; an effect changes how the image
    # REPRESENTS tone, and running one before the baseline would let the baseline partly
    # undo it -- the same mistake the grading order already avoids (baseline, then look).
    #
    # Effects run per shot, on that shot's whole clip, because the temporal ones need a
    # clip to have history and a chain re-started per frame would break them.
    steps = list((project.get("settings") or {}).get("effects") or [])
    bad = effects.validate_chain(steps) if steps else []
    if bad:
        raise SystemExit("the effect chain names unknown effects: " + "; ".join(bad))
    if steps:
        print(f"\n=== EFFECTS ({len(steps)} step(s)) ===")
        out = []
        for i, (o, clip) in enumerate(zip(order, stretched), 1):
            print(f"  shot {i} ({o['record']['id']})")
            out.append(effects.apply_chain(clip, steps, verbose=True))
        stretched = out
    else:
        print("\n=== EFFECTS === none set (settings.effects is empty)")

    # ---- assemble into the production's renders/ --------------------------- #
    print("\n=== ASSEMBLING ===")
    name = a.name or f"{a.store}_render"
    frames, d = render(planned, stretched, name)
    meta = {"store": a.store, "name": name, "fps": fps,
            "total_frames": len(frames), "duration_s": round(len(frames) / fps, 2),
            "shots": [{"shot": o["record"]["id"], "take": (o["take"] or None),
                       "state": o["state"], "camera": o["shot"].camera,
                       "scale": o["shot"].scale, "hold_frames": o["hold_frames"],
                       "in_s": round(o["in_s"], 3), "palette": o["palette"]}
                      for o in order]}
    (d / "stats.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  {len(frames)} frames @ {fps}fps = {len(frames)/fps:.1f}s")

    # The render also lands IN the production, not only in output/film. The store is
    # the durable record and output/film is the legacy flat tree -- without this the
    # UI could only find renders by guessing names, which is the workaround the old
    # TIMELINE stage needed ("attach a film that already exists") and the reason a
    # project could be attached to a directory the per-shot UI could not read.
    try:
        import shutil
        rd = fstore.renders_dir(a.store) / name
        rd.mkdir(parents=True, exist_ok=True)
        for f in ("film.mp4", "stats.json", "shots.png", "film.gif"):
            src = d / f
            if src.exists():
                shutil.copy2(src, rd / f)
        project = fstore.load(a.store)
        project["deliverable"] = {
            "render": name, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": round(len(frames) / fps, 2), "frames": len(frames),
            "path": str(rd), "silent": True}
        fstore.log_history(project, act="render", name=name,
                           frames=len(frames), seconds=round(len(frames) / fps, 2))
        fstore.save(project)
        print(f"  and into the production: {rd}")
    except Exception as exc:
        print(f"    store copy failed ({type(exc).__name__}: {exc}); the render is "
              f"still at {d}")

    print(f"\nwrote {d/'film.mp4'}")
    print("DONE")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--name", default="film")
    ap.add_argument("--only", default=None,
                    help="render ONLY these shots (comma list, 1-based) and stop "
                         "before grading and assembly. Their per-shot stages are "
                         "kept on disk, so a single shot can be re-rendered "
                         "without touching the others")
    ap.add_argument("--assemble", action="store_true",
                    help="skip generation entirely and assemble the shots already "
                         "on disk. Generate once per shot, assemble as often as "
                         "you like -- the pool's economy, without the pool")
    ap.add_argument("--from-project", default=None,
                    help="render the FROZEN plan in workspace/<name>.json instead "
                         "of recomputing the director and the planner. This is "
                         "what makes an approved plan the plan that renders; any "
                         "edit made in the workbench would otherwise be discarded")
    ap.add_argument("--store", default=None,
                    help="render a PRODUCTION from productions/<slug>/ (the current "
                         "model). Reads the shots themselves in timeline order -- no "
                         "frozen copy to drift -- writes a NEW immutable take per "
                         "shot instead of overwriting, and re-finishes shots whose "
                         "grade/hold/shake moved without re-generating them")
    ap.add_argument("--shots", default="landscape",
                    choices=["landscape", "figure", "test", "llm"],
                    help="landscape/figure/test are hand-written lists; 'llm' "
                         "has the local model write the shot list from --concept")
    ap.add_argument("--concept", default=None,
                    help="a single line of text. With --shots llm the model "
                         "writes the shot list from it")
    ap.add_argument("--llm", default="llama3:latest",
                    help="which ollama model writes the shot list")
    ap.add_argument("--writers", default="ensemble",
                    choices=["single", "ensemble"],
                    help="single = one model writes everything; ensemble = "
                         "several cheap 'idea farm' models supply raw visual "
                         "ideas and one showrunner authors the final shot list "
                         "and owns cohesion")
    ap.add_argument("--farms", default=",".join(
        ["qwen2.5:3b", "phi3:latest", "mistral:latest"]),
        help="comma list of farm models, used when --writers ensemble. Three, "
             "not four: llama3 gave one idea where the others give six, and the "
             "small qwen3 models were measured worse still")
    ap.add_argument("--showrunner", default="granite4.1:8b",
                    help="the one model that authors the final shot list")
    ap.add_argument("--llm-temp", type=float, default=0.8)
    ap.add_argument("--models", default="off",
                    help="off | auto | a comma list of checkpoint names. 'auto' "
                         "runs model_select.py to pick per shot for variety; "
                         "renderable checkpoints only")
    ap.add_argument("--looks", default="off",
                    help="off | auto | a comma list of palette names from "
                         "looks.py. 'auto' walks the ARC, one look per scene")
    ap.add_argument("--look-post", default="technical",
                    choices=["off", "technical"],
                    help="technical = normalise exposure/contrast/grain across "
                         "shots so no cut reads as a technical error, then apply "
                         "the look on top")
    ap.add_argument("--allow-unproven", action="store_true",
                    help="let model_select propose models whose output has not "
                         "been accepted (t2v-zero)")
    ap.add_argument("--direct", action="store_true",
                    help="let director.py choose each shot's motion from fit, "
                         "cohesiveness and continuity, instead of the shot list")
    ap.add_argument("--director", default="global", choices=["global", "greedy"],
                    help="global optimises the whole shot list at once (the "
                         "moving budget goes where it matters most); greedy is "
                         "the shipped left-to-right behaviour, for comparison")
    ap.add_argument("--no-energy-arc", action="store_true",
                    help="drop the motion-energy-arc term (NOT story progression -- see "
                         "story.BEATS), so the moving budget is spent only where "
                         "each shot individually benefits")
    ap.add_argument("--max-movers", type=int, default=None,
                    help="cap how many shots may carry a camera move. Default: "
                         "2 for a short film, scaled up for longer ones")
    ap.add_argument("--strict-evidence", action="store_true",
                    help="refuse any subject motion whose localisation has not "
                         "been measured")
    ap.add_argument("--length", default="custom",
                    choices=["tiktok", "reels", "shorts", "custom"],
                    help="platform duration preset. Sets the rhythm curve's "
                         "total, so the film lands on the target length without "
                         "making every shot equal (R4 forbids that)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="explicit duration, overriding --length")
    ap.add_argument("--no-budget", action="store_true",
                    help="skip the duration -> shots -> hours report")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and direct, print the result, generate nothing")
    ap.add_argument("--base", default="models/dreamshaper8",
                    choices=["models/dreamshaper8", "models/sd15"],
                    help="SD1.5-family checkpoint. dreamshaper8 is cleaner and "
                         "does not leak the SHUTTERSTOCK watermark; sd15 is "
                         "harsher and more saturated.")
    ap.add_argument("--shake", type=float, default=1.0,
                    help="handheld shake SCALE, 1.0 = the researched default "
                         "hold, 0 to disable. Not pixels: amplitude and roll are "
                         "per shot now, derived from the shot itself")
    ap.add_argument("--frames", type=int, default=None,
                    choices=[8, 12, 16, 20, 24],
                    help="raw frames the model generates per shot, before RAFT x2. "
                         "The store's `settings.frames` is used when this is omitted. "
                         "Measured: 12 = 510s against 16 = 695s, with alignment and "
                         "coherence equal or better. 8 was rejected on motion quality")
    a = ap.parse_args()
    if a.store:
        return run_store(a)
    # ---- render a FROZEN plan from a project file ------------------------- #
    # The workbench's whole point is that the plan you approved is the plan that
    # renders. Without this, `make_film` recomputes the director and the planner
    # on every run, so an approved plan -- and any edit made in the UI -- would be
    # silently discarded.
    import film_project as fp
    project = None
    proposal = None      # the text-layer scorecard is skipped for a frozen plan
    looks = None
    if a.from_project:
        project = fp.load(a.from_project)
        planned = fp.frozen_shots(project)
        if not planned:
            raise SystemExit(
                f"project {a.from_project!r} has no frozen plan. Plan it and "
                f"approve the plan first (the PLAN stage), then render.")
        report = fp.frozen_report(project)
        shots = planned
        scenes = [s.scene for s in planned]
        looks = project.get("grade", {}).get("looks") or None
        if looks:
            a.looks = "off"          # the project's palettes win; see below
        # the baseline on/off is also a project setting, not a CLI flag: the GRADE
        # stage owns it, and a control there must reach the render.
        a.look_post = ("technical" if project.get("grade", {}).get("technical", True)
                       else "off")
        print(f"  grade: baseline {a.look_post}, "
              f"palettes {looks if looks else '(none)'}")
        print(f"=== FROM PROJECT {a.from_project} ===")
        print(f"  FROZEN plan: {len(planned)} shots, approved "
              f"{project.get('plan', {}).get('approved_at') or 'NOT APPROVED'}")
        print(f"  rendering exactly these; the director and the planner are "
              f"NOT re-run")
        print(describe(planned))
    target_seconds = a.seconds or LENGTHS.get(a.length)
    if project is None:

        shots = {"figure": FIGURE_SHOTS, "test": TEST_SHOTS}.get(a.shots, SHOTS)
        proposal = None
        if a.shots == "llm":
            from story import (CONCEPT_DEFAULT, scorecard, write_shots,
                               write_shots_ensemble)
            concept = a.concept or CONCEPT_DEFAULT
            who = (f"ensemble: {len(a.farms.split(','))} farms -> {a.showrunner}"
                   if a.writers == "ensemble" else a.llm)
            print(f'=== STORY ({who}) ===\n  concept: "{concept}"')
            # shot count is derived from the target duration, because the holds clamp
            # and duration is therefore set by how many shots there are
            want = None
            if target_seconds:
                want = max(4, round(target_seconds / 2.5))
                print(f"  target {target_seconds:.0f}s -> asking for {want} shots")
            if a.writers == "ensemble":
                proposal = write_shots_ensemble(
                    concept,
                    farms=[m.strip() for m in a.farms.split(",") if m.strip()],
                    showrunner=a.showrunner, temperature=a.llm_temp, n_shots=want)
            else:
                proposal = write_shots(concept, model=a.llm,
                                       temperature=a.llm_temp, n_shots=want)
            print(f"  logline: {proposal.logline}")
            shots = proposal.shots
        if a.test:
            shots = [Shot(shots[0].prompt, "wide", "none", 2.0),
                     Shot(shots[1].prompt, "macro", "none", 2.0)]

        # scene ids, before the planner runs, so a model switch and a palette land on
        # the same boundary the planner cannot move
        scenes = [s.scene for s in shots]

        print("=== PROPOSED ===")
        print(describe(shots))

        if a.direct:
            from director import choose, FitScorer
            print("\n=== DIRECTING ===")
            # `global` (the default) optimises the whole shot list at once, so the
            # moving-shot budget goes to the shots that most deserve it instead of
            # the earliest ones. `greedy` is the shipped left-to-right behaviour,
            # kept so the two can be compared on the same list.
            shots, report = choose(shots, strict_evidence=a.strict_evidence,
                                   scorer=FitScorer(), mode=a.director,
                                   max_movers=a.max_movers,
                                   energy_arc=not a.no_energy_arc)
            print("\n" + report.show())

        planned, report = plan(shots, total=target_seconds)
    print("\n=== PLANNED ===")
    achieved = sum(s.seconds for s in planned)
    if target_seconds and not a.no_budget:
        print(budget_report(target_seconds))
        # A target duration is only reachable if there are enough shots. Holds
        # are clamped to MAX_SHOT (3.6s), so N shots cannot exceed 3.6N seconds
        # however the curve is scaled. Without this check the preset silently
        # under-delivers -- 6 shots asked for 30s delivers 21.6s and says
        # nothing, which is the same class of bug as the refused-model fallback.
        need = max(1, round(target_seconds / 2.5))
        if achieved < target_seconds * 0.9:
            print(f"  !! target {target_seconds:.0f}s NOT reachable with "
                  f"{len(planned)} shots: holds clamp at "
                  f"{__import__('edit_planner').MAX_SHOT}s, so the ceiling is "
                  f"{len(planned) * __import__('edit_planner').MAX_SHOT:.1f}s")
            print(f"  !! achieved {achieved:.1f}s. Ask the writer for about "
                  f"{need} shots, or accept a shorter piece.")
        else:
            print(f"  achieved {achieved:.1f}s across {len(planned)} shots")
    print(describe(planned))
    print("\n" + report.show())
    print(energy_profile(planned))

    if proposal is not None:
        print("\n=== TEXT-LAYER SCORECARD (what the model cost the pipeline) ===")
        scorecard(proposal, planned, report.changes)

    # ---- pick a checkpoint per shot, for variety --------------------------- #
    models = None
    if a.models != "off":
        if a.models == "auto":
            from model_select import ShotNeed, WEIGHTS, choose as choose_models, report as mreport
            needs = [ShotNeed(p.prompt, p.scale, motion=(p.energy if p.energy is not None
                                                         else 0.5),
                              detail=0.7, subject=p.subject,
                              scene=(scenes[i] if i < len(scenes) else ""))
                     for i, p in enumerate(planned)]
            models, masg, mmeta = choose_models(
                needs, technical_matched=(a.look_post != "off"),
                allow_unproven=a.allow_unproven)
            print("\n=== MODEL ASSIGNMENT (variety) ===")
            print(mreport(masg, mmeta))
        else:
            models = [m.strip() for m in a.models.split(",")]

    # ---- pick a palette per scene ------------------------------------------ #
    # a project's per-scene palettes survive this block
    looks = looks if project is not None else None
    if a.looks != "off":
        import looks as looks_mod
        if a.looks == "auto":
            # one look per scene, walking the ARC. Scenes are numbered by first
            # appearance, so scene 0 gets ARC[0], scene 1 gets ARC[1], and the
            # measured legibility of each adjacent ARC step is what makes the
            # progression safe.
            order, idx = [], {}
            for s in scenes:
                if s not in idx:
                    idx[s] = len(order)
                    order.append(s)
            looks = [looks_mod.ARC[idx[(scenes[i] if i < len(scenes) else "")]
                                   % len(looks_mod.ARC)]
                     for i in range(len(planned))]
        else:
            from itertools import cycle
            looks = list(cycle(l.strip() for l in a.looks.split(",")))
        looks = looks[:len(planned)]
        print("\n=== LOOKS (per scene) ===")
        for i, p in enumerate(planned):
            print(f"  {i+1}. scene={scenes[i] if i < len(scenes) else ''!r:<10} "
                  f"look={looks[i]}")

    if a.dry_run:
        print("\n(dry run -- nothing generated)")
        return

    print("\n=== GENERATING ===")
    from keep_awake import keep_awake
    with keep_awake(f"{len(planned)}-shot film"):
        # `--assemble` needs NO generator: it reads frames that already exist.
        # Building the diffusion pipeline first cost 7 GB and minutes of model
        # loading to cut a film out of finished frames -- measured, not guessed.
        gen = None if a.assemble else Generator(a.base)
        print(f"  starting checkpoint: {a.base}")
        clips, stats = [], []
        only = ({int(x) for x in str(a.only).split(",") if x.strip()}
                if a.only else None)
        if a.assemble:
            # nothing is generated: the shaken frames from earlier per-shot runs
            # are the input. This is what makes "re-render just that shot" real.
            clips = [_ld(a.name, i) for i in range(len(planned))]
            if any(c is None for c in clips):
                missing = [i + 1 for i, c in enumerate(clips) if c is None]
                raise SystemExit(
                    f"--assemble needs every shot on disk; missing {missing}. "
                    f"Render those first (RENDER stage, or --only).")
            print(f"  assembling {len(clips)} shots from disk, no generation")
            stats = [{"shot": i + 1, "prompt": s.prompt, "scale": s.scale,
                      "camera": s.camera, "kind": s.kind,
                      "model": (((project or {}).get("film") or {}).get("model")
                                or Path(a.base).name),
                      "generator_loaded": False,
                      "look": (looks[i] if looks and i < len(looks) else None),
                      "hold_s": s.seconds, "gen_seconds": 0,
                      "raw_motion": 0.0, "smooth_frames": len(c),
                      "shake": False, "shake_spec": "reused from disk",
                      "assembled_from_disk": True}
                     for i, (s, c) in enumerate(zip(planned, clips))]
        for i, s in enumerate(planned if not a.assemble else []):
            model = models[i] if models else None
            frames, dt = gen.clip(s, SEED + i, model=model)

            # Save the RAW clip before RAFT and handheld shake touch it.
            #
            # Measuring the assembled film cannot answer a question about the
            # model, because the pipeline has since added global translational
            # motion (shake) and interpolated frames. Every shot in the first
            # footage test measured global_share 0.62-0.76 including a
            # near-static fog road, i.e. the shake dominated the metric. Keep the
            # raw clip so model-level questions stay answerable.
            raw_dir = OUT / a.name / "raw" / f"shot{i+1:02d}"
            raw_dir.mkdir(parents=True, exist_ok=True)
            for j, f in enumerate(frames):
                f.save(raw_dir / f"f{j:03d}.png")

            smooth = raft_upsample(frames)
            # keep the per-shot stages on disk. `raw/` was already saved above;
            # these two are what make "raw -> RAFT -> shaken" inspectable per shot,
            # and what `--assemble` reads instead of regenerating.
            _st(smooth, a.name, i, "smooth")
            # A per-shot camera edit from the workbench must reach the render, or
            # the CAMERA stage is a control that does nothing. Parameters come from
            # the project when rendering from one; the CLI scale is the fallback.
            sp = (fp.shake_params(project, i) if project is not None
                  else {"amount_scale": a.shake, "seed": i + 3, "enabled": True})
            if a.shake > 0 and sp.get("enabled", True):
                # Stretch to the delivered length FIRST, then shake. `a.shake`
                # is a scale, not a pixel count: 1.0 is the researched default
                # hold, and the character (amplitude, band mix, roll) is drawn
                # per shot from the framing scale, the motion kind, the duration
                # and the shot's own seed. See handheld.py.
                hold = hold_frames(s, FPS)
                smooth = stretch_to(smooth, hold)
                spec = spec_for_shot(s.scale, s.kind, s.seconds,
                                     seed=int(sp["seed"]), fps=FPS,
                                     amount_scale=float(sp["amount_scale"]))
                smooth = apply_handheld(smooth, spec, fps=FPS)
                shake_note = (f"{spec.amount_px:.1f}px/{spec.roll_deg:.2f}deg "
                              f"floor {spec.floor:.2f}")
            else:
                shake_note = None
            _st(smooth, a.name, i, "shaken")
            d = np.mean([np.abs(np.asarray(frames[j]).astype(np.float32)
                                - np.asarray(frames[j - 1]).astype(np.float32)).mean()
                         for j in range(1, len(frames))])
            clips.append(smooth)
            stats.append({"shot": i + 1, "prompt": s.prompt, "scale": s.scale,
                          "camera": s.camera, "kind": s.kind,
                          "model": model or gen.base,
                          "look": looks[i] if looks else None,
                          "hold_s": s.seconds,
                          "gen_seconds": round(dt, 1), "raw_motion": round(float(d), 2),
                          "smooth_frames": len(smooth), "shake": a.shake > 0,
                          "shake_spec": shake_note})
            if only is not None and (i + 1) not in only:
                continue
            print(f"  {i+1}/{len(planned)} {s.camera:<22} {dt:5.0f}s  "
                  f"motion {d:5.2f}  model={model or gen.base}  "
                  f"-> {len(smooth)} frames"
                  f"{'  +shake ' + shake_note if shake_note else ''}", flush=True)

    if only is not None:
        print(f"\n=== PER-SHOT RUN: shots {sorted(only)} ===")
        print(f"  their stages are on disk under output/film/{a.name}/. Grading and "
              f"assembly need EVERY shot, so they were skipped -- render the rest, "
              f"then use --assemble (which regenerates nothing).")
        return

    # ---- the look stage: baseline first, then palette ---------------------- #
    # Order matters. `match_technical` gives every shot one photographic baseline
    # so no cut reads as an exposure/grain error; `apply_look` then adds the
    # deliberate palette. Doing it the other way round would let the palette
    # shift be partly undone by the baseline match.
    if a.look_post != "off" or looks:
        import grade_match as gm
        from looks import apply_look
        print("\n=== GRADING ===")
        sigs = [gm.signature(c) for c in clips]
        tgt = None
        if a.look_post == "technical":
            tgt = gm.technical_target(sigs)
            clips = [gm.match_technical(c, tgt) for c in clips]
            # Per-shot conformance to the SHARED TARGET is valid: every shot is
            # compared to the same reference, not to another shot. Reported
            # ABSOLUTELY -- the technical distance is already in physical units,
            # and both normalisers available here are degenerate on exactly this
            # kind of film (see grade_match.technical_offset).
            print("  shared technical baseline applied "
                  "(exposure/contrast/grain normalised)")
            for i, c in enumerate(clips):
                d = gm.technical_offset(gm.signature(c), tgt)
                print(f"    shot {i+1}: technical offset from baseline "
                      f"{d:.2f}  ({d * gm.SCALES[1]:.1f} grey levels of contrast)")
        if looks:
            clips = [apply_look(c, lk) for c, lk in zip(clips, looks)]
            print("  palette applied: "
                  + ", ".join(f"{i+1}={lk}" for i, lk in enumerate(looks)))
            # Deliberately NOT reporting a cross-shot style verdict here.
            #
            # `decomp` between two DIFFERENT shots measures content, not look --
            # the confound documented in handoff 9.9. The proof is in this very
            # film: two shots that share a look measured 2.87 and 10.28 apart
            # (nothing deliberate changed between them), while the one cut that
            # does carry a palette change measured 40.74 -- against 1.80 for the
            # same palette change on held content in looks.py. A 22x inflation
            # from content alone. Any verdict from those numbers would be
            # fiction, so the film reports the assignment and leaves legibility
            # to `looks.py`, which measures it on a held source.
            print("  (cross-shot style distance is content-confounded and is NOT "
                  "reported; look legibility is validated on held content by "
                  "`python looks.py` -- 6/6 transitions legible)")

    print("\n=== ASSEMBLING ===")
    frames, d = render(planned, clips, a.name)
    print(f"  {len(frames)} frames @ {FPS}fps = {len(frames)/FPS:.1f}s")
    meta = {"shots": stats, "changes": report.changes,
            "total_frames": len(frames), "fps": FPS,
            "duration_s": round(len(frames) / FPS, 2)}
    (d / "stats.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # contact sheet: 3 frames from each shot
    T = 150
    canvas = Image.new("RGB", (3 * (T + 4) + 4, len(clips) * (T + 22) + 22), (20, 20, 24))
    dr = ImageDraw.Draw(canvas)
    dr.text((5, 3), f"{a.name}: {len(clips)} shots, {len(frames)/FPS:.1f}s, "
                    f"hard cuts, planner applied", fill=(235, 235, 235))
    for r, (s, c) in enumerate(zip(stats, clips)):
        y = 20 + r * (T + 22)
        dr.text((5, y), f"{r+1}. {s['scale']:<8} {s['camera']:<12} "
                        f"{s['hold_s']}s  motion {s['raw_motion']}",
                fill=(215, 215, 215))
        picks = [0, len(c) // 2, len(c) - 1]
        for k, pi in enumerate(picks):
            canvas.paste(c[pi].resize((T, T), Image.LANCZOS), (4 + k * (T + 4), y + 16))
    canvas.save(d / "shots.png")
    print(f"\nwrote {d/'film.mp4'} and {d/'shots.png'}")
    print("DONE")


if __name__ == "__main__":
    raise SystemExit(main())
