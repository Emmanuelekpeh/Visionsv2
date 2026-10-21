"""
generate_sequence.py -- render an actual animation sequence from the locked spec.

This is the end-to-end proof of ANIMATION_SPEC.md:

  * Drawings are generated once per distinct STATE, never chained (INV-1).
  * Timing comes from an exposure sheet; holds are real and free (INV-2/INV-3).
  * Generation calls equal the number of distinct drawings, not frames (INV-5).

Two states that quantise to the same prompt are the same drawing: they are
collapsed and generated once. That is the spec doing real work, not decoration.

It also renders the same sequence the LEGACY way -- each drawing produced by
img2img from the previous drawing -- so the drift the spec exists to prevent is
visible rather than merely asserted.

Subject: the bouncing ball, the canonical squash-and-stretch exercise.

    python generate_sequence.py                 # both variants
    python generate_sequence.py --no-compare    # authored only
"""

from __future__ import annotations

import argparse
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
SIZE = 256
STEPS = 8
SEED = 1234

BASE_PROMPT = (
    "flat 2D cel animation drawing of a single red ball above a horizontal black "
    "ground line, thick black outlines, flat solid colors, plain white background, "
    "minimal, simple shapes"
)


# ---------------------------------------------------------------------------
# State -> prompt. This is the bridge from animation state to drawing model, and
# it is the honest bottleneck: tiny-sd is text-conditioned only, so a continuous
# state has to be quantised into descriptive bands. That quantisation is itself
# in the spirit of the spec -- cel animation deals in discrete states.
# ---------------------------------------------------------------------------

HEIGHT_BANDS = [
    (0.06, "ball resting on the ground line"),
    (0.18, "ball just above the ground line"),
    (0.35, "ball in the lower part of the frame"),
    (0.52, "ball slightly below mid height"),
    (0.68, "ball at mid height"),
    (0.85, "ball in the upper part of the frame"),
    (0.95, "ball high in the frame"),
    (1.01, "ball near the very top of the frame"),
]

# Resolution matters here. Coarse bands make distinct animation states
# quantise to the same phrase, and then canonical_states() will -- correctly --
# merge them into one drawing, silently destroying the motion. See the note in
# ANIMATION_SPEC.md about the state -> prompt bridge being the real bottleneck.
SQUASH_BANDS = [
    (-0.80, "ball stretched into a long thin vertical oval"),
    (-0.55, "ball stretched into a tall vertical oval"),
    (-0.30, "ball clearly stretched vertically"),
    (-0.12, "ball slightly stretched vertically"),
    (0.12, "ball perfectly round"),
    (0.45, "ball slightly squashed into a wide oval"),
    (0.75, "ball squashed into a wide flat oval"),
    (1.01, "ball squashed flat against the ground"),
]


def _band(value: float, bands) -> str:
    for threshold, phrase in bands:
        if value <= threshold:
            return phrase
    return bands[-1][1]


def prompt_for(state: AnimationState) -> str:
    return (
        f"{BASE_PROMPT}, {_band(state['height'], HEIGHT_BANDS)}, "
        f"{_band(state['squash'], SQUASH_BANDS)}"
    )


# ---------------------------------------------------------------------------
# The animation plan: key poses, an authored breakdown, acceleration into the
# contact, a hard squash, and two held apexes. The stillness is deliberate.
# ---------------------------------------------------------------------------

def build_plan() -> MotionPlan:
    return MotionPlan((
        KeyPose("apex_high", AnimationState.of(height=1.00, squash=0.0), hold=8),
        KeyPose("fall", AnimationState.of(height=0.55, squash=-0.50),
                ease_into=Easing.ACCELERATE, inbetweens=2, hold=2),
        KeyPose("pre_contact", AnimationState.of(height=0.22, squash=-0.75),
                ease_into=Easing.ACCELERATE, hold=2),
        # The hard squash: contact held for three frames. This is the pose that
        # sells the weight, and it cannot come out of interpolation.
        KeyPose("squash", AnimationState.of(height=0.02, squash=1.00),
                ease_into=Easing.SNAP, hold=3),
        KeyPose("launch", AnimationState.of(height=0.16, squash=-0.90),
                ease_into=Easing.SNAP),
        KeyPose("rise", AnimationState.of(height=0.62, squash=-0.40),
                ease_into=Easing.DECELERATE, hold=2),
        KeyPose("apex_low", AnimationState.of(height=0.78, squash=0.0),
                ease_into=Easing.DECELERATE, hold=8),
    ))


def canonical_states(plan: MotionPlan):
    """Collapse states that quantise to the same prompt into a single drawing.

    Two states that describe the same picture are the same cel. Generating it
    twice would be paying for a drawing we already have (INV-5).

    Returns (order, canonical, prompts): the distinct drawing keys in first
    appearance order, the state-key -> drawing-key map, and the prompts.
    """
    order: list = []
    prompts: dict = {}
    canonical: dict = {}
    for pose in plan.states():
        prompt = prompt_for(pose.state)
        match = next((k for k in order if prompts[k] == prompt), None)
        if match is None:
            match = pose.key
            order.append(match)
            prompts[match] = prompt
        canonical[pose.key] = match
    return order, canonical, prompts


def build_sheet(order, canonical, prompts, plan) -> ExposureSheet:
    sheet = ExposureSheet(sheet_id="bounce", fps=FPS)
    for key in order:
        sheet.add_drawing(Drawing(
            id=key, prompt=prompts[key], source=DrawingSource.AUTHORED, state_key=key
        ))
    track = ExposureTrack.mixed([(canonical[p.key], p.hold) for p in plan.states()])
    sheet.add_layer(Layer(name="ball", track=track))
    return sheet


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class Renderer:
    def __init__(self, device: str = "cpu"):
        from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

        print("loading tiny-sd ...", flush=True)
        self.txt2img = StableDiffusionPipeline.from_pretrained(
            "models/tiny-sd", torch_dtype=torch.float32, safety_checker=None
        ).to(device)
        self.img2img = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img.vae,
            text_encoder=self.txt2img.text_encoder,
            tokenizer=self.txt2img.tokenizer,
            unet=self.txt2img.unet,
            scheduler=self.txt2img.scheduler,
            safety_checker=None,
            feature_extractor=None,
        ).to(device)
        self.device = device
        self.calls = 0

    def generate(self, prompt: str, index: int) -> Image.Image:
        """One AUTHORED drawing: generated from the state, tied to nothing else."""
        self.calls += 1
        generator = torch.Generator(device=self.device).manual_seed(SEED + index)
        with torch.no_grad():
            return self.txt2img(
                prompt, num_inference_steps=STEPS, guidance_scale=7.0,
                height=SIZE, width=SIZE, generator=generator,
            ).images[0]

    def chain(self, init_image: Image.Image, prompt: str, index: int) -> Image.Image:
        """The LEGACY path: derive this drawing from the previous drawing.

        This is the INV-1 violation, rendered so the drift is visible.
        """
        self.calls += 1
        generator = torch.Generator(device=self.device).manual_seed(SEED + index)
        with torch.no_grad():
            return self.img2img(
                prompt=prompt, image=init_image, strength=0.35,
                num_inference_steps=12, guidance_scale=7.0, generator=generator,
            ).images[0]


def render_authored(renderer: Renderer, order, prompts, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    images = {}
    for i, key in enumerate(order):
        t0 = time.time()
        image = renderer.generate(prompts[key], i)
        image.save(out / f"{i:02d}_{key.replace('->', '_to_').replace('#', '_')}.png")
        images[key] = image
        print(f"  [{i+1}/{len(order)}] {key:24s} {time.time()-t0:5.1f}s  "
              f"{prompts[key].split(', ', 1)[1][:56]}", flush=True)
    return images


def render_chained(renderer: Renderer, order, prompts, out: Path):
    """The legacy autoregressive path, for comparison only."""
    out.mkdir(parents=True, exist_ok=True)
    images = {}
    previous = None
    for i, key in enumerate(order):
        t0 = time.time()
        if previous is None:
            image = renderer.generate(prompts[key], i)
            tag = "root"
        else:
            image = renderer.chain(previous, prompts[key], i)
            tag = "chained"
        image.save(out / f"{i:02d}_{key.replace('->', '_to_').replace('#', '_')}.png")
        images[key] = image
        previous = image
        print(f"  [{i+1}/{len(order)}] {key:24s} {time.time()-t0:5.1f}s  {tag}", flush=True)
    return images


# ---------------------------------------------------------------------------
# Exposure -> frames
# ---------------------------------------------------------------------------

def frames_from_sheet(sheet: ExposureSheet, images) -> list:
    """Apply exposure. Every frame is a lookup; nothing is generated here."""
    return [images[sheet.frame_state(f)["ball"]] for f in range(sheet.total_frames)]


# ---------------------------------------------------------------------------
# A procedural stand-in for the drawing model.
#
# tiny-sd does not obey shape or position modifiers: "stretched tall and thin"
# returns a round ball, "near the top of the frame" returns a centred one. It
# renders "a red ball" and ignores the rest of the prompt, so it cannot express
# the states at all -- see ANIMATION_SPEC.md 6.1.
#
# Drawing the state directly instead isolates what this spec actually governs:
# which distinct states exist, in what order, and for how many frames each one
# is exposed. That is the claim under test, and it is testable without a
# competent image model.
# ---------------------------------------------------------------------------

def procedural_draw(state: AnimationState, size: int = SIZE) -> Image.Image:
    """Cel-style ball: flat fill, hard outline, real squash and stretch."""
    from PIL import ImageDraw

    img = Image.new("RGB", (size, size), (250, 248, 244))
    draw = ImageDraw.Draw(img)

    ground_y = size * 0.86
    top_y = size * 0.12
    draw.line([(0, ground_y), (size, ground_y)], fill=(20, 20, 20), width=3)

    squash = float(state["squash"])
    height = float(state["height"])

    contact = ground_y - height * (ground_y - top_y)
    radius = size * 0.13
    rx = radius * (1.0 + 0.55 * squash)
    ry = radius * (1.0 - 0.55 * squash)
    cx = size / 2.0
    cy = contact - ry

    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry],
                 fill=(214, 45, 45), outline=(20, 20, 20), width=4)
    return img


def render_procedural(plan: MotionPlan, canonical, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    states = {}
    for pose in plan.states():
        states.setdefault(canonical[pose.key], pose.state)
    images = {}
    for i, (key, state) in enumerate(states.items()):
        image = procedural_draw(state)
        image.save(out / f"{i:02d}_{key.replace('->', '_to_').replace('#', '_')}.png")
        images[key] = image
        print(f"  [{i+1}/{len(states)}] {key:24s} height={state['height']:.2f} "
              f"squash={state['squash']:+.2f}", flush=True)
    return images


def contact_sheet(images, cols: int = 5, pad: int = 6) -> Image.Image:
    n = len(images)
    rows = (n + cols - 1) // cols
    w, h = images[0].size
    canvas = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad),
                       (30, 30, 34))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        canvas.paste(img, (pad + c * (w + pad), pad + r * (h + pad)))
    return canvas


def write_outputs(frames, out: Path, stem: str):
    out.mkdir(parents=True, exist_ok=True)
    gif = out / f"{stem}.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0)
    print(f"  wrote {gif}", flush=True)
    try:
        from moviepy.editor import ImageSequenceClip
        mp4 = out / f"{stem}.mp4"
        ImageSequenceClip([np.array(f) for f in frames], fps=FPS).write_videofile(
            str(mp4), fps=FPS, codec="libx264", audio=False, logger=None
        )
        print(f"  wrote {mp4}", flush=True)
    except Exception as exc:  # a codec problem should not lose the whole run
        print(f"  (mp4 skipped: {exc})", flush=True)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/sequence")
    parser.add_argument("--no-compare", action="store_true", help="Skip the legacy chained render")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--renderer", choices=["procedural", "diffusion"], default="procedural",
                        help="procedural draws each state directly (fast, legible). diffusion "
                             "uses tiny-sd, which cannot currently express these states.")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plan = build_plan()
    order, canonical, prompts = canonical_states(plan)
    sheet = build_sheet(order, canonical, prompts, plan)
    sheet.validate()

    print("\n=== THE PLAN ===")
    print(f"poses authored      : {len(plan.poses)}")
    print(f"states in the plan  : {plan.drawing_count()}")
    print(f"same prompt merged  : {plan.drawing_count() - len(order)}")
    print(f"distinct drawings   : {len(order)}")
    print(f"exposure sheet      : {sheet.total_frames} frames @ {FPS}fps "
          f"({sheet.duration_seconds:.2f}s)")

    if args.renderer == "procedural":
        print("\n=== PROCEDURAL RENDER (stand-in drawing model) ===")
        t0 = time.time()
        authored = render_procedural(plan, canonical, out / "drawings_procedural")
        authored_time = time.time() - t0
        write_outputs(frames_from_sheet(sheet, authored), out, "bounce")
        print("\n=== EXPOSURE SHEET ===")
        print(sheet.to_ascii())
        print()
        print(f"distinct drawings   : {len(order)}")
        print(f"frames              : {sheet.total_frames}")
        print(f"frames held/reused  : {sheet.total_frames - len(order)}")
        print(f"on-screen drawings  : {len(order) / sheet.duration_seconds:.1f}/sec")
        cs_img = contact_sheet([authored[k] for k in order])
        cs_img.save(out / "contact_sheet.png")
        print(f"\nDONE  ({authored_time:.1f}s)")
        return

    renderer = Renderer(args.device)

    print("\n=== AUTHORED RENDER (INV-1 compliant) ===")
    t0 = time.time()
    authored = render_authored(renderer, order, prompts, out / "drawings_authored")
    authored_time = time.time() - t0

    frames = frames_from_sheet(sheet, authored)

    print("\n=== EXPOSURE SHEET ===")
    print(sheet.to_ascii())
    print()
    report = sheet.cost_report()
    print(f"frames              : {report['frames']}")
    print(f"generation calls    : {report['generation_calls']}")
    print(f"frames held/reused  : {report['held_or_reused_frames']} "
          f"({report['reduction_vs_on_ones']:.0%} fewer generations than on ones)")
    print(f"wall clock          : {authored_time:.0f}s")
    write_outputs(frames, out, "bounce_authored")

    cs = out / "contact_sheet_authored.png"
    cs_img = contact_sheet([authored[k] for k in order])
    cs_img.save(cs)
    print(f"  wrote {cs}", flush=True)

    if not args.no_compare:
        print("\n=== LEGACY CHAINED RENDER (INV-1 violation, for comparison) ===")
        renderer.calls = 0
        t0 = time.time()
        chained = render_chained(renderer, order, prompts, out / "drawings_chained")
        print(f"  wall clock        : {time.time() - t0:.0f}s")
        write_outputs(frames_from_sheet(sheet, chained), out, "bounce_chained")

        cs2 = out / "contact_sheet_chained.png"
        cs2_img = contact_sheet([chained[k] for k in order])
        cs2_img.save(cs2)
        print(f"  wrote {cs2}", flush=True)

        side = Image.new(
            "RGB",
            (max(cs_img.width, cs2_img.width), cs_img.height + cs2_img.height + 24),
            (30, 30, 34),
        )
        side.paste(cs_img, (0, 0))
        side.paste(cs2_img, (0, cs_img.height + 24))
        side.save(out / "comparison.png")
        print(f"  wrote {out / 'comparison.png'}  (top = authored, bottom = chained)",
              flush=True)

    print("\nDONE")


if __name__ == "__main__":
    main()
