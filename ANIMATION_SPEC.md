# ANIMATION_SPEC.md

**Status: LOCKED (v1.0.0).** This document is normative. `animation_spec.py` is its
executable form and `tests/test_animation_spec.py` is its enforcement arm. A change
to the doctrine below requires editing this file in the same commit; a change to
`production_system.py` that contradicts it is a bug, not a preference.

The long-form reasoning behind this contract lives in [plan.md](plan.md). Read that
for *why*. This file is *what*, and it is binding.

---

## 0. The one-sentence doctrine

> You are designing a sequence of discrete visual states, and deciding exactly when
> the world is allowed to change.

Frame-by-frame animation is **not** "make a bunch of pictures and play them
quickly." Everything below follows from taking that sentence literally.

The corollary that matters most for this codebase:

> A video model asks *"what image comes next?"*
> This system asks *"what should happen, what pose communicates it, how long do I
> show it, and should I even bridge it or cut?"*

---

## 1. Normative invariants

These are MUST / MUST NOT. `ExposureSheet.validate()` enforces what is mechanically
checkable.

### INV-1 — No autoregressive drawing chains

**MUST NOT** produce a drawing by feeding the previously drawn frame back into a
generator as its source.

Drawings come from exactly one of:

| Source | Meaning | Cost |
| --- | --- | --- |
| `AUTHORED` | Generated from a declared state. Independent of every other drawing. | 1 generation |
| `ANCHORED` | Pulled toward a **fixed root** `AUTHORED` drawing for continuity. | 1 generation |
| `REUSED` | The same drawing shown again (on twos, a hold, a cycle). | 0 generations |
| `CHAINED` | **Forbidden.** Derived from the immediately preceding drawn frame. | — |

`DrawingSource.CHAINED` raises `AutoregressiveChainError`. So does the loophole: an
`ANCHORED` drawing whose anchor is itself `ANCHORED`. Anchoring must reach an
`AUTHORED` root in one hop, or it is a chain wearing a disguise.

*Why this is the hardest rule:* chaining feels like it is maintaining continuity. It
is not. It is accumulating drift. The failure is
`character → slightly different → different person → texture soup`, and it is a known,
actively-researched problem in autoregressive video generation — see
[BAgger: Backwards Aggregation for Mitigating Drift in Autoregressive Video Diffusion Models](https://browse-export.arxiv.org/pdf/2512.12080).

### INV-2 — Timing is authored, never implied

**MUST NOT** derive per-segment frame counts by dividing a duration by a segment
count. Frame counts come from an exposure sheet.

### INV-3 — Stillness is a first-class citizen

A hold is a normal, expressible construct. A hold of N frames is **one** drawing and
**zero** generation calls. Animation language includes held poses, snaps, and cuts,
not only motion.

### INV-4 — No universal clock

Different layers **MAY** run at different rates. `Layer.clock` is a `Fraction` of real
time: `1` = real time, `1/2` = half rate, `1/3` = third rate, `0` = static.

```
background(t)        clock = 0     never changes
body(t)              clock = 1
face(t / 2)          clock = 1/2
eyes(event-driven)   explicit holds at explicit frames
effects(t / 3)       clock = 1/3
```

A sheet where every layer shares one clock is legal but is flagged as a warning —
limited animation is where this technique lives.

### INV-5 — The model draws states, not frames

The number of generation calls equals the number of **distinct drawn states**, not the
number of output frames. `ExposureSheet.render_plan()` returns exactly that list.
`ExposureSheet.cost_report()` quantifies the saving.

---

## 2. Vocabulary → code

| Traditional animation | Code |
| --- | --- |
| key pose | `KeyPose(role=PoseRole.KEY)` |
| breakdown | `KeyPose(role=PoseRole.BREAKDOWN)` — authored, not computed |
| inbetween | `PoseState(role=PoseRole.BREAKDOWN)` — generated between two poses |
| exposure sheet | `ExposureSheet` |
| cel layer | `Layer` |
| held frame | `Hold` |
| on ones / twos / threes | `ExposureTrack.on_ones / .on_twos / .on_threes` |
| mixed exposure | `ExposureTrack.mixed([("A", 2), ("B", 1), ...])` |
| dope sheet | `ExposureSheet.to_ascii()` |
| "should I cut?" | `inbetweens=0` |

### The pipeline

```
POSE PLAN  (keys + authored breakdowns, with motion character)
    ↓
STATE SEQUENCE  (sparse: ~7 states, not 24)
    ↓
EXPOSURE  (drawings + holds, per layer, per clock)
    ↓
DRAWINGS  (render_plan: only what must actually be generated)
    ↓
FRAME SEQUENCE  (exposure applied at fps)
    ↓
VIDEO
```

`IMAGE_t → IMAGE_(t+1)` appears nowhere in that diagram. That is the point.

### Motion character

`Easing` describes *how* a movement happens; it is data, not a default.

| Easing | Reads as |
| --- | --- |
| `LINEAR` | mechanical, constant rate |
| `ACCELERATE` | gaining momentum |
| `DECELERATE` | arriving, slowing |
| `SMOOTH` | ease in and out |
| `SNAP` | no visible travel — a cut |
| `OVERSHOOT` | passes the target, settles back |
| `ANTICIPATE` | winds up backwards first |

`OVERSHOOT` and `ANTICIPATE` deliberately sample outside `[0, 1]`. They are the formal
reason a smooth path between two poses is not sufficient, and the tests assert that
they actually do leave the range.

**Authored breakdowns beat computed inbetweens.** An animator's breakdown asks "what
*physical state* should exist here?" — an elbow leading, a squash, an impossible
anatomy. `KeyPose(role=BREAKDOWN)` records that decision. `inbetweens=n` is the
fallback for when a curve genuinely is what you want.

**Stillness is part of the language.** `hold=n` is how a pose is allowed to sit. A
shot that is all motion is not more animated; it is less.

---

## 3. Model decision record

### 3.1 FAR — CONDITIONALLY REJECTED for the drawing stage

**What it is:** *FAR: Frame Autoregressive Model for Both Short- and Long-Context
Video Modeling* ([arXiv 2503.19325](https://arxiv.org/abs/2503.19325),
[weights](https://huggingface.co/guyuchao/FAR_Models)). Small (≈130M-class), predicts
the next frame from an autoregressive context, and reports state-of-the-art video
generation and prediction on **UCF-101** from a single unconditional pretraining run.

**Decision: it MUST NOT be used as the drawing generator.** FAR's objective is
literally next-frame prediction. Adopting it as `drawing_N ← drawing_(N-1)` is a
direct INV-1 violation, and it reinstates the precise drift cascade this spec exists
to eliminate. A 130M UCF-101 research model is also not a general-purpose image
animator: its priors are 101 action classes of short real-world video, not authored
character art.

**Where it is conditionally acceptable:** as a **motion/driver prior** — a source of
suggested *timing and trajectory* that feeds the pose plan, never pixels that feed the
next drawing. Any such use must still pass through `MotionPlan` and produce states,
not chained frames.

> Pending verification before reconsideration: parameter count and the exact
> short/long-context variants, which are stated here from the model card and the
> project's own notes rather than measured locally.

### 3.2 MobileI2V — CONDITIONAL, narrow

**What it is:** *MobileI2V: Fast and High-Resolution Image-to-Video on Mobile Devices*
([arXiv 2511.21475](https://arxiv.org/abs/2511.21475),
[code](https://github.com/hustvl/MobileI2V)). Small image-to-video model.

**Decision: not the animation architecture.** MobileI2V turns one image into one
continuous video on one universal clock. That conflicts with INV-2 (authored timing),
INV-3 (stillness), and INV-4 (multiple clocks) simultaneously. It smooths; this
system's whole advantage is permission *not* to be smooth.

**Where it is acceptable:** as a bounded **bridge segment** fallback when a single
exposure segment must be filled and no other path exists. It may fill a segment; it
may not define the sequence.

### 3.3 Hardware gate

Both decisions above are **conditional on the target GPU and VRAM**, which have not
been specified. The conditions are recorded, not resolved. If a machine profile is
supplied, this section is where the resolved answer goes.

---

## 4. What changed in the pipeline

The legacy path in [production_system.py](production_system.py) violated the
invariants in three specific places:

| Location | Violation | Now |
| --- | --- | --- |
| `produce()` keyframe loop, `img2img(keyframes[-1])` | INV-1 — each keyframe chained off the previous, with histogram matching to a root as a patch | `--strict-timing` anchors every keyframe to the shot's root (`keyframes[0]`) instead. Default mode warns once per shot and still chains. |
| `assemble()`, `segment_duration = duration / (len(keyframes) - 1)` | INV-2 — uniform segments, timing implied by arithmetic | Frame counts come from the sheet; `shot["exposure"]` authors them. |
| `assemble()`, pad / trim to `total_frames_needed` | INV-2 — timing fixed up after the fact | `ExposureSheet.fit_to_frames()` hits the audio duration exactly, up front. A residual mismatch now raises instead of being papered over. |
| `assemble()`, every segment generated | INV-3 — a held pose still cost a full generation pass | The terminal segment is a true hold: zero generation calls. |
| one track for everything | INV-4 — one universal clock | `ExposureSheet`/`Layer` support per-layer clocks; the legacy single-track path declares one layer and is unchanged. |

### Migration policy (deliberately non-breaking)

`Assembler.assemble()` now builds an `ExposureSheet` per shot.

- **Authored timing** (`shot["exposure"]`) is honoured when present.
- **Absent authored timing**, legacy uniform behaviour is reproduced *exactly* and a
  single warning is emitted per shot — the door is open, nothing is silently changed.
- **`strict_timing=True`** escalates a `CHAINED` drawing from a warning to a hard
  `AutoregressiveChainError`. This is the target state; it is off by default only so
  that existing projects keep rendering.

Phase 1 reports. Phase 2 enforces. This spec describes the contract; the flag
describes how far the codebase has got.

`--strict-timing` is the switch between the two phases. It affects both the keyframe
loop (INV-1: anchor to root, not to the previous frame) and assembly (INV-2: raise on a
timing violation rather than warn). It is off by default so that existing projects keep
rendering; turning it on is the intended end state.

---

## 5. Acceptance criteria

The spec is satisfied when all of the following hold:

1. `python -m pytest` passes, including every invariant test.
2. For a shot with `n` output frames and `d` drawings, exactly `d` generation calls
   are made, not `n`. `cost_report()["generation_calls"] == d`.
3. Total assembled frames equal `round(audio_duration * fps)` before concatenation,
   with no pad and no trim step.
4. No code path constructs a `Drawing` with `DrawingSource.CHAINED`.
5. A shot may declare at least two layers on different clocks and render correctly.
6. `strict_timing=True` renders the same output for a spec-conformant shot as
   `strict_timing=False`.

---

## 6. What this spec does not claim

- It does not claim smoothness is bad. It claims smoothness must be *chosen*.
- It does not claim the drawing model is solved. The drawing model is exactly as good
  as your generator; this spec constrains *when it is called and what it is called
  with*, which is where the drift was coming from.
- It does not claim FAR or MobileI2V are bad models. It claims neither is the
  architecture described here, for reasons stated above.

### 6.1 The state → prompt bridge is the real bottleneck

Surfaced by actually rendering a sequence (`generate_sequence.py`), not by theory.

The spec models animation as discrete states with continuous channels (`height`,
`squash`, `arm`, ...). A text-conditioned drawing model cannot consume those. The
state must be **quantised into descriptive bands** before it can be drawn:

```
squash = -0.22  ->  "ball slightly stretched vertically"
height =  0.80  ->  "ball in the upper part of the frame"
```

This quantisation is lossy, and the loss is not cosmetic. With bands that were too
coarse, `apex_high` (height 1.00) and `apex_low` (height 0.78) both became
*"ball near the top of the frame"*. `canonical_states()` then correctly observed that
two states describing the same picture are the same cel, merged them, and generated
one drawing — which silently **destroyed the bounce**, because the whole point of a
bouncing ball is that the second apex is lower.

Three consequences worth carrying forward:

1. **Band resolution is an animation parameter, not a formatting detail.** Too coarse
   and distinct states collapse; too fine and the model ignores the distinction anyway.
2. **Merging must be reported, never silent.** `canonical_states()` prints how many
   states it merged. A silent merge is a motion bug that looks like a saving.
3. **This bottleneck is what a pose- or state-conditioned drawing model would fix.**
   Until then, the achievable temporal resolution of the system is capped by the
   descriptive vocabulary of its prompts, not by the exposure sheet.

### 6.2 Measured: the current drawing model cannot express states at all

Finer bands were not enough, and this is the more serious result. `tiny-sd` was
calibrated directly (`output/calib/grid.png`, 256², 12 steps):

| Prompt | Rendered |
| --- | --- |
| `a red ball, on a plain white background` | a red ball |
| `a red ball stretched tall and thin` | a **round** ball |
| `a red ball squashed wide and flat` | a ball with a spurious dark hole |
| `a red ball high up near the top of the frame` | a centred ball |
| `a red ball near the bottom of the frame` | a centred ball |
| `an elongated red oval, tall and narrow` | texture noise |

It renders *a red ball* and ignores every modifier. Shape and position control do not
exist in this model. So the 8-drawing diffusion render
(`output/sequence/contact_sheet_authored.png`) is 8 near-identical balls, and the
legacy chained render is the same with progressive darkening — drift, visible, but
drift of an image that was never the right image to begin with.

**Conclusion.** The exposure-sheet machinery is sound and proven end-to-end, but no
legible animated sequence can be produced with the currently available drawing model.
The spec's architecture is not the blocker; the renderer is.

This is why `generate_sequence.py` defaults to `--renderer procedural`: it draws each
state directly, which isolates what the spec actually governs — *which* distinct states
exist, in what order, and for how many frames each is exposed. `bounce.gif` and
`filmstrip.png` are that claim demonstrated: 7 authored poses, 9 states, 8 drawings,
28 frames, with a hard squash held 3 frames and two apexes held 8 frames each.

Swapping in a state- or pose-conditioned generator is the single highest-value change
available to this project.

The correctness of the merge logic is not in question; the resolution of the bridge is.
That is a drawing-model problem, and no amount of timing discipline will solve it.
