"""
handheld.py -- procedural camera shake.

People do not hold cameras steady. Adding a little organic drift to every shot is
the cheapest thing that makes AI video stop looking synthetic, and it needs no
model: it is a per-frame sub-pixel transform of ONE frame.

Why procedural rather than a motion LoRA:
  * a LoRA costs a generation slot and has one fixed character
  * this costs nothing, applies to any clip, and is tunable per shot
  * it is not warping between images -- each frame is transformed on its own


WHAT WAS WRONG WITH THE FIRST VERSION
-------------------------------------
Measured on the production call (amount=5.0, drift=0.8, jitter=0.3, seed=3, 31
frames), before this rework:

  * a BINARY envelope. `occasional_envelope` stepped the multiplier from 0 to
    full, and on that seed fired on 5 of 31 frames (16%), not the ~32% claimed.
    Max |d2| jerk was 2.835 px/frame^2 against 0.473 with no envelope at all --
    the envelope multiplied the jerk by 6x. This is what read as a jerk rather
    than a held camera.
  * the crop was INTEGER. `int(cx)` quantised the frame position, so the whole
    31-frame clip resolved to 4 distinct positions. Sub-pixel camera motion was
    not representable, so even a perfect envelope would have stepped.
  * the octaves were PER CLIP, not per second. `_value_noise` fixed the knot
    count from `freq` and ignored `n`, so cycles/clip was always 1, 3, 6: the
    same recipe ran at 0.52/1.55/3.10 Hz on a 31-frame clip and 0.28/0.83/1.66 Hz
    on a 58-frame one. `make_film.render` then stretched 31 frames to the
    shot's hold (0.94x-1.86x on the `shadows` film), shifting every band down
    again. No layer had a real frequency.
  * roll existed but was derived as `amount * 0.12` -- a PIXEL quantity setting a
    DEGREE quantity -- and was gated by the same binary envelope.


WHAT THIS VERSION DOES
----------------------
Bands are specified in Hz at the DELIVERED frame count (`spec.fps`, the exposure
sheet's timeline), so a band means the same thing on a 29-frame shot and a
58-frame one. The envelope is continuous and never reaches zero. Roll is its own
per-shot quantity. One projective resample per frame replaces the
resize+rotate+integer-crop chain, so motion is sub-pixel and there is exactly one
interpolation instead of two.

The amplitude envelope is applied to drift and sway; the tremor band keeps a much
higher floor, and an uncorrelated sub-pixel dither guarantees no two delivered
frames are ever byte-identical -- a true freeze is what makes a shot read as
mechanical.


TWO HONEST LIMITS
-----------------
1. Nyquist. Delivery is 16 fps (HANDOFF: `FPS = 16`, RAFT x2 of an 8 fps base),
   so the highest representable band is 8 Hz. Physiological hand tremor is
   6-12 Hz and therefore CANNOT be carried as a band here. Unity's Cinemachine
   guidance says the same thing from the other side: a frequency above the frame
   rate "falls between the cracks of the Nyquist rate" and reads as choppy, and
   the captured real-shake libraries are 24 fps for the same reason. At 16 fps an
   8-12 Hz tremor aliases; that is faithful rather than a compromise, because
   real 16 fps footage aliases it too, so the aliased remnant is modelled
   explicitly as the uncorrelated dither. Bands above ~4.2 Hz are not used: at
   2-4 frames per cycle they stop reading as tremor and start reading as
   vibration.
2. No parallax. This is a 2D homography on already-generated frames, so a
   rotation can tilt the image but cannot reveal anything behind anything.
   Rotation plus a projective term is the ceiling without depth.

The projective (keystone) term is included but is second-order: at realistic
sub-degree yaw/pitch it produces a differential of ~k*u^2, i.e. ~1.4 px between
frame edge and centre for a 0.4 deg yaw, on top of the uniform shift the same
rotation causes. It is NOT the reason the old shake read as a slide -- the
envelope and the integer crop were. Set `perspective=0.0` in a spec to drop it.


WHAT THE RESEARCH SAYS, AND WHERE IT DISAGREES WITH THE BRIEF
-------------------------------------------------------------
Bands. Cinemachine's guidance is the best-documented parameter set found: three
noise layers per axis, low 0.1-0.5 Hz, mid 0.8-1.5 Hz, high 3-4 Hz, rotation
noise before position noise. The bands here follow that structure.

Amplitude, from TRAJECTORY data. The Blender Camera Shakify library is captured
rather than synthesised, and is the only real handheld trajectory data located:
rotation swings of ~0.5 deg peak-to-peak on one capture and ~4.3 deg on a rougher
one, roll swings of 0.6-1.0 deg p-p. CIPA DC-X011-2024's standard shake waveform,
built from many subjects at 500 Hz, peaks at ~+-0.4 deg on pitch. BASE_ROLL_DEG
sits between those; the delivered shots land at roll 0.62-0.73 deg p-p
(captured range 0.6-1.0). A held camera's residual is never zero -- a
video-analysis patent (US20100189410A1) builds its handheld-vs-tripod classifier
on exactly that premise -- and three operators independently say a fully static
shot "stands out". So the continuous envelope is the phenomenon, not a preference.

**The amplitude is calibrated and comes out marginally conservative.** CIPA's
*average vibration angle* figures look alarming (1.06-3.36 deg) but they are
exposure-time dependent -- 0.019-0.039 deg at 1/60 s, implying a roughly constant
1-3.4 deg/s -- so they describe a long exposure, not a trajectory across a shot.
The applicable comparators are the measured amplitude LAWS, and the model sits
inside them: pooled roll rms 0.086-0.226 deg over the six `shadows` shots against
Xiao 2006's sigma = a*T^0.5632 = 0.129-0.319 deg (a = 0.092-0.155) at the same
durations, i.e. 0.67-0.71x the LOWEST measured estimate and never above it.
Panasonic's 10-subject measurement gives a mean yaw of 0.11 deg at 1/8 s,
distributing to 0.42 deg, and Canon JP2008139640A -- the only explicit stabiliser
design premise found -- dimensions its correction angle against 0.2 deg at 1 Hz.
Both bracket the model, so a modest increase is defensible and an
order-of-magnitude one is not.

There is also a hard reason not to go much bigger: **<=0.05 deg of total rotation
in a normal exposure keeps motion blur under ~1 px at any focal length**, and at
this amplitude the per-frame motion is tenths of a pixel, so the frames stay sharp
WITHOUT motion blur -- which is what a 2D warp can actually deliver. Multi-degree
shake would need per-frame motion blur to look photographic, and a projective
resample cannot produce it.

Rotation versus translation is a DISTANCE THRESHOLD, not a ratio, and that is the
one measured rule that says how framing should change the character. Rotation
displaces the image by theta*f, independent of distance; translation by d*f/Z,
falling off as 1/Z. Han et al. measured the crossover at ~100 cm and CIPA puts it
at ~20x focal length, with translation dominant only near 1:1 macro. ROTATION_SHARE
applies that: near the lens the lateral channel carries more of the jitter, far
from it roll does. Note the lateral channel is not "the operator sliding sideways"
-- to first order it IS yaw/pitch (f*tan(theta)) plus body translation, which
cannot be separated without depth.

Three corrections to the brief this replaces.
  * "Pure translation, no rotation" was NOT true of the old code. It had roll,
    but derived from a pixel amplitude and gated by the same binary envelope. The
    defect was the envelope and the integer crop, not the absence of roll.
  * "Roll is the primary cue that makes a frame read as handheld" is NOT
    supported by anything found. Cinemachine's rationale is that handheld shake
    is "more rotationally than positionally ... because that's where the camera is
    aiming" -- rotation in general, not roll -- and its own seven handheld presets
    set the roll channel to ZERO, with roll appearing only in the engine/rig
    presets. Roll is included because the captured data contains it, not because
    anything measures it as the discriminator. CIPA is the closest thing to
    support: roll is the only blur component that does NOT grow with focal length,
    and it is 23% of combined blur at 24 mm against 3% at 200 mm -- a wide-angle
    effect.
  * "6-12 Hz tremor as one of three bands" is not deliverable at 16 fps; see the
    Nyquist limit above. It is also the wrong number: the widely repeated "hands
    shake at 10-20 Hz" traces to a manufacturer whitepaper with no citation, and
    disagrees with the physiology (8-13 Hz overall, and 7-10 Hz at the wrist,
    which is what grips a camera).

Deliberate moves. No practitioner source states that an operator tightens or
loosens the hold for a pan or zoom, so the damping here is an INFERENCE, not a
finding. It is supported in direction by a MEASURED concentration effect -- RMS
tremor roughly halved as accuracy demand rose, over 6 subjects -- and by the fact
that a deliberate move is already motion. It is contradicted twice over: the same
study notes the opposite result for unilateral postural targeting, so the sign is
contested, and the head has its own pathology (stiction breakaway, overshoot on
small moves) which this model does not attempt.

Framing scale. Operators brace harder when tight, and at the long end the answer
becomes rigging rather than technique -- the practitioner list for long lenses
contains no handheld technique at all. Physics says a tighter frame shows more
displacement for the same angle; the bracing says less. The scale gain is
deliberately modest because the two partly cancel.

What this model does NOT implement, and why.
  * **Walking.** DXOMARK measures a walking operator at +-2 to +-3 deg per axis
    with peaks of 10-20 deg/s, i.e. 5-7x the standing figures here. That is the
    single largest factor found and it is not modelled, because operator
    locomotion is not something the shot list expresses: a walking SUBJECT does
    not imply a walking OPERATOR. Implementing it needs an explicit operator field
    on the shot, not an inference from the subject's motion.
  * **Operator-to-operator spread.** Normal tremor magnitude spans roughly a
    30-fold range between individuals. There is one per-shot seed here, not a
    per-operator character.
  * **A PSD slope.** The "camera shake falls off as 1/f^2" claim could not be
    traced to any measurement, so the band mix is drawn per shot from the ranges
    below rather than derived from a spectral exponent. No source publishes a fit.
  * **A 1/focal-length shutter rule.** Also untraceable; CIPA's measured
    substitute is a 200 um blur / 20 um determination level reported in stops.

Sources. Full list in HANDOFF section 6.1, which is the authoritative copy:
Cinemachine noise profiles; CIPA DC-X011-2024; DXOMARK stabilisation measurements;
Han et al. IEEE TMC 2020; Camera Shakify captured data; US20100189410A1; the
cinematography.net operator threads (Kreines, Adams, Heller, Gallant, Drasin,
Reichline); Foundry Nuke CameraShake defaults.

    python handheld.py                # the before/after A/B on real footage
    python handheld.py --spec         # print a spec per framing scale
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# the physical model
# --------------------------------------------------------------------------- #
#
# Band structure follows Cinemachine's documented guidance -- three layers per
# axis, low 0.1-0.5 Hz, mid 0.8-1.5 Hz, high 3-4 Hz -- shifted to sit inside this
# pipeline's 8 Hz Nyquist limit. Amplitudes come from the captured real handheld
# data (Camera Shakify, 24 fps): roll 0.6-1.0 deg peak-to-peak, rotation ~0.5 deg
# on a calm capture and ~4.3 deg on a rough one. Sources in the module docstring
# and HANDOFF section 6.

BANDS = {                       # Hz, inclusive
    "drift": (0.25, 0.60),      # body weight shifting; sub-cycle over a shot
    "sway": (0.90, 2.20),       # breathing + hand correction + step rate
    "tremor": (2.60, 4.20),     # below the 16fps vibration threshold
}

BAND_FLOOR = 0.05               # a band thinner than this cannot be built at
                                # this length; its weight moves to the trend

# The drift band has no bin at these shot lengths, so drift is a TREND: a lean
# through a few smoothstep knots. The knot RATE is per second, so a 2 s shot and
# a 4 s shot lean at the same rate -- the first version's knot count was fixed
# per clip, which is why the same recipe ran twice as slow on a longer shot.
DRIFT_KNOT_RATE = 0.7           # knots per second ~ 0.35 Hz

# per-shot amplitude, before the per-shot randomisation
BASE_PX = 3.2                   # peak lateral excursion, output pixels
BASE_ROLL_DEG = 0.42            # peak roll. CIPA DC-X011-2024's standard shake
                                # waveform (built from many subjects, 500 Hz
                                # records) peaks at ~+-0.4 deg on pitch; the
                                # captured Camera Shakify roll swings are 0.6-1.0
                                # deg p-p. This sits between them.
SCALE_GAIN = {"wide": 1.00, "medium": 1.00, "close": 1.10, "macro": 1.25}
KIND_GAIN = {"none": 1.00, "subject": 0.85, "camera": 0.50}
DURATION_GAIN = (0.85, 1.35)    # clamped linear in the shot's seconds

# Rotation versus translation is NOT a fixed ratio -- it is a DISTANCE THRESHOLD,
# and that is the one measured rule that says how a shot's framing should change
# the camera's character. Rotation displaces the image by theta*f, independent of
# subject distance; translation displaces it by d*f/Z, falling off as 1/Z. Han et
# al. (IEEE TMC 2020) measured the crossover at ~100 cm and CIPA states it as ~20x
# focal length, with translation dominant only close to 1:1 macro. Their measured
# pixels: 10 deg of rotation moves 314/235/230 px at 10/50/150 cm depth, while 5 cm
# of translation moves 650/130/43 px at the same depths.
#
# So: near the lens, translation carries the jitter; far from it, rotation does.
# (translation multiplier, roll multiplier), applied on top of SCALE_GAIN.
ROTATION_SHARE = {"wide": (0.88, 1.12), "medium": (1.00, 1.00),
                  "close": (1.12, 0.90), "macro": (1.25, 0.78)}

ENVELOPE_FLOOR = (0.20, 0.35)   # never zero -- a switched-off camera is the jerk
ENVELOPE_SLEW = 0.07            # max |d env| per frame; the binary envelope hit 0.81
TREMOR_FLOOR = 0.80
DITHER_PX = (0.08, 0.18)        # uncorrelated, sub-pixel, always present

# for the keystone: k = tan(angle)/focal_px, so a sub-degree yaw is ~1e-5
FOCAL_PX_512 = 569.0            # ~40mm-equivalent on a 512px-wide frame


@dataclass(frozen=True)
class ShakeSpec:
    """One shot's hold character. Everything here is per-shot on purpose."""
    seed: int
    seconds: float
    fps: int = 16
    amount_px: float = BASE_PX          # peak lateral excursion
    roll_deg: float = BASE_ROLL_DEG     # peak roll
    mix: Dict[str, float] = field(default_factory=lambda: {
        "drift": 0.45, "sway": 0.38, "tremor": 0.17})
    roll_mix: Dict[str, float] = field(default_factory=lambda: {
        "drift": 0.35, "sway": 0.45, "tremor": 0.20})
    floor: float = 0.28                  # amplitude envelope floor, > 0
    tremor_floor: float = TREMOR_FLOOR
    dither_px: float = 0.18
    perspective: float = 1.0             # share of roll realised as keystone
    scale: str = ""
    kind: str = "none"

    @property
    def nyquist(self) -> float:
        return self.fps / 2.0


# --------------------------------------------------------------------------- #
# signal construction, in Hz
# --------------------------------------------------------------------------- #

def _band(n: int, lo: float, hi: float, seed: int, fps: int) -> np.ndarray:
    """Band-limited noise in [lo, hi] Hz, unit variance, zero mean.

    Built in the frequency domain so the band means Hz rather than "octaves of
    the clip", which is what the first version got wrong. Returns zeros if the
    band holds no bin at this length -- the caller moves that weight to the
    trend, because at 16 fps a 29-frame shot has a frequency resolution of only
    0.55 Hz.
    """
    if n < 4:
        return np.zeros(n)
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.normal(0, 1, n))
    f = np.fft.rfftfreq(n, d=1.0 / fps)
    df = fps / n
    # soft edges, so a band that is only 2-3 bins wide does not ring
    edge = max(df, 0.25)
    lo_w = np.clip((f - (lo - edge)) / edge, 0, 1)
    hi_w = np.clip(((hi + edge) - f) / edge, 0, 1)
    mask = np.minimum(lo_w, hi_w)
    mask[f < max(lo - edge, df * 0.5)] = 0.0
    mask[f > hi + edge] = 0.0
    if mask.sum() <= 0:
        return np.zeros(n)
    out = np.fft.irfft(spec * mask, n)
    sd = float(out.std())
    return out / sd if sd > 1e-9 else np.zeros(n)


def _trend(n: int, seed: int, knots: int = 3) -> np.ndarray:
    """A slow lean through `knots` random points, unit variance.

    The drift band (0.25-0.60 Hz) has no bin on a 2 s shot, so drift is a trend
    rather than a band. Within one shot that is the honest representation: the
    operator leans, and does not complete a cycle of body weight shift.
    """
    rng = np.random.default_rng(seed)
    k = max(2, knots)
    pts = rng.normal(0, 1, k)
    x = np.linspace(0, k - 1, n)
    i = np.floor(x).astype(int).clip(0, k - 2)
    t = x - i
    s = t * t * (3 - 2 * t)                     # smoothstep: C1 continuous
    out = (1 - s) * pts[i] + s * pts[i + 1]
    sd = float(out.std())
    return (out - out.mean()) / sd if sd > 1e-9 else np.zeros(n)


def _layer(n: int, spec: ShakeSpec, mix: Dict[str, float],
           seed: int) -> np.ndarray:
    """One axis, built from the bands. Unit variance overall."""
    w = {k: max(0.0, float(mix.get(k, 0.0))) for k in BANDS}
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}

    # fixed per-band seeds. NOT hash(): Python randomises str hashing per
    # process, so a hash-derived seed would make the shake irreproducible.
    band_seed = {"drift": 11, "sway": 23, "tremor": 37}

    comp = {}
    for k, (lo, hi) in BANDS.items():
        b = _band(n, lo, hi, seed + band_seed[k], spec.fps)
        if np.abs(b).max() < BAND_FLOOR:
            comp[k] = None                      # not buildable at this length
        else:
            comp[k] = b / max(1e-9, np.abs(b).max())

    # drift is the trend, at a knot RATE, so it means the same on any length
    knots = max(2, int(round(spec.seconds * DRIFT_KNOT_RATE)))
    trend = _trend(n, seed + 4242, knots)
    drift = trend
    if comp.get("drift") is not None:
        drift = 0.7 * trend + 0.3 * comp["drift"]

    out = w["drift"] * drift
    for k in ("sway", "tremor"):
        if comp.get(k) is None:
            out = out + w[k] * trend            # fall back to the slow lean
        else:
            out = out + w[k] * comp[k]
    out = out - out.mean()
    sd = float(out.std())
    return out / sd if sd > 1e-9 else np.zeros(n)


def _limit_slew(e: np.ndarray, floor: float, slew: float,
                rounds: int = 8, passes: int = 40) -> np.ndarray:
    """Smooth until |d env| <= slew, rescaling to [floor, 1] between rounds.

    The rescale is what makes this a loop rather than a filter: rescaling to the
    full range gives the slope back, so the two operations have to converge
    together. This is the difference between a continuous envelope and a switch.
    """
    ker = np.array([0.25, 0.5, 0.25])
    for _ in range(rounds):
        for _ in range(passes):
            if float(np.abs(np.diff(e)).max()) <= slew:
                break
            e = np.convolve(np.pad(e, 1, mode="edge"), ker, mode="valid")
        lo, hi = float(e.min()), float(e.max())
        if hi - lo > 1e-9:
            e = floor + (1.0 - floor) * (e - lo) / (hi - lo)
        if float(np.abs(np.diff(e)).max()) <= slew * 1.05:
            break
    return np.clip(e, floor, 1.0)


def _envelope(n: int, spec: ShakeSpec, seed: int) -> np.ndarray:
    """Continuous amplitude modulation in [floor, 1]. Never zero.

    A slow lean (the operator relaxing or tightening across the shot) carries
    most of the modulation, with a shallow breathing component. The result is
    slew-limited to ENVELOPE_SLEW, so no single frame can step the amplitude the
    way the binary envelope did -- max |d env| was 0.81 there and is <= 0.07 here.
    """
    lean = _trend(n, seed + 77, max(2, int(round(spec.seconds * DRIFT_KNOT_RATE))))
    breathe = _band(n, 0.5, 1.1, seed + 78, spec.fps)
    u = 0.88 * lean + 0.20 * breathe
    u = np.tanh(1.5 * u / max(1e-9, u.std()))
    e = spec.floor + (1.0 - spec.floor) * (0.5 + 0.5 * u)
    return _limit_slew(np.clip(e, spec.floor, 1.0), spec.floor, ENVELOPE_SLEW)


def shake_path(n: int, spec: ShakeSpec, fps: Optional[int] = None) -> Dict[str, np.ndarray]:
    """The camera's own path for a clip of n delivered frames.

    Returns dx, dy (output pixels), roll_deg, the keystone coefficients kx, ky,
    and the amplitude envelope. The tremor band is enveloped only down to
    `tremor_floor` and an uncorrelated sub-pixel dither is added on top, so no
    two frames are ever identical.

    Each axis is normalised so its PEAK equals the spec's amplitude, which makes
    `amount_px` and `roll_deg` mean what they say: the worst moment in the shot.
    The envelope then shapes where that worst moment falls and how much the hold
    varies, rather than silently shrinking the amplitude.
    """
    fps = int(fps or spec.fps)
    if spec.fps != fps:
        spec = replace(spec, fps=fps)
    rng = np.random.default_rng(spec.seed * 7919 + n)

    env = _envelope(n, spec, spec.seed)
    trem_env = spec.tremor_floor + (1.0 - spec.tremor_floor) * (
        (env - spec.floor) / max(1e-9, 1.0 - spec.floor))

    path: Dict[str, np.ndarray] = {"env": env, "tremor_env": trem_env}
    alias = spec.dither_px / 0.7979
    axis_seed = {"dx": 131, "dy": 262, "roll": 393}
    for axis in ("dx", "dy", "roll"):
        mix = spec.roll_mix if axis == "roll" else spec.mix
        amp = {"dx": spec.amount_px, "dy": spec.amount_px,
               "roll": spec.roll_deg}[axis]
        lay = _layer(n, spec, mix, spec.seed + axis_seed[axis])
        w_t = max(0.0, mix.get("tremor", 0.0)) / max(1e-9, sum(mix.values()))
        v = lay * ((1 - w_t) * env + w_t * trem_env)
        # the aliased 8-12 Hz tremor, delivered as uncorrelated frame-to-frame
        # jitter; this is also the guarantee that no two frames are identical
        scale = alias if axis != "roll" else spec.dither_px * 0.25 / 0.7979
        v = v + scale * rng.normal(0, 1, n) * trem_env
        v = v - v.mean()
        pk = float(np.abs(v).max())
        path[axis] = (v / pk * amp) if pk > 1e-9 else v
    path["roll_deg"] = path.pop("roll")

    # keystone: yaw/pitch, second-order. k = tan(theta)/focal_px, applied about
    # the principal point. Measured differential across the frame is ~k*u^2,
    # i.e. ~1.4px at the edge for a 0.4deg yaw -- real, but not the cue. The
    # layers are peak-normalised like the axes, or their peak would grow with
    # the clip length (unit std does not bound a maximum).
    def _peak(v):
        pk = float(np.abs(v).max())
        return v / pk if pk > 1e-9 else v
    yaw = np.radians(_peak(_layer(n, spec, spec.roll_mix, spec.seed + 909))) * spec.roll_deg
    pitch = np.radians(_peak(_layer(n, spec, spec.roll_mix, spec.seed + 910))) * spec.roll_deg
    path["kx"] = spec.perspective * np.tan(yaw) / FOCAL_PX_512
    path["ky"] = spec.perspective * np.tan(pitch) / FOCAL_PX_512
    return path


# --------------------------------------------------------------------------- #
# the transform: one projective resample per frame, sub-pixel
# --------------------------------------------------------------------------- #

def _coeffs(W: int, H: int, dx: float, dy: float, roll_deg: float,
            kx: float, ky: float, zoom: float):
    """PIL PERSPECTIVE coefficients mapping OUTPUT pixels to SOURCE pixels.

    `roll_deg` is the image content's rotation: positive tilts the frame
    clockwise on screen. The content is scaled about the centre by `zoom`
    (which is what buys the headroom), rotated, then translated; the keystone is
    applied in source space about the image centre, because a yaw or pitch
    compresses about the principal point, not about the corner.
    """
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    th = np.radians(roll_deg)
    c, s = np.cos(th), np.sin(th)
    # forward: source -> output
    F = np.array([
        [zoom * c, -zoom * s, dx + cx - (zoom * c) * cx + (zoom * s) * cy],
        [zoom * s, zoom * c, dy + cy - (zoom * s) * cx - (zoom * c) * cy],
        [0.0, 0.0, 1.0],
    ])
    M = np.linalg.inv(F)
    # keystone about the source centre
    K = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [kx, ky, 1.0]])
    T = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
    Ti = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    M = T @ K @ Ti @ M
    M = M / M[2, 2]
    return (float(M[0, 0]), float(M[0, 1]), float(M[0, 2]),
            float(M[1, 0]), float(M[1, 1]), float(M[1, 2]),
            float(M[2, 0]), float(M[2, 1]))


def _sample(coeffs, x, y):
    d = coeffs
    w = d[6] * x + d[7] * y + 1.0
    return ((d[0] * x + d[1] * y + d[2]) / w, (d[3] * x + d[4] * y + d[5]) / w)


def zoom_needed(W: int, H: int, path: Dict[str, np.ndarray],
                pad: float = 2.5) -> float:
    """Smallest zoom that keeps every corner safely inside the source.

    `pad` is required INWARD, not outward, and that distinction is the whole
    point. A mapped source coordinate merely inside [0, W-1] is NOT enough: the
    bicubic kernel reaches 2 px beyond the coordinate it is given, so a corner
    mapping to u = -1.4 still samples the fill colour. Getting this wrong put
    genuine magenta-test fill pixels into 93 frames of one shot and 1242 of
    another while every corner check "passed" -- and the fill is near-black, so
    it would have shipped as a dark smudge in the corners rather than as an
    obvious bug. `pad` therefore keeps the coordinate at least `pad` px inside
    the source on every side.

    Solved per clip from the clip's OWN peak offsets, so a clip that barely moves
    is not needlessly zoomed and softened. Raises rather than returning a zoom
    that would expose a border.
    """
    dx, dy = path["dx"], path["dy"]
    roll = path["roll_deg"]
    kx, ky = path["kx"], path["ky"]
    adx, ady = float(np.abs(dx).max()), float(np.abs(dy).max())
    akx, aky = float(np.abs(kx).max()), float(np.abs(ky).max())
    corners = [(0.0, 0.0), (W - 1.0, 0.0), (W - 1.0, H - 1.0), (0.0, H - 1.0)]
    z = 1.0
    while z < 2.0:
        ok = True
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sr in (float(roll.min()), float(roll.max())):
                    d = _coeffs(W, H, sx * adx, sy * ady, sr, akx, aky, z)
                    for (x, y) in corners:
                        u, v = _sample(d, x, y)
                        if not (pad <= u <= W - 1 - pad and
                                pad <= v <= H - 1 - pad):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return z
        z += 0.0005
    raise RuntimeError(
        f"no zoom under 2.0 keeps a {W}x{H} frame inside the source for "
        f"dx +-{adx:.2f} dy +-{ady:.2f} roll {roll.min():.2f}..{roll.max():.2f} "
        f"keystone {max(akx, aky):.2e} -- the shake is larger than the frame can "
        f"absorb; reduce the amplitude rather than weakening the border margin")


def apply_handheld(frames: Sequence[Image.Image], spec: Optional[ShakeSpec] = None,
                   fps: Optional[int] = None,
                   resample: int = Image.BICUBIC) -> List[Image.Image]:
    """Apply one shot's shake to one shot's frames.

    Frames must already be at the DELIVERED length: call this after the exposure
    sheet's stretch, not before it, or the bands are built against a timeline
    that is about to be rescaled (that was the first version's second bug).

    One projective resample per frame. No margin parameter: the zoom is derived
    from the clip's own peak offsets, so there is no fixed headroom to be wrong
    about and no integer crop to quantise the motion.
    """
    n = len(frames)
    if n == 0:
        return []
    W, H = frames[0].size
    if spec is None:
        spec = spec_for_shot("", "none", n / float(fps or 16), seed=0,
                             fps=int(fps or 16))
    path = shake_path(n, spec, fps)
    z = zoom_needed(W, H, path)
    out = []
    for i, f in enumerate(frames):
        coeffs = _coeffs(W, H, float(path["dx"][i]), float(path["dy"][i]),
                         float(path["roll_deg"][i]), float(path["kx"][i]),
                         float(path["ky"][i]), z)
        out.append(f.convert("RGB").transform(
            (W, H), Image.PERSPECTIVE, coeffs, resample=resample))
    return out


# --------------------------------------------------------------------------- #
# the delivered timeline
# --------------------------------------------------------------------------- #
#
# The shake has to be built against the timeline the AUDIENCE sees. The film
# generates 16 frames, RAFT-upsamples to 31, and then the exposure sheet holds
# each shot for `round(seconds * fps)` delivered frames (29-58 on `shadows`).
# Building bands against the 31-frame clip and rescaling afterwards -- what the
# first version did -- moves every designed frequency by up to 1.86x, so no band
# was ever the frequency it claimed. These two helpers are the single definition
# of that timeline, shared by `make_film.py` and the A/B harness.

def hold_frames(shot, fps: int = 16) -> int:
    """The exposure sheet's hold for one shot, in delivered frames."""
    return max(1, round(shot.seconds * fps))


def stretch_to(frames: Sequence[Image.Image], length: int) -> List[Image.Image]:
    """Resample a clip to `length` delivered frames by index.

    Uses exactly the mapping `make_film.render` uses, so applying it early makes
    render's own resample the identity and the hold the only timeline.
    """
    if len(frames) < 2 or length == len(frames):
        return list(frames)
    idx = [round(k * (len(frames) - 1) / max(1, length - 1)) for k in range(length)]
    return [frames[j] for j in idx]


# --------------------------------------------------------------------------- #
# per-shot character
# --------------------------------------------------------------------------- #

def spec_for_shot(scale: str = "", kind: str = "none", seconds: float = 2.5,
                  seed: int = 0, fps: int = 16,
                  amount_scale: float = 1.0) -> ShakeSpec:
    """Derive one shot's hold from what the shot is doing.

    Never the same tuning twice: the scale and the kind set the centre, the
    duration sets how much the hold loosens, and the seed draws the mix, the
    roll share and the envelope floor within the measured ranges. Two shots with
    the same scale and kind still get different character.
    """
    rng = np.random.default_rng(int(seed) * 104729 + 7)
    sg = SCALE_GAIN.get(scale, 1.0)
    kg = KIND_GAIN.get(kind, 1.0)
    t_mult, r_mult = ROTATION_SHARE.get(scale, (1.0, 1.0))
    lo, hi = DURATION_GAIN
    dg = float(np.clip(lo + (hi - lo) * (seconds - 1.5) / 2.5, lo, hi))
    gain = sg * kg * dg * amount_scale * float(rng.uniform(0.85, 1.15))

    def mix(a, b, c):
        v = np.array([rng.uniform(*a), rng.uniform(*b), rng.uniform(*c)])
        v = v / v.sum()
        return {"drift": float(v[0]), "sway": float(v[1]), "tremor": float(v[2])}

    return ShakeSpec(
        seed=int(seed), seconds=float(seconds), fps=int(fps),
        amount_px=BASE_PX * gain * t_mult,
        roll_deg=BASE_ROLL_DEG * gain * r_mult * float(rng.uniform(0.85, 1.20)),
        mix=mix((0.38, 0.56), (0.28, 0.42), (0.09, 0.16)),
        roll_mix=mix((0.28, 0.48), (0.36, 0.53), (0.11, 0.19)),
        floor=float(rng.uniform(*ENVELOPE_FLOOR)),
        tremor_floor=TREMOR_FLOOR,
        dither_px=float(rng.uniform(*DITHER_PX)),
        scale=scale, kind=kind,
    )


def report(spec: ShakeSpec) -> str:
    mix = "  ".join(f"{k} {v:.2f}" for k, v in spec.mix.items())
    return (f"seed {spec.seed:<3} {spec.scale or '-':<7} {spec.kind:<8} "
            f"{spec.seconds:4.2f}s @{spec.fps}fps  "
            f"{spec.amount_px:5.2f}px  roll {spec.roll_deg:4.2f}deg  "
            f"floor {spec.floor:.2f}  dither {spec.dither_px:.2f}px   {mix}")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="store_true",
                    help="print the per-shot spec for each framing scale")
    ap.add_argument("--shots", default="shot01,shot02",
                    help="forwarded to handheld_ab.py")
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()

    if a.spec:
        print("per-shot hold character (what the seed actually draws)\n")
        for sc in ("wide", "medium", "close", "macro"):
            for kind in ("none", "subject", "camera"):
                for seed in (3, 4):
                    print("  " + report(spec_for_shot(sc, kind, 2.5, seed)))
            print()
    else:
        import sys
        import handheld_ab
        sys.argv = ["handheld_ab.py", "--shots", a.shots, "--every", str(a.every)]
        handheld_ab.main()
