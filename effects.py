"""
effects.py -- image-representation effects, as a composable stage.

WHY THIS EXISTS
---------------
The pipeline produced clean, soft, technically-matched imagery, and the finishing
step in `make_film` actively REMOVES the properties that make an image feel authored:
`grade_match` normalises exposure and grain across shots so that "no cut reads as a
technical error". That is the right default for a film whose only job is to be legible,
and it is exactly wrong for work that wants the artifact to stay visible -- the seam,
the joint, the eroded surface, the thing a machine did that it was not quite asked to.

So this module gives the surface back, as a decision rather than a defect.

FIVE FAMILIES, not thirty filters
---------------------------------
    1. tonal       posterize, threshold, dither, halftone, stipple
    2. spatial     channel separation, displacement, slice, voronoi
    3. temporal    trails, feedback, slit-scan, frame blend
    4. material    xerox, riso, film damage, painterly
    5. structural  edges, pixel sort, patch shuffle, frequency separation

Every effect is `(RGB frames) -> (RGB frames)` over the whole clip, not one frame at a
time, because the temporal ones need history and the spatial ones need to be stable
across frames or they read as noise rather than as a surface.

CONTROLS, and the rule the user set: each effect exposes a FEW MEANINGFUL ones --
`amount` always, plus `mask` and a time behaviour where it makes sense. Nothing here
has a knob that does nothing.

DETERMINISM
-----------
Randomness is drawn from a seeded `numpy.random.default_rng`, seeded per effect and
per clip, so a preview is the export. An effect that looked different each time you
previewed it would be unusable, and one that flickered differently per frame would
read as noise, not as a surface.

COST
----
All of these are numpy on 512x512x16-51 frames -- milliseconds. That matters: it means
an effect can be previewed on REAL rendered frames before committing, which is the
whole point. Only `flow` (RAFT) is expensive and it reuses machinery the pipeline
already loads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# the registry
# --------------------------------------------------------------------------- #

@dataclass
class Effect:
    name: str
    family: str
    blurb: str
    # scales the effect's strength. Every effect has one; it means "how far from the
    # original", so 0.0 is always a no-op and that is asserted in the tests.
    apply: Callable[..., List[Image.Image]]
    params: Dict[str, object] = field(default_factory=dict)
    # temporal effects need the whole clip; spatial ones also do, for stability
    temporal: bool = False

    def run(self, frames: Sequence[Image.Image], **over) -> List[Image.Image]:
        """Apply with the effect's defaults, overridden by any non-None argument.

        FILTER ON `is not None`, NEVER ON TRUTHINESS. The first version did
        `if v is not None` -- correct -- and an earlier draft used a truthiness check,
        which silently dropped `softness=0.0` and `ordered=False`. A parameter whose
        zero value is meaningful must survive being passed, or the effect lies about
        what it is doing: `threshold(softness=0.0)` is a HARD threshold and it was
        quietly running the graded sigmoid instead.
        """
        kw = dict(self.params)
        kw.update({k: v for k, v in over.items() if v is not None})
        return self.apply(frames, **kw)


REGISTRY: Dict[str, Effect] = {}


def register(eff: Effect) -> Effect:
    REGISTRY[eff.name] = eff
    return eff


def by_family() -> Dict[str, List[Effect]]:
    out: Dict[str, List[Effect]] = {}
    for e in REGISTRY.values():
        out.setdefault(e.family, []).append(e)
    return {k: sorted(v, key=lambda e: e.name) for k, v in sorted(out.items())}


FAMILY_ORDER = ["tonal", "spatial", "temporal", "material", "structural"]


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

def to_array(frames: Sequence[Image.Image]) -> np.ndarray:
    """(T,H,W,3) float32 in 0..1."""
    return np.stack([np.asarray(f.convert("RGB"), dtype=np.float32) / 255.0
                     for f in frames])


def to_frames(a: np.ndarray) -> List[Image.Image]:
    a = np.clip(a, 0.0, 1.0)
    return [Image.fromarray((f * 255.0 + 0.5).astype(np.uint8), "RGB") for f in a]


def luma(a: np.ndarray) -> np.ndarray:
    return a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) * 2654435761 % (2 ** 32))


def _mix(orig: np.ndarray, worked: np.ndarray, amount: float) -> np.ndarray:
    """Blend, so `amount=0` is EXACTLY the original. That property is load-bearing:
    it makes an effect chain safe to leave in place and dial to nothing."""
    a = float(np.clip(amount, 0.0, 1.0))
    return orig * (1.0 - a) + worked * a


def apply_mask(a: np.ndarray, worked: np.ndarray, mask: Optional[np.ndarray],
               amount: float) -> np.ndarray:
    """Blend `worked` into `a` through a per-pixel mask, scaled by amount.

    The user asked for a mask on each effect, and it is the difference between a filter
    and a decision: the same distortion everywhere reads as a preset, and the same
    distortion in one region reads as authored.
    """
    if mask is None:
        return _mix(a, worked, amount)
    m = np.clip(mask, 0.0, 1.0)[..., None] * float(np.clip(amount, 0.0, 1.0))
    return a * (1.0 - m) + worked * m


def edge_mask(a: np.ndarray, width: float = 1.0) -> np.ndarray:
    g = luma(a)
    gx = np.gradient(g, axis=2)
    gy = np.gradient(g, axis=1)
    e = np.hypot(gx, gy)
    e = e / (e.max() + 1e-6)
    return np.clip(e * float(width), 0.0, 1.0)


def radial_mask(h: int, w: int, inner: float = 0.35, outer: float = 1.0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.hypot((yy - cy) / cy, (xx - cx) / cx) / np.sqrt(2.0)
    t = np.clip((r - inner) / max(1e-6, outer - inner), 0.0, 1.0)
    return t * t * (3 - 2 * t)                       # smoothstep


MASKS = {"none": None, "edges": edge_mask, "radial": radial_mask}


def resolve_mask(mask: Optional[str], amount_mask: float,
                 a: np.ndarray) -> Optional[np.ndarray]:
    if not mask or mask == "none":
        return None
    if mask == "radial":
        return radial_mask(a.shape[1], a.shape[2])
    if mask == "edges":
        return edge_mask(a)
    return None


# --------------------------------------------------------------------------- #
# 1. TONAL -- reconstruction of tone
# --------------------------------------------------------------------------- #

def _posterize(frames, levels: int = 5, amount: float = 1.0, mask: str = "none",
               seed: int = 0, **_) -> List[Image.Image]:
    a = to_array(frames)
    q = max(2, int(levels))
    worked = np.floor(a * q + 0.5) / q
    return to_frames(apply_mask(a, worked, resolve_mask(mask, amount, a), amount))


def _threshold(frames, level: float = 0.5, softness: float = 0.0, amount: float = 1.0,
               mask: str = "none", local: float = 0.0, seed: int = 0,
               **_) -> List[Image.Image]:
    """Black and white. `softness=0` is a HARD threshold -- two values, no greys.

    The default is hard because that is what the effect is for: `softness` above zero
    turns it into a graded sigmoid, which is a different and much weaker look, and
    defaulting to 0.06 made "threshold" quietly not threshold anything (the test caught
    it: 256 distinct values instead of 2).

    `local > 0` thresholds against a blurred neighbourhood instead of the whole frame,
    which is what makes the graphic detail irregular and organic rather than a
    silhouette. NOTE: the first version carried a `(g - ref) * 0 + ...` term -- a
    leftover that computed the difference and multiplied it by zero, so `local` was
    silently dead. It is now the actual local contrast.
    """
    a = to_array(frames)
    g = luma(a)
    if local > 0:
        k = max(1, int(local * 24))
        pad = np.pad(g, ((0, 0), (k, k), (k, k)), mode="edge")
        acc = np.zeros_like(g)
        n = 0
        for dy in range(-k, k + 1, max(1, k // 2)):
            for dx in range(-k, k + 1, max(1, k // 2)):
                acc += pad[:, k + dy:k + dy + g.shape[1], k + dx:k + dx + g.shape[2]]
                n += 1
        ref = acc / max(1, n)
        t = (g - ref)
    else:
        t = (g - level)
    if softness and softness > 0:
        bw = 1.0 / (1.0 + np.exp(-np.clip(t / float(softness), -30, 30)))
    else:
        bw = (t > 0).astype(np.float32)
    worked = np.repeat(bw[..., None], 3, axis=3)
    return to_frames(apply_mask(a, worked, resolve_mask(mask, amount, a), amount))


_BAYER8 = (np.array([[0, 32, 8, 40, 2, 34, 10, 42],
                     [48, 16, 56, 24, 50, 18, 58, 26],
                     [12, 44, 4, 36, 14, 46, 6, 38],
                     [60, 28, 52, 20, 62, 30, 54, 22],
                     [3, 35, 11, 43, 1, 33, 9, 41],
                     [51, 19, 59, 27, 49, 17, 57, 25],
                     [15, 47, 7, 39, 13, 45, 5, 37],
                     [63, 31, 55, 23, 61, 29, 53, 21]], dtype=np.float32) + 0.5) / 64.0


def _dither(frames, levels: int = 2, ordered: bool = True, amount: float = 1.0,
            mask: str = "none", scale: int = 1, seed: int = 0,
            **_) -> List[Image.Image]:
    """Ordered (Bayer) or error-diffused dithering.

    Ordered is geometric and stable frame to frame; Floyd-Steinberg is organic and
    slightly different everywhere. Both are legitimate and they read very differently,
    so `ordered` is a real choice rather than a flag.
    """
    a = to_array(frames)
    q = max(2, int(levels))
    if ordered:
        th = np.tile(_BAYER8, (int(np.ceil(a.shape[1] / 8)),
                               int(np.ceil(a.shape[2] / 8))))[:a.shape[1], :a.shape[2]]
        th = np.repeat(th[None, :, :, None], 3, axis=3)
        worked = np.floor(a * q + th) / q
    else:
        worked = np.empty_like(a)
        for t in range(a.shape[0]):
            img = a[t] * (q - 1)
            out = np.zeros_like(img)
            buf = img.copy()
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    old = buf[y, x]
                    new = np.round(old)
                    out[y, x] = new
                    err = old - new
                    if x + 1 < img.shape[1]:
                        buf[y, x + 1] += err * 7 / 16
                    if y + 1 < img.shape[0]:
                        if x > 0:
                            buf[y + 1, x - 1] += err * 3 / 16
                        buf[y + 1, x] += err * 5 / 16
                        if x + 1 < img.shape[1]:
                            buf[y + 1, x + 1] += err * 1 / 16
            worked[t] = np.clip(out / (q - 1), 0, 1)
    if scale > 1:
        worked = np.repeat(np.repeat(worked[:, ::scale, ::scale], scale, axis=1),
                           scale, axis=2)[:, :a.shape[1], :a.shape[2]]
    return to_frames(apply_mask(a, worked, resolve_mask(mask, amount, a), amount))


def _halftone(frames, cell: int = 6, angle: float = 45.0, amount: float = 1.0,
              mask: str = "none", irregular: float = 0.0, seed: int = 0,
              **_) -> List[Image.Image]:
    """Dot screening, imitating printmaking. `irregular` jitters cell centres, which is
    the difference between a print and a Photoshop filter.

    The screen is BUILT ONCE and shared across frames; only the tone it carries changes.
    A screen re-randomised per frame would read as noise rather than as a plate.
    """
    a = to_array(frames)
    T, H, W, _ = a.shape
    g = luma(a)
    # CLAMP the cell size. A zero or negative cell divides by zero and poisons the whole
    # array with NaN, which then renders as black and looks like a different bug.
    cell = max(2, int(cell))
    th = np.deg2rad(angle)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx * np.cos(th) + yy * np.sin(th)
    v = -xx * np.sin(th) + yy * np.cos(th)
    rng = _rng(seed)
    jitter = (rng.random((H, W)).astype(np.float32) - 0.5) * 2.0 * float(irregular) * cell
    u = u + jitter
    v = v + jitter
    cu = np.floor(u / cell)
    cv = np.floor(v / cell)
    fu = (u / cell - cu - 0.5) * 2.0
    fv = (v / cell - cv - 0.5) * 2.0
    d = np.hypot(fu, fv)
    # sample each cell's tone from the cell centre in source coordinates, per frame
    src_x = np.clip(np.round((cu + 0.5) * cell * np.cos(th) -
                             (cv + 0.5) * cell * np.sin(th)).astype(int), 0, W - 1)
    src_y = np.clip(np.round((cu + 0.5) * cell * np.sin(th) +
                             (cv + 0.5) * cell * np.cos(th)).astype(int), 0, H - 1)
    out = np.empty_like(a)
    for t in range(T):
        tone = np.clip(g[t][src_y, src_x], 0, 1)
        r = np.sqrt(tone) * 1.15
        dot = (d <= r).astype(np.float32)
        out[t] = np.repeat((dot * tone)[..., None], 3, axis=2)
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


def _stipple(frames, points: int = 4000, amount: float = 1.0, mask: str = "none",
             seed: int = 0, dark: bool = True, **_) -> List[Image.Image]:
    """Tone rebuilt from irregular points. More organic than a screen because the point
    distribution carries the tone instead of a regular grid."""
    a = to_array(frames)
    T, H, W, _ = a.shape
    rng = _rng(seed)
    g = luma(a)
    out = np.ones_like(a)
    n = max(64, int(points))
    ys = rng.integers(0, H, n)
    xs = rng.integers(0, W, n)
    for t in range(T):
        canvas = np.ones((H, W), np.float32)
        tone = g[t][ys, xs]
        keep = rng.random(n) < (1.0 - tone if dark else tone)
        canvas[ys[keep], xs[keep]] = 0.0
        out[t] = canvas[..., None]
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


# --------------------------------------------------------------------------- #
# 2. SPATIAL -- geometry and registration
# --------------------------------------------------------------------------- #

def _channel_separation(frames, offset_px: float = 3.0, angle: float = 0.0,
                        amount: float = 1.0, mask: str = "none", seed: int = 0,
                        **_) -> List[Image.Image]:
    """Misregistration: split into plates and offset them. `angle` sets the direction,
    so a diagonal split reads as a printing error and a horizontal one as a codec."""
    a = to_array(frames)
    T, H, W, _ = a.shape
    th = np.deg2rad(angle)
    dx = int(round(np.cos(th) * offset_px))
    dy = int(round(np.sin(th) * offset_px))
    out = np.empty_like(a)
    for ch, (ox, oy) in enumerate([(dx, dy), (0, 0), (-dx, -dy)]):
        out[..., ch] = np.roll(a[..., ch], (oy, ox), axis=(1, 2))
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


def _aberration(frames, offset_px: float = 4.0, radial: bool = True,
                amount: float = 1.0, mask: str = "none", seed: int = 0,
                **_) -> List[Image.Image]:
    """Fringes that GROW WITH DISTANCE FROM CENTRE when radial, which reads as optical
    depth; a constant offset reads as a mistake. The user's note -- "more interesting
    when displacement varies by depth or motion" -- is what `radial` implements."""
    a = to_array(frames)
    T, H, W, _ = a.shape
    out = np.empty_like(a)
    if not radial:
        return _channel_separation(frames, offset_px=offset_px, angle=0.0,
                                   amount=amount, mask=mask, seed=seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r = np.hypot((yy - cy) / cy, (xx - cx) / cx) / np.sqrt(2.0)
    for ch, sign in ((0, 1.0), (2, -1.0)):
        sh = r * float(offset_px) * sign
        # radial fringe: displace along BOTH axes (a diagonal push away from / toward
        # the centre). The vertical term used to be computed and then overwritten.
        iy = np.clip(np.round(yy + sh).astype(int), 0, H - 1)
        ix = np.clip(np.round(xx + sh).astype(int), 0, W - 1)
        out[..., ch] = a[:, iy, ix, ch]
    out[..., 1] = a[..., 1]
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


def _displace(frames, scale: float = 18.0, frequency: float = 3.0, amount: float = 1.0,
              mask: str = "none", seed: int = 0, drift: float = 0.0,
              **_) -> List[Image.Image]:
    """Push pixels with a smooth noise field. `drift` animates the field across the clip,
    which turns a still ripple into something that moves through the image."""
    a = to_array(frames)
    T, H, W, _ = a.shape
    rng = _rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    f = max(0.2, float(frequency))
    ph = rng.random(2) * 2 * np.pi
    base_u = (np.sin(xx / W * f * 2 * np.pi + ph[0]) *
              np.cos(yy / H * f * 2 * np.pi + ph[1])).astype(np.float32)
    base_v = (np.cos(xx / W * f * 2 * np.pi + ph[1]) *
              np.sin(yy / H * f * 2 * np.pi + ph[0])).astype(np.float32)
    out = np.empty_like(a)
    for t in range(T):
        k = 1.0 + float(drift) * (t / max(1, T - 1))
        u = np.clip(np.round(xx + base_u * scale * k).astype(int), 0, W - 1)
        v = np.clip(np.round(yy + base_v * scale * k).astype(int), 0, H - 1)
        out[t] = a[t][v, u]
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


def _slice(frames, slices: int = 12, shift_px: float = 20.0, amount: float = 1.0,
           mask: str = "none", seed: int = 0, animated: bool = True,
           **_) -> List[Image.Image]:
    """Cut the frame into strips and offset them independently. `animated` re-draws the
    offsets each frame, which is what makes it read as a broken signal rather than a
    static collage."""
    a = to_array(frames)
    T, H, W, _ = a.shape
    rng = _rng(seed)
    n = max(2, int(slices))
    edges = np.linspace(0, H, n + 1).astype(int)
    out = np.empty_like(a)
    for t in range(T):
        if animated or t == 0:
            shifts = (rng.random(n) - 0.5) * 2.0 * shift_px
        for i in range(n):
            y0, y1 = edges[i], edges[i + 1]
            out[t, y0:y1] = np.roll(a[t, y0:y1], int(round(shifts[i])), axis=1)
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


def _voronoi(frames, cells: int = 900, irregular: float = 0.6, amount: float = 1.0,
             mask: str = "none", seed: int = 0, **_) -> List[Image.Image]:
    """Polygonal cells, each filled with the source colour at its own centre.

    `irregular` matters and the user named why: uniform cells give the generic low-poly
    look, and uneven ones read as a construction.
    """
    a = to_array(frames)
    T, H, W, _ = a.shape
    rng = _rng(seed)
    n = max(16, int(cells))
    sy = rng.random(n) * H
    sx = rng.random(n) * W
    if irregular > 0:
        scale = 1.0 + (rng.random(n) - 0.5) * 2.0 * float(irregular)
        sy = np.clip((sy - H / 2) * scale + H / 2, 0, H - 1)
        sx = np.clip((sx - W / 2) * scale + W / 2, 0, W - 1)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    best = np.full((H, W), np.inf, np.float32)
    idx = np.zeros((H, W), np.int32)
    for i in range(n):
        d = (yy - sy[i]) ** 2 + (xx - sx[i]) ** 2
        m = d < best
        best[m] = d[m]
        idx[m] = i
    iy = np.clip(sy[idx].astype(int), 0, H - 1)
    ix = np.clip(sx[idx].astype(int), 0, W - 1)
    out = np.empty_like(a)
    for t in range(T):
        out[t] = a[t][iy, ix]
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


# --------------------------------------------------------------------------- #
# 3. TEMPORAL -- history inside the clip
# --------------------------------------------------------------------------- #

def _trails(frames, decay: float = 0.55, amount: float = 1.0, mask: str = "none",
            seed: int = 0, **_) -> List[Image.Image]:
    """Multiple exposure: each frame keeps a fading trace of the ones before it, so
    movement smears and stillness accumulates density."""
    a = to_array(frames)
    d = float(np.clip(decay, 0.0, 0.97))
    out = np.empty_like(a)
    acc = a[0].copy()
    for t in range(a.shape[0]):
        acc = np.maximum(a[t], acc * d)
        out[t] = acc
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


def _feedback(frames, gain: float = 0.35, zoom: float = 1.015, amount: float = 1.0,
              mask: str = "none", seed: int = 0, **_) -> List[Image.Image]:
    """Feed the previous OUTPUT back in, slightly transformed. The image develops a
    history and starts to organise itself -- recursive structure, not a filter."""
    a = to_array(frames)
    T, H, W, _ = a.shape
    out = np.empty_like(a)
    prev = a[0].copy()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    z = max(1.0, float(zoom))
    ux = np.clip(np.round((xx - cx) / z + cx).astype(int), 0, W - 1)
    uy = np.clip(np.round((yy - cy) / z + cy).astype(int), 0, H - 1)
    g = float(np.clip(gain, 0.0, 0.95))
    for t in range(T):
        zoomed = prev[uy, ux]
        prev = np.clip(a[t] * (1.0 - g) + zoomed * g, 0, 1)
        out[t] = prev
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


def _frame_blend(frames, window: int = 3, amount: float = 1.0, mask: str = "none",
                 seed: int = 0, **_) -> List[Image.Image]:
    """Average across neighbouring frames. Softens motion without losing it -- a gentler
    instrument than `trails` and useful when the model's motion is jittery."""
    a = to_array(frames)
    T = a.shape[0]
    w = max(1, int(window))
    out = np.empty_like(a)
    for t in range(T):
        lo, hi = max(0, t - w), min(T, t + w + 1)
        out[t] = a[lo:hi].mean(axis=0)
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


def _slit_scan(frames, column: int = 2, amount: float = 1.0, mask: str = "none",
               seed: int = 0, **_) -> List[Image.Image]:
    """Build each frame from a different time in the clip, sampled across space. Turns
    time into a spatial deformation: the clearest way to show that the image has a
    duration rather than being a series of pictures."""
    a = to_array(frames)
    T, H, W, _ = a.shape
    c = max(1, int(column))
    out = np.empty_like(a)
    for t in range(T):
        frame = a[t].copy()
        for x in range(0, W, c):
            src = int(round((x / max(1, W - 1)) * (T - 1)))
            frame[:, x:x + c] = a[src][:, x:x + c]
        out[t] = frame
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


# --------------------------------------------------------------------------- #
# 4. MATERIAL -- the surface of a reproduced object
# --------------------------------------------------------------------------- #

def _xerox(frames, contrast: float = 2.2, grain: float = 0.10, amount: float = 1.0,
           mask: str = "none", seed: int = 0, **_) -> List[Image.Image]:
    """Photocopy degradation: crush the contrast, exaggerate grain, dirty the edges.
    Reads as repeatedly reproduced rather than merely noisy."""
    a = to_array(frames)
    g = luma(a)
    c = float(contrast)
    crushed = np.clip((g - 0.5) * c + 0.5, 0, 1)
    # local contrast, so flat areas go pure and edges stay gritty
    k = 2
    pad = np.pad(crushed, ((0, 0), (k, k), (k, k)), mode="edge")
    blur = sum(pad[:, k + dy:k + dy + g.shape[1], k + dx:k + dx + g.shape[2]]
               for dy in range(-k, k + 1) for dx in range(-k, k + 1)) / (2 * k + 1) ** 2
    hi = np.clip(crushed - blur + 0.5, 0, 1)
    worked = np.repeat(hi[..., None], 3, axis=3)
    rng = _rng(seed)
    worked = np.clip(worked + (rng.random(worked.shape).astype(np.float32) - 0.5)
                     * float(grain) * 2.0, 0, 1)
    return to_frames(apply_mask(a, worked, resolve_mask(mask, amount, a), amount))


def _riso(frames, inks: int = 3, offset_px: float = 4.0, grain: float = 0.12,
          amount: float = 1.0, mask: str = "none", seed: int = 0,
          **_) -> List[Image.Image]:
    """Layered stencil printing: few inks, visible grain, imperfect registration, and
    the colour overlaps that fall out of getting it slightly wrong."""
    a = to_array(frames)
    T, H, W, _ = a.shape
    # a riso palette: two or three inks, deliberately not a full gamut
    ink_cols = np.array([[0.05, 0.20, 0.55],     # blue
                         [0.95, 0.20, 0.35],     # fluoro pink
                         [0.98, 0.85, 0.25]],    # yellow
                        dtype=np.float32)[:max(2, min(3, int(inks)))]
    out = np.ones_like(a)
    rng = _rng(seed)
    for i, ink in enumerate(ink_cols):
        # each ink plate takes one band of the tone, with its own registration error
        lo, hi = i / len(ink_cols), (i + 1) / len(ink_cols)
        m = ((a.mean(axis=3) >= lo) & (a.mean(axis=3) < hi)).astype(np.float32)
        ox = int(round((rng.random() - 0.5) * 2 * offset_px))
        oy = int(round((rng.random() - 0.5) * 2 * offset_px))
        m = np.roll(m, (oy, ox), axis=(1, 2))
        out = out * (1.0 - m[..., None]) + ink[None, None, None, :] * m[..., None]
    g = (rng.random(out.shape).astype(np.float32) - 0.5) * float(grain) * 2.0
    out = np.clip(out + g, 0, 1)
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


def _film_damage(frames, dust: float = 0.4, scratches: float = 0.5,
                 weave: float = 1.2, flicker: float = 0.10, amount: float = 1.0,
                 mask: str = "none", seed: int = 0, **_) -> List[Image.Image]:
    """Dust, scratches, gate weave and flicker.

    THE DEFECTS ARE DRAWN PER FRAME, and that is the whole point -- the user's note is
    right that a static overlay reads as a texture rather than as damage. Weave moves
    the whole frame independently of the dust on it.
    """
    a = to_array(frames)
    T, H, W, _ = a.shape
    rng = _rng(seed)
    out = np.empty_like(a)
    for t in range(T):
        frame = a[t].copy()
        if weave > 0:
            ox = int(round((rng.random() - 0.5) * 2 * weave))
            oy = int(round((rng.random() - 0.5) * 2 * weave))
            frame = np.roll(frame, (oy, ox), axis=(0, 1))
        if flicker > 0:
            frame = np.clip(frame * (1.0 + (rng.random() - 0.5) * 2 * flicker), 0, 1)
        if dust > 0:
            n = int(dust * 900)
            ys = rng.integers(0, H, max(0, n))
            xs = rng.integers(0, W, max(0, n))
            frame[ys, xs] = rng.random((len(ys), 3)).astype(np.float32)
        if scratches > 0:
            for _ in range(int(scratches * 3)):
                x = int(rng.integers(0, W))
                y0 = int(rng.integers(0, H))
                ln = int(rng.integers(H // 8, H))
                frame[y0:min(H, y0 + ln), x] = 0.85
        out[t] = frame
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


def _painterly(frames, radius: int = 4, strokes: int = 220, amount: float = 1.0,
               mask: str = "none", seed: int = 0, **_) -> List[Image.Image]:
    """Directional strokes over the frame, so fine detail is replaced by something that
    reads as interpreted rather than merely filtered. Stroke direction follows the
    image gradient, which is what stops it looking like a blur."""
    a = to_array(frames)
    T, H, W, _ = a.shape
    rng = _rng(seed)
    g = luma(a)
    gy = np.gradient(g, axis=1)
    gx = np.gradient(g, axis=2)
    out = a.copy()
    r = max(1, int(radius))
    n = max(16, int(strokes))
    for t in range(T):
        for _ in range(n):
            y = int(rng.integers(r, max(r + 1, H - r)))
            x = int(rng.integers(r, max(r + 1, W - r)))
            ang = np.arctan2(gy[t, y, x], gx[t, y, x] + 1e-6)
            dx, dy = np.cos(ang), np.sin(ang)
            col = a[t, y, x].copy()
            for s in np.linspace(-r, r, 2 * r + 1):
                px = int(np.clip(round(x + dx * s), 0, W - 1))
                py = int(np.clip(round(y + dy * s), 0, H - 1))
                out[t, py, px] = col
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


# --------------------------------------------------------------------------- #
# 5. STRUCTURAL -- the image's own organisation
# --------------------------------------------------------------------------- #

def _edges(frames, keep: float = 0.25, sobel: bool = True, amount: float = 1.0,
           mask: str = "none", seed: int = 0, dark: bool = True,
           **_) -> List[Image.Image]:
    """Contours only. `keep` how much of the original tone survives underneath, so the
    result can be a drawing or a drawing over a photograph."""
    a = to_array(frames)
    g = luma(a)
    gx = np.gradient(g, axis=2)
    gy = np.gradient(g, axis=1)
    e = np.hypot(gx, gy) if sobel else np.abs(gx) + np.abs(gy)
    e = e / (e.max() + 1e-6)
    e = np.clip(e * 3.0, 0, 1)
    ink = (1.0 - e) if dark else e
    base = g * float(np.clip(keep, 0, 1))
    worked = np.repeat((base * ink)[..., None], 3, axis=3)
    return to_frames(apply_mask(a, worked, resolve_mask(mask, amount, a), amount))


def _pixel_sort(frames, threshold: float = 0.55, key: str = "luma",
                reverse: bool = False, amount: float = 1.0, mask: str = "none",
                seed: int = 0, **_) -> List[Image.Image]:
    """Sort runs of pixels by brightness or hue. Stretched, dripping colour fields --
    and the threshold is what makes it read as intentional: sorting everything is noise,
    sorting selected runs is a decision."""
    a = to_array(frames)
    T, H, W, _ = a.shape
    g = luma(a)
    out = np.empty_like(a)
    for t in range(T):
        frame = a[t].copy()
        for y in range(H):
            row = g[t, y]
            hot = row > threshold
            x = 0
            while x < W:
                if not hot[x]:
                    x += 1
                    continue
                x0 = x
                while x < W and hot[x]:
                    x += 1
                seg = frame[y, x0:x]
                if key == "luma":
                    order = np.argsort(seg @ np.array([0.2126, 0.7152, 0.0722],
                                                      dtype=np.float32))
                else:                                   # hue
                    order = np.argsort(seg[:, 0] - seg[:, 2])
                if reverse:
                    order = order[::-1]
                frame[y, x0:x] = seg[order]
        out[t] = frame
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


def _patch_shuffle(frames, patch: int = 32, strength: float = 0.5, amount: float = 1.0,
                   mask: str = "none", seed: int = 0, **_) -> List[Image.Image]:
    """Rearrange blocks while keeping their contents. Fragmentation that still reads as
    the same image -- `strength` is the fraction of patches moved."""
    a = to_array(frames)
    T, H, W, _ = a.shape
    rng = _rng(seed)
    p = max(4, int(patch))
    ny, nx = H // p, W // p
    order = np.arange(ny * nx)
    k = int(len(order) * float(np.clip(strength, 0, 1)))
    if k:
        pick = rng.choice(len(order), size=k, replace=False)
        perm = order.copy()
        perm[pick] = perm[rng.permutation(pick)]
        order = perm
    out = np.empty_like(a)
    for t in range(T):
        for i, src in enumerate(order):
            sy, sx = divmod(int(src), nx)
            dy, dx = divmod(i, nx)
            out[t, dy * p:(dy + 1) * p, dx * p:(dx + 1) * p] = \
                a[t, sy * p:(sy + 1) * p, sx * p:(sx + 1) * p]
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


def _frequency_separation(frames, coarse: int = 9, distort: float = 12.0,
                          amount: float = 1.0, mask: str = "none", seed: int = 0,
                          **_) -> List[Image.Image]:
    """Move the large forms while the fine detail stays put. Lets composition change
    without turning the image to mush -- the reason retouchers separate frequency."""
    a = to_array(frames)
    T, H, W, _ = a.shape
    k = max(1, int(coarse))
    out = np.empty_like(a)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rng = _rng(seed)
    for t in range(T):
        pad = np.pad(a[t], ((k, k), (k, k), (0, 0)), mode="edge")
        low = sum(pad[dy:dy + H, dx:dx + W]
                  for dy in range(0, 2 * k, max(1, k // 3))
                  for dx in range(0, 2 * k, max(1, k // 3)))
        low = low / len(range(0, 2 * k, max(1, k // 3))) ** 2
        high = a[t] - low
        ang = rng.random() * 2 * np.pi
        u = np.clip(np.round(xx + np.cos(ang) * distort).astype(int), 0, W - 1)
        v = np.clip(np.round(yy + np.sin(ang) * distort).astype(int), 0, H - 1)
        out[t] = np.clip(low[v, u] + high, 0, 1)
    return to_frames(apply_mask(a, out, resolve_mask(mask, amount, a), amount))


def _palette_remap(frames, inks: str = "concrete", amount: float = 1.0,
                   mask: str = "none", seed: int = 0, **_) -> List[Image.Image]:
    """Map tone onto a designed ramp. The most direct way to make ordinary footage feel
    authored -- and it is a RAMP, not a tint, so highlights and shadows separate."""
    RAMPS = {
        # the user's own list: dirty white, concrete grey, muddy blue, black
        "concrete": [(0.05, 0.05, 0.06), (0.28, 0.30, 0.33), (0.55, 0.56, 0.54),
                     (0.82, 0.80, 0.76), (0.96, 0.95, 0.92)],
        "sodium":   [(0.04, 0.03, 0.05), (0.28, 0.14, 0.08), (0.72, 0.42, 0.12),
                     (0.95, 0.72, 0.35), (1.00, 0.94, 0.80)],
        "deepwater": [(0.02, 0.04, 0.07), (0.06, 0.14, 0.22), (0.12, 0.30, 0.38),
                      (0.35, 0.58, 0.60), (0.78, 0.88, 0.86)],
        "washed":   [(0.10, 0.12, 0.10), (0.34, 0.38, 0.32), (0.58, 0.62, 0.54),
                     (0.78, 0.80, 0.72), (0.94, 0.94, 0.88)],
    }
    ramp = np.array(RAMPS.get(inks, RAMPS["concrete"]), dtype=np.float32)
    a = to_array(frames)
    g = luma(a)
    idx = np.clip(g * (len(ramp) - 1), 0, len(ramp) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, len(ramp) - 1)
    f = (idx - lo)[..., None]
    worked = ramp[lo] * (1 - f) + ramp[hi] * f
    return to_frames(apply_mask(a, worked, resolve_mask(mask, amount, a), amount))


# --------------------------------------------------------------------------- #
# register them
# --------------------------------------------------------------------------- #

for e in [
    # tonal
    Effect("posterize", "tonal", "flat colour bands; graphic shadows, harsh light",
           _posterize, {"levels": 5}),
    Effect("threshold", "tonal", "hard black and white at softness=0; `local` makes "
           "detail organic", _threshold, {"level": 0.5, "softness": 0.0, "local": 0.0}),
    Effect("dither", "tonal", "Bayer geometry or Floyd-Steinberg error diffusion",
           _dither, {"levels": 2, "ordered": True, "scale": 1}),
    Effect("halftone", "tonal", "dot screening; `irregular` breaks the grid",
           _halftone, {"cell": 6, "angle": 45.0, "irregular": 0.0}),
    Effect("stipple", "tonal", "tone rebuilt from irregular points",
           _stipple, {"points": 4000, "dark": True}),
    # spatial
    Effect("channel_separation", "spatial", "misregistered colour plates",
           _channel_separation, {"offset_px": 3.0, "angle": 0.0}),
    Effect("aberration", "spatial", "fringes that grow with distance from centre",
           _aberration, {"offset_px": 4.0, "radial": True}),
    Effect("displace", "spatial", "pixels pushed by a noise field; `drift` animates it",
           _displace, {"scale": 18.0, "frequency": 3.0, "drift": 0.0}),
    Effect("slice", "spatial", "strips offset independently; `animated` re-draws them",
           _slice, {"slices": 12, "shift_px": 20.0, "animated": True}),
    Effect("voronoi", "spatial", "polygonal cells from the source; `irregular` avoids "
           "the generic low-poly look", _voronoi, {"cells": 900, "irregular": 0.6}),
    # temporal
    Effect("trails", "temporal", "multiple exposure; movement smears, stillness builds",
           _trails, {"decay": 0.55}, temporal=True),
    Effect("feedback", "temporal", "the previous output fed back in, zoomed",
           _feedback, {"gain": 0.35, "zoom": 1.015}, temporal=True),
    Effect("frame_blend", "temporal", "average across neighbouring frames",
           _frame_blend, {"window": 3}, temporal=True),
    Effect("slit_scan", "temporal", "each frame built from a different moment",
           _slit_scan, {"column": 2}, temporal=True),
    # material
    Effect("xerox", "material", "photocopy: crushed contrast, grain, dirty edges",
           _xerox, {"contrast": 2.2, "grain": 0.10}),
    Effect("riso", "material", "layered stencil inks with registration error",
           _riso, {"inks": 3, "offset_px": 4.0, "grain": 0.12}),
    Effect("film_damage", "material", "dust, scratches, weave, flicker -- drawn per "
           "frame, never a static overlay", _film_damage,
           {"dust": 0.4, "scratches": 0.5, "weave": 1.2, "flicker": 0.10}, temporal=True),
    Effect("painterly", "material", "strokes following the image gradient",
           _painterly, {"radius": 4, "strokes": 220}),
    # structural
    Effect("edges", "structural", "contours only, over whatever tone you keep",
           _edges, {"keep": 0.25, "sobel": True, "dark": True}),
    Effect("pixel_sort", "structural", "runs sorted by brightness or hue",
           _pixel_sort, {"threshold": 0.55, "key": "luma", "reverse": False}),
    Effect("patch_shuffle", "structural", "blocks rearranged, contents kept",
           _patch_shuffle, {"patch": 32, "strength": 0.5}),
    Effect("frequency_separation", "structural", "large forms move, fine detail stays",
           _frequency_separation, {"coarse": 9, "distort": 12.0}),
    Effect("palette_remap", "structural", "tone mapped onto a designed ramp",
           _palette_remap, {"inks": "concrete"}),
]:
    register(e)


# --------------------------------------------------------------------------- #
# chains
# --------------------------------------------------------------------------- #

def chain(project: "Dict") -> List[Dict]:
    """The effect chain recorded on a production, as a list of steps."""
    return list(((project.get("settings") or {}).get("effects") or []))


def set_chain(project: "Dict", steps: Sequence[Dict]) -> None:
    project.setdefault("settings", {})["effects"] = [dict(s) for s in steps]


def validate_chain(steps: Sequence[Dict]) -> List[str]:
    """Report steps that name an unknown effect -- never silently skip one."""
    out = []
    for i, s in enumerate(steps):
        if s.get("effect") not in REGISTRY:
            out.append(f"step {i + 1}: no effect called {s.get('effect')!r}")
    return out


def apply_chain(frames: Sequence[Image.Image], steps: Sequence[Dict],
                verbose: bool = True) -> List[Image.Image]:
    """Run the chain in order. `amount` is per step, so a step can be dialled to no-op
    without being removed -- which is the point of building chains rather than picking
    one filter."""
    out = list(frames)
    for i, s in enumerate(steps, 1):
        name = s.get("effect")
        eff = REGISTRY.get(name)
        if eff is None:
            if verbose:
                print(f"    effect step {i}: UNKNOWN {name!r} -- skipped")
            continue
        over = {k: v for k, v in s.items() if k != "effect"}
        try:
            out = eff.run(out, **over)
            if verbose:
                amt = s.get("amount", 1.0)
                print(f"    effect step {i}: {name} amount={amt} "
                      f"-> {len(out)} frames")
        except Exception as exc:
            # an effect that fails must not take the render with it: the takes are the
            # expensive thing and they are already on disk
            print(f"    effect step {i}: {name} FAILED "
                  f"({type(exc).__name__}: {exc}) -- skipped, frames unchanged")
    return out
