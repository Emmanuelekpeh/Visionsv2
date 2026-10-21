"""
motion_metrics.py -- measure WHERE a clip's motion lives, not just how much.

The measurement that killed the fg/bg compositing idea was this one:

    top   5% of pixels hold   7.7% of the temporal variance
    top  10% of pixels hold  14.2%
    top  25% of pixels hold  31.7%

Those numbers are near-uniform (uniform would be 5 / 10 / 25), which proves
AnimateDiff's motion is *global drift* rather than a moving subject. A clip with
a figure walking across frame concentrates its variance in a compact region, so
top-10% share rises well above the uniform 10%.

Metrics
  loc5/10/25   share of temporal variance held by the top k% of pixels
               (uniform = k/100; higher = more localized)
  mag          mean absolute frame-to-frame luminance change (0-255 scale)
  global_share how spatially uniform that change is: 1.0 = the whole frame
               shifts together (brightness drift), low = one region moves
  centroid_travel  std-dev of the per-frame "busiest region" centroid, in px.
               A subject crossing frame drags it; global drift pins it near centre
  mask_spread  mean radius of the busiest region / frame diagonal.
               0.5 = spread over the whole frame (global), <0.3 = a localized blob
  coherence    mean CLIP drift from frame 0 (lower = more self-consistent)
  align        CLIP text-image similarity (higher = matches the prompt)

CAUTION on comparing across harnesses. These statistics depend on resolution,
frame count, fps and video codec. The author's reference renders are compressed
1152x768 mp4s at 16-63 frames; a local run is 512x512 PNGs at 16 frames. `loc10`
measured 0.718 on a local no-LoRA baseline and 0.783 on the author's walking
render -- nearly the same number from two very different clips. Compare only
within a single harness, or the metric will tell you what you want to hear.
"""

from __future__ import annotations

import numpy as np

# A clip counts as separable when its change field is this much more concentrated
# than the no-LoRA baseline's. 0.7 means a 30% reduction. The author's render
# scored 0.18 against a baseline of 0.36 (a 50% reduction); nothing measured on
# this machine has come below 0.42 against a baseline of 0.49 (a 14% reduction).
LOCALISED_GLOBAL_RATIO = 0.7


# --------------------------------------------------------------------------- #
# temporal / spatial structure
# --------------------------------------------------------------------------- #

def temporal_stats(frames) -> dict:
    """frames: sequence of PIL images or HxWx3 arrays, same size."""
    a = np.stack([np.asarray(f.convert("RGB") if hasattr(f, "convert") else f,
                             dtype=np.float32) for f in frames])
    T, H, W, _ = a.shape
    g = a.mean(axis=3)                                  # luminance, T,H,W

    var = g.var(axis=0)                                 # per-pixel temporal variance
    flat = np.sort(var.ravel())[::-1]
    tot = float(flat.sum()) or 1.0
    loc = {}
    for k in (5, 10, 25):
        n = max(1, int(round(flat.size * k / 100.0)))
        loc[f"loc{k}"] = float(flat[:n].sum() / tot)

    d = np.abs(np.diff(g, axis=0))                      # T-1,H,W
    mag = float(d.mean())

    # "Is the whole frame changing together, or one region?"
    #
    # The obvious definition -- |mean(d)| / mean(|d|) -- is DEGENERATE and was
    # wrong here: when every pixel changes in the same direction (a clip that
    # simply brightens), |d| == d, so the ratio is 1.0 no matter how localized
    # the change was. The `fig_none_512` baseline scored exactly 1.00 on it while
    # also scoring loc10=0.72, which is self-contradictory and is how the bug
    # surfaced.
    #
    # Spatial uniformity of the change field is the thing actually wanted: a
    # constant offset has zero spread across pixels, a compact moving blob has a
    # large one relative to its mean.
    uni = []
    for t in range(d.shape[0]):
        m = float(d[t].mean())
        s = float(d[t].std())
        uni.append(1.0 / (1.0 + s / (m + 1e-6)))
    global_share = float(np.mean(uni))

    # busiest region: where does the frame-to-frame change concentrate?
    thr = float(np.percentile(d, 90))
    yy, xx = np.mgrid[0:H, 0:W]
    diag = float(np.hypot(H, W))
    cents, spreads = [], []
    for t in range(d.shape[0]):
        m = d[t] > thr
        if m.sum() < 16:
            continue
        w = d[t] * m
        cx = float((xx * m).sum() / m.sum())
        cy = float((yy * m).sum() / m.sum())
        cents.append((cx, cy))
        r = np.hypot(xx - cx, yy - cy)
        spreads.append(float((r * m).sum() / m.sum() / diag))
    if len(cents) > 1:
        c = np.asarray(cents)
        travel = float(np.hypot(c[:, 0].std(), c[:, 1].std()))
    else:
        travel = 0.0

    return {
        **loc,
        "mag": round(mag, 3),
        "global_share": round(global_share, 3),
        "centroid_travel": round(travel, 2),
        "mask_spread": round(float(np.mean(spreads)) if spreads else 0.0, 3),
        "size": [W, H],
        "n_frames": int(T),
    }


# --------------------------------------------------------------------------- #
# CLIP-based signals -- the handoff found alignment tracks quality almost exactly
# --------------------------------------------------------------------------- #

class ClipScorer:
    """Local CLIP, so no download and no network."""

    def __init__(self, path: str = "models/clip-vit-base-patch32", device="cpu"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.model = CLIPModel.from_pretrained(path).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(path)

    def embed_images(self, frames):
        import torch
        out = []
        for f in frames:
            px = self.proc(images=f, return_tensors="pt").pixel_values
            with torch.no_grad():
                e = self.model.get_image_features(pixel_values=px)
            out.append(torch.nn.functional.normalize(e, dim=-1)[0])
        return torch.stack(out)

    def embed_text(self, text: str):
        import torch
        inp = self.proc(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            e = self.model.get_text_features(**inp)
        return torch.nn.functional.normalize(e, dim=-1)[0]

    def scores(self, frames, prompt: str) -> dict:
        """clip drift (coherence) + text alignment, matching the handoff's units."""
        e = self.embed_images(frames)
        cos0 = (e @ e[0]).clamp(-1, 1)
        drift = float((1.0 - cos0).mean())
        t = self.embed_text(prompt)
        align = float((e @ t).mean())
        return {"coherence": round(drift, 4), "align": round(align, 4)}


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #

def localisation_verdict(m: dict, baseline: dict | None = None) -> dict:
    """Is this clip's motion localized, or is the background moving too?

    PRIMARY SIGNAL IS `global_share`, NOT `loc10`.

    `loc10` is confounded by `mag`, so comparing it across clips is not valid.
    It is the share of temporal variance in the busiest 10% of pixels, so a
    nearly-static clip scores *high* simply because the little change it has is
    concentrated. The no-LoRA baseline (`mag` 3.73) scored loc10 0.718 -- higher
    than every subject-LoRA clip (`mag` 9-49). Read naively that says "the LoRA
    made localisation worse", when it actually means "adding motion spreads the
    variance around". Do not repeat that reading.

    `global_share` is the spatial uniformity of the change field and does not
    depend on how much change there is, so it is the right instrument: a subject
    moving against a held background concentrates the change and drops
    global_share well below the baseline. Measured:

        baseline (no LoRA)            0.49
        walking-frontview             0.47
        walking-frontview @576x384    0.49
        running-frontview             0.42
        ballerina-spin                0.56
        leaf macro + walking LoRA     0.53
        author's walking render       0.18   <- the only truly separable one

    Nothing comes close to the author's 0.18. Compared against an in-run baseline
    when supplied -- absolute numbers from another session are not evidence,
    because the harness may differ.
    """
    base_global = (baseline or {}).get("global_share", 0.49)
    base10 = (baseline or {}).get("loc10", 0.142)
    ratio = m["global_share"] / base_global if base_global else float("inf")
    localized = bool(ratio <= LOCALISED_GLOBAL_RATIO)
    return {
        "global_ratio_vs_baseline": round(ratio, 2),
        "loc10_ratio_vs_baseline": round(m["loc10"] / base10, 2) if base10 else None,
        "loc10_confounded_by_mag": True,
        "localized": localized,
        "verdict": ("LOCALIZED -- separable subject against a held background"
                    if localized else
                    "not separable -- the background moves too"),
    }
