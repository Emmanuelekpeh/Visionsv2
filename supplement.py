"""
supplement.py -- two cheap additions to AnimateDiff output.

raft  Temporal upsampling. AnimateDiff generates 16 real frames; RAFT inserts
      in-betweens so the motion reads smoothly at 16 or 32fps. This is NOT the
      role RAFT was rejected for: AnimateDiff still decides the motion, RAFT only
      resamples it. Cost is ~5% of generation.

lds   Dynamic Textures (Doretto et al. 2003). Fit a linear dynamical system
          x(t+1) = A x(t) + B v(t),   y(t) = C x(t) + w(t)
      to a short clip, then drive it with noise to synthesize ARBITRARILY LONG
      motion -- the "flowing water forever" trick. Fitted to AnimateDiff output,
      so the appearance comes from the diffusion model and the length from the LDS.

    python supplement.py raft river
    python supplement.py lds river
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# torchvision weights land in ~/.cache/torch, which is outside the sandbox
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch_home"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageSequence  # noqa: E402

torch.set_num_threads(12)

SRC = ROOT / "output" / "composite"
OUT = ROOT / "output" / "supplement"
OUT.mkdir(parents=True, exist_ok=True)


def load_frames(name):
    d = SRC / name
    files = sorted(d.glob("f*.png"))
    if not files:
        g = d / "clip.gif"
        if g.exists():
            return [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(g))]
        raise SystemExit(f"no frames in {d}")
    return [Image.open(f).convert("RGB") for f in files]


def save_video(frames, path, fps):
    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, str(path), fps=fps)
        return True
    except Exception as exc:
        print(f"    mp4 failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# RAFT temporal upsampling
# ---------------------------------------------------------------------------

def warp(img, flow):
    import torch.nn.functional as F
    b, _, h, w = img.shape
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    grid = torch.stack((xs, ys), dim=-1).float().unsqueeze(0).repeat(b, 1, 1, 1)
    grid = grid + flow.permute(0, 2, 3, 1)
    grid[..., 0] = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    return F.grid_sample(img, grid, align_corners=True, padding_mode="border")


def to_tensor(img):
    a = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def to_image(t):
    a = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray((a * 255 + 0.5).astype(np.uint8))


def run_raft(name, factor=2, flow_size=256):
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
    from torchvision.transforms.functional import resize

    frames = load_frames(name)
    H, W = frames[0].size[1], frames[0].size[0]
    print(f"  {len(frames)} frames at {W}x{H}, upsampling x{factor}")

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
            import torch.nn.functional as F
            f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            f = f * (W / flow_size)
        return f

    tens = [to_tensor(f) for f in frames]
    t0 = time.time()
    out = [frames[0]]
    for i in range(len(frames) - 1):
        a, b = tens[i], tens[i + 1]
        f_ab = flow(a, b)
        f_ba = flow(b, a)
        for k in range(1, factor):
            t = k / factor
            wa = warp(a, f_ab * t)
            wb = warp(b, f_ba * (1 - t))
            # time-weighted: exact at the endpoints, no double-exposure build-up
            mix = wa * (1 - t) + wb * t
            out.append(to_image(mix))
        out.append(frames[i + 1])
    dt = time.time() - t0

    d = OUT / f"{name}_raft{factor}x"
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(out):
        f.save(d / f"f{i:03d}.png")
    out[0].save(d / "clip.gif", save_all=True, append_images=out[1:],
                duration=max(20, int(1000 / (8 * factor))), loop=0)
    save_video(out, d / "clip.mp4", fps=8 * factor)

    diffs = [float(np.abs(np.asarray(out[i]).astype(np.float32)
                          - np.asarray(out[i - 1]).astype(np.float32)).mean())
             for i in range(1, len(out))]
    meta = {"source": name, "in_frames": len(frames), "out_frames": len(out),
            "factor": factor, "seconds": round(dt, 1),
            "interpolation_overhead_pct": round(100 * dt / (len(frames) * 35.6), 1),
            "frame_diff_mean": round(float(np.mean(diffs)), 2)}
    (d / "stats.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  {len(frames)} -> {len(out)} frames in {dt:.1f}s "
          f"({dt/len(frames):.1f}s per source frame)")
    print(f"  frame_diff {meta['frame_diff_mean']} "
          f"(was {14.62} raw)  -- halved is correct for x2 upsampling")
    return d, out


# ---------------------------------------------------------------------------
# Dynamic Textures / LDS (Doretto et al. 2003)
# ---------------------------------------------------------------------------

def run_lds(name, rank=40, synth=64, noise=1.0):
    frames = load_frames(name)
    T = len(frames)
    H, W = frames[0].size[1], frames[0].size[0]
    print(f"  fitting LDS to {T} frames at {W}x{H}, rank {rank}")

    # small resolution keeps the fit tractable and the texture readable
    sw, sh = 128, 128
    arr = np.stack([np.asarray(f.resize((sw, sh), Image.LANCZOS)).astype(np.float32)
                    for f in frames])                    # T,H,W,3
    Y = arr.reshape(T, -1).T                             # N,T
    mu = Y.mean(axis=1, keepdims=True)
    Yc = Y - mu

    t0 = time.time()
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    n = min(rank, len(S))
    U_n, S_n, V_n = U[:, :n], S[:n], Vt[:n]              # U:Nxn, S:n, V:nxT
    X = (S_n[:, None] * V_n)                             # n x T  (state)

    # A: x(t+1) = A x(t)  -- least squares over the observed sequence
    X1, X2 = X[:, :-1], X[:, 1:]
    A, *_ = np.linalg.lstsq(X1.T, X2.T, rcond=None)
    A = A.T
    resid = X2 - A @ X1
    Q = np.cov(resid) if resid.shape[1] > 1 else np.eye(n) * 1e-4
    fit_s = time.time() - t0

    # synthesize: drive with noise matched to the residual statistics
    rng = np.random.default_rng(0)
    x = X[:, 0].copy()
    gen = []
    for _ in range(synth):
        y = U_n @ x + mu[:, 0]
        gen.append(np.clip(y.reshape(sh, sw, 3), 0, 255).astype(np.uint8))
        v = rng.multivariate_normal(np.zeros(n), Q) if noise > 0 else 0.0
        x = A @ x + noise * v

    # spectral radius tells us whether it will stay bounded or blow up
    eigs = np.abs(np.linalg.eigvals(A))
    rho = float(eigs.max())
    print(f"  fit in {fit_s:.1f}s   spectral radius of A = {rho:.4f} "
          f"({'stable' if rho < 1 else 'UNSTABLE - will diverge'})")

    imgs = [Image.fromarray(g).resize((W, H), Image.LANCZOS) for g in gen]
    d = OUT / f"{name}_lds"
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(imgs):
        f.save(d / f"f{i:03d}.png")
    imgs[0].save(d / "clip.gif", save_all=True, append_images=imgs[1:],
                 duration=125, loop=0)
    save_video(imgs, d / "clip.mp4", fps=8)

    diffs = [float(np.abs(np.asarray(imgs[i]).astype(np.float32)
                          - np.asarray(imgs[i - 1]).astype(np.float32)).mean())
             for i in range(1, len(imgs))]
    meta = {"source": name, "rank": rank, "synthesized": len(imgs),
            "spectral_radius": round(rho, 4), "stable": rho < 1.0,
            "frame_diff_mean": round(float(np.mean(diffs)), 2),
            "fit_seconds": round(fit_s, 1)}
    (d / "stats.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  {T} -> {len(imgs)} frames, frame_diff {meta['frame_diff_mean']}")
    return d, imgs


# ---------------------------------------------------------------------------

def sheet(pairs, title, out_name):
    if not pairs:
        return
    T = 110
    ncols = max(len(v) for _, v in pairs)
    ncols = min(ncols, 24)
    canvas = Image.new("RGB", (ncols * (T + 3) + 3, len(pairs) * (T + 18) + 20),
                       (20, 20, 24))
    dr = ImageDraw.Draw(canvas)
    dr.text((5, 3), title, fill=(235, 235, 235))
    for r, (lab, imgs) in enumerate(pairs):
        y = 18 + r * (T + 18)
        dr.text((5, y), lab, fill=(210, 210, 210))
        for c in range(min(ncols, len(imgs))):
            canvas.paste(imgs[c].resize((T, T), Image.LANCZOS),
                         (3 + c * (T + 3), y + 13))
    canvas.save(OUT / out_name)
    print(f"\nwrote {OUT/out_name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["raft", "lds", "both"])
    ap.add_argument("name", nargs="?", default="river")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--rank", type=int, default=40)
    ap.add_argument("--synth", type=int, default=64)
    a = ap.parse_args()

    pairs = []
    if a.what in ("raft", "both"):
        print(f"\n=== RAFT temporal upsampling: {a.name} ===")
        d, frames = run_raft(a.name, a.factor)
        pairs.append((f"{a.name} RAFT x{a.factor} ({len(frames)} frames)", frames))
    if a.what in ("lds", "both"):
        print(f"\n=== Dynamic Textures / LDS: {a.name} ===")
        d, frames = run_lds(a.name, a.rank, a.synth)
        pairs.append((f"{a.name} LDS (rank {a.rank}, {len(frames)} synthesized)", frames))
    if pairs:
        sheet(pairs, f"supplements on '{a.name}'", "supplement.png")
    print("\nDONE")
