#!/usr/bin/env python3
"""probe_sheet.py -- real rendered frames from a few shots, before the full film.

WHY. Previs stills are a different pipeline (4-step SD15 stills) from the film
(AnimateDiff + motion LoRA + RAFT + finishing). Measured on the squid motif: the
previs looked murky but plausible and the film came back as helmet-creatures.
Previs is a continuity/prompt-alignment proxy, NOT a look proxy. The cheap honest
check is a PROBE RENDER: generate 1 shot per scene through the real store path
(`make_film --store <slug> --only 1,5,9`), then look at the actual take frames.

    python probe_sheet.py --store squid --shots 1,5,9
    # -> output/probe/<slug>_probe.png  (3 frames per shot, prompt under it)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw          # noqa: E402


def _take_frames(take_dir: Path) -> list:
    frames = sorted((take_dir / "frames").glob("f*.png"))
    return frames


def build(slug: str, shots: list, out: Path, width: int = 300) -> Path:
    import film_store as fs
    project = fs.load(slug)
    order = list(project["timeline"])
    picks = [order[i - 1] for i in shots if 1 <= i <= len(order)]
    rows = []
    for sid in picks:
        rec = fs.load_shot(slug, sid)
        take = fs.best_take(project, rec)
        if take is None:
            rows.append((sid, rec.get("prompt", ""), []))
            continue
        files = _take_frames(Path(take["_dir"]))
        idx = sorted({0, len(files) // 2, max(len(files) - 1, 0)})
        rows.append((sid, rec.get("prompt", ""), [files[i] for i in idx]))
    band = 56
    colw = 3 * (width + 4) + 4
    sheet = Image.new("RGB", (colw, len(rows) * (width + band + 8) + 8),
                      (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    for r, (sid, prompt, files) in enumerate(rows):
        y = 8 + r * (width + band + 8)
        d.text((6, y), f"{sid}  {prompt[:96]}", fill=(235, 235, 235))
        d.text((6, y + 16), " " * 6 + prompt[96:170], fill=(170, 170, 175))
        if not files:
            d.text((6, y + 34), "NO TAKE -- render it first", fill=(255, 120, 120))
            continue
        for k, f in enumerate(files):
            im = Image.open(f).convert("RGB").resize((width, width),
                                                     Image.LANCZOS)
            sheet.paste(im, (4 + k * (width + 4), y + band - 22))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Real take frames for probed shots")
    ap.add_argument("--store", required=True)
    ap.add_argument("--shots", default="1",
                    help="comma list of 1-based shot numbers (e.g. 1,5,9)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    shots = [int(x) for x in a.shots.split(",") if x.strip()]
    out = Path(a.out) if a.out else ROOT / "output" / "probe" / f"{a.store}_probe.png"
    path = build(a.store, shots, out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
