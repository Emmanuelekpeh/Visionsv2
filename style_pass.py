#!/usr/bin/env python3
"""style_pass.py -- one deliberate light + palette per scene, and the fog out.

WHY. The first held 13-shot films came back uniformly grey. Measured on previs:
the plans followed place/figure coverage, but every scene's prompts shared the
same fog/mist language and every scene's palette was 'neutral', so the film had
coverage without a look. This pass is the directable fix, applied to an adopted
store BEFORE the render (prompts are render inputs, so this correctly marks
takes stale):

  * strips fog/mist/haze words from the prompt text (the "unclear" look was
    text-level atmosphere, not only the model);
  * inserts ONE concrete light clause per held scene, after the place/figure
    anchors, so each scene reads with its own light;
  * sets one palette per scene from `looks.ARC`, so `make_film` grades the cuts
    apart (previs.py now shows the grade too).

The arc is a legible progression, not a random walk: by default warm dawn ->
silver interior -> amber dusk -> cold night, cycled if the film has more scenes.

    python style_pass.py --store ego_v5
    python style_pass.py --store ego_v5 --keep-fog
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FOG = re.compile(r"\b(?:fog|foggy|mist|misty|haze|hazy|murky|murk|shrouding)\b",
                 re.I)

# One deliberate look per held scene, in film order (light source -> palette).
# These are the FALLBACK samples for an existing store; the real per-film arc is
# the scene plan's `light`/`palette` (written per scene, from its own story). The
# set spans orange, green, blue and magenta so the film is not stuck in one
# warm/cool band. No "dust in the air": it reads as glow/diffusion in the render.
DEFAULT_STYLES: List[Dict[str, str]] = [
    {"light": "clear amber dawn light through tall windows, sharp shadows",
     "palette": "warm-gold"},
    {"light": "hard green neon from a sign, wet reflections, deep shadows",
     "palette": "emerald"},
    {"light": "hard silver light raking across glass and mirrors, cool highlights",
     "palette": "cold-steel"},
    {"light": "magenta neon glow on wet concrete, deep shadows",
     "palette": "magenta"},
    {"light": "red sodium streetlamp, long amber shadows", "palette": "sodium"},
    {"light": "deep blue moonlight, one warm streetlamp, crisp long shadows",
     "palette": "cyan-night"},
]


def clean(text: str) -> str:
    text = FOG.sub("", text)
    text = re.sub(r"\s*,\s*(?=,)", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,")


def restyle_prompt(prompt: str, style: Dict[str, str],
                   other_lights: Sequence[str] = (),
                   remove_fog: bool = True) -> str:
    """Fog out (unless `remove_fog` is False), this scene's light in, anchors kept
    first. Idempotent."""
    drops = set(other_lights) | {style["light"]}
    parts = [(clean(p) if remove_fog else p.strip(" ,"))
             for p in str(prompt or "").split(",")]
    parts = [p for p in parts if p and p not in drops]
    if not parts:
        return style["light"]
    anchors, rest = parts[:2], parts[2:]
    return ", ".join(anchors + [style["light"]] + rest)


def scene_style_map(scene_ids: Sequence[str],
                    styles: Optional[Sequence[Dict[str, str]]] = None
                    ) -> Dict[str, Dict[str, str]]:
    use = list(styles or DEFAULT_STYLES)
    return {sid: use[i % len(use)] for i, sid in enumerate(dict.fromkeys(scene_ids))}


def style_store(slug: str, styles: Optional[Sequence[Dict[str, str]]] = None,
                remove_fog: bool = True,
                out: Optional[Path] = None) -> Dict:
    """Apply the light/palette pass to an adopted store. Returns a summary."""
    import film_store as fs
    project = fs.load(slug)
    order = fs.shots_in_order(project)
    by_scene = scene_style_map([s.get("scene") or "sc1" for s in order], styles)
    lights = [st["light"] for st in (styles or DEFAULT_STYLES)]
    before = {s["id"]: s.get("prompt", "") for s in order}
    for shot in order:
        st = by_scene[shot.get("scene") or "sc1"]
        text = before[shot["id"]]
        new = restyle_prompt(text, st, lights, remove_fog=remove_fog)
        fs.update_shot(project, shot, prompt=new)
    for sc in project.get("scenes") or []:
        st = by_scene.get(sc.get("id"))
        if st:
            fs.set_palette(project, sc["id"], st["palette"])
    project["style_pass"] = {
        "fog_removed": bool(remove_fog),
        "scenes": [{"scene": sc["id"], "palette": sc.get("palette"),
                    "light": by_scene.get(sc["id"], {}).get("light", "")}
                   for sc in project.get("scenes") or []],
    }
    fs.save(project)
    outdir = out or (ROOT / "output" / "style" / slug)
    outdir.mkdir(parents=True, exist_ok=True)
    after = {s["id"]: fs.load_shot(slug, s["id"]).get("prompt", "") for s in order}
    summary = {"slug": slug, "before": before, "after": after,
               "scenes": project["style_pass"]["scenes"]}
    (outdir / "prompts.json").write_text(json.dumps(summary, indent=2),
                                         encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="One light + palette per scene, fog out")
    ap.add_argument("--store", required=True)
    ap.add_argument("--keep-fog", action="store_true",
                    help="do not strip fog/mist/haze words from the prompts")
    a = ap.parse_args()
    summary = style_store(a.store, remove_fog=not a.keep_fog)
    for row in summary["scenes"]:
        print(f"  {row['scene']}: {row['palette']:<11} | {row['light']}")
    print(f"  wrote {ROOT / 'output' / 'style' / a.store / 'prompts.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
