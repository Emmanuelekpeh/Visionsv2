"""
add_narration.py -- write the voice, then mux it onto a rendered film.

The director model in make_montage.py echoed the act names back as the narration
("settling|descent|turn|after") because the JSON schema invited it to. This asks
for the narration on its own, with the beats as context so the lines actually
follow the film's arc rather than describing it.

    python add_narration.py --film output/montage/montage.mp4 \
                            --direction output/montage/direction.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

NARRATION_PROMPT = """Concept: "{concept}"

The film's beats, in order:
{beats}

Write the spoken narration for this film.

Rules:
- Exactly 4 lines.
- Each line is under 12 words. Spare, concrete, no abstractions stacking up.
- Line 1 sets the world BEFORE. Line 2 is going IN. Line 3 is the TURN, the moment
  it accepts. Line 4 is AFTER, changed.
- Do not describe the images. Do not use the words settling, descent, turn, after.
- No rhyme, no greeting, no title.

Return ONLY JSON: {{"lines": ["...", "...", "...", "..."]}}"""


def write_narration(concept: str, beats, model: str):
    import ollama

    beat_text = "\n".join(f"{i+1}. {b['prompt']}" for i, b in enumerate(beats))
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user",
                   "content": NARRATION_PROMPT.format(concept=concept, beats=beat_text)}],
        format="json",
        options={"temperature": 1.0, "num_predict": 500},
    )
    raw = resp["message"]["content"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
    lines = [l.strip() for l in (data.get("lines") or []) if l and l.strip()]
    return lines[:4]


async def tts(text: str, out_path: Path, voice: str, rate: str):
    import edge_tts

    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--voice", default="en-US-GuyNeural")
    ap.add_argument("--rate", default="-10%")
    ap.add_argument("--gap", type=float, default=1.1,
                    help="silence between lines, seconds")
    args = ap.parse_args()

    film = Path(args.film)
    direction = json.loads(Path(args.direction).read_text(encoding="utf-8"))
    concept = direction.get("concept", "")
    beats = direction.get("beats", [])
    out_dir = Path(args.out) if args.out else film.parent

    print("=== NARRATION (ollama) ===", flush=True)
    lines = write_narration(concept, beats, args.model)
    for i, l in enumerate(lines, 1):
        print(f"  {i}. {l}", flush=True)
    if not lines:
        print("no lines produced; aborting")
        return

    print("\n=== VOICE (edge-tts) ===", flush=True)
    clips = []
    for i, line in enumerate(lines):
        p = out_dir / f"line_{i}.mp3"
        asyncio.run(tts(line, p, args.voice, args.rate))
        clips.append(p)
        print(f"  {i+1}: {p.name}  {p.stat().st_size/1024:.0f} KB", flush=True)

    # Lay the lines across the film rather than reading them back to back, so the
    # voice breathes with the edit.
    print("\n=== MUX ===", flush=True)
    from moviepy.editor import (AudioFileClip, CompositeAudioClip, ImageSequenceClip,
                               VideoFileClip, concatenate_audioclips)

    video = VideoFileClip(str(film))
    total = video.duration
    print(f"  film is {total:.2f}s", flush=True)

    # Place the lines across the film at even fractions rather than reading them
    # back to back, so the voice breathes with the edit instead of overrunning it.
    starts = [total * (0.06 + 0.30 * i) for i in range(len(clips))]
    placed = []
    for path, start in zip(clips, starts):
        clip = AudioFileClip(str(path))
        if start >= total:
            continue
        end = min(start + clip.duration, total)
        placed.append(clip.set_start(start).subclip(0, end - start))
        print(f"  line at {start:5.2f}s  ({clip.duration:.2f}s)", flush=True)

    if not placed:
        print("  no line landed inside the film; nothing written")
        return

    mixed = CompositeAudioClip(placed).set_duration(total)
    final = video.set_audio(mixed)
    out_path = out_dir / (film.stem + "_narrated.mp4")
    final.write_videofile(str(out_path), fps=video.fps, codec="libx264",
                          audio_codec="aac", logger=None)
    print(f"\nwrote {out_path}", flush=True)

    (out_dir / "narration.json").write_text(
        json.dumps({"lines": lines, "starts": starts}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
