# Sequence A/B: montage vs coverage, one action, rendered

*Two 4-shot cuts of the same action, shot through the real store path
(`draft.py` -> `film_store` -> `make_film.py`), same model, same seeds, same
16 raw frames per shot. Watched back to back and measured with
`sequence_ab.py`. This is the pixel-level test of the arrangement question; the
corpus study (`PACING_MEASURED_STUDY.md`) only motivates it.*

## The two cuts

Both are a woman crossing a rain-soaked footbridge at night; a second figure is
somewhere ahead of her. They differ only in how the shots are organised.

**Montage (`_seq_current`)** — 4 shots, 4 scenes, a new place every 2.0 s:

| # | image | hold |
|---|---|---:|
| 1 | wide bridge, two distant figures | 2.0 s |
| 2 | close empty railing and converging planks | 2.0 s |
| 3 | medium woman under an umbrella, *warm-lit alley* | 2.0 s |
| 4 | wide distant silhouette on a *different* bridge | 2.0 s |

**Coverage (`_seq_coverage`)** — 4 shots, **1 scene**, one continuous action:

| # | purpose | image | hold |
|---|---|---|---:|
| 1 | establish | wide, woman entering frame on the bridge | 2.5 s |
| 2 | master | medium, same woman walking away down the bridge | 1.7 s |
| 3 | insert/reaction | closer, her face under the streetlight, same bridge | 1.5 s |
| 4 | resolve | medium, her at the far rail in the same fog | 2.2 s |

![per-shot grid](output/sequence_ab/per_shot_grid.png)

The difference is obvious by eye: the montage changes place *and* subject on
every cut (two figures -> empty railing -> a different woman in a warm alley ->
a silhouette on another bridge). The coverage cut stays on one bridge, with the
same figure at four distances, and the cuts land on the action (enter, walk,
face, arrive). Shot 3's face is still a different *generation* of the woman --
this pipeline has no character lock -- but it is the same place, light and
action, so it reads as her.

## Measured

![side-by-side](output/sequence_ab/sequence_ab.gif)

| metric | montage | coverage | reading |
|---|---:|---:|---|
| scenes / shots | 4 / 4 | **1 / 4** | montage = a new place per cut |
| shot-length CV | **0.00** | **0.20** | both far below the human short-form median 0.75; the renderer's 16-frame budget leaves little room |
| cross-cut CLIP similarity (mean) | 0.859 | **0.878** | higher = next shot looks like the same place/thing |
| luma jump across cuts (mean) | 2.2 | **1.5** | lower = smoother |
| colour-histogram jump (mean) | 0.569 | **0.362** | lower = same palette/place |
| motion-direction disagreement (mean) | 90.2° | 88.8° | not meaningful: both cuts are slow (0.1-0.9 px/frame), angle is noise at that magnitude |

Per cut:

| cut | CLIP montage / coverage | hist jump montage / coverage |
|---|---|---|
| 1 -> 2 | 0.832 / **0.894** | **0.343** / 0.510 |
| 2 -> 3 | **0.847** / 0.740 | 0.743 / **0.565** |
| 3 -> 4 | 0.899 / **0.999** | 0.622 / **0.009** |

The last cut is the clearest number: montage jumps from a warm alley to a cold
distant bridge (histogram jump 0.622, luma jump 2.5); coverage cuts from the face
to her at the rail in the same light (histogram jump **0.009**, luma jump 0.0,
CLIP 0.999). The first cut is mixed: CLIP and luma favour coverage, but the
colour histogram favours the montage because the coverage cut goes wide -> medium
on the same bridge (content changes) while the montage stays on similar foggy
tonality. The middle cut is the one place the montage scores higher on CLIP: the
coverage close-up (a face) is semantically further from the wide bridge than the
montage's railing is from the bridge. Together these are the limits of these
metrics: CLIP likes same-content, not same-place, and it is quite insensitive to
the palette continuity that matters here (all these shots are blue-grey fog).
The three numbers point the same way on average (CLIP +0.019, luma -0.7,
histogram -0.207 for coverage), but no single cut is unanimous.

## Verdict

**The coverage arrangement reads better, and the measurable continuity agrees.**
The montage's four new places are exactly the "two seconds, hard stop, new
scenery, repeat" failure; the coverage cut is a scene. The corpus study says
human short video does this (43.8% of clips use <=2 cuts) and mixes holds; this
pair shows the same at the level of one action.

What this does **not** prove:

* **Not watchability.** Four seconds of difference in continuity is not a
  retention measurement; no audience was involved.
* **Not that the grammar is sufficient.** Even the coverage cut has
  shot-length CV 0.20 against a human median of 0.75: the holds are still
  clustered around 2 s because 16 raw frames supply 1.94 s. The next change has
  to be per-shot frame budgeting (8-12 frames for accents, 24-32 for holds), or
  a procedural hold on a static frame, which is what `camera_path.py` can
  already do for moves.
* **Character continuity is still absent.** Coverage makes the *place* cohere;
  the face changes between shots 2 and 3. That is the open character problem,
  deliberately out of scope here.

One design finding: coverage is a **single-scene** film, and `story.validate`
still emits "only one scene: no boundary for a look or model change". The
validator is right about the look/palette stage, but for short films one scene
with coverage is legitimate (this one is the better cut). The message should be
downgraded or scoped to films long enough to need a palette change; it does not
block adoption (only `story:`/`structure:` do), but it is the wrong advice for a
4-8 shot film.

## Reproduce

```powershell
python make_film.py --store _seq_current  --name seq_current
python make_film.py --store _seq_coverage --name seq_coverage
python sequence_ab.py --current _seq_current --coverage _seq_coverage
```

Artifacts: [per_shot_grid.png](output/sequence_ab/per_shot_grid.png) ·
[sequence_ab.gif](output/sequence_ab/sequence_ab.gif) ·
[sequence_ab.mp4](output/sequence_ab/sequence_ab.mp4) ·
[sequence_ab.json](output/sequence_ab/sequence_ab.json) ·
[montage_shots.png](output/sequence_ab/montage_shots.png) ·
sequences: `output/film/seq_current/`, `output/film/seq_coverage/`
