# How human short videos are cut — our own measurement

*Built 2026-10-30 to replace the two commercial blog datasets that the pacing
document previously quoted (551 shots from 9 AI shorts; 1,400 scripts from one
vendor's tool). Those samples were tiny and self-interested. Everything below is
measured on this machine, with code that can be re-run.*

## What was measured, and on what

**Corpus.** `~/Videos/msr_vtt/TrainValVideo` — the MSR-VTT video-captioning
dataset: **7,009 human-made web videos**, 10–30 s each (YouTube: vlogs, gaming,
how-to, clips). That is the closest thing on this machine to the target band this
pipeline shoots for (25–45 s short-form).

* Random samples of **400 + 300 clips** (two independent seeds, 700 total,
  **171.4 minutes**, **3,185 shots**).
* Our own films: the **9 rendered productions** (73 shots, 2.7 min), cut times
  known exactly from their `stats.json`.

**Detector.** `pacing_study.py` samples each video at 8 samples/s (160×90),
computes an HSV histogram distance and a luma difference, and calls a cut when
the change is a one-frame *spike* against its neighbours (fast motion is
sustained; a cut is not). Minimum shot 0.5 s. It detects hard cuts only.

**Validation on known cuts** (our renders): **precision 0.984, recall 1.000,
F1 0.992** at ±0.35 s. It found every detectable cut including the near-monochrome
`project-9` film, after the histogram was fixed to include V (a hue/saturation
histogram had given distance exactly 0.0 on grey films).

**Sampling stability** — two independent random samples:

| | 400 clips, seed 7 | 300 clips, seed 42 |
|---|---:|---:|
| cuts/min | 14.93 | 13.94 |
| pooled ASL | 3.15 s | 3.34 s |
| shot median | 1.63 s | 1.74 s |
| within-clip shot-length CV | 0.74 | 0.77 |

The numbers do not move with the sample, so the shape below is a property of the
corpus, not of one draw.

## The result

![human vs our shot-length distribution](output/pacing_study/distribution.png)

| metric | human short video (700 clips) | our renders (9 films) |
|---|---:|---:|
| shots | 3,185 | 73 |
| cuts per minute (pooled) | **14.5** | **23.8** |
| average shot length (pooled) | **3.23 s** | **2.21 s** |
| shot length, median (p10 / p90) | 1.73 s (0.33 / 8.37) | 2.13 s (1.87 / 2.75) |
| clip ASL, median | 3.54 s | 2.13 s |
| within-clip shot-length CV, median | **0.75** | **0.17** |
| single-shot clips | 6.9% | 22%* |
| clips with ≤2 cuts | 43.8% | 22%* |
| clips cutting faster than 20/min | 29.2% | 78% |

\* our 9-render sample includes two single-shot camera A/B tests; the narrative
films themselves are 0/7 single-shot. Do not read the percentages too hard — the
distribution is the point.

### Shot-length distribution

| shot length | human short video | our renders |
|---|---:|---:|
| under 0.5 s | **14.0%** | 0.0% |
| 0.5–1 s | **22.6%** | 2.7% |
| 1–2 s | 17.8% | 19.2% |
| 2–3 s | 12.0% | **69.9%** |
| 3–5 s | 14.6% | 8.2% |
| over 5 s | **19.0%** | 0.0% |

**55% of human short-video shots are outside 1–5 s. In our films it is 2.7%.**
Human cutting is not "a cut every two seconds": it mixes very short accents
(sub-second inserts, reaction flashes, action beats) with long holds (5–10 s on
one face, one action, one image). The 2–3 s band is the *least* common single
band in the human corpus and the *dominant* band in ours.

### Rhythm

Within-clip shot-length coefficient of variation: **0.75 human vs 0.17 ours**.
Human shorts change pace inside the clip by ~4× more than our films do. Our
timing is close to metronomic: from the earlier audit, 43% of our multi-shot
films had CV < 0.15. The human corpus has almost no such clips (0.7–1.3%).

### What this does and does not prove

* **It proves the arrangement claim**: a uniform 2 s hard-cut montage is not the
  spacing human short video uses. It is a stylistic choice that exists (29% of
  human clips cut faster than 20/min) but it is not the default, and 44% of
  human clips use two cuts or fewer.
* **It does not measure watchability.** Cut rate is not retention; there is no
  audience data here. The earlier blog numbers about completion curves were
  removed for exactly this reason — they were unverifiable and from a vendor.
  What remains is the *arrangement* evidence, which is enough to change the
  rhythm and sequence rules, not enough to claim a retention win.
* **MSR-VTT is web video, not cinema.** Vlogs and gaming montages are not a
  3–7 minute short film. But the pipeline's target is online short-form, and for
  that this is a better reference than nine AI shorts.

## What it means for the pipeline

1. **Widen the rhythm.** The current planner deliberately stays in
   1.7–2.5 s at 16 raw frames — that is the opposite end from human video, which
   is bimodal. A short film needs *both* a fast band (≈0.8–1.5 s: inserts,
   reactions, action accents) and a long band (≈4–8 s: holds, establishing,
   emotional landing), with the 2 s band used sparingly. The renderer can make a
   hold only if it is given more frames (24–32) or a procedural hold; a quick cut
   only by trimming a 16-frame clip (shortening makes motion faster, not slower —
   acceptable for accents). For real sub-second accents, test **8 raw frames**
   as a per-shot setting: 8 frames was rejected as a *global* frame count on
   motion quality, but an accent does not need 2 s of motion. Both are budget
   decisions, not text decisions.
2. **Do not hold every shot for the same time.** The CV target should be a
   check, not just "not all equal": measured human median is 0.75. The current
   R9 check only catches three near-equal holds; it should flag a whole film with
   CV < ~0.3 as metronomic.
3. **The sequence rules are confirmed, not invented.** 43.8% of human clips use
   ≤2 cuts and 6.9% use none — humans hold a place and let an action play. The
   R7–R10 grammar (2–3 scenes, coverage, no new place per cut) is the mechanism
   for that, and the archive replay shows our generated lists violate it
   (25 R7, 43 R8). The two 4-shot sequence A/Bs being rendered test whether a
   coverage arrangement reads better than the montage.

## Reproduce

```powershell
python pacing_study.py --validate
python pacing_study.py --corpus msrvtt --limit 400 --seed 7  --json-out output/pacing_study/msrvtt_400.json
python pacing_study.py --corpus msrvtt --limit 300 --seed 42 --json-out output/pacing_study/msrvtt_300_seed42.json
python pacing_study.py --corpus renders
```
