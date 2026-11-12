# Pacing, shot design and watchability — what the code actually does

*Written 2026-10-29. Every number here is either measured on this machine today
(`pacing_audit.py`) or quoted from the file/line named beside it. The showrunner
model comparison is in [`SHOWRUNNER_TEST.md`](SHOWRUNNER_TEST.md); this document
is about the other half of "making nice videos": what the shots are, how long
they last, and what is stopping the film from flowing.*

---

## 0. TL;DR

1. **A shot contains 1.94 s of real motion.** `settings.frames` (default 16) is
   RAFT-upsampled to `2n-1 = 31` frames at 16 fps. The delivered "length" is a
   separate decision: `hold_frames = round(seconds × 16)`. `make_film.render`
   resamples the 31-frame clip to the hold by index. Any hold above 1.94 s is
   frame duplication — the motion slows down and judders. Above ~1.3× it is
   visible.
2. **The shipped (UI/store) path did not run the planner — fixed today.** It used
   to write the writer's own `seconds` (asked for 1.5–3.6 s) into the store, while
   the prompt told the writer "the planner imposes a rhythm curve and will
   overwrite your seconds". That was true only for the CLI. Now
   `film_store.shots_from_proposal` runs `edit_planner.plan` at adoption, so the
   store film gets R4's beat-shaped rhythm and R1's framing repair, and the UI
   shows exactly what was re-timed (`project["pacing"]`).
3. **The planner curve was disconnected from the renderer — fixed today.** The
   uncommitted beat curve assigned 4.5–6.5 s establish holds and up to 7 s
   resolve holds (`MAX_SHOT = 8`) to a 1.94 s source; that needs 2.4–3.1×
   duplication. `edit_planner.plan()` now takes `frames`/`fps` and clamps every
   hold to `handheld.max_hold_seconds`, reporting an unreachable target in
   `EditReport.notes` instead of faking it. The UI warns on any manual hold that
   would judder.
4. **Length is expensive and the price is almost frame-count independent.** A
   30 s film costs ~3.1 h at 16 frames (16 shots) or ~3.1 h at 32 frames (8
   shots). The choice is **many short cuts vs fewer smooth shots**, not "long
   for free". The UI offers a shot count (2–24) and no length, and never shows
   this arithmetic.
5. **The shot vocabulary is narrower than the pipeline thinks.** All 11 audited
   productions contain **0–1 camera moves**. The store sets `camera="none"` on
   adoption (`film_store.shots_from_proposal`) and the director that chooses
   moves is only invoked by the non-store `make_film` path. R2/R3/R6 (camera
   continuity) are inert in the product. The writer's `camera_suggestion` is
   shown as a caption and then ignored.
6. **The story gate was a warning, not a gate — fixed today.** The showrunner
   re-rolled blindly up to 3 times and kept the least-bad attempt, then `draft.py`
   adopted it even if the beats were broken (9 of 11 productions are
   non-monotonic: `longing` `EEdTdREEETdR`, `seasoning` `dTTTddRRRRRRR`). Now the
   retry is a *repair* retry (`_repair_note` names the validator's complaints and
   the beats it produced), and `draft.py` **refuses to adopt** a proposal with
   `story:` breaks unless `--allow-broken-story` / the UI checkbox says so.
7. **There is no sound.** Every render is silent (`"silent": true`). For
   short-form watchability this is the largest single gap not addressed below.

---

## 0b. What changed in the code today

| change | file | evidence |
|---|---|---|
| one definition of the renderer's motion budget | `handheld.py:615-660` | `natural_seconds`, `max_hold_seconds`, `stretch_ratio`, `fit_hold_seconds` |
| R4 holds clamped to real motion; unreachable targets reported | `edit_planner.py:120-330` | `pacing_audit.py --curve 6` now maxes at 1.23× |
| **the store path now runs the planner at adoption** | `film_store.py:1057` | R4 rhythm + R1 framing repair + budget; `project["pacing"]` records it |
| UI shows the auto-paced holds and warns on manual judder | `ui/studio.py:803` | `N of 31 frames = X×`, "WILL JUDDER" |
| CLI budget/shots use the motion budget | `make_film.py:76-96` | 30 s at 16 frames → ~15 shots, ~3 h |
| **story gate: broken beat order is refused at adoption** | `draft.py` `story_gate` | `--allow-broken-story` is the explicit override |
| **failure-specific repair re-prompt** | `story.py` `_repair_note` | retry names the validator's complaints + previous beats |
| **camera direction reaches the render** | `camera_direction.py`, `draft.py` | 4 policies; hybrid = 2 moves on `seasoning` vs 0 stored before (measured) |
| **A/B tool for camera policy** | `camera_ab.py` | decision-level comparison before spending render hours |
| **visual camera A/B + three-way (LoRA vs procedural path)** | `camera_ab_visual.py`, `camera_path.py` | tilt: −68.9 px vs +2.8 px; procedural settles, LoRA does not (`CAMERA_VISUAL_AB.md`) |
| **sequence grammar: purpose/link fields, sequence brief, R7–R10** | `story.py`, `edit_planner.py`, `draft.py` | gate now refuses R7/R8/R10 like story breaks; planner repairs R10 |
| **organisation metrics in the audit** | `pacing_audit.py` | scene runs/cuts, within vs across overlap, same-subject, hold sd, move→cut |
| **grammar replay tool** | `sequence_audit.py` | 29 cached lists: 25 R7, 43 R8, 23 R10 violations (before the new brief) |
| **own pacing measurement (replaces the blog numbers)** | `pacing_study.py` | 700 MSR-VTT clips, 3,185 shots; detector F1 0.992; human 14.5 cuts/min, CV 0.75 vs ours 23.8/min, CV 0.17 |
| **metronome check** | `story.structure_problems` R9 | a film with shot-length CV < 0.20 is flagged, not just 3 equal holds |
| **sequence A/B measured and rendered** | `sequence_ab.py` | coverage wins continuity (CLIP 0.878 vs 0.859, hist jump 0.362 vs 0.569); `SEQUENCE_AB.md` |
| audit + A/B tools, 40+ new tests | `pacing_audit.py`, `camera_ab.py`, `sequence_audit.py`, `tests/test_*` | `python -m pytest tests` 162 passed |

Still not changed (needs a product decision): character/subject continuity has no
mechanism, renders are silent, and a VLM visual supervisor is designed but not
built (see §8).

---

## 1. How a shot's length is actually decided

There are two independent quantities and the code never compares them.

### 1.1 Generated motion

```
raw frames (settings.frames, default 16)
    → AnimateDiff
    → RAFT x2                (make_film.raft_upsample, factor=2)
    → 2*16 - 1 = 31 frames
    → delivered at FPS = 16  (make_film.py:48)
    → 31 / 16 = 1.94 s of real motion
```

`FPS = 16  # after RAFT x2 of an 8fps base` — `make_film.py:48`.
`raft_upsample` writes the first frame then, per pair, one warp per factor and
the original `b`, so n frames give n + (n-1) = 2n-1 (`make_film.py:402-433`).

This is the amount of motion the model actually invented. Everything after this
point is repetition of it.

### 1.2 Delivered hold

```
hold_frames = max(1, round(seconds * fps))         handheld.py:603
render(): resample src (31 frames) to hold frames by index
          idx = [round(k*(len(src)-1)/(length-1)) for k in range(length)]
          frames.extend(src[j] for j in idx)        make_film.py:448-456
```

The same mapping is applied earlier, before the camera shake, in the store path
(`stretch_to`, `handheld.py:608-617`, called from `make_film.py:633`), so the
hold is the only timeline the shake is built against. That is correct — but it
means the *content* is already duplicated before the shake is added.

A static shot stretched 2× is a still image with smooth camera shake: the eye
accepts it. A **moving** shot stretched 2× is duplicated motion: judder. This is
why the recommendation below distinguishes them.

### 1.3 Which path sets `seconds`

| path | who sets `seconds` | planner applied? |
|---|---|---|
| UI → `draft.py` → store (the product) | the showrunner's raw value, 1.5–3.6 s | **no** |
| CLI `make_film --shots llm` | `edit_planner.plan()` rhythm curve | yes |
| UI shot editor (`ui/studio.py:782`) | human, 1.0–8.0 s | n/a |

Evidence: `draft.py:60-95` adopts `proposal.shots` directly via
`film_store.shots_from_proposal` (`film_store.py:1057-1094`), which stores
`s.seconds`; `make_film --store` reads the plan from
`film_store.export_plan` (`film_store.py:999`), which never calls `plan()`.
Meanwhile `story.SYSTEM` tells the writer:

> "'seconds': do not agonise over it. The planner imposes a rhythm curve and
> will overwrite your values, so just keep them between 1.5 and 3.6."
> — `story.py` system prompt

So the writer is told the number does not matter, and in the product the number
is the only thing that matters. That is the single clearest bug in the pacing
story.

---

## 2. What the films actually look like (measured today)

`python pacing_audit.py` over every production in `productions/`. `stretch` =
delivered hold ÷ generated motion (31 frames). `judder` counts shots over 1.3×.

| production | shots | sec | mean/shot | natural | stretch mean | judder | moves | beats | gen hours |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `_fx_ui` | 5 | 11.2 | 2.24 | 1.44¹ | 1.56 | 5 | 0 | `EdTTR` | 1.0 |
| `ego-death` | 10 | 20.5 | 2.05 | 1.94 | 1.06 | 0 | 0 | `EdTTddTRRR` | 0.8 |
| `film0` | 4 | 10.3 | 2.58 | 1.94 | 1.33 | 2 | 0 | `EdTR` | 0.0 |
| `film2` | 8 | 20.2 | 2.52 | 1.94 | 1.30 | 4 | 0 | `EETTddRR` | 1.0 |
| `flowcheck` | 6 | 12.3 | 2.05 | 1.94 | 1.06 | 0 | 0 | `dETdRR` | 5.2 |
| `frame-probe` | 5 | 10.4 | 2.08 | 1.44¹ | 1.44 | 4 | 0 | `EdTRR` | 0.2 |
| `ionos` | 11 | 23.4 | 2.13 | 1.94 | 1.10 | 0 | 0 | `EEEdTTTTTRR` | 1.9 |
| `longing` | 12 | 24.3 | 2.02 | 1.94 | 1.04 | 0 | 0 | `EEdTdREEETdR` | 2.0 |
| `project-9` | 12 | 27.8 | 2.32 | 1.94 | 1.20 | 3 | 0 | `EEddTTddTTRR` | 10.3² |
| `seasoning` | 13 | 29.2 | 2.25 | 1.94 | 1.16 | 1 | 0 | `dTTTddRRRRRRR` | 3.3 |
| `shadows-take1` | 4 | 10.9 | 2.73 | 1.94 | 1.41 | 2 | 1 | `EdTR` | 0.0 |

¹ rendered at 12 raw frames → natural motion 1.44 s, so the same writer seconds
duplicate more. ² includes re-takes; not one clean pass.

**What the table says**

* The shipped mean shot is **2.0–2.7 s**. By the external short-film data in §4
  (median 3 s) the pipeline cuts *faster* than a typical short film, not slower.
* Films rendered at **12 raw frames** (`_fx_ui`, `frame-probe`) are the worst
  offenders: every shot judders because the writer's ~2.1 s holds are 1.44–1.56×
  the motion available. The 27% speed saving from 12 frames is real, and it is
  paid for in motion.
* Films at 16 frames with writer seconds near 2.0 s (`longing`, `ego-death`,
  `ionos`) are essentially clean — that is the configuration the renderer is
  happiest with.
* **Every production except `shadows-take1` has zero camera moves.** The
  "camera continuity" rules (R2/R3/R6) and the writer's camera suggestions have
  no effect on what is shot.

The writer's `problems` stored in these projects confirm the story half:

* `longing`: `story: 5 establish beats -- only the first shot establishes`,
  `story: beats go backwards (4:turn->develop, 6:resolve->establish,
  10:turn->develop)`, plus a `story-link` warning that the resolve does not pay
  off the turn.
* `seasoning`: `story: beats go backwards (6:turn->develop)` and a beat shape of
  `dTTTddRRRRRRR` — opens on develop, five turns, seven resolves.
* `project-9`: `2 establish beats`, backwards beats.

The re-roll loop in `write_shots_ensemble` (`story.py`) keeps the best of up to
three attempts. When all three are broken it keeps the least-bad and returns it;
`draft.py` adopts it regardless. So the validator **reports** a broken story and
the pipeline **ships** it. The UI does print the problems in prose
(`ui/studio.py:637-641`), which is honest but not a gate.

---

## 3. The renderer is the binding constraint

Measured on this machine: **~43 s per raw frame** (4-step Lightning, 512×512;
README "Performance", `make_film.MIN_PER_SHOT = 9.5`). A 16-frame shot is
~10–12 min in `seasoning`/`longing` history. Fixed per-job costs are amortised
when shots are rendered in one job.

### 3.1 Natural motion per frame setting

| raw frames | delivered frames | natural motion @16 fps | render/shot |
|---:|---:|---:|---:|
| 12 | 23 | 1.44 s | ~8.6 min |
| 16 | 31 | 1.94 s | ~11.5 min |
| 24 | 47 | 2.94 s | ~17.2 min |
| 32 | 63 | 3.94 s | ~23.0 min |

### 3.2 Length → shots → hours (`pacing_audit.py --budget`)

| target | frames | shots needed for stretch ≈ 1.0 | hours | stretch if you insist on 6 shots |
|---|---:|---:|---:|---:|
| 25 s (Reels) | 12 | 18 | 2.6 h | 2.90× |
| 25 s | 16 | 13 | 2.5 h | 2.15× |
| 25 s | 24 | 9 | 2.6 h | 1.42× |
| 25 s | 32 | 7 | 2.7 h | 1.06× |
| 30 s (TikTok) | 12 | 21 | 3.0 h | 3.48× |
| 30 s | 16 | 16 | 3.1 h | 2.58× |
| 30 s | 24 | 11 | 3.2 h | 1.70× |
| 30 s | 32 | 8 | 3.1 h | 1.27× |
| 45 s (Shorts) | 12 | 32 | 4.6 h | 5.22× |
| 45 s | 16 | 24 | 4.6 h | 3.87× |
| 45 s | 24 | 16 | 4.6 h | 2.55× |
| 45 s | 32 | 12 | 4.6 h | 1.90× |

The hours column is almost constant down each block. **Length costs render time;
the frame setting does not change the bill, it changes the shot count.** More
frames = fewer, longer, smoother shots. Fewer frames = more cuts, more variety,
more judder unless holds are shortened.

That is the real creative dial, and the UI exposes neither half of it: it asks
for "shots" (2–24) with no length, and settings asks for raw frames with no
statement of what that does to hold length.

### 3.3 What `edit_planner` assigns — before and after the fix

**Before (the uncommitted worktree change, now corrected).**
`_rhythm_curve_beats` had fixed seconds ranges (establish 4.5–6.5, resolve
4.0–7.0) and a cap of 8 s. `pacing_audit.py --curve 6` showed:

| # | beat | seconds | hold | stretch | verdict |
|---:|---|---:|---:|---:|---|
| 1 | establish | 5.10 | 82 | 2.65 | severe |
| 2 | develop | 3.35 | 54 | 1.74 | judder |
| 3 | turn | 2.82 | 45 | 1.45 | judder |
| 4 | resolve | 6.02 | 96 | 3.10 | severe |
| 5 | resolve | 5.19 | 83 | 2.68 | severe |
| 6 | resolve | 5.56 | 89 | 2.87 | severe |

28 s of film from 11.6 s of generated motion.

**After (this session).** The curve is expressed as factors of the generated
motion — establish 1.00–1.25×, develop 0.88–1.05×, turn 0.78–0.95×, resolve
1.05–1.30× — then clamped by `max_hold_seconds` (1.30× static, 1.15× moving).
Same six beats at 16 frames:

| # | beat | seconds | hold | stretch | verdict |
|---:|---|---:|---:|---:|---|
| 1 | establish | 2.05 | 33 | 1.06 | ok |
| 2 | develop | 1.84 | 29 | 0.94 | trim |
| 3 | turn | 1.71 | 27 | 0.87 | trim |
| 4 | resolve | 2.18 | 35 | 1.13 | ok |
| 5 | resolve | 2.27 | 36 | 1.16 | ok |
| 6 | resolve | 2.36 | 38 | 1.23 | ok |

12.4 s total, mean stretch 1.06, no judder. When a caller asks for `total=30`,
scaling pins every hold to the ceiling — which makes them uniform — so the
planner keeps the shaped curve and reports the shortfall in
`EditReport.notes` instead.

Three places now obey the budget:

* `edit_planner.plan(..., frames=, fps=)` for the CLI path.
* `film_store.shots_from_proposal` for the UI/store path: adoption now RUNS the
  planner, so the stored cut gets R4's rhythm and R1's framing repair, and the
  writer's `seconds` are replaced rather than clamped.
* `make_film.budget_report` / the CLI shot-count derivation, which size a film
  from the motion budget instead of a 2.5 s average.

---

## 4. What our own measurement says about pacing

The blog-derived numbers that were here are gone. They were 551 shots from 9 AI
shorts and 1,400 scripts from one vendor's tool — too small and too interested.
They are replaced by a measured study on this machine:
[`PACING_MEASURED_STUDY.md`](PACING_MEASURED_STUDY.md). Summary:

* **Corpus**: 700 randomly sampled **MSR-VTT** human-made web videos (10–30 s),
  171.4 min, **3,185 shots**, plus our 9 rendered films (73 shots, exact cuts).
* **Detector**: our own histogram + luma-spike cut detector, validated on the
  known cuts of our renders at **F1 0.992** (precision 0.984, recall 1.000).
  Two independent samples agree (cuts/min 14.93 vs 13.94).

![human vs our shot-length distribution](output/pacing_study/distribution.png)

| | human short video | our renders |
|---|---:|---:|
| cuts/min | **14.5** | 23.8 |
| average shot length | **3.23 s** | 2.21 s |
| shot-length p10 / median / p90 | 0.33 / **1.73** / 8.37 s | 1.87 / **2.13** / 2.75 s |
| within-clip shot-length CV | **0.75** | 0.17 |
| shots under 1 s | **36.6%** | 2.7% |
| shots 2–3 s | **12.0%** | **69.9%** |
| shots over 5 s | **19.0%** | 0.0% |

**Human short-form cutting is bimodal, not uniform.** It mixes sub-second
accents with 5–10 s holds and changes pace inside the clip (CV 0.75). Our films
are a spike at 2–3 s (70% of shots), within a narrow 1.9–2.8 s window
(CV 0.17). Fifty-five percent of human shots are outside the 1–5 s band; in ours
it is 2.7%.

What that changes:

* The rhythm target is **two bands** — quick accents (≈0.8–1.5 s) and holds
  (≈4–8 s) — with 2 s used between them, not everywhere. The renderer can only
  do this if frames are budgeted per shot: a 16-frame shot supplies 1.94 s, so a
  hold needs 24–32+ frames (or a procedural hold), and a sub-second accent needs
  trimming (which speeds motion up).
* A film whose shot-length CV is low (say < 0.3) is metronomic even if no three
  holds are identical; the R9 check should flag the whole shape, not just runs.
* **It does not measure retention.** Cut rate is arrangement, not watchability;
  there is no audience data here. The earlier completion-curve numbers were
  unverifiable vendor claims and have been removed.

The caveat in the other direction: MSR-VTT is web video (vlogs, gaming, how-to),
not 3–7 minute narrative shorts. It is the closest local reference to the
platforms this pipeline targets, and a far better one than nine AI shorts.

---

## 5. What is limiting flow, coherence and watchability — ranked

### 5.1 Flow (does one shot lead to the next?)

| cause | where | effect |
|---|---|---|
| Timing not tied to generated motion | §1–3 | duplicated frames, judder; rhythm is accidental |
| Two timing paths disagree | `draft.py` vs `plan()` (**fixed**: adoption runs the planner) | the store cut now gets a rhythm, not the writer's guesses |
| Planner curve ignores the renderer | `edit_planner.py:120-190` (**fixed**: factors of generated motion, clamped) | was 2.4–3.1× stretch; now 0.87–1.23× at 16 frames |
| Store path never normalises holds | `film_store.shots_from_proposal:1057` (**fixed**: runs `plan()` at adoption) | R4 rhythm + R1 framing repair, recorded in `project["pacing"]` |
| Director not in the store path | `camera_direction.py` (**fixed**: adoption stores a camera decision; `hybrid` default) | measured 0–1 moves before, 1–2 after on real films |
| Scale-jump rule exists but no content-level linkage | `edit_planner.R1` only | no match cuts, no eyeline, no movement direction between shots |
| Hard cuts only | `R5` | fine for short-form, but no fades/dissolves means transitions cannot carry time |

### 5.2 Coherence (do the shots belong to one film?)

| cause | where | effect |
|---|---|---|
| No character/subject reference | story prompts only; no image reference | the "recurring figure" is re-invented per shot; faces/wardrobe drift |
| Recurring subject not enforced | story.py comment "REJECTED: subject must be present" | the resolve can be a new location with no one in it |
| Scene boundaries exist but are unused | `scenes` → palette, `settings.models="off"` | no colour/checkpoint anchoring per scene in the audited films |
| Story gate not enforced | `draft.py` `story_gate` (**fixed**: refuses `story:` breaks unless overridden) | broken beats no longer ship by default |
| Single image per shot, LLM may invent instead of using farms | `provenance` measures it | when provenance is low the farm architecture was decoration |

### 5.3 Watchability (would a stranger finish it?)

| cause | where | effect |
|---|---|---|
| No audio | `add_narration.py` exists, not wired; renders `"silent": true` | biggest single gap |
| Hook not modelled | first shot forced `establish`; no test of the opening | weak first second is a hypothesis, not measured |
| Length not chosen | UI has no length; shot count is the only dial | films land at 10–30 s and 0.8–10 h accidentally |
| Shot variety is real but shallow | 5 scales × content; no coverage/insert/reaction grammar | "type of shots" limited by vocabulary, not just count |
| Long stretches of one scale rhythm | writer free to alternate scales only by R1 | no deliberate wide→close→wide pattern |

### 5.4 The organisation problem: "hard stop, new scenery, repeat"

A hard cut is not the problem — hard cuts are the grammar (R5). The problem is
that the film is **not organised**: each shot is an independent two-second
postcard, and the scene label is only a palette group. Measured on the archive
(`pacing_audit.py` organisation table, `sequence_audit.py`):

| production | shots | scene labels | hard stops | within-scene vs across-scene prompt overlap | adjacent same-scene pairs sharing a subject |
|---|---:|---:|---:|---:|---:|
| `seasoning` | 13 | 3 | 2 | **0.05 vs 0.08** | **0 / 10** |
| `project-9` | 12 | 6 | 5 | 0.02 vs 0.03 | 2 / 6 |
| `longing` | 12 | 3 | 6 runs (interleaved) | 0.06 vs 0.01 | 3 / 6 |
| `ionos` | 11 | 3 | 2 | 0.07 vs 0.12 | 5 / 8 |

Within-scene overlap is no higher than across-scene overlap in half the archive:
**the scene labels were cosmetic**, because nothing asked the shots inside a
scene to cover one action. Replaying the new grammar over 29 cached proposals
found **25 R7 (scene structure), 43 R8 (coverage) and 23 R10 (move handoff)**
violations even before counting missing purpose/link fields.

The layer that was missing:

| rule | what it enforces | where |
|---|---|---|
| **R7** scene structure | 2–3 scenes, ≤2 hard stops, no interleaving (A-B-A), no one-shot scenes | `story.structure_problems` |
| **R8** coverage | each multi-shot scene has ≥2 framing scales, a shared subject between adjacent shots, and >1 purpose | `story.structure_problems` |
| **R9** rhythm | scene opening is the longest shot in its scene; no three consecutive near-equal holds | `story.structure_problems` (planner owns timing) |
| **R10** move handoff | a camera move may not be the last shot before new scenery; the planner rewrites it to `none` | `edit_planner.plan` + validator pre-check |

The writer is now briefed to produce **sequences with coverage**: every shot has
a `purpose` (establish/master/insert/reaction/resolve) and a `link` to the
previous shot (open/match-action/eyeline/movement/graphic/scene-cut). The
adoption gate blocks `story:` and `structure:` breaks; the failure-specific
retry names both. `draft.py --allow-broken-story` is still the explicit override.

### 5.5 What the pipeline does well (do not break it)

* The **measured** renderer budget is documented and right (`MIN_PER_SHOT`,
  43 s/frame, four-step Lightning).
* R1 (no adjacent same scale) and R6 (no reversed axis) are real editing rules
  and are enforced on the CLI path.
* The story validator catches exactly the failure it was built for (mood boards
  with no turn) — the problem is that its verdict does not stop the film.
* Provenance measures whether the farms were used; that is the right way to keep
  the ensemble honest.
* Retiming is cheap: changing `seconds` only re-finishes (stretch + shake +
  grade), it does **not** re-render the generator (`film_store.py:631-635`).
  Pacing can therefore be fixed in the cut after an expensive render.

---

## 6. Recommendations, in order of impact per hour of work

### R1 — Make timing one function, and make it obey the renderer — **DONE**

Implemented this session:

* `handheld.natural_seconds()` / `max_hold_seconds()` / `stretch_ratio()` /
  `fit_hold_seconds()` are the single definition of the renderer's motion
  budget (`handheld.py:615-660`).
* `edit_planner.plan(..., frames=, fps=)` clamps every R4 hold to that budget
  and reports a target it cannot reach in `EditReport.notes`
  (`edit_planner.py:194`).
* `film_store.shots_from_proposal()` **runs the planner at adoption** (the UI
  path would otherwise keep the writer's guesses): R4 rhythm, R1 framing repair,
  budget clamp, and a `project["pacing"]` record of writer holds → planned holds
  (`film_store.py:1057`).
* The UI length control shows `N of 31 generated frames = X×` and says
  "WILL JUDDER" above 1.30×, and the write screen shows what adoption re-timed
  (`ui/studio.py:803`).
* `make_film.budget_report()` and the CLI shot-count derivation use the real
  motion budget instead of a hard-coded 2.5 s average (`make_film.py:76`).

**What remains:** existing productions' stored cuts are not migrated (their
judder shots stay until retimed — which is cheap, a re-finish), and the UI still
asks for a shot count, not a target length (R2).

### R2 — Put length and budget on the write screen — **PARTIAL**

Done: the write screen now shows `N shots ≈ N × 1.94 s of film · N × 11 min of
rendering`, derived from `settings.frames`, so the price is visible before the
click (`ui/studio.py`).

Missing: the inverse. There is still no "make me a 30 s film" control that sets
the shot count and shows the hours; the user picks shots and gets a length. The
math exists in `pacing_audit.py --budget` and should be lifted into one helper
used by both.

### R3 — Make the story gate act — **DONE**

Implemented, both halves:

1. **At adoption**, `draft.py::story_gate` refuses a proposal with `story:`
   breaks (non-monotonic beats, no turn, unresolved ending) and leaves the
   production untouched. `--allow-broken-story` (and the UI checkbox "adopt a
   broken story") is the only way to ship one, so it is an explicit decision.
   Warnings (`story-link:`) stay advisory, as designed.
2. **At generation**, the blind re-roll became a repair retry:
   `story._repair_note()` gives the next attempt the validator's exact
   complaints, the beats it produced, and its previous prompts, and asks it to
   fix only those. The retry note is part of the user message, and a clean
   attempt still stops the loop.

**What remains:** the repair prompt is one shot at the model per retry; if all
three attempts still fail, the draft job now exits 2 instead of shipping. That
is the correct behaviour, but the UI has to make "re-run" the obvious next move
— it shows the log's last lines in the failure banner today.

### R4 — Decide what camera moves are for — **DONE, with an honest limit**

The half-state is gone: the store path now stores a real camera decision at
adoption (`camera_direction.py`), and R2/R3/R6 have something to repair. Four
policies, selectable in Settings (`camera_mode`) or per draft (`--cameras`):

| mode | who decides | measured on `seasoning` (13 shots) | on a fresh granite 6-shot list |
|---|---|---|---|
| `off` | nobody (old behaviour) | 0 moves | 0 moves |
| `suggestions` | the writer | 1 move | 2 moves |
| `director` | `director.choose()` | 1 move (different shot) | 2 moves |
| **`hybrid` (default)** | director first, writer where it declines | **2 moves** | 2 moves |

`camera_ab.py` is the A/B harness — it compares the policies on one production or
one raw proposal, reports moves / reversals / planner repairs, and prints the
per-shot decision matrix so the choice can be read against the prompts. No
rendering is implied: a real A/B needs watching, not a metric.

Two measured limits, both deliberate:

1. **The director is conservative.** `none` starts at fit 0.62 and a softmax over
   nine camera candidates rarely beats it without a hint word in the prompt; on
   the 13-shot film it moves 1 shot. That is safe but nearly inert alone, which
   is why `hybrid` is the default rather than `director`.
2. **On a 6-shot draft the director can fill the whole 2-move budget**, so hybrid
   equals director there (measured). The writer's suggestion buys extra movement
   mostly on longer films. Subject-motion LoRAs are opt-in
   (`include_subject=True`) and have not been A/B'd.

The A/B also caught two real bugs before they shipped: uncapped hybrid produced
3 moves in a 6-shot film against a brief of 2, and "none" being a *valid* camera
name let empty suggestions consume the mover budget. Both are regression-tested.

**Visual A/B (done, one pair).** `tilt-down` vs `none` on the same shot, seed
1911, 2.0 s hold, 16 raw frames, rendered through the real store path
(~12 min per take):

| | tilt-down | none |
|---|---:|---:|
| net vertical travel | **−68.9 px** | +2.8 px |
| frame transitions down / up | **15 / 0** | 6 / 7 |
| sharpness (Laplacian var) | 38.1 | 76.7 |

The decision reaches the pixels and moves in one direction repeatedly (the
acceptance test the text A/B could not run). The move is legible in the frames.
The camera-LoRA take is also softer **in this one pair**, but it regenerated a
different composition, so that is not yet explained. Full method, frames and
caveats: [`CAMERA_VISUAL_AB.md`](CAMERA_VISUAL_AB.md). Do not make camera moves
the default for final renders until a second pair separates LoRA cost from
content.

### R5 — Buy smoothness with frames only where it pays (medium, render cost)

Generate long shots with 24–32 frames when the shot is meant to hold (establish,
resolve, emotional beat) and short ones at 16 when the cut should be quick. The
budget table shows total hours are roughly constant per target length, so this is
a redistribution, not an expense. Per-shot `frames` needs the generation hash to
include it (see HANDOFF §0.4 rules) — do not add it casually.

A cheaper version that needs no code: render the whole film at 24–32 frames,
then trim every shot in the cut to the rhythm you want. Trimming is a re-finish
(seconds, not minutes) and trimming never judders — only stretching does.

### R6 — Story order for short-form (a creative decision, not measured)

Allow the first shot to be a **hook detail** (`close`/`macro`, one piece of
information) before the establish, or a non-monotonic cold open that returns to
establish. The current rule ("first shot is establish, beats never go backwards")
is feature-film grammar. That short-form rewards a 0–3 s hook is a reasonable
editing hypothesis, **not something this project has measured** — the earlier
blog-derived claim was removed. Try it as a creative variant, not a rule.

### R7 — Sound (large, separate project)

Silence is the biggest missing watchability lever. `add_narration.py` already
exists; music/ambient bed plus a simple duck would change more in viewer
retention than any of R1–R6.

### R8 — Clean up the worktree defaults — **DONE**

`MAX_SHOT` back to 5.0, the beat curve expressed as factors of generated motion,
14 tests pinning the timing contract. (Kept as a record: this was the change that
made `edit_planner` stop promising 5–7 s holds on a 1.94 s source.)

### R9 — Organise the film into sequences (grammar R7–R10) — **DONE (text+planner), A/B rendering**

Implemented this session:

* `purpose` (establish/master/insert/reaction/resolve) and `link`
  (open/match-action/eyeline/movement/graphic/scene-cut) are schema fields,
  validated, stored on shots and shown in the shot editor.
* The writer brief now demands 2–3 sequences covering one place/action each,
  ≤2 hard stops, no interleaving, scene-opening shot longest, and moves that
  continue in the same scene (`story.SYSTEM`, `SHOWRUNNER_ADDENDUM`).
* R7/R8/R9 are reported by `story.structure_problems`; R10 is *repaired* by
  `edit_planner.plan`; the adoption gate blocks `story:` and `structure:`
  breaks; the retry repair note names them all.
* `pacing_audit.py` reports the organisation numbers; `sequence_audit.py`
  replays the grammar over cached proposals/productions.

**Rendered and compared (SEQUENCE_AB.md).** Two 4-shot cuts of one action, same
seeds through the real store path: a montage (4 scenes, 4 x 2.0 s) vs coverage
(1 scene, 2.5/1.7/1.5/2.2 s). The coverage cut wins on cross-cut CLIP similarity
(0.878 vs 0.859), luma jump (1.5 vs 2.2) and colour-histogram jump (0.362 vs
0.569), and by eye reads as one bridge and one woman instead of four places.
But even the coverage cut has shot-length CV 0.20 against the human median 0.75:
the renderer's 16-frame budget still clusters the holds around 2 s. The grammar
fixes *where* the cuts fall; the next change is per-shot frame budgeting to fix
*how far apart* they are. Full report: `SEQUENCE_AB.md`.

**The rhythm is still wrong even with the grammar.** Our own measurement
(`PACING_MEASURED_STUDY.md`) says human short-form mixes 0.3–1.5 s accents with
5–10 s holds (55% of shots outside 1–5 s), while every hold this planner can
produce is 1.7–2.5 s at 16 frames. The grammar fixes *where* the cuts fall; it
does not fix *how far apart* they are. The next change is per-shot frame
budgeting: 12–16 frames for accents, 24–32 for holds, with the rhythm curve
allowed to reach both bands. R9's CV check should also flag a metronomic film
(CV < ~0.3), not just three equal holds in a row.

### R10 — A visual supervisor / editor — **DESIGNED, not built**

The planner is deliberately narrow: it enforces technical continuity (R1/R2/R6/
R10) and hold timing; it cannot see whether the cut reads as one scene. A separate
evaluation layer, not a smarter planner, is the right shape — see §8. The
deterministic half of it is already here: `story.structure_problems` (script),
`pacing_audit.organisation` (arrangement), `camera_ab_visual` (frame motion).
The VLM half (qwen2.5vl:3b/llava are installed) would look at keyframes and
answer "same place? same action continues?" as an advisory report.

---

## 7. What to measure next

1. **Same shot list, 16 vs 24 vs 32 frames**, one film, same seeds. Measure
   motion continuity (the existing `motion_metrics.py`) and watch it. This is
   the only way to know whether the 2.4× render cost of 32 frames is worth it.
   Then the **per-shot** version our measurement demands: 8–12 frames for
   sub-second accents, 24–32 for holds, and check the delivered film's
   shot-length CV against the human median **0.75**. `pacing_study.py --corpus
   renders` measures the result with the same detector as the human corpus.
2. **Same film retimed three ways** — writer seconds, planner curve, clamped
   rhythm — and watch for which reads smoothest. Retiming is free (re-finish);
   the shot list and takes stay.
3. **A/B the story gate**: 3 blind re-rolls vs failure-specific re-prompts, on
   the same concept. Count how many films reach `EdTR` and how many re-rolls it
   took. (`camera_ab.py`-style decision counting is free; the generation runs are
   the cost.)
4. **Acceptance for R1**: a test that no planned hold exceeds
   `max_hold_seconds(frames)` for any beat shape, and that the stretch report in
   `pacing_audit.py` shows 0 shots over 1.3× at a natural target length. —
   *implemented in `tests/test_edit_planner_pacing.py`.*
5. **Camera direction on pixels** — **done, one pair + three-way**
   (`CAMERA_VISUAL_AB.md`): LoRA tilt travelled −68.9 px net vs +2.8 px for
   `none`, 15/31 downward frame transitions vs 6 down / 7 up — but it never
   settles (2.79 px/f in the last 40%). The procedural path (`camera_path.py`)
   travels 3.55 px/f then holds at 0.43, at the cost of 1.30× zoom headroom.
   Remaining: a second pair of moves and a smaller-travel test before choosing
   the policy; the sequence A/B below is the bigger question.
6. **Subject-motion LoRAs as a direction candidate** (`include_subject=True`):
   measured earlier to work but to be semantically sharp-edged; test whether
   `director.choose` picks sensible ones before enabling them by default.
7. **The sequence A/B (rendering now)**: two 4-shot sequences of one action —
   old montage vs coverage grammar — shot through the real store path. Watch
   them back to back; the decision-level replay already says the montage would
   be refused (R7/R8, one R10 repair). This is the test that decides whether
   R7–R10 become the shipping grammar or need loosening.
8. **Visual supervisor calibration**: hand-label 20–30 cut pairs from existing
   takes (same scene / new scene / action continues) and score the deterministic
   CLIP/phase checks and a local VLM against those labels; only then let the
   film gate advise on real productions.
6. **Subject-motion LoRAs as a direction candidate** (`include_subject=True`):
   measured earlier to work but to be semantically sharp-edged; test whether
   `director.choose` picks sensible ones before enabling them by default.

---

## 8. Open design — a visual supervisor / editor

The question "is it a film or a pile of clips?" cannot be answered by the planner,
and should not be: the planner is a narrow technical layer that rewrites holds and
camera moves. It needs a **separate evaluation layer** that reads the script, the
frames and the finished cut, and reports. Two gates, one report:

```
write ──▶ script gate (already built: story + structure) ──▶ shoot ──▶ film gate
                                                                        │
             deterministic checks ◀──────────────────────────────────────┤
             (CLIP frame similarity, phase-correlation motion direction,
              exposure/sharpness continuity, cut adjacency)
             VLM advisory ◀──────────────────────────────────────────────┘
             (qwen2.5vl:3b / llava on keyframes: "same place? action continues?")
```

* **Script gate** — `story.validate` today: beats, structure R7/R8, rhythm R9.
  Blocking, cheap, and measured against the archive.
* **Film gate** — a new `supervisor.py` that runs after a render:
  * deterministic, first: for every cut, compare the last frame of A with the
    first frame of B (CLIP cosine), estimate motion direction (phase
    correlation), compare exposure/sharpness; a scene cut should look like a
    scene cut, a match cut should not.
  * VLM, advisory only: feed 2–4 keyframes and ask the three questions a human
    editor asks — *same place? same person? does B continue A's action?* Local
    VLMs are small, so this must be scored against a hand-labelled pair before it
    can gate anything.
* **Where it lives** — a report on the production view, not a silent auto-fix:
  "cut 3→4 reads as a new scene but is linked `match-action`"; "the figure's coat
  changes between shots 2 and 5"; "shot 4 is the only one holding 2.8 s, it
  stalls". Suggested repairs (retime, reorder, shoot an insert) are the next
  layer; nothing auto-applies without a human.

Until the VLM half is calibrated, only the deterministic script gate should
block. The rest is measurement — which is the same conclusion this project kept
reaching: measure, then let a person decide.

---

## Appendix A — line references used above

| what | file:line |
|---|---|
| delivered FPS / raw frames default | `make_film.py:46,49` |
| RAFT x2 → 2n-1 frames | `make_film.py:410-441` |
| hold/resample in assembly | `make_film.py:444-464` |
| hold + motion budget definitions | `handheld.py:603-660` |
| budget report / CLI shot count | `make_film.py:76-96` |
| store timing (no planner) | `film_store.py:938-955, 999-1040` |
| **adoption runs the planner, records `project["pacing"]`** | `film_store.py:1057-1140` |
| **camera direction policy (4 modes, hybrid default)** | `camera_direction.py` |
| **camera A/B (decision-level)** | `camera_ab.py` |
| **camera A/B (pixels, phase correlation)** | `camera_ab_visual.py` |
| **procedural move with settle** | `camera_path.py` |
| **sequence grammar R7-R10** | `story.structure_problems`, `edit_planner.plan` |
| **grammar replay over the archive** | `sequence_audit.py` |
| **organisation metrics** | `pacing_audit.organisation` |
| **adoption stores the camera decision** | `film_store.py:1057` (`cameras=`), `draft.py` |
| **story gate / override** | `draft.py` `story_gate`, `--allow-broken-story` |
| **repair re-prompt** | `story.py` `_repair_note`, `_showrunner_user` |
| writer brief on seconds | `story.py` `SYSTEM` (§"Other rules") |
| validate seconds range 1.5–3.6 | `story.py` `validate` |
| retry keeps the best attempt | `story.py` `write_shots_ensemble` |
| token budget scales with shot count | `story.py` `predict_budget` |
| beat factors + renderer clamp | `edit_planner.py:120-215` |
| planner/shortfall notes | `edit_planner.py:194-330` |
| UI shot count / length / frames | `ui/studio.py:584, 803, 1098` |
| UI camera mode setting | `ui/studio.py` (`view_settings`, "how the cut is directed") |
| UI shows problems / pacing / failed-log tail | `ui/studio.py:650, 663, 508` |
| silent renders | `make_film.py` deliverable / `film_store` `deliverable` |

## Appendix B — reproducing the audit

```powershell
python pacing_audit.py                 # every production
python pacing_audit.py --store longing # one production
python pacing_audit.py --budget        # length -> shots -> hours
python pacing_audit.py --curve 6       # what edit_planner would assign
python camera_ab.py --store seasoning  # camera-policy A/B on one production
python camera_ab.py --proposal output/showrunner_test/<stamp>/replies/granite4.1-8b__run2.txt
```
