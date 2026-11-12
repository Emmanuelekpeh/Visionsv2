# Visual organisation: hold a place and figure, or change it — one object, every stage

*The design that came out of the A/B: a montage and a coverage scene are both
good. What was missing was not another rule — it was a single object that says
where the film holds and where it changes, so every stage reads the same thing.*

## The problem: five different ideas of "the same place"

| stage | what it carries | what it actually decides |
|---|---|---|
| `story` | `scene` string, `subject` string | prose labels, nothing checks they are visually true |
| `model_select` | `ShotNeed.scene/subject` | a model switch is only *motivated* at a scene/act/subject/place change |
| `film_store` / `make_film` | one palette per scene id | the look change lands on a scene boundary |
| `director` / `edit_planner` | subject for axis, scene for R10 | camera continuity / move handoff |
| pacing | holds and beat curve | nothing — it never sees the place |
| render prompt | the writer's raw prose | no two shots in a "scene" are told the same place or figure |

Measured consequence on the archive: `seasoning` has 3 scene labels but
within-scene prompt overlap 0.04 vs 0.10 across scenes, and 0/10 adjacent
within-scene pairs share a subject. **The labels were cosmetic.** The rendered
result is a film that changes place, light and person on every cut while the
data says it holds.

## The object: a run

A **run** is a contiguous group of shots that share **one place** and (for a
figure-carrying piece) **one figure**. The film's visual organisation is the
ordered list of its runs; the boundaries are the cuts that mean something.

```
place   [======== bridge, hold 3 ========][ alley ][ bridge, hold 2 ]
figure   woman          woman          woman    figure       -
cut                     | place                          | figure
shot     1     2     3   |   4     5            6
```

- An **insert** with no figure (a hand, a doorway, the rain) does **not** break a
  hold — it is coverage of the same place.
- A **new figure** in the same place is a change point, not drift.
- A **place change** must be a declared `scene-cut`; otherwise it is a jump cut.
- A **run of one** is legitimate: it is the rapid-change/montage mode. It only
  becomes a warning when the piece claims to be a story and every cut resets.
- **Anchors**: each place/figure has a canonical description. Every shot in the
  run gets it appended at freeze time (`effective_prompt`) — the only consistency
  lever this renderer has without image conditioning. The writer writes the
  action; the place/figure wording stays identical.

## What now lines up

| stage | reads the spine |
|---|---|
| story | brief: a scene is a hold, reuse the same nouns/materials; a rapid-change run is one-shot scenes linked `scene-cut` |
| **organise (new stage)** | derives runs, stores anchors, diagnoses alignment, can be approved |
| plan | freezes prompts **with the run's anchors**; R10 move handoff uses the same place boundary |
| look/palette | one palette per **place** (the scene id now has a real meaning) |
| model selection | a switch is motivated by the same place/figure change the spine records |
| pacing | a hold can breathe (long opening), a rapid-change run cuts fast — instead of every shot landing at 2 s |
| audit | hold adherence (within-run overlap vs across), figure drift, undeclared place changes, flat holds |

## The intuitive surface

`strip_png` renders the organisation as one picture (the studio's write view
shows it):

![coverage hold](output/visual_org/_seq_coverage.png)

- **Coverage cut** (left): one place held for 3 shots, real within-run overlap
  **0.387 vs 0.207** across runs, then a figure change. The hold is measurable
  and it is the cut that read as one action.
- **Montage** (`_seq_current`): four single-shot places, 22.5 changes/min. The
  diagnosis is **`intent`** (rapid change is a legitimate tempo), not a failure —
  but it also reports the undeclared cuts and the prompt overlap of **0.0**.
- **`seasoning`** (archive): five runs, within-run overlap **0.043 vs 0.097**
  across → **`cosmetic_hold` break**, plus figure drift, flat 2 s holds, and
  single-shot places in a story. That is the failure the A/B was chasing,
  caught before a render instead of after.

Diagnostics levels: `break` (the intent and the material disagree), `warn`
(probably a mistake, might be a choice), `intent` (a rapid-change rhythm —
reported so it is a decision, not a surprise).

## Where it lives in the process

`film_project.py` gains a stage between story and plan:

```
story -> ORGANISE -> plan -> camera -> grade -> render -> timeline
```

- `organise_project(proj, mode=..., places=..., figures=...)` — derive runs,
  diagnose, store; no shots are rewritten.
- `apply_organisation(proj, "story" | "hybrid" | "montage" | "continuous")` — the
  one-tap rearrangement: merge singles whose prompts actually describe the same
  place, or split everything into a montage; rewrites scene ids/links.
- `approve_organisation(proj)` — refuses to freeze a plan with `break`s unless
  forced (a deliberate montage).
- `plan_project` applies the anchors to the frozen prompts, so what the renderer
  receives is the writer's action + the same place/figure description on every
  shot of a run.

CLI, also on old productions:

```powershell
python visual_org.py --store seasoning
python visual_org.py --store _seq_coverage --png output/visual_org/_seq_coverage.png
python visual_org.py --store _seq_current --suggest hybrid
```

The active studio (`ui/studio.py`) shows the strip and the diagnostics in the
write view.

## Honest limits

- **Anchors are text.** They tell the renderer the same place/figure, but
  AnimateDiff still generates a new face and a new bridge from the same words.
  True figure/location locking needs image conditioning (IP-Adapter / reference
  frames / a locked background), which is the next render-side project, not
  something this object can fake.
- **The overlap proxy is lexical, and needs a boilerplate filter.** "rain,
  cinematic, filmic grain" appears in almost every shot of a film; counting it
  as evidence would make a bridge railing look like a distant figure (measured:
  the first merge rule did exactly that, merging shot 2 into shot 3). The
  suggestion path therefore matches on **distinctive** words (those not shared
  by most of the film), and still only offers merges above a threshold. Two
  prompts can describe the same place with different words; the `break` is a flag
  for the eye, and the A/B is the ground truth.
- **The mode is a tempo, not a quality ranking.** Montage and held scene are
  both correct; the diagnosis only says whether the material, the labels, the
  pacing and the palette agree about which one is happening.

## Reproduce

```powershell
python -m pytest tests/test_visual_org.py -q
python visual_org.py --store _seq_coverage --png output/visual_org/_seq_coverage.png
python visual_org.py --store _seq_current  --png output/visual_org/_seq_current.png
python visual_org.py --store seasoning      --png output/visual_org/seasoning.png
```

Files: `visual_org.py` · `film_project.py` (ORGANISE stage) ·
`ui/studio.py` (strip) · `tests/test_visual_org.py` · strip PNGs in
`output/visual_org/`.

The render-side half — reference-frame conditioning for holds, previs before the
render, and taste capture — is in [STUDIO_LOOP.md](STUDIO_LOOP.md).
