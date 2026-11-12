# The studio loop: holds that hold, previs before render, taste as the oracle

*Follow-up to `VISUAL_ORGANISATION.md` and `SEQUENCE_AB.md`. The organisation
object says **where** to hold or change. These three layers are about making the
decision real on this hardware, before and after the expensive render.*

## 0. Automatic direction: several movements, no mode picker

The studio should not ask a person to choose story/hybrid/montage/continuous.
`sequence_direction.py` reads the shot list and partitions it into **movements**,
choosing a grammar for each from evidence the shots already carry (compound-aware
place/figure affinity with style and atmosphere stripped, beats, energy):

* `hold` — 2-4 shots sharing a place/figure; camera at most one move, and only
  with energy for it; low-energy holds settle.
* `montage` — a rapid-change run; camera none, the cuts carry it.
* `hybrid` — the pairs that continue each other are kept, the rest are cuts.
* `continuous` — the whole piece is one place/figure.

One film can be hold → montage → hybrid → hold; that is the normal output. The
draft path does this automatically (`draft.py` stores `project["direction"]`), and
the studio shows the movements with their reasons and a single "re-direct
automatically" action — no mode selection.

**Camera moves finish the shot.** A generated camera LoRA cannot settle, so the
default path generates every shot STATIC and the movement's one motivated move is
a finishing spec (`camera_move`: move, amount, settle) applied to the delivered
frames by `camera_path.apply_path` — travel, then hold. It is in `finish_inputs`,
so changing it re-finishes in seconds instead of re-generating. Montage movements
get no camera moves. The old CLIP per-shot camera decision is only used when
`--cameras` is explicitly overridden.

Verified in a real render (`_seq_coverage`, tilt-down 45 px, settle 0.62 on shot 1,
no generation):

| shot-1 segment | mean \|dy\| | net dy |
|---|---:|---:|
| baseline, no move (first 62%) | 0.29 px/f | +3.0 px |
| **finishing move, travel** | **1.22 px/f** | **−28.6 px** |
| **finishing move, settle** | **0.47 px/f** | **−0.3 px** |

The move travels and then holds; the settle is indistinguishable from a still
shot's own drift. That is the "tilt and then hold" grammar the A/B asked for,
now automatic and driven by the movement plan.

**Previs is a gate.** `previs.gate()` fingerprints the cut, and rendering is
blocked until a current previs report measures clean. Missing/stale previs is
started automatically; a same-scene boundary with histogram > 0.80 or CLIP < 0.45
blocks with reasons, with a one-shot override. The studio's render buttons call
this gate, so previs is not something a person has to remember.

**The writer plans movements too.** `story.SHOT_SCHEMA` carries an optional
`movement` id, the brief asks for 2–4 movements before shots, and the director
uses the writer's ids when they are contiguous rather than re-segmenting. The
fallback is still content segmentation.

Length scaling lives here too: `shots_for_duration(target)` converts a target
length into shot count at the renderer's measured 1.94 s/shot, so longer films
buy more shots, not longer holds. The studio's write view now asks for **target
length** and derives the shot count (and the render price).

## 1. Hold conditioning: a reference frame per run (IP-Adapter)

**The failure.** Prompt anchors keep the wording consistent, but AnimateDiff still
invents a new face and a new bridge from the same words. The rendered A/B shows it:
the coverage cut holds the place, and the woman's face changes between shots.

**What was missing was already on this machine.** The HF cache has
`h94/IP-Adapter` (SD1.5 weights + its CLIP-ViT-H image encoder) and
`guoyww/animatediff-motion-adapter-v1-5-3`. `AnimateDiffPipeline` supports
IP-Adapter. No network, no new download.

**Controlled test** (one reference frame, same prompt, same seed, 6 frames each,
CPU):

| | CLIP similarity to the reference frame | wardrobe in the output |
|---|---:|---|
| baseline, no conditioning | 0.799 | grey coat |
| IP-Adapter, scale 0.7 | **0.877** | **cream coat, matching the reference** |

![baseline vs conditioned](output/ip_test2/comparison.png)

The conditioning changes the *appearance* (wardrobe, hair, palette, place), not the
face identity exactly; it is an appearance adapter, not a face lock. That is still
the difference between a hold that reads as the same woman and one that does not.
Cost: 257.7 s vs 267.6 s for 6 frames, **+3.8% generation time**, plus the adapter
load (~20 s, once per run boundary).

**Applied to the real failure.** Rerendering the coverage cut's close-up (the shot
that introduced a different woman in a cream coat) conditioned on its run anchor
(shot 1, dark hooded figure on the foggy bridge):

| | CLIP similarity to shot 1 (anchor) | to shot 2 (previous) | what it looks like |
|---|---:|---:|---|
| old shot 3, prompt only | 0.832 | 0.764 | a different woman, cream coat, warm alley light |
| **new shot 3, conditioned** | **0.945** | **0.933** | the same dark hooded figure, same foggy bridge |

![conditioned shot 3](output/ip_test2/shot3_conditioned.png)

Conditioning also pulls the framing toward the reference: the "close-up" came back
as a wider shot of the figure. That is a real tradeoff (a strong reference can
overrule the shot's intended scale); the next tuning is to pair the reference with
scale words in the prompt or blend the reference at a lower scale for inserts.
Identity/place consistency improved enough that the cut reads as one hold.

**How it is wired in**

* `visual_org.reference_plan(shots)` returns, per shot, the anchor shot of its
  run: `{1: 0, 2: 0}` for a 3-shot hold, and **nothing for a rapid change**. A
  one-shot place is not conditioned.
* `make_film.Generator.clip(..., reference=frame)` loads the adapter lazily,
  conditions the clip, and unloads it for an unconditioned clip. The unload is not
  cosmetic: Diffusers 0.32 crashes if the adapter is loaded and a later call passes
  `ip_adapter_image=None`; unloading per run boundary costs ~20 s, not per shot.
* `run_store` reads the reference from the run anchor's take (or from the take
  generated earlier in the same run), and records on the take:
  `conditioned: true`, `hold_reference: "shot 1"`.
* The reference is part of the **generation hash**: `hold_reference: "s01:t01"`
  enters `film_store.generation_inputs`, so a conditioned take and a prompt-only
  take are different inputs, and re-shooting the anchor correctly stales the
  shots that depend on it. Old takes are tolerated as legacy (`ADDED_GEN_KEYS`).
* `make_film --force --only N` exists because rolling conditioning onto takes
  rendered before it existed is not a stored input change. Editing a prompt to
  force a re-render would be a lie about what changed.

Disable with `HOLD_CONDITIONING=0`. Tune with `IP_ADAPTER_SCALE` (0.7 default).

**Limits:** IP-Adapter is appearance conditioning, not identity. A face can still
drift; a new composition can still be generated. The reference frame should be the
run's clearest frame of the figure (the first take's middle). A true face/place
lock needs image-to-video conditioning or a reference-network, which is the next
render-side project.

## 2. Previs: one still per shot, measured at the cuts

A full render is 45-60 minutes for 4 shots; a still is ~20-40 s. The properties
that made the coverage cut win are visible at the boundary frames, so the same
metrics the finished-cut tool uses can be computed from one still per shot.

`previs.py --store <slug>`:

1. renders one SD15 keyframe per shot (4-step Lightning, fixed seed per shot),
   **conditioning held shots on their run's anchor still** — i.e. mirroring the
   render policy, because a naive stills pass does not predict the cut (see the
   measured negative result below);
2. measures cross-cut CLIP similarity, colour-histogram jump and luma jump on the
   stills — the same `sequence_ab` metrics, before motion exists;
3. reports structure and pacing (scenes, cuts, ASL, shot-length CV);
4. writes `report.json` + a contact sheet, and logs the wall time against the
   full-render estimate.

It cannot judge performance or motion; it exists to kill arrangements whose images
do not even hold together, and to rank two candidate arrangements before the
expensive one.

**Measured negative result first.** Naive previs (one unconditioned still per
shot) did **not** reproduce the rendered cut's continuity ranking:

| | montage stills | coverage stills | montage render | coverage render |
|---|---:|---:|---:|---:|
| CLIP cross-cut | 0.891 | 0.889 | 0.859 | **0.878** |
| histogram jump | 0.694 | 0.697 | 0.569 | **0.362** |
| luma jump | 9.6 | **5.0** | 2.2 | **1.5** |

The stills renderer's own style variance is larger than the place difference, so
the naive pass cannot tell a held place from a new one. Only luma favoured the
coverage cut. That is why previs now conditions the held stills exactly as the
renderer conditions the held clips: the previs must simulate the policy, not just
the prompts. The conditioned run is the one below.

**Conditioned previs** (after the fix; shots 2–3 conditioned on shot 1, shot 4 is a
rapid change and is not):

| | montage stills | coverage stills | montage render | coverage render |
|---|---:|---:|---:|---:|
| CLIP cross-cut | 0.891 | **0.923** | 0.859 | **0.878** |
| histogram jump | 0.694 | **0.667** | 0.569 | **0.362** |
| luma jump | **9.6** | 6.8 | 2.2 | **1.5** |

Conditioning recovered **two of the three** signals: CLIP and colour now rank the
coverage cut higher, matching the finished render. Luma still ranks it worse on
stills (6.8 vs 9.6) even though the render ranks it better (1.5 vs 2.2) — a luma
jump on one boundary frame is too noisy a predictor. So previs with the render
policy is a useful *preflight and ranker*, not a guaranteed oracle: use it to kill
an arrangement or to order two candidates, then watch the cut.

![coverage previs stills](output/previs/coverage/contact.png)

Cost: **~185-205 s for 4 stills** vs the ~48-minute full render estimate logged in
the report — **14-16× cheaper**. The previs stage is wired into `film_project`
(`story → organise → previs → plan → ...`), optional per project.

## 3. Taste: the A/B decision becomes a label

The corpus study says montage and held coverage are both legitimate. The thing
that decides between them is editorial taste, which no metric can supply. Every
A/B in this project ended with "watch and decide", and the decision was discarded.

`taste.py` stores one row per comparison:

```json
{"at": "...", "a": "...", "b": "...", "pick": "b", "note": "...",
 "tags": ["montage", "coverage"],
 "metrics": {"a": {"cuts_per_min": 22.5, ...}, "b": {...}}}
```

* `python taste.py add --a <cutA> --b <cutB> --pick a|b|tie --tags ...`
* `python taste.py summary` — win counts, wins by tag, and *the measured
  difference between the pick and the reject* (cut rate, ASL, CV, duration).
* **Studio**: the CUT view has a two-video picker; one click records the same row.
* The first real label (coverage vs montage) is already stored: pick **coverage**,
  note "coverage read as one action; montage changed place and person every cut".

It does not train anything yet. It starts the dataset, and it makes the studio's
preferences queryable instead of folklore. The first useful thing it can answer:
*when this person says "better", is it a pacing difference, a continuity
difference, or neither?* The seeded row already says coverage was +0.19 shot-length
CV and +0.3 cuts/min — i.e. not simply "more cuts".

## How the three compose

```
story -> ORGANISE (hold/change spine)
      -> PREVIS (stills of the candidates; kill bad arrangements)
      -> PLAN (camera/holds; freeze anchors)
      -> materialize_store -> RENDER (the ONE store path: IP conditioning,
         takes, stats, staleness hash)
      -> TASTE (record which cut was better; summarise what separated them)
```

**One render path.** The workbench (`film_project`) no longer renders through its
own loop. `make_film --from-project` calls `film_project.materialize_store`, which
writes the frozen plan into `productions/fp_<name>/`, and then calls `run_store` —
the same renderer the studio uses. So hold conditioning, anchors, the take ladder
and the staleness hash apply to the process that decided the holds, not only to
the studio. Re-materialising preserves takes from a previous run, so a re-render
still only re-shoots what changed.

Holds are decided once and carried through: anchors in the prompt (text), a
reference frame in the generator (pixels), a reference in the take hash
(staleness), and a preference label on the finished cut (taste).

## Reproduce

```powershell
# previs two candidate arrangements
python previs.py --store _seq_current  --name montage
python previs.py --store _seq_coverage --name coverage

# taste
python taste.py summary
```

Files: `make_film.py` (Generator + run_store), `visual_org.reference_plan`,
`previs.py`, `taste.py`, `ui/studio.py` (taste picker), tests
`test_hold_conditioning.py`, `test_previs.py`, `test_taste.py`.
