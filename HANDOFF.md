# AGENT HANDOFF

**Read §0 first — it is the current state and it supersedes §14.** The rest of the
document is reverse-chronological: §1–§13 are the generation stack, the measured model
comparisons and the traps that produced **four false negative verdicts** in an earlier
session. Those still hold; only the UI sections are historical.

## 0. THIS SESSION — read this first, it supersedes §14

### 0.1 CURRENT STATE — the studio, and the store underneath it

**The UI is `ui/studio.py`. Run it: `streamlit run ui/studio.py`.**

`ui/` contains that one file. The two previous UIs are in `ui_legacy/` as
`app_superseded.py` and `workbench_superseded.py`, and **nothing imports them**. §14
describes the seven-stage workbench — that design was superseded and the reason is in
§0.3. Do not rebuild it.

**Four documents, and each one owns something different.** Read the one you need:

| file | what it owns |
|---|---|
| `START_HERE.md` | how to run it, how to migrate, the speed measurements, the known gaps |
| `AUDIT.md` | what was wrong with the OLD data flow, **measured** — the evidence for the rewrite |
| `REDESIGN.md` | what CapCut/Premiere/Resolve/FCP/Kling/Runway/Flow do, what was taken and dropped |
| `video-tool-ux-research.md` | the consumer/AI tool research, with sources |

**The data model is `film_store.py`, not `film_project.py`.** `productions/<slug>/`:
`project.json` (timeline order + settings), `shots/<id>.json` (one file per shot),
`takes/<shot>/<take>/` (immutable: `take.json` + `frames/`), `renders/`. Three
properties hold it up:

- **A shot has a stable string id and its own file.** Order lives only in the timeline.
  Reordering moves no camera, palette, take or approval — the bug that was reproducible
  four ways in the old model.
- **A take is never overwritten.** Re-rendering writes the next one.
- **Staleness is DERIVED, never stored**, by hashing the real dependencies:
  `generation_inputs` (prompt, camera, scale, subject, seed, model, **frames**) and
  `rebuild_hash` (hold, shake, palette, scene palettes, **in-point**).

**New entry points**: `draft.py` (write a shot list and adopt it into the production in
one job), `make_film.py --store <slug>` (the new render path; `--from-project` is the
old one and still works), and `job_runner.py` tasks `draft` and `produce`.

**Verified**: `recon_test/test_film_store.py`, `test_store_render.py`,
`test_studio.py` all pass; `audit_ui.py` still measures the old flow. All four are
self-contained and clean up after themselves.

### 0.2 THE TRAPS THIS SESSION ADDED

Each of these cost real time or real renders. They are in order of how expensive they
were to find.

**1. `next_take_name` was handed a shot RECORD where it wanted a shot ID.** The take
path became `.../takes/{'id': 's01'}` and raised
`TypeError: unsupported operand type(s) for /: 'WindowsPath' and 'dict'` — **after the
diffusion pipeline had loaded**, so 3-5 minutes of work was thrown away, on the exact
branch a first render takes. Every test passed strings, so none touched it. **The fix
is `film_store._sid`**: every store function that takes "a shot" accepts a record or an
id. The lesson is not "be careful" — it is that a function whose argument can be either
thing must accept either thing, because one caller will get it wrong and the failure
will land in the most expensive place available. `test_film_store.py` §8 and
`test_store_render.py` §5 both exercise the record form; §5 stubs `Generator` so the
expensive branch is reachable in a second and therefore actually tested.

**2. The lock masked drift.** `shot_state` returned `approved` **before** checking
whether the finishing inputs had moved, so locking a shot and then changing its scene's
palette produced no warning anywhere. A lock is a claim that you *reviewed* something,
never that it is still current. Drift is now reported *through* approval, and
`unlock_take` exists. `test_film_store.py` §2b.

**3. The re-finish state was append-only — it never self-healed.** Edit a shot's
length, edit it back, and it still claimed a re-finish was needed, forever. The cause:
each shot computed its own in-point by walking the timeline, which made it *read its
own stored `seconds`* as a dependency of its own hash — so changing `seconds` changed
the dependency, not just the value. The in-point is now **passed in** from one shared
`in_points(project)` walk. Change a value and change it back and the state restores
exactly.

**4. `--only` died in the finishing loop.** A partial run generated its shots, then
fell through to a loop that walks the WHOLE timeline and raises on the first shot with
no take yet: *"shot s02 has no take after this run and is needed for assembly"*. That
message is right for its own purpose and wrong as the outcome of asking to shoot two
shots — the user pays ten minutes of generation and loses the run. A partial run now
stops and says what is still missing. `test_store_render.py` §6 uses a film whose
unrendered set is deliberately non-contiguous, because `range(first, first+n)` was the
first, wrong implementation of "the next N".

**5. `_sha1_path`, not `_sha1_file`.** The camera LoRAs are **directories**
(`models/motion_lora/zoom-in/`); the subject ones are single `.safetensors` files.
Hashing a directory the naive way returns `''`, and then every directory-backed adapter
shares one cache key — so one verification would silently stand in for thirteen
others. The helper now walks directories and returns `''` only for unreadable paths,
which callers must treat as *uncacheable*, never as a key.

**6. A Streamlit server started as a child process dies with its parent shell.** Two
symptoms that look unrelated: `run_in_background` produced a job that reported
`Port 8514 is already in use` and exited 1 while a *previous* instance was still
serving, and an HTTP 200 health check passed against a server that was already dead
because it was answering from the dying one. **Check `netstat -ano | Select-String
':8514'` and the pid before believing a URL is live**, and start it detached
(`Start-Process -PassThru`) — this is the same constraint the job runner documents for
jobs.

**7. `${env:PYTHONIOENCODING}='utf-8'` is required for the probe scripts.** A `▸` in a
button label crashed a test harness with `UnicodeEncodeError: 'charmap' codec` on
cp1252. Any glyph in a label is a hazard for logs, consoles and screenshots; prefer
plain words.

**8. A test can measure the earlier tests instead of the code.** `test_film_store.py`
§5 failed three times in a row and **every failure was the expectation, not the code** —
it kept asserting on a timeline that test 1 had reordered. It now builds its own
three-shot film with known states. Worth knowing because a test wrong in that direction
gives false confidence rather than false alarm. Related: `test_studio.py` §6 first
passed *wrongly* because it grepped for the buggy CSS value and matched the explanatory
comment — it now parses the **active CSS rule**.

### 0.3 WHY THE SEVEN STAGES BECAME THREE VERBS

The user's own words: *"it doesn't feel intuitive just packed"* and *"why do we need 7
stages"*. They were right, and they were right twice — the second revision fixed the
data model and then made the same mistake in a different shape (three always-visible
columns, every control on screen at once). **The studio is organised by what a person
is DOING, not by what the pipeline contains**:

| view | what it is | one primary action |
|---|---|---|
| **write** | say what the film is. One field. Then read the shot list it produced. | write the shot list |
| **shoot** | the whole film as a strip, then one shot at a time — 95% of the time goes here | shoot it |
| **cut** | look at it, export | export the film |
| *settings* | frames, palettes, shake, the scouts' raw material, provenance, history | — |

The design rules are in `ui/studio.py`'s header and asserted in `test_studio.py`: one
primary action per view, nothing nested more than two deep, no bordered box that is not
holding a thing, **state as a word not a colour**, the price before the button, nothing
destructive on the canvas. **Read that header before adding a control.**

**The 18 scout ideas were never meant to be on screen.** They are material for the
showrunner — more of them helps the writer and hurts the reader. They now live in
Settings. Measured in `recon_test/idea_diversity.py`: they are *lexically* diverse
(max word overlap 0.21) and still felt repetitive, because the repetition is at the
level of the motif. A "each idea in a different place" rule was tested against three
models and **did not help** (`farm_ab.py`: 100% distinct places both ways, and it
lowered the number of ideas naming a place at all, 17 → 12). **`story.py` was left
alone deliberately.**

### 0.4 SPEED — WHAT WAS MEASURED, AND WHAT TO NOT RETRY

Wall clock was the user's main complaint and the answer was not where it looked. One
shot is ~10 min and **roughly half is fixed cost per job** (`bench_shot.py`):

| | measured |
|---|---|
| pipeline load (7 GB, 14 motion LoRAs) | **57 s** |
| the preflight: 28 UNet passes, 2 per LoRA × 14 | **179 s** |
| fixed total per job | **236 s** |
| across 6 separately-started shots | **24 min of pure overhead** |

The preflight — a **self-check**, not generation — was the single biggest cost in the
system and was re-run from scratch on every job. It is now **cached against the sha1 of
the exact weights it measured** (`output/film/preflight_cache.json`):

```
COLD total load+preflight: 234.6s
WARM total load+preflight:  86.1s      <- 148.5s saved per job
one deliberately-corrupted entry -> re-verified exactly that one, 13 skipped
```

The check is **not removed** — §9.4's failure mode is real — it is only skipped when the
bytes are identical. Delete the cache file to force a full re-check.

**Frame count**: 12 raw frames against 16, one loaded pipeline, same prompt/seed/LoRA
(`bench_frames.py`):

| raw | total | s/frame | align | coherence |
|---|---|---|---|---|
| 16 | 695.4 s | 42.8 | 0.3463 | 0.0240 |
| 12 | **510.0 s** | 41.8 | 0.3484 | **0.0155** |

**27% faster with equal-or-better quality.** Per-frame cost is *identical*, so the
saving is purely fewer frames and scales predictably. `frames` is now a per-production
setting defaulting to 12 for new productions — see trap 9 below for why existing ones
are pinned to 16.

**DO NOT RETRY** (already measured and rejected): 8 raw frames (422s vs 552s, only 24%
faster, and align fell 0.368 → 0.313 with travel 14.8 → 9.0 px); 8-step Lightning
(2.55× slower, less accurate, 2.6× *less* stable than 4-step, no useful midpoint);
256px (tiled green banding — this alone caused three false verdicts).

**9. THE MOST DANGEROUS BUG OF THE SESSION: a change that reaches BACKWARDS into
finished work.** It happened twice, in two layers, and both times the UI offered to
re-render a finished film.

*First layer — the default.* Making 12 frames the default for new productions marked
all six of the user's rendered shots `stale`, because the frame count is part of the
generation hash. The hash was right; the **policy** was wrong. A production that has
never recorded a frame count is now treated as **16** — what its takes were actually
rendered at — and only `new_project` gets 12.

*Second layer — adding a key.* Even with the policy fixed, all six still read `stale`,
because **adding a key to `generation_inputs` changes the hash of every take recorded
before that key existed.** Diagnosed in one command by printing the stored hash beside
the current one and the hash with the key removed:

```
take.generation keys : camera, model, prompt, scale, seed, subject   <- no `frames`
stored input_hash    : dbd1f6a26f16
hash now             : 11590955b95d
hash WITHOUT frames  : dbd1f6a26f16      <- identical to what was stored
```

Nothing about the pixels had changed; only the key set had. A stored hash now counts as
current if it matches the current inputs **or** the current inputs minus the keys that
particular take predates (`ADDED_GEN_KEYS`; `generation_hash`'s `stored_generation` and
`stored_hash` arguments). Deliberately narrow: editing a key the take *did* record still
marks it stale, which `test_film_store.py` §10 proves by editing the prompt of a legacy
record and asserting it goes stale.

**Two rules for anyone adding a render input here:**
1. **Append the key name to `ADDED_GEN_KEYS`.** Not doing it invalidates every existing
   take in every production.
2. **Never write a new default into an existing project.** Pass it explicitly at the
   point of use (`frames=FRAMES`) so there is no window in which a half-applied setting
   can be read — which is exactly how the first layer happened.

**And a trap landed while fixing it:** a *second* definition of `generation_hash` lower
in `film_store.py` silently shadowed the tolerant one, so the legacy comparison was a
no-op and the bug looked unfixed after the fix. Two definitions of one measure is this
project's oldest failure mode — there is now one, with a comment where the duplicate was.

### 0.5 STILL OPEN

1. **Audio.** The export is silent. `add_narration.py` works (ollama writes lines,
   edge-tts speaks them, moviepy muxes) but is not wired to the store, and there is no
   music bed or mix panel. **For a YouTube deliverable this is the largest gap.**
2. **A scrubbable preview.** The viewer shows four frames of a take, not a player.
   Streamlit cannot host a timeline playhead; a real one needs a different client.
3. **The 8-frame and 12-frame results are one prompt and one LoRA each.** Strong
   signals, not laws. Frames are delivered by nearest-index stretching (12 → 23 after
   RAFT → 51 delivered against 16 → 31 → 51), so whether the extra duplication reads as
   judder is something only watching the clip will settle.
4. **The four VAE checkpoints deleted from `recon_test/` this session** —
   `t_plus_1_vae_latest.pt` (×3, re-fetchable from the Kaggle dataset
   `meowkittypaws/latest`) and `t_plus_1_vae_phase2_final.pt`, which
   `compare_phase2_models.py:119` loads as the 30-epoch model. Deleted by cleaning up by
   exclusion without listing what would be removed first. **Do not clean a directory
   that way again.**
5. **No human has looked at the studio's pixels.** Browser automation is unreliable
   here (Chromium's Mojo IPC and Node's `child_process` both need named pipes, which the
   confined sandbox denies; `agent-browser` navigates and then hangs on every
   subsequent command, writing blank frames). The layout is verified **structurally**
   and not visually.

### 0.6 THIS SESSION — showrunner measured, pacing fixed

**Showrunner: keep `granite4.1:8b`.** `compare_showrunners.py` compares candidates
on ONE concept and ONE shared idea pool (3 runs each, retries off). Granite beat
`qwen2.5:7b` on filmcraft 70.6 vs 55.7 (mistral 50.8): jump cuts 0.67 vs 2.33 per
film, scenes 2.67 vs 1.67 (2/3 qwen runs single-scene), self-repetition 0.05 vs
0.14, provenance 0.94 vs 0.83. Qwen is ~50 s/attempt faster and returns valid
JSON, but its lists are 2-4 adjacent medium shots and a looped first/last image.
Story-break rate ties at 1.0/film, so the next quality lever is the story gate,
not the model. Full numbers: `SHOWRUNNER_TEST.md`.

**The "granite malformed JSON" was truncation.** Reproduced in the single-writer
path: 16 shots at `num_predict=1200` → exactly 1200 completion tokens,
`JSONDecodeError ... (13 open vs 11 close brackets) TRUNCATED`; at 3000 → 15
shots, valid raw JSON. `story.predict_budget(n_shots)` now scales the cap
(2000-6000) and both writer paths use it. The comma-repair layer measured **0
uses**; it cannot fix truncation and should not be extended to try.

**Pacing.** A shot contains `(2*frames-1)/fps` seconds of real motion (1.94 s at
16 frames); a longer hold duplicates frames and plays as judder. The uncommitted
planner curve was assigning 4.5-7 s holds (2.4-3.1x) and
`film_store.shots_from_proposal` kept the writer's seconds untouched, so the
store path could ship judder. Fixed: `handheld.natural_seconds` /
`max_hold_seconds` are the one definition; `edit_planner.plan(..., frames=)`
clamps and reports unreachable targets; the UI shows the stretch and warns;
`pacing_audit.py` audits any production. Full analysis:
`PACING_AND_FLOW_RESEARCH.md`.

**Follow-up (same session): the product gaps from that list.**

- **The store path now runs the planner at adoption**
  (`film_store.shots_from_proposal`). R4 gives the cut a beat-shaped rhythm of
  real generated motion, R1 repairs adjacent framing scales, the budget clamps
  every hold, and `project["pacing"]` records writer holds → planned holds so the
  write screen can show what happened. The writer's TEXT is kept; timing and
  framing are paced.
- **The story gate acts.** `draft.py::story_gate` refuses to adopt a proposal
  whose beats still fail (`story:` breaks); `--allow-broken-story` and the UI's
  "adopt a broken story" checkbox are the explicit override. The retry inside
  `write_shots_ensemble` is no longer blind: `story._repair_note` hands the next
  attempt the validator's exact complaints, the beats it produced and its
  previous prompts.
- Rejected drafts leave the production untouched and exit 2; the UI failure
  banner now shows the log's last lines instead of "open the log".
- **Camera direction reaches the render.** `camera_direction.py` decides one
  camera per shot at adoption with four policies (`off`, `suggestions`,
  `director`, `hybrid` default = director first, writer's suggestion where it
  declines, both inside the director's mover cap). `draft.py --cameras` and
  Settings → "how the cut is directed" select it; `film_store` stores the
  decision so R2/R3/R6 act on it; the UI shows the camera count. `camera_ab.py`
  compares policies on one production or one proposal before spending render
  hours. Measured: on `seasoning` (13 shots) stored moves went 0 → 2, reversals
  0, within budget; on a fresh 6-shot list hybrid/director each give 2.
- The A/B caught two bugs before shipping: uncapped hybrid gave 3 moves in a
  6-shot film, and `none` being a valid camera name burned the mover budget on
  empty suggestions. Both regression-tested.
- **151 tests pass** → **162 after the sequence grammar** (`test_camera_direction.py`,
  `test_story_gate.py`, `test_story_structure.py`, `test_edit_planner_pacing.py`).
  The real-store adoption path was smoke-tested end-to-end with a cached proposal.
- **Sequence grammar (R7–R10).** A shot now carries `purpose` (establish/master/
  insert/reaction/resolve) and `link` (open/match-action/eyeline/movement/graphic/
  scene-cut). The brief asks for 2–3 scenes covering ONE place/action each, ≤2
  hard stops, no interleaving, scene-opening shot longest, and moves that continue
  in the same scene. `story.structure_problems` reports R7 (scene runs/cuts/
  interleaving), R8 (coverage: scales, shared subject, purposes), R9 (rhythm);
  `edit_planner.plan` repairs R10 (a move before a scene cut → none); the adoption
  gate blocks `story:` AND `structure:` breaks; the repair retry names both.
  `pacing_audit.py` measures organisation (scene runs/cuts, within vs across
  overlap, same-subject share, hold sd, move→cut); `sequence_audit.py` replays the
  grammar over the archive: 29 cached proposals show 25 R7, 43 R8, 23 R10
  violations before counting the missing new fields. Measured on the shipped
  productions, within-scene prompt overlap is no higher than across-scene (e.g.
  `seasoning` 0.05 vs 0.08, 0/10 adjacent same-scene pairs share a subject) —
  the scene labels were cosmetic.
- **Sequence A/B rendered: coverage beats the montage.** Two 4-shot cuts of one
  action through the real store path (`_seq_current` montage: 4 scenes, 4 x 2.0 s;
  `_seq_coverage`: 1 scene, 2.5/1.7/1.5/2.2 s). Coverage wins cross-cut CLIP
  (0.878 vs 0.859), luma jump (1.5 vs 2.2) and colour-histogram jump (0.362 vs
  0.569), and by eye reads as one bridge/woman instead of four places. Caveat:
  coverage's shot-length CV is only 0.20 (human median 0.75) -- the 16-frame
  budget still pins holds near 2 s, so per-shot frame budgeting is the next
  change, not more text rules. Report/tool: `SEQUENCE_AB.md`,
  `sequence_ab.py`; artifacts `output/sequence_ab/`, `output/film/seq_*`.
- **Visual organisation: the hold/change spine.** The pipeline had five different
  notions of "the same place" (story scene string, model_select's motivation,
  per-scene palette, planner R10 scene, pacing) that did not agree, so a "scene"
  could change place/figure on every cut while the data said it held. New
  `visual_org.py` derives **runs** (one place + one figure), diagnoses alignment
  (`cosmetic_hold`, `undeclared_place_change`, `figure_drift`, `flat_hold`,
  `rapid_change` intent), and owns canonical place/figure **anchors** that
  `film_project` appends to every prompt in a run at freeze. `film_project` gains
  an **ORGANISE** stage (story -> organise -> plan -> camera -> grade -> render ->
  timeline) with `apply_organisation(story|hybrid|montage|continuous)` and
  approval that refuses `break`s unless forced. `ui/studio.py` shows the strip +
  diagnostics; `python visual_org.py --store <slug>` works on old productions.
  Demonstrated: `_seq_coverage` within-run overlap 0.387 vs 0.207 (real hold),
  `_seq_current` 22.5 changes/min (rapid-change intent), `seasoning` 0.043 vs
  0.097 -> cosmetic-hold break, exactly the archive failure. Doc:
  `VISUAL_ORGANISATION.md`; limits: anchors are text, the renderer still cannot
  lock a face/place without image conditioning.
- **Studio loop: hold conditioning, previs, taste.** (a) IP-Adapter reference
  conditioning is wired into `make_film.Generator`/`run_store`: each shot in a
  multi-shot run is conditioned on its run anchor's frame (`visual_org.
  reference_plan`), the take records `conditioned`/`hold_reference`, and the
  reference is part of the generation hash so a conditioned take cannot be
  confused with a prompt-only one (`make_film --force --only N` rolls it onto
  old takes). Measured: controlled test CLIP-to-reference **0.877 vs 0.799**;
  rerendered the coverage close-up (the face that changed) conditioned on shot 1:
  similarity to anchor **0.945 vs 0.832**, to shot 2 **0.933 vs 0.764**, and the
  cream-coated stranger became the dark hooded figure. +3.8% generation cost,
  ~20 s adapter load per run. Quirk: diffusers 0.32 crashes when the adapter is
  loaded and a call has no image, so it is loaded/unloaded per run boundary.
  (b) `previs.py` renders one still per shot and measures the same cut
  continuity/pacing before the render, stored as film_project's PREVIS stage.
  MEASURED: naive unconditioned stills did NOT predict the cut ranking (clip
  0.891 montage vs 0.889 coverage; hist 0.694 vs 0.697) -- the still renderer's
  style variance swamps place difference. Conditioning the held stills on their
  anchor (mirroring the render policy) recovers 2 of 3 signals: clip 0.923 vs
  0.891, hist 0.667 vs 0.694 (luma still disagrees), at ~185-205 s for 4 stills
  vs ~48 min full render (14-16x cheaper).
  (c) `taste.py` + the studio CUT-view picker record A/B choices in
  `output/taste/labels.jsonl` with measured properties and a summary of what
  separated pick from reject. First label stored: coverage > montage. Full write-up:
  `STUDIO_LOOP.md`; tests `test_hold_conditioning.py`, `test_previs.py`,
  `test_taste.py`.
- **Integration pass: one render path, style-relative gate, shared primitives.**
  (a) `make_film --from-project` no longer has its own generation loop: it calls
  `film_project.materialize_store(proj)` -> `productions/fp_<name>/` -> the same
  `run_store` the studio uses, so the process that decides holds also gets
  conditioning, takes, stats and the staleness hash. A real-store integration test
  (`test_render_seam.py`) asserts shot 2's generation input names shot 1 as its
  hold reference, with no internal monkeypatching. (b) `draft.story_gate` blocks
  only `story:` now; `structure:` is an advisory for the ORGANISE stage, because
  the A/B showed a montage is a legitimate arrangement. (c) Palettes are owned by
  the PLACE (`organisation.place_palettes`) and remapped on an organisation
  rewrite, so montage <-> coverage cannot silently drop a look. (d) One
  `conditioning.py` (IP load/scale/unload, shared by renderer and previs) and one
  `continuity.py` (hist/luma/CLIP, shared by previs, sequence_ab and the cut
  detector) replace duplicated copies; `sequence_planner.py` (dead) deleted.
  **190 tests pass.** Remaining deliberate gaps at that point: the ad-hoc
  non-store CLI path (`--shots figure/test/llm`) still has its own loop; previs is
  not an automatic gate (it recovers 2 of 3 signals); taste had one label.
- **Automatic sequence direction: multiple grammars per film, cameras scoped.**
  The manual mode picker is gone. `sequence_direction.py` partitions a shot list
  into movements (compound-aware place/figure affinity with style and atmosphere
  stripped, beats, energy), chooses hold/montage/hybrid/continuous per movement,
  and can return hold -> montage -> hold inside one film. `draft.py` runs it
  automatically before camera direction and stores `project["direction"]`;
  `scope_cameras` enforces the movement policy on `camera_direction`'s decisions
  (montage runs still, a hold keeps at most one move, and low-energy holds settle)
  without inventing moves. The studio shows the movements and their reasons plus
  one "re-direct automatically" action — no mode selection. Scaling:
  `shots_for_duration(target)` derives shot count from the measured 1.94 s/shot,
  and the studio write view now asks for **target length** and computes the shots
  and the render price ("longer needs more shots, not longer holds").
  **Camera implementation v2 (the real fix).** The default path no longer
  generates camera moves at all: a generated camera LoRA cannot settle (measured
  still travelling 2.79 px/f at the last frame), so generation is static and the
  movement plan records at most ONE motivated move per hold as a FINISHING spec
  `camera_move {move, amount, settle}`. `make_film.run_store` applies it with
  `camera_path.apply_path` between stretch and shake (verified in a real render:
  "camera finish: tilt-down 45px, settle 0.62"), and it is part of
  `finish_inputs`, so changing it re-finishes in seconds rather than re-generating.
  The move is chosen from the shot's scale/purpose and the writer's suggestion
  (wide reveal, close push, resolve pull-back); montage movements get none. The
  old CLIP per-shot camera decision remains available only as an explicit
  `--cameras`/`camera_mode` override. **Movements now come from the writer too**:
  `story.SHOT_SCHEMA` has an optional `movement` id, the brief asks for 2-4
  movements before shots, and `sequence_direction` uses the writer's ids when they
  are contiguous instead of re-segmenting. **Previs is a gate**: `previs.gate()`
  fingerprints the cut, treats missing/stale as its own status, and blocks a
  render on a same-scene boundary with histogram > 0.80 or CLIP < 0.45; the studio
  starts previs automatically when the gate closes and allows a one-shot override.
  201 tests pass.
- **Writer throughput: the 30-minute draft.** A real draft showed the farms done
  in ~2.5 min, then 30+ min silent: granite4.1:8b on CPU generates at ~2.7–3
  tok/s, and the 10-shot request used ~2000 completion tokens against Ollama's
  4096 context, so the runner shifted context mid-generation (`n_discard 2045`)
  and hit the cap (`truncated = 1`). The retry loop then re-asked on *advisory*
  warnings/structure (`sb == 0 and sw == 0 and st == 0`), so an adoptable list was
  regenerated 3-4 times at ~13 min each. Fixes: `num_ctx=6144` on every writer
  call (`OLLAMA_NUM_CTX`), retry only on a genuine `story:` break (advisories are
  the director's job now), a visible `showrunner attempt N/M: generating` line and
  job progress phase, and a note when the writer returns fewer shots than
  requested (the count is a request, not enforced). Same draft after the fix:
  farms ~2.5 min + one showrunner attempt 487s, 0 story breaks, 6/6 prompts from
  farm ideas, adopted in ~11 min.
- **The writing loop: enough shots, and the writing fed back.** A real 13-shot
  draft was REFUSED after three 6-shot attempts, all reporting the same
  `beats go backwards (4:turn->develop)`. The retry was already naming that
  failure and attaching the previous prompts, and it changed nothing. Three
  causes, found in order.
  - **A beat needs a position, not advice.** The brief said "beats never
    backwards"; the model tags a beat from the image it just wrote, so the rule
    was unenforceable. `story.beat_layout(n)` now names the roster (establish,
    develop, turn, fall, resolve with shot numbers). It is passed with every ask
    and every repair note, and `_showrunner_user` puts it ABOVE the farm ideas,
    because the count is the brief and the ideas are raw material. The SYSTEM
    prompt's four "six" anchors ("a film of six near-identical shots", "Six equal
    moments", "a sixth restatement", "six variations") were pulling every reply
    toward 6 and are now count-neutral. **Measured on the same concept, same 18
    farm ideas, 9 shots requested, granite4.1:8b: 9/9 shots on attempt 1, shape
    `EddddddTR`, 0 story breaks, 483 s** -- where the same request previously
    returned 6/9 with a break on all three attempts (871 s, refused). Advisories
    remain (2 structure, 1 story-link; the story-link warning is the documented
    "check it by eye" lexical proxy).
  - **The model was writing a beat the taxonomy did not have.** The real 13-shot
    case then returned **13/13 shots on all three attempts** -- the count was
    genuinely fixed -- but one `story:` break survived every time:
    `beats go backwards (9:turn->develop)` / `(10:turn->develop)`, shapes
    `EddddddTTddRR` and `EddddddTTTdRR`. That is not a mood board: establish,
    develop, turn, then the change SETTLING into the new world, then resolve -- a
    four-act short film, with the settling labelled 'develop' because the beat set
    was E/d/T/R. No number of re-rolls fixes a missing word. **`fall` is now a
    first-class beat** between 'turn' and 'resolve' (`BEATS`, marks `E d T f R`,
    the schema enum, the brief, the roster), so the order check accepts
    `E,d,T,f,R` while still refusing the old backwards shape. The brief also says
    the turn is a DOOR: from the first 'turn' on, no shot may be 'develop' again
    -- 'turn' while it happens, 'fall' while it settles, 'resolve' at the end.
    Trade-off measured on the prior-fed rerun: feeding a refused draft back made
    the writer rewrite from that draft instead of the scouts (provenance 0/13
    prompts traced to a farm idea, against 13/13 on the un-primed 9-shot smoke).
    The repair goal and the ensemble's provenance goal pull against each other and
    want a "keep the shots that work, draw the new ones from the scouts" pass.
    **FINAL MEASURED RUN on the real refused case** (`ego`, 13 requested, the
    kept refusal fed back, `fall` live, `--story-retries 0`): **1 writing pass +
    1 extension call -> 13/13 shots, shape `EddddddTTTffR`, 0 story breaks,
    0 story-link warnings, 1137 s, ADOPTED** (the pass returned 12 shots and the
    extension added the 13th). That is the first `ego` draft the gate has ever
    adopted past 6 shots -- its history is now 6 -> 6 -> 13. What remains is
    writer-level advisory (2 jump cuts, 4 scene labels, metronomic holds), which
    adoption repairs: 16 timing/framing repairs and the movement plan scoped one
    finishing tilt. Check the 6 advisories by eye before shooting; the gate's
    blocking failure is gone.
  - **A clean-but-short list had nowhere to go.** Re-rolling throws a good story
    away, so the shortfall is closed by COVERAGE: `story.extend_shot_list` asks
    for inserts that each name the shot they follow (`EXTEND_SCHEMA`,
    `after: integer`) and merges them with **beat, scene, movement and camera
    pinned to the insertion point**. An insert can only repeat the beat it lands
    in, so an arbitrary number of added shots cannot reorder a story that already
    passed; the final shot can never be displaced; duplicate/terse inserts and
    inserts after the ending are dropped with a reason; a partial pass is asked
    again. Measured on an approved 9-shot list, target 12: **+3/3 kept, 0
    dropped, shape `EdddddddddTR`, 0 story breaks**, one call. First smoke run
    exposed that the model wrote camera directions ("zooming out from...") into
    the prompt; the brief now carries BAD/GOOD examples ("write the image, not
    the camera") and the re-run came back as static images.
    `write_shots_ensemble(extend=True)` is opt-in on purpose: the showrunner
    comparison must keep measuring the writer's own count.
  - **The loop survives the process.** A refused draft is written to
    `productions/<slug>/draft_attempt.json` -- beside `project.json`, never in it,
    so "the production was NOT changed" stays true -- and
    `write_shots_ensemble(prior=...)` hands it to the FIRST attempt of the next
    run as a repair brief (and forces the proposal cache open, since replaying the
    refusal is not a loop). `draft.py` prints the whole loop on one line
    (`[draft] writing loop: N attempt(s), M extension call(s) -> X/Y shot(s),
    shape ...`), `job_runner` has an `extension` progress phase and a `refused`
    phase, and the studio's write view says when a refused draft will be fed back.
    `draft.py --story-retries 0` runs one pass only (fast, but a broken draft is
    refused rather than repaired). Also fixed while here:
    `sequence_direction.shots_for_duration` floors at **3** shots, not 2 -- a 4 s
    target used to derive 2 shots, and a 2-shot film can never pass the gate
    (establish + turn (+ fall) + resolve need at least three slots), so the
    minimum length was guaranteed to produce a REFUSED draft.
  - **240 tests pass**, including `tests/test_story_loop.py` (roster, the `fall`
    beat and its gate acceptance, count-aware repair note, insert pinning/junk
    dropping, partial-pass re-ask, real stubbed end-to-end loop, `prior` on the
    first message, pending-draft sidecar, job-runner phases, and a draft-job test
    that refuses, keeps the writing, feeds it to the next run, then adopts and
    clears it) and `tests/test_prompt_shape.py` (shape check, image-clause cut,
    leak-aware best-of, the pass wired into the loop off-by-default, no-leak of
    field values into prompts). Smoke artifacts: `output/film/loop_smoke_result.json`,
    `output/film/loop_smoke_extend_result.json`.
- **The first film, audited: the story held, the prompts did not.** `ego` was
  rendered end to end (13 takes at ~14-35 min each, a **24.38 s, 512², silent**
  cut) and the output was audited against its own prompts -- tool:
  `output/ego_audit/report.py`, metrics: `metrics.json`, sheets
  `contact_film.png` / `flow_strip.png`.
  - **Prompt → frame was weak and the cause was the writing.** CLIP
    prompt-to-take similarity **0.300** against **0.253** for a *different*
    shot's prompt (margin only **+0.018**); top-1 retrieval **9/13**. **12 of 13
    prompts were two-image composites** ("A barefoot man walks across a dusty
    city floor...; the shadow of his ego looms large against a moonlit forest"),
    so the renderer drew one clause or the concept prior. The film ended on a
    **blue vintage car** because the resolve prompt named "an old car" and the
    renderer drew the car instead of the protagonist -- and its CLIP alignment was
    *fine* (+0.05) because the prompt itself was wrong. The best-aligned shots
    were the single-image prompts: the empty chair (+0.085 margin), the campfire
    (+0.059), the parking lot (+0.03).
  - **The flow: the turn lands, the film is a montage, the ending broke.** Beat
    shape `EddddddTTTffR` and the two loudest cuts sit exactly at the turn
    (s07→s08 luma **20.7**, s08→s09 **10.0**) with the motion peak on s08
    (**4.37** vs ~1.8 typical). But **9/13 shots are montage runs** (12 scenes,
    **32 cuts/min** vs human 14.5, hold **CV 0.084** vs 0.75) and there is no
    recurring figure: only s02 is hold-conditioned; the man is naked, then
    suited, then a face, then silhouettes. The one finishing tilt verified at
    **−43.2 px net vs +3.2 px** on the static neighbour.
  - **The render's own `stats.json` was lying.** It is written from the plan
    snapshot taken before the run generated anything, so a run that generates and
    assembles in one process recorded every shot as `unrendered` with `take:
    null`. `make_film.py` now reads state/take **live at assembly**; the stale
    `ego` stats (both copies) were repaired to 13/13 rendered.
- **The prompt pass: one image, and the model was chosen by measurement.**
  `story.prompt_problems` reports `prompt:` advisories (semicolon/second
  sentence, >24 words, meaning words such as ego/confrontation/aftermath/
  symbolic) and the repair brief carries them. `story.rewrite_prompts` is a
  TEXT-ONLY pass on the approved list: same shots, beats, order and timing, a
  per-shot best-of that only accepts a rewrite with fewer measured issues, the
  deterministic image-clause cut for the model habit of returning
  "image; meaning", and leak detection (a prompt carrying "camera: none" or
  "2.0 seconds" is rejected). Measured on the real 13 approved prompts:
  **granite4.1:8b removed the semicolon but kept the meaning (12/13 still
  flagged), qwen2.5:7b got 9/13 clean, and mistral:latest returned 13/13 clean
  single-image prompts in one batch call (~1.5 min)** -- so
  `PROMPT_PASS_MODEL = "mistral:latest"`: this is instruction-following on a tiny
  rewrite, not a writing or filmcraft task. `draft.py` runs the pass by default
  (`--no-prompt-pass`, `--prompt-pass-model`), `job_runner` has a `prompt pass`
  phase, and the check is ADVISORY at the gate -- a text problem must not refuse
  a film the story gate accepted.
- **The resolve re-shot: the audit's biggest defect fixed on the real film.** s13
  was a car. Its prompt was replaced with a single concrete image in the fall's
  place/figure ("a man standing alone on a rain-slicked parking lot at dawn, a
  long shadow stretching behind him on the wet asphalt, cold grey light"), its
  scene set to s12's and scale to medium, so the renderer's hold conditioning
  anchored it on **s12's take** (IP-Adapter) instead of inventing a new figure.
  Only s13 regenerated (693 s, take `t02`, `conditioned: true`,
  `hold_reference: shot 12`); the other 12 takes stayed current; the cut
  re-assembled to `productions/ego/renders/film` (24.38 s). Measured after:
  s13 prompt-to-frame **0.287 → 0.329** (margin +0.025) and the **s12→s13 cut
  CLIP 0.67 → 0.784**; the car is gone and the same suited man stands on the wet
  road at dawn. Before/after strips: `output/ego_audit/flow_strip_before.png` vs
  `flow_strip.png`; the edit is in the production history (`act: edit`,
  `shot: s13`).
- **Making those findings apply to EVERY project, not just `ego`.** The audit and
  its pre-render version are now pipeline tools, not one-off scripts:
  - **`output_audit.py`** -- the generic audit (`--store`, `--render`, `--no-clip`,
    `--stills`). Read-only; writes
    `output/audit/<slug>/{metrics.json,report.txt,contact.png,flow_strip.png}` and
    exits 1 on any `warn`. It reproduces the `ego` findings on
    `--store ego --render film`: prompt→frame margin **+0.011**, prompt-shape on
    **12** prompts, metronomic **CV 0.084**, exposure jumps at s07→s08/s08→s09,
    and the resolve's new imagery as `info` (the car). The verdict logic is pure
    functions (`flags_from`, `cut_by_mode`, `new_words`, `hold_cv`) and unit-tested
    (`tests/test_output_audit.py`), so the report and the exit code cannot drift.
  - **Previs measures the PROMPT before the render.** `previs.measure` now reports
    per-shot prompt→still alignment and margin (a still matching ANOTHER shot's
    prompt better is flagged) and the contact sheet carries each shot's **prompt
    under its still**. The first `ego` resolve (a car) would have been visible for
    ~1 minute of previs stills instead of an hour of render. Gate behaviour is
    unchanged: the numbers are informational because CLIP is a proxy and the
    contact sheet is the arbiter.
  - **`make_film --store` reports the previs status** before it starts (a warning,
    not a block; `--no-previs-check` silences it). The studio render buttons
    already gated on previs, but the CLI path did not -- and the first `ego`
    render was a CLI render ("film" is the CLI default render name).
  - **247 tests pass.**
- **Coverage-first writing: plan the scenes, measure it, bind the anchors.** The
  last structural gap: the first film was a montage of ~12 places in 13 shots, so
  the director montaged it -- correct for the material. The writer had always been
  told "2-3 scenes, one place and one figure each"; nothing measured it. Three
  pieces, all on the draft path:
  - `story.write_scene_plan` (~1 min): 2-3 places, one recurring figure, one
    action each and a shot budget summing to the request. Normalised (2-3 scenes;
    budgets fixed to the count) and attached ABOVE the farm ideas; if the model
    cannot produce a usable plan the writer proceeds without one, as before.
  - `story.coverage_problems`: the sequence director's OWN place/figure affinity
    (`visual_org.affinity`, style/atmosphere stripped) per scene run. Flags a
    scene label that does not hold (adjacent affinity **< 0.10**), a figure change
    inside a scene, a missing `subject`, or a film with no 2-shot hold. `coverage:`
    is ADVISORY at the gate (a montage is a legitimate arrangement);
    `coverage_retries` (default **1** in `draft.py`) re-asks once when the story
    holds but no scene does.
  - **MEASURED, three real runs.** (a) lighthouse n=6, plan first, no anchors: the
    plan was followed in the scene LABELS, but each prompt described a different
    aspect with different nouns -- within-scene affinity **0.06-0.07**, 5-6
    coverage problems, and the one repair re-ask did not fix it: a text model
    cannot re-arrange images it has already written. (b) The fix is the pipeline's
    OWN mechanism: canonical place/figure **ANCHORS**
    (`visual_org.effective_prompt`), used by the workbench at freeze and never
    applied by the store draft path. After the prompt pass, `story.anchor_plan`
    binds every prompt to its planned scene's place+figure clause and CUTS THE FILM
    ON THE PLANNED SCENES (2-3), so coverage passes by construction and the
    director sees high within-scene affinity. (c) The real end-to-end draft
    (`lighthouse`, n=6, default `draft.py` path): plan 3 scenes (2/3/1 shots);
    attempt 1 story clean with 2 coverage problems -> repair -> 3 -> budget spent;
    prompt pass 6/6; **anchors bound 6/6 -> 0 coverage problems**; the director
    then chose **hold for shots 1-4 (affinity 0.44) + hybrid 5-6**, and the 6-shot
    `EdTffR` list was adopted -- a held scene, not a montage. What remains is
    writer-level advisory only (2 jump cuts, metronomic holds, 1 prompt).
  - `draft.py` runs plan + repair + anchors by default (`--no-scene-plan`,
    `--coverage-retries`); `job_runner` has `scene plan` and `coverage repair`
    phases; adoption records `project["anchors"]` so tools re-reading the store do
    not report the appended anchors as writer faults; **259 tests pass**
    (`tests/test_coverage.py`, anchor exemption in `tests/test_output_audit.py`).
    A scratch production `lighthouse` holds the measured example (no takes).
- **THE 13-SHOT TEST (the length that failed): structure solved, images the next
  ceiling.** Same concept, 13 requested, no render; scratch productions `ego_v2`,
  `ego_v3`, `ego_v4` are the runs.
  - **Coverage/holds.** old `ego`: 11 scene runs, **2 held / 9 montage** shots,
    mean in-scene affinity 0.245. `ego_v2` (plan+anchors v1): 4 runs, 12 held / 1
    montage, affinity 0.481, 2 figures. `ego_v3` (after two length fixes): 4 runs
    (2/4/4/3), **13 held / 0 montage**, affinity 0.532, **1 figure**, 0 coverage
    problems, story `EdddddTfffRRR`. The faults the first 13-shot run exposed are
    fixed: `_normalise_scene_plan` keeps ONE figure (most common), and
    `_assign_scenes` assigns shots to planned scenes by CONTENT under the plan's
    order (monotone; budgets only a tie-break) instead of by the plan's shot
    budget -- which had put courtyard prompts into the stained-glass scene.
  - **The images, not the structure.** Previs of `ego_v3` (13 conditioned stills
    ~12 min): cross-cut CLIP 0.92, prompt->still margin **-0.016**, luma jump 19.6,
    stills near-duplicates of one foggy corridor. The plan's places never appeared
    because (a) prompts inside a scene repeated one sentence, and (b) anchors were
    APPENDED to a generic writer sentence and the image model ignored them.
    Fixes in: `prompt_variation_problems` flags same-scene adjacent prompts with
    stemmed overlap **>= 0.65** (a hold is coverage, not a duplicate; fed to the
    repair note and the prompt pass is told to change angle/moment), and
    `_front_anchored_prompt` puts the planned place/figure at the FRONT.
  - **`ego_v4` (front anchors):** 13/13 held again (12 in holds + the resolve
    single), 0 coverage, story `EdddddTTTTffR`; prompts now OPEN with the planned
    places ("dimly lit room with a single window...", "...empty playground with
    rust-covered metal structures..."). Previs: cross-cut CLIP 0.888, luma jump
    **19.6 -> 9.2**, and the contact sheet shows THREE environments (window room /
    rusted playground / peeling house) instead of one corridor -- the plan's places
    reached the pixels. Caveat: the prompt->still margin is still **-0.010**, but
    that metric is now partly invalid: every prompt in a scene shares the front
    anchor by construction, so it trivially matches its scene-mates better. The
    honest metric is variation BEYOND the anchor.
  - **One real flake caught and fixed:** on `ego_v4` mistral returned 12 prompts
    for 13 shots, so the prompt pass was skipped entirely and the writer's
    composites shipped. `rewrite_prompts` now retries once on count-mismatch/parse
    failure (and only then keeps the originals). **265 tests pass.**
  - **`ego_v5`, the clean run** (retry + variation instruction + front anchors):
    13/13 requested, story `EdddTTffffffR`, 13/13 in held runs (12 + the resolve
    single), **0 coverage problems**, one figure; prompt pass **10/13 rewritten,
    0 still flagged**; anchors bound 13/13 and the planned places open every
    prompt. Variation BEYOND the front anchor (anchors stripped, adjacent
    same-scene stemmed overlap): **0.092 with 0 pairs >= 0.65** (the v3 held film's
    repeated text measured 0.429), so the shots inside a hold now differ. Previs:
    cross-cut CLIP 0.902, luma jump **11.4**, three environments visible (window
    room / moonlit park / ruined cathedral) instead of one corridor. A dangling
    clip found here ("...cross-legged in a") is fixed: `_clip_clause` strips
    trailing function words. **266 tests pass.** Remaining image-side weakness:
    the stills are dark/low-contrast and the scenes share too much fog, so the
    next decision is a style/palette pass before paying for a 13-shot render.
  - **The style/palette pass (the user's "the unclearness is the style").** The
    murk had two sources: text-level atmosphere (every scene's prompts shared
    "fog/mist") and one flat grade (every scene palette 'neutral'). Both are now
    directable:
    - `style_pass.py` (new tool, any adopted store): strips fog/mist/haze/
      "shrouding" from the prompts, inserts ONE concrete LIGHT clause per held
      scene after the place/figure anchors, and sets one palette per scene from
      `looks.ARC` (default warm-gold -> cold-steel -> amber-dusk -> cyan-night,
      cycling). Idempotent, snapshots before/after to `output/style/<slug>/`.
    - The PIPELINE now does this at source: `SCENE_PLAN_SCHEMA` requires `light`
      and `palette` per scene (enum = `looks.ARC`), the plan brief says do not
      blanket the film in fog, `_front_anchored_prompt` puts the scene's light
      third at the front of every prompt, and `film_store.shots_from_proposal`
      carries the plan's palettes onto the store scenes (so `make_film` grades
      them). `previs.py` now applies each scene's palette to its still too, i.e.
      shows the graded film rather than the raw model palette.
    - **MEASURED before/after** (ego_v5, 13 stills each, same seeds/conditioning):
      `output/style_ego_v5/before_contact.png` is warm/cold-but-grey and flat;
      `output/previs/ego_v5/contact.png` has four legible looks (amber room ->
      silver mirror -> warm dusk -> cold night). Stills luma jump 11.4 -> 11.5
      (similar, as designed: a palette shift is stylistic, not a technical
      exposure change), CLIP 0.902 -> 0.921. The residual softness is the
      checkpoint's own low-contrast style, not the text or the grade -- the next
      lever there is model/CFG, not the pipeline.
    - **WIDENED after the first comparison** (the first arc stayed orange/blue):
      `looks.LOOKS`/`ARC` now also ship `emerald`, `sodium`, `magenta`, `teal`,
      `violet` (tints up to +/-20 Lab, plus an optional `hue` rotation in
      `apply_look`), `story.SCENE_PALETTES` mirrors the new ARC, the scene-plan
      brief requires each scene's `light` to name a concrete SOURCE and colour,
      and `style_pass.py`'s fallback arc now spans orange/green/blue/magenta.
      `previs`/`make_film` negatives add `haze, glow, bloom, soft focus` and the
      "dust in the air" light clause is gone. MEASURED on the same 13 stills
      (before / arc1 / arc2): mean chroma **2.1 / 10.9 / 14.2**, per-scene hues
      **75 / 165 / 256 / 346 deg** (orange / green / blue / magenta) vs arc1's
      74-77 and 256-264 only. Sharpness did NOT improve (205 -> 191 both arcs):
      the softness is the checkpoint, not the prompt text or negative prompt.
    - **The right fix for "big colour jumps", not a strength knob.** The render
      itself already applies the shared technical baseline (exposure/contrast/
      grain normalised) BEFORE the palette, so a bold palette cannot read as a
      technical error; the remaining cross-scene difference is the edit. Previs
      now mirrors that look stage (`render_stills` applies the same baseline, then
      the per-scene palettes, conditioning holds on the RAW anchor), and `measure`
      reports continuity SEPARATELY for **within a scene** (defects only) and
      **across scene cuts** (style, not an error). Measured on the styled `ego_v5`
      stills: **within-scene luma 1.48 / hist 0.545 / CLIP 0.899 over 9 cuts**
      (holds are continuous), **scene cuts luma 18.7 / hist 1.0 / CLIP 0.807 over
      3 cuts** (deliberate palette separation). Only within-scene jumps warn/
      gate now; a palette change no longer has to be blunted to satisfy a
      confounded single-number "luma jump". Also fixed the gate-path bug this
      exposed: `previs --store` wrote `output/previs/<slug>/` while `gate()` read
      `output/previs/<slug>_previs/`, so a CLI previs could never satisfy the
      render gate; `name_for_store()` now defines the one name both use.
      **279 tests pass.**
- **THE FIRST REAL FILM THROUGH THE WHOLE CHAIN -- DELIVERED.** `ego_v5`, 13
  shots, styled, rendered from the store with
  `make_film.py --store ego_v5 --name film`. The first attempt died mid-shot 2;
  the store resumed from the two finished takes, so only the in-flight shot was
  lost. **Delivered: 13 shots, 399 frames, 24.94 s @ 16 fps, 938 KB**, all takes
  `t01`, hold conditioning 9/13 at IP-Adapter 0.7, technical baseline then
  palettes `warm-gold x4 / emerald x4 / cold-steel x4 / magenta x1`. Files:
  `productions/ego_v5/renders/film/` (`film.mp4`, `film.gif`, `stats.json`).
  `output_audit.py` on the delivered cut: **only one warn -- metronomic
  (shot-length CV 0.074)**; the three luma jumps at scene cuts (15.6 / 18.6 /
  18.0) are classified `style-cut: info` (the palette shift is the edit), the
  prompt->frame margin is `anchored-prompts: info` (shared front anchors make it
  structurally ambiguous), and `resolve-new-imagery` is info (magenta is the
  palette). Remaining craft: pacing CV (the renderer's 1.94 s motion budget) and
  the checkpoint's softness.
- **RHYTHM: from a metronome (CV 0.074) to the measured human distribution
  (CV 0.699, human 0.75).** The delivered film's shots all sat in 1.67-2.26 s.
  A first fix widened R4 but still produced 2.5/1.5 alternation -- better than
  uniform, still not what the research measured. `PACING_MEASURED_STUDY.md`
  (700 MSR-VTT clips, 3,185 shots) says human short-form is BIMODAL:
  <0.5s 14.0% | 0.5-1s 22.6% | 1-2s 17.8% | 2-3s **12.0%** | 3-5s 14.6% |
  >5s 19.0%; median 1.73 s, CV 0.75. The 2-3 s band is the LEAST common one our
  old films used for everything.
  - R4 now assigns by ROLE into the study's bands: accents (inserts, reactions,
    turns, close-ups) at 0.80-1.45 s, mids for other masters, HOLDS for scene
    openings / establishes / resolves. It repairs R9 itself (scene opening
    longest, no triple/adjacent near-equal); holds are floored to 0.01 s.
  - The study's LONG band needed a motion budget, so the fix is a FINISHING
    feature, not another render: `shot["upsample"]` raises the RAFT interpolation
    factor (16 raw frames: x2 = 1.94 s, x4 = 3.81 s, x6 = 5.69 s of smooth motion
    -- the study's "procedural hold"). `upsample` lives in `finish_inputs`, so
    raising it re-finishes (RAFT + shake + grade) and never regenerates. Long
    shots get x4, the resolve x6; accents stay native and are trimmed.
  - `retime.py --store <slug> [--dry-run] [--max-upsample N]` applies this.
  - MEASURED `ego_v5` -> `film_human` (31.81 s, 509 frames): shot lengths
    **[4.06, 1.75, 1.00, 1.81, 4.44, 0.94, 1.25, 1.62, 4.31, 1.19, 1.56, 1.31,
    6.56]**, CV **0.699** (human 0.75), median 1.56 s (human 1.73). Band mix:
    0.5-1 s 7.7% (human 22.6), 1-2 s 61.5% (17.8), 2-3 s 0% (12.0), 3-5 s 23.1%
    (14.6), >5 s 7.7% (19). Provenance of the fix is real: 0.2% near-duplicate
    consecutive frames across 509 -- the long holds are interpolated motion, not
    duplicated stills. `output_audit.py` on `film_human`: **no warns**, CV 0.699,
    all flags info. `film_human` is the deliverable; `film` (uniform) and
    `film_rhythm` (alternation) are kept for comparison.
  - REMAINING GAP, honest: human has 36.6% under 1 s and 19% over 5 s. Our
    accents bottom out at ~0.94 s because the takes are 16 raw frames (trimming
    further is 4x speed); real sub-second accents need **8-frame takes** (the
    study's documented per-shot option), and only one shot is >5 s because only
    the resolve got x6. Those are render-cost decisions; the machinery
    (`upsample`, per-shot frame trimming) is in place.
  - **284 tests pass** (`tests/test_rhythm.py`, pacing tests).
- **What stays manual, by nature.** Changing what a shot is ABOUT is an editorial
  decision: the prompt pass may only rewrite the text, and the audit may only
  measure the result. The s13 fix (car → the man on the wet road) was that
  decision, and it stayed one. The coverage/montage question is now handled for
  the default DRAFT path by the plan + anchors above; and the flat look is now
  handled by the scene plan's light/palette + `style_pass.py`; and rhythm is now
  the measured distribution (CV 0.699 vs human 0.75) with procedural long holds.
  What remains is the checkpoint's own softness (model-level), IP-Adapter
  appearance vs identity inside a hold, and the timing TAILS: human has 36.6%
  sub-1 s accents and 19% >5 s holds; ours are ~0.94 s minimum (16-frame takes;
  sub-second needs 8-frame takes per the study) and one >5 s hold (only the
  resolve got x6).
- **Pacing measured on a real corpus, not blogs.** The two commercial-blog
  datasets in the pacing doc were removed. `pacing_study.py` is our own
  hard-cut detector (HSV+luma spike, validated on our renders' known cuts at
  **F1 0.992**). Run over 700 randomly sampled **MSR-VTT** human short videos
  (171 min, 3,185 shots): **14.5 cuts/min**, ASL **3.23 s**, shot median
  **1.73 s** (p10 0.33, p90 8.37), within-clip shot-length CV **0.75**, 36.6% of
  shots under 1 s and 19.0% over 5 s. Our films: 23.8 cuts/min, ASL 2.21 s,
  CV 0.17, 69.9% of shots 2–3 s, 2.7% outside 1–5 s. Human short-form is
  BIMODAL (quick accents + long holds); ours is a spike at 2 s. Script gate now
  also flags a metronomic whole film (CV<0.20). Full study:
  `PACING_MEASURED_STUDY.md`; artifact: `output/pacing_study/`.
- **Visual camera A/B: one pair rendered.** Same shot/seed/seconds, `tilt-down`
  vs `none`, through the real store path (~12 min per take). The move reached the
  pixels and was direction-consistent: net vertical travel **−68.9 px vs +2.8 px**,
  frame transitions down/up **15/0 vs 6/7**. The tilt-down take also came back
  **softer** (Laplacian var 38 vs 77), but it regenerated a different composition,
  so that is content-confounded. Do not make camera moves the default for final
  renders until a second pair separates the two. Full report:
  `CAMERA_VISUAL_AB.md`, tool: `camera_ab_visual.py`, artifacts:
  `output/camera_ab/visual/`, takes: `productions/_camera_ab/`.

**Still not done (deliberate):** a second/third visual pair to settle whether
camera-LoRA takes are consistently softer and whether pan moves are as clean as
tilt; subject-motion LoRAs are opt-in and unvalidated as a direction candidate;
character/subject continuity has no mechanism; renders are silent.

---

Written at the end of a long session, because the context window is nearly full.
Read this before doing anything. It records what works, what was tried, and —
most importantly — the traps that produced **four false negative verdicts** in
one session. Those cost hours. Do not repeat them.

**THE PRODUCTION UI IS `ui/studio.py`, AND §0 IS THE CURRENT STATE — read §0
first.** §14 describes the seven-stage workbench that §0.3 supersedes; it is kept as a
record of the bugs found while building it, not as a design to rebuild. §7 is the
pipeline status and is still accurate about the GENERATION stack.

---

## 1. The project

Generating watchable YouTube videos from abstract concepts. The user's pipeline:
Ollama writes a story → a model generates imagery → it becomes a video.

The user has been explicit about what they **do not** want:
- stills held on screen (a slideshow) — "I DON'T WANT A STILL FRAME GENERATOR"
- warping / flow interpolation between unrelated images
- training anything ("takes forever just to produce mediocre results")
- buying RAM or a GPU
- Colab free tier ("limitations are ridiculous")

They **do** want: real generated motion, on the machine they already own.

---

## 2. Hardware — the binding constraint

| | |
|---|---|
| GPU | **none.** `torch 2.5.1+cpu`, `cuda.is_available() == False` |
| RAM | **15.3 GB total**, typically 3–8 GB free |
| CPU | 12 cores |
| Disk | ~113 GB free |

**RAM, not compute, is what rules things out.** Every modern video model
(LTX, Wan, CogVideoX) pairs a small video backbone with a **T5-XXL text
encoder ~9.5–19 GB** — larger than this machine's entire RAM. LTX 2B is 1.4 GB
quantized; its text encoder is 19 GB fp32. That is why they cannot run here.

Models that *do* fit are SD1.5/CLIP-based, because CLIP is 250 MB.

---

## 3. What works, with measured numbers

All measured on this machine. Do not re-derive.

### Generators

| method | resolution | s/frame | frames/hour | notes |
|---|---|---|---|---|
| **AnimateDiff v3 + Lightning 4-step** | 512² | **35.6** | **101** | coherent, subtle motion. **Best.** |
| T2V-Zero (SD1.5 + CLIP) | 512² | 175.0 | 21 | works; most motion, least stable |
| SVD-XT | 512×288 | 193.0 | 19 | prettiest frames, almost no motion |
| SD1.5 + LCM-LoRA, 6 steps (stills) | 512² | 84.0 | 43 | for static shots |
| tiny-sd, 25 steps | 512² | 208.0 | 17 | superseded by SD1.5 |

### Post-processing

| | |
|---|---|
| RAFT x2 temporal upsampling | 16→31 frames in **8.8 s** = **1.5% overhead** |
| CLIP prompt-alignment score | predicts quality well; usable as an auto-filter |

### Quality signal (from 4 diverse prompts)

| prompt | motion | coherence (clip drift) | prompt align | verdict |
|---|---|---|---|---|
| city | 23.2 | 0.061 | 0.265 | good |
| river | 14.6 | 0.017 | **0.329** | good |
| grass | 13.1 | 0.024 | 0.280 | weak |
| smoke | 15.1 | 0.091 | **0.259** | **failed** |

**2 of 4 usable.** CLIP alignment tracked quality almost exactly — use it as a gate.

---

## 4. CRITICAL TRAPS — read these first

### 4.1 256px is a trap. Always use 512.

Plain SD1.5 at 256px produces **tiled green banding** — garbage. At 512px the
same prompt/seed produces a proper image. Proof: `output/res_test/compare.png`.

This single fact caused **three false verdicts** in one session:
- "AnimateDiff produces neon mush" → wrong, it was 256px
- "T2V-Zero produces abstract mush" → wrong, it was 256px
- the user's own "the models were shit" conclusion, based on my early runs

**AnimateDiff v3 is also trained at 512.** Never run these at 256.

### 4.2 fp16 on CPU is pathological

Measured: 2574 s CPU time at only **~1.7× average parallelism** across 47
threads — effectively single-threaded. fp32 gives ~10× parallelism.

**Load fp16 checkpoints but cast to fp32**: `variant="fp16", torch_dtype=torch.float32`.

### 4.3 pip is broken on this machine

PyPI is reachable from Python's `urllib` but `pip` hangs (even `pip download six`).
Its temp/target handling also trips the sandbox.

**Workaround**: fetch wheels with `urllib`, unzip into a target dir.
See `.add_pkgs.py`. A wheel is just a zip.

### 4.4 Sandbox: caches must live in the workspace

`HF_HOME`, `TORCH_HOME`, `TEMP`/`TMP` all default outside the writable sandbox
and cause `PermissionError`. Set them to workspace paths. See any script header.

### 4.5 diffusers 0.28 is too old

The project's env lacks `LTXPipeline`, `GGUFQuantizationConfig`,
`AutoencoderKLLTXVideo`. `.ltx_env/` holds **diffusers 0.32.2 + transformers
4.45.2 + tokenizers 0.20.3**, installed by unzipping wheels. Scripts
`sys.path.insert(0, ".ltx_env")`.

### 4.6 Motion LoRA notes

- Weight **1.0 only**. Weight 3.0 *inverts* zoom-in into a 0.74× zoom-**out**.
- "Loading adapter weights led to unexpected keys" is a **false alarm** — the
  LoRA does attach (160 lora params injected). Verified directly.
- The zoom effect is subtle: **x1.04 scale change over 8 frames**. Easy to miss.

### 4.7 The SHUTTERSTOCK watermark — SOLVED

SD1.5 base reproduces stock-photo watermarks. A clean A/B proved it is **NOT**
caused by the motion LoRAs — present with and without, same prompt/seed
(`output/film/watermark_ab/ab.png`). Negative prompts do not fix it
(`watermark, text, signature, logo, shutterstock, alamy, getty, stock photo`
changed nothing).

**The fix is the base checkpoint.** `models/dreamshaper8` (`Lykon/dreamshaper-8`)
produces **no watermark** at the same prompt, seed, steps and LoRA — and the
image quality is markedly better: soft atmospheric light, muted filmic tones,
instead of SD1.5's harsh saturation. See `output/film/checkpoint_ab/compare.png`.

**Use DreamShaper 8 as the base from now on.** Motion adapter and motion LoRAs
are SD1.5-architecture and work unchanged.

Download gotcha: DreamShaper ships `text_encoder` as `.bin` and its scheduler as
`scheduler_config.json`. Downloading with `ignore_patterns=["*.bin"]` leaves
`text_encoder/` empty and the load fails with
`expected str, bytes or os.PathLike object, not NoneType`. Fetch
`text_encoder/*`, `scheduler/*`, `tokenizer/*` explicitly.

Therefore the user's earlier proposal to drop the zoom to fix the watermark was
**not** the cause and must not be applied — the zoom works and should be kept.

---

## 5. Things that were tried and rejected (do not retry)

| approach | verdict | why |
|---|---|---|
| Optical-flow "redrawer" (RAFT warp between keys) | **rejected by user** | warping cannot invent a pose; smears unrelated images |
| ControlNet-openpose + SD1.5 | dead end | **human-only**; and it failed 2 of 4 poses on humans |
| ToonCrafter | not viable | 12–24 GB, needs GPU |
| LTX-Video 2B | blocked | T5-XXL text encoder is 19 GB; GGUF T5 cannot load (`gguf_file` is not a T5 param) |
| SparseCtrl | not pursued | AnimateDiff-based (user had rejected it, on my bad test) |
| fg-over-held-background compositing | **disproven with camera-only LoRAs** | AnimateDiff motion is global drift — no separable foreground. **Reopened by section 9**: the subject LoRAs do produce a localized, articulated figure. Not re-verified on this machine yet. |
| LDS / Dynamic Textures (Doretto 2003) | works, wrong source | stable and infinite, but AnimateDiff clips are ~10-dimensional so output is a smooth undulation. Right tool, needs *real* texture footage |

### 5.1 The key measurement behind the compositing failure

Where does AnimateDiff's motion live?

```
top   5% of pixels hold   7.7% of the temporal variance
top  10% of pixels hold  14.2%
top  25% of pixels hold  31.7%
```

Near-uniform. If a *subject* were moving, the top 10% would hold most of it.
**AnimateDiff's motion is global drift, not subject motion**, plus 33% of the
frame-to-frame change is a global brightness shift.

This is also why adjacent clips look samey — singular values show the clip has
only ~10 effective degrees of freedom (95% energy in the top 10 components).

**Consequence**: for real subject motion, look at the **subject-motion LoRAs**
(`Cseti/AD_Motion_LORAs`: `walking-frontview`, `running-frontview`,
`ballerina-spin`, `drone-*`) — untested so far, and the most promising lead.

---

## 6. Components built

| file | what it is |
|---|---|
| `animation_spec.py` | **exposure sheet engine.** Holds, layers, per-layer clocks, sparse drawings, rejects autoregressive chaining. 105 tests pass. |
| `ANIMATION_SPEC.md` | the normative spec + decision record |
| `edit_planner.py` | **continuity rules R1–R5** for shot sequences + rhythm curve |
| `story.py` | the text layer: single-writer path, and the **ensemble** (idea farms → one showrunner). Also `prompt_leaks` (field values fused into `prompt`) and the shared text measures. See §11.3 |
| `director.py` | picks a motion per shot. `mode="global"` (default) optimises the WHOLE shot list at once; `mode="greedy"` is the shipped left-to-right behaviour, frozen as the A/B control. `compare_modes()` prints the comparison. See §9.6 |
| `handheld.py` | **procedural camera shake** — per-shot, band-limited in Hz, continuous envelope, sub-pixel resample, roll. See 6.1 for what the first version got wrong. |
| `handheld_ab.py` | the shake before/after a human has to look at: steady/before/after panes on real footage + the camera-path plot. `python handheld.py` runs it |
| `writer_audit.py` | **is the writer the ceiling?** fresh proposals from 3 writers, prompt-text diff, self-repetition and cross-writer overlap, and detection of field values fused into `prompt`. No rendering. See §13 |
| `supplement.py` | RAFT temporal upsampling + LDS dynamic textures |
| `make_film.py` | planner → AnimateDiff + camera LoRA → RAFT → assembly |
| `make_composite.py` | AnimateDiff + fg/bg masking (composite disproven, generation code still useful) |
| `compare_video_models.py` | same-prompt/frames/seed harness across generators |
| `make_video.py` | Ollama → SD → video (the still pipeline) |
| `models/` | sd15, dreamshaper8, animatediff_motion_v3, animatediff_lightning, motion_lora/*, svd_xt, ltx_gguf, clip |

### The editing rules (the actual answer to "how do we make it watchable")

- **R1** no two consecutive shots at the same framing scale (that is a jump cut)
- **R2** no opposite camera moves back to back (pan-left then pan-right)
- **R3** no more than 2 moving shots in a row (fatigue)
- **R4** shot lengths follow a rhythm curve, never uniform
- **R5** hard cuts only

**Default to no camera LoRA.** AnimateDiff already drifts; forcing a move on
every shot is exhausting. `none` should be the most common choice.

### Camera LoRA vocabulary

Official (downloaded, `models/motion_lora/`): `zoom-in`, `zoom-out`, `pan-left`,
`pan-right`, `tilt-up`, `tilt-down`, `rolling-clockwise`, `rolling-anticlockwise`.

Additional axes, not yet downloaded, from `Cseti/Basic_camera_motion_LoRAs_sd15-ad2-v1`:
- `camera-lateral_left/right` — **truck** (translates sideways; official `pan` *rotates*)
- `camera-crane_up/down` — **boom** (translates vertically; official `tilt` *rotates*)

### 6.1 The camera rework — what was actually wrong

The complaint: the shake reads as a **jerk**, not as a held camera, and every shot
feels the same. Measured on the production call (`amount=5.0, drift=0.8,
jitter=0.3, seed=3`, 31 frames) before the rework:

| | measured |
|---|---|
| binary envelope on | **5 of 31 frames (16%)** — not the ~32% the table used to claim |
| envelope step | max \|Δ env\| **0.806 in one frame** |
| jerk | max \|Δ²\| **2.835 px/frame²**, against **0.473** with no envelope at all — the envelope *multiplied* the jerk by 6x |
| distinct crop positions | **4 of 31**. `int(cx)` quantised the frame position, so sub-pixel camera motion did not exist |
| roll | present (±0.60° peak), but derived as `amount*0.12` — a PIXEL quantity setting a DEGREE quantity — and gated by that same envelope |
| band structure | **per clip, not per second.** `_value_noise` fixed the knot count from `freq` and ignored `n`: 0.52/1.55/3.10 Hz at 31 frames, 0.28/0.83/1.66 Hz at 58. Then `render()` stretched 31 frames to the hold (0.94x–1.86x on `shadows`), shifting every band again |

So the axis was right and the character was wrong. Four fixes:

1. **Continuous envelope**, slew-limited to \|Δ env\| ≤ 0.07 (measured 0.056–0.072
   across the six shots of `shadows`) and never below 0.20. This is the jerk.
2. **One projective resample per frame** (`Image.PERSPECTIVE`, bicubic) instead of
   resize → rotate → `int()` crop: sub-pixel motion, one interpolation instead of
   two. There is no margin constant any more — the zoom is solved per clip from
   that clip's own peak offsets (1.0215–1.0455 on `shadows`, against a fixed 1.04
   before). Verified by sentinel-`fillcolor`: **0 fill pixels in 266 delivered
   frames**, where "no black pixels on the border" was *not* a valid test (see the
   trap below). A zoom-only transform also *reduces* frame-to-frame change
   (1.201 → 1.168), so the added motion is camera motion and not resample noise.

**TRAP: a mapped coordinate inside the source is not enough for a bicubic
transform.** The first version of `zoom_needed` accepted a corner mapping anywhere
in `[-pad, W-1+pad]` and was declared verified by checking for black border
pixels — which passed on the three shots tested. Run on all six, **two shots
failed**: corners mapping to `u = -1.43` and `v = 511.33`, i.e. inside that pad but
*outside the source*, so the bicubic kernel (which reaches 2 px beyond the
coordinate it is given) sampled the fill colour. 1242 pixels in one shot, 93 in
another. The fill is near-black, so on a dark film it would have shipped as a
subtle corner smudge rather than as an obvious bug. `pad` is now required
**inward** (`pad <= u <= W-1-pad`, pad 2.5), and the acceptance test is a sentinel
`fillcolor` — magenta, which nothing else in the pipeline can produce. Two lessons,
both familiar: **test every case, not the convenient three**, and pick a check the
failure cannot survive.
3. **Bands in Hz at the delivered frame count.** The film now stretches each clip
   to its exposure-sheet hold *before* shaking, so the hold is the only timeline
   and `a.shake` is a **scale** (1.0 = the default hold), not a pixel count.
4. **Per-shot character** from the shot's scale, motion kind, duration and seed:
   amplitude, band mix, roll and the envelope floor are drawn per shot. Two shots
   with the same scale and kind still differ.

**The research, and three corrections to the brief.** Full URLs at the end of this
section.

- **Bands.** Cinemachine's guidance is the best-documented parameter set: three
  noise layers per axis, low 0.1–0.5 Hz, mid 0.8–1.5 Hz, high 3–4 Hz, rotation
  noise before position. The model follows that (lean ≈0.35 Hz, sway 0.9–2.2,
  tremor 2.6–4.2 — measured dominant peaks 0.28–0.55 Hz, then 0.55–2.2 Hz).
- **Amplitude is set from captured real motion, not taste.** The Blender *Camera
  Shakify* library is captured, not synthesised: roll swings **0.6–1.0°
  peak-to-peak**, rotation ~0.5° on one capture and ~4.3° on a rougher one. CIPA
  DC-X011-2024's own standard shake waveform — built from many subjects, 500 Hz,
  32 s records — peaks at **~±0.4°** on pitch. `BASE_ROLL_DEG` sits between them,
  and the six `shadows` shots land at roll p-p **0.56–0.90°** (0.35–0.38° on the
  two deliberately-damped camera moves).
- **Rotation versus translation is a DISTANCE THRESHOLD, not a ratio**, and it is
  the one measured rule that says how framing should change the character.
  Rotation displaces the image by `θ·f`, independent of subject distance;
  translation by `d·f/Z`, falling off as 1/Z. Han et al. measured the crossover at
  **~100 cm**; CIPA puts it at **~20× focal length**, with translation dominant
  only near 1:1 macro. Measured pixels: 10° of rotation moves 314/235/230 px at
  10/50/150 cm depth, while 5 cm of translation moves 650/130/43 px at the same
  depths. `ROTATION_SHARE` applies this, so the rotation:translation balance runs
  **wide 1.54 → medium 1.21 → close 0.97 → macro 0.75**: near the lens the lateral
  channel carries more of the jitter, far from it roll does. Note the lateral
  channel is *not* "the operator sliding sideways" — to first order it **is**
  yaw/pitch (`f·tanθ`) plus body translation, and the two cannot be separated
  without depth.

**The amplitude is calibrated against the measured laws, and comes out
marginally CONSERVATIVE.** CIPA's *average vibration angle* figures look alarming
at first — **1.06–3.36°** — but they are exposure-time dependent (0.019–0.039° at
1/60 s, implying a roughly constant ~1–3.4 deg/s), so they describe a long
exposure, not a trajectory across a shot. The applicable comparators are the
measured amplitude *laws*, and the model sits inside them:

| shot | T (s) | model roll rms | model yaw rms | Xiao 2006 σ = a·T^0.5632 (a = 0.092–0.155) |
|---|---|---|---|---|
| 1 | 3.20 | 0.145° | 0.173° | 0.177–0.298° |
| 3 | 2.29 | 0.147° | 0.143° | 0.147–0.247° |
| 6 | 3.60 | 0.226° | 0.209° | 0.189–0.319° |

Pooled, the model's roll rms is **0.086–0.226°** against Xiao's **0.129–0.319°**
over the same durations — i.e. **0.67–0.71× the lowest measured estimate, never
above it**. Panasonic's measurement (10 subjects, >100 shots, one-handed) gives a
mean yaw of **0.11° at 1/8 s** with a distribution to 0.42°, and the only explicit
stabiliser design premise found — Canon JP2008139640A — dimensions its correction
angle against **0.2° at 1 Hz**. Both bracket the model. So a modest increase is
defensible; an order-of-magnitude one is not.

**And there is a hard reason not to go much bigger.** The blur budget sets what
sharp footage can contain: at a normal exposure, **≤0.05° of total rotation keeps
motion blur under ~1 px at 24, 40 and 85 mm alike** (1 px in 1/50 s = 8.4 / 5.0 /
2.4 deg/s at those focal lengths). At the current amplitude the per-frame motion is
tenths of a pixel, so the frames stay sharp *without* motion blur, which is
exactly what a 2D warp can deliver. Multi-degree shake would need per-frame motion
blur to look photographic, and a projective resample cannot produce it — so the
honest ceiling on "go bigger" is the point where the sharpness stops matching the
motion. For scale: at 40 mm-equivalent on a 512 px frame, **569 px per degree**. CIPA
records **18 µm** of blur at 1/60 s and 200 µm at 1/4 s; CIPA's own OIS hardware
corrects only **1–3°**, which brackets plausible amplitude from above.

**Two more measured facts the model does not implement.** A walking operator is
**±2–3° per axis with peaks of 10–20 deg/s** (DXOMARK), i.e. **5–7× the standing
figures used here** — that is the largest single factor found, and it is not
modelled because operator locomotion is not something the shot list expresses: a
walking *subject* does not imply a walking *operator*, which needs an explicit
field on the shot rather than an inference. And normal tremor magnitude spans a
**~30-fold range between individuals**, against one per-shot seed here.

**Three numbers that are folklore, checked and rejected.** The widely repeated
"hands shake at **10–20 Hz**" traces to a manufacturer whitepaper with no citation
and disagrees with the physiology (**8–13 Hz** overall, **7–10 Hz at the wrist**,
which is what grips a camera) — it nevertheless propagated into OIS bandwidth
design. The "**1/f²** / Brownian" characterisation of camera shake has **no
traceable published fit**; CIPA holds the raw data but synthesised its waveforms
without publishing one, so the band mix here is drawn per shot from ranges rather
than derived from a spectral exponent. And the "**1/focal length**" shutter rule is
likewise untraceable; CIPA's measured substitute is a 200 µm blur / 20 µm
determination level reported in stops.
- **"Pure translation, no rotation" was NOT true of the old code.** It had roll.
  The defect was the envelope and the integer crop, not the absence of roll.
- **"Roll is the primary cue that makes a frame read as handheld" is NOT
  supported.** Cinemachine's rationale is that handheld shake is "more
  rotationally than positionally … because that's where the camera is aiming" —
  rotation in general, not roll — and its own seven handheld presets set the roll
  channel to **zero**, with roll appearing only in the engine/rig presets. Roll is
  in the model because the *captured data* contains it, not because anything
  measures it as the discriminator. Rotation does exceed translation here
  (0.45° roll vs 0.31° yaw-equivalent), which is the sourced claim.
- **"Shake should never be continuous" was searched for and found nowhere.** The
  practitioner corpus says the opposite: continuous sub-noticeable motion is the
  professional norm — "if someone can notice the movement, then it's not subtle
  enough" — and a video-analysis patent (**US20100189410A1**) builds its
  handheld-vs-tripod classifier on the premise that perfect steadiness is
  impossible. Three operators independently say a fully static shot "stands out".
  The continuous envelope is the phenomenon, not a preference.

**Two limits, stated because they bound what any of this can achieve.** At 16 fps
the ceiling is 8 Hz, so the **6–12 Hz physiological tremor cannot be carried as a
band** — it aliases, exactly as it would in real 16 fps footage, and is modelled
as the uncorrelated sub-pixel dither instead (which is also what guarantees no two
delivered frames are identical: verified 0 identical adjacent frames). Cinemachine
states the same failure from the other side: above the frame rate a frequency
"falls between the cracks of the Nyquist rate" and reads as choppy. And there is
no parallax — rotation plus a projective term is the ceiling of a 2D warp.

The **keystone is second-order and is not the fix**: at a 0.4° yaw the
edge-vs-centre differential is `k·u²` ≈ **1.4 px**, on top of the uniform shift the
same rotation causes. Included because it is physically there; `perspective=0.0`
switches it off.

**Damping on deliberate moves is an INFERENCE, not a finding.** No practitioner
source states that an operator tightens or loosens the hold for a pan or zoom. It
is supported in direction by the concentration effect — "Shooting an immobile
object may be tougher", i.e. tracking is *easier* than holding still — and by a
deliberate move already being motion. It is contradicted in part by the head's own
pathology (stiction breakaway, overshoot on small moves), which the model does not
attempt. Kind gain: `none` 1.00, `subject` 0.85, `camera` 0.50.

**Scale gain is deliberately modest.** Physics says a tighter frame shows more
displacement for the same angle; the practitioner corpus says operators brace
harder when tight and that at the long end the answer becomes *rigging* rather
than technique (the long-lens list contains no handheld technique at all). The two
partly cancel: `wide`/`medium` 1.00, `close` 1.10, `macro` 1.25.

**Sources.** CIPA DC-X011-2024 image-stabilisation standard
(https://www.cipa.jp/e/std/image-stabilization/documents/CIPA_DC-X011-2024_E.pdf);
DXOMARK stabilisation measurements
(https://corp.dxomark.com/2026/04/09/stab-series-how-to-create-and-use-realistic-movements-for-stabilization-testing-3-3/);
Han et al., IEEE TMC 2020, on the rotation/translation crossover
(https://cs.nju.edu.cn/yafeng/YinPaper/TMC2020Han.pdf); Unity Cinemachine noise
profiles
(https://docs.unity3d.com/Packages/com.unity.cinemachine@3.1/manual/CinemachineNoiseProfiles.html);
Blender Camera Shakify captured data
(https://raw.githubusercontent.com/EatTheFuture/camera_shakify/master/shake_data.py);
US20100189410A1 (https://patents.google.com/patent/US20100189410); Morrison &
Keogh 2001 on the concentration effect
(http://isbweb.org/images/conf/2001/Longabstracts/PDF/0700_0799/0773.pdf); CML
operator threads
(https://www.cinematography.net/edited-pages/16mm_Handheld.htm); Foundry Nuke
CameraShake defaults
(https://learn.foundry.com/nuke/current/content/reference_guide/transform_nodes/camerashake.html).
The research reads are kept in `research/` — see §13.

---

## 7. Current status / immediate next step

**This section is the status. Sections 9–12 are the evidence behind it.**

**The film has run.** `output/film/test_variety/` (4 shots, 2 scenes, §10.9) and
`output/film/shadows/` (6 shots, 266 frames, 16.6 s, hard cuts, two checkpoints,
two palettes). `shadows` is the first film to keep `raw/shotNN/` — 16
pre-RAFT, pre-shake frames per shot — which is what makes the camera work
verifiable for free instead of for an hour of rendering.

**Base default**: `make_film.py --base` is `models/dreamshaper8` ✓. `make_composite.py`,
`compare_video_models.py` and `make_montage.py` still hardcode `models/sd15`;
those are comparison harnesses, so that is left alone deliberately rather than
fixed silently.

**DONE: the first film rendered through the whole new stack.** `output/film/ensemble1/`
— ensemble story → whole-film director → planner → RAFT → stretch-to-hold →
per-shot shake → shared grade → per-scene palette → hard cuts. 4 shots, 175
frames, 10.9 s, **50 min** of generation (540/788/867/809 s per shot).

| what it proves | the number |
|---|---|
| per-shot shake, with damping on the deliberate move | 3.7/0.45°, 2.5/0.43°, **1.5/0.15°** (the zoom-in), 3.8/0.49° — the mover got a third of the roll |
| the §12.3 grading fix in production | technical offsets **0.02–0.07** decomp units = **0.3–0.8 grey levels**; the old normaliser printed 1e5 on this kind of film |
| stretch-before-shake plumbing | 51/39/27/58 frames = the planned holds exactly |
| the story reads on screen | mirror / identical-figures language carries shot 1 → 4; shot 3 (the turn) is the strongest image and the director spent its one move there |
| the writer's structure survived | 0 structural repairs; only R4 timing rewrites |

The one thing that looked like a defect — shot 2 reading as a bright green garden
in a dark film — is **the user's call and not a defect**. Recorded so it is not
"fixed" later by someone reading a metric.

**DONE this session**: the camera rework (§6.1) — designed, implemented, wired
into `make_film.py`, and measured against the pre-rework generator on real
footage. Before/after GIFs and camera-path plots in `output/film/handheld_ab/`.
Also: the grading report no longer divides by a degenerate normaliser (§12.3).

**FIXED, and worth knowing**: `model_select.py` was left **non-runnable** at the
end of the previous session — the last file written (18:33, after HANDOFF.md at
10:14), with two missing local imports in `measure_spreads` (`grade_match`, and
`PIL.Image`; every other caller in that file imports locally, this one did not).
`--models auto` therefore raised `NameError` on disk even though §10.8 reports it
verified. Fixed; the dry run now reproduces §10.8's assignment exactly
(`dreamshaper8, dreamshaper8, sd15, sd15`, one switch, look shift 7.05). **When a
handoff claims a run was verified, re-run the command** — the file can have been
edited after the verification.

**NEXT (not started)**: the pool memory fix and `--pool` wiring into
`make_film.py`. `pool.load_pool` materialises every frame of every candidate
(786 KB/frame at 512²) and `score()` then needs them all; `load_pool` also needs
a `--prompts` JSON that `make_film` never writes. See §13.

**THEN**: download the Cseti truck/crane LoRAs (`camera-lateral_left/right`,
`camera-crane_up/down`) — the vocabulary already has them, they are just not on
disk, so `director.py` currently can never pick them.

---

### 7.6 THIS SESSION: the production workbench — see §0.1

The UI now exists and is a **production system**, not a report: `film_project.py`
(durable per-film state), `job_runner.py` (detached jobs, one at a time) and
`ui/app.py` (seven stages: STORY, PLAN, CAMERA, GRADE, RENDER, TIMELINE, AUDIT).

**Its immediate next steps are the OPEN list in §0.5**: confirm whether
`story.py` exits non-zero on success (which would make the job panel report FAILED
for a good run), make a film by clicking through the UI end to end, run a full
4-shot assembly from REAL per-shot stages, and decide the fate of `ui_legacy/` and
`project/`.

---

## 9. Subject-motion LoRAs — the untested lead (this session)

### 9.1 What was built

| file | what it is |
|---|---|
| `fetch_subject_loras.py` | pulls the Cseti LoRAs + the author's own candidate renders |
| `motion_loras.py` | key conversion, manual feed-forward attachment, **preflight** |
| `motion_metrics.py` | where-does-the-motion-live measurement (loc5/10/25, spread, travel) |
| `subject_motion_test.py` | the A/B harness: `--refs`, `--preflight`, `--run`, `--report` |
| `motion_vocab.py` | the motion vocabulary: kind, axis, sign, CLIP descriptor, measured evidence |
| `director.py` | chooses a motion per shot on **fit / cohesiveness / continuity** |

### 9.2 The LoRA files are NOT diffusers format

`Cseti/AD_Motion_LORAs` files (400 tensors) differ from the official camera LoRAs
(336 tensors) in three ways: a `temporal_transformer.` segment, attention blocks
named `attention_blocks.0/1` instead of `attn1/attn2`, and no `unet.` prefix.
`motion_loras.convert_keys` rewrites them. **240 attention + 160 feed-forward
tensors, 0 unmapped.**

### 9.3 Two traps that would each have produced a false negative

1. **"Provided path contains more than one weights file."** diffusers refuses an
   adapter directory holding two `.safetensors`. Writing the feed-forward half
   next to the attention weights made *every* subject LoRA silently fail to load
   while still reporting feed-forward blocks attached. The FF file now lives in
   `models/subject_lora/_ff/`, not in the adapter directory.

2. **`ff.net.0.processor.proj_lora` is not a module.** The real target is the
   Linear *inside* the GEGLU (`ff.net.0.proj`). Attaching one level too high
   wraps the GEGLU, whose output is half the size:
   `size of tensor a (1280) must match the size of tensor b (2560)`.

### 9.4 The preflight — do not skip it

`motion_loras.verify_effect` runs the UNet twice on identical random latents,
with and without an adapter, and returns `max |delta|`. **Zero means the weights
are not in the graph.** `make_film.Generator` now raises rather than generate.

Measured on this machine (16 frames, 32×32 latents, ~5 s total):

```
walking-frontview   max|delta| = 1.146687
running-frontview   max|delta| = 1.082610
ballerina-spin      max|delta| = 1.162301
zoom-in             max|delta| = 1.240014
```

Input convention, if you ever call the UNet directly: `sample` is
`(B, C, T, H, W)` and `encoder_hidden_states` must already be expanded to batch
`B*T` — `AnimateDiffPipeline` does
`prompt_embeds.repeat_interleave(num_frames, dim=0)` before calling the UNet.
Passing a single `(1, 77, 768)` raises
`size of tensor a (16384) must match the size of tensor b (1024)`.

### 9.5 Free evidence: the author's own renders

`models/subject_lora/reference/`. The `walking-frontview` render (1152×768,
63 frames) shows a figure walking toward camera with **articulated** motion —
legs alternating, arms swinging, hair and dress moving — against a background
that is essentially locked. Its temporal-variance heat map
(`output/subject_test/reference/ref_walking-frontview_heat.png`) is a single
bright vertical blob on the figure. **That is the separable foreground the
compositing idea was rejected for lacking.** loc10 = 0.783 against a uniform
0.10.

Caveat, so this is not over-claimed: that render is the *author's*, on an
unknown base model, on a low-detail studio background that is unusually easy to
hold static, and the mp4 is compressed. It proves the LoRA is *capable* of a
localized subject. It does not prove it reproduces here, on a textured forest
prompt, on DreamShaper. That is what `fig_none_512` vs `fig_walk_512` measures.

### 9.6 The director: choosing a motion on fit, cohesiveness, continuity

`director.choose()` scores every eligible motion per shot:

- **fit** — a hard gate plus a soft score. A subject LoRA is refused when there
  is no person in the shot, when the framing cannot hold a full body (a walking
  LoRA on a face close-up), or when its *measured* localisation is below
  `LOC10_TRUST` — i.e. it was observed to be global drift. CLIP supplies a soft
  affinity and an explicit content lexicon covers fine distinctions CLIP misses.
- **cohesiveness** — motion energy matched against what the shot's framing scale
  implies, plus a budget on how much of the film moves at all.
- **continuity** — R1–R5 from `edit_planner`, plus R6, the 180-degree rule
  generalised: two shots travelling along the same degree of freedom with
  opposite sign cannot sit across a cut. `walking-frontview` after
  `walk-backview` is refused for the same reason `pan-left` after `pan-right` is.

`none` is a first-class choice with a healthy prior fit (`NONE_FIT = 0.62`), so a
move has to earn its place — the handoff's rule that AnimateDiff already drifts.

**Two corrections worth keeping:**

- `mag` (mean frame-to-frame change) is **not** directorial energy. A `none` shot
  scores *high* on it precisely because global drift is large, while a subject
  LoRA locks the background and scores low. Min-maxing across kinds ranks "nothing
  moves" as the most energetic option. Measured energy is only compared *within*
  a kind; `none` uses a prior.
- The CLIP person gate is a **hybrid**. `"she runs now, full body, urgent,
  medium shot, cinematic, filmic grain"` scored p=0.05 — the style words dominate
  the embedding. The lexicon catches it, the report prints
  `person p=0.90 [lexicon (CLIP said 0.05)]`, and the choice is arguable.

Fit gate sanity check: 5/5 correct on `python director.py`.

**The director is greedy and strictly causal — it holds no representation of the
whole film.** It receives the entire planned shot list (`make_film.py:443`) but
never indexes forward: no lookahead, no global energy arc, no duration budget, no
palette/look state, and it never reads `shot.scene` or `shot.seconds`. It walks
left to right and remembers exactly three things about the decided prefix:

| memory | scope | used for |
|---|---|---|
| `out[-1]` | the immediately previous shot only | the 180° gate, the axis-continuity term `k`, and carrying a subject's direction of travel across the cut |
| trailing run of movers + `budget` | `reversed(out)`, and the running fraction of moving shots (only once `len(out) >= 3`) | R3 / `MAX_MOVING_RUN`, and the fatigue penalty |
| `any(s.subject == shot.subject for s in out)` | the **whole prefix** | the recurring subject raises continuity to 0.9 — the only non-windowed memory it has |

Everything else global is about the *vocabulary*, not the film: the candidate
list, the measured evidence, and `kind_scales` computed once over all motions.
`same_dof_as` requires `axis != "none"`, so a previous `none` shot cannot veto
anything.

**Consequence, and it is a real limitation**: a locally good choice can constrain a
later one. The gate compares only against the immediately previous shot, so
committing a zoom-in on shot 2 can veto a pan-right on shot 3 without ever
weighing the alternative assignment — and the director cannot reserve a move for
the shot that most needs one. Where a whole-film view does exist: `edit_planner`
(the rhythm curve over `n` against a duration target, R1/R2/R6 repairs,
`energy_profile`), `model_select` (the full list, `MAX_SAME_RUN`, scene
boundaries), and `looks`/`make_film` (one palette per scene). The director was the
only stage that was purely left-to-right.

**FIXED — the director now optimises the whole film, not one shot at a time.**
`mode="global"` (the default) is an exact dynamic program over the sequence whose
state is `(previous motion, run of movers, movers used)`. Three things fall out:
the moving budget goes to the shots that most deserve it wherever they are, the
180° rule and R3 become *exact* constraints instead of soft penalties, and
continuity across each cut is optimised rather than merely respected for the
previous choice. The state is ~180 wide and the shot list is 4–12 long, so the
exact answer costs milliseconds. `mode="greedy"` keeps the shipped behaviour,
frozen, as the A/B control — deliberately not refactored, because that is what it
is a control *for*.

On a 6-shot list where several shots hinted at a move (so they competed for the
budget), scored with the same objective function:

| | greedy (shipped) | global (whole film) |
|---|---|---|
| movers | **4 of 6** (budget 2) | **2 of 6** (the cap) |
| rule breaks | **1** — 3 movers in a row (R3) | 0 |
| objective, as found | 4.132 | 4.067 |
| objective **after `edit_planner`'s repair** | 4.044 | **4.067** |
| repairs that changed a motion choice | 1 | **0** |

**The greedy path has no mover cap at all** — only a fatigue penalty that does not
bite until more than 60% of shots move. So on that list it moved 4 shots of 6 and
produced three movers in a row, and the pipeline only stayed legal because
`edit_planner` silently undid one of its choices. The global path is the first
place the brief's "at most two shots may move" is actually enforced, and its lower
raw objective is the *price of obeying the budget*, not a worse answer — hence the
post-repair row, which is the like-for-like comparison. Where greedy was already
optimal it agrees exactly (the demo shot list: identical on all 6 shots, objective
3.679 both), so this is a fix with a measured scope rather than a rewrite that
claims to improve everything.

`--director global|greedy` on `make_film.py`; `mode=` and `max_movers=` on
`director.choose()`; `director.compare_modes()` runs both on the same list and
prints the table above.

**PROGRESSION: the term's SHAPE was wrong, and the fix was not to delete it.** The
fair follow-up question was whether the long view should be about the film *going
somewhere* rather than only about where a bounded budget is spent. The first
attempt scored `target * energy` — a **reward for energy** — and looked inert: the
same choices with it on and off on all four real lists. The reason is that it
pointed the wrong way. A move carries more energy than a still shot whatever the
target says, so rewarding energy *rewards moving early*, which is the opposite of
progression. It fought itself across the film.

**The term has to be a MATCH**, `1 - |target_at_this_shot - energy_of(choice)|`,
which penalises a move in the calm opening instead of paying for one and pays for
one at the peak. Weight 0.25, added on the global path only so the frozen greedy
control is untouched. On six deliberately interchangeable shots — no camera word
in any prompt, no recurring figure, so fit is identical everywhere and placement
is decided purely by the arc:

| | movers | energy profile | arc match | base obj |
|---|---|---|---|---|
| greedy (shipped) | 0/2 | 333333 | 0.70 | 3.717 |
| global, budget only | 0/2 | 333333 | 0.70 | 3.717 |
| **global + progression** | **1/2** | **333733** | **0.78** | 3.639 |

Without the arc the optimiser finds *no reason to move at all* — the film is a
flatline. With it, the one move it spends lands on **shot 4 of 6, exactly the
peak** (target energy 1.00), taking the highest-energy motion available. The base
objective *drops* (3.717 → 3.639): that is the visible price of spending fit on a
move the shots did not individually ask for, which is the trade progression is.

**Honest scope.** On all three real shot lists it still changes nothing, because
their fits already place the moves in the right half of the film (shots 3–6 of 6).
It only decides where placement is genuinely open. And where a shot's *declared*
energy contradicts the arc, the writer wins: `TEST_SHOTS` declares energy 0.7 on
shot 1, and the cohesiveness term then puts a move in the calm opening anyway.
That is correct — the shot's stated intent outranks a house curve — but it means
the arc is a tie-breaker, not an override.

**So progression is asked for in two places.** The writer's brief (`story.SYSTEM`)
now requires energy to rise, peak and fall, never highest on the first or last
shot, with the last shot settling. And the director spends the moves inside that
shape. The diagnostic that came out of this paid for itself immediately: the
ensemble's own output, written with the OLD brief, declared energies **231421**
against a target of **247963** — a peak at shot 4 of 6, i.e. 67% of the way in,
exactly where the target puts it. The writer was already doing this and nobody had
looked. The hand-written lists in `make_film.py` are the ones with no arc: they
carry no `energy` values, so every shot falls back to the energy its scale implies.

### 9.7 What is still unproven

Everything in 9.5 is the author's render. On this machine the decisive numbers
are the `fig_none_512` (baseline) vs `fig_walk_512` pair. **Read
`output/subject_test/metrics.json` before making any claim.** Note that the
handoff's original 7.7 / 14.2 / 31.7 figures came from a different harness, so
verdicts are computed against *this run's own* baseline via
`localisation_verdict`, never against the older numbers.

### 9.7.1 RESULT: the subject LoRAs work, but they do NOT revive compositing

Full A/B, same prompt family, same seed, same 4 steps, DreamShaper at 512 unless
noted (`output/subject_test/comparison.png`, `metrics.json`):

| clip | mag | loc10 | global_share | coherence | align |
|---|---|---|---|---|---|
| none (baseline) | 3.73 | 0.718 | 0.49 | 0.0138 | 0.3247 |
| walking-frontview 512 | 11.47 | 0.426 | 0.47 | 0.0455 | 0.3199 |
| walking-frontview 576x384 (native) | 12.89 | 0.491 | 0.49 | 0.0534 | 0.3354 |
| running-frontview 512 | 9.39 | 0.469 | 0.42 | 0.0576 | **0.3662** |
| ballerina-spin 512 | **49.27** | 0.187 | 0.56 | 0.0409 | 0.2261 |
| leaf macro + walking LoRA (misfit test) | 16.45 | 0.257 | 0.53 | 0.0864 | 0.2826 |
| *author's walking render* | *2.24* | *0.783* | ***0.18*** | *0.0761* | *0.2061* |

**Read `global_share`, not `loc10`.** `loc10` is confounded by `mag`: the
baseline scored 0.718 -- higher than every LoRA clip -- purely because its small
change (mag 3.73) was concentrated. Naively that reads as "the LoRA made
localisation worse", which is wrong; adding motion spreads the variance. This
cost one wrong verdict before it was caught. `localisation_verdict` now uses
`global_share` (spatially uniform change, independent of magnitude) and carries
`loc10_confounded_by_mag: True`. The separation threshold is
`LOCALISED_GLOBAL_RATIO = 0.7`.

On `global_share` nothing gets below 0.86 (running); the author's render is 0.18.
**No subject LoRA produces a separable foreground on a textured scene here.**

What *is* true, from `comparison.png`:

- **walking-frontview works.** A coherent woman in a long coat walking toward
  camera, articulated -- legs, coat, hair all changing. Same for
  **running-frontview**, which also has the best prompt alignment of any clip
  (0.3662). These are real subject-motion generators.
- **ballerina-spin is unusable at these settings.** mag 49.27 (13x baseline),
  loc10 0.187, align 0.2261 (worst), and the frames are abstract colour chaos.
- **The fit gate is validated.** `leaf_walk_512` puts the walking LoRA on a leaf
  macro with no person in it and gets 4.4x the baseline's motion -- a subject
  LoRA does inject motion where no subject exists, which is exactly why
  `director.py` refuses it.

**So the precondition for compositing is background stability, not the LoRA.**
The author's render worked because his scene was a near-empty studio set with
almost no total motion. Compositing stays viable for shots deliberately framed
against low-detail backgrounds (fog, sky, flat walls) -- which is why the film
test's subject shot is framed against fog.

**Open confound, not yet closed:** all of this ran at **4 steps with the Lightning
distillation**. A distilled 4-step schedule can under-express an adapter trained
on a full-step schedule, and the ballerina collapse is at least consistent with
that. Testing one clip at 12-16 steps without Lightning would settle it, at 3-4x
the cost (~30-40 min for one clip). Until then, "the LoRAs cannot separate a
subject" is established *at 4-step Lightning*, and that qualifier belongs on any
claim.

### 9.7.3 THE LEAD IS CLOSED: fog does not rescue separation

The last hypothesis was that a low-detail background (fog, the one thing
AnimateDiff holds well) would let the figure separate where a textured street did
not. Two clips, same prompt, same seed, only the LoRA differing, no RAFT and no
shake:

| condition | scene | LoRA | mag | loc10 | **global_share** | align |
|---|---|---|---|---|---|---|
| `fig_none_512` | street | none | 3.73 | 0.718 | **0.49** | 0.3247 |
| `fig_walk_512` | street | walking | 11.47 | 0.426 | **0.47** | 0.3199 |
| `fog_none_512` | fog | none | 3.85 | 0.557 | **0.52** | 0.3767 |
| `fog_walk_512` | fog | walking | 13.92 | 0.313 | **0.45** | 0.3683 |
| *author's studio render* | *studio* | *walking* | *2.24* | *0.783* | ***0.18*** | *0.2061* |

Two contrasts, which is why the experiment needs four clips and not two:

- **Within the fog scene** (isolates the LoRA): 0.52 -> 0.45, a **13%** reduction.
  The bar is 30% (`LOCALISED_GLOBAL_RATIO = 0.7`); the author achieved 50%.
- **Within the no-LoRA condition** (isolates the scene): 0.49 -> 0.52. A fog
  background is **not** inherently more stable than a textured street.

The direction is right -- fog helps more than the street did (13% vs 4%) -- but it
is nowhere near enough. `fog_walk_512`'s heat map shows tree and fog variance
across the whole frame with the figure barely brighter than its surroundings, and
the strip shows the trees themselves shifting between frames. Metric and eyes
agree, which is worth stating explicitly given how often they have not today.

**Verdict: the subject-motion LoRAs are motion tools, not separation tools.**
Compositing is not revived. Do not spend more render time on this hypothesis at
4-step Lightning.

What the lead *did* pay off, and it is not nothing: `walking-frontview` and
`running-frontview` produce coherent, articulated human motion -- legs, coat,
hair, stride -- which AnimateDiff alone does not. They are worth using for the
motion they create. Note also that a drifting background is a *defect* only for
compositing; as a film shot, a figure walking through shifting fog looks
intentional and good.

Remaining unknowns, in priority order: (1) the 4-step Lightning confound above;
(2) `fog_walk_512` is one seed and one prompt per scene, so none of these numbers
are distributions; (3) the two-crane/truck camera LoRAs are still not downloaded,
so `director.py` can never choose them.

### 9.7.4 The Lightning confound is CLOSED -- and it inverts the hypothesis

Ran the fog pair again at 12 steps, guidance 6.0, **Lightning weights not loaded**
(guarded: `--no-lightning` refuses without both `--steps` and `--guidance`,
because an un-distilled 4-step run is the 256px class of misconfiguration).

| clip | config | mag | loc10 | global_share | align |
|---|---|---|---|---|---|
| `fog_none_512` | 4-step Lightning | 3.85 | 0.557 | 0.52 | 0.3767 |
| `fog_walk_512` | 4-step Lightning | 13.92 | 0.313 | 0.45 | 0.3683 |
| `fog_none_full_512` | 12-step, no Lightning | 5.65 | 0.425 | 0.57 | 0.3699 |
| `fog_walk_full_512` | 12-step, no Lightning | 5.80 | 0.445 | 0.56 | 0.3680 |

Read **down each column** -- full-step sampling changes the no-LoRA baseline too
(global 0.52 -> 0.57 with no LoRA at all), so the rows are not comparable.

```
4-step Lightning      LoRA effect: 0.52 -> 0.45   = 13% reduction
12-step no Lightning  LoRA effect: 0.57 -> 0.56   =  1% reduction
separable needs >= 30%
```

**The hypothesis was that Lightning suppresses the adapter. The opposite is
true.** Without Lightning the LoRA does almost nothing measurable: `mag` rises
only 5.65 -> 5.80 (+3%) where under Lightning it rose 3.85 -> 13.92 (+262%), and
`global_share` moves 0.01. So the 4-step Lightning schedule is the only
configuration in which this adapter does anything substantial at all -- and even
there it fails the 30% bar. **The lead is closed harder, not softer.**

Preflight corroborates it: `walking-frontview` `max|delta|` on the UNet is
1.146687 with Lightning loaded and 0.252381 without -- a 4.5x difference. Not
conclusive on its own (the UNet is a different function in each configuration, so
the delta acts on different weights), but it points the same way as the clips.

Two things learned about the *instruments*, both worth keeping:

- **`heat.png` is max-normalised and therefore lies about spread.** It renders
  `fog_walk_full_512` as a tight bright blob on black, which reads as "fully
  separated". The metric says loc10 0.445 and global_share 0.56 -- i.e. the
  variance is moderately spread and the LoRA *lowered* concentration versus its
  own control (0.557). Extreme dynamic range makes diffuse low-level variance
  render black. Trust the numbers over the picture, or the picture over nothing.
- **`global_share` is not scale-free when total motion is small.** A faint uniform
  fog-brightness drift over a low-`mag` clip can dominate `mean|d|` and make the
  change field look uniform even when the variance sits on the subject. That is
  why `loc10` (0.557 -> 0.445) and `global_share` (0.57 -> 0.56) disagree in
  direction here. Neither is a clean separation measure on its own; both agree the
  LoRA does not help.

**Nothing is left to test on this lead.** Do not re-run it at other step counts,
resolutions, or weights without a new hypothesis: 4-step Lightning, 12-step
full-schedule, 512x512, 576x384 native, fog and textured scenes, three subject
LoRAs, and the fit-gate misfit case have all been measured, and none separate.

### 9.7.2 The watermark gate was wrong

Handoff 4.7 recorded the SHUTTERSTOCK watermark as the reason to abandon sd15,
and `model_select.py` hard-gated it out on that basis. `watermark_check.py`
amplifies the high-frequency residual of the exact frames 4.7 cites
(`watermark_ab/none/f003.png`, byte-identical to
`checkpoint_ab/SD15_base_none/f003.png`) and finds **no legible letterforms**.
sd15 does score 2.3x dreamshaper8 on structured high-frequency content (0.0027 vs
0.0012), but that is foliage detail, not text.

Caveat in the other direction, so this is not over-claimed: 16 frames of one
forest prompt do not prove sd15 never emits a watermark -- it is prompt-dependent
in the training data. So the gate became a per-render check plus a 15% fit
penalty (`watermark_check.text_likeness` on real output), not a blanket refusal.
With sd15 back in the pool the selector immediately uses both checkpoints.

### 9.8 Choosing the model, not just the motion

`model_select.py` answers "can the system pick sd15 for some shots and
dreamshaper8 for others and still read as one film?". It scores fit / cohesion /
continuity like `director.py`, but the cohesion term is not asserted -- it is the
**measured** seam between the two checkpoints' actual renders of the same prompt
and seed.

Two hard gates no fit score can buy: a checkpoint that leaks the SHUTTERSTOCK
watermark (`sd15`, `sd15-lcm`) is refused outright, and a model that cannot make
motion cannot serve a shot that needs it (`svd-xt`, 0.10 motion measured).

`grade_match.py` is what makes intercutting viable at all. Same prompt, same
seed, two checkpoints:

```
sd15           L 130.5±52.7  chroma 13.8  grain 13.8
dreamshaper8   L 113.6±37.5  chroma  6.5  grain  4.4
seam across the cut: 7.76   (1.0 = invisible, >3 = a different film)
after match_to():    1.70   (raw 1.1792 -> 0.2586, 4.56x better)
```

So the raw intercut of those two checkpoints is not a subtle difference to be
argued about, it is 7.8x the within-shot frame-to-frame variation. With grade
matching the selector's switch cost drops from 2.00 to 0.44 and it starts
swapping (`dreamshaper8 → svd-xt` for the macro, `t2v-zero` for the high-motion
shot).

Three things to keep honest about it:

- Only the **sd15 ↔ dreamshaper8** seam is measured. Other pairs fall back to a
  *nominal* 2.00, reported as such. Do not quote the nominal as a measurement.
- The 4.56x reduction is measured on that one pair; `--reduction` extrapolates it
  to other model pairs. Measuring each pair means rendering both models on the
  same prompt and seed, which is the honest way to extend it.
- `match_to` is per-frame, not per-clip. A clip-level transfer cannot fix
  flicker, and sd15 flickers badly: within-clip spread 0.267 vs dreamshaper8's
  0.037. That ratio is itself an argument for DreamShaper.

Grain was 96% of the residual seam until `_match_grain` learned to soften as well
as add: noise can only *raise* grain, so the sd15 clip -- grainier than the
target once its contrast was matched -- overshot and could never come down.

### 9.9 AUDIT: the model-intercutting claim is NOT established

`python grade_match.py --calibrate` builds a ladder of seams whose ordering is
already known. It **fails**:

```
within one clip (noise floor)  : 2.31
same checkpoint, other content : 5.10     <- dreamshaper8 no-lora vs zoom-in
different checkpoint           : 5.58     <- sd15 vs dreamshaper8, zoom-in
required: cross > 2.0x same-model (need > 10.20)
```

A same-checkpoint pair scores 5.10 and a cross-checkpoint pair 5.58 -- a 9%
margin. **The seam metric cannot separate a change of look from a change of
content.** A first version of the check passed this on `min(cross) > max(same)`;
requiring a real margin fails it. Do not re-loosen that check to make it pass.

What follows:

- The 4.56x reduction (seam 7.76 -> 1.70) was measured with prompt, seed **and
  LoRA held fixed** -- only the checkpoint differed. That is the one condition a
  real intercut never has. It is a valid result about aligning two renders *of
  the same content*; it is **not** evidence that intercutting differently-framed
  shots from different checkpoints works.
- Therefore `model_select.py`'s decisions rest on cohesion numbers that are only
  conditionally meaningful, and on `motion`/`detail` values that are invented
  (see the provenance block in that file).
- The "3 switches without grade matching vs 5 with" comparison was partly driven
  by a *nominal* 2.00 switch cost for t2v-zero and svd-xt, which have no renders
  to measure. Treat that comparison as illustrative, not as a result.

To make it real, in order:

1. Make the signature content-robust -- simplest correct version: to know the
   seam between model A and B for shot S, render S with both and measure. A
   cheaper approximation is a per-model look centroid averaged over several
   diverse prompts, so the content component averages out.
2. Wire it up: a multi-base `Generator` (reload per switch, ~1 min), a `--models`
   flag, and `grade_match.match_to` as a stage before assembly. **Neither
   `model_select` nor `grade_match` is imported by `make_film.py` today -- the
   selector decides and nothing acts.**
3. Only then run the end-to-end test: 4 shots, 2 per model, graded and cut
   together, seam-measured *in the assembled film*.

Separately: the subject-motion A/B (section 9.7) is 1 of 6 clips done
(`fig_none_512`, loc10 0.718, global_share 0.49). Resume with
`python subject_motion_test.py --run`. That remains the higher-value lead; it is
the stated one, and it needs no new metric work.

### 9.10 Do not trust wall-clock timings across a sleep

A six-clip run appeared to stall: 632 s of CPU across 103 minutes of wall time,
the working set collapsed from 1.4 GB to 368 MB, and the log went silent. That
looked exactly like a deadlock in the new feed-forward wrappers and was very
nearly recorded as one. The cause was the **host sleeping** while the run was in
progress. Symptoms of a suspended process and a deadlocked one are similar; check
whether the machine slept before blaming the code. The generation itself was
fine -- a single clip is ~130 s/step, ~10 min for 16 frames at 4 steps, exactly
as section 3 predicted.

Add `SM_TEST_PROGRESS=1` to `subject_motion_test.py` for per-step timing, so a
stall is visible within one step instead of one clip.

---

## 10. Variety, not cohesion -- the objective was wrong

### 10.1 The reframe
The user's goal, stated directly: they want **both** sd15 and dreamshaper8 *because
the variety is the point*. They do not have a distinct style yet and cannot
out-resolve other AI video, so deliberate look variation is the thing that holds
attention and differentiates the output. The metric's job is **not** to minimise
difference; it is to keep a difference legible as a **choice** rather than a
**mistake**.

Section 9.9's selector maximised cohesion and therefore concluded "do not
intercut". That was the wrong question. This section supersedes it.

### 10.2 A look has two halves, and they want opposite treatment

```
technical   exposure, contrast, grain   -> a jump reads as an ERROR. MATCH IT.
stylistic   palette, saturation         -> a difference reads as STYLE. KEEP IT.
```

`grade_match.match_technical` normalises only the technical half. Measured on the
checkpoint A/B, with thresholds taken from the **measured noise floor** (technical
1.43, stylistic 0.45 -- two halves of the *same* clip, where no look change
exists) rather than from taste:

```
treatment                        tech  /floor  style  /floor   verdict
as generated                    11.09   7.7x    3.02   6.7x    ERROR
match_to (flatten both halves)   1.87   1.3x    0.16   0.4x    FLAT - variety destroyed
match_technical (baseline only)  1.85   1.3x    3.00   6.7x    LEGIBLE STYLE
```

**`match_to` -- section 9.8's "fix" -- is the wrong tool for this goal.** It
flattens the palette so hard the two clips end up *less* different than two
halves of one clip (0.4x the floor). `match_technical` is the right one: one
photographic baseline, two surviving palettes. See
`output/film/grade_match/variety.png`, colour 3 vs colour 2.

Grain must be matched **upward** (to the grainiest shot), not to the mean.
Blurring to reduce grain also softens contrast and undoes the contrast match; a
mean target left a 2x-floor technical residual. `technical_target` takes the max
for grain and the mean for everything else. Trade-off to be aware of: the cleaner
model inherits the noisier model's grain.

### 10.3 A third instance of the same mistake

Thresholds picked by taste, not measured: the seam cut-offs in 9.8
(`1.0`/`3.0`), `COHESION_TOLERANCE`, and this section's first `TECH_TOL=1.5` /
`STYLE_MIN=1.5`. The last one declared *every* treatment a failure on an
uncalibrated scale -- the numbers were right and the thresholds were fiction.
**Derive every threshold from the within-clip noise floor.** It has now caught
this three times.

### 10.4 The real blocker: the candidate pool is one model deep

`model_select.py` now *rewards* variety (weights fit .40 / technical .15 /
variety .30 / rhythm .15; a switch is only credited at a scene boundary, and
`MAX_SAME_RUN=3` charges monotony). It still switches once. The gates, per shot:

| shot | survivors | gated out |
|---|---|---|
| 1,3,4,5 (wide/medium) | dreamshaper8, t2v-zero | sd15 + sd15-lcm (watermark), svd-xt (0.10 motion) |
| 2 (macro) | dreamshaper8, t2v-zero, svd-xt | sd15 + sd15-lcm (watermark) |
| 6 (energy 0.70) | **t2v-zero only** | dreamshaper8 (0.50 motion < 0.70 needed), svd-xt, sd15s |

So the pool is `{dreamshaper8, t2v-zero}`, and t2v-zero fits badly (0.36) -- one
usable model. **The variety ceiling is the candidate pool, not the metric.** No
amount of metric work fixes that, because sd15 is excluded by a hard quality gate:
it leaks the SHUTTERSTOCK watermark. Wanting "both models" and getting both are
different things.

Two ways forward, and they are not exclusive:

1. **Variety from grading, not from checkpoints.** One checkpoint, N deliberate
   per-scene palettes, with `match_technical` holding a shared baseline. Free,
   deterministic, and *controllable* -- you dial a signature look instead of
   hoping two checkpoints differ pleasingly. This is probably the strongest
   immediate route to "visually distinct".
2. **A second CLEAN checkpoint with a different palette.** The requirement is not
   sd15 specifically, it is a second look that does not leak a watermark. The
   SD1.5 family is full of candidates (RealisticVision, EpicRealism, RevAnimated,
   Dreamlike, OpenJourney, Deliberate...). Any one of them, verified
   watermark-free by the same cheap checkpoint A/B (2 clips, ~20 min), unlocks
   real model variety.

Note also: shot 6 rejected *dreamshaper8* for insufficient motion. AnimateDiff's
motion ceiling is real and measurable, and it is the same limitation the
subject-motion LoRAs in section 9 exist to attack.

### 10.6 The palette stage -- `looks.py` (approved route)

One checkpoint, N deliberate per-scene palettes, `match_technical` holding the
shared baseline. Built and validated, no rendering required:

```
palette floor (same clip, no look change): 0.33

neutral   -> warm-gold       tech 0.07  style 1.80   LEGIBLE
warm-gold -> rich-green      tech 0.13  style 1.64   LEGIBLE
rich-green-> cold-steel      tech 0.01  style 2.51   LEGIBLE
cold-steel-> amber-dusk      tech 0.07  style 4.11   LEGIBLE
amber-dusk-> bleach          tech 0.27  style 3.53   LEGIBLE
bleach    -> cyan-night      tech 0.33  style 1.72   LEGIBLE

6/6 adjacent transitions legible; technical baseline held 6/6
```

Seven looks (`neutral`, `warm-gold`, `cold-steel`, `bleach`, `rich-green`,
`amber-dusk`, `cyan-night`), applied only to the stylistic half so a look can
never introduce the technical seam that reads as an error. See
`output/film/looks/looks.png`. `ARC` is a suggested progression; a film may name
a look per scene instead.

Two units traps hit while building this, both instances of handoff 10.3:

- The first look set was ~6x too strong (stylistic 19-49 against a band of
  1.5-8). A look spec that "seems reasonable" is not; measure it.
- `palette_floor` must return a `decomp` value, not a raw `within_spread`.
  Passing the raw spread mixed two different quantities and inflated every
  distance about 4x. If a distance looks wrong, check the normaliser first.

### 10.7 t2v-zero demoted, and why that matters

The user saw bogus t2v-zero output and asked that it be excluded until
demonstrated. It is now `proven=False` and out of the pool unless
`--allow-unproven` is passed. Caveat in its favour, recorded so it is not written
off wrongly: handoff 4.1 records the earlier "abstract mush" verdict as a
**256px artifact**, so t2v-zero is unknown rather than known-bad.

Consequence worth knowing: with t2v-zero out, the demo's 0.70-energy shot has **no
eligible model** -- every proven model sits at motion 0.50. That is
AnimateDiff's motion ceiling appearing as a hard constraint rather than a scoring
accident, and it is the same ceiling the subject-motion LoRAs exist to attack.
`choose()` now prints this case loudly instead of falling back silently.

### 10.8 Wiring -- DONE, locked into `make_film.py`

Everything decides *and* acts now. `make_film.py` gained:

```
--shots test|figure|landscape     test = 4 shots, 2 scenes, exercises the lot
--models off|auto|a,b             auto runs model_select.py per shot for variety
--looks  off|auto|neutral,gold    auto walks looks.ARC, one palette per scene
--look-post off|technical         technical = shared exposure/contrast/grain
--allow-unproven                  let unproven models into the pool
```

**Checkpoint hot-swap.** `Generator.ensure_model()` rebuilds the pipeline on the
base a shot needs and releases the previous one. Two SD1.5 UNets are NOT held at
once: 15.3 GB total, one fp32 UNet plus motion adapter is already several GB, and
holding both would thrash. Cost is ~1 minute per switch, against ~10 minutes of
sampling per shot, and a film only switches at scene boundaries.

**Order of the grading stages matters.** `match_technical` (shared baseline)
runs *before* `apply_look` (palette). The other order would let the baseline
match partly undo the palette.

**`renderable` flag added to `Model`.** `model_select.choose(renderable_only=True)`
is the default on the film path, so the selector cannot propose svd-xt or
t2v-zero -- a plan nothing can execute. Pass False to see the wider field.

Verified with `--shots test --models auto --looks auto --dry-run`:

```
[models] shot 1: dreamshaper8  variety 1.00  = 0.970
[models] shot 2: dreamshaper8  variety 0.40  = 0.780
[models] shot 3: sd15          variety 1.00  = 0.925  <- SWITCH (look shift 7.76)
[models] shot 4: sd15          variety 0.40  = 0.726
models: ['dreamshaper8','dreamshaper8','sd15','sd15']   switches: 1
walk  -> look neutral      water -> look warm-gold
```

One switch, on the scene boundary, both checkpoints used, one palette per scene.

**A bug worth remembering.** The first version scored *staying* on a model as
`variety = 1.0`, so switching had no advantage and the selector never switched
until monotony forced it at shot 4 -- the objective cancelled itself. Staying now
scores `STAY_VARIETY = 0.4` (it contributes nothing new) and a motivated switch
scores 1.0. If a selector keeps choosing the status quo, check whether the status
quo is being scored as a win.

The footage test (`--shots test --models auto --looks auto --name test_variety`)
is queued behind the subject-motion A/B, serialized on purpose: two pipelines at
once would exceed RAM on this box.

## 10.9 The footage test -- what worked, and two measurement traps

Ran `--shots test --direct --models auto --looks auto`, 4 shots, 2 scenes,
175 frames, 10.9 s. `output/film/test_variety/` (`shots.png`, `film.mp4`).

```
shot 1  walking-frontview  dreamshaper8  neutral     motion 13.98   567s
shot 2  none               dreamshaper8  neutral     motion  4.12   519s
        [base] releasing dreamshaper8, loading sd15 (~1 min)
shot 3  zoom-in            sd15          warm-gold   motion 19.85   551s
shot 4  none               sd15          warm-gold   motion 19.67   523s
```

**It works end to end.** The whole stack ran in one pass: director motion choice,
checkpoint hot-swap, per-scene palette, RAFT, handheld shake, hard-cut assembly.
The film reads coherently -- a figure walking up a foggy forest road, a fog
establishing shot, a waterfall, mossy boulders -- and the look shift lands on the
same cut as the change of environment, so it reads as motivated rather than as a
mistake.

**The checkpoint swap is verified, not assumed.** `ensure_model` rebuilt the
pipeline on sd15 and the preflight re-ran there, proving all 14 adapters live
again on the new base with *different* deltas (zoom-in 1.240 on dreamshaper8 vs
0.912 on sd15; ballerina-spin 1.162 vs 1.403). That is what a real re-attachment
looks like, and it is the check that the earlier false-negative session lacked.

### Trap 1: the content confound, reintroduced

`make_film` originally printed cross-shot style verdicts from `decomp`. They were
fiction:

```
neutral -> neutral       tech 0.81  style  2.87   (no look change at all)
neutral -> warm-gold     tech 0.93  style 40.74   (the real look change)
warm-gold -> warm-gold   tech 1.16  style 10.28   (no look change at all)
```

Two cuts with *nothing deliberate changed* measured 2.87 and 10.28 apart. The
same `neutral -> warm-gold` palette shift measured **1.80 on held content** in
`looks.py` and **40.74 across content** here -- a 22x inflation from content
alone. This is exactly handoff 9.9's confound, and it is easy to reintroduce the
moment you measure across two different shots. `decomp` is only valid between
renderings of the *same* content. The film now reports the palette *assignment*
plus a per-shot technical offset against the shared target (valid, because every
shot is compared to the same reference), and refuses to report a cross-shot style
verdict. Look legibility stays with `looks.py`.

### Trap 2: the assembled film cannot answer a model question

Measuring separation on the assembled frames gave `global_share` 0.68 / 0.62 /
0.55 / 0.76 -- including 0.62 for a near-static fog road. The handheld shake is
global translational motion applied to every frame, so it *is* global drift and it
swamps the metric; RAFT interpolation shifts it further. **The fog hypothesis is
therefore UNTESTED.** The visual impression (shot 1's background looks stable) is
not evidence.

`make_film.py` now writes the raw clip for every shot to
`<name>/raw/shotNN/` before RAFT and shake, so a model-level question stays
answerable from the footage. The first test predates this and has no raw clips.

**Next experiment — TWO clips, not one (`fog_none_512`, `fog_walk_512`):** the
fog-framed prompt with and without the walking LoRA, no RAFT and no shake, ~12
min. The control matters: comparing fog+walk against the *street* baseline would
change the scene and the LoRA at once and be uninterpretable. The pair isolates
the LoRA inside one scene; the existing street pair isolates the scene.

```
fig_none_512   (street, no LoRA)    global 0.49     <- existing
fig_walk_512   (street, walking)    global 0.47     <- existing
fog_none_512   (fog, no LoRA)       global  ?       <- control
fog_walk_512   (fog, walking)       global  ?       <- the question
author's studio render              global 0.18     <- target
```

Read it as two contrasts, not one number:

- **within the fog scene** (fog_walk vs fog_none): does the LoRA stop dragging the
  background? If fog_walk ≈ fog_none the background still moves and the lead is
  closed.
- **within the no-LoRA condition** (fog_none vs fig_none, 0.49): is a low-detail
  scene inherently more stable?

`fog_walk_512` is deliberately NOT added to `EVIDENCE_MAP` in `motion_vocab.py`.
Averaging it into the walking LoRA's evidence would erase the whole point, which
is that separability is scene-dependent.

## 11. Sampling: 4-step Lightning is BOTH faster and better

The obvious assumption -- more steps, better quality -- is **false for this
pipeline**, and getting it wrong costs 4x the render time for a worse film.

`output/compare/fog_walk_512__VS__fog_walk_full_512.png` (and the matching `.gif`)
puts the same prompt, seed and LoRA through both schedules side by side:

| | 4-step Lightning, guidance 1.0 | 12-step full schedule, guidance 6.0 |
|---|---|---|
| time | **~9 min/clip** | ~37 min/clip |
| align | 0.3683 | 0.3680 |
| look | sharp trees, textured bark, wet-road reflections, a clearly rendered figure | soft smears, uniform haze, the figure a featureless blob |

The Lightning distillation was trained *for this motion adapter*. Running the base
SD1.5 schedule is off-distribution for it: `align` is identical, so no metric
catches the difference -- only the picture does. **Do not "improve quality" by
adding steps.** The 4-step configuration is the production setting, and it also
happens to be the fast one.

Consequence for the subject-motion work: `fog_walk_512` (4-step) is the *real*
configuration, and the 12-step runs in 9.7.4 are the off-distribution control.

### 11.1 Judging quality: build the side-by-side, do not argue about metrics

No metric in this project has ever settled a quality question. `align` called
4-step and 12-step identical (0.3683 vs 0.3680); `global_share` and `loc10`
disagreed in direction on the same pair; `heat.png` is max-normalised and renders
a diffuse clip as a blob. Every time the decision mattered, someone had to look.

`compare_clips.py` makes looking cheap -- an animated side-by-side (motion is the
question, and strips cannot show it) plus a metrics panel:

```
python compare_clips.py --pairs "fog_none_512:fog_walk_512,fog_walk_512:fog_walk_full_512"
```

### 11.2 Choosing the text model: repairs, not tokens/sec

`llm_bench.py` measured tokens/sec, which is the wrong criterion. `llm_scorecard.py`
measures what the writer costs the rest of the system. Two metric bugs were fixed
getting there, both worth remembering:

- Counting **R4 rhythm rewrites** as writer errors made every model look equally
  bad -- but the planner owns timing by design and the writer is told so. Split
  into `structural_repairs` (R1/R2/R6, the writer's actual mistakes) and
  `timing_repairs`. A metric that punishes a model for following instructions is
  measuring itself.
- Scoring by `planner_changes` ranked models on the wrong thing. Rank on
  structural repairs.

Over 2 runs each, on a 3-scene-capable concept:

| model | shots | structural repairs | preserved | scenes | tok/s | stability |
|---|---|---|---|---|---|---|
| granite4.1:8b | 4 | **0** | 1.00 | 2 | 2.7 | identical both runs |
| qwen2.5:3b | 6 | **0** | 1.00 | **1** | **5.5** | stable |
| mistral:latest | 7 | 1 | 0.86 | **7** | 3.2 | 6->7 repairs |
| llama3:latest | 6 | 2 | 0.67 | 3 | 2.9 | 6->8 repairs |

The writers are all structurally similar; the differences are shot count, scene
count, speed and **run-to-run stability**. `qwen2.5:3b` is fastest and clean but
always emits ONE scene, which disables the look and checkpoint stages entirely.
`mistral` emitted 7 scenes for 7 shots, i.e. a switch on every cut. `granite` is
the cleanest and most reproducible but only writes 4 shots.

### 11.3 The ensemble: idea farms, then one showrunner

The single-writer path asks ONE model to do everything at once — invent the
images, choose the framing, place the scenes, keep the film coherent — and §11.2
plus §13 show the local models differ a lot at that, with the weaker ones
intermittently fusing other field values into `prompt`.

The ensemble splits the job where the models actually differ:

```
farms       several cheap models, asked only for raw visual ideas (short output,
            so each call is fast). Their disagreements are the point.
showrunner  ONE model -- granite4.1:8b, the only one that never leaked -- given
            every idea and asked to write the final shot list. It decides what
            stays, what is discarded, what is merged, and how the shots cohere.
            It may invent. Its text is the only text that can reach a render.
```

`--writers ensemble` (the new default) on `make_film.py` and `story.py`;
`--writers single` keeps the old path. Farms are pointed at **different briefs**
(places/weather, objects/materials, light, people) so the idea pool is spread out
rather than four near-copies. Measured on the default concept:

| | run 1 | run 2 (after the quota fix) |
|---|---|---|
| farms | qwen 6 ideas/18s, phi3 6/24s, **llama3 1/33s**, mistral 6/50s | 6/20s, 6/27s, **1/34s**, 6/39s |
| showrunner | 6 shots in 183s, leaks 0, structural repairs 0, 3 scenes | 6 shots in 157s, leaks 0 |
| provenance | **qwen 6/6, phi3 0/6, llama3 0/1, mistral 0/6** | **qwen 3/6, phi3 1/6, llama3 0/1, mistral 2/6** |

**The finding that mattered, and the fix.** Run 1's provenance reads 6/0/0/0: the
showrunner took **six of six shots from qwen's list** and ignored the other three
farms entirely. Grouped by farm, the prompt was handing it four tidy little shot
lists and it took one of them whole — the architecture was paying for four farms
and using one. Two changes: the ideas are now **interleaved round-robin** with
their source labelled inline, so no scout's list sits together as a coherent
whole, and the brief now says **take at most two ideas from any one scout**
(`FARM_QUOTA`), with the reason stated to the model. Run 2 draws from three farms.

That is what the provenance measure exists for, and it is worth keeping: without
it, "we added idea farms" would have been true and useless.

**Cost.** Farms ~2 min + showrunner ~2.6 min ≈ **4.5 min**, against ~2.5 min for a
single writer. Text is still under 10% of a ~60 min render, so the original
"don't optimise the writer" arithmetic survives — but this is not free.

**llama3 was dropped from the farms, and the obvious replacements are worse.**
It gave **one idea in 33 s**, twice, where the others give six in 16–50 s — the
fourth slot cost a minute and contributed nothing. Swapping it needed a
measurement, not a guess, and both candidates failed:

| candidate | result |
|---|---|
| `qwen3:1.7b` | **1 idea in 24 s** — its reasoning spends the token budget before the answer |
| `qwen3.5:0.8b` | **empty reply**, 0 ideas |
| `llama3:latest` | 1 idea in 33 s (the one being replaced) |

So the farms run **three** — `qwen2.5:3b`, `phi3:latest`, `mistral:latest` —
and the farm briefs were reordered so the three used are the ones a film actually
needs (where it happens, what is close to the lens, who is in it). Farm time fell
from ~125 s to **~77 s**. Third run: 18 ideas, showrunner 209 s, `prompt_leaks` 0,
provenance **qwen 4 / phi3 2 / mistral 0**.

**STORY PROGRESSION — the layer that did not exist, and the misread that found it.**
When the brief asked "shouldn't the long view be about progression", the first
answer was a *motion-energy* arc in the director. That was the wrong layer twice
over: the question was about the STORY, and months of work had gone into
`energy`/camera-move progression while the film itself was never asked to develop.
The motion term is now named `ENERGY_ARC_*` for exactly this reason.

The gap, measured: the brief demanded **variety** ("do not write six variations of
one image") and never **development** ("something must change"). So the writer
produced mood boards. A real ensemble output, six shots: *shadows on a sidewalk,
no one in sight* → *silhouettes merge at a forest edge* → *two shapes on a bench* →
*two figures step toward each other* → *a lone figure in a crowd* → *shoes alone in
a puddle*. One idea, six restatements, location changing. Nothing in the pipeline
would have noticed — the planner only does technical continuity and the director
only picks camera moves.

Now there is a first-class representation and a check:

| piece | what it does |
|---|---|
| `story.BEATS` = establish / develop / turn / resolve | added to `SHOT_SCHEMA` as a **required** per-shot field |
| the brief | asks for a story: shot 1 establishes, at least one shot is the turn, the last resolves, beats never go backwards, and the **logline states the change** rather than the premise |
| `validate` | reports `story:` problems — no turn, ends unresolved, beats backwards, more than one establish, opens on the wrong beat |
| `story.story_shape` | prints the film's beats as `EdTR`, in the ensemble output and in `scorecard` (`story_breaks`, `story_shape`) |

Verified on constructed cases: a mood board `Eddd` → caught (no turn, no resolve);
`EdTd` → caught (ends unresolved, beats backwards); `ETdR` → caught (backwards);
a real progression `EdTR` → **0 breaks**. The OLD six-shot proposal above is now
caught automatically by the same check.

**A fresh ensemble run with the story brief: `EdTR`, 0 story breaks, 0 rule breaks,
0 prompt leaks, structural repairs 0.** Four shots instead of six, and the logline
now states a change ("a solitary figure confronts a doppelgänger, revealing that
friends and foes are indistinguishable"). The writer *can* produce a story when
asked — it had simply never been asked.

**Honest reading of the result, which is not yet a verdict**: the beats are now
declared and checked, but declaration is not development. Shot 4 (`resolve`, a dim
auction with silhouetted bidders) reads as a *new scene* rather than a resolution
of shot 3's turn (a figure with a knowing look). Whether the resolve pays off the
turn is a judgement for the eye, and it is the next thing to iterate — a much
better problem than six restatements, and now visible in one line of output.

**The weak joint: turn -> resolve, and making a broken story ACT.** Beats alone
were not enough. The first `EdTR` output declared its beats correctly and still
failed, because the turn was *a silhouetted figure with a knowing look* and the
resolve was *a dimly lit auction*. Declared structure is not development.

- **`link_overlap(a, b)`** — stemmed content-word overlap between two prompts.
  Stemmed because "silhouettes" and "silhouetted" are the same thing to a viewer
  and different strings to a computer. Below `TURN_RESOLVE_LINK = 0.08` the
  validator raises a **`story-link:` warning**: the resolve may read as a new
  scene rather than the turn's consequence.
- **Calibration, which is the point.** The real failure: overlap **0.05** -> flagged.
  A genuine payoff ("the box opens and the room floods" -> "she stands in the
  doorway holding the open box"): **0 warnings**. A legitimate ending on the empty
  room the film opened in: **0 warnings**. One warning, two clean.
- **REJECTED: "the recurring subject must be present at both ends."** It fired on
  **2 of 3** constructed cases — a pronoun defeats it ("she" for "woman"), and it
  flagged the empty-room ending. A check that fires on good work is the check
  being wrong, so it is gone and the reason is recorded here rather than in a
  commit nobody reads.
- **The showrunner now RE-ROLLS**: up to `story_retries` (default 2, so 3 attempts)
  if the story has breaks *or* link warnings, keeping the best attempt by
  `(breaks, warnings, problems)` -- so a re-roll can only improve on the first
  attempt, never lose to it. Text costs ~7 min for three attempts against a
  ~60 min render, which is the cheapest quality gate in the system.

**Measured, and it works.** A fresh run: attempt 1 -> 1 warning, attempt 2 -> 1
warning, attempt 3 -> **clean, stopped**. The film it produced:

> **logline:** a woman follows a mysterious figure through shadowed alleys and
> gardens, discovering that friends and foes appear indistinguishable, ultimately
> confronting her own reflection as the true adversary.
>
> 1. **[establish]** moonlit alley, a lone woman observing a masked figure
> 2. **[develop]** shadowed garden party, two figures in matching trench coats,
>    indistinguishable faces
> 3. **[turn]** a mirror of indistinguishable forms, the woman's reflection
>    **diverging at the eye colour**, revealing a hidden foe
> 4. **[resolve]** the woman **confronting her reflection**, two figures identical
>    but for the eye colour

The mirror and the eye colour are set up in shot 3 and **paid off in shot 4**
(overlap 0.11 > 0.08), and the concept is *dramatised* rather than restated: matching
coats, then the eye-colour divergence as the reveal. Compare the baseline mood
board — six restatements of shadows — which had no turn at all.

**Honest limit.** A lexical link cannot tell "linked but worded differently" from
"unrelated", which is why it is a warning with a low threshold and not a verdict.
It only catches near-total disjointness. A resolve that shares no words with its
turn can still be right, and the by-eye read is still the arbiter.

**The idea brief was also feeding the film its own examples.** The GOOD examples
in `IDEA_SYSTEM` were *"wet cobblestones lit by a single streetlamp"* and *"a bare
birch trunk in fog"* — exactly the imagery a fog-and-shadow concept wants, so the
farms copied them. Measured: deepseek-r1:1.5b returned both verbatim, plus the
concept echoed back and a camera direction. Examples now come from a deliberately
unrelated film (*a chrome diner counter at noon*; *a yellow digger bucket*), with
"do not copy the examples" stated. qwen2.5:3b went from copying to **0 copies, 0
concept echoes, 0 camera directions**. A farm copying its brief narrows every film
toward the same handful of images.

**deepseek-r1:1.5b as a farm: rejected.** 6 ideas in 18–28 s, the fastest tested,
but **usable on only 2 of 3 attempts** — the third truncated mid-JSON (`'{"ideas":
["Shadows and friends/foe", "Hard shadows", "Cracked windscre…'`) because its
reasoning eats the token budget. Even when it worked it copied the brief's
examples (a digger bucket and a cracked windscreen, in a film about shadows) and
echoed the concept. Speed is not the criterion — `llama3` taught that with one
idea in 33 s, deepseek teaches it with fast, unusable ones.

**A robustness bug found while testing that.** A farm whose reply was not clean
JSON was dropped *silently*, so the ensemble would quietly run on three farms
instead of four and nothing would say so. `json_from_text` now strips code fences
and takes the outermost braces, a failed farm reports the raw reply and why, and
the farm token budget went 400 → 700 because small models spend the first few
hundred tokens thinking. qwen3:1.7b's clean JSON on a short prompt and truncated
reply on a longer one is exactly the case that used to vanish.

**Do not compare single runs of the showrunner.** Across the three runs its
structural repairs were 0, 0 and **2**, and provenance spread was 6/0/0/0 (grouped,
no quota), 3/2/1/0 (interleaved + quota, four farms) and 4/2/0 (three farms). The
farm *cost* is a stable measurement; the showrunner's output is not, at n=1. The
quota and interleaving demonstrably stopped a 6-of-6 single-scout grab, but a
single run cannot say how much of the remaining spread is the config and how much
is the model.

**Cost.** Farms ~77 s + showrunner ~2.5–3.5 min ≈ **4 min**, against ~2.5 min for a
single writer. Text is still under 10% of a ~60 min render, so the original
"don't optimise the writer" arithmetic survives — but this is not free.

**This makes §13's one-hour render test moot.** It was designed to compare three
writers by rendering their output. Only the showrunner's text can now reach a
render, so the question collapses to "is granite's output good", which is the film
you were going to render anyway. The farms can only waste seconds.

## 12. Error bars, and two levers rejected

Every number before this section was n=1. Three seeds at the production
configuration (`fog_none_s11/12/13`, `fog_walk_s11/12/13`):

| condition | global_share | align |
|---|---|---|
| fog, no LoRA | 0.52 0.56 0.53  mean 0.537, **sd 0.023** | 0.377 0.342 0.365  mean 0.361, sd 0.018 |
| fog, walking LoRA | 0.45 0.46 0.48  mean 0.463, **sd 0.013** | 0.368 0.357 0.361  mean 0.362, sd 0.006 |

**The LoRA's effect is real, and it is still not enough.** The difference is 0.074
against a seed sd of ~0.018 -- about 4x the noise, so not a fluke. But 0.463/0.537
is a **14% reduction**, and separability needs ~30%. So the 9.7 finding is now
properly supported rather than a single sample: the walking LoRA reliably moves
the background-change metric by 14%, and 14% is not separation.

`align` is 0.361 vs 0.362 between conditions -- the LoRA does not hurt prompt
adherence. And align's seed sd is ~0.018, which retroactively confirms that
differences under ~0.02 (e.g. 0.3683 vs 0.3680) were correctly called identical.

### 12.1 REJECTED: 8 raw frames

422s vs 552s -- only **24%** faster, not the hoped-for halving, because model load
and VAE decode do not scale with frame count. `align` drops 0.368 -> 0.313 and
subject travel 14.8px -> 9.0px, for the same motion magnitude. Not worth it.
Halving render time is not available here.

### 12.2 REJECTED: the 8-step midpoint

The user's observation that 12-step is "more stable" is correct (coherence 0.0058
vs 0.0177). But the midpoint does not interpolate between the two -- it is its own,
worse regime:

| config | time | mag | align | coherence (lower = steadier) |
|---|---|---|---|---|
| 4-step Lightning | 552s | 13.92 | **0.368** | **0.0177** |
| 8-step Lightning | 1407s | 19.49 | 0.342 | 0.0462 |
| 12-step no Lightning | 2610s | 5.80 | 0.368 | **0.0058** |

8-step Lightning is **2.55x slower, less prompt-accurate, and 2.6x LESS stable**
than 4-step. There is no useful midpoint: the stable configuration is stable
because it is smeared. Stability and detail trade off, and the middle of the dial
is worse than either end. **4-step Lightning remains the production setting.**

Also note the pipeline already runs RAFT x2, which smooths temporally -- so part
of the 12-step stability advantage is recovered for free in the delivered film,
while its detail loss is not.

### 12.3 The grading normaliser: the result is real, the fix was never wired

The claim carried into this session was that normalising by `within_spread` was
inflating the film's grading report (16.29 where the truth was near the floor),
and that switching to `clip_floor()` fixed it, giving 0.48–0.81 -> 0.11–0.24
across `shadows` shots 1–6.

**The numbers are right. The code path that produces them is not the one in the
file.** Measured three ways on `shadows` (6 clips reconstructed through
RAFT → stretch → shake, post-`match_technical` signatures against the shared
`technical_target`):

| how the distance is reported | shots 1–6 |
|---|---|
| `decomp(..., clip_floor(c))` — **what `make_film.py` actually calls** | 1.29, 8.14, 15.98, 15.99, 175659.85, 682809.09 |
| `decomp(..., within_spread)` — the normaliser the fix replaced | 11.61 for a single pair, i.e. *deflated*, not inflated |
| **absolute, no normaliser** (`technical_offset`) | 0.47–0.70 before, **0.09–0.18 after**, means 0.58 → 0.15 |

So the 3–4x reduction and the 0.1–0.2 residual both reproduce — but only when the
distance is **reported absolutely**. `clip_floor` is degenerate on exactly this
kind of film: it is the *stylistic* distance between two halves of one clip, and a
film with one shared technical baseline plus one palette per scene has an almost
perfectly stable palette across halves. Measured floors on `shadows`: **0.000,
0.000, 0.010, 0.020, 0.060, 0.060** — which is how 793,261 and 814,481 were
produced the first time, and 175,659 the second. The degeneracy was *moved*, from
`within_spread` to `clip_floor`, not removed. And the two are not even
consistently ordered (0.010 vs 0.039 on the same clip), so no general "N x
inflation" factor exists — which is why "about 7x" in the old docstring was never
true in general.

**The fix, now wired.** `grade_match.technical_offset(sig, target)` returns the
technical distance with **no normaliser**, and `make_film.py` prints it with the
grey-level equivalent (0.2 decomp units ≈ 2.4 grey levels). The technical
distance is not a ratio — `SCALES` is already in grey levels, so it has physical
units and dividing it by anything is a category error. The palette control holds
at **1.00x** (stylistic distance 0.04–0.37 before and after), so the matching is
fixing the seam and not flattening the look.

**The lesson, now five instances deep** (hand-picked seam thresholds, the
loc10/mag confound, the look band, the film's 16.29, and this): *if a distance
looks wrong, check the normaliser first* — and when the normaliser can go to zero
on the content you care about, delete it instead of guarding it. `clip_floor` is
kept only for the pools (`model_select`, the A/B harnesses) whose palettes
genuinely vary, and its docstring now carries the measured degeneracy.

## 13. Staged, and not done

Recorded so the next session does not have to rediscover it.

1. **Pool memory + `--pool` wiring — not started.** The symptom was flagged in a
   previous session and never written down, so the failure mode is unconfirmed;
   what is certain from the code: `pool.load_pool` loads every frame of every
   candidate as a PIL image (786 KB/frame at 512², so a 48-clip pool is >1 GB
   before scoring), `pool.score` then needs all of them resident, and the
   `--prompts` JSON that maps clip id -> shot fields has no producer — `make_film`
   would have to write it. Peak RAM falls to one clip at a time if `score()` loads
   per candidate and releases.
2. **The writer test, "S6" — what it actually is, and its cheap half.**
   **SUPERSEDED by §11.3**: the ensemble makes this test moot, because only the
   showrunner's text can reach a render any more — so "which of three writers is
   better" is no longer a question the pipeline asks. The description is kept
   because the *reasoning* still governs how the text layer is judged, and because
   the cheap half below produced two live defects.
   The item was labelled "S6" in conversation; the label was applied to an
   *overridden* decision and does not exist in the code, so this is the record.

   The original call was **"don't invest in LLM optimisation now"**: a proposal
   takes 80–260 s against ~60 min for a 6-shot render, so the text layer is 2–5%
   of runtime and MoE / llama.cpp / fine-tuning would buy seconds on a pipeline
   dominated by diffusion. That was **overridden**: shot quality had to be proven
   not to be the writer's fault *first*, or a weak film leaves you unable to tell
   a bad writer from a bad sampler. The priority inverts — prove the writer is not
   the ceiling, then stop worrying about it.

   The design: 3 writers by scorecard (`granite4.1:8b`, `llama3`, `mistral`;
   `qwen2.5:3b` excluded — it cannot produce ≥2 scenes), same concept, fresh
   proposals, render 2 shots each at identical seed/scale/camera/settings so the
   prompt text is the only variable, then measure `align` and the composite score
   **and look at the pictures**. Cost ≈ 6 clips ≈ 60 min.

   Two caveats that argue for the cheap half first: it is **underpowered** (`align`
   seed sd is 0.018 per §12, so a real difference must exceed ~0.04 and n=2 only
   detects a large effect; genuine certainty wants 4–5 shots per writer ≈ 3 h), and
   **`align` is not quality** — it measures text-image agreement, so the test only
   works if a human looks at the output.

   **Cheap half: done.** `writer_audit.py` generates the fresh proposals, plans
   them, and diffs the prompt text — no rendering. Six samples (one fresh run per
   writer, plus the §11.2 caches folded in free) on the default concept:

   | writer | shots | content words/shot | self-overlap (own prompts) | **leaked, round 1** |
   |---|---|---|---|---|
   | granite4.1:8b | 4 then 6 | 13–17 / 16–20 | **0.038, 0.037** | **0 / 10** |
   | llama3:latest | 6, 6 | 7–12 / 10–14 | 0.249, 0.114 | **6 / 12** |
   | mistral:latest | 7, 6 | 14–18 / 13–18 | 0.063, 0.191 | **6 / 13** |

   Structural repairs were **0 for all three** (the six "timing" rewrites are R4
   rhythm, the planner's job by design — §11.2). Cross-writer content-word overlap
   is only **0.10 mean** (0.08–0.13 pairwise), so the writers describe genuinely
   different films. `granite4.1:8b` is the only writer with a stable low
   self-overlap across two independent runs, and §11.2's "granite only writes 4
   shots" is **not** stable — fresh, it wrote 6.

   **The defect that matters is not a style difference: writers intermittently
   fuse other field values into `prompt`.** llama3 writes "…, yellow
   streetlights, **2.1 seconds, energy 0.2**"; mistral writes "…, cinematic grain,
   **camera: none**". `prompt` is the string CLIP and SD receive, so that is text
   the image model tries to draw — and it would surface in a render test as worse
   `align`, i.e. the test would blame the sampler. **That is the exact
   mis-attribution S6 exists to prevent.**

   **It is variance, not the prompt version — and this correction is the point.**
   Two independent fresh rounds, same configuration:

   | round | granite4.1:8b | llama3:latest | mistral:latest |
   |---|---|---|---|
   | 1 | 0 / 6 leaked | **6 / 6 leaked** | **6 / 6 leaked** |
   | 2 | 0 / 4 leaked | **0 / 6 leaked** | **0 / 6 leaked** |

   The first reading of round 1 was "12 leaks fresh, 0 in 23 older cached prompts,
   and `story.py` was edited between the two sets, so the prompt edit caused it".
   Round 2 rejects that: the same prompt version produced zero. So the leak is
   **intermittent and total per proposal when it happens** — one proposal's worth
   of polluted prompts, with `story.validate` never noticing, showing up later as
   unexplained poor alignment. Two samples per writer bound the rate as
   "occasional to common" and cannot pin it down; a rate needs ~5 proposals per
   writer (≈35 min, or ≈18 min for llama3+mistral only, no rendering).

   **This is the handoff's own failure mode, committed while writing the
   handoff**: a verdict from one run, believed because the split looked clean
   (12/12 vs 0/23). It was not clean, it was two samples. Details and verbatim
   round-1 examples in `output/film/writer_audit/LEAK_NOTES.md`.

   **Now detected, not stripped.** `story.prompt_leaks` records it as a problem in
   `validate` (which by its own docstring collects problems rather than fixing
   them), and `scorecard` reports `prompt_leaks` separately from the structural
   repairs. The leaky prompt is deliberately left intact — detection turns an
   invisible intermittent failure into a counted one; whether to strip it is a
   separate decision, and **until it is stripped the render still receives the
   polluted prompt.**

3. **The truck/crane LoRAs** are still not on disk, so `director.py` cannot pick
   two of its own vocabulary entries.

4. **`render()`'s resample branch is dead code.** Both arms of the `if len(src) >=
   length` test compute the identical index mapping, so the comment's "stretch or
   trim" is really "always stretch". Harmless now that the film stretches before
   shaking, but it is a trap for anyone reading it as a loop-vs-stretch choice.

5. **The proposal cache destroys its own history.** `write_shots` keys the cache
   as `<slug>__<model>__n{n_shots or 'auto'}.json`, with **no run id**, so a fresh
   re-roll of the same model+concept overwrites the previous sample. That is how
   the first round of the writer audit (12 leaked prompts) was lost before it could
   be re-analysed — and it contradicts the cache's own stated purpose, which is to
   tell "whether a change in the output came from the writer or from the sampler".
   For repeated fresh runs it does the opposite. `writer_audit.py` now writes an
   immutable `samples_<timestamp>.json` per round plus an append-only
   `samples.jsonl`; the production cache should probably carry a run id too.

6. **SUSPECTED UNIT MISMATCH: the writer's `energy` and the director's scales do
   not agree.** The brief tells the writer `energy` is 0 to 0.5 and
   `MAX_RENDERABLE_MOTION` clamps it there, but `SCALE_ENERGY` spans 0.20-0.90 and
   `energy_of` returns motion priors spanning 0.30-0.80. The cohesiveness term is
   `1 - |energy_of(name) - want| / 0.6`, where `want` is the writer's 0-0.5 value
   or the scale's 0.2-0.9 fallback — so a shot declaring the maximum 0.5 is
   matched as if it wanted *mid-range*, and the two scales only coincide by
   accident. This is the same shape as the other five units errors and it is
   directly load-bearing for progression, because the declared energies ARE the
   arc. Not fixed here: changing it moves every motion choice, so it needs its own
   measurement, not a guess.

7. **The camera research is on disk**, not just in URLs — `research/`:
   `SHAKE_AMPLITUDE_REPORT.md` (the amplitude study, with its own corrections and
   a 24-URL source list), `txt/` + `pdfs/` (11 primaries, 55 MB), and
   `txt_amplitude/` + `pdfs_amplitude/` (21 more, 159 MB). The two that matter
   most for re-deriving the constants are `cipa_en.txt` (the standard shake
   waveform and its average-vibration-angle curves, whose axis units carry a
   genuine ±100x labelling ambiguity — ratios and rates survive it, absolute
   magnitudes do not) and `han_tmc.txt` (the measured rotation/translation
   crossover at 100 cm). The highest-value primary is still **missing**:
   Golik & Wüller's stabilisation thesis is the one study that measured tremor
   amplitude in subjects actually *holding a camera*, and it 404s.
8. **The open question this work leaves**: the amplitude sits at ~0.7x the lowest
   measured estimate, so if the result ever reads as too subtle the honest move is
   a modest increase (and the blur budget above says where to stop), not an
   order-of-magnitude one. That is a judgement for the eye, and the A/B makes it a
   one-line constant change.

---

## 14. The UI — delete the old one, and build it around how this pipeline works

> **SUPERSEDED. Read §0 instead.** This section describes the seven-stage workbench,
> which was built, verified, used, and then replaced: the user's verdict was *"it
> doesn't feel intuitive just packed"* and *"why do we need 7 stages"*, and §0.3 has
> the reasoning. The workbench is now `ui_legacy/workbench_superseded.py` and the
> production UI is `ui/studio.py`. What follows is kept for two reasons: its
> **interaction model** is still the right one and §0.3 takes from it, and its build
> log records six real bugs (`job_runner`'s lock and pid races, `--assemble` loading 7
> GB it never used, a shadowed `st`) whose fixes are all still live.

**What is there now, and its state.** `ui/app.py` (51 KB), `ui/components.py`
(7 KB), `ui/theme.py` (1 KB), `backend/adapters.py` (15 KB).

**Nothing imports any of them.** Every root-level `.py` was grepped for
`import ui` / `from ui import` / `import backend` / `from backend` — no hits — and
none of the three UI files has an `if __name__ == "__main__"` entry point or a
gradio/streamlit/flask import that a grep can find. They belong to the superseded
"stills + flow-warp" era (§1: the user explicitly does not want a still-frame
generator), so they describe a pipeline this project no longer has.

**Not deleted yet, deliberately.** 60 KB of someone's code should not be deleted
unread. Move `ui/` to `ui_legacy/` once the new shape is agreed, or read it and
delete it knowingly. `gradio`, `streamlit`, `flask`, `fastapi` and `uvicorn` are
all installed, so the choice is open.

### 14.1 CURRENT STATE — the workbench is BUILT (read this first)

**Live**: `streamlit run ui/app.py` -> http://127.0.0.1:8501

**Three modules hold it up.**

| file | what it is |
|---|---|
| `film_project.py` (271 lines) | `workspace/<film>.json` is the source of truth: concept, the proposal as `story.validate` saw it, per-shot records (incl. `motion_pin` and per-shot render state), the **FROZEN plan**, the planner's changes and report, per-stage settings, the film record. |
| `job_runner.py` (544 lines) | Detached jobs writing structured JSON progress; `start()` returns in **0.02 s**, `status()` never blocks, **one job at a time enforced**. Tasks: `story`, `render`, `audit`, `selftest`. |
| `ui/app.py` (1176 lines) | Seven stages, selected in the **sidebar** (not tabs -- streamlit runs every tab body on every rerun, and PLAN loads CLIP). |

**The pipeline changes it required, each verified:**

- **`make_film --from-project <name>`** renders the FROZEN plan and **skips the
  director and the planner**. Without it, an "approve the plan" button is theatre:
  the plan was recomputed on every run, so an approved plan -- and any UI edit --
  would be silently discarded.
- **`make_film --only N` / `--assemble`**: per-shot stages stay on disk
  (`raw/`, `smooth/`, `shaken/`). Generate once per shot, assemble as often as you
  like **with no generation**. `--assemble` does **not** load the diffusion
  pipeline (it used to; that alone was >10 min -> 97 s).
- **`story.py --keep-file/--keep-only`**: a re-roll can be seeded with the scout
  ideas a human kept.

**The seven stages.** STORY (logline, every shot field, add/delete/reorder, the
scouts' ideas with keep/discard, the gate) - PLAN (mode/cap/arc, per-shot motion
pin, the planner's repairs, freeze) - CAMERA (per-shot shake knobs + an A/B preview
on that shot's raw frames) - GRADE (baseline on/off, palette per scene, measured
legibility) - RENDER (per-shot table of what is on disk, thumbnails of raw/RAFT/
shaken, render-this-shot / approve / assemble) - TIMELINE (exposure sheet from
`hold_frames`, reorder, the assembled film, attach an existing render) - AUDIT
(provenance, camera units, grading in decomp AND grey levels, the writer audit,
story shape, the director A/B).

**Verified** (short list; the evidence is in the log below): the frozen plan is what
renders (DIRECTING and SCORECARD absent from the output); `--only 1` gave
raw 16 -> RAFT 31 -> shaken 51 with no other shot touched; `--assemble` cut 4 shots
from disk in **97 s**; the CAMERA preview reported 3.72 px peak translation for
shot 1, against the 3.7 px its real render recorded -- but the ROLL figures are
**not comparable and must not be quoted as agreement**: the preview shows 0.73 deg
as peak-to-peak while section 7's recorded 0.45 deg is a PEAK, a different
statistic. The moving shot damped to 1.55 px / 0.25 deg p2p; TIMELINE computed
10.94 s / 175 frames from the frozen plan,
matching the rendered film; AUDIT reproduced §7's grading offsets
(0.026/0.008/0.047/0.034 decomp = 0.31/0.09/0.57/0.40 grey); and a seeded re-roll
took 18 scout ideas down to the 2 kept, reaching a clean `EdTR` story on **attempt
1** with a logline naming exactly those two.

**OPEN -- do these before trusting the thing:**

1. **A successful seeded re-roll reported exit code 1.** If `story.py` exits
   non-zero on success, `job_runner` will report **FAILED for a good run**. Check
   this before believing the job panel's status.
2. **No film has been made by clicking through the UI end to end.** Every
   workbench prediction matched a rendered artifact, but the renders themselves
   were CLI/background jobs.
3. **A full 4-shot assembly with REAL per-shot stages has not been run.**
   `output/film/ensemble1_pershot/` has shot 1 only; `output/film/_assemble_test/`
   is a MECHANISM test (ensemble1's raw frames staged into `shaken/`, so it proves
   the plumbing, not that those pixels were shaken).
4. **`ui_legacy/` and `project/` still need a keep/delete decision.** There is no
   version control in this workspace, so deletions are irreversible.

**Bugs found this session, all fixed** (the log has the details): a job that never
started **wedged the lock forever** and refused every later job; a **pid race** that
cleared the lock during a job's finish window; `tasklist /FI "PID eq N"` does not
list processes on this machine; **`os.kill(pid, 0)` TERMINATES a process on Windows**
instead of probing it -- it would have killed the very render the lock protects; and
`--assemble` loading 7 GB of generator it never used.

**The log below is REVERSE CHRONOLOGICAL** -- STEP 11 at the top down to STEP 1,
then the original design notes for this section.

---

**STEP 11 DONE, AND PROVEN END TO END: per-scout ideas keep/discard, with a
seeded re-roll. The last unmet item in the brief.**

**The real run** (`--fresh --keep-only --keep-file`, ~2 min):
```
=== IDEAS (3 farms) ===
  18 ideas from 3 farms
  [story] seeded with ONLY the 2 kept idea(s) (the rest discarded)
  [story] showrunner attempt 1: 4 shots, 0 story break(s), 0 story warning(s)
  [story] complete story -- stopping the re-roll
  story shape: EdTR
  logline: In a moonlit alley and a shadowed garden party, a figure confronts
           its own reflection ...
```
18 ideas were generated and 16 discarded; the showrunner saw only the two kept
ones, and its logline names **exactly those two ideas** ("moonlit alley", "shadowed
garden party"). It reached a clean `EdTR` story on **attempt 1** -- where the
unseeded run needed three attempts and 439 s. Seeding the pool is not just
control, it is faster.

**THE FLAW THE REAL RUN EXPOSED, which unit tests could not.** The first
implementation only *matched* kept ideas against the farms' output, and a re-roll
re-runs the farms -- so a fresh farm run does not reproduce a kept idea verbatim,
and the director's selection would have been silently discarded precisely when the
farms had re-run, the normal case. Kept ideas are now **INJECTED** as
`- [kept by hand] ...` lines of their own, with matching used only to avoid
listing one twice. `keep_only=True` is the "discard the rest" meaning, which is
what the UI button does. This is the argument for running the real thing rather
than trusting a green unit test.

**OPEN, UNRESOLVED**: that run reported **exit code 1** while clearly succeeding.
If `story.py` exits non-zero on success, `job_runner` will report **FAILED for a
good run** -- worth checking before trusting the job panel's status colour.


- **`story.write_shots_ensemble(..., keep=[...])`** — the showrunner is shown only
  the kept ideas, prefixed *"(the DIRECTOR kept these ideas by hand and wants the
  film built on them)"*, so a re-roll builds on a selection instead of re-rolling
  the whole pool.
- **`_seed_lines` is deliberately defensive**: it matches on the idea TEXT (not
  the label, which changes when the farms are re-run), and **if nothing matches it
  refuses the filter and passes every idea with a warning**. Handing the
  showrunner an empty idea list because of a string mismatch would look exactly
  like a scout failure, and would be blamed on the scouts.
- **A different kept set is a different proposal**: the keep hash goes into the
  cache filename, so "keep these" cannot silently return the film the user just
  rejected.
- **`story.py --keep-file`**, the runner passes it through, and STORY now has a
  checkbox per idea plus *save the kept set* and *RE-ROLL with the kept ideas*.
  An empty selection writes no file and says so rather than pretending to seed.

**Verified**: `_seed_lines` unit-tested on four cases (keep one -> 1 line, keep two
-> 2, **no match -> all 3 with the warning**, `None` -> all 3); the project carries
18 scout ideas in 3 groups; `write_kept` wrote
`workspace/ensemble1_workbench.kept.json` with the two kept texts; the runner's
argv carries `--keep-file`; an empty kept set returns `None`. A real seeded re-roll
was then run in the background to confirm the path end to end.

**STEP 10 DONE: `--only` and `--assemble` are both PROVEN, and a real
inefficiency was found by measuring rather than reading.**

**`--only 1` works.** `output/film/ensemble1_pershot/` came out as
`raw/shot01` **16 frames** -> `smooth/shot01` **31** (RAFT x2 of 16) ->
`shaken/shot01` **51** (the shot's hold), with no other shot touched. That is the
per-shot path end to end: generate one shot, land its stages, leave the rest alone.

**`--assemble` works, and regenerates nothing.** With the four shots' stages on
disk, `--assemble` printed *"assembling 4 shots from disk, no generation"* and
produced `film.mp4` (549 KB), `stats.json` and `shots.png` — **in 97 s**.

**THE BUG: `--assemble` was loading the entire diffusion pipeline it never used.**
`gen = Generator(a.base)` was built *before* the `if a.assemble:` branch, so
cutting a film out of finished frames loaded 7 GB of AnimateDiff and minutes of
model loading for nothing. Found by measuring, not reading: a job that should be
PIL + ffmpeg was sitting at **7.3 GB RSS and 829 s of CPU**. Guarded — `gen = None
if a.assemble else Generator(a.base)` — which took the same assembly from **>10 min
to 97 s**. My first attempt at that one-line fix broke the file, because the line
was 8-space indented inside `with keep_awake(...)` and I matched it with a 4-space
pattern, emptying the `with` body. Caught by `ast.parse` before anything ran.

**Note on the assembly test's frames**: `output/film/_assemble_test/` stages
ensemble1's RAW frames into the `shaken/` slots, so it is a mechanism test -- it
proves the plumbing reads per-shot stages and cuts a film without generating, NOT
that those particular pixels were shaken. The real per-shot stages are in
`ensemble1_pershot/`.

**STILL OUTSTANDING against the objective: STORY's "per-scout ideas
keep/discard".** The scouts' ideas are displayed, but there is no keep/discard that
influences anything -- and none can, until the re-roll can be SEEDED with kept
ideas, which is a change to `story.write_shots_ensemble`. That is the last item in
the brief that is not done.

**STEP 9 DONE: the AUDIT stage completed. All seven stages now carry real
content.**

- **Provenance**, per farm: ideas, landed, mean best overlap, and the traced
  fraction of final prompts. "Without this, four farms paying for one looks
  identical to an ensemble working."
- **The camera, in units**: per shot, `handheld.shake_path`'s peak px and roll
  peak-to-peak, the hold in frames, and the shake scale or `off`.
- **The grading, in units**: `grade_match.technical_offset` from the shared
  baseline, in **decomp units AND grey levels of contrast** — both, because five
  separate wrong numbers in this project came from a unit confusion.
- **The writer audit**: the pipeline's own `audit.txt` (89 lines) plus
  `LEAK_NOTES.md`, displayed rather than recomputed, since re-running the leak
  measurement costs LLM calls.
- Story shape / breaks / warnings, and the director A/B.

**Verified**: 4 metrics and 3 tables whose columns are exactly the ones above;
the provenance caption read **"3 of 4 final prompts trace back to a scout's
idea"**; and the recomputed grading offsets were **0.026 / 0.008 / 0.047 / 0.034
decomp = 0.31 / 0.09 / 0.57 / 0.40 grey levels**, which reproduces handoff 7's
recorded `ensemble1` measurement (0.02-0.07 decomp = 0.3-0.8 grey) — the
workbench agrees with the earlier by-hand number.

**STILL UNVERIFIED: `--only` end to end.** The per-shot render has now run 13m22s
on shot 1 with **5884 s of CPU** burned (~7.3 cores saturated, 6.4 GB resident), so
it is compute-bound rather than stalled — but it has produced no frames, because
`raw/` is written only after the first clip returns. Two lessons worth keeping:
it is **slower than the measured 540-867 s per shot** while sharing the machine
with the streamlit server, which is the RAM/CPU serialisation rule showing up in
practice; and **`raw/` is written after generation, not during**, so "no directory
yet" says nothing about whether a render is alive. Check CPU, not the filesystem.
`--assemble` behind it is unproven for the same reason.

**STEP 8 DONE: the TIMELINE stage.**

- The exposure sheet: per shot `in`/`out` timecodes, the **hold in frames from
  `handheld.hold_frames`** — the same function the render uses, so a timecode is
  not a second opinion — plus the packshot length, beat, scale and camera.
- **Reorder the FROZEN plan**, which is the only place the cut's order can change
  without re-planning (any story edit invalidates the plan).
- The assembled film: duration, shots, generation cost, `film.mp4`, the contact
  sheet, the per-shot table with model/look/hold/frames/shake — and the **film
  record written back into the project** (`name`, `duration_s`, `frames`), which is
  the artifact's identity.

**Verified, and this is a real check rather than a smoke test**: the TIMELINE
computed **10.94 s / 175 frames** from the frozen plan alone — exactly the
`ensemble1` render's own duration and frame count, with the holds reproducing
51/39/27/58. The frozen plan and the rendered artifact agree, which is what
`--from-project` was built to guarantee.

**A gap found while testing it**: the TIMELINE looked only at
`output/film/<the project's own name>/`, so an existing render living under another
name (`ensemble1`) was invisible and the workbench looked emptier than the disk is.
It now offers to **attach an assembled film that already exists**.

**STEP 7 DONE: per-shot rendering, and the RENDER stage.**

- **`make_film` keeps the per-shot stages on disk**: `output/film/<name>/raw/`,
  `smooth/` (RAFT) and `shaken/`, per shot. That is what makes "raw -> RAFT ->
  shaken" inspectable, and what assembly reads instead of regenerating.
- **`make_film --only N[,M]`** renders just those shots and **stops before grading
  and assembly** (which need every shot), telling you so and pointing at
  `--assemble`.
- **`make_film --assemble`** does no generation at all: it reads the shaken frames
  from disk and runs the existing grade + assemble path, and refuses with the list
  of missing shots if any are absent. **Generate once per shot, assemble as often
  as you like** — the pool's economy without the pool.
- **RENDER stage**: a per-shot table of what is on disk (raw / RAFT / shaken counts
  and status), thumbnails of all three stages for one shot, and
  render-shot-N / approve-shot-N / render-all / ASSEMBLE buttons, all through the
  job runner. `sync_render_state` brings the project's per-shot state in line with
  the disk, and **never downgrades an approved shot**.

**Verified**: the stage helpers round-trip (5 frames written -> 5 loaded, in
`output/film/_selftest_stages/shaken/shot01`), a shot with no stages loads as
`None` (which is how `--assemble` detects a missing shot), and `--only`,
`--assemble` and `--from-project` are all present. The RENDER stage renders under
`AppTest` with its table, thumbnails and buttons.

**BUG, and a worse one than it looks: a job that never started wedged the lock.**
Launching a job from a short-lived command means the parent's process tree is torn
down before the supervisor gets going, so the job sat at `queued` with `pid: null`
— and `status()` only ever marked LOST when a pid existed and was dead. So the job
held the lock **forever** and every later job was refused: one bad start bricked
the runner for the session. `reap()` now also treats "no pid well after creation"
as LOST, with a 45 s grace. Verified: at age 49 s the stuck job reaped to `lost`
with the right message and released the lock.

**The constraint behind it, which belongs in the record**: a supervisor spawned by
a process that then exits does not survive on this machine — the executor tears
down the whole tree. So **jobs must be started from the long-lived UI**, which is
exactly how the workbench uses it. Evidence: started from a short-lived script, the
supervisor never appeared (empty `supervisor.txt`, no pid, no `make_film` process);
started from a parent that stayed alive, the same runner took jobs to `done`.

**NOT yet verified end to end**: the full `--only` -> `--assemble` cycle. The
per-shot render is running as a managed job; its stages landing on disk is the
next round's first check, and assembly needs all four shots (~40 more minutes).

**STEP 6 DONE: the GRADE stage.** The shared photographic baseline on/off, and a
palette per scene, both **read by the render from the project** (`make_film` now
prints `grade: baseline technical, palettes [...]` and applies them, so the stage
is not a display).

- **Legibility is measured, not asserted**: `looks.look_distances` on the film's
  own held frames gives the technical/stylistic separation and the verdict for
  every transition the film will actually make. A one-palette film is told it has
  no transition to measure, which is a valid choice rather than a gap.
- **Preview**: `looks.apply_look` on a shot's raw frame, before and after, with
  `palette_floor` shown.

**Verified**: preview built (206 KB PNG, `preview_look_shot01_warm-gold.png`);
`neutral -> warm-gold` measured **style 3.88** and `warm-gold -> cold-steel`
**7.15**, both verdicts `LEGIBLE STYLE`, floor 0.15; a two-scene assignment wrote
`['neutral','neutral','warm-gold','warm-gold']` into the project — the same
assignment the `ensemble1` render used — and a dry run printed exactly that line
back, which is the proof the render reads it.

**Careful with the numbers**: those separations are measured on THIS film's shot-1
frames, not on `looks.py`'s own reference source, so they are NOT comparable to
handoff 10.6's 1.80 / floor 0.33. The floor and the distance both move with the
source, which is precisely why legibility has to be measured on held content
rather than derived or remembered.

**STEP 4 DONE: the CAMERA stage, and per-shot shake that actually reaches the
render.**

- **CAMERA** — a global shake scale, then per shot: shake on/off, scale, seed,
  save. Each shot has **"preview this shot"**, which builds an A/B GIF of
  **steady against shaken on that shot's RAW frames** by calling
  `handheld.stretch_to` → `handheld.spec_for_shot` → `handheld.apply_handheld` in
  the same order `make_film` does, and prints `handheld.shake_path`'s own numbers
  (peak px, roll p-p, envelope floor, frames at the delivered length).
- **The edit reaches the film.** `film_project.shake_params(proj, i)` supplies the
  per-shot knobs and **`make_film` reads them when rendering from a project**, so
  the CAMERA stage is a control that does something. Before this, `--shake` was a
  single global number and any per-shot edit would have been silently ignored.
  `set_shake_param` drops a key set back to the global default, so an override
  cannot masquerade as a deliberate choice.

**Verified**: a preview built from shot 1's raw frames (5.0 MB GIF, 51 frames at
the delivered length) reported **3.72 px peak translation / 0.73° roll
peak-to-peak / floor 0.23**. **CORRECTED**: section 7 records 3.7 px and 0.45° for
that shot. The translation agrees (3.72 vs 3.7) and that agreement is expected,
because the preview and the render call the same functions -- but the **roll
comparison was wrong and is withdrawn here**: 0.73° is peak-to-peak and 0.45° is a
peak, so the two numbers never measured the same thing. Claiming they matched would
have been a third units confusion in this project, not corroboration. The scale knob doubles the motion
exactly (1.0× → 3.72 px/0.73°, 2.0× → 7.43 px/1.45°). **Damping is visible through
the same path**: shot 1 (`camera=none`) 3.72 px/0.73° against shot 3
(`zoom-in`) **1.55 px/0.25°** — the mover gets a third of the roll, which is the
rule doing its job rather than a claim about it. And a per-shot override
round-trips: `shake_params` returns `{amount_scale: 2.0, seed: 99}` for shot 1 while
shot 2 stays at the default, then clears cleanly.

**A bug found by testing rather than reading**: `FILMS` had been dropped from the
constants when `ui/app.py` was rewritten, so `camera_preview` referenced a name
that did not exist. The CAMERA stage would have failed the first time anyone
pressed preview.

**STEP 3 DONE: the first clickable workbench — `ui/app.py` rewritten.** The
rejected read-only board is gone; this is a stage-per-artifact workbench over the
project file and the job runner.

- **A stage is selected in the SIDEBAR, not as a `st.tabs` tab**, and that is a
  correctness decision, not taste: streamlit executes every tab body on every
  rerun, so a PLAN tab would load CLIP on every keystroke.
- **STORY** — logline, story shape and the `story:`/`story-link:` problems, the
  provenance table, the scouts' raw ideas, and **every shot field editable in
  place** (prompt, beat, scale, camera, seconds, scene, subject, energy) with
  add / delete / reorder. Edits are re-checked with **`story.validate`**, the same
  validator that judges the writer, so a hand edit cannot slip past a rule the
  writer was held to. The gate refuses to approve a structurally broken story.
- **PLAN** — director mode, mover cap, energy arc, "compute the plan" (~16 s,
  behind a button because it loads CLIP), the frozen plan table, the planner's
  repairs split into "changed a CHOICE" vs R4 timing, the energy profile against
  the target arc, a **per-shot motion pin**, and APPROVE AND FREEZE.
- **CAMERA / GRADE / RENDER / TIMELINE** are honest stubs: each says what it will
  do and nothing more. A control that does nothing is worse than an honest gap.
- **AUDIT** — story shape, story breaks, warnings, and the director A/B. The rest
  of the old board's panels come here.
- **Any story edit invalidates the plan** (`frozen: []`, `approved: False`), so a
  stale plan can never render.

**Verified**: every one of the seven stages renders without exception under
`streamlit.testing.v1.AppTest`; STORY loads `pin_test` with a clean story shape
and 22 controls, PLAN shows 4 metrics and the frozen-plan table, and the four stub
stages each show their "not built yet" note with the specific gap.

**A bug worth recording**: `job_panel` assigned `st = jr.status()`, shadowing the
streamlit module inside that function — every `st.sidebar` call below it died with
`UnboundLocalError`. The local is now `job`, and the function says why.

**Also**: `film_project.list_projects()` now filters to files that actually ARE
film projects (they must have `shots` and `plan`), because `workspace/` also holds
`current_project.json` from the retired pygame studio and its schema is different.

**STEP 2 DONE: `job_runner.py` — detached jobs with structured progress.**
Every long step (ensemble story ~4-7 min, generation ~9-14 min/shot,
RAFT+shake+grade ~2-3 min) runs out of the UI process. Not a workaround for
streamlit: only one render fits in 15.3 GB, and a UI must not die with a crashed
job.

- `start(task, **params)` spawns `job_runner.py run <id>` and returns in
  **0.02 s**. `status()` never blocks. `stop()` asks, then kills the tree.
- The supervisor owns one job: it runs the real command with stdout going straight
  to `output/jobs/<id>/log.txt` (**no pipes** — the sandbox forbids piped stdio in
  confined modes, and a file is simpler), polls that log on a timer, folds the
  pipeline's own printed lines into a structured `progress` dict, and writes
  `state.json` until the child exits. The numbers are the pipeline's; nothing
  re-derives a measurement.
- **One job at a time is a hard rule**, enforced with a lock: a second `start`
  raises and names the running job.
- Tasks: `story`, `render` (from a FROZEN plan), `audit`, and `selftest` — the
  runner's own smoke test, because a runner with no self-test cannot be verified.

**Verified** (selftest 6 ticks, then a real pipeline task):
`start` returned in 0.02 s; a second start was refused with *"selftest is already
running … Only one job at a time"*; progress was parsed live as `tick 2/6 … 6/6`
with status `running → done`, rc 0, and the correct log tail; the lock released on
completion; `stop` mid-run took a job from `ticking 3/30` to `stopped`; and the
`audit` task ran the real pipeline through the runner — but the FIRST
attempt reported `lost`, and that was a bug in the runner, not the pipeline.

**THREE TRAPS, all found by running it rather than reading it.**

**The pid race, which is the one that mattered.** The state recorded the *child's*
pid as the job's pid. When the child exited — a second or two before the
supervisor had written its terminal state — the UI saw a live status with a dead
pid and declared the job **LOST**, clearing the lock during the finish window.
That is precisely the hazard the lock exists for: a second render could have
started while the first was still shutting down. The state now carries the
supervisor's `pid` and the command's `child_pid` separately, LOST requires the
*supervisor* to be gone, and the `stop` fallback kills the child. Re-verified: the
same audit task now reaches `done`, rc 0, with `progress {'phase': 'report'}`
parsed from `writer_audit`'s own output.

**TWO PLATFORM TRAPS, both of which would have broken the lock.** `tasklist /FI
"PID eq N"` is not usable on this machine — it runs but does not list the process,
so every job looked dead. And **`os.kill(pid, 0)` is a trap on Windows**: CPython
implements `os.kill` with `TerminateProcess` for any signal other than the console
events, so "signal 0" would KILL the process rather than probe it — it would have
terminated the very render the lock exists to protect. `_alive` now uses
`OpenProcess` + `GetExitCodeProcess`, and **reports ALIVE on any failure**, because
a wrong "dead" releases the lock and allows a second render, which is the one
outcome it must never cause.

**STEP 1 DONE: the project file, and the render reads from it.** This was the
load-bearing pipeline change, not UI work.

- **`film_project.py`** — `workspace/<film>.json`: concept, the proposal as
  validated by `story.validate`, per-shot records (`id, prompt, scale, camera,
  seconds, scene, subject, energy, beat, motion_pin, render{status,seconds,at}`),
  the FROZEN plan (the exact post-R1/R4 shot list), the planner's changes and its
  report, per-stage status, and per-stage settings for camera/grade. `new / load /
  save / list_projects`; `shots_from_raw`, `shot_objects`, `plan_project`,
  `approve_plan`, `frozen_shots`, `frozen_report`.
- **`make_film --from-project <name>`** renders the frozen shots and **skips the
  director and the planner entirely**. Before this, `make_film` recomputed both on
  every run, so an approved plan — and any edit made in a UI — was silently
  discarded.

**Verified.** A project built from the cached ensemble proposal planned in 15.9 s
and froze `['none','none','zoom-in','none']` at holds `[3.2, 2.44, 1.68, 3.6]`;
after approval, `make_film --from-project ensemble1_workbench --dry-run` printed
`FROZEN plan: 4 shots, approved 2026-10-11 21:03:09`, reached the dry run, and its
plan matched the frozen one exactly — with **DIRECTING and SCORECARD both absent
from the output**, which is the proof that neither stage re-ran. The motion-pin
override was exercised on a copy (`pin_test`): with shot 1 pinned to `pan-left`,
the frozen plan keeps it.

**Known limitation, deliberately documented rather than hidden**: the director is
not told about pins, so its moving-shot budget and its 180-degree gate were
evaluated against the motions the pins replace. A pin is an override of the
optimisation, not a constraint it respects. Making it a constraint belongs to the
PLAN stage.

**WHAT IT MUST BE (corrected by the user, and this is the target).** A
**production system** for actually making films with this pipeline, which makes
creating easier and **allows editing at every stage**. It does NOT have to copy
the old pygame app's design, headers or framework -- the pipeline has changed. What
is taken from the old one is its *interaction model*, which is the right one:
a stage per artifact, everything editable in place, a human gate before the next
stage, and work that never blocks the interface.

**A project is a durable file, not a cache.** `workspace/<film>.json`: the concept,
the proposal (editable), per-shot records, the approved plan, per-shot render state,
the film record. Every edit writes it. **The pipeline reads from the project, and
that is the change that matters** -- today the plan is recomputed on every render,
so what you approved is not necessarily what renders.

| stage | produces | editable there | gate |
|---|---|---|---|
| **STORY** | concept -> per-scout ideas -> the showrunner's shot list | concept; each scout's ideas (keep/discard); logline; every shot field (prompt, scale, camera, energy, beat, scene, subject); add / delete / reorder; re-roll | story shape clean (`EdTR`) |
| **PLAN** | the whole-film director's assignment | mode, mover cap, energy arc, and a **per-shot motion pin** -- accept the director's choice or override it; the planner's repairs shown per shot | the plan is FROZEN into the project; the render uses that, not a fresh plan |
| **CAMERA** | the per-shot shake spec | amplitude scale, roll, envelope floor, seed, damp-on-move; preview on that shot's raw frames | look at it |
| **GRADE** | the shared baseline + one palette per scene | baseline on/off, the palette per scene | look at it |
| **RENDER** | per-shot clips, one at a time | the prompt or the shake, then **re-render just that shot** | look -> approve, or re-render |
| **TIMELINE** | the assembled film | order, per-shot in/out, export | export |
| **AUDIT** | provenance, leaks, story shape, the director A/B, units | -- | -- |

**Nothing blocks, and that is required anyway.** Expensive steps (ensemble story
~4-7 min, generation ~9-14 min per shot, RAFT+shake+grade ~2-3 min) run as
**detached jobs writing structured progress to JSON** while the page polls. Not a
workaround: only one render at a time fits in 15.3 GB, and a UI must not die with a
crashed job. Cheap steps (director, planner, shake spec, grading offsets, previews)
run in-process because they take seconds.

**Framework: streamlit + the job runner.** Chosen over pygame even though pygame is
what the old app used: pygame needs every widget hand-drawn (the old app spends
1117 lines on layout maths), has no text selection, no scrolling in a text field
and no video playback -- and this is above all a **prompt-editing tool**, so real
text areas matter. streamlit gives text areas, tables, sliders, images and video for
nothing. Its blocking rerun model is the one real cost, and the job runner removes
it.

**REJECTED, kept as a record: the read-only evidence board.** The first attempt at
`ui/app.py` was a plan-and-evidence dashboard. It was rejected because **you can
decide nothing on it** -- no project, no shot records, no editing, no gates.
Its panels are worth keeping *inside* the workbench as the AUDIT stage, and its
lesson stands: every number comes from calling the pipeline's own functions, and no
measurement is reimplemented. The verification of it (AppTest, HTTP 200) is real but
beside the point.

**What is taken from the old app, unchanged.** `ProjectManager`'s idea that a
project is a resumable file with a rich shot record (`purpose, camera, duration,
action, dialogue, sound, prompt, keyframe, continuation, voice`) rather than a
prompt string; a stage per artifact with an editable copy in between; an APPROVE
gate per shot so a human is in the loop before the expensive step, not after; and
backends that report through callbacks on background threads. Read
`ui_legacy/ui/app.py` for the interaction model before building a stage, and read
that stage's `draw_*` body in full rather than sampling it.

**Hard rule for the implementation**: `ui/app.py` must *call* the pipeline's own
functions -- `story.write_shots_ensemble`, `director.compare_modes`,
`edit_planner.plan`, `handheld.spec_for_shot`, `grade_match.technical_offset` -- and
must not reimplement a single measurement. Two definitions of one measure is how
this project got the normaliser wrong five times, and the UI is the most tempting
place in the codebase to duplicate them.

---

## 8. Working style notes

The user is technically sharp and reads the outputs. They noticed the
SHUTTERSTOCK watermark and the working zoom themselves, both times before I did.
They have pushed back correctly on several of my claims.

**The failure mode to avoid**: stating a verdict from one bad run. Four times in
one session a confident negative ("neon mush", "abstract mush", "no visible
effect", "the models were shit") traced back to a misconfigured test — wrong
resolution, wrong step count, a half-loaded checkpoint. **Measure the
configuration before blaming the model.**

Corollary: when a result looks like garbage, suspect the harness first. Check
resolution, step count, and whether weights actually loaded (count the tensors)
before concluding anything about the method.

## SESSION CHECKPOINT: full-system test on the squid motif

**Where the whole chain stands** (this is the integration run: every fix at once):

1. **Draft — DONE** (`squid`, 10 shots). Scene plan 3 scenes (3/4/3): ocean floor
   → bioluminescent jellyfish field → glowing algae bloom; one figure (diver's
   helmet). Two writing attempts (story clean both), coverage repair fired,
   9→10 extension, prompt pass 10/10, anchors bound 10/10, story `EddddTTffR`,
   3 held movements. **Coverage 0**, one figure.
2. **Integration bugs the full run caught, fixed at source**:
   - **Palette leak**: the plan brief showed `palette: warm-gold`, and the
     writer/prompt pass copied "<colour> palette" into **10/10 prompts**. Fixed:
     `_scene_plan_text` no longer shows palette to the writer; `_prompt_issues`
     now flags "grade palette" as a prompt defect; the squid store was repaired
     (9 leaked names stripped, scene grades re-mapped rich-green / cyan-night /
     warm-gold).
   - **Neutral grades**: director relabelled scenes (planN → movement ids) before
     adoption, so the label-keyed palette map missed and all store scenes fell back
     to `neutral`. Fixed: `edit_planner.Shot` carries `palette`; `anchor_plan`
     sets it per shot; `film_store.shots_from_proposal` grades each scene by
     majority of its shots' planned palettes. Tests added (relabel-safe adoption,
     palette-leak prompt check). **286 tests pass.**
3. **Previs — DONE** (`output/previs/squid_previs/`): baseline-first then
   per-scene palettes; **within a scene luma 1.6 / CLIP 0.889** over 7 cuts
   (holds continuous), **scene cuts luma 9.2** over 2 cuts (grade difference, not
   an error); shot CV 0.28 at adoption. Look is very diffuse — underwater haze +
   the checkpoint's softness, visible in the contact sheet; the grade arc
   (green → cyan → warm) is legible.
4. **Retime — DONE**: study-shaped rhythm, CV **0.277 → 0.605** (human 0.75),
   holds `4.05 / 1.23 / 1.59 / 1.36 / 4.46 / 0.94 / 1.28 / 1.62 / 4.95 / 4.55 s`,
   RAFT x4 on the three scene openings + x6 on the resolve, R9 clean.
5. **Render — DELIVERED BUT REJECTED ON LOOK.** `make_film.py --store squid
   --name film` completed (10 shots, 416 frames, 26.0 s, grades rich/cyan/warm,
   audit CV 0.605, zero technical warns) -- but the film is unusable: the
   "diver's helmet" rendered as glowing helmet-creature heads and the squid as
   vague masses. Root cause: prompt assembly (full anchor block fronted in every
   shot) + plan figure was a glowing prop, not a person (both fixed above).
   Deliverable WITHDRAWN (`project.deliverable = {}`); the render is kept on disk
   as evidence, not as a film.
6. **Now: probe render with rewritten prompts.** 10 prompts rewritten to name
   `a scuba diver in a black wetsuit and round glass helmet` and `a giant squid
   with a long mantle and eight glowing arms`, with distinct actions/framings per
   shot (backup of the failed prompts: `output/squid_failed_prompts.json`).
   Probing shots 1,5,9 through the REAL pipeline (`make_film --store squid --only
   1,5,9`, job `pwsh-82`), then `probe_sheet.py` -> `output/probe/squid_probe.png`.
   Only if those real frames show a diver and a squid does the full film render.
7. **PROBE #1 VERDICT (`output/probe/squid_probe.png`):** s01 is a real scuba
   diver (wetsuit, fins, mask) -- prompt assembly fix works. s05 ("giant squid
   with a long mantle and eight glowing arms") rendered a humanoid squid-goddess,
   and s09 ("ascends through a golden algae bloom") rendered abstract gold shapes.
   Fix: cephalopod anatomy words ("colossal deep-sea squid, long conical body,
   eight long tentacles"), no "arms", no "algae bloom"; plain "gold-lit water /
   bright surface" for the ascent. s05-s10 rewritten
   (`output/rewrite_squid_prompts2.py`), probing 5,9,10 again (job `pwsh-86`);
   full render only after those frames pass.
9. **CORRECTED FILM DELIVERED** (`productions/squid/renders/film/`, 10 shots,
   416 frames, 26.0 s): contact sheet `output/audit/squid/contact.png` shows a real
   scuba diver (s01-s04), cephalopod mantle/tentacles (s05-s08), diver ascending in
   god rays (s09) and rising into light (s10); grades rich-green / cyan-night /
   warm-gold; probe-verified frames `output/probe/squid_probe_final.png`.
   Audit: **CV 0.605**, prompt->frame mean 0.315 / margin -0.001, flags are all
   info (`anchored-prompts`, one `style-cut` s04->s05 luma 12.6, resolve imagery).
   Deliverable set to `squid/film`; the failed helmet-creature render is
   overwritten (prompts backed up in `output/squid_failed_prompts.json`).
   Remaining honest limits: the look is dreamlike/soft (checkpoint), s10 renders
   as a close underwater ascent rather than a literal above-water surface, and
   the sub-1 s accent tail is still absent (16-frame takes).
10. **WORKFLOW CHANGE (important):** look decisions are made ONLY on real probe
    frames (`make_film --only` + `probe_sheet.py` / `frame_strip.py`). Previs
    remains a continuity/prompt-alignment check, never a look gate. The 2-hour
    squid render is the proof.
11. **FRAME-BY-FRAME REVIEW (the real test).** `frame_strip.py --store squid
    --render film` produced `output/inspect/squid_film/*`: a full-film every-4th
    frame strip and EVERY frame of the longest holds. It exposed what contact
    sheets hid: the 4-6 s RAFT "long holds" were 70+ frames of the same soft
    image (mean consecutive diff 1.09-1.61), i.e. dead slow-motion -- the
    procedural-hold design is REJECTED for static AI takes. `film_motion`
    (retime `--max-upsample 2`, no interpolated holds, 16.4 s, 263 frames) brings
    per-shot motion back up (2.16-3.00) with real frames, but the frames still
    show the deeper issue: the model does not follow action semantics. s01
    descends (good); s02 = diver standing facing camera with hand at chest, NOT
    "gloved hand touches a coral branch" and no coral; s03 = diver floating
    upright, arms out, no coral/no swimming. So the sequence is generic diver
    poses, not the encounter. This is prompt adherence / checkpoint quality
    (4-step dreamshaper8), not a pipeline bug: take frames are sharp (squid s01
    sharpness 506 vs 181 for the old ego takes). The next real lever is
    generation quality/adherence (more steps, better checkpoint, or best-of-N
    selected by CLIP against the prompt), not more editing machinery. **Always
    review frame_strip output before calling a film good.**
**Full-system run summary**: the integration run exercised the writing loop
structure (2 attempts, coverage repair, 9→10 extension), scene-plan light+palette,
relabel-safe grade adoption, previs baseline/within-vs-across split, study rhythm +
per-shot RAFT upsample, probe-first look gate, and audit classification. It caught
three real bugs (palette leak into prompts; plan grades lost to relabeling;
previs-not-a-look-gate), all fixed at source and covered by tests (286 pass).
See the scoped PROVENANCE verdict below: prompt TEXT quality was human-authored
for this film, so that remains the open system question.
**PROVENANCE -- IMPORTANT, do not over-read this film as a system test of the
WRITER.** All 10 prompts in the delivered `squid/film` were written BY THE AGENT
BY HAND (two rewrites: `output/rewrite_squid_prompts.py`, then
`output/rewrite_squid_prompts2.py`, then s09/s10 edited in place). The system
supplied the concept, scene plan, beats, coverage/holds, grades, timing and every
render/finish/audit stage -- but NOT the final prompt text. Reason: the writer's
anchored prompts rendered helmet-creatures, and hand-rewriting was the fastest way
to get a watchable film. Consequence: **the film validates the render/finish/
rhythm/grade chain with human prompts; it does NOT validate that the fixed writer
loop now produces legible prompts unaided.** That is the next test: a fresh draft
with the new assembly + person-figure brief, inspect its prompts and probe one
shot before rendering. If it still fails, the fix is the prompt-pass model/
instructions or an explicit human prompt-editing gate -- not more pipeline.

**Honest look caveat (delivered corrected cut)**: coherent and watchable, but
dreamlike/soft: the squid is stylised anatomy, s10 resolves underwater rather than
above the surface, and there is no sub-1 s accent tail. That is checkpoint
softness + 16-frame takes, not the pipeline.

**Full-system verdict (scoped correctly)**: the integration run exercised and
validated: writing loop structure (2 attempts, coverage repair, extension),
scene-plan light+palette, relabel-safe grade adoption, prompt assembly/leak
checks, previs baseline/within-vs-across split, study rhythm + per-shot RAFT
upsample, probe-first look gate, and audit classification. It caught three real
bugs (palette names leaking into prompts; plan grades lost to relabeling;
previs-not-a-look-gate), all fixed at source with tests (286 pass). Prompt TEXT
quality for a new concept remains the open system question.

**PREVIS IS NOT A LOOK GATE.** Measured the hard way on `squid`: previs stills
(4-step SD15) looked murky-but-plausible; the real AnimateDiff film came back as
glowing helmet-creature heads. They are different pipelines -- previs is valid for
continuity, prompt alignment and "is the cut structurally sane", NOT for "will the
film look like this". Look decisions must be made on REAL take frames:
  1. `make_film.py --store <slug> --only 1,5,9`  (one shot per scene, ~11 min each)
  2. `python probe_sheet.py --store <slug> --shots 1,5,9`
     -> `output/probe/<slug>_probe.png`: three real frames per probed shot with the
     prompt underneath. Judge/iterate on that, then render the full film.
The previs `--expect` subject check remains as a weak automatic backstop, not the
gate. The squid failure also exposed prompt-assembly design errors (below).

**ALSO FIXED AFTER THE SQUID FAILURE (look/prompt layer):**
- prompt assembly: place anchor SHORT at the front, the shot's action sentence
  early, light as a TAIL; figure appended only when absent. The old code fronted
  the full place+figure+light block, and every shot in a scene began with the same
  ~20 words.
- scene plan `figure` must be a PERSON/character, never a prop or a light
  ("a diver's helmet glows" was the squid figure; the model drew a helmet head).
- prompt pass instructed to start with ACTION + figure, keep place short, light at
  the end, never copy a repeated anchor block.
- prompt problems now flag grade-palette leakage ("<palette> palette" in prompt).


---
## 2026-11-04 audit fixes (agent b137bda2)
Done, with tests (310 pass in `tests/`):
- studio crash: `st.cache_data(ttl)` race -> lock-guarded `_ttl_cache` in `ui/studio.py`; `load` uncached.
- #1 palette: `story.shot_dict`/`validate` now carry `palette` (was wiped by the re-validate after `anchor_plan`).
- #2 anchors: `_ANCHOR_TRAILING` extended (only/against/near/...).
- #4 take aliasing: `film_store.archive_takes` moves old takes to `takes/_archive/<stamp>/<id>/`; `draft.py` calls it before adopting (refuses if the move fails).
- #6 false restale: `rebuild_hash` tolerates takes predating `upsample`/`camera_move` (`ADDED_FINISH_KEYS`), only while the current value is the neutral default. Real data: restale 105 -> 35 (remaining are genuine upsample/seconds changes).
Open, needs the user's decision: #3 scene assignment (content-vs-budget; current scorer follows its documented content rule), #5 reorder-in-hold re-render (3 failing `recon_test/test_film_store.py` checks).
Next: #7-#13, then unread modules (edit_planner, director, camera_direction, handheld, looks, grade_match, effects, continuity, conditioning, review, taste, style_pass, retime, output_audit, pacing_*).
User decisions: legacy `film_project`/`--from-project`/`workspace/` stay untouched ("LEAVE IT").

## 2026-11-04 audit fixes, part 2 (#7-#13, hygiene, second pass)
Fixed + tested (suite: 334 tests, recon_test film_store/store_render pass):
- Studio: thread-safe `_ttl_cache` (thumb/strip_frames), settings/effects save only when scenes changed, previs gate applies only when produce would generate.
- film_store: `_write_json` retry + direct-write fallback (Windows PermissionError), `archive_takes`, `rebuild_hash` legacy tolerance, single take scan per shot, crc32 seeds.
- draft: ShowrunnerFailure reported cleanly; old takes archived to `takes/_archive/<stamp>/` before adopting a new draft.
- make_film `--frames` override is no longer persisted; job_runner progress regexes match real log lines.
- review.signals tuple bug; output_audit uses `best_take`; sequence_ab `cut_continuity` (was shadowing module); camera_ab_visual reads `frames/`; style_pass `--keep-fog` works.
Open (needs user decision): effects._aberration vertical shift is dead (fix changes visuals); effects chain not in rebuild_hash (UI label only); luma/CLIP thresholds differ between previs/output_audit/gate; review/output_audit/style_pass/retime/taste not wired into job_runner.TASKS.
Known residual: draft.py/run_store write back stale snapshots at job end (mitigated by job lock + UI disabling).
Untouched by decision: legacy film_project, --from-project, workspace/, ui_legacy. Broken & untouched: main.py (missing ui.app), distill_cnn.py (IndentationError L73).

## 2026-11-04 part 3: decisions applied, studio tools, showrunner streaming/salvage
- Decisions: `_aberration` fixed (diagonal radial fringe); effects NOT in rebuild_hash (user: no); thresholds unified in `continuity.HOLD_*` (luma 10, hist 0.80, CLIP 0.45) used by previs, previs.gate, output_audit; review/film_audit/style/retime are `job_runner.TASKS` with Studio panels (shoot: review, cut: audit, write: style+rhythm). taste was already in the CUT view.
- `output_audit --report-only` (runner treats non-zero exit as FAILED). New tasks launch with `python -u`.
- story.py: `chat_streamed` (real token progress, clock starts at first token) and `salvage_shots`/`parse_reply(salvage=True)` (showrunner only: a cut-off reply keeps its complete shots). job_runner parses `showrunner attempt N/M: T/B tokens, R tok/s, up to X min left`; Studio shows it as the outer bar.
- studio "create it" refuses an existing slug (it overwrote film2/project.json once: timeline+concept lost; shots/takes/renders intact; `_fx_ui/project.json` is a near-copy, NOT restored - ask user).
- Not done (user did not pick): lock clean shots on repair attempts. Open: `visual organisation unavailable: KeyError: 'diagnostics'` in Write view (seen on lighthouse).
- Suite: 343 tests pass; recon_test film_store/store_render pass.

## 2026-11-04 part 4: the film reads its own writing
- story.py: prompt_flaws (rules: absence wording, figure missing from closing run), critic_flaws (mistral nominates repeats/off-concept/undrawable/stalls, bounds-checked, max 5), critique_and_repair (per-shot rewrite, kept only if no worse; rule-flagged absence falls back to _strip_absence). Runs after the prompt pass, before anchors. write_shots_ensemble(self_critique=, critic_model=); draft.py --no-self-critique/--critic-model (ON by default); unfixed rule flags become 'critic:' advisories; job_runner parses 'critique round'/'repair shot'. Cache config carries self_critique (old entries count as False).
- Measured on real skydiving prompts (mistral, 256 s, 2 rounds): absence phrases removed, but a small model patches words rather than the story; the stalled tail (figure absent in the ending) is NOT solved by per-shot rewrites. Previs is a still-image proxy only and is not part of this loop.
- Tests: tests/test_self_critique.py (10). Full suite 353 passed.


- part 4b (same day): per-shot repair REPLACED by one batch call (_repair_batch: whole list, flagged shots marked >> with their problems, returns {shot: prompt}); critic (critic_read) now lists the concept's story events first and can flag out-of-order / missing-step. Round 2 only runs if the rules still flag something. Real-model check on the bad 16-shot skydiving draft (read-only, 354 s): inserted the missing leap + free-fall, fixed a landing-in-mid-air shot; round 2 had damaged shot 12 (hence the new guard). The 'skydiving' production still holds the old draft; not re-applied.


- part 4c: order-aware self-critique. plan_events (concept ONLY, before any shot is seen) -> read_events (model locates first/last shot of each GIVEN event; ~5 answers; per-shot labels and object-per-shot were measured unusable on mistral/granite) -> order_flaws (plain code: backwards events, missing events placed in the slot between neighbours; unplaceable ones reported, never guessed) -> ONE _repair_batch call (whole list, flagged shots marked, 'MUST SHOW event n'). Keep-best: a round that leaves more problems is reverted. An unreadable reading flags nothing. Real draft check (skydiving, read-only): event order now correct; it flagged 1 shot, the repair was poor and was reverted. KNOWN MISS: a 7-8B model does not notice the plane-door shot (8) after the landing. Tests: tests/test_self_critique.py. Stale file productions/skydiving/shots/s16.json is an orphan (not in timeline) from the previous draft. 'skydiving' still holds the weak 15-shot draft from job 20261104-152753-draft (it ran BEFORE this order-aware version).



## Part 5 - renderer tiers (fast vs full)
- Root cause of the bad skydiving visuals: Lightning 4-step at guidance 1.0 (negative prompt ignored) draws extra limbs / T-poses on a figure. DreamShaper is not the problem (full sampler on it draws a correct figure). Plain SD1.5 was NOT tested properly (variant D produced noise, inconclusive). AnimateDiff v3 needs SD1.5-family bases only.
- Step count (s05, one seed, 8 frames, `output/audit/skydiving/test_s05_CE.png`): 12 full steps = clean figure; 8 full steps = giant deformed purple wing blob. Not equal -> `FULL_STEPS = 12` (user rule: not the same quality -> do the complete one). One seed only.
- Code: `film_store.render_tier/render_tier_tag/FULL_STEPS/GEN_NEUTRAL`, `sampler` key in `generation_inputs` (legacy takes stay current only for fast shots), per-shot `quality` override ("fast"/"full"/"auto"); `make_film.sampling_plan`, tier-aware `Generator` (rebuild ~1 min on tier switch), tier printed per shot. Tests: `tests/test_render_tier.py`; suite 371 passed.
- Auto rule puts ALL 15 skydiving shots in full (every prompt names the jumpsuit) = ~12 h CPU at 16 frames. Not yet rendered. 16-frame motion of the full tier is unverified (the 8-frame tests barely moved).


## Part 6 - character, keyframes, Studio surfaces
- Skydiving full-tier re-render (s01,s04,s05,s09-s11,s14): anatomy fixed, BUT the figure changed gender (man -> woman by s11/s14) and suit colour drifted. Cause: every prompt says "a skydiver"; `anchor_plan` takes the figure from each SCENE's own plan wording. Fix: project `character` {descriptor, avoid} -> `film_store.set_character/apply_character` put the descriptor before the subject noun in every shot prompt; `avoid` is added to the FULL sampler's negative (fast = guidance 1.0 ignores negatives); `avoid` is a generation input. `draft.py --character/--avoid`; drafts reuse the project's character.
- s04 prompt was cut off mid-phrase (writer defect) -> rewritten by hand.
- Keyframes: `keyframe.py` (full-sampler stills, choose, `--animate` test); `make_film.Generator.clip_from_still` (video-to-video from an approved still). s05 test: figure held perfectly for 16 frames but the clip is ~static at strength 0.7 (34 min). Higher strengths untested. Previs untouched (its stills use the 4-step sampler).
- Studio: shot view has a `render quality` select (auto/fast/full, shows tier + cost); Settings has `the character`. Keyframes are NOT in Studio yet (not proven useful). Tests: tests/test_studio_controls.py.
- Open: green suit in green scenes (negative should help, unconfirmed); clips barely move; render job 20261106-063653-produce (all 15 stale) in progress when written.


## Part 7 - previs gate retired
- User: previs is not the video, so what is it for? Agreed: its stills use the 4-step sampler (the one that draws extra limbs) and the clip does not start from them. Studio `announce` no longer gates or auto-starts previs; `make_film --previs-check` is opt-in info only (the old `--no-previs-check` flag never worked: the code read `previs_check`). `previs.py`, `previs.gate()` and the optional write-view panel remain, relabelled. jynx previs (12:11 PM) showed shots 10-13 failing (blank/smeared) -- real, but unrelated to the gate decision; jynx has no character set and no takes yet.
- Studio fixes this session: length box accepts holds < 1 s; per-tier ETA/price; character panel + self-check toggle on write view. Studio must be RESTARTED after code edits (Streamlit keeps imported modules).

## Part 8 - a chosen take stands (stale semantics)
- `hold_reference` (which take of the anchor a shot was conditioned on) no longer decides staleness: `generation_hash` substitutes the take's recorded value. It is shown as a note (`fs.hold_note`) in the shot panel. Changing an anchor's chosen take never re-shoots its descendants.
- A take whose text/framing/subject/etc. differs from the shot now is still `stale`, but `fs.take_drift` names what differs and the shot panel offers **use this take as-is** (`fs.adopt_take`): frames untouched, the take's inputs are replaced with the current ones, the original record is kept under `adopted` in take.json. A later edit makes it stale again.
- squid: 6 shots wanted re-shooting -> 2 (s01, s09, awaiting the adopt button). Studio must be restarted (film_store changed). Tests in tests/test_render_tier.py + tests/test_studio_controls.py; recon_test/test_store_render.py §6 updated to expect 0.

## Part 9 - aMUSEd look for first frames (branch `amused-look`)
- `keyframe.py --look amused`: aMUSEd-256 still (256 px, 12 steps, CFG 10) -> upscaled -> DreamShaper img2img at 0.5/0.65/0.8 (full sampler). Each strength is its own candidate `c<n>.png` (+json: look, strength, seed); raw aMUSEd kept as `m<n>.png`. Numbering continues, nothing overwritten.
- Studio: shot panel "first frame" (look, count, make candidates, pick one, go back). Job task `keyframes` in job_runner. Picking one sets `shot["keyframe"]` -> full tier, shot re-shoots from that still.
- aMUSEd weights were NOT fully cached; first run downloaded the rest (~12 files). Looks tested on squid s05 only: see output of that run; the diver in the prompt never appeared in any candidate.

### Part 9b - how prompts reach aMUSEd (measured, skydiving s01/s04/s05/s10/s12, aMUSEd only, same seeds)
- aMUSEd-256 uses a CLIP text encoder, 77 tokens; our prompts are 26-46 tokens, so length is not the issue. Order and the negative are.
- A raw+avoid-negative: good figures on s10/s12, no figure on s01. B raw, no negative: s10 became a snail-like body, s12 got a green suit. C subject-first, no negative: figure on s01 but washed out. D subject-first + avoid negative: best on all five. D is the default (`keyframe.amused_prompt`, avoid list from `fs.character_avoid`); the exact prompt is stored on each candidate's json (`amused_prompt`).
- Sheets: output/audit/amused_humans/matrix.png (A,B,C rows) and matrix_D.png. DreamShaper stage on human shots not yet run.

### Part 9c - animating a chosen still (skydiving s01 c1, all runs same seed, full tier, ~43-48 min each)
`keyframe.py --animate 1 --mode {v2v,ip,ramp,ip+ramp}`; frames to output/audit/skydiving/kf_s01_c1[_ip|_ramp|_ipramp]/, sheet compare.png.
- v2v (strength 0.7, `clip_from_still`): figure held through 16 frames, almost no motion (step diff 3.6). Cause: input = 16 identical frames and the first 4 of 12 (high-noise, motion-deciding) steps are skipped.
- ip (IP-Adapter 0.7, from noise): moves (step 7.6) but composition drifts to a fisheye window view, frame 0 is not the still, a doubled body appears mid-clip.
- ramp `(1,0.6,0.3,0.1)` (`make_film.anchor_callback`): frame 0 == still (err 1.7), frame 3 still the diver, from frame ~6 a hard cut to an empty mountain slope. The anchor ends too abruptly; the model cannot bridge the gap.
- ip+ramp: same hard cut, then a fisheye view with 2-3 divers. Not usable.
- Not tried: a long soft ramp (e.g. 1 -> 0.35 across all frames), v2v strength 0.85-0.9, and v2v on an input that already moves (the still with a slow zoom/pan, `camera_path.apply_path`) -- the cause of the static clip is identical input frames.
- "step diff" is not comparable across rows when a clip contains a scene cut.
