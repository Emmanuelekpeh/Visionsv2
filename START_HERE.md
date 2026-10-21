# Start here

## The studio

```
streamlit run ui/studio.py        # running now at http://127.0.0.1:8514
```

**Three verbs, not seven stages.** The old workbench had STORY, PLAN, CAMERA, GRADE,
RENDER, TIMELINE, AUDIT — seven pages in a line with five gates, so you could never
see a change and its consequence at once, and every creative decision was asked for
before a single pixel existed. The studio organises by what you are *doing*:

| view | what it is | one primary action |
|---|---|---|
| **write** | say what the film is. One field. Then read the shot list it produced. | write the shot list |
| **shoot** | the whole film as a strip you can see at once, then one shot at a time. This is where 95% of the time goes, so it is where the space goes. | shoot it |
| **cut** | look at it, then export. Refuses while a shot has never been shot. | export the film |
| *settings* | scenes, palettes, the shake, the writers' raw material, provenance, history. All real, none of it on the main path. | — |

`ui/` contains that one file. The two previous UIs are in `ui_legacy/` with
`_superseded` in their names, and nothing imports them.

**Note on how it stays up.** A server started as a managed background job exits
immediately with *"Port already in use"* if one is already listening, and one started
as an ordinary child process dies with its parent shell — the same constraint the job
runner documents. The instance on 8514 was started detached (`Start-Process -PassThru`)
and is verified listening on pid 6652. Check `netstat -ano | Select-String ':8514'`
before believing a URL is live.

## The page keeps itself fresh

Nothing needs reloading. The view is a self-refreshing fragment that re-reads the
store from disk every 2 seconds, and the job panel does the same.

Two things made that necessary, and the second was costing real time:

- **A job writes into `productions/<slug>/` from another process**, so nothing in the
  browser session knew the shot list had changed. Watching a draft finish and then
  having to reload by hand was the symptom.
- **The scouts printed nothing until all of them had finished.** Three farms run
  serially, so a ~3-minute run produced no output at all and could not be told from a
  hang. `story.py` now announces each farm, flushed, and the runner turns that into a
  real bar: `1/3 scouts`.

Two details worth keeping if you touch it. The fragment takes the **slug**, not a
project dict — passing a dict loaded outside it would hand it a 4-second-old project
and defeat the whole point. And it clears only the project cache, not
`st.cache_data.clear()`, which would also throw away every thumbnail and re-decode the
images twice a second for nothing.

2 seconds is chosen against the supervisor's 1.5-second `state.json` write: polling
slower than the writer is what makes a moving bar look stalled.

## Making it faster

One shot is ~10 minutes and roughly **half of that is fixed cost per job**, measured
with `recon_test/bench_shot.py`:

| | measured |
|---|---|
| pipeline load (7 GB, 14 motion LoRAs) | **57 s** |
| the preflight: 28 UNet passes, 2 per LoRA × 14 | **179 s** |
| fixed total per job | **236 s** |
| across 6 separately-started shots | **24 min of pure overhead** |

The preflight was the single biggest cost in the system and it was being paid on every
job. It is now **cached against the weights**: the result is keyed by the sha1 of the
exact bytes it measured, so it is paid once and skipped while those bytes are
unchanged.

```
COLD total load+preflight: 234.6s
WARM total load+preflight:  86.1s      <- 148.5s saved per job
```

The check is **not** removed, because the thing it protects against is real
(HANDOFF §9.4: a LoRA that silently fails to attach produces a clip that is merely
AnimateDiff, and the null result gets blamed on the method). It is only skipped when
the bytes are identical — and it still fires when they are not:

```
one stale entry -> 96.6s   (re-verified exactly that one LoRA; the other 13 skipped)
```

Delete `output/film/preflight_cache.json` to force a full re-check.

**The second lever is batching.** Every job that needs the model pays the 57 s load
before its first frame, so shots started one at a time pay it once each. In **cut**
there is now *shoot the next N as ONE job*, which pays it once for all of them. It
picks the shots that actually need rendering — with shot 2 already done and 1, 3, 4
missing, "the next 2" means 1 and 3, not 1 and 2.

**A bug that batching exposed, and which would have cost a real render.** `--only 1,3`
generated those two shots, then fell through to the finishing loop, which walks the
**whole** timeline and raises on the first shot with no take yet:

```
SystemExit: shot s02 has no take after this run and is needed for assembly
```

That message is right for its own purpose — an assembled film does need every shot —
and wrong as the outcome of asking to shoot two of them. You would have paid ten
minutes of generation and lost the run. A partial run now generates, lands the takes,
says what is still missing, and stops. `test_store_render.py` test 6 covers it with a
film whose unrendered set is deliberately non-contiguous.

**The raw frame count is now a SETTING, defaulting to 12.** Measured with
`recon_test/bench_frames.py`, one loaded pipeline, same prompt, same seed, same camera
LoRA, so the frame count is the only variable:

| raw frames | diffusion | RAFT | total | s/frame | align | coherence |
|---|---|---|---|---|---|---|
| 16 | 684.4 s | 9.7 s | **695.4 s** | 42.8 | 0.3463 | 0.0240 |
| 12 | 501.7 s | 7.0 s | **510.0 s** | 41.8 | 0.3484 | **0.0155** |

**27% faster, and better on both quality metrics** — alignment marginally up,
coherence (drift away from the first frame) down by a third, which is the direction
that means "steadier". Motion magnitude 7.05 → 8.09 and subject travel 15.4 px →
12.0 px, so there is no loss of movement.

The important structural finding: **per-frame cost is identical** (42.8 vs 41.8 s).
The speed-up is entirely from doing fewer frames, which also means it scales
predictably — 12 frames is 75% of the diffusion work of 16 and comes out 27% faster
overall because RAFT and finishing are nearly free.

Caveats, stated rather than buried: this is **one prompt and one LoRA**, so treat it
as a strong signal and not a law; and frames are delivered by nearest-index stretching
(12 → 23 after RAFT → 51 delivered, against 16 → 31 → 51), so whether the extra
duplication shows as judder is something only watching the clip will tell you. The
contact sheets are at `output/frames_ab/sheet_16.png` and `sheet_12.png` — the 12-frame
draws read at least as well.

Set it in **settings → raw frames per shot**. It is part of the generation hash, so
changing it correctly marks existing takes as needing a re-shoot rather than silently
mixing frame counts in one film. To pin one production at 16, set it back in settings,
or pass `--frames 16` once.

**What is NOT worth doing**, already measured and rejected — do not retry these:

- **8 frames.** 422 s against 552 s for 16 — only 24% faster — and align fell 0.368 →
  0.313 with subject travel 14.8 px → 9.0 px. Rejected on motion quality, which is why
  12 was the point worth measuring rather than 8.
- **8-step Lightning.** 2.55× slower, less prompt-accurate, and 2.6× *less* stable
  than 4-step. There is no useful midpoint.
- **256px.** Tiled green banding. This caused three false verdicts in one session.

The remaining saving with real upside is **fewer shots**, and it is linear: 3 shots
instead of 6 halves the whole render. That is a writing decision, not a compute one —
which is why the `shots` control sits next to the concept with its price on it.

## Two bugs found by actually running it

Both were mine, both only appeared on the path a real user takes, and both were
invisible to the tests that existed at the time.

**1. The story job wrote to a cache nothing read.** `story.py` writes a proposal to
`output/film/proposals/` and prints it. Nothing adopted it into the project you were
looking at, so pressing "write a shot list" ran a real 4-7 minute ensemble and left
the page identical. Fixed: `draft.py` runs the ensemble **and adopts the result into
the production in the same process**, then reports which shot ids it wrote.

**2. `next_take_name` was handed a shot record where it wanted a shot id.** The take
path became `.../takes/{'id': 's01'}` and raised

```
TypeError: unsupported operand type(s) for /: 'WindowsPath' and 'dict'
```

— **after the diffusion pipeline had loaded**, which is 3-5 minutes of work thrown
away, on the exact branch a first render takes. Every test passed strings, so none
of them touched it. Fixed at the source: every store function that takes "a shot" now
accepts a record *or* an id (`film_store._sid`), and `test_film_store.py` test 8 and
`test_store_render.py` test 5 both exercise the record form. Test 5 stubs `Generator`
so the expensive branch is reachable in a second and therefore actually tested.

**3. The writers' output is cached by concept, and the page did not say so.**
`write_shots_ensemble` keys its cache on the concept slug, so pressing the button
twice with the same concept returns the *same proposal, word for word*, in a second
or two. A cache hit was indistinguishable from a fresh run. Fixed: the write view now
says, before the button, whether a proposal for this concept is on disk, when it was
written, and whether you are about to get it back or get a genuinely new one.

**4. The layout was capped at 1180px.** `layout="wide"` was set and then undone by a
`max-width` on the container, so a 1920px window showed the entire app in a narrow
centre column with the right 40% empty. A measure belongs on prose, not on a grid:
the container is now uncapped and `.studio-prose` carries the 68ch measure for the
few places that are actually read as text. `test_studio.py` test 6 asserts on the
**active CSS rule** — its first version failed because the explanatory comment still
contained the buggy value, which is a good reminder that a test can read the
explanation instead of the code.

## The 18 ideas, measured

The complaint was "all the scout ideas are repetitions of the same thing". The
obvious metric disagrees, and that is the interesting part:

```
ideas: 18   word-overlap at 0.42: 0 dropped
word overlap: max 0.21, median 0.00 — LEXICALLY DIVERSE
```

Word overlap cannot see it, because repetition would be at the level of the
**motif**, not the wording. One run did produce five alleys out of six:

```
deep alleyway under neon, shifting shadows, fog rolls in
alley way lit by flickering firelight, puddles reflect broken moonlight
alleyway bathed in red light from street lamps, mist clings to walls
damp alley under neon, puddles reflect cobblestones, streetlamp glare
alley lit by yellow parking lot lights, graffiti etched into damp brick
```

Different words, the same shot, five times. Word overlap passes all of it.

**But that run was the exception, not the rule — and the brief is not the cause.**
`recon_test/farm_ab.py` runs three models against both briefs across two farm angles
and scores the rule in its own terms (distinct *places*, parsed from the locative
phrase in each idea):

```
shipped  ideas=36  named a place=17  distinct places=17  (100% distinct)
spread   ideas=35  named a place=12  distinct places=12  (100% distinct)
VERDICT: the rule does NOT meaningfully change location diversity
```

Adding *"each idea must be in a DIFFERENT PLACE"* did not raise diversity — it
lowered the number of ideas that name a place at all (17 → 12), i.e. it made some
output worse. **So `story.py` was left alone.**

The architectural fix is different: **the 18 ideas are not supposed to be on your
screen.** They are machinery — material for the showrunner, which is why more of them
helps the writer and hurts you. The studio shows the shot list, the artifact, and the
raw material lives in Settings for the one time in ten you want to steer the writer by
hand. That is the whole of the "it doesn't feel intuitive, just packed" fix.

## What to read

| file | what it is |
|---|---|
| [`AUDIT.md`](AUDIT.md) | what was wrong with the old data flow, measured rather than asserted |
| [`REDESIGN.md`](REDESIGN.md) | what CapCut, Premiere, Resolve, FCP, Kling, Runway and Flow do; what was taken and what was dropped |
| [`video-tool-ux-research.md`](video-tool-ux-research.md) | the consumer and AI-tool research, with sources |

## What holds it up

| file | what it is |
|---|---|
| `ui/studio.py` | the studio. The design rules are in its header — read them before adding a control |
| `film_store.py` | the production store: stable shot ids, immutable takes, derived staleness |
| `draft.py` | write a shot list and adopt it into the production, in one job |
| `make_film.py` | the renderer. `--store <slug>` writes new takes; `--from-project` is the old path |
| `job_runner.py` | detached jobs, structured progress, one at a time |

## Verify it

```
python recon_test/audit_ui.py            # measures the OLD flow (read-only)
python recon_test/test_film_store.py     # stable ids, takes, derived staleness
python recon_test/test_store_render.py   # the whole render path with no diffusion model
python recon_test/test_studio.py         # the design rules, as assertions
python recon_test/idea_diversity.py      # word overlap AND motif saturation
python recon_test/farm_ab.py             # does the writers' brief cause repetition?
```

The second one matters most: every check in it is a bug that was reproduced in the
old model, written as an assertion so it cannot come back.

## Known gap in verification

No human has looked at the studio's pixels. Browser automation on this machine is
unreliable: Chromium's Mojo IPC and Node's `child_process` both need named pipes,
which the confined sandbox denies, and `agent-browser` — which does launch — reports
a successful navigation and then hangs on every subsequent command, writing blank
frames. So the layout is verified **structurally** (every view under `AppTest`, with
the design rules as assertions in `test_studio.py`) and **not visually**. Treat the
spacing and hierarchy as unproven until someone opens the URL and says otherwise.

## Still missing

1. **Audio.** The export is silent. `add_narration.py` works but is not wired to the
   store, and there is no music bed. For a YouTube deliverable this is the largest gap.
2. **A scrubbable preview.** The viewer shows four frames of a take, not a player.
   Streamlit cannot host a timeline playhead.
