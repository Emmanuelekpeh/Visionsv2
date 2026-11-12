# Showrunner comparison — measured, 2026-10-29

**Question.** The ensemble's showrunner was changed to `qwen2.5:7b` and the films
seemed worse; the previous choice, `granite4.1:8b`, was blamed for malformed
JSON. Which model should write the final shot list?

**Short answer.** `granite4.1:8b` wins on what actually reaches a render:
**filmcraft 70.6/100 vs qwen 55.7 vs mistral 50.8**. Qwen is ~25% faster and
produces valid JSON, but its lists are visually monotonous — three adjacent
medium shots, one scene, uniform 2.4 s holds, and a first/last shot that are the
same image. Granite costs ~50 s more per attempt and breaks the beat order
slightly more often, but writes 2–3 scenes, varied framing and holds, and a
payoff. **The malformed-JSON claim did not reproduce in the ensemble path — and it
reproduces exactly in the single-writer path.** Asking granite for 16 shots at
the old 1200-token cap wrote exactly 1200 tokens and stopped mid-object; the same
request at 3000 returned 15 shots as clean JSON. It is a token-budget bug, not a
syntax habit — see §4.

Everything below is reproducible with `compare_showrunners.py`; the raw data is
in `output/showrunner_test/20261029-193839/` (gitignored).

---

## 1. Method — and three bugs the first version had

`compare_showrunners.py` runs candidate showrunners on **the same concept** with
**the same idea pool**, repeated, and records the raw replies.

| | |
|---|---|
| concept | `within the shadows, friends and foes appear the same` |
| idea pool | 18 ideas collected ONCE from `qwen2.5:3b`, `phi3:latest`, `mistral:latest` (6 each), then injected identically into every call |
| showrunners | `granite4.1:8b`, `qwen2.5:7b`, `mistral:latest` |
| samples | **3 runs each** (`--runs 3`) |
| shots requested | 6 (`--n-shots 6`), temperature 0.8 |
| retries | **0** — one attempt per run, i.e. the raw model, not best-of-3 |
| renderer | 16 raw frames → 31 delivered frames at 16 fps = 1.94 s of motion per shot |

The first version of the comparison script (from the previous session) produced
numbers that could not support a decision:

1. **The farms were re-run for every model.** Farms are non-deterministic, so
   each model was handed a different idea pool and the comparison measured farm
   luck. Fixed by collecting the pool once and injecting it (`farm_out=`).
2. **It read scorecard keys that do not exist** (`writer_story_breaks`,
   `total_repairs`, `planner_r1_scale`), so every model printed zeros. Fixed.
3. **It ran each model once.** HANDOFF §11.3 already measured that n=1 is
   meaningless here (structural repairs 0, 0, 2 across three runs of the *same*
   model). Fixed with 3 runs and per-run spreads.

One thing was also added: a **filmcraft score**, because sorting on the wrong
column picked the wrong model. See §3.

---

## 2. Results

### Per model (3 runs each)

| model | craft | runs | story breaks (per run) | warnings | rule breaks | jump cuts | scenes | 1-scene rate | self-rep | provenance | holds total | stretch | wall |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **granite4.1:8b** | **70.6** | 3/3 | 1.0 (2, 0, 1) | 0.67 | 2.67 | 0.67 | 2.67 | 0.00 | 0.05 | 0.94 | 14.1 s | 1.22× | 200 s |
| qwen2.5:7b | 55.7 | 3/3 | 1.0 (1, 0, 2) | 0.00 | 4.33 | 2.33 | 1.67 | **0.67** | 0.14 | 0.83 | 14.2 s | 1.22× | **150 s** |
| mistral:latest | 50.8 | 3/3 | 1.67 (1, 1, 3) | 0.67 | 5.67 | 1.33 | 4.33 | 0.00 | 0.11 | 1.00 | 15.7 s | 1.36× | 173 s |

All nine runs returned valid JSON (`parse=raw`). `stretch` is the delivered hold
over the generated motion (1.0 = no duplicated frames); see
[`PACING_AND_FLOW_RESEARCH.md`](PACING_AND_FLOW_RESEARCH.md).

### Every run, so the spread is visible

| model | run | shape | breaks | warns | rules | jump cuts | scenes | self-rep | provenance | holds (s) | wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| granite | 1 | `EETdRR` | 2 | 1 | 4 | 1 | 3 | 0.023 | 1.00 | 2.0 2.5 2.2 2.8 2.3 2.1 | 300 s |
| granite | 2 | `EdTTRR` | **0** | 1 | **1** | **0** | 2 | 0.068 | 1.00 | 2.1 2.3 2.5 2.7 2.2 2.4 | 148 s |
| granite | 3 | `EdTdRR` | 1 | 0 | 3 | 1 | 3 | 0.044 | 0.83 | 2.0 2.5 2.2 2.8 3.0 1.8 | 150 s |
| qwen | 1 | `EdTddR` | 1 | 0 | 5 | **3** | **1** | 0.041 | 1.00 | 2.0 2.5 2.2 2.4 2.1 2.6 | 226 s |
| qwen | 2 | `EdTTTR` | **0** | 0 | 4 | **3** | **1** | **0.206** | 1.00 | **2.4 × 6** | 102 s |
| qwen | 3 | `EETddR` | 2 | 0 | 4 | 1 | 3 | 0.183 | 0.50 | 2.5 2.0 3.0 2.5 2.0 2.5 | 122 s |
| mistral | 1 | `EdTdTR` | 1 | 1 | 4 | 2 | **6** | 0.082 | 1.00 | 3.6 2.1 1.8 2.4 2.7 3.0 | 250 s |
| mistral | 2 | `EdTdRR` | 1 | 1 | 3 | 1 | 2 | 0.009 | 1.00 | 2.8 3.4 2.1 2.5 3.1 2.7 | 130 s |
| mistral | 3 | `EdTdRE` | 3 | 0 | **10** | 1 | 5 | 0.225 | 1.00 | 3.6 1.8 2.7 2.4 3.0 1.5 | 138 s |

Read the spread, not only the mean: **granite and qwen both average 1.0 story
break per film, but granite has one clean run and two slips, qwen the same. The
difference is not the story; it is the filmcraft below.**

---

## 3. Why the score says granite, read off the shot lists

The `filmcraft` number is a **judgement with printed weights**, not a
measurement: −20 per story break, −6 per warning, −5 per jump cut, −10 per film
with only one scene, −10 × self-repetition, −15 × (1 − provenance). The weights
are in `FILMCRAFT_PENALTIES` in `compare_showrunners.py` so they can be argued
with. Three examples are enough to see whether the weights are sane.

### qwen run 2 — the trap of "no story breaks"

Shape `EdTTTR`, zero story breaks, one scene, three jump cuts, all six holds
exactly 2.4 s, self-repetition **0.206** (the highest in the study):

> 1. **establish** wide 2.4 s — *A lone figure in a moonlit garden, statue and person indistinguishable…*
> 2. **develop** medium 2.4 s — *Faded mural on crumbling brick wall, rain smears…*
> 3. **turn** medium 2.4 s — *Group huddled in dim light…*
> 4. **turn** medium 2.4 s — *Cracked mirror reflecting twilight…*
> 5. **turn** medium 2.4 s — *Lone figure in a crowded market…*
> 6. **resolve** wide 2.4 s — *A lone figure in a moonlit garden, statue and person indistinguishable…*

Shots 2–5 are all `medium` (three jump cuts), every hold is identical, every
prompt ends with the same *"cinematic, filmic grain"* tail, and **shot 6 is shot 1
again** — a loop, not a resolution. Declared beats are correct; the film does not
develop. This is precisely the failure `validate` was built to catch and cannot,
because the beats are in the right order.

### granite run 2 — a film with a payoff

Shape `EdTTRR`, zero story breaks, no jump cuts, two scenes, varied holds:

> 1. **establish** medium 2.1 s — *deep alley at dusk… a woman cautiously walks*
> 2. **develop** close 2.3 s — *cracked mirror… reflection shows a different figure*
> 3. **turn** medium 2.5 s — *group huddled in dim light… the woman steps into the center*
> 4. **turn** wide 2.7 s — *faded mural… the mural twists revealing her reflection*
> 5. **resolve** medium 2.2 s — *…her reflection merges with the shadows*
> 6. **resolve** wide 2.4 s — *crowded market… the masks dissolve revealing her own face*

The mirror in shot 2 is paid off in shots 4–6; the recurring woman is present;
scales alternate medium/close/medium/wide/medium/wide. This is a shot list you
would spend an hour rendering.

### mistral run 3 — the leak, again

Ten problems, including every prompt carrying fused field values:

> *faded mural on crumbling brick wall, rain smears, shadowy figures, **energy:
> 0.05, scene: environment 1, subject: recurring figure***

The image model draws *"energy: 0.05"* as text; this is the `prompt_leaks`
failure the pipeline already measures (HANDOFF §11.2/§11.3), and it also ends the
film on `establish` with beats going backwards. Mistral is out on this evidence.

---

## 4. "Granite was giving malformed JSON" — reproduced, and it was truncation

**In the ensemble path it does not happen.** All five granite calls in the main
test (three runs + two stress calls) returned **strictly valid JSON**, parsed in
`raw` mode — no fences, no repairs. A 16-shot stress request returned 8 valid
shots at the 1200 budget (769 tokens) and 6 at 3000 (560 tokens); the model
simply stopped early rather than being cut off.

**In the single-writer path it reproduces exactly.** `write_shots` — used by
`make_film --writers single`, `llm_scorecard.py` and `writer_audit.py` — asks
one model to write everything. Asking granite for 16 shots at the old
`num_predict=1200`:

```
=== num_predict=1200 wall=279s chars=4576 eval_count=1200 mode=FAIL ===
failure: JSONDecodeError: Expecting ',' delimiter at line 113 col 6
         (13 open vs 11 close brackets) TRUNCATED
tail: '... "scale": "medium",\n      "camera": "none",\n      "seconds": 2.2,\n
       "scene": "4",\n      "energy": 0.25,\n      "'
```

`eval_count=1200` — it wrote until the cap and stopped mid-object. The same
request at 3000 returned **15 shots / 1790 tokens as clean JSON** (raw).

So the memory is explained: **the "malformed JSON" was a token cap, not a syntax
habit.** `format=SHOT_SCHEMA` makes Ollama emit grammar-constrained JSON, which is
valid unless the reply is cut off; a cut-off reply has unbalanced brackets, and
no comma rule can repair it. This is also why the repair layer added in the
previous session measured **zero uses**: it treats a failure that is not the one
happening.

**The fix, now in the code:** `story.predict_budget(n_shots)` scales the
completion budget with the request (`max(2000, 400 + 160*n_shots)`, capped at
6000) and both writer paths use it. A flat 3000 would still truncate a 24-shot
request (the UI allows 24), which is the same bug at a larger number. The budget
is a cap, not a target, so a generous ceiling costs nothing when the model
finishes early.

What also changed today to make this diagnosable:

* `story.parse_reply()` returns the parse **mode** (`raw`/`fenced`/`braces`/
  `repaired`) and `story.json_failure()` names truncation explicitly
  (`TRUNCATED` when open brackets outnumber close ones).
* `Proposal` now carries `reply_text`, `parse_mode`, `attempts`, `eval_count`.
* Every attempt's raw reply is saved under
  `output/showrunner_test/<stamp>/replies/`, and a total failure raises
  `ShowrunnerFailure` carrying the replies instead of a bare string.
* Regression test:
  `tests/test_story_parse.py::test_predict_budget_scales_with_the_shot_count`.

**If it recurs**, the raw text will be on disk with a line saying whether it was
truncated or syntactically broken. Do not add a repair rule for truncation.

### Side note: what to do about a model's bad habit

The measured bad habits here are not JSON at all:

* **qwen's habit** is degenerate framing — three adjacent `medium` shots, one
  scene, uniform holds, looped first/last image. A repair layer would hide this;
  the right response is to measure it (done: jump cuts, 1-scene rate,
  self-repetition) and let it cost the score.
* **mistral's habit** is fusing sibling field values into `prompt`. The pipeline
  deliberately records this rather than silently stripping it
  (`story.prompt_leaks`), because the same failure will resurface later as
  unexplained poor alignment. The correct fix is at adoption: reject or strip
  leaked text before it reaches the renderer, and keep the measurement so the
  model choice can account for it.

Both are "what the model did" in a form a person can read and act on — better
evidence to post than a stack trace.

---

## 5. Verdict

1. **Keep `granite4.1:8b` as the showrunner.** It wins every filmcraft column
   that decides what the render looks like: jump cuts (0.67 vs 2.33), scenes
   (2.67 vs 1.67, and 2/3 qwen runs were single-scene), self-repetition (0.05 vs
   0.14), provenance (0.94 vs 0.83). The story-break mean is a tie; the film is
   not.
2. **Qwen is a fallback, not the default.** ~25% faster per attempt (150 s vs
   200 s) and never leaked in this test, but 3/3 runs had 3 jump cuts (two of
   three single-scene), which means more planner repairs and a more static film
   in the shipped path, where the writer's framing IS the cut.
3. **Mistral is out.** Worst story discipline, leaks field values into prompts,
   and switches scene on almost every shot.
4. **The story-break rate (1.0/film for the top two) is the bigger quality
   problem than the model choice.** Neither model reliably produces
   monotonically progressing beats when the validator only reports the failure
   after the fact. **Implemented after this test:** the retry is now a
   failure-specific repair prompt (`story._repair_note`: the validator's exact
   complaints + the previous beats and prompts), and `draft.py` refuses to adopt
   a proposal with `story:` breaks unless `--allow-broken-story` / the UI
   checkbox says so. A rejected draft leaves the production untouched and the
   failure banner shows the log's last lines.
5. **Caveats.** n=3, one concept, one idea pool, temperature 0.8. Re-run on 2–3
   more concepts at n=5 before treating the gap as a law. The model ranking is
   stable enough to keep granite today; the exact scores are not.

### Reproduce

```powershell
python compare_showrunners.py --models "granite4.1:8b,qwen2.5:7b,mistral:latest" `
    --runs 3 --n-shots 6 --stress-shots 16 --stress-models "granite4.1:8b" `
    --stress-budgets "1200,3000"
```

The run directory contains `results.json`, `summary.md`, the shared
`ideas.json`, one raw reply per run in `replies/`, and the stress replies in
`stress/`.
