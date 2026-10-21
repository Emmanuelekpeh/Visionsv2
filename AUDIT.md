# The workbench audit — why the data flow feels inconsistent

Read-only investigation of the current UI. Everything below was **reproduced**, not
read off the source: `recon_test/audit_ui.py` loads the two real projects, renders
all seven stages under `streamlit.testing.v1.AppTest`, and prints the project
against the disk. Run it with:

```
python recon_test/audit_ui.py
```

It generates nothing and writes nothing.

**Headline.** All seven stages render without raising. There is no crash to fix.
The problem is that **there are three sources of truth for the same facts and they
disagree**:

| who claims authority | says what |
|---|---|
| `proj["shots"]` (what STORY edits) | the writer's shot list, cameras as *suggested* |
| `proj["plan"]["frozen"]` (what renders) | the director's/planner's rewrite of it |
| `output/film/<name>/` on disk | what actually exists, as pixels |

The UI reads all three in the same stage, side by side, with **positional** joins
between them. So the same shot shows one camera in STORY and another in PLAN, and an
"approve" written in one stage is read from a different object in another.

---

## 1. `shots` and `frozen` are two copies, and they disagree

Measured on `ensemble1_workbench`:

```
shot 0: camera='none'     -> 'none'      seconds 2.1 -> 3.2   DIFFERENT
shot 1: camera='pan-left' -> 'none'      seconds 2.8 -> 2.44  DIFFERENT
shot 2: camera='tilt-up'  -> 'zoom-in'   seconds 1.9 -> 1.68  DIFFERENT
shot 3: camera='zoom-in'  -> 'none'      seconds 2.3 -> 3.6   DIFFERENT
```

Every shot differs. That is *correct behaviour* — the director overrode the writer's
camera suggestions and the planner rewrote the holds — but the UI presents both as
if each were the shot's camera. `stage_story` renders `d.get("camera")` from
`proj["shots"]`; `stage_plan`'s frozen-plan table renders `d["camera"]` from
`plan.frozen`. Same shot, two cameras, and nothing on screen says which one wins.
**The render uses `frozen`**, so what you read in STORY is not what you get.

The writer's suggestions are also still *live inputs*: `shot_objects()` feeds
`proj["shots"][i]["camera"]` back into `director.choose` on every re-plan, and the
director overwrites it every time. So the value is neither the author's intent nor
the director's decision — it is a suggestion that silently steers the next plan and
is then discarded.

## 2. The RENDER stage writes approval to one object and reads status from another

`sync_render_state` and the approve button both write `proj["shots"][i]["render"]`:

```
ui/app.py:683   for i, d in enumerate(proj["shots"]):
ui/app.py:690       d.setdefault("render", {})["status"] = want
ui/app.py:768   proj["shots"][i].setdefault("render", {})["status"] = "approved"
```

The table that displays status reads it from `frozen`:

```
ui/app.py:730   "status": (proj["shots"][i].get("render") or {}).get("status", ...)
                ... for i, d in enumerate(frozen)
```

That one happens to index `shots` by the `frozen` position — so it is not broken
when the two are the same length, it is broken *as soon as they are not*. And the
row's other columns (`d["scale"]`, `d["camera"]`, `d["seconds"]`) come from `frozen`
while the status comes from `shots`: one row, two objects, joined by integer index.

## 3. Everything downstream is joined by array position — so order is data loss

Four separate settings are stored or read *by index*, not by identity:

| setting | stored as | consumed as |
|---|---|---|
| `camera.overrides` | key `str(i)` over `proj["shots"]` order | `fp.shake_params(project, i)` where `i` is the **frozen** index (`make_film.py:691`) |
| `grade.looks` | a flat list, built from `frozen` order (`ui/app.py:604`) | `looks[i]` where `i` is the **frozen** index (`make_film.py:717`) |
| `motion_pin` | `proj["shots"][i]` | read as `frozen[i]`'s pin (`ui/app.py:370`, `376`) |
| `render` | `proj["shots"][i]` | `frozen[i]`'s status (`ui/app.py:730`) |

TIMELINE's reorder mutates `plan.frozen` in place (`ui/app.py:851-862`). After one
reorder:

- shot N's **camera override** now applies to a different shot;
- `grade.looks` no longer lines up with the scenes, because it was built from the
  *old* order — and `make_film` re-derives `looks` per scene position, so a move
  across a scene boundary changes that scene's palette;
- shot N's **approval** now reads as shot M's;
- a **motion pin** follows the array, not the shot.

The project file confirms the pins live in a different order from the render:
`pin_test` has shot 0 `camera='none'` with a `pan-left` **pin**, and `frozen[0]`
`camera='pan-left'` — for shot 0 that lines up; shot 1 has the pin and `frozen[1]`
is `none`. Reorder once and both silently move.

This is the single most consequential defect: **`shots` and `frozen` are parallel
arrays with no identity, and every settings map is keyed by position in one of
them.** There is no way for any of these joins to be checked.

## 4. Nothing has an identity, so nothing can accumulate

There is no stable key for a shot and no key for a render. Consequences:

- **A re-render overwrites the previous one.** `make_film.py:678-681` writes
  `raw/shot{i+1:02d}/f{j:03d}.png` in place. The generator burns
  **540-788 s per shot** (`ensemble1/stats.json`; 9-14 min measured). One
  regeneration erases the take you were comparing against, so there is no
  "try another camera move and pick the better one".
- **Two layout conventions coexist in the same tree.** `ensemble1/` holds 175 flat
  `f*.png` at its root *and* a `raw/` subdirectory; `ensemble1_pershot/` is properly
  per-shot (`raw/shot01/`, `smooth/`, `shaken/`); `_assemble_test/` is flat with
  empty `raw/`/`shaken/`. `shot_stages()` only understands the per-shot layout, so
  for `ensemble1` it reports 0 raw frames while 175 sit in the same folder.
- **The film record points at the flat one.** `proj["film"]["name"] == "ensemble1"`,
  so the project is attached to the layout the per-shot UI cannot read. RENDER asks
  you to render into `ensemble1` again — which would **overwrite the 175 frames of
  the only finished film** (`film.mp4` + `stats.json` on disk).
- **The UI cannot find media it does not already know the name of.** TIMELINE has
  an "attach an assembled film that already exists" control (`ui/app.py:874-881`)
  because there is no asset index. That is a workaround for the missing concept.

## 5. The stage status is a hand-maintained string that contradicts itself

`proj["status"]` is written from six places and read by nothing that acts on it:

```
{"story": "approved", "plan": "approved", "camera": "edited",
 "grade": "empty", "render": "empty", "timeline": "assembled"}
```

Grade and render are `"empty"` while a graded, assembled 10.94 s film exists and is
attached. The values are inconsistent by construction (`"written"`, `"planned"`,
`"approved"`, `"edited"`, `"set"`, `"stale"`, `"reordered"`, `"assembled"`,
`"empty"`, `"pending"`), the two STAGES lists disagree (`film_project.STAGES` has
six entries, `ui/app.STAGES` has seven with `AUDIT`), and the only thing the UI does
with it is print it in the sidebar. Meanwhile the state that *does* matter —
staleness — is enforced by `touch_story()` clearing the plan, so every edit resets
approval and the user is pushed back through gates. That is the "inconsistent" the
user is feeling: **status is decorative, invalidation is real, and they are not the
same mechanism.**

## 6. There is no audio path at all, and the deliverable is silent

`make_film` writes with `audio=False`. `add_narration.py` exists and works
(ollama writes lines, edge-tts speaks them, moviepy muxes them) but **nothing in the
UI calls it** — it is CLI-only and wired to the `make_montage` direction format, not
to a project. `output/audio/` has 13 timestamped `line_000_*.mp3` files from manual
runs. For a tool whose purpose is watchable YouTube videos, "export" currently means
a silent MP4.

## 7. The workflow shape itself is the wrong shape

Beyond the bugs, the flow does not match how a film gets made:

- **Seven stages in a line, five gates.** STORY -> PLAN -> CAMERA -> GRADE -> RENDER
  -> TIMELINE -> AUDIT. Each stage is a page; the artifact you just made is on the
  next page. You approve a plan you have never seen rendered.
- **All the creative decisions are asked for before the first pixel exists.**
  Camera, grade and hold are decided in PLAN/CAMERA/GRADE against nothing. A
  first-time user reaches RENDER, spends ~45 minutes of generation, and only then
  learns whether the cut works.
- **Feedback is per-stage and late.** There is a camera A/B GIF and a palette A/B
  PNG, but no way to look at the film's cut rhythm — no animatic, no storyboard
  pass, no contact sheet at the point of decision. TIMELINE's contact sheet comes
  after the render.
- **No iteration loop.** Because re-rendering overwrites (§4) and edits invalidate
  the plan (§5), the tool offers "accept" or "start over", not "try again".
- **Nothing shows the cost before you spend it.** ~10 min/shot is discovered from
  the stats *after* assembly (`ui/app.py:913`).
- **The job board is one job at a time with no queue.** With 12 shots at ~10 min
  each, the useful action is "queue shots 3-12 and walk away"; the UI offers
  render-this-one or render-all-remaining.

---

## What is genuinely good and must survive the redesign

Not everything here is wrong, and the redesign should not throw these away:


1. **`film_project` + `--from-project` is the right idea.** What you freeze is what
   renders; the render does not silently re-plan. That contract is the foundation.
2. **Per-shot stages on disk** (`raw/` -> `smooth/` -> `shaken/`) and
   `--assemble` regenerating nothing took assembly from >10 min to 97 s. Keep it.
3. **The exposure sheet.** `handheld.hold_frames` -> `stretch_to` is a real,
   shared definition of the timeline, and the plan's holds (51/39/27/58) match the
   rendered film exactly. This is the project's best idea and it is currently
   buried in stage 6.
4. **The job runner.** Detached, structured JSON progress, a hard one-at-a-time
   lock, and `_alive()` reporting ALIVE on failure rather than risking a second
   render. The Windows traps it avoids (`os.kill(pid, 0)` kills; `tasklist /FI` does
   not list) are real. Keep it and put a queue on it.
5. **Every number comes from the pipeline's own functions.** The UI reimplements no
   measurement. Do not break this in the rewrite.
6. **A/B previews before spending compute.** The camera GIF and palette PNG are
   exactly the right instinct, at exactly the wrong place in the flow.
