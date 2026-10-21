# The redesign — what video tools actually do, and what this one now does

Companion to `AUDIT.md` (what was wrong, measured) and `film_store.py` (the fix).
This is the design: the conventions taken from professional and consumer tools, the
one place this machine forces a different answer, and how to use the thing.

Sources: `video-tool-ux-research.md` (CapCut / Descript / Kapwing / VEED /
Clipchamp, and Kling / Runway / Veo-Flow / Pika / Luma / Sora / Higgsfield) and the
NLE report on Premiere Pro, DaVinci Resolve, Final Cut Pro and Avid Media Composer.

---

## 1. The single finding that decides the design

> **Every NLE treats the timeline as the source of truth and every render as a
> disposable cache.** There is no global "stale / must regenerate" state. Playback
> and export re-evaluate the same edit decision list; the render cache merely
> remembers the last evaluation.

That is the opposite of what this project had, and it is *nearly* right here. The
old workbench had it backwards in a specific way: it made the *plan* the frozen
source of truth and the *shot list* a parallel copy, joined by array index. So the
thing that was supposed to be authoritative (`frozen`) and the thing the user edited
(`shots`) drifted, and every per-shot setting was addressed by position in one of
them. `AUDIT.md` §1-3 measures the drift.

But the NLE answer cannot be adopted whole, because of one number:

| | Premiere / Resolve | this pipeline |
|---|---|---|
| cost of changing a clip | re-read a file from disk | **540-788 s of CPU per shot** |
| render cache | disposable | **the most expensive artifact in the project** |

So the design takes the NLE's *structure* — timeline as truth, non-destructive
edits, takes as first-class — and inverts its *economics*: here a take is a durable
asset that is never overwritten, and the UI's most important job is to tell you the
price of the next change before you commit to it.

That is what `film_store.py` encodes:

- **A shot has a stable string id and its own file.** Order lives only in the
  timeline. Reordering moves no camera, no palette, no take, no approval — the bug
  that was reproducible four ways.
- **A take is an immutable directory.** `takes/<shot>/t01`, `t02`, … Re-rendering
  writes the next one.
- **Staleness is derived, never stored** — by hashing the actual dependencies:
  `generation_hash` (prompt, camera, scale, subject, seed, model → needs the model)
  and `rebuild_hash` (hold, shake, palette, scene palettes, in-point → needs only a
  re-finish). No status string to fall out of date; undoing an edit clears it by
  itself, because a hash of the real dependency is reversible and a counter is not.
- **Two prices, always shown.** `film_status()` returns `to_generate` (minutes) and
  `to_refinish` (seconds) separately. The old UI had one number and it was in a
  stats file after the fact.

## 2. The conventions taken from the tools

| convention | from | how it appears here |
|---|---|---|
| **Four-zone layout**: assets left, viewer centre, inspector right, timeline bottom | Premiere, Resolve, CapCut, FCP — all four converge on it | `ui/workbench.py`'s three columns + the timeline beneath |
| **Timeline is the project**; renders are views of it | every NLE | `project.json` holds `timeline: [ids]` and nothing else about order |
| **Non-destructive edits**; the project references media, does not contain it | every NLE (EDL / AAF / FCPXML) | `productions/<slug>/` references takes; `export_plan()` is the EDL |
| **Takes / auditions / version stacks** | FCP *auditions*, Resolve *take selector* | `takes/<shot>/tNN`, choose one, approve one; the viewer lists them and "use" switches the cut |
| **Offline media is a visible state**, not an error | Premiere "Media Offline", Resolve relink | a shot whose take file is gone reads `stale`, not "missing" — the film never silently shortens |
| **State badges on clips** (cached / rendered / offline) | Premiere render bar, FCP dotted line | `[ ] [~] [!] [x] [*]` per shot in the assets pane and the timeline |
| **Exposure sheet** — drawing identity separated from frame timing | traditional animation, and this project already had it | `handheld.hold_frames` → `stretch_to`, shown as in/out/hold in the timeline |
| **Cost shown on the button before you commit** | Kling prints the credit price on Generate; Kapwing returns a rejectable estimate | the inspector states "about N min (measured from M takes)" before the Generate button |
| **Non-blocking generation queue** | Runway streams generations into a session while you keep working; *every other tool surveyed blocks* | `job_runner` + the always-present queue panel; the page never awaits |
| **Scene as the structural unit** | Descript renders one Scene as script + timeline + storyboard simultaneously | `scenes[]` carry the palette; the assets pane groups by scene |

## 3. The conventions deliberately dropped

- **Modal tool modes** (Avid) and **page-level fragmentation** (Resolve's pages): the
  old workbench already had seven pages and that was the problem. There is one
  workspace with three panels and a mode switch for scenes/delivery.
- **Track targeting vs linked selection vs sync lock** (Avid's three separate
  concepts): this pipeline has one video track and no audio edit, so all three
  collapse to nothing. Adding a fake audio track to look professional would be
  exactly the "honest stub vs fake control" mistake the old code warned about.
- **1980s vocabulary** (splice-in, segment mode, sync lock).
- **A magnetic timeline** (FCPX): it is the single most-complained-about onboarding
  wall in the category, and this cut is 4-12 shots, not 400.

## 4. What is new here that no surveyed tool has

**The two-price model, and the re-finish path.** No AI tool surveyed distinguishes
"this change needs the model" from "this change needs a re-grade". Because this
pipeline keeps the per-shot ladder on disk, a retime or a palette change is a
seconds-long re-finish. Modelling that explicitly is what turns

> edit → 10 minutes → look

into

> edit → 2 seconds → look

for everything except prompt, camera, scale and seed. It is the single biggest
workflow win available on this hardware, and it falls out of the NLE's
"render cache vs deliverable" split applied honestly to a pipeline where the cache
is expensive.

## 5. How to use it
```
streamlit run ui/workbench.py
```

**One-time, to bring an old project across:**

```python
import film_store as fs
fs.import_legacy("workspace/ensemble1_workbench.json", slug="shadows")
```

That reproduces the frozen plan exactly: `shadows-take1` reports 10.94 s / 175
frames for `ensemble1`, which is what the rendered film actually is.

**The loop.**

1. **make** panel → set the concept and the shot count → *write a shot list*
   (~4-7 min, runs detached). Shot count is the budget: 9-14 min per shot.
2. Shots appear in **assets** grouped by scene. Select one; the **inspector** edits
   its prompt, camera, scale, hold, scene. The **timeline** below shows the cut with
   real timecodes and lets you reorder.
3. Press **generate** on a shot. The queue panel shows progress and the log tail.
   You can keep editing while it runs — but only one render at a time, and the
   runner will refuse a second and say why.
4. When a take lands, the **preview** shows five frames from it. Generate again for
   a second take; the preview lists them and **use** switches which one the film
   cuts to. Exactly one take is never thrown away.
5. Changing a palette or a hold does *not* re-generate: the inspector says so, and
   the **deliver** panel re-finishes during export.
6. **deliver** → *export*. It regenerates nothing and refuses while a shot needs
   the model — an export cannot invent frames.

**What is honestly missing**, and should not be faked:

- **Audio.** `film.mp4` is silent. `add_narration.py` works (ollama writes lines,
  edge-tts speaks them, moviepy muxes) but is not wired to the store. A music bed
  and a mix panel are the largest remaining gap for a YouTube deliverable.
- **Scrubbing.** The preview shows a frame strip, not a scrubbable player per shot.
  Streamlit cannot host a timeline playhead; wiring the strip to a player is the
  next step, and a real one needs a different client.
- **Trimming inside a take.** The hold is an exposure-sheet decision, not a trim.
  `--only` renders whole shots.

## 6. Where the old files are

| file | status |
|---|---|
| `ui/workbench.py` | **the new UI.** Store-backed, three panes + timeline, queue |
| `film_store.py` | **the new model.** Stable ids, takes, derived staleness |
| `ui/app.py` | superseded. Kept because `--from-project` and `film_project.py` still work and nothing is deleted without being read |
| `film_project.py` | superseded by `film_store.py`; `make_film --from-project` still reads it |
| `ui_legacy/` | the pygame studio. Its *interaction model* is what §2 already takes from it |

Nothing was deleted. `recon_test/audit_ui.py` measures the old flow,
`recon_test/test_film_store.py` and `recon_test/test_store_render.py` and
`recon_test/test_workbench.py` verify the new one.
