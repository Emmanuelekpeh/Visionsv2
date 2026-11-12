# VISION HANDOFF — you are the visual half of the Visions film system

You are taking over the part of this project that requires **seeing rendered
frames**. The previous agent could not ingest images, so every visual verdict
below is yours to make from the actual frames — do not trust any prior written
conclusion about what the frames look like.

## 1. What this project is

Visions (repo `C:\Users\emman\Downloads\visions`) is a text → video film pipeline:

- **Write** — an LLM ensemble (Ollama) writes a shot list: per shot a `prompt`,
  `scale`, `camera`, `beat`, `purpose`, `link`.
- **Shoot** — each shot is rendered with AnimateDiff (Stable Diffusion 1.5-family
  checkpoint, 4-step Lightning, 16 raw frames at 512px) + optional motion LoRA,
  then RAFT optical-flow upsampling, handheld shake, grade.
- **Cut** — shots assemble into an MP4.

Durable data model: `productions/<slug>/` contains `project.json` (timeline),
`shots/<id>.json` (one file per shot), `takes/<shot>/<take>/` (immutable renders:
`take.json` + `frames/f000.png … f015.png`), and `renders/` (assembled `film.mp4`
+ `stats.json`). Raw generated frames live in `takes/<shot>/<take>/frames/` and are
16 per take.

## 2. Division of labor

- **You own the visuals.** Judging limb/body-plan coherence, whether a frame
  matches its prompt, whether the assembled film is watchable — all "is this
  good?" verdicts are yours, from the pixels.
- **The text side is already handled elsewhere** (writer loop, prompt design,
  story/beat validation). Don't re-do it. If a prompt is bad, say so — but the fix
  for bad text is not your job.

## 3. The goal

The user is reorganizing the system into a **self-organizing loop with a proper
feedback loop** so it reliably produces reasonable videos. There was a recent
function regression, and `ERROR_TESTS.md` turned the complaints into test cases.
**Do not treat that write-up as settled truth — it contains unverified claims. Your
job is to confirm or refute them against the actual frames.**

## 4. Priorities (these override the write-up)

1. **LIMB / BODY-PLAN DRIFT is the #1 problem.** Extra arms, feet, fins that appear
   across frames is an immediate turn-off for a viewer. This matters most.
2. **Softness/blur is NOT a real problem on its own.** A blurry-but-watchable,
   *consistent* frame is acceptable. Sharpness only matters if it breaks
   consistency or legibility. Do not recommend reverting something just because the
   image is soft.
3. **Any "confirmed" label in the write-up is a hypothesis** until you verify it on
   the actual frames.

## 5. What to verify

### Task 1 — C1: limb / body-plan drift (PRIORITY)
Open these raw frames (3 per set: f000 = frame 1, f008 = frame 9, f015 = frame 16).
For each set, state literally what limbs/body parts the human figure has in each
frame, and whether the count or body plan changes across the three frames.

- **Set A (claimed coherent control)**
  `productions\ego\takes\s01\t01\frames\f000.png, f008.png, f015.png`
  prompt: "A barefoot man walks across a dusty city floor, a cracked concrete patch
  reflecting his unsettled gaze; the shadow of his ego looms large against a
  moonlit forest, leaves rustling around him as the confrontation begins."
  ⚠️ The write-up claims this was a "distant silhouette walk" — it was NOT; the
  prompt is a two-image composite. Verify what is actually in the frame.

- **Set B (claimed drift — close diver)**
  `productions\squid\takes\s01\t03\frames\f000.png, f008.png, f015.png`
  prompt: "a scuba diver in a black wetsuit and round glass helmet descends through
  deep blue water, fins kicking, green coral below"

- **Set C (close, one hand raised)**
  `productions\squid\takes\s04\t02\frames\f000.png, f008.png, f015.png`
  prompt: "the scuba diver stops and turns toward a huge dark shape beyond the
  green coral, one hand raised"

- **Set D (hand touches coral)**
  `productions\squid\takes\s02\t02\frames\f000.png, f008.png, f015.png`
  prompt: "the scuba diver's gloved hand touches a glowing green coral branch on
  the ocean floor, silt drifting in the water"

For each set output: limb count per frame, whether it changes, and a verdict of
**COHERENT** (same body plan across all three) or **DRIFT** (limb count / plan
changes).

### Task 2 — C3: prompt match
For Set D (and any other squid shot worth checking): does the frame show what the
prompt names? Specifically — is there a hand physically TOUCHING coral, or just a
diver floating? Is there any coral at all?

### Task 3 — the assembled film (the real viewer-facing test)
Look at these and judge whether the film reads as one coherent
diver-encounters-squid sequence, or whether figures drift / change identity /
change limbs between shots and within holds:

- `output\audit\squid\contact.png` — one frame per shot, prompt under it
- `output\inspect\squid_film_film_strip.png` — every-4th-frame strip of the
  delivered film
- `output\inspect\squid_film_hold_s05.png`, `..._hold_s09.png`, `..._hold_s10.png`
  — every frame of the longest holds

### Task 4 — C4 is numeric; don't use eyes for it
"Dead holds" (frames barely changing) is a measurement (consecutive-frame
difference), not a visual judgment — the text side can compute it. Flag it only if
you see something a number can't capture (e.g. a figure frozen mid-action that
should be moving).

## 6. Ground rules

- Describe what is actually in the pixels, not what the prompt claims.
- If a frame is too soft/abstract to judge, say "can't judge" — don't guess.
- Verdicts in a table: `| set | figure | limb count f000/f008/f015 | verdict |`.

## 7. Deliverable

Answer: (1) which sets are COHERENT vs DRIFT, with the evidence you saw; (2) whether
the prompt-match failures in Task 2 are real; (3) whether the assembled film is
watchable, and if not, exactly which shot(s) break it; (4) a ranked list of what to
fix first, treating limb drift as #1 and softness as low priority.
