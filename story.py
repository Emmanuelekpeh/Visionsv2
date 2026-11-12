"""
story.py -- the text layer the film pipeline was missing.

Every shot list so far was hand-written, which means the cheapest part of the
system to iterate on was never in the loop. A shot list costs seconds to
regenerate and the frames it drives cost ten minutes each, so the text layer is
where tuning actually pays.

The design is deliberately "propose, then let the system repair":

    concept  ->  LLM proposes a shot list (schema-constrained)
             ->  validated and scored against the renderer's known failure modes
             ->  director.choose()   picks a motion per shot
             ->  edit_planner.plan() enforces R1-R6 and repairs what it must
             ->  model_select        picks a checkpoint per shot, for variety
             ->  looks               picks a palette per scene

The LLM is not trusted to obey the rules; it is *measured* on how well it does,
and the repairs it forces are the tuning signal. `scorecard()` reports exactly
that, which is what makes a smaller or fine-tuned model comparable on something
other than vibe.

THE WRITING LOOP. A draft is not one model call. `write_shots_ensemble` re-asks a
rejected attempt with the validator's exact complaints and the previous shot list
attached (`_repair_note`); `beat_layout` puts the requested shot count and a
numbered beat roster in every ask, because "beats never backwards" was advice the
model could not place; `extend_shot_list` brings a CLEAN but short list up to the
requested count with coverage inside its existing scenes; and `prior` lets a
refusal from a previous RUN be fed back into the first attempt. The film the
editor sees is the end of that loop, not the first thing the model said.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from edit_planner import SCALES, Shot, describe, plan

ROOT = Path(__file__).resolve().parent

CONCEPT_DEFAULT = "within the shadows, friends and foes appear the same"

CAMERAS_OK = ["none", "zoom-in", "zoom-out", "pan-left", "pan-right",
              "tilt-up", "tilt-down", "rolling-clockwise", "rolling-anticlockwise"]

# Measured ceiling of the renderer. Every proven checkpoint (dreamshaper8, sd15,
# both running AnimateDiff) produces motion 0.50; only an unproven model claims
# more. A writer that asks for energy above this gets its shot refused by
# model_select and silently repeated -- which is what happened on the first real
# run: two of six shots were promised 0.8 and 1.0 and neither could be rendered.
MAX_RENDERABLE_MOTION = 0.5

# words that carry no image, for the overlap measures. Deliberately short -- this
# is a legibility check, not a language model. One definition, shared by the
# ensemble's provenance measure and writer_audit.py.
STOPWORDS = set("""a an the and or of in on at to for with from into over under
is are was were be been being it its this that these those as by
his her their its they them he she you your we our
shot camera frame scene cuts cut opens close wide angle view
""".split())

# CONTEXT IS A RENDER INPUT FOR THE WRITER. OLLAMA's default 4096 is smaller than
# the system brief + 19 farm ideas + a 2000-token completion. The runner then
# shifts context mid-generation (observed: n_discard 2045 at n_gen 1100), which
# both truncates the JSON and costs a full retry. 6144 holds the current prompt
# (~3k) plus a 2k completion without forcing the model into swap on this 15 GB box.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "6144"))

# The story's progression, as a first-class field. Before this existed, nothing
# in the pipeline asked the shots to DEVELOP: the brief demanded variety (do not
# repeat) and never development (do change), so the writer produced mood boards --
# six restatements of one idea with the location moving. Measured on a real
# ensemble output: "shadows on a sidewalk, no one in sight" -> "silhouettes merge"
# -> "two shapes on a bench" -> "two figures step toward each other" -> "a lone
# figure in a crowd" -> "shoes alone in a puddle". One idea, six times, and no
# check anywhere would have noticed.
#
# `fall` EXISTS BECAUSE THE MODEL KEPT WRITING IT ANYWAY. A real 13-shot draft came
# back on three attempts as `EddddddTTddRR` / `EddddddTTTdRR`: establish, develop,
# turn, then the change SETTLING into the new world, then the resolve. That is a
# four-act short film, and the falling action was labelled `develop` because the
# taxonomy had no word for it -- so the gate called it "beats go backwards
# (10:turn->develop)" and refused a coherent film three times. The beat it kept
# reaching for is real; the schema was the thing that was incomplete.
BEATS = ["establish", "develop", "turn", "fall", "resolve"]
BEAT_MARKS = {"establish": "E", "develop": "d", "turn": "T", "fall": "f",
              "resolve": "R"}

# COVERAGE, as fields. A film is not a row of postcards: it is 2-3 scenes, each
# covering ONE action/place with 2-4 shots. `purpose` is the shot's job inside
# its scene; `link` is how the cut into it is motivated. Measured on the
# productions: `seasoning` has 13 shots over 3 scene labels but 0/10 adjacent
# within-scene pairs share a subject (within-scene prompt overlap 0.05 vs 0.08
# across scenes) -- the labels were cosmetic because nothing asked for coverage.
PURPOSES = ["establish", "master", "insert", "reaction", "resolve"]
LINKS = ["open", "match-action", "eyeline", "movement", "graphic", "scene-cut"]
MAX_SCENES = 3
MAX_SCENE_CUTS = 2

# Below this stemmed overlap the resolve is treated as unrelated to the turn.
# Low on purpose: a legitimate resolve can be worded completely differently, so
# this catches disjointness, not difference.
TURN_RESOLVE_LINK = 0.08

# A PROMPT IS ONE IMAGE, AND THE RENDER MEASURED OTHERWISE. The first real rendered
# film (`ego`, 13 shots) was audited prompt-by-prompt against its takes (CLIP):
# 12 of 13 prompts were TWO-image composites -- "A barefoot man walks across a
# dusty city floor...; the shadow of his ego looms large against a moonlit
# forest..." -- and the image model drew one clause, the other, or the concept
# prior. Mean prompt->frame similarity was 0.30 against 0.25 for a DIFFERENT
# shot's prompt (margin +0.02); four frames matched another shot's prompt better
# than their own, and the film ended on a car because s13's prompt named an "old
# car" and the renderer drew the car, not the man's shadow on it.
#
# These are the measured writing defects, as rules. `prompt:` is an ADVISORY at
# the gate (a prompt can be terse and still be the right image) but it is fed to
# the repair note and to `rewrite_prompts`, which fixes the text without touching
# the story, the beats or the count.
MAX_PROMPT_WORDS = 24
ABSTRACT_WORDS = frozenset("""
aftermath awareness acceptance begins climax confrontation ego emotional fate
feels feeling inner intensifies meaning memory metaphor mood psyche realises
realizes represents resolution self-awareness subconscious surrenders symbolic
symbol tension turmoil understands
""".split())

# COVERAGE, MEASURED. The writer's brief has always asked for 2-3 scenes, each one
# place+figure covered by 2-4 shots; nothing measured it, so the first rendered
# film's 13 shots were 12 places and the director correctly montaged them. The
# measurement is the sequence director's OWN affinity (visual_org.diagnose), so
# the check and the director cannot disagree about whether a scene holds.
#
# ADVISORY at the adoption gate: a montage is a legitimate arrangement
# (SEQUENCE_AB.md), so this must not refuse a film. It is a DEMAND on the WRITING
# loop: `write_shots_ensemble(scene_plan=True)` plans the 2-3 scenes first and
# `coverage_retries` gives one re-ask when the material still comes back as one
# place per shot.
# Adjacent prompts within a scene above this stemmed overlap are the SAME image;
# a hold needs the same PLACE/figure, not the same sentence (measured 0.7+ on the
# 13-shot held film, whose previs stills were near-duplicates).
PROMPT_REPEAT = 0.65
MIN_COVERAGE_HOLD = 2
# Adjacent prompt affinity (place/figure nouns, style+atmosphere stripped) below
# this means the shots inside a "scene" are not the same place. The director's
# hybrid-pair bar is 0.18; a held scene only needs to reuse its place nouns, so the
# writing bar is lower.
COVERAGE_AFFINITY = 0.10

SYSTEM = (
    "You are a director writing a shot list for a short atmospheric film that "
    "will be rendered by AnimateDiff on a CPU-only machine.\n"
    "\n"
    "HARD LIMITS OF THE RENDERER -- these are measured, not opinions:\n"
    "- Maximum motion is 0.5. Never write energy above 0.5; the renderer cannot "
    "deliver it and the shot gets refused.\n"
    "- The renderer already drifts on its own. Prefer camera 'none'. At most two "
    "shots may move.\n"
    "- It renders what is IN FRAME. It cannot render a whisper, a memory, a "
    "feeling, a fade, or a sound.\n"
    "\n"
    "'prompt' must be a CONCRETE VISUAL DESCRIPTION of one single image: physical "
    "things, materials, light, weather, framing. At least 8 words, at most 24, "
    "ONE sentence. Never a title, a scene heading, a sequence of events, or two "
    "images joined by a semicolon. Never its meaning ('symbolic of', 'the ego "
    "surrenders', 'the aftermath') -- that is not in frame.\n"
    "\n"
    "VARIETY -- a film of near-identical shots is a failure even if each shot "
    "is good:\n"    "- At least one shot must NOT contain the recurring figure at all. An "
    "environment or a small detail gives the film air and makes the figure's "
    "return land.\n"
    "- Vary the light and the materials between shots. Do not write every shot of "
    "darkness; contrast is what makes the dark read as dark.\n"
    "- Give at least one shot a camera move (zoom-in, pan-left, tilt-up...), and "
    "leave the rest at 'none'.\n"
    "\n"
    "STORY -- the film must GO somewhere. Equal moments that never accumulate are "
    "a mood board, not a film. This is the point of the whole shot list, and nothing downstream can "
    "fix it for you.\n"
    "- Every shot carries a 'beat': establish, develop, turn, fall, or resolve. "
    "The beats must move FORWARD through that order and never backwards.\n"
    "- Shot 1 is 'establish': the world and the recurring figure as they are now.\n"
    "- At least one shot is 'turn': the moment something CHANGES -- the figure "
    "acts, a discovery lands, the situation inverts. Without a turn the film "
    "restates one idea instead of developing one. The turn is a DOOR: from the "
    "first 'turn' shot to the end, every beat is 'turn' (the change happening), "
    "'fall' (the change settling into the new world), or 'resolve' (the new world "
    "at the end). Nothing after the turn goes back to 'develop'.\n"
    "- 'fall' is the landing after the change: the room after the figure leaves, "
    "the street after the confrontation. It is not a second beginning and not "
    "more develop -- the world does not return to how it was.\n"
    "- The last shot is 'resolve': the world as it is AFTER the turn. It must not "
    "be a restatement -- something about it should be different from shot 1, "
    "and you should be able to say what.\n"
    "- 'logline' states the CHANGE, not the premise. BAD: 'a woman walks through "
    "shadows'. GOOD: 'a woman follows a figure through the shadows and finds it is "
    "her own reflection'.\n"
    "\n"
    "SEQUENCES, NOT POSTCARDS -- this is what separates a film from a slideshow:\n"
    "- Plan the film as 2 to 4 MOVEMENTS first, then write shots. A movement is a "
    "stretch with one job: open the place, escalate in rapid cuts, land the turn, "
    "resolve. Give every shot a 'movement' id (m1, m2...). Consecutive shots with "
    "the same id are one movement; keep ids contiguous, never interleaved.\n"
    "- Use 2 or 3 scenes. A scene is ONE PLACE and ONE ACTION, covered by 2 to 4 "
    "shots. Never jump to a new location on every shot; two seconds then a hard "
    "stop to new scenery is a slideshow, not a film.\n"
    "- Give every shot a 'purpose' inside its scene: 'establish' opens and orients "
    "it, 'master' covers the action, 'insert' shows a detail, 'reaction' shows a "
    "face or body response, 'resolve' closes the scene.\n"
    "- Give every shot a 'link' to the shot before it: 'open' is the first shot of "
    "a scene; 'scene-cut' is a deliberate hard stop to a new place; everything "
    "else continues the same place/action -- 'match-action' (the action carries "
    "across the cut), 'movement' (the same direction continues), 'eyeline' (someone "
    "looks, the next shot shows what they see), 'graphic' (a matching shape or "
    "colour).\n"
    "- Never interleave scenes (A-B-A). Do not give a scene a single shot unless it "
    "is the first or last. No more than two scene cuts in the whole film.\n"
    "- A camera move must lead into a hold or into the next action IN THE SAME "
    "SCENE. Never move the camera on the shot immediately before a scene cut.\n"
    "- A scene is a HOLD: inside one scene, keep the same place and the same "
    "figure -- reuse the same nouns, materials and light in every prompt. The "
    "studio checks that the wording actually holds together and will flag a "
    "scene whose shots describe unrelated places. If you want a rapid-change "
    "run instead, give each shot its own scene and link it 'scene-cut'.\n"
    "- The first shot of a scene is its longest: give the eye time to read a new "
    "place, then cut faster inside the scene.\n"
    "\n"
    "ENERGY keeps its own shape, set by your 'energy' values (the director cannot "
    "create it, because a still shot always carries the same energy whatever it "
    "chooses):\n"
    "- 'energy' (0 to 0.5) must RISE, peak, then fall away. At least one shot near "
    "0.05, at least one at 0.45 or above, the rest spread. Not all equal values.\n"
    "- The highest energy is NOT on the first or the last shot; it belongs in the "
    "middle or just past it. The last shot settles rather than stopping dead.\n"
    "\n"
    "BAD:  'Scene 1: Darkness'\n"
    "BAD:  'Whispers fade'\n"
    "GOOD: 'a bare birch trunk in fog, wet bark, low grey light, thin mist, "
    "cinematic, filmic grain'\n"
    "\n"
    "Other rules:\n"
    "- Never use the same framing scale on two consecutive shots: that is a jump "
    "cut and the planner will rewrite it.\n"
    "- 'seconds': do not agonise over it. The planner imposes a rhythm curve and "
    "will overwrite your values, so just keep them between 1.5 and 3.6.\n"
    "- 'energy' 0 to 0.5: how much movement the shot needs.\n"
    "- 'scene' is one place and one action, not a colour group. Use 2 or 3 "
    "scenes; a look change and a checkpoint change only land at a scene boundary.\n"
    "- 'subject' names the recurring person or thing, or empty string if none.\n"
    "Return JSON only."
)

SHOT_SCHEMA = {
    "type": "object",
    "properties": {
        "logline": {"type": "string"},
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "scale": {"type": "string", "enum": SCALES},
                    "camera": {"type": "string", "enum": CAMERAS_OK},
                    "seconds": {"type": "number"},
                    "subject": {"type": "string"},
                    "scene": {"type": "string"},
                    "energy": {"type": "number"},
                    "beat": {"type": "string", "enum": BEATS},
                    "purpose": {"type": "string", "enum": PURPOSES},
                    "link": {"type": "string", "enum": LINKS},
                    # the writer's own grouping into movements (m1, m2...). It is
                    # optional so old caches still validate; when present the
                    # sequence director uses it instead of re-segmenting.
                    "movement": {"type": "string"},
                },
                "required": ["prompt", "scale", "camera", "seconds", "scene",
                             "energy", "beat", "purpose", "link"],
            },
        },
    },
    "required": ["logline", "shots"],
}


def predict_budget(n_shots: Optional[int]) -> int:
    """Completion-token budget for a shot list of `n_shots` shots.

    MEASURED, not guessed. Asking granite4.1:8b for 16 shots with the old
    `num_predict=1200` made it write until the cap and stop mid-object:
    `JSONDecodeError: Expecting ',' delimiter at line 113 col 6 (13 open vs 11
    close brackets) TRUNCATED`, 1200 completion tokens. The same request at 3000
    returned 15 shots / 1790 tokens as clean JSON. That is the "malformed JSON"
    granite was blamed for: truncation, not a syntax habit, so no repair rule
    could ever have fixed it.

    Measured rate is ~110-130 tokens per shot. The UI offers up to 24 shots, and
    24 x ~120 already exceeds a flat 3000, so the budget has to scale with the
    request. It is a CAP, not a target: a model that finishes earlier stops
    earlier, so a generous ceiling costs nothing on the normal path.
    """
    if not n_shots:
        return 2000                       # ~6 shots plus the logline, with room
    return min(6000, max(2000, 400 + 160 * int(n_shots)))


@dataclass
class Proposal:
    concept: str
    model: str
    logline: str
    shots: List[Shot]
    raw: dict
    seconds: float
    decode_tok_s: float
    repaired: bool = False
    problems: List[str] = None
    used_fallback: bool = False
    cached: bool = False
    # AUDIT FIELDS. Added for the showrunner comparison: when a model's reply
    # fails to parse, the useful evidence is the raw text and WHY it failed, not
    # just a raised exception that the caller catches and forgets. These cost
    # nothing on the normal path and are what `compare_showrunners.py` reports.
    reply_text: str = ""           # the winning attempt's raw reply
    parse_mode: str = ""           # raw | fenced | braces | repaired | (empty)
    attempts: int = 0              # showrunner attempts made
    eval_count: int = 0            # completion tokens in the winning attempt
    # THE LOOP'S OWN EVIDENCE. `requested`/`returned` is the count the request
    # asked for vs what the film actually got (the 13->6 failure); `extended` says
    # the shortfall was closed with coverage rather than by re-writing the story.
    requested: Optional[int] = None
    extended: bool = False
    extend_calls: int = 0
    prompts_rewritten: int = 0
    scene_plan: Optional[Dict] = None
    coverage_repairs: int = 0
    anchored: bool = False
    prior_fed_back: bool = False

    def __post_init__(self):
        if self.problems is None:
            self.problems = []


class ShowrunnerFailure(RuntimeError):
    """Every showrunner attempt failed to parse or validate.

    Carries the raw replies so a comparison harness (and a person reading the
    evidence) can see WHAT the model emitted instead of only that it failed. A
    model whose JSON is malformed and a model whose story is unusable are
    different failures and must not collapse into one exception string.
    """

    def __init__(self, message: str, replies: List[Dict]):
        super().__init__(message)
        self.replies = replies


def _stem(w: str) -> str:
    """Crude stem, enough to see that 'silhouettes' and 'silhouetted' are the same
    thing to a viewer and different strings to a computer."""
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


def link_overlap(a: str, b: str) -> float:
    """Stemmed content-word overlap of two prompts, 0-1 Jaccard."""
    A = {_stem(w) for w in content_words(a)}
    B = {_stem(w) for w in content_words(b)}
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def story_shape(shots) -> str:
    """The film's beats as a row of letters: E establish, d develop, T turn, R
    resolve. A shape with no T is a mood board however good the individual
    images are; a shape that goes backwards is not a story at all."""
    return "".join(BEAT_MARKS.get(getattr(s, "beat", ""), "?") for s in shots)


def beat_layout(n_shots: Optional[int]) -> str:
    """The beat ROSTER the writer is asked to fill, for a given shot count.

    The measured failure this closes: `_repair_note` told the showrunner "beats
    never backwards" and it still returned E,d,d,T,d,R -- "beats go backwards
    (4:turn->develop)" on all three attempts of a real draft. The retry handed it
    the complaint and its own prompts; what it did NOT hand it was a PLAN. A text
    model tags a beat from the image it has just written, not from the image's
    position in the film, so "never backwards" is unenforceable advice. A
    numbered roster turns the rule into a position to obey.

    It also makes the requested shot count concrete: the roster IS the count, so
    "write 13 shots" becomes "shot 13 is the resolve" instead of a number the
    model can decide to ignore once the film feels finished.

    Deliberately a default, not a straitjacket -- starting the turn earlier and
    holding it for several shots is allowed. What is not allowed is develop after
    turn, or turn after resolve.
    """
    if not n_shots:
        return ("The beats must never go backwards: shot 1 is 'establish' and the "
                "last shot is 'resolve', with 'develop', then 'turn' and 'fall' "
                "between them.")
    n = int(n_shots)
    if n < 3:
        return ("A film of fewer than 3 shots cannot hold a turn between an "
                "establish and a resolve. Ask for at least 3 shots.")
    if n == 3:
        return ("The film is exactly 3 shots: shot 1 'establish'; shot 2 'turn'; "
                "shot 3 'resolve'. Beats never go backwards.")
    if n == 4:
        return ("The film is exactly 4 shots: shot 1 'establish'; shot 2 "
                "'develop'; shot 3 'turn'; shot 4 'resolve'. Beats never go "
                "backwards.")
    if n == 5:
        return ("The film is exactly 5 shots: shot 1 'establish'; shot 2 "
                "'develop'; shot 3 'turn'; shot 4 'fall'; shot 5 'resolve'. Beats "
                "never go backwards.")
    # Proportional, not fixed: develop is the first half, the turn lands by
    # ~three-quarters, the falling action settles into the resolve. The model is
    # free to move the turn earlier; what it may not do is return to 'develop'.
    d_end = max(2, int(round(n * 0.5)))
    t_end = max(d_end + 1, min(n - 2, int(round(n * 0.78))))
    return (
        f"The film is exactly {n} shots, so give it {n} beats. Beats move FORWARD "
        f"only: establish -> develop -> turn -> fall -> resolve, never back. A "
        f"safe layout: shot 1 'establish'; shots 2-{d_end} 'develop'; shots "
        f"{d_end + 1}-{t_end} 'turn'; shots {t_end + 1}-{n - 1} 'fall' (the change "
        f"settling into the new world); shot {n} 'resolve'. You MAY begin the turn "
        f"earlier than shot {d_end + 1}: from the first 'turn' on, no shot may be "
        f"'develop' again -- 'turn' while the change happens, 'fall' while it "
        f"settles, 'resolve' for the new world it leaves behind.")


def content_words(text: str) -> set:
    """The words in a prompt that carry an image, for overlap comparisons."""
    words = re.findall(r"[a-z][a-z\-]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def prompt_leaks(text: str) -> List[str]:
    """Field values fused INTO the prompt string. Detection only -- never fixed.

    `prompt` is the exact string CLIP and SD receive, so "..., yellow streetlights,
    2.1 seconds, energy 0.2" or "..., cinematic grain, camera: none" is not
    metadata the pipeline reads and discards -- it is text the image model tries
    to draw: numbers, the word "energy", a colon. A weak writer filling a flat JSON
    schema merges sibling fields into the string field.

    This is measured rather than repaired, for the reason `validate` gives above:
    silently correcting it would hide which model does it, and the leak would
    resurface later as unexplained poor alignment -- the sampler getting blamed
    for the writer. Measured on the default concept (writer_audit.py): 0 of 10
    prompts for granite4.1:8b, 6 of 12 for llama3, 6 of 13 for mistral.
    """
    found = []
    if re.search(r"\b\d+(\.\d+)?\s*(s|sec|secs|second|seconds)\b", text):
        found.append("seconds")
    if re.search(r"\benergy\s*[:=]?\s*\d", text, re.I):
        found.append("energy")
    for f in ("prompt", "camera", "scale", "subject", "scene", "logline"):
        if re.search(rf"\b{f}\s*:", text, re.I):
            found.append(f)
    return found


def _scene_runs(scenes: Sequence[str]) -> List[Tuple[str, List[int]]]:
    """Contiguous scene runs with their shot indices. A revisited scene is a NEW
    run, which is how interleaving (A-B-A) becomes visible."""
    runs: List[Tuple[str, List[int]]] = []
    for i, sc in enumerate(scenes):
        if runs and runs[-1][0] == sc:
            runs[-1][1].append(i)
        else:
            runs.append((sc, [i]))
    return runs


def structure_problems(shots: Sequence[Shot]) -> List[str]:
    """R7-R9: is this a film, or two seconds / hard stop / new scenery, repeat?

    The planner only looks at adjacent technical continuity (R1/R2/R6/R10), and
    the story checks only look at beat order. Neither can see that the film is a
    row of postcards: every shot a new place, no scene covered by more than one
    angle, a camera move that jumps to new scenery. These are the checks that
    make "sequence" mean something. They are REPORTED, not repaired -- inventing
    missing coverage is the writer's job, not the planner's.

    R7 scene structure: 2-3 scenes, at most two hard stops, no interleaving, no
    one-shot scenes except the ends.
    R8 coverage: every multi-shot scene has >=2 framing scales, a shared subject
    between adjacent shots, and more than one purpose.
    R9 rhythm: scene-opening shot is the longest in its scene; no three
    consecutive near-equal holds.
    R10 (pre-check; the planner repairs it): a camera move may not hand off to a
    new scene.
    """
    probs: List[str] = []
    shots = list(shots)
    if len(shots) < 2:
        return probs
    scenes = [s.scene or "scene1" for s in shots]
    runs = _scene_runs(scenes)
    distinct = list(dict.fromkeys(scenes))

    # ---- R7 scene structure ------------------------------------------------ #
    if len(distinct) > MAX_SCENES:
        probs.append(
            f"structure: R7 {len(distinct)} scenes -- a hard stop to new scenery "
            f"every {len(shots) / len(distinct):.1f} shot(s). A scene is one place "
            f"and one action; use 2-{MAX_SCENES} scenes.")
    scene_cuts = sum(1 for i in range(1, len(scenes)) if scenes[i] != scenes[i - 1])
    if scene_cuts > MAX_SCENE_CUTS:
        probs.append(
            f"structure: R7 {scene_cuts} scene cut(s) -- at most {MAX_SCENE_CUTS} "
            f"hard stops to a new place in a short film.")
    seen = set()
    for sc, idx in runs:
        if sc in seen:
            probs.append(
                f"structure: R7 scene {sc!r} is revisited at shot {idx[0] + 1} -- "
                f"scenes must not interleave (A-B-A).")
        seen.add(sc)
    for k, (sc, idx) in enumerate(runs):
        if len(idx) == 1 and 0 < k < len(runs) - 1:
            probs.append(
                f"structure: R7 scene {sc!r} is a single shot -- a scene needs at "
                f"least two shots to be a scene, not a cutaway.")
    # more runs than scenes IS interleaving; already reported above per revisit

    # ---- R8 coverage ------------------------------------------------------- #
    for sc, idx in runs:
        if len(idx) < 2:
            continue
        scales = {shots[j].scale for j in idx}
        if len(scales) < 2:
            probs.append(
                f"structure: R8 scene {sc!r} uses one framing scale "
                f"({next(iter(scales))}) -- give it coverage (master + insert/"
                f"reaction), not the same size on every cut.")
        subjects = [shots[j].subject for j in idx if shots[j].subject]
        if subjects and not any(
                shots[j].subject and shots[j].subject == shots[j - 1].subject
                for j in idx[1:]):
            probs.append(
                f"structure: R8 scene {sc!r}'s shots never share a subject -- they "
                f"are separate places, not coverage of one action.")
        purposes = {shots[j].purpose for j in idx if shots[j].purpose in PURPOSES}
        if len(idx) >= 3 and len(purposes) < 2:
            probs.append(
                f"structure: R8 scene {sc!r} has one purpose "
                f"({purposes.pop() if purposes else '?'}) -- use master, insert and "
                f"reaction to cover the action.")

    # ---- link consistency -------------------------------------------------- #
    for i, s in enumerate(shots):
        if not s.link:
            continue                      # already reported per-shot by validate
        if s.link not in LINKS:
            continue                      # already reported by validate
        same_scene = i > 0 and scenes[i] == scenes[i - 1]
        if i == 0 and s.link != "open":
            probs.append(f"structure: shot 1 must link the scene as 'open', not "
                         f"{s.link!r}")
        elif i > 0 and s.link == "open":
            probs.append(f"structure: shot {i + 1} is linked as 'open' but the "
                         f"scene did not change")
        elif i > 0 and s.link == "scene-cut" and same_scene:
            probs.append(f"structure: shot {i + 1} is linked as 'scene-cut' but it "
                         f"stays in scene {scenes[i]!r}")
        elif (i > 0 and s.link in ("match-action", "eyeline", "movement", "graphic")
              and not same_scene):
            probs.append(f"structure: shot {i + 1} is linked as {s.link!r} across a "
                         f"scene change -- that is a scene cut, not a match")
        elif i > 0 and not same_scene and s.link not in ("scene-cut",):
            probs.append(f"structure: shot {i + 1} changes scene but is not linked "
                         f"as 'scene-cut'")

    # ---- R10 pre-check: a move may not hand off to new scenery ------------- #
    for i in range(len(shots) - 1):
        if (shots[i].camera not in ("", "none")
                and scenes[i + 1] != scenes[i]):
            probs.append(
                f"structure: R10 the camera move on shot {i + 1} hands off to new "
                f"scenery ({scenes[i]!r} -> {scenes[i + 1]!r}) -- a move must "
                f"continue in the same scene or settle before the cut.")

    # ---- R9 rhythm --------------------------------------------------------- #
    holds = [round(s.seconds, 2) for s in shots]
    run = 1
    for i in range(1, len(holds)):
        run = run + 1 if abs(holds[i] - holds[i - 1]) <= 0.1 else 1
        if run == 3:
            probs.append(
                f"rhythm: R9 shots {i - 1}-{i + 1} hold within 0.1s of each other "
                f"-- uniform hard stops are a metronome, not a rhythm.")
    if len(holds) >= 3:
        mean_h = sum(holds) / len(holds)
        sd_h = (sum((x - mean_h) ** 2 for x in holds) / len(holds)) ** 0.5
        cv = sd_h / max(1e-9, mean_h)
        if cv < 0.20:
            probs.append(
                f"rhythm: R9 shot lengths are metronomic (CV={cv:.2f}) -- human "
                f"short-form video sits at CV~0.75 because it mixes sub-second "
                f"accents with multi-second holds; vary the holds, not only the "
                f"framing (see PACING_MEASURED_STUDY.md).")
    for sc, idx in runs:
        if len(idx) < 2:
            continue
        opening = shots[idx[0]].seconds
        longest = max(shots[j].seconds for j in idx[1:])
        if opening < longest - 0.05:
            probs.append(
                f"rhythm: R9 scene {sc!r} opens on {opening:.1f}s but a later "
                f"shot in it holds {longest:.1f}s -- the opening shot orients the "
                f"eye and should be the scene's longest.")
    return probs


def _prompt_issues(prompt: str) -> List[str]:
    """Why this prompt may become a different image than the shot intended.

    One definition, used by the `prompt:` advisory and by `rewrite_prompts`, so
    the pass that repairs text and the check that measures it cannot disagree.
    """
    text = (prompt or "").strip()
    issues: List[str] = []
    if not text:
        return issues
    words = text.split()
    if ";" in text or re.search(r"[.!?]\s+[A-Za-z]", text):
        issues.append("fuses more than one image into one prompt")
    if len(words) > MAX_PROMPT_WORDS:
        issues.append(f"{len(words)} words (a prompt is ONE image, not a "
                      f"sequence; {MAX_PROMPT_WORDS} max)")
    found = sorted({w for w in ABSTRACT_WORDS
                    if re.search(rf"\b{re.escape(w)}\b", text.lower())})
    if found:
        issues.append("interprets the image instead of describing it ("
                      + ", ".join(found) + ") -- the renderer draws what is in "
                      "frame, not what it means")
    # A GRADE IS NOT IMAGE CONTENT. The scene plan says "palette: warm-gold"; a
    # writer/prompt pass that copies "warm-gold palette" into the prompt makes the
    # model paint that colour AND the finish grade it again. Measured on the squid
    # draft: 10/10 prompts carried a palette name because the plan brief showed it.
    grade = sorted({p for p in SCENE_PALETTES
                    if re.search(rf"\b{re.escape(p)}\s+palette\b", text.lower())})
    if grade:
        issues.append("names the grade palette ("
                      + ", ".join(grade) + ") -- the palette is applied AFTER "
                      "generation; describe the light, not the grade")
    return issues


def prompt_problems(shots: Sequence[Shot]) -> List[str]:
    """Is each `prompt` an IMAGE, or a sentence about an image?

    Reported, not repaired, for the same reason `validate` reports everything: an
    advisory that quietly rewrote the writer's text would hide which model writes
    composites. `rewrite_prompts` is the explicit repair pass.
    """
    probs: List[str] = []
    for i, s in enumerate(shots, 1):
        issues = _prompt_issues(getattr(s, "prompt", ""))
        if issues:
            probs.append(f"prompt: shot {i} " + "; ".join(issues))
    return probs


def coverage_problems(shots: Sequence[Shot]) -> List[str]:
    """Is this a film of SCENES, or one new place per shot?

    Uses the sequence director's OWN place/figure affinity
    (`visual_org.affinity`: content words with style and atmosphere stripped), so
    the writer's coverage and the director's hold/montage grammar are judged by
    one definition. The failure this names: `seasoning`/`ego` had scene labels,
    but the prompts inside a scene described unrelated places (within-run overlap
    no higher than across), so the labels were cosmetic and the cut became a
    montage.

    `coverage:` is ADVISORY at the gate (a montage is legitimate) and a DEMAND on
    the writing loop, which plans the scenes first and re-asks once if the material
    still has no coverage.
    """
    shots = list(shots)
    if len(shots) < 3:
        return []
    try:
        from visual_org import affinity, organise
        runs = organise(shots)
    except Exception:                                # noqa: BLE001 -- never block
        return []
    probs: List[str] = []
    any_hold = False
    for run in runs:
        sc, idx = run.place, run.indices
        if len(idx) < MIN_COVERAGE_HOLD:
            continue
        any_hold = True
        pairs = [affinity(shots, idx[k], idx[k + 1])
                 for k in range(len(idx) - 1)]
        mean = sum(pairs) / len(pairs)
        if mean < COVERAGE_AFFINITY:
            probs.append(
                f"coverage: scene {sc!r} holds a LABEL but not a place (adjacent "
                f"prompt affinity {mean:.2f} < {COVERAGE_AFFINITY:.2f}) -- its "
                f"{len(idx)} shots describe unrelated images. A scene is ONE place "
                f"and ONE figure: reuse that place's nouns in every prompt of it, "
                f"and cover the action from different angles.")
        subs = [shots[j].subject.strip() for j in idx
                if shots[j].subject.strip()]
        if len(set(subs)) > 1:
            probs.append(
                f"coverage: scene {sc!r} changes figure inside the scene "
                f"({' / '.join(sorted(set(subs)))}) -- a figure change is a scene "
                f"change; hold one figure for the whole scene.")
        elif not subs:
            probs.append(
                f"coverage: scene {sc!r} names no 'subject' on any shot -- the "
                f"renderer and the director cannot hold a figure they were not "
                f"told about.")
    if len(shots) >= 4 and not any_hold:
        probs.append(
            "coverage: no scene holds one place for 2+ shots -- every cut is a new "
            "place, so this is a montage of images, not 2-3 scenes. Give the film "
            "2-3 scenes of 2-4 shots each, one place and one figure per scene.")
    return probs


def prompt_variation_problems(shots: Sequence[Shot]) -> List[str]:
    """A hold covers ONE place from DIFFERENT angles -- not with the same sentence.

    Measured on the 13-shot held film: the plan/anchors made the shots share their
    place/figure, and the writer (and the prompt pass) then repeated one sentence,
    so the previs stills were near-duplicates (adjacent full-prompt overlap 0.7+;
    prompt->still margin negative because every still matched its scene-mates'
    prompts better than its own). This is the ceiling to coverage's floor: keep the
    place/figure, change the angle/action/moment.
    """
    probs: List[str] = []
    for i in range(1, len(shots)):
        if (shots[i].scene or "") != (shots[i - 1].scene or ""):
            continue
        ov = link_overlap(shots[i - 1].prompt, shots[i].prompt)
        if ov >= PROMPT_REPEAT:
            probs.append(
                f"prompt: shot {i + 1} repeats shot {i}'s image (overlap {ov:.2f} "
                f">= {PROMPT_REPEAT}) -- inside a scene the shots cover ONE place "
                f"and figure from different angles: change the framing, the "
                f"moment or the detail, not the whole sentence.")
    return probs


def story_problems(shots: Sequence[Shot]) -> List[str]:
    """Does the film DEVELOP? Beat order + the turn/resolve joint.

    Extracted from `validate` so the sequence planner and the gate can ask the
    same question without duplicating it. `story:` problems block adoption;
    `story-link:` is a warning because a lexical proxy cannot tell "linked but
    worded differently" from "unrelated".
    """
    probs: List[str] = []
    if not shots:
        return probs
    beats = [s.beat for s in shots]
    if beats[0] != "establish":
        probs.append(f"story: opens on {beats[0]!r}, not 'establish'")
    if beats.count("establish") > 1:
        probs.append(f"story: {beats.count('establish')} establish beats -- "
                     f"only the first shot establishes")
    if "turn" not in beats:
        probs.append("story: no 'turn' -- nothing changes, so the film restates "
                     "one idea instead of developing one")
    if beats[-1] != "resolve":
        probs.append(f"story: ends on {beats[-1]!r}, not 'resolve'")
    order = [BEATS.index(b) for b in beats]
    back = [f"{i+1}:{beats[i]}->{beats[i+1]}"
            for i in range(len(beats) - 1) if order[i + 1] < order[i]]
    if back:
        probs.append("story: beats go backwards (" + ", ".join(back) + ")")

    # THE JOINT THAT WAS WEAK. A turn nobody pays off is a scene change, not a
    # story. Measured on a real output: the turn was "a silhouetted figure with a
    # knowing look" and the resolve was "a dimly lit auction" -- beats declared
    # correctly, and nothing whatever linking them.
    #
    # A lexical proxy cannot tell "linked but worded differently" from
    # "unrelated", so this is a WARNING for the eye, not a hard break: a
    # legitimate resolve can share no words at all with its turn. It only fires on
    # near-total disjointness, which is what the auction was.
    if "turn" in beats and len(shots) > 1:
        ti, ri = beats.index("turn"), len(shots) - 1
        if ri > ti:
            ov = link_overlap(shots[ti].prompt, shots[ri].prompt)
            if ov < TURN_RESOLVE_LINK:
                probs.append(
                    f"story-link: the resolve shares almost no imagery with the "
                    f"turn (overlap {ov:.2f} < {TURN_RESOLVE_LINK}) -- shot "
                    f"{ri+1} may read as a new scene rather than the turn's "
                    f"consequence. Check it by eye.")
    return probs


def validate(raw: dict) -> Tuple[List[Shot], List[str]]:
    """Turn a raw LLM object into Shots, collecting every rule it broke.

    Nothing is silently corrected here. The problems list is the measurement --
    if this function quietly fixed things, the scorecard would report a model as
    well-behaved when the planner was doing all the work.
    """
    problems: List[str] = []
    shots: List[Shot] = []
    items = raw.get("shots")
    if not isinstance(items, list) or not items:
        return [], ["no 'shots' array"]

    # Validation: warn if shot count seems unreasonable, but don't hard-fail
    # The UI allows 2-24 shots and we should respect that
    if len(items) < 2:
        problems.append(f"{len(items)} shot(s) - need at least 2 for a film")
    elif len(items) > 24:
        problems.append(f"{len(items)} shots - exceeds maximum of 24")

    for i, s in enumerate(items, 1):
        if not isinstance(s, dict):
            problems.append(f"shot {i}: not an object")
            continue
        prompt = str(s.get("prompt", "")).strip()
        if len(prompt) < 12:
            problems.append(f"shot {i}: prompt too short")
            continue
        # a title or a heading is not an image. The first real run produced
        # "Scene 1: Darkness" and "Whispers fade", which the renderer can only
        # answer with noise.
        words = prompt.split()
        if len(words) < 8:
            problems.append(f"shot {i}: prompt is {len(words)} words -- too terse "
                            f"to render as an image")
        if words and words[0].lower().rstrip(":").startswith("scene"):
            problems.append(f"shot {i}: prompt starts with a scene heading")
        # other fields' values fused into the prompt string: `prompt` is what CLIP
        # and SD receive, so this is drawn rather than ignored. Recorded, not
        # stripped -- see `prompt_leaks`.
        leak = prompt_leaks(prompt)
        if leak:
            problems.append(
                f"shot {i}: prompt carries field values ({', '.join(leak)}) -- "
                f"the image model draws them")
        scale = s.get("scale", "medium")
        if scale not in SCALES:
            problems.append(f"shot {i}: bad scale {scale!r}")
            scale = "medium"
        cam = s.get("camera", "none")
        if cam not in CAMERAS_OK:
            problems.append(f"shot {i}: bad camera {cam!r}")
            cam = "none"
        try:
            secs = float(s.get("seconds", 2.5))
        except Exception:
            secs = 2.5
            problems.append(f"shot {i}: non-numeric seconds")
        if not (1.5 <= secs <= 3.6):
            problems.append(f"shot {i}: {secs}s outside 1.5-3.6")
        try:
            energy = float(s.get("energy", 0.5))
        except Exception:
            energy = 0.5
            problems.append(f"shot {i}: non-numeric energy")
        energy = max(0.0, min(1.0, energy))
        beat = str(s.get("beat", "")).strip().lower()
        if beat not in BEATS:
            problems.append(f"shot {i}: beat {beat!r} is not one of {BEATS}")
            beat = "develop" 
        purpose = str(s.get("purpose", "")).strip().lower()
        if purpose not in PURPOSES:
            problems.append(f"structure: shot {i}: purpose {purpose!r} is not one "
                            f"of {PURPOSES}")
            purpose = "master"
        link = str(s.get("link", "")).strip().lower()
        if link not in LINKS:
            problems.append(f"structure: shot {i}: link {link!r} is not one of "
                            f"{LINKS}")
            link = "open" if i == 1 else "cut"
        movement = str(s.get("movement", "")).strip()
        if energy > MAX_RENDERABLE_MOTION:
            problems.append(
                f"shot {i}: energy {energy:.2f} exceeds the renderer's measured "
                f"ceiling {MAX_RENDERABLE_MOTION} -- shot would be refused")
            energy = MAX_RENDERABLE_MOTION
        shots.append(Shot(prompt=prompt, scale=scale, camera=cam, seconds=secs,
                          subject=str(s.get("subject", "")).strip(),
                          scene=str(s.get("scene", "scene1")).strip() or "scene1",
                          energy=energy, beat=beat, purpose=purpose, link=link,
                          movement=movement,
                          # carried, never judged: `anchor_plan` sets it and the
                          # re-validate that follows must not wipe it (measured:
                          # lighthouse planned 3 grades, stored all-neutral)
                          palette=str(s.get("palette", "") or "").strip()))
   
    # the rules the planner would otherwise have to repair
    for i in range(len(shots) - 1):
        if shots[i].scale == shots[i + 1].scale:
            problems.append(f"jump cut: shots {i+1},{i+2} both {shots[i].scale}")
    movers = sum(1 for s in shots if s.camera != "none")
    if movers > 2:
        problems.append(f"{movers} moving shots (want <=2)")
    problems.extend(story_problems(shots))
    problems.extend(structure_problems(shots))
    problems.extend(prompt_problems(shots))
    problems.extend(prompt_variation_problems(shots))
    problems.extend(coverage_problems(shots))
    return shots, problems


def shot_dict(s: Shot) -> Dict:
    """A Shot back to the raw JSON shape `validate` reads.

    Needed in two places that are really the same place -- carrying a written
    shot list across a boundary it was not written in: the proposal cache, and
    the refused draft that is fed back into the next run.
    """
    return {"prompt": s.prompt, "scale": s.scale, "camera": s.camera,
            "seconds": s.seconds, "subject": s.subject, "scene": s.scene,
            "energy": s.energy, "beat": s.beat, "purpose": s.purpose,
            "link": s.link, "movement": s.movement, "palette": s.palette}


def shots_from_dicts(items: Sequence[dict]) -> List[Shot]:
    """Raw shot objects -> Shots, through the same validator the writer faces.

    Used to reload a refused draft from disk so the next run can be handed it
    back. The validation problems are ignored on purpose: the draft was measured
    when it was refused, and re-deriving them here would present old faults as
    new information in the next repair brief.
    """
    shots, _problems = validate({"logline": "", "shots": list(items or [])})
    return shots


def write_shots(concept: str = CONCEPT_DEFAULT, model: str = "granite4.1:8b",
                temperature: float = 0.8, attempts: int = 2,
                verbose: bool = True, fresh: bool = False,
                n_shots: Optional[int] = None,
                num_predict: Optional[int] = None) -> Proposal:
    """Concept -> shot list. Always returns something usable.

    `n_shots` makes the writer produce the count the TARGET DURATION needs. Holds
    clamp at 3.6s, so duration is really set by shot count: asking for a 30s
    TikTok and getting 6 shots caps the film at 21.6s no matter what the rhythm
    curve does. Deriving the count from the target is what makes `--length`
    actually mean something.

    Proposals are cached. The text model is non-deterministic, so without a cache
    the shot list behind a 70-minute render is unreproducible -- you cannot
    re-render the same film, and you cannot tell whether a change in the output
    came from the writer or from the sampler. Re-roll deliberately with
    `fresh=True`.
    """
    cache = ROOT / "output" / "film" / "proposals"
    slug = re.sub(r"\W+", "-", concept.lower()).strip("-")[:60]
    # n_shots belongs in the key. Without it a cached 6-shot proposal is served
    # when 12 are requested, and the target duration is silently ignored -- the
    # cache defeats the thing it was added to make reproducible.
    cache_f = cache / (f"{slug}__{model.replace(':', '-')}"
                       f"__n{n_shots or 'auto'}.json")

    if cache_f.exists() and not fresh:
        try:
            raw = json.loads(cache_f.read_text(encoding="utf-8"))
            shots, problems = validate(raw)
            if shots:
                if verbose:
                    print(f"  [story] cached proposal ({cache_f.name}): "
                          f"{len(shots)} shots, {len(problems)} rule(s) broken")
                    for pr in problems:
                        print(f"           - {pr}")
                return Proposal(concept=concept, model=model,
                                logline=str(raw.get("logline", ""))[:200],
                                shots=shots, raw=raw, seconds=0.0,
                                decode_tok_s=0.0, problems=problems, cached=True)
        except Exception:
            pass                       # a corrupt cache must not stop a render

    import ollama

    last_err = ""
    for attempt in range(1, attempts + 1):
        t0 = time.time()
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content":
                           f"Concept: {concept}\n"
                           + (f"Write exactly {n_shots} shots.\n" if n_shots
                              else "")
                           + "Write the shot list."}],
                format=SHOT_SCHEMA,
                options={"temperature": temperature,
                         "num_ctx": OLLAMA_NUM_CTX,
                         "num_predict": int(num_predict or predict_budget(n_shots))},
            )
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {str(exc)[:120]}"
            if verbose:
                print(f"  [story] attempt {attempt} failed: {last_err}")
            continue
        dt = time.time() - t0
        ec = getattr(resp, "eval_count", 0) or 0
        ed = getattr(resp, "eval_duration", 0) or 1
        tok_s = ec / (ed / 1e9)

        content = resp["message"]["content"]
        raw, parse_mode = parse_reply(content)
        if raw is None:
            last_err = f"response was not JSON [{json_failure(content)}]"
            if verbose:
                print(f"  [story] attempt {attempt}: {last_err}")
            continue

        shots, problems = validate(raw)
        if shots:
            try:
                cache.mkdir(parents=True, exist_ok=True)
                cache_f.write_text(json.dumps(raw, indent=2), encoding="utf-8")
            except Exception:
                pass
            p = Proposal(concept=concept, model=model,
                         logline=str(raw.get("logline", ""))[:200],
                         shots=shots, raw=raw, seconds=round(dt, 1),
                         decode_tok_s=round(tok_s, 1),
                         repaired=(attempt > 1), problems=problems,
                         reply_text=content, parse_mode=parse_mode,
                         attempts=attempt, eval_count=int(ec))
            if verbose:
                print(f"  [story] {model}: {len(shots)} shots in {dt:.1f}s "
                      f"({tok_s:.1f} tok/s), {len(problems)} rule(s) broken")
                for pr in problems:
                    print(f"           - {pr}")
            return p

    from make_film import TEST_SHOTS
    if verbose:
        print(f"  [story] no usable proposal ({last_err}); falling back to the "
              f"hand-written shot list")
    return Proposal(concept=concept, model=model, logline="(fallback)",
                    shots=list(TEST_SHOTS), raw={}, seconds=0.0,
                    decode_tok_s=0.0, used_fallback=True,
                    problems=[f"LLM failed: {last_err}"])


# --------------------------------------------------------------------------- #
# the ensemble: idea farms, then one showrunner who owns cohesion
# --------------------------------------------------------------------------- #
#
# The single-writer path asks ONE model to do everything at once: invent the
# images, choose the framing, place the scenes, keep the film coherent. Measured
# (writer_audit.py), the local models differ a lot at that -- and the weak ones
# intermittently fuse other field values into `prompt` (see `prompt_leaks`).
#
# The ensemble splits the job along the line where the models actually differ:
#
#   farms       several cheap models, asked only for raw visual ideas. Short
#               output, so each call is fast, and their disagreements are the
#               point -- more material to choose from.
#   showrunner  ONE model, the cleanest one, given every idea and asked to write
#               the final shot list. It decides what stays, what is discarded,
#               what is merged, and how the shots cohere into one film. It may
#               also invent.
#
# Only the showrunner's output can reach a render, so "is this writer good?" is
# asked of ONE model instead of three, and a bad farm can only ever waste a few
# seconds. The farms are IDEAS, not authors.

# Three farms, not four. Measured (writer_audit / the notes below), llama3 gives
# ONE idea in ~33 s where the others give six in 20-50 s, so the fourth slot was
# costing a minute and contributing nothing. The obvious replacements were tried
# and are worse: qwen3:1.7b gave 1 idea in 24 s (its reasoning eats the budget)
# and qwen3.5:0.8b returned an empty reply. Do not re-add them without measuring.
FARM_MODELS = ["qwen2.5:3b", "phi3:latest", "mistral:latest"]

# Showrunner candidates, restricted to models that are ACTUALLY INSTALLED on this
# machine (`ollama list`, 2026-10-29). The previous list named `llama3.1:8b`,
# which is not installed -- a comparison would have reported it as a model
# failure rather than a typo. llama3 is the installed 8B and is kept only as a
# low baseline: HANDOFF §11.2 measured it leaking field values into `prompt` and
# needing structural repairs, and it decodes at ~2.9 tok/s.
SHOWRUNNER_CANDIDATES = [
    "granite4.1:8b",   # cleanest text, never leaked; the shipped showrunner
    "qwen2.5:7b",      # strong instruction/JSON compliance; weaker on imagery
    "mistral:latest",  # fast-ish, mixed plot; leaks field values historically
    "llama3:latest",   # low baseline, installed as `llama3`, not `llama3.1`
]
SHOWRUNNER = SHOWRUNNER_CANDIDATES[0]  # granite4.1:8b - quality first, JSON repaired

# THE PROMPT PASS HAS ITS OWN MODEL, AND IT WAS MEASURED. The pass is a text
# FORMATTING job ("one sentence, no semicolon, no meaning words"), not an imagery
# job, so the showrunner does not have to write it -- and measured on the real
# 13 approved `ego` prompts, it must not: granite4.1:8b removed the semicolon but
# kept the interpretation (12/13 still flagged), qwen2.5:7b got 9/13 clean, and
# mistral:latest returned **13/13 clean single-image prompts in one batch call**
# (~1.5 min). The output is still validated and per-shot best-of'd, so a bad reply
# costs nothing. Do not swap this for "the best writer": this task rewards
# instruction-following on a tiny rewrite, not prose or filmcraft.
PROMPT_PASS_MODEL = "mistral:latest"

# Each farm is pointed at a different part of the problem, so the pool of ideas is
# spread out rather than three near-copies of the same film. The first three are
# the ones a three-farm run uses: where it happens, what is close to the lens, and
# who is in it. "light" is the fourth, used only if a run is given four farms.
FARM_ANGLES = [
    "places and weather: where this happens, what the air and ground are doing",
    "objects, materials and textures: what is close to the lens, what it is made of",
    "people: what a person is doing, where they are in frame, how much of them is seen",
    "light: time of day, direction, colour, what is lit and what is not",
]

IDEA_SYSTEM = (
    "You are a location scout for a short atmospheric film. You do not write "
    "scenes, stories or shot lists. You give CONCRETE VISUAL IDEAS: physical "
    "things a camera could point at, described so an image can be made of them.\n"
    "\n"
    "Each idea is ONE line of visual description, under 20 words. No titles, no "
    "feelings, no plot. Materials, weather, light, objects, bodies in space.\n"
    "\n"
    "NEVER do these three things:\n"
    "- do not restate the concept, or any phrase from it\n"
    "- do not name a camera move ('the camera pushes in')\n"
    "- do not copy the examples below. They are about a DIFFERENT film, chosen "
    "only to show you the shape of a good line.\n"
    "\n"
    "GOOD: 'a chrome diner counter at noon, ketchup bottles, hard shadows'\n"
    "GOOD: 'a yellow digger bucket, rust and clay, one cracked windscreen'\n"
    "BAD:  'a scene about loneliness'\n"
    "BAD:  'the camera pushes in as she realises'\n"
    "Return JSON only."
)

IDEA_SCHEMA = {
    "type": "object",
    "properties": {"ideas": {"type": "array", "items": {"type": "string"}}},
    "required": ["ideas"],
}

SHOWRUNNER_ADDENDUM = (
    "\n\nYOU ARE THE EDITOR, NOT A SCOUT. A group of location scouts has sent you "
    "raw ideas, each labelled with where it came from. They have not seen each "
    "other's work and none of them has seen the film.\n"
    "\n"
    "Your job is ONE coherent film:\n"
    "- ARRANGE, do not stack. Build 2-3 sequences, each covering ONE place and ONE "
    "action with 2-4 shots whose images relate to each other. Using each idea as a "
    "separate cut to new scenery is a slideshow, not a film. Fill 'purpose' and "
    "'link' for every shot.\n"
    "- Keep what serves it and DISCARD the rest. Taking every idea gives a list of "
    "unrelated images, which is a failure even when each image is good.\n"
    "- Make the kept shots belong together: a consistent recurring figure, "
    "consistent light and materials across a scene, a progression from scene to "
    "scene rather than a jumble.\n"
    "- You may invent, merge or rewrite ideas. The scouts' text is raw material, "
    "not a contract.\n"
    "- Aim for variety WITHIN that coherence: across the shots, vary the framing "
    "scale, the light and the materials. Do not write variations of one image.\n"
    "- THE SHOT COUNT IS BINDING. When the request names a number of shots, write "
    "exactly that many DISTINCT shots. The length of the film is set by that "
    "number, not by your sense of when the film feels finished. A short list is "
    "rejected and rewritten, so count the shots before you return.\n"
    "- USE MORE THAN ONE SCOUT. Taking your shots from a single scout's list means "
    "the others were never asked, and the whole point of asking four was to choose "
    "between them. Take at most two ideas from any one scout.\n"
    "Write the shot list in the same JSON schema."
)

# How many ideas from one farm are allowed before the showrunner is told it is
# leaning too hard on one scout. Measured (see HANDOFF): without this, granite
# took 6 of 6 shots from qwen2.5:3b and ignored the other three farms entirely --
# provenance read 6/0/0/0. One definition, used in the brief and in the report.
FARM_QUOTA = 2


def _repair_missing_commas(candidate: str) -> List[str]:
    """Progressively repair the JSON damage local models actually produce.

    Measured failure modes in this project are all the same family: the model
    emits a structurally-correct-looking object but omits a separator. The
    repairs are therefore separators only -- never values, never braces -- and
    they are attempted only after a strict parse has already failed, so a valid
    reply can never be rewritten.

    Returning a SEQUENCE rather than one string matters: each rule can expose the
    next (adding a comma after `}` can move the failure to the next `}`), so the
    caller tries the cumulative passes in order and stops at the first parse.

    The granite4.1:8b evidence for needing this at all is in
    `output/showrunner_test/` -- the comparison harness records every raw reply,
    so the repair set is kept honest by what the models actually emit.
    """
    steps = []

    def apply(text: str, pattern: str, repl: str) -> str:
        return re.sub(pattern, repl, text)

    # pass 1: the common case -- a separator missing before the next key
    c = candidate
    c = apply(c, r'"\s*\n\s*"', '",\n  "')          # "value"\n "key":
    c = apply(c, r'(\d|true|false)\s*\n\s*"', r'\1,\n  "')  # 123\n "key":
    c = apply(c, r'}\s*\n\s*"', '},\n  "')          # } \n "key":
    c = apply(c, r']\s*\n\s*"', '],\n  "')          # ] \n "key":
    steps.append(c)
    # pass 2: the same rules again; nested objects need a second round
    c2 = c
    c2 = apply(c2, r'"\s*\n\s*"', '",\n  "')
    c2 = apply(c2, r'(\d|true|false)\s*\n\s*"', r'\1,\n  "')
    c2 = apply(c2, r'}\s*\n\s*"', '},\n  "')
    c2 = apply(c2, r']\s*\n\s*"', '],\n  "')
    steps.append(c2)
    # pass 3: same-line omissions, which the newline rules cannot see:
    # `"value" "next":`, `} {`, `123 "next":`. The lookaheads make these fire
    # only when the next token is a key or a new container, so array strings and
    # prose are left alone. Only reached after a strict parse has failed.
    c3 = apply(c2, r'("[^"\n]*"|true|false|-?\d+(?:\.\d+)?)\s+'
                   r'(?="[A-Za-z_][^"\n]*"\s*:)', r'\1, ')
    c3 = apply(c3, r'([}\]])\s*(?=[{\[])', r'\1, ')
    steps.append(c3)
    # cleanup: a trailing comma is the other separator mistake models make
    steps = [apply(s, r',(\s*[}\]])', r'\1') for s in steps]
    # de-duplicate while preserving order
    out, seen = [], set()
    for s in steps:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


PROGRESS_EVERY_S = 8.0       # how often a streamed reply reports its token count


def chat_streamed(label: str, budget: int, **kwargs) -> Tuple[str, int]:
    """`ollama.chat` for a LONG reply, reporting real progress. -> (content, tokens).

    The showrunner's reply is ~10 minutes of silence at ~3 tok/s, and the UI had no
    figure for it ("it is a model writing, not a loop"). Streaming gives one: tokens
    written so far against the cap, the measured rate, and the most that could be
    left (the cap is a ceiling -- a model that finishes sooner stops sooner). Lines
    are `<label>: N/BUDGET tokens, R tok/s, up to M min left`, flushed, which is the
    format `job_runner._story_progress` parses.

    A non-streaming client (a test double, an older server) returns a finished
    response instead of an iterator; that is handled, so streaming is never a new way
    to fail.
    """
    import ollama
    resp = ollama.chat(stream=True, **kwargs)
    if not hasattr(resp, "__next__"):          # already a complete response
        return (resp["message"]["content"],
                int(getattr(resp, "eval_count", 0) or
                    (resp.get("eval_count", 0) if hasattr(resp, "get") else 0) or 0))
    parts: List[str] = []
    chunks = 0
    final_count = 0
    t0: Optional[float] = None       # the clock starts at the FIRST TOKEN: the prompt
    last = time.time()               # is processed first (minutes, ~18 tok/s), and
    for chunk in resp:               # counting that would make the rate absurd
        msg = chunk["message"] if isinstance(chunk, dict) else chunk.message
        piece = (msg["content"] if isinstance(msg, dict) else msg.content) or ""
        if piece:
            if t0 is None:
                t0 = time.time()
            parts.append(piece)
            chunks += 1
        done = chunk.get("done") if isinstance(chunk, dict) else getattr(chunk, "done", False)
        if done:
            ec = (chunk.get("eval_count") if isinstance(chunk, dict)
                  else getattr(chunk, "eval_count", 0))
            final_count = int(ec or 0)
        now = time.time()
        if now - last >= PROGRESS_EVERY_S and chunks and t0 is not None:
            last = now
            rate = chunks / max(1e-6, now - t0)
            left = max(0, budget - chunks) / max(rate, 1e-6) / 60.0
            print(f"  [story] {label}: {chunks}/{budget} tokens, {rate:.1f} tok/s, "
                  f"up to {left:.0f} min left", flush=True)
    return "".join(parts), (final_count or chunks)


def salvage_shots(text: str) -> Optional[Dict]:
    """Recover every COMPLETE shot object from a cut-off showrunner reply.

    A shot list of 15 costs ~10 minutes on this CPU. When the token budget ends the
    reply mid-object, the shots before it are intact JSON -- and the whole attempt
    used to be discarded (`json_failure` calls it TRUNCATED and moves on). This walks
    the `"shots"` array with a string-aware brace counter and keeps each object that
    closes and parses on its own. The unfinished last object is dropped, never
    guessed at. Returns `{"logline", "shots"}`, or None when no shot is complete.
    The caller treats the result like any short reply: `validate` judges it and the
    extension pass closes the shortfall.
    """
    if not text:
        return None
    key = re.search(r'"shots"\s*:\s*\[', text)
    if not key:
        return None
    shots: List[dict] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i in range(key.end(), len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except Exception:
                        obj = None
                    if isinstance(obj, dict):
                        shots.append(obj)
                    start = -1
        elif ch == "]" and depth == 0:
            break
    if not shots:
        return None
    m = re.search(r'"logline"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    try:
        logline = json.loads('"' + m.group(1) + '"') if m else ""
    except Exception:
        logline = m.group(1) if m else ""
    return {"logline": logline, "shots": shots}


def parse_reply(text: str, salvage: bool = False) -> Tuple[object, str]:
    """Parse a model reply. Returns `(value, mode)`; `(None, "")` if unparseable.

    `salvage=True` adds a last resort for the showrunner only: the complete shots of
    a truncated reply (mode "salvaged", see `salvage_shots`). It is off everywhere
    else, because a farm, a scene plan or a prompt pass that comes back partial is a
    failure, not a short film.

    `json_from_text` is the pipeline's raising interface. This is the measuring
    interface: it never raises, because on the normal path nothing needs it and
    on the failure path the *mode* is the evidence. Modes, in increasing
    distance from what the model was asked for:

        raw      the reply was pure JSON
        fenced   it arrived in a ``` fence
        braces   it had text around the outermost {...}
        repaired a separator had to be supplied (see _repair_missing_commas)

    `write_shots_ensemble` records the mode on the Proposal so a comparison can
    say "this model needed repair 3 of 9 attempts" instead of "it failed".
    """
    if text is None:
        return None, ""
    s = text.strip()
    if not s:
        return None, ""
    try:
        return json.loads(s), "raw"
    except Exception:
        pass
    if s.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
        try:
            return json.loads(stripped), "fenced"
        except Exception:
            s = stripped
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None, ""
    candidate = s[i:j + 1]
    try:
        return json.loads(candidate), "braces"
    except json.JSONDecodeError:
        pass
    for fixed in _repair_missing_commas(candidate):
        try:
            return json.loads(fixed), "repaired"
        except Exception:
            continue
    if salvage:
        got = salvage_shots(s)
        if got is not None:
            return got, "salvaged"
    return None, ""


def json_failure(text: str) -> str:
    """Why a reply did not parse, in one line, for the audit.

    Truncation is called out separately because it is not a syntax problem: the
    model was cut off by the token budget (or by its own stop), and the fix is
    `num_predict`/shot count, NOT a repair rule. Conflating the two made the
    first diagnosis wrong.
    """
    if text is None:
        return "empty response"
    s = text.strip()
    if not s:
        return "empty response"
    i, j = s.find("{"), s.rfind("}")
    candidate = s[i:j + 1] if i >= 0 and j > i else s
    try:
        json.loads(candidate)
        return "parsed (should not be a failure)"
    except json.JSONDecodeError as e:
        opens = candidate.count("{") + candidate.count("[")
        closes = candidate.count("}") + candidate.count("]")
        trunc = " TRUNCATED" if opens > closes else ""
        return (f"JSONDecodeError: {e.msg} at line {e.lineno} col {e.colno} "
                f"({opens} open vs {closes} close brackets){trunc}")
    except Exception as e:  # pragma: no cover -- defensive
        return f"{type(e).__name__}: {e}"


def json_from_text(text: str):
    """Pull a JSON object out of a model's reply, or raise.

    Local models do not all honour `format=` the same way. Measured: qwen3:1.7b
    returns clean JSON on a short prompt but can come back wrapped or with
    thinking text in front of it, and a smaller model returned a truncated
    `'{\\n '`. Parsing `resp["message"]["content"]` directly therefore fails for
    reasons that have nothing to do with the model being wrong -- a farm would
    drop out and the ensemble would quietly run on three farms instead of four.
    Strip code fences, take the outermost braces, then supply only missing
    separators.
    """
    value, _mode = parse_reply(text)
    if value is not None:
        return value
    if text is None or not str(text).strip():
        raise ValueError("empty response")
    s = str(text).strip()
    raise ValueError(f"no parseable JSON object in response "
                     f"[{json_failure(text)}]: {s[:120]!r}")


def farm_ideas(concept: str, model: str, angle: str, n_ideas: int = 6,
               temperature: float = 1.0, timeout_predict: int = 700,
               verbose: bool = True) -> Dict:
    """One farm's raw visual ideas. Short output on purpose -- this must be cheap.

    The budget is 700 tokens rather than 300 because some small models spend the
    first few hundred on reasoning before the answer, and a truncated reply means
    a lost farm.
    """
    import ollama

    t0 = time.time()
    content = ""
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": IDEA_SYSTEM},
                      {"role": "user", "content":
                       f"Concept: {concept}\n"
                       f"Your brief: {angle}\n"
                       f"Give {n_ideas} ideas. One line each."}],
            format=IDEA_SCHEMA,
            options={"temperature": temperature, "num_ctx": OLLAMA_NUM_CTX,
                     "num_predict": timeout_predict},
        )
        content = resp["message"]["content"]
        raw = json_from_text(content)
        ideas = [str(x).strip() for x in raw.get("ideas", []) if str(x).strip()]
    except Exception as exc:
        if verbose:
            print(f"    farm {model} ({angle.split(':')[0]}): "
                  f"{type(exc).__name__}: {str(exc)[:90]}")
            print(f"      raw reply was {len(content)} chars: {content[:90]!r}")
        return {"model": model, "angle": angle, "ideas": [], "seconds": 0.0,
                "error": f"{type(exc).__name__}: {str(exc)[:120]}"}

    dt = time.time() - t0
    if verbose:
        print(f"    farm {model:<16} {angle.split(':')[0]:<22} "
              f"{len(ideas)} ideas in {dt:.0f}s")
        for i in ideas[:3]:
            print(f"        - {i}")
        if len(ideas) > 3:
            print(f"        ... and {len(ideas)-3} more")
    return {"model": model, "angle": angle, "ideas": ideas, "seconds": round(dt, 1)}


def _provenance(final_prompts: List[str], farms: List[Dict]) -> Dict:
    """How much of each farm's material survived into the final shot list.

    The point of the architecture is that farms supply ideas and the showrunner
    decides. This checks that it is actually happening rather than the showrunner
    ignoring the material and writing from scratch.
    """
    final_sets = [content_words(p) for p in final_prompts]
    out = {}
    for f in farms:
        if not f["ideas"]:
            out[f["model"]] = {"ideas": 0, "used": 0, "mean_best": 0.0}
            continue
        best = []
        for idea in f["ideas"]:
            iset = content_words(idea)
            best.append(max((jaccard(iset, fs) for fs in final_sets), default=0.0))
        used = sum(1 for b in best if b >= 0.34)
        out[f["model"]] = {"ideas": len(f["ideas"]), "used": used,
                           "mean_best": round(sum(best) / len(best), 3)}
    # and the direct question: does each final prompt trace back to SOME idea?
    traced = 0
    for fs in final_sets:
        b = 0.0
        for f in farms:
            for idea in f["ideas"]:
                b = max(b, jaccard(fs, content_words(idea)))
        traced += 1 if b >= 0.34 else 0
    out["_final_prompts_traced"] = traced
    out["_final_prompts"] = len(final_prompts)
    return out


def _seed_lines(lines: List[str], keep: Optional[Sequence[str]],
                keep_only: bool = False) -> List[str]:
    """Put the ideas a human KEPT in front of the showrunner.

    A kept idea is INJECTED as its own line, not merely matched against the farms'
    output. That is the whole point, and it took a real run to see it: a re-roll
    re-runs the farms, and a fresh farm run does NOT reproduce a kept idea
    verbatim, so text-matching alone discarded the director's selection precisely
    when the farms had re-run -- the normal case. Matching is still used to avoid
    listing a kept idea twice when the farm happens to repeat it.

    `keep_only` is the "discard the rest" meaning: show ONLY the kept ideas.
    """
    if not keep:
        return lines
    wanted = [k.strip() for k in keep if k and k.strip()]
    if not wanted:
        return lines
    matched = [l for l in lines if any(w.lower() in l.lower() for w in wanted)]
    injected = [f"- [kept by hand] {w}" for w in wanted
                if not any(w.lower() in m.lower() for m in matched)]
    if keep_only:
        print(f"  [story] seeded with ONLY the {len(wanted)} kept idea(s) "
              f"(the rest discarded)")
        return injected + matched
    print(f"  [story] {len(wanted)} kept idea(s) guaranteed, plus {len(lines)} "
          f"from the farms alongside them")
    return injected + lines


def _repair_note(problems: Sequence[str], shots: Sequence[Shot],
                 n_shots: Optional[int] = None,
                 returned: Optional[int] = None,
                 prefix: str = "") -> str:
    """Turn a failed story/structure check into a REPAIR BRIEF for the next attempt.

    The old re-roll was blind: the showrunner was asked the identical question
    again and hoped to land differently. Measured (HANDOFF §11.3, and the
    comparison in SHOWRUNNER_TEST.md): the beat-order failure repeats across
    blind attempts often enough that shipping a broken story was the norm — 9 of
    11 productions are non-monotonic. A text model can fix a specific complaint
    it is shown; it cannot fix one it is never told about.

    The note gives it the exact validator complaints (story AND structure), the
    beats/scenes/purposes it actually produced, and its previous prompts so it can
    keep the imagery instead of starting over. Empty string when there is nothing
    to repair — a re-roll for some other reason gets the plain prompt.

    `n_shots` adds the COUNT to the brief: a clean-but-short list has no story
    complaint, so without it the retry was never told the film was too short, and
    a 6-shot reply to a 13-shot request came back as 6 shots every time. `prefix`
    lets a caller say what the previous draft was (e.g. "a previous run of this
    concept was refused"), so a cross-run repair reads as a continuation rather
    than a contradiction.
    """
    relevant = [p for p in problems
                if p.startswith(("story:", "story-link:", "structure:", "prompt:",
                                 "coverage:"))]
    returned = len(shots) if returned is None else int(returned)
    short = bool(n_shots) and returned < int(n_shots)
    if not relevant and not short:
        return ""
    lines: List[str] = []
    if prefix:
        lines.append(prefix)
    if relevant:
        if any(p.startswith(("story:", "story-link:", "structure:"))
               for p in relevant):
            lines.append("YOUR PREVIOUS SHOT LIST WAS REJECTED BY THE "
                         "STORY/STRUCTURE CHECK:")
        else:
            lines.append("YOUR PREVIOUS SHOT LIST NEEDS A REPAIR:")
        lines += [f"- {p}" for p in relevant]
        if any("beats go backwards" in p for p in relevant):
            # The measured failure on a real 13-shot draft: the model wrote a
            # perfectly good three-act film and then labelled the FALLING ACTION
            # after its turn as 'develop' again (shape EddddTTddRR, all three
            # attempts). "Beats never backwards" was already in the brief; what it
            # needs is the rule that makes the falling action legal: after the
            # turn, a shot is 'turn' or 'resolve', never 'develop'.
            lines.append(
                "THE BACKWARDS BEATS ARE A LABEL PROBLEM, NOT AN IMAGE PROBLEM. "
                "Keep every shot's image exactly as it is. A shot after the first "
                "'turn' shows the change happening ('turn'), the change settling "
                "into the new world ('fall'), or the new world at the end "
                "('resolve') -- never 'develop' again. The film does not return "
                "to the world as it was.")
        if any(p.startswith("coverage:") for p in relevant):
            # The measured failure the brief could not enforce: scene labels that
            # named a place and a figure, while the prompts inside the scene
            # described unrelated places (12 places / 13 shots, so the director
            # montaged it). Keep the story; give it scenes.
            lines.append(
                "THE FILM HAS NO SCENES -- IT IS ONE NEW PLACE PER SHOT. Keep the "
                "story, the beats and the shot count. Re-arrange the SAME film "
                "into 2-3 SCENES, each ONE place and ONE figure covered by 2-4 "
                "shots: every prompt in a scene MUST reuse that place's nouns "
                "and the same figure, and the shots inside it are different "
                "angles on that one action (master, insert, reaction). Cut between "
                "scenes -- at most twice -- not inside them.")
    else:
        lines.append("YOUR PREVIOUS SHOT LIST WAS TOO SHORT:")
    lines.append("Its shots were (scene · purpose · link · beat):")
    for i, s in enumerate(shots, 1):
        lines.append(f"{i}. [scene {getattr(s, 'scene', '') or '?'}] "
                     f"[{getattr(s, 'purpose', '') or '?'}] "
                     f"[{getattr(s, 'link', '') or '?'}] "
                     f"[{getattr(s, 'beat', '') or '?'}] "
                     f"{getattr(s, 'prompt', '')[:100]}")
    if n_shots:
        lines.append(f"It returned {returned} shots; the film is exactly "
                     f"{int(n_shots)} shots.")
        if short:
            lines.append(
                f"Add the {int(n_shots) - returned} missing shot(s) as COVERAGE of "
                f"the scenes you already have -- master, insert, reaction -- not as "
                f"new places or a new ending.")
        lines.append(beat_layout(n_shots))
    lines.append(
        "Rewrite the shot list, fixing ONLY the problems above. Keep the film to "
        "2-3 SCENES, each one place and one action covered by 2-4 shots; never "
        "jump to new scenery on every cut; no interleaved scenes; every shot needs "
        "a purpose (establish/master/insert/reaction/resolve) and a link "
        "(open/match-action/eyeline/movement/graphic/scene-cut); the first shot of "
        "a scene is its longest; a camera move must continue in the same scene. "
        "Story rules still apply: exactly one 'establish' at shot 1, at least one "
        "'turn', last shot 'resolve', beats never backwards (nothing after the "
        "first 'turn' may be 'develop' again), and the resolve echoes "
        "the turn. Every 'prompt' must be ONE concrete image in ONE sentence, "
        "8-24 words, no semicolon, no meaning/emotion words -- the renderer draws "
        "only what is in frame. Keep the imagery that already works, keep the same "
        "JSON schema, and return JSON only.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# the scene plan: decide the 2-3 places BEFORE writing shots
# --------------------------------------------------------------------------- #
#
# The writer has always been told "2-3 scenes, one place and one figure each,
# 2-4 shots". Nothing made it do so, and the first rendered film proved the cost:
# 13 shots that were 12 places, so the sequence director -- correctly -- montaged
# them. A cheap plan pass (~1-2 min) commits the writer to the places and the
# figure before it writes a single prompt; the shot list then COVERS the plan
# instead of inventing a new image per beat. The plan is validated and normalised
# (2-3 scenes, shot counts summing to the request); if the model cannot produce a
# usable plan the writer proceeds without one, as before.
# Mirrors looks.ARC. The story layer must not import the cv2-heavy looks module at
# import time, but the plan's palette names must be the ones the grade understands.
SCENE_PALETTES = ["neutral", "warm-gold", "rich-green", "emerald", "sodium",
                  "cold-steel", "teal", "amber-dusk", "magenta", "bleach",
                  "cyan-night", "violet"]

SCENE_PLAN_SCHEMA = {
    "type": "object",
    "properties": {"scenes": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "place": {"type": "string"},
            "figure": {"type": "string"},
            "action": {"type": "string"},
            "light": {"type": "string"},
            "palette": {"type": "string", "enum": SCENE_PALETTES},
            "shots": {"type": "integer"},
        },
        "required": ["place", "figure", "action", "light", "palette", "shots"],
    }}},
    "required": ["scenes"],
}

SCENE_PLAN_SYSTEM = (
    "You are planning the SCENES of a short atmospheric film BEFORE any shots are "
    "written. A scene is ONE PLACE, ONE LIGHT and ONE FIGURE doing ONE action, "
    "covered by 2 to 4 shots from different angles. Plan 2 or 3 scenes -- never "
    "one place per shot.\n"
    "\n"
    "For each scene give:\n"
    "- place: the concrete place with its materials and weather, in reusable nouns "
    "(the same words will appear in every shot of the scene)\n"
    "- figure: the recurring PERSON or character (not a prop, not a light, not an "
    "object), described the same way each time, e.g. 'a scuba diver in a black "
    "wetsuit and round glass helmet'. A prop that glows is not a figure.\n"
    "- action: one line -- what happens there\n"
    "- light: the concrete LIGHT SOURCE and its colour for this scene, e.g. "
    "'green neon sign on wet concrete', 'red sodium streetlamp', 'cold overcast "
    "daylight', 'warm tungsten lamp'. Every scene needs one, and each scene's "
    "light must look different from the others. Do NOT blanket the film in fog or "
    "mist: that is the single most common way these films turn grey and "
    "unreadable.\n"
    "- palette: one grade for the scene, chosen from: neutral, warm-gold, "
    "cold-steel, bleach, rich-green, amber-dusk, cyan-night. Pick an arc across "
    "the film, not the same palette twice in a row.\n"
    "- shots: how many shots cover it. The whole film's shots must sum to the "
    "requested count.\n"
    "\n"
    "The film is the SAME figure through 2-3 places: a 13-shot short does not need "
    "13 characters. If the story develops, it is the ACTION that changes, not the "
    "place on every cut.\n"
    "Return JSON only."
)


def _scene_plan_user(concept: str, n_shots: Optional[int], idea_text: str) -> str:
    n = f"{int(n_shots)} shots" if n_shots else "shots for the target length"
    return (f"Concept: {concept}\n"
            f"The film is exactly {n}, covered by 2-3 scenes.\n"
            f"\nThe scouts' raw places/objects/people (raw material -- keep the "
            f"film to 2-3 places):\n\n{idea_text[:4000]}\n\n"
            f"Plan the scenes. Return JSON only.")


def _normalise_scene_plan(raw, n_shots: Optional[int]) -> Optional[Dict]:
    """2-3 scenes, each with place/figure/action, shot counts summing to the ask.

    A plan that cannot be made to sum is rejected (the writer then proceeds
    without one) rather than silently writing a film of a different length.
    """
    scenes = raw.get("scenes") if isinstance(raw, dict) else None
    if not isinstance(scenes, list) or not (2 <= len(scenes) <= 3):
        return None
    out = []
    for sc in scenes:
        if not isinstance(sc, dict):
            return None
        place = str(sc.get("place") or "").strip()
        figure = str(sc.get("figure") or "").strip()
        action = str(sc.get("action") or "").strip()
        light = str(sc.get("light") or "").strip()
        palette = str(sc.get("palette") or "neutral").strip()
        if palette not in SCENE_PALETTES:
            palette = "neutral"
        try:
            k = int(sc.get("shots") or 0)
        except (TypeError, ValueError):
            return None
        if not place or not figure or not action or k < 1:
            return None
        out.append({"place": place, "figure": figure, "action": action,
                    "light": light, "palette": palette, "shots": k})
    # ONE FIGURE. The plan brief asks for the same recurring figure, but the model
    # writes a description per scene ("man in suit, tied to chair" / "person in dim
    # light"). Anchors bind those words into every prompt, so a varying figure
    # description becomes a varying FIGURE in the film -- measured on the 13-shot
    # run as two distinct subjects. Keep the most common description.
    figures = [sc["figure"] for sc in out]
    if len(set(figures)) > 1:
        from collections import Counter
        common = Counter(figures).most_common(1)[0][0]
        if common:
            for sc in out:
                sc["figure"] = common
    n = int(n_shots or 0)
    if n:
        total = sum(x["shots"] for x in out)
        i, guard = 0, 0
        while total != n and guard < 500:
            sc = out[i % len(out)]
            if total < n:
                sc["shots"] += 1
                total += 1
            elif sc["shots"] > 1:
                sc["shots"] -= 1
                total -= 1
            i += 1
            guard += 1
        if total != n:
            return None
    return {"scenes": out}


def _scene_plan_text(plan: Dict) -> str:
    lines = ["THE SCENES (planned before the shots). Every shot belongs to ONE of "
             "these; inside a scene keep the place nouns and the figure stable:",
             ""]
    for i, sc in enumerate(plan.get("scenes") or [], 1):
        lines.append(f"{i}. {int(sc['shots'])} shot(s) -- place: {sc['place']} | "
                     f"figure: {sc['figure']} | action: {sc['action']} | "
                     f"light: {sc.get('light') or 'its own concrete light'}")
    lines.append("")
    lines.append("The shots inside a scene are DIFFERENT ANGLES on that one place "
                 "and action (master, insert, reaction), not new places. The whole "
                 "film is the SAME figure through these places unless an action "
                 "explicitly changes it. Cut between scenes -- at most twice.\n"
                 "Use each scene's LIGHT words in that scene's prompts, and do not "
                 "give every scene the same fog/mist: the plan's lights are what "
                 "keep the film from turning one flat grey.")
    return "\n".join(lines)


_ANCHOR_TRAILING = frozenset("""
a an the in on at of with and or to from into over under by as for
his her their its this that these those
only against while without near beside behind across through toward towards onto
upon between around along via than but if so where which who whose just also
within beneath beyond above below amid among before after during
""".split())


def _clip_clause(text: str, max_words: int) -> str:
    """A short, reusable anchor phrase (the renderer's CLIP window is finite).

    Cut at a WORD BOUNDARY THAT IS NOT A DANGLING FUNCTION WORD. Measured on the
    13-shot run: a 6-word figure clip of "a man sitting cross-legged in a dimly lit
    room..." came out as "a man sitting cross-legged in a", and that incomplete
    phrase was prepended to every prompt in the scene.
    """
    words = str(text or "").replace(",", " ").split()[:max(1, int(max_words))]
    while words and words[-1].lower().strip(".") in _ANCHOR_TRAILING:
        words.pop()
    return " ".join(words)


def _place_match(prompt: str, place: str) -> float:
    """How much a prompt's place nouns look like a planned scene's place."""
    from visual_org import ATMOSPHERE_WORDS, STYLE_WORDS
    a = content_words(prompt) - STYLE_WORDS - ATMOSPHERE_WORDS
    b = content_words(place) - STYLE_WORDS - ATMOSPHERE_WORDS
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _assign_scenes(shots: Sequence[Shot], scenes: Sequence[Dict]) -> List[int]:
    """Monotone, CONTENT-based assignment of shots to planned scenes.

    Measured on the 13-shot run: the plan said 4/4/5 shots, but the writer's
    images ran 6 courtyard / 2 stained-glass / 5 square. Assigning by the plan's
    SHOT BUDGET put courtyard shots 5-6 into the stained-glass scene, and the
    anchor then appended stained-glass words to courtyard images -- a hold that
    holds two different places. So the assignment maximises place-word match under
    the plan's ORDER (scene indices never decrease), and the plan's budgets are
    only a tie-breaker. The plan still commits the film to 2-3 places; it is the
    shots' own content that says where a shot belongs.
    """
    n, m = len(shots), len(scenes)
    scores = [[_place_match(s.prompt, scenes[j].get("place", ""))
               for j in range(m)] for s in shots]
    budgets = [max(1, int(sc.get("shots") or 1)) for sc in scenes]

    def compositions(total: int, parts: int, prefix=()):
        if parts == 1:
            yield prefix + (total,)
            return
        for k in range(1, total - parts + 2):
            yield from compositions(total - k, parts - 1, prefix + (k,))

    best_sizes, best = None, None
    for sizes in compositions(n, m):
        idx, total = 0, 0.0
        for j, k in enumerate(sizes):
            for _ in range(k):
                total += scores[idx][j]
                idx += 1
        # tie-break toward the plan's own budgets, so an unmatched shot falls where
        # the plan expected it rather than wherever the split happens to land
        total -= 1e-4 * sum(abs(k - budgets[j]) for j, k in enumerate(sizes))
        if best is None or total > best:
            best, best_sizes = total, sizes
    out, idx = [], 0
    for j, k in enumerate(best_sizes or [n]):
        out.extend([j] * k)
        idx += k
    return out[:n]


def _front_anchored_prompt(prompt: str, place: str, figure: str,
                           light: str = "") -> str:
    """Compose the render prompt: SHORT place, then the shot's ACTION, then
    figure/light tails.

    MEASURED FAILURE, twice, in opposite directions:
      * anchors APPENDED to a generic sentence were ignored (ego_v5: places never
        appeared in previs);
      * anchors PREPENDED as full clauses dominated the image, and every shot in a
        scene started with the same ~20 words (squid: all 10 shots drew the same
        glowing helmet-creature head).
    So the place goes first but SHORT (a few words, enough for continuity), the
    writer's shot-specific action sentence stays intact and early, and the light
    goes at the END where it colours the shot without overriding it. The figure is
    only appended when its words are absent, so a writer naming "the diver" is not
    forced to repeat a prop description.
    """
    p = (prompt or "").strip()
    place_short = _clip_clause(place, 6)
    light_short = _clip_clause(light, 8)
    parts: List[str] = []
    if place_short and not p.lower().startswith(place_short.lower()[:12]):
        parts.append(place_short)
    if p:
        parts.append(p)
    # a figure clause that is already present (any shared content noun) is skipped
    fig_words = content_words(figure) if figure else set()
    if fig_words and len(fig_words & content_words(p)) < max(1, len(fig_words) // 2):
        parts.append(_clip_clause(figure, 8))
    if light_short and len(content_words(light_short) & content_words(p)) < 2:
        parts.append(light_short)
    return ", ".join(x for x in parts if x)


# Shared place-noun overlap (Jaccard) below which a shot is not described as being
# at its assigned scene's place. Measured on the lighthouse draft: true matches
# score 0.31-0.43, the shots that were forced into the last scene score 0.0-0.1.
ANCHOR_PLACE_MIN_MATCH = 0.10


def anchor_plan(shots: Sequence[Shot], plan: Optional[Dict]) -> List[Shot]:
    """Bind every prompt to its planned scene's place/figure words.

    THE MEASURED REASON. The first plan-first run (a lighthouse, n=6) followed the
    3-scene plan in its `scene` labels, but each prompt described a different
    aspect of the place with different nouns, so the measured within-scene affinity
    was 0.06-0.07 and the writer still read as a montage. Telling the model to
    "reuse the nouns" is advice; the pipeline already has the mechanism -- canonical
    place/figure ANCHORS appended to every shot in a run at freeze
    (`visual_org.effective_prompt`, used by the workbench). The store draft path
    never applied them.

    Anchors are applied AFTER the prompt pass (so they cannot be rewritten away)
    and only append a clause that is not already present -- a writer who described
    the place is not duplicated. This is what the renderer needs to hold a place,
    and what makes the director's `reference_plan` group the scene into a hold
    (IP-Adapter then conditions the later shots on the anchor frame).
    """
    shots = list(shots)
    scenes = (plan or {}).get("scenes") or []
    if not scenes or not shots:
        return shots
    assignment = _assign_scenes(shots, scenes)
    out: List[Shot] = []
    for i, s in enumerate(shots):
        j = assignment[i] if i < len(assignment) else len(scenes) - 1
        chosen = scenes[j]
        place = _clip_clause(chosen.get("place", ""), 8)
        figure = _clip_clause(chosen.get("figure", ""), 6)
        light = _clip_clause(chosen.get("light", ""), 10)
        subject = _clip_clause(chosen.get("figure", ""), 4)
        # A SHOT THAT DOES NOT BELONG TO ITS SCENE'S PLACE IS NOT TOLD IT DOES. The
        # assignment is order-monotone, so shots after the last strong match all
        # land in the final scene (lighthouse: 1/1/8, and "calm bay at dawn" /
        # "lighthouse balcony" were prefixed "a rocky coastline"). Below the
        # threshold the shot keeps its scene label, figure and light but gets no
        # place words -- the writer's own place is the truth about that image.
        if _place_match(s.prompt, chosen.get("place", "")) < ANCHOR_PLACE_MIN_MATCH:
            place = ""
        out.append(replace(
            s,
            prompt=_front_anchored_prompt(s.prompt, place, figure, light),
            # THE PLAN IS THE SCENE. The writer's own scene labels were the problem
            # (one per shot, or cosmetic); the plan commits to 2-3 places, so the
            # anchored film is cut on those scenes. The sequence director then sees
            # high within-scene affinity and holds them (IP conditioning follows).
            scene=f"plan{j + 1}",
            # The grade travels WITH the shot: the director may rewrite scene
            # labels (planN -> movement ids), and a scene-label-keyed palette map
            # then falls back to neutral. Measured on the squid draft: all three
            # store scenes were neutral because of exactly that rewrite.
            palette=str(chosen.get("palette") or ""),
            subject=(s.subject or subject)))
    return out


def write_scene_plan(concept: str, n_shots: Optional[int], idea_text: str,
                     showrunner: str = SHOWRUNNER, temperature: float = 0.8,
                     attempts: int = 2,
                     verbose: bool = True) -> Tuple[Optional[Dict], Dict]:
    """Concept + ideas -> 2-3 scenes with a place, a figure and a shot budget.

    Cheap on purpose: this is a plan, not prose (a few hundred output tokens). It
    is retried if the reply will not normalise (wrong scene count, unparseable
    shot counts), and the writer proceeds without it if both attempts fail -- the
    plan is a constraint, not a dependency.
    """
    info: Dict = {"attempts": 0, "error": None}
    if not n_shots:
        return None, {"attempts": 0, "error": "no shot count to plan"}
    import ollama
    for i in range(1, max(1, int(attempts)) + 1):
        info["attempts"] = i
        if verbose:
            print(f"  [story] scene plan attempt {i}/{attempts}: choosing 2-3 "
                  f"places before shots (short)", flush=True)
        try:
            resp = ollama.chat(
                model=showrunner,
                messages=[{"role": "system", "content": SCENE_PLAN_SYSTEM},
                          {"role": "user",
                           "content": _scene_plan_user(concept, n_shots, idea_text)}],
                format=SCENE_PLAN_SCHEMA,
                options={"temperature": temperature, "num_ctx": OLLAMA_NUM_CTX,
                         "num_predict": 700},
            )
            content = resp["message"]["content"]
            data, _mode = parse_reply(content)
        except Exception as exc:                     # noqa: BLE001
            info["error"] = f"{type(exc).__name__}: {str(exc)[:140]}"
            if verbose:
                print(f"  [story] scene plan attempt {i} failed: {info['error']}")
            continue
        plan = _normalise_scene_plan(data, n_shots)
        if plan is None:
            info["error"] = "reply was not a usable 2-3 scene plan"
            if verbose:
                print(f"  [story] scene plan attempt {i} unusable: "
                      f"{info['error']}")
            continue
        info["plan"] = plan
        if verbose:
            summary = " · ".join(
                f"{j}. {sc['place'][:34]} ({sc['figure'][:18]}) x{sc['shots']}"
                for j, sc in enumerate(plan["scenes"], 1))
            print(f"  [story] scene plan: {len(plan['scenes'])} scene(s), "
                  f"{sum(sc['shots'] for sc in plan['scenes'])} shots -- {summary}",
                  flush=True)
        return plan, info
    if verbose:
        print("  [story] no usable scene plan -- writing without one", flush=True)
    return None, info


def _showrunner_user(concept: str, n_shots: Optional[int], idea_text: str,
                     repair_ctx: str = "",
                     scene_plan: Optional[Dict] = None) -> str:
    """The showrunner's user message. One definition, so the repair addendum
    cannot drift from the normal prompt.

    The requested count and its beat roster sit ABOVE the scouts' ideas on
    purpose: the count is the brief, the ideas are raw material. When the roster
    was missing from this message the model treated "write exactly N shots" as a
    suggestion and returned whatever number its own sense of pacing wanted (a
    13-shot request came back as 6). The planned scenes sit between them: the
    count and the roster are the shape, the plan is the place/figure constraint,
    the ideas are only material.
    """
    parts = [f"Concept: {concept}"]
    if n_shots:
        parts.append(f"Write exactly {n_shots} shots.")
        parts.append(beat_layout(n_shots))
    if scene_plan:
        parts.append("\n" + _scene_plan_text(scene_plan))
    parts.append("\nThe scouts' raw ideas:\n\n" + idea_text)
    if repair_ctx:
        parts.append("\n" + repair_ctx)
    parts.append("\nWrite the film's shot list.")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# the extension pass: reaching the requested SHOT COUNT without re-writing the story
# --------------------------------------------------------------------------- #
#
# The measured failure: a 13-shot request came back as 6 shots, three times. The
# film that came back could be good -- one of those replies had a clean story --
# but at 6 shots the target length is unreachable, because length is set by shot
# count, not by hold time (holds are clamped to the renderer's motion budget).
#
# Re-rolling a 10-minute generation to ask for a longer film is the wrong tool: it
# throws away a story that already passed. So the extension pass runs AFTER the
# story gate, on an approved list, and only ADDS COVERAGE inside the scenes that
# exist. Its structural fields are PINNED to the insertion point:
#
#   beat, scene, movement  <- copied from the shot the insert follows
#   camera                 <- 'none' (generation is static; moves are finishing specs)
#   subject               <- the model's, or the anchor's when it gives none
#
# That is what makes it incapable of breaking a story that already passed: the
# beats can only repeat a beat they are inserted into (so order cannot go
# backwards), the scene cannot change, and the last shot cannot be displaced.
# Only the image, the framing scale, the purpose and the energy are the model's.
EXTEND_SCHEMA = {
    "type": "object",
    "properties": {
        "shots": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "scale": {"type": "string", "enum": SCALES},
                "seconds": {"type": "number"},
                "energy": {"type": "number"},
                "purpose": {"type": "string", "enum": PURPOSES},
                # the 1-based number of the existing shot this one follows
                "after": {"type": "integer"},
            },
            "required": ["prompt", "scale", "seconds", "energy", "purpose",
                         "after"],
        }},
    },
    "required": ["shots"],
}

EXTEND_SYSTEM = (
    "You are the editor of a shot list for a short atmospheric film. The story "
    "is ALREADY APPROVED and must not change; it is simply shorter than the "
    "target length. Your only job is to add COVERAGE inside the scenes that "
    "already exist: more angles on the same place and action, never a new scene, "
    "never a new story beat, never a new ending.\n"
    "\n"
    "For every added shot:\n"
    "- set 'after' to the 1-based number of the existing shot it follows. Never "
    "follow the FINAL shot: the film must end on the ending it already has.\n"
    "- same place, same figure, same light and materials as the shot it follows: "
    "reuse that shot's nouns. It is another angle on that image, not a new place.\n"
    "- use a framing scale DIFFERENT from the shot it follows, or it reads as a "
    "jump cut.\n"
    "- 'purpose' is what the added shot covers inside the scene: master, insert "
    "or reaction.\n"
    "- 'prompt' is ONE concrete STATIC image, at least 8 words: what is IN FRAME, "
    "not what the camera does. Never write 'zooming', 'panning', 'tilting', "
    "'pushing in' or 'the camera'. BAD: 'zooming out from the man to reveal the "
    "room'. GOOD: 'the dim room from the doorway, the man small under the "
    "glowing window, dust in the light'.\n"
    "- Do not repeat an existing shot's image: a different angle on the same "
    "action, same place and light.\n"
    "- Spread the added shots across the film's scenes instead of piling them all "
    "into the first one.\n"
    "- 'seconds' 1.5 to 3.6; 'energy' 0 to 0.5. This pass adds no camera moves.\n"
    "Return JSON only."
)


def _extension_user(shots: Sequence[Shot], target: int) -> str:
    """The approved film as a numbered list, plus exactly how many shots to add."""
    missing = max(0, int(target) - len(shots))
    lines = [f"THE APPROVED FILM ({len(shots)} shots, in order). It must become "
             f"{int(target)} shots.", ""]
    for i, s in enumerate(shots, 1):
        lines.append(f"{i}. [scene {s.scene or '?'}] [{s.beat or '?'}] "
                     f"[{s.scale}] {s.prompt}")
    lines.append("")
    lines.append(f"Add exactly {missing} shot(s) as coverage inside these scenes. "
                 f"Set 'after' to the number of the shot each one follows.")
    return "\n".join(lines)


def _other_scale(scale: str) -> str:
    """A framing scale next to `scale`, so an insert cannot sit at the same size
    as the shot before it (that is the planner's R1 jump cut)."""
    if scale not in SCALES:
        return SCALES[2]
    i = SCALES.index(scale)
    return SCALES[i + 1] if i + 1 < len(SCALES) else SCALES[i - 1]


def _coverage_shot(item: dict, prev: Shot) -> Shot:
    """One model-supplied insert, with every structural field pinned.

    The beat is the beat it COVERS, not a beat the model chooses. Pinning it is
    the whole safety property of the extension: a run of inserts can only repeat
    an existing beat, so establish/turn/resolve and their order survive an
    arbitrary number of added shots. A shot inserted after the establish is
    labelled develop, because only shot 1 may be an establish.
    """
    beat = prev.beat if prev.beat in BEATS else "develop"
    if beat == "establish":
        beat = "develop"
    scale = str(item.get("scale") or "")
    if scale not in SCALES or scale == prev.scale:
        scale = _other_scale(prev.scale)
    purpose = str(item.get("purpose") or "")
    if purpose not in PURPOSES:
        purpose = "insert"
    try:
        energy = float(item.get("energy", 0.2))
    except (TypeError, ValueError):
        energy = 0.2
    energy = max(0.0, min(MAX_RENDERABLE_MOTION, energy))
    try:
        seconds = float(item.get("seconds", 2.0))
    except (TypeError, ValueError):
        seconds = 2.0
    seconds = max(1.5, min(3.6, seconds))
    subject = str(item.get("subject") or "").strip() or prev.subject
    return Shot(prompt=str(item.get("prompt") or "").strip(), scale=scale,
                camera="none", seconds=seconds, subject=subject,
                scene=prev.scene, energy=energy, beat=beat, purpose=purpose,
                link="match-action", movement=prev.movement)


def _merge_extension(shots: Sequence[Shot], items: Sequence[dict],
                     limit: Optional[int] = None) -> Tuple[List[Shot], List[str]]:
    """Insert the model's coverage shots where it says they go.

    Returns `(merged, dropped)`; `dropped` is the reasons, for the log -- an
    insert is DROPPED rather than allowed to damage the approved film. Duplicate
    images, inserts with no usable `after`, and terse one-line prompts are the
    measured junk a local model adds when asked for "more shots".

    Ordering is by anchor: all inserts naming the same `after` stay in the order
    the model gave them, immediately after that shot.
    """
    dropped: List[str] = []
    if len(shots) < 2:
        return list(shots), ["the film is too short to insert coverage"]
    last = len(shots)
    seen = [content_words(s.prompt) for s in shots]
    inserts: Dict[int, List[Shot]] = {}
    added = 0
    for item in items or []:
        if limit is not None and added >= int(limit):
            break
        if not isinstance(item, dict):
            dropped.append("not an object")
            continue
        prompt = str(item.get("prompt") or "").strip()
        if len(prompt.split()) < 8:
            dropped.append(f"too terse: {prompt[:40]!r}")
            continue
        try:
            after = int(item.get("after"))
        except (TypeError, ValueError):
            dropped.append(f"no usable 'after': {prompt[:40]!r}")
            continue
        # never after the final shot: the ending is not up for revision. An
        # out-of-range index is clamped rather than fatal, because the model's
        # counting is the least reliable part of the reply.
        anchor = max(1, min(last - 1, after))
        words = content_words(prompt)
        if any(jaccard(words, w) >= 0.75 for w in seen):
            dropped.append(f"duplicate image: {prompt[:40]!r}")
            continue
        seen.append(words)
        inserts.setdefault(anchor, []).append(_coverage_shot(item, shots[anchor - 1]))
        added += 1

    out: List[Shot] = []
    for i, s in enumerate(shots, 1):
        out.append(s)
        for add in inserts.get(i, []):
            if add.scale == out[-1].scale:
                add = replace(add, scale=_other_scale(out[-1].scale))
            out.append(add)
    return out, dropped


def extend_shot_list(concept: str, shots: Sequence[Shot], target: int,
                     showrunner: str = SHOWRUNNER, temperature: float = 0.8,
                     attempts: int = 3,
                     verbose: bool = True) -> Tuple[List[Shot], Dict]:
    """Ask the showrunner for coverage until the film reaches `target` shots.

    Only safe -- and only called -- on a list whose story already passed the
    gate; see the note above `EXTEND_SCHEMA`. `attempts` is a cap on the extra
    generations, not a target: the loop stops as soon as the count is reached, or
    the moment a pass adds nothing usable (a second identical ask will not
    differ).
    """
    info: Dict = {"target": int(target), "start": len(shots), "calls": [],
                  "dropped": 0}
    if not target or len(shots) >= int(target):
        return list(shots), info
    import ollama

    current = list(shots)
    for call in range(1, max(1, int(attempts)) + 1):
        missing = int(target) - len(current)
        if missing <= 0:
            break
        budget = min(3000, max(800, 300 * missing + 250))
        print(f"  [story] extension {call}/{attempts}: {len(current)}/{target} "
              f"shot(s) -- asking for {missing} more coverage", flush=True)
        try:
            resp = ollama.chat(
                model=showrunner,
                messages=[{"role": "system", "content": EXTEND_SYSTEM},
                          {"role": "user",
                           "content": _extension_user(current, int(target))}],
                format=EXTEND_SCHEMA,
                options={"temperature": temperature, "num_ctx": OLLAMA_NUM_CTX,
                         "num_predict": budget},
            )
            content = resp["message"]["content"]
            data, mode = parse_reply(content)
        except Exception as exc:  # noqa: BLE001 -- an extension must not kill a draft
            info["calls"].append({"call": call, "error":
                                  f"{type(exc).__name__}: {str(exc)[:160]}"})
            print(f"  [story] extension {call} failed: {type(exc).__name__}: "
                  f"{str(exc)[:110]}", flush=True)
            break
        if not isinstance(data, dict):
            why = json_failure(content)
            info["calls"].append({"call": call, "error": why})
            print(f"  [story] extension {call} reply did not parse -- {why}",
                  flush=True)
            break
        merged, dropped = _merge_extension(current, data.get("shots") or [],
                                           limit=missing)
        gained = len(merged) - len(current)
        info["calls"].append({"call": call, "returned": len(data.get("shots") or []),
                              "added": gained, "dropped": len(dropped),
                              "mode": mode, "reasons": dropped[:6]})
        info["dropped"] += len(dropped)
        if gained:
            current = merged
            print(f"  [story] extension {call}/{attempts}: +{gained} -> "
                  f"{len(current)}/{target} shot(s) (dropped {len(dropped)})",
                  flush=True)
        else:
            print(f"  [story] extension {call}/{attempts}: added nothing usable "
                  f"({len(dropped)} dropped)", flush=True)
            break
    if len(current) < int(target):
        print(f"  [story] extension stopped at {len(current)}/{target} shot(s) -- "
              f"the film stays shorter than the target", flush=True)
    return current, info


# --------------------------------------------------------------------------- #
# the prompt pass: the same film, described as images instead of sentences
# --------------------------------------------------------------------------- #
#
# Measured on the first rendered film (`ego`): the approved prompts were the
# last composite writing in the pipeline. 12 of 13 joined two images with a
# semicolon and many interpreted the shot ("symbolic of the ego's acceptance"),
# so the renderer drew one clause or the concept prior, and CLIP prompt->frame
# alignment had a mean margin of +0.018 over a DIFFERENT shot's prompt. The story,
# beats, scenes and count were fine; only the text was wrong.
#
# So this is a TEXT-ONLY pass that runs on the APPROVED list: one new prompt per
# shot, same order, same subject/place/action. It cannot change the story, the
# beat order, the shot count or the timing -- and a per-shot best-of keeps the
# original whenever the rewrite is not measurably better.
PROMPT_SCHEMA = {
    "type": "object",
    "properties": {"prompts": {"type": "array", "items": {"type": "string"}}},
    "required": ["prompts"],
}

PROMPT_SYSTEM = (
    "You are REWRITING the prompts of an APPROVED shot list for a short "
    "atmospheric film. The story, the order, the beats, the scenes and the shot "
    "count are FIXED. Do not add, remove or reorder shots.\n"
    "\n"
    "Return ONE new prompt per shot, in the same order. Each prompt must be ONE "
    "concrete STATIC image, 8 to 24 words, ONE sentence:\n"
    "- what the CAMERA SEES: physical things, materials, light, weather, framing.\n"
    "- no semicolon, no second sentence, no 'and then'.\n"
    "- no meaning, emotion or inner life: never 'ego', 'confrontation', "
    "'aftermath', 'awareness', 'symbolic', 'surrenders', 'acceptance', "
    "'realises', 'feels'.\n"
    "- no camera moves: never 'zooming', 'panning', 'tilting', 'the camera'.\n"
    "- KEEP the same subject, place and action the current prompt is about; "
    "sharpen it into a frame. Do not invent a new scene and do not fix the story.\n"
    "- INSIDE A SCENE the shots are different angles on ONE place/action: keep "
    "the place and figure words, but do NOT repeat the previous shot's sentence -- "
    "change the framing, the moment or the detail.\n"
    "- START with the shot's ACTION and its FIGURE (a person/character, not a "
    "prop): e.g. 'a scuba diver in a black wetsuit grips a dive light'. Keep the "
    "place to a few concrete words, and put the LIGHT/colour at the END. Never "
    "copy a long identical anchor block into every prompt -- the renderer will "
    "draw that block and every shot will look the same.\n"
    "\n"
    "BAD:  'A barefoot man walks across a dusty city floor; the shadow of his "
    "ego looms large as the confrontation begins.'  (two images + meaning)\n"
    "GOOD: 'A barefoot man walks across a dusty city floor, cracked concrete "
    "reflecting his long shadow.'  (one image)\n"
    "\n"
    "TIP: if your sentence wants a semicolon, the half before it is the image and "
    "the half after it is the meaning. Keep the first half and delete the second.\n"
    "Return JSON only."
)


def _prompt_pass_user(shots: Sequence[Shot]) -> str:
    lines = [f"THE APPROVED FILM ({len(shots)} shots, in order). Rewrite ONE "
             f"prompt per shot, same order:", ""]
    for i, s in enumerate(shots, 1):
        lines.append(f"{i}. [{s.beat or '?'} · {s.scale or '?'}] {s.prompt}")
    lines.append("")
    lines.append(f"Return exactly {len(shots)} prompts, one per shot.")
    return "\n".join(lines)


def _one_image_text(text: str) -> str:
    """The frame half of a rewritten prompt.

    Measured model habit: even told "no semicolon", granite4.1:8b returns
    "image; interpretation" (12 of 13 prompts on the first run). The first clause
    is the image in every measured case, so the pass keeps it and drops the
    interpretation rather than accepting text the renderer will read as two
    images. This is a formatting cut on the REWRITE only -- the writer's approved
    prompt is untouched unless the cleaned rewrite is measurably better.
    """
    t = (text or "").strip()
    if ";" in t:
        t = t.split(";", 1)[0].strip()
    t = re.split(r"(?<=[.!?])\s+", t)[0].strip()
    # a dangling conjunction/comma from the cut is not part of the image
    t = re.sub(r"[,;:\s]+$", "", t).strip()
    return t


def _better_prompt(old: str, new: str) -> Tuple[str, bool]:
    """Pick the better of the approved prompt and the rewritten one.

    Returns `(text, changed)`. A rewrite is accepted only when it has FEWER
    measured prompt-shape issues and is still at least the brief's 8-word floor;
    otherwise the approved text stays exactly as the writer wrote it. Field-value
    leaks count as issues: a model that writes "..., camera: none" into a prompt
    has made it worse, whatever its shape.
    """
    def issues(text: str) -> int:
        return len(_prompt_issues(text)) + len(prompt_leaks(text))

    before = issues(old)
    candidates = [new, _one_image_text(new)]
    best, best_after = old, before
    for cand in candidates:
        if len(cand.split()) < 8:
            continue
        after = issues(cand)
        if after < best_after or (after == best_after and cand != old
                                  and best == old):
            best, best_after = cand, after
    return best, best != old


def rewrite_prompts(shots: Sequence[Shot], showrunner: str = PROMPT_PASS_MODEL,
                    temperature: float = 0.4,
                    verbose: bool = True,
                    attempts: int = 2) -> Tuple[List[Shot], Dict]:
    """Rewrite every prompt as ONE concrete image, keeping the film unchanged.

    Runs after `extend_shot_list` (so it also cleans the inserted coverage shots)
    and only on a shot list whose story already passed. It is a TEXT pass: the
    beat, scene, scale, purpose, link, movement and timing are the approved ones,
    and a per-shot best-of means a rewrite is only accepted when it has fewer
    measured prompt-shape issues than the original. Empty/garbage replies leave
    the approved prompts untouched.
    """
    info: Dict = {"requested": len(shots), "returned": 0, "rewritten": 0,
                  "kept": 0, "flagged": [], "error": None, "attempts": 0}
    if not shots:
        return list(shots), info
    import ollama

    budget = min(4000, max(1200, 300 + 220 * len(shots)))
    print(f"  [story] prompt pass: rewriting {len(shots)} prompt(s) as single "
          f"images (<= {budget} tokens)", flush=True)
    # A count mismatch is a reply-format failure, not a reason to ship the raw
    # writer text: measured once on a real 13-shot draft, mistral returned 12
    # prompts for 13 shots and the whole pass was skipped. Retry on it.
    for attempt in range(1, max(1, int(attempts)) + 1):
        info["attempts"] = attempt
        try:
            resp = ollama.chat(
                model=showrunner,
                messages=[{"role": "system", "content": PROMPT_SYSTEM},
                          {"role": "user", "content": _prompt_pass_user(shots)}],
                format=PROMPT_SCHEMA,
                options={"temperature": temperature, "num_ctx": OLLAMA_NUM_CTX,
                         "num_predict": budget},
            )
            content = resp["message"]["content"]
            data, mode = parse_reply(content)
        except Exception as exc:  # noqa: BLE001 -- a text pass must not kill a draft
            info["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            print(f"  [story] prompt pass attempt {attempt} failed: "
                  f"{info['error']}", flush=True)
            continue
        if not isinstance(data, dict) or not isinstance(data.get("prompts"), list):
            info["error"] = json_failure(content)
            print(f"  [story] prompt pass attempt {attempt} reply did not parse -- "
                  f"{info['error']}", flush=True)
            continue
        prompts = [str(x).strip() for x in data["prompts"]]
        info["returned"] = len(prompts)
        if len(prompts) != len(shots):
            info["error"] = (f"returned {len(prompts)} prompt(s) for "
                             f"{len(shots)} shot(s) -- the count must match")
            print(f"  [story] prompt pass attempt {attempt}: {info['error']}",
                  flush=True)
            continue
        break
    else:
        print(f"  [story] prompt pass gave up after {attempts} attempt(s) -- "
              f"keeping the writer's prompts", flush=True)
        return list(shots), info
    info["error"] = None

    out: List[Shot] = []
    for old, new in zip(shots, prompts):
        text, changed = _better_prompt(old.prompt, new)
        if changed:
            out.append(replace(old, prompt=text))
            info["rewritten"] += 1
        else:
            out.append(old)
            info["kept"] += 1
    info["flagged"] = [i + 1 for i, s in enumerate(out)
                       if _prompt_issues(s.prompt)]
    print(f"  [story] prompt pass: rewrote {info['rewritten']}/{len(shots)} "
          f"prompt(s) as single images"
          + (f"; {len(info['flagged'])} still flagged" if info["flagged"] else ""),
          flush=True)
    if verbose and info["rewritten"]:
        for i, (old, new) in enumerate(zip(shots, out), 1):
            if old.prompt != new.prompt:
                print(f"    {i}. {new.prompt[:120]}")
    return out, info


# --------------------------------------------------------------------------- #
# THE FILM READS ITS OWN WRITING.
#
# Measured on the first "skydiving" draft: the story gate passed it, and the film
# still ended on "skydiver's silhouette vanished" / "no trace of the figure" (the
# renderer draws what a prompt NAMES, so naming an absence draws the thing), the
# last five shots stalled on the same field, and the figure left the frame. None
# of that is a rule the gate checks. Two readers find it, one list of complaints:
#   * RULES   -- absence wording, and the figure missing from the closing run;
#                cheap, exact, and not subject to a small model's mood.
#   * CRITIC  -- a model reads the numbered list against the concept and names
#                shots that repeat a neighbour, do not serve the concept, cannot
#                be photographed, or stall. It only NOMINATES; it never rewrites.
# Each flagged shot is then rewritten ALONE (concept + neighbours + the exact
# complaint) and the rewrite replaces it only if it is measurably no worse. The
# loop repeats up to `rounds` times and keeps what it fixed.
# --------------------------------------------------------------------------- #
ABSENCE_RE = re.compile(
    r"\b(no\s+(?:trace|sign|sight|one|person|figure|body|people)|nobody|"
    r"vanish\w*|disappear\w*|gone|absent|deserted|devoid|"
    r"nothing\s+(?:left|remains))\b", re.I)
GENERIC_SUBJECT = frozenset({"person", "figure", "man", "woman", "someone",
                             "character", "people"})
CRITIC_SCHEMA = {
    "type": "object",
    "properties": {"events": {"type": "array", "items": {
        "type": "object",
        "properties": {"event": {"type": "string"},
                       "first_shot": {"type": "integer"},
                       "last_shot": {"type": "integer"}},
        "required": ["event", "first_shot", "last_shot"]}}},
    "required": ["events"],
}
EVENTS_SCHEMA = {"type": "object",
                 "properties": {"events": {"type": "array",
                                           "items": {"type": "string"}}},
                 "required": ["events"]}
EVENTS_SYSTEM = (
    "List, in the order they physically happen, the 4 to 7 events a short film of "
    "this concept must show (e.g. for a skydiver: leaps from the plane, "
    "free-falls, the parachute opens, glides down, lands). One short phrase "
    "each, in time order. Return JSON only."
)
CRITIC_SYSTEM = (
    "You are a film editor. You are given a concept, its EVENTS in story order, "
    "and a numbered shot list. For EACH event, in the order given, report "
    "first_shot: the number of the FIRST shot whose image literally shows that "
    "event, and last_shot: the number of the LAST shot that shows it. Use 0 for "
    "both if no shot shows it. Judge only by what each image shows. Return one "
    "entry per event, same order, JSON only."
)
REPAIR_SCHEMA = {
    "type": "object",
    "properties": {"fixes": {"type": "array", "items": {
        "type": "object",
        "properties": {"shot": {"type": "integer"}, "prompt": {"type": "string"}},
        "required": ["shot", "prompt"]}}},
    "required": ["fixes"],
}
REPAIR_SYSTEM = (
    "You are fixing SOME shots of an approved short-film shot list. You are "
    "given the whole list, the story events it must show in order, and which "
    "shots have problems (marked >>) with what is wrong with each. Rewrite ONLY "
    "the marked shots so that, read in order with the unmarked shots, the film "
    "tells the concept's story with each event shown once and in order. Each "
    "rewrite is ONE concrete static image, 8 to 24 words, ONE sentence, naming "
    "what the camera SEES and the figure's action; never describe an absence "
    "('no trace', 'vanished', 'gone'); no camera moves, no meaning or emotion, "
    "no semicolon, no framing words like 'a medium shot of'. Keep the place and "
    "light of the shots around it. A shot with 'MUST SHOW' has to depict exactly "
    "that event, even if that changes its content a lot. Change what the frame "
    "SHOWS; do not just delete the bad words. Return one entry per marked shot: {shot, prompt}. "
    "Return JSON only."
)


def _strip_absence(text: str) -> str:
    """Remove the comma-clause that names an absence; the harm is gone, the rest stays.

    The fallback when a model's rewrite is refused: a prompt that merely LACKS the
    phrase "no trace of the figure" is strictly safer than one that has it.
    """
    parts = [p.strip() for p in (text or "").split(",")]
    kept = [p for p in parts if p and not ABSENCE_RE.search(p)]
    return ", ".join(kept)


def _mentions_subject(text: str, subject: str) -> bool:
    """Does this prompt name the film's figure (a prefix match, so plurals count)?"""
    want = [w for w in content_words(subject or "") if w not in GENERIC_SUBJECT]
    if not want:
        return True          # nothing specific to look for
    have = content_words(text or "")
    return any(w in have or any(h.startswith(w[:5]) for h in have if len(w) >= 5)
               for w in want)


def prompt_flaws(shots: Sequence[Shot]) -> Dict[int, List[str]]:
    """The RULE half of the self-critique: `{1-based shot: [reason, ...]}`.

    Absence wording is flagged wherever it appears. The figure is flagged only
    when the film is clearly ABOUT one (it is named in at least half the shots)
    and a closing run of 2+ shots does not name it -- a cutaway mid-film is
    legitimate, a film that forgets its figure by the end is not.
    """
    out: Dict[int, List[str]] = {}
    for i, s in enumerate(shots, 1):
        m = ABSENCE_RE.search(getattr(s, "prompt", "") or "")
        if m:
            out.setdefault(i, []).append(
                f"describes an absence ('{m.group(0)}') -- the renderer draws what "
                f"is named, so it draws that thing; show what IS in frame")
    n = len(shots)
    subjects = [getattr(s, "subject", "") or "" for s in shots]
    subject = max(set(subjects), key=subjects.count) if subjects else ""
    if n >= 4 and subject:
        has = [_mentions_subject(s.prompt, subject) for s in shots]
        if sum(has) * 2 >= n:
            tail = 0
            while tail < n and not has[n - 1 - tail]:
                tail += 1
            if tail >= 2:
                for i in range(n - tail + 1, n + 1):
                    out.setdefault(i, []).append(
                        f"the figure ({subject}) is not in the closing run -- the "
                        f"film ends without the one it is about")
    return out


def _flaw_count(text: str) -> int:
    """How many measurable problems this ONE prompt text has."""
    return (len(_prompt_issues(text)) + len(prompt_leaks(text))
            + (1 if ABSENCE_RE.search(text or "") else 0))


def plan_events(concept: str, model: str, temperature: float = 0.2
                ) -> Tuple[List[str], Optional[str]]:
    """ONE short model call from the CONCEPT ALONE: its events in time order.

    Asked before the model has seen any shot. Measured: when the model listed the
    events after reading the draft, it copied the draft's own order (parachute
    before free-fall) and found nothing wrong.
    """
    import ollama
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": EVENTS_SYSTEM},
                      {"role": "user", "content": f"CONCEPT: {concept}"}],
            format=EVENTS_SCHEMA,
            options={"temperature": temperature, "num_ctx": OLLAMA_NUM_CTX,
                     "num_predict": 250})
        data, _mode = parse_reply(resp["message"]["content"])
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {str(exc)[:160]}"
    items = (data or {}).get("events") if isinstance(data, dict) else None
    events = [str(e).strip()[:80] for e in (items or []) if str(e).strip()][:7]
    if len(events) < 3:
        return [], f"the model listed {len(events)} event(s) for the concept"
    return events, None


def read_events(concept: str, shots: Sequence[Shot], model: str,
                events: Sequence[str],
                temperature: float = 0.1
                ) -> Tuple[List[str], Dict[int, Tuple[int, int]], Optional[str]]:
    """ONE model call: where is each of the GIVEN events shown?

    Returns `(events, {event number: (first shot, last shot)}, error)`; `(0, 0)`
    means no shot shows it. The model only LOCATES events; `order_flaws` (plain
    code) decides what is wrong. Measured on real prompts, per-shot labels were
    unusable from 7-8B models (mistral echoed the shot numbers, an object per shot
    skipped 6 of 15), while "first and last shot of each event" -- about five
    answers -- came back identical across runs. The reply is matched to the
    events BY POSITION and must have one entry per event. Numbers outside the
    list are treated as 0. A reply that locates fewer than half the events is
    unreadable and flags nothing (an empty reading must not mean "no event is
    shown").
    """
    import ollama
    events = list(events)
    lines = [f"CONCEPT: {concept}", "", "EVENTS, in story order:"]
    lines += [f"{i}. {e}" for i, e in enumerate(events, 1)]
    lines += ["", f"SHOT LIST ({len(shots)} shots):"]
    for i, s in enumerate(shots, 1):
        lines.append(f"{i}. {s.prompt[:160]}")
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": CRITIC_SYSTEM},
                      {"role": "user", "content": "\n".join(lines)}],
            format=CRITIC_SCHEMA,
            options={"temperature": temperature, "num_ctx": OLLAMA_NUM_CTX,
                     "num_predict": 500})
        data, _mode = parse_reply(resp["message"]["content"])
    except Exception as exc:  # noqa: BLE001 -- a critic must not kill a draft
        return [], {}, f"{type(exc).__name__}: {str(exc)[:160]}"
    if not isinstance(data, dict) or not isinstance(data.get("events"), list):
        return [], {}, "the critic's reply did not parse"
    items = data["events"]
    if len(items) != len(events):
        return events, {}, (f"the critic answered {len(items)} entr(ies) for "
                            f"{len(events)} event(s) -- unreadable, nothing "
                            f"flagged from it")
    spans: Dict[int, Tuple[int, int]] = {}
    for n_ev, item in enumerate(items, 1):
        try:
            first, last = int(item.get("first_shot")), int(item.get("last_shot"))
        except (TypeError, ValueError, AttributeError):
            first = last = 0
        if not (1 <= first <= len(shots)) or not (1 <= last <= len(shots)):
            first = last = 0
        spans[n_ev] = (first, max(first, last))
    located = sum(1 for f, _l in spans.values() if f)
    if located * 2 < len(events):
        return events, {}, (f"the critic located {located}/{len(events)} event(s) "
                            f"-- unreadable, nothing flagged from it")
    return events, spans, None


def order_flaws(events: Sequence[str], spans: Dict[int, Tuple[int, int]], n: int,
                unplaced: Optional[List[int]] = None) -> Dict[int, List[str]]:
    """Plain code: from `{event: (first shot, last shot)}` find what breaks the order.

    * an event whose LAST shot comes after a later event has already begun, or
      whose FIRST shot comes after a later event's first, goes backwards -- that
      shot is flagged;
    * an event no shot shows is MISSING, and is placed in the slot between the
      shots of its neighbours in the story, on a shot not already flagged. The
      position is chosen here, never by the model: measured, a model asked to
      "add the missing leap" rewrote shot 8 of 15 into a plane door after the
      landing.
    Returns `{shot: [reason]}`. An event with NO free slot (its neighbours are
    adjacent shots) cannot be fixed by rewriting -- that needs a shot inserted --
    so its number is appended to `unplaced` instead of guessed.
    """
    out: Dict[int, List[str]] = {}
    if len(events) < 2 or n < 2 or not spans:
        return out
    E = len(events)
    present = {e: s for e, s in spans.items() if s[0]}
    for e in sorted(present):
        first, last = present[e]
        for f in range(e + 1, E + 1):
            if f not in present:
                continue
            if present[f][0] < last:
                out.setdefault(last, []).append(
                    f"goes backwards: shows '{events[e - 1]}' after "
                    f"'{events[f - 1]}' has already begun (shot {present[f][0]})")
                break
            if present[f][0] < first:
                out.setdefault(first, []).append(
                    f"goes backwards: '{events[e - 1]}' appears only after "
                    f"'{events[f - 1]}' (shot {present[f][0]})")
                break
    claimed = set(out)
    for e in range(1, E + 1):
        if e in present:
            continue
        lo = max([present[x][1] for x in present if x < e] or [0]) + 1
        hi = min([present[x][0] for x in present if x > e] or [n + 1]) - 1
        slot = [k for k in range(lo, hi + 1) if k not in claimed]
        if not slot:
            if unplaced is not None:
                unplaced.append(e)
            continue
        k = slot[len(slot) // 2]
        claimed.add(k)
        out[k] = [f"MUST SHOW event {e}: {events[e - 1]} -- no shot shows it and "
                  f"this is where it belongs in the story order"]
    return out


def _clean_fix(text: str) -> str:
    text = _one_image_text(str(text or ""))
    return re.sub(r"^(?:an?|the)\s+(?:extreme\s+)?(?:close(?:-up)?|medium|wide|"
                  r"long|establishing)\s+(?:shot|view)\s+(?:of|on)\s+", "", text,
                  flags=re.I)


def _repair_batch(concept: str, shots: Sequence[Shot], flagged: Dict[int, List[str]],
                  events: Sequence[str], model: str,
                  temperature: float) -> Dict[int, str]:
    """ONE call: the whole list, the flagged shots marked, `{shot: new prompt}`.

    Every rewrite is written with the FULL film in view, so a fix can line up with
    the shots around it (a single-shot call saw only two neighbours and, measured,
    turned a mid-air shot into a landing). Unknown shot numbers are dropped.
    """
    import ollama
    lines = [f"CONCEPT: {concept}"]
    if events:
        lines.append("STORY EVENTS, IN ORDER: " + " -> ".join(events))
    lines += ["", f"SHOT LIST ({len(shots)} shots; >> marks the shots to fix):"]
    for i, s in enumerate(shots, 1):
        mark = ">> " if i in flagged else "   "
        lines.append(f"{mark}{i}. [{s.beat or '?'} · {s.scale or '?'}] {s.prompt}")
        if i in flagged:
            lines.append("      WRONG: " + " | ".join(flagged[i]))
    lines.append("")
    lines.append("Return fixes for shots " + ", ".join(str(i) for i in sorted(flagged))
                 + " only.")
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": REPAIR_SYSTEM},
                      {"role": "user", "content": "\n".join(lines)}],
            format=REPAIR_SCHEMA,
            options={"temperature": temperature, "num_ctx": OLLAMA_NUM_CTX,
                     "num_predict": 120 + 90 * len(flagged)})
        data, _mode = parse_reply(resp["message"]["content"])
    except Exception:  # noqa: BLE001
        return {}
    out: Dict[int, str] = {}
    if isinstance(data, dict) and isinstance(data.get("fixes"), list):
        for f in data["fixes"]:
            try:
                i = int(f.get("shot"))
            except (TypeError, ValueError, AttributeError):
                continue
            text = _clean_fix(f.get("prompt"))
            if i in flagged and text:
                out[i] = text
    return out


def critique_and_repair(concept: str, shots: Sequence[Shot],
                        model: str = PROMPT_PASS_MODEL, rounds: int = 2,
                        temperature: float = 0.4, max_repairs: int = 6,
                        verbose: bool = True) -> Tuple[List[Shot], Dict]:
    """Check the story order, fix the shots that break it, keep what is better.

    Per round: ONE model call labels every shot with the story event it shows
    (`read_events`); plain code finds what is out of order or missing and picks
    the slot for each missing event (`order_flaws`), and adds the rule flags
    (`prompt_flaws`); then ONE repair call gets the whole list with the flagged
    shots marked (`_repair_batch`). A rewrite replaces the original only when it is
    8-24 words, has no more measurable problems than the original, and (for a
    figure complaint) names the figure. If a later round finds MORE flags than the
    round before, that round's rewrites are reverted. Only `prompt` text changes.

    Returns `(shots, info)`: `info["events"]`, `info["spans"]` (event -> first/last shot), `info["flagged"]` (every nomination and whether it was fixed),
    `info["remaining"]` (what the rules still flag), `info["reverted"]`.
    """
    shots = list(shots)
    info: Dict = {"rounds": 0, "events": [], "spans": {}, "flagged": [],
                  "fixed": 0, "remaining": {}, "critic_error": None,
                  "reverted": False}
    if len(shots) < 2:
        return shots, info
    prev_score: Optional[int] = None
    snapshot: List[Shot] = list(shots)
    planned_events: List[str] = []
    plan_err: Optional[str] = None
    fixed_last = 0
    for rnd in range(1, max(1, int(rounds)) + 1):
        info["rounds"] = rnd
        rule = prompt_flaws(shots)
        if not planned_events and not plan_err:
            planned_events, plan_err = plan_events(concept, model)
        events, spans, err = (read_events(concept, shots, model, planned_events)
                              if planned_events else ([], {}, plan_err))
        if err:
            info["critic_error"] = err
            events, spans = [], {}             # an unreadable reading flags nothing
        if events and rnd == 1:
            info["events"] = events
            info["spans"] = {str(k): list(v) for k, v in sorted(spans.items())}
        unplaced: List[int] = []
        order = (order_flaws(events, spans, len(shots), unplaced)
                 if events else {})
        if rnd == 1 and unplaced:
            info["unplaced"] = [events[e - 1] for e in unplaced]
        merged: Dict[int, List[str]] = {}
        for src in (order, rule):
            for i, rs in src.items():
                merged.setdefault(i, []).extend(rs)
        score = len(merged) + len(unplaced)   # an unplaceable event is still a defect
        if verbose:
            print(f"  [story] critique round {rnd}/{rounds}: {len(order)} shot(s) "
                  f"break the story order, {len(rule)} flagged by the rules"
                  + (f" (critic unavailable: {err})" if err else ""), flush=True)
            if events and rnd == 1:
                print("  [story] story events: " + " -> ".join(events), flush=True)
                print("  [story] event spans (first-last shot): "
                      + " ".join(f"{k}:{a}-{b}" for k, (a, b) in sorted(spans.items())),
                      flush=True)
            if unplaced and rnd == 1:
                print("  [story] no free shot to show: "
                      + ", ".join(events[e - 1] for e in unplaced)
                      + " (its neighbours in the story are adjacent shots; "
                        "fixing it needs a shot inserted)", flush=True)
        if prev_score is not None and score > prev_score:
            shots = list(snapshot)             # the last round made it worse
            info["reverted"] = True
            info["fixed"] -= fixed_last
            if verbose:
                print(f"  [story] round {rnd - 1}'s rewrites left more problems "
                      f"({score} > {prev_score}) -- reverted them", flush=True)
            break
        if not merged:
            break
        snapshot = list(shots)
        prev_score = score
        pick = sorted(merged, key=lambda k: (k not in order, k))[:max(1, max_repairs)]
        flagged = {i: merged[i] for i in pick}
        if verbose:
            print(f"  [story] repair shot 1/1 (#{','.join(str(i) for i in sorted(flagged))}"
                  f"): fixing {len(flagged)} shot(s) against the whole list",
                  flush=True)
        fixes = _repair_batch(concept, shots, flagged, events, model, temperature)
        fixed_now = 0
        for i in sorted(flagged):
            reasons = flagged[i]
            before = shots[i - 1].prompt
            cand = fixes.get(i)
            ok = bool(cand) and 8 <= len(cand.split()) <= MAX_PROMPT_WORDS \
                and cand != before and _flaw_count(cand) <= _flaw_count(before)
            if ok and any("figure" in r and "closing run" in r for r in reasons):
                ok = _mentions_subject(cand, shots[i - 1].subject)
            if ok and i in rule:      # a rule complaint must actually be gone
                ok = not ABSENCE_RE.search(cand)
            if not ok and i in rule and ABSENCE_RE.search(before):
                stripped = _strip_absence(before)
                if len(stripped.split()) >= 8 and _flaw_count(stripped) <= \
                        _flaw_count(before):
                    cand, ok = stripped, True
            info["flagged"].append({"round": rnd, "shot": i, "reasons": reasons,
                                    "source": "order" if i in order else "rule",
                                    "fixed": ok, "before": before,
                                    "after": cand if ok else None})
            if ok:
                shots[i - 1] = replace(shots[i - 1], prompt=cand)
                fixed_now += 1
                if verbose:
                    print(f"    #{i} -> {cand[:120]}", flush=True)
            elif verbose:
                print(f"    #{i} -> kept the original (no better rewrite)",
                      flush=True)
        info["fixed"] += fixed_now
        fixed_last = fixed_now
        if not fixed_now:
            break
    info["remaining"] = {str(i): rs for i, rs in prompt_flaws(shots).items()}
    if verbose:
        print(f"  [story] self-critique: fixed {info['fixed']} shot(s)"
              + (f"; still flagged by the rules: {sorted(info['remaining'])}"
                 if info["remaining"] else ""), flush=True)
    return shots, info


def _cache_matches(raw: dict, want: Dict, farms_supplied: bool) -> bool:
    """Was this cached proposal written for the configuration being asked for now?

    New entries carry `_cache_config` and must match exactly. LEGACY entries (written
    before it existed) cannot be matched exactly, and discarding them would re-run
    10-minute LLM drafts for nothing, so they are accepted unless the evidence in
    them contradicts the request: a different farm set (the ideas record each
    farm's model), or a scene plan demanded when none was made.
    """
    have = raw.get("_cache_config")
    if isinstance(have, dict):
        # entries written before the self-critique existed did not run it
        have = {"self_critique": False, **have}
        if farms_supplied:                       # pair-fair mode: pool is the farms
            have = {k: v for k, v in have.items() if k != "farms"}
            want = {k: v for k, v in want.items() if k != "farms"}
        return have == want
    if not farms_supplied:
        used = sorted(str(f.get("model")) for f in (raw.get("ideas") or [])
                      if isinstance(f, dict) and f.get("model"))
        if used and used != list(want["farms"]):
            return False
    if want["scene_plan"] and not raw.get("scene_plan"):
        return False
    return True


def write_shots_ensemble(concept: str = CONCEPT_DEFAULT,
                         farms: Optional[Sequence[str]] = None,
                         showrunner: str = SHOWRUNNER, n_ideas: int = 6,
                         temperature: float = 0.8, n_shots: Optional[int] = None,
                         story_retries: int = 2, verbose: bool = True,
                         fresh: bool = False,
                         keep: Optional[Sequence[str]] = None,
                         keep_only: bool = False,
                         farm_out: Optional[Sequence[Dict]] = None,
                         use_cache: bool = True,
                         save_cache: bool = True,
                         num_predict: Optional[int] = None,
                         repair_prompt: bool = True,
                         prior: Optional[Dict] = None,
                         extend: bool = False,
                         extend_attempts: int = 3,
                         prompt_pass: bool = False,
                         prompt_pass_model: Optional[str] = None,
                         scene_plan: bool = False,
                         coverage_retries: int = 0,
                         self_critique: bool = False,
                         critic_model: Optional[str] = None) -> Proposal:
    """Idea farms, then a showrunner that authors the film. See the note above.

    `farm_out` injects an already-collected set of farm results, skipping the
    farm calls. This is what makes a SHOWRUNNER COMPARISON fair: every model must
    be handed the same raw material, or the differences measured are the farms'
    non-determinism rather than the showrunner's quality. Callers that pass it
    also normally pass `use_cache=False, save_cache=False` so an experiment cannot
    overwrite the proposal the user is actually working from.

    `prior` is the FED-BACK WRITING loop. It is a previous draft that was
    rejected -- `{"shots": [Shot...], "problems": [...]}` -- and it is handed to
    the FIRST attempt as a repair brief, so a re-run after a refusal continues
    from the rejected draft instead of starting blind. Inside a single call the
    loop already does this between attempts; `prior` is how it survives ACROSS
    calls (the editor sends the refusal back from the next run).

    `extend` closes the other half of the loop: when the story is clean but the
    list is SHORTER than `n_shots`, `extend_shot_list` adds coverage inside the
    approved scenes until the requested count is reached. It is opt-in (off by
    default) because the comparison harness measures the writer's own count, and
    a comparison that silently extended short replies would flatter every model.

    `prompt_pass` is the TEXT half, also opt-in: after the story and count are
    settled, `rewrite_prompts` rewrites each shot's prompt as ONE concrete image
    (the measured failure of the first rendered film was twelve two-image prompts;
    it ended on a car). It cannot change the story, the beat order or the timing,
    and a per-shot best-of keeps the original when the rewrite is not better.
    `prompt:` problems are what it targets; see `prompt_problems`.

    `scene_plan` is the COVERAGE half, opt-in for the same reason: a cheap plan
    pass picks 2-3 places and a recurring figure before any shots, and the shot
    list is written to cover that plan instead of inventing one place per beat.
    `coverage_retries` gives that many re-asks when a clean story still has no
    scene that holds (every cut a new place) -- a montage is legitimate, so this
    is a demand of the writing path, not a gate. See `coverage_problems`.
    """
    farm_models = list(farms) if farms else list(FARM_MODELS)
    if farm_out is not None and not farms:
        # the supplied pool IS the farm set; labelling the proposal with
        # FARM_MODELS here would name farms that were never called
        farm_models = ([str(f.get("model", f"farm{i + 1}"))
                        for i, f in enumerate(farm_out)] or farm_models)
    cache = ROOT / "output" / "film" / "proposals"
    slug = re.sub(r"\W+", "-", concept.lower()).strip("-")[:60]
    cache_f = cache / (f"{slug}__ensemble-{showrunner.replace(':', '-')}"
                       f"-{len(farm_models)}farms__n{n_shots or 'auto'}.json")
    if keep:
        # a different kept set is a DIFFERENT proposal, so it must not hit the same
        # cache entry -- otherwise "keep these" would silently return the film the
        # user just rejected
        tag = hashlib.sha1(("|".join(sorted(keep))).encode("utf-8")).hexdigest()[:8]
        cache_f = cache_f.with_name(cache_f.stem + f"__keep{tag}.json")

    # WHAT THIS RUN WAS ASKED TO DO, recorded in the cache file. The file name only
    # carries the showrunner, the NUMBER of farms and n -- so two different farm
    # sets, or a run with the scene plan / prompt pass / extension switched on vs
    # off, shared one file and the second silently replayed the first.
    cache_config = {"farms": sorted(farm_models), "extend": bool(extend),
                    "prompt_pass": bool(prompt_pass), "scene_plan": bool(scene_plan),
                    "self_critique": bool(self_critique)}

    if use_cache and cache_f.exists() and not fresh:
        try:
            raw = json.loads(cache_f.read_text(encoding="utf-8"))
            shots, problems = validate(raw)
            if shots and not _cache_matches(raw, cache_config, farm_out is not None):
                if verbose:
                    print(f"  [story] cache entry {cache_f.name} was written for a "
                          f"different configuration; writing fresh", flush=True)
                shots = []
            if shots:
                if verbose:
                    print(f"  [story] cached ENSEMBLE proposal ({cache_f.name}): "
                          f"{len(shots)} shots, {len(problems)} rule(s) broken")
                    prov = raw.get("provenance", {})
                    if prov:
                        for k, v in prov.items():
                            if k.startswith("_"):
                                continue
                            print(f"           farm {k:<16} {v['used']}/{v['ideas']} "
                                  f"ideas landed (mean best overlap "
                                  f"{v['mean_best']:.2f})")
                loop = raw.get("loop") or {}
                ext = loop.get("extension") or {}
                pp = loop.get("prompt_pass") or {}
                anch = loop.get("anchors") or raw.get("anchors") or {}
                # the loop's evidence survives the cache: a hit used to report
                # 0 attempts / no plan / not anchored in draft.py's one-line summary
                return Proposal(concept=concept,
                                model=f"ensemble({showrunner}+{len(farm_models)})",
                                logline=str(raw.get("logline", ""))[:200],
                                shots=shots, raw=raw, seconds=0.0,
                                decode_tok_s=0.0, problems=problems, cached=True,
                                attempts=int(loop.get("attempts") or 0),
                                requested=loop.get("requested"),
                                extended=bool(ext.get("calls")),
                                extend_calls=len(ext.get("calls") or []),
                                prompts_rewritten=int(pp.get("rewritten") or 0),
                                scene_plan=raw.get("scene_plan"),
                                coverage_repairs=int(loop.get("coverage_repairs")
                                                     or 0),
                                anchored=bool(anch.get("bound")),
                                prior_fed_back=bool(loop.get("prior_fed_back")))
        except Exception:
            pass

    if farm_out is not None:
        # PAIR-FAIR MODE. The ideas are supplied by the caller, so all showrunner
        # candidates answer the SAME brief from the SAME pool. Nothing is re-run.
        farm_out = [dict(f) for f in farm_out]
        total_ideas = sum(len(f.get("ideas") or []) for f in farm_out)
        print(f"=== IDEAS (supplied: {total_ideas} ideas from "
              f"{len(farm_out)} farm(s), not re-run) ===", flush=True)
    else:
        print(f"=== IDEAS ({len(farm_models)} farms) ===", flush=True)
        farm_out = []
        for i, m in enumerate(farm_models):
            angle = FARM_ANGLES[i % len(FARM_ANGLES)]
            # PROGRESS PER FARM, FLUSHED. The farms are serial and the next line used to be
            # the first output after ALL of them -- so a 3-minute run printed nothing and a
            # watching UI could not tell it from a hang. `job_runner._story_progress` already
            # parses exactly this shape.
            print(f"  [farm {i + 1}/{len(farm_models)}] {m} ...", flush=True)
            f = farm_ideas(concept, m, angle, n_ideas=n_ideas, verbose=verbose)
            farm_out.append(f)
            print(f"  [farm {i + 1}/{len(farm_models)}] {m} -> {len(f['ideas'])} ideas",
                  flush=True)
        total_ideas = sum(len(f["ideas"]) for f in farm_out)
        print(f"  {total_ideas} ideas from {len(farm_models)} farms", flush=True)

    # Showrunner: every idea, labelled by source, plus the hard renderer limits.
    #
    # INTERLEAVED, not grouped. Grouped by farm, the prompt hands the showrunner
    # four tidy little shot lists, and it takes one of them whole -- measured, that
    # is exactly what happened (6 of 6 shots from qwen, provenance 6/0/0/0).
    # Round-robin keeps one scout's list from sitting together as a coherent
    # whole, so choosing between them means actually choosing.
    lines: List[str] = []
    depth = max((len(f["ideas"]) for f in farm_out), default=0)
    for k in range(depth):
        for f in farm_out:
            if k < len(f["ideas"]):
                lines.append(f"- [{f['model']}] {f['ideas'][k]}")
    lines = _seed_lines(lines, keep, keep_only)
    idea_text = "\n".join(lines) if lines else "(no ideas came back)"
    if keep and lines:
        idea_text = ("(the DIRECTOR kept these ideas by hand and wants the film "
                     "built on them"
                     + (", and has DISCARDED the rest" if keep_only else "")
                     + ")\n" + idea_text)

    # PLAN THE SCENES BEFORE THE SHOTS. Cheap (~1-2 min) and it is what makes the
    # shot list COVER 2-3 places instead of inventing one per beat: the writer has
    # always been told to use 2-3 scenes, nothing made it, and the first rendered
    # film came back as 12 places in 13 shots (so the director montaged it).
    planned: Optional[Dict] = None
    plan_info: Optional[Dict] = None
    if scene_plan and n_shots:
        planned, plan_info = write_scene_plan(
            concept, n_shots, idea_text, showrunner=showrunner,
            temperature=temperature, verbose=verbose)

    import ollama
    print(f"\n=== SHOWRUNNER ({showrunner}) ===")
    t0 = time.time()
    # RETRY ON A BROKEN STORY, WITH THE FAILURE NAMED. A story break is not a
    # detail: a film whose beats do not progress is a mood board however good the
    # images are, and text costs ~4 min where a render costs ~60. The retry is not
    # blind -- `_repair_note` hands the next attempt the validator's exact
    # complaints and the beats it produced, so it can repair the order instead of
    # re-rolling the same distribution. A CLEAN story with no coverage gets one
    # targeted re-ask too (the planned scenes are attached): that is the difference
    # between a film of scenes and a montage of postcards.
    raw = None
    problems: List[str] = []
    best = None
    replies: List[Dict] = []       # every attempt's raw text and failure, for audit
    attempts_made = 0
    coverage_repairs = 0
    coverage_seen = 0
    repair_ctx = ""
    # THE FEED-BACK SEED: a draft refused by a PREVIOUS run is attached to the
    # first attempt. Without this, pressing "write new ideas" after a refusal ran
    # the identical brief again and the loop's knowledge died with the process --
    # which is exactly how the same backwards beat survived three attempts.
    prior_shots = list((prior or {}).get("shots") or [])
    prior_fed_back = bool(prior_shots)
    if prior_shots:
        prior_problems = list((prior or {}).get("problems") or [])
        repair_ctx = _repair_note(
            prior_problems, prior_shots, n_shots=n_shots,
            returned=len(prior_shots),
            prefix=("A PREVIOUS DRAFT OF THIS FILM WAS REFUSED. It is attached "
                    "so you REPAIR it rather than start again from nothing:"))
        if repair_ctx:
            print(f"  [story] feeding back the refused draft "
                  f"({len(prior_shots)} shot(s), shape {story_shape(prior_shots)}, "
                  f"{len(prior_problems)} named problem(s))", flush=True)
    max_attempts = 1 + max(0, int(story_retries)) + max(0, int(coverage_retries))
    attempt = 0
    story_used = 0
    coverage_used = 0
    while attempt < max_attempts:
        attempt += 1
        budget = int(num_predict or predict_budget(n_shots))
        print(f"  [story] showrunner attempt {attempt}/{max_attempts}: "
              f"generating (<= {budget} tokens, ctx {OLLAMA_NUM_CTX}, ~3 tok/s on "
              f"CPU -- this is the silent stretch)", flush=True)
        try:
            content, ec = chat_streamed(
                f"showrunner attempt {attempt}/{max_attempts}", budget,
                model=showrunner,
                messages=[{"role": "system",
                           "content": SYSTEM + SHOWRUNNER_ADDENDUM},
                          {"role": "user", "content":
                           _showrunner_user(concept, n_shots, idea_text, repair_ctx,
                                            planned)}],
                format=SHOT_SCHEMA,
                options={"temperature": temperature,
                         "num_ctx": OLLAMA_NUM_CTX,
                         "num_predict": budget},
            )
        except Exception as exc:
            attempts_made = attempt
            replies.append({"attempt": attempt, "text": "",
                            "failure": f"{type(exc).__name__}: {str(exc)[:200]}"})
            print(f"  [story] showrunner attempt {attempt} failed: "
                  f"{type(exc).__name__}: {str(exc)[:110]}")
            continue
        attempts_made = attempt
        data, mode = parse_reply(content, salvage=True)
        if mode == "salvaged":
            print(f"  [story] showrunner attempt {attempt}: reply was cut off "
                  f"({json_failure(content)}); kept its {len(data['shots'])} complete "
                  f"shot(s) instead of discarding the attempt", flush=True)
        if data is None:
            why = json_failure(content)
            replies.append({"attempt": attempt, "text": content, "failure": why,
                            "chars": len(content or ""), "eval_count": ec})
            print(f"  [story] showrunner attempt {attempt}: reply did not parse "
                  f"-- {why} ({len(content or '')} chars, {ec} tok)")
            continue
        raw = data
        shots, problems = validate(raw)
        if not shots:
            replies.append({"attempt": attempt, "text": content,
                            "failure": "valid JSON but no usable shots",
                            "chars": len(content or ""), "eval_count": ec})
            print(f"  [story] showrunner attempt {attempt}: proposal did not validate")
            continue
        sb = sum(1 for pr in problems if pr.startswith("story:"))
        sw = sum(1 for pr in problems if pr.startswith("story-link:"))
        st = sum(1 for pr in problems if pr.startswith("structure:"))
        cov = sum(1 for pr in problems if pr.startswith("coverage:"))
        coverage_seen = max(coverage_seen, cov)
        score = (sb, cov, sw, st, len(problems))
        replies.append({"attempt": attempt, "text": content, "mode": mode,
                        "chars": len(content or ""), "eval_count": ec,
                        "shots": len(shots), "problems": len(problems),
                        "story_breaks": sb, "structure_breaks": st,
                        "coverage_breaks": cov})
        print(f"  [story] showrunner attempt {attempt}: {len(shots)} shots, "
              f"{sb} story break(s), {cov} coverage problem(s), "
              f"{sw} story warning(s), {st} structure problem(s), "
              f"{len(problems)} problem(s) total, parse={mode}")
        if best is None or score < best[0]:
            best = (score, raw, shots, problems, content, mode, ec)
        if sb == 0:
            if cov == 0:
                print(f"  [story] story and coverage complete -- stopping the "
                      f"re-roll (advisories kept: {st} structure, {sw} link)",
                      flush=True)
                break
            if repair_prompt and coverage_used < max(0, int(coverage_retries)):
                coverage_used += 1
                coverage_repairs += 1
                repair_ctx = _repair_note(
                    problems, shots, n_shots=n_shots, returned=len(shots),
                    prefix=("THE PREVIOUS DRAFT WAS A MONTAGE OF PLACES. It needs "
                            "2-3 SCENES, each one place and one figure held for "
                            "2-4 shots; keep the story, the beats and the count."))
                print(f"  [story] story holds but {cov} coverage problem(s) -- "
                      f"re-asking for 2-3 held scenes (coverage repair "
                      f"{coverage_used}/{coverage_retries})", flush=True)
                continue
            print(f"  [story] story holds; {cov} coverage problem(s) remain and "
                  f"the coverage budget is spent -- keeping this list", flush=True)
            break
        if repair_prompt and story_used < max(0, int(story_retries)):
            # TARGETED re-ask, not a blind re-roll: name the exact failures and
            # hand back the previous beats/scenes so the model can fix the
            # arrangement without throwing away imagery. `n_shots` is passed so a
            # clean-but-short list is told the COUNT as well -- it has no story
            # complaint to carry, and without the count it came back short again.
            story_used += 1
            repair_ctx = _repair_note(problems, shots, n_shots=n_shots,
                                      returned=len(shots))
            print(f"  [story] attempt {attempt} failed the story gate ({sb} "
                  f"break(s)) -- re-asking with them named and the previous "
                  f"arrangement attached", flush=True)
            continue
        repair_ctx = ""
        break
    if best is None:
        detail = "; ".join(f"attempt {r['attempt']}: {r.get('failure') or '?'}"
                           for r in replies) or "no reply"
        raise ShowrunnerFailure(
            f"the showrunner produced nothing usable ({detail})", replies)
    _score, raw, shots, problems, reply_text, parse_mode, eval_count = best

    # ---- the count half of the loop ---------------------------------------- #
    # A CLEAN STORY IS NOT ALLOWED TO BE SILENTLY SHORT. The gate only refuses
    # broken stories; a 6-shot film with 13 requested passes the gate and then
    # cannot reach the target length, because length is shot count. Re-rolling
    # would throw the clean story away, so the shortfall is closed with coverage
    # inside the scenes that already exist (see EXTEND_SCHEMA) -- which cannot
    # reorder the beats it has passed on.
    extend_info: Optional[Dict] = None
    story_breaks = [p for p in problems if p.startswith("story:")]
    if (extend and n_shots and len(shots) < int(n_shots) and not story_breaks):
        shots, extend_info = extend_shot_list(
            concept, shots, int(n_shots), showrunner=showrunner,
            temperature=temperature, attempts=extend_attempts, verbose=verbose)
        if extend_info.get("calls"):
            raw["shots"] = [shot_dict(s) for s in shots]
            raw["extension"] = extend_info
            shots, problems = validate(raw)
            if not shots:                       # defensive: never lose the film
                shots, problems = best[2], best[3]

    # ---- the text half: the same film, described as images ----------------- #
    # The prompt pass runs LAST, on the approved list, so it also cleans the
    # extension's inserts. It cannot change the story (it returns one prompt per
    # shot and the best-of keeps the original when the rewrite is not better).
    prompt_info: Optional[Dict] = None
    if prompt_pass and not any(p.startswith("story:") for p in problems):
        new_shots, prompt_info = rewrite_prompts(
            shots, showrunner=(prompt_pass_model or PROMPT_PASS_MODEL),
            temperature=min(float(temperature), 0.5), verbose=verbose)
        if prompt_info.get("rewritten"):
            shots = new_shots
            raw["shots"] = [shot_dict(s) for s in shots]
            raw["prompt_pass"] = prompt_info
            shots, problems = validate(raw)
            if not shots:                       # defensive: never lose the film
                shots, problems = best[2], best[3]

    # ---- the film reads its own writing ------------------------------------ #
    # After the prompt pass (so it judges the final image text) and BEFORE the
    # anchors (which are appended text, not the writer's); see critique_and_repair.
    critique_info: Optional[Dict] = None
    if self_critique and shots and not any(p.startswith("story:") for p in problems):
        new_shots, critique_info = critique_and_repair(
            concept, shots, model=(critic_model or prompt_pass_model
                                   or PROMPT_PASS_MODEL), verbose=verbose)
        if critique_info.get("fixed"):
            raw["shots"] = [shot_dict(s) for s in new_shots]
            raw["critique"] = critique_info
            fixed_shots, fixed_problems = validate(raw)
            if fixed_shots:
                shots, problems = fixed_shots, fixed_problems

    # ---- the coverage mechanism: bind every prompt to its planned scene ----- #
    # Last, because the prompt pass must not rewrite the anchors away. The anchors
    # are the renderer's only place/figure consistency mechanism on this hardware
    # (`visual_org.effective_prompt`); the store draft path never applied them.
    anchor_info: Optional[Dict] = None
    pre_coverage: List[str] = []
    if planned and shots:
        pre_prompt_problems = [p for p in problems if p.startswith("prompt:")]
        pre_coverage = [p for p in problems if p.startswith("coverage:")]
        anchored = anchor_plan(shots, planned)
        bound = sum(1 for a, b in zip(anchored, shots) if a.prompt != b.prompt)
        if bound:
            shots = anchored
            raw["shots"] = [shot_dict(s) for s in shots]
            fresh_shots, fresh_problems = validate(raw)
            if fresh_shots:
                # prompt shape was measured on the WRITER's text; the append is the
                # pipeline's own anchor, so keep that measurement and take every
                # other verdict (story/coverage/structure) from the final prompts
                problems = ([p for p in fresh_problems
                             if not p.startswith("prompt:")]
                            + pre_prompt_problems)
                shots = fresh_shots
            anchor_info = {"bound": bound, "applied": True}
            sizes: Dict[str, int] = {}
            for s in shots:
                sizes[s.scene] = sizes.get(s.scene, 0) + 1
            anchor_info["scenes"] = sizes
            raw["anchors"] = anchor_info
            if verbose:
                print(f"  anchors: bound {bound}/{len(shots)} prompt(s) to their "
                      f"scene's place/figure", flush=True)
    if critique_info:
        # what the self-critique could NOT fix stays visible (advisory, like prompt:)
        problems = list(problems) + [
            f"critic: shot {i} {'; '.join(rs)[:200]}"
            for i, rs in sorted((int(k), v) for k, v in
                                (critique_info.get("remaining") or {}).items())]
    dt = time.time() - t0

    prov = _provenance([s.prompt for s in shots], farm_out)
    raw["ideas"] = farm_out
    raw["provenance"] = prov
    raw["scene_plan"] = planned
    # THE LOOP'S EVIDENCE, for the cache and the editor: what was asked for, what
    # came back, and how the shortfall (if any) was closed.
    raw["loop"] = {
        "requested": int(n_shots) if n_shots else None,
        "returned": len(shots),
        "attempts": attempts_made,
        "shape": story_shape(shots),
        "prior_fed_back": prior_fed_back,
        "extension": extend_info,
        "prompt_pass": prompt_info,
        "critique": critique_info,
        "scene_plan": planned,
        "coverage_repairs": coverage_repairs,
        "coverage_seen": coverage_seen,
        "coverage_problems": pre_coverage or [p for p in problems
                                              if p.startswith("coverage:")],
        "anchors": anchor_info,
    }
    raw["_cache_config"] = cache_config
    if save_cache:
        try:
            cache.mkdir(parents=True, exist_ok=True)
            cache_f.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        except Exception:
            pass

    if verbose:
        print(f"  {len(shots)} shots in {dt:.0f}s, {len(problems)} rule(s) broken")
        for pr in problems:
            print(f"           - {pr}")
        print(f"  story shape: {story_shape(shots)}   "
              f"(E establish, d develop, T turn, R resolve)")
        if n_shots:
            calls = len((extend_info or {}).get("calls") or [])
            print(f"  shot count: {len(shots)}/{int(n_shots)} requested"
                  + (f" after {calls} extension call(s)"
                     if extend_info and calls else ""))
        if prompt_info and prompt_info.get("rewritten") is not None:
            print(f"  prompt pass: {prompt_info.get('rewritten', 0)} prompt(s) "
                  f"rewritten as single images, "
                  f"{len(prompt_info.get('flagged') or [])} still flagged")
        cov_probs = [p for p in problems if p.startswith("coverage:")]
        if planned:
            print(f"  scene plan: {len(planned['scenes'])} scene(s) · coverage "
                  f"repairs {coverage_repairs} · "
                  f"{len(cov_probs)} coverage problem(s) remaining")
        L = prov["_final_prompts_traced"]
        print(f"  provenance: {L}/{prov['_final_prompts']} final prompts trace back "
              f"to a farm idea")
        for k, v in prov.items():
            if k.startswith("_"):
                continue
            print(f"    farm {k:<16} {v['used']}/{v['ideas']} ideas landed "
                  f"(mean best overlap {v['mean_best']:.2f})")
    return Proposal(concept=concept,
                    model=f"ensemble({showrunner}+{len(farm_models)})",
                    logline=str(raw.get("logline", ""))[:200],
                    shots=shots, raw=raw, seconds=round(dt, 1),
                    decode_tok_s=0.0, problems=problems,
                    reply_text=reply_text, parse_mode=parse_mode,
                    attempts=attempts_made, eval_count=eval_count,
                    requested=int(n_shots) if n_shots else None,
                    extended=bool(extend_info and extend_info.get("calls")),
                    extend_calls=len((extend_info or {}).get("calls") or []),
                    prompts_rewritten=int((prompt_info or {}).get("rewritten")
                                         or 0),
                    scene_plan=planned,
                    coverage_repairs=coverage_repairs,
                    anchored=bool(anchor_info and anchor_info.get("bound")),
                    prior_fed_back=prior_fed_back)


def scorecard(p: Proposal, planned: List[Shot], changes: List[str],
              verbose: bool = True) -> Dict:
    """How much work did the LLM leave for the rest of the system?

    This is the number to tune against. A model that obeys the rules costs fewer
    repairs, and repairs are where a shot stops being what the writer intended.
    """
    r1 = sum(1 for c in changes if "R1" in c)
    r4 = sum(1 for c in changes if "R4" in c)
    r6 = sum(1 for c in changes if "R6" in c)
    # R4 (the rhythm curve) is deliberately the planner's job -- the writer is
    # told not to agonise over 'seconds' because it will be overwritten anyway.
    # Counting it as a writer error made every model look equally bad, which is
    # the metric being wrong rather than the models. Structural repairs are the
    # ones that mean the writer made a mistake.
    structural = r1 + r6
    # kept separate from the structural count: a leak is an instruction break but
    # not a planning mistake, and it is the one that pollutes the render prompt
    leaks = sum(1 for pr in p.problems if "carries field values" in pr)
    out = {
        "model": p.model,
        "seconds": p.seconds,
        "decode_tok_s": p.decode_tok_s,
        "shots": len(p.shots),
        "rule_breaks": len(p.problems),
        "prompt_leaks": leaks,
        "story_breaks": sum(1 for pr in p.problems if pr.startswith("story:")),
        "story_warnings": sum(1 for pr in p.problems
                              if pr.startswith("story-link:")),
        "structure_breaks": sum(1 for pr in p.problems
                                if pr.startswith("structure:")),
        "coverage_breaks": sum(1 for pr in p.problems
                               if pr.startswith("coverage:")),
        "prompt_breaks": sum(1 for pr in p.problems
                             if pr.startswith("prompt:")),
        "rhythm_breaks": sum(1 for pr in p.problems
                             if pr.startswith("rhythm:")),
        "story_shape": story_shape(p.shots),
        "planner_changes": len(changes),
        "structural_repairs": structural,
        "timing_repairs": r4,
        "r1_jump_cuts": r1,
        "r4_rhythm_rewrites": r4,
        "r6_reversals": r6,
        "preserved": round(1.0 - structural / max(1, len(p.shots)), 2),
        "scenes": len({s.scene for s in planned}),
        "used_fallback": p.used_fallback,
    }
    if verbose:
        print(f"  logged {out['shots']} shots in {out['seconds']}s "
              f"({out['decode_tok_s']} tok/s)")
        print(f"  broke {out['rule_breaks']} rule(s) the writer was told about")
        print(f"  structural repairs: {structural} "
              f"({r1} jump cut(s), {r6} reversal(s))")
        print(f"  timing rewrites: {r4} -- the planner's job, not a writer error")
        print(f"  -> {int(out['preserved']*100)}% of the writer's structure "
              f"survived to the render")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default=CONCEPT_DEFAULT)
    ap.add_argument("--model", default="granite4.1:8b")
    ap.add_argument("--writers", default="ensemble", choices=["single", "ensemble"],
                    help="single = one model writes everything (the previous "
                         "behaviour); ensemble = idea farms plus one showrunner")
    ap.add_argument("--farms", default=",".join(FARM_MODELS))
    ap.add_argument("--showrunner", default=SHOWRUNNER)
    ap.add_argument("--n-shots", type=int, default=None,
                    help="ask for this many shots. SHOT COUNT IS THE BUDGET: at "
                         "9-14 min per shot on this CPU the count, not the model, "
                         "is what decides whether the film gets made")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--scene-plan", action="store_true",
                    help="plan 2-3 held scenes (one place + one figure each) "
                         "before the shots. This is the coverage demand: without "
                         "it the writer tends to invent one image per beat")
    ap.add_argument("--coverage-retries", type=int, default=0,
                    help="re-asks allowed when the story holds but no scene does")
    ap.add_argument("--keep-only", action="store_true",
                    help="show the showrunner ONLY the kept ideas, discarding the "
                         "rest of the pool")
    ap.add_argument("--keep-file", default=None,
                    help="JSON list of scout ideas the director KEPT. The "
                         "showrunner is then shown only those, labelled as chosen "
                         "by hand, so a re-roll builds on them")
    ap.add_argument("--no-direct", action="store_true",
                    help="skip the motion director, to isolate the text layer")
    a = ap.parse_args()

    print(f'concept: "{a.concept}"')
    if a.writers == "ensemble":
        p = write_shots_ensemble(a.concept, farms=[m.strip() for m in
                                                  a.farms.split(",") if m.strip()],
                                 showrunner=a.showrunner, fresh=a.fresh,
        n_shots=a.n_shots,
        scene_plan=a.scene_plan, coverage_retries=a.coverage_retries,
        keep=(json.loads(Path(a.keep_file).read_text(encoding="utf-8"))
              if a.keep_file else None),
        keep_only=a.keep_only)
    else:
        p = write_shots(a.concept, model=a.model, fresh=a.fresh)
    print("\n=== LLM PROPOSAL ===")
    print(f"  logline: {p.logline}")
    print(describe(p.shots))

    shots = p.shots
    if not a.no_direct:
        from director import FitScorer, choose
        shots, rep = choose(shots, scorer=FitScorer(), verbose=False)
        print("\n=== after the motion director ===")
        print(describe(shots))

    planned, report = plan(shots)
    print("\n=== after the continuity planner ===")
    print(describe(planned))
    print(report.show())

    print("\n=== scorecard ===")
    sc = scorecard(p, planned, report.changes)
    print("\n" + json.dumps(sc, indent=2))

def compare_showrunners(concept: str = CONCEPT_DEFAULT,
                        models: Optional[Sequence[str]] = None,
                        farms: Optional[Sequence[str]] = None,
                        n_shots: Optional[int] = None, runs: int = 1,
                        verbose: bool = True) -> Dict:
    """Compare showrunner candidates on the SAME concept and the SAME ideas.

    Thin entry point kept for discoverability -- the implementation is
    `compare_showrunners.py`, because a fair comparison needs to collect the idea
    pool once, repeat each model, and keep the raw replies as evidence. That is
    more than belongs in the pipeline module.

    This function used to live here and read scorecard keys that do not exist
    (`writer_story_breaks`, `total_repairs`, `planner_r1_scale`), so every
    comparison printed zeros. The metric names are now taken from
    `scorecard()`, and the harness is the single source.
    """
    from compare_showrunners import compare
    return compare(concept=concept, models=models, farms=farms,
                   n_shots=n_shots, runs=runs, verbose=verbose)
