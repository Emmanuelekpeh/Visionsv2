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
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
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

# The story's progression, as a first-class field. Before this existed, nothing
# in the pipeline asked the shots to DEVELOP: the brief demanded variety (do not
# repeat) and never development (do change), so the writer produced mood boards --
# six restatements of one idea with the location moving. Measured on a real
# ensemble output: "shadows on a sidewalk, no one in sight" -> "silhouettes merge"
# -> "two shapes on a bench" -> "two figures step toward each other" -> "a lone
# figure in a crowd" -> "shoes alone in a puddle". One idea, six times, and no
# check anywhere would have noticed.
BEATS = ["establish", "develop", "turn", "resolve"]
BEAT_MARKS = {"establish": "E", "develop": "d", "turn": "T", "resolve": "R"}

# Below this stemmed overlap the resolve is treated as unrelated to the turn.
# Low on purpose: a legitimate resolve can be worded completely differently, so
# this catches disjointness, not difference.
TURN_RESOLVE_LINK = 0.08

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
    "things, materials, light, weather, framing. At least 8 words. Never a title, "
    "a scene heading, or a sequence of events.\n"
    "\n"
    "VARIETY -- a film of six near-identical shots is a failure even if each shot "
    "is good:\n"    "- At least one shot must NOT contain the recurring figure at all. An "
    "environment or a small detail gives the film air and makes the figure's "
    "return land.\n"
    "- Vary the light and the materials between shots. Do not write six shots of "
    "darkness; contrast is what makes the dark read as dark.\n"
    "- Give at least one shot a camera move (zoom-in, pan-left, tilt-up...), and "
    "leave the rest at 'none'.\n"
    "\n"
    "STORY -- the film must GO somewhere. Six equal moments is a mood board, not "
    "a film. This is the point of the whole shot list, and nothing downstream can "
    "fix it for you.\n"
    "- Every shot carries a 'beat': establish, develop, turn, or resolve. The "
    "beats must move FORWARD through that order and never backwards.\n"
    "- Shot 1 is 'establish': the world and the recurring figure as they are now.\n"
    "- At least one shot is 'turn': the moment something CHANGES -- the figure "
    "acts, a discovery lands, the situation inverts. Without a turn the film "
    "restates one idea instead of developing one.\n"
    "- The last shot is 'resolve': the world as it is AFTER the turn. It must not "
    "be a sixth restatement -- something about it should be different from shot 1, "
    "and you should be able to say what.\n"
    "- 'logline' states the CHANGE, not the premise. BAD: 'a woman walks through "
    "shadows'. GOOD: 'a woman follows a figure through the shadows and finds it is "
    "her own reflection'.\n"
    "\n"
    "ENERGY keeps its own shape, set by your 'energy' values (the director cannot "
    "create it, because a still shot always carries the same energy whatever it "
    "chooses):\n"
    "- 'energy' (0 to 0.5) must RISE, peak, then fall away. At least one shot near "
    "0.05, at least one at 0.45 or above, the rest spread. Not six equal values.\n"
    "- The highest energy is NOT on the first or the last shot; it belongs in the "
    "middle or just past it. The last shot settles rather than stopping dead.\n"
    "\n"
    "BAD:  'Scene 1: Darkness'\n"
    "BAD:  'Whispers fade'\n"
    "GOOD: 'a bare birch trunk in fog, wet bark, low grey light, thin mist, "
    "cinematic, filmic grain'\n"
    "\n"
    "Other rules:\n"
    "- 4 to 6 shots.\n"
    "- Never use the same framing scale on two consecutive shots: that is a jump "
    "cut and the planner will rewrite it.\n"
    "- 'seconds': do not agonise over it. The planner imposes a rhythm curve and "
    "will overwrite your values, so just keep them between 1.5 and 3.6.\n"
    "- 'energy' 0 to 0.5: how much movement the shot needs.\n"
    "- 'scene' groups shots by location or act. Use 2 or 3 scenes; a look change "
    "and a checkpoint change only land at a scene boundary.\n"
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
                },
                "required": ["prompt", "scale", "camera", "seconds", "scene",
                             "energy", "beat"],
            },
        },
    },
    "required": ["logline", "shots"],
}


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

    def __post_init__(self):
        if self.problems is None:
            self.problems = []


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

    if not (4 <= len(items) <= 6):
        problems.append(f"{len(items)} shots (want 4-6)")

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
        if energy > MAX_RENDERABLE_MOTION:
            problems.append(
                f"shot {i}: energy {energy:.2f} exceeds the renderer's measured "
                f"ceiling {MAX_RENDERABLE_MOTION} -- shot would be refused")
            energy = MAX_RENDERABLE_MOTION
        shots.append(Shot(prompt=prompt, scale=scale, camera=cam, seconds=secs,
                          subject=str(s.get("subject", "")).strip(),
                          scene=str(s.get("scene", "scene1")).strip() or "scene1",
                          energy=energy, beat=beat))

    # the rules the planner would otherwise have to repair
    for i in range(len(shots) - 1):
        if shots[i].scale == shots[i + 1].scale:
            problems.append(f"jump cut: shots {i+1},{i+2} both {shots[i].scale}")
    movers = sum(1 for s in shots if s.camera != "none")
    if movers > 2:
        problems.append(f"{movers} moving shots (want <=2)")
    # STORY PROGRESSION. Nothing else in the pipeline looks at this, and it is not
    # fixable downstream: the planner only does technical continuity, and the
    # director only picks camera moves. A film with no 'turn' cannot be rescued by
    # either of them.
    if shots:
        beats = [s.beat for s in shots]
        if beats[0] != "establish":
            problems.append(f"story: opens on {beats[0]!r}, not 'establish'")
        if beats.count("establish") > 1:
            problems.append(f"story: {beats.count('establish')} establish beats -- "
                            f"only the first shot establishes")
        if "turn" not in beats:
            problems.append("story: no 'turn' -- nothing changes, so the film "
                            "restates one idea instead of developing one")
        if beats[-1] != "resolve":
            problems.append(f"story: ends on {beats[-1]!r}, not 'resolve'")
        order = [BEATS.index(b) for b in beats]
        back = [f"{i+1}:{beats[i]}->{beats[i+1]}"
                for i in range(len(beats) - 1) if order[i + 1] < order[i]]
        if back:
            problems.append("story: beats go backwards (" + ", ".join(back) + ")")

        # THE JOINT THAT WAS WEAK. A turn nobody pays off is a scene change, not a
        # story. Measured on a real output: the turn was "a silhouetted figure
        # with a knowing look" and the resolve was "a dimly lit auction" -- beats
        # declared correctly, and nothing whatever linking them.
        #
        # A LEXICAL proxy cannot tell "linked but worded differently" from
        # "unrelated", so this is a WARNING for the eye (`story-link:`), not a hard
        # break: a legitimate resolve can share no words at all with its turn. It
        # only fires on near-total disjointness, which is what the auction was.
        if "turn" in beats and len(shots) > 1:
            ti, ri = beats.index("turn"), len(shots) - 1
            if ri > ti:
                ov = link_overlap(shots[ti].prompt, shots[ri].prompt)
                if ov < TURN_RESOLVE_LINK:
                    problems.append(
                        f"story-link: the resolve shares almost no imagery with the "
                        f"turn (overlap {ov:.2f} < {TURN_RESOLVE_LINK}) -- shot "
                        f"{ri+1} may read as a new scene rather than the turn's "
                        f"consequence. Check it by eye.")

        # REJECTED: a "recurring subject must be present at both ends" check. It
        # fired on 2 of 3 constructed cases -- a pronoun defeats it ("she" for
        # "woman"), and it flagged a legitimate ending on the empty room the film
        # opened in. A check that fires on good work is the check being wrong, so
        # the subject is left to the writer's brief and the by-eye read instead.
        # The link check above is the one that earned its place.

    if shots and len({s.scene for s in shots}) < 2:
        problems.append("only one scene: no boundary for a look or model change")
    return shots, problems


def write_shots(concept: str = CONCEPT_DEFAULT, model: str = "granite4.1:8b",
                temperature: float = 0.8, attempts: int = 2,
                verbose: bool = True, fresh: bool = False,
                n_shots: Optional[int] = None) -> Proposal:
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
                options={"temperature": temperature, "num_predict": 1200},
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

        try:
            raw = json.loads(resp["message"]["content"])
        except Exception:
            last_err = "response was not JSON"
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
                         repaired=(attempt > 1), problems=problems)
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
SHOWRUNNER = "granite4.1:8b"

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
    "- Keep what serves it and DISCARD the rest. Taking every idea gives a list of "
    "unrelated images, which is a failure even when each image is good.\n"
    "- Make the kept shots belong together: a consistent recurring figure, "
    "consistent light and materials across a scene, a progression from scene to "
    "scene rather than a jumble.\n"
    "- You may invent, merge or rewrite ideas. The scouts' text is raw material, "
    "not a contract.\n"
    "- Aim for variety WITHIN that coherence: across the shots, vary the framing "
    "scale, the light and the materials. Do not write six variations of one image.\n"
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


def json_from_text(text: str):
    """Pull a JSON object out of a model's reply, or raise.

    Local models do not all honour `format=` the same way. Measured: qwen3:1.7b
    returns clean JSON on a short prompt but can come back wrapped or with
    thinking text in front of it, and a smaller model returned a truncated
    `'{\\n '`. Parsing `resp["message"]["content"]` directly therefore fails for
    reasons that have nothing to do with the model being wrong -- a farm would
    drop out and the ensemble would quietly run on three farms instead of four.
    Strip code fences, then take the outermost braces.
    """
    if text is None:
        raise ValueError("empty response")
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        return json.loads(s[i:j + 1])
    raise ValueError(f"no JSON object in response: {s[:120]!r}")


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
            options={"temperature": temperature, "num_predict": timeout_predict},
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


def write_shots_ensemble(concept: str = CONCEPT_DEFAULT,
                         farms: Optional[Sequence[str]] = None,
                         showrunner: str = SHOWRUNNER, n_ideas: int = 6,
                         temperature: float = 0.8, n_shots: Optional[int] = None,
                         story_retries: int = 2, verbose: bool = True,
                         fresh: bool = False,
                         keep: Optional[Sequence[str]] = None,
                         keep_only: bool = False) -> Proposal:
    """Idea farms, then a showrunner that authors the film. See the note above."""
    farm_models = list(farms) if farms else list(FARM_MODELS)
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

    if cache_f.exists() and not fresh:
        try:
            raw = json.loads(cache_f.read_text(encoding="utf-8"))
            shots, problems = validate(raw)
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
                return Proposal(concept=concept,
                                model=f"ensemble({showrunner}+{len(farm_models)})",
                                logline=str(raw.get("logline", ""))[:200],
                                shots=shots, raw=raw, seconds=0.0,
                                decode_tok_s=0.0, problems=problems, cached=True)
        except Exception:
            pass

    print(f"=== IDEAS ({len(farm_models)} farms) ===", flush=True)
    farm_out: List[Dict] = []
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

    import ollama
    print(f"\n=== SHOWRUNNER ({showrunner}) ===")
    t0 = time.time()
    # RE-ROLL ON A BROKEN STORY. A story break is not a detail: a film whose beats
    # do not progress is a mood board however good the images are, and the whole
    # point of validating the text layer is that text costs ~4 min where a render
    # costs ~60. So the showrunner is asked again, up to `story_retries` times, and
    # the best attempt is kept (fewest story breaks, then fewest problems). An
    # attempt with a clean story stops the loop immediately.
    raw = None
    problems: List[str] = []
    best = None
    for attempt in range(1, story_retries + 2):
        try:
            resp = ollama.chat(
                model=showrunner,
                messages=[{"role": "system",
                           "content": SYSTEM + SHOWRUNNER_ADDENDUM},
                          {"role": "user", "content":
                           f"Concept: {concept}\n"
                           + (f"Write exactly {n_shots} shots.\n" if n_shots else "")
                           + "\nThe scouts' raw ideas:\n\n" + idea_text
                           + "\n\nWrite the film's shot list."}],
                format=SHOT_SCHEMA,
                options={"temperature": temperature, "num_predict": 1200},
            )
            raw = json_from_text(resp["message"]["content"])
        except Exception as exc:
            print(f"  [story] showrunner attempt {attempt} failed: "
                  f"{type(exc).__name__}: {str(exc)[:110]}")
            continue
        shots, problems = validate(raw)
        if not shots:
            print(f"  [story] showrunner attempt {attempt}: proposal did not validate")
            continue
        sb = sum(1 for pr in problems if pr.startswith("story:"))
        sw = sum(1 for pr in problems if pr.startswith("story-link:"))
        score = (sb, sw, len(problems))
        print(f"  [story] showrunner attempt {attempt}: {len(shots)} shots, "
              f"{sb} story break(s), {sw} story warning(s), "
              f"{len(problems)} problem(s) total")
        if best is None or score < best[0]:
            best = (score, raw, shots, problems)
        if sb == 0 and sw == 0:
            print("  [story] complete story -- stopping the re-roll")
            break
    if best is None:
        raise RuntimeError("the showrunner produced nothing usable")
    _score, raw, shots, problems = best
    dt = time.time() - t0

    prov = _provenance([s.prompt for s in shots], farm_out)
    raw["ideas"] = farm_out
    raw["provenance"] = prov
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
                    decode_tok_s=0.0, problems=problems)


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
