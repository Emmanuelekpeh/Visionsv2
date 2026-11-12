"""
director.py -- choose a motion for every shot, on fit, cohesiveness, continuity.

`edit_planner.py` enforces continuity on a sequence someone else proposed. This
is the layer above: it *decides* what each shot should do, then hands the result
to the planner to be repaired if it breaks a rule.

Three scores, each in [0, 1], combined by weight:

  fit           does this motion belong in THIS shot? Two parts. A hard gate --
                a subject LoRA on a shot with no person in it is how you get a
                figure walking through a leaf macro -- and a soft CLIP affinity
                between the shot's prompt and the motion's own description, so
                "a person walking toward the camera" matches a shot that has one.

  cohesiveness  does it sit with the shots around it? Measured motion energy is
                matched against the energy the shot's *scale* implies (a macro
                hold and an extreme wide should not have the same amount of
                movement), plus a budget on how much of the film is moving at all.

  continuity    does it preserve the film's spatial grammar? The 180-degree rule:
                two shots travelling along the same axis must not flip sign
                across a cut. A subject already established must not reverse
                direction. Adjacent subject shots must not repeat a framing.

The evidence loop is the point. `motion_vocab.load_evidence()` folds the
*measured* behaviour of each LoRA back into the vocabulary, so a motion that was
measured to be global drift is refused a subject-carrying shot automatically --
the directory of the film is set by measurement, not by belief. That is the
answer to "make the system decide what to use when": it decides from what the
LoRAs were observed to do on this machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import motion_vocab as mv
from edit_planner import MAX_MOVING_RUN, SCALES, Shot

# How much motion each framing scale implies. A macro hold wants almost nothing
# moving; an extreme wide can carry much more before it reads as chaos.
SCALE_ENERGY = {"macro": 0.20, "close": 0.38, "medium": 0.55,
                "wide": 0.75, "extreme-wide": 0.90}

# fallback energy for a motion with no measurement yet (0-1, relative)
PRIOR_ENERGY = {
    "none": 0.30, "zoom-in": 0.55, "zoom-out": 0.55, "pan-left": 0.65,
    "pan-right": 0.65, "tilt-up": 0.60, "tilt-down": 0.60,
    "rolling-clockwise": 0.80, "rolling-anticlockwise": 0.80,
    "camera-lateral_left": 0.70, "camera-lateral_right": 0.70,
    "camera-crane_up": 0.65, "camera-crane_down": 0.65,
    "walking-frontview": 0.45, "running-frontview": 0.80,
    "ballerina-spin": 0.55, "casualmove-frontview": 0.40,
    "walk-backview": 0.45, "drone-forward": 0.75,
}

WEIGHTS = {"fit": 0.45, "cohesiveness": 0.30, "continuity": 0.25}

# subject motion measured below this top-10% variance share is not trusted to
# carry a shot on its own: it is drifting, not moving a subject.
LOC10_TRUST = mv.LOC10_TRUST

# small preference for doing nothing, per the handoff: AnimateDiff already drifts,
# so forcing a move on every shot is exhausting. `none` wins ties.
NONE_BONUS = 0.03

# Content hints: CLIP text-to-text affinity is good at coarse gating ("is there a
# person here?") and weak at fine distinctions -- "she runs" and "she walks" both
# land near "toward the camera", so the running LoRA loses to the casual one. A
# short explicit lexicon is more reliable than a 0.02 cosine gap, and it is
# legible in the report, so a wrong choice can be argued with.
HINTS = {
    "running-frontview": ("run", "runs", "running", "sprint", "sprints",
                          "sprinting", "chase", "chased", "racing", "fleeing"),
    "walking-frontview": ("walk", "walks", "walking", "stride", "striding",
                          "steps", "marching", "hiking"),
    "casualmove-frontview": ("wander", "wanders", "amble", "ambles",
                             "stroll", "strolls", "casually"),
    "ballerina-spin": ("spin", "spins", "spinning", "twirl", "twirling",
                       "dance", "dances", "dancing", "pirouette"),
    "walk-backview": ("away", "receding", "departing", "leaving", "back"),
    "drone-forward": ("aerial", "drone", "flyover", "flying over"),
    "tilt-up": ("sky", "canopy", "upward", "towering", "above"),
    "tilt-down": ("ground", "floor", "downward", "below", "feet"),
    "zoom-in": ("push in", "closer", "detail", "intimate"),
    "zoom-out": ("pull back", "reveal", "withdrawn", "wider"),
    "pan-left": ("pan left", "leftward"),
    "pan-right": ("pan right", "rightward"),
}

HINT_BOOST = 1.25

# A hinted motion is given a floor rather than a nudge. CLIP's raw gap between
# "she runs" and "a person moving casually at an easy walking pace" can be 0.3 of
# cosine, which no sane multiplier closes; but a shot that says "runs" and gets
# the casual-walk LoRA is simply the wrong shot.
#
# This floor applies to CAMERA moves only. Subject motions get the boost but no
# floor, so they compete on merit instead of winning any prompt that happens to
# contain "walking". The user's call: the walking LoRA earns its place, it should
# not be the default. At 0.95 it was effectively a default for any prompt with the
# word in it.
HINT_FLOOR = 0.95
SUBJECT_HINT_FLOOR = 0.0


def hint_boost(prompt: str, name: str) -> Tuple[float, str]:
    """Explicit content match, e.g. the word 'runs' for running-frontview."""
    import re
    words = HINTS.get(name)
    if not words:
        return 1.0, ""
    for w in words:
        if re.search(rf"\b{re.escape(w)}\b", prompt, re.I):
            return HINT_BOOST, w
    return 1.0, ""


@dataclass
class Choice:
    index: int
    prompt: str
    scale: str
    chosen: str
    runner_up: str
    fit: float
    cohesiveness: float
    continuity: float
    total: float
    runner_total: float = 0.0
    notes: List[str] = field(default_factory=list)
    runner_notes: List[str] = field(default_factory=list)
    rejected: Dict[str, str] = field(default_factory=dict)


@dataclass
class DirectorReport:
    choices: List[Choice]
    weights: Dict[str, float]
    evidence: Dict[str, dict]

    def show(self) -> str:
        w = self.weights
        lines = [f"  director -- weights fit={w['fit']} coh={w['cohesiveness']} "
                 f"cont={w['continuity']}"]
        if self.evidence:
            lines.append("  measured evidence driving the choice:")
            for k, v in self.evidence.items():
                lines.append(f"    {k:<20} loc10={v['loc10']:.3f} "
                             f"energy={v['energy']:.2f}  (from {', '.join(v['clips'])})")
        else:
            lines.append("    (no measured evidence yet -- subject motions UNPROVEN)")
        for c in self.choices:
            flag = "  " if c.chosen == c.runner_up else "* "
            lines.append(f"  {c.index+1:>2}. {flag}{c.chosen:<22} "
                         f"fit={c.fit:.2f} coh={c.cohesiveness:.2f} "
                         f"cont={c.continuity:.2f} -> {c.total:.3f}"
                         f"   (next: {c.runner_up})")
            for n in c.notes:
                lines.append(f"        {n}")
            if c.runner_up != c.chosen:
                lines.append(f"        runner-up {c.runner_up} "
                             f"({c.runner_total:.3f}):")
                for n in c.runner_notes:
                    lines.append(f"          {n}")
            for name, why in c.rejected.items():
                lines.append(f"        rejected {name}: {why}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# fit: CLIP against the motion's own description
# --------------------------------------------------------------------------- #

NO_PERSON_TEXT = ("a photograph of a landscape, an object, a texture or an "
                  "empty scene with no people in it")
PERSON_TEXT = ("a photograph of a person, a human figure, someone standing or "
               "moving in the scene")


class FitScorer:
    """CLIP-based fit. Falls back to a keyword gate if CLIP is unavailable."""

    def __init__(self, clip_path: str = "models/clip-vit-base-patch32",
                 enabled: bool = True):
        self.ok = False
        self._cache: Dict[str, object] = {}
        if not enabled:
            return
        try:
            from motion_metrics import ClipScorer
            self.clip = ClipScorer(clip_path)
            self.ok = True
        except Exception as exc:
            print(f"  [director] CLIP unavailable ({type(exc).__name__}), "
                  f"falling back to keyword fit")

    # -- subject presence ---------------------------------------------------- #
    def person_detail(self, prompt: str) -> Tuple[float, str]:
        """(probability there is a person, which detector said so).

        CLIP zero-shot is good at implicit presence -- "a lone figure on an
        empty road" -- and can miss explicit statements: "she runs now, full
        body" scored 0.05, because the style words ("medium shot, cinematic,
        filmic grain") dominate the embedding. A short lexicon catches the
        explicit cases. Taking the max is a deliberate ensemble: either detector
        firing is enough to let a subject LoRA be considered, and the gate is
        still hard, so a wrong positive costs a rejected candidate rather than a
        person walking through a leaf macro.
        """
        kw = _keyword_person(prompt)
        if not self.ok:
            return (1.0 if kw else 0.0), "lexicon"
        p = self._clip_person(prompt)
        if kw and p < 0.9:
            return 0.9, f"lexicon (CLIP said {p:.2f})"
        return p, "CLIP"

    def person_probability(self, prompt: str) -> float:
        return self.person_detail(prompt)[0]

    def _clip_person(self, prompt: str) -> float:
        import torch
        key = "person:" + prompt
        if key not in self._cache:
            e = self.clip.embed_text(prompt)
            p = self.clip.embed_text(PERSON_TEXT)
            n = self.clip.embed_text(NO_PERSON_TEXT)
            logits = torch.stack([e @ p, e @ n]) * 100.0
            self._cache[key] = float(torch.softmax(logits, dim=0)[0])
        return self._cache[key]

    # -- affinity to each candidate ------------------------------------------ #
    def affinity(self, prompt: str, names: Sequence[str]) -> Dict[str, float]:
        """Softmax over the candidates' own descriptions -- a relative fit."""
        if not self.ok:
            return {n: 0.5 for n in names}
        import torch
        e = self.clip.embed_text(prompt)
        sims = []
        for n in names:
            key = f"desc:{n}"
            if key not in self._cache:
                self._cache[key] = self.clip.embed_text(mv.get(n).descriptor)
            sims.append(float(e @ self._cache[key]))
        t = torch.tensor(sims) * 100.0
        probs = torch.softmax(t, dim=0)
        return {n: float(p) for n, p in zip(names, probs)}

    def prompt_affinity(self, prompt: str, name: str) -> float:
        """Absolute cosine to one motion's description (no softmax)."""
        if not self.ok:
            return 0.5
        key = f"desc:{name}"
        if key not in self._cache:
            self._cache[key] = self.clip.embed_text(mv.get(name).descriptor)
        return float(self.clip.embed_text(prompt) @ self._cache[key])


def _keyword_person(prompt: str) -> bool:
    """Word-boundary match. Naive substring matching reads "the" as "he" and
    turns every landscape into a person."""
    import re
    words = (r"woman|women|man|men|person|people|figure|figures|girl|boy|child|"
             r"crowd|someone|somebody|dancer|ballerina|runner|walker|hiker|"
             r"she|her|he|him|his|they|portrait|face|hands")
    return re.search(rf"\b({words})\b", prompt, re.I) is not None


# --------------------------------------------------------------------------- #
# energy normalisation, from evidence where possible
# --------------------------------------------------------------------------- #

# `none` is a first-class directorial choice, so it starts from a healthy fit
# and a move has to beat it. The handoff's rule: AnimateDiff already drifts, so
# forcing a camera move on every shot is exhausting.
NONE_FIT = 0.62

# a camera move that matches its shot's description is a good choice, but it
# should not win on fit alone -- it still has to suit the shot's energy budget.
CAMERA_FIT_CEILING = 0.85


def energy_scale(candidates: Sequence[str]) -> Tuple[float, float]:
    """min/max of measured energies, for normalisation."""
    vals = [mv.get(n).energy for n in candidates if mv.get(n).measured
            and mv.get(n).energy is not None]
    if len(vals) < 2:
        return 0.0, 0.0
    return min(vals), max(vals)


def energy_scale_by_kind(candidates: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    """Measured energies only compare within a kind.

    Raw frame-to-frame change is NOT a measure of directorial energy. A `none`
    shot scores high on it precisely because AnimateDiff's global drift is large,
    while a subject LoRA locks the background and scores *low* -- so min-maxing
    across kinds would rank "nothing moves" as the most energetic option in the
    vocabulary. Within the subject motions the comparison is apples-to-apples
    (both lock the background), so it is used there and priors are used
    elsewhere. See the `mag` column in subject_motion_test.py's report.
    """
    out: Dict[str, Tuple[float, float]] = {}
    for kind in ("camera", "subject"):
        vals = [mv.get(n).energy for n in candidates
                if mv.get(n).kind == kind and mv.get(n).measured
                and mv.get(n).energy is not None]
        out[kind] = (min(vals), max(vals)) if len(vals) >= 2 else (0.0, 0.0)
    return out


def energy_of(name: str, kind_scales: Dict[str, Tuple[float, float]]) -> float:
    """Motion energy on a 0-1 scale. ALWAYS the prior -- never the measurement.

    Measured `mag` was tried as an energy proxy in two forms and both were wrong:

      - across all motions it ranks `none` (raw drift) as the most energetic;
      - within the subject motions, the ballerina clip's `mag` of 49.27 is not
        energy, it is incoherence (loc10 0.187, align 0.226, frames are colour
        chaos). Normalising against it squeezes walking (12.18) and running
        (9.39) down to 0.07 and 0.00, so the director rejected the walking LoRA
        for a shot whose prompt says "walking" and chose a camera zoom instead.

    `mag` measures how much the frame changed, not how much the film moves. It is
    reported as a diagnostic; it is not scored. `kind_scales` is kept in the
    signature so callers need not change.
    """
    return PRIOR_ENERGY.get(name, 0.5)


# --------------------------------------------------------------------------- #
# the director
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# the long view: choosing motions for the WHOLE film, not one shot at a time
# --------------------------------------------------------------------------- #
#
# The greedy path below decides each shot knowing only the shot before it, which
# is why a good early choice can block a better later one: it spends the film's
# moving-shot budget on whichever shots come first, and it can only avoid an axis
# reversal it has already committed to.
#
# This path optimises the same objective over the whole sequence at once, with a
# dynamic program whose state is (previous motion, run of movers, movers used so
# far). Three things fall out of that:
#
#   * the moving budget goes to the shots that most deserve it, wherever they are
#   * the 180-degree rule and the no-more-than-two-movers-in-a-row rule become
#     exact constraints rather than things the greedy pass hopes it noticed
#   * continuity across a cut is optimised, not just respected for the previous
#     choice
#
# The state is small ((motions x 3 x cap) ~ 180) and the shot list is 4-12 long,
# so the exact answer costs milliseconds. `mode="greedy"` keeps the shipped
# behaviour as the A/B control.
#
# NOTE ON DUPLICATION: the per-shot scoring is written twice on purpose -- once
# here, once frozen in `choose`'s greedy loop. The frozen copy IS the control, so
# it must keep behaving exactly as it shipped rather than being refactored along
# with the new path. If you retune a constant here, change it in both, or the A/B
# stops comparing like with like.

DIVERSITY_PENALTY = 0.06        # soft nudge against using one motion twice

# --------------------------------------------------------------------------- #
# ENERGY ARC (camera moves): NOT story progression. See story.BEATS for that.
# A film has to GO somewhere in its energy too, and the term's SHAPE was the bug
# --------------------------------------------------------------------------- #
#
# The question was fair: a "long view" should be about the film going somewhere,
# not only about where a bounded budget is spent. An energy-arc term was written
# and tested, and it is a no-op -- identical choices with it on and off on all
# four real shot lists tried (production SHOTS, TEST_SHOTS, FIGURE_SHOTS, and the
# ensemble's own output). Two compounding reasons, both measured:
#
#   1. `energy_of` returns PRIOR_ENERGY, a fixed constant per motion. A STILL shot
#      scores 0.30 whatever the director does, so at most two shots can leave the
#      flatline -- and the priors only span 0.30 (none) to 0.80 (running).
#   2. Fit and the hint floor dominate. A motion its own prompt asks for gets its
#      fit floored at 0.95 against a rival near 0.5; that is a ~0.20 swing in the
#      total, where the arc term's entire range is 0.12 x (0.80 - 0.30) = 0.06. To
#      make the arc bite it would need a weight around 0.3, at which point it is
#      choosing motions the shot did not ask for.
#
# THE SHAPE WAS WRONG, NOT THE WEIGHT. The first version scored `target * energy`,
# which is a REWARD for energy -- so it rewarded putting a move early, because a
# move carries more energy than a still shot whatever the target says. It fought
# itself across the film and looked inert. The term has to be a MATCH:
#
#     prog = 1 - |target_at_this_shot - energy_of(choice)|
#
# which penalises a move in the calm opening instead of paying for one, and pays
# for one at the peak. With ENERGY_ARC_WEIGHT below, on a shot list that hints a
# move early: the still shot gains +0.09 more than the hinted early move, so the
# early hint needs a real fit advantage to survive -- and at the peak the same
# move gains +0.088 over staying still. That is the gradient "save the move for
# the peak" needs, and it is what the old shape could not express.
#
# The arc is still partly the SHOT LIST's property -- the writer sets every shot's
# declared energy and `edit_planner` paces duration (R4) -- so energy_arc is ALSO
# asked for in the writer's brief (story.SYSTEM). Both halves are needed: the
# brief gives the film an energy shape, and this term spends the moves inside it.
#
# `energy_arc_target`, `arc_match` and `energy_profile` double as diagnostics.

ENERGY_ARC_PEAK = 0.65         # where in the film the energy should top out
ENERGY_ARC_FLOOR = 0.25        # opening calm
ENERGY_ARC_RESOLVE = 0.35      # and a quieter ending, not a stop

# Added only on the global path, so the frozen greedy control keeps scoring
# exactly as it shipped. Not folded into WEIGHTS.
ENERGY_ARC_WEIGHT = 0.25


def energy_arc_target(i: int, n: int) -> float:
    """The energy a film of n shots should be carrying at shot i.

    Rises from ENERGY_ARC_FLOOR to 1.0 at ENERGY_ARC_PEAK of the film, then
    falls to ENERGY_ARC_RESOLVE. Linear on purpose: a curve nobody can read off
    the report is a curve nobody can argue with.
    """
    if n <= 1:
        return 1.0
    peak = max(1, min(n - 1, int(round(ENERGY_ARC_PEAK * (n - 1)))))
    if i <= peak:
        f = i / peak
        return ENERGY_ARC_FLOOR + (1.0 - ENERGY_ARC_FLOOR) * f
    f = (i - peak) / max(1, n - 1 - peak)
    return 1.0 - (1.0 - ENERGY_ARC_RESOLVE) * f


def arc_match(energies: Sequence[float]) -> float:
    """How closely a film's motion energy follows the target arc, 0-1.

    1.0 = exactly the shape. Legible on purpose, so a report can say "this film
    peaks in the first third" without anyone having to trust a latent number.
    """
    n = len(energies)
    if n == 0:
        return 0.0
    return 1.0 - sum(abs(min(1.0, max(0.0, e)) - energy_arc_target(i, n))
                     for i, e in enumerate(energies)) / n


def energy_profile(energies: Sequence[float]) -> str:
    """A film's energy as a row of digits, so the SHAPE is visible at a glance."""
    return "".join(str(min(9, int(round(e * 9)))) for e in energies)


def _mover_cap(n: int) -> int:
    """How many shots in this film may carry a camera move.

    The writer's brief says "at most two shots may move" and R3 forbids more than
    two movers in a row, which is right for a 4-6 shot short. A longer film needs
    the budget scaled, or a 12-shot film could move only twice.
    """
    return 2 if n <= 6 else max(2, round(n / 3))


def mover_cap(n: int) -> int:
    """Public alias for the same budget. `camera_direction`'s hybrid overlay must
    count against the SAME cap the director optimises under, or the overlay can
    quietly exceed the brief (measured: 3 moves in a 6-shot film)."""
    return _mover_cap(n)


def _shot_options(shot, i, scorer, cands, kind_scales, person_threshold,
                  strict_evidence):
    """History-free scores for every candidate motion on one shot.

    Returns (opts, rejected, info). `opts[name]` carries fit, the raw
    cohesiveness term, the notes, and the Motion. Fatigue and continuity are NOT
    applied here: they depend on what the neighbouring shots ended up choosing,
    which the global path reads from its state and the greedy path reads from the
    shots it has already decided.
    """
    p_person, p_source = scorer.person_detail(shot.prompt)
    has_person = p_person >= person_threshold
    want = shot.energy if shot.energy is not None else \
        SCALE_ENERGY.get(shot.scale, 0.5)

    eligible: List[str] = []
    rejected: Dict[str, str] = {}
    for n in cands:
        m = mv.get(n)
        if not m.fits_scale(shot.scale):
            rejected[n] = (f"does not fit a {shot.scale} shot"
                           + (" (it draws a full body)" if m.full_body else ""))
            continue
        if m.subject_required:
            if not has_person:
                rejected[n] = (f"no person in this shot "
                               f"(p={p_person:.2f} < {person_threshold})")
                continue
            if strict_evidence and not m.measured:
                rejected[n] = "unmeasured, and strict evidence is on"
                continue
        eligible.append(n)
    if "none" not in eligible:
        eligible.append("none")

    # affinity is softmaxed within kind, not across the vocabulary
    by_kind: Dict[str, List[str]] = {"subject": [], "camera": [], "none": []}
    for n in eligible:
        by_kind.setdefault(mv.get(n).kind, []).append(n)
    aff_kind = {k: scorer.affinity(shot.prompt, v) for k, v in by_kind.items() if v}

    opts: Dict[str, dict] = {}
    for n in eligible:
        m = mv.get(n)
        notes: List[str] = []
        if n == "none":
            f = NONE_FIT
        elif m.kind == "subject":
            f = aff_kind["subject"].get(n, 0.5) * (0.55 + 0.45 * p_person)
            if not m.measured:
                f *= 0.75
                notes.append(f"{n}: unmeasured, fit discounted 25%")
            elif m.separable is False:
                f *= 0.8
                notes.append(f"{n}: measured non-separable on a textured scene "
                             f"(global_share {m.global_share:.2f} vs baseline "
                             f"{mv.BASELINE.get('global_share', 0):.2f}) -- scene "
                             f"dependent, penalised not refused")
        else:
            f = aff_kind["camera"].get(n, 0.5) * CAMERA_FIT_CEILING
        boost, hit = hint_boost(shot.prompt, n)
        if boost > 1.0:
            floor = SUBJECT_HINT_FLOOR if m.kind == "subject" else HINT_FLOOR
            if floor > 0.0:
                f = min(1.0, max(f * boost, floor))
                notes.append(f"hint: {hit!r} favours {n} (fit floored at {floor})")
            else:
                f = min(1.0, f * boost)
                notes.append(f"hint: {hit!r} boosts {n} x{boost} "
                             f"(no floor -- subject motions compete)")
        e = energy_of(n, kind_scales)
        match = 1.0 - min(1.0, abs(e - want) / 0.6)
        opts[n] = {"fit": f, "coh": 0.65 * match, "notes": notes, "m": m,
                   "energy": e}
    return opts, rejected, {"p_person": p_person, "p_source": p_source,
                            "has_person": has_person, "want": want}


def _fatigue(coh: float, m, movers_used: int, decided: int,
             notes: List[str]) -> float:
    """The film-so-far fatigue penalty. Identical in both modes.

    `decided >= 3` is not a fudge: the greedy path only starts applying the
    fatigue budget once three shots exist (`len(out) >= 3`), because before that
    the running fraction is meaningless. The global path must use the same
    condition or the two are not optimising the same objective.
    """
    if m.moving and decided >= 3 and (movers_used / decided) > 0.6:
        coh -= 0.25
        notes.append("film is already mostly moving -- fatigue")
    return max(0.0, min(1.0, coh))


def _continuity(m, prev_m, shot, prev_shot, seen_earlier: bool):
    """Continuity term between one motion and its predecessor. Both modes."""
    k, notes = 0.8, []
    if prev_m is not None:
        if (m.moving and prev_m.moving and shot.subject
                and prev_shot is not None and prev_shot.subject == shot.subject):
            if m.sign == prev_m.sign and m.axis == prev_m.axis:
                k = 1.0
                notes.append(f"continues {shot.subject}'s direction"
                             if m.kind == prev_m.kind else
                             f"reinforces the {prev_m.name} "
                             f"(same {m.axis} direction)")
            elif m.axis != prev_m.axis:
                k = 0.55
                notes.append(f"{shot.subject} changes axis "
                             f"({prev_m.axis} -> {m.axis})")
        elif m.moving and prev_m.moving and m.same_dof_as(prev_m) \
                and m.sign * prev_m.sign > 0:
            k = 0.9
    if m.subject_required and shot.subject and seen_earlier:
        k = max(k, 0.9)
    return k, notes


def _forbidden(m, prev_m) -> bool:
    """The 180-degree rule: same axis, opposite sign, straight after."""
    return bool(prev_m is not None and m.same_dof_as(prev_m)
                and m.sign * prev_m.sign < 0)


def _local_total(o: dict, coh: float, k: float, weights: Dict[str, float],
                 name: str, prog: float = 0.0) -> float:
    return (weights["fit"] * o["fit"] + weights["cohesiveness"] * coh
            + weights["continuity"] * k
            + weights.get("energy_arc", 0.0) * prog
            + (NONE_BONUS if name == "none" else 0.0))


def _arc_term(o: dict, i: int, n: int) -> float:
    """How well this motion's energy matches where the film should be by now.

    A MATCH, not a reward for energy -- see the note above. 1.0 is a perfect fit
    to the arc at this position.
    """
    return 1.0 - abs(energy_arc_target(i, n) - o["energy"])


def _viterbi(shots, opts_all, info_all, pen, cap, weights, energy_arc=True):
    """Exact best assignment. State = (previous motion, run, movers used)."""
    n = len(shots)
    cur = {(None, 0, 0): (0.0, None, None)}
    layers = []
    for i in range(n):
        opts = opts_all[i]
        prev_shot = shots[i - 1] if i > 0 else None
        nxt: Dict[tuple, tuple] = {}
        for (prev, run, movers), (val, _bp, _nm) in cur.items():
            prev_m = mv.get(prev) if prev else None
            for name, o in opts.items():
                m = o["m"]
                if _forbidden(m, prev_m):
                    continue
                if m.moving and run + 1 > MAX_MOVING_RUN:
                    continue
                nmovers = movers + (1 if m.moving else 0)
                if nmovers > cap:
                    continue
                nrun = run + 1 if m.moving else 0
                notes: List[str] = []
                coh = _fatigue(o["coh"], m, movers, i, notes)
                k, _kn = _continuity(m, prev_m, shots[i], prev_shot,
                                     info_all[i]["seen_earlier"])
                prog = _arc_term(o, i, n) if energy_arc else 0.0
                v = (val + _local_total(o, coh, k, weights, name, prog)
                     + pen.get(name, 0.0))
                key = (name, nrun, nmovers)
                if key not in nxt or v > nxt[key][0]:
                    nxt[key] = (v, (prev, run, movers), name)
        if not nxt:
            raise RuntimeError(
                "no feasible motion assignment: the mover cap or the run limit "
                "cannot be satisfied with the eligible candidates")
        layers.append(nxt)
        cur = nxt

    best_key, best_val = None, None
    for key, (v, _bp, _nm) in cur.items():
        if best_val is None or v > best_val:
            best_key, best_val = key, v
    choices = [None] * n
    key = best_key
    for i in range(n - 1, -1, -1):
        _v, bp, nm = layers[i][key]
        choices[i] = nm
        key = bp
    return choices, best_val


def _true_value(choices, shots, opts_all, info_all, weights, energy_arc=True) -> float:
    """The objective WITHOUT the diversity penalties, so solutions found under
    different penalties can be compared honestly."""
    n = len(shots)
    n = len(shots)
    val, run, movers = 0.0, 0, 0
    prev_m = None
    for i, name in enumerate(choices):
        o = opts_all[i][name]
        m = o["m"]
        run = run + 1 if m.moving else 0
        movers += 1 if m.moving else 0
        notes: List[str] = []
        coh = _fatigue(o["coh"], m, movers, i, notes)
        k, _ = _continuity(m, prev_m, shots[i], shots[i - 1] if i else None,
                           info_all[i]["seen_earlier"])
        prog = _arc_term(o, i, n) if energy_arc else 0.0
        val += _local_total(o, coh, k, weights, name, prog)
        prev_m = m
    return val


def _choose_global(shots, cands, scorer, kind_scales, weights, person_threshold,
                   strict_evidence, max_movers, evidence, energy_arc=True,
                   verbose=True):
    n = len(shots)
    cap = _mover_cap(n) if max_movers is None else int(max_movers)
    w_opt = dict(weights)
    if energy_arc:
        w_opt["energy_arc"] = ENERGY_ARC_WEIGHT
    opts_all, rej_all, info_all = [], [], []
    for i, s in enumerate(shots):
        o, r, inf = _shot_options(s, i, scorer, cands, kind_scales,
                                  person_threshold, strict_evidence)
        inf["seen_earlier"] = bool(s.subject) and any(
            shots[j].subject == s.subject for j in range(i))
        opts_all.append(o)
        rej_all.append(r)
        info_all.append(inf)

    # Diversity: the exact DP cannot see "has this motion been used before",
    # because that is not in its state. So it is relaxed instead -- find the
    # optimum, penalise any motion the solution used twice, re-solve, and keep
    # whichever solution scores best on the TRUE objective. Four rounds is
    # enough; at the 2-mover cap for a short film it is usually inert.
    pen: Dict[str, float] = {}
    best_choices, best_true = None, None
    rounds = 0
    prev_choices = None
    for _ in range(4):
        rounds += 1
        ch, _v = _viterbi(shots, opts_all, info_all, pen, cap, w_opt, energy_arc)
        tv = _true_value(ch, shots, opts_all, info_all, w_opt, energy_arc)
        if best_true is None or tv > best_true:
            best_choices, best_true = ch, tv
        if ch == prev_choices:
            break                     # the penalty is not moving the solution
        prev_choices = ch
        counts: Dict[str, int] = {}
        for c in ch:
            counts[c] = counts.get(c, 0) + 1
        newpen = dict(pen)
        for name, c in counts.items():
            if c > 1 and name != "none":
                newpen[name] = newpen.get(name, 0.0) + DIVERSITY_PENALTY
        if newpen == pen:
            break
        pen = newpen

    choices_out: List[Choice] = []
    movers = sum(1 for c in best_choices if mv.get(c).moving)
    for i, name in enumerate(best_choices):
        o = opts_all[i][name]
        m = o["m"]
        prev_name = best_choices[i - 1] if i > 0 else None
        prev_m = mv.get(prev_name) if prev_name else None
        notes = list(o["notes"])
        coh = _fatigue(o["coh"], m, 0, i, notes)   # recomputed for the notes only
        k, kn = _continuity(m, prev_m, shots[i], shots[i - 1] if i else None,
                            info_all[i]["seen_earlier"])
        notes.extend(kn)
        rejected = dict(rej_all[i])
        for cand_name, cand_o in opts_all[i].items():
            if _forbidden(cand_o["m"], prev_m):
                rejected[cand_name] = (
                    f"reverses the {prev_m.name} (same {m.axis} axis, "
                    f"opposite sign)")
        # best alternative with its neighbours held fixed -- this is what makes
        # the report arguable, and for a global optimum it is the honest
        # question ("would swapping just this shot have helped?")
        runner, runner_total = name, None
        for cand_name, cand_o in opts_all[i].items():
            if cand_name == name or _forbidden(cand_o["m"], prev_m):
                continue
            cn: List[str] = []
            c_coh = _fatigue(cand_o["coh"], cand_o["m"], 0, i, cn)
            c_k, _ = _continuity(cand_o["m"], prev_m, shots[i],
                                 shots[i - 1] if i else None,
                                 info_all[i]["seen_earlier"])
            t = _local_total(cand_o, c_coh, c_k, weights, cand_name)
            if runner_total is None or t > runner_total:
                runner, runner_total = cand_name, t
        if runner_total is None:
            runner, runner_total = name, _local_total(o, coh, k, weights, name)
        ch = Choice(index=i, prompt=shots[i].prompt, scale=shots[i].scale,
                    chosen=name, runner_up=runner,
                    fit=round(o["fit"], 3), cohesiveness=round(coh, 3),
                    continuity=round(k, 3),
                    total=round(_local_total(o, coh, k, weights, name), 3),
                    runner_total=round(runner_total, 3),
                    notes=notes, runner_notes=[],
                    rejected=rejected)
        choices_out.append(ch)
        if verbose:
            print(f"  [director] shot {i+1}: {name:<20} "
                  f"(fit {o['fit']:.2f} coh {coh:.2f} cont {k:.2f} = "
                  f"{ch.total:.3f})"
                  + (f"   runner-up {runner}" if runner != name else "")
                  + (f"   person p={info_all[i]['p_person']:.2f} "
                     f"[{info_all[i]['p_source']}]" if info_all[i]["has_person"]
                     else ""))

    if verbose:
        energies = [opts_all[i][c]["energy"] for i, c in enumerate(best_choices)]
        base = _true_value(best_choices, shots, opts_all, info_all, weights,
                           energy_arc=False)
        print(f"  [director] global optimum over {n} shots: movers {movers}/{cap} "
              f"(cap), base objective {base:.3f}, {rounds} solve(s)")
        # diagnostic only -- the arc is the SHOT LIST's property, not something
        # the motion choice can create. See the note above.
        print(f"  [director] film energy {energy_profile(energies)}  arc match "
              f"{arc_match(energies):.2f}"
              + ("" if energy_arc else "   (energy_arc is OFF)")
              + f"  (the writer's arc sets the shape; this term spends the moves "
              f"inside it)")
        print(f"  [director] a moving shot's place is chosen by the whole film: the "
              f"budget goes where the energy arc wants it, not to the earliest shots")
    out = [replace(s, camera=c) for s, c in zip(shots, best_choices)]
    return out, DirectorReport(choices_out, weights, evidence)


# --------------------------------------------------------------------------- #
# the A/B: greedy versus global, on the same shot list
# --------------------------------------------------------------------------- #

def _violations(names: Sequence[str]) -> List[str]:
    """Rule breaks in a finished sequence, checked independently of the chooser."""
    v: List[str] = []
    for i in range(1, len(names)):
        a, b = mv.get(names[i - 1]), mv.get(names[i])
        if b.same_dof_as(a) and b.sign * a.sign < 0:
            v.append(f"shot {i+1} reverses shot {i} ({a.name})")
    run = 0
    for i, n in enumerate(names):
        run = run + 1 if mv.get(n).moving else 0
        if run > MAX_MOVING_RUN:
            v.append(f"shot {i+1}: {run} movers in a row (R3)")
    return v


def compare_modes(shots: Sequence[Shot], **kw):
    """Does the film GO somewhere, or does it only obey a budget?

    Three variants on the same shot list, scored with the SAME base objective
    (fit + cohesiveness + continuity) so the columns are comparable:

      greedy (shipped)     no budget at all -- the control
      global (whole film)  the budget enforced exactly, and the 180-degree rule
                           and R3 as real constraints

    The energy arc is reported as a DIAGNOSTIC of the shot list, not as something
    the assignment optimises: see the note above `energy_arc_target` for the
    measurements showing a director-level arc term is a no-op. `base obj` is the
    comparable column; `after repair` adds what edit_planner had to undo.
    """
    kw.setdefault("verbose", False)
    scorer = kw.get("scorer") or FitScorer()
    cands = list(kw.get("candidates") or mv.available())
    kind_scales = energy_scale_by_kind(cands)
    opts_all, info_all = [], []
    for i, s in enumerate(shots):
        o, _r, inf = _shot_options(s, i, scorer, cands, kind_scales,
                                   kw.get("person_threshold", 0.5),
                                   kw.get("strict_evidence", False))
        inf["seen_earlier"] = bool(s.subject) and any(
            shots[j].subject == s.subject for j in range(i))
        opts_all.append(o)
        info_all.append(inf)
    w = kw.get("weights") or WEIGHTS
    n = len(shots)
    cap = _mover_cap(n) if kw.get("max_movers") is None else int(kw["max_movers"])

    variants = [
        ("greedy (shipped)", dict(mode="greedy")),
        ("global, budget only", dict(mode="global", energy_arc=False)),
        ("global + energy_arc", dict(mode="global", energy_arc=True)),
    ]

    from edit_planner import plan as _plan
    L = ["=" * 84,
         "DIRECTOR A/B -- does the film GO somewhere, or only obey a budget?",
         "=" * 84]
    L.append(f"  {n} shots, at most {cap} may move, no more than "
             f"{MAX_MOVING_RUN} in a row")
    # The writer's arc, which no motion choice can create. Hand-written shot lists
    # carry no `energy`, so fall back to what the director actually uses as `want`
    # (the framing scale's implied energy) rather than printing a row of zeros.
    declared = [(s.energy if s.energy is not None
                 else SCALE_ENERGY.get(s.scale, 0.5)) for s in shots]
    L.append(f"  target arc  energy {energy_profile([energy_arc_target(i, n) for i in range(n)])}"
             f"   (calm -> peak at {int(ENERGY_ARC_PEAK*100)}% -> resolve)")
    L.append(f"  SHOT LIST   energy {energy_profile(declared)}   arc match "
             f"{arc_match(declared):.2f}   <- the writer's arc, which no motion "
             f"choice can create")
    L.append("")
    L.append(f"  {'variant':<18} {'movers':>7} {'breaks':>7} {'arc':>6} "
             f"{'base obj':>9} {'after fix':>10} {'undone':>7}")
    rows = []
    for label, call in variants:
        out, _rep = choose(shots, **{**kw, **call})
        names = [s.camera for s in out]
        energies = [opts_all[i][c]["energy"] for i, c in enumerate(names)]
        cand = [replace(s, camera=c) for s, c in zip(shots, names)]
        planned, rep = _plan(cand)
        fixed = [s.camera for s in planned]
        hard = [c for c in rep.changes if "R4" not in c]
        row = {
            "label": label, "names": names, "energies": energies,
            "movers": sum(1 for c in names if mv.get(c).moving),
            "breaks": _violations(names),
            "arc": arc_match(energies),
            "obj": _true_value(names, shots, opts_all, info_all, w),
            "fixed": _true_value(fixed, shots, opts_all, info_all, w),
            "hard": hard,
        }
        rows.append(row)
        L.append(f"  {label:<18} {row['movers']:>4}/{cap:<2} "
                 f"{len(row['breaks']):>7} {row['arc']:>6.2f} {row['obj']:>9.3f} "
                 f"{row['fixed']:>10.3f} {len(hard):>7}")

    for row in rows:
        L.append("")
        L.append(f"  {row['label']}: energy profile {energy_profile(row['energies'])}"
                 f"   arc match {row['arc']:.2f}")
        last = None
        for i, c in enumerate(row["names"]):
            m = mv.get(c)
            if not m.moving:
                continue
            gap = "" if last is None else f", {i - last} shots since the last move"
            L.append(f"    shot {i+1}/{n}  target energy {energy_arc_target(i, n):.2f}  "
                     f"-> {c} (carries {opts_all[i][c]['energy']:.2f}){gap}")
            last = i
        for b in row["breaks"]:
            L.append(f"    RULE BREAK: {b}")
        for c in row["hard"]:
            L.append(f"    planner undid: {c}")

    g, arc_row = rows[0], rows[-1]
    L.append("")
    L.append("  READ THIS:")
    L.append(f"    budget    greedy {g['movers']} movers vs cap {cap}"
             + ("   <-- it exceeded the budget" if g["movers"] > cap else ""))
    L.append(f"    arc       shot list {arc_match(declared):.2f} -> greedy "
             f"{g['arc']:.2f} -> with energy_arc {arc_row['arc']:.2f}"
             + ("   <-- closer to the target shape"
                if arc_row["arc"] > g["arc"] + 0.005 else
                "   (no closer on this list)"))
    L.append("    A higher base objective with MORE movers is not a better film:")
    L.append("    it is the score of an assignment the brief does not allow.")
    return "\n".join(L)


def choose(shots: Sequence[Shot],
           candidates: Optional[Sequence[str]] = None,
           allow: Optional[Sequence[str]] = None,
           scorer: Optional[FitScorer] = None,
           weights: Dict[str, float] = WEIGHTS,
           strict_evidence: bool = False,
           person_threshold: float = 0.5,
           mode: str = "global",
           max_movers: Optional[int] = None,
           energy_arc: bool = True,
           verbose: bool = True) -> Tuple[List[Shot], DirectorReport]:
    """Pick a motion per shot.

    `mode="global"` optimises the whole sequence at once (see the long-view note
    above); `mode="greedy"` is the shipped left-to-right behaviour, kept as the
    A/B control and behaving exactly as it did.
    `max_movers` caps how many shots may move; None means `_mover_cap(len(shots))`.
    `energy_arc=True` also matches the film's energy against an arc that opens
    calm, peaks at ENERGY_ARC_PEAK of the way in, and resolves -- so the budget
    is spent where the shape wants it rather than only where each shot alone
    benefits.
    `allow` restricts the vocabulary (e.g. to what is downloaded).
    `strict_evidence` refuses any subject motion whose localization has not been
    measured at all -- nothing unproven gets to carry a shot.
    """
    from dataclasses import replace as _replace

    scorer = scorer or FitScorer()
    evidence = mv.load_evidence()
    allow = list(allow) if allow else mv.available()
    cands = list(candidates) if candidates else allow
    cands = [c for c in cands if c in allow]
    kind_scales = energy_scale_by_kind(cands)

    if mode == "global":
        return _choose_global(shots, cands, scorer, kind_scales, weights,
                              person_threshold, strict_evidence, max_movers,
                              evidence, energy_arc, verbose)
    if mode != "greedy":
        raise ValueError(f"unknown director mode {mode!r} "
                         f"(expected 'global' or 'greedy')")

    if verbose:
        print("  [director] GREEDY MODE (frozen control): each shot is decided "
              "knowing only the shot before it, so the moving budget goes to "
              "whichever shot comes first and an axis reversal can only be "
              "avoided once it has already been committed to. See _choose_global "
              "for what replaced it.")

    scorer = scorer or FitScorer()
    evidence = mv.load_evidence()
    allow = list(allow) if allow else mv.available()
    cands = list(candidates) if candidates else allow
    cands = [c for c in cands if c in allow]

    out: List[Shot] = []
    choices: List[Choice] = []
    kind_scales = energy_scale_by_kind(cands)

    for i, shot in enumerate(shots):
        p_person, p_source = scorer.person_detail(shot.prompt)
        has_person = p_person >= person_threshold
        want = shot.energy if shot.energy is not None else \
            SCALE_ENERGY.get(shot.scale, 0.5)

        # ---- hard gates ---------------------------------------------------- #
        eligible: List[str] = []
        rejected: Dict[str, str] = {}
        for n in cands:
            m = mv.get(n)
            if not m.fits_scale(shot.scale):
                rejected[n] = (f"does not fit a {shot.scale} shot"
                               + (" (it draws a full body)" if m.full_body else ""))
                continue
            if m.subject_required:
                if not has_person:
                    rejected[n] = (f"no person in this shot "
                                   f"(p={p_person:.2f} < {person_threshold})")
                    continue
                if strict_evidence and not m.measured:
                    rejected[n] = "unmeasured, and strict evidence is on"
                    continue
                # NOTE: there used to be a hard gate here on `m.loc10`, refusing
                # any subject LoRA measured below LOC10_TRUST. It was wrong twice
                # over: loc10 is confounded by magnitude (handoff 9.7.1), and
                # separability is scene-dependent -- every measurement we have is
                # on a textured scene, so gating on it would also refuse the
                # fog-framed shot that is precisely the experiment worth running.
                # It is now a fit penalty below, not a refusal.
            # 180-degree rule: same axis, opposite sign, straight after
            if out:
                prev = mv.get(out[-1].camera)
                if m.same_dof_as(prev) and m.sign * prev.sign < 0:
                    rejected[n] = (f"reverses shot {i}'s {prev.name} "
                                   f"(same {m.axis} axis, opposite sign)")
                    continue
            eligible.append(n)
        if "none" not in eligible:
            eligible.append("none")

        # run of moving shots already chosen (R3)
        run = 0
        for p in reversed(out):
            if mv.get(p.camera).moving:
                run += 1
            else:
                break
        # the fatigue budget only means anything once a few shots exist
        budget = (sum(1 for s in out if mv.get(s.camera).moving) / len(out)
                  if len(out) >= 3 else 0.0)

        scores: Dict[str, Tuple[float, float, float]] = {}
        notes_all: Dict[str, List[str]] = {}
        totals: Dict[str, float] = {}

        # Affinity is softmaxed *within kind*, not across the whole vocabulary.
        # The vocabulary has ~13 camera moves and 6 subject motions, so a flat
        # softmax over all 19 drowns a subject motion that is a perfect match for
        # its shot. Each kind is asked its own question instead, and the answers
        # are combined by the hard gates and the group priors below.
        by_kind: Dict[str, List[str]] = {"subject": [], "camera": [], "none": []}
        for n in eligible:
            by_kind.setdefault(mv.get(n).kind, []).append(n)
        aff_kind: Dict[str, Dict[str, float]] = {
            k: scorer.affinity(shot.prompt, v) for k, v in by_kind.items() if v}

        for n in eligible:
            m = mv.get(n)
            notes: List[str] = []

            # ---- fit ------------------------------------------------------- #
            if n == "none":
                # `none` is a first-class directorial choice, not a fallback.
                # The handoff's rule is that AnimateDiff already drifts, so
                # forcing a move on every shot is exhausting -- a still shot is
                # the *default*, and a move has to earn its place.
                f = NONE_FIT
            elif m.kind == "subject":
                f = aff_kind["subject"].get(n, 0.5)
                f *= 0.55 + 0.45 * p_person
                if not m.measured:
                    f *= 0.75
                    notes.append(f"{n}: unmeasured, fit discounted 25%")
                elif m.separable is False:
                    # evidence, not a verdict: measured on a textured scene where
                    # the background moved. A low-detail background may separate.
                    f *= 0.8
                    notes.append(
                        f"{n}: measured non-separable on a textured scene "
                        f"(global_share {m.global_share:.2f} vs baseline "
                        f"{mv.BASELINE.get('global_share', 0):.2f}) -- scene "
                        f"dependent, penalised not refused")
            else:
                f = aff_kind["camera"].get(n, 0.5) * CAMERA_FIT_CEILING

            boost, hit = hint_boost(shot.prompt, n)
            if boost > 1.0:
                floor = (SUBJECT_HINT_FLOOR if m.kind == "subject"
                         else HINT_FLOOR)
                if floor > 0.0:
                    f = min(1.0, max(f * boost, floor))
                    notes.append(f"hint: {hit!r} favours {n} (fit floored at "
                                 f"{floor})")
                else:
                    f = min(1.0, f * boost)
                    notes.append(f"hint: {hit!r} boosts {n} x{boost} "
                                 f"(no floor -- subject motions compete)")

            # ---- cohesiveness ---------------------------------------------- #
            e = energy_of(n, kind_scales)
            match = 1.0 - min(1.0, abs(e - want) / 0.6)
            c = 0.65 * match
            if m.moving and budget > 0.6:
                c -= 0.25
                notes.append("film is already mostly moving -- fatigue")
            if m.moving and run >= MAX_MOVING_RUN:
                c = 0.0
                notes.append(f"{run} moving shots in a row (R3)")
            c = max(0.0, min(1.0, c))

            # ---- continuity ------------------------------------------------ #
            k = 0.8
            if out:
                prev = mv.get(out[-1].camera)
                if (m.moving and prev.moving and shot.subject
                        and out[-1].subject == shot.subject):
                    # same person on screen: their direction of travel should
                    # carry across the cut, or the audience reads a reset
                    if m.sign == prev.sign and m.axis == prev.axis:
                        k = 1.0
                        if m.kind == prev.kind:
                            notes.append(f"continues {shot.subject}'s direction")
                        else:
                            notes.append(f"reinforces shot {i}'s {prev.name} "
                                         f"(same {m.axis} direction)")
                    elif m.axis != prev.axis:
                        k = 0.55
                        notes.append(f"{shot.subject} changes axis "
                                     f"({prev.axis} -> {m.axis})")
                elif m.moving and prev.moving and m.same_dof_as(prev) \
                        and m.sign * prev.sign > 0:
                    k = 0.9
            if m.subject_required and shot.subject and \
                    any(s.subject == shot.subject for s in out):
                k = max(k, 0.9)

            total = (weights["fit"] * f + weights["cohesiveness"] * c
                     + weights["continuity"] * k)
            if n == "none":
                total += NONE_BONUS
            scores[n] = (f, c, k)
            notes_all[n] = notes
            totals[n] = total

        order = sorted(eligible, key=lambda n: (-totals.get(n, 0), n))
        chosen = order[0]
        runner = order[1] if len(order) > 1 else chosen
        f, c, k = scores[chosen]
        ch = Choice(index=i, prompt=shot.prompt, scale=shot.scale,
                    chosen=chosen, runner_up=runner, fit=round(f, 3),
                    cohesiveness=round(c, 3), continuity=round(k, 3),
                    total=round(totals.get(chosen, 0), 3),
                    runner_total=round(totals.get(runner, 0), 3),
                    notes=notes_all.get(chosen, []),
                    runner_notes=notes_all.get(runner, []),
                    rejected=rejected)
        choices.append(ch)

        if verbose:
            print(f"  [director] shot {i+1}: {chosen:<20} "
                  f"(fit {f:.2f} coh {c:.2f} cont {k:.2f} = "
                  f"{totals.get(chosen,0):.3f})"
                  + (f"   runner-up {runner}" if runner != chosen else "")
                  + (f"   person p={p_person:.2f} [{p_source}]"
                     if has_person else ""))

        out.append(_replace(shot, camera=chosen))

    return out, DirectorReport(choices, weights, evidence)


# --------------------------------------------------------------------------- #
# demo / self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no-clip", action="store_true")
    a = ap.parse_args()

    sc = FitScorer(enabled=not a.no_clip)
    print("=== fit gate sanity check ===")
    probes = [
        ("a lone woman in a long coat walking toward the camera down an empty road", True),
        ("dew on a mossy leaf, tiny water droplets, macro", False),
        ("a vast dense forest at dawn, mist between the tall trees", False),
        ("a ballerina spinning on a bare stage under a single spotlight", True),
        ("a wide river pool reflecting the sky", False),
    ]
    wrong = 0
    for text, expect in probes:
        p = sc.person_probability(text)
        ok = (p >= 0.5) == expect
        wrong += not ok
        print(f"  p={p:.2f} {'OK ' if ok else 'WRONG'}  {text[:58]}")
    print(f"  {len(probes)-wrong}/{len(probes)} correct")

    seq = [
        Shot("a lone woman in a long coat walking down an empty road at dusk",
             "wide", "none", 2.5, subject="woman", energy=0.7),
        Shot("close on the woman's face as she walks, wind in her hair",
             "close", "none", 2.0, subject="woman", energy=0.45),
        Shot("jump cut: she runs, full body, camera tracking beside her",
             "medium", "none", 2.0, subject="woman", energy=0.9),
        Shot("dew on a mossy leaf, macro", "macro", "none", 2.0, energy=0.2),
        Shot("a vast forest at dawn, mist between the trees", "wide", "none", 3.0,
             energy=0.6),
        Shot("tall trees, canopy and sky above", "extreme-wide", "none", 3.5,
             energy=0.75),
    ]
    print("\n=== director ===")
    planned, rep = choose(seq, strict_evidence=a.strict,
                          scorer=sc, verbose=True)
    print("\n" + rep.show())
