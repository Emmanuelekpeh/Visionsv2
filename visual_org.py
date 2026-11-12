#!/usr/bin/env python3
"""
visual_org.py -- the hold/change spine: where a film holds a place and figure,
and where it deliberately changes them.

THE PROBLEM. The pipeline has five different ideas of "the same place" and none
of them agree:

  * `story` carries a `scene` string and a `subject` string -- prose, and measured
    to be cosmetic (within-scene prompt overlap 0.05 vs 0.08 across scenes; 0/10
    adjacent within-scene pairs share a subject in `seasoning`).
  * `model_select` only allows a model switch where a scene/act/subject/place
    changes -- it *assumes* holds exist.
  * `film_store` / `make_film` attach one palette per scene id, at scene
    boundaries.
  * `director`/`edit_planner` use subject for axis continuity and scene for the
    R10 move handoff.
  * pacing decides holds with no knowledge of place.
  * the render prompt is the writer's raw prose, so two shots in one scene do not
    even describe the same place or person.

The result is exactly the failure the archive shows: a "scene" that changes
subject, light and place on every cut. "Hold or change" is a real creative
decision -- sometimes hold a place and figure for two or three shots, sometimes
change fast -- but there is no object that records the decision, so every stage
guesses.

THE OBJECT. A `Run` is a contiguous run of shots that share ONE place (and, for a
figure-carrying film, ONE figure). A film's visual organisation is its sequence
of runs; the change points between them are the cuts that mean something. From
that single spine:

  * prompts get a canonical place/figure anchor (`effective_prompt`) so the
    renderer is told the same thing on every shot of a run;
  * palettes follow the PLACE, not an arbitrary scene index;
  * the planner's R10 / camera continuity and `model_select`'s "motivated
    switch" read the same place/figure ids;
  * pacing can make a hold breathe and a rapid-change run cut fast;
  * `diagnose` can finally say "your scene label is not a hold", "this figure
    changes mid-hold", "this place changes without a scene-cut".

This module is pure: it derives, diagnoses and proposes. It never renders and it
does not silently rewrite a project.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from edit_planner import Shot  # noqa: E402
from story import content_words, jaccard, link_overlap  # noqa: E402

MODES = ("auto", "story", "montage", "hybrid", "continuous")
# A hold that carries a place for fewer shots than this is a cutaway; a run of
# one in a story is usually an accident, not a choice.
MIN_HOLD = 2
# Above this many place/figure changes per minute the piece is a montage/rapid-
# change rhythm: legitimate, but it should be a declared intent, not a surprise.
RAPID_CHANGE_PER_MIN = 20.0


@dataclass
class Run:
    place: str
    figure: str
    indices: List[int]
    change_before: str = "none"     # what broke the previous run: place|figure|none

    @property
    def hold(self) -> int:
        return len(self.indices)

    @property
    def kind(self) -> str:
        return "hold" if self.hold >= MIN_HOLD else "single"

    @property
    def shot_range(self) -> str:
        return f"{self.indices[0] + 1}-{self.indices[-1] + 1}"


# --------------------------------------------------------------------------- #
# Normalising input: both the story `Shot` and the store's shot dicts
# --------------------------------------------------------------------------- #

def as_shots(items: Sequence) -> List[Shot]:
    out: List[Shot] = []
    for it in items:
        if isinstance(it, Shot):
            out.append(it)
            continue
        out.append(Shot(
            prompt=str(it.get("prompt") or ""),
            scale=str(it.get("scale") or "medium"),
            camera=str(it.get("camera_suggestion") or it.get("camera") or "none"),
            seconds=float(it.get("seconds") or 2.5),
            subject=str(it.get("subject") or ""),
            scene=str(it.get("scene") or ""),
            energy=float(it.get("energy") or 0.2),
            beat=str(it.get("beat") or "develop"),
            purpose=str(it.get("purpose") or ""),
            link=str(it.get("link") or ""),
            # carried so a store shot converts to the same Shot the draft path had.
            # `organise`/`reference_plan` do not read these (so hold hashes are
            # unchanged); `sequence_direction` does -- the writer's movement ids,
            # and the finishing camera -- and re-direct used to lose both.
            movement=str(it.get("movement") or ""),
            camera_move=(dict(it["camera_move"])
                         if isinstance(it.get("camera_move"), dict) else None),
            upsample=int(it.get("upsample") or 2),
            palette=str(it.get("palette") or "")))
    return out


# --------------------------------------------------------------------------- #
# The spine
# --------------------------------------------------------------------------- #

def organise(shots: Sequence) -> List[Run]:
    """Group shots into runs of one place and one figure.

    An insert (a shot with an empty subject -- a hand, a doorway, the rain) does
    NOT break a hold; it is coverage of the same place/figure. A non-empty
    subject that differs from the run's figure DOES break it: that is a change
    point, even if the place stays.
    """
    shots = as_shots(shots)
    runs: List[Run] = []
    for i, s in enumerate(shots):
        place = s.scene or "place1"
        figure = (s.subject or "").strip()
        if not runs:
            runs.append(Run(place, figure, [i], "none"))
            continue
        cur = runs[-1]
        if place != cur.place:
            runs.append(Run(place, figure, [i], "place"))
        elif figure and cur.figure and figure != cur.figure:
            runs.append(Run(place, figure, [i], "figure"))
        else:
            cur.indices.append(i)
            if figure and not cur.figure:
                cur.figure = figure
    return runs


def inferred_mode(runs: Sequence[Run], shots: Sequence) -> str:
    if not runs:
        return "auto"
    holds = sum(1 for r in runs if r.kind == "hold")
    if len(runs) == 1:
        return "continuous"
    if holds == 0:
        return "montage"
    if holds >= len(runs) - 1:
        return "story"
    return "hybrid"


def reference_plan(shots: Sequence, min_hold: int = MIN_HOLD) -> Dict[int, int]:
    """Which shot is the reference frame for each shot in a multi-shot run.

    The renderer's image-conditioning half needs one anchor per hold: the first
    shot of the run is generated normally, and every later shot in the same run
    is conditioned on its frame. Keys/values are shot indices. A single-shot
    place has no reference -- which is exactly what distinguishes a rapid change
    from a hold.
    """
    runs = organise(shots)
    plan: Dict[int, int] = {}
    for r in runs:
        if len(r.indices) >= min_hold:
            anchor = r.indices[0]
            for j in r.indices[1:]:
                plan[j] = anchor
    return plan


def organisation_stats(shots: Sequence, runs: Sequence[Run]) -> Dict:
    shots = as_shots(shots)
    duration = sum(s.seconds for s in shots) or 1e-9
    changes = sum(1 for r in runs if r.change_before != "none")
    return {
        "shots": len(shots), "runs": len(runs),
        "holds": sum(1 for r in runs if r.kind == "hold"),
        "singles": sum(1 for r in runs if r.kind == "single"),
        "changes": changes,
        "changes_per_min": round(60.0 * changes / duration, 1),
        "mean_hold": round(sum(r.hold for r in runs) / len(runs), 2),
        "longest_hold": max(r.hold for r in runs),
    }


# --------------------------------------------------------------------------- #
# Anchors: the only lever the renderer has for "the same place/figure"
# --------------------------------------------------------------------------- #

def _boilerplate(shots: Sequence[Shot]) -> set:
    """Words that appear in most shots of the film: `cinematic, filmic grain,
    rain, fog...`. They describe the FILM's look, not a PLACE, so they must not
    count as evidence that two shots are the same place -- measured failure:
    an early merge rule joined a bridge railing to a distant figure because both
    prompts said 'rain, cinematic, filmic grain'."""
    n = len(shots)
    counts: Dict[str, int] = {}
    for s in shots:
        for w in content_words(s.prompt):
            counts[w] = counts.get(w, 0) + 1
    return {w for w, c in counts.items() if c >= max(3, int(0.6 * n))}


def _word_match(a: str, b: str) -> bool:
    """Same entity, allowing a compound: 'bridge' matches 'footbridge'."""
    if a == b:
        return True
    return len(a) >= 5 and len(b) >= 5 and (a in b or b in a)


def _word_overlap(a: set, c: set) -> float:
    if not a and not c:
        return 0.0
    matched = sum(1 for w in a if any(_word_match(w, x) for x in c))
    return matched / max(1, len(a) + len(c) - matched)


def _distinctive_overlap(shots: Sequence[Shot], i: int, j: int,
                         boiler: Optional[set] = None) -> float:
    b = boiler if boiler is not None else _boilerplate(shots)
    a = content_words(shots[i].prompt) - b
    c = content_words(shots[j].prompt) - b
    return _word_overlap(a, c)


def infer_anchor(shots: Sequence, indices: Sequence[int]) -> str:
    """The words every shot in the run already shares, as a draft anchor.

    This is a DRAFT for a human to edit, not something to inject blindly: it is
    the intersection of content words, ordered by how much of the run uses them.
    """
    shots = as_shots(shots)
    boiler = _boilerplate(shots)
    counts: Dict[str, int] = {}
    for i in indices:
        for w in content_words(shots[i].prompt) - boiler:
            counts[w] = counts.get(w, 0) + 1
    common = [w for w, c in counts.items() if c >= max(2, len(indices) - 1)]
    return ", ".join(sorted(common, key=lambda w: (-counts[w], w))[:12])


def _clause_present(anchor: str, prompt: str) -> bool:
    a = content_words(anchor)
    if not a:
        return True
    return jaccard(a, content_words(prompt)) >= 0.5


def effective_prompt(shot: Shot, place_anchor: str = "",
                     figure_anchor: str = "") -> str:
    """The prompt the renderer should receive: action + canonical anchors.

    Only clauses that are NOT already in the prompt are appended, so a writer
    who already described the place is not duplicated, and a shot that only says
    "her hand on the rail" still gets told which rail, in which rain, with which
    light. This is the consistency mechanism available without image
    conditioning on this hardware.
    """
    parts = [shot.prompt.strip()]
    for anchor in (place_anchor, figure_anchor):
        if anchor and not _clause_present(anchor, shot.prompt):
            parts.append(anchor.strip())
    return ", ".join(p for p in parts if p)


def apply_anchors(shots: Sequence, runs: Sequence[Run],
                  places: Dict[str, str],
                  figures: Dict[str, str]) -> List[Shot]:
    shots = as_shots(shots)
    out: List[Shot] = []
    for r in runs:
        for i in r.indices:
            s = shots[i]
            out.append(replace(s, prompt=effective_prompt(
                s, places.get(r.place, ""), figures.get(r.figure, ""))))
    return out


# --------------------------------------------------------------------------- #
# Diagnostics: does the organisation line up end to end?
# --------------------------------------------------------------------------- #

def _adjacent_overlap(shots: Sequence[Shot], runs: Sequence[Run]) -> Tuple[float, float]:
    shots = as_shots(shots)
    inside, across = [], []
    for r in runs:
        for a, b in zip(r.indices, r.indices[1:]):
            inside.append(link_overlap(shots[a].prompt, shots[b].prompt))
    for r0, r1 in zip(runs, runs[1:]):
        across.append(link_overlap(shots[r0.indices[-1]].prompt,
                                   shots[r1.indices[0]].prompt))
    return (sum(inside) / len(inside) if inside else 0.0,
            sum(across) / len(across) if across else 0.0)


def diagnose(shots: Sequence, runs: Optional[Sequence[Run]] = None,
             mode: str = "auto", places: Optional[Dict[str, str]] = None,
             figures: Optional[Dict[str, str]] = None) -> Dict:
    """All the ways the organisation can fail to line up.

    Levels: `break` (the editor's intent and the material disagree), `warn`
    (probably a mistake, but could be a choice), `intent` (a legitimate rapid-
    change rhythm -- reported so it is a decision, not a surprise).
    """
    shots = as_shots(shots)
    runs = list(runs or organise(shots))
    places = places or {}
    figures = figures or {}
    mode = inferred_mode(runs, shots) if mode == "auto" else mode
    inside, across = _adjacent_overlap(shots, runs)
    stats = organisation_stats(shots, runs)
    issues: List[Dict] = []

    def add(level: str, code: str, msg: str, shots_in=None) -> None:
        issues.append({"level": level, "code": code, "message": msg,
                       "shots": list(shots_in or [])})

    # 1. DECLARED VS ACTUAL CHANGE. The cut must say what it does.
    for r0, r1 in zip(runs, runs[1:]):
        boundary = r1.indices[0]
        link = shots[boundary].link
        if r1.change_before == "place":
            if link != "scene-cut":
                level = "warn" if mode == "montage" else "break"
                add(level, "undeclared_place_change",
                    f"shot {boundary + 1} moves to a new place but is linked "
                    f"{link or '(none)'!r}, not 'scene-cut'. Declare it so the "
                    f"palette/planner land the change on the cut."
                    + ("" if mode == "montage" else
                       " Without it the viewer reads a jump cut."),
                    [boundary - 1, boundary])
            if mode in ("story", "hybrid", "continuous") and r0.hold < MIN_HOLD:
                add("warn", "short_hold",
                    f"place {r0.place!r} is held for {r0.hold} shot before the cut "
                    f"-- one-shot places are cutaways; hold {MIN_HOLD}+ or make the "
                    f"rapid-change run deliberate.",
                    r0.indices)
        elif r1.change_before == "figure":
            if link == "scene-cut":
                add("warn", "figure_change_as_scene_cut",
                    f"shot {boundary + 1} changes figure but keeps the place; a "
                    f"scene-cut here claims a new location. Use eyeline/reaction "
                    f"or make it a real place change.",
                    [boundary - 1, boundary])
            elif link in ("eyeline", "movement", "graphic") or \
                    shots[boundary].purpose in ("reaction", "resolve"):
                add("intent", "figure_enters",
                    f"a new figure enters the held place {r1.place!r} at shot "
                    f"{boundary + 1} -- intended if this is the beat where they "
                    f"appear; keep the place description identical across the cut.",
                    [boundary - 1, boundary])
            else:
                add("warn", "figure_drift",
                    f"the figure changes inside place {r1.place!r} at shot "
                    f"{boundary + 1} with no eyeline/reaction or place change "
                    f"declared. Either keep the same figure for the hold or make "
                    f"the introduction a cut the viewer can follow.",
                    [boundary - 1, boundary])

    # 2. A HOLD THAT DOES NOT HOLD. The label says one place; the prompts say
    #    unrelated images. This is the archive failure, caught at script time.
    #    Only meaningful when there actually IS a multi-shot run.
    if any(r.hold >= MIN_HOLD for r in runs):
        if across and inside <= across + 0.02:
            add("break", "cosmetic_hold",
                f"shots inside a run share {inside:.2f} prompt overlap, no more "
                f"than the {across:.2f} between runs -- the place/figure label is "
                f"cosmetic; these are separate pictures, not coverage of one place.")
        elif inside < 0.10:
            add("warn", "thin_hold",
                f"within-run prompt overlap is only {inside:.2f}: the shots do not "
                f"describe the same place/figure well enough to read as a hold.")

    # 3. ONE-SHOT PLACES IN A STORY. The corpus says humans hold; a middle run of
    #    one is either a cutaway/insert or a mistake.
    if mode in ("story", "hybrid"):
        for k, r in enumerate(runs):
            if r.hold == 1 and 0 < k < len(runs) - 1:
                add("warn", "one_shot_place",
                    f"place {r.place!r} is a single shot in a {mode} piece -- if it "
                    f"is a detail of a held place, give it the same place id; if it "
                    f"is a new place, expect the viewer to reset.",
                    r.indices)

    # 4. RAPID CHANGE. Not a defect: a tempo. Report so it is intentional.
    if stats["changes_per_min"] >= RAPID_CHANGE_PER_MIN:
        add("intent", "rapid_change",
            f"{stats['changes_per_min']:.0f} place/figure changes per minute -- "
            f"this is a montage/rapid-change rhythm. Fine as an intent; the pacing "
            f"and the palette must cut with it (hold runs of one, palette changes "
            f"at the cuts) rather than fight it.")
    if mode == "continuous" and len(runs) > 1:
        add("warn", "continuous_mode_changes",
            f"mode is 'continuous' but there are {len(runs)} place/figure changes. "
            f"Either hold the place or declare the montage intent.")
    if mode == "montage" and any(r.kind == "hold" for r in runs):
        add("warn", "montage_mode_holds",
            "mode is 'montage' but some place/figure runs hold for 2+ shots. A "
            "montage is allowed to hold, but that is a hybrid -- declare it.")

    # 5. ANCHORS. If a place/figure has an anchor, every shot in its run should
    #    already carry its words (otherwise the render prompt drifts).
    for r in runs:
        for i in r.indices:
            for kind, anchor in (("place", places.get(r.place, "")),
                                 ("figure", figures.get(r.figure, ""))):
                if anchor and not _clause_present(anchor, shots[i].prompt):
                    add("warn", f"missing_{kind}_anchor",
                        f"shot {i + 1} is in {kind} {r.place if kind == 'place' else r.figure!r} "
                        f"but does not describe its anchor. `effective_prompt` will "
                        f"append it at freeze; the writer should not rely on that.",
                        [i])

    # 6. A HOLD WITH NO BREATH (pacing): equal 2 s holds inside a run cut like a
    #    montage no matter what the place labels say.
    for r in runs:
        if r.hold >= 3:
            holds = [shots[i].seconds for i in r.indices]
            mean = sum(holds) / len(holds)
            sd = (sum((h - mean) ** 2 for h in holds) / len(holds)) ** 0.5
            if sd / max(1e-9, mean) < 0.12:
                add("warn", "flat_hold",
                    f"place {r.place!r} is held for {r.hold} shots but every hold "
                    f"is ~{mean:.1f}s -- a held place needs a long shot and a "
                    f"shorter one, or it still reads as a montage.",
                    r.indices)

    breaks = [i for i in issues if i["level"] == "break"]
    return {"mode": mode, "inferred_mode": inferred_mode(runs, shots),
            "stats": stats, "inside_overlap": round(inside, 3),
            "across_overlap": round(across, 3), "issues": issues,
            "breaks": len(breaks), "warns": sum(1 for i in issues if i["level"] == "warn")}


# --------------------------------------------------------------------------- #
# The intuitive surface: a strip, and one-tap rearrangements
# --------------------------------------------------------------------------- #

def strip_text(shots: Sequence, runs: Optional[Sequence[Run]] = None) -> str:
    shots = as_shots(shots)
    runs = list(runs or organise(shots))
    cells = [(r, max(9, len(r.place) + 2, len(r.figure) + 2,
                     len(" ".join(str(i + 1) for i in r.indices)) + 2))
             for r in runs]

    def row(prefix: str, body) -> str:
        return prefix + "".join(body(r, w) for r, w in cells)

    rows = [
        row("  place  ", lambda r, w: f"{r.place[:w - 2]:<{w}}"),
        row("  figure ", lambda r, w: f"{(r.figure or '-')[:w - 2]:<{w}}"),
        row("  cut    ", lambda r, w: ("|" + " " * (w - 1)) if r.change_before != "none"
            else " " * w),
        row("  shot   ", lambda r, w: " ".join(str(i + 1) for i in r.indices)[:w]
            .ljust(w)),
    ]
    return "\n".join(rows)


def strip_png(shots: Sequence, runs: Optional[Sequence[Run]] = None,
              mode: str = "auto", out: str | Path = "output/visual_org/strip.png",
              title: str = "") -> Path:
    """A picture of the organisation: hold bars per place, figure dots, cuts."""
    from PIL import Image, ImageDraw

    shots = as_shots(shots)
    runs = list(runs or organise(shots))
    PX = 136
    W = max(900, 40 + PX * len(shots))
    H = 250
    im = Image.new("RGB", (W, H), (18, 20, 24))
    d = ImageDraw.Draw(im)
    palette = [(70, 130, 180), (196, 120, 70), (120, 170, 110), (170, 110, 170),
               (190, 170, 80), (90, 160, 170), (180, 100, 120)]
    place_col = {}
    for k, r in enumerate(runs):
        place_col.setdefault(r.place, palette[len(place_col) % len(palette)])
    d.text((14, 10), f"visual organisation  ·  mode {mode}"
           + (f"  ·  {title}" if title else ""), fill=(230, 230, 235))
    d.text((14, 34), "place", fill=(150, 155, 165))
    d.text((14, 104), "figure", fill=(150, 155, 165))
    d.text((14, 172), "cut", fill=(150, 155, 165))
    for r in runs:
        x0 = 80 + r.indices[0] * PX
        x1 = 80 + (r.indices[-1] + 1) * PX
        col = place_col[r.place]
        d.rounded_rectangle((x0 + 2, 30, x1 - 2, 76), radius=8, fill=col)
        d.text((x0 + 8, 42), f"{r.place[:16]}", fill=(10, 10, 12))
        d.text((x0 + 8, 58), f"hold {r.hold}" if r.kind == "hold" else "single",
               fill=(10, 10, 12))
        d.text((x0 + 8, 100), r.figure or "-", fill=(210, 210, 215))
        if r.change_before != "none":
            d.line((x0, 130, x0, 205), fill=(240, 200, 90), width=3)
            d.text((x0 + 4, 140), r.change_before, fill=(240, 200, 90))
    for i, s in enumerate(shots):
        x = 80 + i * PX
        d.text((x + 4, 214), f"shot {i + 1}  {s.seconds:g}s", fill=(180, 185, 195))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    return out


def suggestion(shots: Sequence, mode: str = "hybrid") -> Tuple[List[Shot], List[str]]:
    """Rearrange the scene ids/links to express `mode`. Pure; returns new shots.

    This is the one-tap version of the decision: it does not invent images, it
    groups the existing ones into holds or rapid changes according to the
    material (prompt overlap) and the chosen intent.
    """
    shots = as_shots(shots)
    runs = organise(shots)
    notes: List[str] = []
    new = [replace(s) for s in shots]
    if mode == "montage":
        for i, s in enumerate(new):
            new[i] = replace(s, scene=f"m{i + 1}",
                             link="open" if i == 0 else "scene-cut")
        return new, ["every shot is its own place: a rapid-change montage"]
    if mode == "continuous":
        for i, s in enumerate(new):
            new[i] = replace(s, scene="s1",
                             link="open" if i == 0 else "match-action")
        return new, ["one place for the whole piece; cuts are coverage, not changes"]
    # story/hybrid: merge a single-shot place into its neighbour when the shots
    # actually describe the same place (distinctive overlap), otherwise keep it
    # as a change. Generic atmosphere words are excluded -- otherwise every rainy
    # shot looks like the same place.
    place_of = [s.scene for s in new]
    boiler = _boilerplate(shots)
    for k, r in enumerate(runs):
        if r.hold != 1 or k == 0 or k == len(runs) - 1:
            continue
        prev = runs[k - 1]
        ov_prev = _distinctive_overlap(shots, prev.indices[-1], r.indices[0], boiler)
        nxt = runs[k + 1]
        ov_next = _distinctive_overlap(shots, r.indices[0], nxt.indices[0], boiler)
        if max(ov_prev, ov_next) >= 0.15:
            target = prev.place if ov_prev >= ov_next else nxt.place
            place_of[r.indices[0]] = target
            notes.append(f"shot {r.indices[0] + 1} merged into place {target!r} "
                         f"(distinctive overlap {max(ov_prev, ov_next):.2f})")
        else:
            notes.append(f"shot {r.indices[0] + 1} kept as a change (distinctive "
                         f"overlap {max(ov_prev, ov_next):.2f} -- atmosphere words "
                         f"are not a place)")
    # renumber places contiguously and relink
    seen: Dict[str, str] = {}
    for i, sc in enumerate(place_of):
        seen.setdefault(sc, f"p{len(seen) + 1}")
    prev_sc = None
    for i, s in enumerate(new):
        sc = seen[place_of[i]]
        link = "open" if i == 0 else ("scene-cut" if sc != prev_sc else "match-action")
        new[i] = replace(s, scene=sc, link=link)
        prev_sc = sc
    notes.append(f"{len(seen)} place(s); mode {mode}")
    return new, notes


def apply_to_store(project: Dict, mode: str) -> Tuple[List[Shot], List[str]]:
    """Apply an arrangement to a `film_store` production (the studio's model).

    The workbench version lives in `film_project.apply_organisation`; this is the
    store-side equivalent so the STUDIO can rearrange too. Palettes follow the
    place: each new scene group inherits the palette of its first shot's old
    scene, so a montage <-> coverage change cannot silently drop a look.
    """
    import film_store as fs

    dicts = fs.shots_in_order(project)
    shots = as_shots(dicts)
    old_scene = [s.scene for s in shots]
    old_pal = {s.get("id"): s.get("palette") for s in project.get("scenes") or []}
    new, notes = suggestion(shots, mode)
    new_pal: Dict[str, str] = {}
    for old_sc, ns in zip(old_scene, new):
        new_pal.setdefault(ns.scene, old_pal.get(old_sc, "neutral"))
    for d, ns in zip(dicts, new):
        fs.update_shot(project, d, scene=ns.scene, link=ns.link)
    fs.sync_scenes(project)
    for sc in project.get("scenes") or []:
        sc["name"] = str(sc.get("id") or "?")
    for sid, pal in new_pal.items():
        fs.set_palette(project, sid, pal)
    org = project.setdefault("organisation", {})
    org["mode"] = mode
    org["notes"] = notes
    fs.save(project)
    return new, notes


def apply_anchors_to_store(project: Dict) -> int:
    """Bake the project's place/figure anchors into the store shot prompts.

    Explicit and destructive-on-purpose: the prompt is the render input, so it is
    what the generation hash reads. Rewriting it here means a changed anchor
    correctly marks the take stale instead of drifting behind a side table.
    Returns the number of prompts changed.
    """
    import film_store as fs

    dicts = fs.shots_in_order(project)
    shots = as_shots(dicts)
    org = project.get("organisation") or {}
    places = org.get("places") or {}
    figures = org.get("figures") or {}
    changed = 0
    for r in organise(shots):
        for i in r.indices:
            new_prompt = effective_prompt(shots[i], places.get(r.place, ""),
                                          figures.get(r.figure, ""))
            if new_prompt != shots[i].prompt:
                fs.update_shot(project, dicts[i], prompt=new_prompt)
                changed += 1
    if changed:
        fs.save(project)
    return changed


# Fixed style vocabulary: words that describe the FILM's look, not a place or
# figure. Used by `affinity` (the sequence director) which must keep place/figure
# nouns as evidence even when they repeat across a hold -- unlike `_boilerplate`,
# whose frequency test is right for deciding whether two shots are the same
# *image* but would erase 'footbridge/woman/coat' from a four-shot hold.
STYLE_WORDS = {
    "cinematic", "filmic", "grain", "quality", "detailed", "photograph",
    "realistic", "atmosphere", "lighting", "composition", "shot", "frame",
    "masterpiece", "best", "highly", "8k", "4k", "hd", "bokeh", "style",
}
# Atmosphere is shared by DIFFERENT places (a rainy bridge and a rainy alley are
# not the same place). Excluding it is what stops "rain, fog, night, streetlights"
# from reading as location continuity.
ATMOSPHERE_WORDS = {
    "rain", "raining", "rainy", "rain-soaked", "raindrops", "fog", "foggy",
    "night", "nighttime", "dark", "darkness", "wet", "streetlight",
    "streetlights", "light", "lights", "shadow", "shadows", "mist", "misty",
}


def continuity_overlap(shots: Sequence[Shot], i: int, j: int) -> float:
    """Overlap for 'is this the same place/figure?', style + atmosphere ignored."""
    a = content_words(shots[i].prompt) - STYLE_WORDS - ATMOSPHERE_WORDS
    c = content_words(shots[j].prompt) - STYLE_WORDS - ATMOSPHERE_WORDS
    return _word_overlap(a, c)


def affinity(shots: Sequence, i: int, j: int) -> float:
    """How much two shots look like the same place/figure, from their words.

    Style words are ignored, but place/figure nouns are NOT frequency-filtered --
    a hold repeats its place on purpose. This is the notion the sequence director
    and the studio both use.
    """
    shots = as_shots(shots)
    return continuity_overlap(shots, i, j)


def _shots_from_store(slug: str) -> Tuple[List[Shot], str]:
    import film_store as fs
    project = fs.load(slug)
    return as_shots(fs.shots_in_order(project)), slug


def _shots_from_project(name: str) -> Tuple[List[Shot], str]:
    import film_project as fp
    proj = fp.load(name)
    return as_shots(fp.shot_objects(proj)), f"project:{name}"


def _shots_from_proposal(path: str) -> Tuple[List[Shot], str]:
    from story import validate
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    shots, _ = validate(raw)
    return shots, Path(path).stem


def main() -> int:
    ap = argparse.ArgumentParser(description="Hold/change visual organisation")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--store", help="production slug (film_store)")
    src.add_argument("--project", help="workspace film name (film_project)")
    src.add_argument("--proposal", help="cached proposal json")
    ap.add_argument("--mode", choices=MODES, default="auto")
    ap.add_argument("--png", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--suggest", choices=("story", "hybrid", "montage", "continuous"),
                    default=None, help="propose a rearranged shot list")
    a = ap.parse_args()
    if a.store:
        shots, name = _shots_from_store(a.store)
    elif a.project:
        shots, name = _shots_from_project(a.project)
    else:
        shots, name = _shots_from_proposal(a.proposal)
    if not shots:
        print("no shots")
        return 1
    runs = organise(shots)
    rep = diagnose(shots, runs, mode=a.mode)
    print(f"\n=== visual organisation: {name} ({len(shots)} shots) ===")
    print(strip_text(shots, runs))
    print(f"\n  inferred mode: {rep['inferred_mode']} · "
          f"{rep['stats']['runs']} runs ({rep['stats']['holds']} holds, "
          f"{rep['stats']['singles']} singles) · "
          f"{rep['stats']['changes']} changes "
          f"({rep['stats']['changes_per_min']}/min)")
    print(f"  overlap inside runs {rep['inside_overlap']} vs across "
          f"{rep['across_overlap']}")
    if rep["issues"]:
        print("\n  diagnostics:")
        for it in rep["issues"]:
            print(f"    [{it['level']:<6}] {it['code']}: {it['message']}")
    else:
        print("\n  no organisation problems found.")
    if a.png:
        print(f"\n  strip: {strip_png(shots, runs, rep['mode'], a.png, title=name)}")
    if a.suggest:
        new, notes = suggestion(shots, a.mode if a.mode != "auto" else a.suggest)
        print(f"\n  SUGGESTED ({a.suggest}):")
        for n in notes:
            print(f"    - {n}")
        print(strip_text(new, organise(new)))
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps({"name": name, "report": rep},
                                               indent=2), encoding="utf-8")
        print(f"  json: {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
