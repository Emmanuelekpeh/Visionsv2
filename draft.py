"""
draft.py -- write a film's shot list, and PUT IT IN THE PRODUCTION.

The gap this closes. `story.py` writes a proposal to `output/film/proposals/` and
prints it; nothing consumed it. The workbench's "write a shot list" button therefore
ran a real ensemble job for 4-7 minutes and left the project the user was *looking
at* completely unchanged -- the run only showed up in the old UI, which reads the
cache directly. That is the whole reason a re-run looked like it had done nothing.

So: one job, one effect. Run the ensemble, then adopt the proposal into the store in
the same process, so the page the user is on is different when the job finishes.

    python draft.py --store shadows --concept "..." --n-shots 4

Prints the same lines `story.py` prints, plus `[draft]` markers the job runner folds
into structured progress. It writes nothing until the proposal validates, and it
REFUSES to adopt a proposal whose beat order still fails the story check unless
`--allow-broken-story` is passed: the story gate exists to stop exactly the kind
of film most of the existing productions turned out to be.

THE LOOP. A draft is not a one-shot guess: the writer's own output is fed back
until the film either holds together or the attempts run out.
  * inside one call, `story._repair_note` re-asks with the validator's exact
    complaints and the previous shot list attached, and `beat_layout` gives the
    model a numbered beat roster so the order has a position to obey;
  * a CLEAN but SHORT list is extended with coverage inside its existing scenes
    (never re-written) until the requested shot count is reached;
  * a REFUSED draft is written to `productions/<slug>/draft_attempt.json` --
    beside the production, not inside it -- so the next "write new ideas" is fed
    the rejected writing instead of starting blind.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch_home"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_home"))

# WHERE A REFUSED DRAFT GOES. It is deliberately NOT in project.json: a refusal
# means the production was not changed, and writing the rejected timeline into the
# project to remember it would make that sentence false. A sidecar file in the
# production's own directory is exactly the scope of the thing: this production,
# this concept, the draft that was not good enough yet.
PENDING_DRAFT = "draft_attempt.json"


def _pending_path(slug: str) -> Path:
    import film_store as fs
    return fs.dir_for(slug) / PENDING_DRAFT


def load_pending_draft(slug: str, concept: str) -> Optional[Dict]:
    """The refused draft from a previous run, if it is for THIS concept.

    The concept check matters: a draft refused for "ego death" must not be fed
    back as "the previous attempt" for a completely different film. Same slug,
    different idea -> the loop has nothing to continue from.
    """
    p = _pending_path(slug)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (data.get("concept") or "").strip().lower() != (concept or "").strip().lower():
        return None
    if not data.get("shots"):
        return None
    return data


def save_pending_draft(slug: str, concept: str, proposal, requested) -> Path:
    """Record a refused draft so the NEXT run can be handed it back.

    The draft loop inside one run already feeds a failed attempt into the next;
    this is how the writing survives the process. Before it existed, pressing
    "write new ideas" after a refusal re-ran the identical brief and the same
    backwards beat came back -- the loop's knowledge died with the job.
    """
    from story import shot_dict, story_shape

    p = _pending_path(slug)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "concept": concept,
        "requested": requested,
        "shots": [shot_dict(s) for s in proposal.shots],
        "problems": list(getattr(proposal, "problems", None) or []),
        "story_shape": story_shape(proposal.shots),
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")
    return p


def clear_pending_draft(slug: str) -> None:
    """A draft was adopted, so there is nothing to repair any more."""
    try:
        _pending_path(slug).unlink()
    except OSError:
        pass


def story_gate(problems, allow_broken_story: bool = False):
    """The adoption gate: may this proposal replace the production's timeline?

    Returns `(allowed, breaks, warnings)`. Only `story:` blocks: a film whose
    beats go backwards or never resolves is not a film, and adoption is the last
    place that can refuse it. `structure:` is ADVISORY, not a break -- the A/B
    (`SEQUENCE_AB.md`) and the organisation layer showed that a montage and a
    held coverage scene are both legitimate; refusing one grammar at the gate was
    the "one correct grammar" mistake. Structure problems are surfaced, the
    ORGANISE stage is where they are resolved or deliberately accepted.
    `story-link:`/`rhythm:` stay advisory for the same reason. `prompt:` and
    `coverage:` are advisory too: a prompt that describes two images, or a film
    that is a montage of places with no scene that holds, are quality problems
    (the prompt pass and the coverage repair target them), not broken stories, so
    refusing adoption over them would block a film the story gate accepted.
    """
    breaks = [p for p in (problems or []) if p.startswith("story:")]
    warns = [p for p in (problems or []) if p.startswith(("story-link:", "structure:",
                                                         "rhythm:", "prompt:",
                                                         "coverage:", "critic:"))]
    return (bool(allow_broken_story) or not breaks), breaks, warns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="the production slug to write into")
    ap.add_argument("--concept", required=True)
    ap.add_argument("--n-shots", type=int, default=None)
    ap.add_argument("--farms", default="qwen2.5:3b,phi3:latest,mistral:latest")
    ap.add_argument("--showrunner", default="granite4.1:8b")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the proposal cache. Without this, the same concept "
                         "returns the same proposal -- byte for byte. A refused "
                         "draft from a previous run forces this on, because "
                         "returning the refusal again would not be a loop")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--story-retries", type=int, default=2,
                    help="how many REPAIR passes to ask for after a draft breaks "
                         "the story check (default 2 == up to 3 writing passes). "
                         "0 = one pass only: faster, but a broken draft is refused "
                         "instead of repaired")
    ap.add_argument("--no-prompt-pass", action="store_true",
                    help="skip the text pass that rewrites every prompt as ONE "
                         "concrete image. On by default: the first rendered film "
                         "had twelve two-image prompts and the renderer drew one "
                         "clause of each")
    ap.add_argument("--prompt-pass-model", default=None,
                    help="which model does the text-only prompt pass (default: "
                         "story.PROMPT_PASS_MODEL, measured on the same approved "
                         "prompts -- this is a formatting task, not a writing one)")
    ap.add_argument("--character", default=None,
                    help="ONE descriptor put in front of the figure in every shot "
                         "(e.g. 'male'), so the person does not change between "
                         "shots. Kept on the project; later drafts reuse it")
    ap.add_argument("--avoid", default="",
                    help="words the full sampler must avoid for that character "
                         "(e.g. 'woman, female, long hair')")
    ap.add_argument("--no-self-critique", action="store_true",
                    help="skip the self-critique: after the prompt pass the draft "
                         "is checked by rules (absence wording, the figure missing "
                         "from the ending) and by a critic model, and each flagged "
                         "shot is rewritten alone. On by default")
    ap.add_argument("--critic-model", default=None,
                    help="model that reads and repairs the shot list "
                         "(default: the prompt-pass model)")
    ap.add_argument("--no-scene-plan", action="store_true",
                    help="skip the cheap planning pass that picks 2-3 places and a "
                         "recurring figure BEFORE the shots. Without it the writer "
                         "tends to invent one image per beat -- a montage of places")
    ap.add_argument("--coverage-retries", type=int, default=1,
                    help="re-asks allowed when the story holds but no scene holds "
                         "one place for 2+ shots (default 1). A montage is a "
                         "legitimate film, so this is a writing demand, not a gate")
    ap.add_argument("--allow-broken-story", action="store_true",
                    help="adopt a proposal whose beat order still fails the story "
                         "check. Off by default: the failure is the reason the "
                         "story layer exists, so shipping it must be explicit")
    ap.add_argument("--cameras", default=None,
                    choices=["off", "suggestions", "director", "hybrid"],
                    help="who decides which shots move. off = all still (the old "
                         "behaviour); suggestions = the writer's own suggestion, "
                         "continuity-repaired; director = director.choose(); "
                         "hybrid (default) = director first, writer's suggestion "
                         "where the director declines. Falls back to the project "
                         "setting `camera_mode` when omitted")
    a = ap.parse_args()

    import film_store as fs
    from story import story_shape, write_shots_ensemble

    project = fs.load(a.store)
    print(f"[draft] production {a.store}: {len(project.get('timeline') or [])} "
          f"shot(s) before", flush=True)
    print(f"[draft] concept: {a.concept}", flush=True)

    # THE LOOP CONTINUES ACROSS RUNS. A draft that was refused last time is on
    # disk (never in project.json -- the production really is unchanged). Handing
    # it back as `prior` is what makes "write new ideas" a continuation instead of
    # a blind re-run; a cache hit for this concept would only replay the refusal,
    # so the prior also forces the cache open.
    prior_raw = load_pending_draft(a.store, a.concept)
    prior = None
    if prior_raw:
        from story import shots_from_dicts
        prior_shots = shots_from_dicts(prior_raw.get("shots") or [])
        if prior_shots:
            prior = {"shots": prior_shots,
                     "problems": list(prior_raw.get("problems") or [])}
            print(f"[draft] a previous draft of this concept was refused "
                  f"({len(prior_shots)} shot(s), shape "
                  f"{prior_raw.get('story_shape') or '?'}); feeding it back to the "
                  f"writer", flush=True)
    fresh = bool(a.fresh or prior)

    from story import ShowrunnerFailure
    try:
        proposal = write_shots_ensemble(
            a.concept,
            farms=[m.strip() for m in a.farms.split(",") if m.strip()],
            showrunner=a.showrunner, n_shots=a.n_shots, temperature=a.temperature,
            fresh=fresh, verbose=True, prior=prior, extend=True,
            story_retries=max(0, int(a.story_retries)),
            prompt_pass=not a.no_prompt_pass,
            prompt_pass_model=a.prompt_pass_model,
            scene_plan=not a.no_scene_plan,
            coverage_retries=max(0, int(a.coverage_retries)),
            self_critique=not a.no_self_critique, critic_model=a.critic_model)
    except ShowrunnerFailure as exc:
        # every showrunner attempt failed to parse/validate: say what the model
        # actually emitted, leave the production alone, and exit non-zero so the job
        # shows as failed instead of dying in a traceback with no explanation
        print(f"[draft] REFUSED: {exc}. The production is unchanged.", flush=True)
        for r in (getattr(exc, "replies", None) or [])[:3]:
            print(f"[draft]   attempt {r.get('attempt', '?')}: "
                  f"{str(r.get('failure') or '?')[:200]}", flush=True)
        print(f"[draft] try again (the writer is non-deterministic), a different "
              f"--showrunner, or check that ollama is serving "
              f"{a.showrunner}", flush=True)
        return 1

    shots = list(getattr(proposal, "shots", None) or [])
    if not shots:
        print("[draft] the writer returned no usable shots; the production is "
              "unchanged", flush=True)
        return 1

    # THE LOOP, IN ONE LINE. What was asked for, how many writing attempts it took,
    # how many extension calls closed the count, and the shape that came out. This
    # is the line that makes "did the loop work?" answerable from the log alone.
    if a.n_shots:
        calls = int(getattr(proposal, "extend_calls", 0) or 0)
        rewritten = int(getattr(proposal, "prompts_rewritten", 0) or 0)
        plan = getattr(proposal, "scene_plan", None) or {}
        cov_rep = int(getattr(proposal, "coverage_repairs", 0) or 0)
        cov_left = sum(1 for p in (proposal.problems or [])
                       if p.startswith("coverage:"))
        print(f"[draft] writing loop: {getattr(proposal, 'attempts', 0)} attempt(s)"
              + (f", {calls} extension call(s)" if calls else "")
              + (f", {rewritten} prompt(s) rewritten" if rewritten else "")
              + (f", {len(plan.get('scenes') or [])} scene(s) planned" if plan
                 else "")
              + (f", {cov_rep} coverage repair(s)" if cov_rep else "")
              + (f", prompts anchored to {len(plan.get('scenes') or [])} plan "
                 f"scene(s)" if getattr(proposal, "anchored", False) else "")
              + f" -> {len(shots)}/{a.n_shots} shot(s), story shape "
              f"{story_shape(shots)}"
              + (f", {cov_left} coverage problem(s) left" if cov_left else "")
              + (" (repaired a refused draft)" if prior else ""), flush=True)

    # a real decision point, reported rather than buried: the writer's own shot list
    # is shown here so the log is readable on its own
    allowed, breaks, warns = story_gate(proposal.problems, a.allow_broken_story)
    n_story = sum(1 for b in breaks if b.startswith("story:"))
    n_struct = sum(1 for w in warns if w.startswith("structure:"))
    print(f"[draft] proposal validated: {len(shots)} shots, {n_story} story "
          f"break(s), {n_struct} structure advisory(ies), {len(warns)} warning(s)",
          flush=True)
    if a.n_shots and len(shots) < a.n_shots:
        print(f"[draft] note: the writer returned {len(shots)} of {a.n_shots} shots "
              f"after {getattr(proposal, 'attempts', 0)} attempt(s) and "
              f"{getattr(proposal, 'extend_calls', 0)} extension call(s). The count "
              f"is a request to the writer, not enforced -- the film will be shorter "
              f"than the target.", flush=True)
    print(f"[draft] logline: {proposal.logline}", flush=True)
    if not allowed:
        print(f"[draft] REFUSED: the story check found {len(breaks)} blocking "
              f"problem(s), so this production was NOT changed.", flush=True)
        for b in breaks:
            print(f"[draft]   - {b}", flush=True)
        for w in warns:
            print(f"[draft]   warning: {w}", flush=True)
        try:
            pending = save_pending_draft(a.store, a.concept, proposal, a.n_shots)
            print(f"[draft] the refused draft is kept beside the production "
                  f"({pending.name}) so the next run is fed it back -- the "
                  f"production itself is unchanged", flush=True)
        except Exception as exc:  # noqa: BLE001 -- never lose the refusal over this
            print(f"[draft] note: could not keep the refused draft for the next "
                  f"run ({type(exc).__name__}: {str(exc)[:100]})", flush=True)
        print("[draft] options: press 'write new ideas' to CONTINUE from this draft "
              "(the loop feeds it back and repairs its exact failures), or tick "
              "'adopt a broken story' to keep this list anyway.", flush=True)
        return 2
    if warns:
        print(f"[draft] adopting with {len(warns)} warning(s) -- check them "
              f"by eye before shooting", flush=True)
    if breaks:
        print(f"[draft] adopting a broken story on request: {len(breaks)} "
              f"blocking problem(s) ignored", flush=True)

    # AUTO-DIRECT THE ORGANISATION. Not a mode picker: the film is partitioned
    # into movements and each one gets its own grammar -- hold a place/figure for
    # 2-4 shots, or change fast for a montage run. A 12-shot piece can come back
    # as hold -> montage -> hybrid -> hold, which is the point.
    #
    # CAMERAS: unless the caller explicitly overrides with --cameras (or a
    # project-stored camera_mode), generation is STATIC and the movement plan's
    # one motivated move per hold is a FINISHING spec. A generated camera LoRA
    # cannot settle; the finishing move can (camera_path.apply_path).
    _st = project.get("settings") or {}
    explicit_cameras = a.cameras or _st.get("camera_mode")
    auto_camera = explicit_cameras is None
    direction = None
    try:
        import sequence_direction as sd
        direction = sd.plan(shots, frames=int(_st.get("frames") or 16),
                            fps=int(_st.get("fps") or 16))
        arranged = sd.apply(shots, direction, camera_finish=auto_camera)
        proposal.shots = arranged          # what gets adopted and rendered
        shots = arranged
        print("[draft] auto direction (hold / change, movement by movement):",
              flush=True)
        for line in sd.describe(direction).splitlines():
            print("[draft]" + (" " + line if not line.startswith("  ") else line),
                  flush=True)
    except Exception as exc:  # noqa: BLE001 -- direction must not kill a draft
        print(f"[draft] auto direction unavailable ({type(exc).__name__}: "
              f"{str(exc)[:100]}); keeping the writer's arrangement", flush=True)

    cameras = None
    if auto_camera:
        # all shots generate still; the planned moves are finishing specs
        cameras = ["none"] * len(shots)
        planned_cams = [m.camera_plan for m in (direction.movements if direction
                                                else []) if m.camera_plan]
        if planned_cams:
            print("[draft] cameras (auto, procedural finish): "
                  + "; ".join(f"shot {p['shot'] + 1} {p['move']} "
                              f"{p['amount']}px settle {p['settle']}"
                              for p in planned_cams), flush=True)
        else:
            print("[draft] cameras (auto): no movement earns a move; every shot "
                  "still and settles", flush=True)
        mode = "auto"
    else:
        # DIRECT THE CUT (explicit override): the old per-shot camera decision.
        from camera_direction import DEFAULT_MODE, MODES, decide

        mode = explicit_cameras or DEFAULT_MODE
        if mode not in MODES:
            mode = DEFAULT_MODE
        if mode == "off":
            print("[draft] cameras (off): every shot still", flush=True)
        else:
            try:
                decided, dir_info = decide(shots, mode=mode, verbose=True)
            except Exception as exc:  # noqa: BLE001
                from camera_direction import decide as decide_again
                print(f"[draft] camera direction failed "
                      f"({type(exc).__name__}: {str(exc)[:120]}); using the "
                      f"writer's suggestions", flush=True)
                decided, dir_info = decide_again(shots, mode="suggestions")
            if dir_info.get("fallback"):
                print(f"[draft] director fallback: {dir_info['fallback']}",
                      flush=True)
            cameras = list(decided)
            if direction is not None:
                import sequence_direction as sd_scope
                scoped = sd_scope.scope_cameras(cameras, direction)
                if scoped != cameras:
                    print("[draft] camera moves scoped to the movement plan "
                          "(montage runs still, holds settle)", flush=True)
                cameras = scoped
            movers = sum(1 for c in cameras if c != "none")
            print(f"[draft] cameras ({mode}): {movers} moving shot(s) "
                  f"-> {', '.join(cameras)}", flush=True)

    # ADOPT. The old shot list is replaced and its takes are KEPT -- moved to
    # takes/_archive/<stamp>/<id>/, never deleted (10-minute renders). They cannot
    # stay under takes/<shot_id>/: the new shots reuse s01..sN, so the new shot 1
    # would inherit the old film's takes (measured: best_take showed the previous
    # film's frames and the next take was named t02).
    before = list(project.get("timeline") or [])
    orphans = []
    for sid in before:
        takes = fs.list_takes(project, sid)
        if takes:
            orphans.append((sid, len(takes)))
    try:
        archived = fs.archive_takes(project, before)
    except OSError as exc:
        print(f"[draft] REFUSED: could not move the old takes out of the way "
              f"({exc}). Close anything that has them open (a preview, the "
              f"studio's viewer) and draft again; nothing was changed.", flush=True)
        return 1
    for sid in before:
        fs.remove_shot(project, sid, delete_files=False)
    made = fs.shots_from_proposal(project, proposal, cameras=cameras)
    if a.character:
        project["character"] = {"descriptor": a.character.strip(),
                                "avoid": a.avoid.strip()}
    if project.get("character"):
        print(f"[draft] character '{project['character']['descriptor']}' applied "
              f"to {fs.apply_character(project)} shot prompt(s)", flush=True)
    if direction is not None:
        project["direction"] = direction.to_dict()
    fs.log_history(project, act="draft", concept=a.concept, shots=len(made),
                   model=getattr(proposal, "model", ""), cached=bool(
                       getattr(proposal, "cached", False)),
                   cameras=mode, movers=sum(
                       1 for s in made
                       if bool((s.get("camera_move") or {}).get("move"))
                       or s.get("camera") not in (None, "", "none")))
    fs.save(project)
    # the draft was adopted, so there is no refusal for the next run to repair
    clear_pending_draft(a.store)

    for sid, n in orphans:
        print(f"[draft] note: {sid} had {n} take(s), archived under "
              f"{archived}", flush=True)
    print(f"[draft] wrote {len(made)} shots into {a.store}: "
          f"{[s['id'] for s in made]}", flush=True)
    pace = project.get("pacing") or {}
    if pace.get("changes"):
        print(f"[draft] cut auto-paced to the renderer: "
              f"{pace.get('natural_s')}s of generated motion per shot, "
              f"{len(pace['changes'])} timing/framing repair(s)", flush=True)
    print(f"[draft] cameras stored: {pace.get('movers', 0)} moving shot(s) "
          f"of {len(made)}", flush=True)
    for i, s in enumerate(made, 1):
        print(f"[draft]   {i}. [{s.get('beat','?'):<9}] [{s.get('purpose','?'):<9}] "
              f"[{s.get('link','?'):<11}] {s.get('scale','?'):<8} "
              f"{s.get('prompt','')[:60]}", flush=True)
    print(f"[draft] the shot list is now in the production; nothing has been "
          f"rendered", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
