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
into structured progress. It writes nothing until the proposal validates, so a failed
draft cannot leave a half-written shot list behind.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch_home"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_home"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="the production slug to write into")
    ap.add_argument("--concept", required=True)
    ap.add_argument("--n-shots", type=int, default=None)
    ap.add_argument("--farms", default="qwen2.5:3b,phi3:latest,mistral:latest")
    ap.add_argument("--showrunner", default="granite4.1:8b")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the proposal cache. Without this, the same concept "
                         "returns the same proposal -- byte for byte")
    ap.add_argument("--temperature", type=float, default=0.8)
    a = ap.parse_args()

    import film_store as fs
    from story import validate, write_shots_ensemble

    project = fs.load(a.store)
    print(f"[draft] production {a.store}: {len(project.get('timeline') or [])} "
          f"shot(s) before", flush=True)
    print(f"[draft] concept: {a.concept}", flush=True)

    proposal = write_shots_ensemble(
        a.concept,
        farms=[m.strip() for m in a.farms.split(",") if m.strip()],
        showrunner=a.showrunner, n_shots=a.n_shots, temperature=a.temperature,
        fresh=a.fresh, verbose=True)

    shots = list(getattr(proposal, "shots", None) or [])
    if not shots:
        print("[draft] the writer returned no usable shots; the production is "
              "unchanged", flush=True)
        return 1

    # a real decision point, reported rather than buried: the writer's own shot list
    # is shown here so the log is readable on its own
    breaks = [p for p in (proposal.problems or []) if p.startswith("story:")]
    warns = [p for p in (proposal.problems or []) if p.startswith("story-link:")]
    print(f"[draft] proposal validated: {len(shots)} shots, {len(breaks)} story "
          f"break(s), {len(warns)} warning(s)", flush=True)
    print(f"[draft] logline: {proposal.logline}", flush=True)

    # ADOPT. The old shot list is replaced, but its takes stay on disk under
    # takes/<shot_id>/ -- `remove_shot(delete_files=False)` is deliberate, because
    # throwing away 10-minute renders to adopt a new script would be the worst
    # possible trade. Orphaned takes are reported so they are not a silent leak.
    before = list(project.get("timeline") or [])
    orphans = []
    for sid in before:
        takes = fs.list_takes(project, sid)
        if takes:
            orphans.append((sid, len(takes)))
        fs.remove_shot(project, sid, delete_files=False)
    made = fs.shots_from_proposal(project, proposal)
    fs.log_history(project, act="draft", concept=a.concept, shots=len(made),
                   model=getattr(proposal, "model", ""), cached=bool(
                       getattr(proposal, "cached", False)))
    fs.save(project)

    for sid, n in orphans:
        print(f"[draft] note: {sid} had {n} take(s) kept on disk under "
              f"productions/{a.store}/takes/", flush=True)
    print(f"[draft] wrote {len(made)} shots into {a.store}: "
          f"{[s['id'] for s in made]}", flush=True)
    for i, s in enumerate(made, 1):
        print(f"[draft]   {i}. [{s.get('beat','?'):<9}] {s.get('scale','?'):<8} "
              f"{s.get('prompt','')[:70]}", flush=True)
    print(f"[draft] the shot list is now in the production; nothing has been "
          f"rendered", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
