"""End-to-end check that the store renderer works, cheaply.

Builds a throwaway production with real PNG frames (no generation), then runs the
ACTUAL renderer (`make_film.py --store`) over it. If this passes, the whole store
path -- plan, takes, finishing, grading, assembly -- runs without a diffusion model.

    python recon_test/test_store_render.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_home"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch_home"))

import film_store as fs  # noqa: E402

SLUG = "_store_render_test"
FRESH = "_store_generate_test"
PARTIAL = "_store_partial_test"
FAILS = []


def _unrendered(slug: str):
    """The 1-based indices that still need the model, computed from the store."""
    import film_store as _fs
    pr = _fs.load(slug)
    return [i for i, s in enumerate(_fs.shots_in_order(pr), 1)
            if _fs.needs_generation(_fs.shot_state(pr, s))]


def check(label, got, want=True):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}"
          + ("" if ok else f"   (want {want!r})"))
    if not ok:
        FAILS.append(label)


def main() -> int:
    print("building a throwaway production with real frames")
    shutil.rmtree(fs.dir_for(SLUG), ignore_errors=True)
    pr = fs.new_project(SLUG, "a store render test")
    pr["slug"] = SLUG
    fs.save(pr)

    from PIL import Image
    fs.set_palette(pr, "sc1", "warm-gold")
    pr["scenes"] = [{"id": "sc1", "name": "scene 1", "palette": "warm-gold"}]

    for i, (cam, secs) in enumerate([("none", 2.0), ("pan-left", 1.5)]):
        shot = fs.add_shot(pr, prompt=f"a test frame number {i}", camera=cam,
                           seconds=secs, scene="sc1", scale="wide")
        take = "t01"
        d = fs.take_dir(pr, shot, take) / "frames"
        d.mkdir(parents=True, exist_ok=True)
        # three real frames, each a different colour so the assemble has something
        for j in range(3):
            Image.new("RGB", (512, 512),
                      (40 + 60 * i + 10 * j, 90, 140 - 20 * j)).save(
                d / f"f{j:03d}.png")
        rec = fs.take_record(pr, shot, seconds=3.0, frames=5,
                             ladder={"raw": 3, "smooth": 5, "delivered": 32},
                             take=take)
        fs._write_json(fs.take_dir(pr, shot, take) / "take.json", rec)
        fs.choose_take(pr, shot["id"], take)
    fs.save(pr)

    st = fs.film_status(fs.load(SLUG))
    print(f"  {st['shots']} shots, {st['frames']} frames, "
          f"{st['duration_s']:.2f}s, to generate {st['to_generate']}, "
          f"to re-finish {st['to_refinish']}")
    assert st["to_generate"] == 0, "nothing should need the model in this test"

    print("\n1. dry run (no generation, no assembly)")
    r = subprocess.run([sys.executable, "make_film.py", "--store", SLUG,
                        "--dry-run"], cwd=ROOT, capture_output=True, text=True)
    print("\n".join(r.stdout.strip().splitlines()[-8:]))
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
        return 1
    if "2 shots" not in r.stdout:
        print("FAIL: the plan did not report 2 shots")
        return 1

    print("\n2. the real renderer, from the takes on disk (no model loaded)")
    r = subprocess.run([sys.executable, "make_film.py", "--store", SLUG,
                        "--name", SLUG], cwd=ROOT, capture_output=True, text=True)
    tail = "\n".join(r.stdout.strip().splitlines()[-14:])
    print(tail)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-3000:])
        return 1

    out = ROOT / "output" / "film" / SLUG
    mp4 = out / "film.mp4"
    stats = out / "stats.json"
    print(f"\n  film.mp4 exists: {mp4.exists()}  "
          f"({mp4.stat().st_size / 1024:.0f} KB)" if mp4.exists() else "  no mp4")
    print(f"  stats.json exists: {stats.exists()}")
    if not mp4.exists():
        print("FAIL: no film.mp4 was written")
        return 1

    print("\n3. the renderer did NOT overwrite the takes it read")
    for shot_id in fs.load(SLUG)["timeline"]:
        takes = fs.list_takes(fs.load(SLUG), shot_id)
        print(f"  {shot_id}: takes {[t['take'] for t in takes]}, "
              f"frames {[t['frames_on_disk'] for t in takes]}")
        if len(takes) != 1:
            print("FAIL: the renderer created or removed a take it should not have")
            return 1

    print("\n4. re-running is a no-op: every shot is current")
    st2 = fs.film_status(fs.load(SLUG))
    print(f"  to generate {st2['to_generate']}, to re-finish {st2['to_refinish']}")
    # the render stamped nothing back onto the shots, so they are still current --
    # the render is a pure function of the store, which is the point
    if st2["to_generate"] != 0:
        print("FAIL: a re-run would regenerate")
        return 1

    print("\n5. the GENERATE path -- the one the crash was on")
    # The bug this exercises: `run_store` passed a shot RECORD to `next_take_name`,
    # which wanted an id, so the take path became `.../takes/{'id': 's01'}`. That
    # raised TypeError only on the path that NEEDS the model -- so it fired after the
    # diffusion pipeline had loaded, throwing away minutes of work, and no test
    # touched it because every test shot already had a take.
    #
    # Stubbing `Generator` and `raft_upsample` is the point, not a shortcut: it makes
    # the expensive branch reachable in a second, so the branch is actually tested.
    shutil.rmtree(fs.dir_for(FRESH), ignore_errors=True)
    pr = fs.new_project(FRESH, "a fresh shot with no take")
    pr["slug"] = FRESH
    fs.save(pr)
    fs.add_shot(pr, prompt="a shot that has never been rendered", camera="zoom-in",
                seconds=2.0, scene="sc1")
    fs.save(pr)
    check("the shot starts unrendered",
          fs.film_status(fs.load(FRESH))["to_generate"], 1)

    import make_film as mf
    from PIL import Image as _Im

    calls = {"generated": 0}

    class FakeGenerator:
        """Stands in for `make_film.Generator`. Never loads a model."""

        def __init__(self, base, loras=None, **kw):
            self.base = base
            self.adapters = []

        def clip(self, shot, seed, model=None, reference=None, **_kw):
            # same signature as make_film.Generator.clip (hold conditioning)
            self.conditioned = reference is not None
            calls["generated"] += 1
            return ([_Im.new("RGB", (64, 64), (30 + 8 * i, 60, 120))
                     for i in range(mf.FRAMES)], 0.01)

    real_gen, real_raft = mf.Generator, mf.raft_upsample
    mf.Generator = FakeGenerator
    mf.raft_upsample = lambda frames, factor=2, flow_size=256: list(frames) * 2
    try:
        import argparse
        a = argparse.Namespace(
            store=FRESH, name=FRESH, base="models/dreamshaper8", only=None,
            dry_run=False, shake=1.0)
        rc = mf.run_store(a)
    finally:
        mf.Generator, mf.raft_upsample = real_gen, real_raft

    check("run_store returned 0", rc, 0)
    check("the model was actually called once", calls["generated"], 1)
    pr = fs.load(FRESH)
    shot = fs.get_shot(pr, pr["timeline"][0])
    takes = fs.list_takes(pr, shot)
    check("a take was written", [t["take"] for t in takes], ["t01"])
    check("its frames are on disk", takes[0]["frames_on_disk"] if takes else 0, 16)
    check("the shot now cuts to it", shot.get("take"), "t01")
    check("and it is current, so a re-run would not regenerate",
          fs.shot_state(pr, shot) in ("rendered", "approved"), True)
    out = ROOT / "output" / "film" / FRESH / "film.mp4"
    check("a film was assembled", out.exists(), True)

    print("\n6. a PARTIAL run stops cleanly instead of dying in the finishing loop")
    # The bug this exists for: `--only 1,3` generated those two shots, then fell
    # through to the finishing loop, which walks the WHOLE timeline and raises on the
    # first shot with no take yet. A user batching two shots while a third was
    # unrendered would pay ten minutes of generation and lose the run -- and that is
    # exactly the shape of the "shoot the next N as ONE job" button in the studio.
    shutil.rmtree(fs.dir_for(PARTIAL), ignore_errors=True)
    pr = fs.new_project(PARTIAL, "three shots, one taken")
    pr["slug"] = PARTIAL
    fs.save(pr)
    made = [fs.add_shot(pr, prompt=f"p{i}", camera="none", seconds=2.0, scene="sc1")
            for i in range(3)]
    # give the MIDDLE shot a take, so the unrendered set is 1 and 3 -- not contiguous
    d = fs.take_dir(pr, made[1], "t01") / "frames"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (d / f"f{i:03d}.png").write_bytes(b"x")
    fs._write_json(fs.take_dir(pr, made[1], "t01") / "take.json",
                   fs.take_record(pr, made[1], seconds=4.0, frames=3, take="t01"))
    fs.choose_take(pr, made[1]["id"], "t01")
    fs.save(pr)
    check("the unrendered set is 1 and 3, not contiguous",
          _unrendered(PARTIAL), [1, 3])

    generated = []
    mf.Generator = FakeGenerator
    mf.raft_upsample = lambda frames, factor=2, flow_size=256: list(frames) * 2
    try:
        a = argparse.Namespace(store=PARTIAL, name=PARTIAL,
                               base="models/dreamshaper8", only="1,3",
                               dry_run=False, shake=1.0)
        rc = mf.run_store(a)
    finally:
        mf.Generator, mf.raft_upsample = real_gen, real_raft
    check("the partial run returned 0 rather than dying", rc, 0)
    pr = fs.load(PARTIAL)
    takes = {s["id"]: fs.list_takes(pr, s) for s in fs.shots_in_order(pr)}
    check("shots 1 and 3 got takes", [len(takes[made[0]["id"]]),
                                      len(takes[made[2]["id"]])], [1, 1])
    check("the shot it was NOT asked to shoot was left alone",
          len(takes[made[1]["id"]]), 1)
    check("no film was assembled from an incomplete set",
          (ROOT / "output" / "film" / PARTIAL / "film.mp4").exists(), False)
    # The three shots are ONE hold (scene sc1). The middle take was recorded while
    # its anchor (shot 1) had no take; shot 1 has now been shot, so the middle shot's
    # hold_reference changed. That is PROVENANCE, not a render input: the user's take
    # stands and nothing needs re-shooting (it used to count as 1 to re-generate).
    check("a changed anchor take does not ask for a re-shoot",
          fs.film_status(pr)["to_generate"], 0)

    print("\n7. `--frames` is stamped on the take it produced, and not persisted")
    # Audit #9. `--frames 12` made the renderer generate 12 frames, but the take was
    # stamped with the production's own setting (16), so the studio called a 12-frame
    # take current for a 16-frame film. The override must reach the plan and the
    # take's hash -- and must NOT be saved into the production's settings.
    OVR = "_store_frames_override_test"
    shutil.rmtree(fs.dir_for(OVR), ignore_errors=True)
    pr = fs.new_project(OVR, "frames override")
    pr["slug"] = OVR
    fs.save(pr)
    fs.add_shot(pr, prompt="a shot rendered at an overridden frame count",
                camera="none", seconds=2.0, scene="sc1")
    fs.save(pr)
    recorded_before = (fs.load(OVR).get("settings") or {}).get("frames")
    mf.Generator = FakeGenerator
    mf.raft_upsample = lambda frames, factor=2, flow_size=256: list(frames) * 2
    real_frames = mf.FRAMES
    try:
        a = argparse.Namespace(store=OVR, name=OVR, base="models/dreamshaper8",
                               only=None, dry_run=False, shake=1.0, frames=12)
        rc = mf.run_store(a)
    finally:
        mf.Generator, mf.raft_upsample = real_gen, real_raft
        mf.FRAMES = real_frames
    check("the override run returned 0", rc, 0)
    pr = fs.load(OVR)
    take = fs.list_takes(pr, fs.shots_in_order(pr)[0])[-1]
    check("the take was generated at 12 raw frames", take["ladder"]["raw"], 12)
    check("and its generation inputs say 12, not the production's setting",
          take["generation"]["frames"], 12)
    check("the one-off override was not persisted into the production",
          (pr.get("settings") or {}).get("frames"), recorded_before)
    shutil.rmtree(fs.dir_for(OVR), ignore_errors=True)
    shutil.rmtree(ROOT / "output" / "film" / OVR, ignore_errors=True)

    print("\nall store-render checks passed" if not FAILS
          else f"\n{len(FAILS)} store-render check(s) FAILED: {FAILS}")
    for s in (SLUG, FRESH, PARTIAL):
        shutil.rmtree(fs.dir_for(s), ignore_errors=True)
        shutil.rmtree(ROOT / "output" / "film" / s, ignore_errors=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
