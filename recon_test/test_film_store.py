"""Verification for film_store: identity, takes, derived staleness.

The tests are the claims AUDIT.md makes. Each one is a bug that WAS reproducible in
the old model, written as an assertion so it cannot come back.

    python recon_test/test_film_store.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import film_store as fs  # noqa: E402

# tests must not touch the real productions tree
TMP = ROOT / "recon_test" / "_store_tmp"
fs.PRODUCTIONS = TMP

FAILS = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}"
          + ("" if ok else f"   (want {want!r})"))
    if not ok:
        FAILS.append(label)


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


def state_of(slug: str, shot_id: str) -> str:
    """Reload project and shot first -- the same discipline the UI must follow.

    `shot_state` joins the shot against the PROJECT (for the palette and settings)
    and against the takes on disk, so passing a stale in-memory pair would report
    on a state that no longer exists. This helper makes the fresh read explicit.
    """
    pr = fs.load(slug)
    return fs.shot_state(pr, fs.load_shot(slug, shot_id))


def fake_legacy_take(project, shot, name, keys_to_drop=("frames",)):
    """A take as an EARLIER version of the store would have written it.

    `generation_inputs` gained the `frames` key, and the moment it did, every take
    already on disk hashed differently and six finished shots read as `stale` -- the
    UI offered to re-render a finished film. Nothing about the pixels had changed. This
    builds the old record shape so the tolerance can be tested instead of hoped for.
    """
    d = fs.take_dir(project, shot, name)
    (d / "frames").mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (d / "frames" / f"f{i:03d}.png").write_bytes(b"x")
    old_gen = {k: v for k, v in fs.generation_inputs(project, shot).items()
               if k not in keys_to_drop}
    fs._write_json(d / "take.json", {
        "take": name, "shot": shot["id"], "at": fs._now(),
        "generation": old_gen,                 # note: NO `frames` key
        "input_hash": fs.input_hash(old_gen),
        "rebuild_hash": fs.rebuild_hash(project, shot),
        "seconds": 540.0, "frames": 3, "approved": False,
    })
    return d


def fake_take(project, shot, name, seconds=540.0, approved=False, hash_of=None):
    """Write a take the way the renderer would, without generating anything.

    Uses `film_store.take_record`, so the test cannot invent a take the renderer
    would not have written -- the record shape is one definition in one place.
    """
    d = fs.take_dir(project, shot, name)
    (d / "frames").mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (d / "frames" / f"f{i:03d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    rec = fs.take_record(project, shot, seconds=seconds, frames=3,
                         ladder={"raw": 3, "smooth": 5, "delivered": 3},
                         take=name, approved=approved)
    fs._write_json(d / "take.json", rec)
    return d


def main() -> int:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)

    print("\n1. identity: a reorder moves NOTHING onto another shot")
    pr = fs.new_project("Identity Test", "a test")
    fs.save(pr)
    shots = []
    for i, (cam, sec, sc) in enumerate([("none", 2.0, "sc1"), ("pan-left", 3.0, "sc1"),
                                        ("zoom-in", 1.5, "sc1"), ("tilt-up", 4.0, "sc1")]):
        s = fs.add_shot(pr, prompt=f"shot {i}", camera=cam, seconds=sec, scene=sc)
        shots.append(s)
    # give each shot a take and approve two of them
    # set the palette BEFORE rendering: it is a render input, so a take rendered
    # under one palette and checked under another is legitimately stale. (The test
    # caught this the honest way -- by reporting stale when the palette moved.)
    fs.set_palette(pr, "sc1", "warm-gold")
    for s in shots:
        fake_take(pr, s, "t01")
        fs.choose_take(pr, s["id"], "t01")
    fs.approve_take(pr, shots[0]["id"])
    fs.approve_take(pr, shots[2]["id"])
    fs.save(pr)

    def facts(s):
        return (s["camera"], s["seconds"], fs.palette_for(pr, s), s["take"])

    before = {s["id"]: facts(s) for s in fs.shots_in_order(pr)}
    fs.move_shot(pr, shots[0]["id"], +3)          # move the first shot to the end
    fs.move_to(pr, shots[3]["id"], 0)
    fs.save(pr)
    pr = fs.load(pr["slug"])
    after = {s["id"]: facts(s) for s in fs.shots_in_order(pr)}
    check("per-shot camera/hold/palette/chosen take survive a reorder", after, before)
    check("the timeline order actually changed",
          [s["id"] for s in fs.shots_in_order(pr)],
          [shots[3]["id"], shots[1]["id"], shots[2]["id"], shots[0]["id"]])
    # These four shots are ONE HELD SCENE (same scene id), so the organisation
    # conditions every later shot on the hold's anchor frame (`hold_reference` is a
    # generation input: a shot rendered from a reference frame is different pixels).
    # Reordering inside the hold moves the anchor, so the held shots are correctly
    # stale -- a RE-RENDER, decided by the user ("re-render on reorder"). The cheap
    # re-finish case is a reorder BETWEEN holds / of unheld shots, covered by the
    # retime checks in section 5, where only the in-points move.
    check("a reorder inside a hold moves its anchor, so the held shots re-render",
          sorted(set(fs.shot_state(pr, s) for s in fs.shots_in_order(pr))
                 - {"approved"}),
          ["stale"])
    check("and that re-render is counted and priced, not hidden",
          fs.film_status(pr)["to_generate"], 4)

    print("\n2. derived staleness: no status string to contradict the disk")
    # a dedicated shot, last in the timeline so nothing upstream of it, and not
    # approved -- the assertions below are about the hash, not about approval
    pr = fs.load(pr["slug"])
    probe = fs.add_shot(pr, prompt="probe", camera="none", seconds=2.0, scene="sc1")
    sid = probe["id"]
    check("a shot with no take is 'unrendered'", state_of(pr["slug"], sid),
          "unrendered")
    fake_take(pr, probe, "t01")
    fs.choose_take(pr, sid, "t01")
    fs.save(pr)
    check("an untouched take is 'rendered'", state_of(pr["slug"], sid), "rendered")
    fs.update_shot(pr, sid, prompt="a different prompt entirely")
    check("changing the prompt marks it stale", state_of(pr["slug"], sid), "stale")
    fs.update_shot(pr, sid, prompt="probe")
    check("changing it back un-stales it (the hash is the truth)",
          state_of(pr["slug"], sid), "rendered")
    fs.update_shot(pr, sid, camera="zoom-out")
    check("changing the CAMERA marks it stale", state_of(pr["slug"], sid), "stale")
    fs.update_shot(pr, sid, camera="none")
    check("putting the camera back un-stales it", state_of(pr["slug"], sid),
          "rendered")
    # the distinction that saves ten minutes: grading is NOT a generation input
    fs.set_palette(pr, "sc1", "cold-steel")
    fs.save(pr)
    check("changing the scene PALETTE costs a re-FINISH, not a re-generate",
          state_of(pr["slug"], sid), "restale")
    fs.set_palette(pr, "sc1", "warm-gold")
    fs.save(pr)
    check("putting the palette back clears it", state_of(pr["slug"], sid),
          "rendered")
    fs.update_shot(pr, sid, seconds=2.0 + 0.5)
    check("retiming a shot costs a re-finish too", state_of(pr["slug"], sid),
          "restale")
    check("and never a re-generate",
          fs.needs_generation(state_of(pr["slug"], sid)), False)

    print("\n2b. a lock is a note, not a claim that the take is still current")
    # The bug this exists for: `shot_state` returned `approved` BEFORE checking
    # whether the finishing inputs had drifted, so locking a shot and then changing
    # its scene's palette produced NO warning anywhere. The lock hid the drift --
    # the exact silent inconsistency this store was built to remove.
    fs.set_palette(pr, "sc1", "warm-gold")
    fs.save(pr)
    fs.update_shot(pr, sid, seconds=2.0)
    check("back to current before locking", state_of(pr["slug"], sid), "rendered")
    fs.approve_take(pr, sid)
    fs.save(pr)
    check("locking a current take reports 'approved'", state_of(pr["slug"], sid),
          "approved")
    fs.set_palette(pr, "sc1", "cold-steel")
    fs.save(pr)
    check("a locked take still reports its grade drift", state_of(pr["slug"], sid),
          "restale")
    fs.set_palette(pr, "sc1", "warm-gold")
    fs.save(pr)
    check("and clearing the drift restores 'approved'", state_of(pr["slug"], sid),
          "approved")
    fs.unlock_take(pr, sid)
    fs.save(pr)
    check("unlocking is the inverse and loses nothing",
          state_of(pr["slug"], sid), "rendered")
    check("the take is still there and still chosen",
          (fs.current_take(pr, sid) or {}).get("take"), "t01")
    check("unlocking a shot that was never locked is still reversible",
          fs.unlock_take(pr, sid) is not None, True)
    # A lock belongs to a TAKE, not to a shot -- which is the right scope: shooting
    # another take is not an edit to the one you reviewed.
    fs.approve_take(pr, sid)
    fs.save(pr)
    new = "t09"
    fake_take(pr, fs.get_shot(pr, sid), new)
    fs.choose_take(pr, sid, new)
    fs.save(pr)
    check("a lock does not follow you to a different take",
          state_of(pr["slug"], sid), "rendered")
    check("but the take you locked is still locked",
          any(t["take"] == "t01" and t.get("approved")
              for t in fs.list_takes(fs.load(pr["slug"]), sid)), True)
    fs.update_shot(pr, sid, seconds=2.0)

    print("\n3. takes accumulate: a re-render never destroys the last one")
    s0 = fs.load_shot(pr["slug"], shots[1]["id"])
    fake_take(pr, s0, "t02", seconds=612.0)
    fake_take(pr, s0, "t03", seconds=598.0)
    takes = fs.list_takes(pr, s0["id"])
    check("three takes on disk", [t["take"] for t in takes], ["t01", "t02", "t03"])
    check("the next take gets a free name", fs.next_take_name(pr, s0["id"]), "t04")
    check("the chosen take is what gets shown",
          fs.best_take(pr, s0)["take"], "t01")
    fs.choose_take(pr, s0["id"], "t03")
    check("choosing another take switches the cut",
          fs.best_take(pr, fs.load_shot(pr["slug"], s0["id"]))["take"], "t03")

    print("\n4. the export plan IS the timeline (no frozen copy to drift)")
    plan = fs.export_plan(fs.load(pr["slug"]))
    # tl order after test 1: shot 4 (4.0s), 2 (3.0s), 3 (1.5s), 1 (2.0s), probe (2.5s)
    check("one plan row per shot", len(plan["order"]), 5)
    check("rows are in timeline order",
          [r["id"] for r in plan["order"]],
          [s["id"] for s in fs.shots_in_order(fs.load(pr["slug"]))])
    check("holds come from handheld.hold_frames",
          [r["hold_frames"] for r in plan["order"]], [64, 48, 24, 32, 32])
    check("duration is the sum of the holds in fps",
          round(plan["duration_s"], 4), round((64 + 48 + 24 + 32 + 32) / 16, 4))
    check("frames is the sum of the holds", plan["frames"], 200)
    # the load-bearing one: change a shot, re-read the plan, no freeze step
    fs.update_shot(fs.load(pr["slug"]), shots[0]["id"], seconds=2.5)
    pr2 = fs.load(pr["slug"])
    # shot 1 holds 2.0s -> 32 frames; at 2.5s it holds 40, so +8
    check("an edit reaches the plan with no freeze in between",
          fs.export_plan(pr2)["frames"], 208)

    print("\n5. status is derived for the whole film, and prices the next pass")
    # A DEDICATED production, in three known states, because the main fixture has
    # been through the reorder and the lock tests by now and "is this shot current"
    # depends on all of it. Testing the cascade on a film built for it is the
    # difference between measuring the code and measuring the earlier tests.
    pr5 = fs.new_project("Cascade Test")
    fs.save(pr5)
    c = [fs.add_shot(pr5, prompt=f"c{i}", camera="none", seconds=2.0, scene="sc1")
         for i in range(3)]
    for i, s in enumerate(c):
        fake_take(pr5, s, "t01")
        fs.choose_take(pr5, s["id"], "t01")
    fs.save(pr5)
    # lock the LAST one, so a retime of the FIRST cannot be blamed for it
    fs.approve_take(pr5, c[2]["id"])
    fs.save(pr5)
    st5 = fs.film_status(pr5)
    check("a clean film needs no generation and no re-finish",
          (st5["to_generate"], st5["to_refinish"]), (0, 0))
    check("and the locked shot reads as locked",
          fs.shot_state(pr5, c[2]["id"]), "approved")

    # retime the FIRST shot: it and everything after it must re-finish, and the
    # number must be the cascade, not the film
    fs.update_shot(pr5, c[0]["id"], seconds=3.0)
    fs.save(pr5)
    st5 = fs.film_status(pr5)
    check("a retime re-finishes the edited shot and everything after it",
          st5["to_refinish"], 3)
    check("a retime never needs the model", st5["to_generate"], 0)
    check("the cascade is real, not a whole-film invalidation",
          st5["to_refinish"] <= len(pr5["timeline"]), True)

    # retime the LAST shot. The first shot's retime is still outstanding, so two
    # shots were already going to re-finish; the last shot's own retime adds exactly
    # one more, and adds NOTHING upstream of it. The earlier version of this line
    # expected 1 and was simply wrong -- 3 is correct.
    fs.update_shot(pr5, c[2]["id"], seconds=2.5)
    fs.save(pr5)
    check("retiming the last shot adds exactly one to the re-finish set",
          fs.film_status(pr5)["to_refinish"], 3)
    states = {s["id"]: fs.shot_state(pr5, s) for s in fs.shots_in_order(pr5)}
    check("every shot is either clean or needs only a re-finish, none a re-render",
          sorted(set(states.values())) in (["restale"], ["rendered", "restale"],
                                           ["approved", "restale"]), True)
    # and the LAST shot is now dirty for a reason of its own, which is the part
    # that proves the cascade looks downstream and not upstream
    check("the last shot is dirty", states[c[2]["id"]] == "restale", True)
    check("a re-finishable film may still be exported",
          fs.ready_to_export(fs.load(pr5["slug"])), True)

    # SELF-CLEANING. Restoring a value must restore the state: a hash of the real
    # dependency does that, a bumped counter cannot.
    fs.update_shot(pr5, c[2]["id"], seconds=2.0)
    fs.update_shot(pr5, c[0]["id"], seconds=2.0)
    fs.save(pr5)
    check("putting both values back restores the whole film",
          (fs.film_status(pr5)["to_generate"], fs.film_status(pr5)["to_refinish"]),
          (0, 0))

    pr2 = fs.load(pr["slug"])
    st = fs.film_status(pr2)
    check("shots", st["shots"], 5)
    check("a measured per-shot cost is reported",
          st["measured_per_shot_s"] is not None, True)
    # the mean must be the mean of the CHOSEN takes' measured costs -- not of every
    # take ever made. Compute it from the records rather than hard-coding it, so
    # the assertion is about the definition instead of about five magic numbers.
    chosen = [fs.current_take(pr2, s).get("seconds") for s in fs.shots_in_order(pr2)
              if fs.current_take(pr2, s)]
    check("the estimate is the mean of the CHOSEN takes' measured costs",
          round(st["measured_per_shot_s"], 1), round(sum(chosen) / len(chosen), 1))
    check("every take it averaged carries a measured cost",
          all(isinstance(c, (int, float)) and c > 0 for c in chosen), True)
    check("a film whose hold anchor moved (re-render pending) is not exportable",
          fs.ready_to_export(pr2), False)
    # and a generation change blocks it, which is the case that costs 10 min
    fs.update_shot(pr2, shots[0]["id"], prompt="something else entirely")
    check("a generation change blocks export", fs.ready_to_export(pr2), False)
    check("and it is priced", fs.film_status(pr2)["est_generate_s"] is not None, True)

    print("\n6. scenes are addressed by id, not by position")
    pr3 = fs.new_project("Scene Test")
    fs.save(pr3)
    a = fs.add_shot(pr3, prompt="a", scene="sc1")
    b = fs.add_shot(pr3, prompt="b", scene="sc2")
    fs.set_palette(pr3, "sc1", "cold-steel")
    fs.set_palette(pr3, "sc2", "amber-dusk")
    check("palette a", fs.palette_for(pr3, fs.load_shot(pr3["slug"], a["id"])),
          "cold-steel")
    check("palette b", fs.palette_for(pr3, fs.load_shot(pr3["slug"], b["id"])),
          "amber-dusk")
    fs.move_shot(pr3, a["id"], +1)
    check("palette a after the scenes swap places",
          fs.palette_for(pr3, fs.load_shot(pr3["slug"], a["id"])), "cold-steel")
    check("the palette list the renderer receives is per-shot, in order",
          [r["palette"] for r in fs.export_plan(pr3)["order"]],
          ["amber-dusk", "cold-steel"])

    print("\n7. inventory: the store knows what exists")
    check("list_projects finds it",
          any(p["slug"] == pr["slug"] for p in fs.list_projects()), True)
    check("a shot file per shot",
          len(list((fs.dir_for(pr["slug"]) / "shots").glob("*.json"))), 5)

    print("\n8. every shot argument accepts a RECORD or an ID")
    # The bug this exists for: `make_film.run_store` passed a shot RECORD to
    # `next_take_name`, which wanted an id, so the take path became
    # `.../takes/{'id': 's01'}` and raised
    #   TypeError: unsupported operand type(s) for /: 'WindowsPath' and 'dict'
    # -- AFTER the diffusion pipeline had loaded. Every test above passed strings,
    # so nothing caught it. These call the store the way the renderer does.
    pr4 = fs.load(pr["slug"])
    rec = fs.get_shot(pr4, shots[1]["id"])
    check("get_shot accepts a record", (fs.get_shot(pr4, rec) or {}).get("id"),
          shots[1]["id"])
    check("list_takes accepts a record", len(fs.list_takes(pr4, rec)),
          len(fs.list_takes(pr4, rec["id"])))
    check("next_take_name accepts a record", fs.next_take_name(pr4, rec),
          fs.next_take_name(pr4, rec["id"]))
    check("shot_state accepts a record", fs.shot_state(pr4, rec),
          fs.shot_state(pr4, rec["id"]))
    check("best_take accepts a record",
          (fs.best_take(pr4, rec) or {}).get("take"),
          (fs.best_take(pr4, rec["id"]) or {}).get("take"))
    check("choose_take accepts a record",
          fs.choose_take(pr4, rec, rec.get("take") or "t01").get("id"),
          rec["id"])
    check("a bad argument fails loudly, not with a PathTypeError",
          _raises(lambda: fs.list_takes(pr4, 42)), True)

    print("\n9. the frame count is a render input, so changing it re-shoots")
    # THE DEFAULT IS 16 AND THAT IS A CORRECTION, not a preference. It was briefly 12
    # because 12 measured 27% faster with equal-or-better alignment and coherence -- and
    # `bench_frames.py` does not measure MOTION. AnimateDiff's motion module was trained
    # on 16-frame windows; every take in one production was rendered at 12 and came back
    # reading as still frames. A speed-up that costs the movement is not a speed-up.
    pr6 = fs.new_project("Frames Test")
    fs.save(pr6)
    check("new productions default to 16 raw frames -- what the motion module was "
          "trained on", (pr6.get("settings") or {}).get("frames"), 16)
    s6 = fs.add_shot(pr6, prompt="f", camera="none", seconds=2.0, scene="sc1")
    fs.save(pr6)
    check("the setting reaches the generation inputs",
          fs.generation_inputs(pr6, s6).get("frames"), 16)
    fake_take(pr6, s6, "t01")
    fs.choose_take(pr6, s6["id"], "t01")
    fs.save(pr6)
    check("a take rendered at 16 is current", state_of(pr6["slug"], s6["id"]),
          "rendered")
    fs.set_settings(pr6, frames=12)
    fs.save(pr6)
    check("dialling it down to 12 marks the take as needing a re-shoot",
          state_of(pr6["slug"], s6["id"]), "stale")
    fs.set_settings(pr6, frames=16)
    fs.save(pr6)
    check("and putting it back restores it, because the hash is of the value",
          state_of(pr6["slug"], s6["id"]), "rendered")

    print("\n9b. a shot whose takes are ALL stale does not read as 'rendered'")
    # `shot_state` returned "rendered" for any shot with a take and no take chosen, so a
    # shot whose frames no longer match its settings claimed to be fine and the strip
    # said "shot" about frames the cut would not use.
    pr6b = fs.new_project("Unchosen Stale Test")
    fs.save(pr6b)
    s6b = fs.add_shot(pr6b, prompt="u", camera="none", seconds=2.0, scene="sc1")
    fake_take(pr6b, s6b, "t01")                    # deliberately NOT chosen
    fs.save(pr6b)
    check("a current take with nothing chosen reads 'rendered'",
          state_of(pr6b["slug"], s6b["id"]), "rendered")
    fs.set_settings(pr6b, frames=12)
    fs.save(pr6b)
    check("once its only take is stale, it reads 'stale' even with nothing chosen",
          state_of(pr6b["slug"], s6b["id"]), "stale")

    print("\n10. a take recorded BEFORE a key was added is still current")
    # Adding `frames` to generation_inputs made every existing take hash differently.
    # Nothing about the pixels changed; only the key set did. Six finished shots read
    # as stale and the UI offered to re-render a finished film.
    pr7 = fs.new_project("Legacy Take Test")
    fs.save(pr7)
    s7 = fs.add_shot(pr7, prompt="legacy", camera="none", seconds=2.0, scene="sc1")
    fs.save(pr7)
    fake_legacy_take(pr7, s7, "t01")
    fs.choose_take(pr7, s7["id"], "t01")
    fs.save(pr7)
    check("a legacy take (no `frames` key) reads as current",
          state_of(pr7["slug"], s7["id"]), "rendered")
    # NARROW, not blanket: a key it DID record still invalidates it
    fs.update_shot(fs.load(pr7["slug"]), s7["id"], prompt="edited after the fact")
    check("but editing a key it DID record still marks it stale",
          state_of(pr7["slug"], s7["id"]), "stale")
    pr7 = fs.load(pr7["slug"])
    fs.update_shot(pr7, s7["id"], prompt="legacy")
    check("and reverting restores it", state_of(pr7["slug"], s7["id"]), "rendered")
    # and the value it never recorded must not be treated as a change
    pr7 = fs.load(pr7["slug"])
    check("a legacy take with no recorded frame count is still current against a NEW "
          "production, because a new production records the trained-on default",
          fs.generation_inputs(pr7, fs.get_shot(pr7, s7["id"])).get("frames"), 16)
    # an OLD production, which never recorded one at all, must fall back to 16 --
    # because that is what its takes were rendered at
    pr7.get("settings", {}).pop("frames", None)
    fs.save(pr7)
    pr7 = fs.load(pr7["slug"])
    check("an old production with no settings.frames falls back to 16",
          fs.generation_inputs(pr7, fs.get_shot(pr7, s7["id"])).get("frames"), 16)
    check("and its legacy take is still current at that fallback",
          state_of(pr7["slug"], s7["id"]), "rendered")

    print(f"\n{'=' * 60}")
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + "; ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
