"""Sampler tiers: which shots render FAST (Lightning) vs FULL, and what that does
to take staleness. The shots below are the real skydiving shot shapes."""
import film_store as fs
import make_film


def shot(prompt, scale="medium", subject="skydiver", **kw):
    s = fs.new_shot("s01", prompt, scale=scale, subject=subject)
    s.update(kw)
    return s


def project():
    return {"slug": "t", "shots": {}, "timeline": [], "seed": 7, "model": "dreamshaper8"}


def test_scale_decides():
    for scale, want in (("macro", "full"), ("close", "full"), ("medium", "full"),
                        ("wide", "fast"), ("extreme-wide", "fast")):
        assert fs.render_tier(shot("a skydiver in an orange jumpsuit, arms spread",
                                   scale=scale)) == want, scale


def test_prompt_text_does_not_matter():
    assert fs.render_tier(shot("green field", scale="wide")) == \
        fs.render_tier(shot("a skydiver, jumpsuit, helmet, limbs", scale="wide"))


def test_override_wins_both_ways():
    assert fs.render_tier(shot("green field", scale="wide", quality="full")) == "full"
    assert fs.render_tier(shot("a skydiver, arms out", scale="close",
                               quality="fast")) == "fast"


def test_tag_carries_steps(monkeypatch):
    s = shot("a skydiver, arms out", scale="close")
    assert fs.render_tier_tag(s) == f"full{fs.FULL_STEPS}"
    assert fs.render_tier_tag(shot("green field", scale="wide")) == "fast"
    monkeypatch.setattr(fs, "FULL_STEPS", 8)
    assert fs.render_tier_tag(s) == "full8"


def test_legacy_take_stays_current_for_fast_and_goes_stale_for_full():
    p = project()
    fast = shot("green field at dawn", scale="wide", seed=3)
    full = shot("a skydiver in a jumpsuit, arms out", scale="close", seed=3)
    for s, current in ((fast, True), (full, False)):
        legacy = dict(fs.generation_inputs(p, s))
        legacy.pop("sampler")                      # a take recorded before tiers
        stored = fs.input_hash(legacy)
        got = fs.generation_hash(p, s, stored_generation=legacy, stored_hash=stored)
        assert (got == stored) == current, s["prompt"]


def test_sampling_plan_fast_vs_full():
    f, u = make_film.sampling_plan("fast"), make_film.sampling_plan("full")
    assert f["lightning"] and f["cfg"] == 1.0 and f["steps"] == make_film.STEPS
    assert not u["lightning"] and u["cfg"] > 1.0 and u["steps"] == fs.FULL_STEPS
    assert "extra limbs" in u["neg"]
    assert make_film.sampling_plan("bogus")["tier"] == "fast"

def test_keyframe_makes_shot_full_and_is_an_input(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "PRODUCTIONS", tmp_path)
    p = project()
    p["slug"] = "kf"
    d = fs.dir_for("kf") / "keyframes" / "s01"
    d.mkdir(parents=True)
    (d / "c0.png").write_bytes(b"one")
    s = shot("green field", scale="wide", seed=3)
    assert fs.render_tier(s) == "fast"
    base = fs.generation_inputs(p, s)
    assert base["keyframe"] is None
    s["keyframe"] = "keyframes/s01/c0.png"
    assert fs.render_tier(s) == "full"
    a = fs.generation_inputs(p, s)
    assert a["keyframe"] and a["keyframe"] != base["keyframe"]
    (d / "c0.png").write_bytes(b"two")             # a different approved image
    assert fs.generation_inputs(p, s)["keyframe"] != a["keyframe"]
    (d / "c0.png").unlink()                         # approved but deleted
    assert fs.generation_inputs(p, s)["keyframe"] == "missing"


def test_legacy_take_without_keyframe_key_stays_current_only_without_keyframe(
        tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "PRODUCTIONS", tmp_path)
    p = project()
    p["slug"] = "kf"
    d = fs.dir_for("kf") / "keyframes" / "s01"
    d.mkdir(parents=True)
    (d / "c0.png").write_bytes(b"x")
    plain = shot("green field", scale="wide", seed=3)
    legacy = dict(fs.generation_inputs(p, plain))
    legacy.pop("keyframe")
    stored = fs.input_hash(legacy)
    assert fs.generation_hash(p, plain, stored_generation=legacy,
                              stored_hash=stored) == stored
    framed = dict(plain, keyframe="keyframes/s01/c0.png")
    legacy2 = dict(legacy, sampler=fs.render_tier_tag(framed))
    assert fs.generation_hash(p, framed, stored_generation=legacy2,
                              stored_hash=fs.input_hash(legacy2)) != \
        fs.input_hash(legacy2)


def test_character_goes_in_front_of_the_subject_once():
    n = "skydiver"
    assert fs.character_prompt("A skydiver in an orange jumpsuit", n, "male") == \
        "A male skydiver in an orange jumpsuit"
    once = fs.character_prompt("Skydiver's silhouette against the sky", n, "male")
    assert once == "male Skydiver's silhouette against the sky"
    assert fs.character_prompt(once, n, "male") == once          # idempotent


def test_character_leaves_a_prompt_without_the_subject_alone():
    assert fs.character_prompt("green field at dawn", "skydiver", "male") == \
        "green field at dawn"
    assert fs.character_prompt("a skydiver", "", "male") == "a skydiver"
    assert fs.character_prompt("a skydiver", "skydiver", "") == "a skydiver"


def test_apply_character_rewrites_shots_and_marks_takes_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "PRODUCTIONS", tmp_path)
    p = fs.new_project("char", "a skydiver lands")
    a = fs.new_shot("s01", "A skydiver falls through cloud", subject="a skydiver")
    b = fs.new_shot("s02", "green field at dawn", subject="a skydiver")
    for s in (a, b):
        fs.save_shot(p["slug"], s, p)
        p.setdefault("shots", {})[s["id"]] = s
        p["timeline"].append(s["id"])
    before = fs.generation_inputs(p, fs.get_shot(p, "s01"))
    assert before["avoid"] == ""
    n = fs.set_character(p, "male", avoid="woman, female")
    assert n == 1                                    # s02 never names the subject
    assert fs.get_shot(p, "s01")["prompt"] == "A male skydiver falls through cloud"
    assert fs.get_shot(p, "s02")["prompt"] == "green field at dawn"
    after = fs.generation_inputs(p, fs.get_shot(p, "s01"))
    assert after["avoid"] == "woman, female" and after["prompt"] != before["prompt"]
    assert fs.set_character(p, "male", avoid="woman, female") == 0   # re-run: no-op


def test_avoid_is_a_neutral_added_key():
    assert fs.GEN_NEUTRAL["avoid"] == "" and "avoid" in fs.ADDED_GEN_KEYS

import job_runner as jr


def _feed(lines):
    prog = {}
    for ln in lines:
        jr._store_progress(ln, prog)
    return prog


REAL = [
    "=== GENERATING 3 shot(s) ===",
    "  [plan] tiers: full,fast,full",
    "  [base] releasing models/dreamshaper8/fast, loading models/dreamshaper8/full (~1 min)",
    "  1/3 s01 none                 [full]  3000s -> take t02 (16 raw, 31 smooth) +hold ref shot 1",
    "  2/3 s02 none                 [fast]   600s -> take t01 (16 raw, 31 smooth)",
]


def test_progress_parses_the_tier_tagged_line_and_stays_generating():
    # regression: the "[full]" tag broke the per-shot regex and a mid-run model
    # reload reset the phase, so the bar sat on "loading the model" for 5 hours
    prog = _feed(REAL)
    assert prog["phase"] == "generating"
    assert prog["done"] == 2 and prog["total"] == 3
    assert prog["tier_seconds"] == {"full": [3000.0], "fast": [600.0]}


def test_progress_eta_prices_the_remaining_shots_per_tier():
    prog = _feed(REAL[:2])
    assert prog["eta_seconds"] == 2900 + 660 + 2900          # defaults, nothing timed
    prog = _feed(REAL)
    assert prog["eta_shots"] == 1 and prog["eta_seconds"] == 3000.0   # one full left
    assert prog["current_est_s"] == 3000.0


def test_progress_still_parses_an_untagged_legacy_line():
    prog = _feed(["  1/2 s01 none                  677s -> take t01 (16 raw)"])
    assert prog["done"] == 1 and prog["tier_seconds"] == {"fast": [677.0]}


def test_studio_progress_uses_the_tier_eta():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("studio_mod", root / "ui" / "studio.py")
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    prog = _feed(REAL)
    job = {"status": jr.RUNNING, "child_started": 1.0, "progress": prog}
    f = st.progress_facts(job)
    assert f["eta"] == (1, 3000.0)
    assert f["line"] == "generating shot 3 of 3"


def test_estimate_prices_each_tier_from_its_own_takes():
    def row(sid, scale, secs=None, sampler="fast"):
        shot = {"id": sid, "scale": scale, "subject": "", "prompt": "x"}
        take = ({"seconds": secs, "generation": {"sampler": sampler}}
                if secs else None)
        return {"shot": shot, "take": take}
    rows = [row("a", "wide", 600), row("b", "close", 3000, "full12"),
            row("c", "wide"), row("d", "close"), row("e", "medium")]
    need = [rows[2], rows[3], rows[4]]
    total, by = fs.estimate_generate_s(rows, need)
    assert by["fast"] == {"shots": 1, "each_s": 600.0, "measured": True}
    assert by["full"] == {"shots": 2, "each_s": 3000.0, "measured": True}
    assert total == 600 + 2 * 3000
    # nothing timed yet for a tier -> the measured default, flagged as an estimate
    total2, by2 = fs.estimate_generate_s([row("a", "wide", 600)], [rows[3]])
    assert by2["full"]["measured"] is False
    assert total2 == fs.TIER_DEFAULT_S["full"]


def test_take_tier_treats_an_old_take_as_fast():
    assert fs.take_tier({"generation": {}}) == "fast"
    assert fs.take_tier(None) == "fast"
    assert fs.take_tier({"generation": {"sampler": "full12"}}) == "full"


# ---- a chosen take stands: anchor changes are a note, text drift can be adopted ----

import json as _json


def _take_on_disk(p, shot, name="t01", **gen_over):
    d = fs.take_dir(p, shot, name)
    (d / "frames").mkdir(parents=True)
    (d / "frames" / "f000.png").write_bytes(b"pixels")
    gen = dict(fs.generation_inputs(p, shot))
    gen.update(gen_over)
    rec = {"take": name, "shot": shot["id"], "generation": gen,
           "input_hash": fs.input_hash(gen), "finish": fs.finish_inputs(p, shot),
           "rebuild_hash": fs.rebuild_hash(p, shot), "seconds": 60, "frames": 31,
           "approved": False}
    (d / "take.json").write_text(_json.dumps(rec), encoding="utf-8")
    fs.update_shot(p, shot["id"], take=name)
    return d


def _proj(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "PRODUCTIONS", tmp_path)
    p = fs.new_project("adopt", "a diver")
    fs.save(p)
    fs.add_shot(p, prompt="a diver descends", scale="close", subject="diver",
                scene="sc1")
    fs.save(p)
    return p, fs.get_shot(p, p["timeline"][0])


def test_a_changed_anchor_take_does_not_make_a_shot_stale(tmp_path, monkeypatch):
    p, first = _proj(tmp_path, monkeypatch)
    fs.add_shot(p, prompt="the diver keeps falling", scale="close", subject="diver",
                scene="sc1")
    fs.save(p)
    _take_on_disk(p, first)                              # the anchor's chosen take: t01
    second = fs.get_shot(p, p["timeline"][1])
    assert fs.generation_inputs(p, second)["hold_reference"] == f"{first['id']}:t01"
    _take_on_disk(p, second, hold_reference=f"{first['id']}:t05")   # older anchor take
    second = fs.get_shot(p, second["id"])
    assert fs.shot_state(p, second) == "rendered"
    take = fs.best_take(p, second)
    assert fs.take_drift(p, second, take) == []
    note = fs.hold_note(p, second, take)
    assert f"{first['id']}:t05" in note and "Nothing needs re-shooting" in note


def test_text_that_changed_since_the_take_is_still_reported_as_stale(tmp_path,
                                                                     monkeypatch):
    p, shot = _proj(tmp_path, monkeypatch)
    _take_on_disk(p, shot)
    fs.update_shot(p, shot["id"], prompt="a diver stops and looks up", scale="wide")
    shot = fs.get_shot(p, shot["id"])
    assert fs.shot_state(p, shot) == "stale"
    assert fs.take_drift(p, shot, fs.best_take(p, shot)) == ["prompt", "framing", "render quality"]


def test_adopt_keeps_the_pixels_and_the_original_record(tmp_path, monkeypatch):
    p, shot = _proj(tmp_path, monkeypatch)
    d = _take_on_disk(p, shot)
    old_prompt = fs.get_shot(p, shot["id"])["prompt"]
    fs.update_shot(p, shot["id"], prompt="a diver stops and looks up")
    assert fs.shot_state(p, shot["id"]) == "stale"
    fs.adopt_take(p, shot["id"])
    assert fs.shot_state(p, shot["id"]) in ("rendered", "approved")
    assert (d / "frames" / "f000.png").read_bytes() == b"pixels"      # untouched
    rec = _json.loads((d / "take.json").read_text(encoding="utf-8"))
    assert rec["adopted"][0]["generation"]["prompt"] == old_prompt    # what it was
    assert rec["generation"]["prompt"] == "a diver stops and looks up"
    assert fs.film_status(p)["to_generate"] == 0
    # a LATER text edit still makes it stale: adopting is not a permanent waiver
    fs.update_shot(p, shot["id"], prompt="a diver turns away")
    assert fs.shot_state(p, shot["id"]) == "stale"
