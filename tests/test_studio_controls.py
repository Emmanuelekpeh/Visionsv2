"""Studio surfaces for the render tier and the film's character (driven through the
real app with AppTest, against a temporary store)."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

import film_store as fs  # noqa: E402

APP = str(ROOT / "ui" / "studio.py")


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "PRODUCTIONS", tmp_path)
    p = fs.new_project("ui-tier", "a skydiver lands")
    fs.save(p)
    fs.add_shot(p, prompt="A skydiver falls through cloud", scale="wide",
                subject="a skydiver", scene="sc1")
    fs.add_shot(p, prompt="green field at dawn", scale="close",
                subject="a skydiver", scene="sc1")
    fs.save(p)
    return p


def run(view, shot=None):
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["slug"] = "ui-tier"
    at.session_state["view"] = view
    if shot:
        at.session_state["shot"] = shot
    at.run()
    return at


def test_shot_shows_its_tier_and_the_override_sticks(store):
    sid = store["timeline"][0]                       # wide -> auto means fast
    at = run("shoot", sid)
    assert not at.exception
    sel = [s for s in at.selectbox if s.label == "render quality"]
    assert sel and sel[0].value == "auto"
    assert any("renders **fast**" in c.value for c in at.caption)
    sel[0].select("full").run()
    assert not at.exception
    assert fs.load_shot("ui-tier", sid)["quality"] == "full"
    assert any("renders **full**" in c.value and "set by you" in c.value
               for c in at.caption)


def test_settings_applies_the_character_to_every_shot(store):
    at = run("settings")
    assert not at.exception
    who = [t for t in at.text_input if t.label == "who"][0]
    avoid = [t for t in at.text_input if t.label == "never draw"][0]
    who.input("male")
    avoid.input("woman, female")
    [b for b in at.button if b.label == "apply to every shot"][0].click().run()
    assert not at.exception
    p = fs.load("ui-tier")
    first = fs.get_shot(p, p["timeline"][0])
    second = fs.get_shot(p, p["timeline"][1])
    assert first["prompt"] == "A male skydiver falls through cloud"
    assert second["prompt"] == "green field at dawn"      # never names the figure
    assert p["character"] == {"descriptor": "male", "avoid": "woman, female"}
    assert any("1 shot prompt(s) rewritten" in i.value for i in at.info)


def test_write_view_has_character_and_selfcheck_controls(store):
    at = run("write")
    assert not at.exception
    assert any(c.label.startswith("let it re-read its own shot list")
               for c in at.checkbox)
    assert [t for t in at.text_input if t.label == "who"]       # inside the expander


def test_shoot_view_opens_a_shot_shorter_than_one_second(store):
    # regression: the length box had min 1.0, so a 0.94 s shot (the auto-pacer
    # writes those) crashed the whole page with StreamlitValueBelowMinError
    sid = store["timeline"][0]
    fs.update_shot(store, sid, seconds=0.94)
    fs.save(store)
    at = run("shoot", sid)
    assert not at.exception
    length = [n for n in at.number_input if n.label == "length"][0]
    assert abs(length.value - 0.94) < 1e-9


def test_no_studio_view_mentions_previs(store):
    # the user does not want previs in Studio: no panel, no button, no text
    sid = store["timeline"][0]
    for view in ("write", "shoot", "cut", "settings", "effects"):
        at = run(view, sid if view == "shoot" else None)
        assert not at.exception, view
        page = " ".join(str(getattr(x, "value", "")) + str(getattr(x, "label", ""))
                        for kind in ("markdown", "caption", "button", "info",
                                     "warning", "error", "text")
                        for x in getattr(at, kind, [])).lower()
        assert "previs" not in page, view
        assert not any("previs" in str(e.label).lower() for e in at.expander), view


def test_stale_shot_offers_use_this_take_as_is(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(fs, "PRODUCTIONS", tmp_path)
    p = fs.new_project("ui-adopt", "a diver")
    fs.save(p)
    fs.add_shot(p, prompt="a diver descends", scale="close", subject="diver",
                scene="sc1")
    fs.save(p)
    sid = p["timeline"][0]
    shot = fs.get_shot(p, sid)
    d = fs.take_dir(p, shot, "t01")
    (d / "frames").mkdir(parents=True)
    from PIL import Image
    Image.new("RGB", (16, 16), (9, 9, 9)).save(d / "frames" / "f000.png")
    gen = dict(fs.generation_inputs(p, shot))
    rec = {"take": "t01", "shot": sid, "generation": gen,
           "input_hash": fs.input_hash(gen), "finish": fs.finish_inputs(p, shot),
           "rebuild_hash": fs.rebuild_hash(p, shot), "seconds": 60, "frames": 31,
           "approved": False}
    (d / "take.json").write_text(json.dumps(rec), encoding="utf-8")
    fs.update_shot(p, sid, take="t01", prompt="a diver turns away")
    at = run_named("ui-adopt", "shoot", sid)
    assert not at.exception
    assert any("older **prompt**" in w.value for w in at.warning)
    [b for b in at.button if b.label == "use this take as-is"][0].click().run()
    assert not at.exception
    assert fs.shot_state(fs.load("ui-adopt"), sid) in ("rendered", "approved")
    assert not [b for b in at.button if b.label == "use this take as-is"]


def run_named(slug, view, shot=None):
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["slug"] = slug
    at.session_state["view"] = view
    if shot:
        at.session_state["shot"] = shot
    at.run()
    return at
