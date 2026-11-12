"""The aMUSEd look and its Studio panel: numbering, job task, choosing a still."""
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import film_store as fs      # noqa: E402
import job_runner as jr      # noqa: E402
import keyframe as kf        # noqa: E402

APP = str(ROOT / "ui" / "studio.py")


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "PRODUCTIONS", tmp_path)
    p = fs.new_project("kfl", "a squid")
    fs.save(p)
    fs.add_shot(p, prompt="a squid drifts", scale="medium", scene="sc1")
    fs.save(p)
    return p, p["timeline"][0]


def _cand(p, sid, n, **meta):
    d = kf.keyframe_dir(p, sid)
    d.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (n * 20, 0, 0)).save(d / f"c{n}.png")
    (d / f"c{n}.json").write_text(json.dumps({"shot": sid, **meta}), encoding="utf-8")


def test_new_candidates_never_overwrite_old_ones(proj):
    p, sid = proj
    assert kf.next_candidate(p, sid) == 0
    _cand(p, sid, 0, look="plain")
    _cand(p, sid, 1, look="amused", strength=0.5)
    _cand(p, sid, 10, look="amused", strength=0.8)       # numeric, not alphabetic
    assert kf.next_candidate(p, sid) == 11
    info = kf.candidate_info(p, sid)
    assert [c["candidate"] for c in info] == ["c0", "c1", "c10"]
    assert info[1]["strength"] == 0.5


def test_render_rejects_an_unknown_look(proj):
    p, sid = proj
    with pytest.raises(ValueError):
        kf.render(p, [sid], look="midjourney")


def test_choose_sets_the_keyframe_and_makes_the_shot_full_and_stale(proj):
    p, sid = proj
    _cand(p, sid, 0, look="amused", strength=0.65)
    fs.update_shot(p, sid, quality="fast")
    assert fs.render_tier(fs.get_shot(p, sid)) == "fast"
    kf.choose(p, sid, 0)
    shot = fs.get_shot(p, sid)
    assert shot["keyframe"].endswith("keyframes/%s/c0.png" % sid)
    assert fs.render_tier(shot) == "full"                # a keyframe is always full


def test_keyframes_job_task_and_progress():
    argv = jr.TASKS["keyframes"]["argv"](
        {"store": "squid", "shot": "s05", "look": "amused", "candidates": 2,
         "strengths": "0.5,0.7"})
    assert argv[-6:] == ["--candidates", "2", "--look", "amused",
                         "--looks-strengths", "0.5,0.7"]
    prog = {}
    jr._keyframe_progress("  [keyframe] s05 aMUSEd seed 9: 8s", prog)
    jr._keyframe_progress("  [keyframe] s05 c3 seed 9 amused strength 0.65: 140s", prog)
    assert prog["stills"] == 2 and "c3" in prog["last"]


def test_studio_panel_lists_candidates_and_choosing_one_sets_the_keyframe(proj):
    from streamlit.testing.v1 import AppTest
    p, sid = proj
    _cand(p, sid, 0, look="plain")
    _cand(p, sid, 1, look="amused", strength=0.65)
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["slug"] = p["slug"]
    at.session_state["view"] = "shoot"
    at.session_state["shot"] = sid
    at.run()
    assert not at.exception
    assert any(b.label == "make candidates" for b in at.button)
    use = [b for b in at.button if b.label == "use as first frame"]
    assert len(use) == 2
    use[1].click().run()
    assert not at.exception
    assert fs.get_shot(fs.load(p["slug"]), sid)["keyframe"].endswith("c1.png")
