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


def test_amused_prompt_puts_the_subject_clause_first():
    p = "mountain ridge mid-afternoon wind gusts up, A male skydiver in orange jumps, sun"
    out = kf.amused_prompt(p, "a skydiver")
    assert out.startswith("A male skydiver in orange jumps")
    assert out.endswith("mountain ridge mid-afternoon wind gusts up")   # moved, not dropped
    assert sorted(out.split(", ")) == sorted(p.split(", "))
    # already first, never named, or no subject: unchanged
    assert kf.amused_prompt("A male skydiver, fog", "a skydiver") == "A male skydiver, fog"
    assert kf.amused_prompt("a vast field, mist", "a skydiver") == "a vast field, mist"
    assert kf.amused_prompt("tags, a skydiver", "") == "tags, a skydiver"


def test_ip_animation_passes_the_still_as_reference_and_refuses_when_unconditioned(
        tmp_path, monkeypatch):
    import make_film
    p, sid = proj_two(tmp_path, monkeypatch)
    _cand(p, sid, 0, look="amused")
    seen = {}

    class Fake:
        def __init__(self, *a, **k):
            self.conditioned = seen.get("cond", True)

        def clip(self, shot, seed, model=None, reference=None, tier="fast", avoid="",
                 anchor=None, ramp=None):
            seen.update(reference=reference, tier=tier, avoid=avoid, anchor=anchor,
                        ramp=ramp)
            return [Image.new("RGB", (8, 8))] * 2, 1.0

    monkeypatch.setattr(make_film, "Generator", Fake)
    monkeypatch.setattr(kf, "ROOT", tmp_path)
    assert kf.animate_test(p, sid, 0, None, mode="ip", ip_scale=0.9) == 0
    assert seen["tier"] == "full" and seen["reference"].size == (32, 32)
    assert make_film.IP_ADAPTER_SCALE == 0.9
    assert (tmp_path / "output" / "audit" / p["slug"] / f"kf_{sid}_c0_ip"
            / "f000.png").exists()
    seen["cond"] = False                      # adapter failed to load
    assert kf.animate_test(p, sid, 0, None, mode="ip") == 3
    seen.update(cond=True)
    assert kf.animate_test(p, sid, 0, None, mode="ramp", ramp=(1.0, 0.4)) == 0
    assert seen["reference"] is None and seen["anchor"].size == (32, 32)
    assert seen["ramp"] == (1.0, 0.4)
    assert kf.animate_test(p, sid, 0, None, mode="ip+ramp") == 0
    assert seen["reference"] is not None and seen["anchor"] is not None
    with pytest.raises(ValueError):
        kf.animate_test(p, sid, 0, None, mode="bogus")


def proj_two(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "PRODUCTIONS", tmp_path)
    p = fs.new_project("ipt", "a diver")
    fs.save(p)
    fs.add_shot(p, prompt="a diver leaps", scale="medium", subject="a diver",
                scene="sc1")
    fs.save(p)
    return p, p["timeline"][0]



def test_anchor_callback_pins_frame0_and_leaves_the_free_frames_alone():
    import torch
    import make_film
    from diffusers import DPMSolverMultistepScheduler

    class Dist:
        def __init__(self, m):
            self.mean = m

    class Out:
        def __init__(self, m):
            self.latent_dist = Dist(m)

    class FakeVae:
        class config:
            scaling_factor = 1.0

        def encode(self, x):                      # a constant latent, easy to check
            return Out(torch.full((1, 4, 64, 64), 0.5))

    sch = DPMSolverMultistepScheduler()
    sch.set_timesteps(6)

    class Pipe:
        scheduler = sch

    cb = make_film.anchor_callback(FakeVae(), Image.new("RGB", (64, 64)),
                                   (1.0, 0.5), seed=3)
    lat = torch.randn(1, 4, 8, 64, 64)
    before = lat.clone()
    out = cb(Pipe, 0, sch.timesteps[0], {"latents": lat})["latents"]
    z0 = torch.full((1, 4, 64, 64), 0.5)
    eps = torch.randn(z0.shape, generator=torch.Generator("cpu").manual_seed(3 + 7919))
    known = sch.add_noise(z0, eps, sch.timesteps[1].reshape(1))
    assert torch.allclose(out[:, :, 0], known, atol=1e-5)                 # pinned
    assert torch.allclose(out[:, :, 1], 0.5 * known + 0.5 * before[:, :, 1], atol=1e-5)
    assert torch.equal(out[:, :, 2:], before[:, :, 2:])                   # free
    last = cb(Pipe, len(sch.timesteps) - 1, sch.timesteps[-1],
              {"latents": torch.randn(1, 4, 8, 64, 64)})["latents"]
    assert torch.allclose(last[:, :, 0], z0)           # clean still after the last step


def test_move_mode_hands_video_to_video_a_moving_input(tmp_path, monkeypatch):
    import numpy as np
    import make_film
    p, sid = proj_two(tmp_path, monkeypatch)
    d = kf.keyframe_dir(p, sid)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    still = Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8"))
    still.save(d / "c0.png")
    seen = {}

    class Fake:
        def __init__(self, *a, **k):
            pass

        def clip_from_still(self, shot, seed, src, strength=None, avoid=""):
            seen["src"] = src
            return [Image.new("RGB", (8, 8))] * 2, 1.0

    monkeypatch.setattr(make_film, "Generator", Fake)
    monkeypatch.setattr(kf, "ROOT", tmp_path)
    assert kf.animate_test(p, sid, 0, None, mode="move", move="zoom-in") == 0
    src = seen["src"]
    assert isinstance(src, list) and len(src) == make_film.FRAMES
    a = [np.asarray(f, dtype=np.float32) for f in src]
    assert np.abs(a[0] - a[-1]).mean() > 5            # the input really moves
    assert all(np.abs(a[i] - a[i + 1]).mean() > 0 for i in range(len(a) - 1))
    ref = np.asarray(still.resize(src[0].size), dtype=np.float32)
    assert np.abs(ref - a[0]).mean() < np.abs(ref - a[-1]).mean()   # frame 0 nearest
    assert (tmp_path / "output" / "audit" / p["slug"] / f"kf_{sid}_c0_move"
            / "f000.png").exists()
