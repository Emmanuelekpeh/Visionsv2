"""The automatic sequence director: several movements, several grammars, no mode
picker. The studio should not ask a person to choose hold/montage; it should read
the material, choose per movement, and scope camera moves to match."""

from __future__ import annotations

from pathlib import Path

import film_store as fs
import sequence_direction as sd
from edit_planner import Shot

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "output" / "_test_direction"


def _shot(prompt, subject="", scene="s1", energy=0.2, beat="develop",
          scale="medium", camera="none"):
    return Shot(prompt=prompt, scale=scale, camera=camera, seconds=2.0,
                subject=subject, scene=scene, beat=beat, energy=energy)


def test_length_is_fitted_with_shots_not_longer_holds():
    n30 = sd.shots_for_duration(30, 16, 16)
    n8 = sd.shots_for_duration(8, 16, 16)
    assert n8 == 4
    assert 14 <= n30 <= 16          # ~1.94 s of motion per shot
    est = sd.estimate(30, 16, 16)
    assert est["shots"] == n30
    assert est["render_minutes"] > 150


def test_the_shortest_length_asks_for_a_film_the_gate_can_pass():
    """A 4 s target used to derive 2 shots -- and a 2-shot film can never pass
    the story gate: it needs an establish and a resolve at the ends and a turn
    between them, which is three slots. The write view's minimum length was
    therefore guaranteed to produce a REFUSED draft."""
    assert sd.shots_for_duration(4, 16, 16) == 3
    assert sd.shots_for_duration(6, 16, 16) == 3


def test_hold_then_montage_inside_one_film():
    shots = [
        _shot("a rain-soaked footbridge, a woman in a dark coat stepping on",
              "woman", "s1", 0.2, "establish"),
        _shot("the same woman walking along the footbridge, iron railing",
              "woman", "s1", 0.2, "develop"),
        _shot("the woman's hand on the footbridge railing, rain", "woman", "s1",
              0.2, "turn"),
        _shot("a neon sign in a narrow alley, rain", "", "s2", 0.45, "develop"),
        _shot("a chrome diner at noon, hard shadows", "", "s3", 0.45, "develop"),
        _shot("a yellow digger bucket in clay", "", "s4", 0.45, "resolve"),
    ]
    d = sd.plan(shots)
    modes = [m.mode for m in d.movements]
    assert sd.CONTINUOUS in modes or sd.HOLD in modes, modes
    assert sd.MONTAGE in modes, modes
    # the hold movement shares one scene; the montage run does not
    arranged = sd.apply(shots, d)
    hold_idx = [i for m in d.movements if m.mode in (sd.HOLD, sd.CONTINUOUS)
                for i in m.indices]
    mont_idx = [i for m in d.movements if m.mode == sd.MONTAGE for i in m.indices]
    if hold_idx:
        assert len({arranged[i].scene for i in hold_idx}) == 1
    if mont_idx:
        assert len({arranged[i].scene for i in mont_idx}) == len(mont_idx)
    assert arranged[0].link == "open"
    assert arranged[1].link in ("match-action", "eyeline")


def test_camera_moves_are_scoped_to_the_movements_that_can_carry_them():
    shots = [_shot("a bridge, rain", "", "s1", 0.5, "develop"),
             _shot("a diner at noon", "", "s2", 0.5, "develop"),
             _shot("a barn at dusk", "", "s3", 0.5, "develop")]
    d = sd.plan(shots)
    # force one montage movement and one hold movement for the policy test
    d.movements[0].mode = sd.HOLD
    d.movements[0].camera_budget = 1
    d.movements[1].mode = sd.MONTAGE
    d.movements[1].camera_budget = 0
    scoped = sd.scope_cameras(["pan-left", "tilt-up", "drone-forward"], d)
    hold = d.movements[0].indices
    mont = d.movements[1].indices
    assert sum(1 for i in hold if scoped[i] != "none") <= 1
    assert all(scoped[i] == "none" for i in mont)


def test_hold_gets_one_finishing_camera_move_with_settle():
    """The renderer cannot settle a generated camera move. The hold's move is a
    FINISHING spec on one shot, generation stays still."""
    shots = [
        _shot("a rain-soaked footbridge, a woman stepping on", "woman", "s1",
              0.45, "establish", scale="wide"),
        _shot("the same woman on the footbridge, iron railing", "woman", "s1",
              0.30, "develop"),
        _shot("her hand on the footbridge railing", "woman", "s1", 0.20, "turn"),
    ]
    d = sd.plan(shots)
    assert d.movements[0].mode in (sd.HOLD, sd.CONTINUOUS)
    cam = d.movements[0].camera_plan
    assert cam and cam["move"] in sd.CAMERA_MOVES
    arranged = sd.apply(shots, d)
    assert all(s.camera == "none" for s in arranged)     # generation static
    moves = [(i, s.camera_move) for i, s in enumerate(arranged) if s.camera_move]
    assert len(moves) == 1
    i, spec = moves[0]
    assert spec["settle"] >= 0.5 and spec["amount"] <= 60
    assert i == cam["shot"]


def test_montage_gets_no_camera_moves():
    shots = [_shot("a bridge", "", "s1", 0.5, "develop"),
             _shot("a diner", "", "s2", 0.5, "develop"),
             _shot("a barn", "", "s3", 0.5, "resolve")]
    d = sd.plan(shots)
    assert all(m.mode == sd.MONTAGE for m in d.movements)
    assert all(m.camera_plan is None for m in d.movements)
    assert all(s.camera_move is None for s in sd.apply(shots, d))


def test_writer_movements_are_used_instead_of_resegmenting():
    shots = [
        _shot("a bridge at night", "", "s1", 0.3, "establish"),
        _shot("a diner at noon", "", "s2", 0.3, "develop"),
        _shot("a barn at dusk", "", "s3", 0.3, "turn"),
        _shot("a statue in a hall", "", "s4", 0.3, "resolve"),
    ]
    shots[0] = shots[0].__class__(**{**shots[0].__dict__, "movement": "m1"})
    shots[1] = shots[1].__class__(**{**shots[1].__dict__, "movement": "m1"})
    shots[2] = shots[2].__class__(**{**shots[2].__dict__, "movement": "m2"})
    shots[3] = shots[3].__class__(**{**shots[3].__dict__, "movement": "m2"})
    d = sd.plan(shots)
    assert [len(m.indices) for m in d.movements] == [2, 2]
    assert any("writer's movement ids respected" in n for n in d.notes)


def test_camera_move_is_a_finishing_input():
    project = {"slug": "x", "settings": {"fps": 16},
               "scenes": [{"id": "sc1", "palette": "neutral"}], "timeline": []}
    shot = {"id": "s01", "scene": "sc1", "seconds": 2.0, "camera": "none",
            "camera_move": {"move": "tilt-down", "amount": 45, "settle": 0.62}}
    fin = fs.finish_inputs(project, shot)
    assert fin["camera_move"]["move"] == "tilt-down"
    shot2 = dict(shot, camera_move=None)
    assert fs.finish_inputs(project, shot2)["camera_move"] is None
    assert fin != fs.finish_inputs(project, shot2)


def test_writer_movement_field_survives_validation():
    """The writer now plans movements; validate must carry the ids through."""
    from story import validate
    raw = {"logline": "x", "shots": [
        {"prompt": "a rain-soaked footbridge at night, a woman stepping on, fog",
         "scale": "wide", "camera": "none", "seconds": 2.5, "subject": "woman",
         "scene": "s1", "energy": 0.2, "beat": "establish", "purpose": "establish",
         "link": "open", "movement": "m1"},
        {"prompt": "the same woman walking along the footbridge, rain and fog",
         "scale": "medium", "camera": "none", "seconds": 1.7, "subject": "woman",
         "scene": "s1", "energy": 0.2, "beat": "develop", "purpose": "master",
         "link": "match-action", "movement": "m1"},
        {"prompt": "a second figure under the streetlight on the same bridge, fog",
         "scale": "medium", "camera": "none", "seconds": 2.0, "subject": "figure",
         "scene": "s2", "energy": 0.3, "beat": "turn", "purpose": "reaction",
         "link": "scene-cut", "movement": "m2"},
        {"prompt": "the two figures on the rain-soaked bridge, fog",
         "scale": "close", "camera": "none", "seconds": 1.5, "subject": "figure",
         "scene": "s2", "energy": 0.1, "beat": "resolve", "purpose": "resolve",
         "link": "eyeline", "movement": "m2"},
    ]}
    shots, _problems = validate(raw)
    assert [s.movement for s in shots] == ["m1", "m1", "m2", "m2"]
    d = sd.plan(shots)
    assert [len(m.indices) for m in d.movements] == [2, 2]


def test_direct_project_writes_a_plan_and_rewrites_the_store(monkeypatch):
    prod = SCRATCH / "productions"
    if prod.exists():
        import shutil
        shutil.rmtree(prod)
    prod.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(fs, "PRODUCTIONS", prod)
    p = fs.new_project("directiontest", "x")
    fs.save(p)
    for i, (prompt, subj, beat, en) in enumerate([
            ("a rain-soaked footbridge, a woman stepping on", "woman", "establish", 0.2),
            ("the same woman on the footbridge railing", "woman", "develop", 0.2),
            ("a neon sign in an alley", "", "turn", 0.5),
            ("a chrome diner at noon", "", "resolve", 0.4)]):
        fs.add_shot(p, prompt=prompt, subject=subj, beat=beat, energy=en,
                    scene=f"s{i + 1}", seconds=2.0, scale="wide")
    d = sd.direct_project(p)
    assert p.get("direction", {}).get("movements")
    saved = fs.load("directiontest")
    assert saved.get("direction", {}).get("movements")
    assert len({s.get("scene") for s in fs.shots_in_order(saved)}) >= 1


def _redirect_store(monkeypatch, tmp_path, camera_mode=None, movements=None):
    monkeypatch.setattr(fs, "PRODUCTIONS", tmp_path)
    p = fs.new_project("redirectparity", "x")
    if camera_mode:
        p["settings"]["camera_mode"] = camera_mode
    fs.save(p)
    rows = [("a rain-soaked footbridge, a woman stepping on", "woman", "establish", 0.2),
            ("the same woman on the footbridge railing", "woman", "develop", 0.2),
            ("a neon sign in an alley at night", "", "turn", 0.5),
            ("a chrome diner at noon, steam from a cup", "", "resolve", 0.4)]
    for i, (prompt, subj, beat, en) in enumerate(rows):
        fs.add_shot(p, prompt=prompt, subject=subj, beat=beat, energy=en,
                    scene=f"s{i + 1}", seconds=2.0, scale="wide",
                    movement=(movements[i] if movements else ""),
                    camera="pan-left" if i == 3 else "none",
                    camera_move={"move": "zoom-in", "amount": 38, "settle": 0.6}
                    if i == 0 else None)
    return p


def test_redirect_respects_the_writers_movement_ids(monkeypatch, tmp_path):
    """`as_shots` dropped `movement`, so re-direct re-derived boundaries the writer
    had already planned and the draft path respected."""
    p = _redirect_store(monkeypatch, tmp_path, movements=["m1", "m1", "m2", "m2"])
    d = sd.direct_project(p)
    assert [len(m.indices) for m in d.movements] == [2, 2]
    assert any("writer's movement ids respected" in n for n in d.notes)


def test_redirect_rewrites_camera_move_like_the_draft_path(monkeypatch, tmp_path):
    """Auto cameras: generation static, the plan's one move is a finishing spec.
    The old store move on shot 1 must not survive a plan that no longer gives it."""
    p = _redirect_store(monkeypatch, tmp_path, movements=["m1", "m1", "m2", "m2"])
    d = sd.direct_project(p)
    saved = fs.load("redirectparity")
    shots = fs.shots_in_order(saved)
    assert [s["camera"] for s in shots] == ["none"] * 4
    planned = [m.camera_plan for m in d.movements if m.camera_plan]
    moved = [i for i, s in enumerate(shots) if (s.get("camera_move") or {}).get("move")]
    assert moved == sorted(int(c["shot"]) for c in planned)
    for c in planned:
        assert shots[int(c["shot"])]["camera_move"]["move"] == c["move"]


def test_redirect_with_an_explicit_camera_mode_keeps_generation_cameras(
        monkeypatch, tmp_path):
    p = _redirect_store(monkeypatch, tmp_path, camera_mode="suggestions",
                        movements=["m1", "m1", "m2", "m2"])
    sd.direct_project(p)
    shots = fs.shots_in_order(fs.load("redirectparity"))
    # the stored finishing move is left alone, as in the draft's explicit-camera path
    assert (shots[0].get("camera_move") or {}).get("move") == "zoom-in"
