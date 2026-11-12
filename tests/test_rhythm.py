"""R4 must sound like a rhythm, not a metronome.

MEASURED: the delivered 13-shot film had shot-length CV 0.074. The cause was R4's
own narrow beat bands (develop 0.88-1.05 of the 1.94 s source), i.e. the planner
called a smooth curve "rhythm" while R9's own floor is CV 0.20. The renderer caps
the long end (1.30x static / 1.15x moving), so rhythm must come from alternating
near-cap holds with floor accents -- these tests pin that.
"""

from __future__ import annotations

from dataclasses import replace

from edit_planner import Shot, rhythm_curve
from handheld import max_hold_seconds, natural_seconds

FPS = 16
FRAMES = 16


def cv(values):
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    return sd / mean


def _delivered_list():
    """The delivered ego_v5 shape: 4/4/4/1 scene runs over 13 shots."""
    spec = [
        ("sc1", "establish", "establish", "wide"),
        ("sc1", "develop", "master", "medium"),
        ("sc1", "develop", "insert", "close"),
        ("sc1", "develop", "master", "medium"),
        ("sc2", "turn", "master", "wide"),
        ("sc2", "turn", "insert", "close"),
        ("sc2", "fall", "reaction", "medium"),
        ("sc2", "fall", "master", "wide"),
        ("sc3", "fall", "insert", "extreme-wide"),
        ("sc3", "fall", "insert", "medium"),
        ("sc3", "fall", "master", "wide"),
        ("sc3", "fall", "insert", "close"),
        ("sc4", "resolve", "master", "medium"),
    ]
    return [Shot(f"a concrete image {i + 1}", scale, "none", 2.0,
                 scene=scene, beat=beat, purpose=purpose)
            for i, (scene, beat, purpose, scale) in enumerate(spec)]


def test_the_delivered_13_shot_shape_now_has_rhythm():
    shots = _delivered_list()
    values = rhythm_curve(shots, total=24.94, frames=FRAMES, fps=FPS)
    assert cv(values) >= 0.20, values
    # the rhythm is actually bimodal: accents near the floor, holds near the cap
    caps = [max_hold_seconds(FRAMES, FPS) for _ in shots]
    assert max(values) - min(values) >= 0.8
    assert sum(1 for v in values if v <= 1.75) >= 3
    assert sum(1 for v in values if v >= 0.95 * max(caps)) >= 4
    assert abs(sum(values) - 24.94) <= 0.15 * 24.94
    # every scene opens on its longest shot (R9)
    for sc in ("sc1", "sc2", "sc3"):
        idx = [i for i, s in enumerate(shots) if s.scene == sc]
        assert values[idx[0]] >= max(values[i] for i in idx[1:]) - 0.02
    # no three consecutive holds within 0.1 s (R9)
    for i in range(1, len(values) - 1):
        assert not (abs(values[i] - values[i - 1]) <= 0.1
                    and abs(values[i] - values[i + 1]) <= 0.1)


def test_every_hold_is_inside_the_renderer_budget():
    shots = _delivered_list()
    values = rhythm_curve(shots, total=28.0, frames=FRAMES, fps=FPS)
    for s, v in zip(shots, values):
        cap = max_hold_seconds(FRAMES, FPS, moving=s.moving)
        assert v <= cap + 1e-6
        # accents may be TRIMMED below the old 1.5 s writer floor (fast cuts)
        assert v >= 0.9 - 1e-6


def test_a_moving_shot_gets_the_tighter_ceiling_in_the_curve():
    from edit_planner import _rhythm_curve_beats
    moving = [Shot("image", "wide", "zoom-in", 2.0, beat="establish")]
    static = [Shot("image", "wide", "none", 2.0, beat="establish")]
    v_mv = _rhythm_curve_beats(moving, frames=FRAMES, fps=FPS)[0]
    v_st = _rhythm_curve_beats(static, frames=FRAMES, fps=FPS)[0]
    assert v_mv <= max_hold_seconds(FRAMES, FPS, moving=True) + 1e-6
    assert v_st > v_mv


# --------------------------------------------------------------------------- #
# the pacing study's LONG band: procedural holds via finishing RAFT
# --------------------------------------------------------------------------- #

def test_upsampling_widens_the_motion_budget():
    """16 raw frames at x4 = 61 delivered frames = 3.81 s of smooth motion; the
    pacing study calls this the procedural-hold route to its 4-8 s band."""
    assert natural_seconds(16, FPS, upsample=4) == (
        61 / FPS)
    assert max_hold_seconds(16, FPS, upsample=4) > (
        1.9 * max_hold_seconds(16, FPS, upsample=2))
    long = [Shot("image", "wide", "none", 2.0, beat="establish", upsample=4)]
    from edit_planner import _rhythm_curve_beats
    v = _rhythm_curve_beats(long, frames=FRAMES, fps=FPS)[0]
    assert 3.0 < v <= max_hold_seconds(16, FPS, upsample=4)
    # the shot list may now reach the study's >5 s band without re-generating
    assert rhythm_curve([replace(long[0], upsample=6)], frames=FRAMES,
                        fps=FPS)[0] > 5.0


def test_with_procedural_holds_the_curve_reaches_the_human_distribution():
    """The delivered shape with x4 long takes at scene openings/establishes and
    x6 resolve: accents near the floor, holds in the study's 3-5 s band, one >5 s.
    CV must land near the human 0.75, not the 0.2 floor."""
    shots = _delivered_list()
    ups = []
    for i, s in enumerate(shots):
        if s.beat == "resolve" and i == len(shots) - 1:
            ups.append(6)
        elif i in (0, 4, 8) or s.beat in ("establish", "resolve"):
            ups.append(4)
        else:
            ups.append(2)
    shots = [replace(s, upsample=u) for s, u in zip(shots, ups)]
    vals = rhythm_curve(shots, frames=FRAMES, fps=FPS)
    assert cv(vals) >= 0.5, vals
    assert max(vals) > 5.0
    assert sum(1 for v in vals if v <= 1.5) >= 4
    assert sum(1 for v in vals if v >= 3.0) >= 3
    for s, v in zip(shots, vals):
        assert v <= max_hold_seconds(FRAMES, FPS, moving=s.moving,
                                     upsample=s.upsample) + 1e-6
