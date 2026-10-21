"""
Spec-level tests for redrawn drawings and the key-span link.

The RAFT warping implementation (`redrawer.py`) was removed: warping two keys
smears unrelated pixels and cannot invent a pose that appears in neither key.
What survives is the *abstraction*, which is what a generative redrawer needs:

  * a REDRAWN drawing names the two fixed keys it sits between, and the eased
    position between them -- so a generator can be conditioned on exactly that
  * a REDRAWN drawing may never be derived from another redraw, for the same
    reason an img2img chain is forbidden (INV-1)
  * PoseState carries origin/target/t so the exposure sheet can drive whatever
    produces the inbetween, warp or generator
"""

from __future__ import annotations

import pytest

from animation_spec import (
    AnimationState,
    AutoregressiveChainError,
    Drawing,
    DrawingSource,
    Easing,
    ExposureSheet,
    ExposureTrack,
    KeyPose,
    Layer,
    MotionPlan,
    SpecViolation,
)


# ---------------------------------------------------------------------------
# The engine tells the redrawer where each inbetween sits
# ---------------------------------------------------------------------------

def test_inbetweens_carry_their_key_span_and_position():
    plan = MotionPlan((
        KeyPose("a", AnimationState.of(x=0.0)),
        KeyPose("b", AnimationState.of(x=1.0), inbetweens=3),
    ))
    states = plan.states()
    keys = [s for s in states if not s.is_inbetween]
    mids = [s for s in states if s.is_inbetween]

    assert [s.key for s in keys] == ["a", "b"]
    assert all(s.origin == s.key == s.target for s in keys)
    assert len(mids) == 3
    assert all(s.origin == "a" and s.target == "b" for s in mids)
    assert [round(s.t, 3) for s in mids] == [0.25, 0.5, 0.75]


def test_eased_inbetweens_are_not_evenly_spaced():
    """Acceleration is the point: the t values must bunch up."""
    plan = MotionPlan((
        KeyPose("a", AnimationState.of(x=0.0)),
        KeyPose("b", AnimationState.of(x=1.0), ease_into=Easing.ACCELERATE, inbetweens=3),
    ))
    ts = [s.t for s in plan.states() if s.is_inbetween]
    gaps = [ts[0]] + [ts[i] - ts[i - 1] for i in range(1, len(ts))]
    assert gaps == sorted(gaps), f"accelerating motion should widen its steps, got {gaps}"


def test_key_span_helper_matches_the_state():
    plan = MotionPlan((
        KeyPose("a", AnimationState.of(x=0.0)),
        KeyPose("b", AnimationState.of(x=1.0), inbetweens=2),
    ))
    for state in plan.states():
        origin, target, t = plan.key_span(state)
        assert origin == state.origin and target == state.target and t == state.t


def test_keys_are_not_inbetweens():
    plan = MotionPlan((
        KeyPose("a", AnimationState.of(x=0.0)),
        KeyPose("b", AnimationState.of(x=1.0), inbetweens=1),
    ))
    for state in plan.states():
        assert state.is_inbetween == (state.origin != state.target)


# ---------------------------------------------------------------------------
# REDRAWN is a first-class drawing source, and it cannot chain
# ---------------------------------------------------------------------------

def _redrawn_sheet(extra_drawing=None):
    sheet = ExposureSheet(sheet_id="r")
    sheet.add_drawing(Drawing("key_a", source=DrawingSource.AUTHORED))
    sheet.add_drawing(Drawing("key_b", source=DrawingSource.AUTHORED))
    sheet.add_drawing(Drawing("mid", source=DrawingSource.REDRAWN,
                              redraw_from=("key_a", "key_b"), redraw_t=0.5))
    if extra_drawing is not None:
        sheet.add_drawing(extra_drawing)
    return sheet


def test_redrawn_drawing_is_valid_and_needs_no_generation():
    sheet = _redrawn_sheet()
    sheet.add_layer(Layer("main", ExposureTrack.mixed(
        [("key_a", 1), ("mid", 1), ("key_b", 1)])))
    sheet.validate()

    assert not sheet.drawings["mid"].needs_generation
    assert sheet.drawings["mid"].needs_redraw
    assert [d.id for d in sheet.render_plan()] == ["key_a", "key_b"]
    assert [d.id for d in sheet.redraw_plan()] == ["mid"]


def test_redrawn_drawing_records_its_position_between_the_keys():
    sheet = _redrawn_sheet()
    mid = sheet.drawings["mid"]
    assert mid.redraw_from == ("key_a", "key_b")
    assert mid.redraw_t == 0.5


def test_redraw_without_two_keys_is_rejected():
    sheet = ExposureSheet(sheet_id="r")
    sheet.add_drawing(Drawing("key_a", source=DrawingSource.AUTHORED))
    sheet.add_drawing(Drawing("mid", source=DrawingSource.REDRAWN, redraw_from=("key_a",)))
    sheet.add_layer(Layer("main", ExposureTrack.mixed([("key_a", 1), ("mid", 1)])))
    with pytest.raises(SpecViolation):
        sheet.validate()


def test_redraw_from_unknown_key_is_rejected():
    sheet = ExposureSheet(sheet_id="r")
    sheet.add_drawing(Drawing("key_a", source=DrawingSource.AUTHORED))
    sheet.add_drawing(Drawing("mid", source=DrawingSource.REDRAWN,
                              redraw_from=("key_a", "ghost")))
    sheet.add_layer(Layer("main", ExposureTrack.mixed([("key_a", 1), ("mid", 1)])))
    with pytest.raises(SpecViolation):
        sheet.validate()


def test_redrawing_from_another_redraw_is_a_chain():
    """Deriving a drawing from another derived drawing compounds error."""
    sheet = _redrawn_sheet(
        Drawing("mid2", source=DrawingSource.REDRAWN, redraw_from=("key_a", "mid"))
    )
    sheet.add_layer(Layer("main", ExposureTrack.mixed(
        [("key_a", 1), ("mid", 1), ("mid2", 1), ("key_b", 1)]
    )))
    with pytest.raises(AutoregressiveChainError):
        sheet.validate()


def test_redrawing_from_a_chained_drawing_is_rejected():
    sheet = ExposureSheet(sheet_id="r")
    sheet.add_drawing(Drawing("key_a", source=DrawingSource.AUTHORED))
    sheet.add_drawing(Drawing("bad", source=DrawingSource.CHAINED))
    sheet.add_drawing(Drawing("mid", source=DrawingSource.REDRAWN,
                              redraw_from=("key_a", "bad")))
    sheet.add_layer(Layer("main", ExposureTrack.mixed([("key_a", 1), ("mid", 1)])))
    with pytest.raises(AutoregressiveChainError):
        sheet.validate()


def test_redrawn_may_be_built_from_an_anchored_key():
    """key_b anchored to the root is a fixed key, so a redraw from it is legal."""
    sheet = ExposureSheet(sheet_id="r")
    sheet.add_drawing(Drawing("key_a", source=DrawingSource.AUTHORED))
    sheet.add_drawing(Drawing("key_b", source=DrawingSource.ANCHORED, anchor="key_a"))
    sheet.add_drawing(Drawing("mid", source=DrawingSource.REDRAWN,
                              redraw_from=("key_a", "key_b"), redraw_t=0.5))
    sheet.add_layer(Layer("main", ExposureTrack.mixed(
        [("key_a", 1), ("mid", 1), ("key_b", 1)])))
    sheet.validate()


def test_cost_report_separates_generation_from_redraw():
    sheet = ExposureSheet(sheet_id="r", fps=24)
    sheet.add_drawing(Drawing("key_a", source=DrawingSource.AUTHORED))
    sheet.add_drawing(Drawing("key_b", source=DrawingSource.AUTHORED))
    for i in range(5):
        sheet.add_drawing(Drawing(f"mid{i}", source=DrawingSource.REDRAWN,
                                  redraw_from=("key_a", "key_b"), redraw_t=i / 5))
    holds = [("key_a", 4)] + [(f"mid{i}", 2) for i in range(5)] + [("key_b", 4)]
    sheet.add_layer(Layer("main", ExposureTrack.mixed(holds)))
    sheet.validate()

    report = sheet.cost_report()
    assert report["frames"] == 18
    assert report["generation_calls"] == 2   # only the keys cost diffusion
    assert report["redraw_calls"] == 5       # the middle is cheap
    assert report["held_or_reused_frames"] == 11
