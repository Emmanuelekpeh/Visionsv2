"""
Tests for the locked animation contract (animation_spec.py / ANIMATION_SPEC.md).

These tests are the enforcement arm of the spec. If one of them fails, the
project has drifted away from the doctrine -- not merely from an implementation
detail.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from animation_spec import (
    AnimationState,
    AutoregressiveChainError,
    Drawing,
    DrawingSource,
    Easing,
    ExposureSheet,
    ExposureTrack,
    Hold,
    KeyPose,
    Layer,
    LoopPolicy,
    MotionPlan,
    PoseRole,
    SpecViolation,
    sheet_from_keyframes,
)


# ---------------------------------------------------------------------------
# Exposure: on ones / twos / threes / mixed
# ---------------------------------------------------------------------------


def test_on_ones_exposes_every_drawing_for_one_frame():
    track = ExposureTrack.on_ones(["A", "B", "C"])
    assert track.total_frames == 3
    assert track.run_lengths() == (1, 1, 1)
    assert [track.drawing_at(f) for f in range(3)] == ["A", "B", "C"]


def test_on_twos_halves_the_drawing_count_for_the_same_duration():
    drawings = ["A", "B", "C", "D"]
    ones = ExposureTrack.on_ones(drawings)
    twos = ExposureTrack.on_twos(drawings)
    assert twos.total_frames == 8
    assert twos.run_lengths() == (2, 2, 2, 2)
    assert [twos.drawing_at(f) for f in range(8)] == ["A", "A", "B", "B", "C", "C", "D", "D"]
    # The whole point: same span, half the drawings.
    assert ones.drawing_count == twos.drawing_count == 4
    assert twos.total_frames == 2 * ones.total_frames


def test_on_threes_is_eight_drawings_per_second_at_24fps():
    track = ExposureTrack.on_threes([f"d{i}" for i in range(8)])
    assert track.total_frames == 24
    assert track.drawing_count == 8


def test_mixed_exposure_preserves_uneven_timing():
    # A:0-1  B:2-3  C:4  D:5  E:6-8  F:9
    track = ExposureTrack.mixed([("A", 2), ("B", 2), ("C", 1), ("D", 1), ("E", 3), ("F", 1)])
    assert track.run_lengths() == (2, 2, 1, 1, 3, 1)
    assert track.total_frames == 10
    assert [track.drawing_at(f) for f in range(10)] == [
        "A", "A", "B", "B", "C", "D", "E", "E", "E", "F"
    ]


def test_a_hold_is_a_first_class_thing():
    # INV-3: stillness is expressible and costs nothing.
    held = ExposureTrack.held("A", 12)
    assert held.total_frames == 12
    assert held.drawing_count == 1
    assert set(held.drawing_at(f) for f in range(12)) == {"A"}


def test_zero_length_hold_is_rejected():
    with pytest.raises(SpecViolation):
        Hold("A", 0)


def test_empty_track_is_rejected():
    with pytest.raises(SpecViolation):
        ExposureTrack(())


# ---------------------------------------------------------------------------
# Loop policies
# ---------------------------------------------------------------------------


def test_hold_last_keeps_the_final_drawing_on_screen():
    track = ExposureTrack.on_ones(["A", "B"])
    assert track.drawing_at(99, loop=LoopPolicy.HOLD_LAST) == "B"


def test_repeat_cycles_the_track():
    track = ExposureTrack.on_ones(["A", "B"])
    assert track.drawing_at(4, loop=LoopPolicy.REPEAT) == "A"
    assert track.drawing_at(5, loop=LoopPolicy.REPEAT) == "B"


def test_error_policy_rejects_a_short_track():
    track = ExposureTrack.on_ones(["A"])
    with pytest.raises(SpecViolation):
        track.drawing_at(1, loop=LoopPolicy.ERROR)


def test_negative_frame_is_rejected():
    with pytest.raises(SpecViolation):
        ExposureTrack.on_ones(["A"]).drawing_at(-1)


# ---------------------------------------------------------------------------
# Rescaling (this replaces pad/trim)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [1, 2, 3, 5, 7, 11, 23, 24, 25, 47, 100, 997])
def test_scaled_to_hits_the_target_exactly(target):
    track = ExposureTrack.mixed([("A", 3), ("B", 7), ("C", 2), ("D", 5)])
    scaled = track.scaled_to(max(target, 4))
    assert scaled.total_frames == max(target, 4)


@pytest.mark.parametrize("target", [4, 5, 9, 16, 24, 33, 64])
def test_scaled_to_never_drops_a_drawing(target):
    track = ExposureTrack.mixed([("A", 1), ("B", 1), ("C", 1), ("D", 1)])
    scaled = track.scaled_to(target)
    assert scaled.drawing_count == 4
    assert all(h.frames >= 1 for h in scaled.holds)


def test_scaled_to_is_identity_when_already_correct():
    track = ExposureTrack.on_twos(["A", "B"])
    assert track.scaled_to(4) is track


def test_scaled_to_refuses_to_drop_drawings():
    track = ExposureTrack.on_ones(["A", "B", "C", "D"])
    with pytest.raises(SpecViolation):
        track.scaled_to(3)


def test_extend_to_stretches_only_the_tail():
    track = ExposureTrack.mixed([("A", 2), ("B", 2)])
    extended = track.extend_to(10)
    assert extended.total_frames == 10
    assert extended.run_lengths() == (2, 8)


# ---------------------------------------------------------------------------
# Easing: the character of movement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("easing", list(Easing))
def test_every_easing_starts_at_zero_and_ends_at_one(easing):
    assert easing.sample(0.0) == pytest.approx(0.0, abs=1e-9)
    assert easing.sample(1.0) == pytest.approx(1.0, abs=1e-9)


def test_easing_clamps_out_of_range_time():
    assert Easing.LINEAR.sample(-5.0) == 0.0
    assert Easing.LINEAR.sample(5.0) == 1.0


def test_overshoot_actually_passes_the_target():
    # The reason linear interpolation is not enough on its own.
    peak = max(Easing.OVERSHOOT.sample(t / 100) for t in range(101))
    assert peak > 1.0


def test_anticipate_pulls_back_before_moving():
    trough = min(Easing.ANTICIPATE.sample(t / 100) for t in range(101))
    assert trough < 0.0


def test_snap_does_not_travel_until_the_end():
    assert Easing.SNAP.sample(0.5) == 0.0
    assert Easing.SNAP.sample(1.0) == 1.0


def test_accelerate_is_slower_than_linear_early_on():
    assert Easing.ACCELERATE.sample(0.25) < Easing.LINEAR.sample(0.25)


def test_decelerate_is_faster_than_linear_early_on():
    assert Easing.DECELERATE.sample(0.25) > Easing.LINEAR.sample(0.25)


# ---------------------------------------------------------------------------
# AnimationState
# ---------------------------------------------------------------------------


def test_lerp_interpolates_numeric_channels():
    a = AnimationState.of(arm=0.0, lean=1.0)
    b = AnimationState.of(arm=10.0, lean=3.0)
    mid = AnimationState.lerp(a, b, 0.5)
    assert mid["arm"] == pytest.approx(5.0)
    assert mid["lean"] == pytest.approx(2.0)


def test_lerp_snaps_discrete_channels():
    a = AnimationState.of(mouth="closed", arm=0.0)
    b = AnimationState.of(mouth="open", arm=1.0)
    assert AnimationState.lerp(a, b, 0.4)["mouth"] == "closed"
    assert AnimationState.lerp(a, b, 0.6)["mouth"] == "open"


def test_lerp_handles_channels_present_in_only_one_state():
    a = AnimationState.of(x=0.0)
    b = AnimationState.of(y=5.0)
    mid = AnimationState.lerp(a, b, 0.5)
    assert mid["x"] == 0.0
    assert mid["y"] == 5.0


# ---------------------------------------------------------------------------
# MotionPlan: keys + breakdowns -> sparse discrete states
# ---------------------------------------------------------------------------


def test_plan_without_inbetweens_is_a_cut():
    plan = MotionPlan((
        KeyPose("stand", AnimationState.of(arm=0.0)),
        KeyPose("release", AnimationState.of(arm=1.0), ease_into=Easing.SNAP, inbetweens=0),
    ))
    states = plan.states()
    assert [s.pose_name for s in states] == ["stand", "release"]
    assert all(s.role is PoseRole.KEY for s in states)


def test_inbetweens_create_intermediate_states():
    plan = MotionPlan((
        KeyPose("stand", AnimationState.of(arm=0.0)),
        KeyPose("release", AnimationState.of(arm=1.0), inbetweens=3),
    ))
    states = plan.states()
    assert len(states) == 5  # 2 keys + 3 inbetweens
    assert [s.role for s in states] == [
        PoseRole.KEY, PoseRole.BREAKDOWN, PoseRole.BREAKDOWN, PoseRole.BREAKDOWN, PoseRole.KEY
    ]
    arms = [s.state["arm"] for s in states]
    assert arms == sorted(arms)
    assert arms[0] == 0.0 and arms[-1] == 1.0


def test_authored_breakdown_is_kept_as_a_key():
    # The animator's own breakdown, not a computed one.
    plan = MotionPlan((
        KeyPose("arm_back", AnimationState.of(arm=0.0)),
        KeyPose("elbow_leads", AnimationState.of(arm=0.8, elbow=1.0), role=PoseRole.BREAKDOWN),
        KeyPose("release", AnimationState.of(arm=1.0, elbow=0.0)),
    ))
    states = plan.states()
    assert [s.key for s in states] == ["arm_back", "elbow_leads", "release"]
    assert states[1].role is PoseRole.BREAKDOWN
    assert states[1].state["elbow"] == 1.0


def test_hold_expands_exposure_not_drawing_count():
    plan = MotionPlan((
        KeyPose("still", AnimationState.of(x=0.0), hold=12),
        KeyPose("move", AnimationState.of(x=1.0), ease_into=Easing.SNAP),
    ))
    assert plan.drawing_count() == 2  # two drawings...
    track = plan.to_track()
    assert track.total_frames == 13    # ...spanning thirteen frames


def test_plan_is_sparse_relative_to_frames():
    # One second of 24 fps output carried by seven drawings.
    plan = MotionPlan(tuple(
        KeyPose(f"p{i}", AnimationState.of(x=float(i)),
                ease_into=Easing.SNAP, hold=h)
        for i, h in enumerate([4, 3, 3, 4, 4, 3, 3])
    ))
    track = plan.to_track()
    assert track.total_frames == 24
    assert plan.drawing_count() == 7
    # The model operates in a lower-dimensional temporal space than the video.
    assert track.drawing_count < track.total_frames


def test_plan_requires_at_least_one_pose():
    with pytest.raises(SpecViolation):
        MotionPlan(())


def test_pose_rejects_negative_inbetweens_and_empty_hold():
    with pytest.raises(SpecViolation):
        KeyPose("x", AnimationState.of(a=0.0), inbetweens=-1)
    with pytest.raises(SpecViolation):
        KeyPose("x", AnimationState.of(a=0.0), hold=0)


# ---------------------------------------------------------------------------
# INV-1: no autoregressive chaining
# ---------------------------------------------------------------------------


def _sheet_with(drawings, holds):
    sheet = ExposureSheet(sheet_id="t")
    for d in drawings:
        sheet.add_drawing(d)
    sheet.add_layer(Layer(name="main", track=ExposureTrack.mixed(holds)))
    return sheet


def test_chained_drawing_is_rejected():
    # This is the exact failure the spec exists to prevent:
    # character -> slightly different -> different person -> texture soup.
    sheet = _sheet_with(
        [
            Drawing("kf0", source=DrawingSource.AUTHORED),
            Drawing("kf1", source=DrawingSource.CHAINED),
        ],
        [("kf0", 1), ("kf1", 1)],
    )
    with pytest.raises(AutoregressiveChainError):
        sheet.validate()


def test_anchoring_to_the_root_is_allowed():
    sheet = _sheet_with(
        [
            Drawing("kf0", source=DrawingSource.AUTHORED),
            Drawing("kf1", source=DrawingSource.ANCHORED, anchor="kf0"),
            Drawing("kf2", source=DrawingSource.ANCHORED, anchor="kf0"),
        ],
        [("kf0", 1), ("kf1", 1), ("kf2", 1)],
    )
    sheet.validate()  # must not raise


def test_anchoring_to_another_anchor_is_a_chain_in_disguise():
    sheet = _sheet_with(
        [
            Drawing("kf0", source=DrawingSource.AUTHORED),
            Drawing("kf1", source=DrawingSource.ANCHORED, anchor="kf0"),
            Drawing("kf2", source=DrawingSource.ANCHORED, anchor="kf1"),
        ],
        [("kf0", 1), ("kf1", 1), ("kf2", 1)],
    )
    with pytest.raises(AutoregressiveChainError):
        sheet.validate()


def test_anchor_without_a_target_is_rejected():
    sheet = _sheet_with([Drawing("kf0", source=DrawingSource.ANCHORED)], [("kf0", 1)])
    with pytest.raises(SpecViolation):
        sheet.validate()


def test_anchor_to_unknown_drawing_is_rejected():
    sheet = _sheet_with(
        [
            Drawing("kf0", source=DrawingSource.AUTHORED),
            Drawing("kf1", source=DrawingSource.ANCHORED, anchor="nope"),
        ],
        [("kf0", 1), ("kf1", 1)],
    )
    with pytest.raises(SpecViolation):
        sheet.validate()


def test_exposing_an_undeclared_drawing_is_rejected():
    sheet = _sheet_with([Drawing("kf0")], [("kf0", 1), ("ghost", 1)])
    with pytest.raises(SpecViolation):
        sheet.validate()


def test_sheet_with_no_layers_is_rejected():
    with pytest.raises(SpecViolation):
        ExposureSheet(sheet_id="empty").validate()


# ---------------------------------------------------------------------------
# INV-4: multiple clocks
# ---------------------------------------------------------------------------


def test_static_layer_never_changes():
    sheet = ExposureSheet(sheet_id="s")
    sheet.add_drawing(Drawing("bg"))
    sheet.add_drawing(Drawing("a"))
    sheet.add_layer(Layer(name="background", track=ExposureTrack.held("bg", 1), static=True))
    sheet.add_layer(Layer(name="body", track=ExposureTrack.on_ones(["a"] * 8)))
    assert [sheet.frame_state(f)["background"] for f in range(8)] == ["bg"] * 8


def test_half_rate_layer_changes_on_every_second_frame():
    sheet = ExposureSheet(sheet_id="s")
    for d in ["p0", "p1", "p2", "p3"]:
        sheet.add_drawing(Drawing(d))
    sheet.add_layer(
        Layer(name="face", track=ExposureTrack.on_ones(["p0", "p1", "p2", "p3"]),
              clock=Fraction(1, 2))
    )
    seen = [sheet.frame_state(f)["face"] for f in range(8)]
    assert seen == ["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3"]


def test_third_rate_layer_and_total_frame_count():
    sheet = ExposureSheet(sheet_id="s")
    for d in ["fx0", "fx1", "fx2"]:
        sheet.add_drawing(Drawing(d))
    sheet.add_layer(
        Layer(name="effects", track=ExposureTrack.on_ones(["fx0", "fx1", "fx2"]),
              clock=Fraction(1, 3))
    )
    assert sheet.total_frames == 9
    assert [sheet.frame_state(f)["effects"] for f in range(9)] == (
        ["fx0"] * 3 + ["fx1"] * 3 + ["fx2"] * 3
    )


def test_zero_clock_is_the_same_as_static():
    layer = Layer(name="bg", track=ExposureTrack.held("bg", 1), clock=Fraction(0))
    assert layer.is_static
    assert layer.drawing_at(500) == "bg"


def test_negative_clock_is_rejected():
    with pytest.raises(SpecViolation):
        Layer(name="x", track=ExposureTrack.held("a", 1), clock=Fraction(-1))


def test_mixed_clocks_coexist_in_one_sheet():
    sheet = ExposureSheet(sheet_id="limited")
    for d in ["bg", "a0", "a1", "f0", "e0", "e1"]:
        sheet.add_drawing(Drawing(d))
    sheet.add_layer(Layer(name="background", track=ExposureTrack.held("bg", 1), static=True))
    sheet.add_layer(Layer(name="body", track=ExposureTrack.on_ones(["a0", "a1"] * 3)))
    sheet.add_layer(Layer(name="face", track=ExposureTrack.on_ones(["f0"] * 3), clock=Fraction(1, 2)))
    sheet.add_layer(Layer(name="effects", track=ExposureTrack.on_ones(["e0", "e1"]), clock=Fraction(1, 3)))

    state = sheet.frame_state(0)
    assert set(state) == {"background", "body", "face", "effects"}
    # INV-4: no single universal clock.
    assert len({lay.clock for lay in sheet.layers}) > 1


def test_event_driven_track_holds_between_events():
    track = ExposureTrack.from_events(default="eyes_open", events={5: "blink"}, total_frames=12)
    assert [track.drawing_at(f) for f in range(12)] == (
        ["eyes_open"] * 5 + ["blink"] + ["blink"] * 6
    )


def test_event_driven_track_respects_an_initial_event():
    track = ExposureTrack.from_events(default="open", events={0: "wide", 3: "open"}, total_frames=5)
    assert [track.drawing_at(f) for f in range(5)] == ["wide", "wide", "wide", "open", "open"]


# ---------------------------------------------------------------------------
# Render plan: what actually has to be drawn
# ---------------------------------------------------------------------------


def test_render_plan_deduplicates_drawings():
    sheet = ExposureSheet(sheet_id="s")
    for d in ["A", "B", "C"]:
        sheet.add_drawing(Drawing(d))
    sheet.add_layer(Layer(name="main", track=ExposureTrack.on_twos(["A", "B", "C"])))
    plan = sheet.render_plan()
    assert [d.id for d in plan] == ["A", "B", "C"]


def test_held_frames_do_not_appear_in_the_render_plan():
    sheet = ExposureSheet(sheet_id="s")
    for d in ["A", "B"]:
        sheet.add_drawing(Drawing(d))
    sheet.add_layer(Layer(name="main", track=ExposureTrack.mixed([("A", 10), ("B", 2)])))
    assert [d.id for d in sheet.render_plan()] == ["A", "B"]


def test_reused_drawings_are_not_regenerated():
    sheet = ExposureSheet(sheet_id="s")
    sheet.add_drawing(Drawing("A", source=DrawingSource.AUTHORED))
    sheet.add_drawing(Drawing("A'", source=DrawingSource.REUSED))
    sheet.add_layer(Layer(name="main", track=ExposureTrack.mixed([("A", 1), ("A'", 1)])))
    assert [d.id for d in sheet.render_plan()] == ["A"]


def test_cost_report_quantifies_the_saving():
    sheet = ExposureSheet(sheet_id="s")
    for d in ["A", "B", "C", "D", "E", "F", "G"]:
        sheet.add_drawing(Drawing(d))
    sheet.add_layer(Layer(name="main", track=ExposureTrack.mixed(
        [("A", 3), ("B", 2), ("C", 1), ("D", 1), ("E", 4), ("F", 1), ("G", 2)]
    )))
    report = sheet.cost_report()
    assert report["frames"] == 14
    assert report["generation_calls"] == 7
    assert report["held_or_reused_frames"] == 7
    assert report["reduction_vs_on_ones"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Fitting to audio duration
# ---------------------------------------------------------------------------


def test_fit_to_frames_hits_the_duration_exactly():
    sheet = ExposureSheet(sheet_id="s")
    for d in ["bg", "a", "b", "c"]:
        sheet.add_drawing(Drawing(d))
    sheet.add_layer(Layer(name="background", track=ExposureTrack.held("bg", 1), static=True))
    sheet.add_layer(Layer(name="body", track=ExposureTrack.on_ones(["a", "b", "c"])))
    sheet.fit_to_frames(50)
    assert sheet.total_frames == 50


def test_fit_to_frames_leaves_static_layers_alone():
    sheet = ExposureSheet(sheet_id="s")
    for d in ["bg", "a", "b"]:
        sheet.add_drawing(Drawing(d))
    bg = Layer(name="background", track=ExposureTrack.held("bg", 1), static=True)
    sheet.add_layer(bg)
    sheet.add_layer(Layer(name="body", track=ExposureTrack.on_ones(["a", "b"])))
    sheet.fit_to_frames(20)
    assert sheet.layer("background").track.total_frames == 1


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_sheet_round_trips_through_json():
    sheet = ExposureSheet(sheet_id="round", fps=24)
    for d in ["bg", "a", "b"]:
        sheet.add_drawing(Drawing(d, prompt=f"prompt {d}"))
    sheet.add_layer(Layer(name="background", track=ExposureTrack.held("bg", 1), static=True))
    sheet.add_layer(
        Layer(name="body", track=ExposureTrack.mixed([("a", 2), ("b", 2)]), clock=Fraction(1, 2))
    )
    restored = ExposureSheet.from_dict(sheet.to_dict())
    assert restored.sheet_id == sheet.sheet_id
    assert restored.total_frames == sheet.total_frames
    assert restored.layer("body").clock == Fraction(1, 2)
    assert restored.layer("background").is_static
    assert restored.timeline() == sheet.timeline()


# ---------------------------------------------------------------------------
# Bridge from approved keyframes
# ---------------------------------------------------------------------------


def test_sheet_from_keyframes_roots_at_the_first_keyframe():
    sheet = sheet_from_keyframes("shot0", ["a.png", "b.png", "c.png"], duration_frames=24)
    assert sheet.drawings["kf0"].source is DrawingSource.AUTHORED
    assert sheet.drawings["kf1"].anchor == "kf0"
    assert sheet.drawings["kf2"].anchor == "kf0"
    sheet.validate()  # anchored to the root, not to the previous frame


def test_sheet_from_keyframes_fits_the_duration_exactly():
    sheet = sheet_from_keyframes("shot0", ["a.png", "b.png", "c.png"], duration_frames=37)
    assert sheet.total_frames == 37


def test_sheet_from_keyframes_accepts_authored_timing():
    sheet = sheet_from_keyframes(
        "shot0", ["a.png", "b.png", "c.png"], duration_frames=6, exposure=[3, 1, 2]
    )
    assert sheet.layer("main").track.run_lengths() == (3, 1, 2)


def test_sheet_from_keyframes_rejects_mismatched_exposure():
    with pytest.raises(SpecViolation):
        sheet_from_keyframes("s", ["a.png", "b.png"], duration_frames=8, exposure=[1, 1, 1])


def test_sheet_from_keyframes_requires_a_keyframe():
    with pytest.raises(SpecViolation):
        sheet_from_keyframes("s", [], duration_frames=8)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def test_ascii_dope_sheet_renders_every_frame():
    sheet = ExposureSheet(sheet_id="s")
    for d in ["bg", "a", "b"]:
        sheet.add_drawing(Drawing(d))
    sheet.add_layer(Layer(name="background", track=ExposureTrack.held("bg", 1), static=True))
    sheet.add_layer(Layer(name="body", track=ExposureTrack.on_twos(["a", "b"])))
    text = sheet.to_ascii()
    assert "background" in text and "body" in text
    assert "clock=1/2" in text or "static" in text
    assert text.count("\n") >= 4
