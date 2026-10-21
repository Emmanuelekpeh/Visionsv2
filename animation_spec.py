"""
animation_spec.py -- the executable form of ANIMATION_SPEC.md.

This module is the LOCKED animation contract for the project. It is deliberately
pure Python: no torch, no PIL, no numpy, no I/O. It describes *what should exist
and when*, and nothing about how a pixel is produced.

The central claim it encodes:

    An animation is a sequence of discrete visual states, plus a decision about
    how long each state is allowed to exist before reality changes.

Consequences that are enforced here, not merely documented:

  INV-1  A drawing is never derived from the previous drawn frame.
         Drawings come from a state (AUTHORED) or from a fixed root reference
         (ANCHORED, rooted at an AUTHORED drawing). CHAINED drawings raise.

  INV-2  Timing is authored, not implied. Frame counts come from an exposure
         sheet, never from dividing a duration by a segment count.

  INV-3  Stillness is first class. A hold is a normal, expressible thing, and a
         hold costs zero generation calls.

  INV-4  Layers may run on different clocks. There is no single universal clock.

Vocabulary maps to traditional animation as follows:

    key pose / breakdown  ->  KeyPose / PoseRole
    inbetween             ->  generated intermediate PoseState
    exposure sheet        ->  ExposureSheet
    on ones/twos/threes   ->  ExposureTrack.on_ones / .on_twos / .on_threes
    cel layer             ->  Layer
    held frame            ->  Hold
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SPEC_VERSION = "1.0.0"

__all__ = [
    "SPEC_VERSION",
    "SpecViolation",
    "AutoregressiveChainError",
    "AnimationState",
    "Easing",
    "PoseRole",
    "KeyPose",
    "PoseState",
    "MotionPlan",
    "DrawingSource",
    "Drawing",
    "Hold",
    "LoopPolicy",
    "ExposureTrack",
    "Layer",
    "ExposureSheet",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SpecViolation(Exception):
    """The sheet violates a normative rule of ANIMATION_SPEC.md."""


class AutoregressiveChainError(SpecViolation):
    """INV-1 violated: a drawing was derived from the previous drawn frame."""


# ---------------------------------------------------------------------------
# Animation state
# ---------------------------------------------------------------------------


def _lerp_value(a: Any, b: Any, t: float) -> Any:
    """Interpolate one channel.

    Numeric channels interpolate. Everything else (strings, enums, bools)
    switches at the halfway point -- a discrete visual state cannot be
    half-way between two poses, so it must snap.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return a if t < 0.5 else b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + (b - a) * t
    return a if t < 0.5 else b


@dataclass(frozen=True)
class AnimationState:
    """The world at one instant: pose, velocity, expression, camera, lighting.

    Deliberately a flat channel bag rather than a fixed schema, because the
    channels a shot cares about differ per shot (a talking head has `mouth`,
    a falling object has `velocity`).
    """

    channels: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, **channels: Any) -> "AnimationState":
        return cls(dict(channels))

    def __getitem__(self, key: str) -> Any:
        return self.channels[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.channels.get(key, default)

    def keys(self) -> Iterable[str]:
        return self.channels.keys()

    @staticmethod
    def lerp(a: "AnimationState", b: "AnimationState", t: float) -> "AnimationState":
        """Blend two states.

        Note that this is *not* the normal way to produce inbetweens -- see
        MotionPlan. Authored breakdowns are preferred; lerp exists so that pure
        easing curves (accelerate, overshoot, anticipate) remain expressible.
        """
        keys = list(a.channels.keys())
        for key in b.channels.keys():
            if key not in a.channels:
                keys.append(key)
        out: Dict[str, Any] = {}
        for key in keys:
            if key not in a.channels:
                out[key] = b.channels[key]
            elif key not in b.channels:
                out[key] = a.channels[key]
            else:
                out[key] = _lerp_value(a.channels[key], b.channels[key], t)
        return AnimationState(out)

    def __repr__(self) -> str:  # keeps test failures readable
        inner = ", ".join(f"{k}={v!r}" for k, v in self.channels.items())
        return f"AnimationState({inner})"


# ---------------------------------------------------------------------------
# Easing: the *character* of a movement
# ---------------------------------------------------------------------------


class Easing(str, Enum):
    """How a movement happens between two poses.

    OVERSHOOT and ANTICIPATE intentionally leave the [0, 1] range: they are the
    whole reason linear interpolation is insufficient. OVERSHOOT passes the
    target and settles; ANTICIPATE pulls back before moving.
    """

    LINEAR = "linear"          # mechanical, constant rate
    ACCELERATE = "accelerate"  # gaining momentum
    DECELERATE = "decelerate"  # arriving, slowing
    SMOOTH = "smooth"          # ease in and out
    SNAP = "snap"              # no visible travel: a cut to the next pose
    OVERSHOOT = "overshoot"    # goes past the target, then settles
    ANTICIPATE = "anticipate"  # winds up backwards first

    def sample(self, t: float) -> float:
        """Map normalized time in [0, 1] to normalized progress.

        Progress may fall outside [0, 1] for OVERSHOOT / ANTICIPATE.
        """
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        if self is Easing.LINEAR:
            return t
        if self is Easing.ACCELERATE:
            return t * t
        if self is Easing.DECELERATE:
            return 1.0 - (1.0 - t) ** 2
        if self is Easing.SMOOTH:
            return t * t * (3.0 - 2.0 * t)
        if self is Easing.SNAP:
            return 0.0 if t < 1.0 else 1.0
        if self is Easing.OVERSHOOT:
            c1, c3 = 1.70158, 2.70158
            return 1.0 + c3 * (t - 1.0) ** 3 + c1 * (t - 1.0) ** 2
        if self is Easing.ANTICIPATE:
            c1, c3 = 1.70158, 2.70158
            return c3 * t ** 3 - c1 * t * t
        raise ValueError(f"unhandled easing: {self!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Poses -> sparse state sequence
# ---------------------------------------------------------------------------


class PoseRole(str, Enum):
    KEY = "key"              # an animator-chosen extreme
    BREAKDOWN = "breakdown"  # an animator-chosen inbetween that shapes the motion


@dataclass(frozen=True)
class KeyPose:
    """One authored pose in the plan.

    `ease_into` and `inbetweens` describe the transition *arriving at* this
    pose from the previous one.

    `inbetweens == 0` means the animator declined to bridge the two poses: the
    sequence cuts. That is a legitimate and frequently correct choice.
    """

    name: str
    state: AnimationState
    role: PoseRole = PoseRole.KEY
    ease_into: Easing = Easing.LINEAR
    inbetweens: int = 0
    hold: int = 1
    note: str = ""

    def __post_init__(self) -> None:
        if self.inbetweens < 0:
            raise SpecViolation(f"pose {self.name!r}: inbetweens must be >= 0")
        if self.hold < 1:
            raise SpecViolation(f"pose {self.name!r}: hold must be >= 1 frame")


@dataclass(frozen=True)
class PoseState:
    """A realized discrete state: one drawing, and how long it is held.

    `origin` / `target` / `t` are the link to a redrawer. When a state is an
    inbetween, it names the two key drawings it sits between and its eased
    position along that span, so the drawing can be produced by warping the keys
    rather than by running a generator. When a state *is* a key, origin == target
    and t == 0: nothing needs to be redrawn.
    """

    index: int
    key: str
    pose_name: str
    state: AnimationState
    role: PoseRole
    hold: int
    note: str = ""
    origin: Optional[str] = None
    target: Optional[str] = None
    t: float = 0.0

    @property
    def is_inbetween(self) -> bool:
        """True when this state must be produced by redrawing the key pair."""
        return self.origin is not None and self.target is not None and self.origin != self.target


@dataclass(frozen=True)
class MotionPlan:
    """Keys and breakdowns, expanded into a sparse sequence of discrete states.

    This is the layer that replaces `IMAGE_t -> IMAGE_(t+1)`. The future is not
    inferred from the present drawing; it is read off the plan.
    """

    poses: Tuple[KeyPose, ...]

    def __post_init__(self) -> None:
        if not self.poses:
            raise SpecViolation("a MotionPlan needs at least one pose")

    def states(self) -> Tuple[PoseState, ...]:
        """Expand the plan into its discrete states, in order."""
        out: List[PoseState] = []
        prev: Optional[KeyPose] = None

        for pose in self.poses:
            if prev is not None and pose.inbetweens > 0:
                steps = pose.inbetweens + 1
                for k in range(1, pose.inbetweens + 1):
                    t = k / steps
                    eased = pose.ease_into.sample(t)
                    blended = AnimationState.lerp(prev.state, pose.state, eased)
                    key = f"{prev.name}->{pose.name}#{k}"
                    out.append(
                        PoseState(
                            index=len(out),
                            key=key,
                            pose_name=pose.name,
                            state=blended,
                            role=PoseRole.BREAKDOWN,
                            hold=1,
                            note=f"{pose.ease_into.value} inbetween {k}/{pose.inbetweens}",
                            origin=prev.name,
                            target=pose.name,
                            t=eased,
                        )
                    )

            out.append(
                PoseState(
                    index=len(out),
                    key=pose.name,
                    pose_name=pose.name,
                    state=pose.state,
                    role=pose.role,
                    hold=pose.hold,
                    note=pose.note,
                    origin=pose.name,
                    target=pose.name,
                    t=0.0,
                )
            )
            prev = pose

        return tuple(out)

    def key_span(self, state: PoseState) -> Tuple[Optional[str], Optional[str], float]:
        """(origin_key, target_key, eased_t) for a state, for a redrawer to use."""
        return state.origin, state.target, state.t

    def to_track(self) -> "ExposureTrack":
        return ExposureTrack.from_poses(self.states())

    def drawing_count(self) -> int:
        """How many distinct drawings this plan actually requires."""
        return len(self.states())


# ---------------------------------------------------------------------------
# Drawings and their provenance (the INV-1 mechanism)
# ---------------------------------------------------------------------------


class DrawingSource(str, Enum):
    AUTHORED = "authored"  # generated from a state; no dependence on a prior drawing
    ANCHORED = "anchored"  # pulled toward a fixed AUTHORED root reference
    REDRAWN = "redrawn"    # warped between two fixed keys by the redrawer
    REUSED = "reused"      # the same drawing shown again; costs nothing
    CHAINED = "chained"    # FORBIDDEN: derived from the previous drawn frame


@dataclass(frozen=True)
class Drawing:
    """A discrete visual state that must be drawn, plus where it came from."""

    id: str
    prompt: str = ""
    source: DrawingSource = DrawingSource.AUTHORED
    anchor: Optional[str] = None   # required for ANCHORED; must name an AUTHORED root
    state_key: Optional[str] = None
    note: str = ""
    # For REDRAWN: the two fixed keys this drawing is warped between, and the
    # eased position. Both must be roots, so a redraw can never chain.
    redraw_from: Optional[Tuple[str, str]] = None
    redraw_t: float = 0.0

    @property
    def needs_generation(self) -> bool:
        """Does this drawing require a generator (diffusion) call?"""
        return self.source in (DrawingSource.AUTHORED, DrawingSource.ANCHORED)

    @property
    def needs_redraw(self) -> bool:
        """Does this drawing require the cheap redrawer instead?"""
        return self.source is DrawingSource.REDRAWN


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hold:
    """A drawing, and how many frames it is exposed for."""

    drawing: str
    frames: int = 1

    def __post_init__(self) -> None:
        if self.frames < 1:
            raise SpecViolation(f"hold on {self.drawing!r}: frames must be >= 1 (got {self.frames})")


class LoopPolicy(str, Enum):
    HOLD_LAST = "hold_last"  # keep the final drawing on screen
    REPEAT = "repeat"        # cycle the track
    ERROR = "error"          # a short track is a bug


@dataclass(frozen=True)
class ExposureTrack:
    """Drawings mapped onto frames: this is the exposure sheet for one layer."""

    holds: Tuple[Hold, ...]

    def __post_init__(self) -> None:
        if not self.holds:
            raise SpecViolation("an ExposureTrack needs at least one hold")

    # -- constructors -------------------------------------------------------

    @classmethod
    def on_ones(cls, drawings: Sequence[str]) -> "ExposureTrack":
        return cls(tuple(Hold(d, 1) for d in drawings))

    @classmethod
    def on_twos(cls, drawings: Sequence[str]) -> "ExposureTrack":
        return cls(tuple(Hold(d, 2) for d in drawings))

    @classmethod
    def on_threes(cls, drawings: Sequence[str]) -> "ExposureTrack":
        return cls(tuple(Hold(d, 3) for d in drawings))

    @classmethod
    def mixed(cls, pairs: Sequence[Tuple[str, int]]) -> "ExposureTrack":
        """Explicit timing, e.g. [("A", 2), ("B", 2), ("C", 1), ("D", 4)].

        This is where animation becomes expressive rather than merely correct.
        """
        return cls(tuple(Hold(d, n) for d, n in pairs))

    @classmethod
    def held(cls, drawing: str, frames: int) -> "ExposureTrack":
        return cls((Hold(drawing, frames),))

    @classmethod
    def from_poses(cls, states: Sequence[PoseState]) -> "ExposureTrack":
        return cls(tuple(Hold(s.key, s.hold) for s in states))

    @classmethod
    def from_events(cls, default: str, events: Mapping[int, str], total_frames: int) -> "ExposureTrack":
        """Event-driven exposure, e.g. eyes that change occasionally.

        `events` maps a frame index to the drawing that starts at that frame.
        """
        if total_frames < 1:
            raise SpecViolation("from_events needs total_frames >= 1")
        points = sorted((f, d) for f, d in events.items() if 0 <= f < total_frames)
        holds: List[Hold] = []
        cursor = 0
        current = default
        for frame, drawing in points:
            if frame > cursor:
                holds.append(Hold(current, frame - cursor))
                cursor = frame
            current = drawing
        if cursor < total_frames:
            holds.append(Hold(current, total_frames - cursor))
        return cls(tuple(holds))

    # -- queries ------------------------------------------------------------

    @property
    def total_frames(self) -> int:
        return sum(h.frames for h in self.holds)

    @property
    def drawings(self) -> Tuple[str, ...]:
        """Distinct drawing ids, in first-exposure order."""
        seen: List[str] = []
        for h in self.holds:
            if h.drawing not in seen:
                seen.append(h.drawing)
        return tuple(seen)

    @property
    def drawing_count(self) -> int:
        return len(self.drawings)

    def drawing_at(self, local_frame: int, loop: LoopPolicy = LoopPolicy.HOLD_LAST) -> str:
        """Which drawing is exposed at `local_frame` on this track."""
        if local_frame < 0:
            raise SpecViolation(f"negative frame index: {local_frame}")
        total = self.total_frames
        if local_frame >= total:
            if loop is LoopPolicy.HOLD_LAST:
                return self.holds[-1].drawing
            if loop is LoopPolicy.REPEAT:
                local_frame %= total
            else:
                raise SpecViolation(
                    f"frame {local_frame} is past the end of a {total}-frame track "
                    f"and loop policy is ERROR"
                )
        cursor = 0
        for h in self.holds:
            if local_frame < cursor + h.frames:
                return h.drawing
            cursor += h.frames
        return self.holds[-1].drawing  # pragma: no cover

    def segments(self) -> Tuple[Tuple[str, int, int], ...]:
        """(drawing, start_frame, length_in_frames) for every hold."""
        out = []
        cursor = 0
        for h in self.holds:
            out.append((h.drawing, cursor, h.frames))
            cursor += h.frames
        return tuple(out)

    def run_lengths(self) -> Tuple[int, ...]:
        """The exposure pattern, e.g. (1, 1, 2, 3) -- handy for asserting timing."""
        return tuple(h.frames for h in self.holds)

    def scaled_to(self, total_frames: int) -> "ExposureTrack":
        """Rescale all holds to hit `total_frames` exactly.

        Proportional (largest-remainder) allocation. Every drawing keeps at
        least one frame, so a rescale can never silently drop a pose. This
        exists so that assembly never pads or trims frames after the fact.
        """
        n = len(self.holds)
        if total_frames < n:
            raise SpecViolation(
                f"cannot fit {n} holds into {total_frames} frames without dropping drawings"
            )
        current = self.total_frames
        if current == total_frames:
            return self

        exact = [h.frames * total_frames / current for h in self.holds]
        counts = [max(1, int(x)) for x in exact]

        # Correct upward using the largest dropped fraction, downward by
        # shaving the longest holds that can still spare a frame.
        guard = 0
        while sum(counts) != total_frames:
            guard += 1
            if guard > 100_000:  # pragma: no cover
                raise SpecViolation("scaled_to failed to converge")
            if sum(counts) < total_frames:
                idx = max(range(n), key=lambda i: (exact[i] - int(exact[i]), self.holds[i].frames))
                counts[idx] += 1
            else:
                spare = [i for i in range(n) if counts[i] > 1]
                if not spare:  # pragma: no cover
                    raise SpecViolation("scaled_to cannot shrink further")
                idx = max(spare, key=lambda i: counts[i])
                counts[idx] -= 1

        return ExposureTrack(tuple(Hold(h.drawing, c) for h, c in zip(self.holds, counts)))

    def extend_to(self, total_frames: int, loop: LoopPolicy = LoopPolicy.HOLD_LAST) -> "ExposureTrack":
        """Grow (or shrink) the track to `total_frames` by stretching the tail."""
        if total_frames < self.total_frames:
            return self.scaled_to(total_frames)
        extra = total_frames - self.total_frames
        if extra == 0:
            return self
        head = self.holds[:-1]
        tail = self.holds[-1]
        return ExposureTrack(head + (Hold(tail.drawing, tail.frames + extra),))


# ---------------------------------------------------------------------------
# Layers: multiple clocks inside one frame
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Layer:
    """One cel layer, with its own clock.

    `clock` is the layer's rate relative to real time:

        clock = 1     -> body(t)
        clock = 1/2   -> face(t / 2)      changes at half rate
        clock = 1/3   -> effects(t / 3)
        clock = 0     -> background       completely static

    There is no single universal clock: that is the point of this class.
    """

    name: str
    track: ExposureTrack
    clock: Fraction = Fraction(1)
    loop: LoopPolicy = LoopPolicy.HOLD_LAST
    static: bool = False

    def __post_init__(self) -> None:
        if self.clock < 0:
            raise SpecViolation(f"layer {self.name!r}: clock must be >= 0")

    @property
    def is_static(self) -> bool:
        return self.static or self.clock == 0

    def local_frame(self, frame: int) -> int:
        if self.is_static:
            return 0
        return int(frame * self.clock)

    def drawing_at(self, frame: int) -> str:
        return self.track.drawing_at(self.local_frame(frame), loop=self.loop)


# ---------------------------------------------------------------------------
# The exposure sheet
# ---------------------------------------------------------------------------


@dataclass
class ExposureSheet:
    """A whole shot: drawings, layers, and when each layer is exposed.

        DRAWINGS  ->  TIMING  ->  EXPOSURE  ->  FINAL VIDEO
    """

    sheet_id: str
    drawings: Dict[str, Drawing] = field(default_factory=dict)
    layers: List[Layer] = field(default_factory=list)
    fps: int = 24
    notes: str = ""

    # -- assembly -----------------------------------------------------------

    def add_drawing(self, drawing: Drawing) -> Drawing:
        self.drawings[drawing.id] = drawing
        return drawing

    def add_layer(self, layer: Layer) -> Layer:
        self.layers.append(layer)
        return layer

    def layer(self, name: str) -> Layer:
        for lay in self.layers:
            if lay.name == name:
                return lay
        raise KeyError(f"no layer named {name!r}")

    # -- timing -------------------------------------------------------------

    @property
    def total_frames(self) -> int:
        """Frames needed to fully expose every layer on its own clock."""
        if not self.layers:
            return 0
        longest = 0
        for lay in self.layers:
            if lay.is_static or lay.clock == 0:
                longest = max(longest, 1)
                continue
            longest = max(longest, int(lay.track.total_frames / lay.clock))
        return longest

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.fps

    def frame_state(self, frame: int) -> Dict[str, str]:
        """Which drawing is on every layer at `frame`. These are the cels."""
        return {lay.name: lay.drawing_at(frame) for lay in self.layers}

    def timeline(self) -> List[Dict[str, str]]:
        return [self.frame_state(f) for f in range(self.total_frames)]

    def fit_to_frames(self, total_frames: int) -> "ExposureSheet":
        """Rescale every clocked layer so the sheet lasts exactly `total_frames`."""
        new_layers: List[Layer] = []
        for lay in self.layers:
            if lay.is_static or lay.clock == 0:
                new_layers.append(lay)
                continue
            target = int(round(total_frames * lay.clock))
            target = max(1, target)
            new_layers.append(replace(lay, track=lay.track.scaled_to(target)))
        self.layers = new_layers
        return self

    # -- what actually has to be drawn --------------------------------------

    def render_plan(self) -> List[Drawing]:
        """The drawings that require generation, in first-exposure order.

        This is the payoff of the whole architecture: for animation shot on
        twos, this is roughly half the frames, and held frames cost nothing.
        """
        order: List[str] = []
        for lay in self.layers:
            for drawing_id in lay.track.drawings:
                if drawing_id not in order:
                    order.append(drawing_id)
        plan = []
        for drawing_id in order:
            drawing = self.drawings.get(drawing_id)
            if drawing is None:
                raise SpecViolation(
                    f"drawing {drawing_id!r} is exposed by a layer but is not declared "
                    f"in sheet {self.sheet_id!r}"
                )
            if drawing.needs_generation:
                plan.append(drawing)
        return plan

    def redraw_plan(self) -> List[Drawing]:
        """Drawings the redrawer must produce, in first-exposure order.

        These cost no generator calls at all -- they are warped from keys whose
        flow has already been computed.
        """
        order: List[str] = []
        for lay in self.layers:
            for drawing_id in lay.track.drawings:
                if drawing_id not in order:
                    order.append(drawing_id)
        return [self.drawings[d] for d in order if self.drawings[d].needs_redraw]

    def cost_report(self) -> Dict[str, Any]:
        frames = self.total_frames
        generated = len(self.render_plan())
        redrawn = len(self.redraw_plan())
        reused = max(0, frames - generated - redrawn)
        return {
            "sheet_id": self.sheet_id,
            "fps": self.fps,
            "frames": frames,
            "generation_calls": generated,
            "redraw_calls": redrawn,
            "held_or_reused_frames": reused,
            "drawings_per_second": round(generated / (frames / self.fps), 2) if frames else 0.0,
            "reduction_vs_on_ones": round(1 - (generated / frames), 3) if frames else 0.0,
        }

    # -- the contract -------------------------------------------------------

    def validate(self, strict: bool = True) -> List[str]:
        """Check INV-1..INV-4. Returns warnings; raises SpecViolation on errors."""
        warnings: List[str] = []

        if not self.layers:
            raise SpecViolation(f"sheet {self.sheet_id!r} has no layers")
        if self.fps < 1:
            raise SpecViolation("fps must be >= 1")

        # Every exposed drawing must be declared.
        for lay in self.layers:
            for drawing_id in lay.track.drawings:
                if drawing_id not in self.drawings:
                    raise SpecViolation(
                        f"layer {lay.name!r} exposes undeclared drawing {drawing_id!r}"
                    )

        # INV-1: no chained drawings, and anchors must be fixed roots.
        for drawing in self.drawings.values():
            if drawing.source is DrawingSource.CHAINED:
                raise AutoregressiveChainError(
                    f"drawing {drawing.id!r} is CHAINED. A drawing may not be derived "
                    f"from the previous drawn frame (INV-1). Generate it from a state "
                    f"(AUTHORED) or anchor it to a fixed root (ANCHORED)."
                )
            if drawing.source is DrawingSource.ANCHORED:
                if not drawing.anchor:
                    raise SpecViolation(
                        f"drawing {drawing.id!r} is ANCHORED but names no anchor"
                    )
                if drawing.anchor not in self.drawings:
                    raise SpecViolation(
                        f"drawing {drawing.id!r} anchors to unknown drawing "
                        f"{drawing.anchor!r}"
                    )
                root = self.drawings[drawing.anchor]
                if root.source is DrawingSource.ANCHORED:
                    raise AutoregressiveChainError(
                        f"drawing {drawing.id!r} anchors to {root.id!r}, which is itself "
                        f"ANCHORED. That is a chain in disguise (INV-1): anchor to the "
                        f"AUTHORED root {root.anchor!r} instead."
                    )
                if root.source is DrawingSource.CHAINED:
                    raise AutoregressiveChainError(
                        f"drawing {drawing.id!r} anchors to CHAINED drawing {root.id!r}"
                    )
            if drawing.source is DrawingSource.REDRAWN:
                if not drawing.redraw_from or len(drawing.redraw_from) != 2:
                    raise SpecViolation(
                        f"drawing {drawing.id!r} is REDRAWN but does not name exactly two keys"
                    )
                for anchor_id in drawing.redraw_from:
                    if anchor_id not in self.drawings:
                        raise SpecViolation(
                            f"drawing {drawing.id!r} is redrawn from unknown drawing "
                            f"{anchor_id!r}"
                        )
                    root = self.drawings[anchor_id]
                    if root.source in (DrawingSource.CHAINED, DrawingSource.REDRAWN):
                        raise AutoregressiveChainError(
                            f"drawing {drawing.id!r} is redrawn from {root.id!r}, which is "
                            f"itself {root.source.value}. A redraw must be anchored to fixed "
                            f"keys, never to another redraw (INV-1)."
                        )

        # INV-2/INV-3: holds are the timing, and they must be whole frames.
        for lay in self.layers:
            if any(h.frames < 1 for h in lay.track.holds):
                raise SpecViolation(f"layer {lay.name!r} has a zero-length hold")

        # INV-4 is structural (Layer.clock); warn when everything shares one clock.
        clocks = {lay.clock for lay in self.layers if not lay.is_static}
        if len(self.layers) > 1 and len(clocks) == 1 and len(clocks) > 0:
            warnings.append(
                "all layers share one clock -- INV-4 permits this, but limited "
                "animation is where the technique lives"
            )

        if strict and self.total_frames < 1:
            raise SpecViolation(f"sheet {self.sheet_id!r} exposes zero frames")

        return warnings

    # -- inspection ---------------------------------------------------------

    def to_ascii(self, max_frames: Optional[int] = None, marks: str = ".:-=+*#%@") -> str:
        """Render the exposure sheet the way a dope sheet is read."""
        frames = self.total_frames if max_frames is None else min(max_frames, self.total_frames)
        names = [lay.name for lay in self.layers]
        width = max([len(n) for n in names] + [8])
        header = "frame | " + " | ".join(n.ljust(width) for n in names)
        rule = "-" * len(header)

        # Assign each distinct drawing id per layer a stable single character.
        glyphs: Dict[str, Dict[str, str]] = {}
        for lay in self.layers:
            table = {}
            for i, drawing_id in enumerate(lay.track.drawings):
                table[drawing_id] = marks[i % len(marks)]
            glyphs[lay.name] = table

        lines = [header, rule]
        for f in range(frames):
            cells = []
            for lay in self.layers:
                drawing_id = lay.drawing_at(f)
                cells.append((glyphs[lay.name][drawing_id] + " " + drawing_id).ljust(width))
            lines.append(f"{f:5d} | " + " | ".join(cells))
        lines.append(rule)
        for lay in self.layers:
            legend = ", ".join(f"{v}={k}" for k, v in glyphs[lay.name].items())
            clock = "static" if lay.is_static else f"clock={lay.clock}"
            lines.append(f"{lay.name} ({clock}): {legend}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_version": SPEC_VERSION,
            "sheet_id": self.sheet_id,
            "fps": self.fps,
            "notes": self.notes,
            "drawings": {
                d.id: {
                    "prompt": d.prompt,
                    "source": d.source.value,
                    "anchor": d.anchor,
                    "state_key": d.state_key,
                    "note": d.note,
                }
                for d in self.drawings.values()
            },
            "layers": [
                {
                    "name": lay.name,
                    "clock": [lay.clock.numerator, lay.clock.denominator],
                    "loop": lay.loop.value,
                    "static": lay.is_static,
                    "holds": [[h.drawing, h.frames] for h in lay.track.holds],
                }
                for lay in self.layers
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExposureSheet":
        sheet = cls(
            sheet_id=data["sheet_id"],
            fps=int(data.get("fps", 24)),
            notes=data.get("notes", ""),
        )
        for drawing_id, d in data.get("drawings", {}).items():
            sheet.add_drawing(
                Drawing(
                    id=drawing_id,
                    prompt=d.get("prompt", ""),
                    source=DrawingSource(d.get("source", "authored")),
                    anchor=d.get("anchor"),
                    state_key=d.get("state_key"),
                    note=d.get("note", ""),
                )
            )
        for lay in data.get("layers", []):
            num, den = lay.get("clock", [1, 1])
            sheet.add_layer(
                Layer(
                    name=lay["name"],
                    track=ExposureTrack.mixed([(h[0], int(h[1])) for h in lay["holds"]]),
                    clock=Fraction(int(num), int(den)),
                    loop=LoopPolicy(lay.get("loop", "hold_last")),
                    static=bool(lay.get("static", False)),
                )
            )
        return sheet

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Bridge from the legacy "a list of approved keyframes" representation
# ---------------------------------------------------------------------------


def sheet_from_keyframes(
    sheet_id: str,
    keyframe_paths: Sequence[str],
    duration_frames: int,
    fps: int = 24,
    exposure: Sequence[int] = (),
    prompts: Sequence[str] = (),
) -> ExposureSheet:
    """Build a sheet from approved keyframes.

    Each keyframe becomes a Drawing. The first is AUTHORED (the fixed root); the
    rest are ANCHORED to that root rather than to their immediate predecessor,
    which is the difference between anchored continuity and autoregressive
    drift.

    `exposure` optionally gives the held frame count for each keyframe. When it
    is omitted the timing is uniform, reproducing the legacy behaviour -- the
    door is open, but nothing is silently changed.
    """
    n = len(keyframe_paths)
    if n == 0:
        raise SpecViolation("need at least one keyframe")
    if exposure and len(exposure) != n:
        raise SpecViolation(f"exposure has {len(exposure)} entries but there are {n} keyframes")

    sheet = ExposureSheet(sheet_id=sheet_id, fps=fps)
    ids = [f"kf{i}" for i in range(n)]
    for i, path in enumerate(keyframe_paths):
        sheet.add_drawing(
            Drawing(
                id=ids[i],
                prompt=prompts[i] if i < len(prompts) else "",
                source=DrawingSource.AUTHORED if i == 0 else DrawingSource.ANCHORED,
                anchor=None if i == 0 else ids[0],
                state_key=_path_stem(path),
                note=str(path),
            )
        )

    if exposure:
        track = ExposureTrack.mixed(list(zip(ids, [int(e) for e in exposure])))
    else:
        base = max(1, duration_frames // n)
        counts = [base] * n
        counts[-1] += duration_frames - sum(counts)
        track = ExposureTrack.mixed(list(zip(ids, counts)))

    sheet.add_layer(Layer(name="main", track=track.scaled_to(duration_frames)))
    return sheet


def _path_stem(path: str) -> str:
    """Filename stem, without dragging pathlib into the hot path."""
    tail = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    return tail.rsplit(".", 1)[0] if "." in tail else tail
