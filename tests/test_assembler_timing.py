"""
Integration tests: the Assembler must obey the exposure sheet.

These stub out `_generate_frames` so the timing contract can be tested without
loading any models and without writing a video file.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ps = pytest.importorskip("production_system", reason="pipeline dependencies unavailable")

from animation_spec import AutoregressiveChainError, ExposureSheet, SpecViolation


@pytest.fixture
def workdir():
    """A scratch directory inside the workspace.

    pytest's own tmp_path lives outside the sandbox-writable area here, so these
    tests bring their own.
    """
    path = Path(__file__).resolve().parent.parent / "output" / "pytest_timing_scratch"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _make_keyframes(workdir: Path, count: int) -> list:
    paths = []
    for i in range(count):
        path = workdir / f"kf{i}.png"
        Image.fromarray(np.full((8, 8, 3), i * 20 % 256, dtype=np.uint8)).save(path)
        paths.append(str(path))
    return paths


def _assembler(workdir: Path, calls: list) -> "ps.Assembler":
    """An Assembler with model loading skipped and the generator stubbed."""
    asm = ps.Assembler.__new__(ps.Assembler)  # bypass __init__: no model loading
    asm.output_dir = workdir
    asm.device = "cpu"
    asm.visuals = None
    asm.strict_timing = False
    asm.timing_reports = []

    def fake_generate(img1_path, img2_path, prompt, duration, fps=24):
        n = int(duration * fps)
        calls.append({"img1": img1_path, "img2": img2_path, "prompt": prompt, "frames": n})
        return [np.full((8, 8, 3), 128, dtype=np.uint8) for _ in range(n)]

    asm._generate_frames = fake_generate
    return asm


def _shot(paths):
    return {"keyframes": list(paths), "action_sequence": ["a", "b", "c", "d"][: len(paths)]}


def test_even_timing_matches_the_legacy_shape(workdir):
    paths = _make_keyframes(workdir, 4)
    calls = []
    asm = _assembler(workdir, calls)

    sheet = asm.build_exposure_sheet(0, _shot(paths), total_frames=24)
    assert sheet.total_frames == 24
    assert sheet.layer("main").track.run_lengths() == (6, 6, 6, 6)

    frames = asm._render_sheet(sheet)
    assert len(frames) == 24


def test_authored_exposure_controls_the_holds(workdir):
    paths = _make_keyframes(workdir, 4)
    calls = []
    asm = _assembler(workdir, calls)

    shot = _shot(paths)
    shot["exposure"] = [12, 4, 4, 4]
    sheet = asm.build_exposure_sheet(0, shot, total_frames=24)
    assert sheet.layer("main").track.run_lengths() == (12, 4, 4, 4)
    assert len(asm._render_sheet(sheet)) == 24


def test_rendered_frame_count_is_always_exact(workdir):
    paths = _make_keyframes(workdir, 5)
    for total in [5, 9, 17, 24, 41, 97, 120]:
        calls = []
        asm = _assembler(workdir, calls)
        sheet = asm.build_exposure_sheet(0, _shot(paths), total_frames=total)
        frames = asm._render_sheet(sheet)
        # No pad step, no trim step: the sheet has to be right up front (INV-2).
        assert len(frames) == total, f"failed at total={total}"


def test_transitions_cost_generations_and_holds_do_not(workdir):
    paths = _make_keyframes(workdir, 4)
    calls = []
    asm = _assembler(workdir, calls)

    shot = _shot(paths)
    shot["exposure"] = [10, 2, 10, 2]  # a long opening and closing hold
    sheet = asm.build_exposure_sheet(0, shot, total_frames=24)
    asm._render_sheet(sheet)

    # INV-5: n-1 transitions for n drawings, regardless of frame count.
    assert len(calls) == 3
    assert sum(c["frames"] for c in calls) == 10 + 2 + 10  # the terminal hold is free
    assert sheet.cost_report()["generation_calls"] == 4


def test_single_keyframe_is_a_pure_hold_and_generates_nothing(workdir):
    paths = _make_keyframes(workdir, 1)
    calls = []
    asm = _assembler(workdir, calls)

    sheet = asm.build_exposure_sheet(0, _shot(paths), total_frames=24)
    frames = asm._render_sheet(sheet)

    assert len(frames) == 24
    assert calls == []  # INV-3: stillness costs nothing


def test_assembler_never_produces_a_chained_drawing(workdir):
    """The regression guard for INV-1.

    The legacy keyframe loop fed `keyframes[-1]` back into img2img, which is a
    chain. Sheets built from keyframes must anchor to the root instead.
    """
    paths = _make_keyframes(workdir, 6)
    calls = []
    asm = _assembler(workdir, calls)

    sheet = asm.build_exposure_sheet(0, _shot(paths), total_frames=48)
    sheet.validate()  # must not raise
    assert sheet.drawings["kf0"].source.value == "authored"
    assert all(sheet.drawings[f"kf{i}"].anchor == "kf0" for i in range(1, 6))


def test_mismatched_authored_exposure_falls_back_to_even_timing(workdir, capsys):
    paths = _make_keyframes(workdir, 3)
    calls = []
    asm = _assembler(workdir, calls)

    shot = _shot(paths)
    shot["exposure"] = [1, 1]  # wrong length for 3 keyframes
    sheet = asm.build_exposure_sheet(0, shot, total_frames=24)

    assert sheet.total_frames == 24
    assert "ignoring it" in capsys.readouterr().out


def test_authored_exposure_is_rescaled_to_the_audio(workdir, capsys):
    paths = _make_keyframes(workdir, 3)
    calls = []
    asm = _assembler(workdir, calls)

    shot = _shot(paths)
    shot["exposure"] = [5, 5, 5]  # sums to 15, audio needs 24
    sheet = asm.build_exposure_sheet(0, shot, total_frames=24)

    assert sheet.total_frames == 24
    assert "rescaling" in capsys.readouterr().out


def test_cost_report_is_recorded_for_every_shot(workdir):
    paths = _make_keyframes(workdir, 4)
    calls = []
    asm = _assembler(workdir, calls)

    shot = _shot(paths)
    shot["exposure"] = [6, 6, 6, 6]
    sheet = asm.build_exposure_sheet(0, shot, total_frames=24)
    asm.timing_reports.append(sheet.cost_report())

    report = asm.timing_reports[-1]
    assert report["frames"] == 24
    assert report["generation_calls"] == 4
    assert report["held_or_reused_frames"] == 20
