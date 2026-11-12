"""Previs: one still per shot, measured at the cuts, before the render hour.

No generation in these tests -- the point is the measurement contract: two stills
from the same place should not look like a jump cut to the same boundary metrics
the finished-cut tool uses, and two unrelated stills should.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

import previs
from edit_planner import Shot

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "output" / "_test_previs"


def _img(path: Path, colour, stripe=False) -> Path:
    im = Image.new("RGB", (64, 64), colour)
    if stripe:
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 64, 20), fill=(255, 255, 255))
    im.save(path)
    return path


def test_measure_separates_same_place_from_new_place():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    bridge1 = _img(SCRATCH / "b1.png", (40, 50, 60))
    bridge2 = _img(SCRATCH / "b2.png", (42, 52, 62))
    alley = _img(SCRATCH / "a1.png", (200, 120, 40), stripe=True)
    shots = [Shot(prompt="x", scene="s1"), Shot(prompt="y", scene="s1"),
             Shot(prompt="z", scene="s2"), Shot(prompt="w", scene="s2")]
    rep = previs.measure([bridge1, bridge2, alley, alley], shots, scorer=None)
    assert rep["shots"] == 4 and rep["cuts"] == 3
    d = rep["cuts_detail"]
    # same place: tiny histogram/luma change; new place: large
    assert d[0]["hist_jump"] < 0.05
    assert d[1]["hist_jump"] > 0.3
    assert d[1]["same_scene"] is False and d[0]["same_scene"] is True


def test_gate_and_cli_agree_where_a_store_previs_lives():
    """Regression: `previs --store` wrote previs/<slug>/ but gate() read
    previs/<slug>_previs/, so a CLI previs never satisfied the render gate."""
    assert previs.report_path("ego_v5") == (
        previs.OUT / "ego_v5_previs" / "report.json")
    assert previs.name_for_store("ego_v5") == "ego_v5_previs"


def test_continuity_splits_hold_jumps_from_scene_cuts():
    """A jump inside a hold is a defect; a jump at a scene cut is the edit (and,
    with deliberate palettes, style). The render normalises the technical half, so
    the two must never be averaged into one number again."""
    rows = [
        {"same_scene": True, "luma_jump": 1.0, "hist_jump": 0.01,
         "clip_sim": 0.9},
        {"same_scene": True, "luma_jump": 2.0, "hist_jump": 0.02,
         "clip_sim": 0.8},
        {"same_scene": False, "luma_jump": 30.0, "hist_jump": 0.6,
         "clip_sim": 0.5},
    ]
    s = previs.split_continuity(rows)
    assert s["within_scene"] == {"cuts": 2, "luma_jump": 1.5,
                                 "hist_jump": 0.015, "clip_sim": 0.85}
    assert s["across_scenes"]["cuts"] == 1
    assert s["across_scenes"]["luma_jump"] == 30.0


def test_contact_sheet_is_written():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    files = [_img(SCRATCH / f"f{i}.png", (20 * i, 30, 40)) for i in range(3)]
    out = previs._contact_sheet(files, SCRATCH / "contact.png")
    assert out.exists() and out.stat().st_size > 500


class _FakeScorer:
    """CLIP-shaped stand-in: no model, deterministic embeddings."""

    def embed_text(self, text: str):
        import torch
        v = torch.zeros(8)
        for i, ch in enumerate((text or "x").lower()):
            v[i % 8] += (ord(ch) % 7) / 7.0
        return torch.nn.functional.normalize(v, dim=0)

    def embed_images(self, imgs):
        import numpy as np
        import torch
        out = []
        for im in imgs:
            a = np.asarray(im, dtype=np.float32).mean(axis=(0, 1)) / 255.0
            v = torch.tensor([a[0], a[1], a[2], 1.0, a.mean(), a.max(),
                              a.min(), 0.5])
            out.append(torch.nn.functional.normalize(v, dim=0))
        return torch.stack(out)


def test_previs_reports_prompt_alignment_before_the_render():
    """The first rendered film's resolve was a car; its prompt named a car.

    A still per shot (this pass) shows that for seconds instead of the render
    hour -- the numbers go in the report, and the prompts go on the contact
    sheet, so a still that answers a different prompt is visible.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    files = [_img(SCRATCH / f"p{i}.png", (30 + 20 * i, 40, 50)) for i in range(4)]
    shots = [Shot(prompt="a wet footbridge with a woman in a dark coat",
                  scene="s1", beat="establish"),
             Shot(prompt="the same woman walking along the footbridge",
                  scene="s1", beat="develop"),
             Shot(prompt="a second figure under the streetlight on the bridge",
                  scene="s2", beat="turn"),
             Shot(prompt="the two figures facing each other on the bridge",
                  scene="s2", beat="resolve")]
    rep = previs.measure(files, shots, scorer=_FakeScorer())
    assert len(rep["prompt_align"]) == 4
    assert len(rep["prompt_margin"]) == 4
    assert rep["prompt_margin_mean"] is not None
    assert isinstance(rep["warnings"], list)
    # the contact sheet carries the prompt, not just "shot N"
    sheet = previs._contact_sheet(files, SCRATCH / "labelled.png", shots=shots)
    assert sheet.exists() and sheet.stat().st_size > 800


def test_previs_is_a_first_class_studio_job():
    """The studio starts previs as a background job, not a blocked UI call."""
    import job_runner as jr
    assert "previs" in jr.TASKS
    argv = jr.TASKS["previs"]["argv"]({"store": "demo"})
    assert "previs.py" in argv and "--store" in argv and "demo" in argv


def test_gate_blocks_missing_stale_and_measured_breaks(monkeypatch):
    import json
    out = SCRATCH / "previs_out"
    if out.exists():
        import shutil
        shutil.rmtree(out)
    monkeypatch.setattr(previs, "OUT", out)
    monkeypatch.setattr(previs, "fingerprint", lambda project: "fp123")

    project = {"slug": "gate-demo", "settings": {"frames": 16, "fps": 16}}
    g = previs.gate(project)
    assert g["ok"] is False and g["status"] == "missing"

    rep = {"fingerprint": "fp123",
           "cuts_detail": [{"cut": "1->2", "same_scene": True,
                            "hist_jump": 0.3, "clip_sim": 0.9}]}
    path = previs.report_path("gate-demo")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep), encoding="utf-8")
    assert previs.gate(project)["ok"] is True

    rep["fingerprint"] = "other"
    path.write_text(json.dumps(rep), encoding="utf-8")
    assert previs.gate(project)["status"] == "stale"

    rep["fingerprint"] = "fp123"
    rep["cuts_detail"][0]["hist_jump"] = 0.92
    path.write_text(json.dumps(rep), encoding="utf-8")
    g = previs.gate(project)
    assert g["ok"] is False and g["status"] == "failed" and g["reasons"]


def test_film_project_previs_stage(monkeypatch):
    import film_project as fp

    proj = fp.new("previstest", "a woman crosses a bridge")
    raw = {"logline": "x", "shots": [
        {"prompt": "a rain-soaked footbridge at night, a woman stepping on, "
                   "streetlights and fog", "scale": "wide", "camera": "none",
         "seconds": 2.5, "subject": "woman", "scene": "s1", "energy": 0.2,
         "beat": "establish", "purpose": "establish", "link": "open"},
        {"prompt": "the same woman walking along the footbridge, rain and fog",
         "scale": "medium", "camera": "none", "seconds": 1.7, "subject": "woman",
         "scene": "s1", "energy": 0.2, "beat": "develop", "purpose": "master",
         "link": "match-action"},
        {"prompt": "a second figure under the streetlight on the same bridge, fog",
         "scale": "medium", "camera": "none", "seconds": 2.0, "subject": "figure",
         "scene": "s2", "energy": 0.3, "beat": "turn", "purpose": "reaction",
         "link": "scene-cut"},
        {"prompt": "the two figures on the rain-soaked bridge, fog",
         "scale": "close", "camera": "none", "seconds": 1.5, "subject": "figure",
         "scene": "s2", "energy": 0.1, "beat": "resolve", "purpose": "resolve",
         "link": "eyeline"},
    ]}
    fp.shots_from_raw(proj, raw)
    calls = []
    monkeypatch.setattr(previs, "run",
                        lambda shots, name: calls.append((len(shots), name))
                        or {"shots": len(shots), "clip_sim_mean": 0.9})
    fp.previs_project(proj, candidate="as_written", name="previstest")
    assert calls and calls[0][0] == 4
    assert proj["status"]["previs"] == "done"
    assert proj["previs"]["report"]["clip_sim_mean"] == 0.9

