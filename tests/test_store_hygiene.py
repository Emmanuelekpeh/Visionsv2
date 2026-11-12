"""Silent store-state bugs found by the audit: take aliasing on re-draft (#4) and
false "needs re-grading" for takes recorded before a finishing key existed (#6)."""
import json

import film_store as fs


def _project(tmp_path, monkeypatch, shots=2):
    monkeypatch.setattr(fs, "PRODUCTIONS", tmp_path)
    project = fs.new_project("hygiene", "a test film")
    fs.save(project)
    ids = []
    for _ in range(shots):
        sid = fs.next_id(ids)
        shot = fs.new_shot(sid, prompt=f"a lighthouse at dusk number {sid}")
        fs.save_shot(project["slug"], shot, project)
        ids.append(sid)
    project["timeline"] = list(ids)
    fs.save(project)
    return project


def _write_take(project, shot, name="t01", drop_finish=()):
    """A take as a renderer writes it, optionally minus finishing keys it predates."""
    rec = fs.take_record(project, shot, seconds=600, frames=16, take=name)
    for key in drop_finish:
        rec["finish"].pop(key, None)
    # the legacy take's hash was taken over the finish dict it recorded
    if drop_finish:
        legacy = dict(fs.finish_inputs(project, shot))
        for key in drop_finish:
            legacy.pop(key, None)
        scenes = {s.get("id"): s.get("palette") for s in project.get("scenes") or []}
        rec["rebuild_hash"] = fs.input_hash({
            "finish": legacy, "pixel": fs.pixel_deps(shot),
            "in_s": round(float(fs.in_points(project).get(shot["id"], 0.0)), 4),
            "scenes": scenes})
    d = fs.take_dir(project, shot, name)
    (d / "frames").mkdir(parents=True, exist_ok=True)
    (d / "frames" / "f000.png").write_bytes(b"x")
    (d / "take.json").write_text(json.dumps(rec), encoding="utf-8")
    return rec


def test_archive_takes_moves_them_out_of_the_id_keyed_folder(tmp_path, monkeypatch):
    project = _project(tmp_path, monkeypatch)
    shot = fs.get_shot(project, "s01")
    _write_take(project, shot)
    assert len(fs.list_takes(project, "s01")) == 1

    archive = fs.archive_takes(project, ["s01", "s02"])

    # the new film's shot s01 starts with NO takes, so it cannot inherit the old one
    assert fs.list_takes(project, "s01") == []
    assert fs.next_take_name(project, "s01") == "t01"
    # and nothing was deleted: the render is under the archive
    assert archive is not None
    assert (archive / "s01" / "takes" / "t01" / "take.json").exists()
    assert (archive / "s01" / "takes" / "t01" / "frames" / "f000.png").exists()
    assert (archive / "s01" / "shot.json").exists()


def test_archive_takes_with_nothing_to_move_is_a_noop(tmp_path, monkeypatch):
    project = _project(tmp_path, monkeypatch, shots=1)
    assert fs.archive_takes(project, ["s01"]) is not None   # shot.json exists
    fresh = _project(tmp_path / "x", monkeypatch, shots=0)
    assert fs.archive_takes(fresh, []) is None


def test_two_archives_in_one_second_do_not_collide(tmp_path, monkeypatch):
    project = _project(tmp_path, monkeypatch, shots=1)
    shot = fs.get_shot(project, "s01")
    _write_take(project, shot)
    first = fs.archive_takes(project, ["s01"])
    _write_take(project, shot)
    second = fs.archive_takes(project, ["s01"])
    assert first != second and first.exists() and second.exists()


def test_a_take_from_before_upsample_and_camera_move_is_not_restale(tmp_path,
                                                                    monkeypatch):
    project = _project(tmp_path, monkeypatch, shots=1)
    shot = fs.get_shot(project, "s01")
    _write_take(project, shot, drop_finish=("upsample", "camera_move"))
    shot["take"] = "t01"
    fs.save_shot(project["slug"], shot, project)
    assert fs.shot_state(project, "s01") == "rendered"


def test_legacy_tolerance_stops_where_the_finish_really_changed(tmp_path, monkeypatch):
    project = _project(tmp_path, monkeypatch, shots=1)
    shot = fs.get_shot(project, "s01")
    _write_take(project, shot, drop_finish=("upsample", "camera_move"))
    shot["take"] = "t01"
    fs.save_shot(project["slug"], shot, project)

    # slow motion raised AFTER the old take: that is a real finishing change
    shot["upsample"] = 4
    fs.save_shot(project["slug"], shot, project)
    assert fs.shot_state(project, "s01") == "restale"

    # a finishing camera move added after the old take: also real
    shot["upsample"] = 2
    shot["camera_move"] = {"move": "push", "amount": 38, "settle": 0.6}
    fs.save_shot(project["slug"], shot, project)
    assert fs.shot_state(project, "s01") == "restale"

    # and a re-grade of a take that DOES predate the keys is still caught
    shot["camera_move"] = None
    fs.save_shot(project["slug"], shot, project)
    assert fs.shot_state(project, "s01") == "rendered"
    fs.set_palette(project, shot["scene"], "warm-gold")
    assert fs.shot_state(project, "s01") == "restale"


def test_a_current_take_still_goes_restale_on_a_regrade(tmp_path, monkeypatch):
    project = _project(tmp_path, monkeypatch, shots=1)
    shot = fs.get_shot(project, "s01")
    _write_take(project, shot)
    shot["take"] = "t01"
    fs.save_shot(project["slug"], shot, project)
    assert fs.shot_state(project, "s01") == "rendered"
    fs.set_palette(project, shot["scene"], "cyan-night")
    assert fs.shot_state(project, "s01") == "restale"


# --- #8 concurrent-writer hardening ------------------------------------------ #

def test_write_json_retries_a_transient_windows_handle_race(tmp_path, monkeypatch):
    from pathlib import Path
    target = tmp_path / "project.json"
    real_replace = Path.replace
    calls = {"n": 0}

    def flaky(self, dest):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "Access is denied")   # a reader has it open
        return real_replace(self, dest)

    monkeypatch.setattr(Path, "replace", flaky)
    fs._write_json(target, {"ok": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": 1}
    assert calls["n"] == 3
    assert [p.name for p in tmp_path.iterdir()] == ["project.json"]   # no stray .tmp


def test_write_json_falls_back_to_a_direct_write(tmp_path, monkeypatch):
    from pathlib import Path
    target = tmp_path / "shot.json"
    target.write_text("{}", encoding="utf-8")

    def always(self, dest):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(Path, "replace", always)
    fs._write_json(target, {"edit": "kept"})            # must not raise, must not lose it
    assert json.loads(target.read_text(encoding="utf-8")) == {"edit": "kept"}
    assert [p.name for p in tmp_path.iterdir()] == ["shot.json"]


def test_two_writers_do_not_share_a_temp_file(tmp_path):
    import threading
    target = tmp_path / "project.json"
    errors = []

    def work(n):
        try:
            for i in range(60):
                fs._write_json(target, {"writer": n, "i": i})
        except Exception as exc:                          # noqa: BLE001
            errors.append(repr(exc))

    ts = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert errors == []
    assert "writer" in json.loads(target.read_text(encoding="utf-8"))


# --- #13 one take-folder scan per shot --------------------------------------- #

def test_film_status_scans_each_shots_takes_once_and_agrees_with_shot_state(
        tmp_path, monkeypatch):
    project = _project(tmp_path, monkeypatch, shots=3)
    first = fs.get_shot(project, "s01")
    _write_take(project, first)
    first["take"] = "t01"
    fs.save_shot(project["slug"], first, project)

    calls = {"n": 0}
    real = fs.list_takes

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(fs, "list_takes", counting)
    status = fs.film_status(project)
    assert calls["n"] == 3                       # was 2-3 scans PER shot
    monkeypatch.setattr(fs, "list_takes", real)
    for row in status["rows"]:
        assert row["state"] == fs.shot_state(project, row["shot"])
        assert row["take"] == fs.best_take(project, row["shot"])
    assert status["to_generate"] == 2 and status["states"]["rendered"] == 1


def test_seed_for_a_digitless_id_is_the_same_in_every_process(tmp_path):
    """`hash(str)` is randomised per process (PYTHONHASHSEED), so the studio and the
    render job would have minted different seeds for the same shot."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    code = ("import film_store as fs, sys\n"
            "fs.PRODUCTIONS = __import__('pathlib').Path(sys.argv[1])\n"
            "p = fs.new_project('seedcase'); fs.save(p)\n"
            "s = fs.new_shot('hero'); fs.save_shot(p['slug'], s, p)\n"
            "print(fs.ensure_seed(p, s))")
    seeds = set()
    for hashseed in ("1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONPATH=str(root))
        out = subprocess.run([sys.executable, "-c", code, str(tmp_path / hashseed)],
                             capture_output=True, text=True, env=env, cwd=str(root),
                             timeout=120)
        assert out.returncode == 0, out.stderr[-400:]
        seeds.add(out.stdout.strip().splitlines()[-1])
    assert len(seeds) == 1
