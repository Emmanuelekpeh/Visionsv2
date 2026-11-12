"""Studio no longer gates renders on previs, and the studio's thread-safe cache
replaces st.cache_data(ttl) (the KeyError crash)."""
import threading

import pytest

import film_store as fs
import previs

studio = pytest.importorskip("ui.studio")


@pytest.fixture
def seams(monkeypatch):
    started = []
    monkeypatch.setattr(studio.jr, "start",
                        lambda task, **p: started.append((task, p)))
    monkeypatch.setattr(studio.jr, "status", lambda: {"status": "idle"})
    monkeypatch.setattr(fs, "load", lambda slug: {"slug": slug})
    # previs is missing: the gate is CLOSED for any cut that is about to generate
    monkeypatch.setattr(previs, "gate", lambda project: {
        "ok": False, "status": "missing", "reasons": ["previs has not been run"]})
    monkeypatch.setattr(studio.st, "error", lambda *a, **k: None)
    monkeypatch.setattr(studio.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(studio.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(studio.st, "button", lambda *a, **k: False)
    return started


def _status(monkeypatch, to_generate):
    monkeypatch.setattr(fs, "film_status", lambda p: {"to_generate": to_generate})


def test_no_previs_gate_any_render_starts(seams, monkeypatch):
    # previs used to block every generating render (and auto-start itself). It uses
    # the 4-step sampler, so it cannot predict the render: nothing gates on it now
    for to_generate, extra in ((0, {}), (3, {}), (0, {"only": 1})):
        _status(monkeypatch, to_generate)
        seams.clear()
        assert studio.announce("produce", store="film", **extra) is True
        assert [t for t, _ in seams] == ["produce"]      # never "previs"


def test_previs_is_never_started_automatically(seams, monkeypatch):
    _status(monkeypatch, 5)
    studio.announce("produce", store="film")
    assert "previs" not in [t for t, _ in seams]


def test_thumbnail_cache_survives_concurrent_readers(tmp_path):
    """The crash: fragment + main script hit the cache from two threads."""
    from PIL import Image
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(3):
        Image.new("RGB", (32, 32), (i * 40, 10, 10)).save(frames / f"f{i:03d}.png")
    studio.thumb.clear()
    errors = []

    def work():
        try:
            for i in range(200):
                assert studio.thumb(str(tmp_path), 16) is not None
                if i % 40 == 0:
                    studio.thumb.clear()
        except Exception as exc:            # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
