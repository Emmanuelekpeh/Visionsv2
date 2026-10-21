"""Read-only audit of the workbench's data flow.

Renders each stage with streamlit's AppTest and reports, per stage, whether it
raised or rendered, plus the key UI text. Proves the stage->store index mismatch
and the two-copy drift WITHOUT running any generation.

    python recon_test/audit_ui.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# caches must live inside the workspace (see HANDOFF 4.4)
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_home"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch_home"))
os.environ.setdefault("TEMP", str(ROOT / ".piptmp"))
os.environ.setdefault("TMP", str(ROOT / ".piptmp"))

import film_project as fp  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

STAGES = ["STORY", "PLAN", "CAMERA", "GRADE", "RENDER", "TIMELINE", "AUDIT"]
APP = str(ROOT / "ui" / "app.py")


def show(title: str, value) -> None:
    print(f"  {title}: {value}")


def audit_project(name: str) -> None:
    print(f"\n{'=' * 72}\nPROJECT {name}\n{'=' * 72}")
    proj = fp.load(name)
    print("  project shape")
    show("shots", len(proj["shots"]))
    show("frozen", len(proj["plan"].get("frozen") or []))
    show("plan approved", proj["plan"].get("approved"))
    show("film record", proj.get("film"))
    show("status", json.dumps(proj["status"]))

    # ---- the two-copy question: does shots[i] == frozen[i]? --------------- #
    for i, (a, b) in enumerate(zip(proj["shots"], proj["plan"].get("frozen") or [])):
        same_cam = a.get("camera") == b.get("camera")
        same_sec = float(a.get("seconds") or 0) == float(b.get("seconds") or 0)
        same_prompt = a.get("prompt") == b.get("prompt")
        show(f"shot {i}", f"prompt={'same' if same_prompt else 'DIFFERENT'} "
                          f"camera={a.get('camera')!r}->{b.get('camera')!r} "
                          f"{'same' if same_cam else 'DIFFERENT'} "
                          f"seconds={a.get('seconds')}->{b.get('seconds')} "
                          f"{'same' if same_sec else 'DIFFERENT'}")

    # ---- render state: where is it written, where is it read? ------------- #
    print("  render state")
    show("shots[i].render (written by sync_render_state/approve)",
         [ (d.get("render") or {}).get("status") for d in proj["shots"] ])
    show("frozen[i].render (read by the RENDER table)",
         [ (d.get("render") or {}).get("status")
           for d in (proj["plan"].get("frozen") or []) ])

    # ---- camera / grade are POSITIONAL ----------------------------------- #
    print("  positional settings")
    show("camera.overrides keys", list((proj.get("camera") or {}).get("overrides", {})))
    show("grade.looks (indexed by position)",
         (proj.get("grade") or {}).get("looks"))
    frozen = proj["plan"].get("frozen") or []
    scenes_in_frozen_order = [d.get("scene") for d in frozen]
    show("frozen scene order", scenes_in_frozen_order)
    show("grade.looks aligned to frozen order",
         list(zip(scenes_in_frozen_order, (proj.get("grade") or {}).get("looks") or [])))

    # ---- what is on disk, and can the UI find it? ------------------------- #
    films = ROOT / "output" / "film"
    want = (proj.get("film") or {}).get("name") or proj["name"]
    print("  media")
    show("project's film name", want)
    show("output/film/<that name> exists", (films / want).exists())
    if (films / want).exists():
        show("  has stats.json", (films / want / "stats.json").exists())
        show("  has film.mp4", (films / want / "film.mp4").exists())
        show("  per-shot layout",
             sorted(p.name for p in (films / want).iterdir() if p.is_dir()))
        show("  flat layout (f*.png at the root)",
             len(list((films / want).glob("f*.png"))))
    show("every dir under output/film", sorted(p.name for p in films.iterdir()
                                               if p.is_dir()))

    # ---- the UI itself ---------------------------------------------------- #
    print("  stages under AppTest")
    for stage in STAGES:
        at = AppTest.from_file(APP, default_timeout=120)
        at.session_state["stage"] = stage
        try:
            at.run()
        except Exception as exc:                      # noqa: BLE001
            show(stage, f"RAISED {type(exc).__name__}: {exc}")
            continue
        if at.exception:
            show(stage, "EXCEPTION " + "; ".join(str(e.value) for e in at.exception))
            continue
        heads = [h.value for h in at.header]
        errs = [e.value for e in at.error]
        show(stage, f"ok  headers={heads}"
                    + (f"  errors={errs}" if errs else ""))


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["ensemble1_workbench", "pin_test"]):
        try:
            audit_project(name)
        except Exception as exc:                      # noqa: BLE001
            print(f"  FAILED on {name}: {type(exc).__name__}: {exc}")
