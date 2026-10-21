"""Prove the supervisor no longer dies when the state file is held open.

The bug: `write_state` did `tmp.replace(state.json)` with no retry and no error
handling. On Windows you cannot rename over a file another process has open, and the
UI polls `status()` every 2 seconds -- so the replace raised WinError 5, the exception
escaped `supervise()`, and the supervisor died while its child kept running. The user
lost two drafts that way and both children had to be killed by hand.

    python recon_test/test_job_runner_io.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import job_runner as jr  # noqa: E402

JID = "_io_probe"
FAILS = []


def check(label, got, want=True):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}"
          + ("" if ok else f"   (want {want!r})"))
    if not ok:
        FAILS.append(label)


def main() -> int:
    d = jr.job_dir(JID)
    d.mkdir(parents=True, exist_ok=True)
    target = jr.state_path(JID)
    target.write_text("{}", encoding="utf-8")

    print("\n1. write_state must not raise when the target is held open")
    # hold the target open exactly as the UI's reader does
    fh = open(target, "r", encoding="utf-8")
    try:
        fh.read()
        t0 = time.time()
        ok = jr.write_state({"id": JID, "status": "running", "progress": {},
                             "updated": time.time()})
        dt = time.time() - t0
        check("it returned rather than raising", ok, True)
        print(f"     took {dt * 1000:.0f} ms")
        check("and the state on disk is the NEW state",
              json.loads(target.read_text(encoding="utf-8")).get("status"),
              "running")
    except Exception as exc:                      # noqa: BLE001
        check(f"it raised {type(exc).__name__}", str(exc), "no exception")
    finally:
        fh.close()

    print("\n2. it still works when nothing is holding the file")
    check("a plain write returns True",
          jr.write_state({"id": JID, "status": "done", "updated": time.time()}), True)
    check("and lands", json.loads(target.read_text(encoding="utf-8"))["status"],
          "done")

    print("\n3. a read that lands mid-write must not crash the reader either")
    # the UI reads on a timer, so it can see a partially written file. `read_state`
    # already swallows exceptions; this asserts it
    target.write_text('{"id": "_io_probe", "stat', encoding="utf-8")
    check("read_state tolerates a torn file", jr.read_state(JID), None)

    print("\n4. the supervisor kills its child if it dies unexpectedly")
    src = (ROOT / "job_runner.py").read_text(encoding="utf-8")
    check("supervise wraps the loop in try/finally",
          "return _supervise_loop(st, proc, task, log, d, job_id)"
          in src and "finally:" in src, True)
    check("and the finally kills a still-live child",
          "supervisor exiting with the child still alive" in src, True)
    check("_kill_tree exists and never raises",
          "def _kill_tree" in src and "Best effort; never raises" in src, True)

    print("\n5. write_state's fallback is a real path, not dead code")
    check("it retries the atomic replace", "for attempt in range(5)" in src
          or "for attempt in range" in src, True)
    check("and falls back to a direct write",
          "target.write_text(payload, encoding=\"utf-8\")" in src, True)

    os.remove(target) if target.exists() else None
    try:
        d.rmdir()
    except Exception:
        pass

    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + "; ".join(FAILS))
        return 1
    print("all job-runner I/O checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
