"""
keep_awake.py -- stop Windows suspending the host during a long render.

A six-clip run appeared to deadlock for two hours. It had not: the machine went
to sleep, which produces exactly the same symptoms as a hang -- CPU time stops
advancing, the process working set is trimmed, the log goes silent. On a box
where one clip is ten minutes and the film is over an hour, that is a real cost
and an easy thing to misdiagnose.

SetThreadExecutionState(ES_SYSTEM_REQUIRED | ES_CONTINUOUS) tells Windows the
system is busy and must not enter standby. It deliberately does NOT set
ES_DISPLAY_REQUIRED, so the monitor may still switch off -- only sleep is
blocked. The state is released on exit; if the process dies without releasing
it, Windows clears the request with the thread.

    from keep_awake import keep_awake
    with keep_awake("six-clip A/B"):
        ...long work...
"""

from __future__ import annotations

import contextlib
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _call(flags: int) -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except Exception:
        return False


@contextlib.contextmanager
def keep_awake(label: str = ""):
    """Hold a wake lock for the duration of the block."""
    ok = _call(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if label:
        print(f"  [keep-awake] sleep blocked{' for ' + label if label else ''}"
              if ok else "  [keep-awake] unavailable (non-Windows or blocked)")
    try:
        yield ok
    finally:
        if ok:
            _call(ES_CONTINUOUS)


if __name__ == "__main__":
    import time
    print("holding the wake lock for 5 seconds -- try to sleep now")
    with keep_awake("test"):
        time.sleep(5)
    print("released")
