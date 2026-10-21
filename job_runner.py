"""
job_runner.py -- detached jobs with structured progress.

Every long step in this pipeline (the ensemble story ~4-7 min, generation ~9-14
min per shot, RAFT+shake+grade ~2-3 min) has to run OUT of the UI process. Not as
a workaround for streamlit's blocking rerun: only one render at a time fits in
15.3 GB, and a UI must not die with a crashed job. So:

    start(task, **params)   spawns `job_runner.py run <job_id>` and returns
    status()                reads the job's state.json -- never blocks
    stop(job_id)            asks it to stop, then kills the tree if it will not

The supervisor process owns one job: it runs the real command with its stdout
going straight to the job's log file (**no pipes** -- the sandbox forbids piped
stdio in confined modes, and a file is simpler anyway), polls that log on a timer,
parses the pipeline's own printed output into a structured `progress` dict, and
writes `state.json` until the child exits. The numbers are the pipeline's; nothing
here re-derives a measurement.

One job at a time is a hard rule, enforced with a lock file: the machine cannot
run two renders and the second would silently thrash.

    python job_runner.py status
    python job_runner.py run <job_id>          # the supervisor, spawned for you
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / "output" / "jobs"
LOCK = JOBS / ".current"

# statuses
QUEUED, RUNNING, DONE, FAILED, STOPPED, LOST = (
    "queued", "running", "done", "failed", "stopped", "lost")

# how long the supervisor waits between log polls
POLL_S = 1.5

# A supervisor writes its pid within a second or two of being spawned. If a job is
# still QUEUED with no pid well after that, its supervisor never started -- the
# parent's process tree was torn down before it could (this happens when a job is
# started from a short-lived command rather than from the long-lived UI). Without
# this the job would sit in `queued` and hold the lock FOREVER, blocking every
# later job. Generous, because a false "never started" would be wrong.
QUEUED_GRACE_S = 45


# --------------------------------------------------------------------------- #
# what can be run
# --------------------------------------------------------------------------- #
#
# `progress(line, prog)` folds one line of the pipeline's own output into the
# structured dict. Permissive on purpose: an unrecognised line changes nothing,
# so a new printout in the pipeline degrades to "running" rather than breaking.

def _render_progress(line: str, prog: Dict) -> None:
    if "=== GENERATING ===" in line:
        prog.update(phase="generating")
    elif "=== GRADING ===" in line:
        prog.update(phase="grading")
    elif "=== ASSEMBLING ===" in line:
        prog.update(phase="assembling")
    m = re.match(r"\s*(\d+)/(\d+)\s+(\S+)\s+(\d+)s", line)
    if m:
        prog.update(done=int(m.group(1)), total=int(m.group(2)),
                    last=f"shot {m.group(1)}/{m.group(2)} "
                         f"({m.group(3)}, {m.group(4)}s)")
    m = re.search(r"wrote .*film\.mp4", line)
    if m:
        prog.update(phase="finished")


def _story_progress(line: str, prog: Dict) -> None:
    m = re.search(r"farm (\S+).*?(\d+) ideas in (\d+)s", line)
    if m:
        prog["ideas"] = prog.get("ideas", 0) + int(m.group(2))
        prog.update(phase="ideas",
                    last=f"{m.group(1)}: {m.group(2)} ideas in {m.group(3)}s")
    # per-farm progress, so a 3-minute serial loop is visibly alive rather than silent
    m = re.search(r"\[farm (\d+)/(\d+)\] (\S+) -> (\d+) ideas", line)
    if m:
        prog.update(phase="ideas", farm_done=int(m.group(1)),
                    farm_total=int(m.group(2)), done=int(m.group(1)),
                    total=int(m.group(2)),
                    last=f"farm {m.group(1)}/{m.group(2)} done "
                         f"({m.group(3)}, {m.group(4)} ideas)")
        return
    m = re.search(r"\[farm (\d+)/(\d+)\] (\S+) \.\.\.", line)
    if m:
        prog.update(phase="ideas", farm_done=int(m.group(1)) - 1,
                    farm_total=int(m.group(2)), done=int(m.group(1)) - 1,
                    total=int(m.group(2)),
                    last=f"farm {m.group(1)}/{m.group(2)}: {m.group(3)} thinking")
        return
    m = re.search(r"showrunner attempt (\d+): (\d+) shots, (\d+) story break", line)
    if m:
        prog.update(phase="showrunner", attempt=int(m.group(1)),
                    shots=int(m.group(2)), story_breaks=int(m.group(3)),
                    last=f"attempt {m.group(1)}: {m.group(2)} shots, "
                         f"{m.group(3)} story break(s)")
    if "complete story" in line or "clean story shape" in line:
        prog.update(phase="story clean")
    if "showrunner attempt" in line and "shots," not in line:
        prog.update(phase="showrunner")


def _audit_progress(line: str, prog: Dict) -> None:
    m = re.search(r"farm (\S+).*?(\d+) ideas", line)
    if m:
        prog.update(phase="farms", last=f"{m.group(1)}: {m.group(2)} ideas")
    if "prompts carrying leaked fields" in line:
        prog.update(phase="report")


def _selftest_progress(line: str, prog: Dict) -> None:
    m = re.search(r"TICK (\d+)/(\d+)", line)
    if m:
        prog.update(phase="ticking", done=int(m.group(1)), total=int(m.group(2)),
                    last=f"tick {m.group(1)}/{m.group(2)}")


def _store_progress(line: str, prog: Dict) -> None:
    """Progress for a production render (`make_film --store`)."""
    m = re.search(r"=== FROM STORE (\S+) ===", line)
    if m:
        prog.update(phase="planning", store=m.group(1))
    m = re.search(r"this run: (\d+) to generate, (\d+) to re-finish", line)
    if m:
        prog.update(to_generate=int(m.group(1)), to_refinish=int(m.group(2)),
                    total=int(m.group(1)) + int(m.group(2)),
                    last=f"{m.group(1)} to generate, {m.group(2)} to re-finish")
    # the model load is the longest silent stretch before the first frame, so it gets
    # a phase of its own rather than sharing one with generation
    if "[base] " in line or "preflight:" in line:
        prog.update(phase="loading the model",
                    last="loading the model and its motion LoRAs")
    if "=== GENERATING" in line:
        prog.update(phase="generating")
    if "=== FINISHING ===" in line:
        prog.update(phase="finishing")
    if "=== GRADING ===" in line:
        prog.update(phase="grading")
    if "=== ASSEMBLING ===" in line:
        prog.update(phase="assembling")
    # PER-SHOT WALL CLOCK, which the pipeline already prints. Kept as a list so the
    # UI can show a measured mean and an ETA instead of a fixed guess -- and so the
    # estimate improves as the run goes on rather than staying at 11 minutes forever.
    m = re.search(r"^\s*(\d+)/(\d+) (\S+) +(\S+) +([\d.]+)s -> take", line)
    if m:
        secs = float(m.group(5))
        times = list(prog.get("shot_seconds") or [])
        times.append(secs)
        prog.update(done=int(m.group(1)), total=int(m.group(2)),
                    shot_seconds=times,
                    mean_shot_seconds=round(sum(times) / len(times), 1),
                    last=f"generated {m.group(1)}/{m.group(2)} "
                         f"({m.group(3)}, {secs:.0f}s)")
    m = re.search(r"^\s*(\d+)\. (\S+) +take (\S+)", line)
    if m:
        prog.update(finishing_shot=int(m.group(1)),
                    last=f"finishing take {m.group(3)}")
    if "wrote" in line and "film.mp4" in line:
        prog.update(phase="finished")


def _draft_progress(line: str, prog: Dict) -> None:
    """Progress for `draft.py` -- the ensemble, then the adoption into the store."""
    if "[draft]" in line:
        m = re.search(r"\[draft\] wrote (\d+) shots", line)
        if m:
            prog.update(phase="adopted", shots=int(m.group(1)),
                        last=f"adopted {m.group(1)} shots into the production")
            return
        m = re.search(r"\[draft\] proposal validated: (\d+) shots, (\d+) story "
                      r"break\(s\), (\d+) warning", line)
        if m:
            prog.update(phase="validated", shots=int(m.group(1)),
                        story_breaks=int(m.group(2)),
                        last=f"proposal: {m.group(1)} shots, "
                             f"{m.group(2)} story break(s)")
            return
    _story_progress(line, prog)


TASKS: Dict[str, Dict] = {
    "draft": {
        "desc": "write a shot list AND put it in the production you are looking at",
        "cost": "~4-7 min",
        "progress": _draft_progress,
        "argv": lambda p: [sys.executable, "draft.py",
                           "--store", p["store"],
                           "--concept", p.get("concept", "")]
                          + (["--n-shots", str(p["n_shots"])] if p.get("n_shots")
                             else [])
                          + (["--farms", p["farms"]] if p.get("farms") else [])
                          + (["--showrunner", p["showrunner"]]
                             if p.get("showrunner") else [])
                          + (["--fresh"] if p.get("fresh") else [])
                          + (["--temperature", str(p["temperature"])]
                             if p.get("temperature") is not None else []),
    },
    "story": {
        "desc": "ensemble story: idea farms, then the showrunner, re-rolling "
                "while the story shape is broken",
        "cost": "~4-7 min",
        "progress": _story_progress,
        "argv": lambda p: [sys.executable, "story.py", "--writers", "ensemble"]
                          + (["--fresh"] if p.get("fresh", True) else [])
                          + (["--concept", p["concept"]] if p.get("concept") else [])
                          + (["--n-shots", str(p["n_shots"])] if p.get("n_shots")
                             else [])
                          + (["--farms", p["farms"]] if p.get("farms") else [])
                          + (["--showrunner", p["showrunner"]]
                             if p.get("showrunner") else [])
                          + (["--keep-file", p["keep_file"]] if p.get("keep_file")
                             else [])
                          + (["--keep-only"] if p.get("keep_only") else []),
    },
    "render": {
        "desc": "render a film from its FROZEN plan (the plan you approved)",
        "cost": "~9-14 min per shot",
        "progress": _render_progress,
        "argv": lambda p: [sys.executable, "make_film.py",
                           "--from-project", p["project"],
                           "--name", p.get("name") or "film",
                           "--models", p.get("models", "off"),
                           "--looks", p.get("looks", "auto"),
                           "--shake", str(p.get("shake", 1.0))]
                          + (["--no-energy-arc"] if p.get("no_energy_arc") else [])
                          + (["--only", str(p["only"])] if p.get("only") else [])
                          + (["--assemble"] if p.get("assemble") else []),
    },
    "produce": {
        "desc": "render a production from productions/ -- only the shots that need "
                "it, writing a new take each time and never overwriting one",
        "cost": "seconds if every shot is current; ~9-14 min per shot that "
                "needs generating",
        "progress": _store_progress,
        "argv": lambda p: [sys.executable, "make_film.py",
                           "--store", p["store"],
                           "--name", p.get("name") or f"{p['store']}_render",
                           "--shake", str(p.get("shake", 1.0))]
                          + (["--only", str(p["only"])] if p.get("only") else [])
                          + (["--dry-run"] if p.get("dry_run") else []),
    },
    "audit": {
        "desc": "writer audit: fresh proposals, prompt diff, leaks, provenance",
        "cost": "~1-5 min",
        "progress": _audit_progress,
        "argv": lambda p: [sys.executable, "writer_audit.py",
                           "--runs", str(p.get("runs", 1))]
                          + (["--quiet"] if p.get("quiet") else []),
    },
    "selftest": {
        "desc": "the runner's own smoke test: a short job that prints progress",
        "cost": "~10 s",
        "progress": _selftest_progress,
        "argv": lambda p: [sys.executable, "-c",
                           "import time\n"
                           f"n={int(p.get('ticks', 6))}\n"
                           "for i in range(1, n+1):\n"
                           "    print(f'TICK {i}/{n}', flush=True)\n"
                           "    time.sleep(1.2)\n"
                           "print('selftest done')\n"],
    },
}


def describe_tasks() -> List[Dict]:
    return [{"task": k, "desc": v["desc"], "cost": v["cost"]}
            for k, v in TASKS.items() if not k.startswith("_")]


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #

def job_dir(job_id: str) -> Path:
    return JOBS / job_id


def state_path(job_id: str) -> Path:
    return job_dir(job_id) / "state.json"


def log_path(job_id: str) -> Path:
    return job_dir(job_id) / "log.txt"


def read_state(job_id: str) -> Optional[Dict]:
    p = state_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_state(st: Dict) -> bool:
    """Write the job's state. NEVER RAISES -- that is the whole point of this version.

    THE BUG THIS FIXES, and it cost the user two drafts. `tmp.replace(state_path(...))`
    raises `PermissionError: [WinError 5] Access is denied` if ANY other process has the
    target open at that instant, because Windows will not rename over an open handle.
    The UI polls `status()` every 2 seconds, so the target is regularly open -- and when
    the replace raised, the exception escaped `supervise()`, killed the SUPERVISOR, and
    left the actual render running as an orphan that nobody would ever read the output
    of. The UI then reported "the supervisor is gone but never reported finishing", and
    the job looked like it had failed when it was still working.

    It became frequent only when the heartbeat landed: writes went from one per pipeline
    print (every 30-60 s) to every 1.5 s, so the tiny race window stopped being rare.

    So: retry a few times -- the reader releases the handle in milliseconds -- and if the
    atomic replace still will not go, fall back to a DIRECT write, which does not need to
    rename anything. A torn read of a JSON file for one poll is a far smaller problem
    than a dead supervisor and an orphaned render.
    """
    d = job_dir(st["id"])
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    payload = json.dumps(st, indent=2)
    tmp = d / "state.json.tmp"
    target = state_path(st["id"])
    for attempt in range(4):
        try:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(target)
            return True
        except PermissionError:
            # another process (the UI, the reaper) holds the target open. Back off
            # briefly and retry: this clears in milliseconds, so the delays are small.
            # Measured 763 ms for the whole call at 50/100/150/200 ms; at these values
            # a held file costs ~60 ms, which a 1.5 s heartbeat does not notice.
            time.sleep(0.02 * (attempt + 1))
        except Exception:
            break
    try:
        target.write_text(payload, encoding="utf-8")
        return True
    except Exception:
        return False


def _alive(pid: Optional[int]) -> bool:
    """Is that pid still running?

    Two traps here, both hit while building this. `tasklist /FI "PID eq N"` is not
    usable on this machine -- it runs but does not list the process, so every job
    looked dead. And `os.kill(pid, 0)` is a TRAP on Windows: CPython implements
    os.kill with TerminateProcess for any signal other than the console events, so
    "signal 0" KILLS the process instead of probing it. That would have terminated
    the very render this lock exists to protect.

    So: OpenProcess + GetExitCodeProcess, and on any failure report ALIVE. A wrong
    "dead" would release the lock and let a second render start, which is the one
    outcome this function must never cause.
    """
    if not pid:
        return False
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False,
                                int(pid))
            if not h:
                return False
            code = wintypes.DWORD()
            ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
            k32.CloseHandle(h)
            return bool(ok) and code.value == STILL_ACTIVE
        except Exception:
            return True
    try:
        os.kill(pid, 0)                      # POSIX: a real probe
        return True
    except OSError:
        return False


def _age_s(job_id: str) -> float:
    p = state_path(job_id)
    return time.time() - p.stat().st_mtime if p.exists() else 1e9


def reap(st: Dict) -> Dict:
    """Mark a job LOST if its supervisor cannot be running any more.

    Two ways that happens, and the second one is easy to miss: the supervisor was
    alive and has gone without writing a terminal state, OR it never started at all
    -- which is what happens when a job is launched from a short-lived command and
    the parent's process tree is torn down before the supervisor gets going.
    `pid is None` must NOT be treated as "still fine forever", or the job holds the
    lock and every later job is refused.
    """
    if st.get("status") not in (QUEUED, RUNNING):
        return st
    pid = st.get("pid")
    if pid and not _alive(pid):
        st.update(status=LOST, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                  message="the supervisor is gone but never reported finishing "
                          "-- see the log tail")
    elif not pid and _age_s(st["id"]) > QUEUED_GRACE_S:
        st.update(status=LOST, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                  message=f"the supervisor never started (no pid after "
                          f"{QUEUED_GRACE_S}s). Starting a job from a short-lived "
                          f"process does this: the parent's tree is torn down "
                          f"before the supervisor runs. Start jobs from the UI, "
                          f"which outlives them.")
    if st.get("status") == LOST:
        write_state(st)
    return st


def active_job() -> Optional[Dict]:
    """The job holding the lock, if it is still alive. Clears a stale lock."""
    if not LOCK.exists():
        return None
    st = read_state(LOCK.read_text(encoding="utf-8").strip())
    if st is None:
        LOCK.unlink(missing_ok=True)
        return None
    st = reap(st)
    if st.get("status") in (QUEUED, RUNNING):
        return st
    LOCK.unlink(missing_ok=True)
    return None


# --------------------------------------------------------------------------- #
# the API the UI uses
# --------------------------------------------------------------------------- #

def start(task: str, **params) -> str:
    """Spawn a detached job. Raises if a job is already running."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; have {sorted(TASKS)}")
    busy = active_job()
    if busy:
        raise RuntimeError(
            f"{busy['task']} is already running (job {busy['id']}, "
            f"{busy.get('progress', {}).get('last') or busy['status']}). Only one "
            f"job at a time: the machine cannot run two renders and the second "
            f"would silently thrash.")
    job_id = time.strftime("%Y%m%d-%H%M%S") + f"-{task}"
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    log_path(job_id).write_text("", encoding="utf-8")
    write_state({"id": job_id, "task": task, "params": params, "status": QUEUED,
                 "created": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": None,
                 "finished": None, "returncode": None, "progress": {},
                 "desc": TASKS[task]["desc"], "cost": TASKS[task]["cost"]})
    JOBS.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(job_id, encoding="utf-8")
    kwargs = {}
    if os.name == "nt":
        # own process group, so a Ctrl-C in the UI cannot reach it
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    sup = d / "supervisor.txt"
    with sup.open("ab") as fh:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                          "run", job_id],
                         cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, **kwargs)
    return job_id


def status(job_id: Optional[str] = None) -> Dict:
    """Never blocks. `job_id=None` means the active job, else the most recent."""
    if job_id is None:
        a = active_job()
        if a:
            return a
        job_id = latest_id()
        if job_id is None:
            return {"status": "idle", "progress": {}, "message": "no jobs yet"}
    st = read_state(job_id)
    if st is None:
        return {"status": "unknown", "id": job_id, "progress": {},
                "message": "no such job"}
    # LOST requires the SUPERVISOR to be gone (or to have never started). It always
    # writes a terminal state before exiting, so if it is alive the job is still
    # finishing, however dead the command itself looks.
    return reap(st)


def latest_id() -> Optional[str]:
    if not JOBS.exists():
        return None
    dirs = [d for d in JOBS.iterdir() if d.is_dir() and state_path(d.name).exists()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: state_path(d.name).stat().st_mtime).name


def recent(n: int = 8) -> List[Dict]:
    if not JOBS.exists():
        return []
    dirs = [d for d in JOBS.iterdir() if d.is_dir() and state_path(d.name).exists()]
    dirs.sort(key=lambda d: state_path(d.name).stat().st_mtime, reverse=True)
    out = []
    for d in dirs[:n]:
        st = read_state(d.name)
        if st:
            out.append(st)
    return out


def tail(job_id: str, lines: int = 40) -> str:
    p = log_path(job_id)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def stop(job_id: Optional[str] = None, grace_s: float = 8.0) -> Dict:
    """Ask the supervisor to stop, then kill the process tree if it will not."""
    if job_id is None:
        a = active_job()
        job_id = a["id"] if a else latest_id()
    if job_id is None:
        return {"status": "idle"}
    d = job_dir(job_id)
    (d / "stop").write_text("1", encoding="utf-8")
    deadline = time.time() + grace_s
    while time.time() < deadline:
        st = read_state(job_id) or {}
        if st.get("status") in (DONE, FAILED, STOPPED, LOST):
            return st
        time.sleep(0.5)
    st = read_state(job_id) or {}
    child = st.get("child_pid")
    if child and _alive(child):
        try:
            subprocess.run(["taskkill", "/PID", str(int(child)), "/T", "/F"],
                           capture_output=True, timeout=20)
        except Exception:
            pass
    st.update(status=STOPPED, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
              message="stopped by request")
    write_state(st)
    if LOCK.exists() and LOCK.read_text(encoding="utf-8").strip() == job_id:
        LOCK.unlink(missing_ok=True)
    return st


# --------------------------------------------------------------------------- #
# the supervisor
# --------------------------------------------------------------------------- #

def supervise(job_id: str) -> int:
    st = read_state(job_id)
    if st is None:
        print(f"no state for {job_id}", file=sys.stderr)
        return 2
    task = TASKS[st["task"]]
    argv = task["argv"](st.get("params", {}))
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    log = log_path(job_id)
    st.update(status=RUNNING, pid=None,
              started=time.strftime("%Y-%m-%d %H:%M:%S"),
              argv=[str(a) for a in argv])
    write_state(st)
    print(f"[job] {st['task']} -> {' '.join(str(a) for a in argv)}", flush=True)

    with log.open("ab") as fh:
        proc = subprocess.Popen(argv, cwd=str(ROOT), stdout=fh,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
        try:
            return _supervise_loop(st, proc, task, log, d, job_id)
        finally:
            # NEVER LEAVE THE CHILD BEHIND. If anything at all escapes the loop -- and
            # `tmp.replace` raising WinError 5 did exactly that -- the supervisor used to
            # die silently while the child kept rendering into a log nobody would read.
            # Two of the user's drafts ran on as orphans for 20+ minutes that way. A
            # supervisor's last act is to make sure its command is not still running.
            if proc.poll() is None:
                print("[job] supervisor exiting with the child still alive -- "
                      "terminating it", flush=True)
                _kill_tree(proc.pid)


def _kill_tree(pid: int) -> None:
    """End a process and its children. Best effort; never raises."""
    try:
        subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                       capture_output=True, timeout=20)
    except Exception:
        try:
            os.kill(int(pid), 9)
        except Exception:
            pass


def _supervise_loop(st: Dict, proc, task: Dict, log, d, job_id: str) -> int:
    """Poll the child's log into structured progress until it exits."""
    # `pid` is the SUPERVISOR's own pid and `child_pid` is the command's.
    # Recording the child's pid as the job's pid was a real bug: when the child
    # exited, and before the supervisor had written the terminal state, the UI
    # saw a live status with a dead pid and declared the job LOST -- releasing
    # the lock during the finish window, which is the one thing the lock is
    # there to prevent.
    st["pid"] = os.getpid()
    st["child_pid"] = proc.pid
    # A HEARTBEAT, and the progress bars need it. The supervisor writes state
    # every POLL_S while the child runs, but the state carried no clock, so the
    # UI could not tell a job that had been rendering for eight minutes from one
    # that finished a moment ago. With `child_started` the UI can compute elapsed
    # time and estimate how far into the current shot it is.
    st["child_started"] = time.time()
    st["child_started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_state(st)
    print(f"[job] supervisor pid {os.getpid()}, child pid {proc.pid}",
          flush=True)
    pos, prog = 0, dict(st.get("progress") or {})
    stop_flag = d / "stop"

    while proc.poll() is None:
        # tail the log, parse only what is new
        try:
            with log.open("r", encoding="utf-8", errors="replace") as rh:
                rh.seek(pos)
                chunk = rh.read()
                pos = rh.tell()
        except Exception:
            chunk = ""
        if chunk:
            for line in chunk.splitlines():
                try:
                    task["progress"](line, prog)
                except Exception:
                    pass
            st["progress"] = prog
            st["log_lines"] = (st.get("log_lines", 0) + len(chunk.splitlines()))
            st["updated"] = time.time()
            write_state(st)
        else:
            # heartbeat even when nothing was printed, so the UI's elapsed clock and
            # ETA keep moving through a long single-shot generation where the pipeline
            # prints nothing for 9-14 minutes
            st["updated"] = time.time()
            write_state(st)
        if stop_flag.exists():
            print("[job] stop requested -- terminating", flush=True)
            _kill_tree(proc.pid)
            st.update(status=STOPPED, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                      message="stopped by request")
            write_state(st)
            if LOCK.exists() and LOCK.read_text(encoding="utf-8").strip() == job_id:
                LOCK.unlink(missing_ok=True)
            return 0
        time.sleep(POLL_S)

    # drain whatever arrived between the last poll and the exit
    try:
        with log.open("r", encoding="utf-8", errors="replace") as rh:
            rh.seek(pos)
            for line in rh.read().splitlines():
                try:
                    task["progress"](line, prog)
                except Exception:
                    pass
    except Exception:
        pass

    rc = proc.returncode
    failed = bool(rc)
    st.update(status=FAILED if failed else DONE, returncode=rc,
              finished=time.strftime("%Y-%m-%d %H:%M:%S"), progress=prog,
              message=("the command exited non-zero -- read the log tail. On "
                       "Windows a killed process also settles as 1."
                       if failed else "finished"))
    write_state(st)
    if LOCK.exists() and LOCK.read_text(encoding="utf-8").strip() == job_id:
        LOCK.unlink(missing_ok=True)
    print(f"[job] exit {rc}", flush=True)
    return rc


# --------------------------------------------------------------------------- #

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["status", "run", "stop", "tasks", "log",
                                     "recent"])
    ap.add_argument("job_id", nargs="?", default=None)
    ap.add_argument("--lines", type=int, default=40)
    a = ap.parse_args()

    if a.what == "tasks":
        for t in describe_tasks():
            print(f"  {t['task']:<10} {t['cost']:<16} {t['desc']}")
    elif a.what == "status":
        st = status(a.job_id)
        print(json.dumps(st, indent=2)[:2000])
    elif a.what == "recent":
        for st in recent():
            print(f"  {st['id']:<28} {st['status']:<8} "
                  f"{(st.get('progress') or {}).get('last') or st.get('message','')}")
    elif a.what == "log":
        print(tail(a.job_id or latest_id() or "", a.lines))
    elif a.what == "stop":
        print(json.dumps(stop(a.job_id), indent=2)[:800])
    elif a.what == "run":
        if not a.job_id:
            raise SystemExit("run needs a job id")
        return supervise(a.job_id)


if __name__ == "__main__":
    sys.exit(main() or 0)
