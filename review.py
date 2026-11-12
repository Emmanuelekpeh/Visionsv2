#!/usr/bin/env python3
"""
review.py -- judge TAKES, not shot lists, and record the verdict.

WHY THIS EXISTS. On `squid` s09/t04 the delivered shot is empty water for about
half its length, and where the diver is in frame it is upside down; on s01/t03 the
diver grows a third hand at frame 12. The system accepted all of it, because
`film_store.shot_state` only knows a take EXISTS, and a contact sheet offers one
flattering frame per shot. The thing a viewer notices first -- is the subject
there, is it human, is it the same creature -- was never measured.

So this module reviews the take, which is the unit that decides whether the video
is watchable, and it is deliberately production-agnostic: it walks
`productions/<slug>/`, reads the chosen take of each shot, and never assumes a
subject, a shot count or a story. Three layers, cheapest first:

  1. NUMBERS (seconds, no model). Motion, sharpness, and a foreground estimate
     from subtracting the take's own per-pixel median. Reported as signals, not a
     verdict -- on a light-shaft shot the foreground can be the shaft, so a number
     alone must not pass or fail a take.
  2. EVIDENCE (seconds). Every frame of the take as a labelled grid with the
     detected subject boxed, so a frame with no subject and a subject at the edge
     are both visible. This is the artifact the contact sheet should have been.
  3. VISION (the arbiter). A local VLM is asked, per sampled frame, the only
     question that matters: is there a figure, how many hands, is it upright. The
     project has had `qwen2.5vl`, `granite3.2-vision`, `llava` and `moondream`
     installed all along and never called one. A recorded verdict -- from the
     model or from a person -- is what the gate uses.

    python review.py --store squid                       # numbers + evidence
    python review.py --store squid --vision qwen2.5vl:3b # + the model's verdict
    python review.py --store squid --shot s09            # every take of one shot
    python review.py --store squid --set-verdict s09 t04 fail --why "no figure"
    python review.py --all                               # every production, table
    python review.py --store squid --gate                # may this be rendered?

Writes `productions/<slug>/review.json` (fingerprinted, so a changed take stales
it) and evidence sheets under `output/review/<slug>/`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import film_store as fs                          # noqa: E402

OUT = ROOT / "output" / "review"

# A frame is "at the edge" when the subject's box touches the border this closely:
# the subject is leaving, or half out.
EDGE_PX = 3
# Below this mean consecutive-frame difference a take is effectively static, so a
# temporal background model cannot separate subject from background.
STATIC_DIFF = 0.35
# A dark blob smaller than this fraction of the frame is not a subject to crop.
MIN_DARK_BLOB = 0.002

# The question the VLM answers. Strict JSON, and the disallowed reading is named,
# because "a fin is a hand" is exactly the mistake to avoid.
VISION_PROMPT = (
    "You are checking one frame of an AI-generated video for a viewer-facing "
    "defect. Answer with JSON only, no prose:\n"
    '{"figure": true or false, "hands": <integer or null>, '
    '"orientation": "upright" or "inverted" or "unclear", '
    '"note": "<10 words>"}\n'
    "Rules: figure is true only if a person/creature body is visible. Count every "
    "visible hand: a flipper, fin, glove or paw is NOT a hand. orientation is "
    "about the figure's head relative to its body. If figure is false, hands=null."
)
DEFAULT_VISION = "qwen2.5vl:3b"

# Words that mean a figure is expected in the shot. This only decides whether an
# absent figure is a DEFECT; the measurement itself is content-agnostic.
FIGURE_WORDS = frozenset("""
man woman person people figure figures silhouette silhouettes diver child boy girl
creature animal human character body someone somebody he she they him her his hers
""".split())


# --------------------------------------------------------------------------- #
# numbers: cheap, and honest about what they can and cannot see
# --------------------------------------------------------------------------- #

def _gray(frames: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]


def dark_foreground(frames: Sequence[np.ndarray]) -> Dict:
    """The subject as the region DARKER than the take's own median background.

    A moving subject is absent from the per-pixel median (it moves), and a figure
    in these films is darker than the water/air behind it. Signed on purpose: the
    unsigned residual lights up a moving light shaft as much as a figure, which is
    exactly how s09's "empty" frames read as occupied. Signed does not.

    Returns per frame the largest dark component's area/box and whether it touches
    the border. It is a SIGNAL: a lit figure can be missed, so nothing is passed or
    failed on this alone.
    """
    if not frames:
        return {"frames": 0, "error": "no frames"}
    grays = _gray(frames)
    h, w = grays[0].shape
    diffs = [float(np.mean(np.abs(grays[i] - grays[i - 1])))
             for i in range(1, len(grays))]
    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    if len(grays) < 3:
        return {"frames": len(grays), "mean_diff": mean_diff,
                "status": "too_few_frames"}
    background = np.median(np.stack(grays), axis=0)
    rows = []
    for im in grays:
        d = background - im                  # positive = darker than background
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med)))
        thr = max(6.0, med + 3.0 * 1.4826 * mad)
        mask = (d > thr).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        if n > 1:
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            x, y = int(stats[k, cv2.CC_STAT_LEFT]), int(stats[k, cv2.CC_STAT_TOP])
            bw, bh = int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT])
            area = int(stats[k, cv2.CC_STAT_AREA])
            cx, cy = float(cents[k][0]), float(cents[k][1])
        else:
            x = y = bw = bh = area = 0
            cx = cy = float("nan")
        edge = bool(area and (x <= EDGE_PX or y <= EDGE_PX
                              or x + bw >= w - EDGE_PX or y + bh >= h - EDGE_PX))
        rows.append({"area_frac": area / float(h * w), "bbox": [x, y, bw, bh],
                     "edge": edge,
                     "centroid": [round(cx, 1), round(cy, 1)] if area else None})
    areas = [r["area_frac"] for r in rows]
    typical = float(np.percentile(areas, 75)) if areas else 0.0
    status = ("static" if mean_diff < STATIC_DIFF
              else "no_subject" if typical < MIN_DARK_BLOB
              else "ok")
    return {"frames": len(rows), "rows": rows, "mean_diff": round(mean_diff, 2),
            "typical_area": round(typical, 4), "status": status}


def signals(measure: Dict) -> Dict:
    """Report the numbers as signals. No pass/fail -- that is the vision pass."""
    if "rows" not in measure:
        return {"status": measure.get("status") or "not_measurable",
                "flags": [], "suspicious": True}
    rows = measure["rows"]
    areas = [r["area_frac"] for r in rows]
    edge = [i for i, r in enumerate(rows) if r["edge"]]
    typical = measure.get("typical_area") or 0.0
    low = [i for i, a in enumerate(areas) if a < 0.35 * typical] if typical else []
    flags: List[str] = []
    if measure.get("status") == "no_subject":
        flags.append("no dark subject found in any frame (could be lit, or absent)")
    if measure.get("status") == "static":
        flags.append("take barely moves; presence not measurable temporally")
    if typical and max(areas) > 3.0 * max(min(areas), 1e-4):
        flags.append(f"foreground area swings {min(areas):.4f}-{max(areas):.4f} "
                     f"across the take")
    if low:
        flags.append(f"foreground smallest in {len(low)}/{len(rows)} frame(s)")
    if len(edge) > 0.25 * len(rows):
        flags.append(f"subject at the edge in {len(edge)}/{len(rows)} frame(s)")
    if measure.get("mean_diff") is not None and measure["mean_diff"] < STATIC_DIFF:
        flags.append(f"very little motion (mean diff {measure['mean_diff']})")
    return {"status": measure.get("status"), "flags": flags,
            "suspicious": bool(flags), "low_frames": low, "edge_frames": edge}


# --------------------------------------------------------------------------- #
# evidence: every frame, subject boxed
# --------------------------------------------------------------------------- #

def crop_for_frame(frame: np.ndarray, bbox: Optional[Sequence[int]],
                   margin: float = 0.55, max_side: int = 640) -> Dict:
    """A tight crop around the detected subject, for the vision model.

    The crop is what makes hand-counting possible: on a whole 512 frame a hand is
    ~15 px and a VLM misses it. The box comes from the dark-foreground measure, not
    from a fresh threshold, so the crop and the evidence sheet agree about what the
    subject is.
    """
    h, w = frame.shape[:2]
    if not bbox or len(bbox) != 4:
        return {"image": frame.copy(), "box": [0, 0, w, h]}
    x, y, bw, bh = [int(v) for v in bbox]
    if bw <= 0 or bh <= 0:
        return {"image": frame.copy(), "box": [0, 0, w, h]}
    mx, my = int(margin * bw), int(margin * bh)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
    im = frame[y0:y1, x0:x1].copy()
    side = max(im.shape[:2])
    if side > max_side:
        s = max_side / float(side)
        im = cv2.resize(im, (max(1, int(im.shape[1] * s)),
                             max(1, int(im.shape[0] * s))),
                        interpolation=cv2.INTER_AREA)
    return {"image": im, "box": [x0, y0, x1, y1]}


def evidence_sheet(shot: Dict, take: Dict, frames: Sequence[np.ndarray],
                   measure: Dict, out_dir: Path) -> Path:
    """All frames of a take, labelled, with the detected subject boxed."""
    tile, cols = 300, 4
    rows_n = (len(frames) + cols - 1) // cols
    canvas = np.full((rows_n * (tile + 22) + 6, cols * (tile + 6) + 6, 3),
                     (18, 18, 20), dtype=np.uint8)
    mrows = (measure.get("rows") or [])
    h, w = frames[0].shape[:2] if frames else (1, 1)
    for k, frame in enumerate(frames):
        r, c = divmod(k, cols)
        x0, y0 = 6 + c * (tile + 6), 6 + r * (tile + 22)
        thumb = cv2.resize(frame, (tile, tile), interpolation=cv2.INTER_AREA)
        canvas[y0:y0 + tile, x0:x0 + tile] = thumb
        if k < len(mrows) and mrows[k].get("bbox"):
            bx, by, bw, bh = mrows[k]["bbox"]
            sx, sy = tile / float(w), tile / float(h)
            cv2.rectangle(canvas, (int(bx * sx), int(by * sy)),
                          (int((bx + bw) * sx), int((by + bh) * sy)),
                          (60, 220, 60), 1)
        label = f"{shot.get('id')}/{take.get('take')} f{k:03d}"
        if k < len(mrows) and mrows[k].get("area_frac") is not None:
            label += f"  area {mrows[k]['area_frac']:.3f}"
            if mrows[k].get("edge"):
                label += " edge"
        cv2.putText(canvas, label, (x0, y0 + tile + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (235, 235, 235), 1,
                    cv2.LINE_AA)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{shot.get('id')}_{take.get('take')}.png"
    cv2.imwrite(str(path), canvas)
    return path


# --------------------------------------------------------------------------- #
# vision: the local VLM, the arbiter
# --------------------------------------------------------------------------- #

def _parse_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        d = json.loads(text[i:j + 1])
        return d if isinstance(d, dict) else None
    except Exception:                                # noqa: BLE001
        return None


def vision_frame(model: str, image_path: Path) -> Dict:
    """One VLM call. Returns the parsed answer and the raw reply as evidence."""
    import ollama
    t0 = time.time()
    try:
        resp = ollama.chat(model=model, messages=[{
            "role": "user", "content": VISION_PROMPT,
            "images": [str(image_path)]}],
            options={"temperature": 0, "num_predict": 160})
        text = resp["message"]["content"]
    except Exception as exc:                         # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}",
                "raw": "", "seconds": round(time.time() - t0, 1)}
    return {"answer": _parse_json(text), "raw": text.strip()[:400],
            "seconds": round(time.time() - t0, 1)}


def _sample_indices(n: int, k: int) -> List[int]:
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def vision_take(shot: Dict, take: Dict, frames: Sequence[np.ndarray],
                measure: Dict, model: str, n_frames: int = 3,
                out_dir: Optional[Path] = None, trusted: bool = False) -> Dict:
    """Ask the VLM about a few frames of a take. EVIDENCE, not a verdict by default.

    MEASURED ON THIS MACHINE (`qwen2.5vl:3b`, tight crops, 2026-11-04):
      * s09/t04 f004 is EMPTY WATER and the model answered `figure: true,
        orientation: inverted` -- it hallucinated the diver it was asked about;
      * s01/t03 f013 has THREE hands and the model answered `hands: 2`;
      * s09/t04 f000 is a real inverted diver and it answered `inverted` -- the
        one thing it got right is orientation;
      * cost: 87-180 s per crop on CPU.
    So the model is recorded as advisory, and `trusted=False` by default: an
    untrusted answer must never pass or fail a take. A stronger model (or a GPU)
    can be trusted explicitly with `--trust-vision`.
    """
    picks = _sample_indices(len(frames), n_frames)
    out_dir = out_dir or (OUT / "frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    mrows = measure.get("rows") or []
    rows = []
    for i in picks:
        bbox = mrows[i].get("bbox") if i < len(mrows) else None
        crop = crop_for_frame(frames[i], bbox)
        p = out_dir / f"{shot.get('id')}_{take.get('take')}_f{i:03d}.png"
        cv2.imwrite(str(p), crop["image"])
        ans = vision_frame(model, p)
        rows.append({"frame": i, "path": str(p), **ans})
    answers = [r.get("answer") or {} for r in rows]
    parsed = [a for a in answers if a]
    figures = [bool(a.get("figure")) for a in parsed]
    hands = [a.get("hands") for a in parsed if isinstance(a.get("hands"), int)]
    orientations = [str(a.get("orientation") or "?") for a in parsed]
    expect = figure_expected(shot)
    notes: List[str] = []
    verdict = None
    if parsed:
        if expect and not all(figures):
            absent = [rows[k]["frame"] for k, a in enumerate(answers)
                      if a and not a.get("figure")]
            notes.append(f"model reports no figure in frame(s) {absent}")
        if hands and max(hands) > 2:
            notes.append(f"model reports {max(hands)} hands")
        if "inverted" in orientations and "upright" not in orientations:
            notes.append("model reads the figure inverted in every sampled frame")
        if not notes:
            notes.append("model reports a figure, two hands, no orientation fault")
        if trusted:
            if expect and not all(figures):
                verdict = "fail"
            elif hands and max(hands) > 2:
                verdict = "fail"
            elif "inverted" in orientations and "upright" not in orientations:
                verdict = "review"
            else:
                verdict = "pass"
    return {"model": model, "frames": [r["frame"] for r in rows],
            "trusted": bool(trusted), "verdict": verdict, "notes": notes,
            "hands": hands, "figure": figures, "orientation": orientations,
            "answers": rows}


def figure_expected(shot: Dict) -> bool:
    """Does this shot name a figure that must be on screen?

    `subject` is the writing layer's own signal; when it is empty, a small generic
    vocabulary decides. Only absence-is-a-defect depends on this.
    """
    if str((shot or {}).get("subject") or "").strip():
        return True
    prompt = str((shot or {}).get("prompt") or "").lower()
    return bool(set(prompt.replace(",", " ").replace(".", " ").split())
                & FIGURE_WORDS)


# --------------------------------------------------------------------------- #
# one take, one film
# --------------------------------------------------------------------------- #

def take_frames(take: Dict) -> List[np.ndarray]:
    d = Path(take.get("_dir") or "")
    out = []
    for f in sorted((d / "frames").glob("f*.png")):
        im = cv2.imread(str(f))
        if im is not None:
            out.append(im)
    return out


def sharpness(frames: Sequence[np.ndarray]) -> float:
    if not frames:
        return 0.0
    mid = frames[len(frames) // 2]
    return round(float(cv2.Laplacian(cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY),
                                     cv2.CV_64F).var()), 1)


def review_take(shot: Dict, take: Dict, evidence_dir: Optional[Path] = None,
                vision_model: Optional[str] = None, vision_frames: int = 3,
                vision_trusted: bool = False) -> Dict:
    frames = take_frames(take)
    measure = dark_foreground(frames)
    sig = signals(measure)
    row = {
        "shot": shot.get("id"), "take": take.get("take"),
        "frames": len(frames), "frames_on_disk": take.get("frames_on_disk"),
        "prompt": shot.get("prompt", ""),
        "expect_figure": figure_expected(shot),
        "sharpness": sharpness(frames), "mean_diff": measure.get("mean_diff"),
        "conditioned": bool(take.get("conditioned")),
        "signals": sig,
    }
    if evidence_dir is not None:
        row["evidence"] = str(evidence_sheet(shot, take, frames, measure,
                                             evidence_dir / shot.get("id")))
    if vision_model and frames:
        row["vision"] = vision_take(shot, take, frames, measure, vision_model,
                                    n_frames=vision_frames, out_dir=evidence_dir,
                                    trusted=vision_trusted)
    return row


def fingerprint(project: Dict) -> str:
    rows = []
    for shot in fs.shots_in_order(project):
        take = fs.best_take(project, shot)
        rows.append({"shot": shot.get("id"), "prompt": shot.get("prompt"),
                     "take": (take or {}).get("take"),
                     "frames": (take or {}).get("frames_on_disk"),
                     "at": (take or {}).get("at")})
    return hashlib.sha1(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]


def _verdict_of(row: Dict) -> str:
    """A recorded verdict wins; then the model's; otherwise the take is UNREVIEWED.

    The numbers are deliberately NOT a verdict. On a light-shaft shot the
    foreground area measures the shaft, and on a lit figure it measures nothing at
    all -- so an automatic pass/fail from it would be a guess dressed as a
    measurement, and a gate that guesses is worse than no gate. The numbers flag
    where to look; a vision verdict is what the gate uses.
    """
    if row.get("verdict"):
        return row["verdict"]
    vision = row.get("vision") or {}
    if vision.get("trusted") and vision.get("verdict"):
        return vision["verdict"]
    if not row.get("expect_figure", True):
        return "skip"
    return "unreviewed"


def review_film(store: str, vision_model: Optional[str] = None,
                vision_frames: int = 3, evidence: bool = True,
                vision_trusted: bool = False) -> Dict:
    project = fs.load(store)
    shots = fs.shots_in_order(project)
    ev = OUT / store
    # A recorded verdict survives a re-run of the measurements -- but only for the
    # same take. Re-shooting the shot invalidates the verdict, because the pixels
    # it judged are gone.
    path = fs.dir_for(store) / "review.json"
    recorded: Dict = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            recorded = {(r.get("shot"), r.get("take")): r
                        for r in old.get("rows") or [] if r.get("verdict_by")}
        except Exception:                            # noqa: BLE001
            recorded = {}
    rows: List[Dict] = []
    print(f"[review] {store}: {len(shots)} shot(s)"
          + (f", vision {vision_model}" if vision_model else ""), flush=True)
    for n, shot in enumerate(shots, 1):
        print(f"[review] shot {n}/{len(shots)} {shot.get('id')}", flush=True)
        take = fs.best_take(project, shot)
        if take is None:
            rows.append({"shot": shot.get("id"), "take": None,
                         "verdict": "skip", "reason": "not rendered",
                         "signals": {"flags": ["no take"]}})
            continue
        row = review_take(shot, take, evidence_dir=ev if evidence else None,
                          vision_model=vision_model,
                          vision_frames=vision_frames,
                          vision_trusted=vision_trusted)
        prev = recorded.get((row["shot"], row["take"]))
        if prev:
            row["verdict"] = prev.get("verdict")
            row["reason"] = list(prev.get("reason") or [])
            row["verdict_by"] = prev.get("verdict_by")
            row["verdict_at"] = prev.get("verdict_at")
        rows.append(row)
    for r in rows:
        if not r.get("verdict"):
            r["verdict"] = _verdict_of(r)
            r["reason"] = (r.get("signals", {}).get("flags") or
                           ["needs a recorded verdict (see evidence sheet)"])
    counts = {v: sum(1 for r in rows if r["verdict"] == v)
              for v in ("fail", "review", "pass", "unreviewed", "skip")}
    rep = {"store": store, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "fingerprint": fingerprint(project), "shots": len(rows),
           "vision_model": vision_model, "verdicts": counts, "rows": rows,
           "failing": [{"shot": r["shot"], "take": r["take"],
                        "why": "; ".join(r["reason"]),
                        "evidence": r.get("evidence")}
                       for r in rows if r["verdict"] == "fail"]}
    path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


def gate(project: Dict) -> Dict:
    slug = project.get("slug") or ""
    path = fs.dir_for(slug) / "review.json"
    if not path.exists():
        return {"ok": False, "status": "missing",
                "reasons": [f"no take review; run `python review.py --store {slug}`"]}
    try:
        rep = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                         # noqa: BLE001
        return {"ok": False, "status": "unreadable",
                "reasons": [f"review.json unreadable: {type(exc).__name__}"]}
    if rep.get("fingerprint") != fingerprint(project):
        return {"ok": False, "status": "stale",
                "reasons": ["a take changed since this review -- re-run it"]}
    reasons = []
    for r in rep.get("rows") or []:
        if r.get("verdict") == "fail":
            reasons.append(f"{r['shot']}/{r['take']}: {'; '.join(r.get('reason') or [])}")
        elif r.get("expect_figure") and r.get("verdict") == "unreviewed" and \
                not r.get("signals", {}).get("status") == "static":
            reasons.append(f"{r['shot']}/{r['take']}: not vision-reviewed")
    if reasons:
        return {"ok": False, "status": "failed", "reasons": reasons[:8]}
    return {"ok": True, "status": "ok", "reasons": [], "path": str(path)}


def set_verdict(store: str, shot_id: str, take: Optional[str], verdict: str,
                why: str = "") -> Dict:
    """Record a person's/agent's verdict on a take -- the durable feedback edge.

    This is how the loop learns from a frame that a metric cannot judge. It is
    stored in review.json beside the measurements, so the gate and the next writer
    pass can read the same decision.
    """
    path = fs.dir_for(store) / "review.json"
    project = fs.load(store)
    if path.exists():
        rep = json.loads(path.read_text(encoding="utf-8"))
    else:
        rep = {"store": store, "fingerprint": fingerprint(project), "rows": []}
    sid = shot_id
    take = take or (fs.best_take(project, fs.get_shot(project, sid)) or {}).get("take")
    found = None
    for r in rep.setdefault("rows", []):
        if r.get("shot") == sid and (r.get("take") == take or take is None):
            found = r
            break
    if found is None:
        found = {"shot": sid, "take": take, "signals": {}, "reason": []}
        rep["rows"].append(found)
    found["verdict"] = verdict
    found["reason"] = [why] if why else []
    found["verdict_by"] = "human"
    found["verdict_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return found


# --------------------------------------------------------------------------- #
# every production (this is the point: not squid-specific)
# --------------------------------------------------------------------------- #

def review_all(vision_model: Optional[str] = None, evidence: bool = False,
               vision_trusted: bool = False) -> List[Dict]:
    out = []
    for d in sorted(fs.PRODUCTIONS.iterdir()):
        if not (d / "project.json").exists():
            continue
        try:
            out.append(review_film(d.name, vision_model=vision_model,
                                   evidence=evidence,
                                   vision_trusted=vision_trusted))
        except Exception as exc:                     # noqa: BLE001
            out.append({"store": d.name,
                        "error": f"{type(exc).__name__}: {str(exc)[:120]}"})
    print_table(out)
    return out


def print_table(reps: Sequence[Dict]) -> None:
    print(f"\n{'production':<18} {'shots':>5} {'fail':>4} {'revi':>4} "
          f"{'pass':>4} {'unrv':>4} {'skip':>4}  worst")
    for r in reps:
        if "error" in r:
            print(f"{r['store']:<18} {r['error']}")
            continue
        v = r["verdicts"]
        worst = r["failing"][0] if r["failing"] else {}
        print(f"{r['store']:<18} {r['shots']:>5} {v['fail']:>4} {v['review']:>4} "
              f"{v['pass']:>4} {v['unreviewed']:>4} {v['skip']:>4}  "
              f"{str(worst.get('why') or '')[:60]}")


def print_film(rep: Dict) -> None:
    v = rep["verdicts"]
    print(f"\n=== take review: {rep['store']} ({rep['shots']} shots)"
          + (f"  vision={rep['vision_model']}" if rep.get("vision_model") else "")
          + " ===")
    print(f"  fail {v['fail']} · review {v['review']} · pass {v['pass']} · "
          f"unreviewed {v['unreviewed']} · skip {v['skip']}")
    print(f"\n  {'shot':<6} {'take':<5} {'verdict':<10} {'sharp':>6} {'diff':>5} "
          f"why")
    for r in rep["rows"]:
        why = "; ".join(r.get("reason") or [])[:78]
        print(f"  {str(r.get('shot')):<6} {str(r.get('take')):<5} "
              f"{r['verdict']:<10} {str(r.get('sharpness')):>6} "
              f"{str(r.get('mean_diff')):>5} {why}")
    for r in rep["rows"]:
        if r.get("evidence"):
            print(f"    evidence {r['shot']}/{r['take']}: {r['evidence']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Judge takes, not shot lists")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--store")
    src.add_argument("--all", action="store_true")
    ap.add_argument("--shot", default=None)
    ap.add_argument("--take", default=None)
    ap.add_argument("--vision", nargs="?", const=DEFAULT_VISION, default=None,
                    help="ask a local VLM per sampled frame (default "
                         f"{DEFAULT_VISION})")
    ap.add_argument("--vision-frames", type=int, default=3)
    ap.add_argument("--trust-vision", action="store_true",
                    help="let the VLM's answer decide pass/fail. OFF by default: "
                         "measured on this machine, qwen2.5vl:3b reported a figure "
                         "on an empty-water frame and 2 hands on a 3-hand frame")
    ap.add_argument("--no-evidence", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--set-verdict", nargs=2, metavar=("SHOT", "VERDICT"),
                    default=None)
    ap.add_argument("--why", default="")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    if a.all:
        reps = review_all(vision_model=a.vision,
                          vision_trusted=a.trust_vision)
        if a.json_out:
            Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.json_out).write_text(json.dumps(reps, indent=2), encoding="utf-8")
        return 0

    project = fs.load(a.store)

    if a.set_verdict:
        shot_id, verdict = a.set_verdict
        row = set_verdict(a.store, shot_id, a.take, verdict, a.why)
        print(f"recorded {a.store} {shot_id}/{row.get('take')}: {verdict}"
              + (f" ({a.why})" if a.why else ""))
        return 0

    if a.gate:
        g = gate(project)
        print(f"review gate: {g['status']}")
        for r in g.get("reasons") or []:
            print(f"  - {r}")
        return 0 if g.get("ok") else 1

    if a.shot:
        shot = fs.get_shot(project, a.shot)
        if shot is None:
            raise SystemExit(f"no shot {a.shot!r} in {a.store}")
        takes = fs.list_takes(project, shot)
        if a.take:
            takes = [t for t in takes if t.get("take") == a.take]
        if not takes:
            raise SystemExit(f"no takes for {a.store}/{a.shot}")
        print(f"\n=== {a.store} {a.shot}: {len(takes)} take(s) ===")
        for t in takes:
            row = review_take(shot, t, evidence_dir=OUT / a.store,
                              vision_model=a.vision,
                              vision_frames=a.vision_frames,
                              vision_trusted=a.trust_vision)
            v = row.get("vision") or {}
            print(f"  {t.get('take'):<4} {_verdict_of(row):<10} "
                  f"sharp {row['sharpness']:>6}  diff {row['mean_diff']}  "
                  f"{'; '.join((v.get('notes') or
                                row['signals'].get('flags') or [])[:3])}")
            if row.get("evidence"):
                print(f"       {row['evidence']}")
        return 0

    rep = review_film(a.store, vision_model=a.vision,
                      vision_frames=a.vision_frames,
                      evidence=not a.no_evidence,
                      vision_trusted=a.trust_vision)
    print_film(rep)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return 1 if rep["verdicts"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
