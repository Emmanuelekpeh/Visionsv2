"""
llm_bench.py -- which local model should write the shot list, and why?

The film pipeline has been taking hand-written shot lists. That skips the whole
text layer, and the text layer is where a system can actually be tuned cheaply:
a shot list costs seconds to regenerate, where the frames it drives cost ten
minutes each. So the model that writes the list matters more than it looks.

This measures, on the real task (a one-line concept -> a structured shot list):

  wall time, prefill and decode speed, whether the output parses, and whether it
  satisfies the schema the pipeline needs.

Notes on the stack question, since they inform what is worth measuring:

  - Ollama IS llama.cpp underneath (its runner is llama.cpp plus a server and
    model management). Swapping to raw llama.cpp mainly buys finer control over
    threads, context and batching, not a fundamentally faster kernel. Expect a
    small delta, not a large one.
  - The dominant cost in a pipeline like this is usually *retries*, not tok/s. If
    a model returns JSON the code can trust first time, a slow model beats a fast
    one that needs three attempts.
  - The genuine architectural win on a CPU-only box is a mixture-of-experts model
    with few ACTIVE parameters: large-model quality at small-model decode speed,
    provided the whole thing fits in RAM. On 15.3 GB total that is a hard ceiling,
    and a model that spills to disk is slower than a small dense one.

    python llm_bench.py
    python llm_bench.py --models granite4.1:8b,qwen2.5:3b
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

CONCEPT = "within the shadows, friends and foes appear the same"

# the schema the film pipeline actually needs. scene and subject are what let
# director.py apply continuity rules and model_select.py switch at a boundary.
SHOT_SCHEMA = {
    "type": "object",
    "properties": {
        "logline": {"type": "string"},
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "scale": {"type": "string",
                              "enum": ["macro", "close", "medium", "wide",
                                       "extreme-wide"]},
                    "camera": {"type": "string",
                               "enum": ["none", "zoom-in", "zoom-out", "pan-left",
                                        "pan-right", "tilt-up", "tilt-down",
                                        "rolling-clockwise",
                                        "rolling-anticlockwise"]},
                    "seconds": {"type": "number"},
                    "subject": {"type": "string"},
                    "scene": {"type": "string"},
                    "energy": {"type": "number"},
                },
                "required": ["prompt", "scale", "camera", "seconds", "scene",
                             "energy"],
            },
        },
    },
    "required": ["logline", "shots"],
}

SYSTEM = (
    "You are a director writing a shot list for a short atmospheric film.\n"
    "Rules, from hard experience with the renderer:\n"
    "- 4 to 6 shots. Each shot is ONE image description, not a sequence of events.\n"
    "- Vary the framing scale between consecutive shots; never repeat one.\n"
    "- Prefer camera 'none'. The renderer already drifts; a move on every shot is "
    "exhausting.\n"
    "- 'seconds' is 1.5 to 3.6 and must follow a rhythm, not be uniform.\n"
    "- 'energy' is 0 to 1: how much movement the shot needs.\n"
    "- 'scene' groups shots into locations or acts. A look change and a model "
    "change only land at a scene boundary.\n"
    "- 'subject' is the recurring person or thing, or empty string for none.\n"
    "- Write concrete visual descriptions. No dialogue, no camera directions "
    "inside 'prompt'."
)

DEFAULT_MODELS = ["granite4.1:8b", "llama3:latest", "mistral:latest",
                  "qwen2.5:3b", "qwen3:1.7b"]


def run_one(model: str, warm: bool = True) -> Dict:
    import ollama

    def call():
        t0 = time.time()
        resp = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content":
                       f"Concept: {CONCEPT}\nWrite the shot list."}],
            format=SHOT_SCHEMA,
            options={"temperature": 0.8, "num_predict": 1200},
        )
        return resp, time.time() - t0

    out: Dict = {"model": model}
    try:
        if warm:
            call()                      # pay the model load outside the timing
        resp, wall = call()
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        return out

    out["wall_s"] = round(wall, 1)
    out["load_s"] = round(getattr(resp, "load_duration", 0) / 1e9, 1)
    pe = getattr(resp, "prompt_eval_count", 0) or 0
    pe_d = getattr(resp, "prompt_eval_duration", 0) or 1
    ec = getattr(resp, "eval_count", 0) or 0
    ec_d = getattr(resp, "eval_duration", 0) or 1
    out["prefill_tok_s"] = round(pe / (pe_d / 1e9), 1)
    out["decode_tok_s"] = round(ec / (ec_d / 1e9), 1)
    out["out_tokens"] = ec

    txt = resp["message"]["content"]
    try:
        data = json.loads(txt)
        out["json"] = True
    except Exception:
        out["json"] = False
        data = None
    if data:
        shots = data.get("shots", [])
        out["n_shots"] = len(shots)
        req = SHOT_SCHEMA["properties"]["shots"]["items"]["required"]
        out["schema_ok"] = all(all(k in s for k in req) for s in shots)
        sc = [s.get("scale") for s in shots]
        out["scales_repeat_adjacent"] = any(
            sc[i] == sc[i + 1] for i in range(len(sc) - 1))
        out["n_scenes"] = len({s.get("scene") for s in shots})
        out["logline"] = data.get("logline", "")[:70]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--dump", action="store_true",
                    help="print the winning shot list in full")
    a = ap.parse_args()
    models: List[str] = [m.strip() for m in a.models.split(",") if m.strip()]

    print(f'concept: "{CONCEPT}"\n')
    print(f"  {'model':<20} {'wall':>6} {'load':>6} {'pre':>7} {'dec':>7} "
          f"{'json':>5} {'schema':>7} {'shots':>6} {'scenes':>7} {'rep':>4}")
    print("  " + "-" * 92)
    rows = []
    for m in models:
        r = run_one(m)
        rows.append(r)
        if "error" in r:
            print(f"  {m:<20} {r['error']}")
            continue
        print(f"  {m:<20} {r['wall_s']:>6.1f} {r['load_s']:>6.1f} "
              f"{r['prefill_tok_s']:>7.1f} {r['decode_tok_s']:>7.1f} "
              f"{str(r['json']):>5} {str(r.get('schema_ok')):>7} "
              f"{r.get('n_shots', 0):>6} {r.get('n_scenes', 0):>7} "
              f"{str(r.get('scales_repeat_adjacent')):>4}")

    ok = [r for r in rows if r.get("json") and r.get("schema_ok")]
    print(f"\n  {len(ok)}/{len(rows)} returned schema-valid JSON first try")
    if ok:
        best = max(ok, key=lambda r: r["decode_tok_s"])
        print(f"  fastest valid: {best['model']} "
              f"({best['decode_tok_s']} tok/s decode, {best['wall_s']}s wall)")
        print("\n  NOTE: 'rep' True means two adjacent shots share a framing scale,")
        print("  which is a jump cut -- the planner will fix it, but a model that")
        print("  does not repeat scales saves a repair and reads the rules.")
    if a.dump and ok:
        best = max(ok, key=lambda r: r["decode_tok_s"])
        print(f"\n=== {best['model']} output ===")
        print(best.get("logline", ""))


if __name__ == "__main__":
    main()
