"""
fetch_subject_loras.py -- pull the subject-motion LoRAs from Cseti/AD_Motion_LORAs.

These are the untested lead: AnimateDiff's measured weakness is that its motion is
*global drift*, not subject motion (top 10% of pixels hold only 14.2% of the
temporal variance). A walking/running/spinning-subject LoRA is the only remaining
mechanism on this hardware that could put a *localized* moving subject in frame,
which is what would revive the disproven fg/bg compositing idea.

Also fetches the author's own candidate renders (mp4). Those are ground truth for
what the LoRA is supposed to do, so the intended motion can be measured before
generating anything on this machine.

Run:  python fetch_subject_loras.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# caches must live inside the workspace or the sandbox raises PermissionError
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_home"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch_home"))
os.environ.setdefault("TEMP", str(ROOT / ".piptmp"))
os.environ.setdefault("TMP", str(ROOT / ".piptmp"))

REPO = "Cseti/AD_Motion_LORAs"
DEST = ROOT / "models" / "subject_lora"

# local name -> file in the repo
WANT = {
    "walking-frontview":
        "600_cseti_4941259_walking-frontview-6vid_mv2_r64_w576_h384_fr16.safetensors",
    "running-frontview":
        "1800_cseti_1118902_running-frontview-8vid_mv2_r64_w768_h512_fr16.safetensors",
    "ballerina-spin":
        "250_3723723_cseti_ballerina-spin_w512_h768_r64.safetensors",
    "casualmove-frontview":
        "500_cseti_3734954_casualmove-frontview-3vid_mv2_r64_w576_h384_fr16.safetensors",
    "walk-backview":
        "400_cseti_5294370_walk_backview-1_w576_h384_fr32_r56.safetensors",
    "drone-forward":
        "260_cseti_8890531_drone-forward-mv2_r64_w576_h384_fr16.safetensors",
}

# the author's own renders -- ground truth for the intended motion
REFERENCE = {
    "ref_walking-frontview.mp4": "walking-frontview-candidate.mp4",
    "ref_running-frontview.mp4": "running-8vid-frontview-candidate.mp4",
    "ref_nolora.mp4": "nolora-candidate_00001.mp4",
}


def main() -> int:
    from huggingface_hub import hf_hub_download

    DEST.mkdir(parents=True, exist_ok=True)
    ok, fail = [], []

    for name, remote in WANT.items():
        out = DEST / f"{name}.safetensors"
        if out.exists() and out.stat().st_size > 1_000_000:
            print(f"  [skip] {name}  ({out.stat().st_size/1e6:.1f} MB)")
            ok.append(name)
            continue
        try:
            p = hf_hub_download(REPO, remote, local_dir=str(DEST),
                                local_dir_use_symlinks=False)
            src = Path(p)
            if src != out and src.exists():
                src.replace(out)
            mb = out.stat().st_size / 1e6
            print(f"  [ok]   {name:<22} {mb:7.1f} MB")
            ok.append(name)
        except Exception as exc:
            print(f"  [FAIL] {name}: {type(exc).__name__}: {str(exc)[:140]}")
            fail.append(name)

    refdir = DEST / "reference"
    refdir.mkdir(parents=True, exist_ok=True)
    for local, remote in REFERENCE.items():
        out = refdir / local
        if out.exists() and out.stat().st_size > 100_000:
            print(f"  [skip] {local}")
            continue
        try:
            hf_hub_download(REPO, remote, local_dir=str(refdir),
                            local_dir_use_symlinks=False)
            # hf keeps the original filename inside local_dir
            src = refdir / remote
            if src.exists() and src != out:
                src.replace(out)
            print(f"  [ok]   {local}")
        except Exception as exc:
            print(f"  [FAIL] {local}: {type(exc).__name__}: {str(exc)[:140]}")

    print(f"\n{len(ok)}/{len(WANT)} LoRAs present in {DEST}")
    if fail:
        print("missing: " + ", ".join(fail))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
