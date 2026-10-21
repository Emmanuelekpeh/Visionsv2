"""Build visions-code.zip for Kaggle (code + checkpoints + CLIP, no internet needed on Kaggle).

  python package_kaggle_direct_t2i.py

First run downloads CLIP (~600MB) to models/ if missing. Re-upload zip to Kaggle after.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "kaggle_upload" / "visions-code.zip"
CLIP_DIR = ROOT / "models" / "clip-vit-base-patch32"
CLIP_REPO = "openai/clip-vit-base-patch32"

CODE_FILES = [
    "train_direct_cnn_t2i.py",
    "alternative_t2i.py",
    "vision.py",
    "training_scheduler.py",
    "engines.py",
    "kaggle_dataset.py",
    "kaggle_train_direct_t2i.py",
    "caption_datasets.py",
    "import_caption_datasets.py",
    "activities.txt",
]

CHECKPOINT_PREFIXES = {
    "direct_cnn_t2i": "direct_cnn_t2i.pt",
    "direct_discriminator": "direct_discriminator.pt",
    "text_embed_cache": "text_embed_cache.pt",
}


def _newest_checkpoint(prefix: str) -> Path | None:
    matches = [
        p for p in ROOT.iterdir()
        if p.is_file() and p.suffix.lower() == ".pt" and p.name.startswith(prefix)
    ]
    if not matches:
        return None
    # Prefer explicit kaggle checkpoint if present
    kaggle = [p for p in matches if "kaggle" in p.name.lower()]
    if kaggle:
        return max(kaggle, key=lambda p: p.stat().st_mtime)
    return max(matches, key=lambda p: p.stat().st_mtime)


def ensure_clip() -> Path:
    if CLIP_DIR.is_dir() and (CLIP_DIR / "config.json").is_file():
        print(f"CLIP already at {CLIP_DIR}")
        return CLIP_DIR
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {CLIP_REPO} (~600MB, one-time)...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Run: pip install huggingface_hub")
    snapshot_download(repo_id=CLIP_REPO, local_dir=str(CLIP_DIR))
    print("CLIP ready.")
    return CLIP_DIR


def _add_dir(zf: zipfile.ZipFile, folder: Path, arc_prefix: str) -> int:
    count = 0
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            arc = f"{arc_prefix}/{path.relative_to(folder).as_posix()}"
            zf.write(path, arc)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-clip", action="store_true", help="Omit CLIP (Kaggle will fail without internet)")
    args = parser.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    included: list[str] = []

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in CODE_FILES:
            path = ROOT / name
            if not path.is_file():
                raise FileNotFoundError(path)
            zf.write(path, name)

        for prefix, zip_name in CHECKPOINT_PREFIXES.items():
            path = _newest_checkpoint(prefix)
            if path is not None:
                zf.write(path, zip_name)
                included.append(f"{path.name} -> {zip_name}")

        if not args.skip_clip:
            clip = ensure_clip()
            n = _add_dir(zf, clip, "models/clip-vit-base-patch32")
            included.append(f"CLIP ({n} files)")

    nb = ROOT / "kaggle_direct_t2i.ipynb"
    if nb.is_file():
        shutil.copy2(nb, OUT.parent / nb.name)

    mb = OUT.stat().st_size / (1024 * 1024)
    print(f"\nCreated {OUT} ({mb:.0f} MB)")
    for line in included:
        print(f"  + {line}")
    print("\nRe-upload this zip to Kaggle (replace visions-code dataset). Then run notebook.")


if __name__ == "__main__":
    main()
