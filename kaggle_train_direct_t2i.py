"""
Kaggle runner for DirectCNNT2I.

  --prep   CPU + internet: copy code, link images, download CLIP
  --train  GPU: train (run after --prep)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import types
from pathlib import Path

WORK = Path("/kaggle/working/visions")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
CODE_MARKER = "train_direct_cnn_t2i.py"


def _is_kaggle() -> bool:
    return Path("/kaggle/input").is_dir()


def _count_images(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def _find_code_dir() -> Path | None:
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if not root.is_dir():
            continue
        for match in sorted(root.rglob(CODE_MARKER)):
            if match.is_file():
                return match.parent
    return None


def _debug_input_tree() -> str:
    lines = []
    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        return "/kaggle/input does not exist"
    for path in sorted(input_root.rglob("*")):
        rel = path.relative_to(input_root)
        if path.is_dir():
            lines.append(f"  [dir]  {rel}/")
        else:
            lines.append(f"  [file] {rel} ({path.stat().st_size} bytes)")
    return "\n".join(lines) if lines else "  (empty — click Add Data and attach visions-code)"


def _maybe_extract_code_zip() -> None:
    """Some uploads keep a .zip on disk instead of extracted files."""
    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        return
    for zp in sorted(input_root.rglob("*.zip")):
        name = zp.name.lower()
        if "vision" in name or "code" in name or "direct" in name:
            out = zp.parent / zp.stem
            if (out / CODE_MARKER).exists():
                continue
            import zipfile
            out.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(out)
            print(f"Extracted {zp.name} -> {out}")


def _dataset_roots() -> list[Path]:
    """Kaggle new layout: /kaggle/input/datasets/{user}/{slug} plus legacy /kaggle/input/{slug}."""
    roots: list[Path] = []
    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        return roots
    datasets_dir = input_root / "datasets"
    if datasets_dir.is_dir():
        for user_dir in datasets_dir.iterdir():
            if user_dir.is_dir():
                for slug_dir in user_dir.iterdir():
                    if slug_dir.is_dir():
                        roots.append(slug_dir)
    for path in input_root.iterdir():
        if path.is_dir() and path.name != "datasets":
            roots.append(path)
    return roots


def _find_image_dir(code_dir: Path | None) -> Path | None:
    code_resolved = code_dir.resolve() if code_dir else None
    skip_names = ("vision", "code", "checkpoint", "flickr", "coco")
    best_path, best_count = None, 0
    for root in _dataset_roots():
        if code_resolved and root.resolve() == code_resolved:
            continue
        if (root / CODE_MARKER).exists():
            continue
        if any(k in root.name.lower() for k in skip_names):
            continue
        count = _count_images(root)
        if count > best_count:
            best_path, best_count = root, count
    return best_path


def _find_checkpoints() -> list[Path]:
    found = []
    for base in [Path("/kaggle/input"), WORK]:
        if not base.is_dir():
            continue
        for path in base.rglob("*.pt"):
            if path.name.startswith(("direct_cnn_t2i", "direct_discriminator", "text_embed_cache")):
                found.append(path)
    # Prefer newest duplicate names
    by_name: dict[str, Path] = {}
    for path in sorted(found, key=lambda p: p.stat().st_mtime):
        by_name[path.name] = path
    return list(by_name.values())


def _link_or_copy(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)
    try:
        os.symlink(src.resolve(), dest, target_is_directory=src.is_dir())
        print(f"Linked {src.name} -> {dest}")
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)
        print(f"Copied {src.name} -> {dest}")


def prep(work_dir: Path = WORK, code_dir: Path | None = None) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)

    if code_dir is None:
        script_dir = Path(__file__).resolve().parent
        if (script_dir / CODE_MARKER).exists():
            code_dir = script_dir
        else:
            _maybe_extract_code_zip()
            code_dir = _find_code_dir()

    if code_dir is None:
        raise FileNotFoundError(
            "train_direct_cnn_t2i.py not found.\n"
            "Attach visions-code under Add Data, then re-run.\n\n"
            + _debug_input_tree()
        )

    print(f"Using code from {code_dir}")

    for item in code_dir.iterdir():
        if item.name.startswith("."):
            continue
        dest = work_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    print(f"Code ready in {work_dir}")

    image_dir = _find_image_dir(code_dir if _is_kaggle() else None)
    dataset_dest = work_dir / "dataset"
    if image_dir is not None:
        # Use the dataset as-is: scraped_images/, nested folders, whatever.
        inner = image_dir / "dataset"
        src = inner if inner.is_dir() else image_dir
        _link_or_copy(src, dataset_dest)
        print(f"Images: {_count_images(dataset_dest)} files from {image_dir.name}")
    else:
        print("No image dataset attached — add your scraped images under Add Data.")

    clip_dir = work_dir / "models" / "clip-vit-base-patch32"
    if not clip_dir.is_dir() or not any(clip_dir.iterdir()):
        clip_dir.parent.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="openai/clip-vit-base-patch32",
            local_dir=str(clip_dir),
            local_dir_use_symlinks=False,
        )
        print("CLIP downloaded.")
    else:
        print("CLIP already present.")

    for ckpt in _find_checkpoints():
        dest = work_dir / ckpt.name
        if not dest.exists():
            shutil.copy2(ckpt, dest)
            print(f"Checkpoint: {ckpt.name}")
        # Also copy to generic names for resume fallback
        if ckpt.name.startswith("direct_cnn_t2i") and "discriminator" not in ckpt.name:
            for name in ("direct_cnn_t2i.pt", "direct_cnn_t2i_kaggle.pt"):
                d = work_dir / name
                if not d.exists() or d.stat().st_mtime < ckpt.stat().st_mtime:
                    shutil.copy2(ckpt, d)
        elif ckpt.name.startswith("direct_discriminator"):
            for name in ("direct_discriminator.pt", "direct_discriminator_kaggle.pt"):
                d = work_dir / name
                if not d.exists() or d.stat().st_mtime < ckpt.stat().st_mtime:
                    shutil.copy2(ckpt, d)

    manifest = work_dir / "caption_manifest.jsonl"
    import subprocess
    import_script = work_dir / "import_caption_datasets.py"
    if import_script.is_file():
        cmd = [
            sys.executable,
            str(import_script),
            "--search",
            str(Path("/kaggle/input")),
            "--out",
            str(manifest),
            "--no-coco",
            "--captions-per-image",
            "1",
        ]
        print("Building caption manifest...")
        subprocess.run(cmd, cwd=str(work_dir), check=False)
        if manifest.is_file():
            n = sum(1 for _ in manifest.open(encoding="utf-8"))
            print(f"Caption manifest: {n} pairs")

    os.chdir(work_dir)
    if str(work_dir) not in sys.path:
        sys.path.insert(0, str(work_dir))
    return work_dir


def _stub_train_tiny_t2i() -> None:
    from kaggle_dataset import FolderImageDataset

    stub = types.ModuleType("train_tiny_t2i")
    stub.ScrapedActivityDataset = FolderImageDataset
    sys.modules["train_tiny_t2i"] = stub


def train(
    work_dir: Path = WORK,
    epochs: int = 200,
    batch_size: int = 64,
    tag: str = "",
    max_hours: float = 8.0,
    compile_models: bool = False,
    grad_accum: int = 1,
) -> None:
    os.chdir(work_dir)
    if str(work_dir) not in sys.path:
        sys.path.insert(0, str(work_dir))

    _stub_train_tiny_t2i()
    from train_direct_cnn_t2i import train_direct_cnn_t2i
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU. Turn GPU on in notebook settings, then re-run this cell.")

    train_direct_cnn_t2i(
        device="cuda",
        epochs=epochs,
        batch_size=batch_size,
        use_activities=False,
        use_dataset=True,
        tag=tag,
        use_amp=True,
        max_hours=max_hours,
        preload_images=True,
        compile_models=compile_models,
        grad_accum_steps=grad_accum,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DirectCNNT2I on Kaggle")
    parser.add_argument("--prep", action="store_true", help="Setup (CPU, needs internet)")
    parser.add_argument("--train", action="store_true", help="Train (GPU)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tag", type=str, default="kaggle")
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--compile", dest="compile_models", action="store_true")
    parser.add_argument("--grad-accum", type=int, default=1)
    args = parser.parse_args()

    if not args.prep and not args.train:
        parser.error("Pass --prep (CPU) then --train (GPU)")

    if args.prep:
        prep()
        print("\nDone. Turn GPU on and run:  python kaggle_train_direct_t2i.py --train")

    if args.train:
        if not (WORK / CODE_MARKER).exists():
            prep(code_dir=Path(__file__).resolve().parent if (Path(__file__).resolve().parent / CODE_MARKER).exists() else None)
        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            tag=args.tag,
            max_hours=args.max_hours,
            compile_models=args.compile_models,
            grad_accum=args.grad_accum,
        )


if __name__ == "__main__":
    main()
