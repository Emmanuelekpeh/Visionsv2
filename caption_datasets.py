"""Load Flickr8k and COCO caption datasets into (image_path, caption) pairs."""

from __future__ import annotations

import json
import random
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _norm_caption(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def find_flickr8k_root(search_roots: list[Path | str]) -> Path | None:
    """Return directory containing Flickr8k images + token file."""
    token_names = {"flickr8k.token.txt", "Flickr8k.token.txt", "captions.txt"}
    image_dir_names = {"flicker8k_dataset", "flickr8k_dataset", "images"}

    for root in search_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for token in root.rglob("*"):
            if not token.is_file() or token.name not in token_names:
                continue
            base = token.parent
            for sub in base.iterdir():
                if sub.is_dir() and sub.name.lower() in image_dir_names:
                    if any(sub.glob("*.jpg")):
                        return base
            for sub in base.rglob("*"):
                if sub.is_dir() and sub.name.lower() in image_dir_names:
                    if any(sub.glob("*.jpg")):
                        return base
    return None


def find_coco_root(search_roots: list[Path | str]) -> Path | None:
    """Return directory containing train2017 (or val2017) and captions JSON."""
    for root in search_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for ann in root.rglob("captions_train2017.json"):
            if ann.is_file():
                return ann.parent.parent if ann.parent.name == "annotations" else ann.parent
        for ann in root.rglob("captions_val2017.json"):
            if ann.is_file():
                return ann.parent.parent if ann.parent.name == "annotations" else ann.parent
    return None


def parse_flickr8k_token(token_path: Path, images_dir: Path | None = None) -> dict[str, list[str]]:
    """Parse Flickr8k.token.txt -> {filename: [captions]}."""
    token_path = Path(token_path)
    if images_dir is None:
        for name in ("Flicker8k_Dataset", "Flickr8k_Dataset", "images", "Images"):
            candidate = token_path.parent / name
            if candidate.is_dir():
                images_dir = candidate
                break
        if images_dir is None:
            for candidate in token_path.parent.rglob("*"):
                if candidate.is_dir() and any(candidate.glob("*.jpg")):
                    images_dir = candidate
                    break
    if images_dir is None:
        raise FileNotFoundError(f"No Flickr8k image folder near {token_path}")

    images_dir = Path(images_dir)
    by_image: dict[str, list[str]] = {}
    
    lines = token_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if lines and lines[0].strip().lower() == "image,caption":
        import csv
        for row in csv.reader(lines[1:]):
            if len(row) < 2:
                continue
            filename, caption = row[0].strip(), row[1].strip()
            if not (images_dir / filename).is_file():
                continue
            cap = _norm_caption(caption)
            if cap:
                by_image.setdefault(filename, []).append(cap)
    else:
        for line in lines:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            key, caption = line.split("\t", 1)
            filename = key.split("#", 1)[0].strip()
            if not (images_dir / filename).is_file():
                continue
            cap = _norm_caption(caption)
            if not cap:
                continue
            by_image.setdefault(filename, []).append(cap)
            
    return by_image


def flickr8k_samples(
    root: Path | str,
    captions_per_image: int = 1,
    split: str = "train",
    seed: int = 42,
) -> list[dict]:
    """Build training samples from Flickr8k."""
    root = Path(root)
    token = next(
        (p for p in root.rglob("*") if p.is_file() and p.name.lower() in ("flickr8k.token.txt", "captions.txt")),
        None,
    )
    if token is None:
        raise FileNotFoundError(f"Flickr8k.token.txt or captions.txt not found under {root}")

    images_dir = None
    for name in ("Flicker8k_Dataset", "Flickr8k_Dataset", "images", "Images"):
        candidate = token.parent / name
        if candidate.is_dir():
            images_dir = candidate
            break
    if images_dir is None:
        for candidate in token.parent.rglob("*"):
            if candidate.is_dir() and list(candidate.glob("*.jpg")):
                images_dir = candidate
                break
    if images_dir is None:
        raise FileNotFoundError(f"Flickr8k images folder not found under {root}")

    split_files = {
        "train": ("Flickr_8k.trainImages.txt", "flickr_8k.trainimages.txt"),
        "val": ("Flickr_8k.devImages.txt", "flickr_8k.devimages.txt"),
        "test": ("Flickr_8k.testImages.txt", "flickr_8k.testimages.txt"),
    }
    allowed: set[str] | None = None
    for fname in split_files.get(split, split_files["train"]):
        split_path = next((p for p in root.rglob("*") if p.name.lower() == fname.lower()), None)
        if split_path and split_path.is_file():
            allowed = {
                line.strip()
                for line in split_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            break

    by_image = parse_flickr8k_token(token, images_dir)
    rng = random.Random(seed)
    samples: list[dict] = []
    for filename, captions in sorted(by_image.items()):
        if allowed is not None and filename not in allowed:
            continue
        img_path = images_dir / filename
        if not img_path.is_file():
            continue
        picks = captions if captions_per_image <= 0 else captions[:captions_per_image]
        if 0 < captions_per_image < len(captions):
            picks = rng.sample(captions, captions_per_image)
        for cap in picks:
            samples.append({"path": str(img_path.resolve()), "text": cap, "source": "flickr8k"})
    return samples


def coco_samples(
    root: Path | str,
    max_images: int = 15000,
    captions_per_image: int = 1,
    split: str = "train",
    seed: int = 42,
) -> list[dict]:
    """Build training samples from COCO captions JSON + image folder."""
    root = Path(root)
    ann_name = f"captions_{split}2017.json"
    ann_path = root / "annotations" / ann_name
    if not ann_path.is_file():
        ann_path = next((p for p in root.rglob(ann_name) if p.is_file()), None)
    if ann_path is None:
        raise FileNotFoundError(f"{ann_name} not found under {root}")

    img_dir = root / f"{split}2017"
    if not img_dir.is_dir():
        img_dir = next((p for p in root.rglob(f"{split}2017") if p.is_dir()), None)
    if img_dir is None:
        raise FileNotFoundError(f"{split}2017 image folder not found under {root}")

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    id_to_file = {img["id"]: img["file_name"] for img in data.get("images", [])}
    by_image: dict[int, list[str]] = {}
    for ann in data.get("annotations", []):
        image_id = ann["image_id"]
        cap = _norm_caption(str(ann.get("caption", "")))
        if cap:
            by_image.setdefault(image_id, []).append(cap)

    rng = random.Random(seed)
    image_ids = sorted(by_image.keys())
    if max_images > 0 and len(image_ids) > max_images:
        image_ids = rng.sample(image_ids, max_images)

    samples: list[dict] = []
    for image_id in image_ids:
        filename = id_to_file.get(image_id)
        if not filename:
            continue
        img_path = img_dir / filename
        if not img_path.is_file():
            continue
        captions = by_image[image_id]
        picks = captions if captions_per_image <= 0 else captions[:captions_per_image]
        if 0 < captions_per_image < len(captions):
            picks = rng.sample(captions, captions_per_image)
        for cap in picks:
            samples.append({"path": str(img_path.resolve()), "text": cap, "source": "coco"})
    return samples


def discover_and_build(
    search_roots: list[Path | str],
    flickr8k: bool = True,
    coco: bool = True,
    coco_max_images: int = 15000,
    captions_per_image: int = 1,
    seed: int = 42,
) -> list[dict]:
    """Auto-find Flickr8k / COCO under search roots and return combined samples."""
    roots = [Path(r) for r in search_roots if Path(r).is_dir()]
    all_samples: list[dict] = []

    if flickr8k:
        fk_root = find_flickr8k_root(roots)
        if fk_root:
            try:
                samples = flickr8k_samples(fk_root, captions_per_image=captions_per_image, seed=seed)
                print(f"Flickr8k: {len(samples)} samples from {fk_root}")
                all_samples.extend(samples)
            except FileNotFoundError as e:
                print(f"Flickr8k skipped: {e}")
        else:
            print("Flickr8k: not found in search paths")

    if coco:
        coco_root = find_coco_root(roots)
        if coco_root:
            try:
                samples = coco_samples(
                    coco_root,
                    max_images=coco_max_images,
                    captions_per_image=captions_per_image,
                    seed=seed,
                )
                print(f"COCO: {len(samples)} samples from {coco_root}")
                all_samples.extend(samples)
            except FileNotFoundError as e:
                print(f"COCO skipped: {e}")
        else:
            print("COCO: not found in search paths")

    return all_samples


def write_manifest(samples: list[dict], out_path: Path | str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps({"path": row["path"], "text": row["text"]}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples)} entries -> {out_path}")
