"""Folder-only image dataset for Kaggle (no Pinterest scraping).

Also loads caption_manifest.jsonl entries (Flickr8k, COCO, etc.).
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def _on_kaggle() -> bool:
    return Path("/kaggle/working").is_dir()


def file_content_hash(path: str | os.PathLike) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_image(path: str | os.PathLike, min_size: int = 32) -> bool:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            if w < min_size or h < min_size:
                return False
            if max(w, h) / max(1, min(w, h)) > 4.0:
                return False
        return True
    except Exception:
        return False


def load_caption_manifest(manifest_path: str | os.PathLike) -> list[dict]:
    """Load JSONL manifest: one {"path": "...", "text": "..."} per line."""
    path = os.fspath(manifest_path)
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            img_path = row.get("path", "")
            text = row.get("text", "")
            if img_path and text:
                rows.append({"path": img_path, "text": text})
    return rows


def find_caption_manifests(dataset_dir: str | os.PathLike) -> list[str]:
    """Default manifest locations relative to dataset root or project root."""
    dataset_dir = os.fspath(dataset_dir)
    candidates = [
        os.path.join(dataset_dir, "caption_manifest.jsonl"),
        "caption_manifest.jsonl",
        os.path.join(os.path.dirname(dataset_dir), "caption_manifest.jsonl"),
    ]
    return [p for p in candidates if os.path.isfile(p)]


def estimate_preload_ram_gb(num_samples, img_size=128, return_256=False, uint8=True):
    bytes_per_pixel = 1 if uint8 else 4
    per_image = 3 * img_size * img_size * bytes_per_pixel
    if return_256:
        per_image += 3 * 256 * 256 * bytes_per_pixel
    return num_samples * per_image / (1024 ** 3)


class FolderImageDataset(Dataset):
    """Load images from dataset/<prompt_folder>/... or dataset/<file>.jpg (prompt = stem)."""

    def __init__(
        self,
        img_size=128,
        samples_per_activity=90,
        activities_file="activities.txt",
        use_activities=False,
        use_dataset=True,
        return_256=False,
        preload=False,
        text_to_idx=None,
        dataset_dir="dataset",
        caption_manifest=None,
        ram_budget_gb=4.0,
        dedupe=None,
        validate_images=None,
        verbose=None,
        fast_load=None,
        **_kwargs,
    ):
        if fast_load is None:
            fast_load = _on_kaggle()
        if dedupe is None:
            dedupe = not fast_load
        if validate_images is None:
            validate_images = not fast_load
        if verbose is None:
            verbose = not fast_load

        self.samples = []
        self.img_size = img_size
        self.return_256 = return_256
        self.text_to_idx = text_to_idx
        self.preloaded = None
        self.dataset_dir = dataset_dir
        self.skipped_corrupt = 0
        self.skipped_duplicate = 0

        if not use_dataset:
            print("use_dataset=False — no images loaded (Kaggle mode expects --dataset).")
            return

        if not os.path.isdir(dataset_dir):
            print(f"Warning: dataset directory not found: {dataset_dir}")
        else:
            mode = "fast" if fast_load else "full"
            print(f"Loading images from {dataset_dir} ({mode} scan)...")
            loaded_paths = set()
            content_hashes = set()
            folder_count = 0
            for root, _dirs, files in os.walk(dataset_dir):
                prompt_text = os.path.basename(root).replace("_", " ")
                count = 0
                for filename in files:
                    if os.path.splitext(filename)[1].lower() not in IMAGE_EXT:
                        continue
                    filepath = os.path.join(root, filename)
                    if filepath in loaded_paths:
                        continue
                    if validate_images and not is_valid_image(filepath):
                        self.skipped_corrupt += 1
                        continue
                    if dedupe:
                        try:
                            digest = file_content_hash(filepath)
                        except OSError:
                            self.skipped_corrupt += 1
                            continue
                        if digest in content_hashes:
                            self.skipped_duplicate += 1
                            continue
                        content_hashes.add(digest)
                    if root == dataset_dir:
                        p_text = os.path.splitext(filename)[0].replace("_", " ")
                    else:
                        p_text = prompt_text
                    self.samples.append({"path": filepath, "text": p_text})
                    loaded_paths.add(filepath)
                    count += 1
                    if not verbose and len(self.samples) % 5000 == 0:
                        print(f"  indexed {len(self.samples)} images...")
                if count > 0:
                    folder_count += 1
                    if verbose:
                        print(f"  {count} images from {root}")

            if not verbose and folder_count:
                print(f"  {len(self.samples)} images in {folder_count} folders")

            manifest_paths = []
            if caption_manifest:
                manifest_paths.append(caption_manifest)
            manifest_paths.extend(find_caption_manifests(dataset_dir))
            seen_manifests = set()
            for manifest in manifest_paths:
                manifest = os.path.abspath(manifest)
                if manifest in seen_manifests:
                    continue
                seen_manifests.add(manifest)
                rows = load_caption_manifest(manifest)
                added = 0
                manifest_seen: set[tuple] = set()
                for row in rows:
                    filepath = row["path"]
                    text = row["text"]
                    if (filepath, text) in manifest_seen:
                        continue
                    if not os.path.isfile(filepath):
                        continue
                    if validate_images and not is_valid_image(filepath):
                        self.skipped_corrupt += 1
                        continue
                    if dedupe:
                        try:
                            digest = file_content_hash(filepath)
                        except OSError:
                            self.skipped_corrupt += 1
                            continue
                        if (digest, text) in manifest_seen:
                            self.skipped_duplicate += 1
                            continue
                        manifest_seen.add((digest, text))
                    manifest_seen.add((filepath, text))
                    self.samples.append({"path": filepath, "text": text})
                    added += 1
                if rows:
                    print(f"  {added} manifest samples from {manifest}")

        if self.skipped_corrupt or self.skipped_duplicate:
            print(
                f"Skipped {self.skipped_corrupt} corrupt/invalid, "
                f"{self.skipped_duplicate} duplicate images."
            )
        print(f"Loaded {len(self.samples)} training samples.")

        if preload and self.samples:
            if fast_load and ram_budget_gb < 8.0:
                ram_budget_gb = 8.0
            est_gb = estimate_preload_ram_gb(len(self.samples), self.img_size, self.return_256, uint8=True)
            if est_gb > ram_budget_gb:
                print(f"Preload would use ~{est_gb:.1f}GB (budget {ram_budget_gb:.1f}GB). Streaming from disk.")
            else:
                print(f"Preloading {len(self.samples)} images as uint8 (~{est_gb:.1f}GB)...")
                self.preloaded = []
                for i, sample in enumerate(self.samples):
                    img_128, img_256 = self._decode_sample(sample, store_uint8=True)
                    self.preloaded.append((img_128, img_256))
                    if (i + 1) % 2000 == 0 or i + 1 == len(self.samples):
                        print(f"  Preloaded {i + 1}/{len(self.samples)}")

    def _decode_sample(self, sample, store_uint8=False):
        try:
            img = Image.open(sample["path"])
            draft_size = 256 if self.return_256 else self.img_size
            img.draft("RGB", (draft_size, draft_size))
            img = img.convert("RGB")
        except Exception:
            raise RuntimeError(f"Failed to decode image: {sample['path']}")

        img_128 = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_tensor_128 = torch.from_numpy(np.array(img_128, dtype=np.uint8)).permute(2, 0, 1)

        if self.return_256:
            img = img.resize((256, 256), Image.BILINEAR)
            img_tensor_256 = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)
        else:
            img_tensor_256 = None

        if not store_uint8:
            img_tensor_128 = img_tensor_128.float().div_(255.0)
            if img_tensor_256 is not None:
                img_tensor_256 = img_tensor_256.float().div_(255.0)

        return img_tensor_128, img_tensor_256

    @staticmethod
    def _uint8_to_float(tensor):
        return tensor.float().div_(255.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.preloaded is not None:
            img_tensor_128, img_tensor_256 = self.preloaded[idx]
            img_tensor_128 = self._uint8_to_float(img_tensor_128)
            if img_tensor_256 is not None:
                img_tensor_256 = self._uint8_to_float(img_tensor_256)
        else:
            img_tensor_128, img_tensor_256 = self._decode_sample(sample)

        text_out = self.text_to_idx[sample["text"]] if self.text_to_idx is not None else sample["text"]

        if self.return_256:
            return img_tensor_128, img_tensor_256, text_out
        return img_tensor_128, text_out
