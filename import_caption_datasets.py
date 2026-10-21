"""Import Flickr8k + COCO caption datasets into caption_manifest.jsonl for training.

Local:
  python import_caption_datasets.py --search dataset --search /path/to/kaggle/download

Kaggle (after attaching datasets):
  python import_caption_datasets.py --search /kaggle/input --out /kaggle/working/visions/dataset/caption_manifest.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from caption_datasets import discover_and_build, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build caption_manifest.jsonl from Flickr8k / COCO.")
    parser.add_argument("--search", action="append", default=[], help="Root dirs to scan (repeatable)")
    parser.add_argument("--flickr8k", action="store_true", default=True, help="Include Flickr8k (default on)")
    parser.add_argument("--no-flickr8k", action="store_false", dest="flickr8k")
    parser.add_argument("--coco", action="store_true", default=True, help="Include COCO subset (default on)")
    parser.add_argument("--no-coco", action="store_false", dest="coco")
    parser.add_argument("--coco-max-images", type=int, default=15000, help="Max COCO train images (0 = all)")
    parser.add_argument("--captions-per-image", type=int, default=1, help="Captions per image (1 = one random)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=str,
        default="dataset/caption_manifest.jsonl",
        help="Output manifest path",
    )
    args = parser.parse_args()

    search = list(args.search)
    if not search:
        search = ["dataset", "/kaggle/input", "."]

    samples = discover_and_build(
        search,
        flickr8k=args.flickr8k,
        coco=args.coco,
        coco_max_images=args.coco_max_images,
        captions_per_image=args.captions_per_image,
        seed=args.seed,
    )
    if not samples:
        print("No caption samples found. Attach Flickr8k / COCO datasets or pass --search paths.")
        return

    write_manifest(samples, Path(args.out))


if __name__ == "__main__":
    main()
