"""
Audit, dedupe, and clean scraped images.

  python audit_dataset.py --root dataset/scraped_images
  python audit_dataset.py --clean --dedup
  python audit_dataset.py --merge path/to/other/scraped_images

Kaggle: upload NEW VERSION of images dataset (not a second zip).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import zipfile
from pathlib import Path

from kaggle_dataset import IMAGE_EXT, file_content_hash, is_valid_image

DEFAULT_ROOT = Path("dataset/scraped_images")


def prompt_to_folder(prompt: str) -> str:
    text = prompt.strip().strip('"').strip("'")
    folder = re.sub(r"[^\w\s-]", "", text)
    folder = re.sub(r"\s+", "_", folder).strip("_")
    return folder[:50]


def folder_to_query(folder: str) -> str:
    q = folder.replace("_", " ")
    if q.endswith(" aesthetic"):
        q = q[: -len(" aesthetic")]
    return q.strip()


def load_activities(path: Path) -> list[str]:
    if not path.is_file():
        return []
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip().strip('"').strip("'")
        if not text or text.startswith("---"):
            continue
        prompts.append(text)
    return prompts


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXT:
            yield path


def scan_dataset(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        n = sum(1 for _ in iter_images(folder))
        rows.append({"folder": folder.name, "path": folder, "count": n})
    return rows


def print_report(rows: list[dict], threshold: int) -> None:
    if not rows:
        print("No folders found.")
        return
    total = sum(r["count"] for r in rows)
    thin = [r for r in rows if r["count"] < threshold]
    print(f"\n{'=' * 60}")
    print(f"DATASET AUDIT — {len(rows)} folders, {total} images")
    print(f"Thin (<{threshold}): {len(thin)} folders")
    print(f"{'=' * 60}")
    for r in sorted(rows, key=lambda x: x["count"])[:20]:
        flag = " <<" if r["count"] < threshold else ""
        print(f"  {r['count']:4d}  {r['folder']}{flag}")


def clean_corrupt(root: Path, dry_run: bool) -> int:
    removed = 0
    paths = list(iter_images(root))
    total = len(paths)
    print(f"Checking {total} images for corruption...")
    for i, path in enumerate(paths, 1):
        if i % 1000 == 0 or i == total:
            print(f"  clean {i}/{total}...")
        if is_valid_image(path):
            continue
        removed += 1
        if removed <= 20:
            print(f"  corrupt: {path.name}")
        elif removed == 21:
            print("  ... (more corrupt files, not listing all)")
        if not dry_run:
            path.unlink(missing_ok=True)
    return removed


def dedup_exact(root: Path, dry_run: bool) -> tuple[int, int]:
    """Remove exact byte duplicates. Keeps earliest path per hash."""
    seen: dict[str, Path] = {}
    removed = 0
    paths = sorted(iter_images(root))
    total = len(paths)
    print(f"Hashing {total} images for duplicates...")
    for i, path in enumerate(paths, 1):
        if i % 1000 == 0 or i == total:
            print(f"  dedup {i}/{total}...")
        try:
            digest = file_content_hash(path)
        except OSError:
            continue
        if digest in seen:
            removed += 1
            if removed <= 10:
                print(f"  dupe: {path.name} (same as {seen[digest].name})")
            if not dry_run:
                path.unlink(missing_ok=True)
        else:
            seen[digest] = path
    return removed, len(seen)


def zip_dataset(root: Path, out: Path) -> None:
    """Zip dataset/scraped_images for Kaggle upload."""
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    print(f"Zipping {root} -> {out} ...")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXT:
                arc = f"scraped_images/{path.relative_to(root).as_posix()}"
                zf.write(path, arc)
                count += 1
                if count % 2000 == 0:
                    print(f"  zipped {count}...")
    mb = out.stat().st_size / (1024 * 1024)
    print(f"Created {out} ({count} images, {mb:.0f} MB)")


def merge_tree(canonical: Path, source: Path, dry_run: bool) -> tuple[int, int, int]:
    """Merge source into canonical. Skip corrupt and exact duplicates."""
    canonical.mkdir(parents=True, exist_ok=True)
    known = set()
    for path in iter_images(canonical):
        try:
            known.add(file_content_hash(path))
        except OSError:
            pass

    copied = 0
    skipped_dupe = 0
    skipped_bad = 0

    for path in iter_images(source):
        if not is_valid_image(path):
            skipped_bad += 1
            continue
        try:
            digest = file_content_hash(path)
        except OSError:
            skipped_bad += 1
            continue
        if digest in known:
            skipped_dupe += 1
            continue

        rel = path.relative_to(source)
        if len(rel.parts) > 1:
            dest_dir = canonical / rel.parent
        else:
            dest_dir = canonical / "imported_misc"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / path.name
        stem, suffix = path.stem, path.suffix
        n = 1
        while dest.exists():
            dest = dest_dir / f"{stem}_{n}{suffix}"
            n += 1
        if not dry_run:
            shutil.copy2(path, dest)
        known.add(digest)
        copied += 1

    return copied, skipped_dupe, skipped_bad


def fill_folders(targets: list[dict], target_count: int, add_aesthetic: bool) -> None:
    from pinterest_scraper import scrape_images_pil

    for i, row in enumerate(targets, 1):
        query = folder_to_query(row["folder"])
        print(f"\n[{i}/{len(targets)}] '{query}' ({row['count']} -> {target_count})")
        scrape_images_pil(query, max_results=target_count, add_aesthetic=add_aesthetic, return_paths=True)


def find_missing_activities(activities: list[str], root: Path, threshold: int) -> list[str]:
    missing = []
    for prompt in activities:
        folder = root / prompt_to_folder(prompt)
        n = sum(1 for _ in iter_images(folder)) if folder.is_dir() else 0
        if n < threshold:
            missing.append(prompt)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit, clean, dedupe, merge scraped images.")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--activities-file", type=str, default="activities.txt")
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--clean", action="store_true", help="Delete corrupt/unopenable images")
    parser.add_argument("--dedup", action="store_true", help="Delete exact duplicate files")
    parser.add_argument("--merge", type=str, default=None, help="Merge another scraped_images tree in")
    parser.add_argument("--fill", action="store_true", help="Re-scrape thin folders")
    parser.add_argument("--fill-missing", action="store_true", help="Scrape thin activities.txt prompts")
    parser.add_argument("--aesthetic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--zip", type=str, default=None, nargs="?", const="kaggle_upload/scraped_images.zip",
                        help="Zip dataset for Kaggle (default: kaggle_upload/scraped_images.zip)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"Dataset root not found: {root.resolve()}")
        print("Point --root at dataset/scraped_images (or your Kaggle download).")
        return

    if args.clean:
        n = clean_corrupt(root, args.dry_run)
        print(f"\n{'Would remove' if args.dry_run else 'Removed'} {n} corrupt images.")

    if args.dedup:
        removed, unique = dedup_exact(root, args.dry_run)
        print(f"\n{'Would remove' if args.dry_run else 'Removed'} {removed} duplicates ({unique} unique remain).")

    if args.merge:
        src = Path(args.merge)
        if not src.is_dir():
            print(f"Merge source not found: {src}")
        else:
            copied, dupes, bad = merge_tree(root, src, args.dry_run)
            print(
                f"\nMerge: +{copied} new, skipped {dupes} dupes, {bad} corrupt "
                f"({'dry run' if args.dry_run else 'applied'})"
            )

    rows = scan_dataset(root)
    print_report(rows, args.threshold)

    activities = load_activities(Path(args.activities_file))
    if activities:
        missing = find_missing_activities(activities, root, args.threshold)
        print(f"\n{len(missing)}/{len(activities)} activity prompts below {args.threshold} images")

    if args.fill:
        thin = [r for r in rows if r["count"] < args.threshold]
        if thin:
            print(f"\nFilling {len(thin)} thin folders...")
            fill_folders(thin, args.target, args.aesthetic)

    if args.fill_missing and activities:
        missing = find_missing_activities(activities, root, args.threshold)
        if missing:
            from pinterest_scraper import scrape_images_pil
            print(f"\nScraping {len(missing)} missing/thin prompts...")
            for i, prompt in enumerate(missing, 1):
                print(f"[{i}/{len(missing)}] {prompt}")
                scrape_images_pil(prompt, max_results=args.target, add_aesthetic=args.aesthetic, return_paths=True)

    if not any([args.clean, args.dedup, args.merge, args.fill, args.fill_missing, args.zip is not None]):
        print("\n--- Suggested workflow ---")
        print("  python audit_dataset.py --clean --dedup --zip")
        print("  python package_kaggle_direct_t2i.py")

    if args.zip is not None:
        zip_dataset(root, Path(args.zip))


if __name__ == "__main__":
    main()
