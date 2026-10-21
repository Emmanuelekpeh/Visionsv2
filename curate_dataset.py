import os
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import time
import cv2

def calculate_metrics(prev_path, next_path):
    """Calculate motion (MSE) and complexity (variance)."""
    try:
        # Read images
        prev_img = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
        next_img = cv2.imread(next_path, cv2.IMREAD_GRAYSCALE)
        
        if prev_img is None or next_img is None:
            return None, None
            
        # Calculate Motion (MSE)
        mse = np.mean((prev_img.astype(np.float32) - next_img.astype(np.float32)) ** 2)
        
        # Calculate Complexity (Variance of the first frame)
        variance = np.var(prev_img)
        
        return mse, variance
    except Exception as e:
        return None, None

def prune_dataset(
    source_dir="dataset/t_plus_1_training",
    dest_dir="dataset/t_plus_1_curated",
    min_mse=5.0,        # Filter static frames
    max_mse=500.0,      # Filter scene cuts
    min_variance=20.0,  # Filter blank walls/dark frames
    max_pairs_per_video=100
):
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    manifest_path = source_dir / "t_plus_1_pairs.jsonl"
    
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        return

    print("=" * 60)
    print("DATASET CURATION & PRUNING")
    print("=" * 60)
    
    # 1. Load manifest
    print("Loading manifest...")
    pairs = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    
    print(f"Total pairs in manifest: {len(pairs)}")
    
    # 2. Evaluate pairs
    print("\nEvaluating pairs for motion and complexity...")
    valid_pairs = []
    
    # To speed things up, we'll group by video first and only evaluate up to a limit
    # But since we want the *best* pairs, we evaluate them all, then sort.
    
    video_groups = defaultdict(list)
    
    for pair in tqdm(pairs, desc="Analyzing"):
        prev_path = pair.get("prev")
        next_path = pair.get("next")
        video_id = pair.get("video", "unknown")
        
        # Handle relative paths properly
        if prev_path.startswith("dataset\\"):
            prev_path = prev_path.replace("dataset\\t_plus_1_training\\", "")
            next_path = next_path.replace("dataset\\t_plus_1_training\\", "")
            
        full_prev = source_dir / Path(prev_path).name
        full_next = source_dir / Path(next_path).name
        
        if not full_prev.exists() or not full_next.exists():
            continue
            
        mse, variance = calculate_metrics(str(full_prev), str(full_next))
        
        if mse is None:
            continue
            
        # Apply filters
        if min_mse <= mse <= max_mse and variance >= min_variance:
            pair['mse'] = float(mse)
            pair['variance'] = float(variance)
            pair['_full_prev'] = full_prev
            pair['_full_next'] = full_next
            video_groups[video_id].append(pair)

    # 3. Select top pairs per video
    print("\nSelecting best pairs per video...")
    selected_pairs = []
    
    for video_id, v_pairs in video_groups.items():
        # Sort by MSE (we want moderate, predictable motion, not too low, not too high)
        # We'll sort by how close they are to an "ideal" MSE of around 50
        v_pairs.sort(key=lambda x: abs(x['mse'] - 50.0))
        
        # Take the top N
        kept = v_pairs[:max_pairs_per_video]
        selected_pairs.extend(kept)
        
    print(f"Selected {len(selected_pairs)} highly curated pairs.")
    
    # 4. Copy files to new directory
    print(f"\nCopying files to {dest_dir}...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    new_manifest_path = dest_dir / "t_plus_1_pairs.jsonl"
    copied_files = set()
    
    with open(new_manifest_path, 'w', encoding='utf-8') as f:
        for pair in tqdm(selected_pairs, desc="Copying"):
            full_prev = pair.pop('_full_prev')
            full_next = pair.pop('_full_next')
            
            # New paths
            new_prev_name = full_prev.name
            new_next_name = full_next.name
            
            new_prev_path = dest_dir / new_prev_name
            new_next_path = dest_dir / new_next_name
            
            # Copy if not already copied (frames might be shared across pairs)
            if new_prev_name not in copied_files:
                shutil.copy2(full_prev, new_prev_path)
                copied_files.add(new_prev_name)
                
            if new_next_name not in copied_files:
                shutil.copy2(full_next, new_next_path)
                copied_files.add(new_next_name)
            
            # Update paths in manifest
            pair['prev'] = f"dataset/t_plus_1_curated/{new_prev_name}"
            pair['next'] = f"dataset/t_plus_1_curated/{new_next_name}"
            
            # Remove temporary metrics before saving
            pair.pop('mse', None)
            pair.pop('variance', None)
            
            f.write(json.dumps(pair) + '\n')

    print("\n" + "=" * 60)
    print("CURATION COMPLETE")
    print("=" * 60)
    print(f"Total curated pairs: {len(selected_pairs)}")
    print(f"Total physical images: {len(copied_files)}")
    print(f"Saved to: {dest_dir}")
    
    # Calculate size
    total_size = sum(f.stat().st_size for f in dest_dir.glob("*.png"))
    print(f"New dataset size: {total_size / (1024**2):.2f} MB")
    
    print("\n[OK] Curation complete. Review the files in dataset/t_plus_1_curated/")
    print("When you are ready, you can zip it manually or run a zip script.")

if __name__ == "__main__":
    prune_dataset()
