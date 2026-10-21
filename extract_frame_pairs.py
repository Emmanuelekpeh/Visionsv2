"""
Extract consecutive frame pairs from videos for T+1 VAE temporal training
Saves pairs at 64x64 resolution with manifest file
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import time

def extract_frame_pairs(video_path, output_dir, max_pairs_per_video=500, skip_frames=2, quality_threshold=20, duplicate_threshold=10.0):
    """
    Extract consecutive frame pairs from a video.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frame pairs
        max_pairs_per_video: Maximum pairs to extract per video
        skip_frames: Sample every N frames (1=consecutive, 2=every other frame, etc.)
        quality_threshold: Minimum variance threshold to filter blank/static frames
        duplicate_threshold: Maximum Mean Squared Error (MSE) to filter duplicate frames
    
    Returns:
        List of pair records for manifest
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"  [ERROR] Could not open video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"  Video: {video_name}")
    print(f"    Duration: {duration:.1f}s | FPS: {fps:.1f} | Frames: {total_frames}")
    
    pairs = []
    frame_idx = 0
    saved_count = 0
    prev_frame = None
    
    # Progress bar
    pbar = tqdm(total=min(max_pairs_per_video, total_frames // skip_frames), 
                desc=f"  Extracting {video_name[:30]}", leave=False)
    
    while saved_count < max_pairs_per_video:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames if needed
        if frame_idx % skip_frames != 0:
            frame_idx += 1
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Quality check: skip nearly blank frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)
        if variance < quality_threshold:
            frame_idx += 1
            continue
        
        # Create pair with previous frame
        if prev_frame is not None:
            # Check for duplicate frames using Mean Squared Error (MSE)
            mse = np.mean((frame_rgb.astype(np.float32) - prev_frame.astype(np.float32)) ** 2)
            if mse < duplicate_threshold:
                frame_idx += 1
                continue
            
            # Resize to 64x64
            prev_pil = Image.fromarray(prev_frame).resize((64, 64), Image.BILINEAR)
            curr_pil = Image.fromarray(frame_rgb).resize((64, 64), Image.BILINEAR)
            
            # Generate filenames
            timestamp = int(time.time() * 1000)
            prev_filename = f"{video_name}_f{frame_idx-skip_frames:06d}_{timestamp}.png"
            curr_filename = f"{video_name}_f{frame_idx:06d}_{timestamp+1}.png"
            
            # Save frames
            prev_path = output_dir / prev_filename
            curr_path = output_dir / curr_filename
            
            prev_pil.save(prev_path)
            curr_pil.save(curr_path)
            
            # Record pair
            pairs.append({
                "prev": str(prev_path),
                "next": str(curr_path),
                "video": video_name,
                "frame_idx": frame_idx,
                "fps": fps,
                "source": "video"
            })
            
            saved_count += 1
            pbar.update(1)
        
        prev_frame = frame_rgb
        frame_idx += 1
    
    pbar.close()
    cap.release()
    
    print(f"    [OK] Extracted {saved_count} frame pairs")
    return pairs


def find_videos(directory, extensions=None):
    """Find all video files in directory"""
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    
    directory = Path(directory)
    videos = []
    
    for ext in extensions:
        videos.extend(list(directory.rglob(f"*{ext}")))
    
    return sorted(videos)


def create_manifest(pairs, output_path):
    """Create JSONL manifest file for dataset"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"\n[OK] Manifest saved: {output_path}")
    print(f"  Total pairs: {len(pairs)}")


def main():
    parser = argparse.ArgumentParser(description="Extract frame pairs from videos")
    parser.add_argument("--video-dir", type=str, default="videos",
                       help="Directory containing videos")
    parser.add_argument("--output-dir", type=str, default="dataset/t_plus_1_training",
                       help="Output directory for frame pairs")
    parser.add_argument("--max-pairs", type=int, default=500,
                       help="Maximum pairs per video")
    parser.add_argument("--skip-frames", type=int, default=2,
                       help="Sample every N frames (2=every other frame)")
    parser.add_argument("--quality-threshold", type=float, default=20,
                       help="Minimum variance to filter blank frames")
    parser.add_argument("--duplicate-threshold", type=float, default=10.0,
                       help="Maximum MSE to filter duplicate frames")
    parser.add_argument("--max-videos", type=int, default=None,
                       help="Process only first N videos")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Video Frame Pair Extraction for T+1 VAE Training")
    print("=" * 60)
    
    # Find videos
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"[ERROR] Video directory not found: {video_dir}")
        print("\nTrying common locations:")
        common_paths = [
            Path("videos"),
            Path("dataset/videos"),
            Path("../videos"),
            Path.home() / "Videos",
        ]
        for p in common_paths:
            if p.exists():
                print(f"  Found: {p}")
                video_dir = p
                break
        else:
            print("  No video directories found!")
            print("\nCreate a 'videos' folder and add video files, then run:")
            print(f"  python {Path(__file__).name} --video-dir videos")
            return
    
    print(f"\nSearching for videos in: {video_dir}")
    videos = find_videos(video_dir)
    
    if not videos:
        print(f"[ERROR] No video files found in {video_dir}")
        print("\nSupported formats: .mp4, .avi, .mov, .mkv, .webm, .flv")
        return
    
    if args.max_videos:
        videos = videos[:args.max_videos]
    
    print(f"[OK] Found {len(videos)} videos")
    
    # Extract frame pairs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExtracting frame pairs to: {output_dir}")
    print(f"  Max pairs per video: {args.max_pairs}")
    print(f"  Skip frames: {args.skip_frames} (every {args.skip_frames} frames)")
    print(f"  Quality threshold: {args.quality_threshold}")
    print(f"  Duplicate threshold: {args.duplicate_threshold}")
    print()
    
    all_pairs = []
    
    for video_idx, video_path in enumerate(videos, 1):
        print(f"[{video_idx}/{len(videos)}] Processing: {video_path.name}")
        
        try:
            pairs = extract_frame_pairs(
                video_path=video_path,
                output_dir=output_dir,
                max_pairs_per_video=args.max_pairs,
                skip_frames=args.skip_frames,
                quality_threshold=args.quality_threshold,
                duplicate_threshold=args.duplicate_threshold
            )
            all_pairs.extend(pairs)
            
        except Exception as e:
            print(f"  [ERROR] Error processing {video_path.name}: {e}")
            continue
    
    # Create manifest
    if all_pairs:
        manifest_path = output_dir / "t_plus_1_pairs.jsonl"
        create_manifest(all_pairs, manifest_path)
        
        # Summary
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Total videos processed: {len(videos)}")
        print(f"Total frame pairs: {len(all_pairs)}")
        print(f"Output directory: {output_dir}")
        print(f"Manifest file: {manifest_path}")
        
        # Calculate dataset size
        total_size = sum(f.stat().st_size for f in output_dir.glob("*.png"))
        print(f"Dataset size: {total_size / (1024**2):.2f} MB")
        
        print("\n[OK] Ready for Kaggle upload!")
        print("\nNext steps:")
        print("  1. Compress the dataset:")
        print(f"     zip -r t_plus_1_training.zip {output_dir}")
        print("  2. Upload to Kaggle as a new dataset")
        print("  3. Add to your training notebook as input")
        
    else:
        print("\n[ERROR] No frame pairs extracted!")
        print("Check that videos are readable and not corrupted.")


if __name__ == "__main__":
    main()
