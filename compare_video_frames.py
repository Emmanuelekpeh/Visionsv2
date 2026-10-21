import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def get_generated_frames(video_path, num_frames=10, start_frame=10):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def get_real_frames(manifest_path, num_frames=10, start_idx=50):
    with open(manifest_path, 'r') as f:
        lines = f.readlines()
        
    # We want a sequence of frames. t_plus_1_pairs has (prev, next). 
    # Let's string them together if possible, or just grab the next ones.
    # Actually, we can just grab one pair, then another pair, etc.
    # But since they are pairs of consecutive frames, we can just grab 'prev' from consecutive lines 
    # assuming they are ordered.
    
    frames = []
    for i in range(start_idx, start_idx + num_frames):
        if i < len(lines):
            pair = json.loads(lines[i])
            img_path = Path(pair['prev'])
            if img_path.exists():
                img = Image.open(img_path).convert('RGB')
                frames.append(np.array(img))
    return frames

def create_comparison():
    gen_frames = get_generated_frames('workspace/final_movie.mp4', num_frames=8, start_frame=24)
    real_frames = get_real_frames('dataset/t_plus_1_curated/t_plus_1_pairs.jsonl', num_frames=8, start_idx=100)
    
    # We will resize all to 128x128 for easy comparison
    target_size = (128, 128)
    gen_frames = [cv2.resize(f, target_size) for f in gen_frames]
    real_frames = [cv2.resize(f, target_size) for f in real_frames]
    
    # Check if we got enough frames
    n = min(len(gen_frames), len(real_frames))
    if n == 0:
        print("No frames found to compare.")
        return
        
    fig, axes = plt.subplots(2, n, figsize=(n*2, 4.5))
    
    for i in range(n):
        # Top row: Generated
        axes[0, i].imshow(gen_frames[i])
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Gen t+{i}")
        
        # Bottom row: Real
        axes[1, i].imshow(real_frames[i])
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Real t+{i}")
        
    plt.suptitle("Generated Video (Blending) vs Real Video (Motion)", fontsize=16)
    plt.tight_layout()
    plt.savefig('blending_vs_real_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison to blending_vs_real_comparison.png")

if __name__ == "__main__":
    create_comparison()