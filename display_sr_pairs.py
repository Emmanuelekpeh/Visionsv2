import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from pathlib import Path
import random

def display_random_pairs(dataset_dir, num_pairs=3):
    dataset_dir = Path(dataset_dir)
    target_files = sorted(list(dataset_dir.glob("target_*.png")))
    degraded_files = sorted(list(dataset_dir.glob("degraded_*.png")))
    
    if not target_files or len(target_files) != len(degraded_files):
        print(f"Error: mismatch in target/degraded files in {dataset_dir}")
        return
        
    indices = random.sample(range(len(target_files)), min(num_pairs, len(target_files)))
    
    fig, axes = plt.subplots(num_pairs, 2, figsize=(8, 4 * num_pairs))
    if num_pairs == 1:
        axes = [axes]
        
    for i, idx in enumerate(indices):
        target_img = Image.open(target_files[idx])
        degraded_img = Image.open(degraded_files[idx])
        
        # Display degraded
        axes[i][0].imshow(degraded_img)
        axes[i][0].set_title(f"Degraded (T+1 VAE output, 64x64)\n{degraded_files[idx].name}")
        axes[i][0].axis('off')
        
        # Display target
        axes[i][1].imshow(target_img)
        axes[i][1].set_title(f"Target (Original, 128x128)\n{target_files[idx].name}")
        axes[i][1].axis('off')
        
    plt.tight_layout()
    plt.savefig('sr_dataset_preview.png', bbox_inches='tight')
    print("Saved preview to sr_dataset_preview.png")

if __name__ == '__main__':
    display_random_pairs("dataset/sr_training_true_vae", num_pairs=4)
