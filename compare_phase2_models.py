"""
Compare 5-epoch vs 30-epoch predictions side-by-side
"""
import torch
import torch.nn as nn
from pathlib import Path
import json
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from test_phase2_local import TPlus1VAE

def compare_models(checkpoint1, checkpoint2, dataset_path, device='cpu', num_samples=3):
    print("=" * 60)
    print("MODEL COMPARISON: 5-Epoch vs 30-Epoch")
    print("=" * 60)
    
    # Load models
    print(f"\n[LOADING] Model 1: {checkpoint1}")
    model1 = TPlus1VAE(latent_dim=64)
    model1.load_state_dict(torch.load(checkpoint1, map_location=device, weights_only=True))
    model1 = model1.to(device)
    model1.eval()
    
    print(f"[LOADING] Model 2: {checkpoint2}")
    model2 = TPlus1VAE(latent_dim=64)
    model2.load_state_dict(torch.load(checkpoint2, map_location=device, weights_only=True))
    model2 = model2.to(device)
    model2.eval()
    print("[OK] Models loaded")
    
    # Load dataset
    print(f"[LOADING] Dataset: {dataset_path}")
    with open(dataset_path, 'r') as f:
        pairs = [json.loads(line) for line in f]
    
    transform = transforms.ToTensor()
    
    # Select samples
    import random
    random.seed(42)
    sample_pairs = random.sample(pairs, num_samples)
    
    # Create comparison
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    print(f"\n[COMPARING] {num_samples} predictions...\n")
    
    with torch.no_grad():
        for idx, pair in enumerate(sample_pairs):
            # Load images
            prev_img = Image.open(pair['prev']).convert('RGB')
            next_img = Image.open(pair['next']).convert('RGB')
            
            prev_tensor = transform(prev_img).unsqueeze(0).to(device)
            next_tensor = transform(next_img).unsqueeze(0).to(device)
            next_np = next_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            
            # Predict with both models
            out1 = model1(prev_tensor, next_tensor)
            pred1 = out1['x_pred'].squeeze(0).cpu().numpy().transpose(1, 2, 0)
            pred1 = np.clip(pred1, 0, 1)
            mse1 = np.mean((pred1 - next_np) ** 2)
            
            out2 = model2(prev_tensor, next_tensor)
            pred2 = out2['x_pred'].squeeze(0).cpu().numpy().transpose(1, 2, 0)
            pred2 = np.clip(pred2, 0, 1)
            mse2 = np.mean((pred2 - next_np) ** 2)
            
            improvement = ((mse1 - mse2) / mse1) * 100
            
            # Plot
            axes[idx, 0].imshow(prev_img)
            axes[idx, 0].set_title(f"Frame t (Input)", fontsize=10, weight='bold')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(pred1)
            axes[idx, 1].set_title(f"5-Epoch Pred\nMSE: {mse1:.4f}", fontsize=10)
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(pred2)
            axes[idx, 2].set_title(f"30-Epoch Pred\nMSE: {mse2:.4f}", fontsize=10)
            axes[idx, 2].axis('off')
            axes[idx, 2].add_patch(plt.Rectangle((0, 0), 1, 1, transform=axes[idx, 2].transAxes,
                                                  fill=False, edgecolor='green', linewidth=3))
            
            axes[idx, 3].imshow(next_img)
            axes[idx, 3].set_title(f"Ground Truth", fontsize=10, weight='bold')
            axes[idx, 3].axis('off')
            
            # Difference map
            diff = np.abs(pred2 - next_np)
            diff_vis = np.mean(diff, axis=2)  # Average across RGB
            im = axes[idx, 4].imshow(diff_vis, cmap='hot', vmin=0, vmax=0.3)
            axes[idx, 4].set_title(f"Error Map\n(30-Epoch)", fontsize=10)
            axes[idx, 4].axis('off')
            plt.colorbar(im, ax=axes[idx, 4], fraction=0.046, pad=0.04)
            
            print(f"Sample {idx+1}:")
            print(f"  5-Epoch MSE:  {mse1:.4f}")
            print(f"  30-Epoch MSE: {mse2:.4f}")
            print(f"  Improvement:  {improvement:.1f}%")
            print()
    
    plt.suptitle("Training Progress Comparison: 5 Epochs → 30 Epochs", 
                 fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = 'phase2_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] Comparison: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    checkpoint1 = "t_plus_1_vae_phase2_local_test.pt"  # 5-epoch model
    checkpoint2 = "t_plus_1_vae_phase2_final.pt"       # 30-epoch model
    dataset_path = "dataset/t_plus_1_training/t_plus_1_pairs.jsonl"
    
    if not Path(checkpoint1).exists() or not Path(checkpoint2).exists():
        print("[ERROR] One or both checkpoints not found")
        print(f"  Need: {checkpoint1}")
        print(f"  Need: {checkpoint2}")
        exit(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    compare_models(checkpoint1, checkpoint2, dataset_path, device, num_samples=3)
