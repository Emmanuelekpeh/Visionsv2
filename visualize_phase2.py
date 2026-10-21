"""
Visualize Phase 2 Predictions - See actual temporal learning
"""
import torch
import torch.nn as nn
from pathlib import Path
import json
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import argparse

# Import model architecture from test script
import sys
sys.path.append('.')
from test_phase2_local import TPlus1VAE

def visualize_predictions(checkpoint_path, dataset_path, device='cpu', num_samples=5, output_path='phase2_predictions_final.png'):
    print("=" * 60)
    print("PHASE 2 PREDICTION VISUALIZATION")
    print("=" * 60)
    
    # Load model
    print(f"\n[LOADING] Checkpoint: {checkpoint_path}")
    model = TPlus1VAE(latent_dim=64)
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if 'decoder.deconv.9.weight' in state_dict and state_dict['decoder.deconv.9.weight'].shape[1] != 5:
        print("  [WARNING] Architecture change detected! Dropping last decoder layer to start fresh.")
        del state_dict['decoder.deconv.9.weight']
        del state_dict['decoder.deconv.9.bias']
        
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print(f"[OK] Model loaded")
    
    # Load dataset
    print(f"[LOADING] Dataset: {dataset_path}")
    with open(dataset_path, 'r') as f:
        pairs = [json.loads(line) for line in f]
    print(f"[OK] Loaded {len(pairs)} pairs")
    
    transform = transforms.ToTensor()
    
    # Select random samples
    import random
    random.seed(42)
    sample_pairs = random.sample(pairs, min(num_samples, len(pairs)))
    
    print(f"\n[GENERATING] {len(sample_pairs)} predictions...")
    
    # Create visualization
    fig, axes = plt.subplots(len(sample_pairs), 3, figsize=(12, 4*len(sample_pairs)))
    if len(sample_pairs) == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for idx, pair in enumerate(sample_pairs):
            # Load images
            prev_img = Image.open(pair['prev']).convert('RGB')
            next_img = Image.open(pair['next']).convert('RGB')
            
            # Transform
            prev_tensor = transform(prev_img).unsqueeze(0).to(device)
            next_tensor = transform(next_img).unsqueeze(0).to(device)
            
            # Predict
            out = model(prev_tensor, next_tensor)
            pred_img = out['x_pred'].squeeze(0).cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0, 1)
            
            # Calculate error
            next_np = next_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            mse = np.mean((pred_img - next_np) ** 2)
            
            # Plot
            axes[idx, 0].imshow(prev_img)
            axes[idx, 0].set_title(f"Frame t\n(Input)")
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(pred_img)
            axes[idx, 1].set_title(f"Frame t+1 (Predicted)\nMSE: {mse:.4f}")
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(next_img)
            axes[idx, 2].set_title(f"Frame t+1 (Ground Truth)")
            axes[idx, 2].axis('off')
            
            print(f"  Sample {idx+1}: MSE={mse:.4f}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] Visualization: {output_path}")
    
    print("\n" + "=" * 60)
    print(f"[COMPLETE] Check {output_path} to see results!")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="t_plus_1_vae_phase2_final.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="dataset/t_plus_1_training/t_plus_1_pairs.jsonl",
                       help="Path to dataset manifest")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of predictions to visualize")
    parser.add_argument("--output", type=str, default="phase2_predictions_final.png",
                       help="Output image path")
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint
    dataset_path = args.dataset
    
    # Try different checkpoint names if not found
    if not Path(checkpoint_path).exists():
        alternatives = [
            "t_plus_1_vae_phase2_final.pt",
            "t_plus_1_vae_phase2_latest.pt",
            "t_plus_1_vae_phase2_epoch_30.pt",
            "t_plus_1_vae_phase2_local_test.pt"
        ]
        found = False
        for alt in alternatives:
            if Path(alt).exists():
                print(f"[INFO] Using checkpoint: {alt}")
                checkpoint_path = alt
                found = True
                break
        if not found:
            print(f"[ERROR] No checkpoint found. Tried:")
            for alt in alternatives:
                print(f"  - {alt}")
            exit(1)
    
    if not Path(dataset_path).exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        exit(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    visualize_predictions(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        device=device,
        num_samples=args.num_samples,
        output_path=args.output
    )
