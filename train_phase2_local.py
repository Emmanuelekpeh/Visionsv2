"""
Phase 2 Local Training - Extended Run
Trains with periodic checkpoints and progress logging
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import time
from test_phase2_local import TPlus1VAE, FramePairDataset

def train_phase2_extended(
    checkpoint_path,
    dataset_path,
    device='cpu',
    epochs=30,
    batch_size=32,
    lr=1e-4,
    save_every=5
):
    print("=" * 60)
    print("PHASE 2 EXTENDED LOCAL TRAINING")
    print("=" * 60)
    
    # Load model
    print(f"\n[LOADING] Checkpoint: {checkpoint_path}")
    model = TPlus1VAE(latent_dim=64)
    
    # Check if resuming from previous phase 2 training
    phase2_checkpoint = Path("t_plus_1_vae_phase2_latest.pt")
    if phase2_checkpoint.exists():
        print(f"[RESUMING] Found phase2_latest.pt, continuing training...")
        model.load_state_dict(torch.load(phase2_checkpoint, map_location=device, weights_only=True))
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    
    model = model.to(device)
    print(f"[OK] Model loaded on {device}")
    
    # Load dataset
    print(f"\n[LOADING] Dataset: {dataset_path}")
    dataset = FramePairDataset(dataset_path)
    print(f"[OK] Loaded {len(dataset)} frame pairs")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    
    kl_beta = 1e-3
    
    print(f"\n[CONFIG]")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Batches per epoch: {len(dataloader)}")
    print(f"  Save every: {save_every} epochs")
    print(f"  Device: {device}")
    print("\n" + "=" * 60)
    print("TRAINING START")
    print("=" * 60 + "\n")
    
    training_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = {'total': 0, 'recon': 0, 'pred': 0, 'latent': 0, 'kl': 0}
        epoch_start = time.time()
        
        for batch_idx, (x_t, x_next) in enumerate(dataloader):
            x_t = x_t.to(device, non_blocking=True)
            x_next = x_next.to(device, non_blocking=True)
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    out = model(x_t, x_next)
                    
                    loss_recon = F.mse_loss(out['x_recon'], x_t)
                    loss_pred = F.mse_loss(out['x_pred'], x_next)
                    loss_latent = F.mse_loss(out['z_pred'], out['z_next_target'])
                    loss_kl = -0.5 * torch.mean(1 + out['logvar'] - out['mu'].pow(2) - out['logvar'].exp())
                    
                    loss = loss_recon + loss_pred + 0.1 * loss_latent + kl_beta * loss_kl
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                model.update_target_encoder()
            else:
                out = model(x_t, x_next)
                
                loss_recon = F.mse_loss(out['x_recon'], x_t)
                loss_pred = F.mse_loss(out['x_pred'], x_next)
                loss_latent = F.mse_loss(out['z_pred'], out['z_next_target'])
                loss_kl = -0.5 * torch.mean(1 + out['logvar'] - out['mu'].pow(2) - out['logvar'].exp())
                
                loss = loss_recon + loss_pred + 0.1 * loss_latent + kl_beta * loss_kl
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                model.update_target_encoder()
            
            epoch_losses['total'] += loss.item()
            epoch_losses['recon'] += loss_recon.item()
            epoch_losses['pred'] += loss_pred.item()
            epoch_losses['latent'] += loss_latent.item()
            epoch_losses['kl'] += loss_kl.item()
        
        # Epoch summary
        n_batches = len(dataloader)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - training_start
        
        avg_total = epoch_losses['total'] / n_batches
        avg_recon = epoch_losses['recon'] / n_batches
        avg_pred = epoch_losses['pred'] / n_batches
        avg_latent = epoch_losses['latent'] / n_batches
        avg_kl = epoch_losses['kl'] / n_batches
        
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.1f}s | Total: {total_time/60:.1f}m")
        print(f"  Loss: {avg_total:.4f} | Recon: {avg_recon:.4f} | Pred: {avg_pred:.4f} | Latent: {avg_latent:.4f} | KL: {avg_kl:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_name = f"t_plus_1_vae_phase2_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_name)
            print(f"  [SAVED] {checkpoint_name}")
        
        # Always save latest
        torch.save(model.state_dict(), "t_plus_1_vae_phase2_latest.pt")
        
        print("-" * 60)
    
    # Final save
    final_path = "t_plus_1_vae_phase2_final.pt"
    torch.save(model.state_dict(), final_path)
    
    total_time = time.time() - training_start
    print("\n" + "=" * 60)
    print("[COMPLETE] Phase 2 Training Finished!")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Final checkpoint: {final_path}")
    print("=" * 60 + "\n")
    
    return model


if __name__ == "__main__":
    # Configuration
    checkpoint_path = "t_plus_1_vae_recon_latest (10).pt"
    dataset_path = "dataset/t_plus_1_training/t_plus_1_pairs.jsonl"
    
    # Check files
    if not Path(checkpoint_path).exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        exit(1)
    
    if not Path(dataset_path).exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        exit(1)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train
    batch_size = 64 if device == 'cuda' else 32
    
    model = train_phase2_extended(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        device=device,
        epochs=30,
        batch_size=batch_size,
        lr=1e-4,
        save_every=5
    )
    
    print("[OK] Training complete! Model saved.")
