"""
T+1 VAE: Clean VAE-based frame prediction architecture.

Based on recommended architecture:
- 64×64 resolution (CPU-friendly)
- 64D latent space
- Explicit dynamics MLP
- Multi-objective training (recon + latent pred + pred-frame recon)
- EMA target encoder for stability

Input: frame_t → Output: frame_{t+1}
"""
import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import time
import random
import cv2
from copy import deepcopy

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class Encoder(nn.Module):
    """4-layer CNN encoder → 64D latent (μ, logσ²)"""
    
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 64×64 → 32 → 16 → 8 → 4
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 64→32
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32→16
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16→8
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8→4
            nn.GroupNorm(32, 256),
            nn.SiLU(),
        )
        
        # 256 * 4 * 4 = 4096 → latent
        self.flatten_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
    def forward(self, x):
        """x: [B, 3, 64, 64] → mu, logvar: [B, latent_dim]"""
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """CNN decoder: 64D latent → 64×64 RGB"""
    
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # 4 → 8 → 16 → 32 → 64
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 4→8
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8→16
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16→32
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 32→64
            nn.Sigmoid(),  # Output [0, 1]
        )
        
    def forward(self, z):
        """z: [B, latent_dim] → x: [B, 3, 64, 64]"""
        h = self.fc(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.deconv(h)


class DynamicsMLP(nn.Module):
    """Simple MLP: z_t → ẑ_{t+1}"""
    
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
    def forward(self, z):
        """z_t → ẑ_{t+1}"""
        return self.net(z)


class TPlus1VAE(nn.Module):
    """
    Complete T+1 VAE model.
    
    Training mode:
        Input: (x_t, x_{t+1})
        Output: losses dict
        
    Inference mode:
        Input: x_t
        Output: x̂_{t+1}
    """
    
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.dynamics = DynamicsMLP(latent_dim)
        
        # EMA target encoder (updated slowly)
        self.target_encoder = deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
            
        self.ema_decay = 0.99
        
    def reparameterize(self, mu, logvar):
        """Sample z from q(z|x) using reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update of target encoder"""
        for p, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
    
    def forward(self, x_t, x_next=None):
        """
        Forward pass.
        
        Training (x_next provided):
            Returns dict of losses
            
        Inference (x_next=None):
            Returns predicted next frame
        """
        # Encode current frame
        mu_t, logvar_t = self.encoder(x_t)
        z_t = self.reparameterize(mu_t, logvar_t)
        
        # Predict next latent
        z_pred = self.dynamics(z_t)
        
        # Decode predicted latent to next frame
        x_pred = self.decoder(z_pred)
        
        if x_next is None:
            # Inference mode
            return x_pred
        
        # Training mode - compute all losses
        
        # 1. Reconstruction of current frame
        x_recon = self.decoder(z_t)
        loss_recon = F.mse_loss(x_recon, x_t)
        
        # 2. KL divergence
        loss_kl = -0.5 * torch.mean(1 + logvar_t - mu_t.pow(2) - logvar_t.exp())
        
        # 3. Latent prediction loss (against target encoder)
        with torch.no_grad():
            mu_next_target, _ = self.target_encoder(x_next)
        loss_latent = F.mse_loss(z_pred, mu_next_target.detach())
        
        # 4. Predicted frame reconstruction
        loss_pred_recon = F.mse_loss(x_pred, x_next)
        
        return {
            'loss_recon': loss_recon,
            'loss_kl': loss_kl,
            'loss_latent': loss_latent,
            'loss_pred_recon': loss_pred_recon,
            'x_recon': x_recon,
            'x_pred': x_pred,
            'z_t': z_t,
            'z_pred': z_pred,
        }
    
    def predict_next(self, x_t):
        """Inference: predict next frame from current"""
        self.eval()
        with torch.no_grad():
            return self.forward(x_t, x_next=None)
    
    def rollout(self, x_start, num_steps: int):
        """Multi-step rollout for evaluation"""
        self.eval()
        frames = [x_start]
        x = x_start
        
        with torch.no_grad():
            for _ in range(num_steps):
                x = self.forward(x, x_next=None)
                frames.append(x)
                
        return torch.stack(frames, dim=1)  # [B, T, C, H, W]


# ============================================================
# DATASET
# ============================================================

class FramePairDataset64(Dataset):
    """Dataset for 64×64 frame pairs - fast loading"""
    
    def __init__(self, data_dir: str, max_pairs: int = None):
        self.data_dir = Path(data_dir)
        self.pairs = []
        
        # Load manifest
        manifest_path = self.data_dir / "t_plus_1_pairs.jsonl"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        if "prev" in record and "next" in record:
                            self.pairs.append(record)
        
        # Shuffle and limit
        random.shuffle(self.pairs)
        if max_pairs:
            self.pairs = self.pairs[:max_pairs]
            
        # Count sources
        video_count = sum(1 for p in self.pairs if p.get("source") == "video")
        sd_count = sum(1 for p in self.pairs if p.get("source") == "sd_synthetic")
        
        print(f"Loaded {len(self.pairs)} pairs (video: {video_count}, SD: {sd_count})")
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        record = self.pairs[idx]
        
        # Load and resize to 64×64
        prev_img = Image.open(record["prev"]).convert("RGB").resize((64, 64), Image.BILINEAR)
        next_img = Image.open(record["next"]).convert("RGB").resize((64, 64), Image.BILINEAR)
        
        # Convert to tensors [0, 1]
        prev_tensor = torch.from_numpy(np.array(prev_img)).permute(2, 0, 1).float() / 255.0
        next_tensor = torch.from_numpy(np.array(next_img)).permute(2, 0, 1).float() / 255.0
        
        # Simple augmentation
        if random.random() < 0.5:
            prev_tensor = torch.flip(prev_tensor, dims=[2])  # Horizontal flip
            next_tensor = torch.flip(next_tensor, dims=[2])
        
        return prev_tensor, next_tensor


# ============================================================
# TRAINING
# ============================================================

def train_t_plus_1_vae(
    device: str = "cpu",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    latent_dim: int = 64,
    checkpoint_path: str = "t_plus_1_vae.pt",
    data_dir: str = "dataset/t_plus_1_training",
    scheduled_sampling: bool = True,
):
    """Train the T+1 VAE model with scheduled sampling"""
    
    print("=" * 60)
    print("Training T+1 VAE")
    print(f"  Resolution: 64x64")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Scheduled sampling: {scheduled_sampling}")
    print(f"  Device: {device}")
    print("=" * 60)
    
    # Load dataset
    dataset = FramePairDataset64(data_dir)
    if len(dataset) < 10:
        print("Not enough data. Run extraction first.")
        return False
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=0, pin_memory=False, drop_last=True)
    
    # Create model
    model = TPlus1VAE(latent_dim=latent_dim).to(device)
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        try:
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
    
    # Optimizer with lower LR for continued training
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # KL warmup schedule
    kl_warmup_epochs = 10
    kl_target = 1e-3
    
    # Scheduled sampling: probability of using model's own prediction
    # Starts at 0, increases to 0.5 over training
    ss_start_epoch = 20  # Start scheduled sampling after epoch 20
    ss_max_prob = 0.5    # Max probability of using own prediction
    
    # Loss weights
    lambda_recon = 1.0
    lambda_latent = 0.5
    lambda_pred = 1.0
    lambda_ss = 0.5  # Weight for scheduled sampling loss
    
    print(f"\nStarting training: {len(dataset)} samples, {len(dataloader)} batches/epoch")
    print(f"KL warmup: {kl_warmup_epochs} epochs -> beta={kl_target}")
    if scheduled_sampling:
        print(f"Scheduled sampling: starts epoch {ss_start_epoch}, max prob {ss_max_prob}")
    sys.stdout.flush()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        
        # KL warmup
        kl_beta = min(1.0, epoch / kl_warmup_epochs) * kl_target
        
        # Scheduled sampling probability
        if scheduled_sampling and epoch >= ss_start_epoch:
            ss_prob = min(ss_max_prob, (epoch - ss_start_epoch) / (epochs - ss_start_epoch) * ss_max_prob)
        else:
            ss_prob = 0.0
        
        epoch_losses = {
            'total': 0, 'recon': 0, 'kl': 0, 'latent': 0, 'pred': 0, 'ss': 0
        }
        ss_count = 0
        
        for batch_idx, (x_t, x_next) in enumerate(dataloader):
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            
            optimizer.zero_grad()
            
            # Forward with ground truth input
            outputs = model(x_t, x_next)
            
            # Base loss
            loss = (
                lambda_recon * outputs['loss_recon'] +
                kl_beta * outputs['loss_kl'] +
                lambda_latent * outputs['loss_latent'] +
                lambda_pred * outputs['loss_pred_recon']
            )
            
            # Scheduled sampling: sometimes use model's own prediction as input
            loss_ss = torch.tensor(0.0, device=device)
            if scheduled_sampling and random.random() < ss_prob:
                ss_count += 1
                with torch.no_grad():
                    # Get model's prediction for current frame
                    x_pred_detached = outputs['x_pred'].detach()
                
                # Use prediction as input for next prediction
                # This teaches the model to handle its own outputs
                outputs_ss = model(x_pred_detached, x_next)
                loss_ss = lambda_ss * outputs_ss['loss_pred_recon']
                loss = loss + loss_ss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Update EMA target encoder
            model.update_target_encoder()
            
            # Accumulate
            epoch_losses['total'] += loss.item()
            epoch_losses['recon'] += outputs['loss_recon'].item()
            epoch_losses['kl'] += outputs['loss_kl'].item()
            epoch_losses['latent'] += outputs['loss_latent'].item()
            epoch_losses['pred'] += outputs['loss_pred_recon'].item()
            epoch_losses['ss'] += loss_ss.item() if isinstance(loss_ss, torch.Tensor) else loss_ss
        
        # Average
        n_batches = len(dataloader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        
        # Print progress
        ss_info = f" | SS: {epoch_losses['ss']:.4f} ({ss_count}/{n_batches})" if ss_prob > 0 else ""
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Total: {epoch_losses['total']:.4f} | "
              f"Recon: {epoch_losses['recon']:.4f} | "
              f"Pred: {epoch_losses['pred']:.4f} | "
              f"Latent: {epoch_losses['latent']:.4f} | "
              f"KL: {epoch_losses['kl']:.4f} (beta={kl_beta:.4f}){ss_info}")
        sys.stdout.flush()
        
        # Save every epoch
        torch.save(model.state_dict(), checkpoint_path)
        if (epoch + 1) % 10 == 0:
            print(f"  -> Saved checkpoint")
            sys.stdout.flush()
        
        # Track best
        if epoch_losses['pred'] < best_loss:
            best_loss = epoch_losses['pred']
            torch.save(model.state_dict(), checkpoint_path.replace(".pt", "_best.pt"))
    
    print(f"\nTraining complete. Best pred loss: {best_loss:.4f}")
    torch.save(model.state_dict(), checkpoint_path)
    
    return True


def evaluate_rollout(
    checkpoint_path: str = "t_plus_1_vae.pt",
    data_dir: str = "dataset/t_plus_1_training",
    num_steps: int = 10,
    device: str = "cpu"
):
    """Evaluate multi-step rollout quality"""
    
    print("=" * 60)
    print(f"Evaluating {num_steps}-step rollout")
    print("=" * 60)
    
    # Load model
    model = TPlus1VAE(latent_dim=64).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    # Load a few test samples
    dataset = FramePairDataset64(data_dir, max_pairs=100)
    
    # Pick random start frames
    test_indices = random.sample(range(len(dataset)), min(5, len(dataset)))
    
    output_dir = Path("eval_rollouts")
    output_dir.mkdir(exist_ok=True)
    
    for i, idx in enumerate(test_indices):
        x_t, x_next_gt = dataset[idx]
        x_t = x_t.unsqueeze(0).to(device)
        
        # Rollout
        frames = model.rollout(x_t, num_steps)  # [1, T, C, H, W]
        frames = frames.squeeze(0).cpu()  # [T, C, H, W]
        
        # Save as grid
        grid = torch.cat([f for f in frames], dim=2)  # Horizontal concat
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(grid_np).save(output_dir / f"rollout_{i}.png")
        
        print(f"  Saved rollout {i} ({num_steps} steps)")
    
    print(f"\nRollouts saved to {output_dir}/")


# ============================================================
# MAIN
# ============================================================

def train_reconstruction_phase(
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    latent_dim: int = 64,
    checkpoint_path: str = "t_plus_1_vae_recon.pt",
    scraped_dir: str = "dataset/scraped_images",
    virtual_epoch_size: int = None,  # Samples per virtual epoch
):
    """Phase 1: Train encoder-decoder on reconstruction only using scraped images."""
    
    print("=" * 60)
    print("PHASE 1: Reconstruction Pretraining")
    print(f"  Resolution: 64x64")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Batch size: {batch_size}")
    print("=" * 60)
    
    # Collect all images from scraped_images
    scraped_path = Path(scraped_dir)
    all_images = []
    for ext in ['*.jpg', '*.png']:
        all_images.extend(scraped_path.rglob(ext))
    
    print(f"Found {len(all_images)} images for reconstruction training")
    
    if len(all_images) < 100:
        print("Not enough images!")
        return False
    
    # Simple dataset for single images
    class SingleImageDataset(Dataset):
        def __init__(self, image_paths, max_images=None):
            self.paths = list(image_paths)
            random.shuffle(self.paths)
            if max_images:
                self.paths = self.paths[:max_images]
            print(f"Using {len(self.paths)} images")
            
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            try:
                img = Image.open(self.paths[idx]).convert("RGB").resize((64, 64), Image.BILINEAR)
                tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                # Augmentation
                if random.random() < 0.5:
                    tensor = torch.flip(tensor, dims=[2])
                return tensor
            except:
                # Return random tensor if image fails
                return torch.rand(3, 64, 64)
    
    # Virtual epochs: train on subset each epoch for faster iteration
    if virtual_epoch_size:
        dataset = SingleImageDataset(all_images, max_images=len(all_images))
        print(f"Virtual epochs: {virtual_epoch_size} samples per epoch")
    else:
        dataset = SingleImageDataset(all_images, max_images=None)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    # Create model (encoder + decoder only, no dynamics yet)
    model = TPlus1VAE(latent_dim=latent_dim).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # KL warmup
    kl_warmup = 10
    kl_target = 1e-4  # Very low KL for reconstruction phase
    
    print(f"\nStarting Phase 1: {len(dataset)} images, {len(dataloader)} batches/epoch")
    sys.stdout.flush()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        
        kl_beta = min(1.0, epoch / kl_warmup) * kl_target
        
        max_batches = virtual_epoch_size // batch_size if virtual_epoch_size else len(dataloader)
        
        for batch_idx, x in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            x = x.to(device)
            
            optimizer.zero_grad()
            
            # Encode
            mu, logvar = model.encoder(x)
            z = model.reparameterize(mu, logvar)
            
            # Decode
            x_recon = model.decoder(z)
            
            # Losses
            loss_recon = F.mse_loss(x_recon, x)
            loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = loss_recon + kl_beta * loss_kl
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += loss_recon.item()
            epoch_kl += loss_kl.item()
        
        n_batches = min(max_batches, len(dataloader))
        print(f"Phase1 Epoch {epoch+1:3d}/{epochs} | Loss: {epoch_loss/n_batches:.4f} | Recon: {epoch_recon/n_batches:.4f} | KL: {epoch_kl/n_batches:.4f} (beta={kl_beta:.5f})")
        sys.stdout.flush()
        
        # Save every epoch
        torch.save(model.state_dict(), checkpoint_path)
        if (epoch + 1) % 5 == 0:
            print(f"  -> Saved checkpoint")
            sys.stdout.flush()
    
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nPhase 1 complete. Saved to {checkpoint_path}")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="T+1 VAE Training")
    parser.add_argument("--train", action="store_true", help="Train the model (Phase 2)")
    parser.add_argument("--phase1", action="store_true", help="Phase 1: Reconstruction pretraining")
    parser.add_argument("--eval", action="store_true", help="Evaluate rollouts")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--no-scheduled-sampling", action="store_true", help="Disable scheduled sampling")
    parser.add_argument("--rollout-steps", type=int, default=10, help="Steps for rollout eval")
    parser.add_argument("--from-phase1", action="store_true", help="Load Phase 1 checkpoint before Phase 2")
    parser.add_argument("--virtual-epoch", type=int, default=None, help="Samples per virtual epoch (faster training)")
    args = parser.parse_args()
    
    device = "cpu"
    
    if args.phase1:
        train_reconstruction_phase(
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            latent_dim=args.latent_dim,
            virtual_epoch_size=args.virtual_epoch,
        )
    elif args.train:
        # If --from-phase1, copy the phase1 checkpoint first
        if args.from_phase1 and os.path.exists("t_plus_1_vae_recon.pt"):
            import shutil
            shutil.copy("t_plus_1_vae_recon.pt", "t_plus_1_vae.pt")
            print("Loaded Phase 1 checkpoint for Phase 2 training")
        
        train_t_plus_1_vae(
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            latent_dim=args.latent_dim,
            scheduled_sampling=not args.no_scheduled_sampling,
        )
    elif args.eval:
        evaluate_rollout(device=device, num_steps=args.rollout_steps)
    else:
        print("T+1 VAE - Two-Phase Training")
        print()
        print("Phase 1 (Reconstruction):")
        print("  python t_plus_1_vae.py --phase1 --epochs 50 --batch-size 64")
        print()
        print("Phase 2 (Temporal):")
        print("  python t_plus_1_vae.py --train --from-phase1 --epochs 200 --batch-size 32")
        print()
        print("Evaluate:")
        print("  python t_plus_1_vae.py --eval --rollout-steps 25")
