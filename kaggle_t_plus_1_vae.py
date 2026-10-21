# Kaggle T+1 VAE Training
# Upload this + dataset to Kaggle, enable GPU, and run

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from PIL import Image
import numpy as np
from pathlib import Path
import random
from copy import deepcopy
from torchvision import transforms

# ============================================================
# Perceptual Loss (Fixes Blurry VAEs)
# ============================================================

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        # Use VGG16 features (pretrained on ImageNet)
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        blocks.append(vgg[:4].eval())
        blocks.append(vgg[4:9].eval())
        blocks.append(vgg[9:16].eval())
        blocks.append(vgg[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.mse_loss(x, y)
        return loss

# ============ MODEL ARCHITECTURE ============

class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 64->32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32->16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16->8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8->4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
    
    def forward(self, x):
        h = self.conv(x).reshape(x.size(0), -1)  # Use reshape() instead of view() for channels_last
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4->8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8->16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16->32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 5, 4, 2, 1),  # 32->64, 5 channels (2 flow, 3 RGB residual)
        )
    
    def forward(self, z):
        h = self.fc(z).reshape(z.size(0), 256, 4, 4)  # Use reshape() instead of view() for channels_last
        out = self.deconv(h)
        
        # Split into flow and residual
        # BOUND FLOW tightly to [-0.1, 0.1] to prevent wild distortions/pixel explosions
        flow = torch.tanh(out[:, :2, :, :]) * 0.1
        
        # Bound residual to [-0.5, 0.5] to prevent crazy color shifts
        residual = torch.tanh(out[:, 2:, :, :]) * 0.5
        
        return flow, residual


class DynamicsMLP(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_dim),
        )
    
    def forward(self, z):
        return z + self.net(z)  # Residual


class TPlus1VAE(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.dynamics = DynamicsMLP(latent_dim)
        self.target_encoder = deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.ema_decay = 0.99
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    @torch.no_grad()
    def update_target_encoder(self):
        for p, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
            
    def warp(self, x, flow):
        """Warps image x using optical flow field"""
        B, C, H, W = x.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), 2).float().to(x.device)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        
        vgrid = grid + flow.permute(0, 2, 3, 1)
        vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
        vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
        
        return F.grid_sample(x, vgrid, align_corners=True, padding_mode='border')
    
    def forward(self, x_t, x_next=None):
        mu_t, logvar_t = self.encoder(x_t)
        z_t = self.reparameterize(mu_t, logvar_t)
        
        # Recon (Phase 1)
        flow_recon, res_recon = self.decoder(z_t)
        x_recon = torch.clamp(self.warp(x_t, flow_recon) + res_recon, 0, 1)
        
        # Pred (Phase 2)
        z_pred = self.dynamics(z_t)
        flow_pred, res_pred = self.decoder(z_pred)
        x_pred = torch.clamp(self.warp(x_t, flow_pred) + res_pred, 0, 1)
        
        results = {
            'x_recon': x_recon,
            'x_pred': x_pred,
            'flow_pred': flow_pred,
            'res_pred': res_pred,
            'z_t': z_t,
            'z_pred': z_pred,
            'mu': mu_t,
            'logvar': logvar_t,
        }
        
        if x_next is not None:
            with torch.no_grad():
                mu_next, _ = self.target_encoder(x_next)
            results['z_next_target'] = mu_next
        
        return results
    
    def predict_next(self, x_t):
        self.eval()
        with torch.no_grad():
            mu, _ = self.encoder(x_t)
            z_pred = self.dynamics(mu)
            flow_pred, res_pred = self.decoder(z_pred)
            return torch.clamp(self.warp(x_t, flow_pred) + res_pred, 0, 1)


# ============ DATASET ============
import io
import mmap
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Prevent PIL from crashing on corrupted jpegs

class FastImageLoader:
    """Helper to memory-map files for faster initial reads if OS supports it"""
    @staticmethod
    def load(path):
        try:
            # We skip memory mapping on Kaggle network drives because it actually 
            # slows down NFS/FUSE mounts. Standard PIL open is faster for network storage.
            img = Image.open(str(path))
            # Force load into memory before resizing to close file handle quickly
            img.load() 
            return img.convert("RGB")
        except Exception:
            return Image.new('RGB', (64, 64), color='black')

class FramePairDataset(Dataset):
    def __init__(self, image_paths):
        self.pairs = []
        paths = sorted(image_paths)
        for i in range(len(paths) - 1):
            self.pairs.append((paths[i], paths[i+1]))
        random.shuffle(self.pairs)
        
        # Pre-allocate a tensor for faster transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        try:
            p1, p2 = self.pairs[idx]
            # Use OpenCV for much faster image loading than PIL
            import cv2
            import warnings
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Use IMREAD_UNCHANGED for faster raw reading before color conversion
                img1 = cv2.imread(str(p1), cv2.IMREAD_UNCHANGED)
                img2 = cv2.imread(str(p2), cv2.IMREAD_UNCHANGED)
            
            if img1 is None or img2 is None:
                raise ValueError("Could not read image")
                
            # Convert to RGB (handles BGR from cv2.imread and BGRA/RGBA with alpha channels)
            if len(img1.shape) == 3:
                if img1.shape[2] == 4: # Has alpha channel
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2RGB)
                elif img1.shape[2] == 3: # Standard BGR
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                else:
                    # Fallback for weird channel counts (e.g. 5)
                    img1 = img1[:, :, :3]
            elif len(img1.shape) == 2: # Grayscale
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
                
            if len(img2.shape) == 3:
                if img2.shape[2] == 4:
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2RGB)
                elif img2.shape[2] == 3:
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                else:
                    img2 = img2[:, :, :3]
            elif len(img2.shape) == 2:
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
            
            # Final safety check to absolutely guarantee 3 channels
            if len(img1.shape) == 3 and img1.shape[2] != 3:
                img1 = img1[:, :, :3]
            if len(img2.shape) == 3 and img2.shape[2] != 3:
                img2 = img2[:, :, :3]
            
            # Use INTER_NEAREST for massive speedup (we are only resizing to 64x64 anyway)
            img1 = cv2.resize(img1, (64, 64), interpolation=cv2.INTER_NEAREST)
            img2 = cv2.resize(img2, (64, 64), interpolation=cv2.INTER_NEAREST)
            
            t1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
            t2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
            
            return t1, t2
        except Exception as e:
            return torch.rand(3, 64, 64), torch.rand(3, 64, 64)


def get_all_images_in_dir(directory: Path):
    """Safely get all images in a directory."""
    imgs = []
    import os
    # Convert back to pathlib Path to be safe
    base_dir = Path(directory)
    if base_dir.exists():
        for ext in ("*.jpg", "*.png", "*.jpeg", "*.webp"):
            # Use glob instead of walk for better symlink resolution in python 3
            imgs.extend(list(base_dir.rglob(ext)))
    return imgs

# ============ TRAINING ============

def train_phase1(device, epochs=50, batch_size=128, lr=1e-3, use_compile=True, grad_accum_steps=1):
    """Phase 1: Reconstruction pretraining on scraped images"""
    print("=" * 60)
    print("PHASE 1: Reconstruction Pretraining (GPU)")
    print(f"  Batch size: {batch_size} (effective: {batch_size * grad_accum_steps})")
    print(f"  Gradient accumulation: {grad_accum_steps} steps")
    if grad_accum_steps > 1:
        print(f"  NOTE: Processing {batch_size} samples in {grad_accum_steps} steps of {batch_size // grad_accum_steps}")
    print(f"  Model compilation: {use_compile}")
    print("=" * 60)
    
    # Create model first (don't compile yet)
    model = TPlus1VAE(latent_dim=64).to(device)
    
    # DON'T use channels_last for now - testing if it causes issues
    # if device == "cuda":
    #     model = model.to(memory_format=torch.channels_last)
    
    start_epoch = 0
    
    # Check for checkpoint in Kaggle input datasets FIRST (uploaded models)
    kaggle_checkpoint_paths = [
        Path("/kaggle/input/datasets/meowkittypaws/latest/t_plus_1_vae_recon_latest.pt"),
        Path("/kaggle/input/datasets/meowkittypaws/latest/t_plus_1_vae_recon.pt"),
        Path("/kaggle/input/latest/t_plus_1_vae_recon_latest.pt"),
        Path("/kaggle/input/latest/t_plus_1_vae_recon.pt"),
    ]
    
    checkpoint_loaded = False
    for ckpt_path in kaggle_checkpoint_paths:
        if ckpt_path.exists():
            print(f"Loading checkpoint from Kaggle input: {ckpt_path}")
            state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=True)
            
            # DIAGNOSTIC: Check checkpoint contents
            print(f"\nCHECKPOINT DIAGNOSTIC:")
            print(f"  Keys in checkpoint: {len(state_dict.keys())}")
            print(f"  First few keys: {list(state_dict.keys())[:5]}")
            
            # Check parameter counts
            total_params = sum(p.numel() for p in state_dict.values())
            print(f"  Total parameters in checkpoint: {total_params:,}")
            print(f"  Checkpoint size: {sum(p.numel() * 4 for p in state_dict.values()) / 1e6:.2f}MB")
            
            # Handle architecture change (Decoder output channels 3 -> 5)
            if 'decoder.deconv.9.weight' in state_dict and state_dict['decoder.deconv.9.weight'].shape[1] != 5:
                print("  ⚠️ Architecture change detected! Dropping last decoder layer to start fresh.")
                del state_dict['decoder.deconv.9.weight']
                del state_dict['decoder.deconv.9.bias']
            
            model.load_state_dict(state_dict, strict=False)
            
            # Check loaded model
            model_params = sum(p.numel() for p in model.parameters())
            print(f"  Model parameters after loading: {model_params:,}")
            print(f"  Model size: {sum(p.numel() * 4 for p in model.parameters()) / 1e6:.2f}MB")
            
            if model_params < 1_000_000:
                print(f"  ⚠️ WARNING: Model is suspiciously small!")
            
            checkpoint_loaded = True
            break
    
    # Then check current working directory (for resume during same run)
    if not checkpoint_loaded and Path("t_plus_1_vae_recon_latest.pt").exists() and not Path("t_plus_1_vae_recon.pt").exists():
        print("Resuming from t_plus_1_vae_recon_latest.pt!")
        state_dict = torch.load("t_plus_1_vae_recon_latest.pt", map_location=device, weights_only=True)
        
        # Handle architecture change (Decoder output channels 3 -> 5)
        if 'decoder.deconv.9.weight' in state_dict and state_dict['decoder.deconv.9.weight'].shape[1] != 5:
            print("  ⚠️ Architecture change detected! Dropping last decoder layer to start fresh.")
            del state_dict['decoder.deconv.9.weight']
            del state_dict['decoder.deconv.9.bias']
            
        model.load_state_dict(state_dict, strict=False)
        # Optional: could parse epoch from file if you want accurate numbering, 
        # but starting display from 0 is fine as long as weights are loaded.
    
    # ALWAYS print model info (whether loaded from checkpoint or fresh)
    print(f"\nMODEL INFO:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model memory: {total_params * 4 / 1e6:.2f}MB")
    
    # Expected size for TPlus1VAE with latent_dim=64:
    # Encoder: ~1.1M params (trainable)
    # Decoder: ~1.1M params (trainable)
    # Dynamics: ~0.02M params (trainable)
    # Target Encoder: ~1.1M params (FROZEN - not trainable)
    # Total: ~3.4M params, ~2.2M trainable
    expected_params = 3_400_000
    expected_trainable = 2_200_000
    
    if total_params < expected_params * 0.8 or total_params > expected_params * 1.2:
        print(f"  ⚠️ WARNING: Model has {total_params:,} params, expected ~{expected_params:,}")
        print(f"  Model might be corrupted or from different architecture!")
    else:
        print(f"  ✓ Model size looks correct (~{expected_params/1e6:.1f}M params)")
    
    if trainable_params < expected_trainable * 0.8 or trainable_params > expected_trainable * 1.2:
        print(f"  ⚠️ WARNING: {trainable_params:,} trainable params, expected ~{expected_trainable:,}")
    else:
        print(f"  ✓ Trainable params look correct (~{expected_trainable/1e6:.1f}M)")
    
    print()
    
    # NOW compile the model AFTER loading checkpoint (if any)
    if use_compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode='reduce-overhead')
    
    # In Kaggle, we symlinked the datasets to /kaggle/working/datasets/scraped-images
    # but the glob might fail if it's following symlinks incorrectly. Let's point directly to Kaggle's input!
    
    # We want ALL images from ALL available datasets, not just the first one it finds
    all_paths_to_check = [
        Path("/kaggle/input/datasets/meowkittypaws/scraped-images"),
        Path("/kaggle/input/datasets/srinivasac/flickr8k-dataset"),
        Path("/kaggle/input/datasets/reemjunaid/coco14-cc12m"),
        Path("/kaggle/working/datasets/scraped-images"),
        Path("/kaggle/working/datasets/flickr8k-dataset"),
        Path("/kaggle/working/datasets/coco14-cc12m")
    ]
    
    all_images_set = set()
    image_dirs = []
    
    for p in all_paths_to_check:
        if p.exists():
            print(f"Checking path: {p}")
            imgs = get_all_images_in_dir(p)
            if imgs:
                image_dirs.append(p)
                # Use resolve() to deduplicate symlinks pointing to the same file
                for img_path in imgs:
                    all_images_set.add(img_path.resolve())
                    
    all_images = list(all_images_set)
                
    if not all_images:
        # Final fallback - search everywhere in /kaggle/input for ANY images
        print("Falling back to full /kaggle/input search...")
        imgs = get_all_images_in_dir(Path("/kaggle/input"))
        if imgs:
            image_dirs.append(Path("/kaggle/input"))
            all_images.extend(imgs)
                    
    if not all_images:
        raise ValueError(f"No scraped images found anywhere! Please check dataset paths.")

    print(f"Loading {len(all_images)} images from {len(image_dirs)} directories...")

    dataset = SingleImageDataset(all_images)
    
    # Optimize dataloader for maximum throughput
    # Kaggle T4 has 4 CPU cores - use 4 workers max (not 6!)
    num_workers = 4 if device == "cuda" else 2
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=3,      # Prefetch more batches
        persistent_workers=True if num_workers > 0 else False
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    kl_warmup = 10
    kl_target = 1e-4
    
    # Enable mixed precision for 2x memory savings + speed
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        kl_beta = min(1.0, epoch / kl_warmup) * kl_target
        
        # GPU diagnostics
        if device == "cuda":
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            print(f"GPU Memory cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")
        
        batch_times = []
        import time
        
        for batch_idx, x in enumerate(dataloader):
            batch_start = time.time()
            
            # FORCE everything to GPU explicitly
            x = x.to(device)
            
            # Print DETAILED diagnostic for first batch
            if batch_idx == 0 and epoch == 0:
                print(f"\n{'='*60}")
                print(f"FIRST BATCH DIAGNOSTIC:")
                print(f"{'='*60}")
                print(f"Batch shape: {x.shape}, device: {x.device}, dtype: {x.dtype}")
                print(f"Model device: {next(model.parameters()).device}")
                print(f"GPU Memory (just data): {torch.cuda.memory_allocated()/1e9:.2f}GB")
                
                # Test a simple forward pass WITHOUT autocast or channels_last
                print(f"\nTesting RAW forward pass (no autocast, no channels_last)...")
                with torch.no_grad():
                    test_mu, test_logvar = model.encoder(x[:32])  # Small batch
                    print(f"Encoder output shape: {test_mu.shape}")
                    print(f"GPU Memory (after encoder): {torch.cuda.memory_allocated()/1e9:.2f}GB")
                    test_z = model.reparameterize(test_mu, test_logvar)
                    test_recon = model.decoder(test_z)
                    print(f"Decoder output shape: {test_recon.shape}")
                    print(f"GPU Memory (after decoder): {torch.cuda.memory_allocated()/1e9:.2f}GB")
                    del test_mu, test_logvar, test_z, test_recon
                
                print(f"RAW forward pass successful!")
                print(f"GPU Memory (after cleanup): {torch.cuda.memory_allocated()/1e9:.2f}GB")
                print(f"{'='*60}\n")
            
            # TRAINING with mixed precision (FP16)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    mu, logvar = model.encoder(x)
                    z = model.reparameterize(mu, logvar)
                    flow_recon, res_recon = model.decoder(z)
                    x_recon = torch.clamp(model.warp(x, flow_recon) + res_recon, 0, 1)
                    
                    loss_recon = F.mse_loss(x_recon, x)
                    loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss_flow_reg = torch.mean(torch.abs(flow_recon)) * 0.01 # Regularize flow to be small
                    loss = (loss_recon + kl_beta * loss_kl + loss_flow_reg) / grad_accum_steps
                
                # Check GPU memory during actual training
                if batch_idx == 0 and epoch == 0:
                    print(f"TRAINING forward pass - GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                    
                    # Calculate expected memory
                    batch_mem = x.numel() * 4 / 1e9  # Input batch (FP32)
                    model_mem = sum(p.numel() * 4 for p in model.parameters()) / 1e9
                    # With mixed precision, activations use ~2 bytes instead of 4
                    # Rough estimate: batch_size * channels * spatial * layers * 2 bytes
                    expected_activation = (batch_size * 256 * 16 * 16 * 10 * 2) / 1e9  # ~10 layers avg
                    print(f"Expected components (FP16):")
                    print(f"  - Batch: {batch_mem:.2f}GB (FP32)")
                    print(f"  - Model: {model_mem:.2f}GB (FP32)")
                    print(f"  - Activations (est): {expected_activation:.2f}GB (FP16)")
                    print(f"  - Total expected: {batch_mem + model_mem + expected_activation:.2f}GB")
                    print(f"  - With gradients: {(batch_mem + model_mem + expected_activation) * 2:.2f}GB")
                
                # SCALE the loss before backward!
                scaler.scale(loss).backward()
                
                if batch_idx == 0 and epoch == 0:
                    print(f"TRAINING backward pass - GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                    if torch.cuda.memory_allocated()/1e9 > 4.0:
                        print(f"[OK] GPU memory usage looks good!")
                    else:
                        print(f"[WARNING] Still low - try even larger batch size?")
                
                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                # CPU fallback (no mixed precision)
                mu, logvar = model.encoder(x)
                z = model.reparameterize(mu, logvar)
                flow_recon, res_recon = model.decoder(z)
                x_recon = torch.clamp(model.warp(x, flow_recon) + res_recon, 0, 1)
                
                loss_recon = F.mse_loss(x_recon, x)
                loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss_flow_reg = torch.mean(torch.abs(flow_recon)) * 0.01
                loss = (loss_recon + kl_beta * loss_kl + loss_flow_reg) / grad_accum_steps
                
                loss.backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += loss.item() * grad_accum_steps
            
            # Track batch time
            batch_times.append(time.time() - batch_start)
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_batch_time = sum(batch_times[-10:]) / len(batch_times[-10:])
                if device == "cuda":
                    # Get actual GPU utilization
                    try:
                        import subprocess
                        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                              capture_output=True, text=True, timeout=1)
                        gpu_util = result.stdout.strip().split('\n')[0] if result.returncode == 0 else "N/A"
                    except:
                        gpu_util = "N/A"
                    
                    print(f"  Batch {batch_idx+1}/{len(dataloader)} | "
                          f"Loss: {(epoch_loss/(batch_idx+1)):.4f} | "
                          f"Time: {avg_batch_time:.3f}s/batch | "
                          f"GPU Mem: {torch.cuda.memory_allocated()/1e9:.2f}GB | "
                          f"GPU Util: {gpu_util}%")
                else:
                    print(f"  Batch {batch_idx+1}/{len(dataloader)} | "
                          f"Loss: {(epoch_loss/(batch_idx+1)):.4f} | "
                          f"Time: {avg_batch_time:.3f}s/batch")
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
        
        # Save every epoch instead of every 10 epochs
        torch.save(model.state_dict(), f"t_plus_1_vae_recon_ep{epoch+1}.pt")
        # Also maintain a 'latest' pointer file for easy resuming
        torch.save(model.state_dict(), "t_plus_1_vae_recon_latest.pt")
    
    torch.save(model.state_dict(), "t_plus_1_vae_recon.pt")
    print("Phase 1 complete!")
    return model


def train_phase2(model, device, epochs=100, batch_size=64, lr=5e-4, use_compile=True, grad_accum_steps=1):
    """Phase 2: Temporal dynamics on video frame pairs"""
    print("=" * 60)
    print("PHASE 2: Temporal Training (GPU)")
    print(f"  Batch size: {batch_size} (effective: {batch_size * grad_accum_steps})")
    print(f"  Gradient accumulation: {grad_accum_steps} steps")
    print("=" * 60)
    
    # Check for checkpoint in Kaggle input datasets FIRST (BEFORE compiling)
    kaggle_checkpoint_paths = [
        Path("/kaggle/input/datasets/meowkittypaws/latest/t_plus_1_vae_final.pt"),
        Path("/kaggle/input/datasets/meowkittypaws/latest/t_plus_1_vae_latest.pt"),
        Path("/kaggle/input/datasets/meowkittypaws/latest/t_plus_1_vae_recon.pt"),
        Path("/kaggle/input/latest/t_plus_1_vae_final.pt"),
        Path("/kaggle/input/latest/t_plus_1_vae_latest.pt"),
        Path("/kaggle/input/latest/t_plus_1_vae_recon.pt"),
    ]
    
    checkpoint_loaded = False
    for ckpt_path in kaggle_checkpoint_paths:
        if ckpt_path.exists():
            print(f"Loading Phase 2 checkpoint from Kaggle input: {ckpt_path}")
            state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=True)
            if 'decoder.deconv.9.weight' in state_dict and state_dict['decoder.deconv.9.weight'].shape[1] != 5:
                print("  ⚠️ Architecture change detected! Dropping last decoder layer to start fresh.")
                del state_dict['decoder.deconv.9.weight']
                del state_dict['decoder.deconv.9.bias']
            model.load_state_dict(state_dict, strict=False)
            checkpoint_loaded = True
            if "final" in str(ckpt_path):
                print("Phase 2 already finished! (Found t_plus_1_vae_final.pt)")
                return model
            break
    
    # Then check current working directory
    if not checkpoint_loaded:
        def load_ckpt(path):
            state_dict = torch.load(path, map_location=device, weights_only=True)
            if 'decoder.deconv.9.weight' in state_dict and state_dict['decoder.deconv.9.weight'].shape[1] != 5:
                print("  ⚠️ Architecture change detected! Dropping last decoder layer to start fresh.")
                del state_dict['decoder.deconv.9.weight']
                del state_dict['decoder.deconv.9.bias']
            model.load_state_dict(state_dict, strict=False)
            
        # Check latest first!
        if Path("/kaggle/working/t_plus_1_vae_latest.pt").exists():
            print("Resuming Phase 2 from /kaggle/working/t_plus_1_vae_latest.pt!")
            load_ckpt("/kaggle/working/t_plus_1_vae_latest.pt")
        elif Path("t_plus_1_vae_latest.pt").exists():
            print("Resuming Phase 2 from t_plus_1_vae_latest.pt!")
            load_ckpt("t_plus_1_vae_latest.pt")
        elif Path("t_plus_1_vae_phase2_final.pt").exists():
            print("Starting Phase 2 from t_plus_1_vae_phase2_final.pt (Local Checkpoint)!")
            load_ckpt("t_plus_1_vae_phase2_final.pt")
        elif Path("t_plus_1_vae_final.pt").exists():
            print("Starting Phase 2 from t_plus_1_vae_final.pt!")
            load_ckpt("t_plus_1_vae_final.pt")
        elif Path("t_plus_1_vae_recon.pt").exists():
            print("Starting Phase 2 from finished Phase 1 weights.")
            load_ckpt("t_plus_1_vae_recon.pt")
            
    # Same for frames - try direct paths
    possible_paths = [
        Path("/kaggle/input/datasets/meowkittypaws/t-plus-1-curated/dataset/t_plus_1_curated"),
        Path("/kaggle/input/datasets/meowkittypaws/t-plus-1-curated"),
        Path("/kaggle/input/t-plus-1-curated/dataset/t_plus_1_curated"),
        Path("/kaggle/input/t-plus-1-curated"),
        Path("/kaggle/input/datasets/visions-frames/t_plus_1_training"),
        Path("/kaggle/input/visions-frames/t_plus_1_training"),
        Path("/kaggle/working/datasets/visions-frames"),
        Path("/kaggle/working/datasets")
    ]
    
    all_frames_set = set()
    frames_dir = None
    
    # ALWAYS do the full fallback search to ensure we get ALL datasets attached to the notebook
    print("Searching /kaggle/input for all available frame datasets...")
    for p in Path("/kaggle/input").rglob("*"):
        if p.is_dir() and ("frame" in p.name.lower() or "curated" in str(p).lower() or "scraped" in str(p).lower() or "flickr" in str(p).lower() or "coco" in str(p).lower() or "cc12m" in str(p).lower() or "dataset" in str(p).lower()):
            # Check if this is a directory containing images directly
            imgs = get_all_images_in_dir(p)
            if len(imgs) >= 2:
                # IMPORTANT: Only add images if they are in the same directory, 
                # to avoid recursively adding the same images multiple times
                is_leaf_dir = True
                for child in p.iterdir():
                    if child.is_dir():
                        is_leaf_dir = False
                        break
                
                if is_leaf_dir:
                    frames_dir = p
                    for img_path in imgs:
                        all_frames_set.add(img_path.resolve())
                    print(f"Found {len(imgs)} images in: {p}")
                
    all_frames = list(all_frames_set)
    
    # Suppress libpng warnings globally for OpenCV
    import os
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"
    os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    
    # Also suppress Python warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    if not all_frames:
        print("WARNING: Phase 2 skipping - no frames directory found or not enough frames.")
        return model
        
    print(f"Loading {len(all_frames)} frames from {frames_dir}...")
    
    dataset = FramePairDataset(all_frames)
    
    # Optimize dataloader
    # Kaggle T4x2 instances have 4 CPU cores.
    num_workers = 4 if device == "cuda" else 2
    
    # KAGGLE SPEED FIX: 
    # 1. Drop batch size slightly so the CPU can keep up with resizing
    # 2. Drop prefetch factor to 2 to prevent RAM thrashing
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, # Use the argument passed to the function
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=2, # Lowered to prevent RAM bottlenecks
        persistent_workers=True
    )
    
    model = model.to(device)
    
    # Use channels_last for convnets
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    
    # Compile if not already compiled (skip if already compiled in Phase 1)
    if use_compile and hasattr(torch, 'compile'):
        try:
            # Check if already compiled (avoid double-compile)
            is_compiled = hasattr(model, '_orig_mod')
            if not is_compiled:
                print("Compiling model with torch.compile()...")
                model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            print(f"Could not compile model: {e}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Add VGG Perceptual Loss to fix blurriness
    print("Loading VGG16 for Perceptual Loss...")
    vgg_loss = VGGPerceptualLoss(resize=False).to(device)
    print("VGG16 loaded successfully.")
    
    # Add a learning rate scheduler to help it settle into fine details
    # Reduced patience to 3 so it drops the LR much faster if it gets stuck
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    kl_beta = 1e-3
    
    # Enable mixed precision
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    
    for epoch in range(epochs):
        model.train()
        losses = {'total': 0, 'recon': 0, 'pred': 0, 'latent': 0}
        
        for batch_idx, (x_t, x_next) in enumerate(dataloader):
            x_t = x_t.to(device, non_blocking=True)
            x_next = x_next.to(device, non_blocking=True)
            
            # Use channels_last
            if device == "cuda":
                x_t = x_t.to(memory_format=torch.channels_last)
                x_next = x_next.to(memory_format=torch.channels_last)
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    out = model(x_t, x_next)
                    
                    loss_recon = F.mse_loss(out['x_recon'], x_t)
                    loss_pred = F.mse_loss(out['x_pred'], x_next)
                    
                    # Add L1 loss (Mean Absolute Error) to force sharper edges
                    loss_recon_l1 = F.l1_loss(out['x_recon'], x_t)
                    loss_pred_l1 = F.l1_loss(out['x_pred'], x_next)
                    
                    # Add Perceptual Loss (VGG) to force texture/edge matching
                    loss_perc_recon = vgg_loss(out['x_recon'], x_t)
                    loss_perc_pred = vgg_loss(out['x_pred'], x_next)
                    loss_perceptual = (loss_perc_recon + loss_perc_pred) * 0.1
                    
                    loss_latent = F.mse_loss(out['z_pred'], out['z_next_target'])
                    loss_kl = -0.5 * torch.mean(1 + out['logvar'] - out['mu'].pow(2) - out['logvar'].exp())
                    loss_flow_reg = torch.mean(torch.abs(out['flow_pred'])) * 0.01
                    
                    # Combine MSE, L1, and Perceptual for maximum sharpness
                    loss = (loss_recon + loss_recon_l1 + loss_pred + loss_pred_l1 + loss_perceptual + 0.1 * loss_latent + kl_beta * loss_kl + loss_flow_reg) / grad_accum_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Update EMA after optimizer step
                    model.update_target_encoder()
            else:
                # CPU fallback
                out = model(x_t, x_next)
                
                loss_recon = F.mse_loss(out['x_recon'], x_t)
                loss_pred = F.mse_loss(out['x_pred'], x_next)
                
                # Add L1 loss (Mean Absolute Error) to force sharper edges
                loss_recon_l1 = F.l1_loss(out['x_recon'], x_t)
                loss_pred_l1 = F.l1_loss(out['x_pred'], x_next)
                
                # Add Perceptual Loss (VGG) to force texture/edge matching
                loss_perc_recon = vgg_loss(out['x_recon'], x_t)
                loss_perc_pred = vgg_loss(out['x_pred'], x_next)
                loss_perceptual = (loss_perc_recon + loss_perc_pred) * 0.1
                
                loss_latent = F.mse_loss(out['z_pred'], out['z_next_target'])
                loss_kl = -0.5 * torch.mean(1 + out['logvar'] - out['mu'].pow(2) - out['logvar'].exp())
                loss_flow_reg = torch.mean(torch.abs(out['flow_pred'])) * 0.01
                
                loss = (loss_recon + loss_recon_l1 + loss_pred + loss_pred_l1 + loss_perceptual + 0.1 * loss_latent + kl_beta * loss_kl + loss_flow_reg) / grad_accum_steps
                
                loss.backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    model.update_target_encoder()
            
            losses['total'] += loss.item() * grad_accum_steps
            losses['recon'] += loss_recon.item()
            losses['pred'] += loss_pred.item()
            losses['latent'] += loss_latent.item()
        
        n = len(dataloader)
        avg_total_loss = losses['total']/n
        print(f"Epoch {epoch+1}/{epochs} | Total: {avg_total_loss:.4f} | Recon: {losses['recon']/n:.4f} | Pred: {losses['pred']/n:.4f}")
        
        # Step the scheduler based on the total loss
        scheduler.step(avg_total_loss)
        
        # Save every epoch
        torch.save(model.state_dict(), f"t_plus_1_vae_ep{epoch+1}.pt")
        torch.save(model.state_dict(), "t_plus_1_vae_latest.pt")
    
    torch.save(model.state_dict(), "t_plus_1_vae_final.pt")
    print("Phase 2 complete!")
    return model


# ============ MAIN ============
import shutil

def setup_kaggle_checkpoints():
    """Copy checkpoints from input datasets to working directory to allow resuming."""
    input_dir = Path("/kaggle/input")
    work_dir = Path("/kaggle/working/visions") if Path("/kaggle/working/visions").exists() else Path("/kaggle/working")
    
    print(f"Checking for existing checkpoints to resume in {input_dir}...")
    found = False
    
    # Check for all our typical checkpoint names
    checkpoint_names = [
        "t_plus_1_vae_latest.pt",
        "t_plus_1_vae_phase2_final.pt",
        "t_plus_1_vae_final.pt",
        "t_plus_1_vae_recon.pt",
        "t_plus_1_vae_recon_latest.pt"
    ]
    
    for pt in input_dir.rglob("*.pt"):
        if any(name in pt.name for name in checkpoint_names):
            dest = work_dir / pt.name
            # Only copy if we don't already have a newer version in working
            if not dest.exists() or dest.stat().st_mtime < pt.stat().st_mtime:
                print(f"Copying {pt.name} from {pt.parent.name} to {work_dir}")
                try:
                    shutil.copy2(pt, dest)
                    found = True
                except Exception as e:
                    print(f"Failed to copy {pt.name}: {e}")
                
    if not found:
        print("No prior checkpoints found in /kaggle/input.")


if __name__ == "__main__":
    # Move checkpoints first before anything else
    setup_kaggle_checkpoints()
    
    # Enable cuDNN benchmark for faster convolutions on static input sizes
    torch.backends.cudnn.benchmark = True
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # GPU verification test
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Quick tensor test
        test_tensor = torch.randn(100, 100).cuda()
        test_result = test_tensor @ test_tensor
        print(f"✓ GPU tensor ops working! Result shape: {test_result.shape}")
        del test_tensor, test_result
        torch.cuda.empty_cache()
        print(f"✓ GPU memory cleared")
        print("=" * 60)
    
    if Path("t_plus_1_vae_recon.pt").exists() and not Path("t_plus_1_vae_recon_latest.pt").exists():
        # Edge case: finished phase 1 but notebook restarted before phase 2
        print("Phase 1 already finished! Skipping to Phase 2.")
        model = TPlus1VAE(latent_dim=64).to(device)
        model.load_state_dict(torch.load("t_plus_1_vae_recon.pt", map_location=device))
    elif Path("t_plus_1_vae_phase2_final.pt").exists() or Path("t_plus_1_vae_final.pt").exists() or Path("t_plus_1_vae_latest.pt").exists():
        # Phase 2 already started or finished, skip Phase 1 entirely
        print("Phase 2 checkpoint found! Skipping Phase 1.")
        model = TPlus1VAE(latent_dim=64).to(device)
        
        # Load the latest checkpoint to ensure we have the correct architecture
        latest_ckpt = None
        for p in ["t_plus_1_vae_latest.pt", "t_plus_1_vae_final.pt", "t_plus_1_vae_phase2_final.pt"]:
            if Path(p).exists():
                latest_ckpt = p
                break
                
        if latest_ckpt:
            state_dict = torch.load(latest_ckpt, map_location=device, weights_only=True)
            if 'decoder.deconv.9.weight' in state_dict and state_dict['decoder.deconv.9.weight'].shape[1] != 5:
                print("  ⚠️ Architecture change detected! Dropping last decoder layer to start fresh.")
                del state_dict['decoder.deconv.9.weight']
                del state_dict['decoder.deconv.9.bias']
            model.load_state_dict(state_dict, strict=False)
    else:
        # Start in the right directory depending on Kaggle setup
        work_dir = Path("/kaggle/working/visions") if Path("/kaggle/working/visions").exists() else Path("/kaggle/working")
        import os
        os.chdir(work_dir)
        
        # Phase 1: TRULY MAXIMIZE GPU with massive batch
        # - Model is 3.4M params (tiny!)
        # - Batch 2048 only uses 2-3GB
        # - T4 has 16GB available
        # - Testing batch 4096 to saturate GPU
        
        print("Starting Phase 1!")
        model = train_phase1(
            device, 
            epochs=50, 
            batch_size=4096,     # ← 2x larger! Should use 4-6GB
            lr=1e-3,
            use_compile=False,
            grad_accum_steps=1
        )   
        
    
    # Phase 2: OPTIMIZED
    if not Path("t_plus_1_vae_final.pt").exists():
        model = train_phase2(
            model, 
            device, 
            epochs=1000, 
            batch_size=256,      # ← Dropped to 256 to ensure it fits in GPU memory with VGG loss
            lr=5e-5,             
            use_compile=False,   
            grad_accum_steps=4   # ← Accumulate 4 steps to simulate batch 1024
        )
    else:
        print("Everything is complete!")
        
    print("Training complete! Download t_plus_1_vae_final.pt")
