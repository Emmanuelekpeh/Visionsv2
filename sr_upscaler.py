"""
Super-Resolution Upscaler for T+1 VAE
Upscales 64x64 → 128x128 while preserving sharp details

Architecture: Efficient Sub-Pixel CNN (ESPCN) variant
- Fast inference (single forward pass)
- Perceptual loss (VGG) for sharp textures
- No blur, no hallucination
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torchvision.models as models
# Avoid hanging download by explicitly skipping pre-trained load if unavailable or caching issues occur, but we assume it's already cached.
class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for sharp texture preservation"""
    def __init__(self):
        super().__init__()
        # Use VGG16 features
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.feature_blocks = nn.ModuleList([
            vgg[:4],   # conv1_2
            vgg[4:9],  # conv2_2
            vgg[9:16], # conv3_3
        ])
        for block in self.feature_blocks:
            for param in block.parameters():
                param.requires_grad = False
            block.eval()
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, pred, target):
        """Compute perceptual loss between prediction and target"""
        # Normalize for VGG
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        loss = 0.0
        for block in self.feature_blocks:
            pred = block(pred)
            target = block(target)
            loss += F.mse_loss(pred, target)
        
        return loss




class SRUpscaler(nn.Module):
    """
    Super-Resolution Upscaler: 64x64 → 128x128
    
    Design principles:
    - ESPCN architecture (ultra-fast, CPU-friendly)
    - Sharp (trained with perceptual loss on real VAE artifacts)
    - Single forward pass with sub-pixel convolution upsampling at the end
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        
        # True ESPCN architecture (shallow, no residual blocks, sub-pixel upsampling at the very end)
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 3 * (scale_factor ** 2), 3, 1, 1),
            nn.PixelShuffle(scale_factor),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor [B, 3, 64, 64] - T+1 VAE output
        Returns:
            Tensor [B, 3, 128, 128] - Sharp upscaled image
        """
        return self.net(x)




# ============================================================
# TRAINING
# ============================================================

def train_sr_upscaler(
    device='cuda',
    epochs=50,
    batch_size=32,
    lr=1e-4,
    dataset_path='dataset/sr_training_true_vae'
):
    """
    Train SR upscaler on T+1 dataset
    
    Strategy:
    - Input: T+1 VAE predictions (blurry 64x64)
    - Target: Ground truth frames (sharp 128x128 downsampled to 64x64 then upsampled)
    - Loss: MSE + Perceptual (VGG) + L1
    """
    print("=" * 60)
    print("TRAINING SR UPSCALER")
    print("=" * 60)
    
    from pathlib import Path
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    import json
    
    # Load model
    model = SRUpscaler().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Dataset
    class SRDataset(Dataset):
        def __init__(self, pairs_path):
            self.pairs = []
            with open(pairs_path, 'r') as f:
                for line in f:
                    self.pairs.append(json.loads(line))
            
            self.transform = transforms.ToTensor()
        
        def __getitem__(self, idx):
            pair = self.pairs[idx]
            
            # Load TARGET frame (sharp 128x128 ground truth)
            target_img = Image.open(pair['target_128']).convert('RGB')
            target_128 = target_img
            
            # Load DEGRADED frame (actual T+1 VAE output, 64x64)
            blurry_64 = Image.open(pair['degraded_64']).convert('RGB')
            
            return self.transform(blurry_64), self.transform(target_128)
        
        def __len__(self):
            return len(self.pairs)
    
    pairs_file = Path(dataset_path) / 't_plus_1_sr_pairs.jsonl'
    if not pairs_file.exists():
        raise FileNotFoundError(f"Dataset not found: {pairs_file}. Did you run generate_sr_dataset.py first?")
    
    dataset = SRDataset(pairs_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"Loaded {len(dataset)} training pairs")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = {'total': 0, 'mse': 0, 'l1': 0, 'perceptual': 0}
        
        for batch_idx, (blurry_64, sharp_128) in enumerate(dataloader):
            blurry_64 = blurry_64.to(device)
            sharp_128 = sharp_128.to(device)
            
            # CPU fallback (No scaler)
            pred_128 = model(blurry_64)
            
            loss_mse = F.mse_loss(pred_128, sharp_128)
            loss_l1 = F.l1_loss(pred_128, sharp_128)
            loss_perceptual = perceptual_loss(pred_128, sharp_128)
            
            loss = loss_mse + 0.5 * loss_l1 + 0.1 * loss_perceptual
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Track losses
            epoch_losses['total'] += loss.item()
            epoch_losses['mse'] += loss_mse.item()
            epoch_losses['l1'] += loss_l1.item()
            epoch_losses['perceptual'] += loss_perceptual.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  [{epoch+1}/{epochs}] Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | MSE: {loss_mse.item():.4f} | "
                      f"Perceptual: {loss_perceptual.item():.4f}")
        
        # Epoch summary
        n = len(dataloader)
        avg_loss = epoch_losses['total'] / n
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Total: {avg_loss:.4f} | "
              f"MSE: {epoch_losses['mse']/n:.4f} | "
              f"L1: {epoch_losses['l1']/n:.4f} | "
              f"Perceptual: {epoch_losses['perceptual']/n:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'sr_upscaler_best.pt')
            print(f"  -> Saved best model (loss: {best_loss:.4f})")
        
        # Save epoch checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'sr_upscaler_ep{epoch+1}.pt')
    
    # Save final model
    torch.save(model.state_dict(), 'sr_upscaler_final.pt')
    print("\n" + "=" * 60)
    print("SR UPSCALER TRAINING COMPLETE")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    device = 'cpu' # force cpu for this test
    print(f"Using device: {device}")
    
    # Train upscaler
    model = train_sr_upscaler(
        device=device,
        epochs=10,
        batch_size=4,
        lr=1e-4
    )
