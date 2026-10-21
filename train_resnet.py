import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
import time
import sys

sys.path.append('.')
from sr_resnet import SRResNet
from sr_upscaler import PerceptualLoss

class SRDataset(Dataset):
    def __init__(self, pairs_path):
        self.pairs = []
        with open(pairs_path, 'r') as f:
            for line in f:
                self.pairs.append(json.loads(line))
        self.transform = transforms.ToTensor()
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        target_img = Image.open(pair['target_128']).convert('RGB')
        blurry_64 = Image.open(pair['degraded_64']).convert('RGB')
        return self.transform(blurry_64), self.transform(target_img)
    
    def __len__(self):
        return len(self.pairs)

def train_resnet(device='cpu', epochs=20, batch_size=4, lr=1e-4, resume_path=None):
    print("=" * 60)
    print(f"TRAINING SR RESNET (Device: {device})")
    print("=" * 60)
    
    print("Initializing ResNet...")
    model = SRResNet().to(device)
    
    if resume_path and Path(resume_path).exists():
        print(f"Resuming from {resume_path}...")
        model.load_state_dict(torch.load(resume_path, map_location=device, weights_only=True))
        
    print("Initializing Perceptual Loss...")
    perceptual_loss = PerceptualLoss().to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    pairs_file = Path('dataset/sr_training_true_vae/t_plus_1_sr_pairs.jsonl')
    dataset = SRDataset(pairs_file)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # CPU friendly on Windows
        pin_memory=False
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = {'total': 0, 'mse': 0, 'l1': 0, 'perceptual': 0}
        start_time = time.time()
        
        for batch_idx, (blurry_64, sharp_128) in enumerate(dataloader):
            blurry_64 = blurry_64.to(device)
            sharp_128 = sharp_128.to(device)
            
            pred_128 = model(blurry_64)
            
            loss_mse = F.mse_loss(pred_128, sharp_128)
            loss_l1 = F.l1_loss(pred_128, sharp_128)
            loss_perceptual = perceptual_loss(pred_128, sharp_128)
            
            # Increased L1 weight (1.0 vs 0.5) and decreased perceptual slightly 
            # to prioritize exact color match and reduce the 'cooler' tone shift
            loss = loss_mse + 1.0 * loss_l1 + 0.05 * loss_perceptual
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            epoch_losses['total'] += loss.item()
            epoch_losses['mse'] += loss_mse.item()
            epoch_losses['l1'] += loss_l1.item()
            epoch_losses['perceptual'] += loss_perceptual.item()
            
        n = len(dataloader)
        avg_loss = epoch_losses['total'] / n
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) | Total: {avg_loss:.4f} | MSE: {epoch_losses['mse']/n:.4f} | L1: {epoch_losses['l1']/n:.4f} | Perc: {epoch_losses['perceptual']/n:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'sr_resnet_best.pt')
            
    torch.save(model.state_dict(), 'sr_resnet_final.pt')
    print("Training complete.")
    return model

if __name__ == '__main__':
    device = 'cpu'
    train_resnet(device=device, epochs=25, batch_size=4, resume_path='sr_resnet_final.pt')
