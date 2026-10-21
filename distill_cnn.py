import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from alternative_t2i import DirectCNNT2I
from vision import VisionSystem

class DistillationDataset(Dataset):
    def __init__(self, manifest_path: str, img_size: int = 128):
        self.manifest_path = Path(manifest_path)
        self.data_dir = self.manifest_path.parent
        self.data = []
        
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
                        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = self.data_dir / item['file_name']
        
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image on error
            image_tensor = torch.zeros(3, 128, 128)
            
        return {
            'image': image_tensor,
            'prompt': item['prompt']
        }

def distill_cnn(
    batch_size: int = 16,
    epochs: int = 50,
    lr: float = 1e-4,
    device: str = "cpu"
):
    manifest_path = Path("dataset/distillation/manifest.jsonl")
    if not manifest_path.exists():
        print(f"No distillation data found at {manifest_path}. Generate and approve some frames first!")
        return
        
    dataset = DistillationDataset(manifest_path)
    if len(dataset) == 0:
        print("Dataset is empty.")
        return
        
    print(f"Found {len(dataset)} approved frames for distillation.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    # Initialize models
    print("Loading models...")
    vision = VisionSystem(device=device, load_vae=False)
        model = DirectCNNT2I().to(device)
        
        # Try to load latest checkpoint
        checkpoints = list(Path(".").glob("direct_cnn_t2i*.pt"))
        if checkpoints:
            latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"Resuming from {latest_ckpt}")
            state_dict = torch.load(latest_ckpt, map_location=device, weights_only=True)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict, strict=False)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.L1Loss() # Simple pixel loss for distillation
    
    print("\nStarting Distillation Training...")
    print("The Direct CNN is now learning to mimic the approved Mobile Diffusion outputs.")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            images = batch['image'].to(device)
            prompts = batch['prompt']
            
            # Get text embeddings
            with torch.no_grad():
                inputs = vision.clip_processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                text_embeds = vision.clip_model.get_text_features(**inputs)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            
            optimizer.zero_grad()
            
            # Generate
            # We don't use CFG during basic distillation training to save compute
            pred_images = model(text_embeds, guidance_scale=1.0)
            
            # Loss (compare prediction to the Mobile Diffusion target)
            loss = criterion(pred_images, images)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            save_path = f"direct_cnn_t2i_distilled_ep{epoch+1}.pt"
            torch.save({'model_state_dict': model.state_dict()}, save_path)
            print(f"Saved checkpoint: {save_path}")

if __name__ == "__main__":
    distill_cnn()
