import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
from pathlib import Path

from alternative_t2i import DirectCNNT2I, DirectContinuationCNN, SimpleDiscriminator
from vision import VisionSystem
import torchvision.transforms.functional as TF
import random

def distill_student_model(device="cpu", epochs=15, lr=2e-4, checkpoint_path="direct_cnn_t2i_kaggle_ep830.pt", t_plus_1_path="t_plus_1_student.pt"):
    """
    Trains TWO models simultaneously:
    1. The base Direct CNN (Text -> Image)
    2. The new T+1 Continuation CNN (Image + Text -> Next Image)
    """
    distill_dir = Path("dataset/distillation")
    manifest_path = distill_dir / "manifest.jsonl"
    
    if not manifest_path.exists():
        print("No distillation manifest found. Skipping distillation.")
        return False
        
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    if not records:
        print("Manifest is empty. Skipping distillation.")
        return False
        
    print(f"--- Starting Real-time Distillation on {len(records)} approved frames ---")
    
    # Load Vision System for text embeddings
    vision = VisionSystem(device=device, load_vae=False)
    
    # Load Base Student Model
    model = DirectCNNT2I(text_dim=512, img_size=128).to(device)
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict, strict=False)
        except: pass
            
    # Load T+1 Student Model
    t_plus_1_model = DirectContinuationCNN(text_dim=512, img_size=128).to(device)
    if os.path.exists(t_plus_1_path):
        try:
            t_plus_1_model.load_state_dict(torch.load(t_plus_1_path, map_location=device, weights_only=True))
        except: pass
            
    model.train()
    t_plus_1_model.train()
    
    optimizer_G = optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_T1 = optim.AdamW(t_plus_1_model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    
    # Prepare data
    images = []
    text_embeds = []
    
    with torch.no_grad():
        for record in records:
            img_path = distill_dir / record["file_name"]
            if not img_path.exists():
                continue
                
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((128, 128), Image.BILINEAR)
                img_tensor = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float().div_(255.0)
                images.append(img_tensor)
                
                inputs = vision.clip_processor(text=[record["prompt"]], return_tensors="pt", padding=True, truncation=True).to(device)
                embed = vision.clip_model.get_text_features(**inputs)
                embed = embed / embed.norm(p=2, dim=-1, keepdim=True)
                text_embeds.append(embed.squeeze(0))
            except Exception as e:
                pass
                
    if not images:
        return False
        
    images = torch.stack(images).to(device)
    text_embeds = torch.stack(text_embeds).to(device)
    
    # Fast training loop
    for epoch in range(epochs):
        optimizer_G.zero_grad()
        optimizer_T1.zero_grad()
        
        z = torch.randn(text_embeds.size(0), 128, device=device)
        
        # 1. Train Base Model (Text -> Image)
        fake_images = model(text_embeds, z=z)
        loss_base = F.l1_loss(fake_images, images)
        
        # 2. Train T+1 Model
        t_plus_1_manifest = distill_dir / "t_plus_1_manifest.jsonl"
        t1_loss_val = 0.0
        
        if t_plus_1_manifest.exists():
            # We have real Teacher interruptions to learn from!
            t1_records = []
            with open(t_plus_1_manifest, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip(): t1_records.append(json.loads(line))
                    
            if t1_records:
                prev_imgs, next_imgs, t1_embeds = [], [], []
                for rec in t1_records:
                    try:
                        p_img = Image.open(rec["prev"]).convert("RGB").resize((128, 128), Image.BILINEAR)
                        n_img = Image.open(rec["next"]).convert("RGB").resize((128, 128), Image.BILINEAR)
                        prev_imgs.append(torch.from_numpy(np.array(p_img, dtype=np.uint8)).permute(2, 0, 1).float().div_(255.0))
                        next_imgs.append(torch.from_numpy(np.array(n_img, dtype=np.uint8)).permute(2, 0, 1).float().div_(255.0))
                        
                        inputs = vision.clip_processor(text=[rec["prompt"]], return_tensors="pt", padding=True, truncation=True).to(device)
                        embed = vision.clip_model.get_text_features(**inputs)
                        embed = embed / embed.norm(p=2, dim=-1, keepdim=True)
                        t1_embeds.append(embed.squeeze(0))
                    except: pass
                
                if prev_imgs:
                    prev_imgs = torch.stack(prev_imgs).to(device)
                    next_imgs = torch.stack(next_imgs).to(device)
                    t1_embeds = torch.stack(t1_embeds).to(device)
                    
                    z_t1 = torch.randn(t1_embeds.size(0), 128, device=device)
                    fake_next = t_plus_1_model(prev_imgs, t1_embeds, z=z_t1)
                    loss_t1 = F.l1_loss(fake_next, next_imgs)
                    t1_loss_val = loss_t1.item()
                    loss_t1.backward()
                    optimizer_T1.step()
        
        if t1_loss_val == 0.0:
            # Fallback to synthetic affine transforms if no real interruptions exist yet
            angle = random.uniform(-2, 2)
            translate = [random.uniform(-0.02, 0.02) * 128, random.uniform(-0.02, 0.02) * 128]
            scale = random.uniform(0.98, 1.02)
            
            target_images = TF.affine(images, angle, translate, scale, 0)
            fake_next_frames = t_plus_1_model(images, text_embeds, z=z)
            loss_t1 = F.l1_loss(fake_next_frames, target_images)
            t1_loss_val = loss_t1.item()
            loss_t1.backward()
            optimizer_T1.step()
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Distill Epoch {epoch+1}/{epochs} | Base Loss: {loss_base.item():.4f} | T+1 Loss: {t1_loss_val:.4f}")
            
    torch.save(model.state_dict(), checkpoint_path)
    torch.save(t_plus_1_model.state_dict(), t_plus_1_path)
    print(f"--- Distillation complete. Updated Base and T+1 Student models ---")
    return True

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    distill_student_model(device=device)
