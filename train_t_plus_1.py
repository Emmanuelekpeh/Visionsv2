"""
Train the DirectContinuationCNN (T+1 Student) model.

This model learns: Given frame_t + prompt → predict frame_t+1

Training data is extracted from REAL VIDEOS:
1. Extract consecutive frames from video clips
2. Train T+1 to predict frame_t+1 given frame_t
"""
import os
import sys
import json
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

from alternative_t2i import DirectContinuationCNN, SimpleDiscriminator
from vision import VisionSystem
from pathlib import Path

# Video data source
VIDEO_DATA_PATH = Path.home() / "Projects" / "latent_flow" / "data"


class FramePairDataset(Dataset):
    """Dataset of consecutive frame pairs for T+1 training - supports mixed sources."""
    
    def __init__(self, data_dir: str, vision: VisionSystem, device="cpu"):
        self.data_dir = Path(data_dir)
        self.device = device
        self.pairs = []
        self.vision = vision
        
        # Load from manifest
        manifest_path = self.data_dir / "t_plus_1_pairs.jsonl"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        if "prev" in record and "next" in record:
                            self.pairs.append(record)
        
        # Count sources
        video_count = sum(1 for p in self.pairs if p.get("source") == "video")
        sd_count = sum(1 for p in self.pairs if p.get("source") == "sd_synthetic")
        other_count = len(self.pairs) - video_count - sd_count
        
        print(f"Loaded {len(self.pairs)} frame pairs from {manifest_path}")
        print(f"  Video frames: {video_count}")
        print(f"  SD synthetic: {sd_count}")
        if other_count > 0:
            print(f"  Other/legacy: {other_count}")
        
        # Pre-compute common text embeddings
        print("Pre-computing text embeddings...")
        self.embed_cache = {}
        
        # Generic video embedding
        with torch.no_grad():
            for prompt in ["continuous motion video frame", "video frame with motion"]:
                inputs = vision.clip_processor(
                    text=[prompt], 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(device)
                embed = vision.clip_model.get_text_features(**inputs)
                embed = embed / embed.norm(p=2, dim=-1, keepdim=True)
                self.embed_cache[prompt] = embed.squeeze(0).cpu()
        
        # Cache unique SD prompts (limited to avoid memory issues)
        unique_prompts = set(p.get("prompt", "") for p in self.pairs if p.get("source") == "sd_synthetic")
        for prompt in list(unique_prompts)[:100]:  # Limit to 100 unique prompts
            if prompt and prompt not in self.embed_cache:
                with torch.no_grad():
                    inputs = vision.clip_processor(
                        text=[prompt], 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True
                    ).to(device)
                    embed = vision.clip_model.get_text_features(**inputs)
                    embed = embed / embed.norm(p=2, dim=-1, keepdim=True)
                    self.embed_cache[prompt] = embed.squeeze(0).cpu()
        
        print(f"Cached {len(self.embed_cache)} text embeddings")
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        record = self.pairs[idx]
        
        # Load images
        prev_img = Image.open(record["prev"]).convert("RGB").resize((128, 128), Image.BILINEAR)
        next_img = Image.open(record["next"]).convert("RGB").resize((128, 128), Image.BILINEAR)
        
        # Convert to tensors [0, 1]
        prev_tensor = torch.from_numpy(np.array(prev_img)).permute(2, 0, 1).float() / 255.0
        next_tensor = torch.from_numpy(np.array(next_img)).permute(2, 0, 1).float() / 255.0
        
        # Get text embedding
        prompt = record.get("prompt", "continuous motion video frame")
        if prompt in self.embed_cache:
            text_embed = self.embed_cache[prompt].clone()
        else:
            # Fallback to generic
            text_embed = self.embed_cache["continuous motion video frame"].clone()
        
        return prev_tensor, next_tensor, text_embed, prompt


def extract_frames_from_videos(
    video_dir: str = VIDEO_DATA_PATH,
    output_dir: str = "dataset/t_plus_1_training",
    max_videos: int = 200,
    frames_per_video: int = 20,
    frame_skip: int = 2,  # Skip frames for more noticeable motion
    device: str = "cpu"
):
    """Extract consecutive frame pairs from real videos."""
    
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_files = list(video_dir.glob("*.mp4"))
    if not video_files:
        print(f"No MP4 files found in {video_dir}")
        return 0
    
    print(f"Found {len(video_files)} videos in {video_dir}")
    
    # Shuffle and limit
    random.shuffle(video_files)
    video_files = video_files[:max_videos]
    
    manifest_path = output_dir / "t_plus_1_pairs.jsonl"
    pair_count = 0
    
    with open(manifest_path, "a", encoding="utf-8") as f:  # Append mode
        for vid_idx, video_path in enumerate(video_files):
            print(f"\nProcessing video {vid_idx + 1}/{len(video_files)}: {video_path.name}")
            
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"  Could not open {video_path.name}")
                    continue
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < frames_per_video * frame_skip + 10:
                    print(f"  Video too short ({total_frames} frames)")
                    cap.release()
                    continue
                
                # Start from a random position
                start_frame = random.randint(0, max(0, total_frames - frames_per_video * frame_skip - 5))
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                prev_frame = None
                prev_path = None
                extracted = 0
                
                for i in range(frames_per_video + 1):
                    # Skip frames for more motion
                    for _ in range(frame_skip):
                        ret, frame = cap.read()
                        if not ret:
                            break
                    
                    if not ret or frame is None:
                        break
                    
                    # Convert BGR to RGB and resize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb).resize((128, 128), Image.BILINEAR)
                    
                    # Save frame
                    ts = int(time.time() * 1000) + i
                    frame_path = output_dir / f"vid{vid_idx:04d}_frame{i:03d}_{ts}.png"
                    pil_frame.save(frame_path)
                    
                    # Create pair
                    if prev_path is not None:
                        record = {
                            "prev": str(prev_path),
                            "next": str(frame_path),
                            "prompt": "continuous motion video frame",
                            "source": "video",
                            "video": video_path.name,
                            "frame_idx": i
                        }
                        f.write(json.dumps(record) + "\n")
                        pair_count += 1
                        extracted += 1
                    
                    prev_path = frame_path
                
                cap.release()
                print(f"  Extracted {extracted} pairs")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
    
    print(f"\n{'='*50}")
    print(f"Extracted {pair_count} frame pairs from {len(video_files)} videos")
    print(f"Saved to {manifest_path}")
    
    return pair_count


def generate_sd_sequences(
    num_sequences: int = 50,
    frames_per_sequence: int = 10,
    output_dir: str = "dataset/t_plus_1_training",
    device: str = "cpu",
    use_amused: bool = False
):
    """Generate synthetic frame pairs using SD or Amused img2img - teaches SD distribution."""
    
    from production_system import VisualGenerator
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    teacher_id = "amused-256" if use_amused else "tiny-sd"
    print(f"\nGenerating {num_sequences} sequences x {frames_per_sequence} frames using {teacher_id}...")
    
    # Load Teacher
    visuals = VisualGenerator(device=device, teacher_id=teacher_id)
    
    # Load prompts
    prompts = []
    activities_path = Path("activities.txt")
    if activities_path.exists():
        with open(activities_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("---"):
                    prompts.append(line)
    if not prompts:
        prompts = [
            "A person walking through a city street",
            "Ocean waves crashing on rocks",
            "A bird flying across the sky",
            "Fire burning in a fireplace",
            "Trees swaying in the wind"
        ]
    
    manifest_path = output_dir / "t_plus_1_pairs.jsonl"
    pair_count = 0
    
    # Append to existing manifest
    with open(manifest_path, "a", encoding="utf-8") as f:
        for seq_idx in range(num_sequences):
            prompt = random.choice(prompts)
            model_label = "Amused" if use_amused else "SD"
            print(f"\n{model_label} Sequence {seq_idx + 1}/{num_sequences}: {prompt[:50]}...")
            
            try:
                # Generate initial keyframe with SD
                keyframe = visuals.generate_keyframe(prompt, num_inference_steps=12)
                keyframe = keyframe.resize((128, 128), Image.BILINEAR)
                
                curr_frame = keyframe
                prev_path = None
                
                for frame_idx in range(frames_per_sequence):
                    # Save current frame
                    ts = int(time.time() * 1000)
                    curr_path = output_dir / f"sd_seq{seq_idx:03d}_frame{frame_idx:03d}_{ts}.png"
                    curr_frame.save(curr_path)
                    
                    # Save pair if we have previous
                    if prev_path is not None:
                        record = {
                            "prev": str(prev_path),
                            "next": str(curr_path),
                            "prompt": prompt,
                            "source": "amused_synthetic" if use_amused else "sd_synthetic",
                            "seq_idx": seq_idx,
                            "frame_idx": frame_idx
                        }
                        f.write(json.dumps(record) + "\n")
                        pair_count += 1
                    
                    # Generate next frame using img2img
                    if frame_idx < frames_per_sequence - 1:
                        if use_amused:
                            # Amused img2img has different parameters
                            next_frame = visuals.core_model.img2img_pipe(
                                prompt=prompt,
                                image=curr_frame,
                                strength=0.30,
                            ).images[0].resize((128, 128), Image.BILINEAR)
                        else:
                            # Standard SD img2img
                            next_frame = visuals.core_model.img2img_pipe(
                                prompt=prompt,
                                image=curr_frame,
                                strength=0.30,
                                num_inference_steps=6,
                                guidance_scale=5.0
                            ).images[0].resize((128, 128), Image.BILINEAR)
                        
                        prev_path = curr_path
                        curr_frame = next_frame
                        
            except Exception as e:
                print(f"  Error: {e}")
                continue
                
            print(f"  Generated {frames_per_sequence - 1} pairs")
    
    print(f"\n{'='*50}")
    model_label = "Amused" if use_amused else "SD"
    print(f"Generated {pair_count} {model_label} synthetic pairs")
    
    return pair_count


def train_t_plus_1(
    device: str = "cpu",
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 2e-4,
    checkpoint_path: str = "t_plus_1_student.pt"
):
    """Train the T+1 model on frame pairs."""
    
    print("=" * 60)
    print("Training DirectContinuationCNN (T+1 Student)")
    print("=" * 60)
    
    # Load Vision System
    vision = VisionSystem(device=device, load_vae=False)
    
    # Load dataset
    data_dir = Path("dataset/t_plus_1_training")
    if not (data_dir / "t_plus_1_pairs.jsonl").exists():
        print("No training data found. Extracting from videos...")
        extract_frames_from_videos(max_videos=100, frames_per_video=15, frame_skip=2, device=device)
    
    dataset = FramePairDataset(str(data_dir), vision, device)
    
    if len(dataset) < 10:
        print(f"Not enough training data ({len(dataset)} pairs). Generate more first.")
        return False
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Load model
    model = DirectContinuationCNN(text_dim=512, img_size=128).to(device)
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            print(f"Loaded checkpoint from {checkpoint_path}")
        except:
            print("Could not load checkpoint, starting fresh")
    
    # Load discriminator
    disc = SimpleDiscriminator(img_size=128, text_embed_dim=512).to(device)
    disc_path = "t_plus_1_discriminator.pt"
    if os.path.exists(disc_path):
        try:
            disc.load_state_dict(torch.load(disc_path, map_location=device, weights_only=True))
        except:
            pass
    
    # Optimizers
    optimizer_G = optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_D = optim.AdamW(disc.parameters(), lr=lr * 0.5, betas=(0.5, 0.999), weight_decay=1e-4)
    
    # Training loop
    model.train()
    disc.train()
    
    best_loss = float('inf')
    
    total_batches = len(dataloader)
    print(f"Starting training: {len(dataset)} samples, {total_batches} batches/epoch")
    sys.stdout.flush()
    
    for epoch in range(epochs):
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        
        for batch_idx, (prev_frames, next_frames, text_embeds, _) in enumerate(dataloader):
            prev_frames = prev_frames.to(device)
            next_frames = next_frames.to(device)
            text_embeds = text_embeds.to(device)
            
            batch_size_actual = prev_frames.size(0)
            
            # Progress indicator
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1} batch {batch_idx}/{total_batches}...", end="\r")
            
            # ========== Train Discriminator ==========
            optimizer_D.zero_grad()
            
            z = torch.randn(batch_size_actual, 128, device=device)
            fake_next = model(prev_frames, text_embeds, z=z)
            
            # Real
            d_real = disc(next_frames, text_embeds)
            loss_d_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real) * 0.9)
            
            # Fake
            d_fake = disc(fake_next.detach(), text_embeds)
            loss_d_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake) + 0.1)
            
            loss_D = (loss_d_real + loss_d_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            
            # ========== Train Generator ==========
            optimizer_G.zero_grad()
            
            z = torch.randn(batch_size_actual, 128, device=device)
            fake_next = model(prev_frames, text_embeds, z=z)
            
            # Adversarial loss
            d_fake = disc(fake_next, text_embeds)
            loss_adv = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
            
            # Reconstruction loss (L1)
            loss_l1 = F.l1_loss(fake_next, next_frames)
            
            # Perceptual consistency (CLIP similarity between fake and real)
            # This encourages the model to preserve semantic content
            
            # Total generator loss
            loss_G = loss_l1 * 10.0 + loss_adv * 1.0
            loss_G.backward()
            optimizer_G.step()
            
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
        
        avg_loss_G = epoch_loss_G / len(dataloader)
        avg_loss_D = epoch_loss_D / len(dataloader)
        
        # Print every epoch
        print(f"Epoch {epoch + 1}/{epochs} | G Loss: {avg_loss_G:.4f} | D Loss: {avg_loss_D:.4f}      ")
        sys.stdout.flush()
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(disc.state_dict(), disc_path)
            print(f"  Saved checkpoint")
            sys.stdout.flush()
            
        if avg_loss_G < best_loss:
            best_loss = avg_loss_G
            torch.save(model.state_dict(), checkpoint_path.replace(".pt", "_best.pt"))
    
    print(f"\nTraining complete. Best G Loss: {best_loss:.4f}")
    print(f"Saved to {checkpoint_path}")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true", help="Extract frames from videos only")
    parser.add_argument("--generate-sd", action="store_true", help="Generate SD synthetic sequences only")
    parser.add_argument("--use-amused", action="store_true", help="Use Amused (faster) instead of SD")
    parser.add_argument("--mixed", action="store_true", help="Generate mixed dataset (videos + SD)")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--videos", type=int, default=150, help="Number of videos to process")
    parser.add_argument("--sd-sequences", type=int, default=50, help="Number of SD sequences")
    parser.add_argument("--frames", type=int, default=15, help="Frames per video/sequence")
    parser.add_argument("--skip", type=int, default=2, help="Frame skip for videos")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    args = parser.parse_args()
    
    device = "cpu"
    
    if args.extract:
        extract_frames_from_videos(
            max_videos=args.videos,
            frames_per_video=args.frames,
            frame_skip=args.skip,
            device=device
        )
    elif args.generate_sd:
        generate_sd_sequences(
            num_sequences=args.sd_sequences,
            frames_per_sequence=args.frames,
            device=device,
            use_amused=args.use_amused
        )
    elif args.mixed:
        # Generate mixed dataset: real videos + SD synthetic
        print("=" * 60)
        print("GENERATING MIXED DATASET")
        print("=" * 60)
        print("\n[1/2] Extracting from real videos...")
        video_pairs = extract_frames_from_videos(
            max_videos=args.videos,
            frames_per_video=args.frames,
            frame_skip=args.skip,
            device=device
        )
        model_name = "Amused" if args.use_amused else "SD"
        print(f"\n[2/2] Generating {model_name} synthetic sequences...")
        sd_pairs = generate_sd_sequences(
            num_sequences=args.sd_sequences,
            frames_per_sequence=args.frames,
            device=device,
            use_amused=args.use_amused
        )
        print(f"\n{'='*60}")
        print(f"MIXED DATASET COMPLETE")
        print(f"  Video pairs: {video_pairs}")
        print(f"  SD pairs:    {sd_pairs}")
        print(f"  Total:       {video_pairs + sd_pairs}")
        print(f"{'='*60}")
    elif args.train:
        train_t_plus_1(
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        # Default: show help
        print("Usage:")
        print("  --extract      Extract frames from real videos")
        print("  --generate-sd  Generate SD synthetic sequences")
        print("  --mixed        Generate mixed dataset (recommended)")
        print("  --train        Train the T+1 model")
        print("\nExample workflow:")
        print("  python train_t_plus_1.py --mixed --videos 150 --sd-sequences 50")
        print("  python train_t_plus_1.py --train --epochs 100 --batch-size 8")
