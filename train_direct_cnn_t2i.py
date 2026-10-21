import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import torchvision.utils as vutils
import time
import argparse
import math

from alternative_t2i import DirectCNNT2I, SimpleDiscriminator
from train_tiny_t2i import ScrapedActivityDataset
from vision import VisionSystem


def _unwrap_module(module):
    """Return the underlying nn.Module when torch.compile wrapped it."""
    return getattr(module, "_orig_mod", module)


def _state_dict_for_save(module):
    return _unwrap_module(module).state_dict()


def save_samples(model, vision, test_prompts, epoch, device, output_dir="training_samples_direct", z=None):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    clip_device = next(vision.clip_model.parameters()).device
    if clip_device != torch.device(device):
        vision.clip_model.to(device)
    
    with torch.no_grad():
        inputs = vision.clip_processor(text=test_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        text_embeds = vision.clip_model.get_text_features(**inputs)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        if z is None:
            z = torch.randn(len(test_prompts), 128, device=device)
        
        samples = model(text_embeds, z=z, guidance_scale=3.0)
        samples = torch.clamp(samples, 0, 1)
        
        filepath = os.path.join(output_dir, f"samples_epoch_{epoch+1}.png")
        vutils.save_image(samples, filepath, nrow=min(len(test_prompts), 4))
        print(f"Saved generated samples to {filepath}")
    
    if torch.device(device).type == "cuda":
        vision.clip_model.to("cpu")
        torch.cuda.empty_cache()
    model.train()


def load_activity_prompts(activities_file="activities.txt", limit=None):
    """Load non-empty prompt lines from activities.txt, skipping section headers."""
    if not os.path.exists(activities_file):
        raise FileNotFoundError(f"Activities file not found: {activities_file}")

    prompts = []
    with open(activities_file, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip().strip('"').strip("'")
            if not text or text.startswith("---"):
                continue
            prompts.append(text)

    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def generate_direct_cnn_t2i(checkpoint, prompts, device="cpu", output="generated.png",
                            guidance_scale=3.0, seed=None, nrow=4, batch_size=16):
    """Load a DirectCNNT2I checkpoint and generate images from text prompts."""
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not prompts:
        raise ValueError("No prompts provided for generation.")

    model = DirectCNNT2I(text_dim=512, img_size=128).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {checkpoint} ({sum(p.numel() for p in model.parameters()):,} params)")

    vision = VisionSystem(device=device, load_vae=False)
    if seed is not None:
        torch.manual_seed(seed)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    base, ext = os.path.splitext(output)
    if not ext:
        ext = ".png"
        output = output + ext

    saved_paths = []
    total = len(prompts)
    for start in range(0, total, batch_size):
        batch_prompts = prompts[start:start + batch_size]
        with torch.no_grad():
            inputs = vision.clip_processor(
                text=batch_prompts, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            text_embeds = vision.clip_model.get_text_features(**inputs)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            z = torch.randn(len(batch_prompts), 128, device=device)
            samples = model(text_embeds, z=z, guidance_scale=guidance_scale)
            samples = torch.clamp(samples, 0, 1)

        if total <= batch_size:
            out_path = output
        else:
            batch_num = start // batch_size + 1
            out_path = f"{base}_batch{batch_num:02d}{ext}"

        vutils.save_image(samples, out_path, nrow=min(nrow, len(batch_prompts)))
        saved_paths.append(out_path)
        print(f"Saved {len(batch_prompts)} image(s) ({start + 1}-{start + len(batch_prompts)}/{total}) to {out_path}")

    return saved_paths


def _clamp_batch_size(batch_size: int, gpu_mem_gb: float, compile_models: bool) -> int:
    """Keep GAN training within typical T4/P100 VRAM."""
    if gpu_mem_gb <= 0:
        return batch_size
    if compile_models:
        cap = 32 if gpu_mem_gb <= 12 else 128 if gpu_mem_gb <= 16 else 256
    else:
        cap = 128 if gpu_mem_gb <= 12 else 256 if gpu_mem_gb <= 16 else 512
    if batch_size > cap:
        print(f"Clamping batch-size {batch_size} -> {cap} for {gpu_mem_gb:.1f}GB VRAM (compile={compile_models}).")
        return cap
    return batch_size


def train_direct_cnn_t2i(device="cpu", epochs=90, batch_size=16, lr=2e-4, use_activities=True,
                         use_dataset=True, activities_file="activities.txt", tag="", use_amp=True,
                         max_hours=None, preload_images=None, compile_models=False, grad_accum_steps=1,
                         save_every=10):
    print(f"--- Starting Training for DirectCNNT2I on {device} ---")
    # Covers setup as well as training, keeping the run inside a limited Colab session.
    deadline = time.monotonic() + max_hours * 3600 if max_hours else None
    
    is_cuda = torch.device(device).type == "cuda"
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if is_cuda else 0
    if preload_images is None:
        preload_images = is_cuda
    if is_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB VRAM)")
        print(f"CUDA fast path: cuDNN benchmark, TF32, mixed precision {'enabled' if use_amp else 'disabled'}.")
        batch_size = _clamp_batch_size(batch_size, gpu_mem_gb, compile_models)
        effective_batch = batch_size * grad_accum_steps
        if grad_accum_steps > 1:
            print(f"Gradient accumulation: {grad_accum_steps} steps (effective batch {effective_batch}).")
    else:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        use_amp = False
        preload_images = False
        print(f"Using {torch.get_num_threads()} CPU compute threads.")
    
    # Setup Vision System
    # DirectCNNT2I only uses CLIP text features; do not load the unused Tiny VAE.
    vision = VisionSystem(device=device, load_vae=False)
    
    # Load Dataset — preload uses compact uint8 (~0.5GB for 10k images at 128px)
    dataset = ScrapedActivityDataset(
        img_size=128,
        samples_per_activity=25,
        activities_file=activities_file,
        use_activities=use_activities,
        use_dataset=use_dataset,
        return_256=False,
        preload=preload_images,
    )
    
    if len(dataset) == 0:
        print("No data available for training. Exiting.")
        return
    
    # Dataloader
    virtual_epoch_size = 5000
    if len(dataset) > virtual_epoch_size:
        print(f"Using Virtual Epochs: Randomly sampling {virtual_epoch_size} images per epoch out of {len(dataset)} total.")
        sampler = RandomSampler(dataset, replacement=False, num_samples=virtual_epoch_size)
    else:
        sampler = RandomSampler(dataset, replacement=False)
    
    num_workers = min(4, max(2, (os.cpu_count() or 4) - 1)) if is_cuda else min(3, max(0, (os.cpu_count() or 4) // 2 - 1))
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                           pin_memory=is_cuda, persistent_workers=num_workers > 0,
                           prefetch_factor=8 if num_workers > 0 else None)
    print(f"DataLoader: {num_workers} workers, batch={batch_size}, preload={preload_images}, pinned memory={is_cuda}.")
    
    # Setup Model
    model = DirectCNNT2I(text_dim=512, img_size=128).to(device)
    discriminator = SimpleDiscriminator(img_size=128, text_embed_dim=512).to(device)
    if is_cuda:
        model = model.to(memory_format=torch.channels_last)
        discriminator = discriminator.to(memory_format=torch.channels_last)

    print(f"DirectCNNT2I Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Discriminator Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Load checkpoint before torch.compile — compiled modules use _orig_mod.* keys
    save_path = "direct_cnn_t2i.pt" if not tag else f"direct_cnn_t2i_{tag}.pt"
    disc_save_path = "direct_discriminator.pt" if not tag else f"direct_discriminator_{tag}.pt"

    load_model_path = save_path
    if not os.path.exists(load_model_path) and os.path.exists("direct_cnn_t2i.pt"):
        load_model_path = "direct_cnn_t2i.pt"
    if os.path.exists(load_model_path):
        try:
            model.load_state_dict(torch.load(load_model_path, map_location=device, weights_only=True))
            print(f"Resumed DirectCNNT2I from {load_model_path}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Training from scratch.")

    load_disc_path = disc_save_path
    if not os.path.exists(load_disc_path) and os.path.exists("direct_discriminator.pt"):
        load_disc_path = "direct_discriminator.pt"
    if os.path.exists(load_disc_path):
        try:
            discriminator.load_state_dict(torch.load(load_disc_path, map_location=device, weights_only=True))
            print(f"Resumed Discriminator from {load_disc_path}")
        except Exception as e:
            print(f"Could not load discriminator: {e}. Training from scratch.")

    if compile_models and hasattr(torch, "compile"):
        print("Compiling models with torch.compile...")
        model = torch.compile(model)
        discriminator = torch.compile(discriminator)
    
    # Optimizers
    optimizer_G = optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda and use_amp)
    
    # Load text embedding cache
    cache_file = "text_embed_cache.pt"
    text_embed_cache = {}
    
    if os.path.exists(cache_file):
        print(f"Loading cached text embeddings from {cache_file}...")
        text_embed_cache = torch.load(cache_file, map_location="cpu", weights_only=True)
    
    unique_texts = list(set([s.get('text', '') for s in dataset.samples]))
    missing_texts = [t for t in unique_texts if t not in text_embed_cache]
    
    if missing_texts:
        print(f"Computing CLIP text embeddings for {len(missing_texts)} missing prompts...")
        with torch.no_grad():
            for i in range(0, len(missing_texts), 32):
                batch_texts = missing_texts[i:i+32]
                inputs = vision.clip_processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                embeds = vision.clip_model.get_text_features(**inputs)
                embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                for t, e in zip(batch_texts, embeds):
                    text_embed_cache[t] = e.detach().cpu()
        torch.save(text_embed_cache, cache_file)

    # Move CLIP off GPU — only needed again for sample generation
    if is_cuda:
        vision.clip_model.to("cpu")
        torch.cuda.empty_cache()
        print("Moved CLIP to CPU to free GPU VRAM for training.")

    unique_texts = sorted(text_embed_cache.keys())
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}
    dataset.text_to_idx = text_to_idx
    embed_matrix = torch.stack([text_embed_cache[t] for t in unique_texts])
    if is_cuda:
        embed_matrix = embed_matrix.pin_memory()
    print(f"Text embeddings: {embed_matrix.shape[0]} prompts (CPU pinned, copied per batch).")
    
    model.train()
    discriminator.train()
    
    # Training Loop
    timed_out = False
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss_G = 0
        total_loss_D = 0
        total_loss_pixel = 0
        batches_done = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                print("Time limit reached; saving a resumable checkpoint after this partial epoch.")
                break
            batch_start_time = time.time()
            images, text_indices = batch
            images = images.to(device, non_blocking=is_cuda)
            if is_cuda:
                images = images.to(memory_format=torch.channels_last)
            
            text_embeds = embed_matrix[text_indices].to(device, non_blocking=is_cuda)
            
            # Classifier-Free Guidance training
            keep = (torch.rand(text_embeds.size(0), 1, device=device) >= 0.1).float()
            text_embeds_train = text_embeds * keep

            if batch_idx % grad_accum_steps == 0:
                optimizer_G.zero_grad(set_to_none=True)
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
            
            # Train Generator
            with torch.autocast(device_type="cuda" if is_cuda else "cpu", dtype=torch.float16 if is_cuda else torch.bfloat16, enabled=use_amp):
                z = torch.randn(text_embeds_train.size(0), 128, device=device)
                fake_images = model(text_embeds_train, z=z)
                pred_fake = discriminator(fake_images, text_embeds_train)
                loss_G_GAN = -pred_fake.mean()
                loss_G_pixel = F.l1_loss(fake_images, images)
                loss_G = (loss_G_GAN + 0.5 * loss_G_pixel) / grad_accum_steps
            scaler.scale(loss_G).backward()
            
            total_loss_G += loss_G.item() * grad_accum_steps
            total_loss_pixel += loss_G_pixel.item()
            
            # Train Discriminator (after full accumulated generator step)
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer_G)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer_G)
                scaler.update()

                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
                optimizer_D.zero_grad(set_to_none=True)
                
                noise_std = 0.05 * (1.0 - (epoch / epochs))
                noisy_real = images + torch.randn_like(images) * noise_std
                
                with torch.autocast(device_type="cuda" if is_cuda else "cpu", dtype=torch.float16 if is_cuda else torch.bfloat16, enabled=use_amp):
                    pred_real = discriminator(noisy_real, text_embeds_train)
                    loss_D_real = F.relu(1.0 - pred_real).mean()
                    noisy_fake = fake_images.detach() + torch.randn_like(fake_images) * noise_std
                    pred_fake = discriminator(noisy_fake, text_embeds_train)
                    loss_D_fake = F.relu(1.0 + pred_fake).mean()
                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                scaler.scale(loss_D).backward()
                
                scaler.unscale_(optimizer_D)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                scaler.step(optimizer_D)
                scaler.update()
                
                total_loss_D += loss_D.item()
            batches_done += 1
            
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            
            if batch_idx % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx} | "
                      f"D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f} | "
                      f"{batch_duration:.2f}s/batch")
        
        avg_loss_G = total_loss_G / max(1, batches_done)
        avg_loss_D = total_loss_D / max(1, batches_done)
        avg_loss_pixel = total_loss_pixel / max(1, batches_done)
        epoch_duration = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1} completed in {epoch_duration/60:.2f} minutes "
              f"({epoch_duration/max(1, batches_done):.2f}s/batch). "
              f"Avg D Loss: {avg_loss_D:.4f} | Avg G Loss: {avg_loss_G:.4f} | "
              f"Avg Pixel L1: {avg_loss_pixel:.4f}")
        
        if math.isnan(avg_loss_G) or math.isnan(avg_loss_D) or math.isnan(avg_loss_pixel):
            print("❌ NaN detected in epoch loss! Aborting before saving corrupted checkpoint.")
            break
            
        # Save samples
        test_prompts = ["a person running", "a cat sleeping", "a beautiful sunset", "a car driving", "playing guitar"]
        fixed_z = torch.randn(len(test_prompts), 128, device=device)
        
        output_dir = f"training_samples_direct_{tag}" if tag else "training_samples_direct"
        save_samples(model, vision, test_prompts, epoch, device, output_dir=output_dir, z=fixed_z)
        
        # Save checkpoint (always unwrapped keys for compatibility with non-compiled inference)
        torch.save(_state_dict_for_save(model), save_path)
        torch.save(_state_dict_for_save(discriminator), disc_save_path)
        print(f"Saved checkpoints to {save_path} and {disc_save_path} after Epoch {epoch+1}")
        
        if (epoch + 1) % save_every == 0:
            history_path = save_path.replace('.pt', f'_ep{epoch+1}.pt')
            disc_history_path = disc_save_path.replace('.pt', f'_ep{epoch+1}.pt')
            torch.save(_state_dict_for_save(model), history_path)
            torch.save(_state_dict_for_save(discriminator), disc_history_path)
            print(f"Archived milestone checkpoint to {history_path}")
            
        if timed_out:
            break
    
    print(f"Training complete. Final models saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DirectCNNT2I model.")
    parser.add_argument("--activities", action="store_true", help="Scrape and train on activities from activities.txt")
    parser.add_argument("--activities-file", type=str, default="activities.txt", help="Path to activities file")
    parser.add_argument("--dataset", action="store_true", help="Train on locally saved datasets")
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (lower it if CUDA runs out of memory)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (lower is more stable for GANs)")
    parser.add_argument("--tag", type=str, default="", help="Tag for checkpoint files")
    parser.add_argument("--save-every", type=int, default=10, help="Save an archived milestone checkpoint every N epochs")
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=True,
                        help="Disable CUDA mixed precision for debugging")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Stop cleanly after this many hours and save a checkpoint")
    parser.add_argument("--no-preload", dest="preload", action="store_false", default=True,
                        help="Disable RAM preloading of images (slower on GPU)")
    parser.add_argument("--compile", dest="compile_models", action="store_true", default=False,
                        help="Use torch.compile for faster GPU kernels (PyTorch 2+)")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Accumulate gradients over N batches to simulate larger batch with less VRAM")
    parser.add_argument("--generate", action="store_true",
                        help="Generate images from a checkpoint instead of training")
    parser.add_argument("--checkpoint", type=str, default="direct_cnn_t2i.pt",
                        help="Checkpoint path for --generate")
    parser.add_argument("--prompts", type=str, nargs="+",
                        default=["a person running", "a cat sleeping", "a beautiful sunset"],
                        help="Text prompts for --generate")
    parser.add_argument("--output", type=str, default="generated.png",
                        help="Output image path for --generate")
    parser.add_argument("--guidance-scale", type=float, default=3.0,
                        help="Classifier-free guidance scale for --generate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for --generate (optional)")
    parser.add_argument("--from-activities", action="store_true",
                        help="Use prompts from --activities-file instead of --prompts")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of activity prompts to generate (with --from-activities)")
    parser.add_argument("--gen-batch-size", type=int, default=16,
                        help="Prompts per output grid when generating many activities")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.generate:
        if args.from_activities:
            prompts = load_activity_prompts(args.activities_file, limit=args.limit)
            print(f"Loaded {len(prompts)} prompts from {args.activities_file}")
        else:
            prompts = args.prompts

        generate_direct_cnn_t2i(
            checkpoint=args.checkpoint,
            prompts=prompts,
            device=device,
            output=args.output,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            batch_size=args.gen_batch_size,
        )
        raise SystemExit(0)

    # Default to both if neither specified
    use_act = args.activities
    use_dat = args.dataset
    
    if not args.activities and not args.dataset:
        print("No data flags provided, defaulting to using both.")
        use_act = True
        use_dat = True
    
    train_direct_cnn_t2i(
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_activities=use_act,
        use_dataset=use_dat,
        activities_file=args.activities_file,
        tag=args.tag,
        use_amp=args.amp,
        max_hours=args.max_hours,
        preload_images=args.preload,
        compile_models=args.compile_models,
        grad_accum_steps=args.grad_accum,
        save_every=args.save_every,
    )
