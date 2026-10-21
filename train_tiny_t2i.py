import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torchvision.utils as vutils
import torchvision.models as models
from PIL import Image
import numpy as np
import random
import argparse
import time
import copy

from tiny_t2i import TinyT2I, CascadedUpsampler, TinyDiscriminator, load_t2i_state_dict, load_upsampler_state_dict
from vision import VisionSystem
from pinterest_scraper import scrape_images_pil as scrape_images
from kaggle_dataset import IMAGE_EXT, file_content_hash, is_valid_image


def estimate_preload_ram_gb(num_samples, img_size=128, return_256=True, uint8=True):
    """Rough RAM needed to cache decoded images (uint8 uses 1 byte/channel/pixel)."""
    bytes_per_pixel = 1 if uint8 else 4
    per_image = 3 * img_size * img_size * bytes_per_pixel
    if return_256:
        per_image += 3 * 256 * 256 * bytes_per_pixel
    return num_samples * per_image / (1024 ** 3)


def resolve_preload(preload, num_samples, img_size=128, return_256=True, ram_budget_gb=4.0):
    """Auto-disable preload when the dataset would exceed a safe RAM budget."""
    if not preload or num_samples == 0:
        return False
    est_gb = estimate_preload_ram_gb(num_samples, img_size, return_256, uint8=True)
    if est_gb > ram_budget_gb:
        print(f"Preload would use ~{est_gb:.1f}GB RAM (budget {ram_budget_gb:.1f}GB). Streaming from disk instead.")
        print("Pass --preload to force preload anyway (uses compact uint8 storage).")
        return False
    return True


def _unwrap_module(module):
    """Return the underlying nn.Module when torch.compile wrapped it."""
    return getattr(module, "_orig_mod", module)


def _state_dict_for_save(module):
    return _unwrap_module(module).state_dict()


class EMA:
    """Exponential Moving Average of model weights for stable generation."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        # Cache state dict references to avoid creating new dicts every step
        self.model_states = {name: tensor for name, tensor in model.state_dict().items() if tensor.dtype.is_floating_point}
        self.shadow = {name: tensor.clone().detach() for name, tensor in self.model_states.items()}
        self.backup = {}
        
    def update(self):
        with torch.no_grad():
            for name, tensor in self.model_states.items():
                self.shadow[name].lerp_(tensor, 1.0 - self.decay)
                    
    def apply_shadow(self):
        self.backup = {name: tensor.clone().detach() for name, tensor in self.model_states.items()}
        with torch.no_grad():
            for name, tensor in self.model_states.items():
                tensor.copy_(self.shadow[name])
        
    def restore(self):
        with torch.no_grad():
            for name, tensor in self.model_states.items():
                tensor.copy_(self.backup[name])
        self.backup = {}

class EdgeLoss(nn.Module):
    """Computes spatial gradients (edges) using Sobel filters to penalize structural mismatch."""
    def __init__(self, device):
        super().__init__()
        # Sobel kernels
        k_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        k_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer('weight_x', k_x.to(device))
        self.register_buffer('weight_y', k_y.to(device))

    def forward(self, pred, target, mask=None):
        # Calculate spatial gradients (edges) for both predictions and targets
        pred_x = F.conv2d(pred, self.weight_x, padding=1, groups=3)
        pred_y = F.conv2d(pred, self.weight_y, padding=1, groups=3)
        
        with torch.no_grad():
            target_x = F.conv2d(target, self.weight_x, padding=1, groups=3)
            target_y = F.conv2d(target, self.weight_y, padding=1, groups=3)
            
        loss_x = F.l1_loss(pred_x, target_x, reduction='none')
        loss_y = F.l1_loss(pred_y, target_y, reduction='none')
        
        loss = loss_x + loss_y
        
        if mask is not None:
            # mask is [B, 1, H, W], loss is [B, 3, H-2, W-2]
            # Interpolate mask to match the slightly smaller edge map dimensions
            mask_resized = F.interpolate(mask, size=loss.shape[2:], mode='bilinear', align_corners=False)
            # Weight the foreground more heavily (e.g. 3x weight for foreground)
            weight = 1.0 + 2.0 * mask_resized
            loss = loss * weight
            
        return loss.mean()

class VGGLoss(nn.Module):
    """Perceptual loss using VGG16 features."""
    def __init__(self, device, chunk_size=8):
        super().__init__()
        self.chunk_size = chunk_size
        
        # Define local path for VGG weights
        local_vgg_path = "vgg16_weights.pth"
        
        # Load VGG16 architecture
        vgg = models.vgg16()
        
        # Check if local weights exist, download if they don't
        if os.path.exists(local_vgg_path):
            print(f"Loading VGG16 weights from local file: {local_vgg_path}")
            state_dict = torch.load(local_vgg_path, map_location="cpu")
            vgg.load_state_dict(state_dict)
        else:
            print("Downloading VGG16 weights for the first time...")
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            print(f"Saving VGG16 weights to {local_vgg_path} for offline use...")
            torch.save(vgg.state_dict(), local_vgg_path)
            
        vgg = vgg.features
        # Extract features at relu1_2, relu2_2, relu3_3, relu4_3
        self.blocks = nn.ModuleList([
            vgg[:4],
            vgg[4:9],
            vgg[9:16],
            vgg[16:23]
        ]).to(device).eval()
        for param in self.parameters():
            param.requires_grad = False
            
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, x, y):
        if x.size(0) <= self.chunk_size:
            return self._forward_batch(x, y)
        total = 0.0
        n = x.size(0)
        for i in range(0, n, self.chunk_size):
            end = min(i + self.chunk_size, n)
            total = total + self._forward_batch(x[i:end], y[i:end]) * (end - i)
        return total / n

    def _forward_batch(self, x, y):
        x = (x - self.mean) / self.std
        with torch.no_grad():
            y = (y - self.mean) / self.std
            
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            with torch.no_grad():
                y = block(y)
            loss += F.l1_loss(x, y)
        return loss

class ScrapedActivityDataset(Dataset):
    def __init__(self, img_size=128, samples_per_activity=90, activities_file="activities.txt", use_activities=True, use_dataset=True,
                 return_256=True, preload=False, text_to_idx=None):
        self.samples = []
        self.img_size = img_size
        self.return_256 = return_256
        self.text_to_idx = text_to_idx
        self.preloaded = None
        loaded_paths = set()
        content_hashes = set()
        
        # 1. First, recursively load EVERYTHING from the local dataset folder to get the FULL dataset
        base_dataset_dir = "dataset"
        
        if use_dataset and os.path.exists(base_dataset_dir):
            print(f"Loading ALL available images recursively from {base_dataset_dir}...")
            skipped_corrupt = 0
            skipped_dupe = 0
            for root, dirs, files in os.walk(base_dataset_dir):
                prompt_text = os.path.basename(root).replace("_", " ")
                count = 0
                for filename in files:
                    if os.path.splitext(filename)[1].lower() not in IMAGE_EXT:
                        continue
                    filepath = os.path.join(root, filename)
                    if filepath in loaded_paths:
                        continue
                    if not is_valid_image(filepath):
                        skipped_corrupt += 1
                        continue
                    try:
                        digest = file_content_hash(filepath)
                    except OSError:
                        skipped_corrupt += 1
                        continue
                    if digest in content_hashes:
                        skipped_dupe += 1
                        continue
                    content_hashes.add(digest)
                    if root == base_dataset_dir:
                        p_text = os.path.splitext(filename)[0].replace("_", " ")
                    else:
                        p_text = prompt_text
                    self.samples.append({'path': filepath, 'text': p_text})
                    loaded_paths.add(filepath)
                    count += 1
                if count > 0:
                    try:
                        print(f"Loaded {count} images from {root}")
                    except UnicodeEncodeError:
                        print(f"Loaded {count} images from [Unicode path]")
            if skipped_corrupt or skipped_dupe:
                print(f"Skipped {skipped_corrupt} corrupt, {skipped_dupe} duplicate images.")
        
        # 2. Then, run the scraper for anything in activities.txt that might be missing
        if use_activities:
            activities = []
            if os.path.exists(activities_file):
                with open(activities_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        text = line.strip().strip('"').strip("'")
                        if text and not text.startswith('---'):
                            activities.append(text)
            else:
                print(f"Warning: {activities_file} not found. Using fallback.")
                activities = ["music video performance"]
                
            print(f"--- Checking for missing data from {len(activities)} queries in {activities_file} ---")
            for activity in activities:
                # scrape_images will automatically skip if folder exists and has enough images
                image_paths = scrape_images(activity, max_results=samples_per_activity, add_aesthetic=False, return_paths=True)
                
                added_count = 0
                for path in image_paths:
                    # Only add if we didn't already load it in step 1!
                    if path not in loaded_paths:
                        self.samples.append({'path': path, 'text': activity})
                        loaded_paths.add(path)
                        added_count += 1
                        
                if added_count > 0:
                    print(f"Added {added_count} NEW scraped images for '{activity}'")
                    
        print(f"Successfully loaded {len(self.samples)} total samples for pretraining.")

        if preload and len(self.samples) > 0:
            est_gb = estimate_preload_ram_gb(len(self.samples), self.img_size, self.return_256, uint8=True)
            print(f"Preloading {len(self.samples)} images into RAM as uint8 (~{est_gb:.1f}GB)...")
            self.preloaded = []
            for i, sample in enumerate(self.samples):
                img_128, img_256 = self._decode_sample(sample, store_uint8=True)
                self.preloaded.append((img_128, img_256))
                if (i + 1) % 500 == 0 or i + 1 == len(self.samples):
                    print(f"  Preloaded {i + 1}/{len(self.samples)}")

    def _decode_sample(self, sample, store_uint8=False):
        path_or_img = sample.get('path')
        if path_or_img:
            try:
                img = Image.open(path_or_img)
                draft_size = 256 if self.return_256 else self.img_size
                img.draft('RGB', (draft_size, draft_size))
                img = img.convert('RGB')
            except Exception as e:
                raise RuntimeError(f"Failed to decode image {path_or_img}: {e}") from e
        else:
            img = sample.get('image').convert('RGB')

        img_128 = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_tensor_128 = torch.from_numpy(np.array(img_128, dtype=np.uint8)).permute(2, 0, 1)

        if self.return_256:
            img = img.resize((256, 256), Image.BILINEAR)
            img_tensor_256 = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)
        else:
            img_tensor_256 = None

        if not store_uint8:
            img_tensor_128 = img_tensor_128.float().div_(255.0)
            if img_tensor_256 is not None:
                img_tensor_256 = img_tensor_256.float().div_(255.0)

        return img_tensor_128, img_tensor_256

    @staticmethod
    def _uint8_to_float(tensor):
        return tensor.float().div_(255.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.preloaded is not None:
            img_tensor_128, img_tensor_256 = self.preloaded[idx]
            img_tensor_128 = self._uint8_to_float(img_tensor_128)
            if img_tensor_256 is not None:
                img_tensor_256 = self._uint8_to_float(img_tensor_256)
        else:
            img_tensor_128, img_tensor_256 = self._decode_sample(sample)

        text_out = self.text_to_idx[sample['text']] if self.text_to_idx is not None else sample['text']

        if self.return_256:
            return img_tensor_128, img_tensor_256, text_out
        return img_tensor_128, text_out

def save_samples(model, upsampler, vision, test_prompts, epoch, device, output_dir="training_samples", z=None):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    upsampler.eval()
    clip_device = next(vision.clip_model.parameters()).device
    if clip_device != torch.device(device):
        vision.clip_model.to(device)

    with torch.no_grad():
        inputs = vision.clip_processor(text=test_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        text_embeds = vision.clip_model.get_text_features(**inputs)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        if z is None:
            z = torch.randn(len(test_prompts), 128, device=device)
            
        # Use CFG during generation
        preds_128 = model(text_embeds, z=z, guidance_scale=3.0)
        preds_128 = torch.clamp(preds_128, 0, 1)
        
        preds_256 = upsampler(preds_128, text_embeds, z=z)
        preds_256 = torch.clamp(preds_256, 0, 1)
        
        filepath = os.path.join(output_dir, f"samples_epoch_{epoch+1}.png")
        vutils.save_image(preds_256, filepath, nrow=min(len(test_prompts), 4))
        print(f"Saved generated samples to {filepath}")

    if torch.device(device).type == "cuda":
        vision.clip_model.to("cpu")
        torch.cuda.empty_cache()
    model.train()
    upsampler.train()

def pretrain_tiny_t2i(device="cpu", epochs=20, batch_size=8, lr=1e-3, use_activities=True, use_dataset=True,
                      activities_file="activities.txt", attn_downsample=2, max_batches=None, tag="", seed=None, vgg_freq=1, d_freq=2, use_amp=True,
                      max_hours=None, preload_images=None, compile_models=False, grad_accum_steps=1, vgg_chunk_size=8):
    print(f"--- Starting Pretraining for TinyT2I on {device} ---")
    # Includes model/data setup so this remains safe within a time-limited Colab session.
    deadline = time.monotonic() + max_hours * 3600 if max_hours else None

    if seed is not None:
        # Fixes weight init, condition dropout, instance noise and the sampler's
        # data order, so two runs differ only by the setting under test.
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f"Deterministic run with seed {seed}.")

    is_cuda = torch.device(device).type == "cuda"
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if is_cuda else 0
    if preload_images is None:
        preload_images = is_cuda
    if is_cuda and gpu_mem_gb <= 12 and vgg_freq == 1 and batch_size > 16:
        vgg_freq = 2
        print(f"12GB VRAM detected: defaulting vgg-freq to {vgg_freq} to reduce memory spikes.")
    # These settings only affect CUDA.  They let cuDNN choose the best kernels for
    # our fixed image sizes and allow fast Tensor Core matmuls on Ampere+ Colab GPUs.
    if is_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_amp = bool(use_amp)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB VRAM)")
        print("CUDA fast path: cuDNN benchmark, TF32, and mixed precision "
              f"{'enabled' if use_amp else 'disabled'}.")
        effective_batch = batch_size * grad_accum_steps
        if gpu_mem_gb <= 12:
            cap = 16 if compile_models else 24
            if batch_size > cap:
                print(f"WARNING: batch-size {batch_size} may OOM on {gpu_mem_gb:.0f}GB VRAM. Try <= {cap}, --vgg-freq 2, --vgg-chunk-size 4.")
        elif batch_size <= 32:
            suggested = min(64, max(16, batch_size * 2))
            print(f"NOTE: TinyT2I + VGG is VRAM-heavy. Try --batch-size {suggested} if you have headroom.")
        if grad_accum_steps > 1:
            print(f"Gradient accumulation: {grad_accum_steps} steps (effective batch {effective_batch}).")
        print(f"VGG loss every {vgg_freq} batch(es), chunked to {vgg_chunk_size} images.")
    else:
        # Hyperthreads hurt CPU convolution/attention kernels through oversubscription.
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        use_amp = False
        preload_images = False
        print(f"Using {torch.get_num_threads()} CPU compute threads.")
    
    # 1. Setup Vision System for Text Embeddings
    vision = VisionSystem(device=device)
    
    # 2. Gather Data via Scraper (Human Activities) and/or Local Dataset
    dataset = ScrapedActivityDataset(
        img_size=128, 
        samples_per_activity=25, 
        activities_file=activities_file,
        use_activities=use_activities,
        use_dataset=use_dataset,
        return_256=True,
        preload=preload_images,
    )
                    
    if len(dataset) == 0:
        print("No data available for training. Exiting.")
        return
        
    # Batch 16 gives good CPU vectorization without trashing the cache.
    # "Virtual Epochs": RandomSampler draws a fresh random subset of the corpus every
    # epoch, so one persistent DataLoader (and its worker pool) covers all epochs.
    virtual_epoch_size = 5000
    if len(dataset) > virtual_epoch_size:
        print(f"Using Virtual Epochs: Randomly sampling {virtual_epoch_size} images per epoch out of {len(dataset)} total.")
        sampler = RandomSampler(dataset, replacement=False, num_samples=virtual_epoch_size)
    else:
        sampler = RandomSampler(dataset, replacement=False)

    # Decoding runs in worker processes so it overlaps with the backward pass instead
    # of stalling it. Kept below the compute thread count to avoid starving the cores.
    num_workers = min(4, max(2, (os.cpu_count() or 4) - 1)) if is_cuda else min(3, max(0, (os.cpu_count() or 4) // 2 - 1))
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                            pin_memory=is_cuda, persistent_workers=num_workers > 0,
                            prefetch_factor=8 if num_workers > 0 else None)
    print(f"DataLoader: {num_workers} workers, batch={batch_size}, preload={preload_images}, pinned memory={is_cuda}.")

    # 3. Setup Model, Upsampler, and Discriminator
    model = TinyT2I(text_embed_dim=512, img_size=128, attn_downsample=attn_downsample).to(device)
    upsampler = CascadedUpsampler(in_size=128, out_size=256).to(device)
    discriminator = TinyDiscriminator(img_size=256, text_embed_dim=512).to(device)
    if is_cuda:
        model = model.to(memory_format=torch.channels_last)
        upsampler = upsampler.to(memory_format=torch.channels_last)
        discriminator = discriminator.to(memory_format=torch.channels_last)
    
    # Setup Perceptual Loss
    print("Initializing VGG Perceptual Loss...")
    vgg_loss = VGGLoss(device, chunk_size=vgg_chunk_size)
    edge_loss = EdgeLoss(device)
    
    # LOAD EXISTING CHECKPOINT
    load_paths = ("tiny_t2i_base.pt", "cascaded_upsampler.pt", "tiny_discriminator.pt")
    # A tagged run writes to its own files so an experiment never overwrites the real model.
    save_path, up_save_path, d_save_path = (
        [p.replace(".pt", f"_{tag}.pt") for p in load_paths] if tag else list(load_paths)
    )

    # Loaded independently: an architecture change to one module must not be
    # reported as, or silently discard, a successful load of the others.
    for label, module, path, loader in (
        ("TinyT2I", model, load_paths[0], load_t2i_state_dict),
        ("CascadedUpsampler", upsampler, load_paths[1], load_upsampler_state_dict),
        ("TinyDiscriminator", discriminator, load_paths[2], None),
    ):
        if not os.path.exists(path):
            print(f"No checkpoint for {label} at {path}. Training it from scratch.")
            continue
        try:
            sd = torch.load(path, map_location=device, weights_only=True)
            if loader is not None:
                loader(module, sd)
            else:
                module.load_state_dict(sd)
            print(f"Resumed {label} from {path}.")
        except Exception as e:
            print(f"Could not load {label} from {path} ({e}). Training it from scratch.")

    if compile_models and hasattr(torch, "compile"):
        print("Compiling models with torch.compile...")
        model = torch.compile(model)
        upsampler = torch.compile(upsampler)
        discriminator = torch.compile(discriminator)

        
    # GANs require specific betas for stability (0.5, 0.999)
    optimizer_t2i = optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_up = optim.AdamW(upsampler.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    # fp16 needs dynamic loss scaling; without it small GAN gradients can silently
    # underflow on Colab GPUs.  CPU runs remain full precision.
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda and use_amp)
    
    # Initialize EMA trackers (on unwrapped modules so weights stay compatible with checkpoints)
    ema_model = EMA(_unwrap_module(model))
    ema_upsampler = EMA(_unwrap_module(upsampler))
    
    # Hinge Loss is used instead of BCE to prevent the discriminator from overpowering the generator
    
    model.train()
    upsampler.train()
    discriminator.train()
    
    # 4. Training Loop
    # --> PRE-COMPUTE TEXT EMBEDDINGS OUTSIDE THE LOOP <--
    unique_texts = list(set([s.get('text', '') for s in dataset.samples]))
    
    cache_file = "text_embed_cache.pt"
    text_embed_cache = {}
    
    if os.path.exists(cache_file):
        print(f"Loading cached text embeddings from {cache_file}...")
        text_embed_cache = torch.load(cache_file, map_location="cpu", weights_only=True)
        
    missing_texts = [t for t in unique_texts if t not in text_embed_cache]
    
    if missing_texts:
        print(f"Computing CLIP text embeddings for {len(missing_texts)} missing prompts...")
        with torch.no_grad():
            for i in range(0, len(missing_texts), 32):
                if i % 3200 == 0 and i > 0:
                    print(f"  ...computed {i}/{len(missing_texts)} embeddings...")
                
                batch_texts = missing_texts[i:i+32]
                inputs = vision.clip_processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                embeds = vision.clip_model.get_text_features(**inputs)
                embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                for t, e in zip(batch_texts, embeds):
                    text_embed_cache[t] = e.detach().cpu()
        
        print(f"Saving updated text embeddings cache to {cache_file}...")
        torch.save(text_embed_cache, cache_file)
    else:
        print("All text embeddings found in cache!")

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
                
    timed_out = False
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss_G = 0
        total_loss_D = 0
        total_loss_pixel = 0
        batches_done = 0
        d_updates = 0

        # RandomSampler hands the loader a fresh random subset of the corpus each epoch.
        current_dataloader = dataloader
        total_batches = len(current_dataloader) if max_batches is None else min(max_batches, len(current_dataloader))

        batch_start_time = time.time()
        for batch_idx, (images_128, images_256, text_indices) in enumerate(current_dataloader):
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                print("Time limit reached; saving a resumable checkpoint after this partial epoch.")
                break
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            # Measure data loading time and add it to the total batch duration
            data_load_time = time.time() - batch_start_time
            
            images_128 = images_128.to(device, non_blocking=is_cuda)
            images_256 = images_256.to(device, non_blocking=is_cuda)
            if is_cuda:
                images_128 = images_128.to(memory_format=torch.channels_last)
                images_256 = images_256.to(memory_format=torch.channels_last)
            
            text_embeds = embed_matrix[text_indices].to(device, non_blocking=is_cuda)

            # Classifier-Free Guidance training: drop the condition per sample, not per
            # batch, so roughly 10% of every batch teaches the unconditional branch.
            keep = (torch.rand(text_embeds.size(0), 1, device=device) >= 0.1).float()
            text_embeds = text_embeds * keep

            accum_boundary = (batch_idx + 1) % grad_accum_steps == 0
            if batch_idx % grad_accum_steps == 0:
                optimizer_t2i.zero_grad(set_to_none=True)
                optimizer_up.zero_grad(set_to_none=True)
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)

            # -----------------
            #  Train Generator
            # -----------------
            z = torch.randn(text_embeds.size(0), 128, device=device)
            
            amp_device = 'cuda' if 'cuda' in str(device) else 'cpu'
            amp_dtype = torch.float16 if amp_device == 'cuda' else torch.bfloat16
            
            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                preds_128, out_fg_128, out_bg_128, out_mask_128, _ = model(text_embeds, z=z, return_breakdown=True)
                preds_256 = upsampler(preds_128, text_embeds, z=z)
                
                preds_128_up = F.interpolate(preds_128, size=(256, 256), mode='bilinear', align_corners=False)
                
                pred_fake_G_256 = discriminator(preds_256, text_embeds)
                pred_fake_G_128 = discriminator(preds_128_up, text_embeds)
                
                loss_G_GAN = -(pred_fake_G_256.mean() + pred_fake_G_128.mean()) * 0.5
                
                if batch_idx % vgg_freq == 0:
                    loss_G_vgg = vgg_loss(preds_256, images_256)
                    loss_G = loss_G_GAN + 0.1 * loss_G_vgg
                else:
                    loss_G_vgg = torch.tensor(0.0, device=device)
                    loss_G = loss_G_GAN
                    
                loss_G_edge = edge_loss(preds_128, images_128, mask=out_mask_128)
                loss_G = loss_G + (0.005 * loss_G_edge)
                    
                loss_pixel_nudge = (F.l1_loss(preds_128, images_128) + F.l1_loss(preds_256, images_256)) * 0.5
                loss_G = (loss_G + (0.05 * loss_pixel_nudge)) / grad_accum_steps

            scaler.scale(loss_G).backward()
            
            total_loss_G += loss_G.item() * grad_accum_steps
            with torch.no_grad():
                loss_pixel_128 = F.l1_loss(preds_128, images_128)
                loss_pixel_256 = F.l1_loss(preds_256, images_256)
                total_loss_pixel += (loss_pixel_128.item() + loss_pixel_256.item()) * 0.3

            if accum_boundary:
                scaler.step(optimizer_t2i)
                scaler.step(optimizer_up)
                scaler.update()
                ema_model.update()
                ema_upsampler.update()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            if accum_boundary and batch_idx % d_freq == 0:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
                optimizer_D.zero_grad(set_to_none=True)
                
                noise_std = 0.02 * (1.0 - (epoch / epochs))
                noisy_real_256 = images_256 + torch.randn_like(images_256) * noise_std
                
                with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                    pred_real = discriminator(noisy_real_256, text_embeds)
                    loss_D_real = F.relu(1.0 - pred_real).mean()
                    
                    noisy_fake_256 = preds_256.detach() + torch.randn_like(preds_256) * noise_std
                    pred_fake_256 = discriminator(noisy_fake_256, text_embeds)
                    
                    noisy_fake_128 = preds_128_up.detach() + torch.randn_like(preds_128_up) * noise_std
                    pred_fake_128 = discriminator(noisy_fake_128, text_embeds)
                    
                    loss_D_fake = (F.relu(1.0 + pred_fake_256).mean() + 
                                   F.relu(1.0 + pred_fake_128).mean()) * 0.5
                    
                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                    
                scaler.scale(loss_D).backward()
                scaler.step(optimizer_D)
                scaler.update()
                
                total_loss_D += loss_D.item()
                d_updates += 1

            batches_done += 1
            batch_end_time = time.time()
            # Include data loading time in the batch duration calculation
            batch_duration = batch_end_time - batch_start_time
            
            if batch_idx % 5 == 0:
                # If loss_D isn't defined yet (e.g. first batch if we changed the modulo), just print 0
                d_loss_val = loss_D.item() if 'loss_D' in locals() else 0.0
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{total_batches} | D Loss: {d_loss_val:.4f} | G Loss: {loss_G.item():.4f} | {batch_duration:.2f}s/batch (Data load: {data_load_time:.2f}s)")
                
            # Reset timer for the next batch's data loading phase
            batch_start_time = time.time()
                
        avg_loss_G = total_loss_G / max(1, batches_done)
        # D is only updated half the time
        avg_loss_D = total_loss_D / max(1, d_updates)
        avg_loss_pixel = total_loss_pixel / max(1, batches_done)
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_duration/60:.2f} minutes "
              f"({epoch_duration/max(1, batches_done):.2f}s/batch). Avg D Loss: {avg_loss_D:.4f} | "
              f"Avg G Loss: {avg_loss_G:.4f} | Avg Pixel L1: {avg_loss_pixel:.4f}")
        
        # Save generated samples for visual evaluation using EMA weights
        test_prompts = ["a person running", "a cat sleeping", "a beautiful sunset", "a car driving", "playing guitar"]
        
        # Apply EMA weights for generation
        ema_model.apply_shadow()
        ema_upsampler.apply_shadow()
        
        # Generate with a fixed noise vector so we can see how the SAME latent space evolves over time
        fixed_z = torch.randn(len(test_prompts), 128, device=device)
        save_samples(model, upsampler, vision, test_prompts, epoch, device,
                     output_dir=f"training_samples_{tag}" if tag else "training_samples", z=fixed_z)
        
        # Save checkpoint after every epoch (using EMA weights, unwrapped keys for compatibility)
        torch.save(_state_dict_for_save(model), save_path)
        torch.save(_state_dict_for_save(upsampler), up_save_path)
        torch.save(_state_dict_for_save(discriminator), d_save_path)
        print(f"Saved EMA checkpoints to {save_path}, {up_save_path}, and {d_save_path} after Epoch {epoch+1}")
        
        # Restore active weights for next epoch
        ema_model.restore()
        ema_upsampler.restore()
        if timed_out:
            break
        
    print(f"Pretraining complete. Final models saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain TinyT2I model.")
    parser.add_argument("--activities", action="store_true", help="Scrape and train on activities from activities.txt")
    parser.add_argument("--activities-file", type=str, default="activities.txt", help="Path to activities file (default: activities.txt)")
    parser.add_argument("--dataset", action="store_true", help="Train on locally saved datasets (dataset/scraped_images/)")
    parser.add_argument("--full", action="store_true", help="Run a deep training session with more epochs")
    parser.add_argument("--attn-downsample", type=int, default=2,
                        help="Pool the latent by this factor before world-model attention (1=32x32 tokens, 2=16x16)")
    parser.add_argument("--max-batches", type=int, default=None, help="Cap batches per epoch (for benchmarking/A-B runs)")
    parser.add_argument("--tag", type=str, default="", help="Write checkpoints and samples to tagged files instead of the real ones")
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch count")
    parser.add_argument("--seed", type=int, default=None, help="Seed weight init, augmentation and data order for reproducible comparisons")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (default: 32; lower it if CUDA runs out of memory)")
    parser.add_argument("--vgg-freq", type=int, default=1, help="Calculate VGG loss every N batches (default: 1, increase to speed up training)")
    parser.add_argument("--d-freq", type=int, default=2, help="Update discriminator every N batches (default: 2, prevents mode collapse)")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True,
                        help="Use CUDA fp16 mixed precision (enabled by default on CUDA)")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Disable CUDA mixed precision for debugging")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Stop cleanly after this many hours and save a checkpoint")
    parser.add_argument("--no-preload", dest="preload", action="store_false", default=True,
                        help="Disable RAM preloading of images (slower on GPU)")
    parser.add_argument("--compile", dest="compile_models", action="store_true", default=False,
                        help="Use torch.compile for faster GPU kernels (PyTorch 2+)")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Accumulate gradients over N batches to simulate larger batch with less VRAM")
    parser.add_argument("--vgg-chunk-size", type=int, default=8,
                        help="Process VGG loss in chunks of N images to reduce VRAM spikes")
    
    args = parser.parse_args()
    
    # Default behavior if neither is specified, assume both for backward compatibility, 
    # but since flags are added, if either is used, respect them.
    use_act = args.activities
    use_dat = args.dataset
    
    # If no flags provided, maybe default to doing both so it works like before
    if not args.activities and not args.dataset:
        print("No data flags provided (--activities or --dataset), defaulting to using both.")
        use_act = True
        use_dat = True
        
    epochs = args.epochs if args.epochs is not None else (15 if args.full else 8)
    lr = 5e-4 if args.full else 1e-3
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pretrain_tiny_t2i(
        device=device,
        epochs=epochs, 
        batch_size=args.batch_size,
        lr=lr, 
        use_activities=use_act, 
        use_dataset=use_dat,
        activities_file=args.activities_file,
        attn_downsample=args.attn_downsample,
        max_batches=args.max_batches,
        tag=args.tag,
        seed=args.seed,
        vgg_freq=args.vgg_freq,
        d_freq=args.d_freq,
        use_amp=args.amp,
        max_hours=args.max_hours,
        preload_images=args.preload,
        compile_models=args.compile_models,
        grad_accum_steps=args.grad_accum,
        vgg_chunk_size=args.vgg_chunk_size,
    )
