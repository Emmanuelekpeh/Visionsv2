import torch
import json
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms

def generate_true_vae_degradation_pairs(teacher_images_dir, output_dir, device="cpu"):
    """
    Generates training pairs for SRUpscaler by pushing real images through 
    the actual T+1 VAE to capture its specific artifacts (blur, color bleed) 
    instead of synthetic bilinear downsampling.
    """
    import sys
    sys.path.append('.')
    from experiment_phase2_local import TPlus1VAE
    
    print("Loading T+1 VAE...")
    vae = TPlus1VAE(latent_dim=64).to(device)
    state_dict = torch.load("t_plus_1_vae_ep52.pt", map_location=device, weights_only=True)
    if 'decoder.deconv.9.weight' in state_dict and state_dict['decoder.deconv.9.weight'].shape[1] != 5:
        del state_dict['decoder.deconv.9.weight']
        del state_dict['decoder.deconv.9.bias']
    vae.load_state_dict(state_dict, strict=False)
    vae.eval()
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = out_dir / "t_plus_1_sr_pairs.jsonl"
    
    img_paths = list(Path(teacher_images_dir).glob("*.png")) + list(Path(teacher_images_dir).glob("*.jpg"))
    print(f"Found {len(img_paths)} teacher images for dataset generation.")
    
    if len(img_paths) == 0:
        print("No images found. Please provide a directory with high-res Teacher images (e.g. 128x128).")
        return
        
    with open(manifest_path, "w") as f:
        with torch.no_grad():
            for i, img_path in enumerate(img_paths):
                # Target: 128x128 Teacher Image (Ground Truth)
                target_img = Image.open(img_path).convert('RGB').resize((128, 128), Image.BILINEAR)
                target_path = out_dir / f"target_{i:04d}.png"
                target_img.save(target_path)
                
                # Downsample to 64x64 for VAE input (simulating the Flow output)
                input_64 = target_img.resize((64, 64), Image.BILINEAR)
                tensor_64 = transforms.ToTensor()(input_64).unsqueeze(0).to(device)
                
                # Push through VAE to get REAL artifacts (blur, bleed)
                mu, _ = vae.encoder(tensor_64)
                flow, img_res = vae.decoder(mu)
                
                # FIXED: The VAE doesn't output the image directly! It outputs flow + residual.
                # We have to warp the input and add the residual to get the final frame.
                reconstructed = torch.clamp(vae.warp(tensor_64, flow) + img_res, 0, 1)
                
                recon_np = (reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                degraded_img = Image.fromarray(recon_np)
                
                degraded_path = out_dir / f"degraded_{i:04d}.png"
                degraded_img.save(degraded_path)
                
                # Save pair mapped to the real degraded image, NOT a synthetic bilinear scale
                f.write(json.dumps({
                    "degraded_64": str(degraded_path),
                    "target_128": str(target_path)
                }) + "\n")
                
                if (i+1) % 10 == 0:
                    print(f"Processed {i+1}/{len(img_paths)} images...")
                    
    print(f"Dataset generated at {manifest_path}")

if __name__ == '__main__':
    # Defaulting to dataset/distillation which holds our HumanApproved teacher frames
    generate_true_vae_degradation_pairs("dataset/distillation", "dataset/sr_training_true_vae")