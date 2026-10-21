import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import random
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage

import sys
sys.path.append('.')
from sr_upscaler import SRUpscaler
from experiment_phase2_local import TPlus1VAE

def test_upscale(device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load VAE
    vae = TPlus1VAE(latent_dim=64).to(device)
    state_dict = torch.load("t_plus_1_vae_ep52.pt", map_location=device, weights_only=True)
    if 'decoder.deconv.9.weight' in state_dict and state_dict['decoder.deconv.9.weight'].shape[1] != 5:
        del state_dict['decoder.deconv.9.weight']
        del state_dict['decoder.deconv.9.bias']
    vae.load_state_dict(state_dict, strict=False)
    vae.eval()

    # Load current ESPCN upscaler
    sr_model = SRUpscaler().to(device)
    sr_model.load_state_dict(torch.load("sr_upscaler_final.pt", map_location=device, weights_only=True))
    sr_model.eval()

    # Find some test images
    img_paths = list(Path("dataset/distillation").glob("*.png"))
    if not img_paths:
        print("No images found in dataset/distillation")
        return

    test_paths = random.sample(img_paths, min(4, len(img_paths)))
    
    fig, axes = plt.subplots(len(test_paths), 4, figsize=(16, 4 * len(test_paths)))
    
    with torch.no_grad():
        for i, img_path in enumerate(test_paths):
            # Original
            img_pil = Image.open(img_path).convert('RGB').resize((128, 128), Image.BILINEAR)
            img_tensor = ToTensor()(img_pil).unsqueeze(0).to(device)
            
            # VAE degraded
            img_64 = F.interpolate(img_tensor, size=(64, 64), mode='bilinear', align_corners=False)
            mu, _ = vae.encoder(img_64)
            _, img_res = vae.decoder(mu)
            degraded_tensor = torch.clamp(img_res, 0, 1)
            
            # Bilinear Upscale
            bilinear_tensor = F.interpolate(degraded_tensor, size=(128, 128), mode='bilinear', align_corners=False)
            
            # SR Upscale
            sr_tensor = sr_model(degraded_tensor)
            sr_tensor = torch.clamp(sr_tensor, 0, 1)
            
            axes[i][0].imshow(img_pil)
            axes[i][0].set_title("Original (128x128)")
            axes[i][0].axis('off')
            
            axes[i][1].imshow(ToPILImage()(degraded_tensor.squeeze(0).cpu()))
            axes[i][1].set_title("T+1 VAE Output (64x64)")
            axes[i][1].axis('off')
            
            axes[i][2].imshow(ToPILImage()(bilinear_tensor.squeeze(0).cpu()))
            axes[i][2].set_title("Bilinear Upscale (128x128)")
            axes[i][2].axis('off')
            
            axes[i][3].imshow(ToPILImage()(sr_tensor.squeeze(0).cpu()))
            axes[i][3].set_title("Custom ESPCN Upscale (128x128)")
            axes[i][3].axis('off')
            
    plt.tight_layout()
    plt.savefig("upscale_comparison_1.png", bbox_inches='tight')
    print("Saved upscale_comparison_1.png")

if __name__ == '__main__':
    test_upscale()
