import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from pathlib import Path
import random
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage

import sys
sys.path.append('.')
from sr_upscaler import SRUpscaler
from experiment_phase2_local import TPlus1VAE

def enhance_image(pil_img):
    # Option 1: Unsharp Mask
    img_unsharp = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    # Option 2: EDGE_ENHANCE
    img_edge = pil_img.filter(ImageFilter.EDGE_ENHANCE)
    return img_unsharp, img_edge

def test_upscale_options(device='cuda' if torch.cuda.is_available() else 'cpu'):
    vae = TPlus1VAE(latent_dim=64).to(device)
    state_dict = torch.load("t_plus_1_vae_ep52.pt", map_location=device, weights_only=True)
    if 'decoder.deconv.9.weight' in state_dict and state_dict['decoder.deconv.9.weight'].shape[1] != 5:
        del state_dict['decoder.deconv.9.weight']
        del state_dict['decoder.deconv.9.bias']
    vae.load_state_dict(state_dict, strict=False)
    vae.eval()

    sr_model = SRUpscaler().to(device)
    sr_model.load_state_dict(torch.load("sr_upscaler_final.pt", map_location=device, weights_only=True))
    sr_model.eval()

    img_paths = list(Path("dataset/distillation").glob("*.png"))
    if not img_paths: return

    test_paths = random.sample(img_paths, min(3, len(img_paths)))
    
    fig, axes = plt.subplots(len(test_paths), 5, figsize=(20, 4 * len(test_paths)))
    if len(test_paths) == 1: axes = [axes]
    
    with torch.no_grad():
        for i, img_path in enumerate(test_paths):
            img_pil = Image.open(img_path).convert('RGB').resize((128, 128), Image.BILINEAR)
            img_tensor = ToTensor()(img_pil).unsqueeze(0).to(device)
            
            img_64 = F.interpolate(img_tensor, size=(64, 64), mode='bilinear', align_corners=False)
            mu, _ = vae.encoder(img_64)
            _, img_res = vae.decoder(mu)
            degraded_tensor = torch.clamp(img_res, 0, 1)
            
            sr_tensor = sr_model(degraded_tensor)
            sr_tensor = torch.clamp(sr_tensor, 0, 1)
            
            sr_pil = ToPILImage()(sr_tensor.squeeze(0).cpu())
            unsharp_pil, edge_pil = enhance_image(sr_pil)
            
            axes[i][0].imshow(img_pil)
            axes[i][0].set_title("Ground Truth (128x128)")
            axes[i][0].axis('off')
            
            axes[i][1].imshow(ToPILImage()(degraded_tensor.squeeze(0).cpu()))
            axes[i][1].set_title("VAE Output (64x64)")
            axes[i][1].axis('off')
            
            axes[i][2].imshow(sr_pil)
            axes[i][2].set_title("Current ESPCN (128x128)")
            axes[i][2].axis('off')
            
            axes[i][3].imshow(unsharp_pil)
            axes[i][3].set_title("ESPCN + Unsharp Mask")
            axes[i][3].axis('off')
            
            axes[i][4].imshow(edge_pil)
            axes[i][4].set_title("ESPCN + Edge Enhance")
            axes[i][4].axis('off')
            
    plt.tight_layout()
    plt.savefig("upscale_options_2.png", bbox_inches='tight')
    print("Saved upscale_options_2.png")

if __name__ == '__main__':
    test_upscale_options()
