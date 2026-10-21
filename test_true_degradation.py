import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import random
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage

import sys
sys.path.append('.')
from sr_resnet import SRResNet
from experiment_phase2_local import TPlus1VAE

def test_upscale_options(device='cuda' if torch.cuda.is_available() else 'cpu'):
    vae = TPlus1VAE(latent_dim=64).to(device)
    state_dict = torch.load("t_plus_1_vae_ep52.pt", map_location=device, weights_only=True)
    if 'decoder.deconv.9.weight' in state_dict and state_dict['decoder.deconv.9.weight'].shape[1] != 5:
        del state_dict['decoder.deconv.9.weight']
        del state_dict['decoder.deconv.9.bias']
    vae.load_state_dict(state_dict, strict=False)
    vae.eval()

    sr_model = SRResNet().to(device)
    sr_model.load_state_dict(torch.load("sr_resnet_final.pt", map_location=device, weights_only=True))
    sr_model.eval()

    img_paths = list(Path("dataset/distillation").glob("*.png"))
    if not img_paths: return

    # Test with 5 different images this time
    test_paths = random.sample(img_paths, min(5, len(img_paths)))
    
    fig, axes = plt.subplots(len(test_paths), 3, figsize=(12, 4 * len(test_paths)))
    if len(test_paths) == 1: axes = [axes]
    
    with torch.no_grad():
        for i, img_path in enumerate(test_paths):
            img_pil = Image.open(img_path).convert('RGB').resize((128, 128), Image.BILINEAR)
            img_tensor = ToTensor()(img_pil).unsqueeze(0).to(device)
            
            img_64 = F.interpolate(img_tensor, size=(64, 64), mode='bilinear', align_corners=False)
            mu, _ = vae.encoder(img_64)
            flow, img_res = vae.decoder(mu)
            
            degraded_tensor = torch.clamp(vae.warp(img_64, flow) + img_res, 0, 1)
            
            axes[i][0].imshow(img_pil)
            axes[i][0].set_title("Ground Truth (128x128)")
            axes[i][0].axis('off')
            
            axes[i][1].imshow(ToPILImage()(degraded_tensor.squeeze(0).cpu()))
            axes[i][1].set_title("True VAE Output (64x64)")
            axes[i][1].axis('off')
            
            resnet_upscale = sr_model(degraded_tensor)
            resnet_upscale = torch.clamp(resnet_upscale, 0, 1)
            
            axes[i][2].imshow(ToPILImage()(resnet_upscale.squeeze(0).cpu()))
            axes[i][2].set_title("ResNet Upscale")
            axes[i][2].axis('off')
            
    plt.tight_layout()
    plt.savefig("true_vae_degradation_extended.png", bbox_inches='tight')
    print("Saved true_vae_degradation_extended.png")

if __name__ == '__main__':
    test_upscale_options()