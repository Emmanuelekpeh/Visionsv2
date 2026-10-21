import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import io

from test_phase2_local import TPlus1VAE

def generate_multi_step_debug(image_path, num_frames=5, output_path="visual_test_t1_multistep.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load model
    model = TPlus1VAE(latent_dim=64).to(device)
    state_dict = torch.load("t_plus_1_vae_ep52.pt", map_location=device, weights_only=True)
    if 'decoder.deconv.9.weight' in state_dict and state_dict['decoder.deconv.9.weight'].shape[1] != 5:
        print("Dropping last decoder layer from checkpoint due to architecture change.")
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 2. Load starting image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    init_img = Image.open(image_path).convert("RGB")
    x_t = transform(init_img).unsqueeze(0).to(device)
    
    frames_all = []
    
    with torch.no_grad():
        for i in range(num_frames):
            if i == 0:
                frames_all.append({"type": "input", "img": x_t.squeeze(0).cpu()})
                continue
                
            x_next, flow_pred, res_pred, warped = model.predict_next_debug(x_t)
            
            frames_all.append({
                "type": "pred",
                "img": x_next.squeeze(0).cpu(),
                "flow": flow_pred.squeeze(0).cpu(),
                "res": res_pred.squeeze(0).cpu(),
                "warped": warped.squeeze(0).cpu()
            })
            x_t = x_next
            
    # Visualize grid
    rows = 4  # Img, Flow, Warped, Res
    cols = num_frames
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    
    for c in range(cols):
        item = frames_all[c]
        
        # Row 0: Img
        ax = axes[0, c]
        img_np = item["img"].permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        ax.set_title(f"Frame {c}")
        ax.axis('off')
        
        if c > 0:
            # Row 1: Flow
            ax = axes[1, c]
            flow_np = item["flow"].permute(1, 2, 0).numpy()
            flow_rgb = np.zeros((64, 64, 3))
            flow_rgb[:, :, :2] = (flow_np - flow_np.min()) / (flow_np.max() - flow_np.min() + 1e-5)
            ax.imshow(flow_rgb)
            ax.set_title("Flow (normalized)")
            ax.axis('off')
            
            # Row 2: Warped
            ax = axes[2, c]
            warped_np = item["warped"].permute(1, 2, 0).numpy()
            warped_np = np.clip(warped_np, 0, 1)
            ax.imshow(warped_np)
            ax.set_title("Warped")
            ax.axis('off')
            
            # Row 3: Res
            ax = axes[3, c]
            res_np = item["res"].permute(1, 2, 0).numpy()
            res_norm = (res_np - res_np.min()) / (res_np.max() - res_np.min() + 1e-5)
            ax.imshow(res_norm)
            ax.set_title("Residual")
            ax.axis('off')
        else:
            for r in range(1, rows):
                axes[r, c].axis('off')
                
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved multistep debug visualization to {output_path}")

if __name__ == "__main__":
    import glob
    possible_imgs = glob.glob("dataset/**/*.png", recursive=True)
    if possible_imgs:
        test_img = possible_imgs[0]
        print(f"Testing on {test_img}")
        generate_multi_step_debug(test_img, num_frames=6)
