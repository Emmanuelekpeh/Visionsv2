import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class SimpleFlowInterpolator(nn.Module):
    """
    A lightweight, CPU-friendly optical flow interpolator.
    Instead of full RIFE/FILM, this uses a tiny CNN to predict a flow map and blending mask.
    """
    def __init__(self):
        super().__init__()
        # Ensure torchvision is imported
        import torchvision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load RAFT model instead of the simple CNN
        try:
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            print("Loading RAFT Optical Flow Model...")
            self.weights = Raft_Small_Weights.DEFAULT
            self.raft = raft_small(weights=self.weights, progress=False).to(self.device)
            self.raft.eval()
            self.use_raft = True
        except Exception as e:
            print(f"Could not load RAFT: {e}. Falling back to simple CNN.")
            self.use_raft = False
            
            # Legacy CNN Fallback
            self.conv1 = nn.Conv2d(6, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
            self.upconv = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
            self.out_conv = nn.Conv2d(32, 4, 3, padding=1)
        
    def forward(self, img1, img2, t):
        """
        img1, img2: [B, 3, H, W] in range [0, 1]
        t: float between 0 and 1
        """
        if self.use_raft:
            return self._raft_forward(img1, img2, t)
        else:
            return self._legacy_forward(img1, img2, t)
            
    def _raft_forward(self, img1, img2, t):
        # Convert [0, 1] inputs back to [0, 255] for RAFT transforms
        i1 = (img1 * 255).to(torch.uint8)
        i2 = (img2 * 255).to(torch.uint8)
        
        # We need to process image by image to apply transforms
        from torchvision.transforms.functional import resize
        B, C, H, W = img1.shape
        
        # Resize to RAFT compatible dimensions (needs to be divisible by 8)
        # Using 128x128 which is compatible
        new_h, new_w = 128, 128
        i1_resized = resize(i1, [new_h, new_w])
        i2_resized = resize(i2, [new_h, new_w])
        
        # Apply RAFT transforms (scaling/normalization)
        transforms = self.weights.transforms()
        batch1, batch2 = transforms(i1_resized, i2_resized)
        batch1, batch2 = batch1.to(self.device), batch2.to(self.device)
        
        with torch.no_grad():
            list_of_flows = self.raft(batch1, batch2)
            predicted_flow = list_of_flows[-1] # shape [B, 2, H, W]
            
        # Scale flow by time t
        scaled_flow = predicted_flow * t
        
        # If original dimensions differ, resize flow
        if H != new_h or W != new_w:
            scaled_flow = F.interpolate(scaled_flow, size=(H, W), mode='bilinear', align_corners=False)
            scaled_flow[:, 0] *= float(W) / new_w
            scaled_flow[:, 1] *= float(H) / new_h
        
        # Warp original [0, 1] float tensor
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), 2).float().to(img1.device)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        
        # RAFT outputs absolute pixel displacement
        vgrid = grid + scaled_flow.permute(0, 2, 3, 1)
        vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
        vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
        
        warped_img1 = F.grid_sample(img1, vgrid, align_corners=True, padding_mode='border')
        
        return warped_img1

    def _legacy_forward(self, img1, img2, t):
        x = torch.cat([img1, img2], dim=1)
        
        f = F.leaky_relu(self.conv1(x), 0.2)
        f = F.leaky_relu(self.conv2(f), 0.2)
        f = F.leaky_relu(self.conv3(f), 0.2)
        
        f = F.leaky_relu(self.upconv(f), 0.2)
        out = self.out_conv(f)
        
        # Scale flow by time t
        flow = out[:, :2] * t
        mask = torch.sigmoid(out[:, 2:3])
        res = out[:, 3:4] * 0.1 # Small residual correction
        
        # Warp img1
        B, C, H, W = img1.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), 2).float().to(img1.device)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Add flow and normalize to [-1, 1] for grid_sample
        vgrid = grid + flow.permute(0, 2, 3, 1)
        vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
        vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
        
        warped_img1 = F.grid_sample(img1, vgrid, align_corners=True, padding_mode='border')
        
        # Simple blend for now (in a real model, mask would choose between warped img1 and warped img2)
        blended = warped_img1 * (1 - t) + img2 * t
        return torch.clamp(blended + res, 0, 1)

def interpolate_frames(img1_path, img2_path, num_frames=10, output_prefix="interp"):
    """Test function for the interpolator"""
    device = "cpu"
    model = SimpleFlowInterpolator().to(device)
    # In a real scenario, we would load trained weights here
    
    img1 = Image.open(img1_path).convert('RGB').resize((256, 256))
    img2 = Image.open(img2_path).convert('RGB').resize((256, 256))
    
    t1 = torch.from_numpy(np.array(img1)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t2 = torch.from_numpy(np.array(img2)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    frames = []
    with torch.no_grad():
        for i in range(num_frames):
            t = i / float(num_frames - 1)
            out = model(t1, t2, t)
            
            out_img = (out.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frames.append(Image.fromarray(out_img))
            
    for i, frame in enumerate(frames):
        frame.save(f"{output_prefix}_{i:03d}.png")
    print(f"Saved {num_frames} interpolated frames.")
    
if __name__ == "__main__":
    # Create dummy images for testing
    i1 = Image.new('RGB', (256, 256), color = 'red')
    i2 = Image.new('RGB', (256, 256), color = 'blue')
    i1.save("dummy1.png")
    i2.save("dummy2.png")
    interpolate_frames("dummy1.png", "dummy2.png")
