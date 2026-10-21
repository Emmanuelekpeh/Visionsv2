import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperT2I(nn.Module):
    """Text embedding generates the weights of an image generator network
    
    The text creates a custom "generator network" that produces the image.
    This is a hypernetwork approach where the model learns to create models.
    """
    def __init__(self, text_dim=512, img_size=128, noise_dim=128):
        super().__init__()
        self.img_size = img_size
        self.noise_dim = noise_dim
        
        # Text + Noise → Hypernetwork features
        self.feature_extractor = nn.Sequential(
            nn.Linear(text_dim + noise_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Weight predictors for the generated image network
        # We generate a small decoder network: 256 -> 128 -> 64 -> RGB
        self.gen_decoder_w1 = nn.Linear(2048, 256 * 128)      # First decoder layer
        self.gen_decoder_b1 = nn.Linear(2048, 128)
        
        self.gen_decoder_w2 = nn.Linear(2048, 128 * 64)       # Second decoder layer
        self.gen_decoder_b2 = nn.Linear(2048, 64)
        
        self.gen_decoder_w3 = nn.Linear(2048, 64 * 32)        # Third decoder layer
        self.gen_decoder_b3 = nn.Linear(2048, 32)
        
        self.gen_to_rgb = nn.Linear(2048, 32 * 3 * 8 * 8)     # Final layer to spatial RGB
        self.gen_to_rgb_bias = nn.Linear(2048, 3 * 8 * 8)
        
        # Fixed noise pattern that the generated network will transform
        init_size = 8
        self.register_buffer('base_noise', torch.randn(1, 256, init_size, init_size))
        
        # Static upsampler (not generated, for stability)
        # 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.upsampler = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(8, 3, 3, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, text_embeds, z=None, guidance_scale=1.0):
        B = text_embeds.size(0)
        device = text_embeds.device
        
        if z is None:
            z = torch.randn(B, self.noise_dim, device=device)
        
        # Classifier-free guidance
        if guidance_scale > 1.0:
            text_embeds = torch.cat([torch.zeros_like(text_embeds), text_embeds])
            z = torch.cat([z, z])
            B_expanded = B * 2
        else:
            B_expanded = B
        
        # Extract hypernetwork features
        hyper_input = torch.cat([text_embeds, z], dim=1)
        features = self.feature_extractor(hyper_input)
        
        # Generate network weights from features
        w1 = self.gen_decoder_w1(features).view(B_expanded, 128, 256)
        b1 = self.gen_decoder_b1(features).view(B_expanded, 128, 1)
        
        w2 = self.gen_decoder_w2(features).view(B_expanded, 64, 128)
        b2 = self.gen_decoder_b2(features).view(B_expanded, 64, 1)
        
        w3 = self.gen_decoder_w3(features).view(B_expanded, 32, 64)
        b3 = self.gen_decoder_b3(features).view(B_expanded, 32, 1)
        
        w_rgb = self.gen_to_rgb(features).view(B_expanded, 3 * 8 * 8, 32)
        b_rgb = self.gen_to_rgb_bias(features).view(B_expanded, 3 * 8 * 8, 1)
        
        # Run the generated network
        # Start with base noise
        x = self.base_noise.expand(B_expanded, -1, -1, -1)
        
        # Global average pooling to get a feature vector
        x_vec = x.mean(dim=(2, 3))  # B, 256
        
        # Layer 1: 256 -> 128
        x_vec = torch.bmm(w1, x_vec.unsqueeze(2)) + b1  # B, 128, 1
        x_vec = F.leaky_relu(x_vec, 0.2)
        
        # Layer 2: 128 -> 64
        x_vec = torch.bmm(w2, x_vec) + b2  # B, 64, 1
        x_vec = F.leaky_relu(x_vec, 0.2)
        
        # Layer 3: 64 -> 32
        x_vec = torch.bmm(w3, x_vec) + b3  # B, 32, 1
        x_vec = F.leaky_relu(x_vec, 0.2)
        
        # To RGB: 32 -> 3*8*8
        x_vec = torch.bmm(w_rgb, x_vec) + b_rgb  # B, 3*8*8, 1
        x_vec = x_vec.squeeze(2).view(B_expanded, 3, 8, 8)
        x_vec = torch.sigmoid(x_vec)
        
        # Upsample to target resolution using static network
        out = self.upsampler(x_vec)
        
        # Apply CFG
        if guidance_scale > 1.0:
            uncond, cond = out.chunk(2)
            out = uncond + guidance_scale * (cond - uncond)
            out = torch.clamp(out, 0, 1)
        
        return out


class DirectCNNT2I(nn.Module):
    """Dead simple: Text embedding → CNN → RGB pixels
    
    No latents, no attention, no tricks. Just conv layers.
    """
    def __init__(self, text_dim=512, img_size=128, noise_dim=128):
        super().__init__()
        self.img_size = img_size
        self.noise_dim = noise_dim
        
        # Text + Noise → Spatial features
        init_size = img_size // 16  # Start at 8x8 for 128x128
        self.fc = nn.Sequential(
            nn.Linear(text_dim + noise_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512 * init_size * init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Simple progressive upsampling
        self.decoder = nn.Sequential(
            # 8x8 → 16x16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 → 32x32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 → 64x64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 → 128x128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final RGB
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, text_embeds, z=None, guidance_scale=1.0):
        B = text_embeds.size(0)
        device = text_embeds.device
        
        if z is None:
            z = torch.randn(B, self.noise_dim, device=device)
        
        # Classifier-free guidance
        if guidance_scale > 1.0:
            text_embeds = torch.cat([torch.zeros_like(text_embeds), text_embeds])
            z = torch.cat([z, z])
            B_expanded = B * 2
        else:
            B_expanded = B
        
        # Generate
        x = self.fc(torch.cat([text_embeds, z], dim=1))
        x = x.view(B_expanded, 512, self.img_size // 16, self.img_size // 16)
        out = self.decoder(x)
        
        # Apply CFG
        if guidance_scale > 1.0:
            uncond, cond = out.chunk(2)
            out = uncond + guidance_scale * (cond - uncond)
            out = torch.clamp(out, 0, 1)
        
        return out


class DirectContinuationCNN(nn.Module):
    """Generative T+1 Student Model.
    Takes (Previous Frame + Text + Noise) -> Generates Next Frame.
    """
    def __init__(self, text_dim=512, img_size=128, noise_dim=128):
        super().__init__()
        self.img_size = img_size
        self.noise_dim = noise_dim
        
        # Encode previous frame
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), # 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), # 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), # 8
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Text + Noise -> Spatial features
        self.cond_fc = nn.Sequential(
            nn.Linear(text_dim + noise_dim, 256 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decode combined features
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 128x128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final RGB
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img_t, text_embeds, z=None, guidance_scale=1.0):
        B = text_embeds.size(0)
        device = text_embeds.device
        
        if z is None:
            z = torch.randn(B, self.noise_dim, device=device)
            
        # Classifier-free guidance
        if guidance_scale > 1.0:
            text_embeds = torch.cat([torch.zeros_like(text_embeds), text_embeds])
            z = torch.cat([z, z])
            img_t = torch.cat([img_t, img_t])
            B_expanded = B * 2
        else:
            B_expanded = B
            
        # Extract features
        img_feats = self.img_encoder(img_t) # [B, 256, 8, 8]
        
        cond = self.cond_fc(torch.cat([text_embeds, z], dim=1))
        cond = cond.view(B_expanded, 256, 8, 8)
        
        # Combine and decode
        x = torch.cat([img_feats, cond], dim=1) # [B, 512, 8, 8]
        out = self.decoder(x)
        
        # Apply CFG
        if guidance_scale > 1.0:
            uncond, cond = out.chunk(2)
            out = uncond + guidance_scale * (cond - uncond)
            out = torch.clamp(out, 0, 1)
            
        return out

class SimpleDiscriminator(nn.Module):
    """Lightweight discriminator for both architectures"""
    def __init__(self, img_size=128, text_embed_dim=512):
        super().__init__()
        
        # Image processing path
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),    # 128 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),   # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),  # 32 -> 16
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Text conditioning
        self.text_proj = nn.utils.spectral_norm(nn.Linear(text_embed_dim, 256))
        
        # Final classification
        self.out = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1)),  # 16 -> 8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1)  # 8 -> 7
        )
        
    def forward(self, img, text_embeds):
        B = img.size(0)
        
        # Process image
        x = self.conv(img)  # [B, 256, 16, 16]
        
        # Process text and expand spatially
        t = self.text_proj(text_embeds)  # [B, 256]
        t = t.view(B, 256, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        
        # Combine and classify
        combined = torch.cat([x, t], dim=1)  # [B, 512, 16, 16]
        return self.out(combined)


if __name__ == "__main__":
    # Quick test
    device = "cpu"
    
    print("Testing HyperT2I...")
    hyper = HyperT2I().to(device)
    text = torch.randn(2, 512, device=device)
    out = hyper(text, guidance_scale=1.0)
    print(f"  Output shape: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in hyper.parameters()):,}")
    
    print("\nTesting DirectCNNT2I...")
    direct = DirectCNNT2I().to(device)
    out = direct(text, guidance_scale=1.0)
    print(f"  Output shape: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in direct.parameters()):,}")
    
    print("\nTesting SimpleDiscriminator...")
    disc = SimpleDiscriminator().to(device)
    out = disc(torch.randn(2, 3, 128, 128, device=device), text)
    print(f"  Output shape: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in disc.parameters()):,}")
