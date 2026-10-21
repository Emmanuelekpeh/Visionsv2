import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        # Spatial noise scaling parameter
        self.noise_weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
    def forward(self, x):
        skip = x
        x = self.act(self.bn1(self.conv1(x)))
        
        # Inject spatial noise
        B, _, H, W = x.shape
        noise = torch.randn(B, 1, H, W, device=x.device, dtype=x.dtype)
        x = x + noise * self.noise_weight
        
        x = self.bn2(self.conv2(x))
        return self.act(x + skip)

class LatentWorldModel(nn.Module):
    """
    Acts as the 'Imagination' layer in latent space.
    Takes the base spatial latent features and allows them to globally attend to each other
    and the text prompt to 'invent' or 'resolve' features before decoding.
    """
    def __init__(self, channels=32, text_dim=512, num_heads=4, depth=3, latent_size=32, attn_downsample=2):
        super().__init__()
        self.channels = channels
        self.latent_size = latent_size
        self.noise_dim = text_dim // 4 # e.g. 128

        # Attention is O(tokens^2). Pooling the latent before attending cuts the
        # cost quadratically; the residual then restores the fine detail that
        # pooling discarded, so only the global reasoning happens at low res.
        self.attn_downsample = attn_downsample
        self.attn_size = latent_size // attn_downsample
        num_tokens = self.attn_size * self.attn_size
        
        # Positional embedding for spatial tokens + 1 text token
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens + 1, channels) * (1.0 / (channels ** 0.5)))
        self.text_proj = nn.Linear(text_dim, channels)
        
        # Noise injection layers for superposition
        # We project the noise to match the channel dimension and add it before the transformer
        self.noise_proj = nn.Linear(self.noise_dim, channels)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, 
            nhead=num_heads, 
            dim_feedforward=channels * 4, 
            batch_first=True,
            norm_first=True # Better stability for deep transformers
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x, text_embeds, z=None):
        B, C, H, W = x.shape

        skip = x
        if self.attn_downsample > 1:
            x = F.avg_pool2d(x, self.attn_downsample)

        # Flatten spatial dimensions: B, C, H, W -> B, H*W, C
        x_flat = x.reshape(B, C, -1).permute(0, 2, 1)
        
        # Noise Superposition: Add noise directly to the spatial latents
        if z is not None:
            noise_token = self.noise_proj(z).unsqueeze(1) # B, 1, C
            x_flat = x_flat + noise_token # Broadcasts across all spatial tokens
        
        # Project text to token: B, text_dim -> B, 1, C
        text_token = self.text_proj(text_embeds).unsqueeze(1)
        
        # Concatenate text token with spatial tokens
        tokens = torch.cat([text_token, x_flat], dim=1) # B, (H*W + 1), C
        
        # Add positional embeddings
        tokens = tokens + self.pos_embed
        
        # Transform (Imagination happens here)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        
        # Separate text token and reshape spatial tokens back to image dimensions
        out_spatial = tokens[:, 1:, :].permute(0, 2, 1).reshape(B, C, x.size(2), x.size(3))

        if self.attn_downsample > 1:
            out_spatial = F.interpolate(out_spatial, size=(H, W), mode='bilinear', align_corners=False)
            out_spatial = out_spatial + skip

        return out_spatial

# 1. LATENT SPACE PROCESSING (Mini-VAE)
class TinyVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        # Decoder (Latent 32x32 -> RGB + Mask 128x128)
        # Replaced ConvTranspose2d with Upsample+Conv2d to eliminate checkerboard artifacts
        self.decoder_fg = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(latent_dim, 64, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True), # 64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True), # 128
            nn.Conv2d(32, 4, 3, 1, 1),
            nn.Sigmoid()
        )
        # Outputs 3 channels for Background (RGB)
        self.decoder_bg = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(latent_dim, 64, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

class TinyT2I(nn.Module):
    """
    Integrates: 
    1. Latent Space Processing (VAE)
    2. Foreground/Background Split (Alpha Compositor)
    3. Classifier-Free Guidance (CFG)
    """
    def __init__(self, text_embed_dim=512, img_size=128, hidden_dim=256, latent_dim=16, attn_downsample=2, noise_dim=128):
        super().__init__()
        self.img_size = img_size
        self.latent_size = img_size // 4 # 32 for 128 image
        self.init_size = self.latent_size // 4 # 8
        self.noise_dim = noise_dim
        
        # Shared Foundation (Text + Noise -> Latent Space)
        self.fc = nn.Sequential(
            nn.Linear(text_embed_dim + noise_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Replaced ConvTranspose2d with Upsample+Conv2d to eliminate checkerboard artifacts
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim, 128, 3, 1, 1)
        ) # 8 -> 16
        self.res1 = ResBlock(128)
        self.cond1 = nn.Linear(text_embed_dim, 128 * 2)

        # BG Latent Stream
        self.up2_bg = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1)
        ) # 16 -> 32
        self.res2_bg = ResBlock(64)
        self.cond2_bg = nn.Linear(text_embed_dim, 64 * 2)
        self.to_latent_bg = nn.Conv2d(64, latent_dim, 3, 1, 1)

        # FG Latent Stream
        self.up2_fg = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1)
        ) # 16 -> 32
        self.res2_fg = ResBlock(64)
        self.cond2_fg = nn.Linear(text_embed_dim, 64 * 2)
        self.latent_dim = latent_dim
        self.to_latent_fg = nn.Conv2d(64, latent_dim, 3, 1, 1)
        
        # World Model (Imagination Engine)
        # Using State Superposition: we project FG and BG to a shared dimension and sum them
        self.superposition_dim = latent_dim * 2
        self.proj_fg = nn.Conv2d(latent_dim, self.superposition_dim, 1)
        self.proj_bg = nn.Conv2d(latent_dim, self.superposition_dim, 1)
        
        self.world_model = LatentWorldModel(
            channels=self.superposition_dim, 
            text_dim=text_embed_dim, 
            num_heads=4, 
            depth=3,
            latent_size=self.latent_size,
            attn_downsample=attn_downsample
        )
        
        # Project back from superposition space to individual latents
        self.split_fg = nn.Conv2d(self.superposition_dim, latent_dim, 1)
        self.split_bg = nn.Conv2d(self.superposition_dim, latent_dim, 1)
        
        self.vae = TinyVAE(latent_dim=latent_dim)
        
    def generate_latent(self, text_embeds, z=None):
        B = text_embeds.size(0)
        if z is None:
            z = torch.randn(B, self.noise_dim, device=text_embeds.device, dtype=text_embeds.dtype)
            
        combined_input = torch.cat([text_embeds, z], dim=1)
        x = self.fc(combined_input).view(B, -1, self.init_size, self.init_size)
        
        x = self.up1(x)
        x = self.res1(x)
        cond1 = self.cond1(text_embeds).view(B, 256, 1, 1)
        x = x * (1 + cond1[:, :128]) + cond1[:, 128:]
        
        # BG
        bg = self.up2_bg(x)
        bg = self.res2_bg(bg)
        cond2_bg = self.cond2_bg(text_embeds).view(B, 128, 1, 1)
        bg = bg * (1 + cond2_bg[:, :64]) + cond2_bg[:, 64:]
        latent_bg = self.to_latent_bg(bg)
        
        # FG
        fg = self.up2_fg(x)
        fg = self.res2_fg(fg)
        cond2_fg = self.cond2_fg(text_embeds).view(B, 128, 1, 1)
        fg = fg * (1 + cond2_fg[:, :64]) + cond2_fg[:, 64:]
        latent_fg = self.to_latent_fg(fg)
        
        return latent_fg, latent_bg
        
    def generate_and_imagine(self, text_embeds, z=None):
        # 1. Base Latent Generation (Encoder equivalent)
        latent_fg, latent_bg = self.generate_latent(text_embeds, z)
        
        # 2. State Superposition: Project and Sum
        proj_fg = self.proj_fg(latent_fg)
        proj_bg = self.proj_bg(latent_bg)
        latent_superposed = proj_fg + proj_bg # B, superposition_dim, 32, 32
        
        # 3. Imagination / Transformation (World Model)
        # This is where cross-talk happens between background, foreground, and the prompt
        latent_imagined = self.world_model(latent_superposed, text_embeds, z)
        
        # 4. Split back for decoding
        latent_fg_new = self.split_fg(latent_imagined)
        latent_bg_new = self.split_bg(latent_imagined)
        
        return latent_fg_new, latent_bg_new
        
    def forward(self, text_embeds, z=None, guidance_scale=1.0, return_breakdown=False, cached_bg_latent=None):
        if z is None:
            z = torch.randn(text_embeds.size(0), self.noise_dim, device=text_embeds.device, dtype=text_embeds.dtype)
            
        latent_fg, latent_bg = self.generate_and_imagine(text_embeds, z)
        
        # Override the background latent with the epistemic cache if provided
        if cached_bg_latent is not None:
            latent_bg = cached_bg_latent
        
        # 2. CLASSIFIER-FREE GUIDANCE (CFG)
        # We apply CFG *after* imagination so it amplifies the imagined details
        if guidance_scale > 1.0:
            uncond_embeds = torch.zeros_like(text_embeds)
            uncond_fg, uncond_bg = self.generate_and_imagine(uncond_embeds, z)
            
            # Extrapolate in latent space
            latent_fg = uncond_fg + guidance_scale * (latent_fg - uncond_fg)
            if cached_bg_latent is None:
                latent_bg = uncond_bg + guidance_scale * (latent_bg - uncond_bg)
            
        decoded_fg = self.vae.decoder_fg(latent_fg)
        out_fg = decoded_fg[:, :3, :, :]
        out_mask = decoded_fg[:, 3:, :, :]
        
        out_bg = self.vae.decoder_bg(latent_bg)
        
        # Alpha Compositing
        composite = (out_fg * out_mask) + (out_bg * (1.0 - out_mask))
        
        if return_breakdown:
            return composite, out_fg, out_bg, out_mask, latent_bg
        return composite


# 3. PROGRESSIVE UPSAMPLING (Cascaded Generation)
class CascadedUpsampler(nn.Module):
    """Upscales 128x128 -> 256x256 conditioned on text embeddings.

    The feature convolutions run at the input resolution and the 2x upscale is
    deferred to a final PixelShuffle. A 3x3 kernel at 128x128 covers the same
    output area as a 3x3 at 256x256 but costs a quarter of the compute.
    """
    def __init__(self, in_size=128, out_size=256, text_embed_dim=512):
        super().__init__()
        self.entry = nn.Conv2d(3, 64, 3, 1, 1)
        self.cond = nn.Linear(text_embed_dim + 128, 64 * 2) # Added 128 for noise vector

        self.body = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.to_rgb = nn.Conv2d(32, 3 * 4, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)
        self.out_act = nn.Sigmoid()
        self._icnr_init(self.to_rgb, upscale=2)
        
        # Spatial noise scaling parameters for upsampler
        self.noise_weight_1 = nn.Parameter(torch.zeros(1, 64, 1, 1))
        self.noise_weight_2 = nn.Parameter(torch.zeros(1, 32, 1, 1))

    @staticmethod
    def _icnr_init(conv, upscale=2):
        # Every pixel of a shuffled upscale x upscale block must start from the
        # same filter, otherwise PixelShuffle emits a checkerboard before it trains.
        out_c, in_c, kh, kw = conv.weight.shape
        sub = torch.zeros(out_c // (upscale ** 2), in_c, kh, kw)
        nn.init.kaiming_normal_(sub, a=0.2)
        with torch.no_grad():
            conv.weight.copy_(sub.repeat_interleave(upscale ** 2, dim=0))
            conv.bias.zero_()

    def forward(self, img, text_embeds, z=None):
        B = img.size(0)
        x = self.entry(img)

        if z is None:
            z = torch.randn(B, 128, device=img.device, dtype=img.dtype)
            
        combined_cond = torch.cat([text_embeds, z], dim=1)
        cond = self.cond(combined_cond).view(B, 128, 1, 1)
        x = x * (1 + cond[:, :64]) + cond[:, 64:]

        # Inject spatial noise into the body
        x = self.body[0](x)
        noise1 = torch.randn(B, 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        x = x + noise1 * self.noise_weight_1
        x = self.body[1](x)
        
        x = self.body[2](x)
        noise2 = torch.randn(B, 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        x = x + noise2 * self.noise_weight_2
        x = self.body[3](x)

        out = self.out_act(self.shuffle(self.to_rgb(x)))

        # Skip connection for stability
        base_up = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
        return out * 0.5 + base_up * 0.5

class TinyDiscriminator(nn.Module):
    """
    Conditional PatchGAN Discriminator with Spectral Normalization.
    Takes a 256x256 image and text embeddings, outputs a real/fake map.
    """
    def __init__(self, img_size=256, text_embed_dim=512):
        super().__init__()
        
        # Image processing path (Spectral Norm added for stability)
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)), # 256 -> 128
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), # 128 -> 64
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Text processing path
        self.text_proj = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(text_embed_dim, 256)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Combined processing path
        self.conv4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1)), # 32 -> 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1) # 16 -> 15 (PatchGAN output)
        )

    def forward(self, img, text_embeds):
        B = img.size(0)
        
        # Process image
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x) # [B, 256, 32, 32]
        
        # Process text and expand spatially
        t = self.text_proj(text_embeds) # [B, 256]
        t = t.view(B, 256, 1, 1).expand(-1, -1, x.size(2), x.size(3)) # [B, 256, 32, 32]
        
        # Combine and classify
        combined = torch.cat([x, t], dim=1) # [B, 512, 32, 32]
        out = self.conv4(combined)
        return out

def load_t2i_state_dict(model, state_dict):
    """Load TinyT2I weights, resizing the world model position embedding if the
    attention resolution differs from the one the checkpoint was trained at.
    Returns the load result from torch.nn.Module.load_state_dict."""
    key = 'world_model.pos_embed'
    if key in state_dict:
        saved = state_dict[key]
        target = model.world_model.pos_embed
        if saved.shape != target.shape:
            old_side = int(round((saved.size(1) - 1) ** 0.5))
            new_side = model.world_model.attn_size
            C = saved.size(2)
            text_tok = saved[:, :1, :]
            grid = saved[:, 1:, :].reshape(1, old_side, old_side, C).permute(0, 3, 1, 2)
            grid = F.interpolate(grid, size=(new_side, new_side), mode='bicubic', align_corners=False)
            grid = grid.permute(0, 2, 3, 1).reshape(1, new_side * new_side, C)
            state_dict = dict(state_dict)
            state_dict[key] = torch.cat([text_tok, grid], dim=1)
            print(f"Resized {key} from {old_side}x{old_side} to {new_side}x{new_side} grid.")
            
    # Handle the transition from deterministic (Text -> Latent) to stochastic (Text + Noise -> Latent)
    fc_key = 'fc.0.weight'
    if fc_key in state_dict:
        saved_fc = state_dict[fc_key]
        target_fc = model.fc[0].weight
        if saved_fc.shape != target_fc.shape:
            # We added noise_dim (128) to the input of the first linear layer
            print(f"Adapting {fc_key} to accept noise vector (added {model.noise_dim} input channels, zeroed for backward compatibility)")
            new_fc = torch.zeros_like(target_fc)
            # Copy the old weights for the text embedding part
            new_fc[:, :saved_fc.shape[1]] = saved_fc
            # Initialize the new noise weights with zeros so old checkpoints behave deterministically
            nn.init.zeros_(new_fc[:, saved_fc.shape[1]:])
            state_dict = dict(state_dict)
            state_dict[fc_key] = new_fc
            
    # Handle the transition to State Superposition
    proj_fg_key = 'proj_fg.weight'
    if proj_fg_key not in state_dict and hasattr(model, 'proj_fg'):
        print(f"Initializing new State Superposition layers")
        state_dict = dict(state_dict)
        # Initialize projections as identity mappings to the first/second half of the channels
        # to mimic the old concatenation behavior initially
        
        # proj_fg: map latent_dim to first half of superposition_dim
        nn.init.dirac_(model.proj_fg.weight[:model.latent_dim, :, :, :])
        nn.init.zeros_(model.proj_fg.weight[model.latent_dim:, :, :, :])
        nn.init.zeros_(model.proj_fg.bias)
        state_dict['proj_fg.weight'] = model.proj_fg.weight
        state_dict['proj_fg.bias'] = model.proj_fg.bias
        
        # proj_bg: map latent_dim to second half of superposition_dim
        nn.init.zeros_(model.proj_bg.weight[:model.latent_dim, :, :, :])
        nn.init.dirac_(model.proj_bg.weight[model.latent_dim:, :, :, :])
        nn.init.zeros_(model.proj_bg.bias)
        state_dict['proj_bg.weight'] = model.proj_bg.weight
        state_dict['proj_bg.bias'] = model.proj_bg.bias
        
        # split_fg: extract from first half
        nn.init.dirac_(model.split_fg.weight[:, :model.latent_dim, :, :])
        nn.init.zeros_(model.split_fg.weight[:, model.latent_dim:, :, :])
        nn.init.zeros_(model.split_fg.bias)
        state_dict['split_fg.weight'] = model.split_fg.weight
        state_dict['split_fg.bias'] = model.split_fg.bias
        
        # split_bg: extract from second half
        nn.init.zeros_(model.split_bg.weight[:, :model.latent_dim, :, :])
        nn.init.dirac_(model.split_bg.weight[:, model.latent_dim:, :, :])
        nn.init.zeros_(model.split_bg.bias)
        state_dict['split_bg.weight'] = model.split_bg.weight
        state_dict['split_bg.bias'] = model.split_bg.bias
            
    # Handle the transition for the world model to accept noise
    noise_proj_key = 'world_model.noise_proj.weight'
    if noise_proj_key not in state_dict and hasattr(model.world_model, 'noise_proj'):
        print(f"Initializing new noise projection layer for world model (zeroed for backward compatibility)")
        state_dict = dict(state_dict)
        nn.init.zeros_(model.world_model.noise_proj.weight)
        nn.init.zeros_(model.world_model.noise_proj.bias)
        state_dict[noise_proj_key] = model.world_model.noise_proj.weight
        state_dict['world_model.noise_proj.bias'] = model.world_model.noise_proj.bias
        
    # Handle Spatial Noise weights initialization
    for key in ['res1.noise_weight', 'res2_bg.noise_weight', 'res2_fg.noise_weight']:
        if key not in state_dict and hasattr(model, key.split('.')[0]):
            print(f"Initializing new spatial noise weight for {key}")
            state_dict = dict(state_dict)
            module = getattr(model, key.split('.')[0])
            state_dict[key] = module.noise_weight

    return model.load_state_dict(state_dict, strict=False)


def load_upsampler_state_dict(model, state_dict):
    """Load CascadedUpsampler weights, adapting them for the added noise vector."""
    cond_key = 'cond.weight'
    if cond_key in state_dict:
        saved_cond = state_dict[cond_key]
        target_cond = model.cond.weight
        if saved_cond.shape != target_cond.shape:
            # We added 128 to the input of the cond layer
            print(f"Adapting upsampler {cond_key} to accept noise vector (zeroed for backward compatibility)")
            new_cond = torch.zeros_like(target_cond)
            # Copy the old weights for the text embedding part
            new_cond[:, :saved_cond.shape[1]] = saved_cond
            # Initialize the new noise weights with zeros
            nn.init.zeros_(new_cond[:, saved_cond.shape[1]:])
            state_dict = dict(state_dict)
            state_dict[cond_key] = new_cond
            
    # Handle Spatial Noise weights initialization for upsampler
    for key in ['noise_weight_1', 'noise_weight_2']:
        if key not in state_dict and hasattr(model, key):
            print(f"Initializing new spatial noise weight for upsampler {key}")
            state_dict = dict(state_dict)
            state_dict[key] = getattr(model, key)
            
    return model.load_state_dict(state_dict, strict=False)

if __name__ == "__main__":
    # Test the model
    model = TinyT2I()
    dummy_text_embeds = torch.randn(2, 512)
    
    # Test CFG
    output = model(dummy_text_embeds, guidance_scale=3.0)
    print(f"TinyT2I CFG output shape: {output.shape}") 
    
    # Test Upsampler
    upsampler = CascadedUpsampler()
    up_out = upsampler(output, dummy_text_embeds)
    print(f"Upsampler output shape: {up_out.shape}")
