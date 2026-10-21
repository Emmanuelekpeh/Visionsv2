import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
import math

class PatchEmbedding(nn.Module):
    """CNN encoder to extract patch features from an image."""
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Simple CNN to extract features from patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class PatchDecoder(nn.Module):
    """CNN decoder to map patch features back to pixels."""
    def __init__(self, img_size=512, patch_size=16, embed_dim=256, out_chans=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim // 4, out_chans, kernel_size=patch_size // 4, stride=patch_size // 4)
        )

    def forward(self, x):
        # x: (B, num_patches, embed_dim)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, self.grid_size, self.grid_size)
        x = self.decode(x)
        return torch.sigmoid(x) # Output in [0, 1]

class HybridMAE(nn.Module):
    """
    Masked Autoencoder with Autoregressive capabilities.
    Learns to reconstruct masked patches, allowing for combination of images.
    """
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=256, depth=6, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = PatchDecoder(img_size, patch_size, embed_dim, in_chans)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the boolean mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_kept, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer
        x = self.transformer(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x_ = x_ + self.pos_embed[:, 1:, :]

        # decode to pixels
        x_decoded = self.decoder(x_)
        return x_decoded

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return pred, mask
        
    def generate_from_blank(self):
        """
        Generate an image from a blank canvas.
        Uses the learned mask tokens and positional embeddings to autoregressively
        or simultaneously predict the entire image.
        """
        self.eval()
        with torch.no_grad():
            # Batch size 1, sequence length = num_patches
            N = 1
            L = self.patch_embed.num_patches
            
            # Start with all mask tokens (a blank canvas)
            mask_tokens = self.mask_token.repeat(N, L, 1)
            
            # Add positional embeddings so the model knows *where* each blank patch is
            x_ = mask_tokens + self.pos_embed[:, 1:, :]
            
            # Add cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(N, -1, -1)
            x = torch.cat((cls_tokens, x_), dim=1)
            
            # Run through transformer
            x = self.transformer(x)
            x = self.norm(x)
            
            # Decode back to pixels
            pred = self.decoder(x[:, 1:, :])
            return pred
            
    def recombine(self, img_a, img_b, mask_ratio=0.5):
        """
        Recombine two images. 
        Takes patches from img_a, masks out some, and asks the model to fill them in,
        optionally conditioned on img_b (simplified here by just blending in pixel space for the masked regions).
        """
        self.eval()
        with torch.no_grad():
            # Embed both
            patches_a = self.patch_embed(img_a)
            patches_b = self.patch_embed(img_b)
            
            # Add pos embed
            patches_a = patches_a + self.pos_embed[:, 1:, :]
            patches_b = patches_b + self.pos_embed[:, 1:, :]
            
            # Create a spatial mask (e.g., top half A, bottom half B, or random)
            N, L, D = patches_a.shape
            
            # Simple random spatial mask for recombination
            noise = torch.rand(N, L, device=img_a.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
            len_keep = int(L * (1 - mask_ratio))
            ids_keep_a = ids_shuffle[:, :len_keep]
            
            # Keep patches from A
            kept_a = torch.gather(patches_a, dim=1, index=ids_keep_a.unsqueeze(-1).repeat(1, 1, D))
            
            # We want to fill the rest. In a true conditional MAE, we'd attend to B.
            # For this hybrid, we can initialize the mask tokens with patches from B!
            mask_tokens = self.mask_token.repeat(N, L - len_keep, 1)
            
            # Mix B into the mask tokens to condition the generation
            ids_masked = ids_shuffle[:, len_keep:]
            masked_b = torch.gather(patches_b, dim=1, index=ids_masked.unsqueeze(-1).repeat(1, 1, D))
            
            # Blend B into mask tokens (alpha blending)
            mixed_mask_tokens = mask_tokens * 0.5 + masked_b * 0.5
            
            # Combine A and mixed B
            x_ = torch.cat([kept_a, mixed_mask_tokens], dim=1)
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D)) # unshuffle
            
            # Add cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(N, -1, -1)
            x = torch.cat((cls_tokens, x_), dim=1)
            
            # Run through transformer
            x = self.transformer(x)
            x = self.norm(x)
            
            # Decode
            pred = self.decoder(x[:, 1:, :])
            return pred

def load_image(path, size=512):
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

def save_image(tensor, path):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)

# --- Stylistic Filters ---

def apply_posterize(img: Image.Image, bits: int = 3) -> Image.Image:
    """Reduces the number of bits per color channel, creating a comic-book/retro feel."""
    return ImageOps.posterize(img, bits)

def apply_dither(img: Image.Image, colors: int = 16) -> Image.Image:
    """Applies Floyd-Steinberg dithering for a retro 8-bit/stippled look."""
    # Convert to palette mode with dithering, then back to RGB
    return img.convert('P', dither=Image.FLOYDSTEINBERG, palette=Image.ADAPTIVE, colors=colors).convert('RGB')

def apply_halftone(img: Image.Image, sample: int = 6) -> Image.Image:
    """Applies a classic newspaper B&W halftone dot pattern."""
    img_gray = img.convert('L')
    arr = np.array(img_gray)
    h, w = arr.shape
    
    # Create a grid
    y, x = np.mgrid[0:h, 0:w]
    
    # Modulo to get local cell coordinates
    cx = (x % sample) - sample / 2.0
    cy = (y % sample) - sample / 2.0
    
    # Distance from center of the local cell
    dist = np.sqrt(cx**2 + cy**2)
    
    # Max distance in a cell
    max_dist = np.sqrt(2 * (sample / 2.0)**2)
    
    # Normalize pixel intensity to radius (darker = larger dot)
    radius = (1.0 - arr / 255.0) * max_dist
    
    # Create halftone mask (True where distance > radius, meaning background/white)
    mask = dist > radius
    
    # Apply to image (White background, Black dots)
    halftoned = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(halftoned).convert('RGB')

