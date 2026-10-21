import torch
from PIL import Image
import numpy as np

# transformers >= recent requires torch >= 2.6 for torch.load; patch on older CPU installs
if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
    import transformers.utils.import_utils as _iu
    import transformers.modeling_utils as _mu
    _iu.check_torch_load_is_safe = lambda: None
    _mu.check_torch_load_is_safe = lambda: None

from transformers import CLIPProcessor, CLIPModel
from diffusers import AutoencoderTiny
from training_scheduler import run_on_pytorch_thread

class VisionSystem:
    def __init__(self, device="cpu", load_vae=True):
        self.device = device
        print("Loading Vision System (CPU-aware)...")
        
        # Check for local models first
        import os
        clip_path = "models/clip-vit-base-patch32" if os.path.exists("models/clip-vit-base-patch32") else "openai/clip-vit-base-patch32"
        taesd_path = "models/taesd" if os.path.exists("models/taesd") else "madebyollin/taesd"
        
        # Concept Extractor (CLIP)
        self.clip_model = CLIPModel.from_pretrained(clip_path).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
        
        # Monkey-patch for transformers >= 5.0 where get_text_features returns an object instead of a tensor
        original_get_text_features = self.clip_model.get_text_features
        def get_text_features_wrapper(*args, **kwargs):
            out = original_get_text_features(*args, **kwargs)
            return out if isinstance(out, torch.Tensor) else out.pooler_output if hasattr(out, 'pooler_output') else out[1]
        self.clip_model.get_text_features = get_text_features_wrapper

        original_get_image_features = self.clip_model.get_image_features
        def get_image_features_wrapper(*args, **kwargs):
            out = original_get_image_features(*args, **kwargs)
            return out if isinstance(out, torch.Tensor) else out.pooler_output if hasattr(out, 'pooler_output') else out[1]
        self.clip_model.get_image_features = get_image_features_wrapper
        
        # Semantic Vocabulary for Zero-Shot Classification
        self.vocabulary = [
            "animal", "plant", "building", "landscape", "person", "vehicle", "food", 
            "fractal", "crystal", "fluid", "fire", "smoke", "machine", "robot", "cloud",
            "mountain", "ocean", "forest", "city", "space", "geometric", "abstract",
            "organic", "synthetic", "metallic", "wooden", "glass", "glowing", "dark",
            "bright", "colorful", "monochrome", "pattern", "texture", "smooth", "rough",
            "symetrical", "chaotic", "peaceful", "scary", "alien", "familiar", "ancient",
            "futuristic", "microscopic", "macroscopic", "liquid", "solid", "gaseous"
        ]
        print(f"Loaded semantic vocabulary with {len(self.vocabulary)} terms.")
        
        # Pre-compute text features (init runs on main thread before evolve starts)
        with torch.no_grad():
            text_inputs = self.clip_processor(text=self.vocabulary, return_tensors="pt", padding=True).to(self.device)
            self.text_features = self.clip_model.get_text_features(**text_inputs)
            self.text_features /= self.text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Visual Encoder/Decoder is not needed by the text-only DirectCNN trainer.
        self.vae = AutoencoderTiny.from_pretrained(taesd_path).to(self.device) if load_vae else None
        
        # Freeze models
        self.clip_model.eval()
        if self.vae is not None:
            self.vae.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        if self.vae is not None:
            for param in self.vae.parameters():
                param.requires_grad = False

    def encode_image_to_latent(self, image: Image.Image) -> torch.Tensor:
        """Converts an image to a VAE latent vector."""
        return run_on_pytorch_thread(self._encode_image_to_latent_impl, image)

    def _encode_image_to_latent_impl(self, image: Image.Image) -> torch.Tensor:
        img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_tensor = img_tensor * 2.0 - 1.0
        with torch.no_grad():
            output = self.vae.encode(img_tensor)
            return output.latents

    def decode_latent_to_image(self, latent: torch.Tensor) -> Image.Image:
        """Converts a VAE latent vector back to an image."""
        return run_on_pytorch_thread(self._decode_latent_to_image_impl, latent)

    def roundtrip_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """VAE decode → encode on a batch of latents. Returns tensor matching input batch size."""
        return run_on_pytorch_thread(self._roundtrip_latent_impl, latent)

    def _roundtrip_latent_impl(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)
        latent = torch.clamp(latent, min=-10.0, max=10.0)
        with torch.no_grad():
            decoded = self.vae.decode(latent).sample
            return self.vae.encode(decoded).latents

    def _decode_latent_to_image_impl(self, latent: torch.Tensor) -> Image.Image:
        try:
            latent = torch.clamp(latent, min=-10.0, max=10.0)
            with torch.no_grad():
                output = self.vae.decode(latent).sample
            output = (output / 2 + 0.5).clamp(0, 1)
            output = output.cpu().permute(0, 2, 3, 1).numpy()[0]
            output = (output * 255).astype(np.uint8)
            return Image.fromarray(output)
        except Exception as e:
            print(f"[Vision] CRITICAL DECODE ERROR: {e}")
            return Image.new('RGB', (512, 512), color='black')

    def extract_concept(self, image: Image.Image) -> torch.Tensor:
        """Extracts CLIP embedding for Reality Anchor and Concept Graph."""
        return run_on_pytorch_thread(self._extract_concept_impl, image)

    def _extract_concept_impl(self, image: Image.Image) -> torch.Tensor:
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
        return embedding

    def extract_concept_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decodes latent and extracts its concept embedding."""
        return run_on_pytorch_thread(self._extract_concept_from_latent_impl, latent)

    def _extract_concept_from_latent_impl(self, latent: torch.Tensor) -> torch.Tensor:
        img = self._decode_latent_to_image_impl(latent)
        return self._extract_concept_impl(img)

    def generate_tags(self, image_embedding: torch.Tensor, top_k: int = 3) -> list:
        """Zero-shot classify the image embedding against the vocabulary."""
        return run_on_pytorch_thread(self._generate_tags_impl, image_embedding, top_k)

    def _generate_tags_impl(self, image_embedding: torch.Tensor, top_k: int = 3) -> list:
        with torch.no_grad():
            if image_embedding.dim() == 1:
                image_embedding = image_embedding.unsqueeze(0)
            similarity = (100.0 * image_embedding @ self.text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(top_k)
        return [self.vocabulary[idx] for idx in indices]

    def _ensure_aesthetic_text_features(self):
        """Lazy-init CLIP text banks for multi-facet aesthetic scoring."""
        if hasattr(self, "_aesthetic_ready") and self._aesthetic_ready:
            return

        self.aesthetic_prompts = {
            "striking": [
                "award winning photograph",
                "dramatic lighting masterpiece",
                "iconic powerful visual image",
            ],
            "harmonious": [
                "balanced elegant composition",
                "serene harmonious scenery",
                "tasteful color harmony",
            ],
            "vivid": [
                "sharp detailed high resolution photo",
                "rich vivid colors and texture",
                "crisp clear professional photo",
            ],
            "composition": [
                "rule of thirds composition",
                "well framed artistic photograph",
                "intentional cinematic framing",
            ],
            "overall": [
                "a beautiful high quality photograph",
                "aesthetically pleasing masterpiece",
                "stunning visually appealing image",
            ],
            "anti": [
                "blurry out of focus photo",
                "heavy jpeg compression artifacts",
                "noisy grainy low quality mess",
                "overexposed washed out garbage image",
            ],
        }

        self._aesthetic_features = {}
        with torch.no_grad():
            for facet, prompts in self.aesthetic_prompts.items():
                inputs = self.clip_processor(
                    text=prompts, return_tensors="pt", padding=True
                ).to(self.device)
                feats = self.clip_model.get_text_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                self._aesthetic_features[facet] = feats.mean(dim=0, keepdim=True)
                self._aesthetic_features[facet] = (
                    self._aesthetic_features[facet]
                    / self._aesthetic_features[facet].norm(p=2, dim=-1, keepdim=True)
                )
        self._aesthetic_ready = True

    def score_aesthetics(self, image_embedding: torch.Tensor) -> dict:
        """Multi-facet CLIP aesthetic scores for Beauty and sibling heads."""
        return run_on_pytorch_thread(self._score_aesthetics_impl, image_embedding)

    def _score_aesthetics_impl(self, image_embedding: torch.Tensor) -> dict:
        self._ensure_aesthetic_text_features()
        emb = image_embedding
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        emb = emb.to(self.device)
        emb = emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        scores = {}
        with torch.no_grad():
            anti_sim = float((emb @ self._aesthetic_features["anti"].T).squeeze().item())
            facet_sims = {}
            for facet in ("striking", "harmonious", "vivid", "composition", "overall"):
                facet_sims[facet] = float(
                    (emb @ self._aesthetic_features[facet].T).squeeze().item()
                )

            mean_pos = float(np.mean(list(facet_sims.values())))
            scores["anti"] = float(1.0 / (1.0 + np.exp(-12.0 * (anti_sim - mean_pos))))

            for facet, pos_sim in facet_sims.items():
                contrast = pos_sim - anti_sim
                scores[facet] = float(1.0 / (1.0 + np.exp(-12.0 * contrast)))

            scores["overall"] = float(
                max(0.0, min(1.0, scores["overall"] * (1.0 - 0.35 * scores["anti"])))
            )
            scores["clip_confidence"] = float(
                max(0.0, min(1.0, abs(mean_pos - anti_sim) / 0.08))
            )
        return scores

    def tag_confidence_peak(self, image_embedding: torch.Tensor) -> float:
        """How peaked the semantic tag distribution is (clear concept vs mush)."""
        return run_on_pytorch_thread(self._tag_confidence_peak_impl, image_embedding)

    def _tag_confidence_peak_impl(self, image_embedding: torch.Tensor) -> float:
        emb = image_embedding
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        with torch.no_grad():
            emb = emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            similarity = (100.0 * emb @ self.text_features.T).softmax(dim=-1)[0]
            top = similarity.topk(3).values
            peak = float(top[0] / (top.sum() + 1e-8))
        return float(max(0.0, min(1.0, peak)))
