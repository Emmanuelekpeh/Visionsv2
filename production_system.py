import os
import json
import asyncio
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional

import ollama
import edge_tts
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip

from vision import VisionSystem
# Assuming we use TinyT2I or similar for generation
from alternative_t2i import DirectCNNT2I

# The locked animation contract (see ANIMATION_SPEC.md). Timing comes from an
# exposure sheet, never from dividing a duration by a segment count.
from animation_spec import (
    AutoregressiveChainError,
    DrawingSource,
    ExposureSheet,
    SpecViolation,
    sheet_from_keyframes,
)

# =========================================================================
# Shared helpers for img2img correctness fixes
# =========================================================================

def safe_img2img_steps(num_inference_steps: int, strength: float, min_effective_steps: int = 8) -> int:
    """
    diffusers img2img pipelines only actually run int(num_inference_steps * strength)
    denoising steps. Below ~8 effective steps, small SD models reliably produce
    oversaturated / contrast-blown "inkblot" output regardless of prompt.
    """
    if strength <= 0:
        return num_inference_steps
    effective = int(num_inference_steps * strength)
    if effective >= min_effective_steps:
        return num_inference_steps
    return int(np.ceil(min_effective_steps / strength))

def match_histogram_to_reference(image: Image.Image, reference: Image.Image) -> Image.Image:
    """
    Fixes VAE color drift by forcing the output's color distribution back to match
    a fixed reference using per-channel histogram matching in LAB space.
    """
    import cv2
    import numpy as np
    
    src = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2LAB)
    ref = cv2.cvtColor(np.array(reference.convert("RGB")), cv2.COLOR_RGB2LAB)

    matched = np.zeros_like(src)
    for c in range(3):
        src_channel = src[:, :, c]
        ref_channel = ref[:, :, c]

        src_hist, _ = np.histogram(src_channel.flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(ref_channel.flatten(), 256, [0, 256])

        src_cdf = np.cumsum(src_hist).astype(np.float64)
        src_cdf /= src_cdf[-1] if src_cdf[-1] > 0 else 1
        ref_cdf = np.cumsum(ref_hist).astype(np.float64)
        ref_cdf /= ref_cdf[-1] if ref_cdf[-1] > 0 else 1

        lookup = np.zeros(256, dtype=np.uint8)
        ref_idx = 0
        for src_idx in range(256):
            while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_idx]:
                ref_idx += 1
            lookup[src_idx] = ref_idx

        matched[:, :, c] = lookup[src_channel]

    matched_rgb = cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(matched_rgb)

class Director:
    """Uses Ollama to generate scripts and shot plans. Falls back to simple parsing if Ollama is not available."""
    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.model_name = model_name
        self.ollama_available = True
        try:
            # Quick check if Ollama is running
            ollama.list()
        except Exception:
            print("Warning: Ollama is not running or not accessible. Using fallback text generation.")
            self.ollama_available = False
            
        self.aesthetic_prompt = """VISUAL & CREATIVE IDENTITY: "The Uncanny Documentary"
The visual language is observational, surreal, tactile and slightly uncanny.
Images should feel discovered rather than manufactured.
Treat strange subjects with the visual seriousness normally given to ordinary reality.

Favor imperfect, physical imagery over pristine digital perfection. Use soft detail, natural texture, filmic grain, subtle optical imperfections, restrained color and imperfect exposure when appropriate.
Composition should sometimes be awkward, asymmetrical, partially obstructed or unexpectedly framed. Do not force every image into conventional cinematic composition.
Use darkness, distance, shadows, reflections, environmental obstruction and partial visibility to preserve uncertainty. Do not reveal or explain strange subjects immediately.

Prefer environments that feel inhabited and physically real: ordinary rooms, streets, forests, oceans, laboratories, industrial spaces, homes and mundane landscapes containing something subtly impossible.
The strange element should coexist with normal reality rather than transforming the entire world into spectacle.

Favor human-scale observation, quiet reactions, hesitation, stillness and moments where nothing happens.
Use visual contradictions: beautiful and ugly, cute and disturbing, mundane and alien, organic and mechanical, enormous and tiny, funny and unsettling.
Color should generally be restrained rather than hyper-saturated, with unusual colors appearing selectively as visual events.

Avoid generic cinematic spectacle, glossy AI imagery, excessive sharpness, perfect skin, heroic compositions, constant camera movement, generic horror imagery, fantasy excess, gratuitous VHS effects and artificial "cinematic" filters.

The image should create curiosity before providing explanation. The audience should occasionally wonder: "What exactly am I looking at?"
CRITICAL: Do not make an image strange merely by adding strange objects. Make the relationship between ordinary things strange. Observe strangeness as a documentary would."""

    def generate_script(self, idea: str) -> List[Dict[str, str]]:
        """Generates a script with dialogue/narration lines."""
        if not self.ollama_available:
            return [{"speaker": "Narrator", "text": idea}]
            
        prompt = f"""{self.aesthetic_prompt}

You are a video producer following the aesthetic above. Based on the following idea, write a short, grounded script. Focus heavily on observation and restraint. Let silence and holds do the heavy lifting.
Output ONLY a JSON array of objects, where each object has a 'speaker' and 'text'.
No markdown, no explanation.
Idea: {idea}
        """
        try:
            response = ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}])
            content = response['message']['content']
            # Find JSON array
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end != 0:
                return json.loads(content[start:end])
            return [{"speaker": "Narrator", "text": content}]
        except Exception as e:
            print(f"Error parsing script: {e}")
            return [{"speaker": "Narrator", "text": idea}]

    def plan_shots(self, script_with_timings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generates a shot plan based on the script and audio timings."""
        if not self.ollama_available:
            return [{
                "duration": seg["duration"],
                "subject": "scene",
                "location": "unknown",
                "camera": "static",
                "lighting": "natural",
                "action_sequence": [seg["text"], seg["text"], seg["text"]],
                "audio_file": seg["audio_file"]
            } for seg in script_with_timings]
            
        shots_prompt = f"{self.aesthetic_prompt}\n\nCreate a shot plan for the following script segments and their durations.\n"
        for i, segment in enumerate(script_with_timings):
            shots_prompt += f"Segment {i+1} ({segment['duration']:.1f}s): {segment['text']}\n"
        
        shots_prompt += """
        For each segment, define the visual shot applying the 'uncanny documentary' aesthetic heavily to the camera angles, framing, lighting, and action details.
        Output ONLY a JSON array of objects, one for each segment.
        Each object must have:
        - 'duration': (number) same as the segment duration
        - 'subject': (string) main subject
        - 'location': (string) setting
        - 'camera': (string) camera angle/movement (include awkward framing or observational holds)
        - 'lighting': (string) lighting style (restrained, imperfect)
        - 'action_sequence': (array of 3-5 strings) Break the shot down into a sequence of keyframes. For EACH keyframe, write a full image generation prompt describing the exact physical state of the scene at that moment. The sequence must show logical movement, action, and reaction over time. CRITICAL: Each prompt MUST end with a consistent global visual fingerprint (e.g., 'filmic grain, soft focus') to maintain visual continuity.
        No markdown, no explanation.
        """
        
        response = ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': shots_prompt}])
        try:
            content = response['message']['content']
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end != 0:
                shots = json.loads(content[start:end])
                # Ensure durations match
                for i, shot in enumerate(shots):
                    if i < len(script_with_timings):
                        shot['duration'] = script_with_timings[i]['duration']
                        shot['audio_file'] = script_with_timings[i]['audio_file']
                return shots
            raise ValueError("No JSON array found")
        except Exception as e:
            print(f"Error parsing shot plan: {e}")
            # Fallback
            return [{
                "duration": seg["duration"],
                "subject": "scene",
                "location": "unknown",
                "camera": "static",
                "lighting": "natural",
                "action_sequence": [seg["text"], seg["text"], seg["text"]],
                "audio_file": seg["audio_file"]
            } for seg in script_with_timings]


class VoiceModule:
    """Generates TTS audio and calculates durations. Falls back to dummy audio if edge-tts fails."""
    def __init__(self, output_dir: str = "output/audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def _generate_audio(self, text: str, voice: str, output_path: str, rate: str = "+15%"):
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(output_path)
        
    def _create_dummy_audio(self, text: str, output_path: str, duration: float = 3.0):
        """Creates a silent dummy audio file for testing."""
        from moviepy.editor import AudioClip
        import numpy as np
        
        # Create a silent audio clip
        make_frame = lambda t: np.zeros((len(t), 2)) if isinstance(t, np.ndarray) else np.array([0, 0])
        clip = AudioClip(make_frame, duration=duration)
        clip.write_audiofile(output_path, fps=44100, logger=None)

    def generate_audio_for_script(self, script: List[Dict[str, str]], default_voice: str = "en-US-GuyNeural", rate: str = "+15%") -> List[Dict[str, Any]]:
        """Generates audio files and appends duration to the script."""
        script_with_timings = []
        
        for i, line in enumerate(script):
            text = line['text']
            # Simple voice mapping
            voice = "en-US-AriaNeural" if line.get('speaker', '').lower() == 'female' else default_voice
            
            import time
            ts = int(time.time() * 1000)
            output_file = self.output_dir / f"line_{i:03d}_{ts}.mp3"
            
            # Run edge-tts
            try:
                asyncio.run(self._generate_audio(text, voice, str(output_file), rate=rate))
            except Exception as e:
                print(f"Warning: edge-tts failed ({e}). Using dummy audio.")
                # Estimate duration based on word count (roughly 3 words per second)
                est_duration = max(2.0, len(text.split()) / 3.0)
                self._create_dummy_audio(text, str(output_file), duration=est_duration)
            
            # Get duration using moviepy
            audio_clip = AudioFileClip(str(output_file))
            duration = audio_clip.duration
            audio_clip.close()
            
            segment = line.copy()
            segment['audio_file'] = str(output_file)
            segment['duration'] = duration
            script_with_timings.append(segment)
            
        return script_with_timings


from diffusers import StableDiffusionPipeline, AutoencoderTiny
from diffusers import EulerDiscreteScheduler

class AMusedGenerator:
    """Wraps the aMUSEd model, which uses a completely different architecture (MUSE / Masked Image Modeling) instead of Diffusion."""
    def __init__(self, device="cpu"):
        self.device = device
        print("Loading Teacher Model (aMUSEd - Masked Image Modeling)...")
        from diffusers import AmusedPipeline, AmusedImg2ImgPipeline
        self.pipe = AmusedPipeline.from_pretrained(
            "amused/amused-256", 
            variant="fp16", 
            torch_dtype=torch.float32
        ).to(device)
        
        self.img2img_pipe = AmusedImg2ImgPipeline(
            vqvae=self.pipe.vqvae,
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder,
            transformer=self.pipe.transformer,
            scheduler=self.pipe.scheduler,
        ).to(device)
        
    def generate_keyframe(self, prompt: str, num_inference_steps: int = 12, guidance_scale: float = 7.0) -> Image.Image:
        print(f"Teacher Model (aMUSEd) generating: {prompt} (Steps: {num_inference_steps})")
        import time
        with torch.no_grad():
            # aMUSEd uses discrete tokens and a transformer
            image = self.pipe(
                prompt,
                generator=torch.Generator(self.device).manual_seed(int(time.time())),
            ).images[0]
            return image

class MobileDiffusionGenerator:
    """Wraps a fast CPU-friendly diffusion model (like tiny-sd) for high-quality keyframe generation."""
    def __init__(self, device="cpu", model_id="models/tiny-sd", is_1step=False):
        self.device = device
        self.is_1step = is_1step
        print(f"Loading Teacher Diffusion Model from {model_id}...")
        
        # Load standard Text-to-Image Pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float32,
            safety_checker=None
        ).to(device)
        
        # Load Image-to-Image Pipeline using the exact same components
        from diffusers import StableDiffusionImg2ImgPipeline
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
            unet=self.pipe.unet,
            scheduler=self.pipe.scheduler,
            safety_checker=None,
            feature_extractor=None
        ).to(device)

        # ---------------------------------------------------------------
        # FIXED: Switch to deterministic Euler scheduler for both txt2img
        # and img2img paths. Previously, img2img silently kept the default 
        # scheduler (which was buggy for low-strength passes), and txt2img 
        # used Ancestral which constantly injects fresh noise.
        # ---------------------------------------------------------------
        if not self.is_1step:
            from diffusers import EulerDiscreteScheduler
            deterministic_scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.scheduler = deterministic_scheduler
            self.img2img_pipe.scheduler = deterministic_scheduler
        
    def generate_keyframe(self, prompt: str, num_inference_steps: int = 15, guidance_scale: float = 7.0) -> Image.Image:
        if self.is_1step:
            print(f"Teacher Model (MobileDiffusion) generating: {prompt}")
            with torch.no_grad():
                # The downloaded weights are not the 1-step GAN weights. They are the 4-to-8 
                # step distilled weights. We use Euler A for coherent generation.
                from diffusers import EulerAncestralDiscreteScheduler
                self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
                
                image = self.pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=7.0, 
                ).images[0]
                return image
        else:
            print(f"Teacher Model generating: {prompt} (Steps: {num_inference_steps})")
            with torch.no_grad():
                image = self.pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale, 
                ).images[0]
                return image

class VisualGenerator:
    """Wraps the T2I models. Uses a Teacher as core, TinyT2I as support."""
    def __init__(self, device="cpu", teacher_id="tiny-sd"):
        self.device = device
        
        # Core Generator (Teacher)
        try:
            if teacher_id == "bk-sdm-tiny":
                self.core_model = MobileDiffusionGenerator(device=device, model_id="nota-ai/bk-sdm-tiny", is_1step=False)
            elif teacher_id == "amused-256":
                # aMUSEd uses a completely different architecture (Masked Image Modeling)
                self.core_model = AMusedGenerator(device=device)
            else:
                self.core_model = MobileDiffusionGenerator(device=device, model_id="models/tiny-sd", is_1step=False)
        except Exception as e:
            print(f"Warning: Could not load Teacher ({e}). Falling back to Student only.")
            self.core_model = None
            
        # Support Generator (Direct CNN)
        self.support_model = DirectCNNT2I().to(device)
        self.vision = VisionSystem(device=device, load_vae=False)
        
        # Try to load latest checkpoint for Direct CNN
        ckpt_path = self._find_latest_checkpoint()
        if ckpt_path:
            print(f"Loading Direct CNN checkpoint: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.support_model.load_state_dict(state_dict, strict=False)
        else:
            print("WARNING: No Direct CNN checkpoint found (*.pt). The Student model is UNTRAINED and will output pure noise until trained.")
            
        self.support_model.eval()

    def _find_latest_checkpoint(self) -> Optional[Path]:
        # First check for the specific kaggle checkpoint
        specific_ckpt = Path("direct_cnn_t2i_kaggle_ep830.pt")
        if specific_ckpt.exists():
            return specific_ckpt
            
        checkpoints = list(Path(".").glob("direct_cnn_t2i*.pt"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)

    def generate_keyframe(self, prompt: str, num_inference_steps: int = 15) -> Image.Image:
        """Generates an image from a prompt using the core model."""
        if self.core_model is not None:
            # Generate with Teacher
            img = self.core_model.generate_keyframe(prompt, num_inference_steps=num_inference_steps)
            
            # (Optional) We can run TinyT2I here in the background to learn/compare
            # self.generate_support_keyframe(prompt)
            
            return img
        else:
            return self.generate_support_keyframe(prompt)
            
    def generate_support_keyframe(self, prompt: str) -> Image.Image:
        """Generates an image using the support model (Direct CNN)."""
        print(f"Direct CNN generating: {prompt}")
        with torch.no_grad():
            # Get text embeddings using VisionSystem's CLIP
            inputs = self.vision.clip_processor(text=[prompt], return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_embeds = self.vision.clip_model.get_text_features(**inputs)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            
            # Generate image
            output = self.support_model(text_embeds, guidance_scale=3.0)
            
            # Convert to PIL
            img_tensor = output.squeeze(0).cpu().clamp(0, 1)
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(img_np)


class AutoGate:
    """Filters bad generations before human review."""
    def __init__(self, vision_system: VisionSystem):
        self.vision = vision_system
        self.reference_embedding = None

    def set_reference(self, image: Image.Image):
        """Sets the character/style reference embedding."""
        self.reference_embedding = self.vision.extract_concept(image)

    def check_quality(self, image: Image.Image, threshold: float = 0.6) -> bool:
        """Checks if the image matches the reference style/character."""
        if self.reference_embedding is None:
            return True # Pass if no reference
            
        current_embedding = self.vision.extract_concept(image)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(self.reference_embedding, current_embedding, dim=-1).item()
        print(f"AutoGate similarity score: {similarity:.3f}")
        
        return similarity >= threshold


class HumanApproval:
    """CLI for human approval. Also saves approved frames for distillation training."""
    @staticmethod
    def approve(image: Image.Image, current_prompt: str, save_path: str, auto_approve: bool = False) -> bool:
        image.save(save_path)
        print(f"\n--- HUMAN REVIEW REQUIRED ---")
        print(f"Current Prompt: {current_prompt}")
        print(f"Image saved to: {save_path}")
        
        approved = False
        if auto_approve:
            print("Auto-approving for test mode.")
            approved = True
        else:
            while True:
                choice = input("Approve this frame? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    approved = True
                    break
                elif choice in ['n', 'no']:
                    approved = False
                    break
                print("Please enter 'y' or 'n'.")
                
        if approved:
            # Save to distillation dataset for the Direct CNN to learn from later
            distill_dir = Path("dataset/distillation")
            distill_dir.mkdir(parents=True, exist_ok=True)
            
            # Use a hash of the prompt or timestamp for unique naming
            import time
            ts = int(time.time() * 1000)
            img_name = f"distill_{ts}.png"
            
            # Save image
            image.save(distill_dir / img_name)
            
            # Append to manifest
            manifest_path = distill_dir / "manifest.jsonl"
            with open(manifest_path, "a", encoding="utf-8") as f:
                record = {
                    "file_name": img_name,
                    "prompt": current_prompt,
                    "timestamp": ts
                }
                f.write(json.dumps(record) + "\n")
                
        return approved


class Assembler:
    """Assembles audio and images into a final video. Uses T+1 VAE + SR Upscaler for continuation."""
    def __init__(self, output_dir: str = "output/video", device="cpu", visuals=None,
                 use_transformer: bool = False, strict_timing: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.visuals = visuals # Pass VisualGenerator to allow Teacher Interruption
        self.use_transformer = use_transformer
        # INV-1 enforcement level. False = report a chained drawing and carry on
        # (so existing projects still render). True = raise. See ANIMATION_SPEC.md.
        self.strict_timing = strict_timing
        self.timing_reports: List[Dict[str, Any]] = []
        self.t_plus_1_model = None
        self.sr_upscaler = None
        self.vision = None
        self._load_t_plus_1_model()
        self._load_sr_upscaler()
        self._load_latent_flow_models()

    def _load_t_plus_1_model(self):
        try:
            import sys
            import os
            import torch
            from torchvision import transforms
            
            # Import model directly from test script
            sys.path.append('.')
            from test_phase2_local import TPlus1VAE
            
            print("Loading T+1 VAE...")
            self.t_plus_1_model = TPlus1VAE(latent_dim=64).to(self.device)
            
            # Find latest checkpoint
            checkpoints = [
                "t_plus_1_vae_ep52.pt",
                "t_plus_1_vae_ep47.pt",
                "t_plus_1_vae_final.pt",
                "t_plus_1_vae_latest.pt",
                "t_plus_1_vae_phase2_final.pt",
                "t_plus_1_vae_ep6.pt",
                "t_plus_1_vae_recon.pt"
            ]
            loaded = False
            for ckpt in checkpoints:
                if os.path.exists(ckpt):
                    print(f"Loading T+1 checkpoint: {ckpt}")
                    state_dict = torch.load(ckpt, map_location=self.device, weights_only=True)
                    
                    if 'decoder.deconv.9.weight' in state_dict and state_dict['decoder.deconv.9.weight'].shape[1] != 5:
                        print("  [WARNING] Architecture change detected! Dropping last decoder layer to start fresh.")
                        del state_dict['decoder.deconv.9.weight']
                        del state_dict['decoder.deconv.9.bias']
                        
                    self.t_plus_1_model.load_state_dict(state_dict, strict=False)
                    loaded = True
                    break
            
            if not loaded:
                print("No T+1 VAE checkpoint found! Generation will use untrained weights.")
                
            self.t_plus_1_model.eval()
            self.has_t_plus_1 = True
            
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        except Exception as e:
            print(f"Failed to load T+1 VAE model: {e}")
            self.has_t_plus_1 = False

    def _load_sr_upscaler(self):
        """Load Super-Resolution upscaler for 64x64 → 128x128"""
        try:
            import sys
            sys.path.append('.')
            from sr_resnet import SRResNet
            
            print("Loading SR Upscaler (SRResNet)...")
            self.sr_upscaler = SRResNet().to(self.device)
            
            # Find best checkpoint
            checkpoints = [
                "sr_resnet_best.pt",
                "sr_resnet_final.pt",
                "sr_upscaler_best.pt",
                "sr_upscaler_final.pt",
            ]
            
            loaded = False
            for ckpt in checkpoints:
                if os.path.exists(ckpt):
                    print(f"Loading SR checkpoint: {ckpt}")
                    self.sr_upscaler.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
                    loaded = True
                    break
            
            if not loaded:
                print("WARNING: No SR upscaler checkpoint found. Falling back to bilinear upscaling.")
                self.has_sr_upscaler = False
            else:
                self.sr_upscaler.eval()
                self.has_sr_upscaler = True
                print("SR Upscaler loaded successfully!")
            
        except Exception as e:
            print(f"Failed to load SR upscaler: {e}")
            print("Falling back to bilinear upscaling.")
            self.has_sr_upscaler = False

    def _load_latent_flow_models(self):
        import os
        from pathlib import Path
        lf_path = Path.home() / "Projects" / "latent_flow"
        
        # FIXED: explicitly check path exists instead of relying on try/except
        # to silently skip, making bugs hardware-dependent.
        if not lf_path.is_dir():
            print(f"Latent Flow path not found ({lf_path}); skipping Latent Flow models.")
            self.has_latent_flow = False
            return
            
        try:
            import sys
            import torch
            
            if lf_path not in sys.path:
                sys.path.append(lf_path)
            
            import config as cfg
            from models.vae_improved import ImprovedVAE
            from models.imagination_engine import ImaginationEngine
            from models.transformer_temporal import TransformerTemporalEngine
            
            print("Loading Latent Flow VAE...")
            self.latent_vae = ImprovedVAE(z_dim=cfg.Z_DIM).to(self.device)
            self.latent_vae.load_state_dict(torch.load(os.path.join(lf_path, "models", "vae_improved_weights.pth"), map_location=self.device, weights_only=True))
            self.latent_vae.eval()
            
            print("Loading Latent Flow Imagination Engine (RNN)...")
            self.latent_rnn = ImaginationEngine(
                z_dim=cfg.Z_DIM, 
                hidden_dim=cfg.RNN_HIDDEN, 
                num_mixtures=cfg.MIXTURES,
                use_latent_velocity=cfg.USE_LATENT_VELOCITY
            ).to(self.device)
            self.latent_rnn.load_state_dict(torch.load(os.path.join(lf_path, "models", "imagination_engine_weights.pth"), map_location=self.device, weights_only=True))
            self.latent_rnn.eval()
            
            # Load Transformer variant if available for swapping
            self.has_transformer = False
            transformer_path = os.path.join(lf_path, "models", "transformer_temporal_weights.pth")
            if os.path.exists(transformer_path):
                print("Loading Latent Flow Temporal Transformer...")
                self.latent_transformer = TransformerTemporalEngine(
                    z_dim=cfg.Z_DIM,
                    hidden_dim=cfg.RNN_HIDDEN, # ignored by transformer but required by signature
                    num_mixtures=cfg.MIXTURES,
                    use_latent_velocity=cfg.USE_LATENT_VELOCITY
                ).to(self.device)
                self.latent_transformer.load_state_dict(torch.load(transformer_path, map_location=self.device, weights_only=True))
                self.latent_transformer.eval()
                self.has_transformer = True
                
            # Default to RNN, can be swapped manually or via UI later
            self.active_temporal_model = self.latent_transformer if self.use_transformer and self.has_transformer else self.latent_rnn
            
            print(f"Latent Flow Active Temporal Model: {'Transformer' if self.use_transformer and self.has_transformer else 'RNN'}")
            
            print("Loading Latent Flow Hybrid Controller...")
            from models.controller import HybridController
            self.latent_controller = HybridController(
                z_dim=cfg.Z_DIM, 
                rnn_hidden=cfg.RNN_HIDDEN
            ).to(self.device)
            self.latent_controller.load_state_dict(torch.load(os.path.join(lf_path, "models", "hybrid_controller_weights.pth"), map_location=self.device, weights_only=True))
            self.latent_controller.eval()
            
            self.latent_cfg = cfg
            self.has_latent_flow = True
            print("Successfully loaded video-trained Latent Flow models!")
        except Exception as e:
            print(f"Failed to load Latent Flow models: {e}. Falling back to simple morphing.")
            self.has_latent_flow = False

    def _save_drift_correction(self, prev_z, corrected_z, prompt: str, frame_idx: int, prev_pil=None, corrected_pil=None):
        """
        Saves drift correction pairs for T+1 student distillation training.
        
        The DirectContinuationCNN (T+1 student) trains on pixel-space images, so we need
        to save both the pre-drift image and the Teacher-corrected image for it to learn:
        "Given this drifted frame, the correct next frame should look like this."
        """
        import time
        try:
            distill_dir = Path("dataset/distillation")
            distill_dir.mkdir(parents=True, exist_ok=True)
            
            ts = int(time.time() * 1000)
            
            # Decode latents to pixel images if PIL images not provided
            if prev_pil is None and hasattr(self, 'latent_vae'):
                with torch.no_grad():
                    prev_tensor = self.latent_vae.decode(prev_z)
                    # ImprovedVAE.decode() already outputs [0, 1]
                    prev_tensor = prev_tensor.clamp(0, 1)
                    prev_np = (prev_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    prev_pil = Image.fromarray(prev_np)
                    
            if corrected_pil is None and hasattr(self, 'latent_vae'):
                with torch.no_grad():
                    corr_tensor = self.latent_vae.decode(corrected_z)
                    # ImprovedVAE.decode() already outputs [0, 1]
                    corr_tensor = corr_tensor.clamp(0, 1)
                    corr_np = (corr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    corrected_pil = Image.fromarray(corr_np)
            
            # Save pixel images for T+1 training (required by distill_student.py)
            prev_path = distill_dir / f"drift_prev_{ts}.png"
            corr_path = distill_dir / f"drift_corr_{ts}.png"
            
            if prev_pil is not None:
                prev_pil.save(prev_path)
            if corrected_pil is not None:
                corrected_pil.save(corr_path)
            
            # Write to T+1 manifest in the format distill_student.py expects
            manifest_path = distill_dir / "t_plus_1_manifest.jsonl"
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "prev": str(prev_path),
                    "next": str(corr_path),
                    "prompt": prompt,
                    "frame_idx": frame_idx,
                    "timestamp": ts
                }) + "\n")
                
            # Also save latent vectors for future latent-space training experiments
            torch.save({
                'prev_z': prev_z.cpu(),
                'corrected_z': corrected_z.cpu(),
                'prompt': prompt,
                'frame_idx': frame_idx,
                'timestamp': ts
            }, distill_dir / f"latent_drift_{ts}.pt")
            
        except Exception as e:
            print(f"  Warning: Failed to save drift correction: {e}")

    def _generate_frames(self, img1_path: str, img2_path: str, prompt: str, duration: float, fps: int = 24) -> List[np.ndarray]:
        """
        Generates continuation frames using HYBRID ANCHORING strategy:
        - Fixed Teacher anchors every ANCHOR_INTERVAL frames (guarantees ~25-33% teacher)
        - Adaptive CLIP drift injection when similarity drops (targets 60-70% teacher total)
        - Latent Flow (VAE + RNN/Transformer + Controller) for T+1 interpolation between anchors
        """
        from PIL import Image
        import numpy as np
        import torch
        import time
        
        # =====================================================================
        # HYBRID ANCHORING CONFIGURATION (for 60-70% teacher grounding at 25fps)
        # =====================================================================
        ANCHOR_INTERVAL = 3          # Fixed anchor every 3 frames (33% minimum teacher)
        DRIFT_THRESHOLD = 0.20       # CLIP similarity threshold for adaptive anchor injection
        ADAPTIVE_CHECK_INTERVAL = 2  # Check drift every 2 frames between fixed anchors
        MIN_PULL_STRENGTH = 0.15     # Minimum guidance pull (was 0 at alpha=0)
        MAX_PULL_STRENGTH = 0.45     # Maximum guidance pull at segment end (was 0.3)
        
        img1_pil = Image.open(img1_path).convert('RGB').resize((128, 128), Image.BILINEAR)
        img2_pil = Image.open(img2_path).convert('RGB').resize((128, 128), Image.BILINEAR)
        num_frames = int(duration * fps)
        frames = []
        
        # Stats for logging
        teacher_frames = 0
        student_frames = 0

        # ---------------------------------------------------------------
        # Keep False until latent_vae is retrained on SD-generated (not raw video) frames.
        # ---------------------------------------------------------------
        USE_LATENT_FLOW = False 
        
        if USE_LATENT_FLOW and getattr(self, 'has_latent_flow', False):
            # ---------------------------------------------------------
            # ADVANCED VIDEO GENERATION (USING LATENT FLOW + DENSE ANCHORS)
            # ---------------------------------------------------------
            
            # Ensure we have VisionSystem for CLIP drift detection
            if self.vision is None:
                from vision import VisionSystem
                self.vision = VisionSystem(device=self.device, load_vae=False)
            
            # Pre-compute text embedding for drift detection
            with torch.no_grad():
                text_inputs = self.vision.clip_processor(text=[prompt], return_tensors="pt", padding=True, truncation=True).to(self.device)
                text_embed = self.vision.clip_model.get_text_features(**text_inputs)
                text_embed = text_embed / text_embed.norm(p=2, dim=-1, keepdim=True)
            
            # Normalize to [-1, 1] for ImprovedVAE
            t1 = (torch.from_numpy(np.array(img1_pil)).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0) * 2.0 - 1.0
            t2 = (torch.from_numpy(np.array(img2_pil)).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0) * 2.0 - 1.0
            
            with torch.no_grad():
                mu1, _ = self.latent_vae.encode(t1)
                mu2, _ = self.latent_vae.encode(t2)
                
            # =====================================================================
            # DENSE FIXED ANCHORING: Generate Teacher anchor every ANCHOR_INTERVAL frames
            # =====================================================================
            num_fixed_anchors = max(1, num_frames // ANCHOR_INTERVAL)
            print(f"  [Hybrid Anchor] Generating {num_fixed_anchors} fixed anchors (every {ANCHOR_INTERVAL} frames) for {num_frames} total frames...")
            
            # anchor_data stores (frame_idx, mu_tensor, pil_image) for each anchor
            anchor_data = [(0, mu1, img1_pil)]
            
            if self.visuals and self.visuals.core_model and hasattr(self.visuals.core_model, 'img2img_pipe'):
                curr_anchor_pil = img1_pil
                
                for anchor_idx in range(1, num_fixed_anchors):
                    target_frame = anchor_idx * ANCHOR_INTERVAL
                    if target_frame >= num_frames - 1:
                        break
                        
                    try:
                        init_image = curr_anchor_pil.resize((128, 128), Image.BILINEAR)
                        
                        # Generate next fixed anchor with img2img
                        # Lower strength (0.45) = more structure retention from previous anchor
                        anchor_strength = 0.45
                        anchor_steps = safe_img2img_steps(6, anchor_strength, MIN_EFFECTIVE_IMG2IMG_STEPS)
                        mid_pil = self.visuals.core_model.img2img_pipe(
                            prompt=prompt,
                            image=init_image,
                            strength=anchor_strength,
                            num_inference_steps=anchor_steps,
                            guidance_scale=5.0
                        ).images[0]

                        # Correct color drift back to shot's first keyframe
                        mid_pil = match_histogram_to_reference(mid_pil, img1_pil)
                        
                        curr_anchor_pil = mid_pil.resize((128, 128), Image.BILINEAR)
                        mid_tensor = (torch.from_numpy(np.array(curr_anchor_pil)).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0) * 2.0 - 1.0
                        
                        with torch.no_grad():
                            mid_mu, _ = self.latent_vae.encode(mid_tensor)
                        anchor_data.append((target_frame, mid_mu, curr_anchor_pil))
                        teacher_frames += 1
                    except Exception as e:
                        print(f"  Warning: Fixed anchor at frame {target_frame} failed: {e}")
                        
            # Always include the final frame as an anchor
            anchor_data.append((num_frames - 1, mu2, img2_pil))
            
            # Sort anchors by frame index (should already be sorted, but be safe)
            anchor_data.sort(key=lambda x: x[0])
            
            print(f"  [Hybrid Anchor] {len(anchor_data)} total anchors placed: {[a[0] for a in anchor_data]}")
            
            # =====================================================================
            # LATENT FLOW GENERATION WITH ADAPTIVE DRIFT INJECTION
            # =====================================================================
            with torch.no_grad():
                curr_z = mu1
                prev_z = mu1
                hidden = None
                curr_anchor_idx = 0
                frames_since_anchor = 0
                last_anchor_pil = img1_pil
                
                for i in range(num_frames):
                    # Check if we've reached the next fixed anchor
                    if curr_anchor_idx < len(anchor_data) - 1 and i >= anchor_data[curr_anchor_idx + 1][0]:
                        curr_anchor_idx += 1
                        curr_z = anchor_data[curr_anchor_idx][1].clone()
                        last_anchor_pil = anchor_data[curr_anchor_idx][2]
                        frames_since_anchor = 0
                        # Decode anchor directly (it's a teacher frame)
                        img_tensor = self.latent_vae.decode(curr_z)
                        # ImprovedVAE.decode() already outputs [0, 1]
                        img_tensor = img_tensor.clamp(0, 1)
                        out_np = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        frames.append(out_np)
                        continue
                    
                    # Get current and next anchor for interpolation targeting
                    curr_anchor = anchor_data[curr_anchor_idx]
                    next_anchor = anchor_data[min(curr_anchor_idx + 1, len(anchor_data) - 1)]
                    
                    # Calculate local progress within this anchor segment
                    segment_start = curr_anchor[0]
                    segment_end = next_anchor[0]
                    segment_length = max(1, segment_end - segment_start)
                    segment_alpha = (i - segment_start) / segment_length
                    
                    target_z = next_anchor[1]
                    
                    # Prepare RNN input
                    if self.latent_cfg.USE_LATENT_VELOCITY:
                        delta = curr_z - prev_z if frames_since_anchor > 0 else torch.zeros_like(curr_z)
                        rnn_in = torch.cat([curr_z, delta], dim=-1).unsqueeze(1)
                    else:
                        rnn_in = curr_z.unsqueeze(1)
                        
                    prev_z = curr_z.clone()
                    
                    # Rollout using the active temporal model
                    out = self.active_temporal_model(rnn_in, hidden, training=False)
                    pi, mu_out, sigma_out, hidden = out[0], out[1], out[2], out[3]
                    next_z = self.active_temporal_model.sample(pi, mu_out, sigma_out, temperature=0.8).squeeze(1)
                    
                    if self.latent_cfg.PREDICT_DELTA_Z:
                        curr_z = curr_z + next_z
                    else:
                        curr_z = next_z
                        
                    # Ensure batch dimension
                    if curr_z.dim() == 1:
                        curr_z = curr_z.unsqueeze(0)
                    if target_z.dim() == 1:
                        target_z = target_z.unsqueeze(0)
                        
                    # Controller steering
                    action = self.latent_controller(curr_z, target_z, rnn_hidden=hidden, trans_context=None).squeeze(1)
                    curr_z = curr_z + action * 0.4
                    
                    # =========================================================
                    # STRENGTHENED PULL GUIDANCE (for continuous grounding)
                    # =========================================================
                    # Use quadratic curve with minimum floor for consistent pull
                    pull_strength = MIN_PULL_STRENGTH + (MAX_PULL_STRENGTH - MIN_PULL_STRENGTH) * (segment_alpha ** 2)
                    curr_z = curr_z * (1 - pull_strength) + target_z * pull_strength
                    
                    # Decode to pixels
                    img_tensor = self.latent_vae.decode(curr_z)
                    # ImprovedVAE.decode() already outputs [0, 1]
                    img_tensor = img_tensor.clamp(0, 1)
                    out_np = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    
                    # =========================================================
                    # ADAPTIVE DRIFT DETECTION + TEACHER INJECTION
                    # =========================================================
                    frames_since_anchor += 1
                    inject_anchor = False
                    
                    if frames_since_anchor >= ADAPTIVE_CHECK_INTERVAL and self.visuals and self.visuals.core_model:
                        # Check CLIP similarity between generated frame and prompt
                        out_pil = Image.fromarray(out_np)
                        img_inputs = self.vision.clip_processor(images=out_pil, return_tensors="pt").to(self.device)
                        img_embed = self.vision.clip_model.get_image_features(**img_inputs)
                        img_embed = img_embed / img_embed.norm(p=2, dim=-1, keepdim=True)
                        
                        similarity = torch.nn.functional.cosine_similarity(img_embed, text_embed, dim=-1).item()
                        
                        if similarity < DRIFT_THRESHOLD:
                            inject_anchor = True
                            print(f"    [Drift @ frame {i}] sim={similarity:.3f} < {DRIFT_THRESHOLD}, injecting Teacher anchor")
                            
                            try:
                                # Generate correction anchor
                                correction_strength = 0.55
                                correction_steps = safe_img2img_steps(6, correction_strength, MIN_EFFECTIVE_IMG2IMG_STEPS)
                                correction_pil = self.visuals.core_model.img2img_pipe(
                                    prompt=prompt,
                                    image=out_pil,
                                    strength=correction_strength,
                                    num_inference_steps=correction_steps,
                                    guidance_scale=5.5
                                ).images[0].resize((128, 128), Image.BILINEAR)

                                # Correct color drift back to shot's first keyframe
                                correction_pil = match_histogram_to_reference(correction_pil, img1_pil)
                                
                                # Update latent state to correction
                                corr_tensor = (torch.from_numpy(np.array(correction_pil)).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0) * 2.0 - 1.0
                                curr_z, _ = self.latent_vae.encode(corr_tensor)
                                
                                # Use corrected frame
                                out_np = np.array(correction_pil)
                                last_anchor_pil = correction_pil
                                frames_since_anchor = 0
                                teacher_frames += 1
                                
                                # Save for distillation - pass PIL images directly for efficiency
                                self._save_drift_correction(prev_z, curr_z, prompt, i, prev_pil=out_pil, corrected_pil=correction_pil)
                            except Exception as e:
                                print(f"    Warning: Drift correction failed: {e}")
                    
                    if not inject_anchor:
                        student_frames += 1
                        
                    frames.append(out_np)
            
            total = teacher_frames + student_frames
            teacher_pct = (teacher_frames / total * 100) if total > 0 else 0
            print(f"  [Hybrid Anchor] Complete: {teacher_frames}/{total} teacher frames ({teacher_pct:.1f}%), {student_frames} student frames")
            return frames

        # ---------------------------------------------------------
        # CLEAN AUTOREGRESSIVE GENERATION (FLOW + T+1 VAE + SR UPSCALER)
        # ---------------------------------------------------------
        print(f"  [Flow + T+1 Generation] Generating {num_frames} frames...")
        
        # Start with first keyframe at native 128x128
        current_frame_128 = img1_pil
        frames.append(np.array(current_frame_128))
        
        # Initialize Flow Interpolator
        from flow_interpolator import SimpleFlowInterpolator
        flow_model = SimpleFlowInterpolator().to(self.device)
        flow_model.eval()
        
        # Pre-process images for flow model (expects 0-1 tensors)
        t1_flow = torch.from_numpy(np.array(img1_pil)).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        t2_flow = torch.from_numpy(np.array(img2_pil)).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        
        # The Flow Interpolator expects to warp from A to B.
        # But if we just feed it the same A and B every time, it will always output the same warp.
        # We need to warp the *current* frame towards B, or warp A towards B with increasing alpha.
        # The SimpleFlowInterpolator is designed to warp A towards B with a time parameter t.
        
        for i in range(1, num_frames):
            # Calculate interpolation alpha (0 to 1)
            alpha = i / num_frames
            
            with torch.no_grad():
                # Smooth ease-in-out interpolation for natural motion
                smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
                
                # 1. Macro Interpolation (Global structure motion)
                if hasattr(flow_model, 'use_raft') and flow_model.use_raft:
                    # RAFT requires exact [0, 1] bounds, SimpleFlowInterpolator._raft_forward handles it
                    macro_warped = flow_model(t1_flow, t2_flow, smooth_alpha)
                    
                    # Fill occlusion holes with target frame structure
                    # When RAFT pushes pixels away, it leaves a black smear. We blend the target frame in
                    # based on time to fill the gaps naturally.
                    macro_warped = macro_warped * (1 - smooth_alpha * 0.5) + t2_flow * (smooth_alpha * 0.5)
                else:
                    macro_warped = flow_model(t1_flow, t2_flow, smooth_alpha)
                
                # 2. Micro Details & Correction (T+1 VAE)
                if self.t_plus_1_model is not None and self.has_t_plus_1:
                    # Resize macro frame to 64x64 for T+1 VAE
                    macro_64 = torch.nn.functional.interpolate(macro_warped, size=(64, 64), mode='bilinear', align_corners=False)
                    
                    # Encode current frame state
                    mu, _ = self.t_plus_1_model.encoder(macro_64)
                    
                    # Instead of autoregressive dynamics, we decode the *interpolated* latent 
                    # state to get high-frequency details (micro flow & residual)
                    flow_res, img_res = self.t_plus_1_model.decoder(mu)
                    
                    # Apply micro-corrections
                    final_64 = torch.clamp(self.t_plus_1_model.warp(macro_64, flow_res) + img_res, 0, 1)
                    
                    # Upscale back to 128x128
                    if getattr(self, 'has_sr_upscaler', False) and self.sr_upscaler is not None:
                        # Use SR upscaler if available
                        final_128 = self.sr_upscaler(final_64)
                        final_128 = torch.clamp(final_128, 0, 1)
                    else:
                        # Fallback bilinear
                        final_128 = torch.nn.functional.interpolate(final_64, size=(128, 128), mode='bilinear', align_corners=False)
                else:
                    final_128 = macro_warped
                
                # Convert back to PIL
                final_np = (final_128.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                current_frame_128 = Image.fromarray(final_np)
                
            # NEW: Run all frames (even pure teacher keyframes) through the VAE and SRResNet
            # so that every single frame in the video shares the exact same color/texture properties.
            if getattr(self, 'has_t_plus_1', False) and self.t_plus_1_model is not None and getattr(self, 'has_sr_upscaler', False) and self.sr_upscaler is not None:
                with torch.no_grad():
                    # If this frame didn't already go through the VAE (like anchor frames), push it through
                    if i == 0 or (curr_anchor_idx < len(anchor_data) and i == anchor_data[curr_anchor_idx][0]):
                        # Resize to 64x64 for VAE
                        frame_tensor = torch.from_numpy(np.array(current_frame_128)).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
                        frame_64 = torch.nn.functional.interpolate(frame_tensor, size=(64, 64), mode='bilinear', align_corners=False)
                        
                        # Encode and decode through VAE
                        mu, _ = self.t_plus_1_model.encoder(frame_64)
                        flow, img_res = self.t_plus_1_model.decoder(mu)
                        vae_out = torch.clamp(self.t_plus_1_model.warp(frame_64, flow) + img_res, 0, 1)
                        
                        # Push through SRResNet
                        final_128 = self.sr_upscaler(vae_out)
                        final_128 = torch.clamp(final_128, 0, 1)
                        
                        # Update current frame
                        final_np = (final_128.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        current_frame_128 = Image.fromarray(final_np)
                        # Replace the already-appended frame to maintain consistency
                        frames[-1] = np.array(current_frame_128)
                        
            # Update for next iteration
            frames.append(np.array(current_frame_128))
            student_frames += 1
            
        print(f"  [Flow Generation] Complete: {num_frames} frames generated with spatial warping")
        return frames

    def build_exposure_sheet(
        self,
        shot_idx: int,
        shot: Dict[str, Any],
        total_frames: int,
        fps: int = 24,
    ) -> ExposureSheet:
        """Turn a shot's approved keyframes into an exposure sheet.

        Timing comes from the sheet (INV-2). A shot may author it explicitly as
        `shot["exposure"]`, one held-frame count per keyframe. When that is
        absent the duration is divided evenly across the keyframes, which
        reproduces the legacy timing -- the door is open, but nothing changes
        silently.
        """
        keyframes = list(shot.get("keyframes", []))
        if not keyframes:
            raise SpecViolation(f"shot {shot_idx} has no keyframes")

        actions = shot.get("action_sequence") or [shot.get("prompt", "A scene")]
        if not actions:
            actions = ["A scene"]

        n = len(keyframes)
        prompts = []
        for kf_idx in range(n):
            action_idx = min(int((kf_idx / max(1, n - 1)) * len(actions)), len(actions) - 1)
            prompts.append(actions[action_idx])

        exposure = shot.get("exposure")
        if exposure:
            exposure = [max(1, int(e)) for e in exposure]
            if len(exposure) != n:
                print(f"    [timing] authored exposure has {len(exposure)} entries for {n} "
                      f"keyframes; ignoring it and dividing evenly.")
                exposure = None
            elif sum(exposure) != total_frames:
                print(f"    [timing] authored exposure sums to {sum(exposure)} frames but "
                      f"the audio needs {total_frames}; rescaling proportionally.")
        else:
            print(f"    [timing] no authored exposure for shot {shot_idx}; dividing "
                  f"{total_frames} frames evenly across {n} keyframes. Author "
                  f"shot['exposure'] to choose holds (INV-2).")

        return sheet_from_keyframes(
            sheet_id=f"shot{shot_idx}",
            keyframe_paths=keyframes,
            duration_frames=total_frames,
            fps=fps,
            exposure=exposure or (),
            prompts=prompts,
        )

    def _render_sheet(
        self,
        sheet: ExposureSheet,
        fps: int = 24,
    ) -> List[np.ndarray]:
        """Generate frames by walking the exposure sheet's segments.

        The terminal segment is a genuine hold and costs nothing: it repeats the
        drawing already on screen. Only transitions between distinct drawings
        call the generator (INV-3, INV-5).
        """
        segments = sheet.layer("main").track.segments()
        all_frames: List[np.ndarray] = []

        for seg_idx, (drawing_id, _start, length) in enumerate(segments):
            drawing = sheet.drawings[drawing_id]
            is_last = seg_idx == len(segments) - 1

            if is_last:
                if all_frames:
                    all_frames.extend([all_frames[-1]] * length)
                else:
                    all_frames.extend(self._static_frames(drawing, length))
                print(f"    Segment {seg_idx+1}/{len(segments)}: hold {drawing_id} for "
                      f"{length} frames (0 generations)")
                continue

            next_id = segments[seg_idx + 1][0]
            next_drawing = sheet.drawings[next_id]
            segment_duration = length / fps

            print(f"    Segment {seg_idx+1}/{len(segments)}: {drawing_id} -> {next_id} "
                  f"over {length} frames ({segment_duration:.2f}s)")

            frames = self._generate_frames(
                drawing.note,
                next_drawing.note,
                drawing.prompt,
                segment_duration,
                fps=fps,
            )
            all_frames.extend(self._fit_exact(frames, length, drawing))

        return all_frames

    @staticmethod
    def _static_frames(drawing, length: int) -> List[np.ndarray]:
        """One drawing held for `length` frames, with no generation call."""
        arr = np.array(Image.open(drawing.note).convert("RGB"))
        return [arr] * length

    @staticmethod
    def _fit_exact(frames: List[np.ndarray], length: int, drawing) -> List[np.ndarray]:
        """Force a generated segment to exactly `length` frames.

        The sheet decides the count; the generator's internal rounding does not.
        """
        if not frames:
            return Assembler._static_frames(drawing, length)
        if len(frames) == length:
            return frames
        if len(frames) > length:
            return frames[:length]
        return frames + [frames[-1]] * (length - len(frames))

    def assemble(
        self,
        shots: List[Dict[str, Any]],
        output_filename: str = "final_video.mp4",
        strict_timing: Optional[bool] = None,
    ):
        print("Assembling video from exposure sheets...")
        from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_videoclips
        clips = []
        strict = self.strict_timing if strict_timing is None else strict_timing

        for shot_idx, shot in enumerate(shots):
            if 'keyframes' not in shot or 'audio_file' not in shot:
                print(f"  Warning: Shot {shot_idx} missing keyframes or audio, skipping.")
                continue

            print(f"\n  Processing shot {shot_idx+1}/{len(shots)}...")

            # Load audio to get duration
            audio_clip = AudioFileClip(shot['audio_file'])
            duration = audio_clip.duration
            total_frames_needed = int(duration * 24)  # 24 fps

            print(f"    Duration: {duration:.2f}s ({total_frames_needed} frames @ 24fps)")
            print(f"    Keyframes: {len(shot['keyframes'])}")

            # -----------------------------------------------------------------
            # Timing is authored, not implied (INV-2).
            # -----------------------------------------------------------------
            sheet = self.build_exposure_sheet(shot_idx, shot, total_frames_needed)
            for warning in sheet.validate(strict=strict):
                print(f"    [spec] {warning}")

            report = sheet.cost_report()
            print(f"    [spec] {report['frames']} frames from "
                  f"{report['generation_calls']} drawings "
                  f"({report['held_or_reused_frames']} frames held or reused, "
                  f"{report['reduction_vs_on_ones']:.0%} fewer generations than on ones)")
            self.timing_reports.append(report)
            print(sheet.to_ascii(max_frames=min(12, sheet.total_frames)))

            all_frames = self._render_sheet(sheet)

            # The sheet is authoritative, so this must already be exact. The old
            # pad/trim step is gone; if we still miss, the timing is wrong.
            if len(all_frames) != total_frames_needed:
                raise SpecViolation(
                    f"sheet {sheet.sheet_id} rendered {len(all_frames)} frames but the "
                    f"audio needs {total_frames_needed} (INV-2)"
                )

            print(f"    Generated {len(all_frames)} frames (target: {total_frames_needed})")
            
            # The all_frames list contains numpy arrays.
            # Make sure we convert everything strictly to uint8 arrays for MoviePy.
            final_frames_np = []
            for frame in all_frames:
                if isinstance(frame, np.ndarray):
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                    final_frames_np.append(frame)
                else:
                    # If it's a PIL Image for some reason
                    final_frames_np.append(np.array(frame))
                    
            # Create video clip
            video_clip = ImageSequenceClip(final_frames_np, fps=24)
            video_clip = video_clip.set_audio(audio_clip)
            
            clips.append(video_clip)
        
        if not clips:
            print("No clips to assemble.")
            return
        
        # Concatenate all shots
        final_video = concatenate_videoclips(clips, method="compose")
        final_path = self.output_dir / output_filename
        
        # Write video
        final_video.write_videofile(str(final_path), fps=24, codec="libx264", audio_codec="aac")
        print(f"\n[OK] Video saved to {final_path}")


class ProductionSystem:
    """The main orchestration layer."""
    def __init__(self, device="cpu", strict_timing: bool = False):
        self.director = Director()
        self.voice = VoiceModule()
        self.visuals = VisualGenerator(device=device)
        self.gate = AutoGate(self.visuals.vision)
        # INV-1 enforcement level, threaded through to the Assembler.
        self.strict_timing = strict_timing
        self.assembler = Assembler(visuals=self.visuals, strict_timing=strict_timing)
        
        self.output_dir = Path("output/frames")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def produce(self, idea: str, auto_approve: bool = False):
        print(f"Starting production for idea: {idea}")
        
        # 1. Script
        print("\n--- PHASE 1: SCRIPT ---")
        script = self.director.generate_script(idea)
        print(json.dumps(script, indent=2))
        
        # 2. Voice & Timing
        print("\n--- PHASE 2: VOICE & TIMING ---")
        script_with_timings = self.voice.generate_audio_for_script(script)
        
        # 3. Shot Planning
        print("\n--- PHASE 3: SHOT PLANNING ---")
        shots = self.director.plan_shots(script_with_timings)
        print(json.dumps(shots, indent=2))
        
        # 4. Visual Generation & Approval (DENSE KEYFRAMES for consistency)
        print("\n--- PHASE 4: VISUAL GENERATION (DENSE ANCHORING) ---")
        for i, shot in enumerate(shots):
            print(f"\n=== Shot {i+1}/{len(shots)} ===")
            print(f"Duration: {shot['duration']:.1f}s")
            
            # Handle legacy 'prompt' or new 'action_sequence'
            actions = shot.get('action_sequence', [shot.get('prompt', 'A scene')])
            if not actions: actions = ["A scene"]
            print(f"Action Sequence: {json.dumps(actions, indent=2)}")
            
            # Calculate how many keyframes needed for this shot
            # Strategy: Generate keyframe every ANCHOR_INTERVAL seconds
            ANCHOR_INTERVAL = 1.0  # Generate Teacher keyframe every 1 second
            num_keyframes = max(2, int(shot['duration'] / ANCHOR_INTERVAL) + 1)
            
            print(f"Generating {num_keyframes} anchor keyframes for temporal consistency...")
            
            keyframes = []
            chain_warned = False
            
            for kf_idx in range(num_keyframes):
                approved = False
                attempts = 0
                max_attempts = 3
                
                # Map the keyframe index to the closest action prompt in the sequence
                action_idx = min(int((kf_idx / max(1, num_keyframes - 1)) * len(actions)), len(actions) - 1)
                current_prompt = actions[action_idx]
                
                while not approved and attempts < max_attempts:
                    attempts += 1
                    
                    if kf_idx == 0:
                        # First keyframe: use main shot prompt
                        print(f"\n  Keyframe {kf_idx+1}/{num_keyframes} (Shot Start) -> {current_prompt}")
                        img = self.visuals.generate_keyframe(current_prompt)
                    else:
                        # Intermediate keyframes: use img2img from previous keyframe for continuity
                        print(f"\n  Keyframe {kf_idx+1}/{num_keyframes} (Anchor @ {kf_idx * ANCHOR_INTERVAL:.1f}s) -> {current_prompt}")
                        
                        # First keyframe of the shot, kept only as the fixed color/structure
                        # reference for histogram matching
                        anchor_keyframe = Image.open(keyframes[0])
                        
                        if self.visuals.core_model and hasattr(self.visuals.core_model, 'img2img_pipe'):
                            # INV-1. Chaining img2img off keyframes[-1] accumulates drift:
                            # character -> slightly different -> different person -> texture
                            # soup. Anchoring every drawing to the shot's root instead keeps
                            # one fixed visual identity for the whole shot.
                            if self.strict_timing:
                                init_image = anchor_keyframe
                            else:
                                if not chain_warned:
                                    print("    [spec] INV-1: anchoring this keyframe to the previous "
                                          "keyframe (chained). Run with --strict-timing to anchor to "
                                          "the shot root instead. See ANIMATION_SPEC.md.")
                                    chain_warned = True
                                init_image = Image.open(keyframes[-1])
                            
                            # Use helper to ensure enough steps to prevent inkblot effect
                            strength = 0.35
                            steps = safe_img2img_steps(30, strength, min_effective_steps=8)
                            
                            img = self.visuals.core_model.img2img_pipe(
                                prompt=current_prompt,
                                image=init_image,
                                strength=strength,
                                num_inference_steps=steps,
                                guidance_scale=6.0
                            ).images[0]
                            
                            # Correct per-frame VAE color drift back to this shot's first keyframe
                            img = match_histogram_to_reference(img, anchor_keyframe)
                        else:
                            # Fallback: generate fresh keyframe
                            img = self.visuals.generate_keyframe(current_prompt)
                    
                    # Check AutoGate - only skip for the very first keyframe overall
                    is_first_keyframe_overall = (i == 0 and kf_idx == 0)
                    if not is_first_keyframe_overall and not self.gate.check_quality(img):
                        print("    AutoGate rejected. Retrying...")
                        continue
                    
                    # Human Approval
                    save_path = str(self.output_dir / f"shot_{i:03d}_kf{kf_idx:02d}_v{attempts}.png")
                    if HumanApproval.approve(img, current_prompt, save_path, auto_approve=auto_approve):
                        approved = True
                        keyframes.append(save_path)
                        print(f"    [OK] Keyframe {kf_idx+1} approved")
                        
                        # Set reference from first keyframe of first shot
                        if i == 0 and kf_idx == 0:
                            self.gate.set_reference(img)
                            print("    Reference embedding set for AutoGate.")
                    else:
                        print("    Human rejected. Retrying...")
                
                if not approved:
                    print(f"    Failed after {max_attempts} attempts. Using last generated.")
                    keyframes.append(save_path)
            
            # Store all keyframes for this shot
            shot['keyframes'] = keyframes
            shot['approved_image'] = keyframes[0]  # Keep for backward compatibility
                
        # 5. Assembly
        print("\n--- PHASE 5: ASSEMBLY ---")
        self.assembler.assemble(shots)
        print("\nProduction complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--idea", type=str, default="", help="The video idea")
    parser.add_argument("--auto-approve", action="store_true", help="Skip human approval for testing")
    parser.add_argument("--strict-timing", action="store_true",
                        help="Enforce ANIMATION_SPEC.md: anchor every drawing to the shot root "
                             "(INV-1) and raise on a timing violation instead of warning.")
    args = parser.parse_args()
    
    system = ProductionSystem(strict_timing=args.strict_timing)
    system.produce(args.idea, auto_approve=args.auto_approve)
