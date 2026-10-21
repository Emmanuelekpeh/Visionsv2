import torch
import time
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from alternative_t2i import DirectCNNT2I
from vision import VisionSystem

def download_and_benchmark():
    device = "cpu"
    print("1. Downloading segmind/tiny-sd...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "segmind/tiny-sd", 
        torch_dtype=torch.float32, 
        safety_checker=None
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.save_pretrained("models/tiny-sd")
    pipe = pipe.to(device)
    print("Saved to models/tiny-sd")

    print("\n2. Loading Direct CNN (Student)...")
    student = DirectCNNT2I().to(device)
    ckpt_path = "direct_cnn_t2i_kaggle_ep830.pt"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    student.load_state_dict(state_dict, strict=False)
    student.eval()
    
    print("Loading Vision System for Student text embeddings...")
    vision = VisionSystem(device=device, load_vae=False)

    prompts = [
        "a beautiful landscape with mountains and a river",
        "cyberpunk city street at night with neon lights"
    ]

    print("\n3. Benchmarking segmind/tiny-sd (Teacher)")
    for i, prompt in enumerate(prompts):
        start = time.time()
        # 15 steps is usually good for Euler A on tiny-sd
        img = pipe(prompt, num_inference_steps=15, guidance_scale=7.0).images[0]
        end = time.time()
        print(f"  Prompt {i+1}: {end - start:.2f} seconds")
        img.save(f"benchmark_teacher_{i+1}.png")

    print("\n4. Benchmarking Direct CNN (Student)")
    for i, prompt in enumerate(prompts):
        start = time.time()
        with torch.no_grad():
            inputs = vision.clip_processor(text=[prompt], return_tensors="pt", padding=True, truncation=True).to(device)
            text_embeds = vision.clip_model.get_text_features(**inputs)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            
            output = student(text_embeds, guidance_scale=3.0)
            
            img_tensor = output.squeeze(0).cpu().clamp(0, 1)
            import numpy as np
            from PIL import Image
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
        end = time.time()
        print(f"  Prompt {i+1}: {end - start:.2f} seconds")
        img.save(f"benchmark_student_{i+1}.png")

if __name__ == "__main__":
    download_and_benchmark()
