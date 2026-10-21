import torch
from production_system import MobileDiffusionGenerator

def generate_sample():
    print("Loading 1-step MobileDiffusion GAN...")
    md = MobileDiffusionGenerator(device="cpu", model_id="models/MobileDiffusion", is_1step=True)
    
    prompt = "a highly detailed portrait of a cyberpunk astronaut, neon lights, 4k resolution"
    print(f"Generating: {prompt}")
    
    img = md.generate_keyframe(prompt)
    img.save("md_sample_output.png")
    print("Saved md_sample_output.png")

if __name__ == "__main__":
    generate_sample()
