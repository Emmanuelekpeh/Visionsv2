from production_system import MobileDiffusionGenerator
import time

def test_steps():
    print("Loading generator...")
    try:
        gen = MobileDiffusionGenerator(device="cpu", model_id="models/tiny-sd")
    except Exception as e:
        print(f"Error loading models/tiny-sd: {e}")
        return

    prompt = "abandoned theater, dusty stage, distant figure in shadow, filmic grain, muted colors, soft focus"
    
    configs = [
        (4, 4.0),   # 4 steps, CFG 4.0
    ]
    
    for steps, cfg in configs:
        print(f"Testing {steps} steps, CFG={cfg}...")
        start = time.time()
        img = gen.generate_keyframe(prompt, num_inference_steps=steps, guidance_scale=cfg)
        elapsed = time.time() - start
        
        filename = f"test_s{steps}_cfg{cfg}.png"
        img.save(filename)
        print(f"Saved {filename} in {elapsed:.1f}s")

if __name__ == "__main__":
    test_steps()
