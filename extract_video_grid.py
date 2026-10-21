import cv2
import numpy as np
from PIL import Image

def create_video_grid(video_path, output_path, grid_size=(4, 6)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    num_samples = grid_size[0] * grid_size[1]
    # Sample evenly across the video
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).resize((128, 128)))
        else:
            frames.append(Image.new('RGB', (128, 128), color='red'))
    cap.release()
    
    # Stitch into a grid
    w, h = 128, 128
    grid = Image.new('RGB', (w * grid_size[0], h * grid_size[1]))
    for i, frame in enumerate(frames):
        x = (i % grid_size[0]) * w
        y = (i // grid_size[0]) * h
        grid.paste(frame, (x, y))
        
    grid.save(output_path)
    print(f"Saved {output_path} with {len(frames)} frames sampled from {total_frames} total frames.")

if __name__ == "__main__":
    create_video_grid("output/video/final_video.mp4", "video_grid.jpg")
