import os
import numpy as np
from PIL import Image

for file in os.listdir("."):
    if file.startswith("test_md_") and file.endswith(".png"):
        img = Image.open(file).convert("RGB")
        arr = np.array(img)
        std = np.std(arr)
        tv_x = np.mean(np.abs(arr[:, 1:, :] - arr[:, :-1, :]))
        print(f"{file}: std={std:.2f}, tv_x={tv_x:.2f}")
