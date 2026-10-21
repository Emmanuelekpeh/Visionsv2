import argparse
import shutil
import zipfile
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "colab_upload"
ZIP_OUT = OUT_DIR / "visions-colab.zip"
NB_OUT = OUT_DIR / "colab_direct_t2i.ipynb"

CODE_FILES = [
    "train_direct_cnn_t2i.py",
    "alternative_t2i.py",
    "vision.py",
    "training_scheduler.py",
    "engines.py",
    "kaggle_dataset.py",
    "kaggle_train_direct_t2i.py",
    "caption_datasets.py",
    "import_caption_datasets.py",
    "activities.txt",
]

CHECKPOINT_PREFIXES = {
    "direct_cnn_t2i": "direct_cnn_t2i.pt",
    "direct_discriminator": "direct_discriminator.pt",
    "text_embed_cache": "text_embed_cache.pt",
}

def _newest_checkpoint(prefix: str) -> Path | None:
    matches = [
        p for p in ROOT.iterdir()
        if p.is_file() and p.suffix.lower() == ".pt" and p.name.startswith(prefix)
    ]
    if not matches:
        return None
    # For colab, prioritize Kaggle checkpoints if they exist, as they are the latest
    kaggle = [p for p in matches if "kaggle" in p.name.lower()]
    if kaggle:
        return max(kaggle, key=lambda p: p.stat().st_mtime)
    return max(matches, key=lambda p: p.stat().st_mtime)

def generate_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# DirectT2I Training on Google Colab\n",
                    "\n",
                    "**Setup Instructions:**\n",
                    "1. On the left sidebar, click the **Folder** icon.\n",
                    "2. Click the **Upload** icon and upload `visions-colab.zip`.\n",
                    "3. (Optional) If you have a zip file of images, upload that too.\n",
                    "4. Run the cells below!\n",
                    "\n",
                    "*Note: We mount Google Drive so you don't lose your `.pt` checkpoints when Colab disconnects!*"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Mount Google Drive to save checkpoints safely\n",
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')\n",
                    "\n",
                    "# Install dependencies\n",
                    "!pip install -q transformers diffusers accelerate huggingface_hub"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "import zipfile\n",
                    "import os\n",
                    "import shutil\n",
                    "\n",
                    "WORK_DIR = '/content/visions'\n",
                    "\n",
                    "# Check both ephemeral storage and Google Drive\n",
                    "POSSIBLE_ZIP_PATHS = [\n",
                    "    '/content/visions-colab.zip',\n",
                    "    '/content/drive/MyDrive/visions-colab.zip'\n",
                    "]\n",
                    "\n",
                    "ZIP_PATH = next((p for p in POSSIBLE_ZIP_PATHS if os.path.exists(p)), None)\n",
                    "\n",
                    "if not ZIP_PATH:\n",
                    "    print(\"⚠️ Please upload visions-colab.zip to the Colab files pane, OR place it in your Google Drive root!\")\n",
                    "else:\n",
                    "    print(f\"Found code zip at: {ZIP_PATH}\")\n",
                    "    os.makedirs(WORK_DIR, exist_ok=True)\n",
                    "    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:\n",
                    "        zf.extractall(WORK_DIR)\n",
                    "    print(f\"✅ Extracted codebase to {WORK_DIR}\")\n",
                    "\n",
                    "# Check for dataset zip\n",
                    "POSSIBLE_DATA_PATHS = [\n",
                    "    '/content/images.zip',\n",
                    "    '/content/drive/MyDrive/images.zip',\n",
                    "    '/content/dataset.zip',\n",
                    "    '/content/drive/MyDrive/dataset.zip'\n",
                    "]\n",
                    "DATASET_ZIP = next((p for p in POSSIBLE_DATA_PATHS if os.path.exists(p)), None)\n",
                    "\n",
                    "if DATASET_ZIP:\n",
                    "    print(f\"Found images zip at: {DATASET_ZIP}\")\n",
                    "    dataset_dir = os.path.join(WORK_DIR, 'dataset')\n",
                    "    os.makedirs(dataset_dir, exist_ok=True)\n",
                    "    with zipfile.ZipFile(DATASET_ZIP, 'r') as zf:\n",
                    "        zf.extractall(dataset_dir)\n",
                    "    print(f\"✅ Extracted images to {dataset_dir}\")\n",
                    "else:\n",
                    "    print(\"ℹ️ No images.zip found (skipping local dataset extraction)\")"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "%cd /content/visions\n",
                    "\n",
                    "# Link the checkpoints from Google Drive if they exist\n",
                    "DRIVE_DIR = '/content/drive/MyDrive/visions_checkpoints'\n",
                    "!mkdir -p {DRIVE_DIR}\n",
                    "\n",
                    "# Start training!\n",
                    "# (Colab T4 handles batch size 64 well. Adjust grad-accum if needed)\n",
                    "!python train_direct_cnn_t2i.py --epochs 200 --batch-size 64 --max-hours 8 --compile\n",
                    "\n",
                    "# Save everything back to Google Drive when done/interrupted\n",
                    "!cp *.pt {DRIVE_DIR}/\n",
                    "!cp -r training_samples_direct* {DRIVE_DIR}/\n",
                    "print(f\"Saved checkpoints to {DRIVE_DIR}\")"
                ],
                "execution_count": None,
                "outputs": []
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(NB_OUT, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    included = []

    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in CODE_FILES:
            path = ROOT / name
            if not path.is_file():
                print(f"Warning: {name} not found, skipping.")
                continue
            zf.write(path, name)

        for prefix, zip_name in CHECKPOINT_PREFIXES.items():
            path = _newest_checkpoint(prefix)
            if path is not None:
                zf.write(path, zip_name)
                included.append(f"{path.name} -> {zip_name}")

    generate_notebook()

    mb = ZIP_OUT.stat().st_size / (1024 * 1024)
    print(f"\\nCreated {OUT_DIR.name}/")
    print(f"  {ZIP_OUT.name} ({mb:.1f} MB)")
    for line in included:
        print(f"    + {line}")
    print(f"  {NB_OUT.name}")
    print("\\nUpload these to Google Colab!")

if __name__ == "__main__":
    main()
