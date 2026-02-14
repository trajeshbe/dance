#!/usr/bin/env python3
"""
Download pre-trained AI models for Dance Video Generator
"""
import os
import sys
from pathlib import Path
import urllib.request
from tqdm import tqdm


MODEL_DIR = Path(__file__).parent.parent / "models"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    print(f"Downloading {url}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_models():
    """Download all required models"""

    print("🎬 Dance Video Generator - Model Downloader")
    print("=" * 50)

    # Create model directories
    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / "fomm").mkdir(exist_ok=True)
    (MODEL_DIR / "liveportrait").mkdir(exist_ok=True)
    (MODEL_DIR / "magic-animate").mkdir(exist_ok=True)
    (MODEL_DIR / "magic-dance").mkdir(exist_ok=True)
    (MODEL_DIR / "sam").mkdir(exist_ok=True)

    models_to_download = []

    # FOMM (First Order Motion Model)
    models_to_download.append({
        "name": "FOMM - VOX 256",
        "url": "https://huggingface.co/spaces/first-order-model/demo/resolve/main/vox-256.pth.tar",
        "output": MODEL_DIR / "fomm" / "vox-256.pth",
        "size": "~350MB"
    })

    # SAM (Segment Anything Model)
    models_to_download.append({
        "name": "SAM - ViT-H",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "output": MODEL_DIR / "sam" / "sam_vit_h_4b8939.pth",
        "size": "~2.4GB"
    })

    # YOLOv8 (will be downloaded automatically by ultralytics)
    print("\n📦 Other models will be downloaded automatically on first use:")
    print("  - YOLOv8x (person detection)")
    print("  - MagicAnimate (from HuggingFace)")
    print("  - MagicDance (from HuggingFace)")
    print("  - LivePortrait (from HuggingFace)")
    print("  - Stable Video Diffusion (from HuggingFace)")
    print("  - AnimateDiff (from HuggingFace)")

    print(f"\n📁 Models will be saved to: {MODEL_DIR}")
    print("\nReady to download:")

    for i, model in enumerate(models_to_download, 1):
        print(f"  {i}. {model['name']} ({model['size']})")

    response = input("\nProceed with download? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return

    # Download models
    for model in models_to_download:
        output_path = model["output"]

        if output_path.exists():
            print(f"\n✅ {model['name']} already exists, skipping")
            continue

        print(f"\n📥 Downloading {model['name']}...")
        try:
            download_url(model["url"], output_path)
            print(f"✅ Downloaded {model['name']}")
        except Exception as e:
            print(f"❌ Error downloading {model['name']}: {e}")
            print(f"   Please download manually from: {model['url']}")

    print("\n✅ Model download complete!")
    print("\nNote: HuggingFace models (MagicAnimate, MagicDance, etc.) will be")
    print("downloaded automatically on first use via diffusers library.")
    print("\nYou can now start the application with:")
    print("  docker-compose up -d")


if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm for progress bars...")
        os.system(f"{sys.executable} -m pip install tqdm")
        from tqdm import tqdm

    download_models()
