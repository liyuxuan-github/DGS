#!/usr/bin/env python3
"""
Compute average pairwise LPIPS distance for images in each folder.
No command-line arguments. Edit FOLDER_PATHS directly.
Requires: pip install lpips torch torchvision pillow tqdm
"""

import os
import sys
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

# ========== CONFIG: EDIT THIS ==========
FOLDER_PATHS = [
    "./dgs_0/",
]
IMAGE_SIZE = 512
CHUNK_SIZE = 10       # Reduce if OOM, increase if GPU memory is abundant
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NET = "vgg"          # "vgg", "alex", or "squeeze"
# =======================================

try:
    import lpips
except ImportError:
    print("Error: `lpips` not installed. Run: pip install lpips")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def load_images(folder, image_size=512, device="cuda"):
    """Load all images from a folder into a tensor [N, 3, H, W] in [-1, 1]."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".gif"}
    paths = sorted([
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ])

    if len(paths) == 0:
        print(f"  [Warning] No images found in: {folder}")
        return None

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0, 1]
    ])

    tensors = []
    for p in tqdm(paths, desc=f"  Loading", leave=False):
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(transform(img))
        except Exception as e:
            print(f"  [Warning] Skip {p.name}: {e}")

    if len(tensors) < 2:
        print(f"  [Warning] Need at least 2 images in: {folder}")
        return None

    images = torch.stack(tensors, dim=0).to(device)  # [N, 3, H, W]
    images = images * 2.0 - 1.0  # LPIPS expects [-1, 1]
    return images


def compute_avg_pairwise_lpips(images, loss_fn, chunk_size=8):
    """
    Memory-friendly pairwise LPIPS using chunked computation.
    Only computes upper-triangle (a < b) to avoid redundancy.
    """
    N = images.shape[0]
    total_dist = 0.0
    total_pairs = 0

    num_chunks = (N + chunk_size - 1) // chunk_size
    total_iters = num_chunks * (num_chunks + 1) // 2
    pbar = tqdm(total=total_iters, desc="  Pairwise", leave=False)

    for i in range(0, N, chunk_size):
        batch_i = images[i : i + chunk_size]
        n_i = batch_i.shape[0]

        for j in range(i, N, chunk_size):
            batch_j = images[j : j + chunk_size]
            n_j = batch_j.shape[0]

            with torch.no_grad():
                if i == j:
                    # Same chunk: upper-triangle only (a < b)
                    for ii in range(n_i):
                        rem = n_j - ii - 1
                        if rem <= 0:
                            continue
                        img_a = batch_i[ii : ii + 1].expand(rem, -1, -1, -1)
                        img_b = batch_j[ii + 1 :]
                        dists = loss_fn(img_a, img_b)  # [rem, 1]
                        total_dist += dists.sum().item()
                        total_pairs += rem
                else:
                    # Different chunks: all cross pairs
                    for ii in range(n_i):
                        img_a = batch_i[ii : ii + 1].expand(n_j, -1, -1, -1)
                        dists = loss_fn(img_a, batch_j)  # [n_j, 1]
                        total_dist += dists.sum().item()
                        total_pairs += n_j

            pbar.update(1)

    pbar.close()
    return total_dist / total_pairs if total_pairs > 0 else 0.0


def main():
    print(f"Device: {DEVICE} | Backbone: {NET} | Image size: {IMAGE_SIZE}x{IMAGE_SIZE} | Chunk: {CHUNK_SIZE}")
    print("=" * 60)

    loss_fn = lpips.LPIPS(net=NET).to(DEVICE)
    loss_fn.eval()

    results = []
    for folder in FOLDER_PATHS:
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            print(f"[SKIP] Not a directory: {folder}")
            continue

        images = load_images(folder, image_size=IMAGE_SIZE, device=DEVICE)
        if images is None:
            continue

        print(f"Folder: {folder}  ({images.shape[0]} images)")
        avg_lpips = compute_avg_pairwise_lpips(images, loss_fn, chunk_size=CHUNK_SIZE)

        # Clean up
        del images
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        results.append((str(folder), avg_lpips))
        print(f"  Average Pairwise LPIPS: {avg_lpips:.6f}")
        print("-" * 60)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for folder, val in results:
        print(f"{val:.6f}  |  {folder}")


if __name__ == "__main__":
    main()