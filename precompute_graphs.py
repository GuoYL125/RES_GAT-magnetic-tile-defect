"""Precompute graphs from dataset and cache to disk.
This avoids running SLIC every epoch.
"""
import os
import sys
import time
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util

# Constants
NUM_FEATURES = 5
CACHE_FILE = "precomputed_graphs.npz"
DATA_ROOT = "./data"


def load_dataset_raw():
    """Load dataset without transform - returns raw PIL images and labels."""
    from my_dataset_class import Mydataset
    # Load without transform to get raw images
    dset = Mydataset(DATA_ROOT, train=True, transform=None)
    images = []
    labels = []
    for i in tqdm(range(len(dset)), desc="Loading raw data"):
        img, label = dset[i]
        images.append(img)
        labels.append(label)
    return images, labels


def precompute_graphs(images, labels):
    """Run get_graph_from_image on all images and cache results."""
    n = len(images)
    all_h = []
    all_edges = []
    all_imgs = []
    all_labels = []

    for i in tqdm(range(n), desc="Precomputing graphs (SLIC)"):
        h, edges, img_arr = util.get_graph_from_image(images[i])
        all_h.append(h)
        all_edges.append(edges)
        all_imgs.append(img_arr)
        all_labels.append(labels[i])

    # Save as arrays - pad both nodes and edges to same length
    max_nodes = max(h.shape[0] for h in all_h)
    max_edges = max(e.shape[0] for e in all_edges)

    h_array = np.zeros((n, max_nodes, NUM_FEATURES), dtype=np.float32)
    h_mask = np.zeros((n, max_nodes), dtype=bool)
    edges_array = np.zeros((n, max_edges, 2), dtype=np.int64)
    edges_mask = np.zeros((n, max_edges), dtype=bool)
    for i in range(n):
        ni = all_h[i].shape[0]
        h_array[i, :ni] = all_h[i]
        h_mask[i, :ni] = True
        ei = all_edges[i].shape[0]
        edges_array[i, :ei] = all_edges[i]
        edges_mask[i, :ei] = True

    labels_array = np.array(all_labels, dtype=np.int64)

    # Images: each is (3, 224, 224)
    img_array = np.stack(all_imgs)  # (N, 3, 224, 224)

    np.savez_compressed(
        CACHE_FILE,
        h=h_array,
        h_mask=h_mask,
        edges=edges_array,
        edges_mask=edges_mask,
        imgs=img_array,
        labels=labels_array,
    )

    n_nodes = [h.shape[0] for h in all_h]
    print(f"\nPrecomputed {n} graphs:")
    print(f"  Nodes per graph: min={min(n_nodes)}, max={max(n_nodes)}, avg={np.mean(n_nodes):.1f}")
    print(f"  Edges per graph: min={min(e.shape[0] for e in all_edges)}, max={max(e.shape[0] for e in all_edges)}")
    print(f"  Saved to: {CACHE_FILE}")
    print(f"  File size: {os.path.getsize(CACHE_FILE) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    print("Loading raw dataset (no transform)...")
    t0 = time.time()
    images, labels = load_dataset_raw()
    print(f"Loaded {len(images)} images in {time.time() - t0:.1f}s")

    print("\nPrecomputing graphs...")
    t0 = time.time()
    precompute_graphs(images, labels)
    print(f"Precomputation done in {time.time() - t0:.1f}s")
    print("\nNow run: python train_with_cache.py --train=True --epochs=500 --batch_size=8 --use_cuda=True")
