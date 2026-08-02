"""Visualize SLIC superpixel segmentation and the resulting graph structure.

Outputs:
    slic_segmentation.png  - SLIC segments overlaid on original image
    graph_structure.png    - graph with nodes (superpixel centroids) and edges
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util

NUM_COLORS = 10


def load_one_image(index=0):
    """Load a single raw image (HWC) from the dataset."""
    from my_dataset_class import Mydataset
    dset = Mydataset('./data', train=True, transform=None)
    img, label = dset[index]
    return np.asarray(img), int(label)


def visualize_slic(img_hwc, label, save_path='slic_segmentation.png'):
    """Run SLIC and overlay segment boundaries on the original image."""
    # SLIC on HWC image
    segments = slic(img_hwc, n_segments=224, slic_zero=True, start_label=0)
    num_segments = np.max(segments) + 1

    # Overlay boundaries
    img_overlay = mark_boundaries(img_hwc, segments, color=(255, 0, 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(f'SLIC Superpixel Segmentation (Image index={0}, Label={label})', fontsize=13)

    axes[0].imshow(img_hwc)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(img_overlay)
    axes[1].set_title(f'SLIC Segmentation ({num_segments} segments)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] SLIC segmentation saved to {save_path}")
    print(f"     {num_segments} superpixels generated")
    return segments


def visualize_graph(img_hwc, segments, label, save_path='graph_structure.png'):
    """Build the graph (nodes + edges) and draw it over the image."""
    height, width = img_hwc.shape[:2]

    # --- Compute node centroids (average pixel position per segment) ---
    seg_ids = np.unique(segments)
    centroids = {}
    for sid in seg_ids:
        ys, xs = np.nonzero(segments == sid)
        centroids[sid] = (xs.mean(), ys.mean())

    # --- Build edges: right & below neighbors (same logic as util.py) ---
    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    edges = set()
    for i in range(bneighbors.shape[1]):
        a, b = int(bneighbors[0, i]), int(bneighbors[1, i])
        if a != b:
            edges.add((min(a, b), max(a, b)))

    # --- Draw ---
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.suptitle(f'Image → Graph: Nodes = Superpixel Centroids, Edges = Spatial Neighbors', fontsize=12)

    # Background: faint image
    ax.imshow(img_hwc, alpha=0.35)

    # Draw edges as lines
    for a, b in edges:
        xa, ya = centroids[a]
        xb, yb = centroids[b]
        ax.plot([xa, xb], [ya, yb], color='#888888', linewidth=0.4, alpha=0.7, zorder=1)

    # Draw nodes as colored circles (color = node id)
    cmap = plt.cm.get_cmap('tab20', len(seg_ids))
    for i, sid in enumerate(seg_ids):
        x, y = centroids[sid]
        ax.scatter(x, y, s=18, color=cmap(i % 20), edgecolors='black',
                   linewidths=0.3, zorder=2)

    ax.set_title(f'Graph: {len(seg_ids)} nodes, {len(edges)} edges (+self-loops)')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Graph structure saved to {save_path}")
    print(f"     {len(seg_ids)} nodes, {len(edges)} edges")
    return edges


def visualize_node_features(segments, img_hwc, label, save_path='node_features.png'):
    """Visualize the 5-dim node features as a colored overlay (mean RGB per segment)."""
    height, width = img_hwc.shape[:2]
    seg_ids = np.unique(segments)

    # Reconstruct image where each pixel shows its segment's mean color
    recon = np.zeros_like(img_hwc)
    for sid in seg_ids:
        mask = segments == sid
        mean_rgb = img_hwc[mask].mean(axis=0)
        recon[mask] = mean_rgb

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(f'Node Features: Mean RGB + Centroid (x, y) per Superpixel', fontsize=13)

    axes[0].imshow(img_hwc)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(recon.astype(np.uint8))
    axes[1].set_title('Reconstructed from node mean-RGB')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Node features visualization saved to {save_path}")


if __name__ == '__main__':
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'vis_output'

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading image index {idx}...")
    img, label = load_one_image(idx)
    print(f"  Shape: {img.shape}, dtype: {img.dtype}, label: {label}")

    # 1) SLIC segmentation overlay
    segments = visualize_slic(
        img, label,
        save_path=os.path.join(out_dir, 'slic_segmentation.png'),
    )

    # 2) Graph structure
    visualize_graph(
        img, segments, label,
        save_path=os.path.join(out_dir, 'graph_structure.png'),
    )

    # 3) Node features (mean RGB reconstruction)
    visualize_node_features(
        segments, img, label,
        save_path=os.path.join(out_dir, 'node_features.png'),
    )

    print("\nAll visualizations saved to:", out_dir)
