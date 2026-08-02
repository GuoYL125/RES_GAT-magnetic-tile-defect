"""Visualize SLIC segmentation + graph structure for a single arbitrary image.

Usage:
    python visualize_single_image.py <image_path> <output_dir> [desired_nodes]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from skimage.segmentation import slic, mark_boundaries
from PIL import Image

# Use a Chinese-capable font so Chinese filenames render in titles
for fp in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf',
           r'C:\Windows\Fonts\simsun.ttc', r'C:\Windows\Fonts\Deng.ttf']:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        try:
            plt.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=fp).get_name()]
        except Exception:
            pass
        break
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util  # reuse project's graph construction logic

NUM_FEATURES = 5  # RGB + x,y


def load_image(path):
    """Load image as HWC uint8 numpy array."""
    img = Image.open(path).convert('RGB')
    return np.asarray(img)


def build_graph_structure(img_hwc, segments):
    """Build nodes/edges exactly like util.get_graph_from_image but return the
    networkx graph for visualization."""
    import networkx as nx
    height, width = img_hwc.shape[:2]

    # nodes: collect per-segment pixels
    seg_ids = np.unique(segments)
    nodes = {sid: {'rgb_list': [], 'pos_list': []} for sid in seg_ids}
    for y in range(height):
        for x in range(width):
            sid = segments[y, x]
            nodes[sid]['rgb_list'].append(img_hwc[y, x, :])
            nodes[sid]['pos_list'].append(np.array([x / width, y / height]))

    G = nx.Graph()
    centroids = {}
    for sid in seg_ids:
        rgb = np.stack(nodes[sid]['rgb_list']).mean(axis=0)
        pos = np.stack(nodes[sid]['pos_list']).mean(axis=0)
        G.add_node(sid, features=np.concatenate([rgb.reshape(-1), pos.reshape(-1)]))
        # centroid in pixel coordinates
        ys, xs = np.nonzero(segments == sid)
        centroids[sid] = (xs.mean(), ys.mean())

    # edges: right + below neighbors
    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
    for i in range(bneighbors.shape[1]):
        a, b = int(bneighbors[0, i]), int(bneighbors[1, i])
        if a != b:
            G.add_edge(a, b)

    # self loops
    for sid in seg_ids:
        G.add_edge(sid, sid)

    return G, centroids


def visualize_slic(img_hwc, segments, save_path, img_name):
    num_segments = np.max(segments) + 1
    img_overlay = mark_boundaries(img_hwc, segments, color=(255, 0, 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'SLIC Superpixel Segmentation — {img_name}', fontsize=13)

    axes[0].imshow(img_hwc)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(img_overlay)
    axes[1].set_title(f'SLIC: {num_segments} segments')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] SLIC segmentation -> {save_path}")
    return num_segments


def visualize_graph(img_hwc, G, centroids, save_path, img_name):
    seg_ids = list(G.nodes)
    edges = [(a, b) for a, b in G.edges if a != b]  # exclude self-loops for drawing

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.suptitle(f'Image → Graph — {img_name}', fontsize=13)

    # faint background image
    ax.imshow(img_hwc, alpha=0.35)

    # edges
    for a, b in edges:
        xa, ya = centroids[a]
        xb, yb = centroids[b]
        ax.plot([xa, xb], [ya, yb], color='#888888', linewidth=0.4, alpha=0.7, zorder=1)

    # nodes
    cmap = plt.cm.get_cmap('tab20', len(seg_ids))
    for i, sid in enumerate(seg_ids):
        x, y = centroids[sid]
        ax.scatter(x, y, s=18, color=cmap(i % 20), edgecolors='black',
                   linewidths=0.3, zorder=2)

    ax.set_title(f'Graph: {len(seg_ids)} nodes, {len(edges)} edges (+self-loops)')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Graph structure -> {save_path}")
    return len(seg_ids), len(edges)


if __name__ == '__main__':
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'vis_output'
    desired_nodes = int(sys.argv[3]) if len(sys.argv) > 3 else 224

    if img_path is None:
        print("Usage: python visualize_single_image.py <image_path> <output_dir> [desired_nodes]")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    print(f"Loading image: {img_path}")
    img_hwc = load_image(img_path)
    print(f"  Shape: {img_hwc.shape}, dtype: {img_hwc.dtype}")

    print(f"Running SLIC (desired_nodes={desired_nodes})...")
    segments = slic(img_hwc, n_segments=desired_nodes, slic_zero=True, start_label=0)

    # 1) SLIC overlay
    n_seg = visualize_slic(
        img_hwc, segments,
        os.path.join(out_dir, f'{img_name}_slic.png'),
        img_name,
    )

    # 2) Graph structure (reuse the same node/edge logic as util.py)
    G, centroids = build_graph_structure(img_hwc, segments)
    n_nodes, n_edges = visualize_graph(
        img_hwc, G, centroids,
        os.path.join(out_dir, f'{img_name}_graph.png'),
        img_name,
    )

    print(f"\nDone! {n_seg} superpixels -> {n_nodes} nodes, {n_edges} edges (+self-loops)")
    print(f"Output saved to: {out_dir}")
