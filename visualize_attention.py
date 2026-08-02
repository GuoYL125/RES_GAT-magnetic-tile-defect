"""Visualize graph nodes colored by GAT attention weights.

Runs the trained RES_GAT model on an image, extracts per-node attention
coefficients from the GAT layers via forward hooks, and colors the graph
nodes by how much attention they receive.

Usage:
    python visualize_attention.py <image_path> <output_dir> [desired_nodes]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from skimage.segmentation import slic
from PIL import Image

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util
from model import RES_GAT

# Chinese font
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


def build_single_graph(img_hwc):
    """Build (h, edges, image) exactly like util.get_graph_from_image,
    but return the segmentation map too for centroid computation."""
    from skimage.segmentation import slic
    height, width = img_hwc.shape[:2]

    # resize to 224x224 first? The model expects 224. We handle resize before calling.
    desired_nodes = 224
    segments = slic(img_hwc, n_segments=desired_nodes, slic_zero=True, start_label=0)

    num_nodes = np.max(segments)
    nodes = {node: {'rgb_list': [], 'pos_list': []} for node in range(num_nodes + 1)}
    for y in range(height):
        for x in range(width):
            node = segments[y, x]
            nodes[node]['rgb_list'].append(img_hwc[y, x, :])
            nodes[node]['pos_list'].append(np.array([x / width, y / height]))

    h = np.zeros([num_nodes + 1, util.NUM_FEATURES]).astype(np.float32)
    centroids = {}
    for node in nodes:
        rgb = np.mean(np.stack(nodes[node]['rgb_list']), axis=0)
        pos = np.mean(np.stack(nodes[node]['pos_list']), axis=0)
        h[node, :] = np.concatenate([rgb.reshape(-1), pos.reshape(-1)])
        ys, xs = np.nonzero(segments == node)
        centroids[node] = (xs.mean(), ys.mean())

    # edges
    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
    edge_pairs = set()
    for i in range(bneighbors.shape[1]):
        a, b = int(bneighbors[0, i]), int(bneighbors[1, i])
        if a != b:
            edge_pairs.add((a, b))
    # include self loops
    for node in range(num_nodes + 1):
        edge_pairs.add((node, node))

    edges = np.array(sorted(edge_pairs)).astype(np.int64)  # (m, 2)
    return h, edges, segments, centroids


def batch_single_graph(h, edges):
    """Collate a single graph into the model's expected batched tensors."""
    N = h.shape[0]
    M = edges.shape[0]
    adj = np.zeros([N, N])
    src = np.zeros([M])
    tgt = np.zeros([M])
    Msrc = np.zeros([N, M])
    Mtgt = np.zeros([N, M])
    Mgraph = np.zeros([N, 1])
    for e, (s, t) in enumerate(edges):
        adj[s, t] = 1
        adj[t, s] = 1
        src[e] = s
        tgt[e] = t
        Msrc[s, e] = 1
        Mtgt[t, e] = 1
    Mgraph[:, 0] = 1

    return (h.astype(np.float32),
            adj.astype(np.float32),
            src.astype(np.int64),
            tgt.astype(np.int64),
            Msrc.astype(np.float32),
            Mtgt.astype(np.float32),
            Mgraph.astype(np.float32))


def extract_attention(model, h, adj, src, tgt, Msrc, Mtgt, Mgraph, imgs):
    """Run model forward, capturing per-node attention per GAT layer.

    Returns: list of (layer_idx, node_attention) where node_attention[i]
    = total incoming softmax attention for node i (averaged over heads).
    """
    device = next(model.parameters()).device
    tensors = [torch.from_numpy(t) for t in (h, adj, src, tgt, Msrc, Mtgt, Mgraph)]
    h_t, adj_t, src_t, tgt_t, Msrc_t, Mtgt_t, Mgraph_t = [t.to(device) for t in tensors]
    imgs_t = torch.from_numpy(imgs).float().to(device)

    captured = []  # list of node-attention arrays per GAT multi-head layer

    def make_hook(accumulator):
        def hook(module, input, output):
            # module is GATLayerEdgeSoftmax
            x, adj_, src_, tgt_, Msrc_, Mtgt_ = input
            hsrc = x[src_]
            htgt = x[tgt_]
            hcat = torch.cat([hsrc, htgt], dim=1)
            a = module.w(hcat)  # (M,1) raw scores
            a_base, _ = torch.max(a, 0, keepdim=True)
            a_norm = a - a_base
            a_exp = torch.exp(a_norm)
            a_sum = torch.mm(Mtgt_, a_exp) + module.eps  # (N,1)
            att = (a_exp / (a_sum[tgt_] + module.eps))   # normalized per target node
            # accumulate attention by target node
            n_nodes = x.shape[0]
            node_att = torch.zeros(n_nodes, device=x.device)
            node_att = node_att.scatter_add(0, tgt_, att.squeeze(1))
            accumulator.append(node_att.detach().cpu().numpy())
        return hook

    hooks = []
    layer_accumulators = []
    # For each multi-head GAT layer, group its head accumulators
    head_accs = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod is not getattr(model, 'fclinear', None):
            pass
    # Simpler: iterate the model's GAT_layers structure
    gat_layers = model.GAT_layers  # ModuleList of GATLayerMultiHead
    layer_idx = 0
    for mh in gat_layers:
        accums = []
        for head in mh.GAT_heads:  # list of GATLayerEdgeSoftmax
            acc = []
            hk = head.register_forward_hook(make_hook(acc))
            hooks.append(hk)
            accums.append(acc)
        layer_accumulators.append(accums)

    with torch.no_grad():
        out = model(h_t, adj_t, src_t, tgt_t, Msrc_t, Mtgt_t, Mgraph_t, imgs_t)

    for hk in hooks:
        hk.remove()

    # Average heads within each layer
    per_layer = []
    for accums in layer_accumulators:
        stacked = np.stack([a for a in accums if len(a) > 0])  # (heads, N)
        if stacked.size == 0:
            continue
        if stacked.ndim == 1:
            per_layer.append(stacked)
        else:
            per_layer.append(stacked.mean(axis=0))

    return per_layer, out.detach().cpu().numpy()


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'vis_output'
    desired_nodes = int(sys.argv[3]) if len(sys.argv) > 3 else 224

    if img_path is None:
        print("Usage: python visualize_attention.py <image_path> <output_dir> [desired_nodes]")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    # Load and resize to 224x224 (model expects this)
    pil = Image.open(img_path).convert('RGB')
    pil_resized = pil.resize((224, 224), Image.BILINEAR)
    img_hwc = np.asarray(pil_resized)

    print(f"Image: {img_path} -> resized to {img_hwc.shape}")

    # Build graph
    h, edges, segments, centroids = build_single_graph(img_hwc)
    N = h.shape[0]
    print(f"Graph: {N} nodes, {edges.shape[0]} edges")

    # Batch + run model
    batch = batch_single_graph(h, edges)
    imgs = img_hwc.transpose(2, 0, 1).astype(np.float32)  # CHW
    imgs = imgs[None]  # (1,3,224,224)

    # Load model
    model = RES_GAT(num_features=util.NUM_FEATURES, num_classes=util.NUM_CLASSES)
    state = torch.load('best.pt', map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded best.pt ({sum(p.numel() for p in model.parameters()):,} params)")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        print("Running on GPU")
    else:
        print("Running on CPU")

    per_layer_att, pred = extract_attention(model, *batch, imgs)
    print(f"Prediction: class {np.argmax(pred, axis=1)[0]}, probs: {np.round(pred[0], 3)}")

    # Use LAST GAT layer's attention for coloring
    att = np.asarray(per_layer_att[-1]).reshape(-1)  # (N,)  force 1D
    print(f"Attention per-layer shapes: {[np.asarray(a).shape for a in per_layer_att]}")
    # Normalize to [0,1]
    att_norm = (att - att.min()) / (att.max() - att.min() + 1e-9)

    # ---- Plot 1: SLIC segmentation ----
    from skimage.segmentation import mark_boundaries
    overlay = mark_boundaries(img_hwc, segments, color=(255, 0, 0))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'SLIC Superpixel Segmentation — {img_name}', fontsize=13)
    axes[0].imshow(img_hwc)
    axes[0].set_title('Original (resized 224×224)')
    axes[0].axis('off')
    axes[1].imshow(overlay)
    axes[1].set_title(f'SLIC: {N} segments')
    axes[1].axis('off')
    plt.tight_layout()
    slic_path = os.path.join(out_dir, f'{img_name}_slic.png')
    plt.savefig(slic_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] SLIC -> {slic_path}")

    # ---- Plot 2: Graph colored by attention ----
    edge_pairs = [(int(e[0]), int(e[1])) for e in edges if e[0] != e[1]]

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.suptitle(f'Graph Nodes Colored by GAT Attention — {img_name}', fontsize=13)

    ax.imshow(img_hwc, alpha=0.25)

    # edges (thin gray)
    for a, b in edge_pairs:
        xa, ya = centroids[a]
        xb, yb = centroids[b]
        ax.plot([xa, xb], [ya, yb], color='#bbbbbb', linewidth=0.4, alpha=0.6, zorder=1)

    # nodes colored by attention (inferno: dark=low, yellow=bright=high attention)
    cmap = plt.cm.inferno
    norm = Normalize(vmin=0, vmax=1)
    sc = ax.scatter([centroids[i][0] for i in range(N)],
                    [centroids[i][1] for i in range(N)],
                    c=att_norm, cmap=cmap, norm=norm,
                    s=90, edgecolors='white', linewidths=0.4, zorder=2)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized GAT attention weight')

    # Highlight nodes with genuinely high attention (avoid zero-tie artifacts
    # from rank-based top-k, since GAT attention is extremely peaked).
    threshold = 0.1 * att.max()
    sig_nodes = np.where(att > threshold)[0]
    for i in sig_nodes:
        x, y = centroids[int(i)]
        ax.scatter(x, y, s=120, facecolors='none', edgecolors='red', linewidths=1.2, zorder=3)

    ax.set_title(f'{N} nodes — color=attention (red ring = att > {threshold:.2f})')
    ax.axis('off')
    plt.tight_layout()
    graph_path = os.path.join(out_dir, f'{img_name}_graph_attention.png')
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Attention graph -> {graph_path}")

    # ---- Plot 3: attention map on the image (smooth) ----
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_hwc)
    # build attention image by flooding each superpixel with its attention value
    att_img = np.zeros(img_hwc.shape[:2])
    for i in range(N):
        mask = segments == i
        att_img[mask] = att_norm[i]
    im = ax.imshow(att_img, cmap='jet', alpha=0.55)
    ax.set_title('Attention heatmap over superpixels (last GAT layer)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    heat_path = os.path.join(out_dir, f'{img_name}_attention_heatmap.png')
    plt.savefig(heat_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Attention heatmap -> {heat_path}")

    print("\nAll visualizations saved to:", out_dir)


if __name__ == '__main__':
    main()
