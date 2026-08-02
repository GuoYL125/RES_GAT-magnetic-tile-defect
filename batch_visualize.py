"""Batch visualize: SLIC / graph-attention / heatmap for N samples per class.

Usage:
    python batch_visualize.py <checkpoint.pt> <output_root> [samples_per_class] [use_cuda]
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from skimage.segmentation import slic, mark_boundaries
from PIL import Image

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util
from model import RES_GAT
import visualize_attention as va

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

CLASSES = ['Blowhole', 'Break', 'Crack', 'Fray', 'Uneven']


def load_correct_images():
    """Load correct 224x224 mask images + labels from batch files (transpose, not reshape)."""
    import pickle
    base = 'data/batch_save_train_masks/'
    files = [
        ('data_batch_MT_Blowhole_masks', 0),
        ('data_batch_MT_Crack_masks', 1),
        ('data_batch_MT_Break_masks', 2),
        ('data_batch_MT_Fray_masks', 3),
        ('data_batch_MT_Uneven_masks', 4),
    ]
    images = []   # (N, 3, 224, 224) CHW float
    labels = []
    for fname, label in files:
        with open(os.path.join(base, fname), 'rb') as f:
            e = pickle.load(f, encoding='latin1')
        data = e['data']  # (N, 150528)
        for i in range(data.shape[0]):
            img = data[i].reshape(3, 224, 224).astype(np.float32)  # correct CHW
            images.append(img)
            labels.append(label)
    return np.stack(images), np.array(labels)


def pick_samples(labels, per_class=5):
    """Return list of (global_idx, label) picking per_class per class."""
    picked = []
    for c in range(5):
        idxs = np.where(labels == c)[0]
        # spread across the class for variety
        step = max(1, len(idxs) // per_class)
        chosen = idxs[::step][:per_class]
        if len(chosen) < per_class:
            chosen = idxs[:per_class]  # fallback
        for gi in chosen:
            picked.append((int(gi), c))
    return picked


def build_and_visualize(model, img_chw, label, save_prefix):
    """Build graph from correct mask, run model, save 3 figures."""
    img_hwc = img_chw.transpose(1, 2, 0).astype(np.uint8)
    # SLIC + graph
    h, edges, segments, centroids = va.build_single_graph(img_hwc)
    N = h.shape[0]

    # batch
    batch = va.batch_single_graph(h, edges)
    imgs_t = torch.from_numpy(img_chw[None]).float()

    device = next(model.parameters()).device
    h_t, adj_t, src_t, tgt_t, Msrc_t, Mtgt_t, Mgraph_t = [torch.from_numpy(x).to(device) for x in batch]
    imgs_t = imgs_t.to(device)

    per_layer_att, pred = va.extract_attention(model, *batch, imgs_t.cpu().numpy())
    att = np.asarray(per_layer_att[-1]).reshape(-1)
    att_norm = (att - att.min()) / (att.max() - att.min() + 1e-9)
    pred_class = int(np.argmax(pred, axis=1)[0])

    # --- SLIC ---
    overlay = mark_boundaries(img_hwc, segments, color=(255, 0, 0))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(overlay)
    ax.set_title(f'{CLASSES[label]} — SLIC: {N} segments')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_prefix + '_slic.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    # --- Graph attention ---
    edge_pairs = [(int(e[0]), int(e[1])) for e in edges if e[0] != e[1]]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_hwc, alpha=0.35)
    for a, b in edge_pairs:
        xa, ya = centroids[a]
        xb, yb = centroids[b]
        ax.plot([xa, xb], [ya, yb], color='#bbbbbb', linewidth=0.4, alpha=0.6, zorder=1)
    sc = ax.scatter([centroids[i][0] for i in range(N)],
                    [centroids[i][1] for i in range(N)],
                    c=att_norm, cmap='inferno', norm=Normalize(0, 1),
                    s=80, edgecolors='white', linewidths=0.4, zorder=2)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='attention')
    # Red-circle only nodes with genuinely high attention (GAT attention is very
    # peaked, so top-k by rank would catch many zero-attention ties).
    threshold = 0.1 * att.max()
    sig_nodes = np.where(att > threshold)[0]
    for i in sig_nodes:
        x, y = centroids[int(i)]
        ax.scatter(x, y, s=110, facecolors='none', edgecolors='red', linewidths=1.2, zorder=3)
    ax.set_title(f'{CLASSES[label]} — Graph: {N} nodes / {len(edge_pairs)} edges (pred={CLASSES[pred_class]})')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_prefix + '_graph.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img_hwc)
    att_img = np.zeros(img_hwc.shape[:2])
    for i in range(N):
        att_img[segments == i] = att_norm[i]
    im = ax.imshow(att_img, cmap='jet', alpha=0.6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'{CLASSES[label]} — Attention heatmap (pred={CLASSES[pred_class]})')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_prefix + '_heatmap.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    return N, pred_class


def main():
    ckpt = sys.argv[1]
    out_root = sys.argv[2]
    per_class = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    use_cuda = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True

    print(f"Checkpoint: {ckpt}")
    print(f"Output: {out_root}")
    print(f"Samples/class: {per_class}")

    # Load model
    model = RES_GAT(num_features=util.NUM_FEATURES, num_classes=util.NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    device = 'cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu'
    model = model.to(device)
    print(f"Device: {device}")

    # Load correct images
    images, labels = load_correct_images()
    print(f"Loaded {len(labels)} images")

    # Pick samples
    picked = pick_samples(labels, per_class)
    print(f"Picked {len(picked)} samples")

    stats = {}
    t0 = time.time()
    for gi, cls in picked:
        class_dir = os.path.join(out_root, CLASSES[cls])
        os.makedirs(class_dir, exist_ok=True)
        prefix = os.path.join(class_dir, f'sample_{gi}')
        img_chw = images[gi]
        N, pred = build_and_visualize(model, img_chw, cls, prefix)
        key = f'{CLASSES[cls]}:{gi}'
        stats[key] = {'nodes': N, 'label': CLASSES[cls], 'pred': CLASSES[pred], 'correct': pred == cls}
        mark = 'OK' if pred == cls else 'XX'
        print(f"  [{CLASSES[cls]}] sample {gi}: {N} nodes, pred={CLASSES[pred]} [{mark}]")

    elapsed = time.time() - t0
    correct = sum(1 for s in stats.values() if s['correct'])
    print(f"\nDone in {elapsed:.0f}s. Correct predictions: {correct}/{len(stats)}")
    print(f"Output saved to: {out_root}")


if __name__ == '__main__':
    main()
