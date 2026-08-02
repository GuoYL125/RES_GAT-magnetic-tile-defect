"""Analyze WHY certain nodes get high GAT attention.

Compares features (position, RGB, region size) of high-attention vs
low-attention nodes to find what the model attends to.
"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util
from model import RES_GAT
import visualize_attention as va

IMG = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\24353\Pictures\Saved Pictures\微信图片_20220411090513.jpg"


def main():
    pil = Image.open(IMG).convert("RGB").resize((224, 224))
    img_hwc = np.asarray(pil)
    h, edges, segments, centroids = va.build_single_graph(img_hwc)
    N = h.shape[0]

    # Run model & get attention
    batch = va.batch_single_graph(h, edges)
    imgs = img_hwc.transpose(2, 0, 1).astype(np.float32)[None]

    model = RES_GAT(num_features=util.NUM_FEATURES, num_classes=util.NUM_CLASSES)
    model.load_state_dict(torch.load('best.pt', map_location='cpu'))
    model.eval()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    per_layer_att, pred = va.extract_attention(model, *batch, imgs)
    att = np.asarray(per_layer_att[-1]).reshape(-1)

    print(f"N nodes: {N}, attention range: [{att.min():.4f}, {att.max():.4f}]")
    print(f"attention mean: {att.mean():.4f}, std: {att.std():.4f}")

    # Top / bottom nodes
    k = 15
    top_idx = np.argsort(att)[::-1][:k]
    bot_idx = np.argsort(att)[:k]

    # node features: h[:, :3] = RGB (0-255), h[:, 3:] = normalized x,y
    rgb = h[:, :3]
    pos = h[:, 3:]

    def describe(indices, label):
        print(f"\n{'='*60}")
        print(f"  {label} ({len(indices)} nodes)")
        print(f"{'='*60}")
        print(f"  {'node':>5} {'att':>8} {'R':>4} {'G':>4} {'B':>4} {'x':>5} {'y':>5} {'bright':>7}")
        for i in indices:
            a = att[i]
            r, g, b = rgb[i]
            x, y = pos[i]
            bright = (r + g + b) / 3
            print(f"  {i:>5} {a:>8.4f} {r:>4.0f} {g:>4.0f} {b:>4.0f} {x:>5.3f} {y:>5.3f} {bright:>7.0f}")

    describe(top_idx, "TOP attention nodes")
    describe(bot_idx, "BOTTOM attention nodes")

    # Compare statistics
    top_rgb = rgb[top_idx]
    bot_rgb = rgb[bot_idx]
    top_pos = pos[top_idx]
    bot_pos = pos[bot_idx]

    print(f"\n{'='*60}")
    print("  STATISTICAL COMPARISON")
    print(f"{'='*60}")
    print(f"  Mean brightness  top={top_rgb.mean():.1f}  bottom={bot_rgb.mean():.1f}  delta={top_rgb.mean()-bot_rgb.mean():+.1f}")
    print(f"  Mean R           top={top_rgb[:,0].mean():.1f}  bottom={bot_rgb[:,0].mean():.1f}")
    print(f"  Mean G           top={top_rgb[:,1].mean():.1f}  bottom={bot_rgb[:,1].mean():.1f}")
    print(f"  Mean B           top={top_rgb[:,2].mean():.1f}  bottom={bot_rgb[:,2].mean():.1f}")
    print(f"  Mean x (center)  top={top_pos[:,0].mean():.3f}  bottom={bot_pos[:,0].mean():.3f}")
    print(f"  Mean y (center)  top={top_pos[:,1].mean():.3f}  bottom={bot_pos[:,1].mean():.3f}")

    # spatial distribution
    from skimage.segmentation import mark_boundaries
    hgt, wdt = img_hwc.shape[:2]
    top_mask = np.zeros((hgt, wdt), bool)
    bot_mask = np.zeros((hgt, wdt), bool)
    for i in top_idx:
        top_mask[segments == i] = True
    for i in bot_idx:
        bot_mask[segments == i] = True
    print(f"  Top nodes cover {top_mask.sum()/top_mask.size*100:.1f}% of image area")
    print(f"  Bottom nodes cover {bot_mask.sum()/bot_mask.size*100:.1f}% of image area")

    # region sizes
    sizes = []
    for sid in range(N):
        sizes.append((segments == sid).sum())
    sizes = np.array(sizes)
    print(f"\n  Superpixel sizes: mean={sizes.mean():.0f}px, min={sizes.min():.0f}, max={sizes.max():.0f}")
    print(f"  Top-att nodes avg size: {sizes[top_idx].mean():.0f}px")
    print(f"  Bot-att nodes avg size: {sizes[bot_idx].mean():.0f}px")

    # correlation between attention and size/brightness
    corr_size = np.corrcoef(att, sizes)[0, 1]
    bright_all = rgb.mean(axis=1)
    corr_bright = np.corrcoef(att, bright_all)[0, 1]
    print(f"\n  Corr(attention, region_size): {corr_size:+.3f}")
    print(f"  Corr(attention, brightness):  {corr_bright:+.3f}")


if __name__ == '__main__':
    import torch
    main()
