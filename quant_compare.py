"""Quantitative comparison of buggy vs fixed model attention on the 25 samples."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util
from model import RES_GAT
import visualize_attention as va
import batch_visualize as bv

CLASSES = ['Blowhole', 'Break', 'Crack', 'Fray', 'Uneven']


def get_attention(model, img_chw):
    """Return last-layer node attention for one image."""
    img_hwc = img_chw.transpose(1, 2, 0).astype(np.uint8)
    h, edges, segments, centroids = va.build_single_graph(img_hwc)
    batch = va.batch_single_graph(h, edges)
    per_layer_att, pred = va.extract_attention(model, *batch, img_chw[None])
    att = np.asarray(per_layer_att[-1]).reshape(-1)
    return att, int(np.argmax(pred, axis=1)[0])


def main():
    images, labels = bv.load_correct_images()
    picked = bv.pick_samples(labels, 5)

    buggy = RES_GAT(num_features=util.NUM_FEATURES, num_classes=util.NUM_CLASSES)
    buggy.load_state_dict(torch.load('best_buggy.pt', map_location='cpu'))
    buggy.eval()
    fixed = RES_GAT(num_features=util.NUM_FEATURES, num_classes=util.NUM_CLASSES)
    fixed.load_state_dict(torch.load('best_fixed.pt', map_location='cpu'))
    fixed.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    buggy = buggy.to(device)
    fixed = fixed.to(device)

    print(f"{'class':<9}{'idx':>5} {'true':<9}{'buggy':<9}{'fixed':<9}{'att_corr':>9}")
    print('-' * 60)

    per_class_corr = {c: [] for c in range(5)}
    total_correct_buggy = 0
    total_correct_fixed = 0

    for gi, cls in picked:
        img = images[gi]
        att_b, pred_b = get_attention(buggy, img)
        att_f, pred_f = get_attention(fixed, img)
        corr = np.corrcoef(att_b, att_f)[0, 1] if len(att_b) == len(att_f) else float('nan')

        # only compare if node counts match
        if len(att_b) == len(att_f):
            per_class_corr[cls].append(corr)

        cb = pred_b == cls
        cf = pred_f == cls
        total_correct_buggy += int(cb)
        total_correct_fixed += int(cf)

        print(f"{CLASSES[cls]:<9}{gi:>5} {CLASSES[cls]:<9}{CLASSES[pred_b]:<9}{CLASSES[pred_f]:<9}{corr:>9.3f}")

    print('-' * 60)
    print(f"Correct predictions: buggy={total_correct_buggy}/25, fixed={total_correct_fixed}/25")

    print("\nAttention correlation per class (buggy vs fixed attention):")
    for c in range(5):
        vals = per_class_corr[c]
        if vals:
            print(f"  {CLASSES[c]:<9}: mean={np.mean(vals):.3f}, min={np.min(vals):.3f}, max={np.max(vals):.3f}")

    all_corr = [v for lst in per_class_corr.values() for v in lst]
    print(f"\nOverall mean attention correlation: {np.mean(all_corr):.3f}")
    print("(1.0 = identical attention, 0 = unrelated, <0 = opposite)")


if __name__ == '__main__':
    main()
