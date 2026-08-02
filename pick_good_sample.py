"""Find a training image that the trained model classifies correctly with high confidence."""
import os
import sys
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util
from model import RES_GAT
from my_dataset_class import Mydataset

CLASS_NAMES = ['Blowhole', 'Break', 'Crack', 'Fray', 'Uneven']


def main():
    model = RES_GAT(num_features=util.NUM_FEATURES, num_classes=util.NUM_CLASSES)
    model.load_state_dict(torch.load('best.pt', map_location='cpu'))
    model.eval()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    # Use precomputed graphs to avoid re-running SLIC on every sample
    import train_with_cache as twc
    cache = np.load(twc.CACHE_FILE)
    N_samples = cache['h'].shape[0]
    labels_all = cache['labels']
    imgs_all = cache['imgs']
    h_all = cache['h']
    h_mask_all = cache['h_mask']
    edges_all = cache['edges']
    edges_mask_all = cache['edges_mask']

    print(f"Precomputed dataset size: {N_samples}")

    results = []
    correct_defects = []
    for idx in range(N_samples):
        label = int(labels_all[idx])
        imgs_arr = imgs_all[idx]            # (3,224,224)
        h = h_all[idx][h_mask_all[idx]]      # (N, 5)
        e_mask = edges_mask_all[idx]
        edges = edges_all[idx][e_mask]      # (M, 2)

        # batch single graph
        N = h.shape[0]
        M = edges.shape[0]
        adj = np.zeros([N, N], dtype=np.float32)
        src = np.zeros([M], dtype=np.int64)
        tgt = np.zeros([M], dtype=np.int64)
        Msrc = np.zeros([N, M], dtype=np.float32)
        Mtgt = np.zeros([N, M], dtype=np.float32)
        Mgraph = np.zeros([N, 1], dtype=np.float32)
        for e, (s, t) in enumerate(edges):
            adj[s, t] = 1
            adj[t, s] = 1
            src[e] = s
            tgt[e] = t
            Msrc[s, e] = 1
            Mtgt[t, e] = 1
        Mgraph[:, 0] = 1

        tensors = [torch.from_numpy(x) for x in (h, adj, src, tgt, Msrc, Mtgt, Mgraph)]
        imgs_t = torch.from_numpy(imgs_arr[None]).float()  # add batch dim

        with torch.no_grad():
            if use_cuda:
                tensors = [t.cuda() for t in tensors]
                imgs_t = imgs_t.cuda()
            out = model(tensors[0], tensors[1], tensors[2], tensors[3],
                        tensors[4], tensors[5], tensors[6], imgs_t)
            probs = torch.softmax(out, dim=1)[0]
            pred = int(torch.argmax(probs).item())

        correct = (pred == label)
        conf = float(probs[pred])
        results.append((idx, int(label), pred, correct, conf))

        if correct and int(label) > 0:
            correct_defects.append((idx, int(label), conf))

    print(f"\nTotal: {len(results)}")
    print(f"Correct: {sum(1 for r in results if r[3])}/{len(results)} "
          f"({sum(1 for r in results if r[3])/len(results)*100:.1f}%)")
    print(f"Correct DEFECT samples (label>0): {len(correct_defects)}")

    # Per-class stats
    print("\nPer-class:")
    for c in range(5):
        cls_results = [r for r in results if r[1] == c]
        cls_correct = [r for r in cls_results if r[3]]
        if cls_results:
            print(f"  {CLASS_NAMES[c]}: {len(cls_correct)}/{len(cls_results)} "
                  f"({len(cls_correct)/len(cls_results)*100:.0f}%) "
                  f"avg_conf={np.mean([r[4] for r in cls_correct]):.3f}")

    # Pick the best defect sample (highest confidence correct defect)
    if correct_defects:
        correct_defects.sort(key=lambda r: -r[2])
        best = correct_defects[0]
        print(f"\nBest correct defect sample: idx={best[0]}, "
              f"class={CLASS_NAMES[best[1]]}, conf={best[2]:.3f}")
        # Save the image for visualization (imgs_all is CHW float; convert back)
        img_arr = imgs_all[best[0]].transpose(1, 2, 0)  # HWC
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        Image.fromarray(img_arr).save(f"sample_defect_{CLASS_NAMES[best[1]]}_{best[0]}.jpg")
        print(f"Saved: sample_defect_{CLASS_NAMES[best[1]]}_{best[0]}.jpg")


if __name__ == '__main__':
    main()
