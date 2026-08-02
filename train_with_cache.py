"""Training script that uses precomputed graphs (no SLIC recomputation)."""
import os
import sys
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RES_GAT, FocalLoss
import util

CACHE_FILE = "precomputed_graphs.npz"


def load_precomputed():
    """Load precomputed graphs from npz file."""
    data = np.load(CACHE_FILE)
    print(f"Loaded precomputed data: {data['h'].shape[0]} samples")
    return data


class PrecomputedDataset(torch.utils.data.Dataset):
    """Dataset that returns precomputed graph data."""
    def __init__(self, data):
        self.h = data['h']           # (N, max_nodes, 5)
        self.h_mask = data['h_mask']  # (N, max_nodes) bool
        self.edges = data['edges']   # (N, max_edges, 2)
        self.edges_mask = data['edges_mask']  # (N, max_edges)
        self.imgs = data['imgs']     # (N, 3, 224, 224)
        self.labels = data['labels'] # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Filter nodes and edges to valid ones
        h_mask = self.h_mask[idx]
        h = self.h[idx][h_mask]
        e_mask = self.edges_mask[idx]
        edges = self.edges[idx][e_mask]
        img = self.imgs[idx]
        label = self.labels[idx]
        return (h, edges, img, label)


def collate_precomputed(batch):
    """Collate function for precomputed graphs - same logic as util.batch_graphs."""
    gs = [(h, e, img) for h, e, img, l in batch]
    labels = np.asarray([l for _, _, _, l in batch], dtype=np.int64)
    imgs = np.asarray([g[2] for g in gs])

    G = len(gs)
    N = sum(g[0].shape[0] for g in gs)
    M = sum(g[1].shape[0] for g in gs)
    adj = np.zeros([N, N])
    src = np.zeros([M])
    tgt = np.zeros([M])
    Msrc = np.zeros([N, M])
    Mtgt = np.zeros([N, M])
    Mgraph = np.zeros([N, G])
    h = np.concatenate([g[0] for g in gs])

    n_acc = 0
    m_acc = 0
    for g_idx, g in enumerate(gs):
        n = g[0].shape[0]
        m = g[1].shape[0]
        for e, (s, t) in enumerate(g[1]):
            adj[n_acc + s, n_acc + t] = 1
            adj[n_acc + t, n_acc + s] = 1
            src[m_acc + e] = n_acc + s
            tgt[m_acc + e] = n_acc + t
            Msrc[n_acc + s, m_acc + e] = 1
            Mtgt[n_acc + t, m_acc + e] = 1
        Mgraph[n_acc:n_acc + n, g_idx] = 1
        n_acc += n
        m_acc += m

    return (
        h.astype(np.float32),
        adj.astype(np.float32),
        src.astype(np.int64),
        tgt.astype(np.int64),
        Msrc.astype(np.float32),
        Mtgt.astype(np.float32),
        Mgraph.astype(np.float32),
        labels,
        imgs.astype(np.float32),
    )


def train_epoch(model, optimiser, loader, use_cuda):
    """Single training epoch."""
    losses, accs, acc1s, pres, recalls, f1s = [], [], [], [], [], []
    from sklearn import metrics

    for b in loader:
        optimiser.zero_grad()
        h, adj, src, tgt, Msrc, Mtgt, Mgraph, np_labels, imgs = b
        h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels, imgs = map(
            torch.from_numpy, (h, adj, src, tgt, Msrc, Mtgt, Mgraph, np_labels, imgs)
        )
        if use_cuda:
            h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels, imgs = map(
                lambda t: t.cuda(), (h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels, imgs)
            )

        y = model(h, adj, src, tgt, Msrc, Mtgt, Mgraph, imgs)
        loss = FocalLoss()(inputs=y, targets=pyt_labels)

        pred = torch.argmax(y, dim=1).detach().cpu().numpy()
        acc = np.sum((pred == np_labels).astype(float)) / pyt_labels.shape[0]
        pre = metrics.precision_score(np_labels, pred, average='micro', zero_division=0)
        acc1 = metrics.accuracy_score(np_labels, pred)
        recall = metrics.recall_score(np_labels, pred, average='micro', zero_division=0)
        f1 = metrics.f1_score(np_labels, pred, average='micro', zero_division=0)

        loss.backward()
        optimiser.step()

        losses.append(loss.detach().cpu().item())
        accs.append(acc)
        acc1s.append(acc1)
        pres.append(pre)
        recalls.append(recall)
        f1s.append(f1)

    return losses, accs, acc1s, pres, recalls, f1s


def test_epoch(model, loader, use_cuda, desc="Validation"):
    """Single evaluation epoch."""
    accs, acc1s, pres, recalls, f1s = [], [], [], [], []
    from sklearn import metrics

    for b in tqdm(loader, desc=desc, leave=False):
        with torch.no_grad():
            h, adj, src, tgt, Msrc, Mtgt, Mgraph, np_labels, imgs = b
            h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels, imgs = map(
                torch.from_numpy, (h, adj, src, tgt, Msrc, Mtgt, Mgraph, np_labels, imgs)
            )
            if use_cuda:
                h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels, imgs = map(
                    lambda t: t.cuda(), (h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels, imgs)
                )

            y = model(h, adj, src, tgt, Msrc, Mtgt, Mgraph, imgs)
            pred = torch.argmax(y, dim=1).detach().cpu().numpy()
            acc = np.sum((pred == np_labels).astype(float)) / np_labels.shape[0]
            pre = metrics.precision_score(np_labels, pred, average='micro', zero_division=0)
            acc1 = metrics.accuracy_score(np_labels, pred)
            recall = metrics.recall_score(np_labels, pred, average='micro', zero_division=0)
            f1 = metrics.f1_score(np_labels, pred, average='micro', zero_division=0)

            accs.append(acc)
            acc1s.append(acc1)
            pres.append(pre)
            recalls.append(recall)
            f1s.append(f1)

    return accs, acc1s, pres, recalls, f1s


def main(epochs=500, batch_size=8, use_cuda=True):
    use_cuda = use_cuda and torch.cuda.is_available()
    print(f"use_cuda: {use_cuda}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load precomputed data
    print("Loading precomputed graphs...")
    t0 = time.time()
    data = load_precomputed()
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Split dataset
    valid_split = 0.2
    dataset = PrecomputedDataset(data)
    n_total = len(dataset)
    n_valid = int(n_total * valid_split)
    n_train = n_total - n_valid
    dset_train, dset_valid = torch.utils.data.random_split(
        dataset, [n_train, n_valid],
        generator=torch.Generator().manual_seed(72),
    )

    train_loader = torch.utils.data.DataLoader(
        dset_train, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, collate_fn=collate_precomputed,
    )
    valid_loader = torch.utils.data.DataLoader(
        dset_valid, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=collate_precomputed,
    )

    print(f"Dataset: {n_total} total, {n_train} train, {n_valid} valid")
    print(f"Train batches/epoch: {len(train_loader)}, Valid batches/epoch: {len(valid_loader)}")

    # Model
    model = RES_GAT(num_features=util.NUM_FEATURES, num_classes=util.NUM_CLASSES)
    if use_cuda:
        model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    best_valid_acc = 0.0
    best_model = copy.deepcopy(model)

    # Training loop
    epoch_times = []
    for e in range(1, epochs + 1):
        t_epoch = time.time()

        train_losses, train_accs, train_acc1s, train_pres, train_recalls, train_f1s = \
            train_epoch(model, opt, train_loader, use_cuda)

        valid_accs, valid_acc1s, valid_pres, valid_recalls, valid_f1s = \
            test_epoch(model, valid_loader, use_cuda, desc=f"Val {e}/{epochs}")

        t_epoch = time.time() - t_epoch
        epoch_times.append(t_epoch)

        train_loss = np.mean(train_losses)
        train_acc = 100 * np.mean(train_accs)
        train_acc1 = 100 * np.mean(train_acc1s)
        train_pre = 100 * np.mean(train_pres)
        train_recall = 100 * np.mean(train_recalls)
        train_f1 = 100 * np.mean(train_f1s)

        valid_acc = 100 * np.mean(valid_accs)
        valid_acc1 = 100 * np.mean(valid_acc1s)
        valid_pre = 100 * np.mean(valid_pres)
        valid_recall = 100 * np.mean(valid_recalls)
        valid_f1 = 100 * np.mean(valid_f1s)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model)

        eta = np.mean(epoch_times[-10:]) * (epochs - e) / 60

        log = (f"E{e:3d}/{epochs} | Loss:{train_loss:.4f} | "
               f"TrA:{train_acc:.2f}% TrF1:{train_f1:.2f}% | "
               f"VAcc:{valid_acc:.2f}% VF1:{valid_f1:.2f}% | "
               f"ETA:{eta:.1f}min | "
               f"BestVAcc:{best_valid_acc:.2f}%")
        print(log)

    # Save results
    util.save_model("best_fixed", best_model)
    torch.save(model.state_dict(), "final_fixed.pt")
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_valid_acc:.2f}%")
    print(f"Total time: {sum(epoch_times)/60:.1f} minutes")
    print(f"Average epoch time: {np.mean(epoch_times):.1f}s")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
