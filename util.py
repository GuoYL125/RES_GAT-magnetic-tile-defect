import time
import numpy as np
import scipy as sp
import networkx as nx
from skimage.segmentation import slic
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn.functional as F

from model import FocalLoss


NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int64

NUM_FEATURES = 5   # RGB (3) + position (2)
NUM_CLASSES = 5


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(fname, model):
    torch.save(model.state_dict(), "{fname}.pt".format(fname=fname))


def load_model(fname, model):
    model.load_state_dict(torch.load("{fname}.pt".format(fname=fname)))


def to_cuda(x):
    return x.cuda()


# ---------------------------------------------------------------------------
# Graph construction from image (superpixel segmentation)
# ---------------------------------------------------------------------------

def get_graph_from_image(PIL_image, desired_nodes=224):
    """Convert a PIL image to a graph representation using SLIC superpixels.

    Returns:
        h: node features (RGB mean + position mean)
        edges: adjacency matrix edges
        image: raw image array
    """
    image = np.asarray(PIL_image)
    segments = slic(image, n_segments=desired_nodes, slic_zero=True, start_label=0)

    num_nodes = np.max(segments)
    nodes = {
        node: {"rgb_list": [], "pos_list": []}
        for node in range(num_nodes + 1)
    }

    height, width = image.shape[:2]
    for y in range(height):
        for x in range(width):
            node = segments[y, x]
            nodes[node]["rgb_list"].append(image[y, x, :])
            nodes[node]["pos_list"].append(np.array([float(x) / width, float(y) / height]))

    G = nx.Graph()
    for node in nodes:
        nodes[node]["rgb_list"] = np.stack(nodes[node]["rgb_list"])
        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
        rgb_mean = np.mean(nodes[node]["rgb_list"], axis=0)
        pos_mean = np.mean(nodes[node]["pos_list"], axis=0)
        features = np.concatenate([np.reshape(rgb_mean, -1), np.reshape(pos_mean, -1)])
        G.add_node(node, features=list(features))

    # Build edges from neighboring superpixels
    segments_ids = np.unique(segments)
    centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])
    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    for i in range(bneighbors.shape[1]):
        if bneighbors[0, i] != bneighbors[1, i]:
            G.add_edge(bneighbors[0, i], bneighbors[1, i])

    for node in nodes:
        G.add_edge(node, node)  # self-loops

    n = len(G.nodes)
    m = len(G.edges)
    h = np.zeros([n, NUM_FEATURES]).astype(NP_TORCH_FLOAT_DTYPE)
    edges = np.zeros([2 * m, 2]).astype(NP_TORCH_LONG_DTYPE)
    for e, (s, t) in enumerate(G.edges):
        edges[e, 0] = s
        edges[e, 1] = t
        edges[m + e, 0] = t
        edges[m + e, 1] = s

    for i in G.nodes:
        h[i, :] = G.nodes[i]["features"]

    image = image.reshape(3, 224, 224)
    return h, edges, image


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def batch_graphs(batch):
    """Collate function: combine graphs from a batch into block-diagonal matrices."""
    gs = [g for g, l in batch]
    labels = np.asarray([l for g, l in batch], dtype=NP_TORCH_LONG_DTYPE)
    imgs = np.asarray([gs[0][2] for g, l in batch])

    NUM_FEATURES = gs[0][0].shape[-1]
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
        h.astype(NP_TORCH_FLOAT_DTYPE),
        adj.astype(NP_TORCH_FLOAT_DTYPE),
        src.astype(NP_TORCH_LONG_DTYPE),
        tgt.astype(NP_TORCH_LONG_DTYPE),
        Msrc.astype(NP_TORCH_FLOAT_DTYPE),
        Mtgt.astype(NP_TORCH_FLOAT_DTYPE),
        Mgraph.astype(NP_TORCH_FLOAT_DTYPE),
        labels,
        imgs.astype(NP_TORCH_FLOAT_DTYPE),
    )


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train(model, optimiser, dataset_loader, use_cuda, batch_size=1, disable_tqdm=False, profile=False):
    train_losses, train_accs, train_acc1s = [], [], []
    train_pres, train_recalls, train_f1s = [], [], []

    for b in tqdm(dataset_loader, desc="Instances ", disable=disable_tqdm):
        optimiser.zero_grad()
        h, adj, src, tgt, Msrc, Mtgt, Mgraph, np_labels, imgs = b
        h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels, imgs = map(
            torch.from_numpy, (h, adj, src, tgt, Msrc, Mtgt, Mgraph, np_labels, imgs)
        )
        if use_cuda:
            h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels, imgs = map(
                to_cuda, (h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels, imgs)
            )

        y = model(h, adj, src, tgt, Msrc, Mtgt, Mgraph, imgs)
        loss = FocalLoss()(inputs=y, targets=pyt_labels)

        pred = torch.argmax(y, dim=1).detach().cpu().numpy()
        acc = np.sum((pred == np_labels).astype(float)) / pyt_labels.shape[0]
        pre = metrics.precision_score(np_labels, pred, average='micro')
        acc1 = metrics.accuracy_score(np_labels, pred)
        recall = metrics.recall_score(np_labels, pred, average='micro')
        f1 = metrics.f1_score(np_labels, pred, average='micro')

        loss.backward()
        optimiser.step()

        train_losses.append(loss.detach().cpu().item())
        train_accs.append(acc)
        train_acc1s.append(acc1)
        train_pres.append(pre)
        train_recalls.append(recall)
        train_f1s.append(f1)

    return train_losses, train_accs, train_acc1s, train_pres, train_recalls, train_f1s


def test(model, dataset_loader, use_cuda, desc="Test ", disable_tqdm=False):
    test_accs, test_acc1s = [], []
    test_pres, test_recalls, test_f1s = [], [], []

    for b in tqdm(dataset_loader, desc=desc, disable=disable_tqdm):
        with torch.no_grad():
            h, adj, src, tgt, Msrc, Mtgt, Mgraph, np_labels, imgs = b
            h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels, imgs = map(
                torch.from_numpy, (h, adj, src, tgt, Msrc, Mtgt, Mgraph, np_labels, imgs)
            )
            if use_cuda:
                h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels, imgs = map(
                    to_cuda, (h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels, imgs)
                )

            y = model(h, adj, src, tgt, Msrc, Mtgt, Mgraph, imgs)
            pred = torch.argmax(y, dim=1).detach().cpu().numpy()
            acc = np.sum((pred == np_labels).astype(float)) / np_labels.shape[0]
            pre = metrics.precision_score(np_labels, pred, average='micro')
            acc1 = metrics.accuracy_score(np_labels, pred)
            recall = metrics.recall_score(np_labels, pred, average='micro')
            f1 = metrics.f1_score(np_labels, pred, average='micro')

            test_accs.append(acc)
            test_acc1s.append(acc1)
            test_pres.append(pre)
            test_recalls.append(recall)
            test_f1s.append(f1)

    return test_accs, test_acc1s, test_pres, test_recalls, test_f1s
