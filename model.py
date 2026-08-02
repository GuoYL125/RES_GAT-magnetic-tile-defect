import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


# ---------------------------------------------------------------------------
# Simple CNN Baseline
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """Simple 3-layer CNN baseline for defect classification."""
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Linear(3 * 3 * 64, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        return self.out(x)


# ---------------------------------------------------------------------------
# GAT Layer Variants
# ---------------------------------------------------------------------------

class GATLayerAdj(nn.Module):
    """GAT layer using full adjacency matrix attention."""
    def __init__(self, d_i, d_o, act=F.relu, eps=1e-6):
        super(GATLayerAdj, self).__init__()
        self.f = nn.Linear(2*d_i, d_o)
        self.w = nn.Linear(2*d_i, 1)
        self.act = act
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, x, adj, src, tgt, Msrc, Mtgt):
        N = x.size()[0]
        hsrc = x.unsqueeze(0).expand(N, -1, -1)
        htgt = x.unsqueeze(1).expand(-1, N, -1)
        h = torch.cat([hsrc, htgt], dim=2)
        a = self.w(h)
        a_sqz = a.squeeze(2)
        a_zro = -1e16 * torch.ones_like(a_sqz)
        a_msk = torch.where(adj > 0, a_sqz, a_zro)
        a_att = F.softmax(a_msk, dim=1)
        y = self.act(self.f(h))
        y_att = a_att.unsqueeze(-1) * y
        return y_att.sum(dim=1).squeeze()


class GATLayerEdgeAverage(nn.Module):
    """GAT layer with average (instead of softmax) attention distribution."""
    def __init__(self, d_i, d_o, act=F.relu, eps=1e-6):
        super(GATLayerEdgeAverage, self).__init__()
        self.f = nn.Linear(2*d_i, d_o)
        self.w = nn.Linear(2*d_i, 1)
        self.act = act
        self.eps = eps
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, x, adj, src, tgt, Msrc, Mtgt):
        hsrc = x[src]
        htgt = x[tgt]
        h = torch.cat([hsrc, htgt], dim=1)
        y = self.act(self.f(h))
        a = self.w(h)
        a_sum = torch.mm(Mtgt, a) + self.eps
        return torch.mm(Mtgt, y * a) / a_sum


class GATLayerEdgeSoftmax(nn.Module):
    """GAT layer with softmax attention distribution over edges."""
    def __init__(self, d_i, d_o, act=F.relu, eps=1e-6):
        super(GATLayerEdgeSoftmax, self).__init__()
        self.f = nn.Linear(2*d_i, d_o)
        self.w = nn.Linear(2*d_i, 1)
        self.act = act
        self.eps = eps
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, x, adj, src, tgt, Msrc, Mtgt):
        hsrc = x[src]
        htgt = x[tgt]
        h = torch.cat([hsrc, htgt], dim=1)
        y = self.act(self.f(h))
        a = self.w(h)
        a_base, _ = torch.max(a, 0, keepdim=True)
        a_norm = a - a_base
        a_exp = torch.exp(a_norm)
        a_sum = torch.mm(Mtgt, a_exp) + self.eps
        return torch.mm(Mtgt, y * a_exp) / a_sum


class GATLayerMultiHead(nn.Module):
    """Multi-head GAT layer."""
    def __init__(self, d_in, d_out, num_heads):
        super(GATLayerMultiHead, self).__init__()
        self.GAT_heads = nn.ModuleList([
            GATLayerEdgeSoftmax(d_in, d_out) for _ in range(num_heads)
        ])

    def forward(self, x, adj, src, tgt, Msrc, Mtgt):
        return torch.cat([l(x, adj, src, tgt, Msrc, Mtgt) for l in self.GAT_heads], dim=1)


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ---------------------------------------------------------------------------
# Pure GAT Models
# ---------------------------------------------------------------------------

class GAT_MNIST(nn.Module):
    """Pure GAT model with multi-head attention and MLP classifier."""
    def __init__(self, num_features, num_classes, num_heads=[2, 2, 2]):
        super(GAT_MNIST, self).__init__()
        self.layer_heads = [1] + num_heads
        self.GAT_layer_sizes = [num_features, 32, 64, 64]
        self.MLP_layer_sizes = [self.layer_heads[-1] * self.GAT_layer_sizes[-1], 32, num_classes]
        self.MLP_acts = [F.relu, lambda x: x]

        self.GAT_layers = nn.ModuleList([
            GATLayerMultiHead(d_in * heads_in, d_out, heads_out)
            for d_in, d_out, heads_in, heads_out in zip(
                self.GAT_layer_sizes[:-1], self.GAT_layer_sizes[1:],
                self.layer_heads[:-1], self.layer_heads[1:],
            )
        ])
        self.MLP_layers = nn.ModuleList([
            nn.Linear(d_in, d_out)
            for d_in, d_out in zip(self.MLP_layer_sizes[:-1], self.MLP_layer_sizes[1:])
        ])

    def forward(self, x, adj, src, tgt, Msrc, Mtgt, Mgraph):
        for l in self.GAT_layers:
            x = l(x, adj, src, tgt, Msrc, Mtgt)
        x = torch.mm(Mgraph.t(), x)
        for layer, act in zip(self.MLP_layers, self.MLP_acts):
            x = act(layer(x))
        return x


# ---------------------------------------------------------------------------
# GAT-CNN Hybrid Models
# ---------------------------------------------------------------------------

class RES_GAT(nn.Module):
    """
    Hybrid GAT-CNN model for defect classification.

    Architecture:
    - GAT branch: processes superpixel graph structure (node features: RGB + position)
    - CNN branch: ResNet18 extracts global image features
    - Fusion: concatenate GAT (128-dim) + ResNet (512-dim) → 640-dim → classifier
    """
    def __init__(self, num_features, num_classes, num_heads=[1, 4, 2]):
        super(RES_GAT, self).__init__()

        self.layer_heads = [1] + num_heads
        self.GAT_layer_sizes = [num_features, 32, 64, 64]

        self.GAT_layers = nn.ModuleList([
            GATLayerMultiHead(d_in * heads_in, d_out, heads_out)
            for d_in, d_out, heads_in, heads_out in zip(
                self.GAT_layer_sizes[:-1], self.GAT_layer_sizes[1:],
                self.layer_heads[:-1], self.layer_heads[1:],
            )
        ])

        # ResNet18 backbone for global image features
        self.model_resnet18 = models.resnet18(pretrained=False)
        for param in self.model_resnet18.parameters():
            param.requires_grad = True
        self.layer_resnet = nn.Sequential(*list(self.model_resnet18.children())[:-1])

        # Classifier: 640-dim (128 GAT + 512 ResNet) -> num_classes
        self.fclinear = nn.Sequential(
            nn.Linear(640, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, adj, src, tgt, Msrc, Mtgt, Mgraph, imgs):
        # CNN branch: global features via ResNet18
        imgs = self.layer_resnet(imgs)
        imgs = imgs.view(imgs.shape[0], -1)

        # GAT branch: graph structure features
        for l in self.GAT_layers:
            x = l(x, adj, src, tgt, Msrc, Mtgt)
        x = torch.mm(Mgraph.t(), x)

        # Fusion and classification
        x = torch.cat([x, imgs], dim=1)
        x = self.fclinear(x)
        return F.log_softmax(x, dim=1)


class CNN_GAT(nn.Module):
    """Hybrid model: simple CNN first, then GAT on features."""
    def __init__(self, num_features, num_classes, num_heads=[1, 3, 2]):
        super(CNN_GAT, self).__init__()
        self.layer_heads = [1] + num_heads
        self.GAT_layer_sizes = [num_features, 32, 64, 64]

        self.GAT_layers = nn.ModuleList([
            GATLayerMultiHead(d_in * heads_in, d_out, heads_out)
            for d_in, d_out, heads_in, heads_out in zip(
                self.GAT_layer_sizes[:-1], self.GAT_layer_sizes[1:],
                self.layer_heads[:-1], self.layer_heads[1:],
            )
        ])

        # Simple CNN backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
        )

        self.fc1 = nn.Linear(3 * 3 * 64 + 128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 5)

    def forward(self, x, adj, src, tgt, Msrc, Mtgt, Mgraph, imgs):
        imgs = self.conv1(imgs)
        imgs = self.conv2(imgs)
        imgs = self.conv3(imgs)
        imgs = imgs.view(imgs.shape[0], -1)

        for l in self.GAT_layers:
            x = l(x, adj, src, tgt, Msrc, Mtgt)
        x = torch.mm(Mgraph.t(), x)
        x = torch.cat([x, imgs], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class GAT_CNN(nn.Module):
    """Hybrid model: GAT first, then CNN on features (deprecated, kept for reference)."""
    def __init__(self, num_features, num_classes, num_heads=[2, 2, 2]):
        super(GAT_CNN, self).__init__()
        self.layer_heads = [1] + num_heads
        self.GAT_layer_sizes = [num_features, 32, 64, 64]
        self.MLP_layer_sizes = [self.layer_heads[-1] * self.GAT_layer_sizes[-1] * 2, 32, num_classes]
        self.MLP_acts = [F.relu, lambda x: x]

        self.GAT_layers = nn.ModuleList([
            GATLayerMultiHead(d_in * heads_in, d_out, heads_out)
            for d_in, d_out, heads_in, heads_out in zip(
                self.GAT_layer_sizes[:-1], self.GAT_layer_sizes[1:],
                self.layer_heads[:-1], self.layer_heads[1:],
            )
        ])
        self.MLP_layers = nn.ModuleList([
            nn.Linear(d_in, d_out)
            for d_in, d_out in zip(self.MLP_layer_sizes[:-1], self.MLP_layer_sizes[1:])
        ])
        self.CNN = SimpleCNN()

    def forward(self, x, adj, src, tgt, Msrc, Mtgt, Mgraph, imgs):
        for l in self.GAT_layers:
            x = l(x, adj, src, tgt, Msrc, Mtgt)
        x = torch.mm(Mgraph.t(), x)
        cnn_output = self.CNN(imgs)
        x = torch.cat([x, cnn_output], dim=1)
        for layer, act in zip(self.MLP_layers, self.MLP_acts):
            x = act(layer(x))
        return x
