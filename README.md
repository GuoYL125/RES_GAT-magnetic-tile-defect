# RES-GAT: Hybrid GAT-CNN for Surface Defect Classification

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)]()

A hybrid **Graph Attention Network (GAT) + ResNet18** model for **magnetic tile surface defect classification**. The model combines superpixel-based graph representations with global CNN features to achieve high-accuracy defect classification.

## Architecture

```
Input Image
  ├──→ ResNet18 → Global Image Features (512-dim)
  └──→ SLIC Superpixels → GAT → Graph Structure Features (128-dim)
                                ↓
                    Concatenate (640-dim)
                                ↓
                    FC Layer → Classification Result
```

- **GAT branch**: SLIC superpixels → graph construction (node features: RGB + position) → 3-layer multi-head GAT (1→4→2 heads, per-head dim: 32→64→64, concatenated: 32→256→128) → graph-level pooling (128-dim)
- **CNN branch**: ResNet18 backbone (randomly initialized, global feature extraction, 512-dim)
- **Fusion**: 128-dim (GAT) + 512-dim (ResNet) = 640-dim, then through a 3-layer MLP classifier

## Dataset

**Magnetic Tile Defect Dataset** ([source](https://github.com/lpj-github-io/Magnetic-tile-defect-datasets))

6 defect types with severe class imbalance:

| Class | Blowhole | Break | Crack | Fray | Uneven | Free (Normal) |
|-------|----------|-------|-------|------|--------|---------------|
| Count | 230 | 170 | 114 | 64 | 206 | 1904 |

## Results

### RES-GAT (GAT + ResNet18 hybrid)

| Metric | Best Epoch (475) | Final (500) |
|--------|-----------------|-------------|
| Accuracy | **99.40%** | 98.19% |
| F1 Score | 86.02% | **89.52%** |

Loss decreases from 0.9553 (epoch 1) to 0.0209 (epoch 500).

### Comparison: Pure GAT vs Hybrid

| Model | Best Accuracy | Convergence |
|-------|--------------|-------------|
| Pure GAT (superpixel only) | ~87% | ~400 epochs |
| **RES-GAT (GAT + ResNet18)** | **99.40%** | ~200 epochs |

## Installation

```bash
git clone https://github.com/yourusername/RES_GAT-magnetic-tile-defect.git
cd RES_GAT-magnetic-tile-defect
pip install -r requirements.txt
```

### Data Preparation

The preprocessed dataset is included in `data/batch_save_train_masks/`. To use the original raw dataset, download from [Magnetic-tile-defect-datasets](https://github.com/lpj-github-io/Magnetic-tile-defect-datasets) and preprocess using SLIC superpixel segmentation.

## Usage

### Train

```bash
python prototype.py --train=True --epochs=500 --batch_size=32 --use_cuda=True
```

### Quick Test (2 epochs)

```bash
python prototype.py --train=True --epochs=2 --batch_size=8 --use_cuda=True
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train` | True | Enable training |
| `--test` | False | Enable testing |
| `--epochs` | 500 | Number of epochs |
| `--batch_size` | 32 | Batch size |
| `--use_cuda` | True | Use GPU if available |

## Project Structure

```
├── prototype.py          # Main entry: training & evaluation
├── model.py              # Model definitions (RES_GAT, GAT_MNIST, CNN_GAT, etc.)
├── layers.py             # GAT layer implementations
├── models.py             # Standard GAT model (from pyGAT)
├── util.py               # Graph construction, data batching, train/test loops
├── my_dataset_class.py   # Custom dataset loader
├── data_process.py       # Alternative data loading utilities
├── pingjia.py            # Evaluation metrics
├── requirements.txt      # Dependencies
└── data/
    └── batch_save_train_masks/   # Preprocessed training data
```

## Model Variants

| Model | Description |
|-------|-------------|
| `RES_GAT` | **Main model** - ResNet18 + multi-head GAT hybrid (default) |
| `GAT_MNIST` | Pure GAT model (no CNN branch) |
| `CNN_GAT` | Simple CNN first, then GAT on features |
| `GAT_CNN` | GAT first, then CNN on features |
| `SimpleCNN` | Pure CNN baseline |

## Notes

- The model uses **Focal Loss** (`gamma=2`) to handle severe class imbalance
- Input images are resized to 224×224
- Superpixels are generated via SLIC (~75 segments/image, configurable)
- Node features: 5-dim (R, G, B + normalized x, y position)
- ResNet18 uses randomly initialized weights (`pretrained=False`)

## Citation

If you use this code in your research, please cite:

```
@software{res_gat_defect,
  author = {Guo Yilong},
  title = {RES-GAT: Hybrid GAT-CNN for Magnetic Tile Defect Classification},
  year = {2024},
  url = {https://github.com/yourusername/RES_GAT-magnetic-tile-defect}
}
```

## License

MIT
