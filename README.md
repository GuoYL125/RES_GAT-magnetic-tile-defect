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

## Data Pipeline (Verified)

The training data in `data/batch_save_train_masks/` is **not** the original RGB photos — it is the **binary defect masks**, downscaled to 224×224.

```
Raw dataset (lpj-github-io/Magnetic-tile-defect-datasets)
  ├── MT_XXX/images/exp*_num_*.jpg      ← original photos (e.g. 327×502)
  └── MT_XXX/masks/exp*_num_*.png       ← binary masks (white = defect region)

Preprocessed → data/batch_save_train_masks/data_batch_MT_*_masks
  └── 224×224 binary mask (≈69% black background, ≈31% white defect)
```

**Key finding:** both model branches see the **mask**, not the original photo and not an
"Image × Mask" product. The `mask` pixel content (white=defect, black=background) is what
SLIC segments into superpixels and what ResNet18 extracts features from.

### ✅ Fixed: `get_graph_from_image` image channel conversion

`util.py` previously used:

```python
image = image.reshape(3, 224, 224)   # WRONG: scrambles channel layout
```

For an HWC `(224,224,3)` array this must be `image.transpose(2, 0, 1)`. The old `reshape`
interleaved RGB channels and produced a scrambled (blue-tinted) image fed to the ResNet18
branch. This has been **fixed** (`transpose(2, 0, 1)`) and the model retrained. Effects:

- The **GAT branch was never affected** — it uses graph node features `h` (mean RGB + position),
  computed correctly from the mask.
- The **ResNet18 branch now receives a clean channel-correct image**, improving best validation
  accuracy from 88.75% → 92.50%.

Note: both branches see the **binary mask**, not the original photo (see Key finding above).

## Results

### RES-GAT (GAT + ResNet18 hybrid, after reshape fix)

| Metric | Best Epoch (277) | Final (500) |
|--------|-----------------|-------------|
| Accuracy | **92.50%** | — |
| F1 Score | **92.50%** | — |

> Reproduced with the corrected `transpose` image conversion (see Data Pipeline).
> Best validation accuracy 92.50% at epoch 277; the model overfits if trained past ~epoch 350
> (loss climbs and validation accuracy collapses), so early stopping is recommended.

### Comparison: Pure GAT vs Hybrid

| Model | Best Accuracy | Convergence |
|-------|--------------|-------------|
| Pure GAT (superpixel only) | ~87% | ~400 epochs |
| **RES-GAT (GAT + ResNet18)** | **92.50%** | ~200 epochs |

### Before/after reshape fix

| Model | Best Validation Accuracy |
|-------|:-----------------------:|
| Buggy (reshape scrambles ResNet input) | 88.75% |
| **Fixed (transpose)** | **92.50%** (+3.75%) |

Fixing the channel-conversion bug improved accuracy by 3.75 points and made the two
models' attention maps essentially uncorrelated (Pearson ≈ -0.02), confirming the bug
was fundamentally changing what the model learned to attend to.

## Installation

```bash
git clone https://github.com/GuoYL125/RES_GAT-magnetic-tile-defect.git
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
├── prototype.py          # Original main entry: training & evaluation
├── train_with_cache.py   # Recommended training (precomputed SLIC graphs, faster)
├── precompute_graphs.py  # Precompute SLIC graphs once and cache to disk
├── model.py              # Model definitions (RES_GAT, GAT_MNIST, CNN_GAT, etc.)
├── layers.py             # GAT layer implementations
├── models.py             # Standard GAT model (from pyGAT)
├── util.py               # Graph construction, data batching, train/test loops
├── my_dataset_class.py   # Custom dataset loader
├── data_process.py       # Alternative data loading utilities
├── pingjia.py            # Evaluation metrics
├── visualize_slic_graph.py      # SLIC + graph visualization (dataset images)
├── visualize_single_image.py    # SLIC + graph visualization (arbitrary image)
├── visualize_attention.py       # Node coloring by GAT attention weight
├── analyze_attention_nodes.py   # Why high-attention nodes (feature analysis)
├── batch_visualize.py           # Batch SLIC/graph/heatmap for many samples
├── compare_visuals.py           # Side-by-side comparison of two models
├── quant_compare.py             # Quantitative attention correlation
├── analyze_results.py           # Parse training log + plot curves
├── requirements.txt      # Dependencies
└── data/
    └── batch_save_train_masks/   # Preprocessed training data (224×224 masks)
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
- Superpixels are generated via SLIC (`desired_nodes=224`, typically ~220 segments/image)
- Node features: 5-dim (R, G, B + normalized x, y position)
- ResNet18 uses randomly initialized weights (`pretrained=False`)
- **Break ↔ Crack are frequently confused** — these two defect classes are visually similar
  in mask space; consider merging them or adding more samples
- Training is run with `train_with_cache.py` (precomputed SLIC graphs) to avoid recomputing
  SLIC every epoch; use early stopping around epoch 300 to avoid the late-training collapse

## Citation

If you use this code in your research, please cite:

```
@software{res_gat_defect,
  author = {Guo Yilong},
  title = {RES-GAT: Hybrid GAT-CNN for Magnetic Tile Defect Classification},
  year = {2024},
  url = {https://github.com/GuoYL125/RES_GAT-magnetic-tile-defect}
}
```

## License

MIT
