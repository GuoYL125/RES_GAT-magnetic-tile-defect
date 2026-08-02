"""Visualize original image, mask annotation, overlay, and stored training data."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

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

ORIG_IMG = r"D:\Desktop\bishe\dataset\Magnetic-tile-defect-datasets.-master\MT_Uneven\images\exp5_num_45084.jpg"
MASK_IMG = r"D:\Desktop\bishe\dataset\Magnetic-tile-defect-datasets.-master\MT_Uneven\masks\exp5_num_45084.png"
OUT_DIR = r"D:\Desktop\磁瓦图_vis"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    orig = np.asarray(Image.open(ORIG_IMG).convert('RGB'))
    mask = np.asarray(Image.open(MASK_IMG).convert('L'))
    print(f"Original: {orig.shape}, dtype={orig.dtype}")
    print(f"Mask: {mask.shape}, dtype={mask.dtype}, "
          f"white(foreground)={np.sum(mask > 127)}px = {np.sum(mask > 127)/mask.size*100:.1f}%")

    # 1) Grid: Original | Mask | Overlay
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Original Magnetic Tile Image & Defect Mask (exp5_num_45084, class=Uneven)', fontsize=14)

    axes[0].imshow(orig)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'Defect Mask ({np.sum(mask>127)} px defect)')
    axes[1].axis('off')

    # Overlay: red mask boundary + translucent mask on original
    axes[2].imshow(orig)
    axes[2].imshow(mask, cmap='jet', alpha=0.4, interpolation='nearest')
    axes[2].set_title('Overlay (mask on original)')
    axes[2].axis('off')

    plt.tight_layout()
    grid_path = os.path.join(OUT_DIR, 'exp5_num_45084_orig_mask_overlay.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] grid -> {grid_path}")

    # 2) Mask boundary overlay (contour) on original - clean version
    from skimage.segmentation import find_boundaries
    from skimage.morphology import dilation
    boundary = find_boundaries(mask > 127, mode='outer')
    boundary = dilation(boundary)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(orig)
    ax.imshow(np.ma.masked_where(~boundary, boundary), cmap='Reds', alpha=0.9, interpolation='nearest')
    ax.set_title('Defect Region Boundary (red contour)')
    ax.axis('off')
    plt.tight_layout()
    contour_path = os.path.join(OUT_DIR, 'exp5_num_45084_boundary.png')
    plt.savefig(contour_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] boundary -> {contour_path}")

    # 3) The stored training data (binary mask used by the model)
    import pickle
    base = 'data/batch_save_train_masks/'
    with open(os.path.join(base, 'data_batch_MT_Uneven_masks'), 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
    data = entry['data']  # (103, 150528)
    fnames = entry['filenames']
    # find our file
    idx = fnames.index('exp5_num_45084.png')
    stored = data[idx].reshape(3, 224, 224).transpose(1, 2, 0)
    print(f"Stored training data for exp5_num_45084: index={idx}, shape={stored.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('What the MODEL actually sees (batch_save_train_masks)', fontsize=14)
    axes[0].imshow(stored)
    axes[0].set_title('Stored "data" (looks like binary mask)')
    axes[0].axis('off')
    # grayscale view
    stored_gray = stored.mean(axis=2)
    axes[1].imshow(stored_gray, cmap='gray')
    axes[1].set_title('Grayscale view')
    axes[1].axis('off')
    plt.tight_layout()
    stored_path = os.path.join(OUT_DIR, 'exp5_num_45084_stored_training.png')
    plt.savefig(stored_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] stored -> {stored_path}")

    print("\nAll saved to:", OUT_DIR)


if __name__ == '__main__':
    main()
