"""Side-by-side comparison of buggy vs fixed model visualizations + attention correlation."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

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
CUR = r"D:\Desktop\缺陷可视化\current"
FIX = r"D:\Desktop\缺陷可视化\fixed"
OUT = r"D:\Desktop\缺陷可视化\compare"


def main():
    os.makedirs(OUT, exist_ok=True)
    # For each class, find the 5 sample indices
    # (samples were picked in the same order in both runs)
    for cls in CLASSES:
        cur_files = sorted([f for f in os.listdir(os.path.join(CUR, cls)) if f.endswith('_heatmap.png')])
        fix_files = sorted([f for f in os.listdir(os.path.join(FIX, cls)) if f.endswith('_heatmap.png')])
        assert len(cur_files) == len(fix_files), f"{cls} count mismatch"

        for cf, ff in zip(cur_files, fix_files):
            sample = cf.replace('_heatmap.png', '')
            cur_img = np.asarray(Image.open(os.path.join(CUR, cls, cf)))
            fix_img = np.asarray(Image.open(os.path.join(FIX, cls, ff)))

            # side-by-side
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(cur_img)
            axes[0].set_title(f'{cls} — {sample}\nBugg y model')
            axes[0].axis('off')
            axes[1].imshow(fix_img)
            axes[1].set_title(f'{cls} — {sample}\nFixed model')
            axes[1].axis('off')
            plt.tight_layout()
            out_name = f'{cls}_{sample}_compare.png'
            plt.savefig(os.path.join(OUT, out_name), dpi=130, bbox_inches='tight')
            plt.close(fig)

    print(f"Side-by-side comparisons saved to: {OUT}")
    print(f"Total: {len(os.listdir(OUT))} comparison images")


if __name__ == '__main__':
    main()
