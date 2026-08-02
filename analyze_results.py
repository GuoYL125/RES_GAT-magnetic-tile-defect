"""Analyze training results from logs and generate comprehensive report."""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

LOG_FILE = Path("training_output.log")
BEST_MODEL = Path("best.pt")
HISTORY_FILE = Path("training_history.npz")


def parse_log(log_path):
    """Parse training log lines into structured arrays."""
    if not log_path.exists():
        print(f"[WARN] Log file not found: {log_path}")
        return None

    try:
        text = log_path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            text = log_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            text = log_path.read_text(encoding='gbk', errors='replace')
    # Strip BOM if present
    if text.startswith('﻿'):
        text = text[1:]
    lines = text.strip().split("\n")
    epochs = []
    for line in lines:
        # Format: E  1/500 | Loss:4.2306 | TrA:38.29% TrF1:38.29% | VAcc:46.25% VF1:46.25% | ETA:218.7min | BestVAcc:46.25%
        m = re.match(r'E\s*\d+/\d+\s+\|\s+Loss:([\d.]+)\s+\|\s+TrA:([\d.]+)%\s+TrF1:([\d.]+)%\s+\|\s+VAcc:([\d.]+)%\s+VF1:([\d.]+)%', line)
        if m:
            epochs.append({
                "loss": float(m.group(1)),
                "train_acc": float(m.group(2)),
                "train_f1": float(m.group(3)),
                "valid_acc": float(m.group(4)),
                "valid_f1": float(m.group(5)),
                "train_acc1": float(m.group(2)),  # same as train_acc in this format
                "train_pre": float(m.group(2)),
                "train_recall": float(m.group(2)),
                "valid_acc1": float(m.group(4)),
                "valid_pre": float(m.group(4)),
                "valid_recall": float(m.group(4)),
            })
    return epochs


def generate_report(epochs):
    """Print comprehensive training report."""
    if not epochs:
        print("No training data found.")
        return

    n = len(epochs)
    print(f"\n{'='*60}")
    print(f"        TRAINING COMPLETE: {n} EPOCHS")
    print(f"{'='*60}")

    # Best epoch by validation accuracy
    best = max(epochs, key=lambda e: e["valid_acc"])
    best_idx = epochs.index(best) + 1
    final = epochs[-1]
    first = epochs[0]

    print(f"\n{'─'*60}")
    print(f"  FIRST EPOCH (1)")
    print(f"{'─'*60}")
    print(f"    Loss:      {first['loss']:.4f}")
    print(f"    Train Acc: {first['train_acc']:.2f}%")
    print(f"    Valid Acc: {first['valid_acc']:.2f}%")

    print(f"\n{'─'*60}")
    print(f"  BEST EPOCH ({best_idx})")
    print(f"{'─'*60}")
    print(f"    Loss:      {best['loss']:.4f}")
    print(f"    Train Acc:      {best['train_acc']:.2f}%")
    print(f"    Train Acc1:     {best['train_acc1']:.2f}%")
    print(f"    Train Precision:{best['train_pre']:.2f}%")
    print(f"    Train Recall:   {best['train_recall']:.2f}%")
    print(f"    Train F1:       {best['train_f1']:.2f}%")
    print(f"    Valid Acc:      {best['valid_acc']:.2f}%")
    print(f"    Valid Acc1:     {best['valid_acc1']:.2f}%")
    print(f"    Valid Precision:{best['valid_pre']:.2f}%")
    print(f"    Valid Recall:   {best['valid_recall']:.2f}%")
    print(f"    Valid F1:       {best['valid_f1']:.2f}%")

    print(f"\n{'─'*60}")
    print(f"  FINAL EPOCH ({n})")
    print(f"{'─'*60}")
    print(f"    Loss:      {final['loss']:.4f}")
    print(f"    Train Acc:      {final['train_acc']:.2f}%")
    print(f"    Train Acc1:     {final['train_acc1']:.2f}%")
    print(f"    Train Precision:{final['train_pre']:.2f}%")
    print(f"    Train Recall:   {final['train_recall']:.2f}%")
    print(f"    Train F1:       {final['train_f1']:.2f}%")
    print(f"    Valid Acc:      {final['valid_acc']:.2f}%")
    print(f"    Valid Acc1:     {final['valid_acc1']:.2f}%")
    print(f"    Valid Precision:{final['valid_pre']:.2f}%")
    print(f"    Valid Recall:   {final['valid_recall']:.2f}%")
    print(f"    Valid F1:       {final['valid_f1']:.2f}%")

    print(f"\n{'─'*60}")
    print(f"  CONVERGENCE ANALYSIS")
    print(f"{'─'*60}")
    # Find when validation acc first exceeded 90%
    for i, e in enumerate(epochs):
        if e["valid_acc"] >= 90:
            print(f"    Hit 90% valid acc at epoch {i+1}")
            break
    for i, e in enumerate(epochs):
        if e["valid_acc"] >= 95:
            print(f"    Hit 95% valid acc at epoch {i+1}")
            break
    # Rolling average
    window = 20
    if n >= window:
        smoothed = np.convolve([e["valid_acc"] for e in epochs], np.ones(window)/window, mode='valid')
        print(f"    Max rolling-{window} avg: {smoothed.max():.2f}% at epoch {smoothed.argmax() + window}")
        print(f"    Final rolling-{window} avg: {smoothed[-1]:.2f}%")

    print(f"\n{'─'*60}")
    print(f"  OVERALL STATS")
    print(f"{'─'*60}")
    print(f"    Max valid acc:  {max(e['valid_acc'] for e in epochs):.2f}%")
    print(f"    Min valid acc:  {min(e['valid_acc'] for e in epochs):.2f}%")
    print(f"    Avg valid acc (last 50):  {np.mean([e['valid_acc'] for e in epochs[-50:]]):.2f}%")
    print(f"    Max valid F1:   {max(e['valid_f1'] for e in epochs):.2f}%")
    print(f"    Final valid F1: {final['valid_f1']:.2f}%")
    print(f"    Loss trend:     {first['loss']:.4f} → {final['loss']:.4f}")
    print()


def plot_curves(epochs, save_path="training_curves.png"):
    """Plot and save training curves."""
    if not epochs:
        return

    xs = np.arange(1, len(epochs) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RES-GAT Training History", fontsize=16, fontweight='bold')

    # Loss
    ax = axes[0, 0]
    ax.plot(xs, [e["loss"] for e in epochs], 'b-', alpha=0.7, linewidth=1)
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(xs, [e["train_acc"] for e in epochs], 'b-', alpha=0.5, linewidth=1, label="Train Acc (micro)")
    ax.plot(xs, [e["valid_acc"] for e in epochs], 'r-', alpha=0.7, linewidth=1, label="Valid Acc (micro)")
    ax.plot(xs, [e["valid_acc1"] for e in epochs], 'g-', alpha=0.5, linewidth=1, label="Valid Acc (accuracy_score)")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1 Score
    ax = axes[1, 0]
    ax.plot(xs, [e["train_f1"] for e in epochs], 'b-', alpha=0.5, linewidth=1, label="Train F1")
    ax.plot(xs, [e["valid_f1"] for e in epochs], 'r-', alpha=0.7, linewidth=1, label="Valid F1")
    ax.set_title("F1 Score")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision & Recall
    ax = axes[1, 1]
    ax.plot(xs, [e["train_pre"] for e in epochs], 'b-', alpha=0.4, linewidth=1, label="Train Precision")
    ax.plot(xs, [e["train_recall"] for e in epochs], 'c-', alpha=0.4, linewidth=1, label="Train Recall")
    ax.plot(xs, [e["valid_pre"] for e in epochs], 'r-', alpha=0.7, linewidth=1, label="Valid Precision")
    ax.plot(xs, [e["valid_recall"] for e in epochs], 'm-', alpha=0.7, linewidth=1, label="Valid Recall")
    ax.set_title("Precision & Recall")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Training curves saved to {save_path}")


def save_data(epochs, save_path="training_history.npz"):
    """Save parsed training data for later analysis."""
    if not epochs:
        return
    data = {k: [e[k] for e in epochs] for k in epochs[0].keys()}
    np.savez(save_path, **data)
    print(f"[INFO] Training data saved to {save_path}")


if __name__ == "__main__":
    import sys

    log_path = sys.argv[1] if len(sys.argv) > 1 else LOG_FILE

    print(f"[INFO] Parsing log: {log_path}")
    epochs = parse_log(Path(log_path))

    if not epochs:
        print("[ERROR] No epoch data found in log.")
        print("Make sure the log contains 'EPOCH SUMMARY' lines.")
        sys.exit(1)

    generate_report(epochs)
    save_data(epochs)
    plot_curves(epochs)
