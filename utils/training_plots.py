import json
import os
import math
from typing import Dict, List

import matplotlib.pyplot as plt


def save_training_history(history: Dict[str, List[float]], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def plot_training_history(history: Dict[str, List[float]], output_dir: str = "outputs") -> None:
    os.makedirs(output_dir, exist_ok=True)
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        raise ValueError("History is empty. Nothing to plot.")

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    bleu1 = history.get("bleu1", [])
    bleu2 = history.get("bleu2", [])
    bleu3 = history.get("bleu3", [])
    bleu4 = history.get("bleu4", [])
    bleu_total = history.get("bleu_total", [])
    accuracy = history.get("accuracy", [])
    f1 = history.get("f1", [])

    # Figure 1: convergence with train/validation loss
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    if len(val_loss) == len(epochs) and any(not math.isnan(v) for v in val_loss):
        plt.plot(epochs, val_loss, marker="s", label="Validation Loss")
    plt.title("Loss Curve (Convergence Check)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=200)
    plt.close()

    # Figure 2: BLEU metrics
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, bleu1, marker="o", label="BLEU@1")
    plt.plot(epochs, bleu2, marker="o", label="BLEU@2")
    plt.plot(epochs, bleu3, marker="o", label="BLEU@3")
    plt.plot(epochs, bleu4, marker="o", label="BLEU@4")
    plt.plot(epochs, bleu_total, marker="d", linestyle="--", label="BLEU Total")
    plt.title("BLEU Metrics per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bleu_metrics.png"), dpi=200)
    plt.close()

    # Figure 3: answer quality metrics
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, accuracy, marker="o", label="Accuracy")
    plt.plot(epochs, f1, marker="s", label="F1-score")
    plt.title("Accuracy and F1-score per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_metrics.png"), dpi=200)
    plt.close()
