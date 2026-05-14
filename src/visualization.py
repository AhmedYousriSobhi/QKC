"""
visualization.py
────────────────
All plotting helpers for the project:
  • Circuit diagrams
  • Kernel matrix heatmaps
  • Comparison bar charts
  • Confusion matrices
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from qiskit.circuit import ParameterVector


# ── Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"  : "DejaVu Sans",
    "figure.dpi"   : 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


# ──────────────────────────────────────────────────────────────
# 1. Circuit diagrams
# ──────────────────────────────────────────────────────────────

def draw_circuit(qc, sample: np.ndarray, title: str, save_path: str):
    """
    Bind a parameter sample to the circuit, draw it, and save to disk.

    Parameters
    ----------
    qc        : QuantumCircuit (parametrized)
    sample    : 1-D array of floats to bind to parameters
    title     : figure title
    save_path : full path for the saved PNG
    """
    bound = qc.assign_parameters(dict(zip(qc.parameters, sample)))
    fig = bound.draw(output="mpl", style="iqp", fold=-1)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved circuit → {save_path}")


# ──────────────────────────────────────────────────────────────
# 2. Kernel matrix heatmaps
# ──────────────────────────────────────────────────────────────

def plot_kernel_heatmap(K: np.ndarray,
                        labels: np.ndarray,
                        title: str,
                        save_path: str,
                        n_show: int = 15):
    """
    Plot the top-left n_show × n_show block of the kernel matrix as a heatmap.
    Rows/columns are sorted by class label so structure is visible.

    The kernel K(x,z) = |⟨φ(x)|φ(z)⟩|² lies in [0,1].
    High values → states are close in Hilbert space.
    Low values  → states are nearly orthogonal.
    """
    # Sort by label for visual clarity
    idx = np.argsort(labels)[:n_show]
    K_sub = K[np.ix_(idx, idx)]
    lbl_sub = labels[idx]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(K_sub, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Kernel value K(x,z)")

    # Mark class boundary
    boundary = int(np.sum(lbl_sub == lbl_sub[0]))
    ax.axhline(boundary - 0.5, color="red", lw=1.5, ls="--", alpha=0.7)
    ax.axvline(boundary - 0.5, color="red", lw=1.5, ls="--", alpha=0.7)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Sample index (sorted by class)")
    ax.set_ylabel("Sample index (sorted by class)")

    # Compact tick labels
    ticks = np.arange(0, n_show, max(1, n_show // 5))
    ax.set_xticks(ticks); ax.set_yticks(ticks)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap → {save_path}")


# ──────────────────────────────────────────────────────────────
# 3. Comparison bar chart
# ──────────────────────────────────────────────────────────────

def plot_comparison_bar(df, save_path: str):
    """
    Horizontal bar chart comparing all models by accuracy.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [PALETTE[0] if "QSVM" in m or "Quantum" in m else PALETTE[1]
              for m in df["Model"]]

    bars = ax.barh(df["Model"], df["Accuracy"], color=colors, edgecolor="white", height=0.55)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Model Comparison — Test Accuracy", fontweight="bold")

    for bar, val in zip(bars, df["Accuracy"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=PALETTE[0], label="Quantum SVM"),
        Patch(facecolor=PALETTE[1], label="Classical SVM"),
    ]
    ax.legend(handles=legend_items, loc="lower right")

    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison chart → {save_path}")


# ──────────────────────────────────────────────────────────────
# 4. Confusion matrices
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrices(results_list: list, save_path: str):
    """
    Plot a row of confusion matrices — one per model in results_list.

    Parameters
    ----------
    results_list : list of dicts with keys 'label' and 'cm'.
    save_path    : output PNG path.
    """
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        cm = res["cm"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Malignant", "Benign"],
                    yticklabels=["Malignant", "Benign"],
                    linewidths=0.5)
        ax.set_title(res["label"], fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrices → {save_path}")


# ──────────────────────────────────────────────────────────────
# 5. PCA explained variance
# ──────────────────────────────────────────────────────────────

def plot_pca_variance(pca, save_path: str):
    """Bar chart of explained variance ratio per component."""
    if pca is None:
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(1, len(pca.explained_variance_ratio_) + 1),
           pca.explained_variance_ratio_ * 100,
           color=PALETTE[2], edgecolor="white")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA — Explained Variance per Component", fontweight="bold")
    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.plot(range(1, len(cumulative) + 1), cumulative, "o--", color=PALETTE[3],
            label=f"Cumulative: {cumulative[-1]:.1f}%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved PCA variance chart → {save_path}")


# ──────────────────────────────────────────────────────────────
# 6. Scatter of 2-D PCA data
# ──────────────────────────────────────────────────────────────

def plot_pca_scatter(X_train, y_train, save_path: str):
    """2-D scatter of the PCA-reduced training data colored by class."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, color, name in zip([-1, 1], [PALETTE[3], PALETTE[0]],
                                  ["Malignant (−1)", "Benign (+1)"]):
        mask = y_train == label
        ax.scatter(X_train[mask, 0], X_train[mask, 1],
                   c=color, label=name, alpha=0.65, edgecolors="white", s=50)
    ax.set_xlabel("PC 1 (scaled to [0, π])")
    ax.set_ylabel("PC 2 (scaled to [0, π])")
    ax.set_title("Training Data — 2-D PCA Projection", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved PCA scatter → {save_path}")
