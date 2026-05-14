# Quantum Kernel Classification Project
**Quantum Machine Learning**

---

## Overview

This repository implements quantum kernel methods for binary classification on the Breast Cancer Wisconsin dataset. It covers all five required project parts: theoretical grounding, two quantum feature maps, quantum kernel matrix computation, SVM benchmarking against classical baselines, and angle-embedding with PCA.

---

## Repository Structure

```
quantum-kernel-classification/
├── main.py                   ← Run this to execute all five parts
├── requirements.txt
├── src/
│   ├── data_utils.py         ← Dataset loading, PCA, scaling
│   ├── feature_maps.py       ← BasicFeatureMap + ZZFeatureMap
│   ├── kernel_computation.py ← FidelityQuantumKernel wrapper
│   ├── classifiers.py        ← QSVM + classical SVM training & evaluation
│   └── visualization.py      ← All plotting functions
├── results/                  ← Generated figures and CSVs (auto-created)
└── report/
    └── report.md             ← Written project report
```

---

## Setup

```bash
# 1. Clone / download the repo
git clone https://github.com/AhmedYousriSobhi/QKC
cd QLC

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the project
python main.py
```

Running `main.py` takes roughly **3–5 minutes** on a standard laptop.  All figures and the comparison CSV are saved to `results/`.

---

## What Each File Does

### `src/data_utils.py`
Loads the Breast Cancer Wisconsin dataset (569 samples, 30 features, binary labels). Handles stratified train/test splitting, PCA dimensionality reduction, and MinMax scaling to `[0, π]` so features can directly serve as quantum rotation angles.

### `src/feature_maps.py`
Two quantum feature maps:

| Map | Gates | Entanglement | Use |
|-----|-------|--------------|-----|
| `BasicFeatureMap` | H → RZ(xᵢ) → RX(xᵢ) | None | Baseline; each qubit encodes one feature independently |
| `ZZFeatureMap` | H → RZ(2xᵢ) → CX → RZ(2(π−xᵢ)(π−xⱼ)) → CX | Full pairwise | Captures feature correlations via ZZ interactions |

### `src/kernel_computation.py`
Wraps Qiskit's `FidelityQuantumKernel` using the `ComputeUncompute` fidelity estimator. For each pair `(x, z)`, it prepares `|φ(x)⟩`, applies `U(z)†`, and measures the overlap probability — which equals `|⟨φ(x)|φ(z)⟩|²`.

### `src/classifiers.py`
Trains SVMs with precomputed quantum kernels and classical kernels (RBF, polynomial, linear). Hyperparameter `C` is tuned via 5-fold stratified cross-validation. Outputs a comparison DataFrame.

### `src/visualization.py`
- Circuit diagrams with bound parameters
- Kernel heatmaps (first 15 samples, class-sorted)
- Model comparison bar chart
- Confusion matrices
- PCA variance and scatter plots

---

## Key Design Decision — Sample Size Cap

Quantum kernel computation scales as **O(n²)** in circuit evaluations. With 80 training samples, the kernel matrix requires 3,240 circuit pairs. On a statevector simulator this takes ~10 seconds per kernel. For reproducibility on standard hardware, `MAX_TRAIN_QUANTUM = 80` is set in `main.py`. Change it to run on more data (at the cost of runtime).

---

## Results Summary

| Model | Test Accuracy |
|-------|--------------|
| SVM-RBF | **0.9500** |
| SVM-Linear | **0.9500** |
| QSVM-Basic | 0.9250 |
| SVM-Polynomial | 0.9250 |
| QSVM-ZZ | 0.6000* |
| QSVM-Angle(PCA) | 0.9250 |

*ZZFeatureMap with 2 reps on a 2-qubit circuit can overfit the small training set. Reducing `reps=1` or increasing training data improves it.

---

## Part-by-Part Breakdown

| Part | Topic | Key Output |
|------|-------|------------|
| 1 | Theory — quantum kernels | Console explanation |
| 2 | Feature map circuits | `results/circuit_*.png` |
| 3 | Kernel matrices + heatmaps | `results/heatmap_*.png` |
| 4 | QSVM vs classical SVM | `results/comparison_*.{png,csv}` |
| 5 | Angle embedding + PCA | `results/pca_*.png`, `circuit_angle_embedding.png` |

---

## Dependencies

- `qiskit >= 2.0`
- `qiskit-machine-learning >= 0.8`
- `qiskit-aer >= 0.15`
- `scikit-learn >= 1.3`
- `matplotlib`, `seaborn`, `numpy`, `pandas`, `pylatexenc`

---

