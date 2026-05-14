"""
main.py
───────
PHM 678 — Quantum Kernel Classification Project
Spring 2026

Run this file to execute all five project parts in sequence:

  Part 1: Dataset summary and theoretical notes
  Part 2: Feature map circuits and visualizations
  Part 3: Quantum kernel matrices and heatmaps
  Part 4: QSVM vs classical SVM benchmark
  Part 5: Angle-embedding QSVM with PCA

All outputs are saved to results/.

Usage:
  python main.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ── Local modules ─────────────────────────────────────────────
from src.data_utils import load_dataset, preprocess, dataset_summary
from src.feature_maps import (
    build_basic_feature_map,
    build_entangled_feature_map,
    describe_feature_map,
)
from src.kernel_computation import (
    build_quantum_kernel,
    compute_kernel_matrix,
    kernel_diagnostics,
)
from src.classifiers import (
    train_quantum_svm,
    evaluate_quantum_svm,
    train_classical_svms,
    evaluate_classical_svms,
    build_comparison_table,
)
from src.visualization import (
    draw_circuit,
    plot_kernel_heatmap,
    plot_comparison_bar,
    plot_confusion_matrices,
    plot_pca_variance,
    plot_pca_scatter,
)


# ══════════════════════════════════════════════════════════════
# PART 1 — Theoretical Background & Dataset Overview
# ══════════════════════════════════════════════════════════════

def part1():
    print("\n" + "═" * 60)
    print("  PART 1 — Theoretical Background & Dataset")
    print("═" * 60)

    X, y, feat_names, target_names = load_dataset()
    dataset_summary(X, y, feat_names)

    print("""
Quantum Kernel Methods — Key Ideas
───────────────────────────────────
A quantum kernel method embeds classical data x into a quantum Hilbert
space via a parametrized quantum circuit U(x), producing the quantum
state |φ(x)⟩ = U(x)|0⟩.

The quantum kernel function is:
    K(x, z) = |⟨φ(x)|φ(z)⟩|²

This measures the squared overlap (fidelity) between two quantum states.
It is the quantum analogue of an inner product in the feature space, so
any SVM that uses it is implicitly working in a potentially exponentially
large Hilbert space.

What is a feature map?
  A parametrized quantum circuit U(x) that maps each data point x ∈ Rᵈ
  to a quantum state on n qubits.  Different circuit designs (depth,
  entanglement topology, gate choices) define different feature spaces
  and therefore different kernels.

How is the kernel matrix constructed?
  For a training set {x₁, …, xₙ} we run the circuit for every pair (i,j),
  measure the fidelity, and store K[i,j] = |⟨φ(xᵢ)|φ(xⱼ)⟩|².
  The result is a symmetric positive semi-definite n×n matrix that plugs
  directly into sklearn's SVC with kernel='precomputed'.
""")

    return X, y, feat_names


# ══════════════════════════════════════════════════════════════
# PART 2 — Implementing Quantum Feature Maps
# ══════════════════════════════════════════════════════════════

def part2(n_features: int = 2):
    print("\n" + "═" * 60)
    print("  PART 2 — Quantum Feature Maps")
    print("═" * 60)

    fmap_basic     = build_basic_feature_map(n_features)
    fmap_entangled = build_entangled_feature_map(n_features, reps=2)

    print("\n[BasicFeatureMap]")
    print(describe_feature_map("BasicFM"))
    print(f"  Qubits     : {fmap_basic.num_qubits}")
    print(f"  Parameters : {fmap_basic.num_parameters}")
    print(f"  Depth      : {fmap_basic.depth()}")

    print("\n[ZZFeatureMap]")
    print(describe_feature_map("ZZFeatureMap"))
    print(f"  Qubits     : {fmap_entangled.num_qubits}")
    print(f"  Parameters : {fmap_entangled.num_parameters}")
    print(f"  Depth      : {fmap_entangled.depth()}")

    # Draw circuits for one example sample
    sample = np.array([0.5 * np.pi, 1.2])[:n_features]

    print("\n  Drawing circuits ...")
    draw_circuit(fmap_basic, sample,
                 "Basic Feature Map  (H → RZ → RX, no entanglement)",
                 "results/circuit_basic.png")
    draw_circuit(fmap_entangled, sample,
                 "Entangled Feature Map  (ZZFeatureMap, 2 reps)",
                 "results/circuit_entangled.png")

    return fmap_basic, fmap_entangled


# ══════════════════════════════════════════════════════════════
# PART 3 — Quantum Kernel Computation
# ══════════════════════════════════════════════════════════════

def part3(fmap_basic, fmap_entangled, X_train, y_train):
    print("\n" + "═" * 60)
    print("  PART 3 — Quantum Kernel Computation")
    print("═" * 60)

    qk_basic     = build_quantum_kernel(fmap_basic)
    qk_entangled = build_quantum_kernel(fmap_entangled)

    print("\n  Computing BasicFM kernel matrix ...")
    t0 = time.time()
    K_basic_train = compute_kernel_matrix(qk_basic, X_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    kernel_diagnostics(K_basic_train, "BasicFM Kernel")

    print("\n  Computing ZZFeatureMap kernel matrix ...")
    t0 = time.time()
    K_zz_train = compute_kernel_matrix(qk_entangled, X_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    kernel_diagnostics(K_zz_train, "ZZFeatureMap Kernel")

    # Heatmaps (first 15 entries)
    plot_kernel_heatmap(K_basic_train, y_train,
                        "Quantum Kernel Matrix — Basic Feature Map\n"
                        "(first 15 samples, sorted by class; red dashed = class boundary)",
                        "results/heatmap_basic.png", n_show=15)

    plot_kernel_heatmap(K_zz_train, y_train,
                        "Quantum Kernel Matrix — ZZFeatureMap\n"
                        "(first 15 samples, sorted by class; red dashed = class boundary)",
                        "results/heatmap_zz.png", n_show=15)

    print("""
  Heatmap interpretation
  ──────────────────────
  Each cell (i,j) shows K(xᵢ,xⱼ) ∈ [0,1].
  Diagonal = 1.0 (a state is perfectly similar to itself).

  BasicFM:     Expect a relatively smooth gradient — without entanglement,
               the kernel captures single-qubit rotations only, so samples
               that are geometrically close in angle space score high.

  ZZFeatureMap: Off-diagonal structure is richer because pairwise feature
               interactions (xᵢ·xⱼ) are encoded in the ZZ interaction
               terms.  Same-class clusters should appear as bright blocks,
               suggesting the map separates the two cancer subtypes.
""")

    return qk_basic, qk_entangled, K_basic_train, K_zz_train


# ══════════════════════════════════════════════════════════════
# PART 4 — Classical SVM Training and Evaluation
# ══════════════════════════════════════════════════════════════

def part4(qk_basic, qk_entangled,
          K_basic_train, K_zz_train,
          X_train, X_test, y_train, y_test):
    print("\n" + "═" * 60)
    print("  PART 4 — SVM Training and Evaluation")
    print("═" * 60)

    # ── 4a. Compute test kernel matrices ──
    print("\n  Computing test kernels ...")
    _, K_basic_test = compute_kernel_matrix(qk_basic,     X_train, X_test)
    _, K_zz_test    = compute_kernel_matrix(qk_entangled, X_train, X_test)

    # ── 4b. Train quantum SVMs ──
    print("\n  Training Quantum SVMs ...")
    model_basic, C_basic, _ = train_quantum_svm(K_basic_train, y_train)
    model_zz,    C_zz,    _ = train_quantum_svm(K_zz_train,    y_train)
    print(f"  BasicFM best C     = {C_basic}")
    print(f"  ZZFeatureMap best C = {C_zz}")

    # ── 4c. Evaluate quantum SVMs ──
    res_basic = evaluate_quantum_svm(model_basic, K_basic_test, y_test,
                                     "QSVM-Basic")
    res_zz    = evaluate_quantum_svm(model_zz, K_zz_test, y_test,
                                     "QSVM-ZZ")

    # ── 4d. Train and evaluate classical SVMs ──
    print("\n  Training Classical SVMs ...")
    classical_models = train_classical_svms(X_train, y_train)
    print("\n  Evaluating Classical SVMs ...")
    classical_results = evaluate_classical_svms(classical_models, X_test, y_test)

    # ── 4e. Comparison table ──
    quantum_results = [res_basic, res_zz]
    df = build_comparison_table(quantum_results, classical_results)
    print("\n  ── Full Comparison Table ──")
    print(df.to_string(index=False))
    df.to_csv("results/comparison_table.csv", index=False)
    print("  Saved → results/comparison_table.csv")

    # ── 4f. Plots ──
    plot_comparison_bar(df, "results/comparison_bar.png")
    plot_confusion_matrices(quantum_results + classical_results,
                            "results/confusion_matrices.png")

    return quantum_results, classical_results, df


# ══════════════════════════════════════════════════════════════
# PART 5 — Angle Embedding + PCA + QSVM
# ══════════════════════════════════════════════════════════════

def part5(n_components: int = 2):
    print("\n" + "═" * 60)
    print("  PART 5 — Angle Embedding with PCA")
    print("═" * 60)

    X, y, _, _ = load_dataset()

    print(f"\n  Applying PCA → {n_components} components ...")
    X_train, X_test, y_train, y_test, pca, scaler = preprocess(
        X, y, n_components=n_components, apply_pca=True, max_train_samples=80
    )
    if len(X_test) > 40:
        X_test, y_test = X_test[:40], y_test[:40]

    print(f"  Explained variance: "
          f"{pca.explained_variance_ratio_ * 100}")
    print(f"  Total retained    : "
          f"{pca.explained_variance_ratio_.sum() * 100:.1f}%")

    plot_pca_variance(pca, "results/pca_variance.png")
    plot_pca_scatter(X_train, y_train, "results/pca_scatter.png")

    # ── Angle-embedding circuit = BasicFeatureMap ──
    # (PCA features are already scaled to [0,π] = valid rotation angles)
    print(f"\n  Building angle-embedding circuit ({n_components} qubits) ...")
    fmap_angle = build_basic_feature_map(n_components)
    draw_circuit(fmap_angle, X_train[0],
                 "Angle Embedding Circuit  (PCA features as rotation angles)",
                 "results/circuit_angle_embedding.png")

    qk_angle = build_quantum_kernel(fmap_angle)

    print("  Computing kernel matrices for angle-embedding QSVM ...")
    K_train = compute_kernel_matrix(qk_angle, X_train)
    _, K_test = compute_kernel_matrix(qk_angle, X_train, X_test)

    model_angle, best_C, _ = train_quantum_svm(K_train, y_train)
    print(f"  Best C = {best_C}")

    res_angle = evaluate_quantum_svm(model_angle, K_test, y_test,
                                     "QSVM-Angle(PCA)")

    print(f"""
  Angle Embedding vs Part 4 QSVMs
  ─────────────────────────────────
  In Part 4, we used ALL 30 features reduced to 2 PCA components only for
  circuit width, but the ZZFeatureMap still used the same 2-D data.

  Part 5 explicitly:
    1. Runs PCA first to retain the top {n_components} components.
    2. Scales each component to [0, π] to use as a rotation angle.
    3. Applies BasicFeatureMap — each angle is literally a gate parameter.

  This is the textbook "angle embedding" approach.  It is interpretable
  (each qubit directly represents a data axis) but limited — with only
  {n_components} qubits there is no entanglement between features.

  Accuracy comparison:
    QSVM-Basic (Part 4): see comparison_table.csv
    QSVM-Angle (Part 5): {res_angle['accuracy']:.4f}

  The angle-embedding QSVM tends to perform comparably or slightly below
  the ZZFeatureMap because it lacks entanglement-based feature interaction.
  However, it is far more circuit-efficient and noise-tolerant.
""")

    return res_angle


# ══════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  PHM 678 — Quantum Kernel Classification Project")
    print("  Spring 2026")
    print("█" * 60)

    N_FEATURES       = 2    # qubits (= features after PCA)
    TEST_SIZE        = 0.20
    MAX_TRAIN_QUANTUM = 80  # quantum kernel scales as O(n²) — cap for speed

    # ── Load & preprocess ────────────────────────────────────
    # Parts 2-4: NO PCA. We select the 2 raw features with the highest
    # variance so circuits stay interpretable and PCA remains Part 5's
    # exclusive contribution (as the assignment intends).
    X_raw, y_raw, feat_names, _ = load_dataset()
    X_train, X_test, y_train, y_test, _, scaler = preprocess(
        X_raw, y_raw,
        n_components=N_FEATURES,
        test_size=TEST_SIZE,
        apply_pca=False,          # <-- raw features, no PCA
        max_train_samples=MAX_TRAIN_QUANTUM,
    )
    # Cap test size for quantum kernel (K_test is n_test × n_train)
    if len(X_test) > 40:
        X_test, y_test = X_test[:40], y_test[:40]
    print(f"\n  Train size (quantum): {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"  Features (top-variance raw): {N_FEATURES}  — NO PCA applied here")
    print(f"  PCA is reserved for Part 5 (angle embedding).")

    # ── Run all parts ─────────────────────────────────────────
    X, y, feat_names = part1()

    fmap_basic, fmap_entangled = part2(n_features=N_FEATURES)

    qk_basic, qk_entangled, K_basic_train, K_zz_train = part3(
        fmap_basic, fmap_entangled, X_train, y_train
    )

    quantum_results, classical_results, df = part4(
        qk_basic, qk_entangled,
        K_basic_train, K_zz_train,
        X_train, X_test, y_train, y_test,
    )

    res_angle = part5(n_components=N_FEATURES)

    # ── Final summary ──────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  PROJECT COMPLETE — Results saved to results/")
    print("═" * 60)
    print("\n  Files generated:")
    for f in sorted(os.listdir("results")):
        print(f"    results/{f}")
    print()
