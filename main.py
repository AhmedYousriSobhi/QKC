"""
main.py
───────
Quantum Kernel Classification Project


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
import random
import warnings
import numpy as np
import pandas as pd

SEED = 42

# ── Global random seed — guarantees identical results on every run ──
random.seed(SEED)
np.random.seed(SEED)

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

    print("""
### Why not use all 30 features?

The breast cancer dataset has 30 features. Running a **30-qubit** quantum circuit on a classical simulator is computationally intractable:

| Qubits | State vector size | Memory needed |
|--------|-------------------|---------------|
| 2 | $2^2 = 4$ amplitudes | negligible |
| 10 | $2^{10} = 1{,}024$ | ~16 KB |
| 20 | $2^{20} \approx 10^6$ | ~16 MB |
| **30** | $2^{30} \approx 10^9$ | **~16 GB** |

Beyond memory, the kernel matrix for $n$ training samples requires $O(n^2)$ full circuit evaluations. At 30 qubits, each evaluation is already slow, and with hundreds of samples the total time is measured in days, not seconds.

**The standard solution in QSVM research:** reduce the feature space to 2–4 dimensions before encoding. This project uses two strategies:

- **Parts 2–4:** select the 2 raw features with the highest training-set variance. Features stay interpretable — they are real, named measurements.
- **Part 5:** apply PCA, project onto the top 2 principal components, and use those as angle-encoding rotation angles. This is where PCA enters the project.
          """)
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
                 "results/part-2-circuit_basic.png")
    draw_circuit(fmap_entangled, sample,
                 "Entangled Feature Map  (ZZFeatureMap, 2 reps)",
                 "results/part-2-circuit_entangled.png")

    return fmap_basic, fmap_entangled


# ══════════════════════════════════════════════════════════════
# PART 3 — Quantum Kernel Computation
# ══════════════════════════════════════════════════════════════

def part3(fmap_basic, fmap_entangled, X_train, y_train):
    print("\n" + "═" * 60)
    print("  PART 3 — Quantum Kernel Computation")
    print("═" * 60)

    qk_basic     = build_quantum_kernel(fmap_basic, seed=SEED)
    qk_entangled = build_quantum_kernel(fmap_entangled, seed=SEED)

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
                        "results/part-3-heatmap_basic.png", n_show=15)

    plot_kernel_heatmap(K_zz_train, y_train,
                        "Quantum Kernel Matrix — ZZFeatureMap\n"
                        "(first 15 samples, sorted by class; red dashed = class boundary)",
                        "results/part-3-heatmap_zz.png", n_show=15)

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
          X_train, X_test, y_train, y_test,
          X_train_full=None, X_test_full=None,
          y_train_full=None, y_test_full=None):
    """
    X_train / y_train     : 80-sample quantum subset (for QSVM kernel computation)
    X_train_full / y_train_full : full training set  (for classical SVMs — no sample cap)
    """
    print("\n" + "═" * 60)
    print("  PART 4 — SVM Training and Evaluation")
    print("═" * 60)

    # Fall back to quantum subset if full arrays not supplied
    if X_train_full is None:
        X_train_full, X_test_full = X_train, X_test
        y_train_full, y_test_full = y_train, y_test

    # ── 4a. Compute test kernel matrices ──
    print("\n  Computing test kernels ...")
    _, K_basic_test = compute_kernel_matrix(qk_basic,     X_train, X_test)
    _, K_zz_test    = compute_kernel_matrix(qk_entangled, X_train, X_test)

    # ── 4b. Train quantum SVMs ──
    print("\n  Training Quantum SVMs ...")
    model_basic, C_basic, _ = train_quantum_svm(K_basic_train, y_train, random_state=SEED)
    model_zz,    C_zz,    _ = train_quantum_svm(K_zz_train,    y_train, random_state=SEED)
    print(f"  BasicFM best C     = {C_basic}")
    print(f"  ZZFeatureMap best C = {C_zz}")

    # ── 4c. Evaluate quantum SVMs ──
    res_basic = evaluate_quantum_svm(model_basic, K_basic_test, y_test,
                                     "QSVM-Basic")
    res_zz    = evaluate_quantum_svm(model_zz, K_zz_test, y_test,
                                     "QSVM-ZZ")

    # ── 4d. Train and evaluate classical SVMs ──
    # Classical SVMs use the full training set — no O(n²) circuit cost.
    print("\n  Training Classical SVMs (full training set) ...")
    classical_models = train_classical_svms(X_train_full, y_train_full, random_state=SEED)
    print("\n  Evaluating Classical SVMs ...")
    classical_results = evaluate_classical_svms(classical_models, X_test_full, y_test_full)

    # ── 4e. Comparison table ──
    quantum_results = [res_basic, res_zz]
    df = build_comparison_table(quantum_results, classical_results)
    print("\n  ── Full Comparison Table ──")
    print(df.to_string(index=False))
    df.to_csv("results/part-4-comparison_table.csv", index=False)
    print("  Saved → results/part-4-comparison_table.csv")

    # ── 4f. Plots ──
    plot_comparison_bar(df, "results/part-4-comparison_bar.png")
    plot_confusion_matrices(quantum_results + classical_results,
                            "results/part-4-confusion_matrices.png")

    return quantum_results, classical_results, df


# ══════════════════════════════════════════════════════════════
# PART 5 — Angle Embedding + PCA + QSVM
# ══════════════════════════════════════════════════════════════

def part5(n_components: int = 2):
    """
    Part 5 is the ONLY place PCA is applied in this project.

    Parts 2-4 used raw feature selection (top-2 by variance: worst area,
    mean area).  Here we apply PCA to all 30 features, project onto the top
    n_components principal components, and use those projections as rotation
    angles — the textbook definition of angle embedding.
    """
    print("\n" + "═" * 60)
    print("  PART 5 — Angle Embedding with PCA")
    print("  (PCA introduced here for the first time)")
    print("═" * 60)

    from sklearn.decomposition import PCA as _PCA

    # Re-load and split identically to Parts 2-4 so comparisons are fair
    X, y, _, _ = load_dataset()
    X_train_full, X_test_full, y_train_full, y_test_full, _, _ = preprocess(
        X, y, n_components=n_components, apply_pca=False
    )

    # ── Apply PCA to the original 30-feature arrays ──────────
    # We need the raw 30-feature split, so reload without feature selection
    from sklearn.model_selection import train_test_split as _tts
    X_tr30, X_te30, y_tr, y_te = _tts(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    print(f"\n  Fitting PCA({n_components}) on {len(X_tr30)} training samples ...")
    pca = _PCA(n_components=n_components, random_state=SEED)
    X_tr_pca = pca.fit_transform(X_tr30)
    X_te_pca = pca.transform(X_te30)

    for i, (v, c) in enumerate(zip(
        pca.explained_variance_ratio_ * 100,
        np.cumsum(pca.explained_variance_ratio_) * 100
    ), 1):
        print(f"  PC{i}: {v:.2f}%  (cumulative: {c:.2f}%)")

    # Subsample for quantum kernel speed
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_tr_pca), size=80, replace=False)
    X_train = X_tr_pca[idx]
    y_train = y_tr[idx]
    X_test  = X_te_pca[:40]
    y_test  = y_te[:40]

    # Scale PCA components to [0, π] — they become the rotation angles
    from sklearn.preprocessing import MinMaxScaler as _MMS
    scaler = _MMS(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    plot_pca_variance(pca, "results/part-5-pca_variance.png")
    plot_pca_scatter(X_train, y_train, "results/part-5-pca_scatter.png")

    # ── Angle-embedding circuit ──────────────────────────────
    # BasicFeatureMap: H → RZ(θ) → RX(θ) per qubit
    # θᵢ = PCA component i, scaled to [0, π]
    print(f"\n  Building angle-embedding circuit ({n_components} qubits) ...")
    fmap_angle = build_basic_feature_map(n_components)
    draw_circuit(fmap_angle, X_train[0],
                 f"Angle Embedding (Part 5)\n"
                 f"PC1={X_train[0,0]:.3f} rad, PC2={X_train[0,1]:.3f} rad",
                 "results/part-5-circuit_angle_embedding.png")

    qk_angle = build_quantum_kernel(fmap_angle, seed=SEED)

    print("  Computing kernel matrices ...")
    K_train = compute_kernel_matrix(qk_angle, X_train)
    _, K_test = compute_kernel_matrix(qk_angle, X_train, X_test)

    model_angle, best_C, _ = train_quantum_svm(K_train, y_train, random_state=SEED)
    print(f"  Best C = {best_C}")

    res_angle = evaluate_quantum_svm(model_angle, K_test, y_test,
                                     "QSVM-Angle(PCA)")

    print(f"""
  Part 5 vs Part 4 — what changed
  ─────────────────────────────────
  Parts 2-4 input: worst area, mean area (2 raw features, named measurements)
  Part 5 input   : PC1, PC2 (projections onto directions of max variance)

  PCA retains {pca.explained_variance_ratio_.sum()*100:.1f}% of total dataset variance.
  Feature selection (Parts 2-4) uses just 2 of the 30 original features.

  The circuit structure is identical (BasicFeatureMap).
  The only difference is what each rotation angle represents.

  QSVM-Angle(PCA) accuracy: {res_angle['accuracy']:.4f}
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
    np.random.seed(SEED)   # re-seed after preprocess subsampling
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

    # Full preprocessed arrays for classical SVMs (no sample cap)
    X_tr_full2, X_te_full2, y_tr_full2, y_te_full2, _, scaler_full = preprocess(
        X_raw, y_raw,
        n_components=N_FEATURES,
        test_size=TEST_SIZE,
        apply_pca=False,
        max_train_samples=None,   # no cap for classical SVMs
    )

    quantum_results, classical_results, df = part4(
        qk_basic, qk_entangled,
        K_basic_train, K_zz_train,
        X_train, X_test, y_train, y_test,
        X_train_full=X_tr_full2, X_test_full=X_te_full2,
        y_train_full=y_tr_full2, y_test_full=y_te_full2,
    )

    res_angle = part5(n_components=N_FEATURES)

    # ── Export comparison_table_final.csv (Parts 4 + 5 combined) ──
    rep_angle = res_angle["report"]
    row_angle = {
        "Model"    : res_angle["label"],
        "Accuracy" : round(res_angle["accuracy"], 4),
        "Precision": round(rep_angle["weighted avg"]["precision"], 4),
        "Recall"   : round(rep_angle["weighted avg"]["recall"], 4),
        "F1-Score" : round(rep_angle["weighted avg"]["f1-score"], 4),
    }
    df_final = pd.concat(
        [df, pd.DataFrame([row_angle])], ignore_index=True
    ).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    df_final.to_csv("results/part-5-comparison_table_final.csv", index=False)
    print("  Saved → results/part-5-comparison_table_final.csv")

    # ── Final summary ──────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  PROJECT COMPLETE — Results saved to results/")
    print("═" * 60)
    print("\n  Files generated:")
    for f in sorted(os.listdir("results")):
        print(f"    results/{f}")
    print()
