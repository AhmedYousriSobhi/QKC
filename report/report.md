# PHM 678 — Quantum Kernel Classification Project
## Project Report · Spring 2026

---

## Part 1 — Theoretical Background

### What is a Quantum Feature Map?

A quantum feature map is a parametrized quantum circuit `U(x)` that takes a classical data point `x ∈ Rᵈ` and produces a quantum state:

```
|φ(x)⟩ = U(x)|0⟩
```

where `|0⟩` is the ground state of an `n`-qubit system. Each data point `x` is encoded into the amplitudes and phases of this state through a series of rotation gates (RX, RZ, RY) and entangling gates (CX, CZ). The choice of circuit architecture — gate types, qubit connectivity, number of repetitions — defines what aspects of the data are amplified in the Hilbert space.

The key intuition: classical ML works with feature vectors in `Rᵈ`. Quantum ML works with feature states in a Hilbert space of dimension `2ⁿ`. The kernel trick lets us work in this exponentially large space without ever explicitly computing the full state vector — only the inner products between pairs of states.

### How is the Quantum Kernel Matrix Constructed?

The quantum kernel between two data points `x` and `z` is defined as the squared overlap (fidelity) of their corresponding quantum states:

```
K(x, z) = |⟨φ(x)|φ(z)⟩|²
```

This is estimated using the **compute-uncompute** method:
1. Prepare the state `|φ(x)⟩` by running `U(x)`.
2. Apply the inverse circuit `U(z)†`.
3. Measure the probability of observing the all-zeros bitstring `|0⟩`.
4. That probability equals `|⟨φ(x)|φ(z)⟩|²`.

For a training set of `n` points, evaluating all pairs produces an `n × n` symmetric positive semi-definite kernel matrix `K`. This matrix plugs directly into scikit-learn's `SVC` with `kernel='precomputed'`, so the entire classical SVM training machinery applies unchanged.

The computational cost is `O(n²)` circuit evaluations. With each evaluation requiring full circuit simulation, this is the main bottleneck distinguishing quantum kernels from classical ones.

---

## Part 2 — Quantum Feature Maps

### Dataset

The Breast Cancer Wisconsin dataset (569 samples, 30 features) was used throughout. Labels were mapped to `{−1, +1}` to match SVM convention. For the quantum circuits, the 30 features were reduced to 2 via PCA (retaining ~99.9% of variance) and scaled to `[0, π]` so they serve directly as rotation angles.

### Feature Map 1 — Basic Feature Map

```
For each qubit i:
    H  →  RZ(xᵢ)  →  RX(xᵢ)
```

**Design intuition:**  
The Hadamard gate places each qubit in the uniform superposition `(|0⟩ + |1⟩)/√2`. The RZ rotation then encodes the feature `xᵢ` as a phase difference between `|0⟩` and `|1⟩`. The RX rotation mixes the amplitudes further, adding a second degree of freedom per qubit. Together, the two rotations place the qubit anywhere on the Bloch sphere in the XZ plane.

There is no entanglement between qubits. Each qubit independently encodes one feature. The resulting feature space is a direct product of single-qubit Hilbert spaces — equivalent in expressiveness to a classical kernel that separates features additively. This map is simple, fast, and noise-tolerant, but limited in the feature interactions it can capture.

**Circuit parameters:** 2 qubits, 2 parameters, depth 3.

### Feature Map 2 — ZZFeatureMap (Entangled)

```
For each rep:
    H on all qubits
    RZ(2·xᵢ) on each qubit i
    For each pair (i,j):
        CX(i,j)  →  RZ(2·(π−xᵢ)·(π−xⱼ))  →  CX(i,j)
```

**Design intuition:**  
The ZZFeatureMap encodes pairwise feature interactions through the ZZ interaction term `(π−xᵢ)·(π−xⱼ)`. When `xᵢ` and `xⱼ` are both large or both small, the ZZ rotation is near zero; when they differ, it is large. This means the kernel function is sensitive to whether two features co-vary — something the basic map completely ignores.

The CX gate before and after the RZ implements a controlled-Z-rotation between pairs of qubits, creating genuine entanglement. In the quantum circuit picture, the qubits become correlated: the state of qubit `i` influences what happens to qubit `j`. In the kernel picture, this means `K(x,z)` captures whether the two data points share similar feature interaction patterns, not just similar individual feature values.

With 2 reps, the encoding block is applied twice, deepening the circuit and making the feature map harder to simulate classically.

**Circuit parameters:** 2 qubits, 2 parameters, depth varies with reps.

---

## Part 3 — Quantum Kernel Computation

Both kernel matrices were computed for 80 training samples using Qiskit's `FidelityQuantumKernel` with the `ComputeUncompute` fidelity estimator on a statevector simulator.

### Kernel Diagnostics

| Property | BasicFM | ZZFeatureMap |
|----------|---------|--------------|
| Shape | 80 × 80 | 80 × 80 |
| Diagonal mean | ≈ 1.0 | ≈ 1.0 |
| Symmetric | Yes | Yes |
| Min eigenvalue | ≈ 0 | ≈ 0 |
| Value range | [0, 1] | [0, 1] |

Both matrices are positive semi-definite and symmetric — confirming they are valid kernel matrices that will produce a well-defined dual SVM problem.

### Heatmap Analysis (first 15 samples, class-sorted)

**BasicFM heatmap:**  
The kernel values follow a smooth gradient. Samples that are geometrically close in the scaled 2-D feature space score high (bright cells); distant samples score low (dark cells). The class boundary (red dashed line) separates the malignant and benign clusters moderately well — there is visible structure, but the blocks are not sharply defined. This reflects the map's lack of entanglement: it separates classes only through the similarity of individual feature angles.

**ZZFeatureMap heatmap:**  
The off-diagonal structure is richer. The ZZ interaction terms introduce asymmetry in how the kernel responds to feature differences, producing a more irregular pattern. Within-class similarity (the two blocks along the diagonal) should be higher than the off-diagonal cross-class similarity, though with only 2 qubits this effect is subtle. The deeper circuit creates a kernel that is harder to approximate classically, which is both a potential advantage and a source of noise sensitivity.

---

## Part 4 — Classical SVM Training and Evaluation

### Quantum SVM Results

Both QSVMs used `SVC(kernel='precomputed')` with the respective kernel matrix. The regularization parameter `C` was tuned via 5-fold stratified cross-validation over `{0.01, 0.1, 1.0, 10.0, 100.0}`.

| Model | Best C | Test Accuracy | Precision | Recall | F1 |
|-------|--------|--------------|-----------|--------|----|
| QSVM-Basic | 10.0 | **0.9250** | 0.9251 | 0.9250 | 0.9244 |
| QSVM-ZZ | 100.0 | 0.6000 | 0.6113 | 0.6000 | 0.6042 |

The ZZFeatureMap underperforms on this small subset. With 80 training samples and a 2-rep circuit, the richer feature space is more prone to overfitting. The BasicFM, despite its simplicity, generalizes better because its kernel values vary smoothly with input distances.

### Classical SVM Baselines

| Model | Best C | Test Accuracy | Precision | Recall | F1 |
|-------|--------|--------------|-----------|--------|----|
| SVM-RBF | 10.0 | **0.9500** | 0.9537 | 0.9500 | 0.9492 |
| SVM-Linear | 100.0 | **0.9500** | 0.9537 | 0.9500 | 0.9492 |
| SVM-Polynomial | 1.0 | 0.9250 | 0.9251 | 0.9250 | 0.9244 |

### Discussion

The RBF and linear kernels outperform both quantum kernels on this dataset. This result is expected and consistent with the current literature on near-term quantum advantage:

1. **Data size matters.** With 80 training samples and only 2 features, the breast cancer dataset is well within the range where classical kernels — especially RBF — are essentially optimal. Quantum advantage in kernel methods requires datasets that are provably hard to kernel with classical methods.

2. **Simulation removes quantum advantage.** Running on a statevector simulator means there is no physical quantum speedup. The quantum kernel is computed exactly via classical linear algebra, which is no faster than the RBF kernel.

3. **Circuit depth and overfitting.** The ZZFeatureMap with 2 reps produces a complex feature space that overfits on a small training set. Reducing `reps=1` or adding regularization would likely improve it.

4. **The BasicFM is competitive.** At 0.925 accuracy, it matches the polynomial SVM. This suggests that even a non-entangling quantum kernel can capture useful structure in this 2-D feature space.

---

## Part 5 — Angle Embedding with PCA

### PCA Analysis

PCA was applied to all 30 features before quantum circuit construction. The first two principal components retain 99.9% of total variance (PC1: 98.2%, PC2: 1.7%). The remaining 28 components carry negligible information, confirming that the breast cancer dataset has strong low-dimensional structure.

Each PC was scaled to `[0, π]` so it can directly serve as a rotation angle in a quantum gate.

### Angle Embedding Circuit

The angle embedding circuit is identical to the BasicFeatureMap: `H → RZ(xᵢ) → RX(xᵢ)` per qubit, where `xᵢ` is now the `i`-th PCA component scaled to `[0, π]`. This is the canonical angle-encoding approach: each qubit's rotation angle literally is a principal component of the data.

### Results

| Model | Test Accuracy |
|-------|--------------|
| QSVM-Angle(PCA) | 0.9250 |

The angle-embedding QSVM matches the BasicFM QSVM from Part 4 — both achieve 0.925. This makes sense: both use the same circuit architecture on the same 2-D PCA-reduced data. The explicit "angle embedding" framing in Part 5 makes the data-to-gate mapping more transparent, but the underlying computation is equivalent.

### Comparison with Part 4

The Part 4 ZZFeatureMap uses the same 2-D data but introduces entanglement and a deeper circuit. That richer feature space hurt rather than helped on this dataset (0.60 vs 0.925). The angle embedding avoids this by staying simple: 2 qubits, no entanglement, shallow circuit.

**When does angle embedding shine?**  
It works best when the data has a clean low-dimensional structure (as here), the number of relevant features is small enough to fit directly on available qubits, and circuit depth is a constraint (e.g., on noisy hardware). It fails when the relevant structure lives in higher-dimensional feature interactions that angle encoding cannot capture without entanglement.

---

## Conclusion

This project implemented and compared quantum and classical kernel methods on a real binary classification task.

The main findings:
- The BasicFeatureMap (no entanglement) produces a competitive quantum kernel, achieving 92.5% test accuracy.
- The ZZFeatureMap, while theoretically richer, underperforms on small training sets due to the higher-dimensional feature space being harder to generalize from.
- Classical RBF and linear SVMs achieve the best accuracy (95%) on this dataset, consistent with the expectation that near-term quantum kernel advantage requires specifically structured data.
- Angle embedding (Part 5) is a clean, interpretable approach that matches the BasicFM baseline while being explicit about the data-to-gate encoding.

The most important takeaway: quantum kernels are not universally better than classical ones. Quantum advantage in machine learning, if it exists, will come from datasets with mathematical structure that is hard to compute classically — not from routine classification benchmarks like breast cancer detection.
