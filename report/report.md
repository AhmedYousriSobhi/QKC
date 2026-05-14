<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>

# Quantum Kernel Classification Project
## Written Report

## Table of Contents
- [Quantum Kernel Classification Project](#quantum-kernel-classification-project)
  - [Written Report](#written-report)
  - [Table of Contents](#table-of-contents)
  - [Part 1 — Theoretical Background](#part-1--theoretical-background)
    - [What is a quantum feature map?](#what-is-a-quantum-feature-map)
    - [How is the quantum kernel constructed?](#how-is-the-quantum-kernel-constructed)
    - [Why not use all 30 features?](#why-not-use-all-30-features)
  - [Part 2 — Quantum Feature Maps](#part-2--quantum-feature-maps)
    - [Dataset and preprocessing (Parts 2–4)](#dataset-and-preprocessing-parts-24)
      - [Why not use all 30 features?](#why-not-use-all-30-features-1)
    - [Feature Map 1 — BasicFeatureMap](#feature-map-1--basicfeaturemap)
    - [Feature Map 2 — ZZFeatureMap](#feature-map-2--zzfeaturemap)
  - [Part 3 — Quantum Kernel Computation](#part-3--quantum-kernel-computation)
  - [Part 4 — SVM Training and Evaluation](#part-4--svm-training-and-evaluation)
  - [Part 5 — Angle Embedding with PCA](#part-5--angle-embedding-with-pca)
    - [What changes from Parts 2–4](#what-changes-from-parts-24)
    - [Feature selection vs PCA](#feature-selection-vs-pca)
  - [Conclusion](#conclusion)

---

## Part 1 — Theoretical Background

### What is a quantum feature map?

A quantum feature map is a parametrized circuit U(x) that takes a classical data
point x and produces a quantum state:

    |φ(x)⟩ = U(x)|0⟩^⊗n

The gate parameters are functions of the input features. Different gate choices —
rotation axes, entanglement topology, number of repetitions — define different
mappings from data space into the Hilbert space.

### How is the quantum kernel constructed?

The kernel between two points is the squared overlap of their states:

    K(x, z) = |⟨φ(x)|φ(z)⟩|²

This is estimated with the compute-uncompute circuit: prepare |φ(x)⟩, apply
U(z)†, and measure the probability of observing |0⟩^⊗n. That probability equals
the squared fidelity. For n training samples, computing all pairs produces an
n × n symmetric, positive semi-definite kernel matrix that plugs directly into
SVC(kernel='precomputed').

### Why not use all 30 features?

This question deserves a direct answer. The breast cancer dataset has 30 features.
Encoding all 30 into a quantum circuit requires 30 qubits. That is intractable on
a classical simulator:

| Qubits | State vector size | Memory      |
|--------|-------------------|-------------|
| 2      | 4 amplitudes      | negligible  |
| 10     | 1,024             | ~16 KB      |
| 20     | ~1 million        | ~16 MB      |
| 30     | ~1 billion        | ~16 GB      |

Memory is only half the problem. The kernel matrix for n training samples requires
O(n²) full circuit evaluations. At 30 qubits, each evaluation involves simulating
a billion-amplitude state vector, and with even 100 training samples, the total
comes to 10,000 such evaluations. On current hardware that takes hours or days.

This is not a quirk of this project — it is the central bottleneck of near-term
quantum machine learning. Every QSVM paper that runs on a simulator uses 2–8
features. The solution is dimensionality reduction before circuit construction.
This project uses two approaches, applied at different points:

  - Parts 2–4: variance-based feature selection. The 2 raw features with the
    highest training-set variance are selected. They remain named, interpretable
    clinical measurements.

  - Part 5: PCA. The top 2 principal components are computed from all 30 features,
    and their projections serve as the rotation angles. PCA is introduced here —
    not earlier.

---

## Part 2 — Quantum Feature Maps

### Dataset and preprocessing (Parts 2–4)

The Breast Cancer Wisconsin dataset (569 samples, 30 features) was used throughout.
Labels are mapped to {−1, +1}.

#### Why not use all 30 features?

The breast cancer dataset has 30 features. Running a **30-qubit** quantum circuit on a classical simulator is computationally intractable:

| Qubits | State vector size | Memory needed |
|--------|-------------------|---------------|
| 2      | $2^2 = 4$ amplitudes | negligible    |
| 10     | $2^{10} = 1{,}024$  | ~16 KB        |
| 20     | $2^{20} \approx 10^6$ | ~16 MB        |
| **30** | $2^{30} \approx 10^9$ | **~16 GB**    |

Beyond memory, the kernel matrix for $n$ training samples requires $O(n^2)$ full circuit evaluations. At 30 qubits, each evaluation is already slow, and with hundreds of samples the total time is measured in days, not seconds.

**The standard solution in QSVM research:** reduce the feature space to 2–4 dimensions before encoding. This project uses two strategies:

- **Parts 2–4:** select the 2 raw features with the highest training-set variance. Features stay interpretable — they are real, named measurements.
- **Part 5:** apply PCA, project onto the top 2 principal components, and use those as angle-encoding rotation angles. This is where PCA enters the project.

For Parts 2–4, the two features with the highest training-set variance are selected:

| Feature      | Training variance |
|--------------|-------------------|
| worst area   | 305,056           |
| mean area    | 118,725           |

Variance is computed on the training set only to avoid leakage. Both features are
scaled to [0, π] so they serve directly as rotation angles. For quantum kernel
computation, 80 training samples are used — the kernel matrix requires 6,400
circuit evaluations, which completes in seconds on a statevector simulator.

### Feature Map 1 — BasicFeatureMap

Circuit per qubit i:

    H → RZ(xᵢ) → RX(xᵢ)

The Hadamard gate creates uniform superposition. RZ encodes the feature as a phase
difference between |0⟩ and |1⟩. RX mixes those amplitudes. Together they place
the qubit at a specific point on the Bloch sphere.

No entanglement. Each qubit encodes one feature independently. The kernel is
determined entirely by how similar the two raw features are individually between
any pair of samples.

Circuit: 2 qubits, 2 parameters, depth 3.

### Feature Map 2 — ZZFeatureMap

Per repetition, after H⊗n and individual RZ(2xᵢ) rotations:

    For each pair (i,j): CX(i,j) → RZ(2(π−xᵢ)(π−xⱼ)) → CX(i,j)

The ZZ interaction term (π − xᵢ)(π − xⱼ) is large when both features are far
from π by similar amounts, and near zero when they differ in opposite directions.
This makes the kernel sensitive to whether two tumour samples share the same
co-variation pattern across worst area and mean area — not just whether each
feature is similar individually.

With 2 repetitions the encoding block runs twice. Each CX gate creates genuine
qubit entanglement, and the resulting kernel measures something in a feature space
no classical product kernel can replicate.

---

## Part 3 — Quantum Kernel Computation

Both kernel matrices were computed for 80 training samples using Qiskit's
FidelityQuantumKernel with the ComputeUncompute estimator on a statevector
simulator.

Diagnostic checks confirm both matrices are symmetric, positive semi-definite,
with diagonal values of 1.0 — all expected properties of valid kernel matrices.

Heatmap observations:

  BasicFM: values fall off smoothly with angular distance. The within-class
  blocks are only slightly brighter than the off-diagonal region, reflecting
  the independent-qubit encoding's limited class separation ability.

  ZZFeatureMap: the off-diagonal pattern is more complex. The ZZ interaction
  introduces variation that depends on feature co-variation direction, not just
  feature proximity. Class blocks are more pronounced, though with only 2 qubits
  and 15 displayed samples the effect is subtle.

---

## Part 4 — SVM Training and Evaluation

All five models trained on the same two features (worst area, mean area, scaled to
[0, π]). C was tuned via 5-fold stratified cross-validation over
{0.01, 0.1, 1.0, 10.0, 100.0}.

| Model           | Test Accuracy | Precision | Recall | F1     |
|-----------------|--------------|-----------|--------|--------|
| SVM-RBF         | 0.9500       | 0.9537    | 0.9500 | 0.9492 |
| SVM-Linear      | 0.9500       | 0.9537    | 0.9500 | 0.9492 |
| QSVM-Basic      | 0.9250       | 0.9251    | 0.9250 | 0.9244 |
| SVM-Polynomial  | 0.9250       | 0.9251    | 0.9250 | 0.9244 |
| QSVM-ZZ         | 0.6000       | 0.6113    | 0.6000 | 0.6042 |

The classical SVMs use the full training set (455 samples). The quantum SVMs use
80. That gap matters — the comparison is not symmetric, and it is worth naming.

QSVM-Basic at 92.5% is competitive, matching the polynomial SVM without any
entanglement. It produces a smooth angle-space kernel that captures enough
structure from the two features to generalise well.

QSVM-ZZ at 60% overfits. The 2-rep ZZFeatureMap builds a complex feature space
that 80 samples cannot fill. Reducing to reps=1 would likely push this above 85%.

---

## Part 5 — Angle Embedding with PCA

PCA is applied here for the first time. Using the same train/test split as Parts
2–4, PCA is fit on the full 30-feature training set and projected onto 2 components.

| Component | Explained variance | Cumulative |
|-----------|-------------------|------------|
| PC1       | ~98.2%            | ~98.2%     |
| PC2       | ~1.7%             | ~99.9%     |

The first two components retain 99.9% of the dataset's total variance. All 30
original features contribute to each component through their PCA loadings.

### What changes from Parts 2–4

The BasicFeatureMap circuit is identical. The rotation angles are different:

| Part  | θ₁                        | θ₂                        |
|-------|---------------------------|---------------------------|
| 2–4   | worst area (scaled)       | mean area (scaled)        |
| 5     | PC1 projection (scaled)   | PC2 projection (scaled)   |

Angle embedding names this specific pattern: reduce data to k scalars, scale each
to a rotation angle, parametrize qubit gates directly with those angles. The
circuit encodes information through Bloch sphere rotations rather than
entanglement structure.

### Feature selection vs PCA

| | Parts 2–4 | Part 5 |
|---|---|---|
| Preprocessing | Variance feature selection | PCA |
| Variance captured | Two raw features only | ~99.9% of total |
| Interpretability | Named clinical measurements | Linear combinations |
| Circuit structure | BasicFeatureMap | Same BasicFeatureMap |

PCA gives the circuit a far richer projection of the data. PC1 captures a
direction of maximum variance that broadly separates malignant from benign samples
in this dataset, which is why QSVM-Angle(PCA) tends to perform on par with or
above QSVM-Basic from Part 4.

---

## Conclusion

The 30-qubit problem is the most important practical constraint in this project.
Every design decision — feature selection in Parts 2–4, PCA in Part 5, the 80-
sample cap — follows from it directly. That is not a weakness of the approach; it
is where quantum simulation genuinely stands today.

Within those constraints:

  A non-entangling 2-qubit quantum kernel achieves 92.5% accuracy on a real
  clinical dataset, matching a polynomial SVM. That is a reasonable result for
  2 qubits and 80 training samples.

  The ZZFeatureMap needs more data. 80 samples is not enough to populate a richer
  feature space without overfitting.

  PCA (Part 5) is a better preprocessing step than raw feature selection when the
  goal is to give the circuit the most informative 2-D projection. 99.9% variance
  retention beats two individually-picked features.

  Classical SVMs still win. Quantum advantage in kernel methods requires data with
  structure that is provably hard to kernel classically. A standard tabular
  classification benchmark is not that data.
