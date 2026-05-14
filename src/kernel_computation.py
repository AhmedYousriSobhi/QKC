"""
kernel_computation.py
─────────────────────
Compute quantum kernel matrices using Qiskit's FidelityQuantumKernel.

The quantum kernel between two data points x and z is defined as:
    K(x, z) = |⟨φ(x)|φ(z)⟩|²

where |φ(x)⟩ is the quantum state produced by the feature map circuit.
This is estimated by preparing both states on a simulator and measuring
the probability of returning to |0⟩ (the "swap test" approach).
"""

import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute


# ──────────────────────────────────────────────────────────────
# 1. Build kernel object
# ──────────────────────────────────────────────────────────────

def build_quantum_kernel(feature_map):
    """
    Wrap a feature map circuit in a FidelityQuantumKernel.

    Uses the ComputeUncompute fidelity estimator:
      - Prepares |φ(x)⟩ and then applies the *inverse* circuit for |φ(z)⟩.
      - Measures the probability of getting the all-zeros bitstring.
      - That probability equals |⟨φ(x)|φ(z)⟩|².

    Parameters
    ----------
    feature_map : QuantumCircuit
        A parametrized circuit (output of build_basic_feature_map or
        build_entangled_feature_map).

    Returns
    -------
    qk : FidelityQuantumKernel
    """
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    qk = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity,
    )
    return qk


# ──────────────────────────────────────────────────────────────
# 2. Evaluate the kernel matrix
# ──────────────────────────────────────────────────────────────

def compute_kernel_matrix(qk, X_train, X_test=None):
    """
    Compute the kernel matrix K for training and (optionally) test data.

    If X_test is None, returns K_train  (shape: n_train × n_train).
    Otherwise also returns K_test        (shape: n_test  × n_train).

    Both are needed by sklearn's SVM when using a precomputed kernel.
    """
    K_train = qk.evaluate(x_vec=X_train)

    if X_test is not None:
        K_test = qk.evaluate(x_vec=X_test, y_vec=X_train)
        return K_train, K_test

    return K_train


# ──────────────────────────────────────────────────────────────
# 3. Kernel diagnostics
# ──────────────────────────────────────────────────────────────

def kernel_diagnostics(K, name: str = "Quantum Kernel"):
    """
    Print basic sanity checks for a kernel matrix:
      - Symmetry (K ≈ Kᵀ)
      - Diagonal values (should be 1.0 for pure states)
      - Value range
    """
    print(f"\n── {name} Diagnostics ──")
    print(f"  Shape         : {K.shape}")
    print(f"  Diagonal mean : {np.diag(K).mean():.4f}  (should be ≈1.0)")
    print(f"  Min / Max     : {K.min():.4f} / {K.max():.4f}")
    print(f"  Symmetric     : {np.allclose(K, K.T, atol=1e-6)}")
    # Check positive semi-definiteness via eigenvalues
    eigvals = np.linalg.eigvalsh(K)
    print(f"  Min eigenvalue: {eigvals.min():.6f}  (should be ≥ 0 for valid kernel)")
