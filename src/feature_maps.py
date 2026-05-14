"""
feature_maps.py
───────────────
Two quantum feature maps used in Parts 2, 3, 4, and 5:

  1. BasicFeatureMap  — angle-encoding via RX/RZ rotations (no entanglement).
  2. EntangledFeatureMap — ZZFeatureMap-style with CX entanglement and two reps.

Both return Qiskit QuantumCircuit objects that encode classical data points
into quantum states |φ(x)⟩ before kernel computation.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap


# ──────────────────────────────────────────────────────────────
# 1. Basic Feature Map  (no entanglement)
# ──────────────────────────────────────────────────────────────

def build_basic_feature_map(n_features: int) -> QuantumCircuit:
    """
    Build a simple angle-encoding circuit with RX and RZ gates.

    For each feature xᵢ the circuit applies:
        H → RZ(xᵢ) → RX(xᵢ)

    No entanglement between qubits — each qubit independently encodes one
    classical feature.  This is the simplest possible quantum feature map and
    acts as our baseline.

    Parameters
    ----------
    n_features : int
        Number of input features (= number of qubits).

    Returns
    -------
    qc : QuantumCircuit
        Parametrized quantum circuit with `n_features` parameters named x[0..n-1].
    """
    x = ParameterVector("x", length=n_features)
    qc = QuantumCircuit(n_features)

    for i in range(n_features):
        qc.h(i)           # Hadamard: puts qubit in superposition
        qc.rz(x[i], i)   # Z-rotation by the feature value
        qc.rx(x[i], i)   # X-rotation by the same value (adds phase mixing)

    qc.name = "BasicFM"
    return qc


# ──────────────────────────────────────────────────────────────
# 2. Entangled Feature Map  (ZZFeatureMap)
# ──────────────────────────────────────────────────────────────

def build_entangled_feature_map(n_features: int, reps: int = 2) -> QuantumCircuit:
    """
    Build a ZZFeatureMap: Qiskit's standard entangled feature map.

    Structure (per repetition):
        H on all qubits
        RZ(2·xᵢ) on each qubit
        For each pair (i,j): CX(i,j) → RZ(2·(π−xᵢ)·(π−xⱼ)) → CX(i,j)

    The ZZ interaction between qubits encodes correlations between feature pairs
    in the Hilbert space, which the basic map completely ignores.  More
    repetitions → richer feature space, but longer circuits.

    Parameters
    ----------
    n_features : int
        Number of input features (= number of qubits).
    reps : int
        Number of times the encoding block is repeated.

    Returns
    -------
    qc : QuantumCircuit (Qiskit library ZZFeatureMap)
    """
    qc = ZZFeatureMap(feature_dimension=n_features, reps=reps)
    qc.name = "ZZFeatureMap"
    return qc


# ──────────────────────────────────────────────────────────────
# 3. Circuit description helpers
# ──────────────────────────────────────────────────────────────

def describe_feature_map(name: str):
    descriptions = {
        "BasicFM": (
            "Basic Feature Map\n"
            "  Gates: H → RZ(xᵢ) → RX(xᵢ) per qubit\n"
            "  Entanglement: None\n"
            "  Expressivity: Low — each qubit is independent.  Fast to\n"
            "  simulate and easy to interpret, but the feature space is\n"
            "  limited to rotations on the Bloch sphere.\n"
        ),
        "ZZFeatureMap": (
            "Entangled Feature Map (ZZFeatureMap)\n"
            "  Gates: H → RZ(2xᵢ) → CX → RZ(2(π−xᵢ)(π−xⱼ)) → CX per pair\n"
            "  Entanglement: Full pairwise via CX gates\n"
            "  Expressivity: High — captures pairwise feature correlations\n"
            "  through ZZ interactions.  The kernel K(x,z) measures overlap\n"
            "  in a space that is exponentially large in qubit count.\n"
        ),
    }
    return descriptions.get(name, "No description available.")
