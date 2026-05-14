"""
data_utils.py
─────────────
Load, inspect, and prepare the Breast Cancer Wisconsin dataset for quantum
classification. Handles PCA reduction for angle-embedding experiments (Part 5).
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# ──────────────────────────────────────────────────────────────
# 1. Raw dataset
# ──────────────────────────────────────────────────────────────

def load_dataset():
    """Return the full Breast Cancer Wisconsin dataset as arrays."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    # Map to {-1, +1} labels, which SVM convention prefers
    y = np.where(y == 0, -1, 1)
    return X, y, data.feature_names, data.target_names


# ──────────────────────────────────────────────────────────────
# 2. Preprocessing pipeline
# ──────────────────────────────────────────────────────────────

def preprocess(X, y,
               n_components: int = 2,
               test_size: float = 0.20,
               random_state: int = 42,
               apply_pca: bool = True,
               max_train_samples: int = None):
    """
    Full preprocessing pipeline:
      1. Train/test split (stratified, optional subsampling for quantum speed).
      2a. apply_pca=False (Parts 2–4): select n_components raw features by
          training-set variance. PCA-free — features stay interpretable.
      2b. apply_pca=True  (Part 5):   PCA onto n_components directions of max
          variance. Components become the angle-embedding rotation angles.
      3. MinMax scaling to [0, π] — maps features to valid rotation angles.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    pca      : fitted PCA object (for inspection / reuse)
    scaler   : fitted MinMaxScaler (for inspection / reuse)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Optional: subsample training set (quantum kernels scale as O(n²))
    if max_train_samples and len(X_train) > max_train_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_train), size=max_train_samples, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    if apply_pca:
        # Part 5 path: genuine PCA — projects onto directions of max variance.
        # The PCA components are linear combinations of ALL 30 features.
        pca = PCA(n_components=n_components, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_test  = pca.transform(X_test)
    else:
        # Parts 2-4 path: pick the n_components raw features with the highest
        # training-set variance. No PCA — features remain interpretable, and
        # no information from X_test leaks into the selection.
        feature_vars = X_train.var(axis=0)
        top_idx = np.argsort(feature_vars)[::-1][:n_components]
        X_train = X_train[:, top_idx]
        X_test  = X_test[:, top_idx]
        pca = None

    # Scale to [0, π] so features can directly serve as rotation angles
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, pca, scaler


# ──────────────────────────────────────────────────────────────
# 3. Quick summary
# ──────────────────────────────────────────────────────────────

def dataset_summary(X, y, feature_names=None):
    """Print a concise overview of the dataset."""
    classes, counts = np.unique(y, return_counts=True)
    print("=" * 50)
    print("Breast Cancer Wisconsin Dataset")
    print("=" * 50)
    print(f"  Samples   : {len(X)}")
    print(f"  Features  : {X.shape[1]}")
    print(f"  Classes   : {dict(zip(classes, counts))}")
    print(f"  Malignant : {counts[0]}  |  Benign : {counts[1]}")
    if feature_names is not None:
        print(f"\n  First 5 features: {list(feature_names[:5])}")
    print("=" * 50)
