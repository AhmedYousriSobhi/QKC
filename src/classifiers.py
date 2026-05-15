"""
classifiers.py
──────────────
Train and evaluate SVMs using:
  • Precomputed quantum kernels (Parts 4 & 5)
  • Classical kernels: RBF, polynomial, linear (Part 4 baseline)

Includes grid-searched hyperparameter tuning for the regularization
parameter C, and a clean comparison table builder.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)


# ──────────────────────────────────────────────────────────────
# 1. Quantum SVM (precomputed kernel)
# ──────────────────────────────────────────────────────────────

def train_quantum_svm(K_train, y_train, C_values=None, random_state: int = 42):
    """
    Train an SVM on a precomputed quantum kernel matrix.

    sklearn's SVC with kernel='precomputed' expects:
      - fit(K_train, y_train)   where K_train[i,j] = K(xᵢ, xⱼ)
      - predict(K_test)         where K_test[i,j]  = K(zᵢ, xⱼ)

    Hyperparameter C (regularization) is tuned via 5-fold cross-validation.

    Parameters
    ----------
    K_train   : ndarray (n_train × n_train)
    y_train   : ndarray (n_train,)
    C_values  : list of float  (grid to search over)

    Returns
    -------
    best_model : fitted SVC
    best_C     : float
    cv_results : dict
    """
    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    param_grid = {"C": C_values}

    svm = SVC(kernel="precomputed")
    search = GridSearchCV(svm, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    search.fit(K_train, y_train)

    return search.best_estimator_, search.best_params_["C"], search.cv_results_


def evaluate_quantum_svm(model, K_test, y_test, label: str = "QSVM"):
    """Predict and return a metrics dict."""
    y_pred = model.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n── {label} Results ──")
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    return {"label": label, "accuracy": acc, "report": report, "cm": cm, "y_pred": y_pred}


# ──────────────────────────────────────────────────────────────
# 2. Classical SVM baselines
# ──────────────────────────────────────────────────────────────

CLASSICAL_KERNELS = {
    "RBF"        : {"kernel": "rbf",   "gamma": "scale"},
    "Polynomial" : {"kernel": "poly",  "degree": 3, "gamma": "scale"},
    "Linear"     : {"kernel": "linear"},
}


def train_classical_svms(X_train, y_train, C_values=None, random_state: int = 42):
    """
    Train RBF, polynomial, and linear SVMs with grid-searched C.

    Returns a dict  {kernel_name: (fitted_model, best_C)}
    """
    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = {}

    for name, kwargs in CLASSICAL_KERNELS.items():
        svm = SVC(**kwargs)
        search = GridSearchCV(
            SVC(**kwargs), {"C": C_values}, cv=cv, scoring="accuracy", n_jobs=-1
        )
        search.fit(X_train, y_train)
        results[name] = (search.best_estimator_, search.best_params_["C"])
        print(f"  {name:12s}: best C={search.best_params_['C']:.4f}  "
              f"CV acc={search.best_score_:.4f}")

    return results


def evaluate_classical_svms(models_dict, X_test, y_test):
    """
    Evaluate each classical SVM and return a list of metrics dicts.
    """
    all_results = []
    for name, (model, best_C) in models_dict.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        all_results.append({
            "label"   : f"SVM-{name}",
            "accuracy": acc,
            "best_C"  : best_C,
            "report"  : report,
            "cm"      : cm,
            "y_pred"  : y_pred,
        })
        print(f"  SVM-{name:12s}: accuracy = {acc:.4f}")
    return all_results


# ──────────────────────────────────────────────────────────────
# 3. Comparison table
# ──────────────────────────────────────────────────────────────

def build_comparison_table(quantum_results: list, classical_results: list) -> pd.DataFrame:
    """
    Combine quantum and classical results into a single summary DataFrame.

    Parameters
    ----------
    quantum_results  : list of dicts (output of evaluate_quantum_svm)
    classical_results: list of dicts (output of evaluate_classical_svms)

    Returns
    -------
    df : pd.DataFrame with columns [Model, Accuracy, Precision, Recall, F1]
    """
    rows = []

    for r in quantum_results + classical_results:
        rep = r["report"]
        # weighted averages across classes
        rows.append({
            "Model"    : r["label"],
            "Accuracy" : round(r["accuracy"], 4),
            "Precision": round(rep["weighted avg"]["precision"], 4),
            "Recall"   : round(rep["weighted avg"]["recall"], 4),
            "F1-Score" : round(rep["weighted avg"]["f1-score"], 4),
        })

    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return df
