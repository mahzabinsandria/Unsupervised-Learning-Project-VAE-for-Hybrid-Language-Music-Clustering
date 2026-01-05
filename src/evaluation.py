from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_pred != -1
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return float("nan")
    total = 0
    for c in np.unique(y_pred):
        idx = (y_pred == c)
        _, counts = np.unique(y_true[idx], return_counts=True)
        total += counts.max()
    return total / len(y_true)

def eval_internal(X: np.ndarray, labels: np.ndarray, silhouette_metric: str = "euclidean") -> dict:
    """
    Handles DBSCAN by ignoring noise points (-1).
    If <2 clusters remain, returns NaNs (so you know DBSCAN failed).
    """
    labels = np.asarray(labels)
    mask = labels != -1
    X2 = X[mask]
    y2 = labels[mask]

    uniq = np.unique(y2)
    if len(uniq) < 2 or len(X2) < 3:
        return {"silhouette": float("nan"), "calinski_harabasz": float("nan"), "davies_bouldin": float("nan")}

    return {
        "silhouette": float(silhouette_score(X2, y2, metric=silhouette_metric)),
        "calinski_harabasz": float(calinski_harabasz_score(X2, y2)),
        "davies_bouldin": float(davies_bouldin_score(X2, y2)),
    }

def eval_with_labels(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = y_pred != -1
    y_true2 = np.asarray(y_true)[mask]
    y_pred2 = np.asarray(y_pred)[mask]

    if len(np.unique(y_pred2)) < 2 or len(y_true2) == 0:
        return {"ari": float("nan"), "nmi": float("nan"), "purity": float("nan")}

    return {
        "ari": float(adjusted_rand_score(y_true2, y_pred2)),
        "nmi": float(normalized_mutual_info_score(y_true2, y_pred2)),
        "purity": float(purity_score(y_true2, y_pred2)),
    }
