from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def kmeans_labels(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    return KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(X)

def agglomerative_labels(X: np.ndarray, k: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k).fit_predict(X)

def dbscan_labels(X: np.ndarray, eps: float = 0.9, min_samples: int = 4) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
