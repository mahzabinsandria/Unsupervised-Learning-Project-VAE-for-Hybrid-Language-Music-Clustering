from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

def kmeans_labels(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    return KMeans(n_clusters=k, random_state=seed, n_init=50).fit_predict(X)

def agglomerative_labels(X: np.ndarray, k: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k).fit_predict(X)

def dbscan_auto(
    X: np.ndarray,
    min_samples: int = 4,
    quantiles = (0.60, 0.70, 0.80, 0.85, 0.90, 0.95),
):
    """
    Standardizes X, then tries multiple eps values based on kNN distance quantiles.
    Picks the eps that yields >=2 clusters and best silhouette (ignoring noise).
    Returns: labels, chosen_eps, X_scaled
    """
    Xs = StandardScaler().fit_transform(X)

    k = max(2, min_samples)
    nn = NearestNeighbors(n_neighbors=k).fit(Xs)
    dists, _ = nn.kneighbors(Xs)
    kth = np.sort(dists[:, -1])

    best = None
    for q in quantiles:
        eps = float(np.quantile(kth, q))
        y = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)

        mask = y != -1
        uniq = np.unique(y[mask])
        if len(uniq) < 2:
            continue

        # silhouette on non-noise only
        s = silhouette_score(Xs[mask], y[mask])
        if (best is None) or (s > best["s"]):
            best = {"labels": y, "eps": eps, "s": s}

    if best is None:
        # fallback: use median eps even if it fails (you will see NaNs)
        eps = float(np.quantile(kth, 0.85))
        y = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)
        return y, eps, Xs

    return best["labels"], best["eps"], Xs
