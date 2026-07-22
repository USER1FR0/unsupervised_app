import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


def elbow_kmeans(X: np.ndarray, k_range: range = range(2, 11)) -> dict:
    """Método del codo para K-Means: inercia (WCSS) para distintos k."""
    inertias = []
    silhouettes = []
    for k in k_range:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        model.fit(X)
        inertias.append(float(model.inertia_))
        silhouettes.append(float(silhouette_score(X, model.labels_)))

    return {
        "k": list(k_range),
        "inertia": inertias,
        "silhouette": silhouettes,
        "suggested_k": _detect_elbow(list(k_range), inertias),
    }


def bic_gmm(X: np.ndarray, k_range: range = range(2, 11), covariance_type: str = "full") -> dict:
    """BIC para GMM en distintos k. Menor BIC = mejor modelo."""
    bics = []
    aics = []
    for k in k_range:
        model = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42)
        model.fit(X)
        bics.append(float(model.bic(X)))
        aics.append(float(model.aic(X)))

    suggested_k = list(k_range)[int(np.argmin(bics))]
    return {
        "k": list(k_range),
        "bic": bics,
        "aic": aics,
        "suggested_k": suggested_k,
    }


def k_distances(X: np.ndarray, k: int = 5) -> dict:
    """Distancias al k-ésimo vecino más cercano para elegir eps de DBSCAN.
    El 'codo' en la curva sugiere el eps óptimo."""
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    kth_distances = np.sort(distances[:, k - 1])

    suggested_eps = _detect_elbow(list(range(len(kth_distances))), kth_distances.tolist())
    suggested_eps_value = float(kth_distances[suggested_eps]) if suggested_eps else float(np.median(kth_distances))

    return {
        "distances": kth_distances.tolist(),
        "k": k,
        "suggested_eps": round(suggested_eps_value, 3),
    }


def hierarchical_linkage_matrix(X: np.ndarray, linkage_method: str = "ward") -> np.ndarray:
    """Matriz de linkage para dendrograma."""
    return linkage(X, method=linkage_method)


def _detect_elbow(x_vals: list, y_vals: list) -> int:
    """Detecta el codo por método de máxima distancia a la línea recta.
    Retorna el índice donde está el codo."""
    if len(x_vals) < 3:
        return x_vals[0] if x_vals else None

    x = np.array(x_vals)
    y = np.array(y_vals)

    # Línea entre primer y último punto
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    distances = []
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        # Distancia punto-línea
        d = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(d)

    elbow_idx = int(np.argmax(distances))
    return x_vals[elbow_idx]