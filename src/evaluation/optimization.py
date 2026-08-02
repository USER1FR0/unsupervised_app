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
        "suggested_k": _kneedle_elbow(list(k_range), inertias, direction="decreasing"),
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

    Retorna 3 sugerencias derivadas de la distribución de k-distancias:
    - eps_aggressive: percentil 50 → clusters chicos y más ruido
    - eps_moderate:   percentil 75 → equilibrio (recomendación principal)
    - eps_conservative: percentil 90 → menos clusters, casi sin ruido

    También intenta detectar un codo real con Kneedle; si el codo cae
    después del p90 (curva casi lineal + salto al final), se ignora y se
    usa el p75 como sugerencia por defecto.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    kth_distances = np.sort(distances[:, k - 1])

    p50 = float(np.percentile(kth_distances, 50))
    p75 = float(np.percentile(kth_distances, 75))
    p90 = float(np.percentile(kth_distances, 90))

    # Codo por Kneedle (solo si cae en zona útil)
    knee_idx = _kneedle_elbow(
        list(range(len(kth_distances))),
        kth_distances.tolist(),
        direction="increasing",
    )
    knee_eps = float(kth_distances[knee_idx]) if knee_idx is not None else None
    if knee_eps is not None and knee_eps > p90:
        # curva demasiado lineal, el "codo" quedó en el jump final
        knee_eps = None

    return {
        "distances": kth_distances.tolist(),
        "k": k,
        "suggested_eps": round(p75, 3),
        "eps_aggressive": round(p50, 3),
        "eps_moderate": round(p75, 3),
        "eps_conservative": round(p90, 3),
        "eps_knee": round(knee_eps, 3) if knee_eps is not None else None,
    }


def hierarchical_linkage_matrix(X: np.ndarray, linkage_method: str = "ward") -> np.ndarray:
    """Matriz de linkage para dendrograma."""
    return linkage(X, method=linkage_method)


# --------- Detección de codo (Kneedle simplificado) ---------

def _kneedle_elbow(x_vals: list, y_vals: list, direction: str = "decreasing") -> int:
    """Detecta el 'codo' (knee point) en una curva monótona.

    Algoritmo Kneedle simplificado:
      1. Normalizar (x, y) al rango [0, 1].
      2. Para curvas decrecientes (típico de inercia en elbow K-Means)
         el codo es donde la diferencia (y_norm - (1 - x_norm)) es máxima.
      3. Para curvas crecientes (típico de k-distancias) el codo es donde
         la diferencia (y_norm - x_norm) es máxima.

    Retorna el x correspondiente al codo. None si no hay suficientes puntos.
    """
    if len(x_vals) < 3:
        return x_vals[0] if x_vals else None

    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)

    x_range = x.max() - x.min() + 1e-9
    y_range = y.max() - y.min() + 1e-9
    x_norm = (x - x.min()) / x_range
    y_norm = (y - y.min()) / y_range

    if direction == "decreasing":
        # Codo = mayor distancia bajo la diagonal (1,0) -> (0,1)
        diff = (1 - x_norm) - y_norm
    else:
        # Codo = mayor distancia sobre la diagonal (0,0) -> (1,1)
        diff = y_norm - x_norm

    idx = int(np.argmax(diff))
    return x_vals[idx]


# Alias por compatibilidad hacia atrás
def _detect_elbow(x_vals: list, y_vals: list) -> int:
    return _kneedle_elbow(x_vals, y_vals, direction="decreasing")
