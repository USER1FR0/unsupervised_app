import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def compute_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Calcula métricas de calidad de clustering.
    Retorna dict con métricas que aplican al modelo."""
    metrics = {}

    # Excluir ruido (label = -1) para métricas
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]

    unique_clusters = set(labels_clean)
    n_clusters = len(unique_clusters)
    n_outliers = int(np.sum(labels == -1))

    metrics["n_clusters"] = n_clusters
    metrics["n_outliers"] = n_outliers
    metrics["n_samples"] = len(X)
    metrics["n_samples_valid"] = len(X_clean)

    # Silhouette y Davies-Bouldin necesitan al menos 2 clusters
    if n_clusters >= 2:
        metrics["silhouette"] = round(float(silhouette_score(X_clean, labels_clean)), 4)
        metrics["davies_bouldin"] = round(float(davies_bouldin_score(X_clean, labels_clean)), 4)
        metrics["calinski_harabasz"] = round(float(calinski_harabasz_score(X_clean, labels_clean)), 4)
    else:
        metrics["silhouette"] = None
        metrics["davies_bouldin"] = None
        metrics["calinski_harabasz"] = None

    return metrics


def interpret_silhouette(score: float) -> str:
    """Interpretación cualitativa del silhouette score."""
    if score is None:
        return "No aplicable"
    if score >= 0.7:
        return "Estructura fuerte"
    if score >= 0.5:
        return "Estructura razonable"
    if score >= 0.25:
        return "Estructura débil"
    return "Sin estructura clara"