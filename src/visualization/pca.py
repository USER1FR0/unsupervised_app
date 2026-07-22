import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def project_2d(X_scaled: np.ndarray) -> tuple[np.ndarray, PCA]:
    """Proyecta los datos escalados a 2D mediante PCA.
    Retorna la matriz proyectada y el modelo PCA."""
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    return X_2d, pca


def variance_explained(pca: PCA) -> dict:
    """Retorna la varianza explicada por cada componente."""
    ratios = pca.explained_variance_ratio_
    return {
        "pc1": round(float(ratios[0]), 4),
        "pc2": round(float(ratios[1]), 4),
        "total": round(float(sum(ratios)), 4),
    }