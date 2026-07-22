from src.clustering.base import ClusteringModel
from src.clustering.kmeans_model import KMeansModel
from src.clustering.hierarchical_model import HierarchicalModel
from src.clustering.dbscan_model import DBSCANModel
from src.clustering.gmm_model import GMMModel


ALGORITHMS = {
    "kmeans": {
        "class": KMeansModel,
        "label": "K-Means",
        "description": "Agrupa puntos alrededor de k centroides. Rápido e interpretable.",
    },
    "hierarchical": {
        "class": HierarchicalModel,
        "label": "Clusterización Jerárquica",
        "description": "Construye una jerarquía anidada de clusters mediante fusión.",
    },
    "dbscan": {
        "class": DBSCANModel,
        "label": "DBSCAN",
        "description": "Basado en densidad. Detecta formas arbitrarias y outliers.",
    },
    "gmm": {
        "class": GMMModel,
        "label": "Modelo de Mezcla Gaussiana (GMM)",
        "description": "Probabilístico. Modela solapamiento entre grupos.",
    },
}


def create_model(algorithm: str, **params) -> ClusteringModel:
    """Instancia un modelo por su nombre."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algoritmo desconocido: {algorithm}")
    return ALGORITHMS[algorithm]["class"](**params)