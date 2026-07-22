import numpy as np
from sklearn.cluster import AgglomerativeClustering
from src.clustering.base import ClusteringModel


class HierarchicalModel(ClusteringModel):
    name = "hierarchical"

    def __init__(self, n_clusters: int = 3, linkage: str = "ward"):
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage = linkage
        self._model = None

    def fit(self, X: np.ndarray) -> None:
        self._model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
        )
        self._labels = self._model.fit_predict(X)
        self._fitted = True

    def get_params(self) -> dict:
        return {
            "n_clusters": self.n_clusters,
            "linkage": self.linkage,
        }

    def get_model(self):
        return self._model