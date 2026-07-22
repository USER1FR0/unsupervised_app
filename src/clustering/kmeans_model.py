import numpy as np
from sklearn.cluster import KMeans
from src.clustering.base import ClusteringModel


class KMeansModel(ClusteringModel):
    name = "kmeans"

    def __init__(self, n_clusters: int = 3, init: str = "k-means++", n_init: int = 10, random_state: int = 42):
        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self._model = None

    def fit(self, X: np.ndarray) -> None:
        self._model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self._model.fit(X)
        self._labels = self._model.labels_
        self._fitted = True

    def get_params(self) -> dict:
        return {
            "n_clusters": self.n_clusters,
            "init": self.init,
            "n_init": self.n_init,
            "random_state": self.random_state,
        }

    def get_model(self):
        return self._model

    def get_inertia(self) -> float:
        return float(self._model.inertia_)

    def get_centroids(self) -> np.ndarray:
        return self._model.cluster_centers_