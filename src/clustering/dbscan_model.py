import numpy as np
from sklearn.cluster import DBSCAN
from src.clustering.base import ClusteringModel


class DBSCANModel(ClusteringModel):
    name = "dbscan"

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self._model = None

    def fit(self, X: np.ndarray) -> None:
        self._model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
        )
        self._labels = self._model.fit_predict(X)
        self._fitted = True

    def get_params(self) -> dict:
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
        }

    def get_model(self):
        return self._model