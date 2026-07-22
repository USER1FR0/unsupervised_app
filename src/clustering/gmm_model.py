import numpy as np
from sklearn.mixture import GaussianMixture
from src.clustering.base import ClusteringModel


class GMMModel(ClusteringModel):
    name = "gmm"

    def __init__(self, n_components: int = 3, covariance_type: str = "full", random_state: int = 42):
        super().__init__()
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self._model = None

    def fit(self, X: np.ndarray) -> None:
        self._model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )
        self._model.fit(X)
        self._labels = self._model.predict(X)
        self._fitted = True

    def get_params(self) -> dict:
        return {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "random_state": self.random_state,
        }

    def get_model(self):
        return self._model

    def get_probabilities(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_bic(self, X: np.ndarray) -> float:
        return float(self._model.bic(X))

    def get_aic(self, X: np.ndarray) -> float:
        return float(self._model.aic(X))