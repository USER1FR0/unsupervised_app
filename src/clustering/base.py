from abc import ABC, abstractmethod
import numpy as np


class ClusteringModel(ABC):
    """Interfaz común para todos los algoritmos de clustering."""

    name: str = "base"

    def __init__(self):
        self._fitted = False
        self._labels = None

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Entrena el modelo con los datos escalados."""
        ...

    @abstractmethod
    def get_params(self) -> dict:
        """Retorna los hiperparámetros usados."""
        ...

    @abstractmethod
    def get_model(self):
        """Retorna el modelo interno de sklearn (para persistencia)."""
        ...

    @property
    def labels(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("El modelo no ha sido entrenado.")
        return self._labels

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_cluster_count(self) -> int:
        """Número de clusters detectados (excluye ruido en DBSCAN)."""
        unique = set(self._labels)
        unique.discard(-1)
        return len(unique)

    def get_outlier_count(self) -> int:
        """Número de puntos etiquetados como ruido. Solo aplica a DBSCAN."""
        return int(np.sum(self._labels == -1))