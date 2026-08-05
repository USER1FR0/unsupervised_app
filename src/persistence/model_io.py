"""Persistencia de modelos, scaler y PCA en filesystem via joblib."""
from datetime import datetime
from pathlib import Path
from typing import Optional
import joblib
import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_model_bundle(model, scaler, algorithm: str, pca=None) -> dict:
    """Guarda modelo, scaler y opcionalmente PCA como .pkl.
    Retorna dict con rutas relativas al root del proyecto."""
    ts = _timestamp()
    model_file = MODELS_DIR / f"{algorithm}_{ts}.pkl"
    scaler_file = MODELS_DIR / f"scaler_{algorithm}_{ts}.pkl"

    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    bundle = {
        "model_file_path": str(model_file.relative_to(MODELS_DIR.parent)),
        "scaler_file_path": str(scaler_file.relative_to(MODELS_DIR.parent)),
        "timestamp": ts,
    }

    if pca is not None:
        pca_file = MODELS_DIR / f"pca_{algorithm}_{ts}.pkl"
        joblib.dump(pca, pca_file)
        bundle["pca_file_path"] = str(pca_file.relative_to(MODELS_DIR.parent))

    return bundle


def load_model_bundle(model_path: str, scaler_path: str,
                       pca_path: Optional[str] = None):
    """Carga modelo, scaler y (si existe) PCA. Retorna tupla (model, scaler, pca)."""
    root = MODELS_DIR.parent
    model = joblib.load(root / model_path)
    scaler = joblib.load(root / scaler_path)
    pca = joblib.load(root / pca_path) if pca_path else None
    return model, scaler, pca


def delete_model_bundle(model_path: str, scaler_path: str,
                          pca_path: Optional[str] = None) -> None:
    root = MODELS_DIR.parent
    paths = [model_path, scaler_path]
    if pca_path:
        paths.append(pca_path)
    for path in paths:
        if not path:
            continue
        f = root / path
        if f.exists():
            f.unlink()


def bundle_exists(model_metadata: dict) -> bool:
    """True si los .pkl esperados por la metadata siguen en disco."""
    root = MODELS_DIR.parent
    for key in ("model_file_path", "scaler_file_path"):
        p = model_metadata.get(key)
        if not p or not (root / p).exists():
            return False
    return True


def predict_with_saved_model(model_metadata: dict, X_new: np.ndarray) -> dict:
    """Aplica un modelo guardado a datos nuevos.

    Retorna:
        {
            'labels': np.ndarray,          # cluster asignado
            'probabilities': np.ndarray|None,  # solo GMM
            'X_scaled': np.ndarray,        # datos escalados
            'X_2d': np.ndarray|None,       # proyeccion PCA 2D si el PCA se guardo
        }
    """
    model, scaler, pca = load_model_bundle(
        model_metadata["model_file_path"],
        model_metadata["scaler_file_path"],
        model_metadata.get("pca_file_path"),
    )
    X_scaled = scaler.transform(X_new)

    labels = model.predict(X_scaled)
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_scaled)
        except Exception:
            probabilities = None

    X_2d = pca.transform(X_scaled) if pca is not None else None

    return {
        "labels": labels,
        "probabilities": probabilities,
        "X_scaled": X_scaled,
        "X_2d": X_2d,
    }
