from datetime import datetime
from pathlib import Path
import joblib

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_model_bundle(model, scaler, algorithm: str) -> dict:
    """Guarda el modelo y su scaler como .pkl en el filesystem.
    Retorna dict con las rutas relativas."""
    ts = _timestamp()
    model_file = MODELS_DIR / f"{algorithm}_{ts}.pkl"
    scaler_file = MODELS_DIR / f"scaler_{algorithm}_{ts}.pkl"

    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    return {
        "model_file_path": str(model_file.relative_to(MODELS_DIR.parent)),
        "scaler_file_path": str(scaler_file.relative_to(MODELS_DIR.parent)),
        "timestamp": ts,
    }


def load_model_bundle(model_path: str, scaler_path: str) -> tuple:
    """Carga un modelo y su scaler desde el filesystem."""
    root = MODELS_DIR.parent
    model = joblib.load(root / model_path)
    scaler = joblib.load(root / scaler_path)
    return model, scaler


def delete_model_bundle(model_path: str, scaler_path: str) -> None:
    """Elimina los archivos del modelo del filesystem."""
    root = MODELS_DIR.parent
    for path in [model_path, scaler_path]:
        f = root / path
        if f.exists():
            f.unlink()