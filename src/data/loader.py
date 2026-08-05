"""Carga y validacion de datasets desde CSV.

Flujo unico: el usuario sube un CSV desde la landing y se guarda en
st.session_state['df']. Todas las paginas consumen esa referencia.
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


DIMENSIONS = ["O", "C", "E", "A", "N"]
DIMENSION_LABELS = {
    "O": "Apertura",
    "C": "Conciencia",
    "E": "Extraversion",
    "A": "Amabilidad",
    "N": "Neuroticismo",
}

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# CSVs de ejemplo disponibles en data/. El usuario puede cargar cualquiera
# de estos con un click en la landing, o subir el suyo propio.
SAMPLE_DATASETS = {
    "real": "real.csv",
    "sintetico": "synthetic.csv",
    "demo": "demo.csv",
}

REQUIRED_COLUMNS = DIMENSIONS + ["edad", "genero", "estado", "municipio"]


# ---------- Validacion y carga ----------

def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas de un dataframe crudo (columnas + tipos + raw_answers)."""
    df = df.copy()

    # Normalizar timestamp -> submitted_at
    if "timestamp" in df.columns and "submitted_at" not in df.columns:
        df = df.rename(columns={"timestamp": "submitted_at"})

    if "submitted_at" in df.columns:
        df["submitted_at"] = pd.to_datetime(df["submitted_at"], errors="coerce",
                                             dayfirst=True)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"El CSV no tiene las columnas obligatorias: {', '.join(missing)}"
        )

    for dim in DIMENSIONS:
        df[dim] = pd.to_numeric(df[dim], errors="coerce")

    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")

    df = df.dropna(subset=DIMENSIONS + ["edad"]).reset_index(drop=True)
    df["edad"] = df["edad"].astype(int)

    # Validar rango OCEAN
    out_of_range = ((df[DIMENSIONS] < 1) | (df[DIMENSIONS] > 5)).any(axis=1).sum()
    if out_of_range > 0:
        raise ValueError(
            f"{out_of_range} filas con valores OCEAN fuera del rango 1-5."
        )

    for col in ["genero", "estado", "municipio"]:
        df[col] = df[col].astype(str)

    # Reconstruir raw_answers a partir de q1..q20 (para consistencia interna / alfa)
    q_cols = [f"q{i}" for i in range(1, 21)]
    if all(c in df.columns for c in q_cols):
        df["raw_answers"] = df[q_cols].values.tolist()

    return df


def load_csv_from_upload(uploaded_file) -> pd.DataFrame:
    """Lee un CSV desde el file_uploader de Streamlit."""
    raw = uploaded_file.read()
    df = pd.read_csv(BytesIO(raw))
    return _normalize_dataframe(df)


def load_csv_from_path(path: Path) -> pd.DataFrame:
    """Lee un CSV desde disco (para datasets de ejemplo en data/)."""
    df = pd.read_csv(path)
    return _normalize_dataframe(df)


def load_sample(sample_key: str) -> pd.DataFrame:
    """Carga uno de los CSVs de ejemplo en data/."""
    if sample_key not in SAMPLE_DATASETS:
        raise ValueError(f"Sample desconocido: {sample_key}")
    path = DATA_DIR / SAMPLE_DATASETS[sample_key]
    if not path.exists():
        raise FileNotFoundError(f"No existe {path.name} en data/")
    return load_csv_from_path(path)


def sample_exists(sample_key: str) -> bool:
    if sample_key not in SAMPLE_DATASETS:
        return False
    return (DATA_DIR / SAMPLE_DATASETS[sample_key]).exists()


def sample_size(sample_key: str) -> int:
    """Cuenta lineas del CSV de ejemplo sin cargarlo entero."""
    if not sample_exists(sample_key):
        return 0
    path = DATA_DIR / SAMPLE_DATASETS[sample_key]
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


# ---------- Estado de sesion ----------

def set_active_dataframe(df: pd.DataFrame, name: str) -> None:
    """Guarda el dataframe activo y limpia estado derivado."""
    st.session_state["df"] = df
    st.session_state["csv_name"] = name
    # Limpiar estado derivado de modelos anteriores
    for key in (
        "df_filtered", "X_scaled", "scaler",
        "current_model", "current_metrics", "current_algorithm",
        "current_params", "current_training_time",
        "current_pca", "df_labeled", "pdf_buffer",
        "clf_df", "clf_labels", "clf_probs", "clf_X_2d",
    ):
        st.session_state.pop(key, None)


def clear_active_dataframe() -> None:
    for key in ("df", "csv_name"):
        st.session_state.pop(key, None)


def get_active_dataframe() -> Optional[pd.DataFrame]:
    return st.session_state.get("df")


# ---------- Helpers de subconjunto ----------

def get_dimensions_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[DIMENSIONS].copy()


def get_demographics_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[["edad", "genero", "estado", "municipio"]].copy()
