import io
import pandas as pd
import streamlit as st
from src.db.response_repository import find_all


DIMENSIONS = ["O", "C", "E", "A", "N"]
DIMENSION_LABELS = {
    "O": "Apertura",
    "C": "Conciencia",
    "E": "Extraversión",
    "A": "Amabilidad",
    "N": "Neuroticismo",
}

REQUIRED_CSV_COLUMNS = DIMENSIONS + ["edad", "genero", "estado", "municipio"]


# ---------- Fuente: MongoDB ----------

def load_from_mongo() -> pd.DataFrame:
    """Carga todos los registros de MongoDB como DataFrame limpio."""
    docs = find_all()
    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    df = df.drop(columns=["_id"], errors="ignore")

    if "submitted_at" in df.columns:
        df["submitted_at"] = pd.to_datetime(df["submitted_at"])

    return df


# ---------- Fuente: CSV cargado por el usuario ----------

def load_csv_into_session(uploaded_file) -> int:
    """Valida y guarda el DataFrame del CSV en session_state.
    Retorna la cantidad de filas cargadas. Lanza ValueError si el CSV
    no tiene las columnas mínimas o si los tipos no son válidos."""
    raw = uploaded_file.read()
    df = pd.read_csv(io.BytesIO(raw))

    faltantes = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Faltan columnas obligatorias: {', '.join(faltantes)}"
        )

    # Convertir OCEAN a float y validar rango
    for dim in DIMENSIONS:
        df[dim] = pd.to_numeric(df[dim], errors="coerce")
    df = df.dropna(subset=DIMENSIONS).reset_index(drop=True)

    fuera_rango = ((df[DIMENSIONS] < 1) | (df[DIMENSIONS] > 5)).any(axis=1).sum()
    if fuera_rango > 0:
        raise ValueError(
            f"{fuera_rango} filas con valores OCEAN fuera del rango 1–5."
        )

    # Convertir edad a int y limpiar categóricas
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df = df.dropna(subset=["edad"]).reset_index(drop=True)
    df["edad"] = df["edad"].astype(int)

    for col in ["genero", "estado", "municipio"]:
        df[col] = df[col].astype(str)

    # Parsear timestamp si viene
    if "submitted_at" in df.columns:
        df["submitted_at"] = pd.to_datetime(df["submitted_at"], errors="coerce")

    st.session_state["df_csv_source"] = df
    return len(df)


def clear_csv_source() -> None:
    """Elimina el CSV cargado de session_state y limpia derivados."""
    for key in (
        "df_csv_source",
        "df_filtered",
        "X_scaled",
        "scaler",
        "current_model",
        "current_metrics",
        "current_algorithm",
        "current_params",
        "current_training_time",
        "df_labeled",
        "pdf_buffer",
    ):
        st.session_state.pop(key, None)


# ---------- Fuente activa (dispatcher) ----------

def load_dataframe() -> pd.DataFrame:
    """Retorna el DataFrame de la fuente activa (Mongo o CSV cargado).
    La fuente se controla desde el sidebar en app.py y se guarda en
    st.session_state['data_source']."""
    source = st.session_state.get("data_source", "MongoDB")
    if source == "CSV cargado":
        return st.session_state.get("df_csv_source", pd.DataFrame())
    return load_from_mongo()


def active_source_label() -> str:
    """Etiqueta legible de la fuente activa, para mostrar en las páginas."""
    source = st.session_state.get("data_source", "MongoDB")
    return "CSV cargado" if source == "CSV cargado" else "MongoDB"


# ---------- Helpers de subconjunto ----------

def get_dimensions_df(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna solo las columnas OCEAN."""
    return df[DIMENSIONS].copy()


def get_demographics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna solo columnas demográficas."""
    return df[["edad", "genero", "estado", "municipio"]].copy()
