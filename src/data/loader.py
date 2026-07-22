import pandas as pd
from src.db.response_repository import find_all


DIMENSIONS = ["O", "C", "E", "A", "N"]
DIMENSION_LABELS = {
    "O": "Apertura",
    "C": "Conciencia",
    "E": "Extraversión",
    "A": "Amabilidad",
    "N": "Neuroticismo",
}


def load_dataframe() -> pd.DataFrame:
    """Carga todos los registros de MongoDB como DataFrame limpio."""
    docs = find_all()
    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    df = df.drop(columns=["_id"], errors="ignore")

    if "submitted_at" in df.columns:
        df["submitted_at"] = pd.to_datetime(df["submitted_at"])

    return df


def get_dimensions_df(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna solo las columnas OCEAN."""
    return df[DIMENSIONS].copy()


def get_demographics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna solo columnas demográficas."""
    return df[["edad", "genero", "estado", "municipio"]].copy()