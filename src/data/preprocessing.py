import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data.loader import DIMENSIONS


def drop_null_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con valores nulos en las variables OCEAN."""
    return df.dropna(subset=DIMENSIONS).reset_index(drop=True)


def scale_dimensions(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Aplica StandardScaler a las 5 dimensiones OCEAN.
    Retorna la matriz escalada y el scaler ajustado."""
    scaler = StandardScaler()
    X = df[DIMENSIONS].values
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def count_nulls_by_column(df: pd.DataFrame) -> dict:
    """Conteo de nulos por columna, ignorando las que no tienen ninguno."""
    nulls = df.isnull().sum()
    return {col: int(n) for col, n in nulls.items() if n > 0}