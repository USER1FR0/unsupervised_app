import numpy as np
import pandas as pd
from src.data.loader import DIMENSIONS, DIMENSION_LABELS


def compute_cluster_profile(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Perfil promedio de cada cluster: media OCEAN + tamaño."""
    df_ = df.copy().reset_index(drop=True)
    df_["cluster"] = labels

    profile = df_.groupby("cluster")[DIMENSIONS].mean().round(2).reset_index()
    sizes = df_.groupby("cluster").size().reset_index(name="n")
    profile = profile.merge(sizes, on="cluster")
    return profile


def interpret_cluster(row: pd.Series, global_means: dict) -> str:
    """Genera interpretación textual de un cluster comparando contra el promedio global."""
    dim_diffs = {}
    for dim in DIMENSIONS:
        dim_diffs[dim] = row[dim] - global_means[dim]

    # Ordenar por magnitud absoluta
    sorted_dims = sorted(dim_diffs.items(), key=lambda x: abs(x[1]), reverse=True)

    # Tomar las 2 dimensiones más distintivas
    top = sorted_dims[:2]

    fragments = []
    for dim, diff in top:
        label = DIMENSION_LABELS[dim]
        if abs(diff) < 0.15:
            continue
        level = "alta" if diff > 0 else "baja"
        fragments.append(f"{label} {level}")

    if not fragments:
        return "Perfil cercano al promedio general."

    n = int(row["n"])
    if row["cluster"] == -1:
        return f"Ruido · {n} puntos que no encajan en ningún cluster denso."

    return f"{n} personas con " + " y ".join(fragments) + "."


def build_interpretations(profile: pd.DataFrame, global_means: dict) -> pd.DataFrame:
    """Añade columna 'interpretación' al DataFrame de perfil."""
    result = profile.copy()
    result["interpretación"] = result.apply(
        lambda row: interpret_cluster(row, global_means), axis=1
    )
    return result


def rename_profile_columns(profile: pd.DataFrame) -> pd.DataFrame:
    """Formato amigable para mostrar en tabla."""
    result = profile.copy()
    result["cluster"] = result["cluster"].apply(
        lambda x: f"Cluster {int(x)}" if x != -1 else "Ruido"
    )
    return result