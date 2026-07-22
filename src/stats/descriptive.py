import numpy as np
import pandas as pd
from src.data.loader import DIMENSIONS


def mean(values: np.ndarray) -> float:
    """Media aritmética: sumatoria dividida entre n."""
    return float(np.sum(values) / len(values))


def median(values: np.ndarray) -> float:
    """Mediana: valor central del conjunto ordenado."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return float((sorted_vals[mid - 1] + sorted_vals[mid]) / 2)
    return float(sorted_vals[mid])


def std_dev(values: np.ndarray) -> float:
    """Desviación estándar muestral: sqrt(sum((x - mean)^2) / (n - 1))."""
    m = mean(values)
    variance = np.sum((values - m) ** 2) / (len(values) - 1)
    return float(np.sqrt(variance))


def quantile(values: np.ndarray, q: float) -> float:
    """Cuantil por interpolación lineal (método estándar)."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    pos = q * (n - 1)
    lower = int(np.floor(pos))
    upper = int(np.ceil(pos))
    if lower == upper:
        return float(sorted_vals[lower])
    weight = pos - lower
    return float(sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight)


def describe_variable(values: np.ndarray) -> dict:
    """Estadística descriptiva completa de una variable numérica."""
    return {
        "n": len(values),
        "media": round(mean(values), 3),
        "mediana": round(median(values), 3),
        "desv_std": round(std_dev(values), 3),
        "min": round(float(np.min(values)), 3),
        "Q1": round(quantile(values, 0.25), 3),
        "Q3": round(quantile(values, 0.75), 3),
        "max": round(float(np.max(values)), 3),
    }


def describe_all_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Estadística descriptiva para las 5 dimensiones OCEAN."""
    rows = []
    for dim in DIMENSIONS:
        stats = describe_variable(df[dim].values)
        stats["variable"] = dim
        rows.append(stats)
    result = pd.DataFrame(rows)
    cols = ["variable", "n", "media", "mediana", "desv_std", "min", "Q1", "Q3", "max"]
    return result[cols]


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlación de Pearson entre las 5 dimensiones.
    Implementación propia: cov(X,Y) / (std_X * std_Y)."""
    dims = DIMENSIONS
    n_dims = len(dims)
    matrix = np.zeros((n_dims, n_dims))

    for i, di in enumerate(dims):
        for j, dj in enumerate(dims):
            xi = df[di].values
            xj = df[dj].values
            mi, mj = mean(xi), mean(xj)
            cov = np.sum((xi - mi) * (xj - mj)) / (len(xi) - 1)
            si = std_dev(xi)
            sj = std_dev(xj)
            matrix[i, j] = cov / (si * sj) if si > 0 and sj > 0 else 0.0

    return pd.DataFrame(matrix, index=dims, columns=dims).round(3)


def internal_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Chequeo de consistencia: promedio de correlaciones entre ítems de la misma dimensión.
    Valores cercanos a 1 indican que los ítems miden lo mismo."""
    from src.data.loader import DIMENSIONS

    if "raw_answers" not in df.columns:
        return pd.DataFrame()

    items_by_dim = {
        "O": [1, 2, 3, 4],
        "C": [5, 6, 7, 8],
        "E": [9, 10, 11, 12],
        "A": [13, 14, 15, 16],
        "N": [17, 18, 19, 20],
    }
    reverse_items = {4, 8, 12, 16, 20}

    def get_item_values(item_num: int) -> np.ndarray:
        vals = np.array([row[item_num - 1] for row in df["raw_answers"]])
        if item_num in reverse_items:
            vals = 6 - vals
        return vals

    rows = []
    for dim, items in items_by_dim.items():
        item_values = [get_item_values(i) for i in items]
        correlations = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                vi, vj = item_values[i], item_values[j]
                mi, mj = mean(vi), mean(vj)
                si, sj = std_dev(vi), std_dev(vj)
                if si > 0 and sj > 0:
                    cov = np.sum((vi - mi) * (vj - mj)) / (len(vi) - 1)
                    correlations.append(cov / (si * sj))
        avg_corr = float(np.mean(correlations)) if correlations else 0.0
        rows.append({"dimensión": dim, "corr_promedio": round(avg_corr, 3)})

    return pd.DataFrame(rows)