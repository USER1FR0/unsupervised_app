import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.loader import DIMENSIONS


_INSTRUMENTO_CACHE = None


def _load_instrumento() -> dict:
    """Carga el mapeo de ítems del instrumento desde config/instrumento.json.
    Fallback a valores por defecto si el archivo no existe."""
    global _INSTRUMENTO_CACHE
    if _INSTRUMENTO_CACHE is not None:
        return _INSTRUMENTO_CACHE

    path = Path(__file__).resolve().parents[2] / "config" / "instrumento.json"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            _INSTRUMENTO_CACHE = json.load(f)
    else:
        _INSTRUMENTO_CACHE = {
            "items_por_dimension": {
                "O": [1, 2, 3, 4], "C": [5, 6, 7, 8], "E": [9, 10, 11, 12],
                "A": [13, 14, 15, 16], "N": [17, 18, 19, 20],
            },
            "reverse_items": [4, 8, 12, 16, 20],
            "escala_min": 1, "escala_max": 5,
        }
    return _INSTRUMENTO_CACHE


# ---------- Estadísticos básicos (implementación propia) ----------

def mean(values: np.ndarray) -> float:
    """Media aritmética: sumatoria dividida entre n."""
    n = len(values)
    if n == 0:
        return 0.0
    return float(np.sum(values) / n)


def median(values: np.ndarray) -> float:
    """Mediana: valor central del conjunto ordenado."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 0:
        return float((sorted_vals[mid - 1] + sorted_vals[mid]) / 2)
    return float(sorted_vals[mid])


def std_dev(values: np.ndarray) -> float:
    """Desviación estándar muestral: sqrt(sum((x - mean)^2) / (n - 1)).
    Retorna 0 si n <= 1 (evita división entre cero)."""
    n = len(values)
    if n <= 1:
        return 0.0
    m = mean(values)
    variance = np.sum((values - m) ** 2) / (n - 1)
    return float(np.sqrt(variance))


def variance(values: np.ndarray) -> float:
    """Varianza muestral (n-1)."""
    return std_dev(values) ** 2


def quantile(values: np.ndarray, q: float) -> float:
    """Cuantil por interpolación lineal (método estándar)."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
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


def _pearson(xi: np.ndarray, xj: np.ndarray) -> float:
    """Correlación de Pearson: cov(X,Y) / (std_X * std_Y). Implementación propia."""
    n = len(xi)
    if n <= 1:
        return 0.0
    mi, mj = mean(xi), mean(xj)
    si, sj = std_dev(xi), std_dev(xj)
    if si == 0 or sj == 0:
        return 0.0
    cov = np.sum((xi - mi) * (xj - mj)) / (n - 1)
    return float(cov / (si * sj))


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlación de Pearson entre las 5 dimensiones. Implementación propia."""
    dims = DIMENSIONS
    n_dims = len(dims)
    matrix = np.zeros((n_dims, n_dims))
    for i, di in enumerate(dims):
        for j, dj in enumerate(dims):
            matrix[i, j] = _pearson(df[di].values, df[dj].values)
    return pd.DataFrame(matrix, index=dims, columns=dims).round(3)


# ---------- Análisis psicométrico ----------

def _get_item_values(df: pd.DataFrame, item_num: int, reverse_items: set,
                      escala_max: int) -> np.ndarray:
    """Extrae la columna del ítem N del array raw_answers.
    Aplica reverse coding (invertir la escala) si el ítem está en reverse_items."""
    vals = np.array([row[item_num - 1] for row in df["raw_answers"]], dtype=float)
    if item_num in reverse_items:
        vals = (escala_max + 1) - vals
    return vals


def internal_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Correlación promedio entre los ítems de cada dimensión.
    Valores cercanos a 1 indican que los ítems miden lo mismo."""
    if "raw_answers" not in df.columns:
        return pd.DataFrame()

    inst = _load_instrumento()
    items_by_dim = inst["items_por_dimension"]
    reverse_items = set(inst["reverse_items"])
    escala_max = inst["escala_max"]

    rows = []
    for dim, items in items_by_dim.items():
        item_values = [_get_item_values(df, i, reverse_items, escala_max) for i in items]
        correlations = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                correlations.append(_pearson(item_values[i], item_values[j]))
        avg_corr = float(np.mean(correlations)) if correlations else 0.0
        rows.append({"dimensión": dim, "corr_promedio": round(avg_corr, 3)})

    return pd.DataFrame(rows)


def cronbach_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """Alfa de Cronbach por dimensión.

    Fórmula:
        α = (k / (k-1)) * (1 - Σvar(itemᵢ) / var(Σitemᵢ))
    donde k es el número de ítems por dimensión.

    Interpretación estándar en psicometría:
        α ≥ 0.9  excelente
        α ≥ 0.8  bueno
        α ≥ 0.7  aceptable
        α ≥ 0.6  cuestionable
        α < 0.6  pobre
    """
    if "raw_answers" not in df.columns:
        return pd.DataFrame()

    inst = _load_instrumento()
    items_by_dim = inst["items_por_dimension"]
    reverse_items = set(inst["reverse_items"])
    escala_max = inst["escala_max"]

    rows = []
    for dim, items in items_by_dim.items():
        k = len(items)
        if k < 2:
            rows.append({"dimensión": dim, "alpha_cronbach": None, "interpretación": "n<2"})
            continue

        item_matrix = np.column_stack([
            _get_item_values(df, i, reverse_items, escala_max) for i in items
        ])
        # Varianza por ítem
        item_variances = [variance(item_matrix[:, c]) for c in range(k)]
        # Varianza del total (suma por fila)
        totals = item_matrix.sum(axis=1)
        total_variance = variance(totals)

        if total_variance == 0:
            alpha = 0.0
        else:
            alpha = (k / (k - 1)) * (1 - sum(item_variances) / total_variance)

        rows.append({
            "dimensión": dim,
            "alpha_cronbach": round(float(alpha), 3),
            "interpretación": _interpret_alpha(alpha),
        })
    return pd.DataFrame(rows)


def _interpret_alpha(alpha: float) -> str:
    if alpha >= 0.9:
        return "Excelente"
    if alpha >= 0.8:
        return "Bueno"
    if alpha >= 0.7:
        return "Aceptable"
    if alpha >= 0.6:
        return "Cuestionable"
    return "Pobre"
