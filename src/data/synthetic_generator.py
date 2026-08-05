"""Generador de datos sinteticos realistas para GMM.

Estrategia:
1. Se definen 5 arquetipos base con medias OCEAN objetivo y pesos.
2. Para cada registro se muestrea un arquetipo ponderado.
3. Se genera un score OCEAN alrededor de la media del arquetipo con ruido gaussiano.
4. Se reconstruyen 4 items Likert por dimension consistentes con el score OCEAN,
   respetando el reverso del 4to item.
5. Se agrega demografia plausible y se asigna arquetipo con la misma logica de
   scoring.determine_archetype (dimension dominante x mas baja).
"""
from __future__ import annotations
from datetime import datetime, timedelta
import random
from typing import List

import numpy as np
import pandas as pd

from src.data.scoring import (
    DIMENSIONS, ITEMS_BY_DIMENSION, REVERSE_ITEMS,
    SCALE_MIN, SCALE_MAX,
    compute_scores_from_answers, determine_archetype,
)


BASE_ARCHETYPES = {
    "explorer":    {"O": 4.5, "C": 2.3, "E": 3.4, "A": 3.5, "N": 2.8, "weight": 0.22},
    "architect":   {"O": 3.2, "C": 4.5, "E": 2.2, "A": 3.4, "N": 2.5, "weight": 0.18},
    "charismatic": {"O": 3.5, "C": 3.6, "E": 4.5, "A": 3.9, "N": 2.0, "weight": 0.20},
    "guardian":    {"O": 2.5, "C": 3.8, "E": 3.2, "A": 4.5, "N": 3.0, "weight": 0.20},
    "intense":     {"O": 3.5, "C": 2.8, "E": 3.0, "A": 3.2, "N": 4.4, "weight": 0.20},
}

DIMENSION_STD = 0.55  # dispersion intra-arquetipo
ITEM_NOISE = 0.35     # ruido intra-item respecto al score objetivo

GENERO_CHOICES = ["Femenino", "Masculino", "Otro", "Prefiero no decir"]
GENERO_WEIGHTS = [0.45, 0.45, 0.08, 0.02]

ESTADO_CHOICES = ["Guanajuato", "Queretaro", "Estado de Mexico", "Ciudad de Mexico", "Jalisco"]
ESTADO_WEIGHTS = [0.75, 0.10, 0.05, 0.05, 0.05]

MUNICIPIOS_GUANAJUATO = [
    "San Luis de la Paz", "Leon", "Celaya", "Irapuato", "Guanajuato",
    "Salamanca", "San Miguel de Allende", "Dolores Hidalgo", "Silao", "Valle de Santiago",
]


def _clip(x: float, lo: float = SCALE_MIN, hi: float = SCALE_MAX) -> float:
    return max(lo, min(hi, x))


def _pick_archetype(rng: random.Random) -> dict:
    keys = list(BASE_ARCHETYPES.keys())
    weights = [BASE_ARCHETYPES[k]["weight"] for k in keys]
    return BASE_ARCHETYPES[rng.choices(keys, weights=weights, k=1)[0]]


def _generate_dimension_score(rng: random.Random, mean: float) -> float:
    """Score OCEAN con ruido gaussiano recortado al rango 1-5."""
    return _clip(rng.gauss(mean, DIMENSION_STD))


def _generate_items_for_dimension(rng: random.Random, target_score: float,
                                   items: List[int]) -> dict:
    """Genera 4 respuestas Likert enteras cuyo promedio (con reverso) se acerca
    al target_score.

    Iteracion sencilla: se muestrean valores gaussianos alrededor del target,
    se ajusta el ultimo item para que el promedio coincida lo mas posible con
    el target, se redondea y se aplica el reverso al ultimo item para el registro.
    """
    # Muestreo inicial (valores continuos)
    raw_values = [rng.gauss(target_score, ITEM_NOISE) for _ in items]

    # Ajuste del ultimo item para acercar el promedio al target
    partial_sum = sum(raw_values[:-1])
    needed_last = 4 * target_score - partial_sum
    raw_values[-1] = needed_last

    # Redondear y clipear a 1-5
    likert = [int(round(_clip(v))) for v in raw_values]

    # El item mostrado (crudo, que se guardaria en la BD) debe respetar el reverso:
    # como el 4to item de cada bloque es reverso, el valor "crudo" es (6 - deseado).
    # Aqui likert[3] representa el valor "deseado" (misma direccion que el score);
    # aplicamos el reverso para dejarlo como respuesta cruda.
    last_item_num = items[-1]
    if last_item_num in REVERSE_ITEMS:
        likert[-1] = (SCALE_MIN + SCALE_MAX) - likert[-1]
        likert[-1] = int(round(_clip(likert[-1])))

    return {items[i]: likert[i] for i in range(len(items))}


def _generate_answers(rng: random.Random, target_scores: dict) -> List[int]:
    """Genera las 20 respuestas Likert en orden q1..q20."""
    answers_by_num = {}
    for dim, items in ITEMS_BY_DIMENSION.items():
        answers_by_num.update(
            _generate_items_for_dimension(rng, target_scores[dim], items)
        )
    return [answers_by_num[i] for i in range(1, 21)]


def _generate_demographics(rng: random.Random) -> dict:
    # Edad con sesgo joven (media ~24, rango 18-45)
    edad = int(round(_clip(rng.gauss(24, 5), 18, 45)))
    genero = rng.choices(GENERO_CHOICES, weights=GENERO_WEIGHTS, k=1)[0]
    estado = rng.choices(ESTADO_CHOICES, weights=ESTADO_WEIGHTS, k=1)[0]
    municipio = rng.choice(MUNICIPIOS_GUANAJUATO) if estado == "Guanajuato" else "N/A"
    return {"edad": edad, "genero": genero, "estado": estado, "municipio": municipio}


def _random_timestamp(rng: random.Random, base: datetime) -> datetime:
    """Timestamp uniforme en los ultimos 30 dias."""
    delta_days = rng.uniform(0, 30)
    return base - timedelta(days=delta_days, seconds=rng.randint(0, 86400))


def generate_synthetic_dataset(n: int, seed: int) -> pd.DataFrame:
    """Genera un dataset sintetico realista.

    n:    numero de registros
    seed: semilla para reproducibilidad
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    now = datetime.now()

    rows = []
    for _ in range(n):
        arch = _pick_archetype(rng)
        target_scores = {d: _generate_dimension_score(rng, arch[d]) for d in DIMENSIONS}

        answers = _generate_answers(rng, target_scores)
        computed_scores = compute_scores_from_answers(answers)
        archetype_info = determine_archetype(computed_scores)
        demog = _generate_demographics(rng)
        ts = _random_timestamp(rng, now)

        row = {"submitted_at": ts.strftime("%d/%m/%Y %H:%M:%S")}
        for i, val in enumerate(answers, start=1):
            row[f"q{i}"] = val
        row.update(computed_scores)
        row["arquetipo"] = archetype_info["nombre"]
        row.update(demog)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def print_validation_stats(df: pd.DataFrame) -> None:
    """Imprime estadisticas descriptivas para validar realismo."""
    print(f"\nRegistros generados: {len(df)}")
    print("\nDistribucion de arquetipos:")
    print(df["arquetipo"].value_counts().to_string())
    print("\nMedias globales OCEAN:")
    for d in DIMENSIONS:
        print(f"  {d}: {df[d].mean():.3f} (std {df[d].std():.3f})")
    print("\nDistribucion demografica:")
    print(df["genero"].value_counts(normalize=True).round(3).to_string())
    print("---")
    print(df["estado"].value_counts(normalize=True).round(3).to_string())
