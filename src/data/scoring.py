"""Logica de scoring y arquetipos del instrumento Big Five.

Replica el comportamiento de Apps Script (Arquetipos.gs) para que la app,
el generador sintetico y el notebook usen la misma verdad.

Instrumento:
- 20 items Likert 1-5.
- 4 items por dimension OCEAN.
- El 4to item de cada bloque es reverso: se invierte con (6 - respuesta).
"""
from __future__ import annotations
from typing import Iterable


DIMENSIONS = ["O", "C", "E", "A", "N"]

# 4 items por dimension; el 4to (indice 3) es reverso.
ITEMS_BY_DIMENSION = {
    "O": [1, 2, 3, 4],
    "C": [5, 6, 7, 8],
    "E": [9, 10, 11, 12],
    "A": [13, 14, 15, 16],
    "N": [17, 18, 19, 20],
}
REVERSE_ITEMS = {4, 8, 12, 16, 20}

SCALE_MIN = 1
SCALE_MAX = 5


# ---------- Arquetipos (25 combinaciones dominante x mas baja + 5 espejo) ----------

ARQUETIPOS = {
    "O-C": {"nombre": "El Explorador Distraído",
            "frase": "Empiezas cinco cosas con la certeza de que esta vez sí."},
    "O-E": {"nombre": "El Filósofo de Bolsillo",
            "frase": "Piensas mucho, hablas poco, y cuando hablas la gente se queda callada."},
    "O-A": {"nombre": "El Crítico Creativo",
            "frase": "Ves problemas antes de que existan y soluciones antes de que las pidan."},
    "O-N": {"nombre": "El Visionario Sereno",
            "frase": "Se te ocurren ideas raras y nada te tiembla al proponerlas."},
    "O-O": {"nombre": "El Curioso Profesional",
            "frase": "Sabes cosas raras y no sabes bien por qué las sabes."},

    "C-O": {"nombre": "El Guardián del Método",
            "frase": "Si funciona, no le muevas. Y funciona porque tú no le mueves."},
    "C-E": {"nombre": "El Arquitecto Silencioso",
            "frase": "Terminas lo que otros están todavía discutiendo cómo empezar."},
    "C-A": {"nombre": "El Estratega Sin Rodeos",
            "frase": "Tus reglas son claras y las aplicas primero contigo mismo."},
    "C-N": {"nombre": "El Ejecutor Imparable",
            "frase": "Todo bajo control, incluso cuando todo está fuera de control."},
    "C-C": {"nombre": "El Detallista Puro",
            "frase": "Ves el detalle chueco antes que el cuadro entero."},

    "E-O": {"nombre": "El Alma Práctica de la Fiesta",
            "frase": "No la piensas mucho, y por eso la gente la pasa bien contigo."},
    "E-C": {"nombre": "El Improvisador de Oficio",
            "frase": "Sin plan, sin problema. Con plan, aburrido."},
    "E-A": {"nombre": "El Líder Directo",
            "frase": "Dices lo que piensas y la gente se acostumbra o se va."},
    "E-N": {"nombre": "El Carismático Estable",
            "frase": "Buena vibra y buen sueño. La combinación es sospechosa."},
    "E-E": {"nombre": "El Conector Nato",
            "frase": "Conoces a alguien que conoce a alguien. Siempre."},

    "A-O": {"nombre": "El Guardián Leal",
            "frase": "Cambias muchas cosas en la vida, menos la gente que quieres."},
    "A-C": {"nombre": "El Buen Corazón Distraído",
            "frase": "Todos te quieren y todos te esperan."},
    "A-E": {"nombre": "El Empático Reservado",
            "frase": "Escuchas más de lo que hablas y te das cuenta de más de lo que dices."},
    "A-N": {"nombre": "El Ancla Emocional",
            "frase": "Todos te cuentan sus problemas y nadie te pregunta por los tuyos."},
    "A-A": {"nombre": "El Puente Humano",
            "frase": "Tu instinto es unir, aunque a veces termines en medio del fuego."},

    "N-O": {"nombre": "El Preocupado Metódico",
            "frase": "Prefieres saber qué esperar, aunque sea lo mismo de siempre."},
    "N-C": {"nombre": "El Sensible Espontáneo",
            "frase": "Sientes fuerte y vives igual. Nadie puede acusarte de tibio."},
    "N-E": {"nombre": "El Introspectivo Profundo",
            "frase": "Tu cabeza es un lugar interesante donde casi nadie está invitado."},
    "N-A": {"nombre": "El Escéptico Emocional",
            "frase": "Sientes intensamente y confías con cuentagotas."},
    "N-N": {"nombre": "El Alma Intensa",
            "frase": "Sientes todo con volumen alto. Los graves no se pueden bajar."},
}


def _reverse(value: int) -> int:
    return (SCALE_MAX + SCALE_MIN) - value


def compute_scores_from_answers(answers: Iterable[int]) -> dict:
    """Recibe 20 respuestas Likert 1-5 y retorna dict con O, C, E, A, N.

    Aplica reverso al 4to item de cada bloque.
    """
    answers = list(answers)
    if len(answers) != 20:
        raise ValueError(f"Se esperaban 20 respuestas, se recibieron {len(answers)}")

    scores = {}
    for dim, items in ITEMS_BY_DIMENSION.items():
        total = 0.0
        for item_num in items:
            val = answers[item_num - 1]
            if item_num in REVERSE_ITEMS:
                val = _reverse(val)
            total += val
        scores[dim] = round(total / len(items), 2)
    return scores


def determine_archetype(scores: dict) -> dict:
    """Retorna el arquetipo (dict con nombre, frase) segun scores OCEAN.

    Regla: dimension dominante (mayor score) x dimension mas baja (menor score).
    Si son la misma dimension (perfil plano en un extremo), se usa el arquetipo
    'espejo' X-X.
    """
    ordered = sorted(scores.items(), key=lambda kv: kv[1])
    lowest = ordered[0][0]
    highest = ordered[-1][0]

    # Empate perfecto: retornar espejo del dominante.
    key = f"{highest}-{lowest}" if highest != lowest else f"{highest}-{highest}"
    if key not in ARQUETIPOS:
        key = f"{highest}-{highest}"
    return ARQUETIPOS[key]


def archetype_name(scores: dict) -> str:
    """Solo el nombre del arquetipo."""
    return determine_archetype(scores)["nombre"]
