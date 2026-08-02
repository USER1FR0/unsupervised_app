"""Tests unitarios de la implementación propia en src/stats/descriptive.py.

Verifica que los cálculos manuales coinciden con numpy dentro de tolerancia
razonable. Justifica la etiqueta "implementación propia" del entregable.

Ejecutar con:
    pytest tests/ -v
o
    python -m unittest tests.test_descriptive
"""
import unittest
import numpy as np
import pandas as pd

from src.stats.descriptive import (
    mean, median, std_dev, variance, quantile, _pearson,
    describe_variable, cronbach_alpha, internal_consistency,
)


class TestBasicStats(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.small = np.array([1, 2, 3, 4, 5], dtype=float)
        self.uniform = np.array([3, 3, 3, 3, 3], dtype=float)
        self.random_big = np.random.rand(500) * 10

    def test_mean_vs_numpy(self):
        self.assertAlmostEqual(mean(self.small), float(np.mean(self.small)), places=6)
        self.assertAlmostEqual(mean(self.random_big), float(np.mean(self.random_big)), places=6)

    def test_median_vs_numpy(self):
        self.assertAlmostEqual(median(self.small), float(np.median(self.small)), places=6)
        even = np.array([1, 2, 3, 4], dtype=float)
        self.assertAlmostEqual(median(even), float(np.median(even)), places=6)

    def test_std_dev_muestral(self):
        # numpy usa ddof=0 por defecto (poblacional). ddof=1 es muestral.
        self.assertAlmostEqual(std_dev(self.small),
                                float(np.std(self.small, ddof=1)),
                                places=6)

    def test_std_dev_edge_cases(self):
        self.assertEqual(std_dev(np.array([5.0])), 0.0)
        self.assertEqual(std_dev(np.array([])), 0.0)
        self.assertEqual(std_dev(self.uniform), 0.0)

    def test_variance_es_std_al_cuadrado(self):
        self.assertAlmostEqual(variance(self.small), std_dev(self.small) ** 2, places=6)

    def test_quantile_vs_numpy_lineal(self):
        # numpy usa "linear" por default (mismo método)
        for q in [0.25, 0.5, 0.75, 0.1, 0.9]:
            self.assertAlmostEqual(
                quantile(self.random_big, q),
                float(np.quantile(self.random_big, q)),
                places=6,
                msg=f"Falló en q={q}",
            )

    def test_describe_variable_estructura(self):
        d = describe_variable(self.small)
        for key in ["n", "media", "mediana", "desv_std", "min", "Q1", "Q3", "max"]:
            self.assertIn(key, d)
        self.assertEqual(d["n"], 5)


class TestPearson(unittest.TestCase):
    def test_perfecta_positiva(self):
        x = np.arange(10, dtype=float)
        y = 2 * x + 3
        self.assertAlmostEqual(_pearson(x, y), 1.0, places=6)

    def test_perfecta_negativa(self):
        x = np.arange(10, dtype=float)
        y = -x
        self.assertAlmostEqual(_pearson(x, y), -1.0, places=6)

    def test_sin_relacion(self):
        np.random.seed(1)
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        self.assertLess(abs(_pearson(x, y)), 0.15)

    def test_matches_numpy_corrcoef(self):
        np.random.seed(3)
        x = np.random.randn(200)
        y = x + np.random.randn(200) * 0.5
        expected = float(np.corrcoef(x, y)[0, 1])
        self.assertAlmostEqual(_pearson(x, y), expected, places=5)

    def test_constante_retorna_cero(self):
        x = np.array([2.0, 2.0, 2.0, 2.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(_pearson(x, y), 0.0)


class TestPsicometria(unittest.TestCase):
    def _fake_raw_answers_df(self, n=100, seed=0):
        """Crea DataFrame con raw_answers de 20 ítems Likert 1–5.
        Los primeros 4 ítems (dimensión O) están correlacionados fuerte,
        para verificar que Cronbach da alto."""
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n):
            base_O = rng.integers(1, 6)
            answers = [
                int(np.clip(base_O + rng.integers(-1, 2), 1, 5)) for _ in range(3)
            ]
            # ítem 4 en reverse, así que su valor "verdadero" es (6 - base_O)
            answers.append(int(np.clip(6 - base_O + rng.integers(-1, 2), 1, 5)))
            # ítems 5-20 aleatorios independientes
            answers += rng.integers(1, 6, size=16).tolist()
            rows.append({"raw_answers": answers})
        return pd.DataFrame(rows)

    def test_cronbach_alpha_devuelve_5_filas(self):
        df = self._fake_raw_answers_df(n=200)
        result = cronbach_alpha(df)
        self.assertEqual(len(result), 5)
        self.assertIn("alpha_cronbach", result.columns)

    def test_cronbach_alpha_alto_para_items_correlacionados(self):
        df = self._fake_raw_answers_df(n=500, seed=7)
        result = cronbach_alpha(df)
        alpha_O = result[result["dimensión"] == "O"]["alpha_cronbach"].iloc[0]
        # Los ítems de O están construidos para correlacionar → α alto.
        self.assertGreater(alpha_O, 0.5,
            f"α de O deberia ser alto (>0.5), obtuve {alpha_O}")

    def test_cronbach_alpha_bajo_para_items_aleatorios(self):
        df = self._fake_raw_answers_df(n=500, seed=7)
        result = cronbach_alpha(df)
        alpha_C = result[result["dimensión"] == "C"]["alpha_cronbach"].iloc[0]
        # Los ítems de C son independientes → α cerca de 0 o incluso negativo.
        self.assertLess(alpha_C, 0.3,
            f"α de C deberia ser bajo (<0.3), obtuve {alpha_C}")

    def test_internal_consistency_estructura(self):
        df = self._fake_raw_answers_df(n=100)
        result = internal_consistency(df)
        self.assertEqual(len(result), 5)
        self.assertIn("corr_promedio", result.columns)


if __name__ == "__main__":
    unittest.main()
