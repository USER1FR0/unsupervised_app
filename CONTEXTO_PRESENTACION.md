# Contexto para la presentación — Análisis No Supervisado de Personalidad

Documento de apoyo para armar la exposición. Cubre qué es el proyecto, cómo funciona, qué hace el algoritmo y cómo interpretar sus resultados. Organizado por secciones temáticas para que puedas ordenar las diapositivas como prefieras.

**Equipo:** Balderas Melchor José Antonio · Oropeza Yepiz Cristian Efraín · Rodríguez Guerrero Juan Francisco
**Grupo:** GIDS6091 · Unidad IV — Extracción de Conocimientos en Bases de Datos
**Institución:** Universidad Tecnológica del Norte de Guanajuato

---

## 1. Objetivo del proyecto

Aplicación web que **descubre grupos naturales de personalidad** en un conjunto de respuestas usando aprendizaje **no supervisado**. No conocemos las etiquetas de antemano; el algoritmo encuentra la estructura por sí solo.

**Aportes concretos:**
- Recolección de datos con instrumento propio (encuesta Likert de 20 ítems).
- Análisis estadístico y psicométrico del instrumento.
- Modelado con **Gaussian Mixture Model (GMM)** como algoritmo central.
- Visualización de los clusters mediante reducción de dimensionalidad (PCA).
- Interpretación probabilística de las asignaciones (soft clustering).
- Persistencia de modelos entrenados y clasificación de datos nuevos.

---

## 2. Marco teórico: Big Five (OCEAN)

Modelo psicológico de personalidad más aceptado en investigación científica. Describe la personalidad en **cinco dimensiones continuas**:

| Dim | Rasgo | Alta puntuación |
|-----|-------|-----------------|
| **O** | Apertura | Curiosidad, creatividad, apertura al cambio |
| **C** | Conciencia | Organización, responsabilidad, planificación |
| **E** | Extraversión | Sociabilidad, energía, asertividad |
| **A** | Amabilidad | Empatía, cooperación, confianza |
| **N** | Neuroticismo | Reactividad emocional, ansiedad |

Cada persona queda representada como un **vector de 5 dimensiones** con valores en el rango [1, 5]. Cinco números por persona = un punto en un espacio de 5D.

---

## 3. Instrumento de recolección

### Encuesta

- **20 preguntas** tipo Likert (escala 1–5).
- **4 ítems por dimensión** OCEAN.
- **5 ítems reverso** (el 4º de cada bloque: q4, q8, q12, q16, q20). Están redactados en dirección opuesta al rasgo para detectar el sesgo de "sí a todo".
- **Datos demográficos** al final: edad, género, estado, municipio.

### Fórmula de scoring

Para cada dimensión $d$ con ítems $q_{d1}, q_{d2}, q_{d3}, q_{d4}$ (el 4º es reverso):

$$\text{Score}_d = \frac{q_{d1} + q_{d2} + q_{d3} + (6 - q_{d4})}{4}$$

El `(6 - q)` invierte la escala: contestar 5 en un ítem reverso cuenta como 1.

### Asignación de arquetipo (25 en total)

- Se identifica la dimensión **dominante** (score más alto) y la **más baja** (score más bajo).
- Se combinan como clave `DOMINANTE-BAJA` y se busca en la tabla de 25 arquetipos.
- Si el perfil es plano (dominante ≈ baja), se usa el arquetipo espejo `X-X`.

**Ejemplo:** dominante O, baja N → `"O-N"` → *"El Visionario Sereno"*.

---

## 4. Arquitectura y stack

**Frontend:** Streamlit (Python 3.12) con multipage — 7 páginas.
**Visualización:** Plotly (interactivo) + Matplotlib (para el PDF).
**ML:** scikit-learn (GaussianMixture, StandardScaler, PCA).
**Datos:** pandas + numpy.
**Persistencia:** MongoDB Atlas (metadata de modelos) + joblib (`.pkl` de modelos, scalers y PCA).
**Reportes:** ReportLab (PDF descargable).
**Recolección de datos:** encuesta HTML servida vía Google Apps Script, exportada como CSV.

### Módulos internos relevantes

- `src/data/loader.py` — validación y carga de CSVs.
- `src/data/scoring.py` — fórmula de scoring y matriz de arquetipos.
- `src/data/synthetic_generator.py` — generador de datos sintéticos realistas.
- `src/clustering/gmm_model.py` — wrapper del GMM.
- `src/evaluation/metrics.py` — silhouette, Davies-Bouldin, Calinski-Harabasz.
- `src/evaluation/optimization.py` — cálculo de BIC/AIC para selección de k.
- `src/persistence/model_io.py` — serialización del bundle modelo + scaler + PCA.
- `src/visualization/pca.py` — proyección 2D.

---

## 5. Flujo del análisis

```
1. Carga de datos (CSV o dataset de ejemplo)
        ↓
2. Exploración (filtros, estadística descriptiva, alfa de Cronbach)
        ↓
3. Entrenamiento (BIC → configuración → GMM.fit())
        ↓
4. Resultados (métricas, PCA 2D, perfiles, probabilidades)
        ↓
5. Modelos (guardar con nombre único, historial)
        ↓
6. Descargas (CSV de datos, CSV con clusters, PDF de reporte)
        ↓
7. Clasificación (aplicar modelo guardado a datos nuevos)
```

Cada paso desbloquea el siguiente. La app valida requisitos y guía con enlaces si falta algo.

---

## 6. Preprocesamiento

### Escalado (StandardScaler)

Los 5 valores OCEAN se estandarizan a media 0 y desviación 1 antes de entrar al modelo:

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

**Por qué:** GMM compara distancias entre puntos. Sin escalado, dimensiones con mayor rango dominarían aunque los valores estén todos en 1–5. Después de escalar, todas las dimensiones pesan lo mismo.

El scaler se persiste junto al modelo para poder transformar datos nuevos de la misma forma.

---

## 7. El algoritmo: Gaussian Mixture Model (GMM)

### Idea central

> "Asumo que los datos vienen de **k distribuciones gaussianas mezcladas**. Estimo los parámetros de cada gaussiana y calculo la probabilidad de cada punto de pertenecer a cada una."

Cada cluster es una **gaussiana multivariada** en 5D con su propia media $\mu_k$, matriz de covarianza $\Sigma_k$ y peso $\pi_k$.

### Algoritmo Expectation-Maximization (EM)

Se itera hasta convergencia:

1. **Inicialización:** k gaussianas con parámetros aleatorios.
2. **E-step (Expectation):** para cada punto $x_i$, calcular la probabilidad posterior (responsabilidad) de que pertenezca al componente $k$:

   $$\gamma_{ik} = \frac{\pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$

3. **M-step (Maximization):** actualizar los parámetros ponderando por las responsabilidades:

   $$\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}$$

   $$\Sigma_k = \frac{\sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_i \gamma_{ik}}$$

   $$\pi_k = \frac{1}{n}\sum_i \gamma_{ik}$$

4. **Repetir** E y M hasta que la log-verosimilitud deje de mejorar.

### Diferencia clave con K-Means

| Aspecto | K-Means | GMM |
|---------|---------|-----|
| Asignación | Dura (un cluster por punto) | Blanda (probabilidades) |
| Forma del cluster | Esfera | Elipse (según covarianza) |
| Función objetivo | Distancia euclidiana | Verosimilitud gaussiana |
| Modelo probabilístico | No | Sí |

### Por qué elegimos GMM

1. **Es probabilístico:** entrega la probabilidad de cada persona de pertenecer a cada cluster, en vez de forzar una asignación dura. Refleja mejor la naturaleza continua de la personalidad.
2. **Modela elipses:** cada cluster puede tener su propia forma y orientación.
3. **Tiene criterio objetivo de selección de k:** el BIC.

### Hiperparámetros

| Nombre | Función | Valor típico |
|--------|---------|--------------|
| `n_components` | Número de gaussianas (clusters) | 3–5, elegido por BIC |
| `covariance_type` | Forma de la matriz de covarianza | `full` (default) |
| `random_state` | Semilla para reproducibilidad | 42 |

**Opciones de `covariance_type`:**
- `full`: elipse libre en cualquier dirección (máxima flexibilidad).
- `tied`: todas las gaussianas comparten la misma elipse.
- `diag`: elipse alineada con los ejes (más estable con pocos datos).
- `spherical`: esfera (equivalente a K-Means).

---

## 8. Selección de k con BIC

### Bayesian Information Criterion

$$\text{BIC} = -2 \ln(\hat{L}) + p \ln(n)$$

- $\hat{L}$: verosimilitud del modelo ajustado.
- $p$: número de parámetros libres del modelo.
- $n$: número de observaciones.

**Menor BIC = mejor modelo.** Penaliza la complejidad: modelos con más componentes tienen más parámetros y son castigados. Evita el sobreajuste.

### AIC (comparación)

$$\text{AIC} = -2 \ln(\hat{L}) + 2p$$

Similar pero penaliza menos → suele sugerir modelos más complejos. Se muestra como comparación; la decisión se toma con BIC.

### Interpretación de la curva

- **U (baja y luego sube):** el mínimo es el k óptimo (caso ideal).
- **Monotónicamente decreciente:** los datos toleran arbitrariamente muchos componentes. Se elige el k donde la mejora se vuelve marginal.
- **Plana:** no hay estructura clara.

En la app se calcula BIC y AIC para k = 2..10 y se grafica automáticamente.

---

## 9. Reducción de dimensionalidad (PCA)

Los datos viven en 5D y no se pueden visualizar directamente. Se aplica **Principal Component Analysis** para proyectar en 2D preservando la máxima varianza posible.

**Proceso:**
1. Se centra la matriz (media 0 por columna).
2. Se calculan los eigenvectores y eigenvalores de la matriz de covarianza.
3. Los 2 eigenvectores con mayores eigenvalores forman los ejes PC1 y PC2.
4. Los datos se proyectan sobre esos ejes.

**Varianza explicada:** cada eje muestra qué porcentaje de la variación total captura. Con OCEAN suele estar entre 50% y 70% en los dos primeros componentes.

**Nota:** la posición de un punto en el scatter PCA **no equivale** a los valores OCEAN originales — es una proyección con pérdida.

---

## 10. Métricas de evaluación

La app reporta cuatro métricas objetivas de calidad de clustering:

### Silhouette Coefficient

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

- $a(i)$: distancia media de $i$ a los puntos de su propio cluster (cohesión).
- $b(i)$: distancia media de $i$ al cluster vecino más cercano (separación).
- Rango $[-1, 1]$. **Mayor = mejor.**

**Interpretación estándar (Kaufman & Rousseeuw):**

| Valor | Significado |
|-------|-------------|
| > 0.7 | Estructura fuerte |
| 0.5 – 0.7 | Estructura razonable |
| 0.25 – 0.5 | Estructura débil |
| < 0.25 | Sin estructura clara |

### Davies-Bouldin Index

$$\text{DB} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(c_i, c_j)}$$

Ratio promedio de dispersión intra-cluster ($\sigma$) sobre distancia entre centroides ($d$). **Menor = mejor.** Un valor < 1 es sólido.

### Calinski-Harabasz Index

$$\text{CH} = \frac{\text{tr}(B_k)}{\text{tr}(W_k)} \cdot \frac{n - k}{k - 1}$$

Ratio de dispersión inter-cluster sobre intra-cluster, corregido por grados de libertad. **Mayor = mejor.** No tiene umbral absoluto; sirve para comparar modelos entre sí.

### Alfa de Cronbach (calidad del instrumento, no del clustering)

$$\alpha = \frac{k}{k-1}\left(1 - \frac{\sum_{i=1}^{k} \sigma_{i}^2}{\sigma_{\text{total}}^2}\right)$$

Mide la consistencia interna de cada dimensión. α ≥ 0.7 se considera aceptable en psicometría. Se muestra en la página de Exploración.

---

## 11. Interpretación de resultados

### Perfil por cluster

Tabla con la **media OCEAN** de cada cluster + una **interpretación textual automática** que compara con la media global.

**Ejemplo de salida:**
> **Cluster 0** · 42 personas con Extraversion alta y Neuroticismo baja.
> O=3.52 · C=3.71 · E=4.20 · A=3.85 · N=2.10

Se lee: hay un grupo de 42 personas caracterizado por ser especialmente extrovertidas y emocionalmente estables comparadas con el promedio del dataset.

### Visualización PCA 2D

Scatter donde cada punto es una persona coloreada por su cluster asignado.

- **Grupos separados visualmente** → el modelo capturó estructura clara.
- **Grupos superpuestos** → estructura débil o los datos no son gaussianos.

### Soft clustering: probabilidades

Cada persona recibe un vector $p = (p_1, p_2, \ldots, p_k)$ donde $\sum p_i = 1$.

**Prob. máxima:** $p_{\max} = \max_i p_i$ es la confianza de la asignación.

| Rango | Interpretación |
|-------|----------------|
| $p_{\max} > 0.9$ | Asignación muy segura (hard clustering en la práctica) |
| $0.7 \le p_{\max} \le 0.9$ | Asignación confiable |
| $p_{\max} < 0.7$ | **Persona fronteriza** — comparte rasgos entre múltiples clusters |

**Los perfiles fronterizos son los más interesantes académicamente:** no encajan limpiamente en un solo arquetipo y revelan la naturaleza continua de la personalidad. La app los resalta.

### Ejemplo real de interpretación fronteriza

> Una persona con probabilidad `[Cluster 0: 0.42, Cluster 1: 0.38, Cluster 2: 0.20]` no es del "cluster 0"; es alguien con rasgos híbridos entre 0 y 1. En una encuesta grande hay muchas personas así — GMM los identifica; K-Means los borra.

---

## 12. Persistencia de modelos

Al guardar un modelo entrenado se persisten tres artefactos:

- `models/gmm_<timestamp>.pkl` — modelo entrenado.
- `models/scaler_gmm_<timestamp>.pkl` — escalador ajustado.
- `models/pca_gmm_<timestamp>.pkl` — PCA ajustado (para visualización de datos nuevos).

En MongoDB se guarda la **metadata** del experimento:
- `model_name` (único), `algorithm`, `hyperparameters`.
- `metrics` (silhouette, DB, CH, BIC, AIC).
- `dataset_source`, `n_records`, `training_time_seconds`.
- Rutas a los `.pkl`.

**Por qué persistir el scaler y el PCA:** para clasificar datos nuevos hay que aplicarles la misma transformación que se usó en entrenamiento; si se recalcula sobre datos nuevos, los ejes y las escalas cambian y las predicciones son inconsistentes.

---

## 13. Clasificación de datos nuevos

Aplicación de un modelo previamente entrenado a un dataset que el modelo **nunca vio** durante el entrenamiento.

### Proceso interno

1. Seleccionar un modelo guardado.
2. Cargar los datos nuevos (CSV, sample, generar demo, o dataset actual).
3. Aplicar el `scaler.transform()` para estandarizar con los mismos parámetros.
4. Llamar `model.predict(X)` y `model.predict_proba(X)`.
5. Aplicar el `pca.transform()` para visualizar en 2D.

### Salidas

- **Cluster asignado** por cada registro.
- **Probabilidades** por cluster (vector de k componentes).
- **PCA 2D** con los nuevos puntos proyectados usando el PCA del entrenamiento (para que sean comparables con la corrida original).
- **Personas fronterizas** filtradas automáticamente (`prob_maxima < 0.7`).

### Valor demostrativo

La página permite generar un dataset sintético en el momento con N configurable. Esto sirve para mostrar en vivo cómo el modelo generaliza a datos que no formaron parte del entrenamiento.

---

## 14. Datasets manejados

La app trabaja con CSVs que el usuario carga. Se incluyen tres datasets de ejemplo listos:

| Dataset | Origen | Uso típico |
|---------|--------|------------|
| **Real** | 211 respuestas reales recolectadas con la encuesta | Análisis principal |
| **Sintético** | 500 registros generados con arquetipos controlados (seed 42) | Entrenamiento con dataset grande |
| **Demo** | 100 registros generados con seed distinta (99) | Clasificación en vivo |

### Generador de datos sintéticos

Ubicado en `src/data/synthetic_generator.py`. Genera registros que **parecen reales** siguiendo estos pasos:

1. Define 5 arquetipos base con medias OCEAN objetivo y pesos:
   - explorer (O alta), architect (C alta), charismatic (E alta), guardian (A alta), intense (N alta).
2. Para cada registro, elige un arquetipo ponderado.
3. Genera scores OCEAN alrededor de la media del arquetipo con ruido gaussiano ($\sigma = 0.55$).
4. Reconstruye 20 respuestas Likert consistentes con esos scores (respetando el reverso del 4º ítem).
5. Genera demografía plausible (edad ~24 con sesgo joven, distribución realista de género y estado).
6. Aplica la misma función de scoring que la app real para asignar el arquetipo.

**Uso en la app:** los datasets pre-generados en `data/` + generación en vivo desde la página de Clasificación con N configurable (10–2000).

---

## 15. Módulos de la aplicación

7 páginas Streamlit con flujo secuencial:

1. **Inicio** — carga de CSV (upload, sample o dataset actual) y previsualización.
2. **Exploración** — filtros por edad/género/estado, estadística descriptiva por dimensión, correlación de Pearson entre dimensiones, alfa de Cronbach.
3. **Entrenamiento** — optimización BIC/AIC (k=2..10), configuración de hiperparámetros GMM, entrenamiento con métricas rápidas.
4. **Resultados** — métricas de evaluación, PCA 2D, perfil por cluster con interpretación textual, tabla de probabilidades con barras visuales, casos fronterizos.
5. **Modelos** — guardar con nombre único, historial con métricas, purga de modelos huérfanos.
6. **Descargas** — CSV filtrado, CSV con clusters, reporte PDF con métricas + PCA + perfiles.
7. **Clasificación** — aplicar modelo guardado a datos nuevos (upload, sample, generación demo, dataset actual), predicciones con probabilidades, PCA 2D, fronterizas destacadas.

Sidebar con navegación limpia y estado (dataset activo, modelos guardados). Botón "Reiniciar aplicación" para partir de cero sin tocar la base de datos.

---

## 16. Conclusiones sugeridas

**El proyecto en una frase:**
> Aplicación que aplica GMM a respuestas Big Five para descubrir grupos naturales de personalidad con asignación probabilística, visualización en 2D y capacidad de clasificar registros nuevos con el modelo entrenado.

**Aportes destacables:**

- Uso de **GMM** justificado teórica y empíricamente: la personalidad es continua, los grupos no tienen fronteras nítidas, el algoritmo debe reflejarlo.
- **Selección de k por BIC**: criterio objetivo, no discrecional.
- **Soft clustering**: cada persona no es solo "del grupo 2", sino "62% del 2 y 30% del 1"; las personas fronterizas son insight valioso, no ruido.
- **Persistencia completa** (modelo + scaler + PCA): los modelos guardados son reutilizables sobre datos nuevos con transformaciones consistentes.
- **Instrumento propio validado** con alfa de Cronbach y consistencia interna.

**Consideraciones sobre el dataset real (n=211):**

Los coeficientes de silhouette observados en datasets sociales de este tamaño y con constructos continuos (como la personalidad) suelen ubicarse en el rango 0.2–0.4. Esto **no indica fallo del modelo**; refleja la naturaleza continua del constructo. La ventaja de GMM es cuantificar esa continuidad mediante las probabilidades de pertenencia, en lugar de forzar asignaciones nítidas donde la realidad es difusa.

---

## 17. Bibliografía sugerida

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. — Capítulo 9 (Mixture Models and EM).
- McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models*. Wiley.
- Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data*. Wiley. — Silhouette.
- Scikit-learn documentation: `sklearn.mixture.GaussianMixture`, `sklearn.decomposition.PCA`.
- John, O. P., & Srivastava, S. (1999). *The Big Five Trait Taxonomy: History, Measurement, and Theoretical Perspectives*.

---

**Documento generado como apoyo para la exposición. No sustituye la comprensión conceptual del algoritmo ni el manejo práctico de la aplicación.**
