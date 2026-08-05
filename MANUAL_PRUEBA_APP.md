# Manual de la app + Glosario

Aplicación **Big Five Analyzer** — Análisis No Supervisado de Personalidad — Unidad IV.

Este manual explica cómo usar la app página por página, qué significa cada resultado, y cómo defender los números frente al jurado. Complementa a `ALGORITMOS_EQUIPO.md` (teoría del algoritmo).

---

## Índice

1. [Requisitos y arranque](#1-requisitos-y-arranque)
2. [El instrumento por dentro](#2-el-instrumento-por-dentro)
3. [Flujo general](#3-flujo-general)
4. [Página por página](#4-página-por-página)
   - [4.1 Inicio](#41-inicio)
   - [4.2 Exploración](#42-exploración)
   - [4.3 Entrenamiento](#43-entrenamiento)
   - [4.4 Resultados](#44-resultados)
   - [4.5 Modelos](#45-modelos)
   - [4.6 Descargas](#46-descargas)
   - [4.7 Clasificación](#47-clasificación)
5. [Glosario](#5-glosario)
6. [Problemas comunes](#6-problemas-comunes)

---

## 1. Requisitos y arranque

**Requisitos:**
- Python 3.12 con dependencias instaladas (`pip install -r requirements.txt`).
- Conexión a MongoDB Atlas (solo para guardar/leer modelos entrenados).
- Un CSV con respuestas Big Five, o alguno de los datasets de ejemplo en `data/`.

**Arranque:**

```bash
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac
streamlit run app.py
```

Abre `http://localhost:8501`. En el sidebar aparece la navegación y el botón **Reiniciar aplicación** por si necesitas partir de cero (limpia el estado en memoria, no la base de datos).

---

## 2. El instrumento por dentro

**Encuesta**: 20 preguntas Likert 1–5, agrupadas en 4 ítems por dimensión OCEAN, más datos demográficos.

| Dimensión | Ítems | Concepto |
|-----------|-------|----------|
| **O** — Apertura | 1, 2, 3, 4 | Curiosidad, creatividad, apertura al cambio |
| **C** — Conciencia | 5, 6, 7, 8 | Organización, responsabilidad, planificación |
| **E** — Extraversión | 9, 10, 11, 12 | Sociabilidad, energía, asertividad |
| **A** — Amabilidad | 13, 14, 15, 16 | Empatía, cooperación, confianza |
| **N** — Neuroticismo | 17, 18, 19, 20 | Reactividad emocional, ansiedad |

**Ítems reverso**: los ítems 4, 8, 12, 16 y 20 (el 4º de cada bloque). Están redactados al revés para detectar el sesgo de "sí a todo".

**Fórmula de scoring** por dimensión $d$ con ítems $q_{d1}, q_{d2}, q_{d3}, q_{d4}$ (donde el 4º es reverso):

$$\text{Score}_d = \frac{q_{d1} + q_{d2} + q_{d3} + (6 - q_{d4})}{4}$$

El resultado queda en el rango [1.0, 5.0].

**Asignación de arquetipo**: la dimensión más alta (dominante) y la más baja se combinan en la clave `DOMINANTE-BAJA` (ej. `O-N` → "El Visionario Sereno"). Si el perfil es plano (dominante ≈ baja), se usa el arquetipo espejo `X-X`. Total: **25 arquetipos**.

---

## 3. Flujo general

La app se recorre en orden. Cada paso desbloquea el siguiente:

```
Inicio (cargar CSV)
    ↓
Exploración (filtrar + estadística)
    ↓
Entrenamiento (optimizar k con BIC → configurar → entrenar GMM)
    ↓
Resultados (métricas, PCA 2D, perfiles, probabilidades)
    ↓
Modelos (guardar con nombre único)
    ↓
Descargas (CSVs + reporte PDF)
    ↓
Clasificación (aplicar modelo guardado a datos nuevos)
```

En cualquier página, si falta el paso anterior, verás un warning y un enlace directo a la página que necesitas visitar primero.

---

## 4. Página por página

### 4.1 Inicio

Landing con carga de datos.

**Zona de carga**:
- **Drag & drop**: sube tu propio CSV.
- **Datasets de ejemplo**: 3 botones para cargar `real.csv` (211 respuestas reales), `synthetic.csv` (500 sintéticos) o `demo.csv` (100 sintéticos con seed distinta).

**Requisitos del CSV**:
- Obligatorias: `O`, `C`, `E`, `A`, `N`, `edad`, `genero`, `estado`, `municipio`.
- Opcionales: `q1`..`q20` (necesarias para el análisis psicométrico), `arquetipo`, `submitted_at`.

Una vez cargado, ves métricas del dataset (registros, edad promedio, arquetipos únicos) y una vista previa. El botón **Limpiar dataset** lo remueve del estado.

### 4.2 Exploración

Filtros + estadística descriptiva.

**Filtros**: rango de edad, género y estado. La tabla y todas las estadísticas se recalculan.

**Tabs**:

- **Tabla**: registros filtrados con columnas configurables.
- **Estadística descriptiva**: por cada dimensión: media, mediana, desviación estándar, min, Q1, Q3, max. Cálculos con implementación propia (fórmulas en `src/stats/descriptive.py`).
- **Distribución**: histograma y boxplot de una dimensión seleccionable + distribución por género y estado.
- **Correlación**: heatmap de correlación de Pearson entre las 5 dimensiones. Valores cercanos a 0 = dimensiones independientes (bueno para OCEAN).

**Análisis psicométrico** (solo si el CSV trae `q1`..`q20`):

- **Consistencia interna**: correlación promedio entre los 4 ítems de una dimensión. Valores altos = los ítems miden lo mismo.
- **Alfa de Cronbach**: métrica psicométrica estándar. Fórmula:

  $$\alpha = \frac{k}{k-1}\left(1 - \frac{\sum_{i=1}^{k} \sigma_{i}^2}{\sigma_{\text{total}}^2}\right)$$

  Interpretación: α ≥ 0.9 excelente, ≥ 0.8 bueno, ≥ 0.7 aceptable, ≥ 0.6 cuestionable.

### 4.3 Entrenamiento

Tres pasos: optimización de hiperparámetros → configuración → entrenamiento.

**Paso 1 — Optimización**: la app entrena GMM con k = 2..10 y grafica BIC y AIC. El **k con menor BIC** se marca como sugerido. Ver `ALGORITMOS_EQUIPO.md` sección 4 para la teoría.

**Paso 2 — Configuración**:
- `n_components`: número de clusters (default = k sugerido por BIC).
- `covariance_type`: forma de las gaussianas (default `full`).
- `random_state`: semilla (default 42).

**Paso 3 — Entrenamiento**: al presionar el botón, se entrena y se muestran las métricas de vista rápida (silhouette, Davies-Bouldin, BIC final, clusters detectados).

**Validaciones**: la app bloquea el entrenamiento si hay menos de 10 registros filtrados o si `n_components` es mayor al número de datos.

### 4.4 Resultados

Análisis completo del modelo recién entrenado.

**Métricas de evaluación**: silhouette, Davies-Bouldin, Calinski-Harabasz, AIC. Ver sección 5 del `ALGORITMOS_EQUIPO.md` para las fórmulas.

**PCA 2D**: scatter de los 5D reducidos a 2 componentes principales. Cada punto es una persona, coloreada por su cluster asignado. El caption indica cuánta varianza se explica con esos 2 ejes.

**Perfil por cluster**: tabla con la media OCEAN de cada cluster + interpretación textual automática (ej. "42 personas con Extraversion alta y Neuroticismo baja").

**Gráfico de barras**: perfil OCEAN de cada cluster en forma visual, comparable entre clusters.

**Probabilidades por cluster (soft clustering)**:
- Tabla con cada persona, su cluster asignado, `prob_maxima` (confianza), y todas las probabilidades.
- Filtro para ver solo personas fronterizas (`prob_maxima < 0.7`).
- Top 3 casos con mayor incertidumbre (menor `prob_maxima`).

Ver `ALGORITMOS_EQUIPO.md` sección 7 para la interpretación.

### 4.5 Modelos

Guardar el modelo actual y consultar el historial.

**Guardar**:
- Nombre único (validaciones: max 80 caracteres, solo alfanuméricos + `._-`).
- Se guarda en MongoDB (metadata) + `.pkl` en `models/` (modelo, scaler, PCA).
- Sugerencia automática: `gmm_<archivo>_<timestamp>`.

**Historial**: expander por cada modelo con hiperparámetros, métricas y botones:
- **Clasificar con este** → abre la página de Clasificación.
- **Eliminar** → borra `.pkl` + metadata.

**Modelos huérfanos**: si un `.pkl` fue eliminado manualmente pero la metadata sigue en Mongo, el modelo aparece marcado `[ARCHIVOS FALTANTES]`. Usa **Purgar modelos huérfanos** para limpiar en bloque.

### 4.6 Descargas

Tres tipos de export:

1. **CSV de datos filtrados**: los registros después de aplicar filtros en Exploración.
2. **CSV con etiquetas de cluster**: los datos + columna `cluster` y `prob_maxima`.
3. **Reporte PDF**: documento con métricas, hiperparámetros, PCA 2D y perfil por cluster.

### 4.7 Clasificación

Aplica un modelo guardado a datos nuevos que el modelo nunca vio durante el entrenamiento. Ideal para la demo en vivo.

**Paso 1 — Modelo**: selectbox con todos los modelos usables (huérfanos filtrados).

**Paso 2 — Fuente de datos**: cuatro modos.

- **Cargar CSV nuevo**: sube un CSV con las mismas columnas OCEAN.
- **Dataset de ejemplo**: cualquiera de los 3 samples (real, sintético, demo).
- **Generar demo**: genera N registros sintéticos en el momento con el generador del proyecto (útil para la exposición). Configura cantidad (10–2000) y semilla.
- **Usar dataset actual**: el CSV cargado en Inicio.

**Paso 3 — Ejecutar**: aplica `scaler` y `predict` del modelo guardado.

**Paso 4 — Resultados**:
- Tabla completa con cluster asignado y probabilidades.
- PCA 2D con los puntos nuevos proyectados usando el PCA del entrenamiento.
- Personas fronterizas (`prob_maxima < 0.7`).
- Descarga CSV con predicciones.

---

## 5. Glosario

**Big Five (OCEAN)**: modelo de personalidad con 5 dimensiones continuas (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism).

**Ítem reverso**: pregunta redactada en dirección opuesta a la dimensión. Se invierte con `6 - respuesta`.

**Escalado (StandardScaler)**: transforma cada dimensión a media 0 y desviación 1. Necesario para GMM porque compara distancias.

**Clustering**: agrupar datos sin etiquetas previas.

**Soft clustering**: asignación probabilística. Cada punto tiene una probabilidad para cada cluster.

**Componente / cluster**: cada una de las k gaussianas del modelo GMM.

**Matriz de covarianza**: describe la forma y orientación de una gaussiana multivariada.

**BIC / AIC**: criterios de selección de modelo. Menor = mejor. BIC penaliza más los modelos complejos.

**Silhouette**: métrica de calidad de clustering. Rango [-1, 1], mayor es mejor.

**Davies-Bouldin**: métrica de calidad de clustering. Menor es mejor. Un valor < 1 es sólido.

**Calinski-Harabasz**: métrica de calidad de clustering. Mayor es mejor. Sin escala absoluta.

**PCA (Principal Component Analysis)**: técnica de reducción de dimensionalidad. Proyecta los datos en menos dimensiones preservando la máxima varianza.

**Varianza explicada**: porcentaje de la variación total que captura cada componente principal.

**Persona fronteriza**: registro cuya `prob_maxima` es baja (< 0.7). Comparte rasgos entre múltiples clusters.

**Arquetipo**: etiqueta descriptiva basada en la combinación dominante × más baja de las dimensiones OCEAN. 25 en total.

**Alfa de Cronbach**: métrica psicométrica que mide la consistencia interna de una escala. Rango [0, 1], ≥ 0.7 aceptable.

**Modelo huérfano**: entrada en MongoDB cuya referencia a `.pkl` no existe en disco. No es utilizable, hay que purgarlo.

---

## 6. Problemas comunes

**"No se pudo conectar con MongoDB"** — Revisa `config/.env` (variables `MONGO_URI`, `MONGO_DB`) y tu conexión a internet. El cluster gratuito de Atlas puede tardar en despertar.

**"Solo hay X registros filtrados. GMM necesita al menos 10"** — Ajusta los filtros en Exploración para incluir más datos, o carga un dataset con más registros.

**"n_components (k) debe ser menor al número de registros"** — Reduce el slider `n_components` o carga más datos.

**"El CSV no tiene las columnas obligatorias"** — Verifica que el archivo tenga: `O`, `C`, `E`, `A`, `N`, `edad`, `genero`, `estado`, `municipio`.

**"X filas con valores OCEAN fuera del rango 1-5"** — Los scores deben estar entre 1 y 5. Revisa el archivo fuente.

**"Ya existe un modelo con el nombre 'X'"** — Elige otro nombre. Los nombres deben ser únicos.

**"Los archivos del modelo no existen"** — El `.pkl` fue eliminado pero la metadata sigue. Ve a **Modelos** → **Purgar modelos huérfanos**.

**El silhouette me sale muy bajo (< 0.3)** — Es esperado con datasets pequeños y datos continuos como los rasgos de personalidad. Ver `ALGORITMOS_EQUIPO.md` sección 9 para cómo defenderlo.

**La navegación se ve rara / hay elementos duplicados** — Reinicia streamlit con `Ctrl+C` y `streamlit run app.py`. Cambios en CSS a veces requieren refresh.

**Quiero empezar desde cero** — En el sidebar, presiona **Reiniciar aplicación**. Limpia todo el estado en memoria (dataset, modelo entrenado, resultados). Los modelos guardados en Mongo/disco no se tocan.

---

**Referencias cruzadas**

- Teoría del algoritmo → `ALGORITMOS_EQUIPO.md`.
- Estructura del código → `README.md`.
- Notebook de demo con código puro → `notebooks/demo_gmm.ipynb`.
