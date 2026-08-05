# GMM en profundidad — Guía para el equipo

Documento técnico sobre el algoritmo central de la aplicación: **Gaussian Mixture Model (GMM)**. Cubre teoría, hiperparámetros, métricas, interpretación de resultados y cómo defenderlos en la exposición.

Complementa a `MANUAL_PRUEBA_APP.md` (uso de la app).

---

## Índice

1. [Por qué GMM](#1-por-qué-gmm)
2. [Cómo funciona](#2-cómo-funciona)
3. [Hiperparámetros](#3-hiperparámetros)
4. [Selección de k con BIC](#4-selección-de-k-con-bic)
5. [Métricas de evaluación](#5-métricas-de-evaluación)
6. [Reducción de dimensionalidad (PCA)](#6-reducción-de-dimensionalidad-pca)
7. [Soft clustering: probabilidades](#7-soft-clustering-probabilidades)
8. [Interpretación de resultados](#8-interpretación-de-resultados)
9. [Cómo defenderlo](#9-cómo-defenderlo)

---

## 1. Por qué GMM

El instrumento de la Unidad IV pide aplicar **el algoritmo de análisis no supervisado deseado**. Elegimos GMM por tres razones:

1. **Es probabilístico**: no dice "eres del cluster 2 y punto". Dice "tienes 68% de ser del cluster 2 y 30% del 1". La personalidad humana no tiene fronteras nítidas, así que esta suavidad refleja mejor la realidad.
2. **Modela clusters elípticos**: cada cluster tiene su propia forma (matriz de covarianza), a diferencia de K-Means que solo modela esferas.
3. **Tiene criterio objetivo de selección de k**: el BIC (Bayesian Information Criterion) sugiere el número óptimo de componentes sin ambigüedad.

Los demás algoritmos (K-Means, Jerárquico, DBSCAN) están implementados en el código pero fuera del flujo principal, para mantener el enfoque.

---

## 2. Cómo funciona

### Idea intuitiva

> "Asumo que los datos vienen de **k distribuciones gaussianas mezcladas**. Voy a estimar los parámetros de cada gaussiana (media, covarianza, peso) y calcular la probabilidad de cada punto de pertenecer a cada una."

Cada cluster es una gaussiana multivariada de 5 dimensiones (una por cada rasgo OCEAN). Un punto puede tener probabilidad no nula en varios clusters.

### Algoritmo EM (Expectation-Maximization)

Se itera hasta convergencia:

1. **Inicialización**: k gaussianas con parámetros aleatorios (media $\mu_k$, covarianza $\Sigma_k$, peso $\pi_k$).
2. **E-step (Expectation)**: para cada punto $x_i$ y cada componente $k$, calcula la probabilidad posterior (responsabilidad):

   $$\gamma_{ik} = \frac{\pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$

3. **M-step (Maximization)**: actualiza los parámetros ponderando por las responsabilidades:

   $$\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}, \quad \Sigma_k = \frac{\sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_i \gamma_{ik}}, \quad \pi_k = \frac{\sum_i \gamma_{ik}}{n}$$

4. **Repetir** hasta que la log-verosimilitud deje de mejorar.

### Diferencia con K-Means

| Aspecto | K-Means | GMM |
|---------|---------|-----|
| Asignación | Dura (un cluster por punto) | Blanda (probabilidades) |
| Forma del cluster | Esfera | Elipse (según covarianza) |
| Función objetivo | Distancia euclidiana | Verosimilitud gaussiana |
| Métrica de selección | Inercia / Silhouette | BIC / AIC |

---

## 3. Hiperparámetros

Los tres que expone la app:

### `n_components` (k)

Número de gaussianas (clusters). Se elige con BIC (ver sección 4). En la app, el slider va de 2 a 10 y el default es el sugerido por BIC.

### `covariance_type`

Estructura de la matriz de covarianza $\Sigma_k$ de cada gaussiana. Controla la **forma** que puede tomar cada cluster:

| Tipo | Forma | Parámetros/comp | Cuándo usar |
|------|-------|-----------------|-------------|
| `full` | Elipse orientada libremente | $d(d+1)/2$ | Default. Máxima flexibilidad. |
| `tied` | Todas comparten la misma elipse | $d(d+1)/2$ total | Clusters de forma similar. |
| `diag` | Elipse con ejes paralelos a los axes | $d$ | Datasets pequeños, más estable. |
| `spherical` | Esfera (un solo valor de radio) | $1$ | Similar a K-Means. |

Con $d=5$ dimensiones y ~200 registros, `full` funciona bien. Con menos de 100 registros conviene `diag` para evitar sobreajuste.

### `random_state`

Semilla del generador de números aleatorios. La inicialización de EM es aleatoria, así que el mismo dataset con distinta semilla puede converger a mínimos locales distintos. Fijar la semilla (default `42`) garantiza reproducibilidad.

---

## 4. Selección de k con BIC

### Bayesian Information Criterion (BIC)

$$\text{BIC} = -2 \ln(\hat{L}) + p \ln(n)$$

donde:
- $\hat{L}$ = verosimilitud del modelo ajustado
- $p$ = número de parámetros libres del modelo
- $n$ = número de observaciones

**Menor BIC = mejor modelo.** BIC penaliza modelos con más parámetros (más componentes), evitando el sobreajuste.

### AIC (Akaike Information Criterion)

$$\text{AIC} = -2 \ln(\hat{L}) + 2p$$

Similar pero penaliza menos que BIC → tiende a sugerir modelos más complejos. Se muestra como comparación pero **la decisión se toma con BIC**.

### Interpretación de la curva

- **Curva en U (baja y luego sube)**: el mínimo es el k óptimo. Es el caso ideal.
- **Curva monótonamente decreciente**: los datos toleran arbitrariamente muchos componentes. Elegir el k más bajo donde la mejora se vuelve marginal (regla del codo).
- **Curva plana**: no hay estructura clara. Pocos datos o distribución uniforme.

---

## 5. Métricas de evaluación

La app reporta cinco métricas:

### Silhouette Coefficient

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

- $a(i)$ = distancia media de $i$ a los puntos de su propio cluster.
- $b(i)$ = distancia media de $i$ al cluster vecino más cercano.
- Rango $[-1, 1]$. **Mayor = mejor**.

Interpretación estándar (Kaufman & Rousseeuw):

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

Ratio de dispersión inter-cluster ($B_k$) sobre intra-cluster ($W_k$), corregido por grados de libertad. **Mayor = mejor.** No tiene umbral fijo, sirve para comparar modelos entre sí.

### BIC y AIC

Ver sección 4. En la página de Resultados aparecen los valores del modelo final entrenado.

---

## 6. Reducción de dimensionalidad (PCA)

Los datos viven en 5D (una dimensión por rasgo OCEAN) y no se pueden visualizar directamente. Se aplica **PCA (Principal Component Analysis)** para proyectar en 2D preservando la máxima varianza posible.

### Cómo funciona

1. Se centra la matriz de datos (media 0 por columna).
2. Se calculan los eigenvectores y eigenvalores de la matriz de covarianza.
3. Los 2 eigenvectores con mayores eigenvalores forman los ejes PC1 y PC2.
4. Los datos se proyectan sobre esos ejes.

### Interpretación

- **PC1** captura la mayor variación posible.
- **PC2** captura la segunda mayor, ortogonal a PC1.
- La **varianza explicada** (que la app muestra en cada eje) indica cuánta información se conserva. Con OCEAN suele estar entre 50% y 70% del total.

**Nota importante**: la posición en el scatter PCA no equivale directamente a los scores OCEAN. Es una proyección, no una equivalencia.

---

## 7. Soft clustering: probabilidades

Cada persona recibe un vector de probabilidades $p = (p_1, p_2, \ldots, p_k)$ donde $\sum p_i = 1$.

### Prob. máxima

$$p_{\max} = \max_i p_i$$

Es la confianza de la asignación. En la app:
- $p_{\max} > 0.9$: asignación muy segura, casi hard clustering.
- $0.7 \le p_{\max} \le 0.9$: asignación confiable.
- $p_{\max} < 0.7$: **persona fronteriza**, comparte rasgos entre múltiples clusters.

### Por qué importa

Los perfiles fronterizos son los más interesantes académicamente: no encajan limpiamente en un solo arquetipo y revelan la naturaleza continua de la personalidad. La app resalta los tres casos con menor $p_{\max}$.

---

## 8. Interpretación de resultados

Cuando el modelo termina de entrenar, la página de Resultados muestra:

### Perfil por cluster

Tabla con la **media OCEAN** de cada cluster + una **interpretación textual automática** que compara con la media global. Ejemplo:

> **Cluster 0** · 42 personas con Extraversion alta y Neuroticismo baja.
> O=3.52 · C=3.71 · E=4.20 · A=3.85 · N=2.10

Esto se lee: hay 42 personas cuyo perfil se caracteriza por ser especialmente extrovertidas y emocionalmente estables comparadas con el promedio.

### PCA 2D

Scatter donde cada punto es una persona coloreada por su cluster. Si los grupos se ven **separados visualmente**, el modelo capturó estructura. Si se ven **superpuestos**, o los clusters no son gaussianos o hay muy pocos datos.

### Probabilidades

Tabla ordenable con las probabilidades por cluster. El filtro de fronterizas (`prob_maxima < 0.7`) revela quiénes están "entre dos aguas".

---

## 9. Cómo defenderlo

**Silhouette bajo (< 0.3) con dataset pequeño**: es esperado. Con n=200 y 5 dimensiones, es común obtener silhouettes en 0.2–0.35 sin que haya nada malo. La estructura de personalidad es continua, no discreta.

Frase para el jurado:

> "El silhouette obtenido refleja la naturaleza continua del constructo de personalidad, no una falla del modelo. La ventaja de GMM sobre algoritmos duros como K-Means es precisamente que cuantifica esa continuidad mediante las probabilidades de pertenencia, en lugar de forzar asignaciones nítidas donde no las hay."

**Por qué GMM y no K-Means**:

> "GMM se apega mejor a la teoría psicométrica: los rasgos de personalidad se distribuyen aproximadamente de forma normal en la población, y las fronteras entre 'tipos' de personalidad son borrosas. GMM modela esas dos características directamente; K-Means solo aproxima."

**Cómo elegimos k**:

> "El BIC penaliza modelos innecesariamente complejos. Escogimos el k del mínimo BIC porque es el punto donde ganar más componentes deja de compensar la pérdida en parsimonia."

---

**Referencias**

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, cap. 9 (Mixture Models and EM).
- McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models*.
- scikit-learn documentation: `sklearn.mixture.GaussianMixture`.
