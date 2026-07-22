# Análisis No Supervisado de Personalidad

Aplicación web para el análisis no supervisado de perfiles de personalidad basados en el modelo **Big Five (OCEAN)**. Cubre el flujo completo: recolección de datos vía encuesta propia, persistencia en base de datos documental, exploración estadística, entrenamiento y comparación de cuatro algoritmos de clustering, evaluación con métricas objetivas, y descarga de resultados.

Proyecto de la Unidad IV — Extracción de Conocimientos en Base de Datos.

---

## Índice

1. [Contexto](#contexto)
2. [Cómo levantar el proyecto desde cero](#cómo-levantar-el-proyecto-desde-cero)
3. [Arquitectura y flujo de datos](#arquitectura-y-flujo-de-datos)
4. [Guía técnica módulo por módulo](#guía-técnica-módulo-por-módulo)
5. [Guía de los 4 algoritmos](#guía-de-los-4-algoritmos)
6. [Cómo interpretar cada métrica](#cómo-interpretar-cada-métrica)
7. [Decisiones de diseño](#decisiones-de-diseño)

---

## Contexto

### El problema

El análisis no supervisado permite descubrir estructura en datos sin etiquetas previas. La pregunta que resuelve este proyecto es: **¿existen grupos naturales de personalidad entre las personas encuestadas?** No sabemos cuántos grupos hay ni cómo se ven — los algoritmos los encuentran.

### El dominio: Big Five (OCEAN)

El modelo de personalidad más aceptado en psicología científica. Describe la personalidad en cinco dimensiones continuas independientes:

| Dim | Nombre | Alta puntuación | Baja puntuación |
|---|---|---|---|
| **O** | Apertura | Creativo, curioso, abierto a nuevas experiencias | Convencional, práctico, resistente al cambio |
| **C** | Conciencia | Organizado, responsable, planificador | Espontáneo, flexible, descuidado |
| **E** | Extraversión | Sociable, energético, asertivo | Reservado, introspectivo, tranquilo |
| **A** | Amabilidad | Empático, cooperativo, confiado | Competitivo, escéptico, directo |
| **N** | Neuroticismo | Emocionalmente reactivo, ansioso | Estable, tranquilo bajo presión |

Cada persona tiene un valor 1–5 en cada dimensión. Cinco números por persona = un punto en un espacio de 5 dimensiones.

### La estrategia

1. **Recolectamos datos**: encuesta web con 20 preguntas Likert (4 por dimensión) más datos demográficos.
2. **Persistimos**: cada respuesta viaja de la encuesta a Google Sheets y de ahí a MongoDB Atlas.
3. **Exploramos**: la app muestra estadística descriptiva propia y correlaciones.
4. **Entrenamos**: cuatro algoritmos de clustering se ejecutan sobre los mismos datos.
5. **Comparamos**: métricas objetivas (silhouette, Davies-Bouldin, etc.) permiten decidir cuál algoritmo describe mejor los datos.
6. **Interpretamos**: cada cluster se caracteriza por el promedio OCEAN de sus miembros.

---

## Cómo levantar el proyecto desde cero

### Requisitos previos

- Python 3.12
- Cuenta gratuita de MongoDB Atlas (M0 cluster)
- Acceso al Google Sheet donde caen las respuestas (URL de export CSV)

### Pasos

**1. Clonar y entrar al proyecto**

```bash
git clone <url-del-repo>
cd unsupervised-app
```

**2. Crear entorno virtual con Python 3.12**

```bash
python -m venv .venv
```

En Windows:
```bash
.venv\Scripts\activate
```

En macOS/Linux:
```bash
source .venv/bin/activate
```

**3. Instalar dependencias**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**4. Configurar variables de entorno**

Crea `config/.env` con este contenido:

```env
MONGO_USER=tu_usuario_atlas
MONGO_PASS=tu_password_atlas
MONGO_HOST=cluster0.xxxxx.mongodb.net
MONGO_DB=unsupervised_app
SHEET_ID=id_del_google_sheet
```

El `SHEET_ID` es la parte de la URL entre `/d/` y `/edit`:
`https://docs.google.com/spreadsheets/d/{ESTE_ES_EL_ID}/edit`

**5. Verificar conexión a MongoDB**

```bash
python test_connection.py
```

Debe imprimir `OK - Conectado a MongoDB Atlas`.

**6. Primer import de datos**

```bash
python test_import.py
```

Descarga las respuestas del Google Sheet y las guarda en MongoDB.

**7. Correr la app**

```bash
streamlit run app.py
```

Abre `http://localhost:8501`.

---

## Arquitectura y flujo de datos

### Vista de alto nivel

```
┌─────────────────────────────────────────────────────┐
│  Encuesta (Google Apps Script)                      │
│  → HTML/JS con 20 preguntas + demográficos          │
│  → Calcula scoring en el navegador                  │
│  → Guarda cada respuesta en Google Sheets           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Google Sheets (fuente cruda)                       │
│  → Una fila por respuesta                           │
│  → Columnas: timestamp, q1..q20, O,C,E,A,N,         │
│              arquetipo, edad, género, estado,       │
│              municipio                              │
└──────────────────────┬──────────────────────────────┘
                       │  (sheets_importer.py)
                       ▼
┌─────────────────────────────────────────────────────┐
│  MongoDB Atlas (unsupervised_app)                   │
│  Colección responses:                               │
│    - un documento por respuesta                     │
│    - contiene O,C,E,A,N + raw_answers + demográf.   │
│  Colección models:                                  │
│    - metadatos de modelos entrenados                │
└──────────────────────┬──────────────────────────────┘
                       │  (loader.py)
                       ▼
┌─────────────────────────────────────────────────────┐
│  App Streamlit                                      │
│  → Exploración → Entrenamiento → Resultados         │
│  → Modelos guardados → Descargas                    │
└─────────────────────────────────────────────────────┘
```

### Flujo dentro de la app

```
[1] Usuario abre la app
      │
      ▼
[2] Landing → Sync desde Sheets si falta actualizar
      │
      ▼
[3] Exploración → Filtros por edad/género/estado
                → Estadística descriptiva
                → El DataFrame filtrado queda en session_state
      │
      ▼
[4] Entrenamiento → Elige algoritmo
                  → Ve gráfica de optimización (elbow, dendrograma, etc.)
                  → Configura hiperparámetros
                  → Entrena el modelo
                  → Modelo y métricas quedan en session_state
      │
      ▼
[5] Resultados → PCA 2D coloreado por cluster
              → Perfil promedio por cluster
              → Interpretación textual automática
      │
      ▼
[6] Modelos → Guarda el modelo actual en MongoDB + filesystem
            → Ve historial de experimentos
      │
      ▼
[7] Descargas → CSV filtrado
              → CSV con etiquetas de cluster
              → Reporte PDF completo
```

### Estructura de carpetas

```
unsupervised-app/
├── app.py                          # Landing page + sidebar global
├── .streamlit/
│   └── config.toml                 # Tema visual (paleta menta)
├── pages/                          # Páginas de Streamlit multipage
│   ├── 1_📊_Exploración.py
│   ├── 2_🧪_Entrenamiento.py
│   ├── 3_📈_Resultados.py
│   ├── 4_💾_Modelos.py
│   └── 5_⬇️_Descargas.py
├── src/                            # Lógica de negocio
│   ├── db/
│   │   ├── connection.py           # Cliente MongoDB
│   │   ├── response_repository.py  # CRUD de respuestas
│   │   └── model_repository.py     # CRUD de metadatos de modelos
│   ├── data/
│   │   ├── loader.py               # Mongo → DataFrame
│   │   └── preprocessing.py        # Limpieza + escalado
│   ├── stats/
│   │   └── descriptive.py          # Estadística con algoritmos propios
│   ├── clustering/
│   │   ├── base.py                 # Interfaz común
│   │   ├── kmeans_model.py
│   │   ├── hierarchical_model.py
│   │   ├── dbscan_model.py
│   │   └── gmm_model.py
│   ├── evaluation/
│   │   ├── metrics.py              # Silhouette, DB, CH
│   │   ├── optimization.py         # Elbow, k-distances, BIC
│   │   └── interpretation.py       # Perfil y texto por cluster
│   ├── visualization/
│   │   ├── charts.py               # Gráficas Plotly
│   │   └── pca.py                  # Reducción de dimensionalidad
│   ├── persistence/
│   │   └── model_io.py             # Save/load con joblib
│   ├── reporting/
│   │   └── pdf_report.py           # Reporte descargable
│   └── sheets_importer.py          # Google Sheets → MongoDB
├── models/                         # Modelos entrenados (.pkl)
├── config/
│   └── .env
├── requirements.txt
└── README.md
```

---

## Guía técnica módulo por módulo

### `src/db/connection.py`

**Qué es**: cliente único de MongoDB para toda la app.

**Por qué existe**: centralizar la conexión evita abrir y cerrar clientes en cada operación. MongoDB recomienda un solo cliente compartido por proceso.

**Cómo funciona**: lee las variables del `.env`, construye la URI codificando el usuario y password (los caracteres especiales requieren `quote_plus`), y crea el cliente perezosamente (solo cuando alguien lo pide por primera vez).

**Qué obtienes**: `get_db()` retorna la base de datos lista para operar. `ping()` te dice si hay conexión.

---

### `src/db/response_repository.py`

**Qué es**: capa de acceso a la colección `responses`.

**Por qué existe**: aísla las operaciones de Mongo del resto del código. Si mañana cambias de MongoDB a Postgres, solo tocas este archivo.

**Métodos clave**:

- `upsert_many(docs)` — inserta o actualiza documentos usando `submitted_at` como clave única. Esto significa que puedes sincronizar cien veces desde Google Sheets sin duplicar nada.
- `find_all()` — retorna todos los documentos ordenados del más reciente al más antiguo.
- `ensure_indexes()` — crea índices en `submitted_at` (único), `arquetipo` y `estado`. Los índices aceleran búsquedas y filtros.

**Por qué upsert y no insert**: si el usuario re-sincroniza, no queremos filas duplicadas. El `submitted_at` es único por respuesta (Google Forms garantiza timestamps distintos).

---

### `src/sheets_importer.py`

**Qué es**: puente entre Google Sheets y MongoDB.

**Cómo funciona**:

1. Construye la URL de export CSV pública de Google Sheets: `https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=0`.
2. `requests.get()` descarga el CSV.
3. Forzamos `response.encoding = "utf-8"` porque Google Sheets no siempre declara el encoding correcto y sin esto los acentos salen corruptos.
4. `pandas` parsea el CSV.
5. Cada fila se transforma en un documento MongoDB con la estructura definida.

**Detalle importante**: el timestamp viene en formato `DD/MM/YYYY HH:MM:SS`. Pandas asume MM/DD por defecto (formato USA), por eso pasamos `dayfirst=True`.

---

### `src/data/loader.py`

**Qué es**: convierte los documentos de MongoDB en un DataFrame de pandas listo para usar.

**Por qué existe**: pandas es el lenguaje común de todos los módulos que hacen cálculos. MongoDB retorna diccionarios; los módulos necesitan DataFrames.

**Constantes clave**:

- `DIMENSIONS = ["O", "C", "E", "A", "N"]` — las variables numéricas del análisis.
- `DIMENSION_LABELS` — mapa de códigos a nombres legibles ("O" → "Apertura").

Estos nombres se usan en toda la app. Si cambias uno, cámbialo aquí y todo lo demás se actualiza.

---

### `src/data/preprocessing.py`

**Qué es**: preparación de datos antes del clustering.

**Funciones**:

- `drop_null_dimensions(df)` — elimina filas con valores nulos en OCEAN. Los algoritmos de clustering fallan con nulos.
- `scale_dimensions(df)` — **este es el paso crítico**. Aplica `StandardScaler` a las 5 dimensiones.

**Por qué escalamos**:

Los algoritmos de clustering calculan **distancias** entre puntos. Si una variable va de 1 a 5 y otra de 0 a 1000, la segunda domina todo. Al escalar, todas las variables quedan con media 0 y desviación estándar 1, contribuyendo por igual al cálculo de distancia.

En nuestro caso las 5 variables ya están en la misma escala (1-5), pero escalar sigue siendo importante porque hace que los algoritmos sean más estables y comparables.

**Detalle**: guardamos el `scaler` porque al hacer predicciones sobre datos nuevos hay que aplicar exactamente la misma transformación. Si entrenaste con `StandardScaler` ajustado sobre 200 respuestas, para clasificar una respuesta nueva necesitas ese mismo scaler.

---

### `src/stats/descriptive.py`

**Qué es**: estadística descriptiva implementada con algoritmos propios.

**Por qué existe**: el instrumento pide explícitamente "algoritmos propios". No basta con `df.describe()`.

**Cálculos implementados manualmente**:

- **Media**: `sum(x) / n`. Suma de todos los valores dividida entre la cantidad.
- **Mediana**: valor central del conjunto ordenado. Si n es par, promedio de los dos centrales.
- **Desviación estándar muestral**: `sqrt(sum((x - mean)^2) / (n - 1))`. Mide qué tan dispersos están los valores respecto a la media. Usamos `n - 1` (desviación muestral, no poblacional) porque nuestros datos son una muestra.
- **Cuantiles (Q1, Q3)**: por interpolación lineal. El Q1 es el valor que deja el 25% de los datos por debajo, Q3 el 75%.
- **Correlación de Pearson**: `cov(X, Y) / (std_X * std_Y)`. Valor entre -1 y 1. Mide la relación lineal entre dos variables.

**Función especial: `internal_consistency(df)`**

Calcula el promedio de correlaciones entre los 4 ítems que miden la misma dimensión. Por ejemplo, para Apertura correlaciona q1, q2, q3 y (6-q4). Si estas correlaciones son altas (>0.4), significa que los ítems realmente miden lo mismo — es una validación del instrumento.

Esto es **valor agregado** para la exposición: puedes decir "validé que mis ítems de una misma dimensión correlacionan entre sí antes de usarlos".

---

### `src/clustering/base.py`

**Qué es**: interfaz abstracta que todos los algoritmos deben implementar.

**Por qué existe**: permite tratar los 4 algoritmos como intercambiables. La página de entrenamiento no necesita saber si es K-Means o GMM; solo llama `model.fit(X)` y `model.labels`.

**Métodos que cada modelo debe implementar**:

- `fit(X)` — entrena.
- `get_params()` — retorna los hiperparámetros usados.
- `get_model()` — retorna el modelo sklearn interno para poder serializarlo.

**Métodos comunes**:

- `labels` — array con la etiqueta de cluster de cada punto.
- `get_cluster_count()` — cuántos clusters únicos hay.
- `get_outlier_count()` — cuántos puntos son ruido (label = -1, solo aplica DBSCAN).

Este patrón se llama **Strategy Pattern**: familias de algoritmos intercambiables detrás de una misma interfaz.

---

### `src/evaluation/metrics.py`

**Qué es**: métricas para evaluar la calidad del clustering.

**Métricas implementadas** (ver sección "Cómo interpretar cada métrica" para detalle):

- Silhouette
- Davies-Bouldin
- Calinski-Harabasz
- Conteo de clusters
- Conteo de outliers

**Detalle técnico**: en DBSCAN los puntos etiquetados como ruido (`-1`) se excluyen del cálculo de silhouette y Davies-Bouldin, porque no pertenecen a ningún cluster real.

---

### `src/evaluation/optimization.py`

**Qué es**: funciones que ayudan a elegir hiperparámetros.

**Funciones**:

- `elbow_kmeans(X)` — corre K-Means con k=2..10 y devuelve inercia (WCSS) y silhouette para cada k. Detecta automáticamente el "codo".
- `bic_gmm(X)` — corre GMM con distintos n_components y calcula BIC y AIC.
- `k_distances(X, k)` — para DBSCAN. Calcula la distancia al k-ésimo vecino más cercano de cada punto, ordena y busca el codo.
- `hierarchical_linkage_matrix(X, method)` — matriz de linkage para el dendrograma.

**Cómo detectamos el codo**: método de máxima distancia a la línea recta. Trazamos una línea imaginaria entre el primer y último punto de la curva. El punto que más se aleja de esa línea es el codo. Es un método simple y sin dependencias.

---

### `src/evaluation/interpretation.py`

**Qué es**: genera texto humano describiendo cada cluster.

**Cómo funciona**:

1. Calcula el promedio de cada dimensión OCEAN por cluster.
2. Compara contra el promedio global de todos los datos.
3. Identifica las 2 dimensiones donde el cluster más se aparta del promedio.
4. Genera texto: "N personas con Apertura alta y Neuroticismo bajo."

**Por qué esto es importante**: los clusters sin interpretación son números. Con interpretación se vuelven insights accionables.

---

### `src/visualization/pca.py`

**Qué es**: reducción de dimensionalidad para visualizar.

**Problema**: no puedes graficar 5 dimensiones. PCA (Principal Component Analysis) toma los 5 valores y los proyecta a 2 nuevos ejes que capturan **la mayor varianza posible**.

**Cómo interpretar el porcentaje de varianza**: si PC1 explica 40% y PC2 explica 25%, la proyección 2D preserva el 65% de la información original. Es una vista aproximada, pero suficiente para ver si los clusters se separan visualmente.

**Detalle**: PCA solo se usa para visualizar. **El clustering se hace sobre los datos originales de 5 dimensiones**, no sobre la proyección 2D. Reducir antes de clusterizar es una decisión válida pero cambia los resultados; en este proyecto elegimos clusterizar en 5D para no perder información.

---

### `src/persistence/model_io.py`

**Qué es**: guarda y carga modelos entrenados como archivos `.pkl`.

**Por qué joblib y no pickle**: joblib está optimizado para objetos con arrays de numpy (todos los modelos de sklearn lo son). Es más rápido y produce archivos más pequeños.

**Qué guarda**: dos archivos por modelo:
- El modelo en sí (KMeans, DBSCAN, etc. ya entrenado).
- El scaler (StandardScaler ya ajustado).

Ambos son necesarios para reproducir predicciones. El scaler es tan importante como el modelo.

---

### `src/db/model_repository.py`

**Qué es**: metadatos de modelos en MongoDB.

**Qué guardamos aquí (no los .pkl)**:

- Algoritmo usado.
- Hiperparámetros.
- Métricas obtenidas.
- Fecha de entrenamiento.
- Ruta a los archivos `.pkl` (guardados aparte).

**Por qué separamos**: MongoDB no está diseñado para almacenar archivos binarios grandes. Guardamos los `.pkl` en el filesystem y solo referenciamos su ruta desde Mongo. Así puedes consultar el historial ("¿qué modelos entrené?") sin cargar los binarios.

---

### `src/reporting/pdf_report.py`

**Qué es**: genera un PDF con los resultados del modelo actual.

**Contenido del reporte**:

1. Portada con fecha.
2. Resumen del modelo (algoritmo, registros, tiempo).
3. Hiperparámetros usados.
4. Métricas de evaluación.
5. Perfil promedio por cluster.
6. Interpretación textual de cada cluster.

Usa **ReportLab**, biblioteca de Python para PDF. Genera todo en memoria (`BytesIO`) y lo entrega directo al usuario sin escribir a disco.

---

## Guía de los 4 algoritmos

### K-Means

**Qué hace**: divide los datos en k grupos, minimizando la distancia de cada punto al centroide de su cluster.

**Cómo funciona (intuitivo)**:

1. Elige k puntos aleatorios como centroides iniciales.
2. Asigna cada punto al centroide más cercano.
3. Recalcula el centroide como el promedio de sus puntos asignados.
4. Repite 2-3 hasta que los centroides dejan de moverse.

**Hiperparámetros**:

- `n_clusters` (k) — cuántos grupos quieres. **Tú lo decides**, apoyado por el método del codo.
- `init` — cómo inicializa. `k-means++` es mejor que random porque distribuye los centroides iniciales.
- `n_init` — cuántas veces reintenta con distintas inicializaciones. Se queda con el mejor resultado. 10 es un buen default.

**Cuándo usar K-Means**:

- ✅ Grupos aproximadamente esféricos.
- ✅ Tamaños similares de cluster.
- ✅ Datasets grandes (es el más rápido).
- ❌ Formas irregulares o alargadas.
- ❌ Grupos muy desiguales en tamaño.
- ❌ Presencia de outliers (los "jala" a los centroides).

**Cómo elegir k**:

- Mira la gráfica del codo. El punto donde la inercia deja de bajar bruscamente es un buen k.
- La gráfica de silhouette por k complementa: k con mayor silhouette suele ser buen candidato.
- La app te sugiere un k automáticamente.

**Métrica clave**: Inercia (WCSS) — suma de distancias cuadradas al centroide. Menor = mejor, pero siempre baja al aumentar k, por eso necesitas el codo.

---

### Clusterización Jerárquica

**Qué hace**: construye un árbol (dendrograma) donde cada punto empieza como su propio cluster y se van fusionando los más cercanos.

**Cómo funciona (intuitivo)**:

1. Cada uno de los n puntos es su propio cluster.
2. Encuentra los dos clusters más cercanos y fusiónalos.
3. Repite hasta que todo sea un solo cluster.
4. El resultado es un árbol. Cortas a cierta altura para obtener k clusters.

**Hiperparámetros**:

- `n_clusters` — dónde cortar el árbol.
- `linkage` — cómo mide la distancia entre clusters:
  - **ward**: minimiza el aumento de varianza al fusionar. Produce clusters compactos y de tamaño similar. **Es el default y suele ser el mejor**.
  - **complete**: distancia máxima entre puntos. Clusters compactos.
  - **average**: distancia promedio. Balanceado.
  - **single**: distancia mínima. Tiende a producir cadenas largas.

**Cuándo usar Jerárquico**:

- ✅ Datasets pequeños o medianos (< 1000 puntos). Lento con muchos datos: O(n³).
- ✅ Cuando la jerarquía en sí es informativa (taxonomías).
- ✅ Análisis exploratorio: el dendrograma revela estructura.
- ❌ Datasets grandes.
- ❌ Cuando ya sabes exactamente cuántos clusters quieres (K-Means es más eficiente).

**Cómo elegir k desde el dendrograma**: buscas los "saltos verticales" grandes. Los saltos grandes indican que estás fusionando clusters muy distintos entre sí. Cortas justo debajo del salto más grande.

---

### DBSCAN

**Qué hace**: agrupa puntos que están densamente rodeados de otros puntos. Los que quedan aislados se marcan como ruido.

**Cómo funciona (intuitivo)**:

1. Para cada punto, cuenta cuántos vecinos tiene dentro de un radio `eps`.
2. Si tiene al menos `min_samples` vecinos, es un "punto núcleo" y forma o extiende un cluster.
3. Los vecinos de un núcleo también entran al cluster.
4. Los puntos que no son núcleo ni frontera se marcan como **ruido** (label = -1).

**Hiperparámetros**:

- `eps` — radio de vecindad. Más grande = clusters más grandes.
- `min_samples` — cuántos vecinos necesita un punto para ser núcleo. Más alto = más estricto, más ruido.

**Cuándo usar DBSCAN**:

- ✅ Formas de cluster irregulares (no esféricas).
- ✅ Presencia de outliers (los identifica automáticamente).
- ✅ Cuando no sabes cuántos clusters hay.
- ❌ Clusters con densidades muy distintas entre sí.
- ❌ Datos de alta dimensionalidad (la distancia euclidiana pierde efectividad).

**Cómo elegir eps**:

- Regla práctica: `min_samples ≈ 2 * dimensiones` = para 5D, min_samples ≈ 5-10.
- Para eps: mira la gráfica de k-distancias. Es la distancia al k-ésimo vecino de cada punto, ordenada. El "codo" de esa curva es un buen eps.

**Detalle importante**: si DBSCAN te marca casi todo como ruido (label = -1), tu `eps` es muy pequeño. Si te da un solo cluster gigante, es muy grande.

---

### GMM (Modelo de Mezcla Gaussiana)

**Qué hace**: asume que los datos vienen de k distribuciones gaussianas mezcladas. Aprende los parámetros de cada distribución y asigna a cada punto una **probabilidad** de pertenecer a cada cluster.

**Cómo funciona (intuitivo)**:

1. Inicializa k gaussianas (con media y covarianza).
2. Calcula la probabilidad de cada punto de pertenecer a cada gaussiana.
3. Recalcula los parámetros de cada gaussiana ponderando por esas probabilidades.
4. Repite hasta convergencia. Esto es el algoritmo EM (Expectation-Maximization).

**Hiperparámetros**:

- `n_components` — cuántas gaussianas (equivalente a k clusters).
- `covariance_type` — forma de las elipses de cada cluster:
  - **full**: cada cluster tiene su propia matriz de covarianza completa. Más flexible.
  - **tied**: todos comparten la misma matriz.
  - **diag**: matrices diagonales (elipses alineadas a los ejes).
  - **spherical**: un solo valor por cluster (esferas, similar a K-Means).

**Cuándo usar GMM**:

- ✅ Cuando esperas clusters solapados.
- ✅ Cuando necesitas **probabilidad de pertenencia**, no solo asignación dura.
- ✅ Cuando los clusters son elípticos (no esféricos como K-Means asume).
- ❌ Cuando los datos claramente no son gaussianos.
- ❌ Datasets muy pequeños (necesita datos para estimar covarianzas confiablemente).

**Cómo elegir n_components**:

- BIC (Bayesian Information Criterion): menor = mejor. Penaliza modelos complejos.
- AIC (Akaike Information Criterion): también menor = mejor, pero penaliza menos la complejidad.
- La app te sugiere el k con menor BIC automáticamente.

**Ventaja única**: te dice "esta persona tiene 70% de probabilidad de ser cluster 1 y 30% de ser cluster 2". K-Means y DBSCAN son asignaciones duras.

---

## Cómo interpretar cada métrica

### Coeficiente de silueta (silhouette score)

**Rango**: -1 a 1.

**Cómo se calcula**: para cada punto, mide qué tan cerca está de los puntos de su propio cluster comparado con el cluster más cercano ajeno. Promedia sobre todos los puntos.

**Interpretación**:

| Valor | Significado |
|---|---|
| **> 0.7** | Estructura fuerte, clusters bien separados |
| **0.5 - 0.7** | Estructura razonable |
| **0.25 - 0.5** | Estructura débil, considera revisar |
| **< 0.25** | Sin estructura clara, probablemente k está mal |
| **Negativo** | Los puntos están mal asignados |

**Cuándo la usas**: para comparar modelos entre sí. Un modelo con silhouette 0.6 es mejor que uno con 0.4.

**Limitación**: penaliza formas no esféricas. DBSCAN puede tener silhouette bajo aun encontrando estructura real.

---

### Índice Davies-Bouldin

**Rango**: 0 a infinito. **Menor = mejor**.

**Cómo se calcula**: promedio de la razón entre "compacidad dentro del cluster" y "separación entre clusters".

**Interpretación**:

- Menor a 1: clusters bien definidos.
- 1 a 2: aceptable.
- Mayor a 2: los clusters se traslapan.

**Cuándo la usas**: complementa silhouette. Si silhouette es alto y DB es bajo, tu modelo es sólido.

---

### Índice Calinski-Harabasz

**Rango**: 0 a infinito. **Mayor = mejor**.

**Cómo se calcula**: razón entre la dispersión entre clusters y dentro de clusters.

**Interpretación**: no tiene umbrales absolutos. Solo sirve para comparar modelos entre sí sobre los mismos datos. El que tenga CH más alto es mejor.

---

### Inercia (WCSS)

**Rango**: 0 a infinito. **Menor = mejor**, pero siempre baja al aumentar k.

**Qué es**: suma de distancias cuadradas de cada punto a su centroide.

**Cuándo la usas**: solo con K-Means, y solo para el método del codo. Nunca la uses aisladamente para comparar modelos: k=10 siempre tendrá menor inercia que k=3.

---

### BIC / AIC (solo GMM)

**Rango**: -infinito a infinito. **Menor = mejor**.

**Diferencia**:

- **BIC** penaliza más la complejidad. Prefiere modelos más simples.
- **AIC** penaliza menos. Puede favorecer modelos más complejos.

**Cuándo la usas**: para elegir n_components en GMM. Recomendación: usa BIC. Si BIC y AIC coinciden, mayor confianza.

---

## Decisiones de diseño

### ¿Por qué Streamlit y no Flask?

Streamlit permite construir apps con Python puro sin escribir HTML/JS/CSS. Para un proyecto de análisis de datos donde la lógica es más importante que el frontend, es mucho más rápido. Cada widget (`st.slider`, `st.selectbox`) es una línea de código.

### ¿Por qué MongoDB y no PostgreSQL?

Las respuestas de la encuesta son documentos naturales (una respuesta = un objeto con campos anidados). MongoDB los guarda como JSON directamente. Con SQL habría que decidir schema, hacer migraciones, y transformar entre formatos. Con MongoDB simplemente insertas y consultas.

Además el instrumento pide trabajar con base de datos, y MongoDB Atlas ofrece hosting gratuito que no requiere configuración de servidor.

### ¿Por qué Google Sheets como intermediario y no captura directa a MongoDB?

Tres razones:

1. **Redundancia**: si algo falla en MongoDB, los datos siguen en Sheets.
2. **Simplicidad del formulario**: Google Apps Script se despliega gratis, sin servidor, sin auth.
3. **Facilidad de auditoría**: puedes ver el Sheet directamente sin abrir MongoDB Compass.

El importador convierte Sheets → MongoDB. Es idempotente: puedes correrlo cuantas veces quieras sin duplicar.

### ¿Por qué PCA solo para visualizar?

Reducir a 2D antes de clusterizar pierde información. Los 4 algoritmos operan sobre las 5 dimensiones originales para tomar decisiones basadas en la máxima información posible. PCA solo se aplica al final para poder mostrar los clusters en un plano visible.

### ¿Por qué Session State para el DataFrame filtrado?

Streamlit re-ejecuta el script completo con cada interacción. Si no guardaras el DataFrame filtrado, cada cambio de página requeriría reaplicar todos los filtros. Session state persiste entre reruns dentro de la misma sesión del navegador.

### ¿Por qué guardar el scaler junto al modelo?

El StandardScaler transforma los datos. Si guardas solo el modelo y mañana quieres clasificar una respuesta nueva, no puedes: la respuesta nueva no está en la misma escala que los datos con los que se entrenó. Guardar el scaler garantiza reproducibilidad.

### ¿Por qué interfaz común para los algoritmos?

Los 4 algoritmos de sklearn tienen APIs ligeramente distintas (KMeans usa `.fit()` + `.labels_`, DBSCAN usa `.fit_predict()`, GMM tiene `.predict_proba()`). La interfaz común homogeneiza esto: la UI llama siempre a `.fit(X)` y `.labels`, sin importar el algoritmo. Esto se llama **Strategy Pattern**.

---

## Guía rápida de uso

1. **Cargar datos**: sidebar → "Sincronizar" descarga las nuevas respuestas del Sheet.
2. **Explorar**: revisa la estadística y correlaciones. Si algo se ve raro (ej. una variable con outliers extremos), toma nota.
3. **Filtrar**: acota tu análisis si quieres solo cierto grupo (ej. solo Guanajuato, solo mayores de 20).
4. **Entrenar los 4 algoritmos** con parámetros iniciales sugeridos:
   - K-Means: k=3 o k=4.
   - Jerárquico: k=3, ward.
   - DBSCAN: eps sugerido por la gráfica, min_samples=5.
   - GMM: k sugerido por BIC, covariance=full.
5. **Comparar métricas**: guarda cada modelo en el historial. El que tenga mejor silhouette y menor Davies-Bouldin es el ganador.
6. **Interpretar**: en Resultados, lee la interpretación textual de cada cluster.
7. **Exportar**: descarga el reporte PDF y el CSV con etiquetas.

---

## Autor

Juan Francisco Rodríguez Guerrero
Universidad Tecnológica del Norte de Guanajuato
GIDS6091-E · Extracción de Conocimientos en Base de Datos