# Analisis No Supervisado de Personalidad

Aplicacion web para el analisis no supervisado de perfiles de personalidad basados
en el modelo **Big Five (OCEAN)**. Enfocada en **Modelo de Mezcla Gaussiana (GMM)**
como algoritmo principal, con K-Means, Clusterizacion Jerarquica y DBSCAN como
comparativa academica.

Proyecto de la Unidad IV — Extraccion de Conocimientos en Base de Datos.

---

## Indice

1. [Contexto](#contexto)
2. [Requisitos](#requisitos)
3. [Instalacion](#instalacion)
4. [Datasets](#datasets)
5. [Ejecutar la app](#ejecutar-la-app)
6. [Notebook de demo](#notebook-de-demo)
7. [Estructura del proyecto](#estructura-del-proyecto)
8. [Persistencia](#persistencia)
9. [Comandos utiles](#comandos-utiles)

---

## Contexto

El modelo **Big Five** describe la personalidad en cinco dimensiones continuas:

| Dim | Nombre | Alta puntuacion |
|-----|--------|-----------------|
| O   | Apertura | Creativo, curioso, abierto al cambio |
| C   | Conciencia | Organizado, responsable, planificador |
| E   | Extraversion | Sociable, energico, asertivo |
| A   | Amabilidad | Empatico, cooperativo, confiado |
| N   | Neuroticismo | Emocionalmente reactivo, ansioso |

Cada persona tiene 5 valores en el rango 1-5 (promedio de 4 items Likert por
dimension, con el 4to item invertido). GMM asigna probabilidades a cada persona
de pertenecer a cada cluster, lo que permite detectar perfiles fronterizos.

---

## Requisitos

- Python 3.12
- Cuenta MongoDB Atlas (opcional: solo se usa para persistir modelos)

---

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Copia tu URI de Mongo en `config/.env`:

```
MONGO_URI=mongodb+srv://usuario:pass@cluster.mongodb.net/
MONGO_DB=big_five
```

---

## Datasets

Los datos viven como CSVs en `data/` y se cargan en memoria. No se persisten
respuestas en la base de datos.

| Dataset      | Archivo               | Descripcion                            |
|--------------|-----------------------|----------------------------------------|
| Real         | `data/real.csv`       | Respuestas de la encuesta publica.     |
| Sintetico    | `data/synthetic.csv`  | Generado con arquetipos controlados.   |
| Demo         | `data/demo.csv`       | Generado con seed distinta para clasificacion en vivo. |

### Generar los sinteticos

```bash
python scripts/generate_synthetic.py --dataset synthetic --n 500 --seed 42
python scripts/generate_synthetic.py --dataset demo --n 100 --seed 99
```

Para eliminarlos:

```bash
python scripts/clear_synthetic.py --dataset synthetic
python scripts/clear_synthetic.py --dataset demo
```

El dataset `real` nunca se elimina por script.

---

## Ejecutar la app

```bash
streamlit run app.py
```

Navegacion:

- **Exploracion**: filtros, tabla, estadistica descriptiva, alfa de Cronbach.
- **Entrenamiento**: GMM como flujo principal. Otros algoritmos en expander.
- **Resultados**: metricas, PCA 2D, perfil por cluster, probabilidades (GMM).
- **Modelos**: guardar el modelo actual con nombre unico, ver historial.
- **Descargas**: CSV filtrados, CSV con clusters, reporte PDF.
- **Comparativa**: ranking de modelos por metrica.
- **Clasificacion**: aplica un modelo guardado a un dataset nuevo.

En el sidebar se selecciona la fuente de datos activa (Real, Sintetico, Demo, Todos).

---

## Notebook de demo

```bash
jupyter notebook notebooks/demo_gmm.ipynb
```

Muestra GMM funcionando en codigo puro: carga, escalado, seleccion de k por BIC,
entrenamiento, probabilidades, PCA 2D y perfil por cluster.

---

## Estructura del proyecto

```
unsupervised-app/
├── app.py                          # Landing y sidebar global
├── .streamlit/config.toml
├── pages/
│   ├── 1_Exploracion.py
│   ├── 2_Entrenamiento.py
│   ├── 3_Resultados.py
│   ├── 4_Modelos.py
│   ├── 5_Descargas.py
│   ├── 6_Comparativa.py
│   └── 7_Clasificacion.py
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   ├── scoring.py               # arquetipos y scoring
│   │   └── synthetic_generator.py
│   ├── db/
│   │   ├── connection.py
│   │   └── model_repository.py      # solo persistencia de modelos
│   ├── clustering/                  # 4 algoritmos con interfaz comun
│   ├── evaluation/                  # metricas, optimizacion, interpretacion
│   ├── visualization/               # charts + PCA
│   ├── persistence/model_io.py      # bundle modelo + scaler + PCA
│   ├── reporting/pdf_report.py
│   ├── stats/descriptive.py
│   └── ui/theme.py                  # CSS global y sidebar comun
├── scripts/
│   ├── generate_synthetic.py
│   └── clear_synthetic.py
├── data/
│   ├── real.csv
│   ├── synthetic.csv
│   └── demo.csv
├── models/                          # .pkl generados al guardar
├── notebooks/demo_gmm.ipynb
└── requirements.txt
```

---

## Persistencia

MongoDB Atlas solo conserva la coleccion `models` con metadata:

- `model_name` (unico)
- `algorithm`, `hyperparameters`, `metrics`
- `dataset_source`, `n_records`, `training_time_seconds`
- Rutas a los `.pkl`: modelo, scaler, PCA

Los archivos `.pkl` viven en `models/`.

---

## Comandos utiles

```bash
# Correr la app
streamlit run app.py

# Regenerar sinteticos con otra semilla
python scripts/generate_synthetic.py --dataset synthetic --n 500 --seed 42

# Notebook
jupyter notebook notebooks/demo_gmm.ipynb
```
