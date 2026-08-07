# Base de datos MongoDB — Guía para exposición

**Objetivo**: entender qué se guarda en la base de datos, por qué se eligió MongoDB, cómo está estructurada la colección y cada documento, y para qué sirve cada campo.

---

## 1. Por qué MongoDB

### 1.1 Naturaleza del dato

Un modelo de machine learning entrenado no es un dato relacional. Es un objeto complejo con:
- Hiperparámetros de distintos tipos.
- Métricas numéricas variables.
- Rutas a archivos binarios.
- Metadatos con estructura irregular.

MongoDB permite almacenar este tipo de información como **documento JSON** de forma directa, sin necesidad de definir un esquema rígido ni normalizar en varias tablas.

### 1.2 Decisiones justificadas

**Por qué documental y no relacional**:
- Cada modelo puede tener campos ligeramente distintos según el algoritmo o la configuración.
- Con SQL habría que hacer JOIN entre varias tablas para reconstruir un modelo completo.
- Con MongoDB un modelo es un solo documento, se lee y escribe en una operación.

**Por qué MongoDB Atlas y no local**:
- Hosting gratuito (tier M0 con 512 MB).
- Accesible desde cualquier lugar sin depender de tener la máquina prendida.
- No requiere configuración de servidor.
- Coherente con un despliegue eventual en la nube.

---

## 2. Estructura general de la base de datos

### 2.1 Base de datos

**Nombre**: `unsupervised_app`

Contiene una única colección relevante para la aplicación.

### 2.2 Colección utilizada

**Nombre**: `models`

**Propósito**: almacenar el historial de modelos entrenados y sus metadatos.

**Naturaleza**: sin esquema fijo (aprovechando la flexibilidad de MongoDB), pero con una estructura consistente por convención definida en el código.

### 2.3 Colecciones que NO existen

En versiones anteriores del proyecto se contemplaba una colección `responses` para almacenar las respuestas de la encuesta. En la versión actual esa colección **no existe**: las respuestas viven exclusivamente como archivos CSV en el sistema de archivos local.

---

## 3. Esquema de la colección `models`

Aunque MongoDB es sin esquema, la aplicación define una estructura fija que todos los documentos cumplen. Esta es la especificación formal.

### 3.1 Esquema formal en notación JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Model Document",
  "type": "object",
  "required": [
    "_id",
    "model_name",
    "algorithm",
    "algorithm_label",
    "trained_at",
    "hyperparameters",
    "metrics",
    "training_time_seconds",
    "n_records",
    "dataset_source",
    "model_file_path",
    "scaler_file_path",
    "pca_file_path"
  ],
  "properties": {
    "_id": {
      "type": "ObjectId",
      "description": "Identificador unico generado por MongoDB"
    },
    "model_name": {
      "type": "string",
      "description": "Nombre asignado por el usuario",
      "maxLength": 100
    },
    "algorithm": {
      "type": "string",
      "enum": ["gmm"],
      "description": "Clave tecnica del algoritmo"
    },
    "algorithm_label": {
      "type": "string",
      "description": "Etiqueta legible del algoritmo"
    },
    "trained_at": {
      "type": "date",
      "description": "Timestamp del entrenamiento en formato ISO 8601 UTC"
    },
    "hyperparameters": {
      "type": "object",
      "required": ["n_components", "covariance_type", "random_state"],
      "properties": {
        "n_components": {
          "type": "integer",
          "minimum": 2,
          "maximum": 10
        },
        "covariance_type": {
          "type": "string",
          "enum": ["full", "tied", "diag", "spherical"]
        },
        "random_state": {
          "type": "integer"
        }
      }
    },
    "metrics": {
      "type": "object",
      "required": [
        "n_clusters",
        "n_outliers",
        "n_samples",
        "n_samples_valid",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz"
      ],
      "properties": {
        "n_clusters": { "type": "integer", "minimum": 1 },
        "n_outliers": { "type": "integer", "minimum": 0 },
        "n_samples": { "type": "integer", "minimum": 1 },
        "n_samples_valid": { "type": "integer", "minimum": 1 },
        "silhouette": { "type": "number", "minimum": -1, "maximum": 1 },
        "davies_bouldin": { "type": "number", "minimum": 0 },
        "calinski_harabasz": { "type": "number", "minimum": 0 }
      }
    },
    "training_time_seconds": {
      "type": "number",
      "minimum": 0
    },
    "n_records": {
      "type": "integer",
      "minimum": 1
    },
    "dataset_source": {
      "type": "string",
      "description": "Nombre del CSV o dataset de origen"
    },
    "model_file_path": {
      "type": "string",
      "description": "Ruta relativa al archivo .pkl del modelo"
    },
    "scaler_file_path": {
      "type": "string",
      "description": "Ruta relativa al archivo .pkl del StandardScaler"
    },
    "pca_file_path": {
      "type": "string",
      "description": "Ruta relativa al archivo .pkl del modelo PCA"
    }
  }
}
```

### 3.2 Documento de ejemplo real

```json
{
  "_id": ObjectId("6a7510cef044021f78a09da6"),
  "model_name": "Modelo1",
  "algorithm": "gmm",
  "algorithm_label": "GMM",
  "trained_at": ISODate("2026-08-06T22:55:10.404Z"),
  "hyperparameters": {
    "n_components": 8,
    "covariance_type": "tied",
    "random_state": 40
  },
  "metrics": {
    "n_clusters": 8,
    "n_outliers": 0,
    "n_samples": 39,
    "n_samples_valid": 39,
    "silhouette": 0.1853,
    "davies_bouldin": 1.0447,
    "calinski_harabasz": 8.8679
  },
  "training_time_seconds": 0.014,
  "n_records": 39,
  "dataset_source": "RespuestasTestPersonalidad - Hoja 1 (1).csv",
  "model_file_path": "models/gmm_20260806_165510.pkl",
  "scaler_file_path": "models/scaler_gmm_20260806_165510.pkl",
  "pca_file_path": "models/pca_gmm_20260806_165510.pkl"
}
```

### 3.3 Descripción de cada campo

| Campo | Tipo | Descripción | Ejemplo |
|-------|------|-------------|---------|
| `_id` | ObjectId | Identificador único autogenerado | `6a7510cef044021f78a09da6` |
| `model_name` | String | Nombre asignado por el usuario | `"Modelo1"` |
| `algorithm` | String | Clave técnica del algoritmo | `"gmm"` |
| `algorithm_label` | String | Etiqueta legible | `"GMM"` |
| `trained_at` | Date | Fecha y hora del entrenamiento | `2026-08-06T22:55:10.404Z` |
| `hyperparameters.n_components` | Integer | Número de gaussianas | `8` |
| `hyperparameters.covariance_type` | String | Tipo de covarianza | `"tied"` |
| `hyperparameters.random_state` | Integer | Semilla aleatoria | `40` |
| `metrics.n_clusters` | Integer | Clusters detectados | `8` |
| `metrics.n_outliers` | Integer | Puntos marcados como outliers | `0` |
| `metrics.n_samples` | Integer | Total de registros usados | `39` |
| `metrics.n_samples_valid` | Integer | Registros válidos tras limpiar nulos | `39` |
| `metrics.silhouette` | Float | Coeficiente de silueta | `0.1853` |
| `metrics.davies_bouldin` | Float | Índice Davies-Bouldin | `1.0447` |
| `metrics.calinski_harabasz` | Float | Índice Calinski-Harabasz | `8.8679` |
| `training_time_seconds` | Float | Duración del entrenamiento | `0.014` |
| `n_records` | Integer | Registros del dataset | `39` |
| `dataset_source` | String | Origen de los datos | `"real.csv"` |
| `model_file_path` | String | Ruta al .pkl del modelo | `"models/gmm_20260806_165510.pkl"` |
| `scaler_file_path` | String | Ruta al .pkl del scaler | `"models/scaler_gmm_..."` |
| `pca_file_path` | String | Ruta al .pkl del PCA | `"models/pca_gmm_..."` |

---

## 4. Justificación del esquema

Cada decisión en el esquema tiene una razón.

### 4.1 Objetos anidados: `hyperparameters` y `metrics`

En lugar de tener campos planos como `hp_n_components`, `hp_covariance_type`, `metric_silhouette`, se agrupan en subdocumentos.

**Ventajas**:
- **Legibilidad**: al leer un documento se distingue claramente qué es configuración (hyperparameters) y qué es resultado (metrics).
- **Escalabilidad**: agregar una métrica nueva no rompe consultas existentes.
- **Consultas específicas**: es fácil filtrar por métricas sin afectar hiperparámetros.

Ejemplo de consulta aprovechando la estructura:

```javascript
db.models.find({ "metrics.silhouette": { $gte: 0.4 } })
db.models.find({ "hyperparameters.n_components": 5 })
```

### 4.2 Duplicación intencional: `algorithm` y `algorithm_label`

Se guardan dos campos porque cumplen roles distintos:
- `algorithm` es la clave que usa el código para instanciar el modelo.
- `algorithm_label` es lo que ve el usuario en la interfaz.

Esta separación permite cambiar textos de UI sin tocar el código interno.

### 4.3 Duplicación intencional: `n_samples` en `metrics` y `n_records` en la raíz

Ambos indican lo mismo pero conviven porque:
- `n_records` es un metadato del dataset origen.
- `n_samples` es un dato calculado durante el entrenamiento.

En algunos casos podrían diferir (por ejemplo, si el modelo hace bootstrap o valida cruzado). Se dejan ambos para robustez.

### 4.4 Rutas relativas y no absolutas

Los `file_path` son rutas relativas al directorio del proyecto, no rutas absolutas del sistema.

**Ventaja**: los documentos son portables. Si el proyecto se mueve a otra máquina, las rutas siguen siendo válidas mientras se mantenga la estructura de carpetas.

### 4.5 Timestamp en UTC

`trained_at` se guarda en UTC con zona horaria explícita.

**Ventaja**: comparaciones y ordenamientos son consistentes sin importar la zona horaria del cliente.

---

## 5. Índices

Aunque MongoDB no requiere índices para funcionar, se definen algunos para mejorar el rendimiento de consultas frecuentes.

### 5.1 Índices definidos

```javascript
db.models.createIndex({ "trained_at": -1 })
db.models.createIndex({ "algorithm": 1 })
db.models.createIndex({ "dataset_source": 1 })
```

### 5.2 Justificación de cada índice

**`trained_at` descendente**: la página de Modelos muestra los modelos ordenados del más reciente al más antiguo. Este índice hace ese ordenamiento eficiente.

**`algorithm`**: aunque actualmente solo hay GMM, en el futuro podrían coexistir varios algoritmos. Este índice permite filtrar por tipo rápidamente.

**`dataset_source`**: útil para responder "¿cuántos modelos se entrenaron con el dataset Real?" o "¿cuál fue el mejor modelo con el dataset Sintético?".

---

## 6. Constraints y validaciones

MongoDB permite validación de esquema a nivel de colección. En el proyecto, la validación se hace en código (Python) antes de insertar, por simplicidad.

### 6.1 Validaciones aplicadas por la aplicación

- **`model_name` no vacío**: se valida antes de guardar.
- **`n_components` en rango [2, 10]**: limitado por el slider en la UI.
- **`covariance_type` en lista predefinida**: limitado por el selectbox en la UI.
- **Métricas presentes**: solo se guarda un modelo si el entrenamiento fue exitoso y produjo todas las métricas.

### 6.2 Constraints implícitos

- `_id` es único por definición de MongoDB.
- `trained_at` se genera automáticamente al momento del guardado.
- Las rutas de los archivos se generan siguiendo el patrón `{algoritmo}_{timestamp}.pkl` para evitar colisiones.

---

## 7. Ciclo de vida de un documento

Un documento pasa por estos estados durante su vida útil.

### 7.1 Creación

Cuando el usuario entrena un modelo y decide guardarlo:

1. Se valida el nombre.
2. Se serializan los objetos (modelo, scaler, PCA) como .pkl en el filesystem.
3. Se construye el documento con todos los campos requeridos.
4. Se inserta en la colección `models`.

### 7.2 Lectura

Ocurre en varios momentos:
- Al abrir la página de Modelos (listar todos).
- Al seleccionar un modelo en la página de Clasificación (leer uno por ID).
- Al mostrar la vista rápida del modelo actual.

### 7.3 Actualización

**No se actualizan documentos existentes**. Cada entrenamiento produce un documento nuevo. Esto preserva el historial completo y evita perder experimentos anteriores.

### 7.4 Eliminación

Cuando el usuario elimina un modelo desde la aplicación:

1. Se lee el documento para obtener las rutas de los archivos.
2. Se eliminan los tres archivos .pkl del filesystem.
3. Se elimina el documento de MongoDB.

Este orden es importante: si se falla en la eliminación de un archivo, el documento se mantiene y el modelo sigue siendo utilizable.

---

## 8. Relación entre MongoDB y filesystem

MongoDB y el filesystem forman un sistema conjunto donde MongoDB actúa como **catálogo** y el filesystem como **almacén**.

### 8.1 Diagrama de referencia

```
   MongoDB (catalogo)                    Filesystem (almacen)
   ─────────────────                     ────────────────────
   Documento del Modelo1
   ├─ model_file_path ─────────►  models/gmm_20260806_165510.pkl
   ├─ scaler_file_path ────────►  models/scaler_gmm_20260806_165510.pkl
   └─ pca_file_path ───────────►  models/pca_gmm_20260806_165510.pkl
```

### 8.2 Cómo funciona en la práctica

Cuando la aplicación necesita usar un modelo (por ejemplo para clasificar):

1. Lee el documento desde MongoDB para obtener las rutas.
2. Con las rutas, carga los tres archivos .pkl desde el filesystem con joblib.
3. Aplica los objetos cargados para procesar datos nuevos.

### 8.3 Por qué esta separación

- **MongoDB es rápido consultando metadatos**: no queremos que sea lento por almacenar blobs.
- **El filesystem es eficiente con archivos binarios**: no tiene overhead de red ni BSON.
- **Cada capa hace lo que hace mejor**.

---

## 9. Ventajas de esta arquitectura

### 9.1 Ligereza

Cada documento pesa alrededor de 1 KB. Aunque haya cientos de modelos, la colección sigue siendo rápida.

### 9.2 Consultas complejas sin cargar binarios

Se pueden hacer análisis como "¿cuál es el mejor modelo por silhouette?" con una sola consulta MongoDB, sin abrir ningún .pkl.

### 9.3 Historial completo

Cada entrenamiento queda registrado. Se puede rastrear cómo evolucionaron los experimentos, qué hiperparámetros se probaron, qué dataset se usó en cada caso.

### 9.4 Auditoría y reproducibilidad

Con la información del documento se puede reproducir exactamente un experimento. Basta con conocer el dataset origen y aplicar los mismos hiperparámetros.

### 9.5 Portabilidad

Al usar rutas relativas y guardar todo el contexto en el documento, el sistema es portable entre máquinas.

---

## 10. Puntos clave para la exposición

### 10.1 Frases útiles

> "La base de datos MongoDB almacena el historial de modelos entrenados. Cada documento representa un experimento con sus hiperparámetros, métricas y referencias a los archivos binarios."

> "El esquema de la colección es consistente aunque MongoDB no lo obligue: nombre, algoritmo, timestamp, hiperparámetros, métricas y rutas. Todo lo necesario para auditar y reproducir un experimento."

> "MongoDB actúa como el catálogo y el filesystem como el almacén: la base guarda los metadatos, los archivos binarios viven en disco. Cada capa hace lo que hace mejor."

### 10.2 Preguntas típicas y respuestas

**"¿Cuál es la estructura de la colección?"**
La colección `models` contiene un documento por cada modelo entrenado. Cada documento incluye identificador, nombre, algoritmo, timestamp, un subdocumento de hiperparámetros, un subdocumento de métricas, tiempo de entrenamiento, dataset origen y rutas a los tres archivos binarios (modelo, scaler, PCA).

**"¿Por qué usan objetos anidados en lugar de campos planos?"**
Los objetos anidados agrupan información relacionada. Al separar `hyperparameters` de `metrics` queda claro qué es configuración de entrada y qué es resultado del entrenamiento. Esto facilita la lectura y las consultas específicas sobre cada grupo.

**"¿Por qué guardan `n_samples` en `metrics` y también `n_records` en la raíz?"**
Ambos indican cantidad de datos pero conceptualmente son distintos. `n_records` es el metadato del dataset origen. `n_samples` es un dato calculado durante el entrenamiento, que en teoría podría diferir si se aplicara alguna técnica como bootstrap. Se mantienen ambos para robustez.

**"¿Tienen validación de esquema en MongoDB?"**
La validación se hace en la aplicación antes de insertar, por simplicidad. MongoDB soporta JSON Schema validation a nivel de colección, pero para este proyecto la validación en código es suficiente y más flexible.

**"¿Qué índices tienen definidos?"**
Tres: `trained_at` descendente para ordenar el historial, `algorithm` y `dataset_source` para filtrar. Son los patrones de consulta más frecuentes en la aplicación.

**"¿Por qué los archivos binarios están fuera de MongoDB?"**
MongoDB soporta blobs pero no está diseñado para archivos grandes. Guardar los .pkl en el filesystem mantiene la base de datos ligera, hace las consultas al catálogo instantáneas y los archivos se cargan solo cuando se necesitan.

**"¿Los documentos se actualizan?"**
No. Cada entrenamiento crea un documento nuevo. Esto preserva el historial completo y evita perder información de experimentos anteriores. Solo se hacen operaciones de inserción, lectura y borrado.

---

## Resumen ejecutivo

**Base de datos**: `unsupervised_app` en MongoDB Atlas.

**Colección única**: `models` (historial de modelos entrenados).

**Estructura del documento**:
- Identificación (`_id`, `model_name`).
- Tipo de algoritmo (`algorithm`, `algorithm_label`).
- Timestamp (`trained_at`).
- Configuración (`hyperparameters`: n_components, covariance_type, random_state).
- Resultados (`metrics`: n_clusters, n_outliers, n_samples, n_samples_valid, silhouette, davies_bouldin, calinski_harabasz).
- Trazabilidad (`training_time_seconds`, `n_records`, `dataset_source`).
- Referencias a binarios (`model_file_path`, `scaler_file_path`, `pca_file_path`).

**Índices**: sobre `trained_at`, `algorithm` y `dataset_source`.

**Filosofía**: MongoDB es catálogo, filesystem es almacén. Cada capa cumple una función específica.

**Operaciones**: solo insertar, leer y borrar. No hay actualizaciones (cada experimento es inmutable).

**Validación**: implementada en la aplicación antes de insertar.

**Portabilidad**: rutas relativas y toda la información contextual en el documento.
