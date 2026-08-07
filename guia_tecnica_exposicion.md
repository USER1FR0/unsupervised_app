# Guía técnica de la aplicación — Manual para exposición

**Aplicación**: Big Five Analyzer
**Objetivo del documento**: entender a fondo cada elemento visible de la aplicación para poder defenderlo en la exposición sin sorpresas.

---

## Índice

1. [Estructura general de la aplicación](#1-estructura-general-de-la-aplicación)
2. [Página de inicio: carga de datos](#2-página-de-inicio-carga-de-datos)
3. [Los tres datasets de ejemplo](#3-los-tres-datasets-de-ejemplo)
4. [Página de exploración](#4-página-de-exploración)
5. [Página de entrenamiento y los hiperparámetros de GMM](#5-página-de-entrenamiento-y-los-hiperparámetros-de-gmm)
6. [Vista rápida del modelo entrenado](#6-vista-rápida-del-modelo-entrenado)
7. [Perfil promedio por cluster e interpretación textual](#7-perfil-promedio-por-cluster-e-interpretación-textual)
8. [Probabilidades por cluster: el diferenciador de GMM](#8-probabilidades-por-cluster-el-diferenciador-de-gmm)
9. [Página de modelos: historial de experimentos](#9-página-de-modelos-historial-de-experimentos)
10. [Página de clasificación: las cuatro opciones](#10-página-de-clasificación-las-cuatro-opciones)
11. [Botón "Reiniciar aplicación"](#11-botón-reiniciar-aplicación)
12. [Preguntas típicas del profesor y respuestas](#12-preguntas-típicas-del-profesor-y-respuestas)

---

## 1. Estructura general de la aplicación

La aplicación está dividida en secciones accesibles desde el panel lateral izquierdo:

- **Inicio**: carga del dataset a analizar.
- **Exploración**: visualización, filtros, estadística descriptiva, correlaciones.
- **Entrenamiento**: configuración y entrenamiento del modelo GMM.
- **Resultados**: métricas, visualización PCA, perfil por cluster, probabilidades.
- **Modelos**: historial de modelos entrenados y guardados.
- **Descargas**: exportación de datos y reportes PDF.
- **Clasificación**: uso de modelos guardados para clasificar personas nuevas.

En el panel también aparece un cuadro de **Estado** con:
- **Dataset**: cantidad de registros cargados actualmente.
- **Archivo**: nombre del archivo activo.
- **Modelos**: cantidad de modelos guardados en el historial.

Y una sección de **Acciones** con el botón **Reiniciar aplicación** (ver sección 11).

Esta estructura sigue el flujo natural del análisis: cargar → explorar → entrenar → interpretar → aplicar.

---

## 2. Página de inicio: carga de datos

### 2.1 Qué se ve

En la pantalla principal aparece el título de la aplicación y una sección llamada **Paso 1: Carga tus datos** con dos opciones:

- Subir un archivo CSV propio (arrastrar y soltar, o buscar).
- Seleccionar uno de los tres datasets de ejemplo: **Real (211)**, **Sintético (500)** o **Demo (100)**.

### 2.2 Qué formato debe tener el CSV

Cualquier CSV cargado debe tener las siguientes columnas para que el análisis funcione:

- **q1 a q20**: las 20 respuestas Likert (1 a 5).
- **O, C, E, A, N**: los cinco puntajes de dimensión (calculados a partir de las respuestas).
- **arquetipo**: nombre del arquetipo asignado por regla.
- **edad, genero, estado, municipio**: datos demográficos.
- **submitted_at**: timestamp de la respuesta (opcional pero recomendado).

Si el archivo no tiene los puntajes OCEAN precalculados, la aplicación puede calcularlos a partir de q1..q20 aplicando la fórmula de scoring (promedio con inversión de reversas).

### 2.3 Por qué esta estructura

El CSV es el único formato que la aplicación lee. Se eligió CSV porque:
- Es un formato universal.
- Permite exportar desde cualquier fuente (Google Sheets, Excel, otras herramientas).
- No depende de una conexión activa a una base de datos.
- Facilita las pruebas y la reproducibilidad.

---

## 3. Los tres datasets de ejemplo

Esta es una de las decisiones de diseño más importantes del proyecto y hay que dominarla para la exposición.

### 3.1 Por qué tres datasets y no uno

Un solo dataset no permite validar la robustez del modelo desde ángulos distintos. Al ofrecer tres, cada uno tiene un propósito específico dentro del análisis.

### 3.2 Dataset Real (211 registros)

**Qué es**: respuestas reales de personas que contestaron la encuesta pública.

**Cómo se generaron**: se difundió la encuesta a través de canales personales (WhatsApp, redes sociales) y se recolectaron respuestas durante varias semanas.

**Para qué sirve**:
- Es el dataset principal del proyecto.
- Con él se entrena el modelo definitivo.
- Refleja la distribución real de personalidades en la muestra encuestada.

**Qué esperar al analizarlo**:
- Métricas más modestas (silhouette bajo, entre 0.2 y 0.4).
- Clusters difusos y solapados.
- Distribución no uniforme entre arquetipos (algunos más frecuentes que otros).

**Por qué las métricas son modestas**: la personalidad humana real es difusa. Nadie es puramente un tipo. Los clusters existen pero no están perfectamente separados. Esto es lo esperado y refleja la realidad psicológica.

### 3.3 Dataset Sintético (500 registros)

**Qué es**: datos artificiales generados por un script.

**Cómo se generaron**:
- Se definieron 5 arquetipos base con medias OCEAN objetivo (por ejemplo, "Explorador" con Apertura alta y Conciencia baja).
- Para cada persona sintética: se elige un arquetipo aleatoriamente, se muestran valores OCEAN con distribución normal alrededor de esas medias, y se agrega ruido controlado.
- Se generan también las 20 respuestas Likert compatibles con esos scores.
- Se asignan demográficos plausibles.

**Para qué sirve**:
- **Validación del algoritmo**: como conocemos la estructura real (sabemos que hay 5 arquetipos base), podemos comprobar que GMM la recupera.
- **Contraste didáctico**: los clusters aquí son mucho más definidos que en el real, lo cual demuestra que el modelo funciona cuando los datos tienen estructura clara.
- **Referencia técnica**: sirve como "control" para comparar contra el dataset real.

**Qué esperar al analizarlo**:
- Métricas mucho mejores (silhouette > 0.5, a veces > 0.7).
- Clusters bien separados en la visualización PCA.
- BIC sugiere claramente k = 5.

**Por qué es útil para la exposición**: al comparar Real vs Sintético se puede decir:

> "Con datos controlados el algoritmo recupera perfectamente los 5 grupos que diseñamos, lo cual valida que funciona. Con datos reales las métricas son más modestas porque la personalidad humana no está tan compartimentada como los datos sintéticos, y eso también es un hallazgo interesante."

### 3.4 Dataset Demo (100 registros)

**Qué es**: datos artificiales generados con el mismo script del Sintético pero con **semilla aleatoria distinta**.

**Cómo se generaron**: mismo proceso que el Sintético, pero con seed=99 en lugar de seed=42. Esto produce registros distintos pero con la misma estructura general.

**Para qué sirve**:
- **Clasificación en vivo**: son datos que **el modelo nunca vio en entrenamiento**.
- **Demostrar generalización**: se entrena con el dataset Real (o Sintético), se guarda el modelo, y se aplica a estos 100 registros del Demo. Si clasifica bien, el modelo generaliza.
- **Contenido de la exposición**: es el momento en el que se dice "este modelo funciona, no solo con los datos que ya vio, sino con datos nuevos".

**Qué esperar al analizarlo**:
- La distribución de asignaciones a clusters debería parecerse a la del entrenamiento.
- Las probabilidades deberían tener la misma "forma" (algunas personas claras, algunas fronterizas).

### 3.5 Diferencias clave entre los tres

| Aspecto | Real | Sintético | Demo |
|---------|------|-----------|------|
| Origen | Personas reales | Script | Script (otra seed) |
| Cantidad | 211 | 500 | 100 |
| Estructura | Difusa, natural | Controlada, con 5 grupos claros | Controlada, con 5 grupos claros |
| Uso principal | Entrenar modelo definitivo | Validar que el algoritmo funciona | Clasificar datos nunca vistos |
| Métricas esperadas | Modestas | Altas | N/A (se clasifica, no se entrena) |

### 3.6 Justificación en la exposición

Si el profesor pregunta por qué tres datasets:

> "Un solo dataset no permite validar el modelo desde ángulos distintos. Con el Real entrenamos con datos auténticos. Con el Sintético demostramos que el algoritmo puede recuperar estructura conocida. Con el Demo probamos que el modelo generaliza a datos que nunca vio. Los tres juntos forman una narrativa completa: entrenar, validar y generalizar."

---

## 4. Página de exploración

### 4.1 Qué se ve

La página muestra el análisis del dataset activo, en este caso `real.csv` con 211 registros.

**Elementos visibles**:
- Aviso de valores nulos detectados (160 en el ejemplo).
- Sección de filtros: rango de edad, género, estado.
- Métricas: filtrados, totales, excluidos.
- Pestañas: Tabla, Estadística descriptiva, Distribución, Correlación.

### 4.2 El aviso de valores nulos

**Qué significa**: la aplicación detecta cuántas celdas están vacías o son inválidas en el CSV.

**Por qué aparece**: al recolectar respuestas reales, no todos completan la encuesta. Algunos dejan campos demográficos vacíos o no responden todas las preguntas.

**Cómo se manejan**: al entrenar el modelo, la aplicación **elimina las filas con valores nulos en las variables OCEAN**. Las filas con nulos solo en demográficos pueden mantenerse (los demográficos no se usan para el clustering).

**Justificación**: los algoritmos de clustering no funcionan con datos faltantes en las variables analizadas. La opción más limpia es descartar esas filas antes de entrenar. Si se imputaran los valores (rellenarlos con la media, por ejemplo), se introduciría información artificial que podría sesgar los clusters.

### 4.3 Los filtros

**Rango de edad**: slider con valores mínimo y máximo del dataset.
**Género**: multiselect con las opciones detectadas en los datos (Femenino, Masculino, Otro, Prefiero no decir).
**Estado**: multiselect con los estados de residencia detectados.

**Para qué sirven**:
- Análisis de subgrupos: por ejemplo, ver si los patrones de personalidad son distintos entre estados o géneros.
- Segmentación: entrenar un modelo solo con un subgrupo específico.
- Exploración: entender la composición demográfica de la muestra.

**Qué cambia al aplicarlos**:
- La tabla, la estadística y todas las gráficas se actualizan al dataset filtrado.
- El contador "Filtrados" refleja cuántos registros pasan los filtros.
- Al entrenar el modelo, se usa el subconjunto filtrado.

**Justificación de este diseño**: se prefirió aplicar los filtros globalmente para evitar inconsistencias. Si el usuario filtra por edad en Exploración, esos mismos filtros aplican en Entrenamiento y Resultados, garantizando que las métricas correspondan al mismo subconjunto de datos.

### 4.4 Las cuatro pestañas

**Tabla**: los registros crudos, con selector de columnas para elegir qué mostrar. Sirve para inspeccionar los datos, buscar valores atípicos, ver la calidad de la información.

**Estadística descriptiva**: cálculos propios (no `df.describe()`) de media, mediana, desviación estándar, mínimo, máximo, Q1 y Q3 por cada dimensión OCEAN. Se implementaron manualmente porque el instrumento pide algoritmos propios. También incluye **consistencia interna**: correlación promedio entre los 4 ítems de cada dimensión. Valores altos (>0.4) validan que los ítems miden lo que dicen medir.

**Distribución**: histogramas y boxplots por dimensión seleccionada. Ayuda a ver si los valores están concentrados en un rango o dispersos.

**Correlación**: matriz de correlación de Pearson entre las 5 dimensiones. Muestra si las dimensiones son independientes (correlaciones cercanas a 0) o si hay dependencia entre algunas.

---

## 5. Página de entrenamiento y los hiperparámetros de GMM

### 5.1 Estructura de la página

La página está dividida en tres pasos claros:

**Paso 1**: Optimización de hiperparámetros (gráficas BIC y AIC).
**Paso 2**: Configuración del modelo (elegir n_components, covariance_type, random_state).
**Paso 3**: Entrenamiento (botón para entrenar el modelo).

### 5.2 Paso 1: Gráficas de BIC y AIC

**Qué muestran**: cómo cambian BIC y AIC al variar k desde 2 hasta 10.

**Cómo se interpretan**:
- **BIC (Bayesian Information Criterion)**: el punto más bajo indica el k óptimo. En la captura, BIC es mínimo en k=2, con la línea punteada indicando la sugerencia.
- **AIC (Akaike Information Criterion)**: penaliza menos la complejidad, puede sugerir un k más alto.

**Por qué se muestran ambos**: BIC y AIC pueden dar sugerencias distintas. Mostrar ambos permite al analista decidir con más información. En general se prefiere BIC porque es más conservador.

**Sugerencia automática**: la aplicación calcula el k con menor BIC y lo sugiere debajo de las gráficas.

**Detalle importante**: la sugerencia es solo una recomendación. El usuario puede elegir un k distinto si tiene razones (por ejemplo, si conoce el dominio y sabe que espera 5 grupos, puede usar k=5 aunque BIC sugiera k=2).

### 5.3 Paso 2: Los hiperparámetros

Aquí se configuran los tres parámetros clave del modelo GMM.

#### 5.3.1 n_components

**Qué es**: el número de gaussianas en la mezcla, equivalente al número de clusters que buscará el modelo.

**Cómo elegirlo**:
- Empezar con la sugerencia automática de BIC.
- Si se conoce el dominio, ajustar según el conocimiento previo.
- Probar valores cercanos y comparar métricas.

**Cómo afecta el modelo**:
- k pequeño (2-3): clusters grandes y genéricos, poca discriminación.
- k grande (7-10): clusters pequeños y específicos, riesgo de sobreajuste.
- k intermedio (4-6): balance entre generalización y detalle.

**Justificación en el proyecto**: para el dataset real usamos el k sugerido por BIC. Para el sintético usamos k=5 (porque diseñamos 5 arquetipos base).

#### 5.3.2 covariance_type

**Qué es**: controla la forma de las gaussianas.

**Opciones**:

| Tipo | Descripción | Cuándo usarlo |
|------|-------------|---------------|
| **full** | Cada cluster tiene su propia matriz completa de covarianza | Datasets grandes, se quiere máxima flexibilidad |
| **tied** | Todos los clusters comparten la misma matriz | Datasets pequeños, se quiere restringir grados de libertad |
| **diag** | Matrices diagonales (elipses alineadas con los ejes) | Casos donde las dimensiones se asumen independientes |
| **spherical** | Un valor de varianza por cluster (esferas) | Similar a K-Means, más restrictivo |

**Justificación**: `full` es el más flexible y suele dar mejores resultados con datos suficientes. Nuestro dataset real (211 registros) está en el límite; con menos datos podría convenir `tied` o `diag` para evitar sobreajuste.

**Interpretación matemática**: la matriz de covarianza controla la forma de las elipses gaussianas. `full` permite elipses alargadas y rotadas en cualquier dirección. `spherical` solo permite círculos perfectos.

#### 5.3.3 random_state

**Qué es**: la semilla del generador de números aleatorios.

**Por qué importa**: el algoritmo EM que usa GMM tiene una fase de inicialización aleatoria. Con distintas semillas puede converger a resultados ligeramente distintos (mínimos locales).

**Por qué se fija**: para garantizar reproducibilidad. Con la misma semilla, mismos datos y mismos hiperparámetros, siempre se obtiene el mismo modelo.

**Elección típica**: 42 es una convención en la comunidad de ciencia de datos (referencia cultural a "La guía del autoestopista galáctico"). Cualquier número entero sirve.

**Impacto en la exposición**: se puede mostrar que al cambiar random_state las métricas varían ligeramente pero la estructura general se mantiene. Esto valida que el modelo es estable, no dependiente de una inicialización afortunada.

### 5.4 Paso 3: Entrenamiento

**Qué pasa al hacer click en "Entrenar GMM"**:

1. Los datos filtrados se cargan desde el estado de la aplicación.
2. Se aplica StandardScaler para escalar las 5 dimensiones.
3. Se instancia el modelo GMM con los hiperparámetros elegidos.
4. Se ejecuta el algoritmo EM hasta convergencia.
5. Se predicen las etiquetas de cluster para cada persona.
6. Se calculan las métricas de evaluación.
7. Se guarda todo en el estado de la aplicación para uso en otras páginas.

**Tiempo esperado**: 0.05 a 2 segundos para datasets de este tamaño.

---

## 6. Vista rápida del modelo entrenado

Después de entrenar, aparece un resumen inmediato con:

- **Clusters**: cantidad de clusters detectados (igual a n_components).
- **Silhouette**: coeficiente de silueta del modelo.
- **Davies-Bouldin**: índice de Davies-Bouldin.
- **BIC final**: BIC del modelo entrenado.
- **Interpretación silhouette**: texto que categoriza el silhouette en "Estructura fuerte", "Estructura razonable", "Estructura débil" o "Sin estructura clara".

En la captura: 8 clusters, silhouette 0.224, Davies-Bouldin 1.437, BIC 2980.0, interpretación "Sin estructura clara".

### 6.1 Cómo interpretar estos números

**Silhouette 0.224**: está en el rango de "sin estructura clara" según los umbrales estándar. Esto no significa que el modelo esté mal, significa que **los clusters no están perfectamente separados**. En datos de personalidad esto es esperado: los rasgos humanos son continuos y difusos.

**Davies-Bouldin 1.437**: está en zona aceptable (menor a 2 pero mayor a 1). Los clusters existen pero se traslapan.

**BIC 2980**: no tiene interpretación absoluta. Solo sirve para comparar modelos entre sí sobre los mismos datos. Un modelo con BIC 2500 sería preferible.

### 6.2 Qué decir si el silhouette sale bajo en la exposición

> "En datos de personalidad reales es normal que el silhouette sea moderado o bajo. La personalidad humana no es una variable con categorías rígidas: las personas tienen perfiles mixtos. Un silhouette de 0.22 nos indica que hay estructura pero es difusa, lo cual es consistente con la naturaleza del dominio. Si el silhouette fuera 0.9 en datos reales de personalidad, sospecharíamos que algo está mal."

---

## 7. Perfil promedio por cluster e interpretación textual

Esta sección aparece en la página de Resultados y es donde se hace la conexión con la interpretación humana.

### 7.1 Qué muestra la tabla

Por cada cluster descubierto, se muestran:

- **Número de cluster** (Cluster 0, Cluster 1, ...).
- **Apertura, Conciencia, Extraversión, Amabilidad, Neuroticismo**: promedio de cada dimensión OCEAN para las personas de ese cluster.
- **n**: cantidad de personas asignadas a ese cluster.
- **Interpretación**: texto generado automáticamente que describe el cluster.

### 7.2 Cómo se calcula el perfil

**Fórmula**: para cada cluster i y cada dimensión d:

```
perfil[i][d] = media(valores de d de todas las personas asignadas al cluster i)
```

Es simplemente el **promedio de los valores OCEAN** de las personas que el algoritmo agrupó en ese cluster.

**Por qué el promedio**: en un modelo GMM, el promedio de un cluster corresponde al vector μ (media) de esa gaussiana. Es el "centro" del cluster. Todas las personas de ese cluster están distribuidas alrededor de ese centro.

### 7.3 Cómo se genera la interpretación textual automática

**Proceso**:

1. Se calcula la **media global** de cada dimensión (promedio de todo el dataset).
2. Para cada cluster, se calcula la **diferencia** entre el perfil del cluster y la media global:
   ```
   diferencia[d] = perfil_cluster[d] - media_global[d]
   ```
3. Se ordenan las dimensiones por el valor **absoluto** de esa diferencia.
4. Se seleccionan las **2 dimensiones más distintivas** (las que más se apartan del promedio).
5. Se determina si cada una está "alta" (diferencia positiva) o "baja" (diferencia negativa).
6. Se genera el texto: "N personas con [Dimensión1] [alta/baja] y [Dimensión2] [alta/baja]."

### 7.4 Ejemplo del cálculo

Supongamos:
- Media global: O=3.2, C=3.4, E=3.1, A=3.6, N=3.0
- Cluster 1: O=4.37, C=3.11, E=2.38, A=4.43, N=2.44, n=31

Diferencias:
- O: 4.37 - 3.2 = +1.17 (alta)
- C: 3.11 - 3.4 = -0.29 (baja, poca diferencia)
- E: 2.38 - 3.1 = -0.72 (baja)
- A: 4.43 - 3.6 = +0.83 (alta)
- N: 2.44 - 3.0 = -0.56 (baja)

Ordenadas por magnitud absoluta:
1. O: 1.17 (más grande)
2. A: 0.83
3. E: 0.72
4. N: 0.56
5. C: 0.29

Se seleccionan las dos primeras: O alta y A alta.

Texto generado: "31 personas con Apertura alta y Amabilidad alta."

### 7.5 Por qué esta lógica y no otra

**Por qué comparar con la media global**: sirve como referencia. Un cluster con valores altos en todo no es distintivo si toda la muestra tiene valores altos. Comparar con la media identifica lo que hace **único** a cada cluster.

**Por qué solo 2 dimensiones**: mantener el texto conciso. Con más dimensiones el texto sería confuso ("N personas con O alta, C baja, E moderada, A alta y N baja..."), sin aportar más valor.

**Por qué "alta" y "baja" y no valores numéricos**: la descripción textual es más humana. Un profesor, un usuario final, o cualquier lector entiende "Apertura alta" mejor que "O = 4.37".

### 7.6 Relación con los arquetipos

Aquí hay que aclarar algo importante que puede confundir en la exposición.

**Los clusters y los arquetipos son cosas distintas**:

- **Arquetipo**: se asigna a cada persona cuando responde la encuesta, usando una regla determinista: dimensión dominante × dimensión más baja mapeada a la matriz 5×5.
- **Cluster**: se asigna cuando el algoritmo GMM analiza los datos y descubre grupos naturales sin conocer los arquetipos.

**¿Deberían coincidir?**: no necesariamente. El arquetipo depende solo de dos dimensiones extremas por persona. El cluster considera las 5 dimensiones globalmente y usa criterios estadísticos.

**Es interesante compararlos**: si los clusters agrupan a personas con arquetipos similares, valida la lógica de la encuesta. Si son muy distintos, sugiere que el modelo descubre estructura que la lógica de arquetipos no captura.

### 7.7 Interpretación práctica de la tabla

Con los datos de la captura:

| Cluster | O | C | E | A | N | n | Interpretación |
|---------|---|---|---|---|---|---|----------------|
| 0 | 1.84 | 1.97 | 3.47 | 4.22 | 3.38 | 8 | 8 personas con Apertura baja y Conciencia baja |
| 1 | 4.37 | 3.11 | 2.38 | 4.43 | 2.44 | 31 | 31 personas con Apertura alta y Amabilidad alta |
| 2 | 3.12 | 3.49 | 2.17 | 3.23 | 4.01 | 21 | 21 personas con Extraversión baja y Neuroticismo alta |
| 3 | 3.3 | 2.38 | 4.28 | 3.12 | 4.29 | 44 | 44 personas con Neuroticismo alta y Extraversión alta |
| 4 | 2.6 | 4.38 | 3.38 | 3.6 | 2.06 | 42 | 42 personas con Conciencia alta y Neuroticismo baja |
| 5 | 1.97 | 3.34 | 3.56 | 2.09 | 2.16 | 8 | 8 personas con Apertura baja y Amabilidad baja |
| 6 | 4.11 | 4.08 | 2.85 | 3.41 | 3.24 | 32 | 32 personas con Conciencia alta y Apertura alta |
| 7 | 4.23 | 3.25 | 3.34 | 3.81 | 2.13 | 25 | 25 personas con Neuroticismo baja y Apertura alta |

**Qué se puede decir en la exposición**:
- Los clusters más pobladosos (3 con 44 personas, 4 con 42, 6 con 32, 1 con 31) representan los perfiles más frecuentes en la muestra.
- Los clusters pequeños (0 con 8, 5 con 8) pueden ser perfiles atípicos.
- Se observa diversidad: hay clusters con Conciencia alta, otros con Neuroticismo alto, etc. No hay un solo tipo dominante.

---

## 8. Probabilidades por cluster: el diferenciador de GMM

Esta es la sección más importante para la exposición porque es el **valor único de GMM** que ningún otro algoritmo puede ofrecer.

### 8.1 Qué muestra

Una tabla con:
- Edad y género de cada persona.
- **cluster_asignado**: el cluster con mayor probabilidad (asignación dura).
- **Prob. máxima**: qué tan alta es la probabilidad más grande.
- **P(C0), P(C1), ..., P(Ck)**: la probabilidad de pertenecer a cada cluster.

Y un indicador general:
- **Prob. máxima promedio**: qué tan seguros están las asignaciones en promedio.

### 8.2 Cómo se calculan las probabilidades

**Fórmula**: para cada persona x y cada cluster i:

```
P(cluster i | x) = πᵢ · N(x | μᵢ, Σᵢ) / Σⱼ πⱼ · N(x | μⱼ, Σⱼ)
```

Donde:
- πᵢ es el peso del cluster i (proporción de personas en ese cluster).
- N(x | μᵢ, Σᵢ) es la densidad de la gaussiana i evaluada en x.
- El denominador es la suma sobre todos los clusters (para normalizar).

**Intuitivamente**: se pregunta "¿qué tan probable es que esta persona haya sido generada por la gaussiana i?" y se normaliza para que las probabilidades sumen 1.

### 8.3 Cómo interpretar los resultados

**Persona clara** (una probabilidad domina):
```
Cluster asignado: 4
Probabilidades: P(C4) = 0.87, P(C6) = 0.10, resto casi 0
Interpretación: encaja muy bien en el cluster 4.
```

**Persona fronteriza** (dos probabilidades similares):
```
Cluster asignado: 2 (probabilidad 0.42)
Probabilidades: P(C2) = 0.42, P(C3) = 0.38, resto bajos
Interpretación: perfil mixto entre cluster 2 y 3. El modelo se decide por el 2 pero con incertidumbre.
```

**Persona muy dispersa** (varias probabilidades bajas):
```
Probabilidades: 0.25, 0.22, 0.20, 0.18, 0.15
Interpretación: no encaja claramente en ningún cluster. Perfil muy general o atípico.
```

### 8.4 Qué es la probabilidad máxima promedio

**Fórmula**:
```
prob_maxima_promedio = media de max(P(cluster i | x)) sobre todas las personas
```

Es un indicador global de cuán "seguro" está el modelo en promedio.

**Interpretación**:
- Cercano a 1 (por ejemplo 0.9): el modelo asigna con alta certeza.
- Cercano a 1/k (por ejemplo 0.2 con k=5): el modelo tiene mucha incertidumbre.

En la captura se ve 0.852, que es un valor alto: la mayoría de las personas tienen un cluster dominante claro.

### 8.5 El filtro de personas fronterizas

**Qué es**: checkbox "Mostrar solo personas fronterizas (prob_maxima < 0.7)".

**Para qué sirve**: identificar rápidamente qué personas tienen perfil mixto, es decir, no encajan claramente en un solo cluster.

**Por qué son interesantes**:
- En psicología, las personas fronterizas suelen ser las más ricas de analizar: tienen rasgos de múltiples perfiles.
- Para el modelo, son las personas donde la incertidumbre es alta.
- Para la exposición, son el mejor ejemplo del valor de GMM: mostrar que el modelo dice "no estoy seguro" en vez de forzar una asignación.

### 8.6 Argumento clave para la exposición

> "GMM es el único algoritmo del que hablamos que entrega probabilidades. K-Means, Jerárquico y DBSCAN solo dan asignaciones duras: 'esta persona es del cluster 3, punto'. GMM dice: 'esta persona es 68% del cluster 3, 24% del cluster 5, y 8% del resto'. Esto refleja la naturaleza real de la personalidad, donde nadie es un tipo puro. Además, al identificar personas fronterizas podemos analizar perfiles mixtos que son psicológicamente interesantes."

---

## 9. Página de modelos: historial de experimentos

### 9.1 Qué se ve

Una lista de todos los modelos que se han entrenado y guardado, cada uno con:

- **Nombre**: asignado por el usuario al guardar (por ejemplo "Modelo1").
- **Algoritmo**: siempre GMM en la versión actual.
- **Métricas resumidas**: clusters, silhouette.
- **Fecha de entrenamiento**.

Al expandir un modelo se ven:

**Hiperparámetros usados**:
- n_components
- covariance_type
- random_state

**Métricas detalladas**:
- n_clusters, n_outliers, n_samples, n_samples_valid
- silhouette, davies_bouldin, calinski_harabasz

**Metadata**:
- Registros usados
- Archivo origen
- Tiempo de entrenamiento

Y dos botones:
- **Clasificar con este**: lleva a la página de Clasificación con este modelo preseleccionado.
- **Eliminar**: borra el modelo del historial (y su archivo .pkl).

### 9.2 Dónde se guardan los modelos

**En el filesystem**: los archivos `.pkl` (modelo serializado + scaler) se guardan en la carpeta `models/` del proyecto.

**En MongoDB**: los metadatos (hiperparámetros, métricas, nombre, rutas a los archivos) se guardan en la colección `models`. **Solo se guardan los metadatos, no los datos crudos de las respuestas**.

### 9.3 Por qué guardar los modelos

- **Reproducibilidad**: se puede volver a los resultados de un experimento sin reentrenar.
- **Comparativa**: se pueden comparar varios modelos entrenados con distintos hiperparámetros o datasets.
- **Aplicación**: el modelo guardado se puede usar para clasificar datos nuevos (página de Clasificación).

### 9.4 Por qué guardar el scaler junto al modelo

Al entrenar se aplica StandardScaler para normalizar los datos. Si mañana se quiere clasificar una persona nueva, hay que aplicarle **exactamente la misma normalización**. Si se recalcula el scaler con datos distintos, los valores no serán comparables y las predicciones no serán consistentes con el entrenamiento.

Por eso el scaler ajustado se guarda como archivo separado (`scaler_gmm_20260806.pkl`) junto al modelo (`gmm_20260806.pkl`) y ambos se cargan juntos en el momento de clasificar.

---

## 10. Página de clasificación: las cuatro opciones

Aquí se aplica un modelo previamente entrenado a datos nuevos para obtener predicciones.

### 10.1 Estructura de la página

**Paso 1**: Seleccionar un modelo del historial.
**Paso 2**: Elegir los datos a clasificar (4 opciones).
**Paso 3**: Ejecutar la clasificación.

### 10.2 Paso 1: Seleccionar el modelo

Se muestra un dropdown con los modelos guardados. Al elegir uno, se muestran sus métricas y detalles: silhouette, Davies-Bouldin, cantidad de registros con los que se entrenó, hiperparámetros usados.

**Por qué mostrar los detalles**: el usuario debe saber con qué modelo va a clasificar. Si el modelo tuvo silhouette bajo, las predicciones también tendrán incertidumbre. Si el modelo se entrenó con 39 registros, quizás no sea tan robusto como uno entrenado con 500.

### 10.3 Paso 2: Las cuatro opciones de datos a clasificar

Esta es la parte más rica de la exposición porque cada opción tiene un propósito distinto.

#### 10.3.1 Opción 1: Cargar CSV nuevo

**Qué es**: subir un archivo CSV con respuestas Big Five.

**Cuándo usarlo**:
- Se tiene un conjunto de respuestas de otras personas (por ejemplo, respuestas nuevas recolectadas después del entrenamiento).
- Se quiere aplicar el modelo a datos completamente externos.

**Requisitos**: el CSV debe tener las 5 dimensiones OCEAN (mismas columnas O, C, E, A, N). Si trae también q1..q20, se pueden validar.

**Escenario de exposición**: "Si mañana el profesor nos entrega respuestas de otro grupo, podemos cargarlas aquí y clasificarlas al instante con nuestro modelo entrenado."

#### 10.3.2 Opción 2: Dataset de ejemplo

**Qué es**: usar uno de los tres datasets base (Real, Sintético o Demo) que ya vienen en el proyecto.

**Cuándo usarlo**:
- Se quiere clasificar un dataset conocido y ver los resultados.
- Se quiere hacer una prueba rápida sin cargar nada.

**Diferencia importante**: si se elige "Real" pero el modelo se entrenó con "Real", en realidad estás pidiéndole que clasifique los mismos datos con los que se entrenó (poco interesante). El caso más útil es entrenar con Real y clasificar con Sintético, o al revés.

**Escenario de exposición**: "Tenemos tres datasets de referencia disponibles. Si el modelo se entrenó con Real, podemos verificar cómo se comporta con Sintético o Demo."

#### 10.3.3 Opción 3: Generar demo

**Qué es**: llamar al generador de datos sintéticos al momento y crear un dataset nuevo desde cero.

**Cómo funciona**: el mismo script que generó los datasets Sintético y Demo se ejecuta al vuelo con parámetros configurables (cantidad de registros, seed).

**Cuándo usarlo**:
- Se quiere clasificar datos nunca antes vistos, generados en el momento.
- Se quiere demostrar la capacidad de generar datos artificiales bajo demanda.
- Se quiere mostrar reproducibilidad: con la misma seed se obtiene el mismo resultado, con distinta seed se obtienen datos distintos.

**Escenario de exposición**: "Podemos generar un dataset nuevo justo ahora, con una semilla que nunca hemos usado, y ver cómo el modelo lo clasifica. Esto demuestra que el modelo no depende de datos específicos, sino que aprendió patrones generales."

#### 10.3.4 Opción 4: Usar dataset actual

**Qué es**: usar el dataset que actualmente está cargado en la aplicación (el que se ve en Exploración).

**Cuándo usarlo**:
- Se quiere clasificar los mismos datos que se están explorando en ese momento.
- Útil para comparar clusters descubiertos vs clusters predichos.

**Escenario de exposición**: "Podemos ver cómo el modelo actual clasifica el dataset que estamos explorando en este momento, sin cambiar de dataset."

### 10.4 Por qué cuatro opciones y no una

Cada opción cubre un caso de uso distinto en la exposición y en el uso real:

- **CSV nuevo**: para el mundo real, donde llegan datos externos.
- **Dataset de ejemplo**: para pruebas rápidas y validación cruzada entre los tres datasets.
- **Generar demo**: para mostrar reproducibilidad y capacidad de generación bajo demanda.
- **Dataset actual**: para conveniencia y comparación directa.

Tener las cuatro opciones demuestra que el sistema es flexible y puede aplicarse en múltiples escenarios.

### 10.5 Cómo se generan los datos sintéticos

**Script**: `src/data/synthetic_generator.py`.

**Proceso**:

1. Se definen 5 arquetipos base con medias OCEAN objetivo. Por ejemplo:
   - Explorador: O=4.5, C=2.3, E=3.4, A=3.5, N=2.8
   - Arquitecto: O=3.2, C=4.5, E=2.2, A=3.4, N=2.5
   - Y así con Carismático, Guardián, Intenso.
2. A cada arquetipo se le asigna un peso (proporción esperada en el dataset).
3. Para cada persona sintética:
   - Se elige un arquetipo aleatoriamente según los pesos.
   - Para cada dimensión OCEAN, se muestra un valor de una distribución normal con media = valor del arquetipo, desviación = 0.55.
   - Se recorta al rango [1, 5].
   - Se generan 20 respuestas Likert 1-5 que producen esos scores al aplicar la fórmula de scoring.
   - Se asigna edad, género, estado, municipio con distribuciones plausibles.
   - Se calcula el arquetipo con la lógica determinista (dominante × más baja).

**Reproducibilidad**: al usar la misma semilla (seed), se obtienen exactamente los mismos datos. Con seed distinta, datos distintos pero con la misma estructura general.

### 10.6 Paso 3: Ejecutar clasificación

Al hacer click en "Clasificar":

1. Se carga el modelo y el scaler desde el filesystem.
2. Se aplica el mismo scaler que se usó en entrenamiento.
3. Se ejecuta `predict()` para obtener las etiquetas.
4. Se ejecuta `predict_proba()` para obtener las probabilidades.
5. Se muestran los resultados: cantidad clasificada, distribución por cluster, tabla con probabilidades.
6. Se ofrece descargar los resultados como CSV.

### 10.7 Qué se obtiene con la clasificación

**Etiquetas**: cada persona nueva recibe un número de cluster (0, 1, 2, ...).
**Probabilidades**: para cada persona, la probabilidad de pertenecer a cada cluster.
**Distribución**: cuántas personas cayeron en cada cluster.
**Personas fronterizas**: aquellas con probabilidad máxima < 0.7.

### 10.8 Diferencia entre entrenar y clasificar

Es crítico entender esta diferencia para la exposición:

**Entrenar**: el modelo "aprende" los patrones de los datos. Descubre los clusters, calcula sus medias, sus covarianzas, sus pesos.

**Clasificar**: el modelo, ya entrenado, aplica lo aprendido a datos nuevos. **No aprende nada nuevo**, solo predice.

**Analogía**: entrenar es como un profesor que estudia mucho para dominar un tema. Clasificar es cuando ese profesor recibe un examen nuevo y responde usando lo que ya sabía. No aprende nada del examen, solo aplica.

---

## 11. Botón "Reiniciar aplicación"

### 11.1 Qué hace

Limpia todo el estado de la aplicación:
- Dataset cargado.
- Modelo actualmente entrenado (no los guardados).
- Filtros aplicados.
- Resultados calculados.
- Selecciones en cada página.

### 11.2 Qué NO hace

- No borra los modelos guardados en el historial.
- No borra los datasets de ejemplo (Real, Sintético, Demo).
- No modifica MongoDB.
- No modifica archivos en el filesystem.

### 11.3 Para qué sirve

- **Empezar de cero**: si se cargó un dataset y ya no se necesita, este botón limpia todo sin tener que cerrar y reabrir la app.
- **Cambiar de contexto**: si se estaba trabajando con el dataset Real y se quiere pasar al Sintético desde cero.
- **Recuperarse de estados extraños**: si algo se ve raro en la interfaz por acumulación de estados previos, este botón resuelve.
- **Durante la exposición**: útil para reiniciar entre demos y no cargar contexto anterior.

### 11.4 Cuándo NO usarlo

No hay que darle a este botón después de entrenar un modelo si aún no se guardó. Porque si el modelo actual no se guardó al historial, se pierde. Los modelos ya guardados se mantienen.

---

## 12. Preguntas típicas del profesor y respuestas

### 12.1 Sobre los datasets

**"¿Por qué tres datasets?"**
Un solo dataset no valida el modelo desde distintos ángulos. Con Real entrenamos con datos auténticos, con Sintético validamos que el algoritmo recupera estructura conocida, y con Demo probamos que el modelo generaliza a datos que nunca vio.

**"¿Cómo saben que sus datos sintéticos son realistas?"**
El generador define arquetipos con medias OCEAN plausibles y agrega ruido gaussiano. Después de generar validamos: distribución no uniforme entre arquetipos, medias globales cerca de 3.0, correlaciones intra-dimensión similares al dataset real. Si fueran demasiado perfectos, los clusters serían triviales.

### 12.2 Sobre los hiperparámetros

**"¿Por qué elegiste covariance_type=full?"**
Full es el más flexible: cada cluster tiene su propia forma, tamaño y orientación de elipse. Con suficientes datos captura mejor la estructura real. Con muestras muy pequeñas se prefiere tied o diag para prevenir sobreajuste.

**"¿Por qué random_state=42?"**
Es una convención en ciencia de datos y garantiza reproducibilidad. Cualquier número entero fijo sirve; lo importante es que sea constante para que el mismo entrenamiento dé siempre el mismo resultado.

**"¿Cómo eliges el n_components?"**
Se prueba k=2 hasta k=10 y se elige el que minimiza BIC. BIC penaliza modelos complejos, previniendo el sobreajuste. En algunos casos también consideramos el silhouette para verificar que los clusters están bien separados.

### 12.3 Sobre los clusters y arquetipos

**"¿Los clusters coinciden con los arquetipos?"**
No necesariamente. Los arquetipos se asignan por regla determinista (dimensión dominante × dimensión más baja). Los clusters los descubre GMM considerando las 5 dimensiones globalmente. Analizar la coincidencia entre ambos es interesante en sí mismo.

**"¿Cómo se calcula la interpretación textual de cada cluster?"**
Se compara el perfil promedio del cluster contra la media global del dataset. Se seleccionan las 2 dimensiones con mayor diferencia absoluta y se determina si están "altas" o "bajas". Con eso se genera el texto.

### 12.4 Sobre las métricas

**"¿Por qué el silhouette es tan bajo en el dataset real?"**
En datos de personalidad reales es esperado. La personalidad humana no tiene categorías rígidas. Un silhouette moderado indica que hay estructura pero es difusa, lo cual refleja la realidad psicológica. Si fuera muy alto en datos reales, sospecharíamos que algo está mal.

**"¿Cómo interpretan Davies-Bouldin?"**
Menor es mejor. Valores por debajo de 1 indican clusters bien definidos. Entre 1 y 2 es aceptable. Por encima de 2 los clusters se traslapan mucho.

### 12.5 Sobre el proceso

**"¿Por qué escalar los datos si ya están en la misma escala 1-5?"**
Aunque estén en el mismo rango, el escalado garantiza que todas las variables tengan media 0 y desviación estándar 1. Esto estabiliza numéricamente el algoritmo, hace las métricas más comparables y es una buena práctica de la industria.

**"¿Por qué guardan solo los modelos y no los datos?"**
Las respuestas son datos crudos que viven como CSV. Los modelos entrenados representan el "conocimiento extraído", que es el activo valioso del análisis. MongoDB solo guarda metadatos de modelos, no datos personales, lo cual también es mejor por privacidad.

### 12.6 Sobre las probabilidades

**"¿Qué diferencia hay entre las probabilidades de GMM y las asignaciones de K-Means?"**
K-Means da asignaciones duras: cada persona es del cluster X, punto. GMM da probabilidades: esta persona es 68% del cluster 3, 24% del cluster 5. Esto es superior para modelar personalidad porque los rasgos humanos son mixtos.

**"¿Qué son las personas fronterizas?"**
Son personas cuya probabilidad máxima es menor a 0.7. El modelo las asigna a un cluster pero con incertidumbre porque tienen rasgos de varios grupos. En psicología suelen ser los casos más interesantes de analizar.

### 12.7 Sobre la clasificación

**"¿Cómo funciona la clasificación con un modelo guardado?"**
Se carga el modelo y el scaler que se guardaron al momento de entrenar. A los datos nuevos se les aplica el mismo scaler para que estén en la misma escala. Después el modelo predice a qué cluster pertenecen y con qué probabilidad.

**"¿Por qué no reentrenar el modelo cada vez?"**
El entrenamiento es costoso y requiere muchos datos. Un modelo entrenado con 200 respuestas puede clasificar miles de personas nuevas sin necesidad de reentrenar. Además, reentrenar con distintos datos cambiaría los clusters, perdiendo comparabilidad.

**"¿Puede el modelo clasificar personas con perfiles muy distintos a los del entrenamiento?"**
Puede intentarlo, pero con incertidumbre. Si una persona nueva tiene un perfil muy distinto a todos los del entrenamiento, sus probabilidades estarán dispersas y su probabilidad máxima será baja. Esto es una advertencia natural de que el modelo no está seguro.

---

## Resumen ejecutivo

**El proyecto**: aplicación para descubrir grupos naturales de personalidad usando GMM sobre datos del modelo Big Five.

**Los tres datasets**:
- **Real (211)**: respuestas auténticas, entrena el modelo definitivo.
- **Sintético (500)**: datos controlados, valida que el algoritmo funciona.
- **Demo (100)**: datos nunca vistos, prueba la generalización.

**Los hiperparámetros de GMM**:
- **n_components**: número de clusters, elegido por BIC.
- **covariance_type**: forma de las gaussianas (full = flexible).
- **random_state**: semilla para reproducibilidad.

**Las etiquetas**:
- **Arquetipo**: se asigna al responder la encuesta, por regla determinista.
- **Cluster**: lo descubre GMM analizando el dataset completo.
- **No siempre coinciden**: y esa comparación es interesante.

**La interpretación textual**: se genera automáticamente comparando el perfil de cada cluster contra la media global, seleccionando las 2 dimensiones más distintivas.

**Las probabilidades**: son el valor único de GMM. Muestran que las personas tienen perfiles mixtos, no puros. Las fronterizas son las más interesantes.

**La clasificación**: aplica un modelo guardado a datos nuevos. Cuatro opciones para elegir esos datos según el escenario.

**El reinicio**: limpia el estado sin borrar modelos guardados. Útil para empezar de cero durante la exposición.
