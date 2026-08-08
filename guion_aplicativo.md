# Guión de exposición del aplicativo — Con interpretación de resultados

Enfoque en cómo leer y explicar cada gráfica, cada métrica y cada resultado. Nivel: ingeniería con base en ML.

---

## Inicio

**Qué es**

Punto de entrada de la aplicación. Aquí se carga el dataset a analizar.

**Qué se muestra en pantalla**

Dos áreas: zona de carga de archivo CSV, y botones para seleccionar uno de los tres datasets de ejemplo (Real, Sintético, Demo).

En el panel lateral: estado de la sesión con contador de registros, nombre del archivo activo y cantidad de modelos guardados.

**Cómo funciona**

Al cargar un CSV se valida que contenga las columnas mínimas requeridas: `q1..q20`, `O, C, E, A, N`, `arquetipo`, y demográficos. Si faltan las dimensiones OCEAN, la app las calcula a partir de q1..q20 aplicando el scoring definido en `src/data/scoring.py`.

Al seleccionar un dataset de ejemplo, se lee directamente del sistema de archivos: `data/real.csv`, `data/synthetic.csv` o `data/demo.csv`.

**Los tres datasets**

**Real (~211 registros)**: respuestas auténticas de personas que contestaron la encuesta pública. Distribución no controlada, refleja la muestra real. Se usa como **dataset principal para entrenar el modelo definitivo**.

**Sintético (500 registros)**: datos generados por `src/data/synthetic_generator.py`. El generador define cinco arquetipos base con medias OCEAN objetivo (Explorer, Architect, Charismatic, Guardian, Intense), les asigna pesos desiguales para reflejar frecuencia realista, y genera personas con distribución normal alrededor de esas medias. Sirve para **validar que el algoritmo recupera estructura conocida**.

**Demo (100 registros)**: mismo generador que el Sintético pero con semilla distinta (seed=99 vs seed=42). Sirve para **clasificación en vivo con datos que el modelo nunca vio**.

**Justificación de tres datasets**

Un solo dataset no valida el modelo desde ángulos distintos. La estrategia:
- Entrenar con datos auténticos (Real).
- Verificar recuperación de estructura conocida (Sintético).
- Probar generalización a datos completamente nuevos (Demo).

---

## Exploración

**Qué es**

Análisis descriptivo del dataset cargado, previo al entrenamiento.

### Aviso de valores nulos

Si el CSV tiene celdas vacías, la app las detecta y muestra un aviso. En la captura: 160 valores nulos detectados.

**Cómo se manejan**: al entrenar, se descartan las filas con nulos en cualquiera de las cinco dimensiones OCEAN. Los algoritmos de clustering no funcionan con datos faltantes.

**Decisión técnica**: no imputamos *(rellenar valores faltantes con estimaciones como la media)* porque introduciría información artificial que sesgaría los clusters. Descartar filas incompletas es más limpio.

### Filtros

Tres controles:
- **Rango de edad**: slider con mínimo y máximo del dataset.
- **Género**: multiselect.
- **Estado**: multiselect.

Al aplicar filtros, todas las páginas siguientes trabajan sobre el subconjunto filtrado. Garantiza consistencia entre exploración, entrenamiento y resultados.

### Pestaña Tabla

Registros crudos con selector de columnas. Útil para inspección visual y detección de outliers.

### Pestaña Estadística descriptiva — Cómo leer cada métrica

Implementaciones propias en `src/stats/descriptive.py`, no `df.describe()`.

**Media**: promedio simple. En Big Five debería estar cerca de 3.0 (centro de la escala). Valores muy alejados sugieren sesgos de la muestra.

**Mediana**: valor central del conjunto ordenado. Si difiere mucho de la media, la distribución es asimétrica. En Big Five media y mediana suelen coincidir dentro de ±0.2.

**Desviación estándar muestral**: usando `(n-1)` en el denominador. Rango típico en Big Five: 0.6 a 1.0. Valores por debajo de 0.4 = muestra homogénea. Valores por encima de 1.2 = muestra muy dispersa.

**Cuartiles Q1 y Q3**: valores que dejan 25% y 75% de los datos debajo. El rango intercuartílico (Q3-Q1) mide dispersión sin ser afectado por outliers. Típicamente 1.0 a 1.5 en Big Five.

**Mínimo y máximo**: si el mínimo es 1.0 exacto y el máximo 5.0 exacto, significa que hay personas en los extremos. Si están en 1.5 y 4.5, la muestra evita respuestas extremas.

### Consistencia interna del instrumento — Cómo interpretarla

Métrica psicométrica: correlación promedio entre los cuatro ítems que miden la misma dimensión.

**Rangos de interpretación**:
- **> 0.7**: excelente consistencia. Los ítems son casi redundantes entre sí.
- **0.5 – 0.7**: buena consistencia. Los ítems miden lo mismo con matices.
- **0.4 – 0.5**: aceptable. Estándar mínimo para instrumentos psicométricos.
- **< 0.4**: baja consistencia. Los ítems podrían no medir lo mismo.

**Qué hacer si sale baja**: revisar la redacción de los ítems. Puede haber uno mal formulado o ambiguo.

**En nuestro caso**: si nuestros valores están entre 0.4 y 0.6, el instrumento es válido para el propósito exploratorio del proyecto.

### Pestaña Distribución — Cómo leer histogramas y boxplots

**Histograma**: cuenta cuántas personas caen en cada rango de valores.

Formas típicas y qué significan:
- **Campana centrada en 3**: distribución normal. La muestra promedia el centro. Es lo esperado en Big Five.
- **Campana sesgada a la derecha (más masa en 4-5)**: la muestra puntúa alto en esa dimensión. Ejemplo típico en Amabilidad: la gente suele auto-reportarse más amable de lo que es.
- **Campana sesgada a la izquierda (más masa en 1-2)**: la muestra puntúa bajo. Menos común, puede aparecer en Neuroticismo si la muestra es joven y estable.
- **Bimodal (dos picos)**: puede indicar dos subgrupos en la muestra. Interesante para analizar.
- **Uniforme (sin forma clara)**: dispersión sin patrón. Puede indicar problema con la dimensión o muestra muy heterogénea.

**Boxplot**: representa la distribución con cinco números.

Elementos:
- **Caja**: del Q1 al Q3. Contiene el 50% central de los datos.
- **Línea dentro de la caja**: mediana.
- **Bigotes**: extienden hasta 1.5 veces el rango intercuartílico.
- **Puntos fuera de los bigotes**: outliers.

Cómo se leen:
- **Caja alta y estrecha**: la mayoría puntúa alto y de forma similar.
- **Caja ancha**: alta dispersión, opiniones diversas.
- **Mediana en el centro de la caja**: distribución simétrica.
- **Mediana hacia el borde superior de la caja**: sesgo a la izquierda.
- **Muchos outliers**: personas atípicas respecto a la mayoría.

### Pestaña Correlación — Cómo leer la matriz

Matriz 5x5 con la correlación de Pearson entre cada par de dimensiones OCEAN. Calculada manualmente con `cov(X,Y) / (std_X · std_Y)`.

**Rango**: -1 a 1.

Cómo leer valores:
- **> 0.7**: correlación fuerte positiva. Las dimensiones se mueven juntas.
- **0.3 – 0.7**: correlación moderada positiva.
- **-0.3 a 0.3**: prácticamente independientes. **Es lo esperado en Big Five**.
- **-0.3 a -0.7**: correlación moderada negativa. Se mueven en direcciones opuestas.
- **< -0.7**: correlación fuerte negativa.

**Qué esperar en el heatmap**:
- Diagonal principal en 1.0 (una variable se correlaciona perfectamente consigo misma).
- Valores fuera de la diagonal cercanos a cero (independencia entre dimensiones).

**Si aparecen correlaciones fuertes**:
- **Positiva fuerte entre O y E**: la muestra podría interpretar "apertura a experiencias" y "extraversión social" como lo mismo. Puede indicar limitación del instrumento en esa muestra.
- **Negativa fuerte entre C y N**: personas organizadas resultan menos ansiosas. Es esperable, es un hallazgo válido.

**Uso del heatmap para el análisis**: si dos dimensiones están muy correlacionadas, aportan información redundante al clustering. GMM podría no distinguir claramente entre ellas.

---

## Entrenamiento

**Qué es**

Configuración y entrenamiento del modelo GMM.

### Paso 1: Optimización de hiperparámetros — Cómo leer las gráficas de BIC y AIC

Se muestran dos curvas: BIC y AIC en función de k de 2 a 10.

**Interpretación de la curva de BIC**:

**Caso 1: Curva en U clara**

Baja hasta un mínimo y después sube. El mínimo es el k óptimo. Es el caso ideal, indica estructura clara en los datos.

Ejemplo: BIC toca fondo en k=5, después sube en k=6, 7, 8. Elegimos k=5.

**Caso 2: Curva monótonamente decreciente**

Baja continuamente sin llegar a mínimo dentro del rango probado. Significa que los datos toleran cada vez más componentes sin sobreajustar demasiado.

Cómo actuar: aplicar regla del codo. Buscar el punto donde la pendiente se aplana. El k del "codo" es una elección razonable aunque no sea el mínimo absoluto.

**Caso 3: Curva plana**

Cambios muy pequeños al variar k. Los datos no tienen estructura clara de clusters, o hay muy pocos datos.

Cómo actuar: reconsiderar si el clustering es apropiado para estos datos, o buscar más datos.

**Interpretación de la curva de AIC**:

Similar pero suele sugerir un k mayor que BIC. Si BIC sugiere k=5 y AIC sugiere k=7, el modelo elegido es k=5 (BIC es más conservador y previene sobreajuste).

**Comparación entre BIC y AIC**:

- Si ambas coinciden en el mínimo: **alta confianza** en el k elegido.
- Si difieren en 1 o 2 componentes: se elige BIC. Diferencia esperada.
- Si difieren en 3 o más componentes: los datos son difusos, ninguna estructura clara.

**En la captura ejemplo**: BIC sugiere k=2 con línea punteada. La curva de AIC va bajando pero de forma irregular. En este caso: pocos datos (39 muestras) y estructura muy débil. El modelo funcionará pero con métricas modestas.

### Paso 2: Configuración del modelo

Tres hiperparámetros:

**`n_components`** (slider 2-10): número de gaussianas. Default: el sugerido por BIC.

**`covariance_type`** (selectbox): forma de las gaussianas. Default: `full`.

Cuándo cambiar el default:
- Cambiar a `diag` si hay pocos datos (menos de 50) y el modelo con `full` da métricas raras.
- Cambiar a `tied` si hay muchos clusters y pocos datos, para reducir parámetros a estimar.
- Cambiar a `spherical` casi nunca; solo si se quiere replicar comportamiento tipo K-Means para comparar.

**`random_state`** (input numérico): semilla aleatoria. Default: 42.

Fija la inicialización del algoritmo EM. Con la misma semilla siempre se obtiene el mismo modelo. Para verificar estabilidad, se puede entrenar con distintas semillas y comparar métricas.

### Paso 3: Entrenamiento — Qué pasa internamente

Al presionar el botón:

1. Los datos filtrados se pasan a StandardScaler (`fit_transform`).
2. Se instancia `GaussianMixture` de scikit-learn con los hiperparámetros elegidos.
3. Se ejecuta `.fit(X_scaled)` que corre internamente el algoritmo EM hasta convergencia.
4. Se calculan etiquetas con `.predict()` y probabilidades con `.predict_proba()`.
5. Se calculan las métricas de evaluación.
6. Todo queda en `st.session_state` para uso en las páginas siguientes.

### Vista rápida después de entrenar — Cómo leerla

Cuatro métricas inmediatas más una interpretación cualitativa del silhouette.

**Cómo leer la vista rápida en conjunto**:

**Modelo bien**: silhouette moderado (>0.3), Davies-Bouldin razonable (<1.5), BIC en la zona baja de la curva.

**Modelo con problemas**: silhouette bajo (<0.15) y Davies-Bouldin alto (>2). Los clusters no están bien definidos, considerar cambiar hiperparámetros.

**Sobreajuste probable**: silhouette alto pero muchos clusters con pocas personas cada uno. El modelo aprendió el ruido, no la estructura.

**Ejemplo de la captura**:
- 8 clusters, silhouette=0.224, Davies-Bouldin=1.437, BIC=2980.
- Interpretación: silhouette bajo pero interpretable en contexto de personalidad. Davies-Bouldin aceptable. BIC moderado.
- Conclusión: modelo con estructura difusa pero utilizable. Coherente con lo esperado en datos reales de personalidad.

---

## Resultados

Página más rica en información. Aquí se hacen las interpretaciones que sostienen la exposición.

### Sección: Métricas de evaluación — Interpretación conjunta

Cinco métricas visibles:

**Silhouette** (con interpretación cualitativa)

Rango -1 a 1.

- **> 0.7**: estructura fuerte. Clusters bien separados y compactos.
- **0.5 – 0.7**: estructura razonable.
- **0.25 – 0.5**: estructura débil. **Típico y esperado en personalidad**.
- **< 0.25**: sin estructura clara.

**Davies-Bouldin**

Rango 0 a infinito.

- **< 1**: clusters bien definidos.
- **1 – 2**: aceptable, clusters existen pero se traslapan.
- **> 2**: clusters muy traslapados o mal definidos.

**Calinski-Harabasz**

Rango 0 a infinito, mayor es mejor. No tiene umbrales absolutos, solo sirve para comparar modelos sobre los mismos datos.

**Outliers**

Cantidad de puntos marcados como ruido. En GMM siempre es 0 (todos los puntos tienen probabilidades de pertenencia).

**BIC final**

Valor absoluto del modelo entrenado. Menor es mejor. Solo sirve para comparar con otros modelos sobre los mismos datos.

**Cómo se leen en conjunto**:

Modelo A: silhouette=0.42, Davies-Bouldin=1.05, BIC=2100.
Modelo B: silhouette=0.35, Davies-Bouldin=1.20, BIC=2050.

¿Cuál es mejor? A es mejor en silhouette y Davies-Bouldin, B es mejor en BIC. La decisión depende del contexto:
- Si priorizas separación entre clusters: A.
- Si priorizas parsimonia (menos parámetros para similar ajuste): B.
- En personalidad, generalmente A porque la separación es lo escaso.

### Sección: Proyección PCA 2D — Cómo interpretarla

**Qué es PCA**

Técnica que proyecta datos de alta dimensión a un espacio de menor dimensión preservando la mayor varianza posible.

Matemáticamente: PCA encuentra los autovectores de la matriz de covarianza con mayor autovalor. Esos autovectores son las direcciones (componentes principales) donde los datos varían más.

Los dos primeros componentes principales son las dos direcciones con mayor varianza. PC1 tiene más varianza que PC2 por construcción.

**Varianza explicada**

Cada componente explica un porcentaje de la varianza total.

Ejemplo típico:
- PC1: 36.4%
- PC2: 25.0%
- Total: 61.4%

Significa que la proyección 2D preserva 61.4% de la información. El resto (38.6%) se pierde al comprimir de 5D a 2D.

**Cómo leer el porcentaje**:
- **Total > 70%**: proyección muy fiel. La vista 2D representa bien los datos.
- **Total 50 – 70%**: proyección razonable. Suficiente para explorar visualmente.
- **Total < 50%**: proyección pobre. Se pierde información importante, interpretar con cuidado.

**Cómo leer el scatter plot**

Cada punto es una persona, coloreado por cluster asignado.

Patrones a buscar:

**Patrón 1: Clusters claramente separados**

Grupos de colores distintos en zonas distintas del plano, con espacio entre ellos. Indica que los clusters son distinguibles incluso en solo 2 dimensiones. Es la mejor señal.

**Patrón 2: Clusters solapados**

Colores mezclados en las zonas de contacto. Es lo típico en datos de personalidad. GMM asigna probabilidades altas al cluster central pero las probabilidades bajas revelan la ambigüedad.

**Patrón 3: Un cluster gigante y otros pequeños**

Un color domina toda la gráfica y los otros aparecen como manchitas laterales. Puede indicar:
- Un cluster mayoritario "genérico" y otros "especializados".
- k está sobreestimado y GMM inventó clusters pequeños.

**Patrón 4: Sin estructura visible**

Todos los colores mezclados sin patrón. Los datos no se separan en 2D. Puede ser porque:
- La estructura real usa dimensiones 3, 4 o 5.
- No hay estructura real de clusters.

**Nota crítica**: el clustering se hizo sobre 5 dimensiones. Si en 2D no se ven separados los clusters, no significa que el modelo esté mal. Solo significa que las direcciones importantes no coinciden con PC1 y PC2. La estructura puede estar en las otras 3 dimensiones que perdimos al comprimir.

### Sección: Perfil promedio por cluster — Cómo leer la tabla

Tabla con:
- Número de cluster.
- Media OCEAN de cada dimensión.
- n: cantidad de personas.
- Interpretación textual automática.

**Cómo se calcula el perfil**

Para cada cluster i y dimensión d:
```
perfil[i][d] = promedio de d en las personas asignadas al cluster i
```

Corresponde al vector μ del cluster en el modelo GMM entrenado.

**Cómo se genera la interpretación textual**

Algoritmo en 5 pasos:

1. Se calcula la media global de cada dimensión sobre todo el dataset.
2. Para cada cluster, se calcula la diferencia:
   ```
   diferencia[d] = perfil_cluster[d] - media_global[d]
   ```
3. Se ordenan las dimensiones por magnitud absoluta de la diferencia.
4. Se seleccionan las **2 dimensiones más distintivas**.
5. Se determina si están "altas" (diferencia positiva) o "bajas" (diferencia negativa) y se genera el texto.

**Por qué comparar contra la media global**

Un cluster con todos los valores en 4.0 no es distintivo si toda la muestra puntúa cerca de 4.0. Comparar con la media global identifica qué hace **único** al cluster.

**Por qué solo 2 dimensiones**

Con más dimensiones el texto sería denso y confuso. Con 2 se comunica lo esencial: las dos características más marcadas.

**Cómo se leen los perfiles reales**

Ejemplo de la captura:

| Cluster | O | C | E | A | N | n | Interpretación |
|---------|---|---|---|---|---|---|----------------|
| 0 | 1.84 | 1.97 | 3.47 | 4.22 | 3.38 | 8 | Apertura baja y Conciencia baja |
| 1 | 4.37 | 3.11 | 2.38 | 4.43 | 2.44 | 31 | Apertura alta y Amabilidad alta |
| 2 | 3.12 | 3.49 | 2.17 | 3.23 | 4.01 | 21 | Extraversión baja y Neuroticismo alta |
| 3 | 3.3 | 2.38 | 4.28 | 3.12 | 4.29 | 44 | Neuroticismo alta y Extraversión alta |
| 4 | 2.6 | 4.38 | 3.38 | 3.6 | 2.06 | 42 | Conciencia alta y Neuroticismo baja |
| 5 | 1.97 | 3.34 | 3.56 | 2.09 | 2.16 | 8 | Apertura baja y Amabilidad baja |
| 6 | 4.11 | 4.08 | 2.85 | 3.41 | 3.24 | 32 | Conciencia alta y Apertura alta |
| 7 | 4.23 | 3.25 | 3.34 | 3.81 | 2.13 | 25 | Neuroticismo baja y Apertura alta |

**Interpretación psicológica de los clusters más grandes**:

**Cluster 3 (n=44)**: personas emocionalmente reactivas pero socialmente activas. Perfil coherente: extravertidos intensos que expresan sus emociones. Podríamos llamarlo "El Expresivo Intenso".

**Cluster 4 (n=42)**: personas organizadas y estables. Alta responsabilidad, baja ansiedad. Perfil coherente: "El Ejecutivo Sereno".

**Cluster 6 (n=32)**: personas creativas y disciplinadas. Combinación potente: apertura para nuevas ideas más conciencia para ejecutarlas. Perfil: "El Innovador Metódico".

**Cluster 1 (n=31)**: personas creativas y empáticas pero introvertidas. Perfil: "El Idealista Reservado".

**Clusters pequeños**:

**Cluster 0 y 5 (n=8 cada uno)**: perfiles atípicos con múltiples dimensiones bajas. Pueden ser:
- Personas que respondieron por inercia con valores bajos.
- Perfiles genuinamente atípicos en la muestra.

Vale la pena revisar sus respuestas crudas para descartar respuestas ruidosas.

**Cómo defender el análisis en la exposición**:

1. Enseñar la tabla.
2. Señalar los clusters grandes con perfil claro.
3. Dar la interpretación psicológica de al menos dos.
4. Reconocer los clusters pequeños como atípicos que valdría la pena estudiar.

### Sección: Probabilidades por cluster (exclusivo GMM) — Cómo interpretarla

Es la sección técnicamente más rica y el diferenciador de GMM.

**Qué muestra la tabla**

Columnas: edad, género, cluster asignado, probabilidad máxima, y una columna por cada cluster con la probabilidad P(Ci).

**Métrica global: probabilidad máxima promedio**

Indica cuán "seguro" está el modelo en promedio.

Rangos:
- **> 0.85**: modelo muy seguro. Mayoría de asignaciones claras.
- **0.70 – 0.85**: modelo confiable. Algunas personas fronterizas.
- **0.55 – 0.70**: mucho perfil mixto. El modelo tiene dudas frecuentes.
- **< 0.55**: modelo con mucha incertidumbre. Reconsiderar hiperparámetros.

En la captura: 0.852. Modelo confiable en promedio.

**Cómo leer una fila individual**

**Persona clara** (una probabilidad domina):
```
Cluster asignado: 4
P(C0)=0.00, P(C1)=0.00, P(C2)=0.00, P(C3)=0.00, P(C4)=0.87, P(C5)=0.10, P(C6)=0.03
Prob. máxima: 0.87
```
Interpretación: encaja claramente en el cluster 4. Perfil prototípico.

**Persona fronteriza** (dos probabilidades similares):
```
Cluster asignado: 2 (con probabilidad 0.42)
P(C0)=0.00, P(C1)=0.02, P(C2)=0.42, P(C3)=0.38, P(C4)=0.15, P(C5)=0.03
Prob. máxima: 0.42
```
Interpretación: perfil híbrido entre cluster 2 y 3. GMM lo asigna al 2 por ser ligeramente mayor, pero reconoce la incertidumbre. En K-Means esta información se perdería completamente.

**Persona muy dispersa**:
```
Cluster asignado: 1 (con probabilidad 0.28)
Probabilidades: 0.28, 0.22, 0.20, 0.15, 0.10, 0.05
Prob. máxima: 0.28
```
Interpretación: no encaja claramente en ningún cluster. Puede ser un perfil genuinamente atípico o una respuesta ruidosa. Vale la pena revisar sus datos crudos.

**Filtro: mostrar solo personas fronterizas (prob_maxima < 0.7)**

Al activarlo, se ve solo los casos donde el modelo tiene dudas. **Estas son las personas más interesantes psicológicamente**.

En la captura hay varias personas con prob_max entre 0.323 y 0.506. Son perfiles claramente mixtos.

**Ejemplo de la captura**:
```
Edad 24, Masculino, cluster=1, prob_max=0.466
P(C1)=0.47, P(C6)=0.28
```
Interpretación: persona que está casi al 50-50 entre el cluster 1 (Apertura alta y Amabilidad alta) y el cluster 6 (Conciencia alta y Apertura alta). Comparte el rasgo de Apertura alta con ambos, pero varía en si su rasgo secundario es Amabilidad o Conciencia. **Es exactamente el tipo de perfil híbrido que GMM captura y otros algoritmos borrarían**.

**Cómo defender esta sección en la exposición**:

1. Explicar qué es la probabilidad máxima y qué significa.
2. Mostrar el filtro de personas fronterizas.
3. Elegir un caso ejemplar y explicarlo con detalle: qué probabilidades tiene, entre qué clusters oscila, cómo se interpreta.
4. Enfatizar: "esta información es única de GMM. Ningún otro algoritmo del clustering estándar la entrega."

---

## Modelos

**Qué es**

Historial de modelos entrenados y guardados. Cada entrada es un experimento reproducible.

**Cómo se guarda un modelo**

Al presionar "Guardar en historial":

1. Se serializan tres objetos como archivos `.pkl` con `joblib`:
   - El modelo GMM entrenado.
   - El StandardScaler ajustado.
   - El modelo PCA ajustado.
2. Se guardan en `models/` con nombres tipo `gmm_YYYYMMDD_HHMMSS.pkl`.
3. Se inserta un documento en la colección `models` de MongoDB con metadatos.

**Por qué tres archivos y no uno**

Sin el **scaler**: al clasificar datos nuevos no podríamos aplicar la misma transformación. Recalcular scaler con datos distintos daría escalas inconsistentes y predicciones sin sentido.

Sin el **PCA**: cada visualización daría un plano distinto. La proyección depende de los datos de entrenamiento; guardarla asegura visualizaciones comparables.

Sin el **modelo**: obviamente no hay predicción posible.

**Qué se muestra en el historial**

Por cada modelo:
- Nombre, algoritmo, clusters, silhouette, fecha.
- Al expandir: hiperparámetros completos, todas las métricas, registros usados, tiempo de entrenamiento, archivo origen.

**Cómo interpretar el ejemplo de la captura**

Modelo1 con GMM, 8 clusters, silhouette=0.185, 39 registros.

- **8 clusters con 39 registros = promedio de 5 personas por cluster**. Poblaciones muy pequeñas, algunos clusters pueden ser poco confiables.
- **Silhouette 0.185**: baja separación. Aceptable si es un dataset pequeño exploratorio.
- **covariance_type=tied** en lugar de full: decisión probable porque con 39 datos, `full` sería inestable.
- **Davies-Bouldin=1.045**: dentro del rango aceptable.
- **Tiempo=0.014 segundos**: entrenamiento rápido, consistente con muestra pequeña.

**Cómo comparar modelos entre sí**

Con varios modelos guardados:
- Ordenar por silhouette descendente para ver los mejores en separación.
- Comparar BIC para elegir el modelo con mejor balance ajuste-complejidad.
- Comparar cantidad de clusters: modelos con más clusters son más granulares pero más sensibles a ruido.
- Comparar dataset origen: modelos entrenados con dataset Sintético suelen tener mejores métricas que Real (estructura más limpia).

**Operaciones disponibles por modelo**

- **Clasificar con este**: lleva a la página de Clasificación con este modelo preseleccionado.
- **Eliminar**: borra el documento de MongoDB y los tres archivos `.pkl` del filesystem.

---

## Clasificación

**Qué es**

Aplicar un modelo previamente entrenado a datos nuevos.

**Diferencia fundamental entre entrenar y clasificar**

Entrenar: el modelo aprende. Descubre parámetros de gaussianas, calcula medias, covarianzas, pesos.

Clasificar: el modelo ya entrenado aplica lo aprendido. No aprende nada nuevo, solo calcula probabilidades para los datos nuevos.

### Paso 1: Seleccionar modelo — Cómo elegir el adecuado

Dropdown con modelos del historial. Al seleccionar uno se muestra su ficha:

Elementos visibles: nombre, dataset origen, hiperparámetros (badges con GMM, k, n, cov), silhouette, Davies-Bouldin, registros con los que se entrenó.

**Cómo elegir**:

- **Para clasificar respuestas nuevas de personas reales**: usar el modelo entrenado con el dataset Real. Es el que mejor generaliza a datos genuinos.
- **Para probar generalización**: usar un modelo entrenado con Real, y clasificar con Sintético o Demo. Si funciona bien, el modelo generaliza.
- **Para experimentar**: cualquier modelo. Si el modelo tuvo silhouette bajo, esperar predicciones con mucha incertidumbre.

**Regla práctica**: modelo con silhouette < 0.15 dará clasificaciones muy inciertas. Preferir uno con silhouette > 0.25.

### Paso 2: Datos a clasificar — Diferencia entre las 4 opciones

**Opción 1: Cargar CSV nuevo**

Subir un archivo externo con respuestas Big Five.

**Uso típico**:
- Datos recolectados después del entrenamiento.
- Respuestas de otro grupo (compañeros, familia).
- Datos preparados manualmente para probar un caso específico.

**Requisitos**: CSV con las 5 dimensiones OCEAN o con q1..q20 para calcular.

**Opción 2: Dataset de ejemplo**

Usar uno de los tres datasets base (Real, Sintético, Demo).

**Uso típico**:
- Probar cómo se comporta un modelo entrenado con Real clasificando el Sintético.
- Ver cómo cambian las predicciones al usar datos con estructura distinta.

**Cuidado**: si eliges el mismo dataset con el que se entrenó el modelo, en realidad estás pidiéndole que clasifique datos que ya conoce. Interesante para verificar consistencia, no para probar generalización.

**Opción 3: Generar demo**

Llamar al generador de datos sintéticos al momento y crear un batch nuevo.

**Uso típico**:
- Crear datos completamente frescos en la exposición.
- Demostrar que el modelo funciona con datos generados con semilla distinta.
- Mostrar reproducibilidad: misma semilla → mismos datos, distinta semilla → datos distintos con misma estructura.

**Cómo funciona internamente**: usa `src/data/synthetic_generator.py` con parámetros configurables (cantidad de registros, seed).

**Opción 4: Usar dataset actual**

Clasificar el mismo dataset que está cargado en la sesión.

**Uso típico**:
- Ver cómo un modelo antiguo del historial clasifica los datos actuales.
- Comparar la asignación de un modelo antiguo vs el que acaba de entrenarse.

**Por qué existen las 4 opciones**

Cada una cubre un caso de uso distinto:
- CSV nuevo → mundo real, datos externos.
- Dataset de ejemplo → pruebas cruzadas entre los tres datasets.
- Generar demo → reproducibilidad bajo demanda.
- Dataset actual → conveniencia y comparación directa.

### Paso 3: Ejecutar clasificación — Qué pasa internamente

1. Se cargan los tres archivos `.pkl` desde el filesystem (modelo, scaler, PCA).
2. A los datos nuevos se les aplica `scaler.transform(X_new)`. **No `fit_transform`**.
3. Se ejecuta `model.predict(X_scaled)` para etiquetas duras.
4. Se ejecuta `model.predict_proba(X_scaled)` para probabilidades.
5. Se genera visualización PCA usando el PCA guardado.

**Diferencia crítica entre `fit_transform` y `transform`**

`fit_transform`: calcula parámetros (media, desviación) desde cero con los datos actuales y aplica.

`transform`: aplica los parámetros ya calculados (los del entrenamiento).

Al clasificar hay que usar `transform`. Si se usara `fit_transform`, los datos nuevos estarían en una escala calculada solo con ellos mismos, distinta a la del entrenamiento. El modelo esperaría valores centrados y escalados según los datos de entrenamiento, y recibiría valores centrados según los datos nuevos. Las predicciones serían inconsistentes.

### Cómo interpretar los resultados de clasificación

**Distribución de asignaciones**

Cuántas personas cayeron en cada cluster.

**Casos posibles**:

**Distribución similar a la del entrenamiento**: el modelo generaliza bien. Los perfiles nuevos son similares a los originales.

**Distribución muy distinta**: los datos nuevos tienen una composición diferente. Puede ser:
- La muestra nueva es sesgada (mayoría de un tipo de perfil).
- Los datos nuevos vienen de una población distinta.

**Un cluster con cero asignaciones**: ese perfil no existe en los datos nuevos. Interesante si es esperado (ese perfil es raro) o preocupante si sugiere sesgo.

**Distribución de probabilidades máximas**

**Probabilidad máxima promedio alta (>0.8)**: el modelo asigna con seguridad. Los datos nuevos encajan en la estructura aprendida.

**Probabilidad máxima promedio media (0.6 – 0.8)**: asignaciones razonables pero con casos fronterizos frecuentes. Los datos nuevos son similares pero no idénticos a los de entrenamiento.

**Probabilidad máxima promedio baja (<0.6)**: mucha incertidumbre. Los datos nuevos podrían ser muy distintos a los de entrenamiento, o el modelo no era muy discriminante.

**Personas fronterizas**

Cantidad de personas con probabilidad máxima < 0.7.

**Muchas personas fronterizas**: los datos nuevos tienen perfiles ambiguos o mixtos. Interpretación válida en personalidad (perfiles genuinamente híbridos).

**Pocas personas fronterizas**: los datos nuevos son claros y encajan bien en la estructura aprendida.

**Visualización PCA en 2D**

Se muestra el scatter de los datos nuevos proyectados en el PCA guardado (no un PCA nuevo).

**Cómo leer**:

**Puntos concentrados en las mismas zonas que los clusters del entrenamiento**: buena generalización. Los datos nuevos ocupan las mismas regiones del espacio.

**Puntos en zonas vacías del entrenamiento**: los datos nuevos son atípicos respecto a los originales. GMM les asigna un cluster con probabilidad baja.

**Puntos completamente separados del área de entrenamiento**: los datos nuevos son muy distintos. El modelo puede no ser adecuado para ellos.

### Descarga de resultados

Botón para descargar CSV con las predicciones. Incluye:
- Datos originales (edad, género, OCEAN).
- Cluster asignado.
- Probabilidad máxima.
- Probabilidad por cada cluster.

Útil para análisis posterior en Excel o para integrar con otras herramientas.

---

## Descargas

**Qué es**

Exportación de resultados en distintos formatos.

**CSV con datos filtrados**: los registros del dataset activo después de aplicar filtros. Uso: análisis externo, integración con BI.

**CSV con etiquetas de cluster**: los datos filtrados más una columna `cluster` que indica la asignación del modelo actual. Uso: análisis externo de los grupos descubiertos.

**Reporte PDF**: documento formal con el análisis completo del modelo actual.

**Cómo se genera el PDF**

Con **ReportLab**, biblioteca de Python que genera PDF programáticamente sin dependencias externas. Se construye en memoria (`BytesIO`) y se entrega sin escribir a disco.

**Contenido del reporte**:
- Portada con fecha.
- Resumen del modelo: algoritmo, registros, tiempo.
- Hiperparámetros usados.
- Métricas de evaluación.
- Perfil promedio por cluster.
- Interpretación textual de cada cluster.

---

## Reiniciar aplicación

**Qué hace**

Limpia el estado de la sesión: dataset cargado, modelo actual, filtros, resultados, selecciones.

**Qué NO hace**

- No borra los modelos guardados.
- No borra los datasets de ejemplo.
- No modifica MongoDB.
- No modifica archivos.

**Cuándo usarlo**

Para empezar de cero, para pasar de una demo a otra durante la exposición, para recuperarse de estados extraños.

**Cuándo NO usarlo**

Después de entrenar un modelo que aún no se ha guardado. Se pierde el modelo actual (no los guardados).

---

## Flujo natural de trabajo

El diseño refleja el proceso completo:

**1. Cargar** (Inicio) → **2. Explorar** (Exploración) → **3. Entrenar** (Entrenamiento) → **4. Evaluar** (Resultados) → **5. Persistir** (Modelos) → **6. Aplicar** (Clasificación) → **7. Exportar** (Descargas)

**Estado compartido entre páginas**

Todas las páginas leen del `st.session_state`. Los objetos que se comparten:
- El DataFrame filtrado (`df_filtered`).
- Los datos escalados (`X_scaled`) y el scaler.
- El modelo actualmente entrenado (`current_model`) y sus métricas.
- Los resultados calculados para descargas.

Esta arquitectura evita recalcular en cada navegación.

---

## Argumentación técnica para la exposición

**La app no es solo interfaz**

Detrás de cada pantalla hay lógica en módulos separados: `src/data/` para procesamiento, `src/clustering/` para modelos, `src/evaluation/` para métricas, `src/visualization/` para gráficas, `src/persistence/` para guardar y cargar, `src/reporting/` para el PDF. Cada módulo tiene responsabilidad única.

**Reproducibilidad total**

Con el dataset origen y los hiperparámetros guardados en MongoDB, cualquier modelo puede reentrenarse exactamente igual. `random_state` fijo garantiza que EM converja al mismo resultado.

**Separación datos-modelos**

Los datos crudos viven como CSV. Los modelos como binarios en filesystem. MongoDB solo guarda metadatos. Cada capa cumple una función específica.

**Interpretación técnica en cada resultado**

Ninguna métrica, gráfica o tabla se muestra sin contexto. Cada elemento tiene una interpretación asociada y un rango de valores esperados. Esto es lo que distingue un análisis técnico de una simple visualización.
