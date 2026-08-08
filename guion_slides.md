# Guión de exposición — 15 slides con profundidad técnica

Estudio dirigido para exposición de 20-40 minutos. Cada slide con detalle técnico defendible, conceptos con definición precisa, y anticipación de preguntas.

---

## Slide 1 — Portada

**Qué decir**:

"Buen día. Presentamos el proyecto de la Unidad IV, análisis no supervisado de personalidad basado en el modelo Big Five. Elegimos como algoritmo el **modelo de mezcla gaussiana**, GMM por sus siglas en inglés."

"Somos Balderas, Oropeza y Rodríguez, del grupo GIDS6091."

**Contexto para tener claro**: el instrumento de evaluación pide aplicar el algoritmo no supervisado deseado, singular. Elegimos GMM con justificación teórica y empírica que iremos desarrollando.

**Transición**: "Empecemos por definir qué construimos exactamente."

---

## Slide 2 — ¿Qué construimos?

**Qué decir**:

"Construimos una aplicación web completa que descubre grupos naturales de personalidad a partir de una encuesta propia. El pipeline cubre las cinco etapas del análisis: recolección, procesamiento, modelado, evaluación y clasificación de personas nuevas."

"El concepto central: en el análisis no supervisado *(rama del machine learning que trabaja sin etiquetas previas)* **no le indicamos al modelo cuántos grupos existen ni cómo se ven**. El algoritmo debe descubrirlos usando únicamente la estructura interna de los datos."

**Diferencia con supervisado (por si preguntan)**: en supervisado tenemos pares entrada-etiqueta y el modelo aprende a predecir la etiqueta. En no supervisado solo tenemos entradas, y el objetivo es encontrar estructura latente *(patrón oculto que no está explicitado en los datos)*.

**Transición**: "Antes de entrar al algoritmo, definimos qué estamos midiendo."

---

## Slide 3 — Modelo Big Five (OCEAN)

**Qué decir**:

"Big Five, también llamado OCEAN por sus siglas, es el modelo de personalidad con mayor respaldo empírico en la psicología científica contemporánea. Fue desarrollado a partir de análisis léxico y confirmado por décadas de estudios factoriales."

*(Análisis factorial: técnica estadística que reduce muchas variables observadas a pocos factores latentes. Es lo que llevó a que las cinco dimensiones aparecieran consistentemente.)*

"El modelo describe la personalidad en cinco dimensiones **continuas e independientes**:"

- **Apertura**: creatividad e imaginación.
- **Conciencia**: organización y responsabilidad.
- **Extraversión**: sociabilidad y energía.
- **Amabilidad**: empatía y cooperación.
- **Neuroticismo**: reactividad emocional.

"Cada persona se convierte en un vector de cinco números entre 1 y 5. Ese vector es el input directo del clustering."

"Elegimos Big Five sobre otros modelos como MBTI o DISC porque cumple tres condiciones necesarias para clustering: variables numéricas *(no categóricas)*, dimensiones estadísticamente independientes *(sin redundancia que sesgue las distancias)*, e interpretabilidad psicológica de los resultados."

**Transición**: "Ahora, cómo obtenemos esos cinco números en la práctica."

---

## Slide 4 — Diseño de la encuesta

**Qué decir**:

"El instrumento consta de 20 preguntas tipo Likert de 1 a 5. Cuatro preguntas por cada dimensión OCEAN, más datos demográficos: edad, género, estado, municipio."

*(Escala Likert: escala de valoración ordinal donde el encuestado indica su grado de acuerdo con una afirmación. La usamos con cinco puntos, permitiendo respuesta neutra en el 3.)*

"El diseño incluye un elemento psicométrico clave: **una pregunta reversa por cada dimensión**."

*(Pregunta reversa: ítem redactado en sentido opuesto a la dimensión que mide. Sirve como control de calidad de respuesta.)*

"El problema que resuelve: las personas responden por inercia sin leer. Alguien que ponga todo cinco sin pensar producirá scores exagerados. La reversa detecta esa inconsistencia porque, al invertirla en el cálculo, un patrón de todo cinco produce contradicción interna que se refleja en el score final."

"Ejemplo: en Apertura, una directa dice 'me gusta el arte complejo', una reversa dice 'prefiero rutinas conocidas'. Alguien que valore alto ambas de manera literal, tras la inversión, tendrá un promedio moderado en vez de artificialmente alto."

**Transición**: "Con esas 20 respuestas se calculan las cinco dimensiones OCEAN."

---

## Slide 5 — Ejemplo de scoring + Implementación

**Qué decir**:

"La fórmula por dimensión es un promedio con inversión de reversa:"

`Score = (q1 + q2 + q3 + (6 - q_reversa)) / 4`

"La regla de inversión general es `max + min - respuesta`. Para Likert 1-5 eso es `6 - respuesta`. Convierte simétricamente: un 1 se vuelve 5, un 2 se vuelve 4, un 3 permanece 3."

"Esta simetría es importante porque preserva la escala original después de invertir. Si usáramos otra regla, la dimensión combinada tendría rangos inconsistentes."

**Sobre el código** (por si preguntan por él):

"El diccionario `ITEMS_BY_DIMENSION` define qué ítems pertenecen a cada dimensión. El set `REVERSE_ITEMS` identifica cuáles son reversos. La función itera sobre cada dimensión, aplica la inversión donde corresponde, y calcula el promedio."

"Elegimos representar los reversos como set y no como lista porque la búsqueda es O(1) en set y O(n) en lista. Detalle de eficiencia, pero muestra intencionalidad."

**Resultado**: "Cada persona se convierte en un vector `[O, C, E, A, N]` con valores continuos entre 1 y 5."

**Transición**: "A partir de ese vector generamos una etiqueta interpretable de arquetipo."

---

## Slide 6 — Asignación de arquetipo

**Qué decir**:

"El arquetipo se asigna con una **lógica determinista**, no con clustering."

*(Determinista: mismo input produce siempre el mismo output. Sin aleatoriedad ni modelo entrenado.)*

"Tres pasos:"

1. "Identificar la dimensión con score más alto (dominante)."
2. "Identificar la dimensión con score más bajo."
3. "Combinar como clave 'dominante-baja' y buscar en la matriz de 25 arquetipos."

"En el ejemplo: dominante O = 4.5, más baja N = 2.25. Clave O-N mapea a 'Visionario Sereno'."

**Punto crítico que hay que dejar claro**:

"El arquetipo asignado por esta regla **no es lo mismo que el cluster que descubre GMM**. Son dos etiquetas distintas para la misma persona."

- "El arquetipo es una etiqueta que asignamos al momento de responder la encuesta, para retroalimentación inmediata. Solo mira dos dimensiones."

- "El cluster es una etiqueta que descubre GMM analizando el dataset completo. Considera las cinco dimensiones simultáneamente y las relaciones estadísticas entre personas."

"Pueden coincidir en algunos casos pero no es una equivalencia. Analizar cómo se relacionan sería un estudio adicional."

**Transición**: "Ahora vemos la encuesta funcionando."

---

## Slide 7 — Encuesta Forms en vivo

**Qué decir mientras demuestran**:

"Este es el formulario público. Está construido con Google Apps Script *(plataforma de desarrollo que permite crear aplicaciones web servidas desde Google, sin infraestructura propia)*. Al enviar, la respuesta se guarda en Google Sheets y quedará disponible para importar al pipeline."

"Muestren rápido: dos o tres preguntas, envían, y aparece el arquetipo asignado con las probabilidades del scoring."

**Puntos técnicos para mencionar mientras demuestran**:

- El scoring corre en el navegador con JavaScript, no en servidor.
- La reversa se invierte antes de calcular el promedio.
- El arquetipo se determina con la matriz 5×5 sin llamadas externas.

**Transición**: "Con los datos capturados, entramos a la parte de análisis."

---

## Slide 8 — Estandarización + GMM: la idea central

**Qué decir sobre estandarización**:

"Antes de entrenar cualquier modelo aplicamos **StandardScaler**. La fórmula es `z = (x - μ) / σ`, resta la media y divide por la desviación estándar."

*(Estandarización: transformación que hace que una variable tenga media cero y desviación estándar uno. También llamada normalización z-score.)*

"Aunque nuestras cinco dimensiones ya están en la misma escala 1 a 5, escalamos por tres razones:"

1. "**Estabilidad numérica**: valores centrados en cero son más estables en cálculos con matrices de covarianza y en la inversión de matrices que hace GMM internamente."

2. "**Equilibrio en distancias**: en un espacio con dimensiones no estandarizadas, las que tengan mayor varianza dominan. Aunque todas están en 1-5, sus varianzas pueden ser distintas."

3. "**Portabilidad del modelo**: al guardar el scaler junto al modelo garantizamos que datos nuevos se transformen exactamente igual antes de clasificar."

**Qué decir sobre GMM**:

"GMM asume que los datos son generados por una **mezcla de k distribuciones gaussianas multivariadas** *(distribución normal en varias dimensiones, extensión de la campana de Gauss al espacio 5D)*."

"Cada cluster corresponde a una gaussiana con tres parámetros propios: peso, media y matriz de covarianza. El algoritmo estima esos parámetros y, con ellos, calcula para cada persona la probabilidad de haber sido generada por cada gaussiana."

**Diferencia clave que hay que enfatizar**:

"A diferencia de K-Means, GMM no asigna cada punto a un solo cluster. Asigna una **distribución de probabilidad**. Por eso decimos que es un algoritmo probabilístico o de clustering suave."

**Transición**: "Vamos a lo interno: qué define exactamente a una gaussiana en el modelo."

---

## Slide 9 — Los 3 parámetros + Tipos de covariance_type

**Qué decir sobre los 3 parámetros**:

"Cada gaussiana en la mezcla queda completamente definida por tres parámetros:"

- **Peso π (pi)**: proporción de la muestra que pertenece a ese cluster. Es una probabilidad a priori.
- **Media μ (mu)**: vector de cinco valores que define el centro del cluster en el espacio de dimensiones.
- **Covarianza Σ (sigma)**: matriz que describe la forma, tamaño y orientación de la elipse *(elipsoide en 5D)*.

"La restricción matemática fundamental es que los pesos suman uno: `Σ πᵢ = 1`. Toda persona pertenece a algún cluster con probabilidad positiva."

*(Matriz de covarianza: describe cómo varían conjuntamente las dimensiones. Los elementos de la diagonal son las varianzas individuales; los de fuera son las covarianzas entre pares de dimensiones.)*

**Qué decir sobre covariance_type**:

"El hiperparámetro `covariance_type` restringe la forma de la matriz de covarianza. Cuatro opciones:"

- **full**: cada cluster tiene su matriz completa. Elipses de cualquier tamaño, forma y orientación.
- **tied**: todos comparten la misma matriz. Misma forma en distintas ubicaciones.
- **diag**: matrices diagonales. Elipses alineadas con los ejes.
- **spherical**: un solo valor de varianza. Esferas.

**Justificación de nuestra elección**:

"Usamos `full` porque es el más flexible y captura la estructura real. Las opciones más restrictivas asumen que las dimensiones son independientes dentro de cada cluster, lo cual no es cierto para datos de personalidad."

"El costo es mayor número de parámetros a estimar: para 5 dimensiones y k clusters, `full` estima 15 valores por matriz *(matriz simétrica de 5x5)*. Con datos suficientes es defendible."

**Transición**: "Nos falta un parámetro más: cuántas gaussianas usar."

---

## Slide 10 — Cómo elegimos el número de clusters (BIC/AIC)

**Qué decir**:

"El número de componentes k no se elige a ojo. Usamos **criterios de información** para decidir con base estadística."

*(Criterio de información: métrica que combina bondad de ajuste con penalización por complejidad, para evitar sobreajuste al elegir modelos.)*

**BIC**:

`BIC = -2 · ln(L) + p · ln(n)`

Donde:
- L es la verosimilitud del modelo ajustado.
- p es el número de parámetros libres.
- n es el número de observaciones.

*(Verosimilitud: probabilidad de haber observado los datos dados los parámetros del modelo. En estadística se maximiza esta cantidad para encontrar los parámetros óptimos.)*

"El primer término premia el ajuste; el segundo castiga la complejidad. **Menor BIC es mejor modelo**. Con más componentes el ajuste mejora pero el castigo por complejidad crece, y en algún punto el segundo término domina."

**AIC**:

`AIC = -2 · ln(L) + 2p`

"AIC es similar pero penaliza menos. Su factor de penalización es constante `2p`, mientras que el de BIC es `p · ln(n)` que crece con el tamaño de la muestra."

"**Priorizamos BIC porque es más conservador**. Elige modelos más simples y refleja mejor la estructura real. AIC podría sugerir demasiados componentes."

**El proceso en la app**:

"Entrenamos GMM con k de 2 a 10, calculamos BIC en cada caso, y elegimos el k que lo minimiza. La app muestra ambas curvas y sugiere el k óptimo automáticamente."

**Interpretación de la curva**:

- **Curva en U**: el mínimo es claro. Es el caso ideal.
- **Curva monótonamente decreciente**: los datos toleran muchos componentes. Elegimos el k donde la mejora se vuelve marginal (regla del codo).
- **Curva plana**: no hay estructura clara. Los datos podrían no tener clusters diferenciables.

**Transición**: "Con los hiperparámetros definidos, ahora cómo aprende internamente el modelo."

---

## Slide 11 — Cómo GMM aprende los parámetros (EM)

**Qué decir**:

"GMM aprende con el algoritmo **EM: Expectation-Maximization**, formalizado por Dempster, Laird y Rubin en 1977."

*(EM: método iterativo para estimar parámetros en modelos con variables latentes. En GMM, la variable latente es a qué cluster pertenece cada punto, que no observamos directamente.)*

**Cuatro pasos**:

**1. Inicialización**

"Se eligen valores iniciales para π, μ y Σ. Scikit-learn arranca desde una asignación tipo K-Means para acelerar la convergencia."

**2. E-step (Expectation)**

"Para cada persona i y cada cluster k, se calcula la **responsabilidad**:"

`γ(i, k) = πk · N(xi | μk, Σk) / Σj πj · N(xi | μj, Σj)`

"Es la probabilidad posterior de que la persona i pertenezca al cluster k, dados los parámetros actuales. `N(x | μ, Σ)` es la densidad gaussiana multivariada evaluada en el punto."

*(Densidad gaussiana multivariada: función que asigna una densidad de probabilidad a cada punto del espacio, en forma de campana centrada en μ con dispersión Σ.)*

**3. M-step (Maximization)**

"Con esas responsabilidades como pesos, se recalculan los parámetros de cada cluster:"

- Nueva media: promedio ponderado de las personas.
- Nueva covarianza: dispersión ponderada respecto a la nueva media.
- Nuevo peso: proporción efectiva del cluster.

"Es una forma de máxima verosimilitud ponderada."

**4. Iterar hasta convergencia**

"Se repiten E y M hasta que la log-verosimilitud del modelo deje de mejorar significativamente. Típicamente entre 10 y 100 iteraciones."

**Garantía teórica**: "EM garantiza que la log-verosimilitud es no decreciente en cada iteración. Puede quedar en un óptimo local, por eso es sensible a inicialización, por eso se fija `random_state`."

**Transición**: "Con el modelo entrenado, qué produce como salida."

---

## Slide 12 — Soft clustering + Métricas

**Qué decir sobre soft clustering**:

"GMM no dice 'perteneces al cluster 2, punto'. Dice: para cada persona, `[0.68, 0.24, 0.05, 0.02, 0.01]`. Es la distribución completa de probabilidades sobre los clusters."

"En este ejemplo la persona es 68% cluster 0, 24% cluster 1, y el resto marginal. La asignación dura sería cluster 0, pero conocemos la incertidumbre."

**Por qué esto importa**:

"Los **perfiles fronterizos** son el hallazgo más valioso académicamente. Son personas con probabilidad máxima menor a 0.7. Casos donde el modelo dice explícitamente 'esta persona tiene rasgos mixtos'."

"En clustering duro (K-Means, jerárquico) esa información se pierde. GMM la preserva."

**Interpretación por rango de probabilidad máxima**:

- **Mayor a 0.9**: asignación muy segura. La persona es prototípica del cluster.
- **0.7 a 0.9**: confiable. Perfil claro pero no extremo.
- **Menor a 0.7**: persona fronteriza. Perfil mixto entre dos o más clusters.

**Qué decir sobre las métricas** (tabla del lado derecho):

"Cuatro métricas complementarias para evaluar el clustering:"

- **Silhouette**: rango -1 a 1, mayor es mejor. Mide separación relativa entre clusters.
- **Davies-Bouldin**: rango 0 a infinito, menor es mejor. Mide compacidad y separación.
- **Calinski-Harabasz**: rango 0 a infinito, mayor es mejor. Razón entre varianza inter e intra cluster.
- **BIC/AIC**: rango real, menor es mejor. Balance entre ajuste y complejidad, específicos de modelos probabilísticos.

"**Ninguna métrica sola decide**. Un modelo con silhouette alto pero BIC muy alto podría estar sobreajustado. Un modelo con silhouette bajo pero interpretación de clusters muy clara podría ser el correcto en datos difusos como personalidad. Se combinan y se contextualiza."

**Transición**: "La métrica más citada en clustering es Silhouette. Vamos a verla en detalle."

---

## Slide 13 — Silhouette

**Qué decir**:

"El coeficiente de silueta se calcula punto por punto:"

`s(i) = (b(i) - a(i)) / max(a(i), b(i))`

Donde:

- **a(i)**: distancia promedio de i a los otros puntos de su mismo cluster. Mide **cohesión intra-cluster**.
- **b(i)**: distancia promedio de i al cluster vecino más cercano (el más cercano al que no pertenece). Mide **separación inter-cluster**.

"El silhouette del modelo es el promedio de todos los `s(i)`."

**Interpretación del rango**:

- **Mayor a 0.7**: estructura fuerte. Clusters bien separados y compactos.
- **0.5 a 0.7**: estructura razonable.
- **0.25 a 0.5**: estructura débil. **Típico en datos de personalidad**.
- **Menor a 0.25**: sin estructura clara. Los datos podrían no tener clusters naturales, o el k está mal elegido.
- **Negativo**: puntos están más cerca de clusters ajenos que del propio. Modelo mal ajustado.

**Anticipación de la pregunta obligada**:

"En datos reales de personalidad **esperamos silhouette moderado o bajo**. La personalidad humana no tiene categorías rígidas: los rasgos son continuos y se distribuyen sin fronteras estrictas."

"Un silhouette de 0.9 en personalidad real sería sospechoso. Un silhouette de 0.2 a 0.4 refleja la realidad: hay estructura, pero es difusa. Es exactamente el escenario donde GMM aporta valor con las probabilidades."

**Transición**: "Con el modelo evaluado, viene la parte de persistencia y reutilización."

---

## Slide 14 — Persistencia del modelo

**Qué decir**:

"Cuando un modelo entrenado tiene valor, lo persistimos para reutilizarlo. La captura muestra un documento real de nuestra colección `models` en MongoDB."

**Filosofía de persistencia**:

"MongoDB **no guarda las respuestas de las personas**. Guarda solo metadatos de los modelos. Los datos crudos viven como CSV en el sistema de archivos."

"Elegimos MongoDB por dos razones:"

1. "Un modelo de ML tiene estructura variable *(distintos hiperparámetros según algoritmo, métricas que pueden evolucionar)*. Encaja natural en un documento JSON. Con SQL habría que normalizar en varias tablas y hacer JOIN."

2. "Consultas sobre metadatos son inmediatas. Podemos preguntar 'cuál es el mejor modelo por silhouette' sin cargar ningún archivo binario."

**Estructura del documento**:

- **Identificación**: `_id`, `model_name`, `algorithm`.
- **Timestamp**: fecha y hora en UTC.
- **Hiperparámetros**: n_components, covariance_type, random_state.
- **Métricas**: n_clusters, silhouette, Davies-Bouldin, Calinski-Harabasz.
- **Trazabilidad**: dataset origen, tiempo de entrenamiento.
- **Rutas a archivos binarios**: `model_file_path`, `scaler_file_path`, `pca_file_path`.

**Por qué guardar tres archivos y no uno**:

"El modelo por sí solo no basta. Necesitamos también el **scaler** para aplicar la misma transformación a datos nuevos, y el **PCA** para visualizar consistentemente."

"Sin el scaler, datos nuevos estarían en escala distinta y las predicciones serían inconsistentes. Sin el PCA guardado, cada visualización daría un plano distinto y no sería comparable."

**Por qué los binarios fuera de MongoDB**:

"MongoDB soporta blobs pero no está diseñado para archivos grandes. La arquitectura es: **MongoDB como catálogo, filesystem como almacén**. Cada capa hace lo que hace mejor."

**Transición**: "Cerramos con conclusiones y aportes."

---

## Slide 15 — Conclusiones

**Qué decir**:

"Los aportes del proyecto son cinco:"

**1. Pipeline completo end-to-end**

"Desde el diseño del instrumento con criterios psicométricos hasta la clasificación de personas nuevas con modelo persistido. No es un ejercicio aislado, es un sistema."

**2. Justificación teórica y empírica de GMM**

"Teórica: la personalidad no tiene categorías rígidas, requiere clustering probabilístico. Empírica: BIC como criterio de selección, silhouette y Davies-Bouldin como validación."

**3. Soft clustering como diferenciador metodológico**

"Ningún otro algoritmo del clustering estándar entrega distribución de probabilidad por cluster. Es una información que en dominios difusos como personalidad no debería descartarse."

**4. Persistencia consistente**

"Guardamos modelo, scaler y PCA como triplete. Garantiza que un modelo entrenado hace un mes sigue produciendo predicciones consistentes hoy."

**5. Tres datasets como estrategia de validación**

- **Real** para entrenar con datos auténticos.
- **Sintético** para verificar que el algoritmo recupera estructura conocida.
- **Demo** para probar generalización con datos que el modelo nunca vio.

**Cierre**:

"Como reflexión final: el proyecto demuestra que un análisis no supervisado no consiste en aplicar un algoritmo, sino en construir un pipeline donde cada decisión, desde la escala Likert hasta el tipo de covarianza, tiene justificación. Estamos abiertos a preguntas."

---

## Preguntas anticipadas y respuestas

**Por qué GMM y no K-Means**
K-Means asume clusters esféricos y hace asignación dura. GMM modela clusters elípticos y entrega probabilidades. En datos de personalidad los perfiles son mixtos: GMM captura esa realidad, K-Means la borra.

**Por qué el silhouette bajo**
En datos reales de personalidad es esperado. Los rasgos humanos son continuos, no discretos. Un silhouette moderado indica estructura difusa, coherente con el dominio. Si fuera 0.9 sospecharíamos que los datos están artificialmente separados.

**Cómo eligen k**
Con BIC. Probamos k de 2 a 10 y elegimos el que lo minimiza. BIC penaliza fuerte la complejidad, previniendo sobreajuste. Es el criterio más conservador y más adecuado para descubrir estructura real.

**Qué es el random_state**
La semilla del generador aleatorio. GMM tiene inicialización aleatoria; con la misma semilla siempre converge al mismo resultado. Permite reproducibilidad y verificación de estabilidad al probar con distintas semillas.

**Por qué guardan el scaler junto al modelo**
Para aplicar la misma transformación exacta a datos nuevos. Si recalculáramos el scaler con datos distintos, las escalas no coincidirían y las predicciones serían inconsistentes con el entrenamiento.

**Por qué covariance_type=full**
Máxima flexibilidad. Cada cluster tiene su propia forma, tamaño y orientación. Con datos suficientes es lo que mejor captura la estructura. Con muestras muy pequeñas preferiríamos `diag` para evitar sobreajustar la covarianza.

**Qué pasa si EM cae en un óptimo local**
Puede pasar porque EM garantiza no decrecimiento pero no óptimo global. Por eso fijamos random_state y podemos verificar con distintas semillas. Si los resultados son estables entre corridas, tenemos confianza en la solución.

**Cuántos datos tienen**
Alrededor de 200 respuestas reales, 500 sintéticas y 100 de demo. El sintético se genera con un script propio que produce personas coherentes con arquetipos base predefinidos y ruido gaussiano controlado.

**Cómo generan los datos sintéticos**
Cinco arquetipos base con medias OCEAN objetivo, pesos desiguales (para reflejar que en la realidad no todos los perfiles son igual de comunes), y ruido normal alrededor de esas medias. Luego se reconstruyen las 20 respuestas Likert coherentes con esos scores.

**Los clusters de GMM coinciden con los arquetipos**
No necesariamente. El arquetipo se determina con dos dimensiones extremas por regla. Los clusters de GMM consideran las cinco dimensiones y las relaciones estadísticas entre personas. Que coincidan sería casualidad; que no coincidan es esperado.

**Por qué no usan más algoritmos para comparar**
El instrumento pide aplicar el algoritmo deseado, singular. Elegimos GMM con justificación. Si se pide una comparativa formal es un ejercicio adicional.
