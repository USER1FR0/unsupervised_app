import time
import streamlit as st

from src.clustering import create_model
from src.evaluation.metrics import compute_metrics, interpret_silhouette
from src.evaluation.optimization import bic_gmm
from src.data.preprocessing import scale_dimensions
from src.visualization.charts import elbow_plot
from src.ui.theme import apply_global_style, render_sidebar, section_head

st.set_page_config(page_title="Entrenamiento", layout="wide")
apply_global_style()
render_sidebar()

st.markdown("## Entrenamiento del modelo")
st.caption(
    "Modelo de Mezcla Gaussiana (GMM). Optimizacion con BIC, configuracion y entrenamiento."
)
st.divider()

# ---------- Requisitos ----------
if "df_filtered" not in st.session_state or st.session_state["df_filtered"].empty:
    st.warning(
        "Primero visita **Exploracion** y aplica los filtros deseados "
        "(aunque sea sin cambios) para cargar los datos."
    )
    st.page_link("pages/1_Exploracion.py", label="Ir a Exploracion")
    st.stop()

df = st.session_state["df_filtered"]

# Validaciones minimas para GMM
if len(df) < 10:
    st.error(
        f"Solo hay {len(df)} registros filtrados. GMM necesita al menos 10 "
        "para producir clusters significativos. Ajusta los filtros en Exploracion."
    )
    st.page_link("pages/1_Exploracion.py", label="Ir a Exploracion")
    st.stop()

col_i1, col_i2 = st.columns([1, 2])
col_i1.metric(
    "Registros a usar", len(df),
    help="Cantidad de registros del dataset filtrado que se usaran para entrenar el modelo.",
)
col_i2.info(
    "GMM asume que los datos provienen de una mezcla de distribuciones gaussianas. "
    "Entrega asignaciones **probabilisticas**, ideal para detectar perfiles fronterizos."
)

try:
    X_scaled, scaler = scale_dimensions(df)
except Exception as e:
    st.error(f"Error al escalar los datos: {e}")
    st.stop()
st.session_state["X_scaled"] = X_scaled
st.session_state["scaler"] = scaler

st.write("")

# ---------- 1. Optimizacion ----------
section_head(
    title="Optimizacion de hiperparametros",
    subtitle="BIC y AIC para k = 2..10. El menor BIC sugiere el numero optimo de componentes.",
    kicker="Paso 1",
)

try:
    with st.spinner("Calculando BIC/AIC..."):
        opt = bic_gmm(X_scaled, covariance_type="full")
except Exception as e:
    st.error(f"Error en la optimizacion BIC/AIC: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(
        elbow_plot(opt["k"], opt["bic"], opt["suggested_k"], "BIC por k", "BIC"),
        use_container_width=True,
    )
with col2:
    st.plotly_chart(
        elbow_plot(opt["k"], opt["aic"], None, "AIC por k", "AIC"),
        use_container_width=True,
    )

st.markdown(
    f'<div class="card-accent">'
    f'<strong>k sugerido (menor BIC): {opt["suggested_k"]}</strong> · Usa este valor '
    f'como punto de partida en la configuracion.'
    f'</div>',
    unsafe_allow_html=True,
)

st.write("")

# ---------- 2. Configuracion ----------
section_head(title="Configuracion del modelo", kicker="Paso 2")

col_a, col_b, col_c = st.columns(3)
n_components = col_a.slider(
    "n_components",
    2, 10, int(opt["suggested_k"]),
    help=(
        "Numero de gaussianas que ajustara el modelo, equivalente al numero de clusters. "
        "Default: el k con menor BIC. "
        "k pequeno (2-3): clusters grandes y genericos. "
        "k moderado (4-6): balance entre generalizacion y detalle. "
        "k grande (7-10): clusters especificos pero con riesgo de sobreajuste; "
        "algunas gaussianas pueden quedar sin puntos asignados."
    ),
)
covariance_type = col_b.selectbox(
    "covariance_type",
    ["full", "tied", "diag", "spherical"],
    index=0,
    help=(
        "Forma de las elipses gaussianas de cada cluster. "
        "full: cada cluster con su matriz completa, elipses libres en cualquier orientacion (mas flexible). "
        "tied: todos comparten la misma matriz, misma forma en distintas ubicaciones. "
        "diag: matrices diagonales, elipses alineadas con los ejes. "
        "spherical: esferas, similar a K-Means. "
        "Recomendado full con datos suficientes."
    ),
)
random_state = col_c.number_input(
    "random_state", value=42, step=1,
    help=(
        "Semilla del generador aleatorio. Fija la inicializacion del algoritmo EM "
        "para garantizar reproducibilidad total: mismo dataset + mismos hiperparametros "
        "+ misma semilla = mismo modelo. Cualquier entero sirve. 42 es convencion."
    ),
)

st.write("")

# ---------- 3. Entrenar ----------
section_head(title="Entrenamiento", kicker="Paso 3")

if st.button(
    "Entrenar GMM", type="primary", use_container_width=True,
    help=(
        "Aplica StandardScaler, ejecuta el algoritmo Expectation-Maximization hasta convergencia, "
        "y calcula las metricas de evaluacion (Silhouette, Davies-Bouldin, Calinski-Harabasz, BIC)."
    ),
):
    if n_components >= len(df):
        st.error(
            f"n_components ({n_components}) debe ser menor al numero de registros "
            f"({len(df)}). Reduce n_components o aumenta los datos."
        )
        st.stop()

    params = {
        "n_components": n_components,
        "covariance_type": covariance_type,
        "random_state": int(random_state),
    }
    try:
        with st.spinner("Entrenando..."):
            start = time.perf_counter()
            model = create_model("gmm", **params)
            model.fit(X_scaled)
            elapsed = time.perf_counter() - start
        metrics = compute_metrics(X_scaled, model.labels)
    except Exception as e:
        st.error(f"Error durante el entrenamiento: {e}")
        st.stop()

    st.session_state["current_model"] = model
    st.session_state["current_metrics"] = metrics
    st.session_state["current_algorithm"] = "gmm"
    st.session_state["current_params"] = params
    st.session_state["current_training_time"] = round(elapsed, 3)

    st.success(f"Modelo entrenado en **{elapsed:.2f} s**")

    st.markdown("#### Vista rapida")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Clusters", metrics["n_clusters"],
        help="Numero de clusters con al menos una persona asignada. Puede ser menor a n_components si hubo sobreajuste.",
    )
    col2.metric(
        "Silhouette",
        f"{metrics['silhouette']:.3f}" if metrics["silhouette"] is not None else "N/A",
        help="Rango -1 a 1. Mayor a 0.7 estructura fuerte. 0.5-0.7 razonable. 0.25-0.5 debil (tipico en personalidad). Menor a 0.25 sin estructura clara.",
    )
    col3.metric(
        "Davies-Bouldin",
        f"{metrics['davies_bouldin']:.3f}" if metrics["davies_bouldin"] is not None else "N/A",
        help="Menor es mejor. Menor a 1 clusters bien definidos. 1-2 aceptable. Mayor a 2 muy solapados.",
    )
    col4.metric(
        "BIC final",
        f"{model.get_bic(X_scaled):.1f}",
        help="BIC del modelo entrenado. Se compara contra otros modelos guardados. Menor es mejor.",
    )

    if metrics["silhouette"] is not None:
        st.info(f"Interpretacion silhouette: **{interpret_silhouette(metrics['silhouette'])}**")

    st.write("")
    st.page_link("pages/3_Resultados.py", label="Ver analisis completo en Resultados")
