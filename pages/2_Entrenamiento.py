import time
import streamlit as st
import numpy as np
from src.clustering import ALGORITHMS, create_model
from src.evaluation.metrics import compute_metrics, interpret_silhouette
from src.evaluation.optimization import (
    elbow_kmeans,
    bic_gmm,
    k_distances,
    hierarchical_linkage_matrix,
)
from src.data.loader import DIMENSIONS, active_source_label
from src.data.preprocessing import scale_dimensions
from src.visualization.charts import elbow_plot, k_distance_plot, dendrogram_plot

st.set_page_config(page_title="Entrenamiento", page_icon=":material/science:", layout="wide")

st.title("Entrenamiento de modelos")
st.caption(f"Selecciona un algoritmo, ajusta hiperparámetros y entrena · Fuente: **{active_source_label()}**")

# ---------- Verificar datos ----------
if "df_filtered" not in st.session_state or st.session_state["df_filtered"].empty:
    st.warning(
        "Primero visita la página **Exploración** y aplica filtros "
        "(aunque sea sin cambios) para cargar los datos."
    )
    st.stop()

df = st.session_state["df_filtered"]
st.info(f"Trabajando con **{len(df)}** registros filtrados.")

# Escalar
X_scaled, scaler = scale_dimensions(df)
st.session_state["X_scaled"] = X_scaled
st.session_state["scaler"] = scaler

st.write("")

# ---------- Selector de algoritmo ----------
st.markdown("### 1. Selecciona un algoritmo")

algo_key = st.radio(
    "Algoritmo",
    options=list(ALGORITHMS.keys()),
    format_func=lambda k: ALGORITHMS[k]["label"],
    horizontal=True,
    label_visibility="collapsed",
)

st.caption(ALGORITHMS[algo_key]["description"])

st.write("")
st.divider()

# ---------- Optimización ----------
st.markdown("### 2. Optimización de hiperparámetros")
st.caption("Antes de entrenar, revisa esta gráfica para elegir mejor los hiperparámetros.")

if algo_key == "kmeans":
    with st.spinner("Calculando método del codo..."):
        opt = elbow_kmeans(X_scaled)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            elbow_plot(opt["k"], opt["inertia"], opt["suggested_k"], "Método del codo (WCSS)", "Inercia"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            elbow_plot(opt["k"], opt["silhouette"], None, "Silhouette por k", "Silhouette"),
            use_container_width=True,
        )
    st.success(f"**k sugerido**: {opt['suggested_k']}")

elif algo_key == "hierarchical":
    linkage_method = st.selectbox("Método de enlace para el dendrograma", ["ward", "complete", "average", "single"])
    with st.spinner("Calculando dendrograma..."):
        Z = hierarchical_linkage_matrix(X_scaled, linkage_method=linkage_method)
    fig_dendro = dendrogram_plot(Z, f"Dendrograma · linkage={linkage_method}")
    st.pyplot(fig_dendro, use_container_width=True)
    st.caption("Los saltos verticales grandes en el dendrograma sugieren buenos puntos de corte.")

elif algo_key == "dbscan":
    min_samples_opt = st.slider("min_samples para gráfica", 3, 15, 5, key="dbscan_ms_opt")
    with st.spinner("Calculando k-distancias..."):
        opt = k_distances(X_scaled, k=min_samples_opt)
    st.plotly_chart(
        k_distance_plot(
            opt["distances"], opt["k"], opt["suggested_eps"],
            eps_aggressive=opt.get("eps_aggressive"),
            eps_conservative=opt.get("eps_conservative"),
        ),
        use_container_width=True,
    )
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("eps agresivo (p50)", opt["eps_aggressive"],
                 help="Clusters más chicos y más puntos como ruido.")
    col_b.metric("eps moderado (p75)", opt["eps_moderate"],
                 help="Recomendación principal. Balance entre estructura y ruido.")
    col_c.metric("eps conservador (p90)", opt["eps_conservative"],
                 help="Menos clusters, casi sin ruido. Puede colapsar todo en uno.")
    if opt.get("eps_knee") is not None:
        st.caption(f"Codo detectado por Kneedle: **{opt['eps_knee']}** (puede coincidir con alguno de los anteriores).")
    st.success(
        f"**eps sugerido**: {opt['suggested_eps']} · "
        "prueba primero el moderado, y si te sale 1 solo cluster baja al agresivo."
    )

elif algo_key == "gmm":
    cov_type_opt = st.selectbox("Tipo de covarianza", ["full", "tied", "diag", "spherical"], key="gmm_cov_opt")
    with st.spinner("Calculando BIC/AIC..."):
        opt = bic_gmm(X_scaled, covariance_type=cov_type_opt)
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
    st.success(f"**k sugerido (menor BIC)**: {opt['suggested_k']}")

st.write("")
st.divider()

# ---------- Hiperparámetros ----------
st.markdown("### 3. Configura los hiperparámetros")

params = {}

if algo_key == "kmeans":
    col1, col2, col3 = st.columns(3)
    params["n_clusters"] = col1.slider("Número de clusters (k)", 2, 10, 3)
    params["init"] = col2.selectbox("Inicialización", ["k-means++", "random"])
    params["n_init"] = col3.slider("Reintentos (n_init)", 1, 20, 10)

elif algo_key == "hierarchical":
    col1, col2 = st.columns(2)
    params["n_clusters"] = col1.slider("Número de clusters", 2, 10, 3)
    params["linkage"] = col2.selectbox("Criterio de enlace", ["ward", "complete", "average", "single"])

elif algo_key == "dbscan":
    col1, col2 = st.columns(2)
    params["eps"] = col1.slider("eps (radio de vecindad)", 0.1, 3.0, 0.5, step=0.05)
    params["min_samples"] = col2.slider("min_samples", 2, 20, 5)

elif algo_key == "gmm":
    col1, col2 = st.columns(2)
    params["n_components"] = col1.slider("Número de componentes", 2, 10, 3)
    params["covariance_type"] = col2.selectbox("Tipo de covarianza", ["full", "tied", "diag", "spherical"])

st.write("")

# ---------- Entrenar ----------
if st.button("Entrenar modelo", type="primary", use_container_width=True):
    with st.spinner("Entrenando..."):
        start = time.perf_counter()
        model = create_model(algo_key, **params)
        model.fit(X_scaled)
        elapsed = time.perf_counter() - start

    metrics = compute_metrics(X_scaled, model.labels)

    # Guardar en session state para la página de Resultados
    st.session_state["current_model"] = model
    st.session_state["current_metrics"] = metrics
    st.session_state["current_algorithm"] = algo_key
    st.session_state["current_params"] = params
    st.session_state["current_training_time"] = round(elapsed, 3)

    st.success(f"Modelo entrenado en **{elapsed:.2f} s**")

    # Preview rápido de métricas
    st.markdown("#### Vista rápida")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clusters detectados", metrics["n_clusters"])
    col2.metric("Silhouette", f"{metrics['silhouette']:.3f}" if metrics["silhouette"] is not None else "N/A")
    col3.metric("Davies-Bouldin", f"{metrics['davies_bouldin']:.3f}" if metrics["davies_bouldin"] is not None else "N/A")
    col4.metric("Outliers", metrics["n_outliers"])

    if metrics["silhouette"] is not None:
        st.info(f"Interpretación silhouette: **{interpret_silhouette(metrics['silhouette'])}**")
    else:
        st.warning(
            "El silhouette no se puede calcular con menos de 2 clusters. "
            "Ajusta los hiperparámetros: en DBSCAN prueba bajar `eps`; "
            "en el resto sube `k` a 2 o más."
        )

    st.write("")
    st.markdown("Ve a la página **Resultados** para el análisis completo.")
