import streamlit as st
import pandas as pd
import numpy as np
from src.data.loader import DIMENSIONS, DIMENSION_LABELS
from src.evaluation.metrics import interpret_silhouette
from src.evaluation.interpretation import (
    compute_cluster_profile,
    build_interpretations,
    rename_profile_columns,
)
from src.visualization.pca import project_2d, variance_explained
from src.visualization.charts import pca_scatter, cluster_profile_bars
from src.clustering import ALGORITHMS

st.set_page_config(page_title="Resultados", page_icon="📈", layout="wide")

st.title("📈 Resultados del modelo")
st.caption("Métricas, visualización y perfiles por cluster")

# ---------- Verificar que hay modelo entrenado ----------
required_keys = ["current_model", "current_metrics", "df_filtered", "X_scaled"]
if not all(k in st.session_state for k in required_keys):
    st.warning("Primero entrena un modelo en la página **🧪 Entrenamiento**.")
    st.stop()

model = st.session_state["current_model"]
metrics = st.session_state["current_metrics"]
df = st.session_state["df_filtered"].reset_index(drop=True)
X_scaled = st.session_state["X_scaled"]
algo_key = st.session_state["current_algorithm"]
params = st.session_state["current_params"]
training_time = st.session_state["current_training_time"]

# ---------- Header con contexto del modelo ----------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f"### {ALGORITHMS[algo_key]['label']}")
    params_str = " · ".join(f"**{k}**: {v}" for k, v in params.items())
    st.caption(params_str)
with col_h2:
    st.metric("Tiempo de entrenamiento", f"{training_time} s")

st.write("")

# ---------- Métricas ----------
st.markdown("### Métricas de evaluación")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Clusters", metrics["n_clusters"])
col2.metric(
    "Silhouette",
    f"{metrics['silhouette']:.3f}" if metrics["silhouette"] is not None else "N/A",
    help="Rango -1 a 1. Mayor = mejor separación entre clusters.",
)
col3.metric(
    "Davies-Bouldin",
    f"{metrics['davies_bouldin']:.3f}" if metrics["davies_bouldin"] is not None else "N/A",
    help="Menor = mejor. Compara compacidad y separación.",
)
col4.metric(
    "Calinski-Harabasz",
    f"{metrics['calinski_harabasz']:.1f}" if metrics["calinski_harabasz"] is not None else "N/A",
    help="Mayor = mejor. Razón entre varianza inter e intra cluster.",
)
col5.metric("Outliers", metrics["n_outliers"])

if metrics["silhouette"] is not None:
    interp = interpret_silhouette(metrics["silhouette"])
    st.info(f"**Interpretación del silhouette**: {interp}")

# Métricas específicas
if algo_key == "kmeans" and hasattr(model, "get_inertia"):
    st.caption(f"**Inercia (WCSS)**: {model.get_inertia():.2f}")
elif algo_key == "gmm":
    st.caption(f"**BIC**: {model.get_bic(X_scaled):.2f} · **AIC**: {model.get_aic(X_scaled):.2f}")

st.write("")
st.divider()

# ---------- PCA 2D ----------
st.markdown("### Proyección en 2D (PCA)")
st.caption("Reducción de dimensionalidad para visualizar los clusters en dos ejes.")

X_2d, pca = project_2d(X_scaled)
variance = variance_explained(pca)

st.plotly_chart(
    pca_scatter(X_2d, model.labels, variance),
    use_container_width=True,
)
st.caption(
    f"Los dos componentes explican **{variance['total']*100:.1f}%** de la varianza total. "
    "Puntos cercanos son personas con perfiles similares."
)

st.write("")
st.divider()

# ---------- Perfil por cluster ----------
st.markdown("### Perfil por cluster")

profile = compute_cluster_profile(df, model.labels)
global_means = {d: float(df[d].mean()) for d in DIMENSIONS}
profile_with_interp = build_interpretations(profile, global_means)
profile_display = rename_profile_columns(profile_with_interp)

# Renombrar columnas OCEAN a nombres completos
profile_display = profile_display.rename(columns={d: DIMENSION_LABELS[d] for d in DIMENSIONS})

st.dataframe(
    profile_display,
    use_container_width=True,
    hide_index=True,
)

st.plotly_chart(
    cluster_profile_bars(profile),
    use_container_width=True,
)

st.write("")
st.divider()

# ---------- Interpretación textual ----------
st.markdown("### Interpretación de cada cluster")

for _, row in profile_with_interp.iterrows():
    cluster_id = row["cluster"]
    if cluster_id == -1:
        st.markdown(f"**🔸 Ruido** · {row['interpretación']}")
    else:
        st.markdown(f"**🔹 Cluster {int(cluster_id)}** · {row['interpretación']}")
    st.caption(
        "  ·  ".join([f"{d}={row[d]:.2f}" for d in DIMENSIONS])
    )
    st.write("")

st.divider()

# ---------- Dataset con etiquetas ----------
st.markdown("### Dataset con etiquetas de cluster")

df_labeled = df.copy()
df_labeled["cluster"] = model.labels

with st.expander("Ver dataset completo con etiquetas", expanded=False):
    st.dataframe(df_labeled, use_container_width=True, hide_index=True)

# Guardar para uso en descargas
st.session_state["df_labeled"] = df_labeled