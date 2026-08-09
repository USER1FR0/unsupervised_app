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
from src.ui.theme import apply_global_style, render_sidebar, section_head

st.set_page_config(page_title="Resultados", layout="wide")
apply_global_style()
render_sidebar()

st.markdown("## Resultados del modelo")
st.caption("Metricas, PCA, perfil por cluster e interpretacion probabilistica")
st.divider()

required_keys = ["current_model", "current_metrics", "df_filtered", "X_scaled"]
if not all(k in st.session_state for k in required_keys):
    st.warning("Primero entrena un modelo en **Entrenamiento**.")
    st.page_link("pages/2_Entrenamiento.py", label="Ir a Entrenamiento")
    st.stop()

model = st.session_state["current_model"]
metrics = st.session_state["current_metrics"]
df = st.session_state["df_filtered"].reset_index(drop=True)
X_scaled = st.session_state["X_scaled"]
params = st.session_state["current_params"]
training_time = st.session_state["current_training_time"]

# ---------- Header del modelo ----------
section_head(
    title="Modelo actual",
    subtitle=f"GMM · {' · '.join(f'{k}={v}' for k, v in params.items())}",
    kicker="Resumen",
)

col_h1, col_h2, col_h3 = st.columns(3)
col_h1.metric(
    "Tiempo de entrenamiento", f"{training_time} s",
    help="Duracion del proceso de ajuste (EM) del modelo GMM en segundos.",
)
col_h2.metric(
    "Registros", len(df),
    help="Cantidad de personas usadas para entrenar el modelo actual.",
)
col_h3.metric(
    "BIC", f"{model.get_bic(X_scaled):.1f}",
    help="Bayesian Information Criterion del modelo entrenado. Menor es mejor. Para comparar contra otros modelos guardados.",
)

st.write("")

# ---------- Metricas ----------
section_head(title="Metricas de evaluacion", kicker="Calidad del clustering")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(
    "Clusters", metrics["n_clusters"],
    help="Numero de clusters con al menos una persona asignada.",
)
col2.metric(
    "Silhouette",
    f"{metrics['silhouette']:.3f}" if metrics["silhouette"] is not None else "N/A",
    help=(
        "Coeficiente de silueta: s(i) = (b(i) - a(i)) / max(a(i), b(i)). "
        "a(i) = distancia promedio dentro del propio cluster (cohesion). "
        "b(i) = distancia promedio al cluster vecino mas cercano (separacion). "
        "Rango -1 a 1. Umbrales: >0.7 fuerte, 0.5-0.7 razonable, 0.25-0.5 debil (tipico en personalidad), <0.25 sin estructura."
    ),
)
col3.metric(
    "Davies-Bouldin",
    f"{metrics['davies_bouldin']:.3f}" if metrics["davies_bouldin"] is not None else "N/A",
    help=(
        "Indice Davies-Bouldin. Promedio del maximo de la razon entre dispersiones internas y separacion entre clusters. "
        "Menor es mejor. <1 clusters bien definidos, 1-2 aceptable, >2 muy solapados."
    ),
)
col4.metric(
    "Calinski-Harabasz",
    f"{metrics['calinski_harabasz']:.1f}" if metrics["calinski_harabasz"] is not None else "N/A",
    help=(
        "Razon entre varianza inter-cluster e intra-cluster. Mayor es mejor. "
        "Sin umbrales absolutos, sirve para comparar modelos entrenados sobre los mismos datos."
    ),
)
col5.metric(
    "AIC", f"{model.get_aic(X_scaled):.1f}",
    help="Akaike Information Criterion. Menor es mejor. Penaliza menos que BIC.",
)

if metrics["silhouette"] is not None:
    st.markdown(
        f'<div class="card-accent">'
        f'<strong>Interpretacion del silhouette:</strong> {interpret_silhouette(metrics["silhouette"])}'
        f'</div>',
        unsafe_allow_html=True,
    )

st.write("")

# ---------- PCA 2D ----------
section_head(
    title="Proyeccion PCA en 2D",
    subtitle="Reduccion de dimensionalidad para visualizar los clusters en dos ejes.",
    kicker="Visualizacion",
)

X_2d, pca = project_2d(X_scaled)
variance = variance_explained(pca)
st.session_state["current_pca"] = pca

st.plotly_chart(pca_scatter(X_2d, model.labels, variance), use_container_width=True)
st.caption(
    f"Los dos componentes explican **{variance['total']*100:.1f}%** de la varianza total. "
    "Puntos cercanos son personas con perfiles similares.",
    help=(
        "Porcentaje de informacion original preservada en la proyeccion 2D. "
        "El resto se pierde al comprimir de 5D a 2D. "
        "Rangos: >70% muy fiel, 50-70% razonable, <50% pobre (interpretar con cuidado)."
    ),
)

st.write("")

# ---------- Perfil por cluster ----------
section_head(title="Perfil promedio por cluster", kicker="Interpretacion")

profile = compute_cluster_profile(df, model.labels)
global_means = {d: float(df[d].mean()) for d in DIMENSIONS}
profile_with_interp = build_interpretations(profile, global_means)
profile_display = rename_profile_columns(profile_with_interp)
profile_display = profile_display.rename(columns={d: DIMENSION_LABELS[d] for d in DIMENSIONS})

st.dataframe(profile_display, use_container_width=True, hide_index=True)
st.plotly_chart(cluster_profile_bars(profile), use_container_width=True)

st.write("")

section_head(title="Interpretacion textual")

for _, row in profile_with_interp.iterrows():
    cluster_id = row["cluster"]
    label = "Ruido" if cluster_id == -1 else f"Cluster {int(cluster_id)}"
    dims_line = "  ·  ".join([f"{d}={row[d]:.2f}" for d in DIMENSIONS])
    st.markdown(
        f'<div class="card" style="margin-bottom: 10px;">'
        f'<strong>{label}</strong> · {row["interpretación"]}<br/>'
        f'<span style="color:var(--text-3); font-size:0.85rem;">{dims_line}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.write("")

# ---------- Probabilidades (GMM) ----------
section_head(
    title="Probabilidades por cluster (soft clustering)",
    subtitle="GMM asigna una probabilidad de pertenencia a cada cluster. Las personas fronterizas son especialmente interesantes.",
    kicker="Exclusivo GMM",
)

probs = model.get_probabilities(X_scaled)
prob_max = probs.max(axis=1)
n_clusters = probs.shape[1]

probs_df = pd.DataFrame(
    probs.round(3),
    columns=[f"P(Cluster {i})" for i in range(n_clusters)],
)

display_cols = [c for c in ["edad", "genero"] if c in df.columns]
detail = pd.concat(
    [
        df[display_cols].reset_index(drop=True),
        pd.Series(model.labels, name="cluster_asignado"),
        pd.Series(prob_max.round(3), name="prob_maxima"),
        probs_df,
    ],
    axis=1,
)

col_o1, col_o2 = st.columns([1, 1])
with col_o1:
    solo_fronterizas = st.checkbox(
        "Mostrar solo personas fronterizas (prob_maxima < 0.7)", value=False,
        help=(
            "Personas cuya probabilidad maxima es menor a 0.7, es decir, "
            "con perfil mixto entre dos o mas clusters. "
            "Son el hallazgo mas valioso academicamente porque revelan "
            "la naturaleza continua de la personalidad. "
            "En clustering duro (K-Means, jerarquico) esta informacion se pierde."
        ),
    )
with col_o2:
    st.metric(
        "Prob. maxima promedio", f"{prob_max.mean():.3f}",
        help=(
            "Indica cuan seguro esta el modelo en promedio. "
            ">0.85 muy seguro, mayoria de asignaciones claras. "
            "0.70-0.85 confiable con casos fronterizos ocasionales. "
            "0.55-0.70 mucho perfil mixto. "
            "<0.55 alta incertidumbre general."
        ),
    )

view = detail[detail["prob_maxima"] < 0.7] if solo_fronterizas else detail

# Barras visuales para las columnas de probabilidad
col_config = {
    "prob_maxima": st.column_config.ProgressColumn(
        "Prob. maxima",
        help="Confianza de la asignacion (0-1). <0.7 = fronteriza.",
        format="%.3f", min_value=0.0, max_value=1.0,
    ),
}
for i in range(n_clusters):
    col_config[f"P(Cluster {i})"] = st.column_config.ProgressColumn(
        f"P(C{i})",
        format="%.2f", min_value=0.0, max_value=1.0,
    )

st.dataframe(
    view.sort_values("prob_maxima"),
    use_container_width=True, hide_index=True, height=340,
    column_config=col_config,
)

st.write("")
st.markdown("**Casos con mayor incertidumbre**")
st.caption("Las 3 personas con la probabilidad maxima mas baja.")
top_uncertain = detail.nsmallest(3, "prob_maxima")
st.dataframe(top_uncertain, use_container_width=True, hide_index=True)

st.write("")
st.divider()

# ---------- Dataset con etiquetas ----------
df_labeled = df.copy()
df_labeled["cluster"] = model.labels
df_labeled["prob_maxima"] = prob_max.round(3)
st.session_state["df_labeled"] = df_labeled

with st.expander("Ver dataset completo con etiquetas"):
    st.dataframe(df_labeled, use_container_width=True, hide_index=True)
