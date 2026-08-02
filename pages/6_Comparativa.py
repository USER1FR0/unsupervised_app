import streamlit as st
import pandas as pd
from src.db.model_repository import find_all, count, ensure_indexes
from src.clustering import ALGORITHMS

st.set_page_config(page_title="Comparativa", page_icon=":material/leaderboard:", layout="wide")

try:
    ensure_indexes()
except Exception as e:
    st.error(f"No se pudo conectar con MongoDB: {e}")
    st.stop()

st.title("Comparativa de modelos")
st.caption("Ranking objetivo de cualquier subconjunto de modelos guardados en el historial")

total = count()
if total == 0:
    st.info(
        "No hay modelos guardados en el historial. Entrena modelos en "
        "**Entrenamiento** y guárdalos desde **Modelos** para compararlos aquí."
    )
    st.stop()

models = find_all()

# ---------- Construir tabla completa ----------
rows = []
for m in models:
    metrics = m.get("metrics", {}) or {}
    hyp = m.get("hyperparameters", {}) or {}
    _id_short = str(m["_id"])[-6:]
    trained_at = m["trained_at"].strftime("%d/%m/%Y %H:%M") if m.get("trained_at") else ""
    algo_label = m.get("algorithm_label", m.get("algorithm", ""))

    rows.append({
        "modelo_id": _id_short,
        "algoritmo": algo_label,
        "silhouette": metrics.get("silhouette"),
        "davies_bouldin": metrics.get("davies_bouldin"),
        "calinski_harabasz": metrics.get("calinski_harabasz"),
        "clusters": metrics.get("n_clusters"),
        "outliers": metrics.get("n_outliers", 0),
        "muestras": metrics.get("n_samples", m.get("n_records", "")),
        "hiperparámetros": ", ".join(f"{k}={v}" for k, v in hyp.items()),
        "fuente": m.get("data_source", "MongoDB"),
        "tiempo (s)": m.get("training_time_seconds"),
        "fecha": trained_at,
        "_id_full": str(m["_id"]),
        # Etiqueta legible para el selector individual
        "_label": f"[{_id_short}] {algo_label} · sil={metrics.get('silhouette', 'N/A')} · {trained_at}",
    })

df = pd.DataFrame(rows)

# ---------- Filtros ----------
st.markdown("### Filtros")

modo = st.radio(
    "Modo de filtrado",
    options=["Por algoritmo (grupos)", "Por modelo (individual)"],
    horizontal=True,
    help=(
        "**Por algoritmo**: incluyes/excluyes todos los modelos de un algoritmo. "
        "**Por modelo**: eliges uno por uno los que quieres comparar, sin importar el algoritmo."
    ),
)

col_f1, col_f2 = st.columns([3, 1])

if modo.startswith("Por algoritmo"):
    with col_f1:
        algos_disponibles = sorted(df["algoritmo"].dropna().unique().tolist())
        algos_sel = st.multiselect(
            "Algoritmos a incluir",
            options=algos_disponibles,
            default=algos_disponibles,
        )
    df_f = df[df["algoritmo"].isin(algos_sel)].copy()
else:
    with col_f1:
        etiquetas = df["_label"].tolist()
        seleccion = st.multiselect(
            "Selecciona los modelos que quieres comparar",
            options=etiquetas,
            default=etiquetas,
            help="Puedes elegir cualquier combinación, mezclando algoritmos y experimentos.",
        )
    df_f = df[df["_label"].isin(seleccion)].copy()

with col_f2:
    orden = st.selectbox(
        "Ordenar por",
        options=[
            "silhouette (↓ mejor)",
            "davies_bouldin (↑ mejor)",
            "calinski_harabasz (↓ mejor)",
            "fecha (más reciente)",
        ],
    )

if orden.startswith("silhouette"):
    df_f = df_f.sort_values("silhouette", ascending=False, na_position="last")
elif orden.startswith("davies_bouldin"):
    df_f = df_f.sort_values("davies_bouldin", ascending=True, na_position="last")
elif orden.startswith("calinski_harabasz"):
    df_f = df_f.sort_values("calinski_harabasz", ascending=False, na_position="last")
else:
    df_f = df_f.sort_values("fecha", ascending=False)

st.write("")
st.info(f"**{len(df_f)}** modelos seleccionados de **{total}** en el historial.")

if df_f.empty:
    st.warning("No hay modelos seleccionados. Ajusta los filtros arriba.")
    st.stop()

# ---------- Ranking ----------
st.markdown("### Ranking de modelos")
st.caption("Cada fila es un modelo individual (no un algoritmo). El primero según el criterio de orden es el ganador.")

show_cols = [
    "modelo_id", "algoritmo", "silhouette", "davies_bouldin", "calinski_harabasz",
    "clusters", "outliers", "muestras", "fuente", "hiperparámetros",
    "tiempo (s)", "fecha",
]

st.dataframe(
    df_f[show_cols],
    use_container_width=True,
    hide_index=True,
    height=min(500, 40 + 35 * len(df_f)),
)

# ---------- Modelo ganador por criterio ----------
st.write("")
st.markdown("### Modelo ganador por criterio")
st.caption("Cada card es un modelo específico, identificado por su ID corto. Puede ser cualquier modelo del subconjunto seleccionado.")

col1, col2, col3 = st.columns(3)

def _best(df_, col, higher_better=True):
    df_v = df_.dropna(subset=[col])
    if df_v.empty:
        return None
    if higher_better:
        return df_v.loc[df_v[col].idxmax()]
    return df_v.loc[df_v[col].idxmin()]

best_sil = _best(df_f, "silhouette", higher_better=True)
best_db = _best(df_f, "davies_bouldin", higher_better=False)
best_ch = _best(df_f, "calinski_harabasz", higher_better=True)


def _render_winner(col, title, winner, metric_key, fmt):
    with col:
        st.markdown(f"**{title}**")
        if winner is not None:
            st.metric(
                label=f"[{winner['modelo_id']}] {winner['algoritmo']}",
                value=fmt.format(winner[metric_key]),
            )
            st.caption(winner["hiperparámetros"] or "sin hiperparámetros")
            st.caption(f"Entrenado: {winner['fecha']}")
        else:
            st.caption("Sin datos.")


_render_winner(col1, "Mejor Silhouette (mayor)", best_sil, "silhouette", "{:.3f}")
_render_winner(col2, "Mejor Davies-Bouldin (menor)", best_db, "davies_bouldin", "{:.3f}")
_render_winner(col3, "Mejor Calinski-Harabasz (mayor)", best_ch, "calinski_harabasz", "{:.1f}")

st.write("")
st.divider()

# ---------- Descarga ----------
st.markdown("### Exportar")
csv_bytes = df_f[show_cols].to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Descargar comparativa (CSV)",
    data=csv_bytes,
    file_name="comparativa_modelos.csv",
    mime="text/csv",
    use_container_width=True,
)
