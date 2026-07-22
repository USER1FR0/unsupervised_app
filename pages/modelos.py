import streamlit as st
from datetime import datetime
from src.db.model_repository import (
    save_metadata, find_all, find_by_id, delete_by_id, ensure_indexes, count,
)
from src.persistence.model_io import save_model_bundle, load_model_bundle, delete_model_bundle
from src.clustering import ALGORITHMS

st.set_page_config(page_title="Modelos", page_icon="💾", layout="wide")

ensure_indexes()

st.title("💾 Modelos guardados")
st.caption("Guarda modelos entrenados y consulta el historial de experimentos")

# ---------- Guardar modelo actual ----------
st.markdown("### Guardar modelo actual")

required = ["current_model", "current_metrics", "scaler", "current_algorithm", "current_params"]
if not all(k in st.session_state for k in required):
    st.info("Para guardar un modelo, primero entrena uno en la página **🧪 Entrenamiento**.")
else:
    algo = st.session_state["current_algorithm"]
    params = st.session_state["current_params"]
    metrics = st.session_state["current_metrics"]
    training_time = st.session_state["current_training_time"]

    st.markdown(f"**Algoritmo actual**: {ALGORITHMS[algo]['label']}")
    st.caption(" · ".join(f"{k}={v}" for k, v in params.items()))

    if st.button("💾 Guardar en historial", type="primary"):
        with st.spinner("Guardando..."):
            model_wrapper = st.session_state["current_model"]
            scaler = st.session_state["scaler"]

            # Serializar el modelo interno de sklearn
            bundle = save_model_bundle(
                model_wrapper.get_model(),
                scaler,
                algo,
            )

            # Guardar metadata
            metadata = {
                "algorithm": algo,
                "algorithm_label": ALGORITHMS[algo]["label"],
                "trained_at": datetime.utcnow(),
                "hyperparameters": params,
                "metrics": metrics,
                "training_time_seconds": training_time,
                "n_records": len(st.session_state.get("df_filtered", [])),
                "model_file_path": bundle["model_file_path"],
                "scaler_file_path": bundle["scaler_file_path"],
            }
            model_id = save_metadata(metadata)

        st.success(f"✅ Modelo guardado con ID `{model_id[:8]}...`")
        st.rerun()

st.write("")
st.divider()

# ---------- Historial ----------
st.markdown("### Historial de modelos")

total = count()
st.caption(f"{total} modelos en el historial")

if total == 0:
    st.info("No hay modelos guardados aún.")
    st.stop()

models = find_all()

for m in models:
    model_id = str(m["_id"])
    algo_label = m.get("algorithm_label", m["algorithm"])
    trained_at = m["trained_at"].strftime("%d/%m/%Y %H:%M")
    metrics = m.get("metrics", {})

    with st.expander(
        f"**{algo_label}** · {trained_at} · "
        f"silhouette={metrics.get('silhouette', 'N/A')} · "
        f"clusters={metrics.get('n_clusters', 'N/A')}",
    ):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("**Hiperparámetros**")
            for k, v in m.get("hyperparameters", {}).items():
                st.caption(f"· {k}: {v}")

            st.markdown("**Métricas**")
            for k, v in metrics.items():
                st.caption(f"· {k}: {v}")

            st.caption(f"Registros: {m.get('n_records', 'N/A')} · "
                       f"Tiempo: {m.get('training_time_seconds', 'N/A')} s")

        with col2:
            if st.button("🗑️ Eliminar", key=f"del_{model_id}", use_container_width=True):
                delete_model_bundle(m["model_file_path"], m["scaler_file_path"])
                delete_by_id(model_id)
                st.rerun()