import streamlit as st
from datetime import datetime, timezone

from src.db.model_repository import (
    save_metadata, find_all, delete_by_id, ensure_indexes, count, name_exists,
    readable_model_name, readable_dataset_name,
)
from src.persistence.model_io import (
    save_model_bundle, delete_model_bundle, bundle_exists,
)
from src.ui.theme import apply_global_style, render_sidebar, section_head

st.set_page_config(page_title="Modelos", layout="wide")
apply_global_style()
render_sidebar()

st.markdown("## Modelos guardados")
st.caption("Guarda el modelo actual con un nombre unico y consulta el historial.")
st.divider()

# ---------- Conexion Mongo ----------
try:
    ensure_indexes()
except Exception as e:
    st.error(f"No se pudo conectar con MongoDB: {e}")
    st.info("Revisa `config/.env` (MONGO_URI, MONGO_DB) y la conexion a internet.")
    st.stop()

# ==========================================================================
#                       GUARDAR MODELO ACTUAL
# ==========================================================================
section_head(title="Guardar modelo actual", kicker="Persistencia")

required = ["current_model", "current_metrics", "scaler",
            "current_algorithm", "current_params"]
if not all(k in st.session_state for k in required):
    st.info("Para guardar un modelo, primero entrena uno en **Entrenamiento**.")
    st.page_link("pages/2_Entrenamiento.py", label="Ir a Entrenamiento")
else:
    params = st.session_state["current_params"]
    metrics = st.session_state["current_metrics"]
    training_time = st.session_state["current_training_time"]
    csv_name = st.session_state.get("csv_name", "sin_archivo")

    col_info = st.columns(3)
    col_info[0].metric(
        "Algoritmo", "GMM",
        help="Algoritmo del modelo actual. En esta version solo GMM (Gaussian Mixture Model).",
    )
    col_info[1].metric(
        "Clusters", metrics.get("n_clusters", "N/A"),
        help="Numero de clusters efectivos del modelo entrenado en la sesion actual.",
    )
    col_info[2].metric(
        "Silhouette",
        f"{metrics['silhouette']:.3f}" if metrics.get("silhouette") is not None else "N/A",
        help="Modelos con Silhouette mas alto tienen clusters mejor separados. Comparar solo entre modelos entrenados sobre datasets similares.",
    )

    st.write("")

    default_name = f"gmm_{csv_name.replace('.csv','')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    col_n, col_b = st.columns([3, 1])
    with col_n:
        model_name = st.text_input(
            "Nombre del modelo",
            value=default_name,
            help="Nombre unico para identificar este modelo en el historial. "
                 "Solo letras, numeros, guiones y guiones bajos.",
        )
    with col_b:
        st.write("")
        save_clicked = st.button(
            "Guardar", type="primary", use_container_width=True,
            help=(
                "Serializa el modelo, scaler y PCA como .pkl en la carpeta models/ "
                "y guarda un documento con los metadatos (hiperparametros, metricas, rutas) en MongoDB."
            ),
        )

    if save_clicked:
        model_name = (model_name or "").strip()

        # Validaciones
        if not model_name:
            st.error("El nombre no puede estar vacio.")
        elif len(model_name) > 80:
            st.error("El nombre es muy largo (max 80 caracteres).")
        elif not all(c.isalnum() or c in "_-." for c in model_name):
            st.error("El nombre solo puede contener letras, numeros, '.', '_' y '-'.")
        elif name_exists(model_name):
            st.error(f"Ya existe un modelo con el nombre '{model_name}'.")
        else:
            try:
                with st.spinner("Guardando..."):
                    wrapper = st.session_state["current_model"]
                    scaler = st.session_state["scaler"]
                    pca = st.session_state.get("current_pca")

                    # Si no tenemos PCA (usuario no visito Resultados),
                    # lo calculamos aqui para que Clasificacion pueda usarlo.
                    if pca is None and "X_scaled" in st.session_state:
                        from src.visualization.pca import project_2d
                        _, pca = project_2d(st.session_state["X_scaled"])

                    bundle = save_model_bundle(
                        wrapper.get_model(), scaler, "gmm", pca=pca,
                    )
                    metadata = {
                        "model_name": model_name,
                        "algorithm": "gmm",
                        "algorithm_label": "GMM",
                        "trained_at": datetime.now(timezone.utc),
                        "hyperparameters": params,
                        "metrics": metrics,
                        "training_time_seconds": training_time,
                        "n_records": len(st.session_state.get("df_filtered", [])),
                        "dataset_source": csv_name,
                        "model_file_path": bundle["model_file_path"],
                        "scaler_file_path": bundle["scaler_file_path"],
                        "pca_file_path": bundle.get("pca_file_path"),
                    }
                    save_metadata(metadata)
                st.success(f"Modelo guardado como '{model_name}'.")
                st.rerun()
            except Exception as e:
                st.error(f"Error al guardar el modelo: {e}")

st.write("")
st.divider()

# ==========================================================================
#                          HISTORIAL DE MODELOS
# ==========================================================================
try:
    all_models = find_all()
except Exception as e:
    st.error(f"No se pudo leer el historial: {e}")
    st.stop()

total = len(all_models)
orphans = [m for m in all_models if not bundle_exists(m)]

section_head(
    title=f"Historial ({total} modelos)",
    subtitle=f"{len(orphans)} con archivos faltantes" if orphans else "",
    kicker="Experimentos",
)

if total == 0:
    st.info("No hay modelos guardados aun.")
    st.stop()

# Purga de huerfanos
if orphans:
    st.warning(
        f"Hay **{len(orphans)}** modelo(s) con archivos .pkl faltantes en disco. "
        "No sirven para clasificar. Puedes purgarlos del historial."
    )
    if st.button("Purgar modelos huerfanos", type="secondary"):
        removed = 0
        for m in orphans:
            try:
                delete_by_id(str(m["_id"]))
                removed += 1
            except Exception:
                pass
        st.success(f"Purgados {removed} modelo(s).")
        st.rerun()

st.write("")

for m in all_models:
    model_id = str(m["_id"])
    name = readable_model_name(m)
    trained_at = m["trained_at"].strftime("%d/%m/%Y %H:%M") if m.get("trained_at") else ""
    metrics = m.get("metrics", {}) or {}
    ds_source = readable_dataset_name(m)
    sil = metrics.get("silhouette")
    algo_label = m.get("algorithm_label", m.get("algorithm", "?"))
    healthy = bundle_exists(m)
    tag = "" if healthy else " [ARCHIVOS FALTANTES]"

    header = (
        f"**{name}**{tag} · {algo_label} · "
        f"clusters={metrics.get('n_clusters', 'N/A')} · "
        f"silhouette={f'{sil:.3f}' if isinstance(sil, (int, float)) else 'N/A'} · "
        f"{trained_at}"
    )

    with st.expander(header):
        col_a, col_b = st.columns([3, 1])

        with col_a:
            if not healthy:
                st.error(
                    "Los archivos .pkl de este modelo no existen en disco. "
                    "No es utilizable. Eliminalo o purga los huerfanos."
                )

            st.markdown("**Hiperparametros**")
            for k, v in (m.get("hyperparameters") or {}).items():
                st.caption(f"· {k}: {v}")

            st.markdown("**Metricas**")
            for k, v in metrics.items():
                if isinstance(v, float):
                    st.caption(f"· {k}: {v:.3f}")
                else:
                    st.caption(f"· {k}: {v}")

            st.caption(
                f"Registros: {m.get('n_records', 'N/A')} · "
                f"Archivo: {ds_source} · "
                f"Tiempo: {m.get('training_time_seconds', 'N/A')} s"
            )

        with col_b:
            if healthy:
                st.page_link("pages/6_Clasificacion.py", label="Clasificar con este")
            if st.button(
                "Eliminar", key=f"del_{model_id}",
                use_container_width=True,
                help="Elimina el documento en MongoDB y los tres archivos .pkl del filesystem (modelo, scaler, PCA). Operacion irreversible.",
            ):
                try:
                    delete_model_bundle(
                        m.get("model_file_path", ""),
                        m.get("scaler_file_path", ""),
                        m.get("pca_file_path"),
                    )
                except Exception:
                    pass
                try:
                    delete_by_id(model_id)
                    st.rerun()
                except Exception as e:
                    st.error(f"No se pudo eliminar: {e}")
