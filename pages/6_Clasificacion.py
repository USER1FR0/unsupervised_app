import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

from src.db.model_repository import (
    find_all, ensure_indexes,
    readable_model_name, readable_dataset_name,
)
from src.persistence.model_io import predict_with_saved_model, bundle_exists
from src.data.loader import (
    load_csv_from_upload, load_sample, get_active_dataframe,
    SAMPLE_DATASETS, sample_exists, sample_size,
    DIMENSIONS, DIMENSION_LABELS, _normalize_dataframe,
)
from src.data.preprocessing import drop_null_dimensions
from src.data.synthetic_generator import generate_synthetic_dataset
from src.visualization.charts import pca_scatter, bar_dimensions_mean, bar_categorical
from src.ui.theme import (
    apply_global_style, render_sidebar, section_head, card,
    chip, cluster_chip,
)

st.set_page_config(page_title="Clasificacion", layout="wide")
apply_global_style()
render_sidebar()

st.markdown("## Clasificacion con modelo guardado")
st.caption(
    "Aplica un modelo previamente entrenado a datos nuevos y observa las probabilidades."
)
st.divider()

# ---------- Conexion Mongo ----------
try:
    ensure_indexes()
except Exception as e:
    st.error(f"No se pudo conectar con MongoDB: {e}")
    st.info("Revisa `config/.env` y tu conexion a internet.")
    st.stop()

try:
    all_models = find_all()
except Exception as e:
    st.error(f"No se pudo leer el historial de modelos: {e}")
    st.stop()

usable_models = [m for m in all_models if bundle_exists(m)]

if not all_models:
    st.info("No hay modelos guardados. Entrena y guarda uno primero.")
    st.page_link("pages/4_Modelos.py", label="Ir a Modelos")
    st.stop()

if not usable_models:
    st.error(
        "Hay modelos en el historial pero ninguno tiene los archivos .pkl en disco. "
        "Ve a **Modelos** y purga los huerfanos, o entrena y guarda uno nuevo."
    )
    st.page_link("pages/4_Modelos.py", label="Ir a Modelos")
    st.stop()


# ==========================================================================
#                          PASO 1: SELECCIONAR MODELO
# ==========================================================================
section_head(title="Selecciona un modelo", kicker="Paso 1")


def _select_label(m: dict) -> str:
    name = readable_model_name(m)
    ds = readable_dataset_name(m)
    sil = (m.get("metrics") or {}).get("silhouette")
    sil_str = f"silhouette {sil:.3f}" if isinstance(sil, (int, float)) else "silhouette N/A"
    return f"{name} — {ds} — {sil_str}"


selected = st.selectbox(
    "Modelo",
    options=usable_models,
    format_func=_select_label,
    index=0,
    help=(
        "Modelo del historial que se usara para clasificar los datos nuevos. "
        "Se cargaran juntos el modelo GMM, el StandardScaler y el PCA guardados. "
        "Un modelo con Silhouette bajo dara clasificaciones con mucha incertidumbre."
    ),
)
if selected is None:
    st.stop()

metrics = selected.get("metrics") or {}
hyperparams = selected.get("hyperparameters") or {}

# Card resumen del modelo seleccionado
_k = metrics.get("n_clusters", "?")
_n = selected.get("n_records", "?")
_cov = hyperparams.get("covariance_type", "?")
_name = readable_model_name(selected)
_origin = readable_dataset_name(selected)

model_summary = (
    '<div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:8px;">'
    + chip("GMM", "indigo")
    + chip(f"k = {_k}", "primary")
    + chip(f"n = {_n}", "emerald")
    + chip(f"cov = {_cov}", "amber")
    + '</div>'
    + '<div style="color:var(--text-2); font-size:0.9rem;">'
    f'<strong>Nombre:</strong> {_name} · '
    f'<strong>Origen:</strong> {_origin}'
    '</div>'
)
card(model_summary, variant="elevated")

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Silhouette",
    f"{metrics.get('silhouette', 0):.3f}"
    if isinstance(metrics.get("silhouette"), (int, float)) else "N/A",
    help="Del modelo seleccionado, no de los datos a clasificar. Un modelo con silhouette bajo dara clasificaciones mas inciertas.",
)
col_m2.metric("Davies-Bouldin",
    f"{metrics.get('davies_bouldin', 0):.3f}"
    if isinstance(metrics.get("davies_bouldin"), (int, float)) else "N/A",
    help="Menor Davies-Bouldin indica clusters mejor definidos en el entrenamiento.",
)
col_m3.metric(
    "Registros entrenados", selected.get("n_records", "N/A"),
    help="Con que cantidad de datos se entreno el modelo. Muestras muy pequenas hacen el modelo menos robusto para generalizar.",
)

st.write("")


# ==========================================================================
#                       PASO 2: DATOS A CLASIFICAR
# ==========================================================================
section_head(
    title="Datos a clasificar",
    subtitle="Sube un CSV, usa un ejemplo, genera un demo o el dataset actual.",
    kicker="Paso 2",
)

MODE_OPTIONS = [
    "Cargar CSV nuevo",
    "Dataset de ejemplo",
    "Generar demo",
    "Usar dataset actual",
]
source_mode = st.radio(
    "Fuente", options=MODE_OPTIONS, horizontal=True,
    label_visibility="collapsed", key="clf_source_mode",
    help=(
        "Cuatro opciones segun el escenario: "
        "CSV nuevo para datos externos, "
        "Dataset de ejemplo para pruebas cruzadas, "
        "Generar demo para crear datos frescos al momento, "
        "Usar dataset actual para clasificar lo que ya esta cargado en la sesion."
    ),
)

df_to_classify = None
source_name = None

# --- CSV cargado ---
if source_mode == "Cargar CSV nuevo":
    uploaded = st.file_uploader(
        "Arrastra o selecciona un CSV con las columnas OCEAN",
        type=["csv"], label_visibility="collapsed", key="clf_uploader",
        help=(
            "El CSV debe contener las 5 dimensiones OCEAN, o q1-q20 para calcular scoring. "
            "Se aplica scaler.transform (NO fit_transform) para mantener la escala del entrenamiento."
        ),
    )
    if uploaded is not None:
        try:
            df_to_classify = load_csv_from_upload(uploaded)
            source_name = uploaded.name
        except Exception as e:
            st.error(f"CSV invalido: {e}")

# --- Sample ---
elif source_mode == "Dataset de ejemplo":
    cols = st.columns(len(SAMPLE_DATASETS))
    for i, (key, filename) in enumerate(SAMPLE_DATASETS.items()):
        exists = sample_exists(key)
        n = sample_size(key) if exists else 0
        label = f"{key.capitalize()} ({n})" if exists else f"{key.capitalize()} (no disp.)"
        if cols[i].button(label, key=f"clf_sample_{key}",
                            use_container_width=True, disabled=not exists):
            st.session_state["_clf_selected_sample"] = key
            st.rerun()
    selected_sample = st.session_state.get("_clf_selected_sample")
    if selected_sample and sample_exists(selected_sample):
        try:
            df_to_classify = load_sample(selected_sample)
            source_name = SAMPLE_DATASETS[selected_sample]
            st.caption(f"Sample activo: **{selected_sample.capitalize()}**")
        except Exception as e:
            st.error(f"Error al cargar sample: {e}")

# --- Generar demo ---
elif source_mode == "Generar demo":
    st.caption(
        "Genera un dataset sintetico en el momento con el mismo generador del proyecto. "
        "Ideal para probar el modelo con datos nunca vistos durante la exposicion."
    )
    col_n, col_seed, col_gen = st.columns([1, 1, 1])
    with col_n:
        n_gen = st.number_input(
            "Cantidad de registros",
            min_value=10, max_value=2000, value=100, step=10,
            help="Entre 50 y 200 es un rango tipico para demostracion. Muy pocos no es representativo, muchos ralentiza sin aportar.",
        )
    with col_seed:
        seed_gen = st.number_input(
            "Seed",
            min_value=0, max_value=999999, value=99, step=1,
            help="Cualquier entero. Misma semilla produce datos identicos (util para reproducir). Semilla nueva produce datos unicos que ningun modelo ha visto.",
        )
    with col_gen:
        st.write("")
        if st.button(
            "Generar", type="secondary", use_container_width=True,
            help="Ejecuta el generador sintetico con la cantidad y seed indicadas y prepara los datos para clasificar.",
        ):
            try:
                with st.spinner(f"Generando {n_gen} registros..."):
                    df_gen = generate_synthetic_dataset(int(n_gen), int(seed_gen))
                    df_gen = _normalize_dataframe(df_gen)
                st.session_state["_clf_generated_df"] = df_gen
                st.session_state["_clf_generated_meta"] = f"demo_n{n_gen}_seed{seed_gen}"
                st.rerun()
            except Exception as e:
                st.error(f"Error al generar: {e}")

    if "_clf_generated_df" in st.session_state:
        df_to_classify = st.session_state["_clf_generated_df"]
        source_name = st.session_state.get("_clf_generated_meta", "demo_generado")

# --- Actual ---
elif source_mode == "Usar dataset actual":
    df_to_classify = get_active_dataframe()
    source_name = st.session_state.get("csv_name", "actual")
    if df_to_classify is None or df_to_classify.empty:
        st.warning("No hay dataset cargado. Ve a **Inicio** para cargarlo.")

# ---------- Preview del dataset elegido ----------
if df_to_classify is not None and not df_to_classify.empty:
    st.write("")
    st.markdown(f"**Vista previa** · {len(df_to_classify)} registros desde `{source_name}`")

    preview_cols = [c for c in ["submitted_at"] + DIMENSIONS +
                    ["arquetipo", "edad", "genero", "estado"]
                    if c in df_to_classify.columns]
    st.dataframe(
        df_to_classify[preview_cols].head(5),
        use_container_width=True, hide_index=True,
    )

    # Mini stats visuales
    st.write("")
    col_stats1, col_stats2 = st.columns([1, 1])
    with col_stats1:
        try:
            means = {d: df_to_classify[d].mean() for d in DIMENSIONS}
            fig = bar_dimensions_mean(means, "Perfil OCEAN promedio del dataset")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
    with col_stats2:
        try:
            if "arquetipo" in df_to_classify.columns:
                top_arq = df_to_classify["arquetipo"].value_counts().head(6)
                st.plotly_chart(
                    bar_categorical(top_arq, "Top arquetipos del dataset"),
                    use_container_width=True,
                )
            elif "genero" in df_to_classify.columns:
                st.plotly_chart(
                    bar_categorical(df_to_classify["genero"].value_counts(),
                                     "Distribucion por genero"),
                    use_container_width=True,
                )
        except Exception:
            pass

st.write("")


# ==========================================================================
#                       PASO 3: EJECUTAR CLASIFICACION
# ==========================================================================
section_head(title="Ejecutar clasificacion", kicker="Paso 3")

can_run = df_to_classify is not None and not df_to_classify.empty

if st.button(
    "Clasificar", type="primary", use_container_width=True, disabled=not can_run,
    help=(
        "Carga modelo, scaler y PCA desde el filesystem. "
        "Aplica scaler.transform a los datos nuevos. "
        "Ejecuta predict y predict_proba. "
        "Proyecta con el PCA guardado para visualizacion."
    ),
):
    try:
        with st.spinner("Aplicando modelo..."):
            df_new = drop_null_dimensions(df_to_classify)
            if df_new.empty:
                st.error("El dataset esta vacio tras remover nulos.")
                st.stop()
            missing = [c for c in DIMENSIONS if c not in df_new.columns]
            if missing:
                st.error(f"Faltan columnas OCEAN en los datos: {missing}")
                st.stop()
            X_new = df_new[DIMENSIONS].values
            result = predict_with_saved_model(selected, X_new)
        st.session_state["clf_df"] = df_new
        st.session_state["clf_labels"] = result["labels"]
        st.session_state["clf_probs"] = result["probabilities"]
        st.session_state["clf_X_2d"] = result["X_2d"]
        st.session_state["clf_source_name"] = source_name
        st.session_state["clf_model_name"] = readable_model_name(selected)
        st.success(f"Clasificados **{len(df_new)}** registros.")
    except FileNotFoundError as e:
        st.error(f"Los archivos del modelo no existen: {e}")
        st.info("Ve a **Modelos** y purga los huerfanos.")
    except Exception as e:
        st.error(f"Error al aplicar el modelo: {e}")


# ==========================================================================
#                             PASO 4: RESULTADOS
# ==========================================================================
if "clf_labels" in st.session_state:
    st.write("")
    st.divider()
    section_head(title="Resultados", kicker="Paso 4")

    df_new = st.session_state["clf_df"]
    labels = st.session_state["clf_labels"]
    probs = st.session_state["clf_probs"]
    X_2d = st.session_state["clf_X_2d"]

    prob_max = probs.max(axis=1) if probs is not None else None
    n_frontier = int((prob_max < 0.7).sum()) if prob_max is not None else 0
    n_confident = int((prob_max >= 0.9).sum()) if prob_max is not None else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Registros", len(df_new),
        help="Cantidad de personas clasificadas por el modelo tras remover nulos.",
    )
    col2.metric(
        "Clusters presentes", int(pd.Series(labels).nunique()),
        help="Cuantos clusters distintos aparecieron en la asignacion. Puede ser menor a k si el dataset es pequeno.",
    )
    if prob_max is not None:
        col3.metric(
            "Muy confiables (>0.9)", n_confident,
            help="Personas cuya probabilidad maxima supera 0.9. El modelo esta muy seguro de su asignacion.",
        )
        col4.metric(
            "Fronterizas (<0.7)", n_frontier,
            help="Personas con perfil mixto entre dos o mas clusters. Son las mas interesantes psicologicamente.",
        )

    st.write("")

    # Distribucion visual de clusters
    st.markdown("**Distribucion de asignaciones**")
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    st.plotly_chart(
        bar_categorical(
            cluster_counts.rename(lambda x: f"Cluster {int(x)}"),
            "Personas por cluster",
        ),
        use_container_width=True,
    )

    st.write("")
    st.markdown("**Tabla de predicciones**")

    show_cols = [c for c in ["edad", "genero"] + DIMENSIONS if c in df_new.columns]
    result_df = df_new[show_cols].copy().reset_index(drop=True)
    result_df["cluster_asignado"] = labels
    if prob_max is not None:
        result_df["prob_maxima"] = prob_max.round(3)
        for i in range(probs.shape[1]):
            result_df[f"P(Cluster {i})"] = probs[:, i].round(3)

    # Configuracion visual de columnas (barras de probabilidad)
    col_config = {}
    if prob_max is not None:
        col_config["prob_maxima"] = st.column_config.ProgressColumn(
            "Prob. maxima",
            help="Confianza de la asignacion (0-1). <0.7 = fronteriza.",
            format="%.3f",
            min_value=0.0,
            max_value=1.0,
        )
        for i in range(probs.shape[1]):
            col_config[f"P(Cluster {i})"] = st.column_config.ProgressColumn(
                f"P(C{i})",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            )

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True,
        height=340,
        column_config=col_config,
    )

    # PCA 2D
    if X_2d is not None:
        st.write("")
        st.markdown("**Proyeccion PCA 2D (puntos nuevos coloreados por cluster)**")
        variance = {"pc1": 0.0, "pc2": 0.0, "total": 0.0}
        st.plotly_chart(
            pca_scatter(X_2d, np.asarray(labels), variance,
                        title="Puntos proyectados con el PCA del entrenamiento"),
            use_container_width=True,
        )

    # Personas fronterizas con chips
    if prob_max is not None:
        st.write("")
        st.markdown("**Personas fronterizas (prob_maxima < 0.7)**")
        frontier = result_df[result_df["prob_maxima"] < 0.7].sort_values("prob_maxima")
        if frontier.empty:
            st.caption("Ninguna persona quedo fronteriza con este umbral. "
                       "Todas las asignaciones son confiables.")
        else:
            st.caption(f"{len(frontier)} personas comparten rasgos entre multiples clusters.")

            # Top 5 con chips visuales
            for _, row in frontier.head(5).iterrows():
                cluster_id = int(row["cluster_asignado"])
                p_max = float(row["prob_maxima"])
                edad = row.get("edad", "?")
                genero = row.get("genero", "?")

                ocean_line = " · ".join(
                    f"{d}={row[d]:.2f}" for d in DIMENSIONS if d in row
                )
                content = (
                    f'<div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">'
                    f'{cluster_chip(cluster_id)}'
                    f'<strong>{edad} años · {genero}</strong>'
                    f'<span style="color:var(--text-2); font-size:0.85rem;">'
                    f'confianza {p_max:.1%}</span>'
                    f'</div>'
                    f'<div style="color:var(--text-3); font-size:0.8rem; margin-top:6px;">'
                    f'{ocean_line}'
                    f'</div>'
                )
                card(content)

            if len(frontier) > 5:
                with st.expander(f"Ver los otros {len(frontier) - 5} casos fronterizos"):
                    st.dataframe(
                        frontier.iloc[5:],
                        use_container_width=True, hide_index=True,
                        column_config=col_config,
                    )

    st.write("")
    st.markdown("**Descargar predicciones**")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Descargar CSV",
        data=csv,
        file_name=f"predicciones_{st.session_state.get('clf_model_name','modelo').replace(' ','_')}_{ts}.csv",
        mime="text/csv",
        use_container_width=True,
        help="CSV con los datos clasificados, cluster asignado, probabilidad maxima y probabilidades por cluster.",
    )
