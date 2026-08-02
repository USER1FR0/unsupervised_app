import os
import sys
import warnings

# --- Silenciar warnings cosméticos antes de importar streamlit / joblib / requests ---
# LOKY_MAX_CPU_COUNT debe estar antes de que joblib se importe. Windows 11 24H2
# quitó wmic y joblib truena buscándolo si esta var no está seteada.
os.environ["LOKY_MAX_CPU_COUNT"] = os.environ.get(
    "LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4)
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", message=".*urllib3.*")

import streamlit as st
from src.db.response_repository import count, ensure_indexes, upsert_many
from src.sheets_importer import fetch_sheet_dataframe, dataframe_to_documents
from src.data.loader import load_csv_into_session, clear_csv_source

st.set_page_config(
    page_title="Análisis No Supervisado",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Intento de crear índices; si Mongo no responde, no rompemos la landing.
try:
    ensure_indexes()
    _mongo_ok = True
    _mongo_error = None
except Exception as e:
    _mongo_ok = False
    _mongo_error = str(e)

def _reset_derived_state():
    """Limpia el estado derivado cuando el usuario cambia la fuente de datos.
    Evita que un modelo entrenado sobre CSV se muestre encima de datos Mongo."""
    for key in (
        "df_filtered", "X_scaled", "scaler",
        "current_model", "current_metrics", "current_algorithm",
        "current_params", "current_training_time",
        "df_labeled", "pdf_buffer",
    ):
        st.session_state.pop(key, None)


# ---------- Sidebar (global) ----------
with st.sidebar:
    st.markdown("### Fuente de datos")

    prev_source = st.session_state.get("data_source", "MongoDB")
    source = st.radio(
        "Origen",
        options=["MongoDB", "CSV cargado"],
        index=0 if prev_source == "MongoDB" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    if source != prev_source:
        _reset_derived_state()
    st.session_state["data_source"] = source

    st.write("")

    if source == "MongoDB":
        if _mongo_ok:
            total = count()
            st.markdown(f"**{total}** registros en Mongo")
            if st.button("Sincronizar desde Sheets", use_container_width=True, type="primary"):
                with st.spinner("Sincronizando..."):
                    try:
                        from src.sheets_importer import get_last_import_errors
                        df_raw = fetch_sheet_dataframe()
                        docs = dataframe_to_documents(df_raw)
                        errors = get_last_import_errors()
                        result = upsert_many(docs)
                        msg = f"{result['inserted']} nuevos · {result['updated']} actualizados"
                        if errors:
                            msg += f" · {len(errors)} filas con error (revisa el log)"
                        st.success(msg)
                        if errors:
                            with st.expander("Filas con error"):
                                for e in errors[:20]:
                                    st.caption(f"Fila {e['row']}: {e['error']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            st.caption("Lee el Google Sheet y actualiza la base sin duplicar.")
        else:
            total = 0
            st.error("No hay conexión con MongoDB.")
            st.caption(_mongo_error or "Revisa config/.env y tu red.")

    else:  # CSV cargado
        uploaded = st.file_uploader(
            "Sube un CSV",
            type=["csv"],
            help="Debe contener al menos las columnas: O, C, E, A, N, edad, genero, estado, municipio.",
        )
        if uploaded is not None:
            try:
                n_rows = load_csv_into_session(uploaded)
                st.success(f"CSV cargado · {n_rows} filas")
            except Exception as e:
                st.error(f"CSV inválido: {e}")

        csv_df = st.session_state.get("df_csv_source")
        total = 0 if csv_df is None else len(csv_df)
        st.markdown(f"**{total}** registros en CSV activo")

        if csv_df is not None:
            if st.button("Quitar CSV cargado", use_container_width=True):
                clear_csv_source()
                st.rerun()

        st.caption(
            "El CSV se usa como fuente alterna a Mongo. "
            "Descarga un CSV de ejemplo en la carpeta `data/`."
        )

# ---------- Landing ----------
st.title("Análisis No Supervisado de Personalidad")
st.caption("Big Five · Unidad IV · Extracción de Conocimientos en Base de Datos")

st.write("")

if total == 0:
    if source == "MongoDB":
        st.info(
            "No hay registros en Mongo aún. Usa **Sincronizar desde Sheets** en el "
            "panel lateral, o cambia a la fuente **CSV cargado**."
        )
    else:
        st.info("Sube un CSV en el panel lateral para trabajar con esa fuente.")
    st.stop()

st.markdown(
    "Aplicación para el análisis no supervisado de perfiles de personalidad. "
    "Los datos provienen de una encuesta propia basada en el modelo Big Five (OCEAN), "
    "y se procesan con cuatro algoritmos de agrupamiento: K-Means, Clusterización "
    "Jerárquica, DBSCAN y GMM."
)

st.write("")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros", total)
col2.metric("Algoritmos", 4)
col3.metric("Dimensiones", 5)
col4.metric("Arquetipos", 25)

st.write("")
st.divider()

st.markdown("### Cómo usar la aplicación")
st.markdown(
    """
    1. **Exploración** — Inspecciona los datos, aplica filtros y revisa la estadística descriptiva.
    2. **Entrenamiento** — Selecciona un algoritmo, ajusta sus hiperparámetros y entrénalo.
    3. **Resultados** — Revisa las métricas, la proyección PCA y la interpretación de los clusters.
    4. **Modelos** — Guarda modelos entrenados y compara experimentos previos.
    5. **Descargas** — Exporta datos filtrados y genera reportes PDF.
    """
)
