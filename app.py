import streamlit as st
from src.db.response_repository import count, ensure_indexes, upsert_many
from src.sheets_importer import fetch_sheet_dataframe, dataframe_to_documents

st.set_page_config(
    page_title="Análisis No Supervisado",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_indexes()

# ---------- Sidebar (global) ----------
with st.sidebar:
    st.markdown("### Fuente de datos")

    total = count()
    st.markdown(f"**{total}** registros disponibles")

    st.write("")

    if st.button("Sincronizar", use_container_width=True, type="primary"):
        with st.spinner("Sincronizando..."):
            try:
                df_raw = fetch_sheet_dataframe()
                docs = dataframe_to_documents(df_raw)
                result = upsert_many(docs)
                st.success(f"{result['inserted']} nuevos · {result['updated']} actualizados")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.caption("Lee el Google Sheet y actualiza la base sin duplicar.")

# ---------- Landing ----------
st.title("Análisis No Supervisado de Personalidad")
st.caption("Big Five · Unidad IV · Extracción de Conocimientos en Base de Datos")

st.write("")

if total == 0:
    st.info("No hay registros aún. Usa **Sincronizar** en el panel lateral para importar respuestas.")
    st.stop()

st.markdown(
    "Aplicación para el análisis no supervisado de perfiles de personalidad. "
    "Los datos provienen de una encuesta propia basada en el modelo Big Five (OCEAN), "
    "y se procesan con cuatro algoritmos de agrupamiento: K-Means, Clusterización Jerárquica, "
    "DBSCAN y GMM."
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
    1. **📊 Exploración** — Inspecciona los datos, aplica filtros y revisa la estadística descriptiva.
    2. **🧪 Entrenamiento** — Selecciona un algoritmo, ajusta sus hiperparámetros y entrénalo.
    3. **📈 Resultados** — Revisa las métricas, la proyección PCA y la interpretación de los clusters.
    4. **💾 Modelos** — Guarda modelos entrenados y compara experimentos previos.
    5. **⬇️ Descargas** — Exporta datos filtrados y genera reportes PDF.
    """
)