import os
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = os.environ.get(
    "LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4)
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", message=".*urllib3.*")

import streamlit as st

from src.data.loader import (
    load_csv_from_upload, load_sample,
    set_active_dataframe, clear_active_dataframe,
    get_active_dataframe,
    SAMPLE_DATASETS, sample_exists, sample_size,
    DIMENSIONS,
)
from src.ui.theme import apply_global_style, render_sidebar, hero, section_head, card

st.set_page_config(
    page_title="Big Five Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_style()
render_sidebar()

# ---------- HERO ----------
hero(
    title="Analisis No Supervisado de Personalidad",
    subtitle=(
        "Descubre grupos naturales de personalidad en tus datos usando el modelo "
        "Big Five (OCEAN) y el algoritmo de Mezcla Gaussiana (GMM). Carga tu CSV, "
        "explora la estadistica, entrena el modelo y clasifica nuevos registros."
    ),
    eyebrow="Unidad IV · Extraccion de Conocimientos",
)

df = get_active_dataframe()

# ==========================================================================
#                              CARGA DE DATOS
# ==========================================================================

section_head(
    title="Carga tus datos",
    subtitle="Sube un CSV con respuestas Big Five o usa uno de los datasets de ejemplo.",
    kicker="Paso 1",
)

col_up, col_sample = st.columns([2, 1])

with col_up:
    uploader_seed = st.session_state.get("_uploader_seed", 0)
    uploaded = st.file_uploader(
        "Arrastra o selecciona un archivo CSV",
        type=["csv"],
        help=(
            "Columnas requeridas: O, C, E, A, N, edad, genero, estado, municipio. "
            "Opcionalmente q1..q20 para reconstruir los items."
        ),
        label_visibility="collapsed",
        key=f"main_uploader_{uploader_seed}",
    )
    if uploaded is not None:
        already_loaded = st.session_state.get("csv_name") == uploaded.name and df is not None
        if not already_loaded:
            try:
                df_new = load_csv_from_upload(uploaded)
                set_active_dataframe(df_new, uploaded.name)
                df = df_new
                st.rerun()
            except Exception as e:
                st.error(f"CSV invalido: {e}")

with col_sample:
    st.markdown("**Datasets de ejemplo**")
    for key, filename in SAMPLE_DATASETS.items():
        exists = sample_exists(key)
        n = sample_size(key) if exists else 0
        label = f"{key.capitalize()} ({n})" if exists else f"{key.capitalize()} (no disponible)"
        sample_help = {
            "real": "Respuestas autenticas de la encuesta publica. Dataset principal para entrenar el modelo definitivo. Distribucion no controlada, refleja la muestra real.",
            "synthetic": "Registros generados por script con 5 arquetipos base (Explorer, Architect, Charismatic, Guardian, Intense) y ruido gaussiano. Sirve para validar que el algoritmo recupera estructura conocida.",
            "demo": "Registros generados con semilla distinta al Sintetico (99 vs 42). Estructura similar pero datos que ningun modelo entrenado ha visto. Para clasificacion en vivo.",
        }.get(key)
        if st.button(label, key=f"load_{key}", use_container_width=True,
                      disabled=not exists, help=sample_help):
            try:
                df_new = load_sample(key)
                set_active_dataframe(df_new, filename)
                st.success(f"Cargados **{len(df_new)}** registros desde `{filename}`.")
                df = df_new
                st.rerun()
            except Exception as e:
                st.error(f"Error al cargar {filename}: {e}")


# ==========================================================================
#                     ESTADO ACTUAL Y SIGUIENTES PASOS
# ==========================================================================

st.write("")
st.divider()

if df is None or df.empty:
    st.info(
        "Todavia no hay datos cargados. Sube tu CSV arriba o selecciona un dataset "
        "de ejemplo para comenzar."
    )

    with st.expander("Que columnas debe tener mi CSV?"):
        st.markdown(
            """
            **Obligatorias**
            - `O`, `C`, `E`, `A`, `N` — scores OCEAN en el rango 1-5.
            - `edad` — entero.
            - `genero`, `estado`, `municipio` — texto.

            **Opcionales (para analisis psicometrico)**
            - `q1` a `q20` — respuestas Likert 1-5 individuales.
            - `arquetipo` — etiqueta descriptiva.
            - `submitted_at` o `timestamp` — fecha del registro.
            """
        )
else:
    section_head(
        title="Dataset activo",
        subtitle=f"Cargado desde `{st.session_state.get('csv_name', '?')}`",
        kicker="Estado",
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Registros", len(df))
    col_b.metric("Dimensiones", len(DIMENSIONS))

    if "edad" in df.columns:
        col_c.metric("Edad promedio", f"{df['edad'].mean():.1f}")

    if "arquetipo" in df.columns:
        col_d.metric("Arquetipos distintos", df["arquetipo"].nunique())

    st.write("")

    # Preview corto
    st.markdown("**Vista previa**")
    preview_cols = [c for c in ["submitted_at"] + DIMENSIONS +
                    ["arquetipo", "edad", "genero", "estado"]
                    if c in df.columns]
    st.dataframe(df[preview_cols].head(5), use_container_width=True, hide_index=True)

    st.write("")

    # CTA a siguientes pasos
    section_head(
        title="Siguientes pasos",
        subtitle="El flujo esta pensado para recorrerse en orden.",
        kicker="Paso 2",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        card(
            '<h4 style="margin-top:0;">Explorar</h4>'
            '<p style="margin: 6px 0 12px 0;">'
            'Filtra por edad, genero y estado. Revisa la estadistica descriptiva, '
            'correlaciones y la consistencia del instrumento.'
            '</p>'
        )
        st.page_link("pages/1_Exploracion.py", label="Ir a Exploracion")

    with col2:
        card(
            '<h4 style="margin-top:0;">Entrenar</h4>'
            '<p style="margin: 6px 0 12px 0;">'
            'Optimiza los hiperparametros de GMM con BIC, configura el modelo y '
            'entrenalo sobre los datos filtrados.'
            '</p>'
        )
        st.page_link("pages/2_Entrenamiento.py", label="Ir a Entrenamiento")

    with col3:
        card(
            '<h4 style="margin-top:0;">Analizar</h4>'
            '<p style="margin: 6px 0 12px 0;">'
            'Metricas, PCA en 2D, perfil por cluster y probabilidades de '
            'pertenencia (soft clustering).'
            '</p>'
        )
        st.page_link("pages/3_Resultados.py", label="Ir a Resultados")

    st.write("")
    col_x, _ = st.columns([1, 5])
    with col_x:
        if st.button(
            "Limpiar dataset", type="secondary", use_container_width=True,
            help="Descarta el dataset activo de la sesion. NO borra los archivos de ejemplo ni los modelos guardados.",
        ):
            clear_active_dataframe()
            # Rotamos la seed del uploader para reciclar el widget (evita re-carga)
            st.session_state["_uploader_seed"] = st.session_state.get("_uploader_seed", 0) + 1
            st.rerun()
