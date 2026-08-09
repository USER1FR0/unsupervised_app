import streamlit as st
import pandas as pd

from src.data.loader import DIMENSIONS, DIMENSION_LABELS
from src.data.preprocessing import drop_null_dimensions, count_nulls_by_column
from src.stats.descriptive import (
    describe_all_dimensions,
    correlation_matrix,
    internal_consistency,
    cronbach_alpha,
)
from src.visualization.charts import (
    histogram,
    boxplot,
    heatmap_correlation,
    bar_categorical,
    bar_dimensions_mean,
)
from src.ui.theme import apply_global_style, render_sidebar, section_head, require_dataset

st.set_page_config(page_title="Exploracion", layout="wide")
apply_global_style()
render_sidebar()

st.markdown("## Exploracion de datos")
st.caption(
    f"Filtros, estadistica descriptiva y correlaciones · "
    f"Archivo: **{st.session_state.get('csv_name', '?')}**"
)
st.divider()

df_full = require_dataset()

# Reporte de nulos
nulls = count_nulls_by_column(df_full)
if nulls:
    with st.expander(f"Se detectaron {sum(nulls.values())} valores nulos"):
        for col, n in nulls.items():
            st.write(f"- **{col}**: {n} valores nulos")

df_clean = drop_null_dimensions(df_full)

# ---------- Filtros ----------
section_head(title="Filtros", kicker="Segmentacion")

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    edad_min, edad_max = int(df_clean["edad"].min()), int(df_clean["edad"].max())
    if edad_min == edad_max:
        edad_range = (edad_min, edad_max)
        st.caption(f"Rango de edad: {edad_min} (unico valor)")
    else:
        edad_range = st.slider(
            "Rango de edad",
            min_value=edad_min, max_value=edad_max,
            value=(edad_min, edad_max),
            help=(
                "Segmenta el analisis por rango de edad. El filtro se propaga a todas las paginas: "
                "Entrenamiento y Resultados usaran solo el subconjunto filtrado."
            ),
        )

with col_f2:
    generos_disponibles = sorted(df_clean["genero"].unique().tolist())
    generos = st.multiselect(
        "Genero", options=generos_disponibles, default=generos_disponibles,
        help="Segmenta por genero. El filtro se aplica en cascada al resto del pipeline.",
    )

with col_f3:
    estados_disponibles = sorted(df_clean["estado"].unique().tolist())
    estados = st.multiselect(
        "Estado", options=estados_disponibles, default=estados_disponibles,
        help="Segmenta por estado de residencia. Util para analizar subgrupos regionales.",
    )

df = df_clean[
    (df_clean["edad"].between(edad_range[0], edad_range[1]))
    & (df_clean["genero"].isin(generos))
    & (df_clean["estado"].isin(estados))
].reset_index(drop=True)

st.write("")

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric(
    "Filtrados", len(df),
    help="Registros que cumplen los filtros de edad, genero y estado seleccionados.",
)
col_m2.metric(
    "Totales", len(df_full),
    help="Total de registros en el dataset cargado antes de aplicar filtros.",
)
col_m3.metric(
    "Excluidos", len(df_full) - len(df),
    help="Registros descartados por no cumplir los filtros o por tener nulos en dimensiones OCEAN.",
)

if df.empty:
    st.warning("No hay registros que cumplan los filtros seleccionados.")
    st.stop()

st.session_state["df_filtered"] = df

st.write("")
st.divider()

# ---------- Tabs ----------
tab_tabla, tab_stats, tab_dist, tab_corr = st.tabs([
    "Tabla",
    "Estadistica descriptiva",
    "Distribucion",
    "Correlacion",
])

with tab_tabla:
    section_head(title="Registros filtrados",
                 subtitle=f"{len(df)} registros, {len(df.columns)} columnas")

    preferred = [
        "submitted_at", "O", "C", "E", "A", "N", "arquetipo",
        "edad", "genero", "estado", "municipio",
    ]
    default_cols = [c for c in preferred if c in df.columns]

    columnas_display = st.multiselect(
        "Columnas a mostrar",
        options=df.columns.tolist(),
        default=default_cols,
        help="Elige que columnas mostrar en la tabla. Util para inspeccion visual y deteccion de respuestas ruidosas.",
    )

    st.dataframe(
        df[columnas_display] if columnas_display else df,
        use_container_width=True, hide_index=True, height=480,
    )

with tab_stats:
    section_head(title="Estadistica descriptiva por dimension",
                 subtitle="Media, mediana, desviacion, cuartiles y extremos")

    stats_df = describe_all_dimensions(df)
    stats_df["variable"] = stats_df["variable"].map(
        {d: f"{d} · {DIMENSION_LABELS[d]}" for d in DIMENSIONS}
    )
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.write("")

    col_left, col_right = st.columns(2)

    with col_left:
        section_head(title="Consistencia interna",
                     subtitle="Correlacion promedio entre items de una misma dimension")
        consistency = internal_consistency(df)
        if not consistency.empty:
            consistency["dimensión"] = consistency["dimensión"].map(
                {d: f"{d} · {DIMENSION_LABELS[d]}" for d in DIMENSIONS}
            )
            st.dataframe(consistency, use_container_width=True, hide_index=True)
        else:
            st.info("Requiere las columnas q1..q20 en el CSV.")

    with col_right:
        section_head(title="Alfa de Cronbach",
                     subtitle="Metrica psicometrica estandar (>= 0.7 aceptable)")
        alpha_df = cronbach_alpha(df)
        if not alpha_df.empty:
            alpha_df["dimensión"] = alpha_df["dimensión"].map(
                {d: f"{d} · {DIMENSION_LABELS[d]}" for d in DIMENSIONS}
            )
            st.dataframe(alpha_df, use_container_width=True, hide_index=True)
        else:
            st.info("Requiere las columnas q1..q20 en el CSV.")

    st.write("")
    section_head(title="Promedio general por dimension")
    try:
        means = {d: df[d].mean() for d in DIMENSIONS}
        st.plotly_chart(bar_dimensions_mean(means), use_container_width=True)
    except Exception as e:
        st.error(f"No se pudo graficar los promedios: {e}")

with tab_dist:
    section_head(title="Distribucion por variable")

    var = st.selectbox(
        "Selecciona una dimension",
        options=DIMENSIONS,
        format_func=lambda d: f"{d} · {DIMENSION_LABELS[d]}",
        help=(
            "Dimension OCEAN a visualizar. Histograma muestra frecuencia por rango de valores. "
            "Boxplot muestra Q1, mediana, Q3, bigotes hasta 1.5·IQR y outliers."
        ),
    )

    col_h, col_b = st.columns(2)
    with col_h:
        st.plotly_chart(
            histogram(df[var].values, f"Histograma · {DIMENSION_LABELS[var]}", "Puntaje"),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            boxplot(df[var].values, f"Boxplot · {DIMENSION_LABELS[var]}", "Puntaje"),
            use_container_width=True,
        )

    st.write("")
    section_head(title="Distribuciones categoricas")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.plotly_chart(
            bar_categorical(df["genero"].value_counts(), "Distribucion por genero"),
            use_container_width=True,
        )
    with col_c2:
        st.plotly_chart(
            bar_categorical(df["estado"].value_counts(), "Distribucion por estado"),
            use_container_width=True,
        )

with tab_corr:
    section_head(title="Matriz de correlacion",
                 subtitle="Correlacion de Pearson entre las 5 dimensiones OCEAN")
    corr = correlation_matrix(df)
    st.plotly_chart(heatmap_correlation(corr), use_container_width=True)

    st.markdown(
        '<div class="card-accent">'
        '<strong>Interpretacion rapida:</strong> valores positivos (azul) indican que '
        'las dimensiones suben juntas; valores negativos (naranja) indican relacion '
        'inversa; valores cercanos a 0 indican independencia.'
        '</div>',
        unsafe_allow_html=True,
    )
