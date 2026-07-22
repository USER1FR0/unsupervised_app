import streamlit as st
import pandas as pd
from src.data.loader import load_dataframe, DIMENSIONS, DIMENSION_LABELS
from src.data.preprocessing import drop_null_dimensions, count_nulls_by_column
from src.stats.descriptive import (
    describe_all_dimensions,
    correlation_matrix,
    internal_consistency,
)
from src.visualization.charts import (
    histogram,
    boxplot,
    heatmap_correlation,
    bar_categorical,
    bar_dimensions_mean,
)

st.set_page_config(page_title="Exploración", page_icon="📊", layout="wide")

st.title("📊 Exploración de datos")
st.caption("Visualización, filtros y estadística descriptiva")

# ---------- Carga ----------
df_full = load_dataframe()

if df_full.empty:
    st.warning("No hay datos cargados. Sincroniza desde el panel lateral en la página principal.")
    st.stop()

# Reporte de nulos
nulls = count_nulls_by_column(df_full)
if nulls:
    with st.expander("⚠️ Se detectaron valores nulos", expanded=False):
        for col, n in nulls.items():
            st.write(f"- **{col}**: {n} valores nulos")

# Eliminar nulos en dimensiones
df_clean = drop_null_dimensions(df_full)

# ---------- Filtros ----------
st.markdown("### Filtros")

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    edad_min, edad_max = int(df_clean["edad"].min()), int(df_clean["edad"].max())
    edad_range = st.slider(
        "Rango de edad",
        min_value=edad_min,
        max_value=edad_max,
        value=(edad_min, edad_max),
    )

with col_f2:
    generos_disponibles = sorted(df_clean["genero"].unique().tolist())
    generos = st.multiselect(
        "Género",
        options=generos_disponibles,
        default=generos_disponibles,
    )

with col_f3:
    estados_disponibles = sorted(df_clean["estado"].unique().tolist())
    estados = st.multiselect(
        "Estado",
        options=estados_disponibles,
        default=estados_disponibles,
    )

col_r1, col_r2 = st.columns([1, 5])
with col_r1:
    if st.button("Reiniciar filtros", use_container_width=True):
        st.rerun()

# Aplicar filtros
df = df_clean[
    (df_clean["edad"].between(edad_range[0], edad_range[1]))
    & (df_clean["genero"].isin(generos))
    & (df_clean["estado"].isin(estados))
].reset_index(drop=True)

st.write("")
st.info(f"**{len(df)}** registros después de aplicar filtros (de {len(df_clean)} totales)")

if df.empty:
    st.warning("No hay registros que cumplan los filtros seleccionados.")
    st.stop()

# Guardar en session state para usar en otras páginas
st.session_state["df_filtered"] = df

st.write("")
st.divider()

# ---------- Tabs ----------
tab_tabla, tab_stats, tab_dist, tab_corr = st.tabs([
    "Tabla de datos",
    "Estadística descriptiva",
    "Distribución por variable",
    "Correlación",
])

with tab_tabla:
    st.markdown("#### Registros")
    st.caption(f"Total: {len(df)} · Columnas: {len(df.columns)}")

    columnas_display = st.multiselect(
        "Columnas a mostrar",
        options=df.columns.tolist(),
        default=["submitted_at", "O", "C", "E", "A", "N", "arquetipo", "edad", "genero", "estado", "municipio"],
    )

    st.dataframe(
        df[columnas_display] if columnas_display else df,
        use_container_width=True,
        hide_index=True,
        height=480,
    )

with tab_stats:
    st.markdown("#### Estadística descriptiva por dimensión")
    st.caption("Cálculos propios: media, mediana, desviación estándar, mínimo, Q1, Q3 y máximo.")

    stats_df = describe_all_dimensions(df)
    stats_df["variable"] = stats_df["variable"].map(
        {d: f"{d} · {DIMENSION_LABELS[d]}" for d in DIMENSIONS}
    )
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.write("")
    st.markdown("#### Consistencia interna del instrumento")
    st.caption(
        "Correlación promedio entre los ítems de cada dimensión. "
        "Valores cercanos a 1 indican que los ítems miden lo mismo."
    )

    consistency = internal_consistency(df)
    if not consistency.empty:
        consistency["dimensión"] = consistency["dimensión"].map(
            {d: f"{d} · {DIMENSION_LABELS[d]}" for d in DIMENSIONS}
        )
        st.dataframe(consistency, use_container_width=True, hide_index=True)
    else:
        st.info("No se pudo calcular (no hay respuestas crudas disponibles).")

    st.write("")
    st.markdown("#### Promedio general por dimensión")
    means = {d: df[d].mean() for d in DIMENSIONS}
    st.plotly_chart(bar_dimensions_mean(means), use_container_width=True)

with tab_dist:
    st.markdown("#### Distribución por variable")

    var = st.selectbox(
        "Selecciona una dimensión",
        options=DIMENSIONS,
        format_func=lambda d: f"{d} · {DIMENSION_LABELS[d]}",
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
    st.markdown("#### Distribuciones categóricas")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.plotly_chart(
            bar_categorical(
                df["genero"].value_counts(),
                "Distribución por género",
            ),
            use_container_width=True,
        )
    with col_c2:
        st.plotly_chart(
            bar_categorical(
                df["estado"].value_counts(),
                "Distribución por estado",
            ),
            use_container_width=True,
        )

with tab_corr:
    st.markdown("#### Matriz de correlación entre dimensiones")
    st.caption("Correlación de Pearson calculada con implementación propia.")

    corr = correlation_matrix(df)
    st.plotly_chart(heatmap_correlation(corr), use_container_width=True)

    st.write("")
    st.markdown("**Interpretación rápida**")
    st.caption(
        "Valores positivos (verde) indican que las dimensiones tienden a subir juntas. "
        "Valores negativos (durazno) indican que cuando una sube, la otra baja. "
        "Valores cercanos a 0 indican independencia estadística."
    )