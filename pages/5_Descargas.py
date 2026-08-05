import streamlit as st
import numpy as np
from datetime import datetime

from src.data.loader import DIMENSIONS
from src.data.preprocessing import scale_dimensions
from src.evaluation.interpretation import (
    compute_cluster_profile,
    build_interpretations,
)
from src.reporting.pdf_report import generate_report
from src.visualization.pca import project_2d, variance_explained
from src.ui.theme import apply_global_style, render_sidebar, section_head

st.set_page_config(page_title="Descargas", layout="wide")
apply_global_style()
render_sidebar()

st.markdown("## Descargas")
st.caption("Exporta los datos filtrados, los resultados y un reporte en PDF.")
st.divider()

# ---------- 1. Datos filtrados ----------
section_head(
    title="Datos filtrados",
    subtitle="Registros despues de aplicar los filtros de Exploracion.",
    kicker="1 · CSV",
)

if "df_filtered" not in st.session_state or st.session_state["df_filtered"].empty:
    st.info("No hay datos filtrados. Ve a **Exploracion** primero.")
    st.page_link("pages/1_Exploracion.py", label="Ir a Exploracion")
else:
    df = st.session_state["df_filtered"]
    csv_filtered = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=f"Descargar {len(df)} registros filtrados",
        data=csv_filtered,
        file_name=f"datos_filtrados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.write("")
st.divider()

# ---------- 2. Datos con clusters ----------
section_head(
    title="Datos con etiquetas de cluster",
    subtitle="Registros filtrados + columna 'cluster' + 'prob_maxima' del modelo actual.",
    kicker="2 · CSV",
)

if "df_labeled" not in st.session_state:
    st.info("No hay modelo entrenado con resultados. Ve a **Resultados** primero.")
    st.page_link("pages/3_Resultados.py", label="Ir a Resultados")
else:
    df_labeled = st.session_state["df_labeled"]
    csv_labeled = df_labeled.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=f"Descargar {len(df_labeled)} registros clusterizados",
        data=csv_labeled,
        file_name=f"datos_clusterizados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.write("")
st.divider()

# ---------- 3. Reporte PDF ----------
section_head(
    title="Reporte PDF completo",
    subtitle="Metricas, hiperparametros, PCA 2D y perfil por cluster en un documento.",
    kicker="3 · PDF",
)

required = ["current_model", "current_metrics", "current_algorithm",
            "current_params", "df_filtered", "current_training_time"]
if not all(k in st.session_state for k in required):
    st.info("Necesitas un modelo entrenado. Ve a **Entrenamiento** primero.")
    st.page_link("pages/2_Entrenamiento.py", label="Ir a Entrenamiento")
else:
    if st.button("Generar reporte PDF", type="primary", use_container_width=True):
        try:
            with st.spinner("Generando reporte..."):
                model = st.session_state["current_model"]
                df = st.session_state["df_filtered"].reset_index(drop=True)
                profile = compute_cluster_profile(df, model.labels)
                global_means = {d: float(df[d].mean()) for d in DIMENSIONS}
                interpretations = build_interpretations(profile, global_means)

                X_scaled = st.session_state.get("X_scaled")
                if X_scaled is None:
                    X_scaled, _ = scale_dimensions(df)
                X_2d, pca = project_2d(X_scaled)
                variance = variance_explained(pca)

                pdf_buffer = generate_report(
                    algorithm=st.session_state["current_algorithm"],
                    params=st.session_state["current_params"],
                    metrics=st.session_state["current_metrics"],
                    profile_df=profile,
                    interpretations_df=interpretations,
                    training_time=st.session_state["current_training_time"],
                    n_records=len(df),
                    X_2d=X_2d,
                    labels=np.asarray(model.labels),
                    variance=variance,
                )
            st.session_state["pdf_buffer"] = pdf_buffer.getvalue()
            st.success("Reporte generado.")
        except Exception as e:
            st.error(f"Error al generar el PDF: {e}")

    if "pdf_buffer" in st.session_state:
        st.download_button(
            label="Descargar reporte PDF",
            data=st.session_state["pdf_buffer"],
            file_name=f"reporte_gmm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
