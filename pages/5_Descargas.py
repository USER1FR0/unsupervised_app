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
from src.clustering import ALGORITHMS

st.set_page_config(page_title="Descargas", page_icon=":material/download:", layout="wide")

st.title("Descargas")
st.caption("Exporta datos y resultados para uso externo")

# ---------- CSV con datos filtrados ----------
st.markdown("### 1. Datos filtrados")
st.caption("Registros después de aplicar los filtros de la página de Exploración.")

if "df_filtered" not in st.session_state or st.session_state["df_filtered"].empty:
    st.info("No hay datos filtrados. Ve a **Exploración** primero.")
else:
    df = st.session_state["df_filtered"]
    csv_filtered = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=f"Descargar {len(df)} registros filtrados (CSV)",
        data=csv_filtered,
        file_name=f"datos_filtrados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.write("")
st.divider()

# ---------- CSV con etiquetas de cluster ----------
st.markdown("### 2. Datos con etiquetas de cluster")
st.caption("Registros filtrados + la columna 'cluster' del modelo actual.")

if "df_labeled" not in st.session_state:
    st.info("No hay modelo entrenado con resultados. Ve a **Resultados** primero.")
else:
    df_labeled = st.session_state["df_labeled"]
    csv_labeled = df_labeled.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=f"Descargar {len(df_labeled)} registros con clusters (CSV)",
        data=csv_labeled,
        file_name=f"datos_clusterizados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.write("")
st.divider()

# ---------- Reporte PDF ----------
st.markdown("### 3. Reporte PDF completo")
st.caption("Documento con métricas, hiperparámetros, PCA en 2D y perfil por cluster.")

required = ["current_model", "current_metrics", "current_algorithm",
            "current_params", "df_filtered", "current_training_time"]
if not all(k in st.session_state for k in required):
    st.info("Necesitas un modelo entrenado. Ve a **Entrenamiento** primero.")
else:
    if st.button("Generar reporte PDF", type="primary", use_container_width=True):
        with st.spinner("Generando reporte..."):
            model = st.session_state["current_model"]
            df = st.session_state["df_filtered"].reset_index(drop=True)
            profile = compute_cluster_profile(df, model.labels)
            global_means = {d: float(df[d].mean()) for d in DIMENSIONS}
            interpretations = build_interpretations(profile, global_means)

            # Preparar datos para las gráficas (rápido, matplotlib puro)
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

    if "pdf_buffer" in st.session_state:
        algo_key = st.session_state["current_algorithm"]
        st.download_button(
            label="Descargar reporte PDF",
            data=st.session_state["pdf_buffer"],
            file_name=f"reporte_{algo_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
