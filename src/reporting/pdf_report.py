"""Generación del PDF de resultados.

Uso matplotlib (no kaleido) para renderizar gráficas al PDF: es más rápido,
no requiere chromium, y está garantizado que ya está instalado por el resto
de dependencias del proyecto.
"""
from datetime import datetime
from io import BytesIO

import matplotlib
matplotlib.use("Agg")  # backend sin GUI
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
)

from src.data.loader import DIMENSIONS, DIMENSION_LABELS
from src.clustering import ALGORITHMS


# Paleta consistente con el theme mint de la app
PALETTE = {
    "primary": "#4dbb8a",
    "peach": "#ffb896",
    "sky": "#a8d5e8",
    "sunny": "#ffd88a",
    "muted": "#c9b8e0",
    "text": "#1f3a2e",
    "grid": "#e8f6ee",
}
CLUSTER_COLORS = [
    PALETTE["primary"], PALETTE["peach"], PALETTE["sky"], PALETTE["sunny"],
    PALETTE["muted"], "#d4c88a", "#8ba888", "#e8a598", "#a5c6d6", "#c8d97b",
]


def _build_styles():
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName="Helvetica",
                          fontSize=10, leading=13, alignment=TA_LEFT, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                        fontSize=14, leading=17, spaceBefore=14, spaceAfter=6, textColor=colors.black)
    title = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold",
                           fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=6)
    subtitle = ParagraphStyle("subtitle", parent=styles["Normal"], fontName="Helvetica",
                              fontSize=10, alignment=TA_CENTER, spaceAfter=14, textColor=colors.grey)
    caption = ParagraphStyle("caption", parent=styles["Normal"], fontName="Helvetica-Oblique",
                              fontSize=8, alignment=TA_CENTER, spaceAfter=10, textColor=colors.grey)
    return {"body": body, "h1": h1, "title": title, "subtitle": subtitle, "caption": caption}


def _table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _mpl_to_image_flow(fig, width_cm=16, height_cm=8):
    """Convierte figura matplotlib a Image de ReportLab en memoria."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width_cm * cm, height=height_cm * cm)


def _render_pca_scatter(X_2d, labels, variance):
    """PCA scatter con matplotlib, estilo mint."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    unique = sorted(set(labels), key=lambda x: (x == -1, x))
    for i, lab in enumerate(unique):
        mask = labels == lab
        if lab == -1:
            color = "#c0c0c0"
            name = "Ruido"
        else:
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            name = f"Cluster {int(lab)}"
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=color, edgecolors="white", linewidths=0.5,
            s=55, alpha=0.85, label=name,
        )
    ax.set_xlabel(f"PC1 · {variance['pc1']*100:.1f}% varianza", color=PALETTE["text"])
    ax.set_ylabel(f"PC2 · {variance['pc2']*100:.1f}% varianza", color=PALETTE["text"])
    ax.set_title("Proyección PCA en 2D", color=PALETTE["text"])
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#d0d0d0")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6)
    ax.tick_params(colors=PALETTE["text"])
    return fig


def _render_cluster_profiles(profile_df):
    """Bar chart agrupado con matplotlib. profile_df: cluster, n, O, C, E, A, N."""
    clusters = profile_df["cluster"].tolist()
    dims = DIMENSIONS
    labels_dim = [DIMENSION_LABELS[d] for d in dims]

    n_clusters = len(clusters)
    n_dims = len(dims)
    width = 0.8 / n_clusters
    x = np.arange(n_dims)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, c in enumerate(clusters):
        values = [profile_df.iloc[i][d] for d in dims]
        if c == -1:
            color = "#c0c0c0"
            name = "Ruido"
        else:
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            name = f"Cluster {int(c)}"
        offset = (i - (n_clusters - 1) / 2) * width
        ax.bar(x + offset, values, width, color=color, label=name,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_dim, color=PALETTE["text"])
    ax.set_ylabel("Promedio (1-5)", color=PALETTE["text"])
    ax.set_ylim(1, 5)
    ax.set_title("Perfil promedio por cluster", color=PALETTE["text"])
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#d0d0d0")
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.6)
    ax.tick_params(colors=PALETTE["text"])
    return fig


def generate_report(
    algorithm: str,
    params: dict,
    metrics: dict,
    profile_df,
    interpretations_df,
    training_time: float,
    n_records: int,
    X_2d=None,
    labels=None,
    variance=None,
) -> BytesIO:
    """Genera el PDF y lo retorna como BytesIO.

    Los parámetros de gráficas (X_2d, labels, variance) son opcionales;
    si están presentes se renderiza el scatter y el perfil por cluster.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Reporte de análisis no supervisado",
    )

    styles = _build_styles()
    story = []

    # Portada
    story.append(Paragraph("Reporte de Análisis No Supervisado", styles["title"]))
    story.append(Paragraph(
        f"Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        styles["subtitle"],
    ))

    # Resumen
    story.append(Paragraph("Resumen del modelo", styles["h1"]))
    story.append(_table([
        ["Campo", "Valor"],
        ["Algoritmo", ALGORITHMS[algorithm]["label"]],
        ["Registros analizados", str(n_records)],
        ["Tiempo de entrenamiento", f"{training_time} s"],
        ["Fecha", datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
    ], col_widths=[6 * cm, 10 * cm]))

    # Hiperparámetros
    story.append(Paragraph("Hiperparámetros utilizados", styles["h1"]))
    params_data = [["Parámetro", "Valor"]] + [[k, str(v)] for k, v in params.items()]
    story.append(_table(params_data, col_widths=[6 * cm, 10 * cm]))

    # Métricas
    story.append(Paragraph("Métricas de evaluación", styles["h1"]))
    metrics_rows = [
        ["Métrica", "Valor"],
        ["Clusters detectados", str(metrics["n_clusters"])],
        ["Outliers", str(metrics["n_outliers"])],
        ["Muestras totales", str(metrics["n_samples"])],
    ]
    if metrics.get("silhouette") is not None:
        metrics_rows.append(["Coeficiente de silueta", f"{metrics['silhouette']:.4f}"])
    if metrics.get("davies_bouldin") is not None:
        metrics_rows.append(["Índice Davies-Bouldin", f"{metrics['davies_bouldin']:.4f}"])
    if metrics.get("calinski_harabasz") is not None:
        metrics_rows.append(["Índice Calinski-Harabasz", f"{metrics['calinski_harabasz']:.2f}"])
    story.append(_table(metrics_rows, col_widths=[6 * cm, 10 * cm]))

    # PCA (opcional)
    if X_2d is not None and labels is not None and variance is not None:
        story.append(PageBreak())
        story.append(Paragraph("Proyección en 2D (PCA)", styles["h1"]))
        try:
            fig = _render_pca_scatter(np.asarray(X_2d), np.asarray(labels), variance)
            story.append(_mpl_to_image_flow(fig, width_cm=16, height_cm=9))
            story.append(Paragraph(
                "Reducción de dimensionalidad de 5D a 2D. Los ejes son "
                "componentes principales que capturan la mayor varianza.",
                styles["caption"],
            ))
        except Exception as e:
            story.append(Paragraph(
                f"No se pudo generar el scatter: {e}",
                styles["body"],
            ))

    # Perfil por cluster (tabla)
    story.append(Paragraph("Perfil promedio por cluster", styles["h1"]))
    profile_header = ["Cluster", "n"] + [DIMENSION_LABELS[d] for d in DIMENSIONS]
    profile_rows = [profile_header]
    for _, row in profile_df.iterrows():
        cid = f"Cluster {int(row['cluster'])}" if row["cluster"] != -1 else "Ruido"
        rowdata = [cid, str(int(row["n"]))] + [f"{row[d]:.2f}" for d in DIMENSIONS]
        profile_rows.append(rowdata)
    col_widths_profile = [2.5 * cm, 1.2 * cm] + [2.4 * cm] * 5
    story.append(_table(profile_rows, col_widths=col_widths_profile))

    # Perfil por cluster (gráfica)
    try:
        fig = _render_cluster_profiles(profile_df)
        story.append(Spacer(1, 8))
        story.append(_mpl_to_image_flow(fig, width_cm=16, height_cm=8))
    except Exception:
        pass

    # Interpretación
    story.append(Paragraph("Interpretación de los clusters", styles["h1"]))
    for _, row in interpretations_df.iterrows():
        cid = f"Cluster {int(row['cluster'])}" if row["cluster"] != -1 else "Ruido"
        story.append(Paragraph(f"<b>{cid}</b>: {row['interpretación']}", styles["body"]))

    doc.build(story)
    buffer.seek(0)
    return buffer
