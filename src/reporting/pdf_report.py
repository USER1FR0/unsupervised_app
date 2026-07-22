from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from src.data.loader import DIMENSIONS, DIMENSION_LABELS
from src.clustering import ALGORITHMS


def _build_styles():
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName="Helvetica",
                          fontSize=10, leading=13, alignment=TA_LEFT, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                        fontSize=14, leading=17, spaceBefore=14, spaceAfter=6, textColor=colors.black)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                        fontSize=11, leading=14, spaceBefore=10, spaceAfter=4, textColor=colors.black)
    title = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold",
                           fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=6)
    subtitle = ParagraphStyle("subtitle", parent=styles["Normal"], fontName="Helvetica",
                              fontSize=10, alignment=TA_CENTER, spaceAfter=14, textColor=colors.grey)
    return {"body": body, "h1": h1, "h2": h2, "title": title, "subtitle": subtitle}


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


def generate_report(
    algorithm: str,
    params: dict,
    metrics: dict,
    profile_df,
    interpretations_df,
    training_time: float,
    n_records: int,
) -> BytesIO:
    """Genera un reporte PDF y lo retorna como BytesIO listo para descarga."""
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
    resumen_data = [
        ["Campo", "Valor"],
        ["Algoritmo", ALGORITHMS[algorithm]["label"]],
        ["Registros analizados", str(n_records)],
        ["Tiempo de entrenamiento", f"{training_time} s"],
        ["Fecha", datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
    ]
    story.append(_table(resumen_data, col_widths=[6 * cm, 10 * cm]))

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

    # Perfil por cluster
    story.append(Paragraph("Perfil promedio por cluster", styles["h1"]))
    profile_header = ["Cluster", "n"] + [DIMENSION_LABELS[d] for d in DIMENSIONS]
    profile_rows = [profile_header]
    for _, row in profile_df.iterrows():
        cid = f"Cluster {int(row['cluster'])}" if row["cluster"] != -1 else "Ruido"
        rowdata = [cid, str(int(row["n"]))] + [f"{row[d]:.2f}" for d in DIMENSIONS]
        profile_rows.append(rowdata)
    col_widths_profile = [2.5 * cm, 1.2 * cm] + [2.4 * cm] * 5
    story.append(_table(profile_rows, col_widths=col_widths_profile))

    # Interpretación
    story.append(Paragraph("Interpretación de los clusters", styles["h1"]))
    for _, row in interpretations_df.iterrows():
        cid = f"Cluster {int(row['cluster'])}" if row["cluster"] != -1 else "Ruido"
        story.append(Paragraph(f"<b>{cid}</b>: {row['interpretación']}", styles["body"]))

    doc.build(story)
    buffer.seek(0)
    return buffer