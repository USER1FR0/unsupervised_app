import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


PALETTE = {
    "primary": "#0ea5e9",
    "primary_hover": "#0284c7",
    "primary_soft": "#f0f9ff",
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "text": "#0f172a",
    "text_secondary": "#475569",
    "text_muted": "#94a3b8",
    "border_soft": "#e2e8f0",
    "border": "#cbd5e1",
    "bg": "#ffffff",
    "bg_elevated": "#f8fafc",
    "grid": "#eef2f6",
    "noise": "#94a3b8",
}

# Serie categórica para clusters (tech / neutral)
SERIES_PALETTE = [
    "#0ea5e9",  # sky
    "#6366f1",  # indigo
    "#10b981",  # emerald
    "#f59e0b",  # amber
    "#8b5cf6",  # violet
    "#14b8a6",  # teal
    "#ef4444",  # rose
    "#3b82f6",  # blue
    "#0891b2",  # cyan
    "#a855f7",  # purple
]

BASE_LAYOUT = {
    "plot_bgcolor": PALETTE["bg"],
    "paper_bgcolor": PALETTE["bg"],
    "font": {"family": "Inter, system-ui, sans-serif", "color": PALETTE["text"], "size": 13},
    "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
}


def _apply_base(fig: go.Figure, title: str = None, height: int = 320) -> go.Figure:
    fig.update_layout(**BASE_LAYOUT, title=title, height=height)
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False,
                     linecolor=PALETTE["border_soft"])
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False,
                     linecolor=PALETTE["border_soft"])
    return fig


def _cluster_color_map(labels_iterable) -> dict:
    unique = sorted(set(labels_iterable), key=lambda x: (x == "Ruido", x))
    color_map = {}
    idx = 0
    for lbl in unique:
        if lbl == "Ruido":
            color_map[lbl] = PALETTE["noise"]
        else:
            color_map[lbl] = SERIES_PALETTE[idx % len(SERIES_PALETTE)]
            idx += 1
    return color_map


def histogram(values, title: str, xlabel: str = None) -> go.Figure:
    fig = px.histogram(x=values, nbins=20, color_discrete_sequence=[PALETTE["primary"]])
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    fig.update_layout(xaxis_title=xlabel, yaxis_title="Frecuencia", showlegend=False)
    return _apply_base(fig, title)


def boxplot(values, title: str, ylabel: str = None) -> go.Figure:
    fig = go.Figure(go.Box(
        y=values,
        marker_color=PALETTE["primary"],
        boxmean="sd",
        name="",
    ))
    fig.update_layout(yaxis_title=ylabel, showlegend=False)
    return _apply_base(fig, title)


def heatmap_correlation(matrix: pd.DataFrame, title: str = "Matriz de correlacion") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale=[[0, "#ea580c"], [0.5, "#ffffff"], [1, PALETTE["primary"]]],
        zmin=-1, zmax=1,
        text=matrix.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 12, "color": PALETTE["text"]},
        colorbar={"title": "r", "outlinewidth": 0},
    ))
    return _apply_base(fig, title, height=380)


def bar_categorical(counts: pd.Series, title: str, xlabel: str = None) -> go.Figure:
    fig = px.bar(x=counts.index, y=counts.values, color_discrete_sequence=[PALETTE["primary"]])
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    fig.update_layout(xaxis_title=xlabel, yaxis_title="Cantidad", showlegend=False)
    return _apply_base(fig, title)


def bar_dimensions_mean(means: dict, title: str = "Promedio por dimension") -> go.Figure:
    from src.data.loader import DIMENSION_LABELS
    labels = [DIMENSION_LABELS[d] for d in means.keys()]
    values = list(means.values())
    fig = px.bar(x=labels, y=values, color_discrete_sequence=[PALETTE["primary"]])
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    fig.update_layout(xaxis_title=None, yaxis_title="Promedio (1-5)", showlegend=False)
    fig.update_yaxes(range=[1, 5])
    return _apply_base(fig, title)


def elbow_plot(k_values, y_values, suggested_k, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_values, y=y_values,
        mode="lines+markers",
        line=dict(color=PALETTE["primary"], width=2),
        marker=dict(size=8, line=dict(color="white", width=1)),
        name=y_label,
    ))
    if suggested_k is not None:
        fig.add_vline(
            x=suggested_k,
            line_dash="dash",
            line_color=PALETTE["warning"],
            annotation_text=f"Sugerido: k={suggested_k}",
            annotation_position="top right",
            annotation_font_color=PALETTE["warning"],
        )
    fig.update_layout(xaxis_title="k", yaxis_title=y_label, showlegend=False)
    return _apply_base(fig, title)


def k_distance_plot(distances, k, suggested_eps, title: str = None,
                     eps_aggressive=None, eps_conservative=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(distances))),
        y=distances,
        mode="lines",
        line=dict(color=PALETTE["primary"], width=2),
    ))
    if eps_aggressive is not None:
        fig.add_hline(y=eps_aggressive, line_dash="dot", line_color=SERIES_PALETTE[2],
                      annotation_text=f"agresivo: {eps_aggressive}",
                      annotation_position="bottom right")
    fig.add_hline(y=suggested_eps, line_dash="dash", line_color=PALETTE["warning"],
                  annotation_text=f"moderado: {suggested_eps}",
                  annotation_position="top right")
    if eps_conservative is not None:
        fig.add_hline(y=eps_conservative, line_dash="dot", line_color=PALETTE["text_muted"],
                      annotation_text=f"conservador: {eps_conservative}",
                      annotation_position="top left")
    fig.update_layout(
        xaxis_title="Puntos ordenados",
        yaxis_title=f"Distancia al {k}o vecino",
        showlegend=False,
    )
    return _apply_base(fig, title or f"Grafica de k-distancias (k={k})")


def dendrogram_plot(linkage_matrix, title: str = "Dendrograma"):
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig_mpl, ax = plt.subplots(figsize=(10, 4))
    dendrogram(linkage_matrix, ax=ax, color_threshold=0,
               above_threshold_color=PALETTE["primary"])
    ax.set_xlabel("Observaciones")
    ax.set_ylabel("Distancia")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig_mpl


def pca_scatter(X_2d: np.ndarray, labels: np.ndarray, variance: dict,
                title: str = None) -> go.Figure:
    """Scatter 2D coloreado por cluster."""
    df_plot = pd.DataFrame({
        "PC1": X_2d[:, 0],
        "PC2": X_2d[:, 1],
        "Cluster": [f"Cluster {int(l)}" if l != -1 else "Ruido" for l in labels],
    })
    color_map = _cluster_color_map(df_plot["Cluster"].tolist())

    fig = px.scatter(
        df_plot, x="PC1", y="PC2", color="Cluster",
        color_discrete_map=color_map,
        opacity=0.85,
    )
    fig.update_traces(marker=dict(size=9, line=dict(color="white", width=1)))
    fig.update_layout(
        xaxis_title=f"PC1 · {variance['pc1']*100:.1f}% varianza",
        yaxis_title=f"PC2 · {variance['pc2']*100:.1f}% varianza",
        legend_title_text="",
    )
    return _apply_base(fig, title or "Proyeccion PCA en 2D", height=420)


def cluster_profile_bars(profile_df: pd.DataFrame,
                         title: str = "Perfil promedio por cluster") -> go.Figure:
    """Barras agrupadas del perfil OCEAN por cluster."""
    from src.data.loader import DIMENSION_LABELS, DIMENSIONS

    profile_df = profile_df.copy()
    profile_df["cluster_label"] = profile_df["cluster"].apply(
        lambda x: f"Cluster {int(x)}" if x != -1 else "Ruido"
    )

    melted = profile_df.melt(
        id_vars=["cluster_label"],
        value_vars=DIMENSIONS,
        var_name="Dimension",
        value_name="Promedio",
    )
    melted["Dimension"] = melted["Dimension"].map(DIMENSION_LABELS)

    color_map = _cluster_color_map(melted["cluster_label"].tolist())

    fig = px.bar(
        melted, x="Dimension", y="Promedio",
        color="cluster_label", barmode="group",
        color_discrete_map=color_map,
    )
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    fig.update_layout(legend_title_text="", yaxis_range=[1, 5])
    return _apply_base(fig, title, height=360)
