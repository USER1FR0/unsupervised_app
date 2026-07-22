import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


PALETTE = {
    "primary": "#4dbb8a",
    "secondary": "#b8dcc8",
    "peach": "#ffb896",
    "sky": "#a8d5e8",
    "sunny": "#ffd88a",
    "text": "#1f3a2e",
    "muted": "#6a8578",
    "grid": "#e8f6ee",
}

BASE_LAYOUT = {
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
    "font": {"family": "Inter, sans-serif", "color": PALETTE["text"], "size": 13},
    "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
}


def _apply_base(fig: go.Figure, title: str = None, height: int = 340) -> go.Figure:
    fig.update_layout(**BASE_LAYOUT, title=title, height=height)
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False)
    return fig


def histogram(values, title: str, xlabel: str = None) -> go.Figure:
    fig = px.histogram(
        x=values,
        nbins=20,
        color_discrete_sequence=[PALETTE["primary"]],
    )
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


def heatmap_correlation(matrix: pd.DataFrame, title: str = "Matriz de correlación") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale=[[0, PALETTE["peach"]], [0.5, "white"], [1, PALETTE["primary"]]],
        zmin=-1, zmax=1,
        text=matrix.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 12},
        colorbar={"title": "r"},
    ))
    return _apply_base(fig, title, height=380)


def bar_categorical(counts: pd.Series, title: str, xlabel: str = None) -> go.Figure:
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color_discrete_sequence=[PALETTE["primary"]],
    )
    fig.update_layout(xaxis_title=xlabel, yaxis_title="Cantidad", showlegend=False)
    return _apply_base(fig, title)


def bar_dimensions_mean(means: dict, title: str = "Promedio por dimensión") -> go.Figure:
    from src.data.loader import DIMENSION_LABELS
    labels = [DIMENSION_LABELS[d] for d in means.keys()]
    values = list(means.values())
    fig = px.bar(
        x=labels,
        y=values,
        color_discrete_sequence=[PALETTE["primary"]],
    )
    fig.update_layout(xaxis_title=None, yaxis_title="Promedio (1-5)", showlegend=False)
    fig.update_yaxes(range=[1, 5])
    return _apply_base(fig, title)

def elbow_plot(k_values, y_values, suggested_k, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_values, y=y_values,
        mode="lines+markers",
        line=dict(color=PALETTE["primary"], width=2),
        marker=dict(size=8),
        name=y_label,
    ))
    if suggested_k is not None:
        fig.add_vline(
            x=suggested_k,
            line_dash="dash",
            line_color=PALETTE["peach"],
            annotation_text=f"Sugerido: k={suggested_k}",
            annotation_position="top right",
        )
    fig.update_layout(xaxis_title="k", yaxis_title=y_label, showlegend=False)
    return _apply_base(fig, title)


def k_distance_plot(distances, k, suggested_eps, title: str = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(distances))),
        y=distances,
        mode="lines",
        line=dict(color=PALETTE["primary"], width=2),
    ))
    fig.add_hline(
        y=suggested_eps,
        line_dash="dash",
        line_color=PALETTE["peach"],
        annotation_text=f"eps sugerido: {suggested_eps}",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="Puntos ordenados",
        yaxis_title=f"Distancia al {k}º vecino",
        showlegend=False,
    )
    return _apply_base(fig, title or f"Gráfica de k-distancias (k={k})")


def dendrogram_plot(linkage_matrix, title: str = "Dendrograma") -> go.Figure:
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig_mpl, ax = plt.subplots(figsize=(10, 4))
    dendrogram(linkage_matrix, ax=ax, color_threshold=0, above_threshold_color=PALETTE["primary"])
    ax.set_xlabel("Observaciones")
    ax.set_ylabel("Distancia")
    ax.set_title(title)
    plt.tight_layout()

    # Retornamos figura matplotlib; Streamlit la muestra con st.pyplot
    return fig_mpl

def pca_scatter(X_2d: np.ndarray, labels: np.ndarray, variance: dict, title: str = None) -> go.Figure:
    """Scatter 2D coloreado por cluster."""
    df_plot = pd.DataFrame({
        "PC1": X_2d[:, 0],
        "PC2": X_2d[:, 1],
        "Cluster": [f"Cluster {int(l)}" if l != -1 else "Ruido" for l in labels],
    })

    color_map = {}
    palette_cycle = [
        PALETTE["primary"], PALETTE["peach"], PALETTE["sky"],
        PALETTE["sunny"], PALETTE["secondary"], PALETTE["muted"],
        "#c9b8e0", "#d4c88a", "#8ba888", "#e8a598",
    ]
    unique_labels = sorted(df_plot["Cluster"].unique(), key=lambda x: (x == "Ruido", x))
    for i, cluster in enumerate(unique_labels):
        if cluster == "Ruido":
            color_map[cluster] = "#c0c0c0"
        else:
            color_map[cluster] = palette_cycle[i % len(palette_cycle)]

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
    return _apply_base(fig, title or "Proyección PCA en 2D", height=460)


def cluster_profile_bars(profile_df: pd.DataFrame, title: str = "Perfil promedio por cluster") -> go.Figure:
    """Barras agrupadas del perfil OCEAN por cluster."""
    from src.data.loader import DIMENSION_LABELS, DIMENSIONS

    profile_df = profile_df.copy()
    profile_df["cluster_label"] = profile_df["cluster"].apply(
        lambda x: f"Cluster {int(x)}" if x != -1 else "Ruido"
    )

    melted = profile_df.melt(
        id_vars=["cluster_label"],
        value_vars=DIMENSIONS,
        var_name="Dimensión",
        value_name="Promedio",
    )
    melted["Dimensión"] = melted["Dimensión"].map(DIMENSION_LABELS)

    palette_cycle = [
        PALETTE["primary"], PALETTE["peach"], PALETTE["sky"],
        PALETTE["sunny"], PALETTE["secondary"], PALETTE["muted"],
        "#c9b8e0", "#d4c88a", "#8ba888", "#e8a598",
    ]
    clusters_sorted = sorted(melted["cluster_label"].unique(), key=lambda x: (x == "Ruido", x))
    color_map = {}
    for i, c in enumerate(clusters_sorted):
        color_map[c] = "#c0c0c0" if c == "Ruido" else palette_cycle[i % len(palette_cycle)]

    fig = px.bar(
        melted, x="Dimensión", y="Promedio",
        color="cluster_label", barmode="group",
        color_discrete_map=color_map,
    )
    fig.update_layout(legend_title_text="", yaxis_range=[1, 5])
    return _apply_base(fig, title, height=380)