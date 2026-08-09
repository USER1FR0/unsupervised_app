"""UI global: paleta tech (slate + sky), CSS injectado y sidebar de navegacion."""
import streamlit as st


GLOBAL_CSS = """
<style>
:root {
    /* Fondos */
    --bg:            #ffffff;
    --bg-elevated:   #f8fafc;
    --bg-hover:      #f1f5f9;

    /* Texto */
    --text:          #0f172a;
    --text-2:        #475569;
    --text-3:        #94a3b8;
    --text-inverse:  #f1f5f9;

    /* Bordes */
    --border:        #e2e8f0;
    --border-strong: #cbd5e1;

    /* Accent */
    --primary:       #0ea5e9;
    --primary-hover: #0284c7;
    --primary-soft:  #f0f9ff;
    --primary-ring:  rgba(14, 165, 233, 0.15);

    /* Secundarios */
    --indigo:        #6366f1;
    --emerald:       #10b981;
    --amber:         #f59e0b;
    --rose:          #ef4444;

    /* Sidebar */
    --sidebar-bg:    #0f172a;
    --sidebar-border:#1e293b;
    --sidebar-text:  #cbd5e1;
    --sidebar-text-strong: #f8fafc;
    --sidebar-active-bg:   rgba(14, 165, 233, 0.15);
    --sidebar-active-text: #38bdf8;
}

/* --------- LAYOUT --------- */
.main .block-container {
    padding-top: 2.4rem;
    padding-bottom: 5rem;
    max-width: 1200px;
}

/* --------- TIPOGRAFIA --------- */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
                 Roboto, "Helvetica Neue", Arial, sans-serif;
    color: var(--text);
    -webkit-font-smoothing: antialiased;
}
h1, h2, h3, h4, h5, h6 {
    color: var(--text);
    font-weight: 650;
    letter-spacing: -0.015em;
}
h1 { font-size: 2rem; }
h2 { font-size: 1.6rem; margin-top: 0.4rem; }
h3 { font-size: 1.15rem; margin-top: 1.6rem; color: var(--text); }
h4 { font-size: 0.95rem; color: var(--text-2); }

p, li { color: var(--text-2); line-height: 1.55; }

/* --------- CAPTIONS --------- */
.st-emotion-cache-16idsys p, small, .st-emotion-cache-1inwz65 {
    color: var(--text-2);
    font-size: 0.875rem;
}

/* --------- HERO (landing) --------- */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0284c7 100%);
    color: var(--sidebar-text-strong);
    padding: 34px 36px;
    border-radius: 12px;
    margin-bottom: 24px;
    border: 1px solid var(--sidebar-border);
    box-shadow: 0 2px 20px rgba(15, 23, 42, 0.08);
}
.hero .eyebrow {
    color: #7dd3fc;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.hero h1 {
    color: #ffffff !important;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 10px 0;
    letter-spacing: -0.02em;
}
.hero p {
    color: #cbd5e1;
    font-size: 1rem;
    max-width: 780px;
    margin: 0;
    line-height: 1.55;
}

/* --------- SECTION HEADERS --------- */
.section-head {
    border-left: 3px solid var(--primary);
    padding: 4px 0 4px 14px;
    margin: 4px 0 18px 0;
}
.section-head .kicker {
    color: var(--primary);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.section-head .title {
    color: var(--text);
    font-size: 1.35rem;
    font-weight: 650;
    letter-spacing: -0.015em;
}
.section-head .subtitle {
    color: var(--text-2);
    font-size: 0.9rem;
    margin-top: 2px;
}

/* --------- CARDS --------- */
.card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
}
.card-elevated {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
}
.card-accent {
    background: var(--primary-soft);
    border-left: 3px solid var(--primary);
    border-radius: 8px;
    padding: 14px 18px;
    color: var(--text);
}

/* --------- METRIC --------- */
[data-testid="stMetric"] {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] {
    color: var(--text-2);
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.03em;
}
[data-testid="stMetricValue"] {
    color: var(--text);
    font-size: 1.55rem;
    font-weight: 650;
}

/* --------- BUTTONS --------- */
.stButton > button, .stDownloadButton > button, .stLinkButton > a {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
    padding: 8px 16px !important;
}
button[kind="primary"] {
    background: var(--primary) !important;
    border: 1px solid var(--primary) !important;
    color: #ffffff !important;
}
button[kind="primary"]:hover {
    background: var(--primary-hover) !important;
    border-color: var(--primary-hover) !important;
    box-shadow: 0 2px 8px var(--primary-ring) !important;
}
button[kind="secondary"] {
    background: var(--bg) !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--text) !important;
}
button[kind="secondary"]:hover {
    background: var(--bg-hover) !important;
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}

/* --------- ALERTS --------- */
div[data-testid="stAlert"] {
    border-radius: 8px;
    padding: 12px 16px;
    border-left-width: 3px;
    background: var(--bg-elevated) !important;
}

/* --------- DIVIDERS --------- */
hr, div[data-testid="stDivider"] {
    border-color: var(--border) !important;
    background: var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* --------- SIDEBAR --------- */
/* Ocultar la navegacion automatica de Streamlit; usamos la nuestra */
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebarNavItems"] { display: none !important; }
[data-testid="stSidebarNavSeparator"] { display: none !important; }
div[data-testid="collapsedControl"] svg { fill: var(--sidebar-text) !important; }

section[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--sidebar-border);
}
section[data-testid="stSidebar"] > div:first-child {
    background: var(--sidebar-bg) !important;
}
/* Reducir padding superior para acercar el brand al boton collapse "<" */
section[data-testid="stSidebar"] .block-container {
    padding-top: 0.3rem !important;
}
[data-testid="stSidebarHeader"] {
    padding: 0 !important;
    height: auto !important;
    min-height: 0 !important;
}
/* Botones internos del sidebar (collapse) alineados a la altura del brand */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
    top: 0.55rem !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}
section[data-testid="stSidebar"] * {
    color: var(--sidebar-text);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] strong {
    color: var(--sidebar-text-strong) !important;
}
section[data-testid="stSidebar"] hr {
    background: var(--sidebar-border) !important;
    border-color: var(--sidebar-border) !important;
}
section[data-testid="stSidebar"] a {
    display: block;
    padding: 8px 12px !important;
    border-radius: 6px !important;
    color: var(--sidebar-text) !important;
    text-decoration: none !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    transition: all 0.12s ease;
    margin-bottom: 2px;
}
section[data-testid="stSidebar"] a:hover {
    background: rgba(148, 163, 184, 0.08) !important;
    color: var(--sidebar-text-strong) !important;
}

/* Sidebar app title: se alinea con el boton collapse "<" reservando espacio a la derecha */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 40px 16px 0;
    border-bottom: 1px solid var(--sidebar-border);
    margin-bottom: 16px;
    min-height: 40px;
}
.sidebar-brand-mark {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, var(--primary), var(--indigo));
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: #ffffff; font-weight: 700; font-size: 0.95rem;
}
.sidebar-brand-text {
    color: var(--sidebar-text-strong) !important;
    font-weight: 600;
    font-size: 0.95rem;
    line-height: 1.2;
}
.sidebar-brand-sub {
    color: var(--text-3) !important;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
}
.sidebar-section-title {
    color: var(--text-3) !important;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 18px 0 8px 4px;
}
.sidebar-status {
    background: rgba(15, 23, 42, 0.35);
    border: 1px solid var(--sidebar-border);
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 4px;
}
.sidebar-status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
    padding: 3px 0;
}
.sidebar-status-label { color: var(--text-3) !important; }
.sidebar-status-value { color: var(--sidebar-text-strong) !important; font-weight: 600; }
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.dot-on  { background: var(--emerald); box-shadow: 0 0 0 3px rgba(16,185,129,0.18); }
.dot-off { background: #64748b; }

/* --------- TABS --------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 16px;
    font-weight: 500;
    color: var(--text-2);
    border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
    background: var(--primary-soft);
}

/* --------- EXPANDER --------- */
[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg);
}
[data-testid="stExpander"] summary { font-weight: 500; }

/* --------- DATAFRAME --------- */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    border: 1px solid var(--border);
}

/* --------- INPUTS --------- */
input, select, textarea {
    accent-color: var(--primary);
}
[data-baseweb="input"] input, [data-baseweb="select"] {
    border-radius: 6px !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-ring) !important;
}

/* --------- FILE UPLOADER --------- */
[data-testid="stFileUploader"] section {
    border: 2px dashed var(--border-strong);
    border-radius: 10px;
    background: var(--bg-elevated);
    padding: 24px;
    transition: all 0.15s ease;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--primary);
    background: var(--primary-soft);
}

/* --------- CHIPS --------- */
.chip {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    line-height: 1.4;
    border: 1px solid transparent;
    margin-right: 4px;
}
.chip-primary { background: #eff6ff; color: #0369a1; border-color: #bae6fd; }
.chip-indigo  { background: #eef2ff; color: #4338ca; border-color: #c7d2fe; }
.chip-emerald { background: #ecfdf5; color: #047857; border-color: #a7f3d0; }
.chip-amber   { background: #fffbeb; color: #b45309; border-color: #fde68a; }
.chip-rose    { background: #fef2f2; color: #b91c1c; border-color: #fecaca; }
.chip-muted   { background: #f1f5f9; color: #475569; border-color: #cbd5e1; }

/* --------- PROB BAR --------- */
.prob-bar {
    position: relative;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 6px;
    height: 22px;
    overflow: hidden;
    min-width: 100px;
}
.prob-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary), var(--indigo));
    border-radius: 5px 0 0 5px;
    transition: width 0.3s ease;
}
.prob-bar-text {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text);
    font-size: 0.75rem;
    font-weight: 600;
    text-shadow: 0 0 3px rgba(255,255,255,0.6);
}

/* --------- MINI METRIC (para dashboards internos) --------- */
.mini-stat {
    display: flex;
    flex-direction: column;
    padding: 10px 14px;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 8px;
}
.mini-stat-label {
    color: var(--text-2);
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-weight: 600;
}
.mini-stat-value {
    color: var(--text);
    font-size: 1.15rem;
    font-weight: 650;
    margin-top: 2px;
}
</style>
"""


def apply_global_style() -> None:
    """CSS global. Se llama al inicio de cada pagina."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "Analisis No Supervisado") -> None:
    """Hero prominente para la landing.
    IMPORTANTE: el HTML va SIN indentacion para evitar que Streamlit lo
    interprete como bloque de codigo (Markdown convierte 4+ espacios en <pre>).
    """
    html = (
        '<div class="hero">'
        f'<div class="eyebrow">{eyebrow}</div>'
        f'<h1>{title}</h1>'
        f'<p>{subtitle}</p>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def section_head(title: str, subtitle: str = "", kicker: str = "") -> None:
    """Header de seccion con acento lateral."""
    parts = ['<div class="section-head">']
    if kicker:
        parts.append(f'<div class="kicker">{kicker}</div>')
    parts.append(f'<div class="title">{title}</div>')
    if subtitle:
        parts.append(f'<div class="subtitle">{subtitle}</div>')
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def card(content_html: str, variant: str = "") -> None:
    """Card generico. variant: '', 'elevated', 'accent'."""
    cls = "card" if not variant else f"card-{variant}"
    st.markdown(f'<div class="{cls}">{content_html}</div>', unsafe_allow_html=True)


def chip(text: str, color: str = "primary") -> str:
    """HTML de un chip pequeño de color. Devuelve string, no lo renderiza.

    color: 'primary' | 'indigo' | 'emerald' | 'amber' | 'rose' | 'muted'
    """
    return f'<span class="chip chip-{color}">{text}</span>'


def cluster_chip(cluster_id, palette_index: int = None) -> str:
    """Chip para etiqueta de cluster, coloreado ciclicamente."""
    if cluster_id == -1 or cluster_id == "Ruido":
        return '<span class="chip chip-muted">Ruido</span>'
    idx = palette_index if palette_index is not None else int(cluster_id)
    colors = ["primary", "indigo", "emerald", "amber", "rose"]
    color = colors[idx % len(colors)]
    return f'<span class="chip chip-{color}">Cluster {int(cluster_id)}</span>'


def prob_bar(prob: float) -> str:
    """Barra visual horizontal para representar una probabilidad 0-1."""
    pct = int(round(float(prob) * 100))
    return (
        f'<div class="prob-bar">'
        f'<div class="prob-bar-fill" style="width:{pct}%;"></div>'
        f'<span class="prob-bar-text">{pct}%</span>'
        f'</div>'
    )


def render_sidebar() -> None:
    """Sidebar de navegacion + estado."""
    from src.db.model_repository import count as count_models

    with st.sidebar:
        # Marca
        st.markdown(
            '<div class="sidebar-brand">'
            '<div class="sidebar-brand-mark">B5</div>'
            '<div>'
            '<div class="sidebar-brand-text">Big Five Analyzer</div>'
            '<div class="sidebar-brand-sub">UNIDAD IV · GMM</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Navegacion
        st.markdown('<div class="sidebar-section-title">Navegacion</div>',
                     unsafe_allow_html=True)

        st.page_link("app.py", label="Inicio")
        st.page_link("pages/1_Exploracion.py", label="Exploracion")
        st.page_link("pages/2_Entrenamiento.py", label="Entrenamiento")
        st.page_link("pages/3_Resultados.py", label="Resultados")
        st.page_link("pages/4_Modelos.py", label="Modelos")
        st.page_link("pages/5_Descargas.py", label="Descargas")
        st.page_link("pages/6_Clasificacion.py", label="Clasificacion")

        # Estado
        st.markdown('<div class="sidebar-section-title">Estado</div>',
                     unsafe_allow_html=True)

        df = st.session_state.get("df")
        n_records = 0 if df is None else len(df)
        csv_name = st.session_state.get("csv_name", "sin cargar")
        dot_data = "dot-on" if n_records > 0 else "dot-off"

        try:
            n_models = count_models()
        except Exception:
            n_models = 0

        st.markdown(
            '<div class="sidebar-status">'
            f'<div class="sidebar-status-row">'
            f'<span class="sidebar-status-label"><span class="status-dot {dot_data}"></span>Dataset</span>'
            f'<span class="sidebar-status-value">{n_records}</span>'
            f'</div>'
            f'<div class="sidebar-status-row">'
            f'<span class="sidebar-status-label">Archivo</span>'
            f'<span class="sidebar-status-value" style="font-size:0.72rem; max-width:120px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{csv_name}</span>'
            f'</div>'
            f'<div class="sidebar-status-row" style="margin-top:6px; padding-top:8px; border-top:1px solid rgba(148,163,184,0.15);">'
            f'<span class="sidebar-status-label">Modelos</span>'
            f'<span class="sidebar-status-value">{n_models}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Reset global
        st.markdown('<div class="sidebar-section-title">Acciones</div>',
                     unsafe_allow_html=True)
        if st.button("Reiniciar aplicacion", key="_reset_all",
                      use_container_width=True,
                      help="Limpia el estado de la sesion (dataset cargado, modelo actual, filtros). NO borra los modelos guardados en el historial ni los datasets de ejemplo."):
            _reset_all_state()
            st.rerun()

        # Footer
        st.markdown(
            '<div style="margin-top:24px; padding-top:16px; color: var(--text-3);'
            ' font-size:0.7rem; text-align:center; border-top:1px solid var(--sidebar-border);">'
            'Big Five · OCEAN<br/>'
            'Extraccion de Conocimientos en BD'
            '</div>',
            unsafe_allow_html=True,
        )


def _reset_all_state() -> None:
    """Limpia TODO el session_state. Ideal para arrancar pruebas desde cero.

    No borra la base de datos ni los .pkl guardados: solo el estado en memoria.
    """
    # Preservar la seed del uploader para no dejar el widget con estado viejo
    seed = st.session_state.get("_uploader_seed", 0) + 1
    st.session_state.clear()
    st.session_state["_uploader_seed"] = seed


def require_dataset() -> "pd.DataFrame":
    """Guardia comun: si no hay CSV cargado, muestra warning + link y detiene."""
    import pandas as pd
    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning(
            "Todavia no has cargado un CSV. Ve a **Inicio** y carga el archivo "
            "de respuestas para comenzar."
        )
        st.page_link("app.py", label="Ir a Inicio")
        st.stop()
    return df
