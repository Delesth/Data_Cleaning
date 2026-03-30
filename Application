import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Application de DataClean",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #161920;
    --surface2: #1e2230;
    --border: #2a2f3d;
    --accent: #00e5a0;
    --accent2: #7c6ffd;
    --accent3: #ff6b6b;
    --text: #e8eaf0;
    --muted: #6b7280;
    --success: #00e5a0;
    --warning: #fbbf24;
    --danger: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0,229,160,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
    line-height: 1.2;
}
.hero-title span { color: var(--accent); }
.hero-subtitle {
    color: var(--muted);
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 300;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
    transform: scaleX(0);
    transition: transform 0.3s;
}
.metric-card:hover::after { transform: scaleX(1); }
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
    display: block;
}
.metric-label {
    font-size: 0.8rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem 0;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}
.badge-success { background: rgba(0,229,160,0.15); color: var(--accent); border: 1px solid rgba(0,229,160,0.3); }
.badge-warning { background: rgba(251,191,36,0.15); color: var(--warning); border: 1px solid rgba(251,191,36,0.3); }
.badge-danger  { background: rgba(239,68,68,0.15);  color: var(--danger);  border: 1px solid rgba(239,68,68,0.3); }

/* Info boxes */
.info-box {
    background: var(--surface);
    border-left: 3px solid var(--accent2);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: var(--muted);
}

/* Buttons overrides */
.stButton > button {
    background: var(--accent);
    color: #0d0f14;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #00c98a;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0,229,160,0.3);
}

/* Download button */
.stDownloadButton > button {
    background: transparent;
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
}
.stDownloadButton > button:hover {
    background: rgba(0,229,160,0.1);
}

/* Dataframe styling */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: var(--muted);
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: var(--surface2);
    color: var(--text);
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface);
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}

/* Select boxes & inputs */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
}

/* Plotly chart bg */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

def load_data(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(uploaded_file)
    elif name.endswith((".xls", ".xlsx")):
        xl = pd.ExcelFile(uploaded_file)
        if len(xl.sheet_names) > 1:
            sheet = st.sidebar.selectbox("Feuille Excel", xl.sheet_names)
        else:
            sheet = xl.sheet_names[0]
        df = xl.parse(sheet)
    else:
        st.error("Format non supporté. Utilisez CSV ou Excel.")
        return None
    return df


def detect_column_types(df):
    numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols  = df.select_dtypes(include=['datetime64']).columns.tolist()
    bool_cols      = df.select_dtypes(include=['bool']).columns.tolist()
    return numeric_cols, categorical_cols, datetime_cols, bool_cols


def plotly_theme():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(22,25,32,1)',
        font=dict(family='DM Sans', color='#e8eaf0', size=12),
        xaxis=dict(gridcolor='#2a2f3d', zerolinecolor='#2a2f3d', tickfont=dict(color='#6b7280')),
        yaxis=dict(gridcolor='#2a2f3d', zerolinecolor='#2a2f3d', tickfont=dict(color='#6b7280')),
        colorway=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24','#38bdf8','#f472b6','#a3e635'],
        margin=dict(t=40, b=40, l=40, r=20)
    )


def download_df(df, fmt):
    if fmt == "CSV":
        return df.to_csv(index=False).encode('utf-8'), "text/csv", ".csv"
    elif fmt == "Excel":
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine='openpyxl')
        return buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"
    else:
        return df.to_json(orient='records').encode(), "application/json", ".json"


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧹 DataClean Pro")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Importer un fichier",
        type=["csv", "xls", "xlsx"],
        help="CSV ou Excel (.xls/.xlsx)"
    )

    if uploaded_file:
        st.markdown("### Navigation")
        page = st.radio(
            "Section",
            [
                "📊 Aperçu & Stats",
                "🔍 Qualité des données",
                "📈 Visualisations",
                "🔧 Nettoyage",
                "🔄 Transformations",
                "📉 Analyse bivariée",
                "⬇️ Export",
            ],
            label_visibility="collapsed"
        )
    else:
        page = None


# ─────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
  <p class="hero-title">Data<span>Clean</span> Pro</p>
  <p class="hero-subtitle">Nettoyage & analyse exploratoire de données — prêt pour la production</p>
</div>
""", unsafe_allow_html=True)

if not uploaded_file:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <span class="metric-value" style="color:#7c6ffd">①</span>
            <div class="metric-label">Importer CSV ou Excel</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <span class="metric-value" style="color:#00e5a0">②</span>
            <div class="metric-label">Explorer & nettoyer</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <span class="metric-value" style="color:#ff6b6b">③</span>
            <div class="metric-label">Exporter la base propre</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        ← Commencez par importer votre fichier dans la barre latérale gauche.<br>
        Formats supportés : <b>CSV</b>, <b>XLS</b>, <b>XLSX</b>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ── Load data ─────────────────────────────────────────────────
if 'df_original' not in st.session_state or st.session_state.get('filename') != uploaded_file.name:
    df_raw = load_data(uploaded_file)
    if df_raw is None:
        st.stop()
    st.session_state.df_original = df_raw.copy()
    st.session_state.df = df_raw.copy()
    st.session_state.filename = uploaded_file.name
    st.session_state.cleaning_log = []

df     = st.session_state.df
df_ori = st.session_state.df_original
num_cols, cat_cols, dt_cols, bool_cols = detect_column_types(df)


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — APERÇU & STATS
# ═══════════════════════════════════════════════════════════════
if page == "📊 Aperçu & Stats":
    # KPI row
    n_rows, n_cols = df.shape
    n_missing = int(df.isnull().sum().sum())
    n_dupl = int(df.duplicated().sum())
    mem = df.memory_usage(deep=True).sum()
    mem_str = f"{mem/1024:.1f} Ko" if mem < 1024**2 else f"{mem/1024**2:.1f} Mo"

    st.markdown('<p class="section-header">Vue d\'ensemble</p>', unsafe_allow_html=True)

    cols = st.columns(5)
    metrics = [
        ("Lignes",         f"{n_rows:,}",              "#00e5a0"),
        ("Colonnes",       f"{n_cols}",                "#7c6ffd"),
        ("Valeurs manq.",  f"{n_missing:,}",           "#fbbf24"),
        ("Doublons",       f"{n_dupl:,}",              "#ff6b6b"),
        ("Mémoire",        mem_str,                    "#38bdf8"),
    ]
    for c, (label, val, color) in zip(cols, metrics):
        with c:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-value" style="color:{color}">{val}</span>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    # Preview
    st.markdown('<p class="section-header">Aperçu des données</p>', unsafe_allow_html=True)
    n_preview = st.slider("Nombre de lignes à afficher", 5, 100, 10)
    preview_opt = st.radio("Afficher", ["Début", "Fin", "Aléatoire"], horizontal=True)
    if preview_opt == "Début":
        st.dataframe(df.head(n_preview), use_container_width=True)
    elif preview_opt == "Fin":
        st.dataframe(df.tail(n_preview), use_container_width=True)
    else:
        st.dataframe(df.sample(min(n_preview, len(df))), use_container_width=True)

    # Types
    st.markdown('<p class="section-header">Types de colonnes</p>', unsafe_allow_html=True)
    dtype_df = pd.DataFrame({
        'Colonne': df.columns,
        'Type':    df.dtypes.astype(str).values,
        'Non-null': df.count().values,
        '% non-null': (df.count().values / len(df) * 100).round(2),
        'Valeurs uniques': [df[c].nunique() for c in df.columns],
        'Exemple': [str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0] > 0 else 'N/A' for c in df.columns]
    })
    st.dataframe(dtype_df, use_container_width=True)

    # Descriptive stats
    st.markdown('<p class="section-header">Statistiques descriptives</p>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Numériques", "Catégorielles"])
    with tab1:
        if num_cols:
            desc = df[num_cols].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T.round(4)
            desc['skewness'] = df[num_cols].skew().round(4)
            desc['kurtosis'] = df[num_cols].kurt().round(4)
            desc['CV (%)']   = (df[num_cols].std() / df[num_cols].mean() * 100).round(2)
            st.dataframe(desc, use_container_width=True)
        else:
            st.info("Aucune colonne numérique détectée.")
    with tab2:
        if cat_cols:
            cat_desc = pd.DataFrame({
                'Colonne':       cat_cols,
                'Count':         [df[c].count() for c in cat_cols],
                'Uniques':       [df[c].nunique() for c in cat_cols],
                'Top valeur':    [df[c].mode()[0] if df[c].count() > 0 else 'N/A' for c in cat_cols],
                'Freq. top (%)': [(df[c].value_counts().iloc[0] / df[c].count() * 100).round(2) if df[c].count() > 0 else 0 for c in cat_cols],
            })
            st.dataframe(cat_desc, use_container_width=True)
        else:
            st.info("Aucune colonne catégorielle.")


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — QUALITÉ DES DONNÉES
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Qualité des données":
    st.markdown('<p class="section-header">Analyse de la qualité</p>', unsafe_allow_html=True)

    # Missing values heatmap
    st.subheader("Valeurs manquantes")
    miss = df.isnull().sum().reset_index()
    miss.columns = ['Colonne', 'Manquantes']
    miss['%'] = (miss['Manquantes'] / len(df) * 100).round(2)
    miss['Présentes'] = len(df) - miss['Manquantes']
    miss = miss.sort_values('Manquantes', ascending=False)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        fig.add_bar(
            x=miss['Colonne'], y=miss['%'],
            marker_color=['#ef4444' if p > 30 else '#fbbf24' if p > 5 else '#00e5a0' for p in miss['%']],
            text=[f"{p}%" for p in miss['%']], textposition='outside'
        )
        fig.update_layout(
            title="% valeurs manquantes par colonne",
            xaxis_tickangle=-45,
            yaxis_title="% manquant",
            **plotly_theme()
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(miss[['Colonne','Manquantes','%']], use_container_width=True, hide_index=True)

    # Missing value matrix (if not too large)
    if len(df) <= 5000 and len(df.columns) <= 50:
        st.subheader("Matrice de présence (blanc = manquant)")
        fig2, ax = plt.subplots(figsize=(12, max(4, len(df.columns)*0.3)))
        fig2.patch.set_facecolor('#0d0f14')
        ax.set_facecolor('#161920')
        sns.heatmap(df.isnull(), cbar=False, ax=ax,
                    cmap=['#00e5a0', '#ef4444'],
                    xticklabels=True, yticklabels=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', color='#6b7280', fontsize=9)
        ax.tick_params(colors='#6b7280')
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    # Duplicates
    st.markdown('<p class="section-header">Doublons</p>', unsafe_allow_html=True)
    n_dupl = df.duplicated().sum()
    if n_dupl > 0:
        st.markdown(f'<span class="badge badge-danger">⚠ {n_dupl} doublons détectés ({n_dupl/len(df)*100:.2f}%)</span>', unsafe_allow_html=True)
        with st.expander("Afficher les doublons"):
            st.dataframe(df[df.duplicated(keep=False)].sort_values(df.columns.tolist()), use_container_width=True)
    else:
        st.markdown('<span class="badge badge-success">✓ Aucun doublon</span>', unsafe_allow_html=True)

    # Valeurs uniques
    st.markdown('<p class="section-header">Distribution des valeurs uniques</p>', unsafe_allow_html=True)
    uniq = pd.DataFrame({
        'Colonne': df.columns,
        'Uniques': [df[c].nunique() for c in df.columns],
        'Total':   len(df)
    })
    uniq['% unicité'] = (uniq['Uniques'] / uniq['Total'] * 100).round(2)
    fig3 = px.bar(uniq.sort_values('Uniques', ascending=True),
                  x='% unicité', y='Colonne', orientation='h',
                  color='% unicité', color_continuous_scale=['#1e2230','#7c6ffd','#00e5a0'],
                  text='Uniques')
    fig3.update_layout(title="Unicité des colonnes", **plotly_theme())
    st.plotly_chart(fig3, use_container_width=True)

    # Outliers (IQR)
    if num_cols:
        st.markdown('<p class="section-header">Détection des outliers (méthode IQR)</p>', unsafe_allow_html=True)
        outlier_data = []
        for c in num_cols:
            q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            out = df[(df[c] < lo) | (df[c] > hi)][c]
            outlier_data.append({
                'Colonne': c,
                'Q1': round(q1,4), 'Q3': round(q3,4), 'IQR': round(iqr,4),
                'Borne inf': round(lo,4), 'Borne sup': round(hi,4),
                'Nb outliers': len(out),
                '% outliers': round(len(out)/len(df)*100,2)
            })
        out_df = pd.DataFrame(outlier_data).sort_values('Nb outliers', ascending=False)
        st.dataframe(out_df, use_container_width=True, hide_index=True)

    # Constant / near-constant columns
    st.markdown('<p class="section-header">Colonnes constantes ou quasi-constantes</p>', unsafe_allow_html=True)
    quasi = []
    for c in df.columns:
        top_freq = df[c].value_counts(normalize=True).iloc[0] if df[c].nunique() > 0 else 1.0
        if top_freq >= 0.95:
            quasi.append({'Colonne': c, 'Valeur dominante': df[c].mode()[0], 'Fréquence (%)': round(top_freq*100,2)})
    if quasi:
        st.dataframe(pd.DataFrame(quasi), use_container_width=True, hide_index=True)
    else:
        st.markdown('<span class="badge badge-success">✓ Aucune colonne quasi-constante</span>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — VISUALISATIONS
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Visualisations":
    st.markdown('<p class="section-header">Visualisations explorartoires</p>', unsafe_allow_html=True)

    viz_type = st.selectbox("Type de visualisation", [
        "Histogramme (distribution)",
        "Boîte à moustaches (boxplot)",
        "Violon plot",
        "Barplot (catégoriel)",
        "Pie chart",
        "Heatmap de corrélation",
        "Pairplot",
        "Série temporelle",
        "Scatter plot",
        "Distribution des manquants (heatmap)",
    ])

    # ── Histogramme ──────────────────────────────────────────
    if viz_type == "Histogramme (distribution)":
        if not num_cols:
            st.warning("Aucune colonne numérique.")
        else:
            col = st.selectbox("Colonne", num_cols)
            bins = st.slider("Bins", 5, 100, 30)
            kde  = st.checkbox("Overlay KDE", True)
            fig = go.Figure()
            fig.add_histogram(x=df[col].dropna(), nbinsx=bins,
                              marker_color='#7c6ffd', opacity=0.8, name='Count')
            if kde:
                from scipy.stats import gaussian_kde
                vals = df[col].dropna().values
                if len(vals) > 1:
                    kde_fn = gaussian_kde(vals)
                    x_range = np.linspace(vals.min(), vals.max(), 300)
                    scale   = len(vals) * (vals.max()-vals.min()) / bins
                    fig.add_scatter(x=x_range, y=kde_fn(x_range)*scale,
                                    mode='lines', line=dict(color='#00e5a0', width=2), name='KDE')
            fig.update_layout(title=f"Distribution — {col}", **plotly_theme())
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df[col].describe().round(4).to_frame(), use_container_width=True)

    # ── Boxplot ───────────────────────────────────────────────
    elif viz_type == "Boîte à moustaches (boxplot)":
        if not num_cols:
            st.warning("Aucune colonne numérique.")
        else:
            cols_sel = st.multiselect("Colonnes", num_cols, default=num_cols[:min(5,len(num_cols))])
            group_by = st.selectbox("Grouper par (optionnel)", ["Aucun"] + cat_cols)
            if cols_sel:
                if group_by == "Aucun":
                    fig = go.Figure()
                    for i, c in enumerate(cols_sel):
                        fig.add_box(y=df[c].dropna(), name=c,
                                    marker_color=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24','#38bdf8'][i%5])
                else:
                    fig = px.box(df, y=cols_sel[0], color=group_by,
                                 color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24'])
                fig.update_layout(title="Boxplot", **plotly_theme())
                st.plotly_chart(fig, use_container_width=True)

    # ── Violin ───────────────────────────────────────────────
    elif viz_type == "Violon plot":
        if not num_cols:
            st.warning("Aucune colonne numérique.")
        else:
            col = st.selectbox("Colonne numérique", num_cols)
            grp = st.selectbox("Grouper par", ["Aucun"] + cat_cols)
            if grp == "Aucun":
                fig = px.violin(df, y=col, box=True, points="outliers",
                                color_discrete_sequence=['#7c6ffd'])
            else:
                fig = px.violin(df, y=col, x=grp, box=True, points="outliers",
                                color=grp, color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24'])
            fig.update_layout(title=f"Violin — {col}", **plotly_theme())
            st.plotly_chart(fig, use_container_width=True)

    # ── Barplot catégoriel ────────────────────────────────────
    elif viz_type == "Barplot (catégoriel)":
        if not cat_cols:
            st.warning("Aucune colonne catégorielle.")
        else:
            col = st.selectbox("Colonne", cat_cols)
            top_n = st.slider("Top N valeurs", 5, 50, 15)
            vc = df[col].value_counts().head(top_n).reset_index()
            vc.columns = ['Valeur', 'Count']
            fig = px.bar(vc, x='Valeur', y='Count',
                         color='Count', color_continuous_scale=['#1e2230','#7c6ffd','#00e5a0'],
                         text='Count')
            fig.update_traces(textposition='outside')
            fig.update_layout(title=f"Distribution — {col}", xaxis_tickangle=-45, **plotly_theme())
            st.plotly_chart(fig, use_container_width=True)

    # ── Pie chart ─────────────────────────────────────────────
    elif viz_type == "Pie chart":
        if not cat_cols:
            st.warning("Aucune colonne catégorielle.")
        else:
            col = st.selectbox("Colonne", cat_cols)
            top_n = st.slider("Top N", 3, 20, 8)
            vc = df[col].value_counts().head(top_n)
            fig = px.pie(values=vc.values, names=vc.index,
                         color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24','#38bdf8','#f472b6'])
            fig.update_layout(title=f"Répartition — {col}", **plotly_theme())
            st.plotly_chart(fig, use_container_width=True)

    # ── Corrélation ───────────────────────────────────────────
    elif viz_type == "Heatmap de corrélation":
        if len(num_cols) < 2:
            st.warning("Il faut au moins 2 colonnes numériques.")
        else:
            method = st.selectbox("Méthode", ["pearson", "spearman", "kendall"])
            corr = df[num_cols].corr(method=method)
            fig = px.imshow(corr, text_auto=".2f",
                            color_continuous_scale=['#ef4444','#161920','#00e5a0'],
                            zmin=-1, zmax=1, aspect='auto')
            fig.update_layout(title=f"Matrice de corrélation ({method})", **plotly_theme())
            st.plotly_chart(fig, use_container_width=True)

            # Strong correlations table
            threshold = st.slider("Seuil de corrélation forte", 0.5, 0.99, 0.7)
            pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    v = corr.iloc[i,j]
                    if abs(v) >= threshold:
                        pairs.append({'Var 1': corr.columns[i], 'Var 2': corr.columns[j], 'Corrélation': round(v,4)})
            if pairs:
                st.markdown(f"**Paires avec |r| ≥ {threshold}:**")
                st.dataframe(pd.DataFrame(pairs).sort_values('Corrélation', key=abs, ascending=False),
                             use_container_width=True, hide_index=True)

    # ── Pairplot ──────────────────────────────────────────────
    elif viz_type == "Pairplot":
        if len(num_cols) < 2:
            st.warning("Il faut au moins 2 colonnes numériques.")
        else:
            sel = st.multiselect("Colonnes", num_cols, default=num_cols[:min(4, len(num_cols))])
            color_by = st.selectbox("Couleur par", ["Aucun"] + cat_cols)
            if len(sel) >= 2:
                fig = px.scatter_matrix(df, dimensions=sel,
                                        color=None if color_by=="Aucun" else color_by,
                                        color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24'])
                fig.update_traces(diagonal_visible=True, showupperhalf=False)
                fig.update_layout(title="Pairplot", **plotly_theme())
                st.plotly_chart(fig, use_container_width=True)

    # ── Série temporelle ──────────────────────────────────────
    elif viz_type == "Série temporelle":
        date_candidates = dt_cols + [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if not date_candidates:
            st.warning("Aucune colonne date détectée. Essayez de convertir une colonne dans 'Transformations'.")
        else:
            date_col = st.selectbox("Colonne date", date_candidates)
            val_col  = st.selectbox("Valeur", num_cols)
            agg      = st.selectbox("Agrégation", ["Somme", "Moyenne", "Médiane", "Count"])
            try:
                tmp = df.copy()
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
                tmp = tmp.dropna(subset=[date_col])
                freq = st.selectbox("Fréquence", ["Jour (D)", "Semaine (W)", "Mois (M)", "Trimestre (Q)", "Année (Y)"])
                freq_map = {"Jour (D)":"D","Semaine (W)":"W","Mois (M)":"M","Trimestre (Q)":"Q","Année (Y)":"Y"}
                agg_map  = {"Somme":"sum","Moyenne":"mean","Médiane":"median","Count":"count"}
                ts = tmp.set_index(date_col)[val_col].resample(freq_map[freq]).agg(agg_map[agg]).reset_index()
                fig = px.line(ts, x=date_col, y=val_col,
                              color_discrete_sequence=['#00e5a0'])
                fig.update_traces(line_width=2)
                fig.add_scatter(x=ts[date_col], y=ts[val_col], mode='markers',
                                marker=dict(color='#7c6ffd', size=5), showlegend=False)
                fig.update_layout(title=f"Série temporelle — {val_col} ({agg})", **plotly_theme())
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur: {e}")

    # ── Scatter ───────────────────────────────────────────────
    elif viz_type == "Scatter plot":
        if len(num_cols) < 2:
            st.warning("Il faut au moins 2 colonnes numériques.")
        else:
            x = st.selectbox("Axe X", num_cols)
            y = st.selectbox("Axe Y", [c for c in num_cols if c != x])
            color_by = st.selectbox("Couleur par", ["Aucun"] + cat_cols + num_cols)
            size_by  = st.selectbox("Taille par (optionnel)", ["Aucun"] + num_cols)
            trendline = st.checkbox("Ligne de tendance (OLS)")
            kwargs = dict(
                x=x, y=y,
                color=None if color_by=="Aucun" else color_by,
                size=None if size_by=="Aucun" else size_by,
                trendline="ols" if trendline else None,
                opacity=0.7,
                color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24'],
                color_continuous_scale=['#1e2230','#7c6ffd','#00e5a0'],
            )
            fig = px.scatter(df, **kwargs)
            fig.update_layout(title=f"Scatter — {x} vs {y}", **plotly_theme())
            st.plotly_chart(fig, use_container_width=True)

    # ── Missing heatmap ───────────────────────────────────────
    elif viz_type == "Distribution des manquants (heatmap)":
        fig = px.imshow(df.isnull().astype(int),
                        color_continuous_scale=['#161920','#ef4444'],
                        aspect='auto')
        fig.update_layout(title="Carte des valeurs manquantes", **plotly_theme())
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — NETTOYAGE
# ═══════════════════════════════════════════════════════════════
elif page == "🔧 Nettoyage":
    st.markdown('<p class="section-header">Opérations de nettoyage</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Les modifications s\'appliquent de manière cumulative. Le journal ci-dessous retrace chaque opération.</div>', unsafe_allow_html=True)

    # ── Doublons ─────────────────────────────────────────────
    with st.expander("🗑 Supprimer les doublons"):
        n_dup = df.duplicated().sum()
        st.write(f"Doublons détectés : **{n_dup}**")
        subset_cols = st.multiselect("Colonnes de référence (vide = toutes)", df.columns.tolist(), key='dup_cols')
        keep_opt = st.radio("Conserver", ["first", "last", "Supprimer tous"], key='dup_keep', horizontal=True)
        if st.button("Supprimer les doublons"):
            keep = False if keep_opt == "Supprimer tous" else keep_opt
            sub  = subset_cols if subset_cols else None
            before = len(df)
            st.session_state.df = df.drop_duplicates(subset=sub, keep=keep).reset_index(drop=True)
            after  = len(st.session_state.df)
            msg = f"Suppression de {before-after} doublons (keep={keep})"
            st.session_state.cleaning_log.append(msg)
            st.success(msg)
            df = st.session_state.df

    # ── Valeurs manquantes ────────────────────────────────────
    with st.expander("🩹 Traiter les valeurs manquantes"):
        col_missing = st.selectbox("Colonne cible", ["Toutes les colonnes"] + df.columns.tolist(), key='miss_col')
        strategy = st.selectbox("Stratégie", [
            "Supprimer les lignes", "Supprimer les colonnes",
            "Remplacer par la moyenne", "Remplacer par la médiane",
            "Remplacer par le mode", "Remplacer par une valeur fixe",
            "Remplissage avant (ffill)", "Remplissage arrière (bfill)"
        ], key='miss_strat')
        custom_val = None
        if strategy == "Remplacer par une valeur fixe":
            custom_val = st.text_input("Valeur de remplacement", key='miss_val')

        if st.button("Appliquer le traitement des valeurs manquantes"):
            df_work = st.session_state.df.copy()
            target_cols = df_work.columns.tolist() if col_missing == "Toutes les colonnes" else [col_missing]
            for c in target_cols:
                if strategy == "Supprimer les lignes":
                    df_work = df_work.dropna(subset=[c])
                elif strategy == "Supprimer les colonnes":
                    df_work = df_work.drop(columns=[c])
                elif strategy == "Remplacer par la moyenne" and c in df_work.select_dtypes(include=np.number):
                    df_work[c].fillna(df_work[c].mean(), inplace=True)
                elif strategy == "Remplacer par la médiane" and c in df_work.select_dtypes(include=np.number):
                    df_work[c].fillna(df_work[c].median(), inplace=True)
                elif strategy == "Remplacer par le mode":
                    df_work[c].fillna(df_work[c].mode()[0] if df_work[c].count() > 0 else np.nan, inplace=True)
                elif strategy == "Remplacer par une valeur fixe":
                    try:
                        fill = float(custom_val) if df_work[c].dtype in [float, int] else custom_val
                    except:
                        fill = custom_val
                    df_work[c].fillna(fill, inplace=True)
                elif strategy == "Remplissage avant (ffill)":
                    df_work[c].fillna(method='ffill', inplace=True)
                elif strategy == "Remplissage arrière (bfill)":
                    df_work[c].fillna(method='bfill', inplace=True)
            st.session_state.df = df_work
            df = df_work
            msg = f"Valeurs manquantes — {strategy} sur {col_missing}"
            st.session_state.cleaning_log.append(msg)
            st.success(msg)

    # ── Outliers ──────────────────────────────────────────────
    with st.expander("📐 Traiter les outliers"):
        if not num_cols:
            st.info("Aucune colonne numérique.")
        else:
            out_col = st.selectbox("Colonne", num_cols, key='out_col')
            out_method = st.selectbox("Méthode de détection", ["IQR (×1.5)", "IQR (×3)", "Z-score (|z|>3)"], key='out_meth')
            out_action = st.selectbox("Action", ["Supprimer les lignes", "Remplacer par les bornes (winsorize)", "Remplacer par NaN"], key='out_act')
            if st.button("Traiter les outliers"):
                df_work = st.session_state.df.copy()
                col_data = df_work[out_col].dropna()
                if "IQR" in out_method:
                    factor = 1.5 if "1.5" in out_method else 3.0
                    q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
                    iqr = q3 - q1
                    lo, hi = q1 - factor*iqr, q3 + factor*iqr
                else:
                    z = (col_data - col_data.mean()) / col_data.std()
                    lo, hi = col_data[z > -3].min(), col_data[z < 3].max()
                mask = (df_work[out_col] < lo) | (df_work[out_col] > hi)
                n_out = mask.sum()
                if out_action == "Supprimer les lignes":
                    df_work = df_work[~mask]
                elif out_action == "Remplacer par les bornes (winsorize)":
                    df_work.loc[df_work[out_col] < lo, out_col] = lo
                    df_work.loc[df_work[out_col] > hi, out_col] = hi
                else:
                    df_work.loc[mask, out_col] = np.nan
                st.session_state.df = df_work
                df = df_work
                msg = f"Outliers ({out_col}) — {out_action} ({n_out} lignes)"
                st.session_state.cleaning_log.append(msg)
                st.success(msg)

    # ── Suppression de colonnes ───────────────────────────────
    with st.expander("🗂 Supprimer des colonnes"):
        cols_to_drop = st.multiselect("Colonnes à supprimer", df.columns.tolist(), key='drop_cols')
        if st.button("Supprimer les colonnes sélectionnées"):
            st.session_state.df = df.drop(columns=cols_to_drop)
            df = st.session_state.df
            msg = f"Colonnes supprimées : {cols_to_drop}"
            st.session_state.cleaning_log.append(msg)
            st.success(msg)

    # ── Renommer ──────────────────────────────────────────────
    with st.expander("✏️ Renommer des colonnes"):
        rename_col = st.selectbox("Colonne à renommer", df.columns.tolist(), key='ren_col')
        new_name   = st.text_input("Nouveau nom", key='ren_name')
        if st.button("Renommer") and new_name:
            st.session_state.df = df.rename(columns={rename_col: new_name})
            df = st.session_state.df
            msg = f"Renommage : {rename_col} → {new_name}"
            st.session_state.cleaning_log.append(msg)
            st.success(msg)

    # ── Filtres ───────────────────────────────────────────────
    with st.expander("🔎 Filtrer les lignes"):
        filter_col = st.selectbox("Colonne", df.columns.tolist(), key='filt_col')
        if filter_col in num_cols:
            lo_f = float(df[filter_col].min())
            hi_f = float(df[filter_col].max())
            rng  = st.slider("Plage de valeurs", lo_f, hi_f, (lo_f, hi_f), key='filt_range')
            if st.button("Appliquer le filtre numérique"):
                st.session_state.df = df[(df[filter_col] >= rng[0]) & (df[filter_col] <= rng[1])]
                df = st.session_state.df
                msg = f"Filtre : {filter_col} ∈ [{rng[0]}, {rng[1]}]"
                st.session_state.cleaning_log.append(msg)
                st.success(msg)
        else:
            unique_vals = df[filter_col].dropna().unique().tolist()
            keep_vals   = st.multiselect("Valeurs à conserver", unique_vals, default=unique_vals, key='filt_vals')
            if st.button("Appliquer le filtre catégoriel"):
                st.session_state.df = df[df[filter_col].isin(keep_vals)]
                df = st.session_state.df
                msg = f"Filtre catégoriel : {filter_col} ∈ {keep_vals}"
                st.session_state.cleaning_log.append(msg)
                st.success(msg)

    # ── Reset ─────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔁 Réinitialiser toutes les modifications"):
        st.session_state.df = st.session_state.df_original.copy()
        st.session_state.cleaning_log = []
        st.success("Données réinitialisées.")

    # ── Log ───────────────────────────────────────────────────
    st.markdown('<p class="section-header">Journal des opérations</p>', unsafe_allow_html=True)
    if st.session_state.cleaning_log:
        for i, entry in enumerate(st.session_state.cleaning_log, 1):
            st.markdown(f"`{i}.` {entry}")
    else:
        st.markdown('<span class="badge badge-warning">Aucune opération effectuée</span>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — TRANSFORMATIONS
# ═══════════════════════════════════════════════════════════════
elif page == "🔄 Transformations":
    st.markdown('<p class="section-header">Transformations & ingénierie des variables</p>', unsafe_allow_html=True)

    # ── Conversion de types ───────────────────────────────────
    with st.expander("🔢 Convertir les types de colonnes"):
        conv_col  = st.selectbox("Colonne", df.columns.tolist(), key='conv_col')
        conv_type = st.selectbox("Nouveau type", ["int", "float", "str", "bool", "datetime"], key='conv_type')
        if st.button("Convertir"):
            try:
                df_work = st.session_state.df.copy()
                if conv_type == "datetime":
                    df_work[conv_col] = pd.to_datetime(df_work[conv_col], errors='coerce')
                elif conv_type == "bool":
                    df_work[conv_col] = df_work[conv_col].astype(bool)
                elif conv_type == "int":
                    df_work[conv_col] = pd.to_numeric(df_work[conv_col], errors='coerce').astype('Int64')
                elif conv_type == "float":
                    df_work[conv_col] = pd.to_numeric(df_work[conv_col], errors='coerce')
                else:
                    df_work[conv_col] = df_work[conv_col].astype(str)
                st.session_state.df = df_work
                df = df_work
                msg = f"Conversion : {conv_col} → {conv_type}"
                st.session_state.cleaning_log.append(msg)
                st.success(msg)
            except Exception as e:
                st.error(f"Erreur: {e}")

    # ── Normalisation / Standardisation ───────────────────────
    with st.expander("📏 Normaliser / Standardiser"):
        if not num_cols:
            st.info("Aucune colonne numérique.")
        else:
            norm_cols = st.multiselect("Colonnes", num_cols, key='norm_cols')
            norm_method = st.radio("Méthode", ["Min-Max [0,1]", "Z-score (standardisation)", "Robust (médiane/IQR)"], key='norm_meth', horizontal=True)
            if st.button("Appliquer la normalisation") and norm_cols:
                df_work = st.session_state.df.copy()
                for c in norm_cols:
                    s = df_work[c].copy()
                    if norm_method == "Min-Max [0,1]":
                        df_work[c] = (s - s.min()) / (s.max() - s.min())
                    elif norm_method == "Z-score (standardisation)":
                        df_work[c] = (s - s.mean()) / s.std()
                    else:
                        df_work[c] = (s - s.median()) / (s.quantile(0.75) - s.quantile(0.25))
                st.session_state.df = df_work
                df = df_work
                msg = f"Normalisation ({norm_method}) : {norm_cols}"
                st.session_state.cleaning_log.append(msg)
                st.success(msg)

    # ── Encodage ──────────────────────────────────────────────
    with st.expander("🏷 Encoder les variables catégorielles"):
        if not cat_cols:
            st.info("Aucune colonne catégorielle.")
        else:
            enc_col    = st.selectbox("Colonne", cat_cols, key='enc_col')
            enc_method = st.radio("Méthode", ["Label Encoding", "One-Hot Encoding", "Ordinal (ordre personnalisé)"], key='enc_meth', horizontal=True)
            if enc_method == "Ordinal (ordre personnalisé)":
                cats = df[enc_col].dropna().unique().tolist()
                st.write("Définissez l'ordre (glissez ou entrez une liste séparée par virgules):")
                order_str = st.text_input("Ordre", value=", ".join([str(c) for c in cats]), key='ord_str')
            if st.button("Encoder"):
                df_work = st.session_state.df.copy()
                if enc_method == "Label Encoding":
                    df_work[enc_col+"_encoded"] = pd.factorize(df_work[enc_col])[0]
                elif enc_method == "One-Hot Encoding":
                    dummies = pd.get_dummies(df_work[enc_col], prefix=enc_col, dtype=int)
                    df_work = pd.concat([df_work, dummies], axis=1)
                else:
                    order = [x.strip() for x in order_str.split(',')]
                    mapping = {v:i for i,v in enumerate(order)}
                    df_work[enc_col+"_ordinal"] = df_work[enc_col].map(mapping)
                st.session_state.df = df_work
                df = df_work
                msg = f"Encodage ({enc_method}) : {enc_col}"
                st.session_state.cleaning_log.append(msg)
                st.success(msg)

    # ── Nouvelle colonne calculée ─────────────────────────────
    with st.expander("➕ Créer une colonne calculée"):
        new_col_name = st.text_input("Nom de la nouvelle colonne", key='new_col_name')
        formula = st.text_area(
            "Formule (utilisez `df['colonne']` ou les noms de colonnes directement)",
            placeholder="Exemple : df['prix'] * df['quantite']",
            key='new_col_formula'
        )
        if st.button("Créer la colonne") and new_col_name and formula:
            try:
                df_work = st.session_state.df.copy()
                df_work[new_col_name] = eval(formula, {"df": df_work, "np": np, "pd": pd})
                st.session_state.df = df_work
                df = df_work
                msg = f"Nouvelle colonne : {new_col_name} = {formula[:50]}..."
                st.session_state.cleaning_log.append(msg)
                st.success(msg)
            except Exception as e:
                st.error(f"Erreur dans la formule : {e}")

    # ── Extraction date ───────────────────────────────────────
    with st.expander("📅 Extraire des composantes de date"):
        date_cols_all = dt_cols + [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if not date_cols_all:
            st.info("Aucune colonne date détectée.")
        else:
            dt_col = st.selectbox("Colonne date", date_cols_all, key='dt_col')
            components = st.multiselect("Composantes à extraire",
                ["Année", "Mois", "Jour", "Heure", "Minute", "Jour de la semaine", "Semaine", "Trimestre"], key='dt_comp')
            if st.button("Extraire") and components:
                df_work = st.session_state.df.copy()
                df_work[dt_col] = pd.to_datetime(df_work[dt_col], errors='coerce')
                map_comp = {
                    "Année": ("year", dt_col+"_annee"),
                    "Mois":  ("month", dt_col+"_mois"),
                    "Jour":  ("day", dt_col+"_jour"),
                    "Heure": ("hour", dt_col+"_heure"),
                    "Minute":("minute", dt_col+"_minute"),
                    "Jour de la semaine": ("dayofweek", dt_col+"_dow"),
                    "Semaine": ("isocalendar().week", dt_col+"_semaine"),
                    "Trimestre": ("quarter", dt_col+"_trimestre"),
                }
                for comp in components:
                    attr, name = map_comp[comp]
                    if '()' in attr:
                        df_work[name] = df_work[dt_col].dt.isocalendar().week.astype(int)
                    else:
                        df_work[name] = getattr(df_work[dt_col].dt, attr)
                st.session_state.df = df_work
                df = df_work
                msg = f"Extraction date ({dt_col}) : {components}"
                st.session_state.cleaning_log.append(msg)
                st.success(msg)

    # ── Nettoyage texte ───────────────────────────────────────
    with st.expander("🔡 Nettoyer les colonnes texte"):
        if not cat_cols:
            st.info("Aucune colonne texte.")
        else:
            txt_col = st.selectbox("Colonne texte", cat_cols, key='txt_col')
            txt_ops = st.multiselect("Opérations", [
                "Mettre en minuscules", "Supprimer les espaces (strip)",
                "Supprimer les caractères spéciaux",
                "Capitaliser (première lettre)", "Mettre en majuscules"
            ], key='txt_ops')
            if st.button("Nettoyer le texte") and txt_ops:
                df_work = st.session_state.df.copy()
                col_s   = df_work[txt_col].astype(str)
                for op in txt_ops:
                    if op == "Mettre en minuscules":         col_s = col_s.str.lower()
                    elif op == "Supprimer les espaces (strip)": col_s = col_s.str.strip()
                    elif op == "Supprimer les caractères spéciaux": col_s = col_s.str.replace(r'[^a-zA-Z0-9\s]','', regex=True)
                    elif op == "Capitaliser (première lettre)": col_s = col_s.str.capitalize()
                    elif op == "Mettre en majuscules":       col_s = col_s.str.upper()
                df_work[txt_col] = col_s
                st.session_state.df = df_work
                df = df_work
                msg = f"Nettoyage texte ({txt_col}) : {txt_ops}"
                st.session_state.cleaning_log.append(msg)
                st.success(msg)

    st.markdown("**Aperçu du dataframe transformé**")
    st.dataframe(st.session_state.df.head(10), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 6 — ANALYSE BIVARIÉE
# ═══════════════════════════════════════════════════════════════
elif page == "📉 Analyse bivariée":
    st.markdown('<p class="section-header">Analyse bivariée & relations entre variables</p>', unsafe_allow_html=True)

    if len(df.columns) < 2:
        st.warning("Il faut au moins 2 colonnes.")
    else:
        num_cols2, cat_cols2, _, _ = detect_column_types(df)

        tab1, tab2, tab3 = st.tabs(["Num × Num", "Cat × Num", "Cat × Cat"])

        with tab1:
            if len(num_cols2) < 2:
                st.info("Il faut au moins 2 colonnes numériques.")
            else:
                x2 = st.selectbox("Variable X", num_cols2, key='bv_x')
                y2 = st.selectbox("Variable Y", [c for c in num_cols2 if c!=x2], key='bv_y')
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.scatter(df, x=x2, y=y2, trendline='ols',
                                     opacity=0.6, color_discrete_sequence=['#7c6ffd'])
                    fig.update_layout(title=f"Scatter + OLS : {x2} vs {y2}", **plotly_theme())
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    from scipy import stats
                    x_clean = df[[x2,y2]].dropna()
                    r, p = stats.pearsonr(x_clean[x2], x_clean[y2])
                    rho, p2 = stats.spearmanr(x_clean[x2], x_clean[y2])
                    st.markdown(f"""
                    <div class="metric-card" style="margin-top:1rem">
                        <span class="metric-value">{r:.4f}</span>
                        <div class="metric-label">Pearson r (p={p:.4f})</div>
                    </div>
                    <div class="metric-card" style="margin-top:0.5rem">
                        <span class="metric-value" style="color:#7c6ffd">{rho:.4f}</span>
                        <div class="metric-label">Spearman ρ (p={p2:.4f})</div>
                    </div>""", unsafe_allow_html=True)

        with tab2:
            if not cat_cols2 or not num_cols2:
                st.info("Il faut au moins une colonne catégorielle et une numérique.")
            else:
                cat_v = st.selectbox("Variable catégorielle", cat_cols2, key='bv_cat')
                num_v = st.selectbox("Variable numérique", num_cols2, key='bv_num')
                top_n = st.slider("Max catégories", 3, 20, 10, key='bv_topn')
                top_cats = df[cat_v].value_counts().head(top_n).index.tolist()
                df_filt  = df[df[cat_v].isin(top_cats)]
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.box(df_filt, x=cat_v, y=num_v, color=cat_v,
                                 color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24','#38bdf8'])
                    fig.update_layout(title="Boxplot groupé", xaxis_tickangle=-30, **plotly_theme())
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    grp = df_filt.groupby(cat_v)[num_v].agg(['mean','median','std','count']).round(3).reset_index()
                    st.dataframe(grp, use_container_width=True, hide_index=True)

        with tab3:
            if len(cat_cols2) < 2:
                st.info("Il faut au moins 2 colonnes catégorielles.")
            else:
                c1 = st.selectbox("Colonne 1", cat_cols2, key='bv_c1')
                c2 = st.selectbox("Colonne 2", [c for c in cat_cols2 if c!=c1], key='bv_c2')
                ct = pd.crosstab(df[c1], df[c2])
                fig = px.imshow(ct, text_auto=True, aspect='auto',
                                color_continuous_scale=['#161920','#7c6ffd','#00e5a0'])
                fig.update_layout(title=f"Table de contingence : {c1} × {c2}", **plotly_theme())
                st.plotly_chart(fig, use_container_width=True)
                # Chi2
                from scipy.stats import chi2_contingency
                chi2, p_chi, dof, _ = chi2_contingency(ct)
                st.markdown(f"""
                <div class="info-box">
                    Test du χ² : χ²={chi2:.4f}, ddl={dof}, <b>p={p_chi:.6f}</b><br>
                    {'✅ Association significative (p < 0.05)' if p_chi < 0.05 else '❌ Pas d\'association significative (p ≥ 0.05)'}
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 7 — EXPORT
# ═══════════════════════════════════════════════════════════════
elif page == "⬇️ Export":
    st.markdown('<p class="section-header">Export de la base nettoyée</p>', unsafe_allow_html=True)

    df_clean = st.session_state.df
    n_ori    = len(st.session_state.df_original)
    n_clean  = len(df_clean)
    n_cols_ori   = len(st.session_state.df_original.columns)
    n_cols_clean = len(df_clean.columns)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <span class="metric-value">{n_ori:,}</span><div class="metric-label">Lignes originales</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <span class="metric-value" style="color:#00e5a0">{n_clean:,}</span><div class="metric-label">Lignes nettoyées</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <span class="metric-value">{n_cols_ori}</span><div class="metric-label">Colonnes orig.</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <span class="metric-value" style="color:#00e5a0">{n_cols_clean}</span><div class="metric-label">Colonnes finales</div></div>""", unsafe_allow_html=True)

    # Missing final
    miss_final = df_clean.isnull().sum().sum()
    st.markdown(f"""
    <div class="info-box">
        Valeurs manquantes restantes : <b>{miss_final}</b> ({miss_final/(df_clean.shape[0]*df_clean.shape[1])*100:.2f}% du total)
    </div>""", unsafe_allow_html=True)

    # Preview
    st.subheader("Aperçu de la base finale")
    st.dataframe(df_clean.head(20), use_container_width=True)

    # Log recap
    st.markdown('<p class="section-header">Journal des transformations</p>', unsafe_allow_html=True)
    if st.session_state.cleaning_log:
        log_text = "\n".join([f"{i+1}. {e}" for i,e in enumerate(st.session_state.cleaning_log)])
        st.code(log_text, language='text')
    else:
        st.info("Aucune transformation enregistrée.")

    # Download
    st.markdown('<p class="section-header">Téléchargement</p>', unsafe_allow_html=True)
    fmt = st.selectbox("Format d'export", ["CSV", "Excel", "JSON"])
    filename = st.text_input("Nom du fichier", value="data_nettoyee")

    data, mime, ext = download_df(df_clean, fmt)
    st.download_button(
        label=f"⬇️  Télécharger {filename}{ext}",
        data=data,
        file_name=f"{filename}{ext}",
        mime=mime,
        use_container_width=True
    )

    # Export du rapport de nettoyage
    if st.session_state.cleaning_log:
        report  = f"# Rapport de nettoyage — DataClean Pro\n\n"
        report += f"Fichier source : {st.session_state.filename}\n"
        report += f"Lignes : {n_ori} → {n_clean}\n"
        report += f"Colonnes : {n_cols_ori} → {n_cols_clean}\n\n"
        report += "## Opérations effectuées\n"
        for i, e in enumerate(st.session_state.cleaning_log, 1):
            report += f"{i}. {e}\n"
        st.download_button(
            label="📄 Télécharger le rapport de nettoyage (.txt)",
            data=report.encode(),
            file_name="rapport_nettoyage.txt",
            mime="text/plain",
            use_container_width=True
        )
