import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal
import io
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DataClean Pro",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');
:root {
    --bg:#0d0f14; --surface:#161920; --surface2:#1e2230;
    --border:#2a2f3d; --accent:#00e5a0; --accent2:#7c6ffd;
    --text:#e8eaf0; --muted:#6b7280;
    --warning:#fbbf24; --danger:#ef4444;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg);color:var(--text);}
.stApp{background-color:var(--bg);}
section[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--border);}
.hero-header{background:linear-gradient(135deg,var(--surface) 0%,var(--surface2) 100%);border:1px solid var(--border);border-radius:16px;padding:2rem 2.5rem;margin-bottom:2rem;position:relative;overflow:hidden;}
.hero-header::before{content:'';position:absolute;top:-40%;right:-10%;width:300px;height:300px;background:radial-gradient(circle,rgba(0,229,160,0.08) 0%,transparent 70%);border-radius:50%;}
.hero-title{font-family:'Space Mono',monospace;font-size:2rem;font-weight:700;color:var(--text);margin:0;}
.hero-title span{color:var(--accent);}
.hero-sub{color:var(--muted);font-size:.95rem;margin-top:.4rem;}
.hero-author{color:var(--accent2);font-size:.85rem;margin-top:.3rem;font-style:italic;}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.2rem 1.5rem;transition:border-color .2s;}
.metric-card:hover{border-color:var(--accent);}
.metric-value{font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;color:var(--accent);display:block;}
.metric-label{font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-top:.25rem;}
.section-header{font-family:'Space Mono',monospace;font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);border-bottom:1px solid var(--border);padding-bottom:.5rem;margin:2rem 0 1rem 0;}
.badge{display:inline-block;padding:.2rem .7rem;border-radius:999px;font-size:.75rem;font-weight:600;font-family:'Space Mono',monospace;}
.badge-success{background:rgba(0,229,160,.15);color:var(--accent);border:1px solid rgba(0,229,160,.3);}
.badge-warning{background:rgba(251,191,36,.15);color:var(--warning);border:1px solid rgba(251,191,36,.3);}
.badge-danger{background:rgba(239,68,68,.15);color:var(--danger);border:1px solid rgba(239,68,68,.3);}
.info-box{background:var(--surface);border-left:3px solid var(--accent2);border-radius:0 8px 8px 0;padding:1rem 1.2rem;margin:1rem 0;font-size:.9rem;color:var(--muted);}
.glossary-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.2rem 1.5rem;margin:.6rem 0;}
.glossary-title{font-family:'Space Mono',monospace;font-size:.85rem;color:var(--accent);font-weight:700;}
.glossary-tag{display:inline-block;padding:.15rem .5rem;border-radius:6px;font-size:.7rem;background:rgba(124,111,253,.2);color:var(--accent2);margin-bottom:.4rem;}
.glossary-body{font-size:.88rem;color:var(--muted);line-height:1.6;}
.course-chapter{background:linear-gradient(135deg,var(--surface) 0%,var(--surface2) 100%);border:1px solid var(--border);border-left:4px solid var(--accent);border-radius:0 12px 12px 0;padding:1.5rem 2rem;margin:1rem 0;}
.course-title{font-family:'Space Mono',monospace;font-size:1rem;color:var(--accent);font-weight:700;margin-bottom:.5rem;}
.course-body{font-size:.9rem;color:var(--text);line-height:1.8;}
.course-formula{background:#0d0f14;border:1px solid var(--border);border-radius:8px;padding:.8rem 1.2rem;font-family:'Space Mono',monospace;font-size:.85rem;color:#fbbf24;margin:.5rem 0;}
.join-diagram{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:1rem;text-align:center;font-family:'Space Mono',monospace;font-size:.8rem;color:var(--accent);margin:.5rem 0;}
.stButton>button{background:var(--accent);color:#0d0f14;border:none;border-radius:8px;font-family:'Space Mono',monospace;font-size:.8rem;font-weight:700;letter-spacing:.05em;padding:.6rem 1.5rem;transition:all .2s;}
.stButton>button:hover{background:#00c98a;transform:translateY(-1px);box-shadow:0 4px 20px rgba(0,229,160,.3);}
.stDownloadButton>button{background:transparent;color:var(--accent);border:1px solid var(--accent);border-radius:8px;font-family:'Space Mono',monospace;font-size:.8rem;font-weight:700;}
.stDownloadButton>button:hover{background:rgba(0,229,160,.1);}
.footer{text-align:center;color:var(--muted);font-size:.8rem;padding:2rem 0 1rem 0;border-top:1px solid var(--border);margin-top:3rem;}
.footer span{color:var(--accent2);}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────
def load_data(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
        df = None
        for enc in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding=enc)
                break
            except Exception:
                continue
        if df is None:
            st.error("Impossible de lire le fichier. Vérifiez l'encodage.")
            return None
    elif name.endswith((".xls", ".xlsx")):
        xl = pd.ExcelFile(uploaded_file)
        sheet = xl.sheet_names[0] if len(xl.sheet_names)==1 else st.sidebar.selectbox("Feuille", xl.sheet_names)
        df = xl.parse(sheet)
    else:
        st.error("Format non supporté."); return None
    return df

def detect_column_types(df):
    return (df.select_dtypes(include=[np.number]).columns.tolist(),
            df.select_dtypes(include=['object','category']).columns.tolist(),
            df.select_dtypes(include=['datetime64']).columns.tolist(),
            df.select_dtypes(include=['bool']).columns.tolist())

def plotly_theme():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(22,25,32,1)',
        font=dict(family='DM Sans', color='#e8eaf0', size=12),
        xaxis=dict(gridcolor='#2a2f3d', zerolinecolor='#2a2f3d', tickfont=dict(color='#6b7280')),
        yaxis=dict(gridcolor='#2a2f3d', zerolinecolor='#2a2f3d', tickfont=dict(color='#6b7280')),
        colorway=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24','#38bdf8','#f472b6','#a3e635'],
        margin=dict(t=40,b=40,l=40,r=20)
    )

def download_df(df, fmt):
    if fmt=="CSV": return df.to_csv(index=False).encode('utf-8'),"text/csv",".csv"
    elif fmt=="Excel":
        buf=io.BytesIO(); df.to_excel(buf,index=False,engine='openpyxl')
        return buf.getvalue(),"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",".xlsx"
    else: return df.to_json(orient='records').encode(),"application/json",".json"

def interpret_p(p, alpha=0.05):
    return f"✅ Significatif (p={p:.4f} < {alpha})" if p < alpha else f"❌ Non significatif (p={p:.4f} ≥ {alpha})"

def interpret_r(r):
    a = abs(r)
    if a >= 0.9: label = "très forte"
    elif a >= 0.7: label = "forte"
    elif a >= 0.5: label = "modérée"
    elif a >= 0.3: label = "faible"
    else: label = "très faible / nulle"
    direction = "positive" if r > 0 else "négative"
    return f"Corrélation {label} {direction}"


# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧹 DataClean Pro")
    st.markdown("---")
    uploaded_file = st.file_uploader("Importer un fichier", type=["csv","xls","xlsx"])
    if uploaded_file:
        st.markdown("### Navigation")
        page = st.radio("Section", [
            "📊 Aperçu & Stats",
            "🔍 Qualité des données",
            "📈 Visualisations",
            "🎨 DataViz Avancée",
            "🔧 Nettoyage",
            "🔄 Transformations",
            "📉 Analyse bivariée",
            "🔗 Jointures & Fusion",
            "📋 Agrégation & Pivot",
            "🤖 Analyse avancée",
            "📚 Guide & Glossaire",
            "🎓 Cours Data Science",
            "⬇️ Export",
        ], label_visibility="collapsed")
    else:
        page = None
    st.markdown("---")
    st.markdown("<div style='color:#6b7280;font-size:.75rem;text-align:center'>par <b style='color:#7c6ffd'>Grâce Delesth NGANGA</b></div>", unsafe_allow_html=True)


# ─── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <p class="hero-title">Data<span>Clean</span> Pro</p>
  <p class="hero-sub">Nettoyage & analyse exploratoire de données — prêt pour la production</p>
  <p class="hero-author">✦ Développé par Grâce Delesth NGANGA</p>
</div>
""", unsafe_allow_html=True)

if not uploaded_file:
    c1,c2,c3 = st.columns(3)
    for c,n,label,color in [(c1,"①","Importer CSV ou Excel","#7c6ffd"),(c2,"②","Explorer & nettoyer","#00e5a0"),(c3,"③","Exporter la base propre","#ff6b6b")]:
        with c:
            st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:{color}">{n}</span><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">← Commencez par importer votre fichier dans la barre latérale.<br>Formats supportés : <b>CSV</b>, <b>XLS</b>, <b>XLSX</b></div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)
    st.stop()

# ─── Load data ────────────────────────────────────────────────
if 'df_original' not in st.session_state or st.session_state.get('filename') != uploaded_file.name:
    df_raw = load_data(uploaded_file)
    if df_raw is None: st.stop()
    st.session_state.df_original = df_raw.copy()
    st.session_state.df = df_raw.copy()
    st.session_state.filename = uploaded_file.name
    st.session_state.cleaning_log = []

df = st.session_state.df
df_ori = st.session_state.df_original
num_cols, cat_cols, dt_cols, bool_cols = detect_column_types(df)


# ══════════════════════════════════════════════════════════════
# PAGE 1 — APERÇU & STATS
# ══════════════════════════════════════════════════════════════
if page == "📊 Aperçu & Stats":
    n_rows, n_cols_count = df.shape
    n_missing = int(df.isnull().sum().sum())
    n_dupl = int(df.duplicated().sum())
    mem = df.memory_usage(deep=True).sum()
    mem_str = f"{mem/1024:.1f} Ko" if mem < 1024**2 else f"{mem/1024**2:.1f} Mo"
    pct_complete = round((1 - n_missing/(n_rows*n_cols_count))*100, 1) if n_rows*n_cols_count > 0 else 100

    st.markdown('<p class="section-header">Vue d\'ensemble</p>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for c,(label,val,color) in zip([c1,c2,c3,c4,c5,c6],[
        ("Lignes",f"{n_rows:,}","#00e5a0"),
        ("Colonnes",f"{n_cols_count}","#7c6ffd"),
        ("Valeurs manq.",f"{n_missing:,}","#fbbf24"),
        ("Doublons",f"{n_dupl:,}","#ff6b6b"),
        ("% complétude",f"{pct_complete}%","#38bdf8"),
        ("Mémoire",mem_str,"#f472b6"),
    ]):
        with c:
            st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:{color}">{val}</span><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Aperçu des données</p>', unsafe_allow_html=True)
    n_prev = st.slider("Lignes à afficher", 5, 100, 10)
    opt = st.radio("Afficher", ["Début","Fin","Aléatoire"], horizontal=True)
    st.dataframe({"Début":df.head,"Fin":df.tail,"Aléatoire":lambda n: df.sample(min(n,len(df)))}[opt](n_prev), use_container_width=True)

    st.markdown('<p class="section-header">Types de colonnes</p>', unsafe_allow_html=True)
    dtype_df = pd.DataFrame({
        'Colonne': df.columns,
        'Type': df.dtypes.astype(str).values,
        'Non-null': df.count().values,
        '% non-null': (df.count().values/len(df)*100).round(2),
        'Valeurs uniques': [df[c].nunique() for c in df.columns],
        'Exemple': [str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0]>0 else 'N/A' for c in df.columns]
    })
    st.dataframe(dtype_df, use_container_width=True)

    st.markdown('<p class="section-header">Statistiques descriptives</p>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Numériques","Catégorielles"])
    with tab1:
        if num_cols:
            desc = df[num_cols].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T.round(4)
            desc['skewness'] = df[num_cols].skew().round(4)
            desc['kurtosis'] = df[num_cols].kurt().round(4)
            desc['CV (%)']   = (df[num_cols].std()/df[num_cols].mean()*100).round(2)
            st.dataframe(desc, use_container_width=True)
            st.markdown('<div class="info-box"><b>Skewness</b> : asymétrie de la distribution (0 = symétrique, >0 = queue à droite, <0 = queue à gauche)<br><b>Kurtosis</b> : aplatissement (>0 = pointue, <0 = aplatie)<br><b>CV</b> : coefficient de variation, mesure la dispersion relative</div>', unsafe_allow_html=True)
        else: st.info("Aucune colonne numérique.")
    with tab2:
        if cat_cols:
            cat_desc = pd.DataFrame({
                'Colonne': cat_cols,
                'Count': [df[c].count() for c in cat_cols],
                'Uniques': [df[c].nunique() for c in cat_cols],
                'Top valeur': [df[c].mode()[0] if df[c].count()>0 else 'N/A' for c in cat_cols],
                'Freq. top (%)': [(df[c].value_counts().iloc[0]/df[c].count()*100).round(2) if df[c].count()>0 else 0 for c in cat_cols],
            })
            st.dataframe(cat_desc, use_container_width=True)
        else: st.info("Aucune colonne catégorielle.")

    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — QUALITÉ
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Qualité des données":
    st.markdown('<p class="section-header">Analyse de la qualité</p>', unsafe_allow_html=True)

    miss = df.isnull().sum().reset_index()
    miss.columns = ['Colonne','Manquantes']
    miss['%'] = (miss['Manquantes']/len(df)*100).round(2)
    miss = miss.sort_values('Manquantes', ascending=False)

    c1,c2 = st.columns([2,1])
    with c1:
        fig = go.Figure()
        fig.add_bar(x=miss['Colonne'], y=miss['%'],
            marker_color=['#ef4444' if p>30 else '#fbbf24' if p>5 else '#00e5a0' for p in miss['%']],
            text=[f"{p}%" for p in miss['%']], textposition='outside')
        fig.update_layout(title="% valeurs manquantes par colonne", xaxis_tickangle=-45, **plotly_theme())
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(miss[['Colonne','Manquantes','%']], use_container_width=True, hide_index=True)

    st.markdown('<p class="section-header">Doublons</p>', unsafe_allow_html=True)
    n_dupl = df.duplicated().sum()
    if n_dupl > 0:
        st.markdown(f'<span class="badge badge-danger">⚠ {n_dupl} doublons ({n_dupl/len(df)*100:.2f}%)</span>', unsafe_allow_html=True)
        with st.expander("Afficher les doublons"):
            st.dataframe(df[df.duplicated(keep=False)], use_container_width=True)
    else:
        st.markdown('<span class="badge badge-success">✓ Aucun doublon</span>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Unicité des colonnes</p>', unsafe_allow_html=True)
    uniq = pd.DataFrame({'Colonne':df.columns,'Uniques':[df[c].nunique() for c in df.columns],'Total':len(df)})
    uniq['% unicité'] = (uniq['Uniques']/uniq['Total']*100).round(2)
    fig2 = px.bar(uniq.sort_values('Uniques',ascending=True), x='% unicité', y='Colonne',
                  orientation='h', color='% unicité', color_continuous_scale=['#1e2230','#7c6ffd','#00e5a0'], text='Uniques')
    fig2.update_layout(title="Unicité des colonnes", **plotly_theme())
    st.plotly_chart(fig2, use_container_width=True)

    if num_cols:
        st.markdown('<p class="section-header">Détection des outliers (IQR)</p>', unsafe_allow_html=True)
        out_rows = []
        for c in num_cols:
            q1,q3 = df[c].quantile(.25), df[c].quantile(.75)
            iqr = q3-q1
            lo,hi = q1-1.5*iqr, q3+1.5*iqr
            out = df[(df[c]<lo)|(df[c]>hi)][c]
            out_rows.append({'Colonne':c,'Q1':round(q1,4),'Q3':round(q3,4),'Borne inf':round(lo,4),
                             'Borne sup':round(hi,4),'Nb outliers':len(out),'%':round(len(out)/len(df)*100,2)})
        st.dataframe(pd.DataFrame(out_rows).sort_values('Nb outliers',ascending=False), use_container_width=True, hide_index=True)

    st.markdown('<p class="section-header">Colonnes quasi-constantes (≥ 95%)</p>', unsafe_allow_html=True)
    quasi = [{'Colonne':c,'Valeur dominante':df[c].mode()[0],'Fréquence (%)':round(df[c].value_counts(normalize=True).iloc[0]*100,2)}
             for c in df.columns if df[c].nunique()>0 and df[c].value_counts(normalize=True).iloc[0]>=0.95]
    if quasi: st.dataframe(pd.DataFrame(quasi), use_container_width=True, hide_index=True)
    else: st.markdown('<span class="badge badge-success">✓ Aucune colonne quasi-constante</span>', unsafe_allow_html=True)

    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Visualisations":
    st.markdown('<p class="section-header">Visualisations exploratoires</p>', unsafe_allow_html=True)
    viz_type = st.selectbox("Type de visualisation", [
        "Histogramme (distribution)", "Boîte à moustaches (boxplot)", "Violon plot",
        "Barplot (catégoriel)", "Pie chart", "Heatmap de corrélation",
        "Pairplot", "Série temporelle", "Scatter plot", "Carte des manquants"
    ])

    if viz_type == "Histogramme (distribution)":
        if not num_cols: st.warning("Aucune colonne numérique.")
        else:
            col = st.selectbox("Colonne", num_cols)
            bins = st.slider("Bins", 5, 100, 30)
            kde  = st.checkbox("Overlay KDE", True)
            fig = go.Figure()
            fig.add_histogram(x=df[col].dropna(), nbinsx=bins, marker_color='#7c6ffd', opacity=0.8, name='Count')
            if kde:
                from scipy.stats import gaussian_kde
                vals = df[col].dropna().values
                if len(vals)>1:
                    kd = gaussian_kde(vals)
                    xr = np.linspace(vals.min(), vals.max(), 300)
                    scale = len(vals)*(vals.max()-vals.min())/bins
                    fig.add_scatter(x=xr, y=kd(xr)*scale, mode='lines', line=dict(color='#00e5a0',width=2), name='KDE')
            fig.update_layout(title=f"Distribution — {col}", **plotly_theme())
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df[col].describe().round(4).to_frame(), use_container_width=True)

    elif viz_type == "Boîte à moustaches (boxplot)":
        if not num_cols: st.warning("Aucune colonne numérique.")
        else:
            cols_sel = st.multiselect("Colonnes", num_cols, default=num_cols[:min(5,len(num_cols))])
            grp = st.selectbox("Grouper par (optionnel)", ["Aucun"]+cat_cols)
            if cols_sel:
                if grp=="Aucun":
                    fig = go.Figure()
                    colors = ['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24','#38bdf8']
                    for i,c in enumerate(cols_sel):
                        fig.add_box(y=df[c].dropna(), name=c, marker_color=colors[i%5])
                else:
                    fig = px.box(df, y=cols_sel[0], color=grp, color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24'])
                fig.update_layout(title="Boxplot", **plotly_theme())
                st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Violon plot":
        if not num_cols: st.warning("Aucune colonne numérique.")
        else:
            col = st.selectbox("Colonne", num_cols)
            grp = st.selectbox("Grouper par", ["Aucun"]+cat_cols)
            if grp=="Aucun":
                fig = px.violin(df, y=col, box=True, points="outliers", color_discrete_sequence=['#7c6ffd'])
            else:
                fig = px.violin(df, y=col, x=grp, box=True, points="outliers", color=grp,
                                color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24'])
            fig.update_layout(title=f"Violin — {col}", **plotly_theme())
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Barplot (catégoriel)":
        if not cat_cols: st.warning("Aucune colonne catégorielle.")
        else:
            col = st.selectbox("Colonne", cat_cols)
            top_n = st.slider("Top N", 5, 50, 15)
            vc = df[col].value_counts().head(top_n).reset_index()
            vc.columns = ['Valeur','Count']
            fig = px.bar(vc, x='Valeur', y='Count', color='Count',
                         color_continuous_scale=['#1e2230','#7c6ffd','#00e5a0'], text='Count')
            fig.update_traces(textposition='outside')
            fig.update_layout(title=f"Distribution — {col}", xaxis_tickangle=-45, **plotly_theme())
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Pie chart":
        if not cat_cols: st.warning("Aucune colonne catégorielle.")
        else:
            col = st.selectbox("Colonne", cat_cols)
            top_n = st.slider("Top N", 3, 20, 8)
            vc = df[col].value_counts().head(top_n)
            fig = px.pie(values=vc.values, names=vc.index,
                         color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24','#38bdf8','#f472b6'])
            fig.update_layout(title=f"Répartition — {col}", **plotly_theme())
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Heatmap de corrélation":
        if len(num_cols)<2: st.warning("Il faut au moins 2 colonnes numériques.")
        else:
            method = st.selectbox("Méthode", ["pearson","spearman","kendall"])
            corr = df[num_cols].corr(method=method)
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=['#ef4444','#161920','#00e5a0'], zmin=-1, zmax=1, aspect='auto')
            fig.update_layout(title=f"Matrice de corrélation ({method})", **plotly_theme())
            st.plotly_chart(fig, use_container_width=True)
            thr = st.slider("Seuil corrélation forte", 0.5, 0.99, 0.7)
            pairs = [{'Var 1':corr.columns[i],'Var 2':corr.columns[j],'Corrélation':round(corr.iloc[i,j],4)}
                     for i in range(len(corr.columns)) for j in range(i+1,len(corr.columns)) if abs(corr.iloc[i,j])>=thr]
            if pairs: st.dataframe(pd.DataFrame(pairs).sort_values('Corrélation',key=abs,ascending=False), use_container_width=True, hide_index=True)

    elif viz_type == "Pairplot":
        if len(num_cols)<2: st.warning("Il faut au moins 2 colonnes numériques.")
        else:
            sel = st.multiselect("Colonnes", num_cols, default=num_cols[:min(4,len(num_cols))])
            color_by = st.selectbox("Couleur par", ["Aucun"]+cat_cols)
            if len(sel)>=2:
                fig = px.scatter_matrix(df, dimensions=sel,
                                        color=None if color_by=="Aucun" else color_by,
                                        color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24'])
                fig.update_traces(diagonal_visible=True, showupperhalf=False)
                fig.update_layout(title="Pairplot", **plotly_theme())
                st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Série temporelle":
        date_candidates = dt_cols+[c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if not date_candidates: st.warning("Aucune colonne date détectée.")
        else:
            date_col = st.selectbox("Colonne date", date_candidates)
            val_col  = st.selectbox("Valeur", num_cols)
            agg      = st.selectbox("Agrégation", ["Somme","Moyenne","Médiane","Count"])
            freq     = st.selectbox("Fréquence", ["Jour (D)","Semaine (W)","Mois (ME)","Trimestre (QE)","Année (YE)"])
            try:
                tmp = df.copy()
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
                tmp = tmp.dropna(subset=[date_col])
                freq_map = {"Jour (D)":"D","Semaine (W)":"W","Mois (ME)":"ME","Trimestre (QE)":"QE","Année (YE)":"YE"}
                agg_map  = {"Somme":"sum","Moyenne":"mean","Médiane":"median","Count":"count"}
                ts = tmp.set_index(date_col)[val_col].resample(freq_map[freq]).agg(agg_map[agg]).reset_index()
                fig = px.line(ts, x=date_col, y=val_col, color_discrete_sequence=['#00e5a0'])
                fig.update_traces(line_width=2)
                fig.update_layout(title=f"Série temporelle — {val_col}", **plotly_theme())
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur : {e}")

    elif viz_type == "Scatter plot":
        if len(num_cols)<2: st.warning("Il faut au moins 2 colonnes numériques.")
        else:
            x = st.selectbox("Axe X", num_cols)
            y = st.selectbox("Axe Y", [c for c in num_cols if c!=x])
            color_by = st.selectbox("Couleur par", ["Aucun"]+cat_cols+num_cols)
            size_by  = st.selectbox("Taille par", ["Aucun"]+num_cols)
            show_trend = st.checkbox("Ligne de tendance (régression linéaire)")
            fig = px.scatter(df, x=x, y=y,
                color=None if color_by=="Aucun" else color_by,
                size=None if size_by=="Aucun" else size_by,
                opacity=0.7,
                color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24'],
                color_continuous_scale=['#1e2230','#7c6ffd','#00e5a0'])
            if show_trend:
                clean = df[[x,y]].dropna()
                if len(clean)>1:
                    m,b = np.polyfit(clean[x], clean[y], 1)
                    xr = np.linspace(clean[x].min(), clean[x].max(), 200)
                    fig.add_scatter(x=xr, y=m*xr+b, mode='lines',
                                    line=dict(color='#ff6b6b', width=2, dash='dash'), name='Tendance')
            fig.update_layout(title=f"Scatter — {x} vs {y}", **plotly_theme())
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Carte des manquants":
        fig = px.imshow(df.isnull().astype(int), color_continuous_scale=['#161920','#ef4444'], aspect='auto')
        fig.update_layout(title="Carte des valeurs manquantes", **plotly_theme())
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — NETTOYAGE
# ══════════════════════════════════════════════════════════════
elif page == "🔧 Nettoyage":
    st.markdown('<p class="section-header">Opérations de nettoyage</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Les modifications s\'appliquent de manière cumulative. Chaque opération est tracée dans le journal.</div>', unsafe_allow_html=True)

    with st.expander("🗑 Supprimer les doublons"):
        st.write(f"Doublons détectés : **{df.duplicated().sum()}**")
        subset_cols = st.multiselect("Colonnes de référence (vide = toutes)", df.columns.tolist(), key='dup_cols')
        keep_opt = st.radio("Conserver", ["first","last","Supprimer tous"], key='dup_keep', horizontal=True)
        if st.button("Supprimer les doublons"):
            keep = False if keep_opt=="Supprimer tous" else keep_opt
            before = len(df)
            st.session_state.df = df.drop_duplicates(subset=subset_cols if subset_cols else None, keep=keep).reset_index(drop=True)
            removed = before - len(st.session_state.df)
            st.session_state.cleaning_log.append(f"Suppression de {removed} doublons")
            st.success(f"{removed} doublons supprimés"); df = st.session_state.df

    with st.expander("🩹 Traiter les valeurs manquantes"):
        col_missing = st.selectbox("Colonne cible", ["Toutes les colonnes"]+df.columns.tolist(), key='miss_col')
        strategy = st.selectbox("Stratégie", [
            "Supprimer les lignes","Supprimer les colonnes",
            "Remplacer par la moyenne","Remplacer par la médiane","Remplacer par le mode",
            "Remplacer par une valeur fixe","Remplissage avant (ffill)","Remplissage arrière (bfill)"
        ], key='miss_strat')
        custom_val = st.text_input("Valeur fixe", key='miss_val') if strategy=="Remplacer par une valeur fixe" else None
        if st.button("Appliquer"):
            df_work = st.session_state.df.copy()
            targets = df_work.columns.tolist() if col_missing=="Toutes les colonnes" else [col_missing]
            for c in targets:
                if strategy=="Supprimer les lignes": df_work = df_work.dropna(subset=[c])
                elif strategy=="Supprimer les colonnes": df_work = df_work.drop(columns=[c])
                elif strategy=="Remplacer par la moyenne" and c in num_cols: df_work[c].fillna(df_work[c].mean(), inplace=True)
                elif strategy=="Remplacer par la médiane" and c in num_cols: df_work[c].fillna(df_work[c].median(), inplace=True)
                elif strategy=="Remplacer par le mode": df_work[c].fillna(df_work[c].mode()[0] if df_work[c].count()>0 else np.nan, inplace=True)
                elif strategy=="Remplacer par une valeur fixe":
                    try: fill = float(custom_val) if df_work[c].dtype in [float,int] else custom_val
                    except: fill = custom_val
                    df_work[c].fillna(fill, inplace=True)
                elif strategy=="Remplissage avant (ffill)": df_work[c] = df_work[c].ffill()
                elif strategy=="Remplissage arrière (bfill)": df_work[c] = df_work[c].bfill()
            st.session_state.df = df_work; df = df_work
            msg = f"Valeurs manquantes — {strategy} sur {col_missing}"
            st.session_state.cleaning_log.append(msg); st.success(msg)

    with st.expander("📐 Traiter les outliers"):
        if num_cols:
            out_col = st.selectbox("Colonne", num_cols, key='out_col')
            out_meth = st.selectbox("Méthode", ["IQR (×1.5)","IQR (×3)","Z-score (|z|>3)"], key='out_meth')
            out_act  = st.selectbox("Action", ["Supprimer les lignes","Remplacer par les bornes (winsorize)","Remplacer par NaN"], key='out_act')
            if st.button("Traiter les outliers"):
                df_work = st.session_state.df.copy()
                s = df_work[out_col].dropna()
                if "IQR" in out_meth:
                    f = 1.5 if "1.5" in out_meth else 3.0
                    q1,q3 = s.quantile(.25), s.quantile(.75)
                    lo,hi = q1-f*(q3-q1), q3+f*(q3-q1)
                else:
                    lo,hi = s[((s-s.mean())/s.std()).abs()<=3].min(), s[((s-s.mean())/s.std()).abs()<=3].max()
                mask = (df_work[out_col]<lo)|(df_work[out_col]>hi)
                n_out = mask.sum()
                if out_act=="Supprimer les lignes": df_work = df_work[~mask]
                elif out_act=="Remplacer par les bornes (winsorize)":
                    df_work.loc[df_work[out_col]<lo, out_col] = lo
                    df_work.loc[df_work[out_col]>hi, out_col] = hi
                else: df_work.loc[mask, out_col] = np.nan
                st.session_state.df = df_work; df = df_work
                msg = f"Outliers ({out_col}) — {out_act} ({n_out} lignes)"
                st.session_state.cleaning_log.append(msg); st.success(msg)

    with st.expander("🗂 Supprimer des colonnes"):
        cols_drop = st.multiselect("Colonnes à supprimer", df.columns.tolist(), key='drop_cols')
        if st.button("Supprimer") and cols_drop:
            st.session_state.df = df.drop(columns=cols_drop); df = st.session_state.df
            st.session_state.cleaning_log.append(f"Colonnes supprimées : {cols_drop}"); st.success("Colonnes supprimées.")

    with st.expander("✏️ Renommer une colonne"):
        ren_col = st.selectbox("Colonne", df.columns.tolist(), key='ren_col')
        new_name = st.text_input("Nouveau nom", key='ren_name')
        if st.button("Renommer") and new_name:
            st.session_state.df = df.rename(columns={ren_col:new_name}); df = st.session_state.df
            st.session_state.cleaning_log.append(f"Renommage : {ren_col} → {new_name}"); st.success("Renommé.")

    with st.expander("🔎 Filtrer les lignes"):
        filter_col = st.selectbox("Colonne", df.columns.tolist(), key='filt_col')
        if filter_col in num_cols:
            lo_f,hi_f = float(df[filter_col].min()), float(df[filter_col].max())
            rng = st.slider("Plage", lo_f, hi_f, (lo_f,hi_f), key='filt_rng')
            if st.button("Appliquer filtre numérique"):
                st.session_state.df = df[(df[filter_col]>=rng[0])&(df[filter_col]<=rng[1])]; df = st.session_state.df
                msg = f"Filtre : {filter_col} ∈ [{rng[0]},{rng[1]}]"
                st.session_state.cleaning_log.append(msg); st.success(msg)
        else:
            uv = df[filter_col].dropna().unique().tolist()
            kv = st.multiselect("Valeurs à garder", uv, default=uv, key='filt_vals')
            if st.button("Appliquer filtre catégoriel"):
                st.session_state.df = df[df[filter_col].isin(kv)]; df = st.session_state.df
                msg = f"Filtre catégoriel : {filter_col}"
                st.session_state.cleaning_log.append(msg); st.success(msg)

    st.markdown("---")
    if st.button("🔁 Réinitialiser toutes les modifications"):
        st.session_state.df = st.session_state.df_original.copy()
        st.session_state.cleaning_log = []
        st.success("Données réinitialisées.")

    st.markdown('<p class="section-header">Journal des opérations</p>', unsafe_allow_html=True)
    if st.session_state.cleaning_log:
        for i,e in enumerate(st.session_state.cleaning_log,1): st.markdown(f"`{i}.` {e}")
    else:
        st.markdown('<span class="badge badge-warning">Aucune opération effectuée</span>', unsafe_allow_html=True)

    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════
elif page == "🔄 Transformations":
    st.markdown('<p class="section-header">Transformations & ingénierie des variables</p>', unsafe_allow_html=True)

    with st.expander("🔢 Convertir les types"):
        conv_col  = st.selectbox("Colonne", df.columns.tolist(), key='conv_col')
        conv_type = st.selectbox("Nouveau type", ["int","float","str","bool","datetime"], key='conv_type')
        if st.button("Convertir"):
            try:
                df_work = st.session_state.df.copy()
                if conv_type=="datetime": df_work[conv_col] = pd.to_datetime(df_work[conv_col], errors='coerce')
                elif conv_type=="bool": df_work[conv_col] = df_work[conv_col].astype(bool)
                elif conv_type=="int": df_work[conv_col] = pd.to_numeric(df_work[conv_col], errors='coerce').astype('Int64')
                elif conv_type=="float": df_work[conv_col] = pd.to_numeric(df_work[conv_col], errors='coerce')
                else: df_work[conv_col] = df_work[conv_col].astype(str)
                st.session_state.df = df_work; df = df_work
                st.session_state.cleaning_log.append(f"Conversion : {conv_col} → {conv_type}"); st.success("Converti.")
            except Exception as e: st.error(f"Erreur : {e}")

    with st.expander("📏 Normaliser / Standardiser"):
        if num_cols:
            norm_cols = st.multiselect("Colonnes", num_cols, key='norm_cols')
            norm_meth = st.radio("Méthode", ["Min-Max [0,1]","Z-score","Robust (médiane/IQR)"], key='norm_meth', horizontal=True)
            if st.button("Normaliser") and norm_cols:
                df_work = st.session_state.df.copy()
                for c in norm_cols:
                    s = df_work[c].copy()
                    if norm_meth=="Min-Max [0,1]": df_work[c] = (s-s.min())/(s.max()-s.min())
                    elif norm_meth=="Z-score": df_work[c] = (s-s.mean())/s.std()
                    else: df_work[c] = (s-s.median())/(s.quantile(.75)-s.quantile(.25))
                st.session_state.df = df_work; df = df_work
                st.session_state.cleaning_log.append(f"Normalisation ({norm_meth}) : {norm_cols}"); st.success("Normalisé.")

    with st.expander("🏷 Encoder les catégorielles"):
        if cat_cols:
            enc_col = st.selectbox("Colonne", cat_cols, key='enc_col')
            enc_meth = st.radio("Méthode", ["Label Encoding","One-Hot Encoding","Ordinal"], key='enc_meth', horizontal=True)
            if enc_meth=="Ordinal":
                cats = df[enc_col].dropna().unique().tolist()
                order_str = st.text_input("Ordre (séparé par virgules)", value=", ".join([str(c) for c in cats]), key='ord_str')
            if st.button("Encoder"):
                df_work = st.session_state.df.copy()
                if enc_meth=="Label Encoding": df_work[enc_col+"_encoded"] = pd.factorize(df_work[enc_col])[0]
                elif enc_meth=="One-Hot Encoding":
                    dummies = pd.get_dummies(df_work[enc_col], prefix=enc_col, dtype=int)
                    df_work = pd.concat([df_work, dummies], axis=1)
                else:
                    order = [x.strip() for x in order_str.split(',')]
                    df_work[enc_col+"_ordinal"] = df_work[enc_col].map({v:i for i,v in enumerate(order)})
                st.session_state.df = df_work; df = df_work
                st.session_state.cleaning_log.append(f"Encodage ({enc_meth}) : {enc_col}"); st.success("Encodé.")

    with st.expander("➕ Colonne calculée"):
        new_col_name = st.text_input("Nom de la nouvelle colonne", key='new_col')
        formula = st.text_area("Formule (ex: df['prix'] * df['qte'])", key='formula')
        if st.button("Créer") and new_col_name and formula:
            try:
                df_work = st.session_state.df.copy()
                df_work[new_col_name] = eval(formula, {"df":df_work,"np":np,"pd":pd})
                st.session_state.df = df_work; df = df_work
                st.session_state.cleaning_log.append(f"Nouvelle colonne : {new_col_name}"); st.success("Créée.")
            except Exception as e: st.error(f"Erreur dans la formule : {e}")

    with st.expander("📅 Extraire composantes de date"):
        date_all = dt_cols+[c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if date_all:
            dt_col = st.selectbox("Colonne date", date_all, key='dt_col')
            comps  = st.multiselect("Composantes", ["Année","Mois","Jour","Heure","Jour de la semaine","Trimestre"], key='dt_comps')
            if st.button("Extraire") and comps:
                df_work = st.session_state.df.copy()
                df_work[dt_col] = pd.to_datetime(df_work[dt_col], errors='coerce')
                mp = {"Année":("year","_annee"),"Mois":("month","_mois"),"Jour":("day","_jour"),
                      "Heure":("hour","_heure"),"Jour de la semaine":("dayofweek","_dow"),"Trimestre":("quarter","_trim")}
                for comp in comps:
                    attr, suffix = mp[comp]
                    df_work[dt_col+suffix] = getattr(df_work[dt_col].dt, attr)
                st.session_state.df = df_work; df = df_work
                st.session_state.cleaning_log.append(f"Extraction date ({dt_col})"); st.success("Extrait.")

    with st.expander("🔡 Nettoyer le texte"):
        if cat_cols:
            txt_col = st.selectbox("Colonne", cat_cols, key='txt_col')
            txt_ops = st.multiselect("Opérations", [
                "Minuscules","Majuscules","Capitaliser","Strip (espaces)","Supprimer caractères spéciaux"
            ], key='txt_ops')
            if st.button("Nettoyer") and txt_ops:
                df_work = st.session_state.df.copy()
                s = df_work[txt_col].astype(str)
                for op in txt_ops:
                    if op=="Minuscules": s = s.str.lower()
                    elif op=="Majuscules": s = s.str.upper()
                    elif op=="Capitaliser": s = s.str.capitalize()
                    elif op=="Strip (espaces)": s = s.str.strip()
                    elif op=="Supprimer caractères spéciaux": s = s.str.replace(r'[^a-zA-Z0-9\s]','',regex=True)
                df_work[txt_col] = s
                st.session_state.df = df_work; df = df_work
                st.session_state.cleaning_log.append(f"Nettoyage texte ({txt_col})"); st.success("Nettoyé.")

    # ── NOUVELLES TRANSFORMATIONS ─────────────────────────────

    with st.expander("🔀 Colonne conditionnelle (IF / ELSE)"):
        st.markdown('<div class="info-box"><b>Équivalent Power Query : Colonne conditionnelle</b><br>Crée une nouvelle colonne selon des règles SI/SINON. Ex : si dateClo est vide → "Ouvert" sinon "Fermé".</div>', unsafe_allow_html=True)
        cond_new_col = st.text_input("Nom de la nouvelle colonne", key='cond_col_name', placeholder="ex: Statut")
        cond_source  = st.selectbox("Colonne source", df.columns.tolist(), key='cond_source')
        cond_type    = st.radio("Type de condition", ["Est vide (NaN/null)", "Contient", "Égal à", "Supérieur à", "Inférieur à", "Compris entre"], horizontal=True, key='cond_type')

        col1, col2 = st.columns(2)
        with col1:
            cond_val1 = st.text_input("Valeur 1 (seuil / texte recherché)", key='cond_val1', placeholder="ex: 0 ou Ouvert")
        with col2:
            cond_val2 = st.text_input("Valeur 2 (borne sup. si 'Compris entre')", key='cond_val2')

        col3, col4 = st.columns(2)
        with col3:
            then_val = st.text_input("Valeur SI condition vraie (THEN)", key='cond_then', placeholder="ex: Ouvert")
        with col4:
            else_val = st.text_input("Valeur SI condition fausse (ELSE)", key='cond_else', placeholder="ex: Fermé")

        # Multi-règles (IF / ELIF / ELSE)
        st.markdown("**Règles supplémentaires (optionnel — équivalent SINON SI)**")
        extra_rules_raw = st.text_area(
            "Format : condition | valeur_si_vrai (une par ligne)\nEx : df['montant'] > 1000 | Grand compte",
            key='cond_extra', height=80
        )

        if st.button("✅ Créer la colonne conditionnelle") and cond_new_col:
            try:
                df_work = st.session_state.df.copy()
                col = df_work[cond_source]

                # Condition principale
                if cond_type == "Est vide (NaN/null)":
                    mask = col.isnull() | (col.astype(str).str.strip() == '') | (col.astype(str).str.strip().str.lower() == 'nan')
                elif cond_type == "Contient":
                    mask = col.astype(str).str.contains(cond_val1, case=False, na=False)
                elif cond_type == "Égal à":
                    try: mask = col == float(cond_val1)
                    except: mask = col.astype(str) == cond_val1
                elif cond_type == "Supérieur à":
                    mask = pd.to_numeric(col, errors='coerce') > float(cond_val1)
                elif cond_type == "Inférieur à":
                    mask = pd.to_numeric(col, errors='coerce') < float(cond_val1)
                elif cond_type == "Compris entre":
                    num_col = pd.to_numeric(col, errors='coerce')
                    mask = (num_col >= float(cond_val1)) & (num_col <= float(cond_val2))
                else:
                    mask = pd.Series([False]*len(df_work))

                result = np.where(mask, then_val, else_val)

                # Règles supplémentaires (SINON SI)
                if extra_rules_raw.strip():
                    for rule_line in extra_rules_raw.strip().split('\n'):
                        if '|' in rule_line:
                            rule_expr, rule_val = rule_line.split('|', 1)
                            try:
                                rule_mask = eval(rule_expr.strip(), {"df": df_work, "np": np, "pd": pd})
                                result = np.where(rule_mask & ~mask, rule_val.strip(), result)
                            except Exception as re:
                                st.warning(f"Règle ignorée : {rule_expr.strip()} → {re}")

                df_work[cond_new_col] = result
                st.session_state.df = df_work; df = df_work
                msg = f"Colonne conditionnelle '{cond_new_col}' créée"
                st.session_state.cleaning_log.append(msg)
                st.success(msg)
                st.dataframe(df_work[[cond_source, cond_new_col]].head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Erreur : {e}")

    with st.expander("🔁 Remplacer des valeurs (Find & Replace)"):
        st.markdown('<div class="info-box"><b>Équivalent Power Query : Remplacer les valeurs</b><br>Remplace une valeur spécifique par une autre dans une colonne.</div>', unsafe_allow_html=True)
        rep_col  = st.selectbox("Colonne", df.columns.tolist(), key='rep_col')
        rep_mode = st.radio("Mode", ["Valeur exacte","Contient (regex)"], horizontal=True, key='rep_mode')
        rep_from = st.text_input("Valeur à remplacer", key='rep_from', placeholder="ex: N/A ou \\bNA\\b")
        rep_to   = st.text_input("Remplacer par", key='rep_to', placeholder="ex: vide → laisser blanc")
        rep_case = st.checkbox("Respecter la casse", True, key='rep_case')
        if st.button("Remplacer") and rep_from:
            try:
                df_work = st.session_state.df.copy()
                is_regex = (rep_mode == "Contient (regex)")
                replace_to = rep_to if rep_to else np.nan
                df_work[rep_col] = df_work[rep_col].astype(str).str.replace(
                    rep_from, str(replace_to) if rep_to else '', regex=is_regex, case=rep_case
                )
                if not rep_to:
                    df_work[rep_col] = df_work[rep_col].replace('', np.nan)
                st.session_state.df = df_work; df = df_work
                msg = f"Remplacement dans {rep_col} : '{rep_from}' → '{rep_to}'"
                st.session_state.cleaning_log.append(msg); st.success(msg)
            except Exception as e:
                st.error(f"Erreur : {e}")

    with st.expander("✂️ Découper / Extraire du texte"):
        st.markdown('<div class="info-box"><b>Équivalent Power Query : Extraire / Fractionner</b><br>Extrait une partie du texte par position, délimiteur, ou motif.</div>', unsafe_allow_html=True)
        split_col = st.selectbox("Colonne texte", df.columns.tolist(), key='split_col')
        split_mode = st.radio("Mode", ["Positions (début:fin)","Délimiteur (split)","N premiers caractères","N derniers caractères","Extraire entre deux textes"], horizontal=False, key='split_mode')

        if split_mode == "Positions (début:fin)":
            c1,c2 = st.columns(2)
            with c1: pos_start = st.number_input("Position début (0=premier)", min_value=0, value=0, key='pos_start')
            with c2: pos_end   = st.number_input("Position fin (-1=jusqu'à la fin)", value=-1, key='pos_end')
            new_split_col = st.text_input("Nouveau nom de colonne", value=f"{split_col}_extrait", key='split_new1')
            if st.button("Extraire par position"):
                df_work = st.session_state.df.copy()
                end = None if pos_end == -1 else int(pos_end)
                df_work[new_split_col] = df_work[split_col].astype(str).str[int(pos_start):end]
                st.session_state.df = df_work; df = df_work
                st.session_state.cleaning_log.append(f"Extraction position [{pos_start}:{pos_end}] sur {split_col}"); st.success("Extrait.")

        elif split_mode == "Délimiteur (split)":
            delim = st.text_input("Délimiteur", value=";", key='delim')
            part  = st.number_input("Numéro de partie (0 = première)", min_value=0, value=0, key='split_part')
            new_split_col = st.text_input("Nouveau nom", value=f"{split_col}_part{part}", key='split_new2')
            if st.button("Fractionner"):
                try:
                    df_work = st.session_state.df.copy()
                    df_work[new_split_col] = df_work[split_col].astype(str).str.split(delim).str[int(part)]
                    st.session_state.df = df_work; df = df_work
                    st.session_state.cleaning_log.append(f"Split '{delim}' partie {part} sur {split_col}"); st.success("Fractionné.")
                except Exception as e: st.error(f"Erreur : {e}")

        elif split_mode == "N premiers caractères":
            n_chars = st.number_input("Nombre de caractères", min_value=1, value=3, key='n_first')
            new_split_col = st.text_input("Nouveau nom", value=f"{split_col}_{n_chars}premiers", key='split_new3')
            if st.button("Extraire premiers"):
                df_work = st.session_state.df.copy()
                df_work[new_split_col] = df_work[split_col].astype(str).str[:int(n_chars)]
                st.session_state.df = df_work; df = df_work
                st.session_state.cleaning_log.append(f"{n_chars} premiers caractères de {split_col}"); st.success("Extrait.")

        elif split_mode == "N derniers caractères":
            n_chars = st.number_input("Nombre de caractères", min_value=1, value=3, key='n_last')
            new_split_col = st.text_input("Nouveau nom", value=f"{split_col}_{n_chars}derniers", key='split_new4')
            if st.button("Extraire derniers"):
                df_work = st.session_state.df.copy()
                df_work[new_split_col] = df_work[split_col].astype(str).str[-int(n_chars):]
                st.session_state.df = df_work; df = df_work
                st.session_state.cleaning_log.append(f"{n_chars} derniers caractères de {split_col}"); st.success("Extrait.")

        elif split_mode == "Extraire entre deux textes":
            c1,c2 = st.columns(2)
            with c1: txt_before = st.text_input("Texte avant", key='txt_before', placeholder="ex: (")
            with c2: txt_after  = st.text_input("Texte après", key='txt_after', placeholder="ex: )")
            new_split_col = st.text_input("Nouveau nom", value=f"{split_col}_between", key='split_new5')
            if st.button("Extraire entre") and txt_before and txt_after:
                try:
                    import re as re_mod
                    pattern = re_mod.escape(txt_before) + r'(.*?)' + re_mod.escape(txt_after)
                    df_work = st.session_state.df.copy()
                    df_work[new_split_col] = df_work[split_col].astype(str).str.extract(pattern, expand=False)
                    st.session_state.df = df_work; df = df_work
                    st.session_state.cleaning_log.append(f"Extraction entre '{txt_before}' et '{txt_after}' sur {split_col}"); st.success("Extrait.")
                except Exception as e: st.error(f"Erreur : {e}")

    with st.expander("🔢 Discrétisation / Binning (découpage en intervalles)"):
        st.markdown('<div class="info-box"><b>Équivalent Power Query : Colonne à partir d\'exemples / regroupement</b><br>Transforme une variable numérique continue en catégories (tranches d\'âge, niveaux, etc.)</div>', unsafe_allow_html=True)
        if num_cols:
            bin_col  = st.selectbox("Colonne numérique", num_cols, key='bin_col')
            bin_mode = st.radio("Mode", ["Intervalles égaux (equal-width)","Quantiles (equal-frequency)","Bornes personnalisées"], horizontal=True, key='bin_mode')
            bin_new  = st.text_input("Nom de la nouvelle colonne", value=f"{bin_col}_tranche", key='bin_new')

            if bin_mode in ["Intervalles égaux (equal-width)", "Quantiles (equal-frequency)"]:
                n_bins = st.slider("Nombre de tranches", 2, 20, 5, key='n_bins')
                bin_labels_raw = st.text_input("Labels personnalisés (optionnel, séparés par virgules)", key='bin_labels')
            else:
                bins_raw   = st.text_input("Bornes (ex: 0,18,35,60,100)", key='bins_raw')
                labels_raw = st.text_input("Labels (ex: Mineur,Jeune,Adulte,Senior)", key='labels_raw')

            if st.button("Discrétiser"):
                try:
                    df_work = st.session_state.df.copy()
                    if bin_mode == "Intervalles égaux (equal-width)":
                        lbl = [l.strip() for l in bin_labels_raw.split(',')] if bin_labels_raw else None
                        df_work[bin_new] = pd.cut(df_work[bin_col], bins=n_bins, labels=lbl)
                    elif bin_mode == "Quantiles (equal-frequency)":
                        lbl = [l.strip() for l in bin_labels_raw.split(',')] if bin_labels_raw else None
                        df_work[bin_new] = pd.qcut(df_work[bin_col], q=n_bins, labels=lbl, duplicates='drop')
                    else:
                        bornes = [float(x.strip()) for x in bins_raw.split(',')]
                        lbl    = [l.strip() for l in labels_raw.split(',')] if labels_raw else None
                        df_work[bin_new] = pd.cut(df_work[bin_col], bins=bornes, labels=lbl)
                    st.session_state.df = df_work; df = df_work
                    msg = f"Discrétisation ({bin_mode}) : {bin_col} → {bin_new}"
                    st.session_state.cleaning_log.append(msg); st.success(msg)
                    vc = df_work[bin_new].value_counts().reset_index()
                    vc.columns = ['Tranche', 'Count']
                    fig_bin = px.bar(vc, x='Tranche', y='Count', color='Count',
                                     color_continuous_scale=['#1e2230','#7c6ffd','#00e5a0'], text='Count')
                    fig_bin.update_layout(title=f"Répartition des tranches — {bin_new}", **plotly_theme())
                    st.plotly_chart(fig_bin, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur : {e}")

    with st.expander("📐 Transformation mathématique"):
        st.markdown('<div class="info-box">Applique une transformation mathématique sur une colonne numérique (log, sqrt, carré, inverse...).</div>', unsafe_allow_html=True)
        if num_cols:
            math_col  = st.selectbox("Colonne", num_cols, key='math_col')
            math_func = st.selectbox("Transformation", [
                "Log naturel ln(x)","Log base 10","Racine carrée √x","Carré x²","Cube x³",
                "Inverse 1/x","Valeur absolue |x|","Exponentielle eˣ","Arrondir"
            ], key='math_func')
            math_new  = st.text_input("Nom de la nouvelle colonne", value=f"{math_col}_transf", key='math_new')
            decimals  = st.number_input("Décimales (pour Arrondir)", min_value=0, max_value=10, value=2, key='math_dec')
            if st.button("Appliquer la transformation"):
                try:
                    df_work = st.session_state.df.copy()
                    s = df_work[math_col]
                    if math_func == "Log naturel ln(x)": df_work[math_new] = np.log(s.replace(0, np.nan))
                    elif math_func == "Log base 10":     df_work[math_new] = np.log10(s.replace(0, np.nan))
                    elif math_func == "Racine carrée √x": df_work[math_new] = np.sqrt(s.clip(lower=0))
                    elif math_func == "Carré x²":        df_work[math_new] = s ** 2
                    elif math_func == "Cube x³":         df_work[math_new] = s ** 3
                    elif math_func == "Inverse 1/x":     df_work[math_new] = 1 / s.replace(0, np.nan)
                    elif math_func == "Valeur absolue |x|": df_work[math_new] = s.abs()
                    elif math_func == "Exponentielle eˣ": df_work[math_new] = np.exp(s)
                    elif math_func == "Arrondir":        df_work[math_new] = s.round(int(decimals))
                    st.session_state.df = df_work; df = df_work
                    msg = f"Transformation {math_func} sur {math_col} → {math_new}"
                    st.session_state.cleaning_log.append(msg); st.success(msg)
                except Exception as e: st.error(f"Erreur : {e}")

    with st.expander("🔗 Concaténer des colonnes texte"):
        st.markdown('<div class="info-box"><b>Équivalent Power Query : Fusionner des colonnes</b><br>Combine plusieurs colonnes en une seule chaîne de texte.</div>', unsafe_allow_html=True)
        concat_cols = st.multiselect("Colonnes à fusionner (dans l'ordre)", df.columns.tolist(), key='concat_text_cols')
        concat_sep  = st.text_input("Séparateur", value=" ", key='concat_sep', placeholder="ex: espace, _, -")
        concat_new  = st.text_input("Nom de la nouvelle colonne", value="colonne_fusionnée", key='concat_text_new')
        if st.button("Fusionner") and len(concat_cols) >= 2:
            try:
                df_work = st.session_state.df.copy()
                df_work[concat_new] = df_work[concat_cols[0]].astype(str)
                for c in concat_cols[1:]:
                    df_work[concat_new] = df_work[concat_new] + concat_sep + df_work[c].astype(str)
                st.session_state.df = df_work; df = df_work
                msg = f"Fusion colonnes {concat_cols} → {concat_new}"
                st.session_state.cleaning_log.append(msg); st.success(msg)
                st.dataframe(df_work[[*concat_cols, concat_new]].head(5), use_container_width=True)
            except Exception as e: st.error(f"Erreur : {e}")

    with st.expander("🔄 Transposer lignes ↔ colonnes"):
        st.markdown('<div class="info-box">Transpose le dataframe : les lignes deviennent des colonnes et inversement.</div>', unsafe_allow_html=True)
        use_first_as_header = st.checkbox("Utiliser la première ligne comme entête", True, key='transp_header')
        if st.button("Transposer le dataframe"):
            try:
                df_work = st.session_state.df.copy()
                df_transposed = df_work.T
                if use_first_as_header:
                    df_transposed.columns = df_transposed.iloc[0]
                    df_transposed = df_transposed[1:].reset_index()
                else:
                    df_transposed = df_transposed.reset_index()
                st.session_state.df = df_transposed; df = df_transposed
                msg = f"Transposition : {df_work.shape} → {df_transposed.shape}"
                st.session_state.cleaning_log.append(msg); st.success(msg)
                st.dataframe(df_transposed.head(5), use_container_width=True)
            except Exception as e: st.error(f"Erreur : {e}")

    with st.expander("📊 Ranking / Classement"):
        st.markdown('<div class="info-box">Ajoute une colonne de rang basée sur une variable numérique. Utile pour créer des tops ou des classements.</div>', unsafe_allow_html=True)
        if num_cols:
            rank_col    = st.selectbox("Colonne à classer", num_cols, key='rank_col')
            rank_method = st.radio("Méthode", ["average","min","max","first","dense"], horizontal=True, key='rank_method')
            rank_asc    = st.checkbox("Ordre croissant", False, key='rank_asc')
            rank_new    = st.text_input("Nom de la colonne rang", value=f"{rank_col}_rang", key='rank_new')
            if st.button("Créer le classement"):
                df_work = st.session_state.df.copy()
                df_work[rank_new] = df_work[rank_col].rank(method=rank_method, ascending=rank_asc).astype(int)
                st.session_state.df = df_work; df = df_work
                msg = f"Ranking {rank_col} → {rank_new}"
                st.session_state.cleaning_log.append(msg); st.success(msg)
                st.dataframe(df_work[[rank_col, rank_new]].sort_values(rank_new).head(10), use_container_width=True)

    st.markdown("**Aperçu du dataframe transformé**")
    st.dataframe(st.session_state.df.head(10), use_container_width=True)
    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE — DATAVIZ AVANCÉE  ← NOUVEAU
# ══════════════════════════════════════════════════════════════
elif page == "🎨 DataViz Avancée":
    st.markdown('<p class="section-header">DataViz Avancée — Dashboard & graphiques enrichis</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Créez des visualisations professionnelles, personnalisables et exportables. Idéal pour préparer vos rapports et dashboards.</div>', unsafe_allow_html=True)

    viz_adv = st.selectbox("🎨 Choisir la visualisation", [
        "📊 Dashboard KPIs automatique",
        "🌡 Heatmap de corrélation annotée",
        "📦 Boxplots multiples côte-à-côte",
        "🌊 Graphique en aires empilées",
        "🔵 Bubble chart (3 variables)",
        "🌳 Treemap (hiérarchie)",
        "🌞 Sunburst (hiérarchie imbriquée)",
        "🎻 Violin plot comparatif",
        "📉 Distribution cumulative (ECDF)",
        "🔥 Heatmap temporelle (calendrier)",
        "📈 Graphique en entonnoir (Funnel)",
        "🗺 Graphique radar / araignée",
        "📊 Waterfall / Cascade",
        "🔗 Réseau de corrélations (graphe)",
        "📐 Q-Q Plot multi-variables",
    ])

    # ── Dashboard KPIs ────────────────────────────────────────
    if viz_adv == "📊 Dashboard KPIs automatique":
        st.markdown('<p class="section-header">Dashboard KPIs automatique</p>', unsafe_allow_html=True)
        if not num_cols:
            st.info("Aucune colonne numérique détectée.")
        else:
            kpi_cols = st.multiselect("Colonnes KPI", num_cols, default=num_cols[:min(4,len(num_cols))], key='kpi_cols')
            group_by_kpi = st.selectbox("Grouper par (optionnel)", ["Aucun"]+cat_cols, key='kpi_grp')
            if kpi_cols:
                # Row de métriques
                cols_disp = st.columns(len(kpi_cols))
                for i, c in enumerate(kpi_cols):
                    with cols_disp[i]:
                        v = df[c].mean()
                        delta = df[c].std()
                        st.metric(label=c, value=f"{v:,.2f}", delta=f"σ={delta:.2f}")
                # Distributions
                fig_dash = make_subplots(rows=1, cols=len(kpi_cols),
                                         subplot_titles=kpi_cols)
                colors = ['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24']
                for i,c in enumerate(kpi_cols):
                    fig_dash.add_histogram(x=df[c].dropna(), nbinsx=20,
                                           marker_color=colors[i%4], opacity=0.8,
                                           row=1, col=i+1, name=c)
                fig_dash.update_layout(title="Distribution des KPIs", showlegend=False, **plotly_theme())
                st.plotly_chart(fig_dash, use_container_width=True)

                if group_by_kpi != "Aucun":
                    for kc in kpi_cols:
                        grp = df.groupby(group_by_kpi)[kc].mean().reset_index().sort_values(kc, ascending=False)
                        fig_g = px.bar(grp, x=group_by_kpi, y=kc, color=kc,
                                       color_continuous_scale=['#1e2230','#7c6ffd','#00e5a0'],
                                       text=grp[kc].round(2))
                        fig_g.update_layout(title=f"Moyenne de {kc} par {group_by_kpi}", **plotly_theme())
                        st.plotly_chart(fig_g, use_container_width=True)

    # ── Heatmap corrélation annotée ───────────────────────────
    elif viz_adv == "🌡 Heatmap de corrélation annotée":
        if len(num_cols) < 2:
            st.warning("Il faut au moins 2 colonnes numériques.")
        else:
            sel_hm = st.multiselect("Colonnes", num_cols, default=num_cols[:min(8,len(num_cols))], key='hm_cols')
            method_hm = st.selectbox("Méthode", ["pearson","spearman","kendall"], key='hm_meth')
            palette   = st.selectbox("Palette", ["RdGn","RdBu","PuOr","viridis"], key='hm_pal')
            pal_map   = {"RdGn":['#ef4444','#161920','#00e5a0'],
                         "RdBu":['#ef4444','#ffffff','#1d4ed8'],
                         "PuOr":['#7c3aed','#ffffff','#ea580c'],
                         "viridis":['#0d0f14','#7c6ffd','#00e5a0']}
            if len(sel_hm) >= 2:
                corr = df[sel_hm].corr(method=method_hm).round(3)
                fig_hm = px.imshow(corr, text_auto=True, aspect='auto', zmin=-1, zmax=1,
                                   color_continuous_scale=pal_map[palette])
                fig_hm.update_traces(textfont_size=11)
                fig_hm.update_layout(title=f"Corrélation {method_hm} — annotations complètes", **plotly_theme())
                st.plotly_chart(fig_hm, use_container_width=True)
                # Table des paires significatives
                alpha = st.slider("Seuil |r| significatif", 0.3, 0.95, 0.5, key='hm_alpha')
                pairs = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        r = corr.iloc[i,j]
                        if abs(r) >= alpha:
                            pairs.append({'Var A': corr.columns[i], 'Var B': corr.columns[j],
                                          'r': r, 'Force': interpret_r(r)})
                if pairs:
                    st.dataframe(pd.DataFrame(pairs).sort_values('r', key=abs, ascending=False), use_container_width=True, hide_index=True)

    # ── Boxplots multiples ────────────────────────────────────
    elif viz_adv == "📦 Boxplots multiples côte-à-côte":
        if not num_cols: st.warning("Aucune colonne numérique.")
        else:
            sel_box = st.multiselect("Variables", num_cols, default=num_cols[:min(6,len(num_cols))], key='adv_box')
            show_pts = st.checkbox("Afficher les points", False, key='adv_box_pts')
            notch    = st.checkbox("Entailles (notch) — IC médiane", False, key='adv_notch')
            if sel_box:
                fig_b = go.Figure()
                colors = ['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24','#38bdf8','#f472b6']
                for i,c in enumerate(sel_box):
                    fig_b.add_box(y=df[c].dropna(), name=c, marker_color=colors[i%6],
                                  notched=notch, points="all" if show_pts else "outliers",
                                  boxmean='sd')
                fig_b.update_layout(title="Boxplots comparatifs (ligne = médiane, croix = moyenne)", **plotly_theme())
                st.plotly_chart(fig_b, use_container_width=True)

    # ── Aires empilées ────────────────────────────────────────
    elif viz_adv == "🌊 Graphique en aires empilées":
        date_cands = dt_cols+[c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if not date_cands or not num_cols:
            st.warning("Il faut une colonne date et au moins une colonne numérique.")
        else:
            date_c = st.selectbox("Colonne date", date_cands, key='area_date')
            val_cs = st.multiselect("Valeurs (séries)", num_cols, default=num_cols[:min(3,len(num_cols))], key='area_vals')
            freq_a = st.selectbox("Fréquence", ["Jour (D)","Semaine (W)","Mois (ME)","Trimestre (QE)"], key='area_freq')
            freq_m = {"Jour (D)":"D","Semaine (W)":"W","Mois (ME)":"ME","Trimestre (QE)":"QE"}
            if val_cs:
                try:
                    tmp = df.copy()
                    tmp[date_c] = pd.to_datetime(tmp[date_c], errors='coerce')
                    tmp = tmp.dropna(subset=[date_c]).set_index(date_c)
                    ts = tmp[val_cs].resample(freq_m[freq_a]).sum().reset_index()
                    fig_area = px.area(ts, x=date_c, y=val_cs,
                                       color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24'])
                    fig_area.update_layout(title="Aires empilées temporelles", **plotly_theme())
                    st.plotly_chart(fig_area, use_container_width=True)
                except Exception as e: st.error(f"Erreur : {e}")

    # ── Bubble chart ──────────────────────────────────────────
    elif viz_adv == "🔵 Bubble chart (3 variables)":
        if len(num_cols) < 3: st.warning("Il faut au moins 3 colonnes numériques.")
        else:
            c1,c2,c3,c4 = st.columns(4)
            with c1: bx = st.selectbox("Axe X", num_cols, key='bub_x')
            with c2: by = st.selectbox("Axe Y", [c for c in num_cols if c!=bx], key='bub_y')
            with c3: bs = st.selectbox("Taille bulle", [c for c in num_cols if c not in [bx,by]], key='bub_s')
            with c4: bc = st.selectbox("Couleur", ["Aucune"]+cat_cols+num_cols, key='bub_c')
            fig_bub = px.scatter(df, x=bx, y=by, size=bs, size_max=50,
                                 color=None if bc=="Aucune" else bc, opacity=0.7,
                                 color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24'],
                                 color_continuous_scale=['#1e2230','#7c6ffd','#00e5a0'],
                                 hover_data=df.columns.tolist()[:5])
            fig_bub.update_layout(title=f"Bubble chart : {bx} × {by} (taille={bs})", **plotly_theme())
            st.plotly_chart(fig_bub, use_container_width=True)

    # ── Treemap ───────────────────────────────────────────────
    elif viz_adv == "🌳 Treemap (hiérarchie)":
        if not cat_cols or not num_cols: st.warning("Colonnes catégorielles et numériques requises.")
        else:
            tree_path = st.multiselect("Chemin hiérarchique (du parent à l'enfant)", cat_cols, key='tree_path')
            tree_val  = st.selectbox("Valeur (taille)", num_cols, key='tree_val')
            tree_col  = st.selectbox("Couleur", ["Aucune"]+num_cols+cat_cols, key='tree_col')
            if tree_path:
                try:
                    fig_tree = px.treemap(df, path=tree_path, values=tree_val,
                                          color=None if tree_col=="Aucune" else tree_col,
                                          color_continuous_scale=['#1e2230','#7c6ffd','#00e5a0'])
                    fig_tree.update_layout(title="Treemap hiérarchique", **plotly_theme())
                    st.plotly_chart(fig_tree, use_container_width=True)
                except Exception as e: st.error(f"Erreur : {e}")

    # ── Sunburst ──────────────────────────────────────────────
    elif viz_adv == "🌞 Sunburst (hiérarchie imbriquée)":
        if not cat_cols or not num_cols: st.warning("Colonnes catégorielles et numériques requises.")
        else:
            sun_path = st.multiselect("Chemin hiérarchique", cat_cols, key='sun_path')
            sun_val  = st.selectbox("Valeur", num_cols, key='sun_val')
            if sun_path:
                try:
                    fig_sun = px.sunburst(df, path=sun_path, values=sun_val,
                                          color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24','#38bdf8'])
                    fig_sun.update_layout(title="Sunburst chart", **plotly_theme())
                    st.plotly_chart(fig_sun, use_container_width=True)
                except Exception as e: st.error(f"Erreur : {e}")

    # ── Violin comparatif ─────────────────────────────────────
    elif viz_adv == "🎻 Violin plot comparatif":
        if not num_cols: st.warning("Aucune colonne numérique.")
        else:
            viol_cols = st.multiselect("Variables", num_cols, default=num_cols[:min(4,len(num_cols))], key='viol_multi')
            viol_grp  = st.selectbox("Grouper par", ["Aucun"]+cat_cols, key='viol_grp')
            if viol_cols:
                if viol_grp == "Aucun":
                    df_melt = df[viol_cols].melt(var_name='Variable', value_name='Valeur').dropna()
                    fig_viol = px.violin(df_melt, x='Variable', y='Valeur', box=True, points="outliers",
                                         color='Variable', color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24'])
                else:
                    fig_viol = px.violin(df, x=viol_grp, y=viol_cols[0], color=viol_grp, box=True, points="outliers",
                                         color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24'])
                fig_viol.update_layout(title="Violin plot comparatif", **plotly_theme())
                st.plotly_chart(fig_viol, use_container_width=True)

    # ── ECDF ──────────────────────────────────────────────────
    elif viz_adv == "📉 Distribution cumulative (ECDF)":
        if not num_cols: st.warning("Aucune colonne numérique.")
        else:
            ecdf_cols = st.multiselect("Variables", num_cols, default=num_cols[:min(4,len(num_cols))], key='ecdf_cols')
            if ecdf_cols:
                fig_ecdf = go.Figure()
                colors_ec = ['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24','#38bdf8']
                for i,c in enumerate(ecdf_cols):
                    s = df[c].dropna().sort_values()
                    ecdf = np.arange(1, len(s)+1)/len(s)
                    fig_ecdf.add_scatter(x=s, y=ecdf, mode='lines', name=c,
                                         line=dict(color=colors_ec[i%5], width=2))
                fig_ecdf.add_hline(y=0.5, line_dash='dash', line_color='#6b7280', annotation_text='Médiane')
                fig_ecdf.add_hline(y=0.25, line_dash='dot', line_color='#6b7280', annotation_text='Q1')
                fig_ecdf.add_hline(y=0.75, line_dash='dot', line_color='#6b7280', annotation_text='Q3')
                fig_ecdf.update_layout(title="Distribution cumulative empirique (ECDF)",
                                       xaxis_title="Valeur", yaxis_title="Probabilité cumulée",
                                       **plotly_theme())
                st.plotly_chart(fig_ecdf, use_container_width=True)

    # ── Heatmap temporelle ────────────────────────────────────
    elif viz_adv == "🔥 Heatmap temporelle (calendrier)":
        date_cands = dt_cols+[c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if not date_cands or not num_cols:
            st.warning("Il faut une colonne date et une colonne numérique.")
        else:
            ht_date = st.selectbox("Colonne date", date_cands, key='ht_date')
            ht_val  = st.selectbox("Valeur", num_cols, key='ht_val')
            ht_x    = st.selectbox("Axe X (granularité)", ["Mois","Semaine","Trimestre","Heure"], key='ht_x')
            ht_y    = st.selectbox("Axe Y", ["Année","Jour de la semaine","Mois"], key='ht_y')
            try:
                tmp = df.copy()
                tmp[ht_date] = pd.to_datetime(tmp[ht_date], errors='coerce')
                tmp = tmp.dropna(subset=[ht_date])
                x_map = {"Mois":"month","Semaine":"isocalendar().week","Trimestre":"quarter","Heure":"hour"}
                y_map = {"Année":"year","Jour de la semaine":"dayofweek","Mois":"month"}

                def safe_get(col, attr):
                    if '().' in attr:
                        parts = attr.split('.')
                        return getattr(getattr(col.dt, parts[0][:-2])(), parts[1]).astype(int)
                    return getattr(col.dt, attr)

                tmp['_x'] = safe_get(tmp[ht_date], x_map[ht_x])
                tmp['_y'] = safe_get(tmp[ht_date], y_map[ht_y])
                pivot_ht  = tmp.groupby(['_y','_x'])[ht_val].mean().unstack(fill_value=0)
                fig_ht = px.imshow(pivot_ht, text_auto=".0f", aspect='auto',
                                   color_continuous_scale=['#161920','#7c6ffd','#00e5a0'])
                fig_ht.update_layout(title=f"Heatmap temporelle — {ht_val} ({ht_y} × {ht_x})", **plotly_theme())
                st.plotly_chart(fig_ht, use_container_width=True)
            except Exception as e: st.error(f"Erreur : {e}")

    # ── Funnel ────────────────────────────────────────────────
    elif viz_adv == "📈 Graphique en entonnoir (Funnel)":
        if not cat_cols or not num_cols: st.warning("Colonnes catégorielles et numériques requises.")
        else:
            fun_cat = st.selectbox("Étapes (catégorielle)", cat_cols, key='fun_cat')
            fun_val = st.selectbox("Valeur", num_cols, key='fun_val')
            fun_agg = st.selectbox("Agrégation", ["sum","mean","count"], key='fun_agg')
            fun_df  = df.groupby(fun_cat)[fun_val].agg(fun_agg).reset_index().sort_values(fun_val, ascending=False)
            fig_fun = px.funnel(fun_df, x=fun_val, y=fun_cat,
                                color_discrete_sequence=['#7c6ffd'])
            fig_fun.update_layout(title=f"Entonnoir : {fun_val} par {fun_cat}", **plotly_theme())
            st.plotly_chart(fig_fun, use_container_width=True)

    # ── Radar ─────────────────────────────────────────────────
    elif viz_adv == "🗺 Graphique radar / araignée":
        if not num_cols: st.warning("Aucune colonne numérique.")
        else:
            rad_cols = st.multiselect("Variables (axes du radar)", num_cols, default=num_cols[:min(6,len(num_cols))], key='rad_cols')
            rad_grp  = st.selectbox("Grouper par (une ligne par groupe)", ["Aucun"]+cat_cols, key='rad_grp')
            if rad_cols:
                if rad_grp == "Aucun":
                    means = df[rad_cols].mean().values
                    fig_rad = go.Figure(go.Scatterpolar(r=means, theta=rad_cols,
                                                        fill='toself', fillcolor='rgba(0,229,160,0.2)',
                                                        line=dict(color='#00e5a0', width=2), name='Moyenne'))
                else:
                    top_grps = df[rad_grp].value_counts().head(6).index.tolist()
                    fig_rad  = go.Figure()
                    colors_r = ['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24','#38bdf8','#f472b6']
                    for i,g in enumerate(top_grps):
                        sub = df[df[rad_grp]==g][rad_cols].mean().values
                        fig_rad.add_scatterpolar(r=sub, theta=rad_cols, fill='toself', name=str(g),
                                                 line=dict(color=colors_r[i%6], width=2))
                fig_rad.update_layout(polar=dict(bgcolor='#161920',
                    radialaxis=dict(visible=True, gridcolor='#2a2f3d', color='#6b7280'),
                    angularaxis=dict(gridcolor='#2a2f3d', color='#6b7280')),
                    title="Graphique radar", paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e8eaf0'), margin=dict(t=60,b=40,l=40,r=40))
                st.plotly_chart(fig_rad, use_container_width=True)

    # ── Waterfall ─────────────────────────────────────────────
    elif viz_adv == "📊 Waterfall / Cascade":
        if not cat_cols or not num_cols: st.warning("Colonnes catégorielles et numériques requises.")
        else:
            wf_cat = st.selectbox("Catégories (étapes)", cat_cols, key='wf_cat')
            wf_val = st.selectbox("Valeur", num_cols, key='wf_val')
            wf_agg = st.selectbox("Agrégation", ["sum","mean"], key='wf_agg')
            wf_df  = df.groupby(wf_cat)[wf_val].agg(wf_agg).reset_index()
            vals   = wf_df[wf_val].values
            colors_wf = ['#00e5a0' if v >= 0 else '#ef4444' for v in vals]
            fig_wf = go.Figure(go.Waterfall(
                name="", measure=["relative"]*len(vals),
                x=wf_df[wf_cat].tolist(), y=vals,
                connector=dict(line=dict(color='#2a2f3d')),
                increasing=dict(marker_color='#00e5a0'),
                decreasing=dict(marker_color='#ef4444'),
            ))
            fig_wf.update_layout(title=f"Waterfall : {wf_val} par {wf_cat}", **plotly_theme())
            st.plotly_chart(fig_wf, use_container_width=True)

    # ── Réseau de corrélations ────────────────────────────────
    elif viz_adv == "🔗 Réseau de corrélations (graphe)":
        if len(num_cols) < 3: st.warning("Il faut au moins 3 colonnes numériques.")
        else:
            net_cols = st.multiselect("Colonnes", num_cols, default=num_cols[:min(8,len(num_cols))], key='net_cols')
            net_thr  = st.slider("Seuil |r| minimum", 0.1, 0.95, 0.5, key='net_thr')
            net_meth = st.selectbox("Méthode", ["pearson","spearman"], key='net_meth')
            if len(net_cols) >= 3:
                corr_net = df[net_cols].corr(method=net_meth)
                # Construire les arêtes
                edges_x, edges_y, annotations = [], [], []
                np.random.seed(42)
                n = len(net_cols)
                angles = np.linspace(0, 2*np.pi, n, endpoint=False)
                pos = {c: (np.cos(a), np.sin(a)) for c,a in zip(net_cols, angles)}

                fig_net = go.Figure()
                for i,c1 in enumerate(net_cols):
                    for j,c2 in enumerate(net_cols):
                        if i < j:
                            r = corr_net.loc[c1,c2]
                            if abs(r) >= net_thr:
                                x0,y0 = pos[c1]
                                x1,y1 = pos[c2]
                                color = '#00e5a0' if r > 0 else '#ef4444'
                                width = abs(r)*5
                                fig_net.add_scatter(x=[x0,x1,None], y=[y0,y1,None],
                                                    mode='lines', line=dict(color=color, width=width),
                                                    hoverinfo='skip', showlegend=False)

                for c,(x,y) in pos.items():
                    fig_net.add_scatter(x=[x], y=[y], mode='markers+text',
                                        marker=dict(size=20, color='#7c6ffd', line=dict(color='#00e5a0', width=2)),
                                        text=[c], textposition='top center',
                                        textfont=dict(color='#e8eaf0', size=10),
                                        name=c, showlegend=False)
                fig_net.update_layout(title=f"Réseau de corrélations (|r| ≥ {net_thr}) — vert=positif, rouge=négatif",
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      **plotly_theme())
                st.plotly_chart(fig_net, use_container_width=True)

    # ── Q-Q Plot multi ────────────────────────────────────────
    elif viz_adv == "📐 Q-Q Plot multi-variables":
        if not num_cols: st.warning("Aucune colonne numérique.")
        else:
            qq_cols = st.multiselect("Variables", num_cols, default=num_cols[:min(4,len(num_cols))], key='qq_cols')
            dist    = st.selectbox("Distribution de référence", ["norm","uniform","expon","lognorm"], key='qq_dist')
            if qq_cols:
                n_cols_qq = min(len(qq_cols), 2)
                n_rows_qq = (len(qq_cols)+1)//2
                fig_qq_m, axes = plt.subplots(n_rows_qq, n_cols_qq, figsize=(6*n_cols_qq, 4*n_rows_qq))
                fig_qq_m.patch.set_facecolor('#0d0f14')
                axes = np.array(axes).flatten() if len(qq_cols) > 1 else [axes]
                for i,c in enumerate(qq_cols):
                    ax = axes[i]
                    ax.set_facecolor('#161920')
                    s = df[c].dropna()
                    stats.probplot(s, dist=dist, plot=ax)
                    ax.get_lines()[0].set(color='#7c6ffd', markersize=3, alpha=0.6)
                    ax.get_lines()[1].set(color='#00e5a0', linewidth=2)
                    ax.set_title(c, color='#e8eaf0', fontsize=10)
                    ax.tick_params(colors='#6b7280')
                    for spine in ax.spines.values(): spine.set_color('#2a2f3d')
                    ax.xaxis.label.set_color('#6b7280')
                    ax.yaxis.label.set_color('#6b7280')
                for j in range(len(qq_cols), len(axes)):
                    axes[j].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_qq_m, use_container_width=True)

    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
elif page == "📉 Analyse bivariée":
    st.markdown('<p class="section-header">Analyse bivariée & relations entre variables</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Explorez les relations entre deux variables. Les tests statistiques sont interprétés automatiquement.</div>', unsafe_allow_html=True)

    if len(df.columns) < 2:
        st.warning("Il faut au moins 2 colonnes.")
    else:
        num_cols2, cat_cols2, _, _ = detect_column_types(df)
        tab1, tab2, tab3 = st.tabs(["🔢 Num × Num", "🏷 Cat × Num", "🗂 Cat × Cat"])

        with tab1:
            if len(num_cols2) < 2:
                st.info("Il faut au moins 2 colonnes numériques.")
            else:
                x2 = st.selectbox("Variable X", num_cols2, key='bv_x')
                y2 = st.selectbox("Variable Y", [c for c in num_cols2 if c!=x2], key='bv_y')
                show_trend = st.checkbox("Afficher la droite de régression", True, key='bv_trend')
                clean = df[[x2,y2]].dropna()
                fig = px.scatter(clean, x=x2, y=y2, opacity=0.6, color_discrete_sequence=['#7c6ffd'])
                if show_trend and len(clean) > 1:
                    m,b = np.polyfit(clean[x2], clean[y2], 1)
                    xr = np.linspace(clean[x2].min(), clean[x2].max(), 200)
                    fig.add_scatter(x=xr, y=m*xr+b, mode='lines',
                                    line=dict(color='#00e5a0', width=2, dash='dash'), name='Tendance OLS')
                fig.update_layout(title=f"Scatter : {x2} vs {y2}", **plotly_theme())
                st.plotly_chart(fig, use_container_width=True)
                if len(clean) > 1:
                    r, p_r   = stats.pearsonr(clean[x2], clean[y2])
                    rho, p_s = stats.spearmanr(clean[x2], clean[y2])
                    tau, p_t = stats.kendalltau(clean[x2], clean[y2])
                    c1,c2,c3 = st.columns(3)
                    with c1: st.markdown(f'<div class="metric-card"><span class="metric-value">{r:.4f}</span><div class="metric-label">Pearson r</div></div>', unsafe_allow_html=True)
                    with c2: st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#7c6ffd">{rho:.4f}</span><div class="metric-label">Spearman ρ</div></div>', unsafe_allow_html=True)
                    with c3: st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#ff6b6b">{tau:.4f}</span><div class="metric-label">Kendall τ</div></div>', unsafe_allow_html=True)
                    st.markdown(f"""<div class="info-box">
                        <b>Pearson r={r:.4f}</b> — {interpret_r(r)} — {interpret_p(p_r)}<br>
                        <b>Spearman ρ={rho:.4f}</b> — {interpret_r(rho)} — {interpret_p(p_s)}<br>
                        <b>Kendall τ={tau:.4f}</b> — {interpret_p(p_t)}<br><br>
                        <b>Régression linéaire :</b> y = {m:.4f}·x + {b:.4f}
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
                viz_choice = st.radio("Visualisation", ["Boxplot","Violin","Barres (moyenne ± écart-type)"], horizontal=True, key='bv_viz')
                if viz_choice == "Boxplot":
                    fig = px.box(df_filt, x=cat_v, y=num_v, color=cat_v, color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24','#38bdf8'])
                elif viz_choice == "Violin":
                    fig = px.violin(df_filt, x=cat_v, y=num_v, color=cat_v, box=True, color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24','#38bdf8'])
                else:
                    grp = df_filt.groupby(cat_v)[num_v].agg(['mean','std']).reset_index()
                    fig = go.Figure()
                    fig.add_bar(x=grp[cat_v], y=grp['mean'], error_y=dict(type='data', array=grp['std'].fillna(0)), marker_color='#7c6ffd', name='Moyenne ± σ')
                fig.update_layout(title=f"{viz_choice} : {cat_v} × {num_v}", xaxis_tickangle=-30, **plotly_theme())
                st.plotly_chart(fig, use_container_width=True)
                grp_stats = df_filt.groupby(cat_v)[num_v].agg(['mean','median','std','count']).round(3).reset_index()
                grp_stats.columns = [cat_v,'Moyenne','Médiane','Écart-type','N']
                st.dataframe(grp_stats, use_container_width=True, hide_index=True)
                groups = [g[num_v].dropna().values for _, g in df_filt.groupby(cat_v) if g[num_v].dropna().shape[0] > 1]
                if len(groups) >= 2:
                    st.markdown('<p class="section-header">Tests statistiques</p>', unsafe_allow_html=True)
                    try:
                        f_stat, p_anova = f_oneway(*groups)
                        h_stat, p_kruskal = kruskal(*groups)
                        st.markdown(f"""<div class="info-box">
                            <b>ANOVA (F-test)</b> : F={f_stat:.4f} — {interpret_p(p_anova)}<br>
                            <i>Compare les moyennes entre groupes (suppose normalité et homogénéité des variances)</i><br><br>
                            <b>Kruskal-Wallis</b> : H={h_stat:.4f} — {interpret_p(p_kruskal)}<br>
                            <i>Alternative non paramétrique à l'ANOVA (ne suppose pas la normalité)</i>
                        </div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Tests non calculables : {e}")

        with tab3:
            if len(cat_cols2) < 2:
                st.info("Il faut au moins 2 colonnes catégorielles.")
            else:
                c1v = st.selectbox("Colonne 1", cat_cols2, key='bv_c1')
                c2v = st.selectbox("Colonne 2", [c for c in cat_cols2 if c!=c1v], key='bv_c2')
                ct  = pd.crosstab(df[c1v], df[c2v])
                viz_ct = st.radio("Afficher", ["Heatmap","Table normalisée"], horizontal=True, key='bv_ct_viz')
                if viz_ct == "Heatmap":
                    fig = px.imshow(ct, text_auto=True, aspect='auto', color_continuous_scale=['#161920','#7c6ffd','#00e5a0'])
                else:
                    ct_norm = pd.crosstab(df[c1v], df[c2v], normalize='index').round(3)
                    fig = px.imshow(ct_norm, text_auto=True, aspect='auto', color_continuous_scale=['#161920','#7c6ffd','#00e5a0'], zmin=0, zmax=1)
                fig.update_layout(title=f"Contingence : {c1v} × {c2v}", **plotly_theme())
                st.plotly_chart(fig, use_container_width=True)
                try:
                    chi2_val, p_chi, dof, expected = chi2_contingency(ct)
                    n = ct.sum().sum()
                    v_cramer = np.sqrt(chi2_val / (n * (min(ct.shape)-1)))
                    st.markdown(f"""<div class="info-box">
                        <b>Test du χ²</b> : χ²={chi2_val:.4f}, ddl={dof} — {interpret_p(p_chi)}<br>
                        <b>V de Cramér</b> : {v_cramer:.4f} — {"Forte" if v_cramer>0.3 else "Modérée" if v_cramer>0.1 else "Faible"} association
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Test χ² non calculable : {e}")

    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 7 — JOINTURES & FUSION  ← NOUVEAU
# ══════════════════════════════════════════════════════════════
elif page == "🔗 Jointures & Fusion":
    st.markdown('<p class="section-header">Jointures, fusions & opérations ensemblistes</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Importez un second fichier pour effectuer des jointures (merge), des concaténations ou des opérations ensemblistes avec le fichier principal.</div>', unsafe_allow_html=True)

    # Schéma visuel des jointures
    with st.expander("📖 Types de jointures — rappel visuel"):
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.markdown("""<div class="join-diagram">
                <b>INNER JOIN</b><br>──────────<br>
                A ∩ B<br>
                Lignes communes<br>seulement
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="join-diagram">
                <b>LEFT JOIN</b><br>──────────<br>
                Tout A<br>+ correspondances B<br>(NaN si absent)
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="join-diagram">
                <b>RIGHT JOIN</b><br>──────────<br>
                Tout B<br>+ correspondances A<br>(NaN si absent)
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown("""<div class="join-diagram">
                <b>OUTER JOIN</b><br>──────────<br>
                A ∪ B<br>Toutes les lignes<br>(NaN si absent)
            </div>""", unsafe_allow_html=True)

    tab_join, tab_concat, tab_ens = st.tabs(["🔀 Merge (Jointure)", "📎 Concaténation", "∪ Opérations ensemblistes"])

    with tab_join:
        st.markdown("**Importer le second fichier (table de droite)**")
        file2 = st.file_uploader("Second fichier", type=["csv","xls","xlsx"], key="join_file")
        if file2:
            df2 = load_data(file2)
            if df2 is not None:
                st.success(f"✓ Fichier chargé : {df2.shape[0]} lignes × {df2.shape[1]} colonnes")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Table gauche (principale)**")
                    st.dataframe(df.head(5), use_container_width=True)
                with col2:
                    st.markdown("**Table droite (importée)**")
                    st.dataframe(df2.head(5), use_container_width=True)

                # Colonnes communes détectées
                common_cols = list(set(df.columns) & set(df2.columns))
                st.markdown("**Configuration de la jointure**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    left_on = st.selectbox("Clé gauche", df.columns.tolist(), key='left_on',
                                           index=df.columns.tolist().index(common_cols[0]) if common_cols else 0)
                with c2:
                    right_on = st.selectbox("Clé droite", df2.columns.tolist(), key='right_on',
                                            index=df2.columns.tolist().index(common_cols[0]) if common_cols else 0)
                with c3:
                    how = st.selectbox("Type de jointure", ["inner","left","right","outer"], key='join_how',
                                       format_func=lambda x: {"inner":"INNER","left":"LEFT","right":"RIGHT","outer":"OUTER (FULL)"}[x])

                suffixes_l = st.text_input("Suffixe gauche (si conflit de noms)", value="_gauche", key='suf_l')
                suffixes_r = st.text_input("Suffixe droite (si conflit de noms)", value="_droite", key='suf_r')

                if st.button("🔀 Effectuer la jointure"):
                    try:
                        df_merged = pd.merge(df, df2, left_on=left_on, right_on=right_on,
                                             how=how, suffixes=(suffixes_l, suffixes_r))
                        st.session_state.df = df_merged
                        df = df_merged
                        msg = f"Jointure {how.upper()} sur {left_on}={right_on} → {df_merged.shape[0]} lignes × {df_merged.shape[1]} colonnes"
                        st.session_state.cleaning_log.append(msg)
                        st.success(msg)
                        # Diagnostic
                        st.markdown('<p class="section-header">Résultat de la jointure</p>', unsafe_allow_html=True)
                        c1,c2,c3 = st.columns(3)
                        with c1: st.markdown(f'<div class="metric-card"><span class="metric-value">{df_merged.shape[0]:,}</span><div class="metric-label">Lignes résultantes</div></div>', unsafe_allow_html=True)
                        with c2: st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#7c6ffd">{df_merged.shape[1]}</span><div class="metric-label">Colonnes</div></div>', unsafe_allow_html=True)
                        with c3:
                            n_miss_new = df_merged.isnull().sum().sum()
                            st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#fbbf24">{n_miss_new}</span><div class="metric-label">NaN créés</div></div>', unsafe_allow_html=True)
                        st.dataframe(df_merged.head(20), use_container_width=True)
                    except Exception as e:
                        st.error(f"Erreur de jointure : {e}")
        else:
            st.info("⬆ Importez un second fichier pour activer la jointure.")

    with tab_concat:
        st.markdown("**Importer le fichier à concaténer**")
        file_concat = st.file_uploader("Fichier à concaténer", type=["csv","xls","xlsx"], key="concat_file")
        if file_concat:
            df_c = load_data(file_concat)
            if df_c is not None:
                axis = st.radio("Axe de concaténation", ["Verticalement (ajouter des lignes)","Horizontalement (ajouter des colonnes)"], key='concat_axis')
                ignore_idx = st.checkbox("Réinitialiser l'index", True, key='concat_idx')
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Fichier principal** : {df.shape}")
                    st.dataframe(df.head(3), use_container_width=True)
                with col2:
                    st.markdown(f"**Fichier à concaténer** : {df_c.shape}")
                    st.dataframe(df_c.head(3), use_container_width=True)

                if st.button("📎 Concaténer"):
                    try:
                        ax = 0 if "Verticalement" in axis else 1
                        df_result = pd.concat([df, df_c], axis=ax, ignore_index=ignore_idx)
                        st.session_state.df = df_result; df = df_result
                        msg = f"Concaténation {'verticale' if ax==0 else 'horizontale'} → {df_result.shape[0]} lignes × {df_result.shape[1]} colonnes"
                        st.session_state.cleaning_log.append(msg)
                        st.success(msg)
                        st.dataframe(df_result.head(10), use_container_width=True)
                    except Exception as e:
                        st.error(f"Erreur : {e}")
        else:
            st.info("⬆ Importez un fichier pour activer la concaténation.")

    with tab_ens:
        st.markdown("**Opérations ensemblistes sur une colonne clé**")
        st.markdown('<div class="info-box">Ces opérations identifient les lignes présentes dans l\'un ou l\'autre des fichiers selon une clé commune.</div>', unsafe_allow_html=True)
        file_ens = st.file_uploader("Second fichier", type=["csv","xls","xlsx"], key="ens_file")
        if file_ens:
            df_e = load_data(file_ens)
            if df_e is not None:
                common = list(set(df.columns) & set(df_e.columns))
                if common:
                    key_col = st.selectbox("Colonne clé", common, key='ens_key')
                    op = st.radio("Opération", ["Intersection (∩)","Union (∪)","Différence A−B","Différence B−A","Différence symétrique (Δ)"], key='ens_op')
                    if st.button("Appliquer"):
                        s1 = set(df[key_col].dropna())
                        s2 = set(df_e[key_col].dropna())
                        if op=="Intersection (∩)": res = s1 & s2
                        elif op=="Union (∪)": res = s1 | s2
                        elif op=="Différence A−B": res = s1 - s2
                        elif op=="Différence B−A": res = s2 - s1
                        else: res = s1.symmetric_difference(s2)
                        st.markdown(f"**{len(res)} valeurs** dans le résultat de l'opération **{op}**")
                        df_res = df[df[key_col].isin(res)] if op not in ["Union (∪)","Différence B−A","Différence symétrique (Δ)"] else pd.concat([df[df[key_col].isin(res)], df_e[df_e[key_col].isin(res)]])
                        st.dataframe(df_res.head(20), use_container_width=True)
                        if st.button("Sauvegarder ce résultat comme table principale"):
                            st.session_state.df = df_res.reset_index(drop=True)
                            st.success("Table principale mise à jour.")
                else:
                    st.warning("Aucune colonne en commun entre les deux fichiers.")
        else:
            st.info("⬆ Importez un fichier pour activer les opérations ensemblistes.")

    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 8 — AGRÉGATION & PIVOT  ← NOUVEAU
# ══════════════════════════════════════════════════════════════
elif page == "📋 Agrégation & Pivot":
    st.markdown('<p class="section-header">Agrégation, GroupBy & Tableaux croisés dynamiques</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 GroupBy & Agrégation", "🔄 Tableau croisé (Pivot)", "🔢 Reshape (Melt / Stack)"])

    with tab1:
        st.markdown("**Agrégation par groupe(s)**")
        if not cat_cols: st.info("Aucune colonne catégorielle disponible pour le groupement.")
        else:
            group_cols = st.multiselect("Grouper par", cat_cols, key='grp_by')
            agg_cols   = st.multiselect("Colonnes à agréger", num_cols, default=num_cols[:min(3,len(num_cols))], key='agg_cols')
            agg_funcs  = st.multiselect("Fonctions", ["mean","median","sum","count","min","max","std","var","nunique"],
                                         default=["mean","count"], key='agg_funcs')
            if group_cols and agg_cols and agg_funcs:
                if st.button("Calculer l'agrégation"):
                    try:
                        result = df.groupby(group_cols)[agg_cols].agg(agg_funcs).round(4)
                        result.columns = ['_'.join(c) for c in result.columns]
                        result = result.reset_index()
                        st.dataframe(result, use_container_width=True)
                        # Visualisation
                        if len(group_cols)==1 and len(agg_cols)>=1:
                            fig = px.bar(result, x=group_cols[0],
                                         y=[c for c in result.columns if c not in group_cols],
                                         barmode='group', color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24'])
                            fig.update_layout(title="Résultat de l'agrégation", **plotly_theme())
                            st.plotly_chart(fig, use_container_width=True)
                        # Export
                        data, mime, ext = download_df(result, "CSV")
                        st.download_button("⬇️ Exporter l'agrégation", data=data, file_name=f"agregation{ext}", mime=mime)
                    except Exception as e:
                        st.error(f"Erreur : {e}")

    with tab2:
        st.markdown("**Tableau croisé dynamique (Pivot Table)**")
        if not cat_cols or not num_cols: st.info("Il faut des colonnes catégorielles et numériques.")
        else:
            c1,c2,c3,c4 = st.columns(4)
            with c1: pivot_idx  = st.selectbox("Index (lignes)", cat_cols, key='piv_idx')
            with c2: pivot_col  = st.selectbox("Colonnes", [c for c in cat_cols if c!=pivot_idx], key='piv_col') if len(cat_cols)>1 else st.info("Besoin d'une 2e catégorielle")
            with c3: pivot_val  = st.selectbox("Valeurs", num_cols, key='piv_val')
            with c4: pivot_agg  = st.selectbox("Agrégation", ["mean","sum","count","median","min","max"], key='piv_agg')

            fill_val = st.number_input("Valeur pour NaN (0 = laisser vide)", value=0, key='piv_fill')

            if st.button("Générer le pivot") and len(cat_cols)>1:
                try:
                    pt = pd.pivot_table(df, values=pivot_val, index=pivot_idx,
                                        columns=pivot_col, aggfunc=pivot_agg, fill_value=fill_val)
                    st.dataframe(pt.round(3), use_container_width=True)
                    fig = px.imshow(pt, text_auto=".1f", aspect='auto',
                                    color_continuous_scale=['#161920','#7c6ffd','#00e5a0'])
                    fig.update_layout(title=f"Pivot : {pivot_val} par {pivot_idx} × {pivot_col}", **plotly_theme())
                    st.plotly_chart(fig, use_container_width=True)
                    data, mime, ext = download_df(pt.reset_index(), "Excel")
                    st.download_button("⬇️ Exporter le pivot", data=data, file_name=f"pivot{ext}", mime=mime)
                except Exception as e:
                    st.error(f"Erreur : {e}")

    with tab3:
        st.markdown("**Reshape : Melt (large → long) & Stack/Unstack**")
        st.markdown('<div class="info-box"><b>Melt</b> : transforme des colonnes en lignes (format large → format long, idéal pour les visualisations).<br><b>Stack</b> : empile les colonnes en lignes multi-niveaux.</div>', unsafe_allow_html=True)

        op_reshape = st.radio("Opération", ["Melt (large → long)","Stack","Unstack"], horizontal=True, key='reshape_op')
        if op_reshape == "Melt (large → long)":
            id_vars  = st.multiselect("Colonnes identifiantes (id_vars)", df.columns.tolist(), key='melt_id')
            val_vars = st.multiselect("Colonnes à fondre (value_vars)", [c for c in df.columns if c not in id_vars], key='melt_val')
            var_name = st.text_input("Nom colonne variable", value="variable", key='melt_vname')
            val_name = st.text_input("Nom colonne valeur", value="valeur", key='melt_valname')
            if st.button("Appliquer Melt") and id_vars and val_vars:
                try:
                    df_melted = pd.melt(df, id_vars=id_vars, value_vars=val_vars, var_name=var_name, value_name=val_name)
                    st.success(f"Melt appliqué → {df_melted.shape[0]} lignes × {df_melted.shape[1]} colonnes")
                    st.dataframe(df_melted.head(20), use_container_width=True)
                    if st.button("💾 Utiliser ce résultat comme table principale", key='melt_save'):
                        st.session_state.df = df_melted.reset_index(drop=True)
                        st.session_state.cleaning_log.append("Melt appliqué")
                except Exception as e:
                    st.error(f"Erreur : {e}")
        else:
            st.info(f"Opération {op_reshape} applicable sur des DataFrames multi-index. Utilisez d'abord un GroupBy pour créer un index hiérarchique.")

    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 9 — ANALYSE AVANCÉE  ← NOUVEAU
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Analyse avancée":
    st.markdown('<p class="section-header">Analyse avancée & profiling automatique</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🩺 Profiling automatique","📐 Tests de normalité","🔬 Régression multiple","📊 Analyse en composantes"])

    with tab1:
        st.markdown("**Rapport de profiling automatique de la base**")
        if st.button("🔍 Générer le profiling complet"):
            with st.spinner("Analyse en cours..."):
                # Résumé global
                st.markdown('<p class="section-header">1. Résumé global</p>', unsafe_allow_html=True)
                total_cells = df.shape[0]*df.shape[1]
                miss_total  = df.isnull().sum().sum()
                dup_total   = df.duplicated().sum()
                c1,c2,c3,c4,c5 = st.columns(5)
                for c,(l,v,col) in zip([c1,c2,c3,c4,c5],[
                    ("Lignes",f"{df.shape[0]:,}","#00e5a0"),
                    ("Colonnes",f"{df.shape[1]}","#7c6ffd"),
                    ("Complétude",f"{round((1-miss_total/total_cells)*100,1)}%","#38bdf8"),
                    ("Doublons",f"{dup_total:,}","#ff6b6b"),
                    ("Mémoire",f"{df.memory_usage(deep=True).sum()/1024:.1f} Ko","#f472b6"),
                ]):
                    with c: st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:{col}">{v}</span><div class="metric-label">{l}</div></div>', unsafe_allow_html=True)

                # Par colonne
                st.markdown('<p class="section-header">2. Profil par colonne</p>', unsafe_allow_html=True)
                profile_rows = []
                for col in df.columns:
                    s = df[col]
                    row = {
                        'Colonne': col,
                        'Type': str(s.dtype),
                        'Non-null': s.count(),
                        '% manquant': round(s.isnull().mean()*100,2),
                        'Uniques': s.nunique(),
                        '% unicité': round(s.nunique()/len(s)*100,2),
                    }
                    if s.dtype in [np.float64, np.int64, np.float32, np.int32]:
                        row.update({'Moyenne':round(s.mean(),4),'Médiane':round(s.median(),4),
                                    'Écart-type':round(s.std(),4),'Min':round(s.min(),4),'Max':round(s.max(),4),
                                    'Skewness':round(s.skew(),4),'Kurtosis':round(s.kurt(),4)})
                    else:
                        row.update({'Top valeur': str(s.mode()[0]) if s.count()>0 else 'N/A',
                                    'Freq. top (%)': round(s.value_counts().iloc[0]/s.count()*100,2) if s.count()>0 else 0})
                    profile_rows.append(row)
                st.dataframe(pd.DataFrame(profile_rows), use_container_width=True)

                # Alertes automatiques
                st.markdown('<p class="section-header">3. Alertes & recommandations</p>', unsafe_allow_html=True)
                alertes = []
                for col in df.columns:
                    s = df[col]
                    miss_pct = s.isnull().mean()*100
                    if miss_pct > 50: alertes.append(f"🔴 **{col}** : {miss_pct:.1f}% de valeurs manquantes → envisager la suppression")
                    elif miss_pct > 20: alertes.append(f"🟠 **{col}** : {miss_pct:.1f}% de valeurs manquantes → imputation nécessaire")
                    if s.nunique()==1: alertes.append(f"⚪ **{col}** : colonne constante (toujours '{s.iloc[0]}') → sans intérêt analytique")
                    if s.nunique()==len(s) and str(s.dtype)=='object': alertes.append(f"🔵 **{col}** : 100% de valeurs uniques → probablement un identifiant")
                    if str(s.dtype) in ['float64','int64']:
                        sk = abs(s.skew())
                        if sk > 2: alertes.append(f"🟡 **{col}** : skewness={sk:.2f} → distribution très asymétrique, envisager une transformation log")
                if dup_total > 0: alertes.append(f"🔴 **Doublons** : {dup_total} lignes dupliquées détectées")
                if alertes:
                    for a in alertes: st.markdown(a)
                else:
                    st.markdown('<span class="badge badge-success">✓ Aucune alerte détectée — données de bonne qualité</span>', unsafe_allow_html=True)

    with tab2:
        st.markdown("**Tests de normalité**")
        st.markdown('<div class="info-box">La normalité est une hypothèse fondamentale de nombreux tests statistiques (t-test, ANOVA, régression linéaire). Ces tests vérifient si vos données suivent une loi normale.</div>', unsafe_allow_html=True)
        if not num_cols: st.info("Aucune colonne numérique.")
        else:
            norm_col = st.selectbox("Colonne à tester", num_cols, key='norm_test_col')
            s_clean = df[norm_col].dropna()
            if len(s_clean) < 3:
                st.warning("Pas assez de données.")
            else:
                results = []
                # Shapiro-Wilk (n < 5000)
                if len(s_clean) <= 5000:
                    stat_sw, p_sw = stats.shapiro(s_clean)
                    results.append({"Test":"Shapiro-Wilk","Statistique":round(stat_sw,6),"p-value":round(p_sw,6),
                                    "Normalité":interpret_p(p_sw),"Usage":"Recommandé pour n < 50"})
                # D'Agostino-Pearson
                stat_dp, p_dp = stats.normaltest(s_clean)
                results.append({"Test":"D'Agostino-Pearson (K²)","Statistique":round(stat_dp,6),"p-value":round(p_dp,6),
                                "Normalité":interpret_p(p_dp),"Usage":"Basé sur skewness + kurtosis"})
                # Kolmogorov-Smirnov
                stat_ks, p_ks = stats.kstest(s_clean, 'norm', args=(s_clean.mean(), s_clean.std()))
                results.append({"Test":"Kolmogorov-Smirnov","Statistique":round(stat_ks,6),"p-value":round(p_ks,6),
                                "Normalité":interpret_p(p_ks),"Usage":"Compare à une normale théorique"})
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                st.markdown('<div class="info-box">⚠ Si p < 0.05 → on rejette H₀ (normalité) : la distribution n\'est PAS normale.<br>Dans ce cas, préférez les tests non-paramétriques (Spearman, Kruskal-Wallis, Mann-Whitney).</div>', unsafe_allow_html=True)

                # QQ-Plot
                fig_qq, ax = plt.subplots(figsize=(6,4))
                fig_qq.patch.set_facecolor('#0d0f14')
                ax.set_facecolor('#161920')
                stats.probplot(s_clean, dist="norm", plot=ax)
                ax.get_lines()[0].set(color='#7c6ffd', markersize=3, alpha=0.6)
                ax.get_lines()[1].set(color='#00e5a0', linewidth=2)
                ax.set_title(f"QQ-Plot — {norm_col}", color='#e8eaf0')
                ax.tick_params(colors='#6b7280')
                ax.spines[:].set_color('#2a2f3d')
                ax.xaxis.label.set_color('#6b7280')
                ax.yaxis.label.set_color('#6b7280')
                st.pyplot(fig_qq, use_container_width=True)

    with tab3:
        st.markdown("**Régression linéaire multiple**")
        st.markdown('<div class="info-box">Modélise la relation entre une variable cible (Y) et plusieurs variables explicatives (X₁, X₂, ...).</div>', unsafe_allow_html=True)
        if len(num_cols) < 2: st.info("Il faut au moins 2 colonnes numériques.")
        else:
            target = st.selectbox("Variable cible (Y)", num_cols, key='reg_target')
            features = st.multiselect("Variables explicatives (X)", [c for c in num_cols if c!=target],
                                       default=[c for c in num_cols if c!=target][:min(3,len(num_cols)-1)], key='reg_features')
            if features and st.button("Calculer la régression"):
                try:
                    clean_reg = df[[target]+features].dropna()
                    X = clean_reg[features].values
                    y = clean_reg[target].values
                    X_b = np.column_stack([np.ones(len(X)), X])
                    # OLS manuel via numpy
                    beta = np.linalg.lstsq(X_b, y, rcond=None)[0]
                    y_pred = X_b @ beta
                    ss_res = np.sum((y - y_pred)**2)
                    ss_tot = np.sum((y - y.mean())**2)
                    r2 = 1 - ss_res/ss_tot
                    r2_adj = 1 - (1-r2)*(len(y)-1)/(len(y)-len(features)-1)
                    rmse = np.sqrt(ss_res/len(y))
                    mae  = np.mean(np.abs(y-y_pred))

                    c1,c2,c3,c4 = st.columns(4)
                    for c,(l,v,col) in zip([c1,c2,c3,c4],[
                        ("R²",f"{r2:.4f}","#00e5a0"),
                        ("R² ajusté",f"{r2_adj:.4f}","#7c6ffd"),
                        ("RMSE",f"{rmse:.4f}","#ff6b6b"),
                        ("MAE",f"{mae:.4f}","#fbbf24"),
                    ]):
                        with c: st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:{col}">{v}</span><div class="metric-label">{l}</div></div>', unsafe_allow_html=True)

                    coef_df = pd.DataFrame({'Variable':['Constante']+features,'Coefficient':beta.round(6)})
                    st.dataframe(coef_df, use_container_width=True, hide_index=True)

                    fig_reg = go.Figure()
                    fig_reg.add_scatter(x=y, y=y_pred, mode='markers', marker=dict(color='#7c6ffd', opacity=0.5, size=5), name='Prédictions')
                    rng_line = [min(y.min(),y_pred.min()), max(y.max(),y_pred.max())]
                    fig_reg.add_scatter(x=rng_line, y=rng_line, mode='lines', line=dict(color='#00e5a0', dash='dash'), name='Parfait')
                    fig_reg.update_layout(title="Valeurs réelles vs prédites", xaxis_title="Réel", yaxis_title="Prédit", **plotly_theme())
                    st.plotly_chart(fig_reg, use_container_width=True)
                    st.markdown(f'<div class="info-box"><b>R²={r2:.4f}</b> : le modèle explique {r2*100:.1f}% de la variance de {target}.<br>{"✅ Bon ajustement" if r2>0.7 else "⚠ Ajustement modéré" if r2>0.4 else "❌ Faible ajustement — envisager d\'autres variables ou transformations"}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Erreur : {e}")

    with tab4:
        st.markdown("**Analyse en Composantes Principales (ACP / PCA)**")
        st.markdown('<div class="info-box">L\'ACP réduit la dimensionnalité en projetant les données dans un espace de moindre dimension tout en conservant le maximum de variance.</div>', unsafe_allow_html=True)
        if len(num_cols) < 2: st.info("Il faut au moins 2 colonnes numériques.")
        else:
            pca_cols = st.multiselect("Colonnes pour l'ACP", num_cols, default=num_cols[:min(6,len(num_cols))], key='pca_cols')
            n_comp   = st.slider("Nombre de composantes", 2, min(len(pca_cols),10) if pca_cols else 2, 2, key='pca_n')
            if pca_cols and len(pca_cols)>=2 and st.button("Lancer l'ACP"):
                try:
                    X_pca = df[pca_cols].dropna()
                    X_std = (X_pca - X_pca.mean()) / X_pca.std()
                    cov_matrix = np.cov(X_std.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                    idx = np.argsort(eigenvalues)[::-1]
                    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:,idx]
                    explained = eigenvalues / eigenvalues.sum() * 100
                    cumulative = np.cumsum(explained)
                    scores = X_std.values @ eigenvectors[:,:n_comp]

                    # Variance expliquée
                    fig_var = go.Figure()
                    fig_var.add_bar(x=[f"PC{i+1}" for i in range(len(explained))], y=explained.round(2),
                                    marker_color='#7c6ffd', name='Variance (%)')
                    fig_var.add_scatter(x=[f"PC{i+1}" for i in range(len(explained))], y=cumulative.round(2),
                                        mode='lines+markers', line=dict(color='#00e5a0', width=2), name='Cumulée (%)')
                    fig_var.update_layout(title="Variance expliquée par composante", **plotly_theme())
                    st.plotly_chart(fig_var, use_container_width=True)
                    st.markdown(f'<div class="info-box">Les {n_comp} premières composantes expliquent <b>{cumulative[n_comp-1]:.1f}%</b> de la variance totale.</div>', unsafe_allow_html=True)

                    # Projection 2D
                    df_scores = pd.DataFrame(scores[:,:2], columns=['PC1','PC2'])
                    if cat_cols:
                        color_col = st.selectbox("Colorier par", ["Aucun"]+cat_cols, key='pca_color')
                        if color_col != "Aucun":
                            df_scores[color_col] = df[color_col].dropna().reset_index(drop=True)
                        fig_sc = px.scatter(df_scores, x='PC1', y='PC2',
                                            color=None if color_col=="Aucun" else color_col,
                                            color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24'],
                                            opacity=0.7)
                    else:
                        fig_sc = px.scatter(df_scores, x='PC1', y='PC2', color_discrete_sequence=['#7c6ffd'], opacity=0.7)
                    fig_sc.update_layout(title=f"Projection ACP — PC1 ({explained[0]:.1f}%) × PC2 ({explained[1]:.1f}%)", **plotly_theme())
                    st.plotly_chart(fig_sc, use_container_width=True)

                    # Loadings
                    loadings = pd.DataFrame(eigenvectors[:,:n_comp], index=pca_cols, columns=[f"PC{i+1}" for i in range(n_comp)]).round(4)
                    st.markdown("**Contributions des variables (loadings)**")
                    st.dataframe(loadings, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur ACP : {e}")

    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 10 — GUIDE & GLOSSAIRE (identique v2)
# ══════════════════════════════════════════════════════════════
elif page == "📚 Guide & Glossaire":
    st.markdown('<p class="section-header">Guide d\'utilisation & Glossaire statistique</p>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📖 Concepts de base", "🧪 Tests statistiques", "🔧 Techniques de nettoyage"])

    with tab1:
        concepts = [
            ("Statistiques descriptives","Fondamentaux","Résument et décrivent les caractéristiques principales d'un jeu de données.",
             ["Moyenne : somme des valeurs divisée par leur nombre","Médiane : valeur centrale qui divise les données en deux",
              "Mode : valeur la plus fréquente","Écart-type (σ) : dispersion autour de la moyenne",
              "Variance (σ²) : carré de l'écart-type","Percentile : valeur sous laquelle tombe X% des observations"]),
            ("Valeurs manquantes (NaN)","Qualité des données","Données absentes dans le jeu de données.",
             ["MCAR : manquantes complètement au hasard","MAR : manquantes selon d'autres variables",
              "MNAR : manquantes selon la valeur elle-même","Imputation : remplacement par une valeur estimée"]),
            ("Outliers","Qualité des données","Valeurs très éloignées du reste des données.",
             ["Méthode IQR : outlier si < Q1-1.5×IQR ou > Q3+1.5×IQR",
              "Z-score : outlier si |z| > 3","Winsorisation : remplacement par les bornes"]),
            ("Distribution","Fondamentaux","Façon dont les valeurs d'une variable sont réparties.",
             ["Normale (gaussienne) : en cloche, symétrique","Skewness > 0 : queue à droite",
              "Kurtosis > 0 : distribution pointue","KDE : estimation lissée de la distribution"]),
            ("Corrélation","Relations","Mesure de la relation linéaire entre deux variables. Entre -1 et +1.",
             ["r ≈ +1 : forte corrélation positive","r ≈ -1 : forte corrélation négative",
              "r ≈ 0 : peu de corrélation","Corrélation ≠ causalité !"]),
            ("Encodage","Préparation","Transformation des variables catégorielles en numérique.",
             ["Label Encoding : entier par catégorie","One-Hot Encoding : colonne binaire par catégorie",
              "Ordinal Encoding : entiers selon un ordre défini"]),
            ("Normalisation vs Standardisation","Préparation","Mise à l'échelle des variables numériques.",
             ["Min-Max [0,1] : (x - min) / (max - min)","Z-score : (x - μ) / σ",
              "Robust : médiane et IQR — résistant aux outliers"]),
        ]
        for title, tag, desc, bullets in concepts:
            st.markdown(f'<div class="glossary-card"><div class="glossary-tag">{tag}</div><div class="glossary-title">📌 {title}</div><div class="glossary-body">{desc}</div></div>', unsafe_allow_html=True)
            with st.expander(f"Détails — {title}"):
                for b in bullets: st.markdown(f"• {b}")

    with tab2:
        tests = [
            ("Pearson (r)","Corrélation","Corrélation linéaire entre deux variables numériques.","Deux numériques continues.","H₀ : r=0. p<0.05 → corrélation significative.","• |r|≥0.7:forte | 0.5-0.7:modérée | <0.3:faible"),
            ("Spearman (ρ)","Corrélation","Alternative non paramétrique à Pearson.","Deux variables ordinales ou numériques.","H₀ : pas de corrélation de rang.","Plus robuste aux outliers que Pearson."),
            ("Kendall (τ)","Corrélation","Concordance entre deux classements.","Deux ordinales ou numériques.","H₀ : pas d'association.","Plus fiable sur petits échantillons."),
            ("ANOVA (F-test)","Comparaison de groupes","Compare les moyennes de plusieurs groupes.","Catégorielle + numérique. Suppose normalité.","H₀ : moyennes égales. p<0.05 → un groupe diffère.","Compléter avec test post-hoc (Tukey)."),
            ("Kruskal-Wallis","Comparaison de groupes","Alternative non paramétrique à l'ANOVA.","Ne suppose pas la normalité.","H₀ : distributions identiques.","Préféré si données non-normales."),
            ("χ² (Chi-deux)","Association catégorielle","Indépendance entre deux variables catégorielles.","Deux catégorielles. Effectifs ≥ 5.","H₀ : variables indépendantes.","Compléter avec V de Cramér."),
            ("V de Cramér","Association catégorielle","Force de l'association entre catégorielles.","Calculé à partir du χ².","Entre 0 et 1.","• >0.3:forte | 0.1-0.3:modérée | <0.1:faible"),
            ("Shapiro-Wilk","Normalité","Teste si une distribution est normale.","Recommandé pour n < 5000.","H₀ : distribution normale. p<0.05 → non normale.","Le plus puissant des tests de normalité."),
        ]
        for title, tag, desc, usage, hyp, interp in tests:
            st.markdown(f'<div class="glossary-card"><div class="glossary-tag">{tag}</div><div class="glossary-title"> {title}</div><div class="glossary-body">{desc}</div></div>', unsafe_allow_html=True)
            with st.expander(f"Détails — {title}"):
                st.markdown(f"**Quand ?** {usage} | **Hypothèse :** {hyp} | **Règle :** {interp}")

    with tab3:
        techniques = [
            ("IQR","Outliers","Zone normale : Q1-1.5×IQR à Q3+1.5×IQR.",["Borne inf = Q1-1.5×IQR","Borne sup = Q3+1.5×IQR","×3 pour les outliers extrêmes"]),
            ("Imputation moyenne","Valeurs manquantes","Remplace NaN par la moyenne. Sensible aux outliers.",["Rapide et simple","Réduit la variance","Éviter si distribution asymétrique"]),
            ("Imputation médiane","Valeurs manquantes","Remplace NaN par la médiane. Robuste.",["Recommandée si distribution asymétrique","Insensible aux outliers"]),
            ("One-Hot Encoding","Encodage","Crée une colonne binaire par modalité.",["Rouge/Bleu/Vert → 3 colonnes","Pour variables nominales","Attention au dummy variable trap"]),
            ("Min-Max","Mise à l'échelle","Ramène les valeurs entre 0 et 1.",["Formule : (x-min)/(max-min)","Sensible aux outliers","Recommandée : réseaux de neurones, KNN"]),
            ("Z-score","Mise à l'échelle","Centre et réduit (μ=0, σ=1).",["Formule : (x-μ)/σ","Pour SVM, régression","Moins sensible aux outliers que Min-Max"]),
        ]
        for title, tag, desc, bullets in techniques:
            st.markdown(f'<div class="glossary-card"><div class="glossary-tag">{tag}</div><div class="glossary-title"> {title}</div><div class="glossary-body">{desc}</div></div>', unsafe_allow_html=True)
            with st.expander(f"Détails — {title}"):
                for b in bullets: st.markdown(f"• {b}")

    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 11 — COURS DATA SCIENCE  ← NOUVEAU
# ══════════════════════════════════════════════════════════════
elif page == "🎓 Pré-requis":
    st.markdown('<p class="section-header">Résumé de cours complet — Data Analyst & Data Scientist</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Ce cours couvre les notions fondamentales qu\'un bon Data Analyst ou Data Scientist doit maîtriser. Chaque chapitre est un résumé dense et actionnable.</div>', unsafe_allow_html=True)

    chapitre = st.selectbox("📖 Choisir un chapitre", [
        "1. Fondements statistiques",
        "2. Probabilités & distributions",
        "3. Tests d'hypothèses",
        "4. Régression & modélisation",
        "5. Machine Learning supervisé",
        "6. Machine Learning non supervisé",
        "7. Préparation & Feature Engineering",
        "8. Évaluation des modèles",
        "9. SQL & manipulation de données",
        "10. Visualisation & storytelling",
        "11. Bonnes pratiques & éthique",
    ])

    if chapitre.startswith("1."):
        st.markdown('<div class="course-chapter"><div class="course-title"> Chapitre 1 — Fondements statistiques</div><div class="course-body">La statistique est la science de la collecte, l\'analyse et de l\'interprétation des données. Deux grandes branches :</div></div>', unsafe_allow_html=True)
        tabs = st.tabs(["Statistiques descriptives","Paramètres de position","Paramètres de dispersion","Forme de la distribution"])
        with tabs[0]:
            st.markdown("""
**Statistiques descriptives** : résument les données sans généraliser.
- **Population** : ensemble complet des éléments étudiés
- **Échantillon** : sous-ensemble représentatif de la population
- **Variable quantitative** : valeurs numériques (âge, revenu)
- **Variable qualitative** : catégories (sexe, couleur)
- **Variable discrète** : valeurs entières (nombre d'enfants)
- **Variable continue** : valeurs sur un intervalle (taille, poids)
""")
        with tabs[1]:
            st.markdown("**Paramètres de position :**")
            st.markdown('<div class="course-formula">Moyenne : x̄ = (Σxᵢ) / n</div>', unsafe_allow_html=True)
            st.markdown('<div class="course-formula">Médiane : valeur centrale quand les données sont triées</div>', unsafe_allow_html=True)
            st.markdown('<div class="course-formula">Mode : valeur la plus fréquente</div>', unsafe_allow_html=True)
            st.markdown("""
- **Moyenne** : sensible aux valeurs extrêmes. À utiliser quand la distribution est symétrique.
- **Médiane** : robuste aux outliers. Préférer pour distributions asymétriques.
- **Mode** : utile pour les variables catégorielles ou discrètes.
- **Percentiles** : P25 (Q1), P50 (médiane), P75 (Q3) découpent les données en 4 quartiles.
""")
        with tabs[2]:
            st.markdown("**Paramètres de dispersion :**")
            st.markdown('<div class="course-formula">Variance : σ² = Σ(xᵢ - x̄)² / n</div>', unsafe_allow_html=True)
            st.markdown('<div class="course-formula">Écart-type : σ = √σ²</div>', unsafe_allow_html=True)
            st.markdown('<div class="course-formula">IQR = Q3 - Q1</div>', unsafe_allow_html=True)
            st.markdown('<div class="course-formula">CV (coefficient de variation) = σ / x̄ × 100%</div>', unsafe_allow_html=True)
            st.markdown("""
- **Variance** : mesure la dispersion au carré (difficile à interpréter directement)
- **Écart-type** : même unité que les données. Petit σ = données regroupées. Grand σ = données étalées.
- **IQR** : plage des 50% centraux. Résistant aux outliers.
- **CV** : permet de comparer deux séries d'unités différentes.
""")
        with tabs[3]:
            st.markdown('<div class="course-formula">Skewness (asymétrie) : S > 0 → queue droite | S < 0 → queue gauche | S ≈ 0 → symétrique</div>', unsafe_allow_html=True)
            st.markdown('<div class="course-formula">Kurtosis (aplatissement) : K > 3 → pointue (leptokurtique) | K < 3 → aplatie (platykurtique) | K = 3 → normale (mésokurtique)</div>', unsafe_allow_html=True)
            st.markdown("""
**Règle empirique (loi normale) :**
- μ ± 1σ contient **68%** des données
- μ ± 2σ contient **95%** des données
- μ ± 3σ contient **99.7%** des données
""")

    elif chapitre.startswith("2."):
        st.markdown('<div class="course-chapter"><div class="course-title"> Chapitre 2 — Probabilités & Distributions</div></div>', unsafe_allow_html=True)
        tabs = st.tabs(["Probabilités de base","Distributions discrètes","Distributions continues","Théorème central limite"])
        with tabs[0]:
            st.markdown("""
**Probabilité** : mesure de la vraisemblance d'un événement. Entre 0 et 1.
""")
            st.markdown('<div class="course-formula">P(A) = Nombre de cas favorables / Nombre de cas possibles</div>', unsafe_allow_html=True)
            st.markdown("""
- **P(A∪B)** = P(A) + P(B) - P(A∩B) — Union
- **P(A∩B)** = P(A) × P(B|A) — Intersection (événements dépendants)
- **P(A∩B)** = P(A) × P(B) — si A et B indépendants
- **Probabilité conditionnelle** : P(A|B) = P(A∩B) / P(B)
- **Théorème de Bayes** : P(A|B) = P(B|A) × P(A) / P(B)
""")
        with tabs[1]:
            st.markdown("""
**Loi de Bernoulli** : expérience à 2 résultats (succès/échec). Paramètre : p (prob. de succès).

**Loi Binomiale** : nombre de succès en n essais indépendants.
""")
            st.markdown('<div class="course-formula">P(X=k) = C(n,k) × p^k × (1-p)^(n-k) | E[X]=np | Var(X)=np(1-p)</div>', unsafe_allow_html=True)
            st.markdown("""
**Loi de Poisson** : nombre d'événements rares dans un intervalle fixé.
""")
            st.markdown('<div class="course-formula">P(X=k) = (λ^k × e^(-λ)) / k! | E[X]=λ | Var(X)=λ</div>', unsafe_allow_html=True)
        with tabs[2]:
            st.markdown("""
**Loi Normale N(μ, σ²)** : la plus importante en statistique. Symétrique en cloche.
""")
            st.markdown('<div class="course-formula">f(x) = (1/σ√(2π)) × exp(-(x-μ)²/(2σ²))</div>', unsafe_allow_html=True)
            st.markdown("""
**Loi Normale Centrée Réduite N(0,1)** : μ=0, σ=1. Variable Z = (X-μ)/σ.

**Loi Student (t)** : utilisée quand σ inconnu et n petit (<30). Paramètre : ddl (degrés de liberté).

**Loi Chi-deux (χ²)** : somme de carrés de normales. Utilisée pour les tests d'adéquation et d'indépendance.

**Loi F (Fisher)** : rapport de deux chi-deux. Utilisée dans l'ANOVA.

**Loi Exponentielle** : temps entre deux événements de Poisson.

**Loi Uniforme** : toutes les valeurs équiprobables entre a et b.
""")
        with tabs[3]:
            st.markdown("""
**Théorème Central Limite (TCL)** : l'un des théorèmes les plus importants en statistique.
""")
            st.markdown('<div class="course-formula">Si X₁,...,Xₙ sont i.i.d. avec μ et σ² finis, alors x̄ → N(μ, σ²/n) quand n → ∞</div>', unsafe_allow_html=True)
            st.markdown("""
**Ce que ça signifie :** la moyenne d'un grand échantillon suit une loi normale, **quelle que soit la distribution initiale**.
- En pratique : n ≥ 30 est généralement suffisant.
- Fondement de nombreux tests statistiques.
- Justifie l'utilisation de la loi normale pour l'inférence.
""")

    elif chapitre.startswith("3."):
        st.markdown('<div class="course-chapter"><div class="course-title"> Chapitre 3 — Tests d\'hypothèses</div></div>', unsafe_allow_html=True)
        tabs = st.tabs(["Logique des tests","Tests paramétriques","Tests non paramétriques","Erreurs & puissance"])
        with tabs[0]:
            st.markdown("""
**Logique d'un test statistique :**
1. **H₀ (hypothèse nulle)** : hypothèse de départ qu'on cherche à réfuter (ex: "les deux groupes ont la même moyenne")
2. **H₁ (hypothèse alternative)** : ce qu'on cherche à montrer (ex: "les groupes diffèrent")
3. **Statistique de test** : valeur calculée à partir des données
4. **p-value** : probabilité d'obtenir un résultat aussi extrême si H₀ est vraie
5. **Décision** : si p < α (seuil, généralement 0.05) → on rejette H₀
""")
            st.markdown('<div class="course-formula">p-value < α → Rejet de H₀ → Résultat statistiquement significatif</div>', unsafe_allow_html=True)
        with tabs[1]:
            st.markdown("""
**Tests paramétriques** (supposent une distribution normale) :

| Test | Usage | H₀ |
|---|---|---|
| **z-test** | Comparer une moyenne à μ₀ (n grand, σ connu) | μ = μ₀ |
| **t-test 1 échantillon** | Comparer une moyenne à μ₀ (σ inconnu) | μ = μ₀ |
| **t-test 2 échantillons indépendants** | Comparer 2 moyennes | μ₁ = μ₂ |
| **t-test apparié** | Avant/après sur les mêmes sujets | μ(diff) = 0 |
| **ANOVA 1 facteur** | Comparer ≥3 moyennes | μ₁=μ₂=...=μₖ |
| **ANOVA 2 facteurs** | 2 facteurs simultanément | Pas d'effet des facteurs |
| **Test de Levene** | Homogénéité des variances | σ₁²=σ₂² |
""")
        with tabs[2]:
            st.markdown("""
**Tests non paramétriques** (sans hypothèse de normalité) :

| Test paramétrique | Équivalent non paramétrique |
|---|---|
| t-test 1 échantillon | Test de Wilcoxon signé |
| t-test 2 échantillons | Test de Mann-Whitney U |
| ANOVA | Test de Kruskal-Wallis |
| Pearson | Spearman, Kendall |
| t-test apparié | Wilcoxon paires |

**Quand les utiliser ?**
- Données non normalement distribuées
- Petit échantillon (n < 30)
- Variables ordinales
- Présence d'outliers importants
""")
        with tabs[3]:
            st.markdown("""
**Erreurs de décision :**
- **Erreur de Type I (α)** : rejeter H₀ alors qu'elle est vraie → faux positif. Contrôlée par le seuil α.
- **Erreur de Type II (β)** : ne pas rejeter H₀ alors qu'elle est fausse → faux négatif.
- **Puissance (1-β)** : probabilité de détecter un vrai effet. Dépend de n, α, et la taille d'effet.
""")
            st.markdown('<div class="course-formula">Puissance ↑ si : n↑ ou α↑ ou effet↑</div>', unsafe_allow_html=True)
            st.markdown("""
**Taille d'effet (effect size)** : mesure l'ampleur pratique d'une différence, indépendamment de n.
- **Cohen's d** : (μ₁-μ₂)/σ_poolé → 0.2=faible, 0.5=moyen, 0.8=fort
- **η² (eta carré)** : dans l'ANOVA, proportion de variance expliquée
- **r de Pearson** : pour les corrélations

⚠ Un résultat significatif ≠ résultat important. Avec n très grand, même une différence infime devient significative.
""")

    elif chapitre.startswith("4."):
        st.markdown('<div class="course-chapter"><div class="course-title"> Chapitre 4 — Régression & Modélisation</div></div>', unsafe_allow_html=True)
        tabs = st.tabs(["Régression linéaire simple","Régression linéaire multiple","Régression logistique","Hypothèses & diagnostics"])
        with tabs[0]:
            st.markdown('<div class="course-formula">Y = β₀ + β₁X + ε</div>', unsafe_allow_html=True)
            st.markdown("""
- **β₀** : ordonnée à l'origine (valeur de Y quand X=0)
- **β₁** : pente (variation de Y pour une unité de X)
- **ε** : terme d'erreur (résidu)
- **Estimation OLS** : minimise Σ(yᵢ - ŷᵢ)²
""")
            st.markdown('<div class="course-formula">R² = 1 - SS_res/SS_tot | interprétation : % de variance de Y expliquée par X</div>', unsafe_allow_html=True)
        with tabs[1]:
            st.markdown('<div class="course-formula">Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε</div>', unsafe_allow_html=True)
            st.markdown("""
**Indicateurs clés :**
- **R² ajusté** : R² pénalisé par le nombre de variables → préférer au R² simple
- **F-statistic** : teste si au moins un βᵢ ≠ 0
- **p-values des coefficients** : teste chaque variable individuellement
- **VIF (Variance Inflation Factor)** : détecte la multicolinéarité. VIF > 10 = problème.

**Sélection de variables :**
- Forward selection : on ajoute les variables une par une
- Backward elimination : on retire les variables une par une
- AIC / BIC : critères d'information (pénalisent la complexité)
""")
        with tabs[2]:
            st.markdown("""
**Régression logistique** : quand Y est binaire (0/1). Prédit une probabilité.
""")
            st.markdown('<div class="course-formula">log(p/(1-p)) = β₀ + β₁X₁ + ... | p = 1/(1+e^(-z))</div>', unsafe_allow_html=True)
            st.markdown("""
- **Odds Ratio** = e^βᵢ → si OddsRatio > 1 : augmente les chances de Y=1
- **Pseudo-R²** (McFadden) : analogue au R² pour la régression logistique
- **Seuil de classification** : par défaut 0.5 (probabilité > 0.5 → classe 1)
- Évaluation : Matrice de confusion, AUC-ROC, précision, rappel
""")
        with tabs[3]:
            st.markdown("""
**Hypothèses de la régression linéaire (BLUE - Gauss-Markov) :**
1. **Linéarité** : la relation entre X et Y est linéaire
2. **Indépendance des erreurs** : les résidus ne sont pas corrélés entre eux (pas d'autocorrélation)
3. **Homoscédasticité** : la variance des résidus est constante (pas de forme en éventail)
4. **Normalité des résidus** : les résidus suivent une loi normale
5. **Absence de multicolinéarité** : les variables X ne sont pas trop corrélées entre elles

**Diagnostics :**
- Graphe résidus vs valeurs ajustées (→ homoscédasticité)
- QQ-plot des résidus (→ normalité)
- Graphe résidus vs ordre (→ indépendance)
- VIF pour chaque variable (→ multicolinéarité)
""")

    elif chapitre.startswith("5."):
        st.markdown('<div class="course-chapter"><div class="course-title"> Chapitre 5 — Machine Learning Supervisé</div></div>', unsafe_allow_html=True)
        tabs = st.tabs(["Principes généraux","Classification","Régression ML","Ensembles"])
        with tabs[0]:
            st.markdown("""
**Machine Learning supervisé** : on apprend à partir de données étiquetées (X, y).

**Pipeline standard :**
1. Collecte et nettoyage des données
2. Feature engineering (création/sélection de variables)
3. Division train/test (80/20 ou 70/30)
4. Entraînement du modèle sur le train
5. Évaluation sur le test
6. Optimisation des hyperparamètres (GridSearch, RandomSearch)
7. Déploiement
""")
            st.markdown('<div class="course-formula">Biais-Variance dilemme : Erreur totale = Biais² + Variance + Bruit irréductible</div>', unsafe_allow_html=True)
            st.markdown("""
- **Sous-ajustement (underfitting)** : biais élevé, variance faible → modèle trop simple
- **Sur-ajustement (overfitting)** : biais faible, variance élevée → modèle trop complexe
- **Régularisation** : L1 (Lasso), L2 (Ridge) — pénalisent la complexité
""")
        with tabs[1]:
            st.markdown("""
| Algorithme | Points forts | Points faibles |
|---|---|---|
| **Régression logistique** | Interprétable, rapide | Linéaire seulement |
| **KNN** | Simple, non paramétrique | Lent sur gros volumes |
| **Arbre de décision** | Interprétable, rapide | Sur-ajustement facile |
| **Random Forest** | Robuste, précis | Moins interprétable |
| **SVM** | Efficace haute dimension | Lent sur gros n |
| **Gradient Boosting (XGBoost)** | Très performant | Nombreux hyperparamètres |
| **Naïf Bayes** | Rapide, texte | Hypothèse forte |
| **Réseau de neurones** | Très flexible | Données massives requises |

**Métriques de classification :**
- **Précision** = VP / (VP + FP) → parmi les prédits positifs, combien sont vrais positifs
- **Rappel (Recall)** = VP / (VP + FN) → parmi les vrais positifs, combien sont détectés
- **F1-score** = 2 × Précision × Rappel / (Précision + Rappel)
- **AUC-ROC** : aire sous la courbe ROC (0.5 = aléatoire, 1 = parfait)
- **Accuracy** = (VP+VN)/(total) — trompeuse si classes déséquilibrées
""")
        with tabs[2]:
            st.markdown("""
**Métriques de régression :**
""")
            st.markdown('<div class="course-formula">MAE = (1/n) × Σ|yᵢ - ŷᵢ| (robuste aux outliers)</div>', unsafe_allow_html=True)
            st.markdown('<div class="course-formula">MSE = (1/n) × Σ(yᵢ - ŷᵢ)² (pénalise les grandes erreurs)</div>', unsafe_allow_html=True)
            st.markdown('<div class="course-formula">RMSE = √MSE (même unité que Y)</div>', unsafe_allow_html=True)
            st.markdown('<div class="course-formula">R² = 1 - Σ(yᵢ-ŷᵢ)²/Σ(yᵢ-ȳ)²</div>', unsafe_allow_html=True)
        with tabs[3]:
            st.markdown("""
**Méthodes d'ensemble** : combinent plusieurs modèles pour améliorer les performances.

- **Bagging** (Bootstrap Aggregating) : entraîne des modèles en parallèle sur des sous-échantillons → Random Forest
- **Boosting** : entraîne des modèles en séquence, chaque modèle corrige les erreurs du précédent → AdaBoost, XGBoost, LightGBM
- **Stacking** : utilise les prédictions de plusieurs modèles comme entrées d'un méta-modèle

**Validation croisée (Cross-validation) :**
- K-Fold : divise les données en K parties, entraîne K fois
- Stratified K-Fold : préserve la proportion des classes
- Leave-One-Out : extrême, chaque observation est un fold
""")

    elif chapitre.startswith("6."):
        st.markdown('<div class="course-chapter"><div class="course-title"> Chapitre 6 — Machine Learning Non Supervisé</div></div>', unsafe_allow_html=True)
        tabs = st.tabs(["Clustering","Réduction de dimension","Règles d'association","Détection d'anomalies"])
        with tabs[0]:
            st.markdown("""
**Clustering** : regrouper des observations similaires sans étiquettes.

**K-Means :**
1. Choisir K centres aléatoires
2. Assigner chaque point au centre le plus proche
3. Recalculer les centres
4. Répéter jusqu'à convergence
- Choisir K : méthode du coude (elbow method) ou silhouette score
""")
            st.markdown('<div class="course-formula">Inertie = Σ ||xᵢ - cₖ||² (à minimiser)</div>', unsafe_allow_html=True)
            st.markdown("""
**DBSCAN** : clustering par densité. Détecte les formes non sphériques et les outliers.
- Points noyaux, points frontières, bruit (outliers)
- Ne nécessite pas de K a priori

**Clustering hiérarchique** :
- Agglomératif : part de n clusters et fusionne
- Divisif : part de 1 cluster et divise
- Dendrogramme pour visualiser les fusions
""")
        with tabs[1]:
            st.markdown("""
**ACP (Analyse en Composantes Principales / PCA) :**
- Projette les données dans un espace de dimension réduite
- Maximise la variance expliquée
- Les composantes sont orthogonales (non corrélées)
- Choisir le nombre de composantes : variance cumulée ≥ 80-95%

**t-SNE** : réduction à 2-3D pour la visualisation. Non linéaire.

**UMAP** : similaire à t-SNE mais plus rapide et préserve la structure globale.

**LDA (Linear Discriminant Analysis)** : réduction supervisée, maximise la séparation entre classes.
""")
        with tabs[2]:
            st.markdown("""
**Règles d'association** : découvrir des patterns "si A alors B" dans des transactions.

**Métriques clés :**
""")
            st.markdown('<div class="course-formula">Support(A→B) = P(A∩B) = fréquence de la règle</div>', unsafe_allow_html=True)
            st.markdown('<div class="course-formula">Confiance(A→B) = P(B|A) = P(A∩B)/P(A)</div>', unsafe_allow_html=True)
            st.markdown('<div class="course-formula">Lift(A→B) = Confiance / P(B) → si >1 : association positive</div>', unsafe_allow_html=True)
            st.markdown("""
**Algorithme Apriori** : extrait les itemsets fréquents, puis génère les règles.
Utilisé en : recommandation, analyse du panier d'achat, médecine.
""")
        with tabs[3]:
            st.markdown("""
**Détection d'anomalies** : identifier les observations inhabituelles.

**Méthodes :**
- **IQR / Z-score** : statistiques univariées
- **Isolation Forest** : isole les anomalies par des coupures aléatoires
- **LOF (Local Outlier Factor)** : compare la densité locale d'un point à ses voisins
- **Autoencoder** : réseau de neurones, l'erreur de reconstruction est grande pour les anomalies
- **One-Class SVM** : apprend la frontière des données normales

**Applications** : fraude bancaire, pannes industrielles, cybersécurité.
""")

    elif chapitre.startswith("7."):
        st.markdown('<div class="course-chapter"><div class="course-title"> Chapitre 7 — Préparation & Feature Engineering</div></div>', unsafe_allow_html=True)
        tabs = st.tabs(["Collecte & exploration","Nettoyage","Feature Engineering","Sélection de variables"])
        with tabs[0]:
            st.markdown("""
**Étapes d'un projet data :**
1. **Business understanding** : comprendre le problème métier
2. **Data understanding** : explorer, visualiser, profiler les données
3. **Data preparation** : nettoyer, transformer, enrichir
4. **Modeling** : choisir et entraîner les modèles
5. **Evaluation** : mesurer les performances
6. **Deployment** : mettre en production

**EDA (Exploratory Data Analysis) :**
- Statistiques descriptives sur chaque variable
- Distribution de chaque variable
- Relations bivariées (corrélations, crosstabs)
- Identification des anomalies, outliers, valeurs manquantes
""")
        with tabs[1]:
            st.markdown("""
**Valeurs manquantes — stratégies :**
- Suppression (si < 5% et MCAR)
- Imputation par moyenne/médiane/mode
- Imputation par modèle (KNN imputation, MICE)
- Créer une colonne indicatrice "est_manquant"

**Outliers — stratégies :**
- Supprimer si erreur de saisie avérée
- Winsorisation (remplacer par les bornes)
- Transformation (log, sqrt) pour réduire l'impact
- Utiliser des algorithmes robustes

**Doublons :** toujours vérifier l'unicité des clés primaires.

**Cohérence des types :** dates, numériques mal interprétées comme texte.
""")
        with tabs[2]:
            st.markdown("""
**Feature Engineering** : créer de nouvelles variables à partir de l'existant.

**Variables numériques :**
- Transformation logarithmique : log(x) — pour distributions asymétriques
- Racine carrée : √x — atténue l'asymétrie
- Binarisation : x > seuil → 0/1
- Interactions : X₁ × X₂, X₁ / X₂

**Variables catégorielles :**
- Label Encoding, One-Hot, Ordinal, Target Encoding
- Groupement de modalités rares en "Autre"
- Fréquence encoding : remplacer par la fréquence d'apparition

**Variables temporelles :**
- Extraire : année, mois, jour, heure, jour_semaine, trimestre
- Ancienneté : jours depuis une date de référence
- Indicateurs : est_weekend, est_ferie, saison

**Texte (NLP basique) :**
- TF-IDF, Bag of Words, embeddings
""")
        with tabs[3]:
            st.markdown("""
**Sélection de variables (Feature Selection) :**

**Méthodes filtre (indépendantes du modèle) :**
- Corrélation avec la cible (Pearson, Spearman)
- Test du χ² (variables catégorielles)
- ANOVA / Kruskal-Wallis
- Variance faible → supprimer (quasi-constantes)

**Méthodes wrapper (évaluent les sous-ensembles) :**
- Forward / Backward / Bidirectional selection
- Recursive Feature Elimination (RFE)

**Méthodes embedded (intégrées au modèle) :**
- Importance des features (Random Forest, XGBoost)
- Coefficients Lasso (L1) — certains deviennent 0
- Coefficients Ridge (L2) — pénalisés mais non nuls

**Règle d'or :** commencer avec peu de variables de qualité plutôt que beaucoup de variables bruitées.
""")

    elif chapitre.startswith("8."):
        st.markdown('<div class="course-chapter"><div class="course-title"> Chapitre 8 — Évaluation des modèles</div></div>', unsafe_allow_html=True)
        tabs = st.tabs(["Métriques classification","Métriques régression","Validation & généralisation","Comparaison de modèles"])
        with tabs[0]:
            st.markdown("""
**Matrice de confusion :**
""")
            st.markdown("""
```
                 Prédit 0    Prédit 1
Réel 0    →    VP (TN)      FP (FP)
Réel 1    →    FN (FN)      VP (TP)
```
""")
            st.markdown("""
| Métrique | Formule | Quand l'utiliser |
|---|---|---|
| **Accuracy** | (TP+TN)/total | Classes équilibrées |
| **Précision** | TP/(TP+FP) | Coût élevé des faux positifs |
| **Rappel (Sensibilité)** | TP/(TP+FN) | Coût élevé des faux négatifs |
| **F1-score** | 2×P×R/(P+R) | Compromis Précision/Rappel |
| **AUC-ROC** | Aire sous ROC | Comparaison de modèles |
| **AUC-PR** | Aire sous PR | Classes déséquilibrées |
| **Spécificité** | TN/(TN+FP) | Taux de vrais négatifs |
""")
        with tabs[1]:
            st.markdown("""
| Métrique | Formule | Interprétation |
|---|---|---|
| **MAE** | Σ\|yᵢ-ŷᵢ\|/n | Erreur moyenne absolue — robuste aux outliers |
| **MSE** | Σ(yᵢ-ŷᵢ)²/n | Pénalise les grandes erreurs |
| **RMSE** | √MSE | Même unité que Y |
| **MAPE** | Σ\|yᵢ-ŷᵢ\|/\|yᵢ\|/n | Erreur relative en % |
| **R²** | 1-SS_res/SS_tot | % variance expliquée (0→1) |
| **R² ajusté** | Pénalise les variables | Comparaison multi-variables |
""")
        with tabs[2]:
            st.markdown("""
**Validation croisée (Cross-validation) :**
- **K-Fold** : divise en K parties, entraîne K fois sur K-1 parties
- **Stratified K-Fold** : préserve la proportion des classes → recommandé pour classification
- **Leave-One-Out (LOO)** : K = n. Très précis mais lent.
- **Time Series Split** : pour données temporelles (le futur ne peut pas être dans le train)

**Hyperparamètres vs paramètres :**
- Paramètres : appris pendant l'entraînement (poids, coefficients)
- Hyperparamètres : définis avant l'entraînement (learning rate, max_depth, n_estimators...)

**Optimisation des hyperparamètres :**
- GridSearchCV : teste toutes les combinaisons
- RandomizedSearchCV : teste un sous-ensemble aléatoire
- Bayesian Optimization (Optuna, Hyperopt) : plus efficace
""")
        with tabs[3]:
            st.markdown("""
**Critères de sélection d'un modèle :**
- Performance sur le jeu de test (ou validation croisée)
- Temps d'entraînement et de prédiction
- Interprétabilité (requis dans certains secteurs : banque, médecine)
- Robustesse (sensibilité aux outliers, aux changements de données)
- Facilité de mise à jour / réentraînement

**AIC / BIC (critères d'information) :**
""")
            st.markdown('<div class="course-formula">AIC = 2k - 2ln(L) | BIC = k×ln(n) - 2ln(L) | (k=nb paramètres, L=vraisemblance)</div>', unsafe_allow_html=True)
            st.markdown("Le modèle avec le **plus petit AIC/BIC** est préféré.")

    elif chapitre.startswith("9."):
        st.markdown('<div class="course-chapter"><div class="course-title"> Chapitre 9 — SQL & Manipulation de données</div></div>', unsafe_allow_html=True)
        tabs = st.tabs(["SQL Fondamentaux","Jointures SQL","Agrégations","Window Functions"])
        with tabs[0]:
            st.markdown("""
**Ordre d'exécution SQL :** FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT

**Requête de base :**
""")
            st.markdown('<div class="course-formula">SELECT col1, col2 FROM table WHERE condition ORDER BY col1 DESC LIMIT 100;</div>', unsafe_allow_html=True)
            st.markdown("""
**Filtres :**
- `WHERE age > 30 AND sexe = \'F\'`
- `WHERE pays IN (\'France\', \'Belgique\')`
- `WHERE nom LIKE \'%Martin%\'`
- `WHERE date BETWEEN \'2023-01-01\' AND \'2023-12-31\'`
- `WHERE valeur IS NULL / IS NOT NULL`

**Opérateurs set :**
- `UNION` : combine et déduplique
- `UNION ALL` : combine sans dédupliquer
- `INTERSECT` : lignes communes
- `EXCEPT` : lignes dans A mais pas dans B
""")
        with tabs[1]:
            st.markdown("""
**Jointures SQL :**
""")
            st.markdown('<div class="course-formula">INNER JOIN → lignes avec correspondance dans les deux tables</div>', unsafe_allow_html=True)
            st.markdown('<div class="course-formula">LEFT JOIN → toutes les lignes de gauche + correspondances droite (NULL si absent)</div>', unsafe_allow_html=True)
            st.markdown('<div class="course-formula">RIGHT JOIN → toutes les lignes de droite + correspondances gauche</div>', unsafe_allow_html=True)
            st.markdown('<div class="course-formula">FULL OUTER JOIN → toutes les lignes des deux tables</div>', unsafe_allow_html=True)
            st.markdown('<div class="course-formula">CROSS JOIN → produit cartésien (chaque ligne A × chaque ligne B)</div>', unsafe_allow_html=True)
            st.markdown('<div class="course-formula">SELF JOIN → une table jointe avec elle-même</div>', unsafe_allow_html=True)
        with tabs[2]:
            st.markdown("""
**Agrégations avec GROUP BY :**
""")
            st.markdown('<div class="course-formula">SELECT pays, COUNT(*), AVG(revenu), MAX(age) FROM clients GROUP BY pays HAVING COUNT(*) > 100;</div>', unsafe_allow_html=True)
            st.markdown("""
**Fonctions d'agrégation :** COUNT, SUM, AVG, MIN, MAX, STDDEV, VAR, MEDIAN

**HAVING** : filtre sur les groupes (après agrégation, contrairement à WHERE)

**Sous-requêtes :**
- Dans WHERE : `WHERE id IN (SELECT id FROM ...)`
- Dans FROM : `FROM (SELECT ... FROM ...) AS t`
- CTE (Common Table Expression) : `WITH t AS (SELECT ...) SELECT * FROM t`
""")
        with tabs[3]:
            st.markdown("""
**Window Functions** (fonctions fenêtre) : calculs sur une fenêtre de lignes sans les regrouper.
""")
            st.markdown('<div class="course-formula">fonction() OVER (PARTITION BY col ORDER BY col2 ROWS BETWEEN ...)</div>', unsafe_allow_html=True)
            st.markdown("""
**Fonctions de ranking :**
- `ROW_NUMBER()` : numéro de ligne unique
- `RANK()` : rang avec ex-aequo (saute les rangs)
- `DENSE_RANK()` : rang avec ex-aequo (ne saute pas)
- `NTILE(n)` : divise en n groupes égaux

**Fonctions analytiques :**
- `LAG(col, n)` : valeur n lignes avant
- `LEAD(col, n)` : valeur n lignes après
- `SUM() OVER (PARTITION BY ...)` : somme cumulée
- `AVG() OVER (ORDER BY date ROWS 7 PRECEDING)` : moyenne glissante
""")

    elif chapitre.startswith("10."):
        st.markdown('<div class="course-chapter"><div class="course-title"> Chapitre 10 — Visualisation & Storytelling</div></div>', unsafe_allow_html=True)
        tabs = st.tabs(["Choisir le bon graphique","Principes de design","Storytelling data","Outils"])
        with tabs[0]:
            st.markdown("""
**Quel graphique pour quel usage ?**

| Objectif | Graphique recommandé |
|---|---|
| Distribution d'une variable | Histogramme, KDE, Boxplot, Violin |
| Comparaison de groupes | Barplot, Boxplot groupé |
| Évolution dans le temps | Ligne, Aires |
| Relation entre 2 variables | Scatter plot, Heatmap |
| Proportions | Pie chart, Treemap, Waffle |
| Corrélation multiple | Heatmap, Pairplot |
| Distribution géographique | Carte choroplèthe |
| Hiérarchie | Treemap, Sunburst |
| Flux | Sankey diagram |
| Réseau | Graph/Network |
""")
        with tabs[1]:
            st.markdown("""
**Principes de Tufte (data-ink ratio) :**
- Maximiser l'encre utile (data) par rapport à l'encre totale
- Supprimer les éléments inutiles (grilles lourdes, 3D, ombres)
- Chaque élément doit avoir un but

**Principes de base :**
- **Titre clair** : ce que montre le graphique
- **Étiquettes d'axes** : toujours avec les unités
- **Légende** : seulement si nécessaire
- **Couleurs** : cohérentes, accessibles (daltonisme), max 6-7 couleurs
- **Échelle** : toujours commencer à 0 pour les barres
- **Annotation** : mettre en évidence les insights clés

**Erreurs courantes :**
- Graphique en camembert avec trop de catégories
- Axe Y tronqué pour exagérer les différences
- Double axe Y trompeur
- Graphiques 3D (difficiles à lire)
""")
        with tabs[2]:
            st.markdown("""
**Storytelling avec les données :**
1. **Contexte** : quelle est la situation actuelle ?
2. **Complication** : quel est le problème ou l'opportunité ?
3. **Résolution** : que montrent les données ? Que recommandez-vous ?

**Structure narrative :**
- Une slide = un message
- Titre de slide = la conclusion (pas la description)
- Données pour prouver, pas pour décorer

**Pyramide de Minto :**
- Conclusion d'abord, arguments ensuite
- Adapté aux décideurs (temps limité)

**Types de graphiques selon l'audience :**
- Technique → graphiques détaillés, statistiques
- Management → graphiques simples, KPIs, tendances
- Grand public → graphiques intuitifs, analogies
""")
        with tabs[3]:
            st.markdown("""
**Outils de visualisation :**

| Outil | Usage |
|---|---|
| **Matplotlib / Seaborn** | Python, graphiques statiques |
| **Plotly** | Python, graphiques interactifs |
| **Tableau** | BI, drag & drop, tableaux de bord |
| **Power BI** | BI Microsoft, intégration Office |
| **Looker / Looker Studio** | BI cloud, Google |
| **D3.js** | JavaScript, visualisations sur mesure |
| **Streamlit** | Web apps data science en Python |
| **Dash** | Dashboards Python interactifs |
| **Flourish / Datawrapper** | Visualisations pour journalistes |
""")

    elif chapitre.startswith("11."):
        st.markdown('<div class="course-chapter"><div class="course-title"> Chapitre 11 — Bonnes pratiques & Éthique</div></div>', unsafe_allow_html=True)
        tabs = st.tabs(["Bonnes pratiques ML","Biais & équité","RGPD & confidentialité","Carrière Data"])
        with tabs[0]:
            st.markdown("""
**Bonnes pratiques en Data Science :**

**Code & reproductibilité :**
- Utiliser des notebooks structurés (avec sections claires)
- Versionner le code avec Git
- Versionner les données avec DVC
- Fixer les seeds aléatoires (`random_state=42`)
- Documenter les choix et hypothèses

**Modélisation :**
- Toujours séparer train/test **avant** tout preprocessing
- Ne jamais utiliser d'informations futures (data leakage)
- Valider sur des données représentatives du contexte de production
- Monitorer les performances en production (model drift)

**Gestion de projet :**
- Définir une baseline simple avant les modèles complexes
- Itérer : commencer simple, complexifier si nécessaire
- Documenter les expériences (MLflow, Weights & Biases)
""")
        with tabs[1]:
            st.markdown("""
**Biais en Machine Learning :**

**Sources de biais :**
- **Biais de sélection** : données non représentatives de la population réelle
- **Biais historique** : les données reflètent des discriminations passées
- **Biais de mesure** : instrument de mesure imparfait ou variable selon les groupes
- **Biais de confirmation** : on cherche à confirmer une hypothèse préexistante

**Exemples réels :**
- Recrutement automatisé favorisant les hommes
- Reconnaissance faciale moins précise sur les personnes noires
- Scoring de crédit discriminant géographiquement

**Métriques d'équité (Fairness) :**
- Parité démographique
- Égalité des opportunités
- Parité prédictive

**Bonne pratique :** analyser les performances du modèle par groupe (sexe, âge, ethnie).
""")
        with tabs[2]:
            st.markdown("""
**RGPD (Règlement Général sur la Protection des Données) :**

**Principes fondamentaux :**
- **Licéité** : base légale pour traiter les données (consentement, contrat, intérêt légitime...)
- **Limitation des finalités** : données collectées pour un but précis
- **Minimisation** : ne collecter que le nécessaire
- **Exactitude** : données à jour
- **Limitation de conservation** : ne pas garder indéfiniment
- **Intégrité et confidentialité** : sécurité des données

**Droits des personnes :**
- Droit d'accès, de rectification, d'effacement (droit à l'oubli)
- Droit à la portabilité
- Droit d'opposition au traitement

**Pour le Data Scientist :**
- Privacy by design : penser à la vie privée dès la conception
- Anonymisation vs pseudonymisation
- Analyse d'impact (DPIA) pour les traitements à risque
- Ne jamais utiliser de vraies données en développement
""")
        with tabs[3]:
            st.markdown("""
**Compétences clés d'un Data Analyst / Data Scientist :**

**Techniques :**
- SQL (indispensable)
- Python ou R
- Statistiques & probabilités
- Machine Learning
- Visualisation (Tableau, Power BI, Plotly)
- Git, ligne de commande

**Méthodes :**
- EDA (Analyse exploratoire)
- Feature Engineering
- Évaluation et sélection de modèles
- Déploiement (API, Streamlit, Docker)

**Soft skills :**
- Curiosité et esprit critique
- Communication (expliquer les résultats à des non-techniques)
- Compréhension du métier (domain knowledge)
- Gestion de projet

**Profils :**
- **Data Analyst** : SQL, Excel/BI, statistiques descriptives, visualisation
- **Data Scientist** : Python/R, ML, statistiques inférentielles, modélisation
- **Data Engineer** : pipelines, ETL, bases de données, cloud
- **ML Engineer** : déploiement de modèles, MLOps, scalabilité

**Certifications utiles :**
- Google Data Analytics (Coursera)
- AWS/Azure/GCP ML certifications
- TensorFlow Developer Certificate
- DataCamp, Kaggle competitions
""")

    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 12 — EXPORT
# ══════════════════════════════════════════════════════════════
elif page == "⬇️ Export":
    st.markdown('<p class="section-header">Export de la base nettoyée</p>', unsafe_allow_html=True)

    df_clean = st.session_state.df
    n_ori, n_clean = len(df_ori), len(df_clean)
    nc_ori, nc_clean = len(df_ori.columns), len(df_clean.columns)
    miss_final = df_clean.isnull().sum().sum()

    c1,c2,c3,c4 = st.columns(4)
    for c,(label,val,color) in zip([c1,c2,c3,c4],[
        ("Lignes originales",f"{n_ori:,}","#6b7280"),
        ("Lignes finales",f"{n_clean:,}","#00e5a0"),
        ("Colonnes orig.",f"{nc_ori}","#6b7280"),
        ("Colonnes finales",f"{nc_clean}","#00e5a0"),
    ]):
        with c:
            st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:{color}">{val}</span><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="info-box">Valeurs manquantes restantes : <b>{miss_final}</b> — Taux de complétude : <b>{round((1-miss_final/(n_clean*nc_clean))*100,2) if n_clean*nc_clean>0 else 100}%</b></div>', unsafe_allow_html=True)

    st.subheader("Aperçu de la base finale")
    st.dataframe(df_clean.head(20), use_container_width=True)

    st.markdown('<p class="section-header">Journal des transformations</p>', unsafe_allow_html=True)
    if st.session_state.cleaning_log:
        st.code("\n".join([f"{i+1}. {e}" for i,e in enumerate(st.session_state.cleaning_log)]), language='text')
    else:
        st.info("Aucune transformation enregistrée.")

    st.markdown('<p class="section-header">Téléchargement</p>', unsafe_allow_html=True)
    fmt      = st.selectbox("Format", ["CSV","Excel","JSON"])
    filename = st.text_input("Nom du fichier", value="data_nettoyee")
    data, mime, ext = download_df(df_clean, fmt)
    st.download_button(f"⬇️ Télécharger {filename}{ext}", data=data,
                       file_name=f"{filename}{ext}", mime=mime, use_container_width=True)

    if st.session_state.cleaning_log:
        report  = "# Rapport de nettoyage — DataClean Pro\n"
        report += f"Auteur de l'outil : Grâce Delesth NGANGA\n"
        report += f"Fichier source : {st.session_state.filename}\n"
        report += f"Lignes : {n_ori} → {n_clean} | Colonnes : {nc_ori} → {nc_clean}\n\n"
        report += "## Opérations effectuées\n"
        for i,e in enumerate(st.session_state.cleaning_log,1): report += f"{i}. {e}\n"
        st.download_button("📄 Télécharger le rapport (.txt)", data=report.encode(),
                           file_name="rapport_nettoyage.txt", mime="text/plain", use_container_width=True)

    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)
