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
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(uploaded_file)
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
            "🔧 Nettoyage",
            "🔄 Transformations",
            "📉 Analyse bivariée",
            "📚 Guide & Glossaire",
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

    # Doublons
    st.markdown('<p class="section-header">Doublons</p>', unsafe_allow_html=True)
    n_dupl = df.duplicated().sum()
    if n_dupl > 0:
        st.markdown(f'<span class="badge badge-danger">⚠ {n_dupl} doublons ({n_dupl/len(df)*100:.2f}%)</span>', unsafe_allow_html=True)
        with st.expander("Afficher les doublons"):
            st.dataframe(df[df.duplicated(keep=False)], use_container_width=True)
    else:
        st.markdown('<span class="badge badge-success">✓ Aucun doublon</span>', unsafe_allow_html=True)

    # Unicité
    st.markdown('<p class="section-header">Unicité des colonnes</p>', unsafe_allow_html=True)
    uniq = pd.DataFrame({'Colonne':df.columns,'Uniques':[df[c].nunique() for c in df.columns],'Total':len(df)})
    uniq['% unicité'] = (uniq['Uniques']/uniq['Total']*100).round(2)
    fig2 = px.bar(uniq.sort_values('Uniques',ascending=True), x='% unicité', y='Colonne',
                  orientation='h', color='% unicité', color_continuous_scale=['#1e2230','#7c6ffd','#00e5a0'], text='Uniques')
    fig2.update_layout(title="Unicité des colonnes", **plotly_theme())
    st.plotly_chart(fig2, use_container_width=True)

    # Outliers IQR
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

    # Quasi-constantes
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
            # Trendline sans statsmodels
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

    st.markdown("**Aperçu du dataframe transformé**")
    st.dataframe(st.session_state.df.head(10), use_container_width=True)
    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 6 — ANALYSE BIVARIÉE (CORRIGÉE)
# ══════════════════════════════════════════════════════════════
elif page == "📉 Analyse bivariée":
    st.markdown('<p class="section-header">Analyse bivariée & relations entre variables</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Explorez les relations entre deux variables. Les tests statistiques sont interprétés automatiquement.</div>', unsafe_allow_html=True)

    if len(df.columns) < 2:
        st.warning("Il faut au moins 2 colonnes.")
    else:
        num_cols2, cat_cols2, _, _ = detect_column_types(df)
        tab1, tab2, tab3 = st.tabs(["🔢 Num × Num", "🏷 Cat × Num", "🗂 Cat × Cat"])

        # ── Num × Num ──────────────────────────────────────────
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
                    with c1:
                        st.markdown(f'<div class="metric-card"><span class="metric-value">{r:.4f}</span><div class="metric-label">Pearson r</div></div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#7c6ffd">{rho:.4f}</span><div class="metric-label">Spearman ρ</div></div>', unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#ff6b6b">{tau:.4f}</span><div class="metric-label">Kendall τ</div></div>', unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="info-box">
                        <b>Pearson r={r:.4f}</b> — {interpret_r(r)} — {interpret_p(p_r)}<br>
                        <b>Spearman ρ={rho:.4f}</b> — {interpret_r(rho)} — {interpret_p(p_s)}<br>
                        <b>Kendall τ={tau:.4f}</b> — {interpret_p(p_t)}<br><br>
                        <b>Régression linéaire :</b> y = {m:.4f}·x + {b:.4f} (si tendance affichée)
                    </div>""", unsafe_allow_html=True)

        # ── Cat × Num ──────────────────────────────────────────
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
                    fig = px.box(df_filt, x=cat_v, y=num_v, color=cat_v,
                                 color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24','#38bdf8'])
                elif viz_choice == "Violin":
                    fig = px.violin(df_filt, x=cat_v, y=num_v, color=cat_v, box=True,
                                    color_discrete_sequence=['#00e5a0','#7c6ffd','#ff6b6b','#fbbf24','#38bdf8'])
                else:
                    grp = df_filt.groupby(cat_v)[num_v].agg(['mean','std']).reset_index()
                    fig = go.Figure()
                    fig.add_bar(x=grp[cat_v], y=grp['mean'],
                                error_y=dict(type='data', array=grp['std'].fillna(0)),
                                marker_color='#7c6ffd', name='Moyenne ± σ')
                fig.update_layout(title=f"{viz_choice} : {cat_v} × {num_v}", xaxis_tickangle=-30, **plotly_theme())
                st.plotly_chart(fig, use_container_width=True)

                grp_stats = df_filt.groupby(cat_v)[num_v].agg(['mean','median','std','count']).round(3).reset_index()
                grp_stats.columns = [cat_v,'Moyenne','Médiane','Écart-type','N']
                st.dataframe(grp_stats, use_container_width=True, hide_index=True)

                # Tests statistiques
                groups = [g[num_v].dropna().values for _, g in df_filt.groupby(cat_v) if g[num_v].dropna().shape[0] > 1]
                if len(groups) >= 2:
                    st.markdown('<p class="section-header">Tests statistiques</p>', unsafe_allow_html=True)
                    try:
                        f_stat, p_anova = f_oneway(*groups)
                        h_stat, p_kruskal = kruskal(*groups)
                        st.markdown(f"""
                        <div class="info-box">
                            <b>ANOVA (F-test)</b> : F={f_stat:.4f} — {interpret_p(p_anova)}<br>
                            <i>Compare les moyennes entre groupes (suppose normalité et homogénéité des variances)</i><br><br>
                            <b>Kruskal-Wallis</b> : H={h_stat:.4f} — {interpret_p(p_kruskal)}<br>
                            <i>Alternative non paramétrique à l'ANOVA (ne suppose pas la normalité)</i>
                        </div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Tests non calculables : {e}")

        # ── Cat × Cat ──────────────────────────────────────────
        with tab3:
            if len(cat_cols2) < 2:
                st.info("Il faut au moins 2 colonnes catégorielles.")
            else:
                c1v = st.selectbox("Colonne 1", cat_cols2, key='bv_c1')
                c2v = st.selectbox("Colonne 2", [c for c in cat_cols2 if c!=c1v], key='bv_c2')
                ct  = pd.crosstab(df[c1v], df[c2v])

                viz_ct = st.radio("Afficher", ["Heatmap","Table de contingence normalisée"], horizontal=True, key='bv_ct_viz')
                if viz_ct == "Heatmap":
                    fig = px.imshow(ct, text_auto=True, aspect='auto',
                                    color_continuous_scale=['#161920','#7c6ffd','#00e5a0'])
                else:
                    ct_norm = pd.crosstab(df[c1v], df[c2v], normalize='index').round(3)
                    fig = px.imshow(ct_norm, text_auto=True, aspect='auto',
                                    color_continuous_scale=['#161920','#7c6ffd','#00e5a0'], zmin=0, zmax=1)
                fig.update_layout(title=f"Contingence : {c1v} × {c2v}", **plotly_theme())
                st.plotly_chart(fig, use_container_width=True)

                try:
                    chi2_val, p_chi, dof, expected = chi2_contingency(ct)
                    n = ct.sum().sum()
                    v_cramer = np.sqrt(chi2_val / (n * (min(ct.shape)-1)))
                    st.markdown(f"""
                    <div class="info-box">
                        <b>Test du χ² (Chi-deux)</b> : χ²={chi2_val:.4f}, ddl={dof} — {interpret_p(p_chi)}<br>
                        <i>Teste l'indépendance entre deux variables catégorielles</i><br><br>
                        <b>V de Cramér</b> : {v_cramer:.4f} — {"Forte" if v_cramer>0.3 else "Modérée" if v_cramer>0.1 else "Faible"} association<br>
                        <i>Mesure la force de l'association (0 = aucune, 1 = parfaite)</i>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Test χ² non calculable : {e}")

    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 7 — GUIDE & GLOSSAIRE
# ══════════════════════════════════════════════════════════════
elif page == "📚 Guide & Glossaire":
    st.markdown('<p class="section-header">Guide d\'utilisation & Glossaire statistique</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 Concepts de base", "🧪 Tests statistiques", "🔧 Techniques de nettoyage"])

    with tab1:
        concepts = [
            ("Statistiques descriptives", "Fondamentaux",
             "Résument et décrivent les caractéristiques principales d'un jeu de données.",
             ["Moyenne : somme des valeurs divisée par leur nombre",
              "Médiane : valeur centrale qui divise les données en deux moitiés égales",
              "Mode : valeur la plus fréquente",
              "Écart-type (σ) : mesure la dispersion des données autour de la moyenne",
              "Variance (σ²) : carré de l'écart-type",
              "Percentile : valeur en-dessous de laquelle tombe X% des observations"]),
            ("Valeurs manquantes (NaN)", "Qualité des données",
             "Données absentes dans le jeu de données. Leur traitement est crucial avant toute analyse.",
             ["MCAR : manquantes complètement au hasard (le plus rare et le moins problématique)",
              "MAR : manquantes aléatoirement selon d'autres variables",
              "MNAR : manquantes selon la valeur elle-même (le plus problématique)",
              "Imputation : remplacement des NaN par une valeur estimée (moyenne, médiane, mode...)"]),
            ("Outliers (valeurs aberrantes)", "Qualité des données",
             "Valeurs très éloignées du reste des données. Peuvent être des erreurs ou des observations rares.",
             ["Méthode IQR : outlier si < Q1-1.5×IQR ou > Q3+1.5×IQR",
              "Z-score : outlier si |z| > 3 (à plus de 3 écarts-types de la moyenne)",
              "Winsorisation : remplace les outliers par les bornes plutôt que de les supprimer"]),
            ("Distribution", "Fondamentaux",
             "Façon dont les valeurs d'une variable sont réparties.",
             ["Normale (gaussienne) : en forme de cloche, symétrique",
              "Skewness (asymétrie) : > 0 queue à droite, < 0 queue à gauche",
              "Kurtosis (aplatissement) : > 0 distribution pointue, < 0 aplatie",
              "KDE (Kernel Density Estimation) : estimation lissée de la distribution"]),
            ("Corrélation", "Relations",
             "Mesure de la relation linéaire entre deux variables numériques. Entre -1 et +1.",
             ["r proche de +1 : forte corrélation positive",
              "r proche de -1 : forte corrélation négative",
              "r proche de 0 : peu ou pas de corrélation",
              "Attention : corrélation ≠ causalité !"]),
            ("Encodage", "Préparation des données",
             "Transformation des variables catégorielles en valeurs numériques pour les algorithmes de ML.",
             ["Label Encoding : attribue un entier à chaque catégorie (1, 2, 3...)",
              "One-Hot Encoding : crée une colonne binaire par catégorie",
              "Ordinal Encoding : attribue des entiers selon un ordre défini"]),
            ("Normalisation vs Standardisation", "Préparation des données",
             "Mise à l'échelle des variables numériques pour les rendre comparables.",
             ["Min-Max [0,1] : (x - min) / (max - min) — résultat entre 0 et 1",
              "Z-score : (x - μ) / σ — moyenne 0, écart-type 1",
              "Robust : utilise la médiane et l'IQR — résistant aux outliers"]),
        ]
        for title, tag, desc, bullets in concepts:
            st.markdown(f"""
            <div class="glossary-card">
                <div class="glossary-tag">{tag}</div>
                <div class="glossary-title">📌 {title}</div>
                <div class="glossary-body">{desc}</div>
            </div>""", unsafe_allow_html=True)
            with st.expander(f"Détails — {title}"):
                for b in bullets:
                    st.markdown(f"• {b}")

    with tab2:
        tests = [
            ("Test de Pearson (r)", "Corrélation",
             "Mesure la corrélation linéaire entre deux variables numériques.",
             "Deux variables numériques continues.",
             "H₀ : pas de corrélation linéaire (r = 0). Si p < 0.05 → corrélation significative.",
             "• |r| ≥ 0.7 : forte | 0.5-0.7 : modérée | 0.3-0.5 : faible | < 0.3 : très faible"),
            ("Test de Spearman (ρ)", "Corrélation",
             "Alternative non paramétrique à Pearson. Mesure la corrélation de rang.",
             "Deux variables ordinales ou numériques (même non normalement distribuées).",
             "H₀ : pas de corrélation de rang. Si p < 0.05 → corrélation significative.",
             "Interprétation similaire à Pearson. Plus robuste aux outliers."),
            ("Test de Kendall (τ)", "Corrélation",
             "Mesure la concordance entre deux classements. Plus robuste que Spearman pour les petits échantillons.",
             "Deux variables ordinales ou numériques.",
             "H₀ : pas d'association. Si p < 0.05 → association significative.",
             "τ entre -1 et +1. Généralement plus petit que ρ mais plus fiable."),
            ("ANOVA (F-test)", "Comparaison de groupes",
             "Compare les moyennes de plusieurs groupes pour voir si elles diffèrent significativement.",
             "Une variable catégorielle (groupes) + une variable numérique. Suppose normalité et homoscédasticité.",
             "H₀ : toutes les moyennes sont égales. Si p < 0.05 → au moins un groupe diffère.",
             "Un p significatif ne dit pas QUELS groupes diffèrent → test post-hoc (Tukey, Bonferroni)."),
            ("Test de Kruskal-Wallis", "Comparaison de groupes",
             "Alternative non paramétrique à l'ANOVA. Compare les distributions de plusieurs groupes.",
             "Une catégorielle + une numérique. Ne suppose pas la normalité.",
             "H₀ : distributions identiques entre groupes. Si p < 0.05 → au moins un groupe diffère.",
             "Préféré quand les données ne suivent pas une loi normale ou avec petits échantillons."),
            ("Test du χ² (Chi-deux)", "Association catégorielle",
             "Teste si deux variables catégorielles sont indépendantes l'une de l'autre.",
             "Deux variables catégorielles. Effectifs attendus ≥ 5 par cellule.",
             "H₀ : les deux variables sont indépendantes. Si p < 0.05 → association significative.",
             "Ne mesure pas la force de l'association → compléter avec le V de Cramér."),
            ("V de Cramér", "Association catégorielle",
             "Mesure la force de l'association entre deux variables catégorielles. Complète le χ².",
             "Calculé à partir du χ².",
             "Toujours entre 0 et 1.",
             "• > 0.3 : forte association | 0.1–0.3 : modérée | < 0.1 : faible"),
            ("Régression linéaire (OLS)", "Modélisation",
             "Modélise la relation entre une variable dépendante Y et une variable indépendante X.",
             "Deux variables numériques continues.",
             "Équation : Y = a·X + b. Le R² mesure la qualité d'ajustement (0 à 1).",
             "OLS = Ordinary Least Squares (Moindres Carrés Ordinaires)."),
        ]
        for title, tag, desc, usage, hyp, interp in tests:
            st.markdown(f"""
            <div class="glossary-card">
                <div class="glossary-tag">{tag}</div>
                <div class="glossary-title">🧪 {title}</div>
                <div class="glossary-body">{desc}</div>
            </div>""", unsafe_allow_html=True)
            with st.expander(f"Détails — {title}"):
                st.markdown(f"**Quand l'utiliser ?** {usage}")
                st.markdown(f"**Hypothèse & interprétation :** {hyp}")
                st.markdown(f"**Règle d'interprétation :** {interp}")

    with tab3:
        techniques = [
            ("Méthode IQR", "Détection d'outliers",
             "L'IQR (Interquartile Range = Q3 - Q1) définit la zone 'normale' des données.",
             ["Borne inférieure = Q1 - 1.5 × IQR", "Borne supérieure = Q3 + 1.5 × IQR",
              "Toute valeur hors de ces bornes est considérée comme outlier",
              "Multiplier par 3 au lieu de 1.5 donne des bornes plus larges (outliers extrêmes)"]),
            ("Imputation par la moyenne", "Valeurs manquantes",
             "Remplace les NaN par la moyenne de la colonne. Simple mais sensible aux outliers.",
             ["Avantage : rapide et simple", "Inconvénient : réduit la variance",
              "À éviter si la distribution est très asymétrique (préférer la médiane)"]),
            ("Imputation par la médiane", "Valeurs manquantes",
             "Remplace les NaN par la médiane. Plus robuste que la moyenne face aux outliers.",
             ["Recommandée pour les distributions asymétriques",
              "Insensible aux valeurs extrêmes"]),
            ("Encodage One-Hot", "Encodage catégoriel",
             "Crée une colonne binaire (0/1) pour chaque modalité de la variable.",
             ["Exemple : 'couleur' avec rouge/bleu/vert → 3 colonnes",
              "Recommandé pour les variables nominales (sans ordre)",
              "Attention au 'dummy variable trap' : supprimer une colonne en présence d'un modèle linéaire"]),
            ("Normalisation Min-Max", "Mise à l'échelle",
             "Ramène toutes les valeurs entre 0 et 1.",
             ["Formule : (x - min) / (max - min)",
              "Sensible aux outliers (ils influencent min et max)",
              "Recommandée pour les réseaux de neurones, KNN"]),
            ("Standardisation Z-score", "Mise à l'échelle",
             "Centre les données (moyenne = 0) et les réduit (écart-type = 1).",
             ["Formule : (x - μ) / σ",
              "Recommandée pour les algorithmes basés sur la distance (SVM, régression logistique)",
              "Moins sensible aux outliers que le Min-Max"]),
        ]
        for title, tag, desc, bullets in techniques:
            st.markdown(f"""
            <div class="glossary-card">
                <div class="glossary-tag">{tag}</div>
                <div class="glossary-title">🔧 {title}</div>
                <div class="glossary-body">{desc}</div>
            </div>""", unsafe_allow_html=True)
            with st.expander(f"Détails — {title}"):
                for b in bullets: st.markdown(f"• {b}")

    st.markdown('<div class="footer">DataClean Pro · <span>Grâce Delesth NGANGA</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 8 — EXPORT
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
