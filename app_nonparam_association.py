import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats.contingency import association
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Uji Nonparametrik Asosiasi", page_icon="🔗", layout="wide")
st.title("🔗 Uji Nonparametrik — Hipotesis Asosiasi")
st.markdown("""
Aplikasi lengkap untuk **uji hipotesis asosiasi/korelasi nonparametrik**:
Spearman, Kendall Tau, Chi-Square Independence, Cramér's V, Koefisien Kontingensi,
Gamma, Somer's D, Point-Biserial, dan lain-lain.
""")

# ============================
# HELPERS
# ============================
def interpret_corr(r):
    r = abs(r)
    if r < 0.1: return "Sangat Lemah"
    elif r < 0.3: return "Lemah"
    elif r < 0.5: return "Sedang"
    elif r < 0.7: return "Kuat"
    elif r < 0.9: return "Sangat Kuat"
    else: return "Sempurna"

def interpret_assoc(v):
    v = abs(v)
    if v < 0.1: return "Sangat Lemah"
    elif v < 0.3: return "Lemah"
    elif v < 0.5: return "Sedang"
    else: return "Kuat"

def cramers_v(contingency_table):
    chi2 = stats.chi2_contingency(contingency_table, correction=False)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0

def tschuprows_t(contingency_table):
    chi2 = stats.chi2_contingency(contingency_table, correction=False)[0]
    n = contingency_table.sum().sum()
    r, c = contingency_table.shape
    return np.sqrt(chi2 / (n * np.sqrt((r-1)*(c-1)))) if n > 0 else 0

def pearsons_contingency(contingency_table):
    chi2 = stats.chi2_contingency(contingency_table, correction=False)[0]
    n = contingency_table.sum().sum()
    return np.sqrt(chi2 / (chi2 + n)) if (chi2 + n) > 0 else 0

def goodman_kruskal_gamma(x, y):
    """Goodman-Kruskal Gamma for ordinal data."""
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0: concordant += 1
            elif prod < 0: discordant += 1
    gamma = (concordant - discordant) / (concordant + discordant) if (concordant + discordant) > 0 else 0
    # Approx SE
    se = 0
    if (concordant + discordant) > 0:
        se = np.sqrt((4 * (concordant + discordant - (concordant - discordant)**2 / (concordant + discordant))) /
                      ((concordant + discordant)**2 * (n * (n - 1)))) if n > 1 else 0
    z = gamma / se if se > 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return gamma, z, p, concordant, discordant

def somers_d(x, y):
    """Somer's D (asymmetric: Y dependent)."""
    n = len(x)
    concordant = 0
    discordant = 0
    ty = 0  # tied on Y only
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dy > 0: concordant += 1
            elif dx * dy < 0: discordant += 1
            elif dy == 0 and dx != 0: ty += 1
    d = (concordant - discordant) / (concordant + discordant + ty) if (concordant + discordant + ty) > 0 else 0
    return d, concordant, discordant, ty

def eta_squared_cat(categories, values):
    """Eta-squared: association between categorical and continuous variable."""
    groups = {}
    for c, v in zip(categories, values):
        groups.setdefault(c, []).append(v)
    grand_mean = np.mean(values)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups.values())
    ss_total = np.sum((values - grand_mean)**2)
    return ss_between / ss_total if ss_total > 0 else 0

# ============================
# SIDEBAR
# ============================
st.sidebar.header("📁 Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])
use_demo = st.sidebar.checkbox("Gunakan data demo", value=True if uploaded_file is None else False)

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    use_demo = False
elif use_demo:
    np.random.seed(42)
    n = 120
    x_cont = np.random.normal(50, 15, n)
    df = pd.DataFrame({
        'Skor_X': x_cont,
        'Skor_Y': 0.6 * x_cont + np.random.normal(0, 12, n),
        'Skor_Z': -0.3 * x_cont + np.random.exponential(10, n),
        'Kategori_A': np.random.choice(['Rendah', 'Sedang', 'Tinggi'], n, p=[0.3, 0.45, 0.25]),
        'Kategori_B': np.random.choice(['Ya', 'Tidak'], n, p=[0.55, 0.45]),
        'Ordinal_X': np.random.choice([1, 2, 3, 4, 5], n),
        'Ordinal_Y': np.random.choice([1, 2, 3, 4, 5], n),
    })
    # make some association in categorical
    for i in range(n):
        if df.loc[i, 'Kategori_A'] == 'Tinggi':
            df.loc[i, 'Kategori_B'] = np.random.choice(['Ya', 'Tidak'], p=[0.8, 0.2])
        elif df.loc[i, 'Kategori_A'] == 'Rendah':
            df.loc[i, 'Kategori_B'] = np.random.choice(['Ya', 'Tidak'], p=[0.3, 0.7])
    # ordinal association
    df['Ordinal_Y'] = np.clip(df['Ordinal_X'] + np.random.choice([-1, 0, 0, 1], n), 1, 5).astype(int)
    st.sidebar.success("Data demo dimuat (120 obs, 7 variabel: kontinu, ordinal, nominal)")
else:
    st.warning("Silakan upload data atau gunakan data demo.")
    st.stop()

# ============================
# 1. EKSPLORASI
# ============================
st.header("1. Eksplorasi Data")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Observasi", df.shape[0])
c2.metric("Jumlah Variabel", df.shape[1])
c3.metric("Missing Values", df.isnull().sum().sum())

tab_d1, tab_d2 = st.tabs(["Data", "Deskriptif"])
with tab_d1:
    st.dataframe(df.head(30), use_container_width=True)
with tab_d2:
    st.dataframe(df.describe(include='all').T.round(4), use_container_width=True)

# ============================
# 2. TIPE ANALISIS
# ============================
st.header("2. Pilih Tipe Analisis Asosiasi")

analysis_type = st.radio("Jenis data variabel:", [
    "🔢 Kontinu / Ordinal vs Kontinu / Ordinal (Korelasi Rank)",
    "📋 Nominal vs Nominal (Tabel Kontingensi)",
    "📊 Nominal vs Kontinu (Eta / Point-Biserial)",
    "🔄 Ordinal vs Ordinal (Gamma, Somer's D, Kendall)",
    "🧮 Matriks Korelasi Multi-Variabel"
], index=0)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
all_cols = df.columns.tolist()

col_a1, col_a2 = st.columns(2)
with col_a1:
    alpha = st.selectbox("Tingkat Signifikansi (α)", [0.01, 0.05, 0.10], index=1)
with col_a2:
    alt_type = st.selectbox("Arah Uji Korelasi", ["two-sided", "greater", "less"], index=0)

summary_rows = []

# ====================================================================
# TYPE A: RANK CORRELATION (Continuous / Ordinal)
# ====================================================================
if "Korelasi Rank" in analysis_type:
    st.header("3. Korelasi Rank: Spearman & Kendall")

    if len(numeric_cols) < 2:
        st.error("Minimal 2 kolom numerik.")
        st.stop()

    vc1, vc2 = st.columns(2)
    with vc1:
        varx = st.selectbox("Variabel X", numeric_cols, index=0)
    with vc2:
        vary_opts = [c for c in numeric_cols if c != varx]
        vary = st.selectbox("Variabel Y", vary_opts, index=0)

    x = df[varx].dropna().values.astype(float)
    y = df[vary].dropna().values.astype(float)
    min_n = min(len(x), len(y))
    x, y = x[:min_n], y[:min_n]
    n_obs = min_n

    # ---- 3a. SPEARMAN ----
    st.subheader("3a. Spearman Rank Correlation (ρ)")
    sp_r, sp_p = stats.spearmanr(x, y, alternative=alt_type)
    t_sp = float(sp_r) * np.sqrt((n_obs - 2) / (1 - float(sp_r)**2)) if abs(float(sp_r)) < 1 else np.inf

    sp_result = pd.DataFrame({
        'Metrik': ['ρ (Spearman)', 't-statistic', 'p-value', 'n', 'Interpretasi',
                   f'Keputusan (α={alpha})'],
        'Nilai': [round(float(sp_r), 6), round(t_sp, 4), round(float(sp_p), 6), n_obs,
                  interpret_corr(sp_r),
                  'Signifikan (Ada asosiasi)' if sp_p < alpha else 'Tidak Signifikan']
    })
    st.dataframe(sp_result, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': 'Spearman ρ', 'Koefisien': round(float(sp_r),6),
                         'p-value': round(float(sp_p),6), 'Interpretasi': interpret_corr(sp_r),
                         'Keputusan': 'Signifikan' if sp_p < alpha else 'Tidak Signifikan'})

    # ---- 3b. KENDALL TAU ----
    st.subheader("3b. Kendall Tau (τ)")
    kt_r, kt_p = stats.kendalltau(x, y, alternative=alt_type)

    kt_result = pd.DataFrame({
        'Metrik': ['τ (Kendall Tau-b)', 'p-value', 'n', 'Interpretasi',
                   f'Keputusan (α={alpha})'],
        'Nilai': [round(float(kt_r), 6), round(float(kt_p), 6), n_obs,
                  interpret_corr(kt_r),
                  'Signifikan' if kt_p < alpha else 'Tidak Signifikan']
    })
    st.dataframe(kt_result, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': 'Kendall τ', 'Koefisien': round(float(kt_r),6),
                         'p-value': round(float(kt_p),6), 'Interpretasi': interpret_corr(kt_r),
                         'Keputusan': 'Signifikan' if kt_p < alpha else 'Tidak Signifikan'})

    # ---- 3c. PEARSON (sebagai pembanding) ----
    st.subheader("3c. Pearson r (Pembanding Parametrik)")
    pr_r, pr_p = stats.pearsonr(x, y, alternative=alt_type)
    pr_result = pd.DataFrame({
        'Metrik': ['r (Pearson)', 'p-value', 'Interpretasi',
                   f'Keputusan (α={alpha})'],
        'Nilai': [round(float(pr_r), 6), round(float(pr_p), 6),
                  interpret_corr(pr_r),
                  'Signifikan' if pr_p < alpha else 'Tidak Signifikan']
    })
    st.dataframe(pr_result, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': 'Pearson r (pembanding)', 'Koefisien': round(float(pr_r),6),
                         'p-value': round(float(pr_p),6), 'Interpretasi': interpret_corr(pr_r),
                         'Keputusan': 'Signifikan' if pr_p < alpha else 'Tidak Signifikan'})

    # Perbandingan tabel
    st.subheader("Perbandingan Koefisien")
    comp_df = pd.DataFrame({
        'Metode': ['Spearman ρ', 'Kendall τ', 'Pearson r'],
        'Koefisien': [round(float(sp_r),6), round(float(kt_r),6), round(float(pr_r),6)],
        'p-value': [round(float(sp_p),6), round(float(kt_p),6), round(float(pr_p),6)],
        'Interpretasi': [interpret_corr(sp_r), interpret_corr(kt_r), interpret_corr(pr_r)]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ---- VISUALISASI ----
    st.header("4. Visualisasi")
    tab1, tab2, tab3, tab4 = st.tabs(["Scatter Plot", "Rank Scatter", "Heatmap Rank", "Residual"])

    with tab1:
        fig = px.scatter(x=x, y=y, trendline="ols", trendline_color_override="red",
                          labels={'x': varx, 'y': vary},
                          title=f"Scatter: {varx} vs {vary} (ρ={float(sp_r):.3f})")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        rx = stats.rankdata(x)
        ry = stats.rankdata(y)
        fig = px.scatter(x=rx, y=ry, trendline="ols", trendline_color_override="red",
                          labels={'x': f'Rank {varx}', 'y': f'Rank {vary}'},
                          title=f"Rank Scatter (Spearman): Rank({varx}) vs Rank({vary})")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = go.Figure(data=go.Heatmap(
            z=np.histogram2d(x, y, bins=15)[0],
            colorscale='Viridis'))
        fig.update_layout(title="2D Density Heatmap", xaxis_title=varx, yaxis_title=vary, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        slope, intercept = np.polyfit(x, y, 1)
        residuals = y - (slope * x + intercept)
        fig = px.scatter(x=x, y=residuals, labels={'x': varx, 'y': 'Residual'},
                          title="Residual Plot (linear fit)")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    # ---- BOOTSTRAP CI ----
    st.header("5. Bootstrap Confidence Interval")
    with st.expander("Jalankan Bootstrap CI untuk Spearman ρ"):
        n_boot = st.number_input("Jumlah bootstrap", 500, 50000, 5000, 500, key='boot_corr')
        if st.button("Run Bootstrap CI", type="primary"):
            boot_rho = np.zeros(int(n_boot))
            for b in range(int(n_boot)):
                idx = np.random.choice(n_obs, n_obs, replace=True)
                boot_rho[b], _ = stats.spearmanr(x[idx], y[idx])
            ci_low = np.percentile(boot_rho, (alpha/2)*100)
            ci_up = np.percentile(boot_rho, (1-alpha/2)*100)
            st.dataframe(pd.DataFrame({
                'Metrik': ['Spearman ρ (observed)', 'Bootstrap SE',
                           f'CI Lower ({(1-alpha)*100:.0f}%)', f'CI Upper ({(1-alpha)*100:.0f}%)'],
                'Nilai': [round(float(sp_r),6), round(np.std(boot_rho),6),
                          round(ci_low,6), round(ci_up,6)]
            }), use_container_width=True, hide_index=True)
            fig_b = go.Figure()
            fig_b.add_trace(go.Histogram(x=boot_rho, nbinsx=50, marker_color='steelblue'))
            fig_b.add_vline(x=float(sp_r), line_color="red", annotation_text=f"ρ={float(sp_r):.3f}")
            fig_b.update_layout(title="Bootstrap Distribution of Spearman ρ", height=400)
            st.plotly_chart(fig_b, use_container_width=True)

# ====================================================================
# TYPE B: NOMINAL vs NOMINAL
# ====================================================================
elif "Nominal" in analysis_type and "Kontingensi" in analysis_type:
    st.header("3. Uji Asosiasi: Nominal vs Nominal")

    all_usable = cat_cols + [c for c in numeric_cols if df[c].nunique() <= 20]
    if len(all_usable) < 2:
        st.error("Minimal 2 variabel kategorikal (atau numerik dgn ≤ 20 unique values).")
        st.stop()

    vc1, vc2 = st.columns(2)
    with vc1:
        varx = st.selectbox("Variabel X (Row)", all_usable, index=0)
    with vc2:
        vary_opts = [c for c in all_usable if c != varx]
        vary = st.selectbox("Variabel Y (Column)", vary_opts, index=0)

    ct = pd.crosstab(df[varx], df[vary])
    ct_arr = ct.values
    n_obs = ct_arr.sum()

    st.subheader("Tabel Kontingensi (Observed)")
    st.dataframe(ct, use_container_width=True)

    # Expected frequencies
    chi2_res = stats.chi2_contingency(ct_arr, correction=False)
    chi2_stat, chi2_p, chi2_df, expected = chi2_res
    exp_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns).round(2)
    with st.expander("Tabel Frekuensi Expected"):
        st.dataframe(exp_df, use_container_width=True)

    # ---- 3a. CHI-SQUARE TEST ----
    st.subheader("3a. Chi-Square Test of Independence")
    chi_result = pd.DataFrame({
        'Metrik': ['χ² statistic', 'df', 'p-value', f'Keputusan (α={alpha})'],
        'Nilai': [round(chi2_stat, 4), int(chi2_df), round(chi2_p, 6),
                  'Terdapat Asosiasi' if chi2_p < alpha else 'Independen (Tidak Ada Asosiasi)']
    })
    st.dataframe(chi_result, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': 'Chi-Square Independence', 'Koefisien': f'χ²={chi2_stat:.4f}',
                         'p-value': round(chi2_p,6), 'Interpretasi': '-',
                         'Keputusan': 'Asosiasi' if chi2_p < alpha else 'Independen'})

    # Likelihood Ratio
    lr_stat = 2 * np.sum(ct_arr * np.log(ct_arr / expected + 1e-15))
    lr_p = 1 - stats.chi2.cdf(lr_stat, chi2_df)
    lr_result = pd.DataFrame({
        'Metrik': ['G² (Likelihood Ratio)', 'df', 'p-value', f'Keputusan (α={alpha})'],
        'Nilai': [round(lr_stat, 4), int(chi2_df), round(lr_p, 6),
                  'Asosiasi' if lr_p < alpha else 'Independen']
    })
    st.dataframe(lr_result, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': 'Likelihood Ratio G²', 'Koefisien': f'G²={lr_stat:.4f}',
                         'p-value': round(lr_p,6), 'Interpretasi': '-',
                         'Keputusan': 'Asosiasi' if lr_p < alpha else 'Independen'})

    # Fisher Exact (only 2x2)
    if ct_arr.shape == (2, 2):
        st.subheader("Fisher's Exact Test (2×2)")
        fe_or, fe_p = stats.fisher_exact(ct_arr, alternative=alt_type)
        fe_result = pd.DataFrame({
            'Metrik': ['Odds Ratio', 'p-value', f'Keputusan (α={alpha})'],
            'Nilai': [round(fe_or, 4), round(fe_p, 6),
                      'Asosiasi' if fe_p < alpha else 'Independen']
        })
        st.dataframe(fe_result, use_container_width=True, hide_index=True)
        summary_rows.append({'Uji': "Fisher's Exact", 'Koefisien': f'OR={fe_or:.4f}',
                             'p-value': round(fe_p,6), 'Interpretasi': '-',
                             'Keputusan': 'Asosiasi' if fe_p < alpha else 'Independen'})

    # ---- 3b. MEASURES OF ASSOCIATION ----
    st.subheader("3b. Ukuran Kekuatan Asosiasi")
    cv = cramers_v(ct)
    tt = tschuprows_t(ct)
    pc = pearsons_contingency(ct)
    phi = np.sqrt(chi2_stat / n_obs) if n_obs > 0 else 0

    assoc_df = pd.DataFrame({
        'Ukuran': ["Cramér's V", "Tschuprow's T", "Pearson's Contingency Coeff. (C)",
                   "Phi (φ) — valid for 2×2"],
        'Nilai': [round(cv, 6), round(tt, 6), round(pc, 6), round(phi, 6)],
        'Interpretasi': [interpret_assoc(cv), interpret_assoc(tt), interpret_assoc(pc), interpret_assoc(phi)]
    })
    st.dataframe(assoc_df, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': "Cramér's V", 'Koefisien': round(cv,6),
                         'p-value': round(chi2_p,6), 'Interpretasi': interpret_assoc(cv),
                         'Keputusan': 'Signifikan' if chi2_p < alpha else 'Tidak Signifikan'})

    # Standardized residuals
    st.subheader("3c. Adjusted Standardized Residuals")
    std_resid = (ct_arr - expected) / np.sqrt(expected * (1 - ct.sum(axis=1).values[:, None]/n_obs) *
                                               (1 - ct.sum(axis=0).values[None, :]/n_obs) + 1e-15)
    std_resid_df = pd.DataFrame(std_resid.round(4), index=ct.index, columns=ct.columns)
    st.dataframe(std_resid_df, use_container_width=True)
    st.markdown("> Nilai > |1.96| menunjukkan sel yang berkontribusi signifikan terhadap asosiasi.")

    # VISUALISASI
    st.header("4. Visualisasi")
    tab1, tab2, tab3, tab4 = st.tabs(["Bar Grouped", "Heatmap Observed", "Heatmap Residual", "Mosaic-Like"])

    with tab1:
        ct_long = ct.reset_index().melt(id_vars=ct.index.name, var_name=vary, value_name='Count')
        fig = px.bar(ct_long, x=ct.index.name, y='Count', color=vary, barmode='group',
                      title=f"Bar Plot: {varx} × {vary}")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure(data=go.Heatmap(z=ct_arr, x=[str(c) for c in ct.columns],
                                         y=[str(r) for r in ct.index], colorscale='Blues',
                                         text=ct_arr, texttemplate="%{text}"))
        fig.update_layout(title="Heatmap Frekuensi Observasi", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = go.Figure(data=go.Heatmap(z=std_resid, x=[str(c) for c in ct.columns],
                                         y=[str(r) for r in ct.index], colorscale='RdBu', zmid=0,
                                         text=std_resid.round(2), texttemplate="%{text}"))
        fig.update_layout(title="Heatmap Adjusted Standardized Residuals", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        ct_pct = ct.div(ct.sum(axis=1), axis=0)
        fig = px.bar(ct_pct.reset_index().melt(id_vars=ct_pct.index.name, var_name=vary, value_name='Proportion'),
                      x=ct_pct.index.name, y='Proportion', color=vary, barmode='stack',
                      title=f"Stacked Bar (Proportion): {varx} × {vary}")
        st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# TYPE C: NOMINAL vs CONTINUOUS
# ====================================================================
elif "Kontinu" in analysis_type and "Nominal" in analysis_type and "Eta" in analysis_type:
    st.header("3. Asosiasi: Nominal vs Kontinu")

    all_cat = cat_cols + [c for c in numeric_cols if df[c].nunique() <= 10]
    if len(all_cat) < 1 or len(numeric_cols) < 1:
        st.error("Minimal 1 variabel kategorikal dan 1 numerik.")
        st.stop()

    vc1, vc2 = st.columns(2)
    with vc1:
        var_cat = st.selectbox("Variabel Kategorikal (Nominal/Ordinal)", all_cat, index=0)
    with vc2:
        cont_opts = [c for c in numeric_cols if c != var_cat]
        var_cont = st.selectbox("Variabel Kontinu", cont_opts, index=0)

    cat_vals = df[var_cat].dropna().values
    cont_vals = df[var_cont].dropna().values
    min_n = min(len(cat_vals), len(cont_vals))
    cat_vals = cat_vals[:min_n]
    cont_vals = cont_vals[:min_n].astype(float)
    n_obs = min_n
    unique_cats = sorted(set(cat_vals))

    # Eta squared
    st.subheader("3a. Eta Squared (η²)")
    eta2 = eta_squared_cat(cat_vals, cont_vals)
    eta = np.sqrt(eta2)
    eta_result = pd.DataFrame({
        'Metrik': ['η (Eta)', 'η² (Eta Squared)', 'Interpretasi', f'Makna'],
        'Nilai': [round(eta, 6), round(eta2, 6), interpret_corr(eta),
                  f'{eta2*100:.1f}% variansi {var_cont} dijelaskan oleh {var_cat}']
    })
    st.dataframe(eta_result, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': 'Eta Squared (η²)', 'Koefisien': round(eta2,6),
                         'p-value': '-', 'Interpretasi': interpret_corr(eta),
                         'Keputusan': f'{eta2*100:.1f}% variansi'})

    # Point-Biserial (if binary)
    if len(unique_cats) == 2:
        st.subheader("3b. Point-Biserial Correlation")
        binary_map = {unique_cats[0]: 0, unique_cats[1]: 1}
        binary = np.array([binary_map[c] for c in cat_vals])
        pb_r, pb_p = stats.pointbiserialr(binary, cont_vals)
        pb_result = pd.DataFrame({
            'Metrik': ['r_pb (Point-Biserial)', 'p-value', 'Interpretasi',
                       f'Keputusan (α={alpha})'],
            'Nilai': [round(float(pb_r), 6), round(float(pb_p), 6), interpret_corr(pb_r),
                      'Signifikan' if pb_p < alpha else 'Tidak Signifikan']
        })
        st.dataframe(pb_result, use_container_width=True, hide_index=True)
        summary_rows.append({'Uji': 'Point-Biserial', 'Koefisien': round(float(pb_r),6),
                             'p-value': round(float(pb_p),6), 'Interpretasi': interpret_corr(pb_r),
                             'Keputusan': 'Signifikan' if pb_p < alpha else 'Tidak Signifikan'})

    # Kruskal-Wallis as association test
    st.subheader(f"3c. Kruskal-Wallis sebagai Uji Asosiasi")
    groups_kw = [cont_vals[cat_vals == c] for c in unique_cats]
    groups_kw = [g for g in groups_kw if len(g) > 0]
    if len(groups_kw) >= 2:
        kw_s, kw_p = stats.kruskal(*groups_kw)
        eps_sq = float(kw_s) / (n_obs - 1) if n_obs > 1 else 0
        kw_result = pd.DataFrame({
            'Metrik': ['H-statistic', 'p-value', 'ε² (Epsilon-squared)',
                       f'Keputusan (α={alpha})'],
            'Nilai': [round(float(kw_s),4), round(float(kw_p),6), round(eps_sq,4),
                      'Asosiasi Signifikan' if kw_p < alpha else 'Tidak Signifikan']
        })
        st.dataframe(kw_result, use_container_width=True, hide_index=True)
        summary_rows.append({'Uji': 'Kruskal-Wallis', 'Koefisien': f'H={float(kw_s):.4f}',
                             'p-value': round(float(kw_p),6), 'Interpretasi': f'ε²={eps_sq:.4f}',
                             'Keputusan': 'Asosiasi' if kw_p < alpha else 'Tidak'})

    # Visualisasi
    st.header("4. Visualisasi")
    tab1, tab2, tab3 = st.tabs(["Box Plot", "Violin Plot", "Mean + CI"])
    melted = pd.DataFrame({'Kategori': cat_vals, 'Nilai': cont_vals})
    with tab1:
        fig = px.box(melted, x='Kategori', y='Nilai', color='Kategori', points='all',
                      title=f"{var_cat} vs {var_cont}")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = px.violin(melted, x='Kategori', y='Nilai', color='Kategori', box=True, points='all')
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        means = melted.groupby('Kategori')['Nilai'].mean()
        sems = melted.groupby('Kategori')['Nilai'].apply(stats.sem)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=means.index.tolist(), y=means.values,
                              error_y=dict(type='data', array=(1.96*sems).values, visible=True)))
        fig.update_layout(title="Mean ± 95% CI", yaxis_title=var_cont, height=400)
        st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# TYPE D: ORDINAL vs ORDINAL (Gamma, Somer's D)
# ====================================================================
elif "Gamma" in analysis_type:
    st.header("3. Asosiasi Ordinal: Gamma, Somer's D, Kendall")

    ord_cols = numeric_cols
    if len(ord_cols) < 2:
        st.error("Minimal 2 variabel ordinal/numerik.")
        st.stop()

    vc1, vc2 = st.columns(2)
    with vc1:
        varx = st.selectbox("Variabel X (Ordinal)", ord_cols, index=min(5, len(ord_cols)-1))
    with vc2:
        vary_opts = [c for c in ord_cols if c != varx]
        vary = st.selectbox("Variabel Y (Ordinal)", vary_opts, index=min(0, len(vary_opts)-1))

    x = df[varx].dropna().values.astype(float)
    y = df[vary].dropna().values.astype(float)
    min_n = min(len(x), len(y))
    x, y = x[:min_n], y[:min_n]
    n_obs = min_n

    # Goodman-Kruskal Gamma
    st.subheader("3a. Goodman-Kruskal Gamma (γ)")
    if n_obs <= 2000:
        gamma, z_g, p_g, conc, disc = goodman_kruskal_gamma(x, y)
        gamma_result = pd.DataFrame({
            'Metrik': ['γ (Gamma)', 'Concordant Pairs', 'Discordant Pairs', 'Z-approx', 'p-value',
                       'Interpretasi', f'Keputusan (α={alpha})'],
            'Nilai': [round(gamma,6), conc, disc, round(z_g,4), round(p_g,6),
                      interpret_corr(gamma),
                      'Signifikan' if p_g < alpha else 'Tidak Signifikan']
        })
        st.dataframe(gamma_result, use_container_width=True, hide_index=True)
        summary_rows.append({'Uji': 'Goodman-Kruskal γ', 'Koefisien': round(gamma,6),
                             'p-value': round(p_g,6), 'Interpretasi': interpret_corr(gamma),
                             'Keputusan': 'Signifikan' if p_g < alpha else 'Tidak'})
    else:
        st.info("Gamma dihitung untuk n ≤ 2000 (komputasi O(n²)).")

    # Somer's D
    st.subheader("3b. Somer's D")
    if n_obs <= 2000:
        d_xy, conc_d, disc_d, ty_d = somers_d(x, y)
        d_yx, _, _, _ = somers_d(y, x)
        d_sym = (d_xy + d_yx) / 2
        sd_result = pd.DataFrame({
            'Metrik': [f"Somer's D(Y|X) — {vary} dependent",
                       f"Somer's D(X|Y) — {varx} dependent",
                       "Somer's D (symmetric)", 'Interpretasi'],
            'Nilai': [round(d_xy,6), round(d_yx,6), round(d_sym,6), interpret_corr(d_sym)]
        })
        st.dataframe(sd_result, use_container_width=True, hide_index=True)
        summary_rows.append({'Uji': "Somer's D (sym)", 'Koefisien': round(d_sym,6),
                             'p-value': '-', 'Interpretasi': interpret_corr(d_sym),
                             'Keputusan': '-'})

    # Kendall Tau
    st.subheader("3c. Kendall Tau-b & Tau-c")
    kt_b, kt_bp = stats.kendalltau(x, y)
    # Tau-c (Stuart's Tau-c)
    ct_ord = pd.crosstab(pd.Categorical(x), pd.Categorical(y))
    m = min(ct_ord.shape)
    tau_c = 2 * m * float(kt_b) * (1 - abs(float(kt_b))) / (m - 1) if m > 1 else float(kt_b)  # approx

    kt_result = pd.DataFrame({
        'Metrik': ['Kendall τ-b', 'p-value (τ-b)', 'Interpretasi',
                   f'Keputusan (α={alpha})'],
        'Nilai': [round(float(kt_b),6), round(float(kt_bp),6),
                  interpret_corr(kt_b),
                  'Signifikan' if kt_bp < alpha else 'Tidak Signifikan']
    })
    st.dataframe(kt_result, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': 'Kendall τ-b', 'Koefisien': round(float(kt_b),6),
                         'p-value': round(float(kt_bp),6), 'Interpretasi': interpret_corr(kt_b),
                         'Keputusan': 'Signifikan' if kt_bp < alpha else 'Tidak'})

    # Spearman
    st.subheader("3d. Spearman ρ (Pembanding)")
    sp_r, sp_p = stats.spearmanr(x, y)
    sp_result = pd.DataFrame({
        'Metrik': ['Spearman ρ', 'p-value', 'Interpretasi'],
        'Nilai': [round(float(sp_r),6), round(float(sp_p),6), interpret_corr(sp_r)]
    })
    st.dataframe(sp_result, use_container_width=True, hide_index=True)

    # Visualisasi
    st.header("4. Visualisasi")
    tab1, tab2 = st.tabs(["Scatter Ordinal", "Heatmap Crosstab"])
    with tab1:
        jitter_x = x + np.random.uniform(-0.2, 0.2, n_obs)
        jitter_y = y + np.random.uniform(-0.2, 0.2, n_obs)
        fig = px.scatter(x=jitter_x, y=jitter_y, opacity=0.5,
                          labels={'x': varx, 'y': vary},
                          title=f"Ordinal Scatter (jittered): {varx} vs {vary}")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        ct_o = pd.crosstab(df[varx], df[vary])
        fig = go.Figure(data=go.Heatmap(z=ct_o.values, x=[str(c) for c in ct_o.columns],
                                         y=[str(r) for r in ct_o.index], colorscale='YlOrRd',
                                         text=ct_o.values, texttemplate="%{text}"))
        fig.update_layout(title=f"Crosstab Heatmap: {varx} × {vary}", height=400)
        st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# TYPE E: CORRELATION MATRIX
# ====================================================================
elif "Matriks" in analysis_type:
    st.header("3. Matriks Korelasi Multi-Variabel")

    sel_vars = st.multiselect("Pilih variabel numerik (min 3)", numeric_cols,
                               default=numeric_cols[:min(6, len(numeric_cols))])
    if len(sel_vars) < 3:
        st.warning("Pilih minimal 3 variabel.")
        st.stop()

    sub_df = df[sel_vars].dropna()
    n_obs = len(sub_df)

    corr_method = st.selectbox("Metode Korelasi", ["Spearman", "Kendall", "Pearson"], index=0)

    if corr_method == "Spearman":
        corr_matrix = sub_df.corr(method='spearman')
        pval_matrix = np.zeros((len(sel_vars), len(sel_vars)))
        for i, j in combinations(range(len(sel_vars)), 2):
            _, p = stats.spearmanr(sub_df.iloc[:, i], sub_df.iloc[:, j])
            pval_matrix[i, j] = p; pval_matrix[j, i] = p
    elif corr_method == "Kendall":
        corr_matrix = sub_df.corr(method='kendall')
        pval_matrix = np.zeros((len(sel_vars), len(sel_vars)))
        for i, j in combinations(range(len(sel_vars)), 2):
            _, p = stats.kendalltau(sub_df.iloc[:, i], sub_df.iloc[:, j])
            pval_matrix[i, j] = p; pval_matrix[j, i] = p
    else:
        corr_matrix = sub_df.corr(method='pearson')
        pval_matrix = np.zeros((len(sel_vars), len(sel_vars)))
        for i, j in combinations(range(len(sel_vars)), 2):
            _, p = stats.pearsonr(sub_df.iloc[:, i], sub_df.iloc[:, j])
            pval_matrix[i, j] = p; pval_matrix[j, i] = p

    st.subheader(f"Matriks Korelasi ({corr_method})")
    st.dataframe(corr_matrix.round(4), use_container_width=True)

    pval_df = pd.DataFrame(pval_matrix, index=sel_vars, columns=sel_vars).round(6)
    with st.expander("Matriks p-value"):
        st.dataframe(pval_df, use_container_width=True)

    # Significance matrix
    sig_matrix = (pval_matrix < alpha).astype(int)
    np.fill_diagonal(sig_matrix, -1)
    sig_labels = np.where(sig_matrix == 1, '★', np.where(sig_matrix == 0, 'ns', '-'))
    sig_df = pd.DataFrame(sig_labels, index=sel_vars, columns=sel_vars)
    with st.expander(f"Matriks Signifikansi (★ = p < {alpha})"):
        st.dataframe(sig_df, use_container_width=True)

    # Heatmap
    st.header("4. Visualisasi")
    tab1, tab2, tab3 = st.tabs(["Heatmap Korelasi", "Heatmap p-value", "Scatter Matrix"])

    with tab1:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_vals = corr_matrix.values.copy()
        corr_vals[mask] = np.nan
        fig = go.Figure(data=go.Heatmap(
            z=corr_vals, x=sel_vars, y=sel_vars, colorscale='RdBu', zmid=0,
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 3), texttemplate="%{text}"))
        fig.update_layout(title=f"Heatmap Korelasi {corr_method}", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure(data=go.Heatmap(
            z=pval_matrix, x=sel_vars, y=sel_vars, colorscale='YlOrRd_r',
            zmin=0, zmax=0.1,
            text=np.round(pval_matrix, 4), texttemplate="%{text}"))
        fig.update_layout(title="Heatmap p-value", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.scatter_matrix(sub_df, dimensions=sel_vars,
                                 title=f"Scatter Matrix ({corr_method})")
        fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.5))
        fig.update_layout(height=200*len(sel_vars))
        st.plotly_chart(fig, use_container_width=True)

# ============================
# SUMMARY
# ============================
st.header("5. Ringkasan Seluruh Uji" if "Matriks" not in analysis_type else "5. Ringkasan")
if len(summary_rows) > 0:
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ============================
# EXPORT
# ============================
st.header("6. Ekspor Hasil")
col_e1, col_e2 = st.columns(2)
with col_e1:
    lines = ["=" * 70, "UJI NONPARAMETRIK - HIPOTESIS ASOSIASI", "=" * 70,
             f"Tipe Analisis  : {analysis_type}",
             f"Alpha          : {alpha}", ""]
    if len(summary_rows) > 0:
        lines.append("=" * 70)
        lines.append("HASIL UJI")
        lines.append("=" * 70)
        for row in summary_rows:
            lines.append(f"  {row['Uji']:35s} | Koef: {str(row['Koefisien']):>12s} | p={str(row['p-value']):<10} | {row['Keputusan']}")
    st.download_button("Download Summary (TXT)", data="\n".join(lines),
                       file_name="nonparam_association_summary.txt", mime="text/plain")
with col_e2:
    st.download_button("Download Data (CSV)", data=df.to_csv(index=False),
                       file_name="nonparam_association_data.csv", mime="text/csv")

# FOOTER
st.markdown("---")
st.markdown("""
**Catatan Metodologis:**

**Korelasi Rank (Kontinu/Ordinal):**
- **Spearman ρ**: korelasi peringkat, robust terhadap outlier dan hubungan nonlinear monoton.
- **Kendall τ**: berbasis concordant/discordant pairs, lebih stabil untuk sampel kecil.
- **Pearson r**: parametrik, disertakan sebagai pembanding.

**Nominal × Nominal:**
- **Chi-Square Independence**: uji standar asosiasi dua variabel nominal.
- **Likelihood Ratio G²**: alternatif Chi-Square, lebih baik untuk sampel kecil.
- **Fisher's Exact**: eksak untuk tabel 2×2.
- **Cramér's V, Tschuprow's T, Pearson's C**: ukuran kekuatan asosiasi.
- **Adjusted Standardized Residuals**: identifikasi sel penyumbang asosiasi.

**Nominal × Kontinu:**
- **Eta Squared (η²)**: proporsi variansi yang dijelaskan.
- **Point-Biserial**: korelasi antara variabel biner dan kontinu.
- **Kruskal-Wallis**: uji perbedaan distribusi sebagai proksi asosiasi.

**Ordinal × Ordinal:**
- **Goodman-Kruskal Gamma (γ)**: berbasis concordant/discordant pairs tanpa koreksi tie.
- **Somer's D**: asimetrik, membedakan variabel dependen vs independen.
- **Kendall τ-b**: dengan koreksi tie.
""")
st.markdown("Dibangun dengan **Streamlit** + **SciPy** + **Plotly** | Python")
