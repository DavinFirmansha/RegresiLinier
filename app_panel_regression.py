import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Regresi Data Panel", page_icon="📐", layout="wide")
st.title("📐 Regresi Data Panel Parametrik")
st.markdown("""
Aplikasi lengkap untuk **regresi data panel** dengan tiga pendekatan:
**Common Effect Model (CEM / Pooled OLS)**, **Fixed Effect Model (FEM)**, dan **Random Effect Model (REM)**.
Dilengkapi uji pemilihan model (Chow, Hausman, Breusch-Pagan LM), uji asumsi klasik, dan visualisasi.
""")

# ============================
# HELPERS
# ============================
def _arr(obj):
    return np.asarray(obj).flatten()

def within_transform(data, entity_col, cols):
    """Demean (within transformation) for FEM."""
    means = data.groupby(entity_col)[cols].transform('mean')
    return data[cols] - means

def between_transform(data, entity_col, cols):
    """Between transformation."""
    return data.groupby(entity_col)[cols].mean()

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
    entities = [f'Firm_{i+1}' for i in range(10)]
    periods = list(range(2015, 2025))
    rows = []
    for e_idx, e in enumerate(entities):
        alpha_i = np.random.normal(0, 5)
        for t in periods:
            x1 = np.random.normal(50 + e_idx * 2, 10)
            x2 = np.random.normal(30, 8)
            x3 = np.random.normal(20 + (t - 2015) * 0.5, 5)
            y = 10 + 0.5 * x1 + 0.3 * x2 - 0.2 * x3 + alpha_i + np.random.normal(0, 4)
            rows.append({'Entity': e, 'Time': t, 'Y': round(y, 2),
                         'X1': round(x1, 2), 'X2': round(x2, 2), 'X3': round(x3, 2)})
    df = pd.DataFrame(rows)
    st.sidebar.success("Data demo: 10 entitas × 10 periode = 100 obs")
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

tab_d1, tab_d2 = st.tabs(["Data", "Statistik Deskriptif"])
with tab_d1:
    st.dataframe(df.head(30), use_container_width=True)
with tab_d2:
    st.dataframe(df.describe(include='all').T.round(4), use_container_width=True)

# ============================
# 2. SETUP PANEL
# ============================
st.header("2. Struktur Data Panel")
all_cols = df.columns.tolist()

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    entity_col = st.selectbox("Variabel Entitas (cross-section, i)", all_cols, index=0)
with col_s2:
    time_col = st.selectbox("Variabel Waktu (time-series, t)",
                             [c for c in all_cols if c != entity_col], index=0)
with col_s3:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    y_options = [c for c in numeric_cols if c != entity_col and c != time_col]
    dep_var = st.selectbox("Variabel Dependen (Y)", y_options, index=0)

indep_options = [c for c in numeric_cols if c != dep_var and c != entity_col and c != time_col]
indep_vars = st.multiselect("Variabel Independen (X₁, X₂, ...)", indep_options, default=indep_options)

if len(indep_vars) < 1:
    st.warning("Pilih minimal 1 variabel independen.")
    st.stop()

# Panel info
n_entities = df[entity_col].nunique()
n_periods = df[time_col].nunique()
N = len(df)
alpha = st.selectbox("Tingkat Signifikansi (α)", [0.01, 0.05, 0.10], index=1)

st.info(f"**Panel:** {n_entities} entitas × {n_periods} periode = {N} observasi | "
        f"**Y:** {dep_var} | **X:** {', '.join(indep_vars)} | α = {alpha}")

# Check balanced
counts = df.groupby(entity_col)[time_col].count()
is_balanced = counts.nunique() == 1
st.markdown(f"**Tipe Panel:** {'Balanced ✅' if is_balanced else 'Unbalanced ⚠️'} "
            f"(min {counts.min()}, max {counts.max()} periode per entitas)")

# Prepare data
panel = df[[entity_col, time_col, dep_var] + indep_vars].dropna().copy()
panel = panel.sort_values([entity_col, time_col]).reset_index(drop=True)
y_data = panel[dep_var].values.astype(float)
X_data = panel[indep_vars].values.astype(float)
X_const = sm.add_constant(X_data)
var_names = ['const'] + indep_vars

# ============================
# 3. VISUALISASI PANEL
# ============================
st.header("3. Visualisasi Data Panel")
tab_v1, tab_v2, tab_v3, tab_v4 = st.tabs([
    "Time Series per Entitas", "Scatter Y vs X", "Box Plot per Entitas", "Heatmap Panel"])

with tab_v1:
    fig = px.line(panel, x=time_col, y=dep_var, color=entity_col,
                   title=f"{dep_var} per Entitas sepanjang Waktu", markers=True)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab_v2:
    sel_x_plot = st.selectbox("Pilih X untuk scatter", indep_vars, index=0, key='scatter_x')
    fig = px.scatter(panel, x=sel_x_plot, y=dep_var, color=entity_col,
                      trendline="ols", title=f"Scatter: {dep_var} vs {sel_x_plot}")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab_v3:
    fig = px.box(panel, x=entity_col, y=dep_var, color=entity_col, points='all',
                  title=f"Box Plot {dep_var} per Entitas")
    st.plotly_chart(fig, use_container_width=True)

with tab_v4:
    pivot = panel.pivot_table(index=entity_col, columns=time_col, values=dep_var, aggfunc='mean')
    fig = go.Figure(data=go.Heatmap(z=pivot.values,
                                     x=[str(c) for c in pivot.columns],
                                     y=[str(r) for r in pivot.index],
                                     colorscale='Viridis',
                                     text=np.round(pivot.values, 2), texttemplate="%{text}"))
    fig.update_layout(title=f"Heatmap {dep_var}: Entitas × Waktu", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ============================
# 4. COMMON EFFECT MODEL (CEM / Pooled OLS)
# ============================
st.header("4. Common Effect Model (CEM / Pooled OLS)")
st.markdown("Mengabaikan dimensi individu dan waktu — semua data diregresikan dengan OLS biasa.")

cem = sm.OLS(y_data, X_const).fit()
cem_params = _arr(cem.params)
cem_bse = _arr(cem.bse)
cem_tvals = _arr(cem.tvalues)
cem_pvals = _arr(cem.pvalues)
cem_ci = np.asarray(cem.conf_int())

cem_coef = pd.DataFrame({
    'Variabel': var_names,
    'Koefisien': cem_params.round(6),
    'Std. Error': cem_bse.round(6),
    't-statistic': cem_tvals.round(4),
    'p-value': cem_pvals.round(6),
    f'CI Lower ({(1-alpha)*100:.0f}%)': cem_ci[:, 0].round(6),
    f'CI Upper ({(1-alpha)*100:.0f}%)': cem_ci[:, 1].round(6),
    f'Signifikan (α={alpha})': ['Ya' if p < alpha else 'Tidak' for p in cem_pvals]
})
st.dataframe(cem_coef, use_container_width=True, hide_index=True)

cem_metrics = pd.DataFrame({
    'Metrik': ['R²', 'Adjusted R²', 'F-statistic', 'F p-value', 'AIC', 'BIC',
               'SSR (Residual)', 'SST (Total)', 'MSE', 'Log-Likelihood', 'Durbin-Watson'],
    'Nilai': [round(cem.rsquared, 6), round(cem.rsquared_adj, 6),
              round(float(cem.fvalue), 4), round(float(cem.f_pvalue), 6),
              round(float(cem.aic), 4), round(float(cem.bic), 4),
              round(float(cem.ssr), 4), round(float(cem.centered_tss), 4),
              round(float(cem.mse_resid), 6), round(float(cem.llf), 4),
              round(float(sm.stats.stattools.durbin_watson(cem.resid)), 4)]
})
with st.expander("Goodness of Fit — CEM"):
    st.dataframe(cem_metrics, use_container_width=True, hide_index=True)

# ============================
# 5. FIXED EFFECT MODEL (FEM)
# ============================
st.header("5. Fixed Effect Model (FEM)")
st.markdown("Mengendalikan heterogenitas individu melalui **within transformation** (demeaning).")

# Within transformation
y_within = within_transform(panel, entity_col, [dep_var])[dep_var].values
X_within = within_transform(panel, entity_col, indep_vars).values

# FEM without constant (within removes it)
fem = sm.OLS(y_within, X_within).fit()
fem_params = _arr(fem.params)
fem_bse = _arr(fem.bse)
fem_tvals = _arr(fem.tvalues)
fem_pvals = _arr(fem.pvalues)

# Correct FEM df: N - n - k
k = len(indep_vars)
df_fem_resid = N - n_entities - k
fem_ssr = float(np.sum(fem.resid**2))
fem_mse = fem_ssr / df_fem_resid if df_fem_resid > 0 else fem_ssr

# Corrected SE
fem_se_corr = np.sqrt(np.diag(fem_mse * np.linalg.inv(X_within.T @ X_within + np.eye(k)*1e-12)))
fem_t_corr = fem_params / fem_se_corr
fem_p_corr = 2 * (1 - stats.t.cdf(np.abs(fem_t_corr), df_fem_resid))

# R-squared within
ss_total_within = np.sum(y_within**2)
fem_r2_within = 1 - fem_ssr / ss_total_within if ss_total_within > 0 else 0

fem_coef = pd.DataFrame({
    'Variabel': indep_vars,
    'Koefisien': fem_params.round(6),
    'Std. Error': fem_se_corr.round(6),
    't-statistic': fem_t_corr.round(4),
    'p-value': fem_p_corr.round(6),
    f'Signifikan (α={alpha})': ['Ya' if p < alpha else 'Tidak' for p in fem_p_corr]
})
st.dataframe(fem_coef, use_container_width=True, hide_index=True)

# Entity fixed effects
entity_means_y = panel.groupby(entity_col)[dep_var].mean()
entity_means_X = panel.groupby(entity_col)[indep_vars].mean()
grand_mean_y = y_data.mean()
grand_mean_X = X_data.mean(axis=0)
entity_fe = entity_means_y.values - entity_means_X.values @ fem_params
entity_fe_df = pd.DataFrame({
    'Entitas': entity_means_y.index.tolist(),
    'Fixed Effect (αᵢ)': entity_fe.round(4)
}).sort_values('Fixed Effect (αᵢ)', ascending=False)

with st.expander("Fixed Effects per Entitas (αᵢ)"):
    st.dataframe(entity_fe_df, use_container_width=True, hide_index=True)
    fig_fe = px.bar(entity_fe_df, x='Entitas', y='Fixed Effect (αᵢ)',
                     color='Fixed Effect (αᵢ)', color_continuous_scale='RdBu',
                     title="Fixed Effects per Entitas")
    st.plotly_chart(fig_fe, use_container_width=True)

fem_metrics = pd.DataFrame({
    'Metrik': ['R² (within)', 'SSR', 'MSE (corrected)', 'df Residual',
               'N entities', 'N periods', 'N total'],
    'Nilai': [round(fem_r2_within, 6), round(fem_ssr, 4), round(fem_mse, 6),
              df_fem_resid, n_entities, n_periods, N]
})
with st.expander("Goodness of Fit — FEM"):
    st.dataframe(fem_metrics, use_container_width=True, hide_index=True)

# ============================
# 6. RANDOM EFFECT MODEL (REM)
# ============================
st.header("6. Random Effect Model (REM)")
st.markdown("Mengasumsikan heterogenitas individu sebagai **komponen acak** — estimasi GLS (Feasible GLS).")

# Estimate variance components for GLS
# sigma_u^2 from between estimator
X_between = sm.add_constant(between_transform(panel, entity_col, indep_vars).values)
y_between = between_transform(panel, entity_col, [dep_var]).values.flatten()
between_ols = sm.OLS(y_between, X_between).fit()
sigma_between2 = float(np.sum(between_ols.resid**2)) / max(n_entities - k - 1, 1)

# sigma_e^2 from within (FEM)
sigma_e2 = fem_mse

# sigma_u^2
T_bar = N / n_entities
sigma_u2 = max(sigma_between2 - sigma_e2 / T_bar, 0.001)

# theta for FGLS
theta = 1 - np.sqrt(sigma_e2 / (T_bar * sigma_u2 + sigma_e2))

# Quasi-demean
y_rem = np.zeros(N)
X_rem = np.zeros((N, len(indep_vars) + 1))
idx = 0
for entity in panel[entity_col].unique():
    mask = panel[entity_col] == entity
    n_i = mask.sum()
    yi = y_data[mask]
    Xi = X_const[mask]
    yi_bar = yi.mean()
    Xi_bar = Xi.mean(axis=0)
    y_rem[idx:idx+n_i] = yi - theta * yi_bar
    X_rem[idx:idx+n_i] = Xi - theta * Xi_bar
    idx += n_i

rem = sm.OLS(y_rem, X_rem).fit()
rem_params = _arr(rem.params)
rem_bse = _arr(rem.bse)
rem_tvals = _arr(rem.tvalues)
rem_pvals = _arr(rem.pvalues)
rem_ci = np.asarray(rem.conf_int())

rem_coef = pd.DataFrame({
    'Variabel': var_names,
    'Koefisien': rem_params.round(6),
    'Std. Error': rem_bse.round(6),
    't-statistic': rem_tvals.round(4),
    'p-value': rem_pvals.round(6),
    f'CI Lower ({(1-alpha)*100:.0f}%)': rem_ci[:, 0].round(6),
    f'CI Upper ({(1-alpha)*100:.0f}%)': rem_ci[:, 1].round(6),
    f'Signifikan (α={alpha})': ['Ya' if p < alpha else 'Tidak' for p in rem_pvals]
})
st.dataframe(rem_coef, use_container_width=True, hide_index=True)

rem_metrics = pd.DataFrame({
    'Metrik': ['R² (GLS)', 'σ²ₑ (idiosyncratic)', 'σ²ᵤ (individual)',
               'θ (quasi-demean factor)', 'ρ (fraction of variance due to uᵢ)'],
    'Nilai': [round(rem.rsquared, 6), round(sigma_e2, 6), round(sigma_u2, 6),
              round(theta, 6), round(sigma_u2 / (sigma_u2 + sigma_e2), 6)]
})
with st.expander("Variance Components & Fit — REM"):
    st.dataframe(rem_metrics, use_container_width=True, hide_index=True)

# ============================
# 7. PERBANDINGAN MODEL
# ============================
st.header("7. Perbandingan Koefisien CEM vs FEM vs REM")

comp_rows = []
for i, v in enumerate(indep_vars):
    j = i + 1  # skip const for CEM/REM
    comp_rows.append({
        'Variabel': v,
        'CEM (β)': round(cem_params[j], 6),
        'CEM (p)': round(cem_pvals[j], 6),
        'FEM (β)': round(fem_params[i], 6),
        'FEM (p)': round(fem_p_corr[i], 6),
        'REM (β)': round(rem_params[j], 6),
        'REM (p)': round(rem_pvals[j], 6),
    })
comp_df = pd.DataFrame(comp_rows)
st.dataframe(comp_df, use_container_width=True, hide_index=True)

# Visual comparison
fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(x=indep_vars, y=cem_params[1:], name='CEM'))
fig_comp.add_trace(go.Bar(x=indep_vars, y=fem_params, name='FEM'))
fig_comp.add_trace(go.Bar(x=indep_vars, y=rem_params[1:], name='REM'))
fig_comp.update_layout(title="Perbandingan Koefisien Regresi", barmode='group',
                        yaxis_title="Koefisien", height=450)
st.plotly_chart(fig_comp, use_container_width=True)

# ============================
# 8. UJI PEMILIHAN MODEL
# ============================
st.header("8. Uji Pemilihan Model")

# ---- 8a. CHOW TEST (CEM vs FEM) ----
st.subheader("8a. Uji Chow (CEM vs FEM)")
st.markdown("**H₀:** CEM lebih baik | **H₁:** FEM lebih baik")

ssr_cem = float(cem.ssr)
ssr_fem = fem_ssr
df1_chow = n_entities - 1
df2_chow = N - n_entities - k
f_chow = ((ssr_cem - ssr_fem) / df1_chow) / (ssr_fem / df2_chow) if df2_chow > 0 else 0
p_chow = 1 - stats.f.cdf(f_chow, df1_chow, df2_chow)

chow_df = pd.DataFrame({
    'Metrik': ['SSR CEM', 'SSR FEM', 'F-statistic', 'df1 (n-1)', 'df2 (N-n-k)',
               'p-value', f'Keputusan (α={alpha})'],
    'Nilai': [round(ssr_cem, 4), round(ssr_fem, 4), round(f_chow, 4),
              df1_chow, df2_chow, round(p_chow, 6),
              'FEM lebih baik (Tolak H₀)' if p_chow < alpha else 'CEM lebih baik (Gagal Tolak H₀)']
})
st.dataframe(chow_df, use_container_width=True, hide_index=True)
chow_decision = 'FEM' if p_chow < alpha else 'CEM'

# ---- 8b. HAUSMAN TEST (FEM vs REM) ----
st.subheader("8b. Uji Hausman (FEM vs REM)")
st.markdown("**H₀:** REM konsisten (REM lebih baik) | **H₁:** FEM lebih baik")

b_fem = fem_params
b_rem = rem_params[1:]  # exclude constant
var_fem = fem_mse * np.linalg.inv(X_within.T @ X_within + np.eye(k) * 1e-12)
var_rem_sub = np.asarray(rem.cov_params())[1:, 1:]

diff_b = b_fem - b_rem
diff_var = var_fem - var_rem_sub

try:
    hausman_stat = float(diff_b @ np.linalg.inv(diff_var) @ diff_b)
    hausman_stat = max(hausman_stat, 0)
except:
    try:
        hausman_stat = float(diff_b @ np.linalg.pinv(diff_var) @ diff_b)
        hausman_stat = max(hausman_stat, 0)
    except:
        hausman_stat = 0

hausman_p = 1 - stats.chi2.cdf(hausman_stat, k)

hausman_df = pd.DataFrame({
    'Metrik': ['Hausman χ²', 'df', 'p-value', f'Keputusan (α={alpha})'],
    'Nilai': [round(hausman_stat, 4), k, round(hausman_p, 6),
              'FEM lebih baik (Tolak H₀)' if hausman_p < alpha else 'REM lebih baik (Gagal Tolak H₀)']
})
st.dataframe(hausman_df, use_container_width=True, hide_index=True)
hausman_decision = 'FEM' if hausman_p < alpha else 'REM'

# ---- 8c. BREUSCH-PAGAN LM (CEM vs REM) ----
st.subheader("8c. Uji Breusch-Pagan Lagrange Multiplier (CEM vs REM)")
st.markdown("**H₀:** CEM lebih baik | **H₁:** REM lebih baik")

resid_cem = cem.resid
sum_ei_bar2 = 0
sum_ei2 = float(np.sum(resid_cem**2))
for entity in panel[entity_col].unique():
    mask = (panel[entity_col] == entity).values
    sum_ei_bar2 += (np.sum(resid_cem[mask]))**2

T_vals = panel.groupby(entity_col).size().values
nT = N
bp_lm = (nT**2 / (2 * np.sum(T_vals * (T_vals - 1)))) * (sum_ei_bar2 / sum_ei2 - 1)**2 if sum_ei2 > 0 else 0
bp_lm = max(bp_lm, 0)
bp_p = 1 - stats.chi2.cdf(bp_lm, 1)

bp_df = pd.DataFrame({
    'Metrik': ['LM statistic', 'df', 'p-value', f'Keputusan (α={alpha})'],
    'Nilai': [round(bp_lm, 4), 1, round(bp_p, 6),
              'REM lebih baik (Tolak H₀)' if bp_p < alpha else 'CEM lebih baik (Gagal Tolak H₀)']
})
st.dataframe(bp_df, use_container_width=True, hide_index=True)
bp_decision = 'REM' if bp_p < alpha else 'CEM'

# ---- DECISION FLOW ----
st.subheader("Alur Keputusan Pemilihan Model")

decisions = []
decisions.append(f"1️⃣ **Uji Chow** (CEM vs FEM): p = {p_chow:.6f} → **{chow_decision}**")
if chow_decision == 'FEM':
    decisions.append(f"2️⃣ **Uji Hausman** (FEM vs REM): p = {hausman_p:.6f} → **{hausman_decision}**")
    best_model = hausman_decision
else:
    decisions.append(f"2️⃣ **Uji BP-LM** (CEM vs REM): p = {bp_p:.6f} → **{bp_decision}**")
    best_model = bp_decision

for d in decisions:
    st.markdown(d)

st.success(f"### ✅ Model Terbaik: **{best_model}** ({'Common Effect' if best_model=='CEM' else 'Fixed Effect' if best_model=='FEM' else 'Random Effect'})")

# ============================
# 9. HASIL MODEL TERBAIK
# ============================
st.header(f"9. Hasil Model Terbaik: {best_model}")
if best_model == 'CEM':
    best_result = cem
    best_coef = cem_coef
    best_resid = cem.resid
    best_fitted = cem.fittedvalues
elif best_model == 'FEM':
    best_coef = fem_coef
    best_resid = fem.resid
    best_fitted = fem.fittedvalues
else:
    best_result = rem
    best_coef = rem_coef
    best_resid = rem.resid
    best_fitted = rem.fittedvalues

st.subheader("Tabel Koefisien")
st.dataframe(best_coef, use_container_width=True, hide_index=True)

# ============================
# 10. UJI ASUMSI KLASIK
# ============================
st.header("10. Uji Asumsi Klasik")

resid = _arr(best_resid)

# ---- 10a. NORMALITAS ----
st.subheader("10a. Normalitas Residual")
sw_stat, sw_p = stats.shapiro(resid[:min(5000, len(resid))])
jb_stat, jb_p = stats.jarque_bera(resid)

norm_result = pd.DataFrame({
    'Uji': ['Shapiro-Wilk', 'Jarque-Bera'],
    'Statistik': [round(sw_stat, 6), round(float(jb_stat), 6)],
    'p-value': [round(sw_p, 6), round(float(jb_p), 6)],
    f'Keputusan (α={alpha})': [
        'Normal' if sw_p > alpha else 'Tidak Normal',
        'Normal' if jb_p > alpha else 'Tidak Normal']
})
st.dataframe(norm_result, use_container_width=True, hide_index=True)

fig_norm = make_subplots(rows=1, cols=2, subplot_titles=("Histogram Residual", "QQ-Plot"))
fig_norm.add_trace(go.Histogram(x=resid, nbinsx=30, marker_color='steelblue',
                                 showlegend=False), row=1, col=1)
qq = stats.probplot(resid, dist="norm")
fig_norm.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                               marker=dict(size=4), showlegend=False), row=1, col=2)
fig_norm.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0]*qq[0][0]+qq[1][1],
                               mode='lines', line=dict(color='red'), showlegend=False), row=1, col=2)
fig_norm.update_layout(height=400)
st.plotly_chart(fig_norm, use_container_width=True)

# ---- 10b. HETEROSKEDASTISITAS ----
st.subheader("10b. Heteroskedastisitas")

if best_model != 'FEM':
    X_het = X_const
else:
    X_het = sm.add_constant(X_within)

try:
    bp_test = sm.stats.diagnostic.het_breuschpagan(resid, X_het)
    white_test = sm.stats.diagnostic.het_white(resid, X_het)
    het_result = pd.DataFrame({
        'Uji': ['Breusch-Pagan', 'White'],
        'Statistik': [round(bp_test[0], 4), round(white_test[0], 4)],
        'p-value': [round(bp_test[1], 6), round(white_test[1], 6)],
        f'Keputusan (α={alpha})': [
            'Homoskedastik' if bp_test[1] > alpha else 'Heteroskedastik',
            'Homoskedastik' if white_test[1] > alpha else 'Heteroskedastik']
    })
    st.dataframe(het_result, use_container_width=True, hide_index=True)
except Exception as e:
    try:
        bp_test = sm.stats.diagnostic.het_breuschpagan(resid, X_het)
        het_result = pd.DataFrame({
            'Uji': ['Breusch-Pagan'],
            'Statistik': [round(bp_test[0], 4)],
            'p-value': [round(bp_test[1], 6)],
            f'Keputusan (α={alpha})': [
                'Homoskedastik' if bp_test[1] > alpha else 'Heteroskedastik']
        })
        st.dataframe(het_result, use_container_width=True, hide_index=True)
    except:
        st.warning(f"Uji heteroskedastisitas gagal: {e}")

# Residual vs Fitted
fitted = _arr(best_fitted)
fig_het = px.scatter(x=fitted, y=resid, labels={'x': 'Fitted Values', 'y': 'Residuals'},
                      title="Residual vs Fitted Values")
fig_het.add_hline(y=0, line_dash="dash", line_color="red")
st.plotly_chart(fig_het, use_container_width=True)

# ---- 10c. AUTOKORELASI ----
st.subheader("10c. Autokorelasi")
dw = sm.stats.stattools.durbin_watson(resid)

try:
    bg_test = sm.stats.diagnostic.acorr_breusch_godfrey(
        sm.OLS(y_data, X_const).fit() if best_model != 'FEM'
        else sm.OLS(y_within, X_within).fit(), nlags=1)
    bg_stat, bg_p = bg_test[0], bg_test[1]
except:
    bg_stat, bg_p = np.nan, np.nan

auto_result = pd.DataFrame({
    'Uji': ['Durbin-Watson', 'Breusch-Godfrey (lag=1)'],
    'Statistik': [round(float(dw), 4), round(float(bg_stat), 4) if not np.isnan(bg_stat) else np.nan],
    'p-value / Keterangan': [
        f'{"Tidak ada autokorelasi" if 1.5 < dw < 2.5 else "Indikasi autokorelasi"}',
        round(float(bg_p), 6) if not np.isnan(bg_p) else 'N/A'],
    'Keputusan': [
        'Tidak ada autokorelasi' if 1.5 < dw < 2.5 else 'Ada indikasi autokorelasi',
        ('Tidak ada autokorelasi' if bg_p > alpha else 'Ada autokorelasi') if not np.isnan(bg_p) else 'N/A']
})
st.dataframe(auto_result, use_container_width=True, hide_index=True)

# ACF Plot
from statsmodels.tsa.stattools import acf
acf_vals = acf(resid, nlags=min(20, len(resid)//3), fft=True)
fig_acf = go.Figure()
fig_acf.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, marker_color='steelblue'))
ci_bound = 1.96 / np.sqrt(len(resid))
fig_acf.add_hline(y=ci_bound, line_dash="dash", line_color="red")
fig_acf.add_hline(y=-ci_bound, line_dash="dash", line_color="red")
fig_acf.update_layout(title="ACF Residual", xaxis_title="Lag", yaxis_title="ACF", height=350)
st.plotly_chart(fig_acf, use_container_width=True)

# ---- 10d. MULTIKOLINIERITAS ----
st.subheader("10d. Multikolinieritas (VIF)")
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
X_vif = panel[indep_vars].values.astype(float)
X_vif_const = sm.add_constant(X_vif)
vif_data['Variabel'] = indep_vars
vif_data['VIF'] = [round(variance_inflation_factor(X_vif_const, i+1), 4) for i in range(len(indep_vars))]
vif_data['Keterangan'] = vif_data['VIF'].apply(
    lambda v: 'Tidak ada multikolinieritas' if v < 5 else ('Moderat' if v < 10 else 'Tinggi — masalah!'))

st.dataframe(vif_data, use_container_width=True, hide_index=True)

# Correlation matrix
corr_X = panel[indep_vars].corr()
fig_corr = go.Figure(data=go.Heatmap(z=corr_X.values, x=indep_vars, y=indep_vars,
                                      colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
                                      text=np.round(corr_X.values, 3), texttemplate="%{text}"))
fig_corr.update_layout(title="Matriks Korelasi antar Variabel Independen", height=400)
st.plotly_chart(fig_corr, use_container_width=True)

# ============================
# 11. DIAGNOSTIK LANJUT
# ============================
st.header("11. Diagnostik Lanjut")

# Residual per entity
st.subheader("11a. Residual per Entitas")
panel_diag = panel.copy()
panel_diag['Residual'] = resid[:len(panel_diag)]
fig_re = px.box(panel_diag, x=entity_col, y='Residual', color=entity_col,
                 title="Box Plot Residual per Entitas")
st.plotly_chart(fig_re, use_container_width=True)

# Residual over time
st.subheader("11b. Residual sepanjang Waktu")
fig_rt = px.scatter(panel_diag, x=time_col, y='Residual', color=entity_col,
                     title="Residual vs Waktu", opacity=0.6)
fig_rt.add_hline(y=0, line_dash="dash", line_color="red")
st.plotly_chart(fig_rt, use_container_width=True)

# Actual vs Fitted
st.subheader("11c. Actual vs Fitted")
fig_af = go.Figure()
fig_af.add_trace(go.Scatter(x=fitted, y=y_data[:len(fitted)] if best_model != 'FEM' else y_within[:len(fitted)],
                              mode='markers', marker=dict(size=5, opacity=0.6), name='Data'))
min_v = min(fitted.min(), (y_data if best_model != 'FEM' else y_within).min())
max_v = max(fitted.max(), (y_data if best_model != 'FEM' else y_within).max())
fig_af.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines',
                              line=dict(color='red', dash='dash'), name='Perfect Fit'))
fig_af.update_layout(title="Actual vs Fitted Values", xaxis_title="Fitted",
                      yaxis_title="Actual", height=450)
st.plotly_chart(fig_af, use_container_width=True)

# ============================
# 12. RINGKASAN
# ============================
st.header("12. Ringkasan Pemilihan Model")

summary_model = pd.DataFrame({
    'Uji': ['Chow Test (CEM vs FEM)', 'Hausman Test (FEM vs REM)',
            'BP-LM Test (CEM vs REM)'],
    'Statistik': [f'F={f_chow:.4f}', f'χ²={hausman_stat:.4f}', f'LM={bp_lm:.4f}'],
    'p-value': [round(p_chow, 6), round(hausman_p, 6), round(bp_p, 6)],
    'Keputusan': [chow_decision, hausman_decision, bp_decision]
})
st.dataframe(summary_model, use_container_width=True, hide_index=True)
st.success(f"**Model Terpilih: {best_model}** ({'Common Effect' if best_model=='CEM' else 'Fixed Effect' if best_model=='FEM' else 'Random Effect'})")

# ============================
# 13. EXPORT
# ============================
st.header("13. Ekspor Hasil")
col_e1, col_e2, col_e3 = st.columns(3)

with col_e1:
    lines = [
        "=" * 70, "REGRESI DATA PANEL PARAMETRIK", "=" * 70,
        f"Entitas        : {entity_col} ({n_entities})",
        f"Waktu          : {time_col} ({n_periods})",
        f"N total        : {N}",
        f"Panel          : {'Balanced' if is_balanced else 'Unbalanced'}",
        f"Dependen (Y)   : {dep_var}",
        f"Independen (X) : {', '.join(indep_vars)}",
        f"Alpha          : {alpha}",
        "", "=" * 70, "PEMILIHAN MODEL", "=" * 70,
        f"Chow Test     : F={f_chow:.4f}, p={p_chow:.6f} → {chow_decision}",
        f"Hausman Test  : χ²={hausman_stat:.4f}, p={hausman_p:.6f} → {hausman_decision}",
        f"BP-LM Test    : LM={bp_lm:.4f}, p={bp_p:.6f} → {bp_decision}",
        f"Model Terpilih: {best_model}",
        "", "=" * 70, f"KOEFISIEN MODEL TERBAIK ({best_model})", "=" * 70]
    for _, row in best_coef.iterrows():
        lines.append(f"  {row['Variabel']:15s} | β={row['Koefisien']:>12.6f} | SE={row['Std. Error']:>10.6f} | "
                     f"t={row['t-statistic']:>8.4f} | p={row['p-value']:>10.6f}")
    st.download_button("📥 Download Summary (TXT)", data="\n".join(lines),
                       file_name="panel_regression_summary.txt", mime="text/plain")

with col_e2:
    panel_export = panel.copy()
    panel_export['Residual'] = resid[:len(panel)]
    panel_export['Fitted'] = fitted[:len(panel)]
    st.download_button("📥 Download Data + Residual (CSV)",
                       data=panel_export.to_csv(index=False),
                       file_name="panel_regression_data.csv", mime="text/csv")

with col_e3:
    all_coef = pd.DataFrame({
        'Variabel': indep_vars,
        'CEM_beta': cem_params[1:].round(6),
        'CEM_pval': cem_pvals[1:].round(6),
        'FEM_beta': fem_params.round(6),
        'FEM_pval': fem_p_corr.round(6),
        'REM_beta': rem_params[1:].round(6),
        'REM_pval': rem_pvals[1:].round(6),
    })
    st.download_button("📥 Download Semua Koefisien (CSV)",
                       data=all_coef.to_csv(index=False),
                       file_name="panel_all_coefficients.csv", mime="text/csv")

# FOOTER
st.markdown("---")
st.markdown("""
**Catatan Metodologis:**

- **CEM (Common Effect / Pooled OLS):** Mengabaikan heterogenitas individu. Estimasi OLS biasa.
- **FEM (Fixed Effect):** Mengendalikan heterogenitas individu via *within transformation* (demeaning). Efek individu diperlakukan sebagai parameter tetap.
- **REM (Random Effect):** Efek individu diperlakukan sebagai komponen acak. Estimasi Feasible GLS (quasi-demeaning dengan θ).

**Uji Pemilihan Model:**
- **Chow Test:** CEM vs FEM (F-test pada restricsi fixed effects).
- **Hausman Test:** FEM vs REM (konsistensi estimator RE).
- **Breusch-Pagan LM:** CEM vs REM (signifikansi komponen variansi individu).

**Uji Asumsi Klasik:**
- Normalitas (Shapiro-Wilk, Jarque-Bera), Heteroskedastisitas (Breusch-Pagan, White),
  Autokorelasi (Durbin-Watson, Breusch-Godfrey), Multikolinieritas (VIF).
""")
st.markdown("Dibangun dengan **Streamlit** + **Statsmodels** + **SciPy** + **Plotly** | Python")
