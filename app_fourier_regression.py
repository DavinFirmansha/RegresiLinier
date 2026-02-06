import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Regresi Nonparametrik Deret Fourier", page_icon="🌊", layout="wide")
st.title("🌊 Regresi Nonparametrik — Deret Fourier")
st.markdown("""
Aplikasi lengkap untuk analisis **regresi nonparametrik** menggunakan estimator **Deret Fourier**, 
dilengkapi berbagai variasi model, pemilihan osilasi optimal via GCV, uji asumsi, confidence band, dan diagnostik model.
""")

# ============================
# MODEL DEFINITIONS
# ============================
MODEL_TYPES = {
    "Cosine + Trend": {
        "formula": "g(x) = β₀/2 + β₁x + Σ αₖ cos(kx)",
        "description": "Model standar Fourier: tren linier + deret cosinus (Bilodeau, 1994)",
        "build": lambda x, K: _build_cos_trend(x, K)
    },
    "Sine + Trend": {
        "formula": "g(x) = β₀/2 + β₁x + Σ γₖ sin(kx)",
        "description": "Tren linier + deret sinus",
        "build": lambda x, K: _build_sin_trend(x, K)
    },
    "Sine + Cosine + Trend": {
        "formula": "g(x) = β₀/2 + β₁x + Σ [αₖ cos(kx) + γₖ sin(kx)]",
        "description": "Model Fourier lengkap: tren linier + deret sinus dan cosinus",
        "build": lambda x, K: _build_sincos_trend(x, K)
    },
    "Cosine Only": {
        "formula": "g(x) = β₀/2 + Σ αₖ cos(kx)",
        "description": "Hanya deret cosinus tanpa tren linier",
        "build": lambda x, K: _build_cos_only(x, K)
    },
    "Sine Only": {
        "formula": "g(x) = β₀/2 + Σ γₖ sin(kx)",
        "description": "Hanya deret sinus tanpa tren linier",
        "build": lambda x, K: _build_sin_only(x, K)
    },
    "Sine + Cosine (No Trend)": {
        "formula": "g(x) = β₀/2 + Σ [αₖ cos(kx) + γₖ sin(kx)]",
        "description": "Deret sinus dan cosinus tanpa tren linier",
        "build": lambda x, K: _build_sincos_only(x, K)
    },
    "Cosine + Quadratic Trend": {
        "formula": "g(x) = β₀/2 + β₁x + β₂x² + Σ αₖ cos(kx)",
        "description": "Tren kuadratik + deret cosinus",
        "build": lambda x, K: _build_cos_qtrend(x, K)
    },
    "Sine + Cosine + Quadratic Trend": {
        "formula": "g(x) = β₀/2 + β₁x + β₂x² + Σ [αₖ cos(kx) + γₖ sin(kx)]",
        "description": "Model terlengkap: tren kuadratik + deret sinus dan cosinus",
        "build": lambda x, K: _build_sincos_qtrend(x, K)
    },
}

# ============================
# BASIS BUILDERS
# ============================
def _build_cos_trend(x, K):
    cols = {'trend_x': x}
    names = ['trend_x']
    for k in range(1, K + 1):
        n = f'cos({k}x)'; cols[n] = np.cos(k * x); names.append(n)
    return pd.DataFrame(cols), names

def _build_sin_trend(x, K):
    cols = {'trend_x': x}
    names = ['trend_x']
    for k in range(1, K + 1):
        n = f'sin({k}x)'; cols[n] = np.sin(k * x); names.append(n)
    return pd.DataFrame(cols), names

def _build_sincos_trend(x, K):
    cols = {'trend_x': x}
    names = ['trend_x']
    for k in range(1, K + 1):
        cn, sn = f'cos({k}x)', f'sin({k}x)'
        cols[cn] = np.cos(k * x); cols[sn] = np.sin(k * x)
        names.extend([cn, sn])
    return pd.DataFrame(cols), names

def _build_cos_only(x, K):
    cols, names = {}, []
    for k in range(1, K + 1):
        n = f'cos({k}x)'; cols[n] = np.cos(k * x); names.append(n)
    return pd.DataFrame(cols), names

def _build_sin_only(x, K):
    cols, names = {}, []
    for k in range(1, K + 1):
        n = f'sin({k}x)'; cols[n] = np.sin(k * x); names.append(n)
    return pd.DataFrame(cols), names

def _build_sincos_only(x, K):
    cols, names = {}, []
    for k in range(1, K + 1):
        cn, sn = f'cos({k}x)', f'sin({k}x)'
        cols[cn] = np.cos(k * x); cols[sn] = np.sin(k * x)
        names.extend([cn, sn])
    return pd.DataFrame(cols), names

def _build_cos_qtrend(x, K):
    cols = {'trend_x': x, 'trend_x2': x**2}
    names = ['trend_x', 'trend_x2']
    for k in range(1, K + 1):
        n = f'cos({k}x)'; cols[n] = np.cos(k * x); names.append(n)
    return pd.DataFrame(cols), names

def _build_sincos_qtrend(x, K):
    cols = {'trend_x': x, 'trend_x2': x**2}
    names = ['trend_x', 'trend_x2']
    for k in range(1, K + 1):
        cn, sn = f'cos({k}x)', f'sin({k}x)'
        cols[cn] = np.cos(k * x); cols[sn] = np.sin(k * x)
        names.extend([cn, sn])
    return pd.DataFrame(cols), names

# ============================
# SAFE ARRAY HELPER
# ============================
def _arr(obj):
    """Safely convert model attribute to flat numpy array."""
    return np.asarray(obj).flatten()

# ============================
# HELPER FUNCTIONS
# ============================
def fit_fourier(x, y, model_type, K):
    basis_df, col_names = MODEL_TYPES[model_type]["build"](x, K)
    X_mat = sm.add_constant(basis_df.values)
    model = OLS(y, X_mat).fit()
    y_hat = np.asarray(model.fittedvalues).flatten()
    residuals = np.asarray(model.resid).flatten()
    param_names = ['const (b0/2)'] + col_names
    return model, y_hat, residuals, X_mat, param_names

def compute_gcv(x, y, model_type, K):
    try:
        basis_df, _ = MODEL_TYPES[model_type]["build"](x, K)
        X_mat = sm.add_constant(basis_df.values)
        n = len(y)
        hat_matrix = X_mat @ np.linalg.pinv(X_mat.T @ X_mat) @ X_mat.T
        y_hat = hat_matrix @ y
        res = y - y_hat
        trace_H = np.trace(hat_matrix)
        mse = np.mean(res**2)
        return mse / ((1 - trace_H / n)**2)
    except:
        return np.inf

def compute_aic_bic(model, n):
    k = model.df_model + 1
    ssr = np.sum(np.asarray(model.resid)**2)
    aic = n * np.log(ssr / n) + 2 * k
    bic = n * np.log(ssr / n) + k * np.log(n)
    return aic, bic

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
    n = 250
    X_demo = np.sort(np.random.uniform(0, 4 * np.pi, n))
    Y_demo = (2.0 * np.sin(X_demo) - 1.5 * np.cos(2 * X_demo) + 0.3 * X_demo
              + 0.8 * np.sin(3 * X_demo) + np.random.normal(0, 0.6, n))
    df = pd.DataFrame({'X': X_demo, 'Y': Y_demo})
    st.sidebar.success("Data demo dimuat (250 obs, pola periodik + tren)")
else:
    st.warning("Silakan upload data atau gunakan data demo.")
    st.stop()

# ============================
# 1. DATA EXPLORATION
# ============================
st.header("1. Eksplorasi Data")
col1, col2, col3 = st.columns(3)
col1.metric("Jumlah Observasi", df.shape[0])
col2.metric("Jumlah Variabel", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())

tab_d1, tab_d2 = st.tabs(["Data", "Statistik Deskriptif"])
with tab_d1:
    st.dataframe(df.head(30), use_container_width=True)
with tab_d2:
    st.dataframe(df.describe().T.round(4), use_container_width=True)

# ============================
# 2. VARIABLE SELECTION
# ============================
st.header("2. Pemilihan Variabel")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Minimal 2 kolom numerik.")
    st.stop()

col_l, col_r = st.columns(2)
with col_l:
    y_var = st.selectbox("Variabel Dependen (Y)", numeric_cols, index=min(1, len(numeric_cols)-1))
with col_r:
    x_options = [c for c in numeric_cols if c != y_var]
    x_var = st.selectbox("Variabel Independen (X)", x_options, index=0)

Y_raw = df[y_var].dropna().values
X_raw = df[x_var].dropna().values
min_len = min(len(Y_raw), len(X_raw))
Y_raw, X_raw = Y_raw[:min_len], X_raw[:min_len]
sort_idx = np.argsort(X_raw)
X = X_raw[sort_idx]
Y = Y_raw[sort_idx]
n_obs = len(X)

# ============================
# 3. SCATTER & LINEARITY
# ============================
st.header("3. Pola Hubungan Data")
fig_sc = px.scatter(x=X, y=Y, labels={'x': x_var, 'y': y_var},
                     title=f"Scatter Plot: {y_var} vs {x_var}", opacity=0.5)
z = np.polyfit(X, Y, 1)
p_line = np.poly1d(z)
x_range = np.linspace(X.min(), X.max(), 300)
fig_sc.add_trace(go.Scatter(x=x_range, y=p_line(x_range), mode='lines',
                              name='Regresi Linier', line=dict(color='red', dash='dash')))
st.plotly_chart(fig_sc, use_container_width=True)

X_ols = sm.add_constant(X)
ols_model = OLS(Y, X_ols).fit()
Y_hat_ols = _arr(ols_model.fittedvalues)
X_reset = np.column_stack([X_ols, Y_hat_ols**2, Y_hat_ols**3])
ols_reset = OLS(Y, X_reset).fit()
f_reset = ((ols_model.ssr - ols_reset.ssr) / 2) / (ols_reset.ssr / (n_obs - X_reset.shape[1]))
p_reset = 1 - stats.f.cdf(f_reset, 2, n_obs - X_reset.shape[1])

st.subheader("Uji Linieritas — Ramsey RESET Test")
reset_df = pd.DataFrame({
    'Uji': ['Ramsey RESET'],
    'F-statistic': [round(float(f_reset), 4)],
    'p-value': [round(float(p_reset), 6)],
    'Keputusan (a=0.05)': ['TIDAK linier -> Gunakan nonparametrik' if p_reset < 0.05
                           else 'Linier -> Regresi parametrik mungkin cukup']
})
st.dataframe(reset_df, use_container_width=True, hide_index=True)

# ============================
# 4. MODEL SETTINGS
# ============================
st.header("4. Pengaturan Model Fourier")
st.subheader("Variasi Model Tersedia")
model_info = pd.DataFrame({
    'Model': list(MODEL_TYPES.keys()),
    'Formula': [v['formula'] for v in MODEL_TYPES.values()],
    'Keterangan': [v['description'] for v in MODEL_TYPES.values()]
})
st.dataframe(model_info, use_container_width=True, hide_index=True)

col_s1, col_s2 = st.columns(2)
with col_s1:
    selected_model = st.selectbox("Pilih Tipe Model Fourier", list(MODEL_TYPES.keys()), index=0)
with col_s2:
    max_K_search = st.number_input("Maks. osilasi (K) untuk pencarian GCV",
                                    min_value=1, max_value=500, value=15, step=1)
st.info(f"**Model terpilih:** {MODEL_TYPES[selected_model]['formula']}")

# ============================
# 5. OSCILLATION SELECTION (GCV)
# ============================
st.header("5. Pemilihan Osilasi Optimal (K)")
with st.spinner("Mencari K optimal menggunakan GCV..."):
    gcv_scores = {}
    for K in range(1, int(max_K_search) + 1):
        gcv_scores[K] = compute_gcv(X, Y, selected_model, K)
    gcv_df = pd.DataFrame({'Osilasi (K)': list(gcv_scores.keys()), 'GCV Score': list(gcv_scores.values())})
    optimal_K = min(gcv_scores, key=gcv_scores.get)
    optimal_gcv = gcv_scores[optimal_K]

fig_gcv = go.Figure()
fig_gcv.add_trace(go.Scatter(x=gcv_df['Osilasi (K)'], y=gcv_df['GCV Score'],
                              mode='lines+markers', name='GCV Score', marker=dict(size=8)))
fig_gcv.add_vline(x=optimal_K, line_dash="dash", line_color="red",
                   annotation_text=f"Optimal K={optimal_K}")
fig_gcv.update_layout(title="GCV Score vs Jumlah Osilasi (K)",
                       xaxis_title="Osilasi (K)", yaxis_title="GCV Score", height=400)
st.plotly_chart(fig_gcv, use_container_width=True)
st.dataframe(gcv_df.round(6), use_container_width=True, hide_index=True)
st.success(f"Osilasi optimal: **K = {optimal_K}** (GCV = {optimal_gcv:.6f})")

manual_K = st.checkbox("Override: gunakan K manual", value=False)
if manual_K:
    use_K = st.number_input("Masukkan nilai K", min_value=1, max_value=500, value=optimal_K)
else:
    use_K = optimal_K

# ============================
# 6. FIT MODEL
# ============================
st.header("6. Hasil Estimasi Deret Fourier")
with st.spinner("Mengestimasi model deret Fourier..."):
    model, Y_fit, residuals, X_mat, param_names = fit_fourier(X, Y, selected_model, use_K)

ss_res = np.sum(residuals**2)
ss_tot = np.sum((Y - np.mean(Y))**2)
r2 = 1 - ss_res / ss_tot
adj_r2 = 1 - (1 - r2) * (n_obs - 1) / (n_obs - model.df_model - 1)
mse = np.mean(residuals**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(residuals))
aic_val, bic_val = compute_aic_bic(model, n_obs)

met1, met2, met3, met4 = st.columns(4)
met1.metric("R-squared", f"{r2:.4f}")
met2.metric("Adj. R-squared", f"{adj_r2:.4f}")
met3.metric("RMSE", f"{rmse:.6f}")
met4.metric("MAE", f"{mae:.6f}")
met5, met6, met7, met8 = st.columns(4)
met5.metric("MSE", f"{mse:.6f}")
met6.metric("AIC", f"{aic_val:.2f}")
met7.metric("BIC", f"{bic_val:.2f}")
met8.metric("Osilasi (K)", f"{use_K}")

# --- COEFFICIENT TABLE (FIXED) ---
st.subheader("Tabel Koefisien (Parameter)")
_params = _arr(model.params)
_bse = _arr(model.bse)
_tvals = _arr(model.tvalues)
_pvals = _arr(model.pvalues)
_ci = np.asarray(model.conf_int())
coef_data = pd.DataFrame({
    'Parameter': param_names,
    'Koefisien': _params,
    'Std. Error': _bse,
    't-value': _tvals,
    'p-value': _pvals,
    'CI Lower (95%)': _ci[:, 0],
    'CI Upper (95%)': _ci[:, 1],
    'Signifikan (a=0.05)': ['Ya' if p < 0.05 else 'Tidak' for p in _pvals]
}).round(6)
st.dataframe(coef_data, use_container_width=True, hide_index=True)

with st.expander("Ringkasan OLS Lengkap", expanded=False):
    st.text(model.summary().as_text())

# Equation
st.subheader("Persamaan Model")
eq_parts = [f"{_params[0]:.4f}"]
for i, name in enumerate(param_names[1:], 1):
    coef = _params[i]
    sign = "+" if coef >= 0 else "-"
    eq_parts.append(f" {sign} {abs(coef):.4f} * {name}")
st.markdown(f"**{y_var}** = {''.join(eq_parts)}")

# ============================
# 6a. REGRESSION CURVE
# ============================
st.subheader("Kurva Regresi Fourier")
X_grid = np.linspace(X.min(), X.max(), 500)
basis_grid, _ = MODEL_TYPES[selected_model]["build"](X_grid, use_K)
X_grid_mat = sm.add_constant(basis_grid.values)
Y_grid = np.asarray(model.predict(X_grid_mat)).flatten()

fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(x=X, y=Y, mode='markers', name='Data Observasi',
                                marker=dict(size=4, opacity=0.4, color='steelblue')))
fig_curve.add_trace(go.Scatter(x=X_grid, y=Y_grid, mode='lines',
                                name=f'Fourier (K={use_K})', line=dict(color='red', width=3)))
fig_curve.add_trace(go.Scatter(x=x_range, y=p_line(x_range), mode='lines',
                                name='Regresi Linier (OLS)', line=dict(color='green', dash='dash', width=2)))
fig_curve.update_layout(title=f"Kurva Regresi Fourier ({selected_model}, K={use_K})",
                         xaxis_title=x_var, yaxis_title=y_var, height=500)
st.plotly_chart(fig_curve, use_container_width=True)

# ============================
# 6b. FOURIER COMPONENTS
# ============================
st.subheader("Komponen Deret Fourier")
with st.expander("Tampilkan Komponen Individual", expanded=False):
    basis_comp, comp_names = MODEL_TYPES[selected_model]["build"](X_grid, use_K)
    fig_comp = go.Figure()
    colors_comp = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1
    for i, cname in enumerate(comp_names):
        weighted = _params[i + 1] * basis_comp[cname].values
        fig_comp.add_trace(go.Scatter(x=X_grid, y=weighted, mode='lines',
                                       name=f'{_params[i+1]:.3f} * {cname}',
                                       line=dict(width=2, color=colors_comp[i % len(colors_comp)])))
    fig_comp.add_hline(y=0, line_color="gray", line_dash="dot")
    fig_comp.update_layout(title="Komponen Fourier Tertimbang", xaxis_title=x_var,
                            yaxis_title="Kontribusi", height=450)
    st.plotly_chart(fig_comp, use_container_width=True)

# ============================
# 7. OSCILLATION SENSITIVITY
# ============================
st.header("7. Sensitivitas Osilasi (K)")
k_values_vis = sorted(set([1, 2, 3, 5, 8, max(1, optimal_K), min(int(max_K_search), 15)]))
fig_ksens = go.Figure()
fig_ksens.add_trace(go.Scatter(x=X, y=Y, mode='markers', name='Data',
                                marker=dict(size=3, opacity=0.3, color='gray')))
ksens_colors = px.colors.qualitative.Set1
for idx, K_val in enumerate(k_values_vis):
    try:
        m_k, _, _, _, _ = fit_fourier(X, Y, selected_model, K_val)
        bg, _ = MODEL_TYPES[selected_model]["build"](X_grid, K_val)
        xg = sm.add_constant(bg.values)
        yg = np.asarray(m_k.predict(xg)).flatten()
        is_opt = (K_val == use_K)
        fig_ksens.add_trace(go.Scatter(x=X_grid, y=yg, mode='lines',
            name=f'K={K_val}' + (' (optimal)' if is_opt else ''),
            line=dict(width=4 if is_opt else 1.5, color=ksens_colors[idx % len(ksens_colors)])))
    except: pass
fig_ksens.update_layout(title=f"Pengaruh Jumlah Osilasi (K) — {selected_model}",
                          xaxis_title=x_var, yaxis_title=y_var, height=500)
st.plotly_chart(fig_ksens, use_container_width=True)

# ============================
# 8. MODEL TYPE COMPARISON
# ============================
st.header("8. Perbandingan Semua Variasi Model")
with st.spinner("Membandingkan semua variasi model Fourier..."):
    comparison_results = {}
    for mname, minfo in MODEL_TYPES.items():
        try:
            best_k, best_gcv = 1, np.inf
            for K_try in range(1, int(max_K_search) + 1):
                g = compute_gcv(X, Y, mname, K_try)
                if g < best_gcv: best_gcv = g; best_k = K_try
            m, yh, res, _, _ = fit_fourier(X, Y, mname, best_k)
            ss_r = np.sum(res**2)
            r2_m = 1 - ss_r / ss_tot
            adj_r2_m = 1 - (1 - r2_m) * (n_obs - 1) / (n_obs - m.df_model - 1)
            aic_m, bic_m = compute_aic_bic(m, n_obs)
            comparison_results[mname] = {'K Optimal': best_k, 'GCV': best_gcv, 'R2': r2_m,
                'Adj_R2': adj_r2_m, 'MSE': np.mean(res**2), 'AIC': aic_m, 'BIC': bic_m,
                'n_params': int(m.df_model + 1)}
        except: pass

tab_mc1, tab_mc2 = st.tabs(["Visualisasi Kurva", "Tabel Perbandingan"])
with tab_mc1:
    fig_mcomp = go.Figure()
    fig_mcomp.add_trace(go.Scatter(x=X, y=Y, mode='markers', name='Data',
                                    marker=dict(size=3, opacity=0.3, color='gray')))
    mcomp_colors = px.colors.qualitative.Plotly
    for idx, (mname, mres) in enumerate(comparison_results.items()):
        try:
            m_c, _, _, _, _ = fit_fourier(X, Y, mname, mres['K Optimal'])
            bg_c, _ = MODEL_TYPES[mname]["build"](X_grid, mres['K Optimal'])
            xg_c = sm.add_constant(bg_c.values)
            yg_c = np.asarray(m_c.predict(xg_c)).flatten()
            fig_mcomp.add_trace(go.Scatter(x=X_grid, y=yg_c, mode='lines',
                name=f"{mname} (K={mres['K Optimal']}, R2={mres['R2']:.3f})",
                line=dict(width=2, color=mcomp_colors[idx % len(mcomp_colors)])))
        except: pass
    fig_mcomp.update_layout(title="Perbandingan Semua Variasi Model Fourier",
                             xaxis_title=x_var, yaxis_title=y_var, height=550)
    st.plotly_chart(fig_mcomp, use_container_width=True)

with tab_mc2:
    comp_df = pd.DataFrame(comparison_results).T.round(6)
    comp_df = comp_df.sort_values('GCV')
    comp_df.insert(0, 'Model', comp_df.index)
    comp_df = comp_df.reset_index(drop=True)
    comp_df['Ranking (GCV)'] = range(1, len(comp_df) + 1)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.success(f"Model terbaik (GCV terendah): **{comp_df.iloc[0]['Model']}** dengan K={int(comp_df.iloc[0]['K Optimal'])}")

# ============================
# 9. FOURIER vs OLS
# ============================
st.header("9. Perbandingan: Fourier vs OLS")
ols_resid = _arr(ols_model.resid)
ols_mse = np.mean(ols_resid**2)
ols_r2 = float(ols_model.rsquared)
ols_adj_r2 = float(ols_model.rsquared_adj)
ols_aic, ols_bic = compute_aic_bic(ols_model, n_obs)

vs_df = pd.DataFrame({
    'Metrik': ['R2', 'Adj. R2', 'MSE', 'RMSE', 'MAE', 'AIC', 'BIC'],
    'OLS Linier': [ols_r2, ols_adj_r2, ols_mse, np.sqrt(ols_mse),
                   np.mean(np.abs(ols_resid)), ols_aic, ols_bic],
    f'Fourier ({selected_model})': [r2, adj_r2, mse, rmse, mae, aic_val, bic_val],
    'Model Terbaik': [
        'Fourier' if r2 > ols_r2 else 'OLS',
        'Fourier' if adj_r2 > ols_adj_r2 else 'OLS',
        'Fourier' if mse < ols_mse else 'OLS',
        'Fourier' if rmse < np.sqrt(ols_mse) else 'OLS',
        'Fourier' if mae < np.mean(np.abs(ols_resid)) else 'OLS',
        'Fourier' if aic_val < ols_aic else 'OLS',
        'Fourier' if bic_val < ols_bic else 'OLS']
}).round(6)
st.dataframe(vs_df, use_container_width=True, hide_index=True)

# ============================
# 10. RESIDUAL DIAGNOSTICS
# ============================
st.header("10. Diagnostik Residual")
met_r1, met_r2_, met_r3, met_r4 = st.columns(4)
met_r1.metric("MSE", f"{mse:.6f}")
met_r2_.metric("RMSE", f"{rmse:.6f}")
met_r3.metric("MAE", f"{mae:.6f}")
met_r4.metric("R2", f"{r2:.4f}")

st.subheader("10a. Plot Residual")
fig_resid = make_subplots(rows=1, cols=2, subplot_titles=("Residual vs Fitted", "Residual vs X"))
fig_resid.add_trace(go.Scatter(x=Y_fit, y=residuals, mode='markers',
    marker=dict(size=4, opacity=0.5, color='steelblue'), showlegend=False), row=1, col=1)
fig_resid.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
fig_resid.add_trace(go.Scatter(x=X, y=residuals, mode='markers',
    marker=dict(size=4, opacity=0.5, color='darkorange'), showlegend=False), row=1, col=2)
fig_resid.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
fig_resid.update_xaxes(title_text="Fitted Values", row=1, col=1)
fig_resid.update_xaxes(title_text=x_var, row=1, col=2)
fig_resid.update_yaxes(title_text="Residuals", row=1, col=1)
fig_resid.update_yaxes(title_text="Residuals", row=1, col=2)
fig_resid.update_layout(height=400)
st.plotly_chart(fig_resid, use_container_width=True)

st.subheader("10b. Uji Normalitas Residual")
tab_n1, tab_n2 = st.tabs(["Visualisasi", "Uji Statistik"])
with tab_n1:
    col_h, col_q = st.columns(2)
    with col_h:
        fig_hist = px.histogram(x=residuals, nbins=30, title="Histogram Residual",
                                 labels={'x': 'Residual'}, marginal="box")
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_q:
        qq = stats.probplot(residuals, dist="norm")
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Data'))
        fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0]*qq[0][0]+qq[1][1],
                                     mode='lines', name='Garis Normal', line=dict(color='red')))
        fig_qq.update_layout(title="QQ-Plot Residual", xaxis_title="Theoretical Quantiles",
                              yaxis_title="Sample Quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)
with tab_n2:
    sw_stat, sw_p = (stats.shapiro(residuals) if len(residuals) <= 5000 else (np.nan, np.nan))
    ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
    jb_stat, jb_p = stats.jarque_bera(residuals)
    ad_result = stats.anderson(residuals, dist='norm')
    norm_df = pd.DataFrame({
        'Uji': ['Shapiro-Wilk', 'Kolmogorov-Smirnov', 'Jarque-Bera', 'Anderson-Darling'],
        'Statistik': [sw_stat, ks_stat, jb_stat, ad_result.statistic],
        'p-value': [sw_p, ks_p, jb_p, np.nan],
        'Keputusan (a=0.05)': [
            'Normal' if sw_p > 0.05 else 'Tidak Normal',
            'Normal' if ks_p > 0.05 else 'Tidak Normal',
            'Normal' if jb_p > 0.05 else 'Tidak Normal',
            'Normal' if ad_result.statistic < ad_result.critical_values[2] else 'Tidak Normal']
    }).round(6)
    st.dataframe(norm_df, use_container_width=True, hide_index=True)

st.subheader("10c. Uji Homoskedastisitas")
bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(residuals, X_mat)
sp_stat, sp_p = stats.spearmanr(X, np.abs(residuals))
pe_stat, pe_p = stats.pearsonr(X, np.abs(residuals))
mid = n_obs // 2
gq_f = np.sum(residuals[mid:]**2) / np.sum(residuals[:mid]**2)
gq_p = 1 - stats.f.cdf(gq_f, mid, mid)
het_df = pd.DataFrame({
    'Uji': ['Breusch-Pagan', 'Goldfeld-Quandt', 'Spearman (|resid| vs X)', 'Pearson (|resid| vs X)'],
    'Statistik': [bp_stat, gq_f, sp_stat, pe_stat],
    'p-value': [bp_p, gq_p, sp_p, pe_p],
    'Keputusan (a=0.05)': [
        'Homoskedastis' if bp_p > 0.05 else 'Heteroskedastis',
        'Homoskedastis' if gq_p > 0.05 else 'Heteroskedastis',
        'Homoskedastis' if sp_p > 0.05 else 'Heteroskedastis',
        'Homoskedastis' if pe_p > 0.05 else 'Heteroskedastis']
}).round(6)
st.dataframe(het_df, use_container_width=True, hide_index=True)

st.subheader("10d. Uji Independensi Residual")
dw_stat = durbin_watson(residuals)
lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
lb_stat = lb_result['lb_stat'].values[0]
lb_p = lb_result['lb_pvalue'].values[0]
median_resid = np.median(residuals)
binary_resid = (residuals >= median_resid).astype(int)
runs = 1 + np.sum(np.diff(binary_resid) != 0)
n1 = np.sum(binary_resid); n0 = len(binary_resid) - n1
runs_mean = (2 * n1 * n0) / (n1 + n0) + 1
runs_var = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0)**2 * (n1 + n0 - 1))
runs_z = (runs - runs_mean) / np.sqrt(runs_var) if runs_var > 0 else 0
runs_p = 2 * (1 - stats.norm.cdf(np.abs(runs_z)))
indep_df = pd.DataFrame({
    'Uji': ['Durbin-Watson', 'Ljung-Box (lag=10)', 'Runs Test'],
    'Statistik': [dw_stat, lb_stat, runs_z],
    'p-value': [np.nan, lb_p, runs_p],
    'Keputusan': [
        'Independen' if 1.5 <= dw_stat <= 2.5 else 'Ada autokorelasi',
        'Independen' if lb_p > 0.05 else 'Ada autokorelasi',
        'Independen' if runs_p > 0.05 else 'Tidak independen']
}).round(4)
st.dataframe(indep_df, use_container_width=True, hide_index=True)

acf_vals = acf(residuals, nlags=25, fft=True)
fig_acf = go.Figure()
for i in range(len(acf_vals)):
    fig_acf.add_trace(go.Bar(x=[i], y=[acf_vals[i]], marker_color='steelblue', showlegend=False, width=0.3))
ci_acf = 1.96 / np.sqrt(n_obs)
fig_acf.add_hline(y=ci_acf, line_dash="dash", line_color="red")
fig_acf.add_hline(y=-ci_acf, line_dash="dash", line_color="red")
fig_acf.add_hline(y=0, line_color="black")
fig_acf.update_layout(title="ACF Residual", xaxis_title="Lag", yaxis_title="ACF", height=350)
st.plotly_chart(fig_acf, use_container_width=True)

# ============================
# 11. SIGNIFICANCE
# ============================
st.header("11. Uji Signifikansi Model")
df_model_resid = n_obs - model.df_model - 1
f_stat = ((ss_tot - ss_res) / model.df_model) / (ss_res / df_model_resid)
f_p = 1 - stats.f.cdf(f_stat, model.df_model, df_model_resid)
ss_ols = np.sum(ols_resid**2)
df_diff = model.df_model - ols_model.df_model
if df_diff > 0:
    f_vs_ols = ((ss_ols - ss_res) / df_diff) / (ss_res / df_model_resid)
    f_vs_ols_p = 1 - stats.f.cdf(f_vs_ols, df_diff, df_model_resid)
else:
    f_vs_ols, f_vs_ols_p = np.nan, np.nan
sig_df = pd.DataFrame({
    'Uji': ['F-test (Model vs Null)', 'F-test (Fourier vs OLS)'],
    'F-statistic': [f_stat, f_vs_ols],
    'df1': [model.df_model, df_diff],
    'df2': [df_model_resid, df_model_resid],
    'p-value': [f_p, f_vs_ols_p],
    'Keputusan (a=0.05)': [
        'Model Signifikan' if f_p < 0.05 else 'Tidak Signifikan',
        'Fourier lebih baik dari OLS' if (not np.isnan(f_vs_ols_p) and f_vs_ols_p < 0.05)
        else 'Fourier tidak signifikan lebih baik']
}).round(6)
st.dataframe(sig_df, use_container_width=True, hide_index=True)

# ============================
# 12. INFLUENTIAL OBS
# ============================
st.header("12. Deteksi Observasi Berpengaruh")
influence = model.get_influence()
inf_summary = influence.summary_frame()
cooks_d = inf_summary['cooks_d']
leverage = inf_summary['hat_diag']
threshold_cook = 4 / n_obs

tab_i1, tab_i2 = st.tabs(["Cook's Distance", "Detail"])
with tab_i1:
    fig_cook = px.bar(x=cooks_d.index, y=cooks_d.values,
                       title=f"Cook's Distance (threshold = {threshold_cook:.4f})",
                       labels={'x': 'Observasi', 'y': "Cook's Distance"})
    fig_cook.add_hline(y=threshold_cook, line_dash="dash", line_color="red")
    st.plotly_chart(fig_cook, use_container_width=True)
    st.info(f"Observasi berpengaruh: **{len(cooks_d[cooks_d > threshold_cook])}** dari {n_obs}")
with tab_i2:
    inf_detail = pd.DataFrame({
        'Obs': range(n_obs), x_var: X, y_var: Y,
        'Fitted': Y_fit, 'Residual': residuals,
        'Leverage': leverage.values, "Cook_D": cooks_d.values,
        'Std_Resid': inf_summary['standard_resid'].values
    }).round(6)
    st.dataframe(inf_detail.nlargest(20, "Cook_D"), use_container_width=True, hide_index=True)

# ============================
# 13. CONFIDENCE BAND
# ============================
st.header("13. Confidence Band (Bootstrap)")
with st.expander("Tampilkan Confidence Band", expanded=False):
    with st.spinner("Menghitung bootstrap confidence band..."):
        n_boot = 200
        boot_curves = np.zeros((n_boot, len(X_grid)))
        for b in range(n_boot):
            idx = np.random.choice(n_obs, size=n_obs, replace=True)
            X_b, Y_b = X[idx], Y[idx]
            si = np.argsort(X_b); X_b, Y_b = X_b[si], Y_b[si]
            try:
                m_b, _, _, _, _ = fit_fourier(X_b, Y_b, selected_model, use_K)
                bg_b, _ = MODEL_TYPES[selected_model]["build"](X_grid, use_K)
                xg_b = sm.add_constant(bg_b.values)
                boot_curves[b, :] = np.asarray(m_b.predict(xg_b)).flatten()
            except:
                boot_curves[b, :] = Y_grid
        ci_low = np.percentile(boot_curves, 2.5, axis=0)
        ci_up = np.percentile(boot_curves, 97.5, axis=0)
    fig_cb = go.Figure()
    fig_cb.add_trace(go.Scatter(x=X, y=Y, mode='markers', name='Data',
                                 marker=dict(size=3, opacity=0.3, color='gray')))
    fig_cb.add_trace(go.Scatter(x=np.concatenate([X_grid, X_grid[::-1]]),
        y=np.concatenate([ci_up, ci_low[::-1]]),
        fill='toself', fillcolor='rgba(255,0,0,0.15)',
        line=dict(color='rgba(255,0,0,0)'), name='95% Confidence Band'))
    fig_cb.add_trace(go.Scatter(x=X_grid, y=Y_grid, mode='lines',
                                 name='Estimasi Fourier', line=dict(color='red', width=3)))
    fig_cb.update_layout(title="Kurva Fourier + 95% Bootstrap Confidence Band",
                          xaxis_title=x_var, yaxis_title=y_var, height=500)
    st.plotly_chart(fig_cb, use_container_width=True)

# ============================
# 14. PREDICTION
# ============================
st.header("14. Prediksi Nilai Baru")
x_pred = st.number_input(f"Masukkan nilai {x_var}",
                          value=float(np.median(X)),
                          step=float(np.std(X) / 10), format="%.4f")
if st.button("Prediksi", type="primary"):
    bg_p, _ = MODEL_TYPES[selected_model]["build"](np.array([x_pred]), use_K)
    xg_p = sm.add_constant(bg_p.values, has_constant='add')
    y_pred = float(np.asarray(model.predict(xg_p)).flatten()[0])
    n_boot_pred = 500
    boot_preds = []
    for _ in range(n_boot_pred):
        idx = np.random.choice(n_obs, size=n_obs, replace=True)
        X_b, Y_b = X[idx], Y[idx]
        si = np.argsort(X_b); X_b, Y_b = X_b[si], Y_b[si]
        try:
            m_b, _, _, _, _ = fit_fourier(X_b, Y_b, selected_model, use_K)
            bg_b, _ = MODEL_TYPES[selected_model]["build"](np.array([x_pred]), use_K)
            xg_b = sm.add_constant(bg_b.values, has_constant='add')
            boot_preds.append(float(np.asarray(m_b.predict(xg_b)).flatten()[0]))
        except: pass
    ci_l = np.percentile(boot_preds, 2.5)
    ci_u = np.percentile(boot_preds, 97.5)
    st.success(f"**Prediksi {y_var} = {y_pred:.4f}**")
    pred_df = pd.DataFrame({
        'Metrik': ['Predicted Value', 'CI Lower (95%)', 'CI Upper (95%)', 'Model', 'Osilasi (K)'],
        'Nilai': [str(round(y_pred, 4)), str(round(float(ci_l), 4)), str(round(float(ci_u), 4)),
                  selected_model, str(use_K)]
    })
    st.dataframe(pred_df, use_container_width=True, hide_index=True)

# ============================
# 15. EXPORT
# ============================
st.header("15. Ekspor Hasil")
col_e1, col_e2, col_e3 = st.columns(3)
with col_e1:
    lines = [
        "=" * 65, "REGRESI NONPARAMETRIK - DERET FOURIER", "=" * 65,
        f"Model              : {selected_model}",
        f"Formula            : {MODEL_TYPES[selected_model]['formula']}",
        f"Osilasi (K)        : {use_K}",
        f"N Observasi        : {n_obs}",
        f"Jumlah Parameter   : {int(model.df_model + 1)}",
        f"R2                 : {r2:.6f}",
        f"Adj. R2            : {adj_r2:.6f}",
        f"MSE                : {mse:.6f}",
        f"RMSE               : {rmse:.6f}",
        f"MAE                : {mae:.6f}",
        f"AIC                : {aic_val:.4f}",
        f"BIC                : {bic_val:.4f}",
        f"GCV                : {optimal_gcv:.6f}",
        "", "=" * 65, "KOEFISIEN", "=" * 65]
    for i, pname in enumerate(param_names):
        lines.append(f"  {pname:25s} : {_params[i]:>12.6f}  (p={_pvals[i]:.6f})")
    st.download_button("Download Summary (TXT)", data="\n".join(lines),
                       file_name="fourier_summary.txt", mime="text/plain")
with col_e2:
    res_df = pd.DataFrame({x_var: X, y_var: Y, 'Fitted': Y_fit, 'Residual': residuals})
    st.download_button("Download Hasil (CSV)", data=res_df.to_csv(index=False),
                       file_name="fourier_results.csv", mime="text/csv")
with col_e3:
    crv_df = pd.DataFrame({'X_grid': X_grid, 'Y_estimated': Y_grid})
    st.download_button("Download Kurva (CSV)", data=crv_df.to_csv(index=False),
                       file_name="fourier_curve.csv", mime="text/csv")

st.markdown("---")
st.markdown("""
**Catatan Metodologis:**
- **Deret Fourier** mengaproksimasi kurva regresi menggunakan kombinasi fungsi trigonometri (sin/cos) dan komponen tren.
- **K (osilasi)** menentukan jumlah fungsi trigonometri — semakin besar K, semakin fleksibel model.
- **GCV** digunakan untuk memilih K optimal secara data-driven.
- Estimasi parameter menggunakan **Ordinary Least Squares (OLS)**.
""")
st.markdown("Dibangun dengan **Streamlit** + **Statsmodels** + **SciPy** + **Plotly** | Python")
