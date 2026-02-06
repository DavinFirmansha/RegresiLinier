import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy import stats
from scipy.optimize import minimize_scalar
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Regresi Nonparametrik Estimator Kernel",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Regresi Nonparametrik — Estimator Kernel")
st.markdown("""
Aplikasi lengkap untuk analisis **regresi nonparametrik** menggunakan metode **Nadaraya-Watson** 
dan **Local Linear** kernel estimator, dilengkapi uji asumsi, pemilihan bandwidth, 
perbandingan fungsi kernel, dan diagnostik model.
""")

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
    n = 300
    X_demo = np.sort(np.random.uniform(0, 10, n))
    Y_demo = np.sin(X_demo) + 0.5 * np.cos(2 * X_demo) + np.random.normal(0, 0.4, n)
    df = pd.DataFrame({'X': X_demo, 'Y': Y_demo})
    st.sidebar.success("✅ Data demo dimuat (300 observasi, fungsi nonlinier)")
else:
    st.warning("⚠️ Silakan upload data atau gunakan data demo.")
    st.stop()

# ============================
# 1. DATA EXPLORATION
# ============================
st.header("1️⃣ Eksplorasi Data")
col1, col2, col3 = st.columns(3)
col1.metric("Jumlah Observasi", df.shape[0])
col2.metric("Jumlah Variabel", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())

tab_data1, tab_data2 = st.tabs(["📋 Data", "📊 Statistik Deskriptif"])
with tab_data1:
    st.dataframe(df.head(30), use_container_width=True)
with tab_data2:
    st.dataframe(df.describe().T.round(4), use_container_width=True)

# ============================
# 2. VARIABLE SELECTION
# ============================
st.header("2️⃣ Pemilihan Variabel")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("❌ Data harus memiliki minimal 2 kolom numerik.")
    st.stop()

col_l, col_r = st.columns(2)
with col_l:
    y_var = st.selectbox("🎯 Variabel Dependen (Y)", numeric_cols, index=min(1, len(numeric_cols)-1))
with col_r:
    x_options = [c for c in numeric_cols if c != y_var]
    x_var = st.selectbox("📌 Variabel Independen (X)", x_options, index=0)

Y = df[y_var].dropna().values
X = df[x_var].dropna().values
min_len = min(len(Y), len(X))
Y = Y[:min_len]
X = X[:min_len]
sort_idx = np.argsort(X)
X_sorted = X[sort_idx]
Y_sorted = Y[sort_idx]

# ============================
# 3. SCATTER PLOT & LINEARITY CHECK
# ============================
st.header("3️⃣ Pola Hubungan Data")

fig_scatter = px.scatter(x=X, y=Y, labels={'x': x_var, 'y': y_var},
                         title=f"Scatter Plot: {y_var} vs {x_var}", opacity=0.5)
# Add OLS line for comparison
z = np.polyfit(X, Y, 1)
p_line = np.poly1d(z)
x_range = np.linspace(X.min(), X.max(), 200)
fig_scatter.add_trace(go.Scatter(x=x_range, y=p_line(x_range),
                                  mode='lines', name='Regresi Linier (OLS)',
                                  line=dict(color='red', dash='dash')))
st.plotly_chart(fig_scatter, use_container_width=True)

# Ramsey RESET test for linearity
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm_api
X_ols = sm_api.add_constant(X)
ols_model = OLS(Y, X_ols).fit()
Y_hat = ols_model.fittedvalues
X_reset = np.column_stack([X_ols, Y_hat**2, Y_hat**3])
ols_reset = OLS(Y, X_reset).fit()
f_reset = ((ols_model.ssr - ols_reset.ssr) / 2) / (ols_reset.ssr / (len(Y) - X_reset.shape[1]))
p_reset = 1 - stats.f.cdf(f_reset, 2, len(Y) - X_reset.shape[1])

st.subheader("Uji Linieritas — Ramsey RESET Test")
reset_df = pd.DataFrame({
    'Uji': ['Ramsey RESET'],
    'F-statistic': [round(f_reset, 4)],
    'p-value': [round(p_reset, 6)],
    'Keputusan (α=0.05)': ['❌ Hubungan TIDAK linier → Gunakan nonparametrik' if p_reset < 0.05
                           else '✅ Hubungan linier → Regresi parametrik mungkin cukup']
})
st.dataframe(reset_df, use_container_width=True, hide_index=True)

# ============================
# 4. KERNEL REGRESSION SETTINGS
# ============================
st.header("4️⃣ Pengaturan Model Kernel")

col_set1, col_set2, col_set3 = st.columns(3)

with col_set1:
    reg_type = st.selectbox("Tipe Estimator", ["lc", "ll"],
                            format_func=lambda x: "Local Constant (Nadaraya-Watson)" if x == "lc" else "Local Linear",
                            index=1)

with col_set2:
    kernel_list = ['gaussian', 'epanechnikov', 'biweight', 'triweight']
    kernel_display = {
        'gaussian': 'Gaussian (Normal)',
        'epanechnikov': 'Epanechnikov',
        'biweight': 'Biweight (Quartic)',
        'triweight': 'Triweight'
    }
    # For statsmodels KernelReg, var_type='c' uses gaussian by default
    # We'll implement manual kernel functions for comparison

with col_set3:
    bw_method = st.selectbox("Metode Bandwidth", ["cv_ls", "aic"],
                             format_func=lambda x: "Cross-Validation (CV-LS)" if x == "cv_ls" else "AIC (Hurvich-Simonoff-Tsai)",
                             index=0)

manual_bw = st.sidebar.checkbox("Set bandwidth manual", value=False)
if manual_bw:
    bw_value = st.sidebar.slider("Bandwidth (h)", 
                                  min_value=0.01, 
                                  max_value=float(np.std(X) * 3),
                                  value=float(np.std(X) * 0.5),
                                  step=0.01)
else:
    bw_value = None

# ============================
# 5. FIT KERNEL REGRESSION
# ============================
st.header("5️⃣ Hasil Estimasi Kernel Regression")

@st.cache_data
def fit_kernel_reg(X_data, Y_data, reg_type, bw_method, bw_val=None):
    if bw_val is not None:
        kr = KernelReg(endog=Y_data, exog=X_data, var_type='c',
                       reg_type=reg_type, bw=[bw_val])
    else:
        kr = KernelReg(endog=Y_data, exog=X_data, var_type='c',
                       reg_type=reg_type, bw=bw_method)
    mean, mfx = kr.fit()
    return kr, mean, mfx

with st.spinner("Mengestimasi model kernel regression..."):
    kr_model, Y_fit, marginal_fx = fit_kernel_reg(X_sorted, Y_sorted, reg_type, bw_method, bw_value)

optimal_bw = kr_model.bw[0]

# Model info
st.subheader("📋 Informasi Model")
info_col1, info_col2, info_col3, info_col4 = st.columns(4)
info_col1.metric("Tipe Estimator", "Nadaraya-Watson" if reg_type == "lc" else "Local Linear")
info_col2.metric("Bandwidth (h)", f"{optimal_bw:.6f}")
info_col3.metric("Jumlah Observasi", len(X_sorted))
info_col4.metric("R² (Nonparametrik)", f"{kr_model.r_squared():.4f}")

# ============================
# 5a. REGRESSION CURVE
# ============================
st.subheader("📈 Kurva Regresi Kernel")
X_grid = np.linspace(X_sorted.min(), X_sorted.max(), 500)
Y_grid, _ = kr_model.fit(X_grid)

fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(x=X_sorted, y=Y_sorted, mode='markers',
                                name='Data Observasi', marker=dict(size=4, opacity=0.4, color='steelblue')))
fig_curve.add_trace(go.Scatter(x=X_grid, y=Y_grid, mode='lines',
                                name=f'Kernel Regression (h={optimal_bw:.4f})',
                                line=dict(color='red', width=3)))
# OLS comparison
fig_curve.add_trace(go.Scatter(x=x_range, y=p_line(x_range), mode='lines',
                                name='Regresi Linier (OLS)', line=dict(color='green', dash='dash', width=2)))
fig_curve.update_layout(title="Perbandingan: Kernel Regression vs OLS",
                         xaxis_title=x_var, yaxis_title=y_var, height=500)
st.plotly_chart(fig_curve, use_container_width=True)

# ============================
# 5b. MARGINAL EFFECTS
# ============================
st.subheader("📊 Efek Marginal (Turunan Parsial ∂Y/∂X)")
fig_mfx = go.Figure()
fig_mfx.add_trace(go.Scatter(x=X_sorted, y=marginal_fx.flatten(), mode='lines',
                              name='Efek Marginal', line=dict(color='purple', width=2)))
fig_mfx.add_hline(y=0, line_dash="dash", line_color="gray")
fig_mfx.update_layout(title="Efek Marginal dari Kernel Regression",
                       xaxis_title=x_var, yaxis_title=f"∂{y_var}/∂{x_var}", height=400)
st.plotly_chart(fig_mfx, use_container_width=True)

# ============================
# 6. BANDWIDTH ANALYSIS
# ============================
st.header("6️⃣ Analisis Bandwidth")

st.subheader("6a. Sensitivitas Bandwidth")

bw_candidates = np.linspace(optimal_bw * 0.2, optimal_bw * 3.0, 9)
fig_bw = go.Figure()
fig_bw.add_trace(go.Scatter(x=X_sorted, y=Y_sorted, mode='markers', name='Data',
                             marker=dict(size=3, opacity=0.3, color='gray')))

colors = px.colors.qualitative.Set1
for i, bw in enumerate(bw_candidates):
    kr_temp = KernelReg(endog=Y_sorted, exog=X_sorted, var_type='c',
                        reg_type=reg_type, bw=[bw])
    y_temp, _ = kr_temp.fit(X_grid)
    is_optimal = np.isclose(bw, optimal_bw, atol=optimal_bw * 0.15)
    fig_bw.add_trace(go.Scatter(
        x=X_grid, y=y_temp, mode='lines',
        name=f'h = {bw:.4f}' + (' ★ optimal' if is_optimal else ''),
        line=dict(width=4 if is_optimal else 1.5, color=colors[i % len(colors)])
    ))

fig_bw.update_layout(title="Pengaruh Bandwidth pada Kurva Regresi",
                      xaxis_title=x_var, yaxis_title=y_var, height=500)
st.plotly_chart(fig_bw, use_container_width=True)

st.markdown("""
> - **Bandwidth kecil** → kurva lebih fleksibel (undersmoothing, risiko overfitting)  
> - **Bandwidth besar** → kurva lebih halus (oversmoothing, risiko underfitting)  
> - **Bandwidth optimal** dipilih melalui metode Cross-Validation atau AIC
""")

# ============================
# 6b. CROSS-VALIDATION SCORE
# ============================
st.subheader("6b. Leave-One-Out Cross-Validation (LOO-CV)")

bw_range_cv = np.linspace(optimal_bw * 0.1, optimal_bw * 4.0, 50)
cv_scores = []

for bw in bw_range_cv:
    kr_cv = KernelReg(endog=Y_sorted, exog=X_sorted, var_type='c',
                      reg_type=reg_type, bw=[bw])
    y_pred_cv, _ = kr_cv.fit()
    residuals_cv = Y_sorted - y_pred_cv
    mse_cv = np.mean(residuals_cv**2)
    cv_scores.append(mse_cv)

fig_cv = go.Figure()
fig_cv.add_trace(go.Scatter(x=bw_range_cv, y=cv_scores, mode='lines+markers',
                             name='MSE', marker=dict(size=3)))
fig_cv.add_vline(x=optimal_bw, line_dash="dash", line_color="red",
                  annotation_text=f"Optimal h={optimal_bw:.4f}")
fig_cv.update_layout(title="MSE vs Bandwidth", xaxis_title="Bandwidth (h)",
                      yaxis_title="Mean Squared Error", height=400)
st.plotly_chart(fig_cv, use_container_width=True)

# ============================
# 7. KERNEL FUNCTION COMPARISON
# ============================
st.header("7️⃣ Perbandingan Fungsi Kernel")

def gaussian_kernel(u):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

def epanechnikov_kernel(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

def biweight_kernel(u):
    return np.where(np.abs(u) <= 1, (15/16) * (1 - u**2)**2, 0)

def triweight_kernel(u):
    return np.where(np.abs(u) <= 1, (35/32) * (1 - u**2)**3, 0)

def nw_estimate(X_data, Y_data, X_eval, bw, kernel_func):
    n = len(X_data)
    Y_hat = np.zeros(len(X_eval))
    for j in range(len(X_eval)):
        u = (X_data - X_eval[j]) / bw
        weights = kernel_func(u)
        w_sum = np.sum(weights)
        if w_sum > 0:
            Y_hat[j] = np.sum(weights * Y_data) / w_sum
        else:
            Y_hat[j] = np.nan
    return Y_hat

kernels = {
    'Gaussian': gaussian_kernel,
    'Epanechnikov': epanechnikov_kernel,
    'Biweight': biweight_kernel,
    'Triweight': triweight_kernel
}

tab_kshape, tab_kcompare = st.tabs(["📊 Bentuk Fungsi Kernel", "📈 Perbandingan Estimasi"])

with tab_kshape:
    u_vals = np.linspace(-3, 3, 300)
    fig_kshape = go.Figure()
    for name, kfunc in kernels.items():
        fig_kshape.add_trace(go.Scatter(x=u_vals, y=kfunc(u_vals), mode='lines', name=name, line=dict(width=2)))
    fig_kshape.update_layout(title="Bentuk Fungsi Kernel K(u)",
                              xaxis_title="u", yaxis_title="K(u)", height=400)
    st.plotly_chart(fig_kshape, use_container_width=True)

with tab_kcompare:
    fig_kcomp = go.Figure()
    fig_kcomp.add_trace(go.Scatter(x=X_sorted, y=Y_sorted, mode='markers', name='Data',
                                    marker=dict(size=3, opacity=0.3, color='gray')))

    kernel_mse = {}
    kernel_colors = {'Gaussian': 'red', 'Epanechnikov': 'blue', 'Biweight': 'green', 'Triweight': 'orange'}
    for name, kfunc in kernels.items():
        Y_k = nw_estimate(X_sorted, Y_sorted, X_grid, optimal_bw, kfunc)
        Y_k_fit = nw_estimate(X_sorted, Y_sorted, X_sorted, optimal_bw, kfunc)
        mse_k = np.nanmean((Y_sorted - Y_k_fit)**2)
        kernel_mse[name] = mse_k
        fig_kcomp.add_trace(go.Scatter(x=X_grid, y=Y_k, mode='lines', name=f'{name} (MSE={mse_k:.4f})',
                                        line=dict(width=2, color=kernel_colors[name])))

    fig_kcomp.update_layout(title=f"Perbandingan Fungsi Kernel (h={optimal_bw:.4f})",
                             xaxis_title=x_var, yaxis_title=y_var, height=500)
    st.plotly_chart(fig_kcomp, use_container_width=True)

    kernel_comp_df = pd.DataFrame({
        'Fungsi Kernel': list(kernel_mse.keys()),
        'MSE': list(kernel_mse.values()),
        'Ranking': pd.Series(list(kernel_mse.values())).rank().astype(int).tolist()
    }).sort_values('MSE').round(6)
    st.dataframe(kernel_comp_df, use_container_width=True, hide_index=True)

# ============================
# 8. RESIDUAL DIAGNOSTICS
# ============================
st.header("8️⃣ Diagnostik Residual")
residuals = Y_sorted - Y_fit

met_r1, met_r2, met_r3, met_r4 = st.columns(4)
mse = np.mean(residuals**2)
mae = np.mean(np.abs(residuals))
rmse = np.sqrt(mse)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((Y_sorted - np.mean(Y_sorted))**2)
r2 = 1 - ss_res / ss_tot

met_r1.metric("MSE", f"{mse:.6f}")
met_r2.metric("RMSE", f"{rmse:.6f}")
met_r3.metric("MAE", f"{mae:.6f}")
met_r4.metric("R²", f"{r2:.4f}")

# --- 8a. Residual plots ---
st.subheader("8a. Plot Residual")
fig_resid = make_subplots(rows=1, cols=2,
                           subplot_titles=("Residual vs Fitted", "Residual vs X"))

fig_resid.add_trace(go.Scatter(x=Y_fit, y=residuals, mode='markers',
                                marker=dict(size=4, opacity=0.5, color='steelblue'), showlegend=False), row=1, col=1)
fig_resid.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

fig_resid.add_trace(go.Scatter(x=X_sorted, y=residuals, mode='markers',
                                marker=dict(size=4, opacity=0.5, color='darkorange'), showlegend=False), row=1, col=2)
fig_resid.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

fig_resid.update_xaxes(title_text="Fitted Values", row=1, col=1)
fig_resid.update_xaxes(title_text=x_var, row=1, col=2)
fig_resid.update_yaxes(title_text="Residuals", row=1, col=1)
fig_resid.update_yaxes(title_text="Residuals", row=1, col=2)
fig_resid.update_layout(height=400)
st.plotly_chart(fig_resid, use_container_width=True)

# --- 8b. Normality of residuals ---
st.subheader("8b. Uji Normalitas Residual")
tab_n1, tab_n2 = st.tabs(["📊 Visualisasi", "📋 Uji Statistik"])

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
        fig_qq.update_layout(title="QQ-Plot", xaxis_title="Theoretical Quantiles",
                              yaxis_title="Sample Quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)

with tab_n2:
    if len(residuals) <= 5000:
        sw_stat, sw_p = stats.shapiro(residuals)
    else:
        sw_stat, sw_p = np.nan, np.nan
    ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
    jb_stat, jb_p = stats.jarque_bera(residuals)
    ad_result = stats.anderson(residuals, dist='norm')

    norm_df = pd.DataFrame({
        'Uji': ['Shapiro-Wilk', 'Kolmogorov-Smirnov', 'Jarque-Bera', 'Anderson-Darling'],
        'Statistik': [sw_stat, ks_stat, jb_stat, ad_result.statistic],
        'p-value': [sw_p, ks_p, jb_p, np.nan],
        'Keputusan (α=0.05)': [
            '✅ Normal' if sw_p > 0.05 else '❌ Tidak Normal',
            '✅ Normal' if ks_p > 0.05 else '❌ Tidak Normal',
            '✅ Normal' if jb_p > 0.05 else '❌ Tidak Normal',
            '✅ Normal' if ad_result.statistic < ad_result.critical_values[2] else '❌ Tidak Normal'
        ]
    }).round(6)
    st.dataframe(norm_df, use_container_width=True, hide_index=True)

# --- 8c. Homoscedasticity ---
st.subheader("8c. Uji Homoskedastisitas Residual")

# Simple test: correlation between |residuals| and X
abs_resid = np.abs(residuals)
spearman_stat, spearman_p = stats.spearmanr(X_sorted, abs_resid)
pearson_stat, pearson_p = stats.pearsonr(X_sorted, abs_resid)

# Breusch-Pagan-like test on nonparametric residuals
X_bp = sm_api.add_constant(X_sorted)
bp_model = OLS(residuals**2, X_bp).fit()
bp_r2 = bp_model.rsquared
bp_stat = len(residuals) * bp_r2
bp_p = 1 - stats.chi2.cdf(bp_stat, 1)

het_df = pd.DataFrame({
    'Uji': ['Breusch-Pagan (residual²)', 'Spearman (|residual| vs X)', 'Pearson (|residual| vs X)'],
    'Statistik': [bp_stat, spearman_stat, pearson_stat],
    'p-value': [bp_p, spearman_p, pearson_p],
    'Keputusan (α=0.05)': [
        '✅ Homoskedastis' if bp_p > 0.05 else '❌ Heteroskedastis',
        '✅ Homoskedastis' if spearman_p > 0.05 else '❌ Heteroskedastis',
        '✅ Homoskedastis' if pearson_p > 0.05 else '❌ Heteroskedastis'
    ]
}).round(6)
st.dataframe(het_df, use_container_width=True, hide_index=True)

# --- 8d. Independence ---
st.subheader("8d. Uji Independensi Residual")

from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

dw_stat = durbin_watson(residuals)
lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
lb_stat = lb_result['lb_stat'].values[0]
lb_p = lb_result['lb_pvalue'].values[0]

# Runs test
median_resid = np.median(residuals)
binary_resid = (residuals >= median_resid).astype(int)
runs = 1 + np.sum(np.diff(binary_resid) != 0)
n1 = np.sum(binary_resid)
n0 = len(binary_resid) - n1
runs_mean = (2 * n1 * n0) / (n1 + n0) + 1
runs_var = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0)**2 * (n1 + n0 - 1))
runs_z = (runs - runs_mean) / np.sqrt(runs_var) if runs_var > 0 else 0
runs_p = 2 * (1 - stats.norm.cdf(np.abs(runs_z)))

indep_df = pd.DataFrame({
    'Uji': ['Durbin-Watson', 'Ljung-Box (lag=10)', 'Runs Test'],
    'Statistik': [dw_stat, lb_stat, runs_z],
    'p-value': [np.nan, lb_p, runs_p],
    'Keputusan': [
        '✅ Independen' if 1.5 <= dw_stat <= 2.5 else '❌ Ada autokorelasi',
        '✅ Independen' if lb_p > 0.05 else '❌ Ada autokorelasi',
        '✅ Independen' if runs_p > 0.05 else '❌ Tidak independen (ada pola)'
    ]
}).round(4)
st.dataframe(indep_df, use_container_width=True, hide_index=True)

# ACF Plot
from statsmodels.tsa.stattools import acf
acf_vals = acf(residuals, nlags=25, fft=True)
fig_acf = go.Figure()
for i in range(len(acf_vals)):
    fig_acf.add_trace(go.Bar(x=[i], y=[acf_vals[i]], marker_color='steelblue', showlegend=False, width=0.3))
ci = 1.96 / np.sqrt(len(residuals))
fig_acf.add_hline(y=ci, line_dash="dash", line_color="red")
fig_acf.add_hline(y=-ci, line_dash="dash", line_color="red")
fig_acf.add_hline(y=0, line_color="black")
fig_acf.update_layout(title="Autocorrelation Function (ACF) Residual",
                       xaxis_title="Lag", yaxis_title="ACF", height=350)
st.plotly_chart(fig_acf, use_container_width=True)

# ============================
# 9. MODEL COMPARISON
# ============================
st.header("9️⃣ Perbandingan Model")

# OLS metrics
ols_pred = ols_model.fittedvalues
ols_resid = Y - ols_pred
ols_mse = np.mean(ols_resid**2)
ols_rmse = np.sqrt(ols_mse)
ols_mae = np.mean(np.abs(ols_resid))
ols_r2 = ols_model.rsquared

# Kernel metrics
kr_mse = mse
kr_rmse = rmse
kr_mae = mae
kr_r2 = r2

comp_df = pd.DataFrame({
    'Metrik': ['MSE', 'RMSE', 'MAE', 'R²'],
    'Regresi Linier (OLS)': [ols_mse, ols_rmse, ols_mae, ols_r2],
    'Kernel Regression': [kr_mse, kr_rmse, kr_mae, kr_r2],
    'Model Terbaik': [
        '🏆 Kernel' if kr_mse < ols_mse else '🏆 OLS',
        '🏆 Kernel' if kr_rmse < ols_rmse else '🏆 OLS',
        '🏆 Kernel' if kr_mae < ols_mae else '🏆 OLS',
        '🏆 Kernel' if kr_r2 > ols_r2 else '🏆 OLS'
    ]
}).round(6)
st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ============================
# 10. SIGNIFICANCE TEST
# ============================
st.header("🔟 Uji Signifikansi Variabel")
st.markdown("Menggunakan bootstrap-based significance test dari `statsmodels`.")

try:
    with st.spinner("Menjalankan bootstrap significance test (mungkin butuh waktu)..."):
        sig = kr_model.sig_test(var_pos=[0], nboot=200)
    sig_df = pd.DataFrame({
        'Variabel': [x_var],
        'Test Statistic': [sig.test_stat],
        'p-value (Bootstrap)': [sig.sig],
        'Keputusan (α=0.05)': ['✅ Signifikan' if sig.sig < 0.05 else '❌ Tidak Signifikan']
    })
    st.dataframe(sig_df, use_container_width=True, hide_index=True)
except Exception as e:
    st.warning(f"Uji signifikansi tidak dapat dijalankan: {e}")

# ============================
# 11. PREDICTION
# ============================
st.header("1️⃣1️⃣ Prediksi Nilai Baru")

pred_col1, pred_col2 = st.columns([1, 2])
with pred_col1:
    x_pred = st.number_input(f"Masukkan nilai {x_var}",
                              value=float(np.median(X)),
                              step=float(np.std(X) / 10),
                              format="%.4f")

if st.button("🔮 Prediksi", type="primary"):
    y_pred, _ = kr_model.fit(np.array([x_pred]))

    # Bootstrap confidence interval
    n_boot = 500
    boot_preds = []
    for _ in range(n_boot):
        idx = np.random.choice(len(X_sorted), size=len(X_sorted), replace=True)
        X_boot, Y_boot = X_sorted[idx], Y_sorted[idx]
        kr_boot = KernelReg(endog=Y_boot, exog=X_boot, var_type='c',
                            reg_type=reg_type, bw=[optimal_bw])
        y_boot, _ = kr_boot.fit(np.array([x_pred]))
        boot_preds.append(y_boot[0])

    ci_lower = np.percentile(boot_preds, 2.5)
    ci_upper = np.percentile(boot_preds, 97.5)

    st.success(f"**Prediksi {y_var} = {y_pred[0]:.4f}**")
    pred_df = pd.DataFrame({
        'Metrik': ['Predicted Value', 'CI Lower (95%)', 'CI Upper (95%)', 'Bandwidth Used'],
        'Nilai': [y_pred[0], ci_lower, ci_upper, optimal_bw]
    }).round(4)
    st.dataframe(pred_df, use_container_width=True, hide_index=True)

# ============================
# 12. CONFIDENCE BAND
# ============================
st.header("1️⃣2️⃣ Confidence Band (Bootstrap)")

with st.expander("📊 Tampilkan Confidence Band", expanded=False):
    with st.spinner("Menghitung bootstrap confidence band..."):
        n_boot_band = 200
        boot_curves = np.zeros((n_boot_band, len(X_grid)))
        for b in range(n_boot_band):
            idx = np.random.choice(len(X_sorted), size=len(X_sorted), replace=True)
            X_b, Y_b = X_sorted[idx], Y_sorted[idx]
            kr_b = KernelReg(endog=Y_b, exog=X_b, var_type='c',
                             reg_type=reg_type, bw=[optimal_bw])
            y_b, _ = kr_b.fit(X_grid)
            boot_curves[b, :] = y_b

        ci_low = np.percentile(boot_curves, 2.5, axis=0)
        ci_up = np.percentile(boot_curves, 97.5, axis=0)

    fig_cb = go.Figure()
    fig_cb.add_trace(go.Scatter(x=X_sorted, y=Y_sorted, mode='markers', name='Data',
                                 marker=dict(size=3, opacity=0.3, color='gray')))
    fig_cb.add_trace(go.Scatter(x=np.concatenate([X_grid, X_grid[::-1]]),
                                 y=np.concatenate([ci_up, ci_low[::-1]]),
                                 fill='toself', fillcolor='rgba(255,0,0,0.15)',
                                 line=dict(color='rgba(255,0,0,0)'),
                                 name='95% Confidence Band'))
    fig_cb.add_trace(go.Scatter(x=X_grid, y=Y_grid, mode='lines', name='Estimasi Kernel',
                                 line=dict(color='red', width=3)))
    fig_cb.update_layout(title="Kurva Regresi Kernel dengan 95% Bootstrap Confidence Band",
                          xaxis_title=x_var, yaxis_title=y_var, height=500)
    st.plotly_chart(fig_cb, use_container_width=True)

# ============================
# 13. EXPORT
# ============================
st.header("1️⃣3️⃣ Ekspor Hasil")

col_e1, col_e2, col_e3 = st.columns(3)

with col_e1:
    summary_lines = [
        "=" * 60,
        "REGRESI NONPARAMETRIK - ESTIMATOR KERNEL",
        "=" * 60,
        f"Tipe Estimator  : {'Nadaraya-Watson (Local Constant)' if reg_type == 'lc' else 'Local Linear'}",
        f"Bandwidth (h)   : {optimal_bw:.6f}",
        f"BW Method       : {bw_method}",
        f"N Observasi     : {len(X_sorted)}",
        f"R² (Nonparam.)  : {r2:.6f}",
        f"MSE             : {mse:.6f}",
        f"RMSE            : {rmse:.6f}",
        f"MAE             : {mae:.6f}",
        "",
        "=" * 60,
        "PERBANDINGAN DENGAN OLS",
        "=" * 60,
        f"OLS R²   : {ols_r2:.6f}  vs  Kernel R²   : {r2:.6f}",
        f"OLS MSE  : {ols_mse:.6f}  vs  Kernel MSE  : {mse:.6f}",
        f"OLS RMSE : {ols_rmse:.6f}  vs  Kernel RMSE : {rmse:.6f}",
    ]
    st.download_button("📄 Download Summary (TXT)",
                       data="\n".join(summary_lines),
                       file_name="kernel_regression_summary.txt", mime="text/plain")

with col_e2:
    result_df = pd.DataFrame({
        x_var: X_sorted, y_var: Y_sorted,
        'Fitted': Y_fit, 'Residual': residuals,
        'Marginal_Effect': marginal_fx.flatten()
    })
    st.download_button("📊 Download Hasil (CSV)",
                       data=result_df.to_csv(index=False),
                       file_name="kernel_regression_results.csv", mime="text/csv")

with col_e3:
    curve_df = pd.DataFrame({'X_grid': X_grid, 'Y_estimated': Y_grid})
    st.download_button("📈 Download Kurva (CSV)",
                       data=curve_df.to_csv(index=False),
                       file_name="kernel_regression_curve.csv", mime="text/csv")

# ============================
# FOOTER
# ============================
st.markdown("---")
st.markdown("""
💡 **Catatan Metodologis:**
- **Nadaraya-Watson** (local constant): estimator kernel klasik yang menghitung rata-rata tertimbang.
- **Local Linear**: memperbaiki bias di batas data (boundary) dibandingkan Nadaraya-Watson.
- **Bandwidth** adalah parameter paling krusial — menentukan trade-off antara bias dan variansi.
- Confidence band dihitung menggunakan **bootstrap resampling** (bukan formula analitik).
""")
st.markdown("🔧 Dibangun dengan **Streamlit** + **Statsmodels** + **SciPy** + **Plotly** | Python 🐍")
