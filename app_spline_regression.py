import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline, splrep, splev
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Regresi Nonparametrik Spline (Least Square)",
    page_icon="📐",
    layout="wide"
)

st.title("📐 Regresi Nonparametrik — Least Square Spline")
st.markdown("""
Aplikasi lengkap untuk analisis **regresi nonparametrik** menggunakan metode **Least Square B-Spline**, 
dilengkapi pemilihan knot optimal (GCV), perbandingan orde spline, uji asumsi klasik, 
confidence band, dan diagnostik model.
""")

# ============================
# HELPER FUNCTIONS
# ============================
def build_bspline_basis(x, knots, degree):
    """Build B-spline basis matrix for given knots and degree."""
    n = len(x)
    # Augmented knot vector
    t = np.concatenate([
        np.repeat(knots[0], degree),
        knots,
        np.repeat(knots[-1], degree)
    ])
    n_basis = len(t) - degree - 1
    basis = np.zeros((n, n_basis))
    for i in range(n_basis):
        coefs = np.zeros(n_basis)
        coefs[i] = 1.0
        spl = BSpline(t, coefs, degree, extrapolate=False)
        basis[:, i] = spl(x)
    basis = np.nan_to_num(basis, nan=0.0)
    return basis, t

def fit_ls_spline(x, y, knots, degree):
    """Fit Least Square Spline and return model details."""
    basis, t = build_bspline_basis(x, knots, degree)
    basis_const = sm.add_constant(basis)
    model = OLS(y, basis_const).fit()
    y_hat = model.fittedvalues
    residuals = model.resid
    return model, y_hat, residuals, basis_const, t

def compute_gcv(x, y, knots, degree):
    """Compute Generalized Cross-Validation score."""
    basis, t = build_bspline_basis(x, knots, degree)
    basis_const = sm.add_constant(basis)
    n = len(y)
    try:
        hat_matrix = basis_const @ np.linalg.pinv(basis_const.T @ basis_const) @ basis_const.T
        y_hat = hat_matrix @ y
        residuals = y - y_hat
        trace_H = np.trace(hat_matrix)
        mse = np.mean(residuals**2)
        gcv = mse / ((1 - trace_H / n)**2)
        return gcv
    except:
        return np.inf

def compute_aic_bic(model, n):
    """Compute AIC and BIC."""
    k = model.df_model + 1
    ssr = np.sum(model.resid**2)
    aic = n * np.log(ssr / n) + 2 * k
    bic = n * np.log(ssr / n) + k * np.log(n)
    return aic, bic

def generate_knot_candidates(x, n_knots):
    """Generate evenly spaced interior knots."""
    quantiles = np.linspace(0, 100, n_knots + 2)[1:-1]
    return np.percentile(x, quantiles)

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
    X_demo = np.sort(np.random.uniform(0, 10, n))
    Y_demo = 2 * np.sin(1.5 * X_demo) + 0.8 * np.cos(3 * X_demo) + 0.3 * X_demo + np.random.normal(0, 0.5, n)
    df = pd.DataFrame({'X': X_demo, 'Y': Y_demo})
    st.sidebar.success("✅ Data demo dimuat (250 observasi, fungsi nonlinier)")
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

tab_d1, tab_d2 = st.tabs(["📋 Data", "📊 Statistik Deskriptif"])
with tab_d1:
    st.dataframe(df.head(30), use_container_width=True)
with tab_d2:
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

Y_raw = df[y_var].dropna().values
X_raw = df[x_var].dropna().values
min_len = min(len(Y_raw), len(X_raw))
Y_raw = Y_raw[:min_len]
X_raw = X_raw[:min_len]
sort_idx = np.argsort(X_raw)
X = X_raw[sort_idx]
Y = Y_raw[sort_idx]
n_obs = len(X)

# ============================
# 3. SCATTER & LINEARITY TEST
# ============================
st.header("3️⃣ Pola Hubungan Data")

fig_sc = px.scatter(x=X, y=Y, labels={'x': x_var, 'y': y_var},
                     title=f"Scatter Plot: {y_var} vs {x_var}", opacity=0.5)
z = np.polyfit(X, Y, 1)
p_line = np.poly1d(z)
x_range = np.linspace(X.min(), X.max(), 300)
fig_sc.add_trace(go.Scatter(x=x_range, y=p_line(x_range), mode='lines',
                              name='Regresi Linier', line=dict(color='red', dash='dash')))
st.plotly_chart(fig_sc, use_container_width=True)

# Ramsey RESET
X_ols = sm.add_constant(X)
ols_model = OLS(Y, X_ols).fit()
Y_hat_ols = ols_model.fittedvalues
X_reset = np.column_stack([X_ols, Y_hat_ols**2, Y_hat_ols**3])
ols_reset = OLS(Y, X_reset).fit()
f_reset = ((ols_model.ssr - ols_reset.ssr) / 2) / (ols_reset.ssr / (n_obs - X_reset.shape[1]))
p_reset = 1 - stats.f.cdf(f_reset, 2, n_obs - X_reset.shape[1])

st.subheader("Uji Linieritas — Ramsey RESET Test")
reset_df = pd.DataFrame({
    'Uji': ['Ramsey RESET'],
    'F-statistic': [round(f_reset, 4)],
    'p-value': [round(p_reset, 6)],
    'Keputusan (α=0.05)': ['❌ TIDAK linier → Gunakan nonparametrik' if p_reset < 0.05
                           else '✅ Linier → Regresi parametrik mungkin cukup']
})
st.dataframe(reset_df, use_container_width=True, hide_index=True)

# ============================
# 4. SPLINE SETTINGS
# ============================
st.header("4️⃣ Pengaturan Model Spline")

col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    spline_degree = st.selectbox("Orde Spline (degree)",
                                  options=[1, 2, 3, 4],
                                  format_func=lambda d: {1: "Linear (d=1)", 2: "Quadratic (d=2)",
                                                          3: "Cubic (d=3)", 4: "Quartic (d=4)"}[d],
                                  index=2)

with col_s2:
    knot_method = st.selectbox("Metode Pemilihan Knot",
                                ["auto_gcv", "manual"],
                                format_func=lambda x: "Otomatis (GCV Optimal)" if x == "auto_gcv" else "Manual")

with col_s3:
    if knot_method == "manual":
        n_knots = st.slider("Jumlah Knot Interior", min_value=1, max_value=20, value=5)
    else:
        max_search = st.slider("Maks. knot untuk pencarian GCV", min_value=1, max_value=25, value=15)

# ============================
# 5. KNOT SELECTION (GCV)
# ============================
st.header("5️⃣ Pemilihan Knot Optimal")

if knot_method == "auto_gcv":
    with st.spinner("Mencari jumlah knot optimal menggunakan GCV..."):
        gcv_scores = {}
        for nk in range(1, max_search + 1):
            knots_cand = generate_knot_candidates(X, nk)
            full_knots = np.concatenate([[X.min()], knots_cand, [X.max()]])
            gcv = compute_gcv(X, Y, full_knots, spline_degree)
            gcv_scores[nk] = gcv

        gcv_df = pd.DataFrame({
            'Jumlah Knot': list(gcv_scores.keys()),
            'GCV Score': list(gcv_scores.values())
        })

        optimal_nk = min(gcv_scores, key=gcv_scores.get)
        optimal_gcv = gcv_scores[optimal_nk]
        interior_knots = generate_knot_candidates(X, optimal_nk)

    # GCV Plot
    fig_gcv = go.Figure()
    fig_gcv.add_trace(go.Scatter(x=gcv_df['Jumlah Knot'], y=gcv_df['GCV Score'],
                                  mode='lines+markers', name='GCV Score',
                                  marker=dict(size=8)))
    fig_gcv.add_vline(x=optimal_nk, line_dash="dash", line_color="red",
                       annotation_text=f"Optimal: {optimal_nk} knot")
    fig_gcv.update_layout(title="Generalized Cross-Validation (GCV) vs Jumlah Knot",
                           xaxis_title="Jumlah Knot Interior",
                           yaxis_title="GCV Score", height=400)
    st.plotly_chart(fig_gcv, use_container_width=True)

    st.dataframe(gcv_df.round(6), use_container_width=True, hide_index=True)
    st.success(f"✅ Jumlah knot optimal: **{optimal_nk}** (GCV = {optimal_gcv:.6f})")
else:
    interior_knots = generate_knot_candidates(X, n_knots)
    optimal_nk = n_knots

# Display knot positions
full_knots = np.concatenate([[X.min()], interior_knots, [X.max()]])
knot_info = pd.DataFrame({
    'Knot ke-': [f"Batas Bawah"] + [f"Interior {i+1}" for i in range(len(interior_knots))] + [f"Batas Atas"],
    'Posisi': full_knots
}).round(4)

with st.expander(f"📍 Posisi Knot ({len(interior_knots)} knot interior)", expanded=False):
    st.dataframe(knot_info, use_container_width=True, hide_index=True)

# ============================
# 6. FIT SPLINE MODEL
# ============================
st.header("6️⃣ Hasil Estimasi Spline Regression")

with st.spinner("Mengestimasi model spline..."):
    model, Y_fit, residuals, basis_matrix, t_knots = fit_ls_spline(X, Y, full_knots, spline_degree)

# Metrics
ss_res = np.sum(residuals**2)
ss_tot = np.sum((Y - np.mean(Y))**2)
r2 = 1 - ss_res / ss_tot
adj_r2 = 1 - (1 - r2) * (n_obs - 1) / (n_obs - model.df_model - 1)
mse = np.mean(residuals**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(residuals))
aic_val, bic_val = compute_aic_bic(model, n_obs)

met1, met2, met3, met4 = st.columns(4)
met1.metric("R²", f"{r2:.4f}")
met2.metric("Adj. R²", f"{adj_r2:.4f}")
met3.metric("RMSE", f"{rmse:.6f}")
met4.metric("MAE", f"{mae:.6f}")

met5, met6, met7, met8 = st.columns(4)
met5.metric("MSE", f"{mse:.6f}")
met6.metric("AIC", f"{aic_val:.2f}")
met7.metric("BIC", f"{bic_val:.2f}")
met8.metric("Jumlah Knot Interior", f"{len(interior_knots)}")

# OLS Summary
with st.expander("📄 Ringkasan Model OLS (Basis Spline)", expanded=False):
    st.text(model.summary().as_text())

# ============================
# 6a. REGRESSION CURVE
# ============================
st.subheader("📈 Kurva Regresi Spline")

X_grid = np.linspace(X.min(), X.max(), 500)
basis_grid, _ = build_bspline_basis(X_grid, full_knots, spline_degree)
basis_grid_const = sm.add_constant(basis_grid)
Y_grid = model.predict(basis_grid_const)

fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(x=X, y=Y, mode='markers', name='Data Observasi',
                                marker=dict(size=4, opacity=0.4, color='steelblue')))
fig_curve.add_trace(go.Scatter(x=X_grid, y=Y_grid, mode='lines',
                                name=f'Spline (d={spline_degree}, knot={len(interior_knots)})',
                                line=dict(color='red', width=3)))
fig_curve.add_trace(go.Scatter(x=x_range, y=p_line(x_range), mode='lines',
                                name='Regresi Linier (OLS)',
                                line=dict(color='green', dash='dash', width=2)))

# Mark knot positions
for k in interior_knots:
    fig_curve.add_vline(x=k, line_dash="dot", line_color="orange", opacity=0.5)

fig_curve.update_layout(title="Kurva Regresi Spline vs OLS (garis vertikal = posisi knot)",
                         xaxis_title=x_var, yaxis_title=y_var, height=500)
st.plotly_chart(fig_curve, use_container_width=True)

# ============================
# 6b. BASIS FUNCTIONS
# ============================
st.subheader("📊 Fungsi Basis B-Spline")

with st.expander("Tampilkan Fungsi Basis", expanded=False):
    basis_plot, _ = build_bspline_basis(X_grid, full_knots, spline_degree)
    fig_basis = go.Figure()
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    for i in range(basis_plot.shape[1]):
        fig_basis.add_trace(go.Scatter(x=X_grid, y=basis_plot[:, i], mode='lines',
                                        name=f'B_{i+1}(x)', line=dict(width=2, color=colors[i % len(colors)])))
    for k in interior_knots:
        fig_basis.add_vline(x=k, line_dash="dot", line_color="gray", opacity=0.3)
    fig_basis.update_layout(title=f"Fungsi Basis B-Spline (degree={spline_degree})",
                             xaxis_title=x_var, yaxis_title="Basis Value", height=450)
    st.plotly_chart(fig_basis, use_container_width=True)

# ============================
# 7. DEGREE COMPARISON
# ============================
st.header("7️⃣ Perbandingan Orde Spline")

degree_results = {}
for d in [1, 2, 3, 4]:
    try:
        m, yh, res, _, _ = fit_ls_spline(X, Y, full_knots, d)
        ss_r = np.sum(res**2)
        r2_d = 1 - ss_r / ss_tot
        mse_d = np.mean(res**2)
        aic_d, bic_d = compute_aic_bic(m, n_obs)
        degree_results[d] = {'R²': r2_d, 'MSE': mse_d, 'AIC': aic_d, 'BIC': bic_d, 'df': m.df_model}
    except:
        pass

tab_dc1, tab_dc2 = st.tabs(["📈 Visualisasi Kurva", "📋 Tabel Perbandingan"])

with tab_dc1:
    fig_deg = go.Figure()
    fig_deg.add_trace(go.Scatter(x=X, y=Y, mode='markers', name='Data',
                                  marker=dict(size=3, opacity=0.3, color='gray')))
    deg_colors = {1: 'blue', 2: 'green', 3: 'red', 4: 'purple'}
    deg_names = {1: 'Linear (d=1)', 2: 'Quadratic (d=2)', 3: 'Cubic (d=3)', 4: 'Quartic (d=4)'}
    for d in degree_results:
        try:
            b_g, _ = build_bspline_basis(X_grid, full_knots, d)
            b_gc = sm.add_constant(b_g)
            m_d, _, _, _, _ = fit_ls_spline(X, Y, full_knots, d)
            y_g = m_d.predict(b_gc)
            fig_deg.add_trace(go.Scatter(x=X_grid, y=y_g, mode='lines',
                                          name=f"{deg_names[d]} (R²={degree_results[d]['R²']:.4f})",
                                          line=dict(color=deg_colors[d], width=2)))
        except:
            pass
    fig_deg.update_layout(title=f"Perbandingan Orde Spline ({len(interior_knots)} knot interior)",
                           xaxis_title=x_var, yaxis_title=y_var, height=500)
    st.plotly_chart(fig_deg, use_container_width=True)

with tab_dc2:
    deg_df = pd.DataFrame(degree_results).T
    deg_df.index = [f"Degree {d}" for d in deg_df.index]
    deg_df = deg_df.round(6)
    best_row = deg_df['AIC'].idxmin()
    st.dataframe(deg_df, use_container_width=True)
    st.info(f"🏆 Orde terbaik berdasarkan AIC: **{best_row}**")

# ============================
# 8. KNOT SENSITIVITY
# ============================
st.header("8️⃣ Sensitivitas Jumlah Knot")

knot_range_vis = [1, 2, 3, 5, 8, 12, 18]
fig_ksens = go.Figure()
fig_ksens.add_trace(go.Scatter(x=X, y=Y, mode='markers', name='Data',
                                marker=dict(size=3, opacity=0.3, color='gray')))

ksens_colors = px.colors.qualitative.Set1
for idx, nk in enumerate(knot_range_vis):
    try:
        ik = generate_knot_candidates(X, nk)
        fk = np.concatenate([[X.min()], ik, [X.max()]])
        mk, _, _, _, _ = fit_ls_spline(X, Y, fk, spline_degree)
        bk, _ = build_bspline_basis(X_grid, fk, spline_degree)
        bkc = sm.add_constant(bk)
        yk = mk.predict(bkc)
        is_opt = (nk == optimal_nk)
        fig_ksens.add_trace(go.Scatter(
            x=X_grid, y=yk, mode='lines',
            name=f'{nk} knot' + (' ★' if is_opt else ''),
            line=dict(width=4 if is_opt else 1.5, color=ksens_colors[idx % len(ksens_colors)])
        ))
    except:
        pass

fig_ksens.update_layout(title=f"Pengaruh Jumlah Knot pada Kurva Spline (degree={spline_degree})",
                          xaxis_title=x_var, yaxis_title=y_var, height=500)
st.plotly_chart(fig_ksens, use_container_width=True)

st.markdown("""
> - **Knot sedikit** → kurva halus, risiko underfitting  
> - **Knot banyak** → kurva fleksibel, risiko overfitting  
> - **Knot optimal** dipilih menggunakan GCV (Generalized Cross-Validation)
""")

# ============================
# 9. RESIDUAL DIAGNOSTICS
# ============================
st.header("9️⃣ Diagnostik Residual")

met_r1, met_r2, met_r3, met_r4 = st.columns(4)
met_r1.metric("MSE", f"{mse:.6f}")
met_r2.metric("RMSE", f"{rmse:.6f}")
met_r3.metric("MAE", f"{mae:.6f}")
met_r4.metric("R²", f"{r2:.4f}")

# --- 9a. Residual Plots ---
st.subheader("9a. Plot Residual")
fig_resid = make_subplots(rows=1, cols=2,
                           subplot_titles=("Residual vs Fitted", "Residual vs X"))

fig_resid.add_trace(go.Scatter(x=Y_fit, y=residuals, mode='markers',
                                marker=dict(size=4, opacity=0.5, color='steelblue'),
                                showlegend=False), row=1, col=1)
fig_resid.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

fig_resid.add_trace(go.Scatter(x=X, y=residuals, mode='markers',
                                marker=dict(size=4, opacity=0.5, color='darkorange'),
                                showlegend=False), row=1, col=2)
fig_resid.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

fig_resid.update_xaxes(title_text="Fitted Values", row=1, col=1)
fig_resid.update_xaxes(title_text=x_var, row=1, col=2)
fig_resid.update_yaxes(title_text="Residuals", row=1, col=1)
fig_resid.update_yaxes(title_text="Residuals", row=1, col=2)
fig_resid.update_layout(height=400)
st.plotly_chart(fig_resid, use_container_width=True)

# --- 9b. Normality ---
st.subheader("9b. Uji Normalitas Residual")
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
        fig_qq.update_layout(title="QQ-Plot Residual",
                              xaxis_title="Theoretical Quantiles",
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

# --- 9c. Homoscedasticity ---
st.subheader("9c. Uji Homoskedastisitas")

bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(residuals, basis_matrix)

sp_stat, sp_p = stats.spearmanr(X, np.abs(residuals))
pe_stat, pe_p = stats.pearsonr(X, np.abs(residuals))

# Goldfeld-Quandt
mid = n_obs // 2
gq_f = np.sum(residuals[mid:]**2) / np.sum(residuals[:mid]**2)
gq_p = 1 - stats.f.cdf(gq_f, mid, mid)

het_df = pd.DataFrame({
    'Uji': ['Breusch-Pagan', 'Goldfeld-Quandt', 'Spearman (|resid| vs X)', 'Pearson (|resid| vs X)'],
    'Statistik': [bp_stat, gq_f, sp_stat, pe_stat],
    'p-value': [bp_p, gq_p, sp_p, pe_p],
    'Keputusan (α=0.05)': [
        '✅ Homoskedastis' if bp_p > 0.05 else '❌ Heteroskedastis',
        '✅ Homoskedastis' if gq_p > 0.05 else '❌ Heteroskedastis',
        '✅ Homoskedastis' if sp_p > 0.05 else '❌ Heteroskedastis',
        '✅ Homoskedastis' if pe_p > 0.05 else '❌ Heteroskedastis'
    ]
}).round(6)
st.dataframe(het_df, use_container_width=True, hide_index=True)

# --- 9d. Independence ---
st.subheader("9d. Uji Independensi Residual")

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
        '✅ Independen' if runs_p > 0.05 else '❌ Tidak independen'
    ]
}).round(4)
st.dataframe(indep_df, use_container_width=True, hide_index=True)

# ACF
acf_vals = acf(residuals, nlags=25, fft=True)
fig_acf = go.Figure()
for i in range(len(acf_vals)):
    fig_acf.add_trace(go.Bar(x=[i], y=[acf_vals[i]], marker_color='steelblue',
                              showlegend=False, width=0.3))
ci_acf = 1.96 / np.sqrt(n_obs)
fig_acf.add_hline(y=ci_acf, line_dash="dash", line_color="red")
fig_acf.add_hline(y=-ci_acf, line_dash="dash", line_color="red")
fig_acf.add_hline(y=0, line_color="black")
fig_acf.update_layout(title="ACF Residual", xaxis_title="Lag", yaxis_title="ACF", height=350)
st.plotly_chart(fig_acf, use_container_width=True)

# ============================
# 10. MODEL COMPARISON
# ============================
st.header("🔟 Perbandingan Model: Spline vs OLS")

ols_resid = ols_model.resid
ols_mse = np.mean(ols_resid**2)
ols_rmse = np.sqrt(ols_mse)
ols_mae = np.mean(np.abs(ols_resid))
ols_r2 = ols_model.rsquared
ols_aic, ols_bic = compute_aic_bic(ols_model, n_obs)

comp_df = pd.DataFrame({
    'Metrik': ['R²', 'Adj. R²', 'MSE', 'RMSE', 'MAE', 'AIC', 'BIC'],
    'Regresi Linier (OLS)': [ols_r2, ols_model.rsquared_adj, ols_mse, ols_rmse, ols_mae, ols_aic, ols_bic],
    'Spline Regression': [r2, adj_r2, mse, rmse, mae, aic_val, bic_val],
    'Model Terbaik': [
        '🏆 Spline' if r2 > ols_r2 else '🏆 OLS',
        '🏆 Spline' if adj_r2 > ols_model.rsquared_adj else '🏆 OLS',
        '🏆 Spline' if mse < ols_mse else '🏆 OLS',
        '🏆 Spline' if rmse < ols_rmse else '🏆 OLS',
        '🏆 Spline' if mae < ols_mae else '🏆 OLS',
        '🏆 Spline' if aic_val < ols_aic else '🏆 OLS',
        '🏆 Spline' if bic_val < ols_bic else '🏆 OLS'
    ]
}).round(6)
st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ============================
# 11. SIGNIFICANCE TEST
# ============================
st.header("1️⃣1️⃣ Uji Signifikansi Model")

# F-test: Spline vs Null (intercept-only)
ss_null = ss_tot
df_null = n_obs - 1
df_model = n_obs - model.df_model - 1
f_stat = ((ss_null - ss_res) / model.df_model) / (ss_res / df_model)
f_p = 1 - stats.f.cdf(f_stat, model.df_model, df_model)

# F-test: Spline vs OLS
ss_ols = np.sum(ols_resid**2)
df_diff = model.df_model - ols_model.df_model
if df_diff > 0:
    f_vs_ols = ((ss_ols - ss_res) / df_diff) / (ss_res / df_model)
    f_vs_ols_p = 1 - stats.f.cdf(f_vs_ols, df_diff, df_model)
else:
    f_vs_ols, f_vs_ols_p = np.nan, np.nan

sig_df = pd.DataFrame({
    'Uji': ['F-test (Model vs Null)', 'F-test (Spline vs OLS)'],
    'F-statistic': [f_stat, f_vs_ols],
    'df1': [model.df_model, df_diff],
    'df2': [df_model, df_model],
    'p-value': [f_p, f_vs_ols_p],
    'Keputusan (α=0.05)': [
        '✅ Model Signifikan' if f_p < 0.05 else '❌ Tidak Signifikan',
        '✅ Spline lebih baik dari OLS' if (not np.isnan(f_vs_ols_p) and f_vs_ols_p < 0.05)
        else '❌ Spline tidak signifikan lebih baik'
    ]
}).round(6)
st.dataframe(sig_df, use_container_width=True, hide_index=True)

# ============================
# 12. CONFIDENCE BAND (Bootstrap)
# ============================
st.header("1️⃣2️⃣ Confidence Band (Bootstrap)")

with st.expander("📊 Tampilkan Confidence Band", expanded=False):
    with st.spinner("Menghitung bootstrap confidence band..."):
        n_boot = 200
        boot_curves = np.zeros((n_boot, len(X_grid)))

        for b in range(n_boot):
            idx = np.random.choice(n_obs, size=n_obs, replace=True)
            X_b, Y_b = X[idx], Y[idx]
            s_idx = np.argsort(X_b)
            X_b, Y_b = X_b[s_idx], Y_b[s_idx]
            try:
                m_b, _, _, _, _ = fit_ls_spline(X_b, Y_b, full_knots, spline_degree)
                bg, _ = build_bspline_basis(X_grid, full_knots, spline_degree)
                bgc = sm.add_constant(bg)
                boot_curves[b, :] = m_b.predict(bgc)
            except:
                boot_curves[b, :] = Y_grid

        ci_low = np.percentile(boot_curves, 2.5, axis=0)
        ci_up = np.percentile(boot_curves, 97.5, axis=0)

    fig_cb = go.Figure()
    fig_cb.add_trace(go.Scatter(x=X, y=Y, mode='markers', name='Data',
                                 marker=dict(size=3, opacity=0.3, color='gray')))
    fig_cb.add_trace(go.Scatter(
        x=np.concatenate([X_grid, X_grid[::-1]]),
        y=np.concatenate([ci_up, ci_low[::-1]]),
        fill='toself', fillcolor='rgba(255,0,0,0.15)',
        line=dict(color='rgba(255,0,0,0)'),
        name='95% Confidence Band'))
    fig_cb.add_trace(go.Scatter(x=X_grid, y=Y_grid, mode='lines',
                                 name='Estimasi Spline', line=dict(color='red', width=3)))
    for k in interior_knots:
        fig_cb.add_vline(x=k, line_dash="dot", line_color="orange", opacity=0.4)
    fig_cb.update_layout(title="Kurva Spline dengan 95% Bootstrap Confidence Band",
                          xaxis_title=x_var, yaxis_title=y_var, height=500)
    st.plotly_chart(fig_cb, use_container_width=True)

# ============================
# 13. INFLUENTIAL OBSERVATIONS
# ============================
st.header("1️⃣3️⃣ Deteksi Observasi Berpengaruh")

influence = model.get_influence()
inf_summary = influence.summary_frame()
cooks_d = inf_summary['cooks_d']
leverage = inf_summary['hat_diag']
threshold_cook = 4 / n_obs

tab_i1, tab_i2 = st.tabs(["📊 Cook's Distance", "📋 Detail"])

with tab_i1:
    fig_cook = px.bar(x=cooks_d.index, y=cooks_d.values,
                       title=f"Cook's Distance (threshold = {threshold_cook:.4f})",
                       labels={'x': 'Observasi', 'y': "Cook's Distance"})
    fig_cook.add_hline(y=threshold_cook, line_dash="dash", line_color="red")
    st.plotly_chart(fig_cook, use_container_width=True)
    influential = cooks_d[cooks_d > threshold_cook]
    st.info(f"📍 Observasi berpengaruh: **{len(influential)}** dari {n_obs}")

with tab_i2:
    inf_detail = pd.DataFrame({
        'Obs': range(n_obs),
        x_var: X, y_var: Y,
        'Fitted': Y_fit, 'Residual': residuals,
        'Leverage': leverage.values,
        "Cook's D": cooks_d.values,
        'Std. Residual': inf_summary['standard_resid'].values
    }).round(6)
    st.dataframe(inf_detail.nlargest(20, "Cook's D"), use_container_width=True, hide_index=True)

# ============================
# 14. PREDICTION
# ============================
st.header("1️⃣4️⃣ Prediksi Nilai Baru")

x_pred = st.number_input(f"Masukkan nilai {x_var}",
                          value=float(np.median(X)),
                          step=float(np.std(X) / 10),
                          format="%.4f")

if st.button("🔮 Prediksi", type="primary"):
    b_pred, _ = build_bspline_basis(np.array([x_pred]), full_knots, spline_degree)
    b_pred_c = sm.add_constant(b_pred, has_constant='add')
    y_pred = model.predict(b_pred_c)[0]

    # Bootstrap CI
    n_boot_pred = 500
    boot_preds = []
    for _ in range(n_boot_pred):
        idx = np.random.choice(n_obs, size=n_obs, replace=True)
        X_b, Y_b = X[idx], Y[idx]
        si = np.argsort(X_b)
        X_b, Y_b = X_b[si], Y_b[si]
        try:
            m_b, _, _, _, _ = fit_ls_spline(X_b, Y_b, full_knots, spline_degree)
            bp, _ = build_bspline_basis(np.array([x_pred]), full_knots, spline_degree)
            bpc = sm.add_constant(bp, has_constant='add')
            boot_preds.append(m_b.predict(bpc)[0])
        except:
            pass

    ci_l = np.percentile(boot_preds, 2.5)
    ci_u = np.percentile(boot_preds, 97.5)

    st.success(f"**Prediksi {y_var} = {y_pred:.4f}**")
    pred_df = pd.DataFrame({
        'Metrik': ['Predicted Value', 'CI Lower (95%)', 'CI Upper (95%)',
                   'Bandwidth (degree)', 'Jumlah Knot Interior'],
        'Nilai': [y_pred, ci_l, ci_u, spline_degree, len(interior_knots)]
    }).round(4)
    st.dataframe(pred_df, use_container_width=True, hide_index=True)

# ============================
# 15. EXPORT
# ============================
st.header("1️⃣5️⃣ Ekspor Hasil")

col_e1, col_e2, col_e3 = st.columns(3)

with col_e1:
    summary_lines = [
        "=" * 60,
        "REGRESI NONPARAMETRIK - LEAST SQUARE SPLINE",
        "=" * 60,
        f"Orde Spline (degree)   : {spline_degree}",
        f"Jumlah Knot Interior   : {len(interior_knots)}",
        f"Posisi Knot Interior   : {np.round(interior_knots, 4).tolist()}",
        f"N Observasi            : {n_obs}",
        f"R²                     : {r2:.6f}",
        f"Adj. R²                : {adj_r2:.6f}",
        f"MSE                    : {mse:.6f}",
        f"RMSE                   : {rmse:.6f}",
        f"MAE                    : {mae:.6f}",
        f"AIC                    : {aic_val:.4f}",
        f"BIC                    : {bic_val:.4f}",
        "",
        "=" * 60,
        "PERBANDINGAN DENGAN OLS",
        "=" * 60,
        f"OLS R²   : {ols_r2:.6f}  vs  Spline R²   : {r2:.6f}",
        f"OLS MSE  : {ols_mse:.6f}  vs  Spline MSE  : {mse:.6f}",
        f"OLS AIC  : {ols_aic:.4f}  vs  Spline AIC  : {aic_val:.4f}",
        "",
        "=" * 60,
        "UJI SIGNIFIKANSI",
        "=" * 60,
        f"F-test (Model vs Null) : F={f_stat:.4f}, p={f_p:.6f}",
        f"F-test (Spline vs OLS) : F={f_vs_ols:.4f}, p={f_vs_ols_p:.6f}",
    ]
    st.download_button("📄 Download Summary (TXT)",
                       data="\n".join(summary_lines),
                       file_name="spline_regression_summary.txt", mime="text/plain")

with col_e2:
    result_df = pd.DataFrame({
        x_var: X, y_var: Y,
        'Fitted': Y_fit, 'Residual': residuals
    })
    st.download_button("📊 Download Hasil (CSV)",
                       data=result_df.to_csv(index=False),
                       file_name="spline_regression_results.csv", mime="text/csv")

with col_e3:
    curve_df = pd.DataFrame({'X_grid': X_grid, 'Y_estimated': Y_grid})
    st.download_button("📈 Download Kurva (CSV)",
                       data=curve_df.to_csv(index=False),
                       file_name="spline_regression_curve.csv", mime="text/csv")

# ============================
# FOOTER
# ============================
st.markdown("---")
st.markdown("""
💡 **Catatan Metodologis:**
- **B-Spline** adalah basis fungsi polinomial yang memiliki sifat compact support dan partisi kesatuan.
- **Knot** adalah titik-titik dimana polinomial bertemu — jumlah dan posisi knot menentukan fleksibilitas model.
- **GCV (Generalized Cross-Validation)** digunakan untuk memilih jumlah knot optimal secara data-driven.
- **Orde/degree** menentukan derajat polinomial: 1 (linier), 2 (kuadratik), 3 (kubik), 4 (kuartik).
- Model ini merupakan **regresi linier dalam basis spline** — koefisien diestimasi dengan metode **Least Square**.
""")
st.markdown("🔧 Dibangun dengan **Streamlit** + **SciPy** + **Statsmodels** + **Plotly** | Python 🐍")
