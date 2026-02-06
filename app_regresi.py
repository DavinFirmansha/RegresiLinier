import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Analisis Regresi Linier Berganda",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Analisis Regresi Linier Berganda")
st.markdown("Aplikasi web lengkap untuk analisis regresi linier berganda dengan output statistik dan uji asumsi klasik.")

# ============================
# SIDEBAR - UPLOAD DATA
# ============================
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx", "xls"])

# Demo data option
use_demo = st.sidebar.checkbox("Gunakan data demo", value=True if uploaded_file is None else False)

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    use_demo = False
elif use_demo:
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(50, 10, n)
    x2 = np.random.normal(30, 5, n)
    x3 = np.random.normal(20, 8, n)
    x4 = np.random.choice([0, 1], size=n, p=[0.4, 0.6])
    noise = np.random.normal(0, 5, n)
    y = 10 + 2.5 * x1 - 1.8 * x2 + 0.7 * x3 + 15 * x4 + noise
    df = pd.DataFrame({
        'Penjualan': y,
        'Harga': x1,
        'Diskon': x2,
        'Stok': x3,
        'Promosi': x4
    })
    st.sidebar.success("✅ Data demo dimuat (200 baris, 5 kolom)")
else:
    st.warning("⚠️ Silakan upload data atau gunakan data demo.")
    st.stop()

# ============================
# DATA PREVIEW
# ============================
st.header("1️⃣ Pratinjau Data")
col1, col2, col3 = st.columns(3)
col1.metric("Jumlah Baris", df.shape[0])
col2.metric("Jumlah Kolom", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())

with st.expander("📋 Lihat Data", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("📊 Statistik Deskriptif"):
    st.dataframe(df.describe().T.round(4), use_container_width=True)

# ============================
# VARIABLE SELECTION
# ============================
st.header("2️⃣ Pilih Variabel")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("❌ Data harus memiliki minimal 2 kolom numerik.")
    st.stop()

col_left, col_right = st.columns(2)
with col_left:
    y_var = st.selectbox("🎯 Variabel Dependen (Y)", numeric_cols, index=0)
with col_right:
    x_options = [c for c in numeric_cols if c != y_var]
    x_vars = st.multiselect("📌 Variabel Independen (X)", x_options, default=x_options)

if len(x_vars) < 1:
    st.warning("⚠️ Pilih minimal 1 variabel independen.")
    st.stop()

# ============================
# CORRELATION ANALYSIS
# ============================
st.header("3️⃣ Analisis Korelasi")
all_vars = [y_var] + x_vars
corr_matrix = df[all_vars].corr().round(4)

fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Matriks Korelasi Pearson"
)
fig_corr.update_layout(width=700, height=500)
st.plotly_chart(fig_corr, use_container_width=True)

# ============================
# REGRESSION MODEL (OLS)
# ============================
st.header("4️⃣ Hasil Regresi OLS")

# Prepare data
Y = df[y_var].dropna()
X = df[x_vars].dropna()
common_idx = Y.index.intersection(X.index)
Y = Y.loc[common_idx]
X = X.loc[common_idx]
X_const = sm.add_constant(X)

# Fit model
model = sm.OLS(Y, X_const).fit()

# Summary
with st.expander("📄 Ringkasan Model Lengkap (OLS Summary)", expanded=True):
    st.text(model.summary().as_text())

# Key metrics
st.subheader("📈 Metrik Utama")
met1, met2, met3, met4 = st.columns(4)
met1.metric("R-squared", f"{model.rsquared:.4f}")
met2.metric("Adj. R-squared", f"{model.rsquared_adj:.4f}")
met3.metric("F-statistic", f"{model.fvalue:.4f}")
met4.metric("Prob (F-stat)", f"{model.f_pvalue:.6f}")

met5, met6, met7, met8 = st.columns(4)
met5.metric("AIC", f"{model.aic:.2f}")
met6.metric("BIC", f"{model.bic:.2f}")
met7.metric("Log-Likelihood", f"{model.llf:.2f}")
met8.metric("Durbin-Watson", f"{durbin_watson(model.resid):.4f}")

# Coefficients table
st.subheader("📊 Tabel Koefisien Regresi")
coef_df = pd.DataFrame({
    'Variabel': model.params.index,
    'Koefisien (B)': model.params.values,
    'Std. Error': model.bse.values,
    't-value': model.tvalues.values,
    'p-value': model.pvalues.values,
    'CI Lower (95%)': model.conf_int()[0].values,
    'CI Upper (95%)': model.conf_int()[1].values,
    'Signifikan (α=0.05)': ['✅ Ya' if p < 0.05 else '❌ Tidak' for p in model.pvalues.values]
}).round(6)
st.dataframe(coef_df, use_container_width=True, hide_index=True)

# Regression equation
st.subheader("📝 Persamaan Regresi")
eq_parts = [f"{model.params['const']:.4f}"]
for var in x_vars:
    coef = model.params[var]
    sign = "+" if coef >= 0 else "-"
    eq_parts.append(f" {sign} {abs(coef):.4f} × {var}")
equation = f"**{y_var}** = {''.join(eq_parts)}"
st.markdown(equation)

# ============================
# ASSUMPTION TESTS
# ============================
st.header("5️⃣ Uji Asumsi Klasik")
residuals = model.resid
fitted = model.fittedvalues

# --- 5a. Normality ---
st.subheader("5a. Uji Normalitas Residual")
tab_norm1, tab_norm2 = st.tabs(["📊 Histogram & QQ-Plot", "📋 Uji Statistik"])

with tab_norm1:
    col_hist, col_qq = st.columns(2)
    with col_hist:
        fig_hist = px.histogram(
            x=residuals, nbins=30, 
            title="Histogram Residual",
            labels={'x': 'Residual', 'y': 'Frekuensi'},
            marginal="box"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_qq:
        qq = stats.probplot(residuals, dist="norm")
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Data'))
        fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0]*qq[0][0]+qq[1][1], mode='lines', name='Garis Normal', line=dict(color='red')))
        fig_qq.update_layout(title="QQ-Plot Residual", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)

with tab_norm2:
    # Shapiro-Wilk
    if len(residuals) <= 5000:
        sw_stat, sw_p = stats.shapiro(residuals)
    else:
        sw_stat, sw_p = np.nan, np.nan

    # Kolmogorov-Smirnov
    ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))

    # Jarque-Bera
    jb_stat, jb_p = stats.jarque_bera(residuals)

    norm_df = pd.DataFrame({
        'Uji': ['Shapiro-Wilk', 'Kolmogorov-Smirnov', 'Jarque-Bera'],
        'Statistik': [sw_stat, ks_stat, jb_stat],
        'p-value': [sw_p, ks_p, jb_p],
        'Keputusan (α=0.05)': [
            '✅ Normal' if sw_p > 0.05 else '❌ Tidak Normal',
            '✅ Normal' if ks_p > 0.05 else '❌ Tidak Normal',
            '✅ Normal' if jb_p > 0.05 else '❌ Tidak Normal'
        ]
    }).round(6)
    st.dataframe(norm_df, use_container_width=True, hide_index=True)

# --- 5b. Multicollinearity ---
st.subheader("5b. Uji Multikolinieritas (VIF)")
if len(x_vars) > 1:
    vif_data = pd.DataFrame({
        'Variabel': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        'Tolerance': [1 / variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }).round(4)
    vif_data['Status'] = vif_data['VIF'].apply(
        lambda x: '✅ Tidak ada multikolinieritas' if x < 5 
        else ('⚠️ Perlu perhatian' if x < 10 else '❌ Multikolinieritas tinggi')
    )
    st.dataframe(vif_data, use_container_width=True, hide_index=True)
else:
    st.info("VIF memerlukan minimal 2 variabel independen.")

# --- 5c. Heteroscedasticity ---
st.subheader("5c. Uji Heteroskedastisitas")
tab_het1, tab_het2 = st.tabs(["📊 Scatter Plot Residual", "📋 Uji Statistik"])

with tab_het1:
    fig_resid = px.scatter(
        x=fitted, y=residuals,
        title="Residual vs Fitted Values",
        labels={'x': 'Fitted Values', 'y': 'Residuals'}
    )
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_resid, use_container_width=True)

with tab_het2:
    # Breusch-Pagan test
    bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(residuals, X_const)

    # White test
    try:
        white_stat, white_p, white_f, white_fp = het_white(residuals, X_const)
    except Exception:
        white_stat, white_p, white_f, white_fp = np.nan, np.nan, np.nan, np.nan

    het_df = pd.DataFrame({
        'Uji': ['Breusch-Pagan', 'White Test'],
        'LM Statistik': [bp_stat, white_stat],
        'LM p-value': [bp_p, white_p],
        'F Statistik': [bp_f, white_f],
        'F p-value': [bp_fp, white_fp],
        'Keputusan (α=0.05)': [
            '✅ Homoskedastis' if bp_p > 0.05 else '❌ Heteroskedastis',
            '✅ Homoskedastis' if white_p > 0.05 else '❌ Heteroskedastis'
        ]
    }).round(6)
    st.dataframe(het_df, use_container_width=True, hide_index=True)

# --- 5d. Autocorrelation ---
st.subheader("5d. Uji Autokorelasi")
dw_stat = durbin_watson(residuals)
dw_interpretation = ""
if dw_stat < 1.5:
    dw_interpretation = "❌ Terdapat autokorelasi positif"
elif dw_stat > 2.5:
    dw_interpretation = "❌ Terdapat autokorelasi negatif"
else:
    dw_interpretation = "✅ Tidak ada autokorelasi"

auto_df = pd.DataFrame({
    'Uji': ['Durbin-Watson'],
    'Statistik': [dw_stat],
    'Interpretasi': [dw_interpretation],
    'Catatan': ['Nilai ideal mendekati 2.0 (range 0-4)']
}).round(4)
st.dataframe(auto_df, use_container_width=True, hide_index=True)

# --- 5e. Linearity ---
st.subheader("5e. Uji Linieritas (Actual vs Predicted)")
fig_lin = px.scatter(
    x=fitted, y=Y,
    title="Actual vs Predicted Values",
    labels={'x': 'Predicted Values', 'y': 'Actual Values'}
)
min_val = min(fitted.min(), Y.min())
max_val = max(fitted.max(), Y.max())
fig_lin.add_trace(go.Scatter(
    x=[min_val, max_val], y=[min_val, max_val],
    mode='lines', name='Garis Ideal (45°)',
    line=dict(color='red', dash='dash')
))
st.plotly_chart(fig_lin, use_container_width=True)

# ============================
# INFLUENTIAL OBSERVATIONS
# ============================
st.header("6️⃣ Deteksi Observasi Berpengaruh")
influence = model.get_influence()
inf_summary = influence.summary_frame()

tab_inf1, tab_inf2 = st.tabs(["📊 Cook's Distance", "📋 Leverage & DFFITS"])

with tab_inf1:
    cooks_d = inf_summary['cooks_d']
    threshold_cook = 4 / len(Y)
    fig_cook = px.bar(
        x=cooks_d.index, y=cooks_d.values,
        title=f"Cook's Distance (threshold = {threshold_cook:.4f})",
        labels={'x': 'Observasi', 'y': "Cook's Distance"}
    )
    fig_cook.add_hline(y=threshold_cook, line_dash="dash", line_color="red")
    st.plotly_chart(fig_cook, use_container_width=True)

    influential = cooks_d[cooks_d > threshold_cook]
    st.info(f"📍 Jumlah observasi berpengaruh (Cook's D > {threshold_cook:.4f}): **{len(influential)}**")

with tab_inf2:
    leverage = inf_summary['hat_diag']
    dffits = inf_summary['dffits']

    inf_detail = pd.DataFrame({
        'Observasi': range(len(Y)),
        'Leverage (hat)': leverage.values,
        'DFFITS': dffits[0].values if isinstance(dffits, tuple) else dffits.values,
        "Cook's D": cooks_d.values,
        'Std. Residual': inf_summary['standard_resid'].values
    }).round(6)

    st.dataframe(inf_detail.nlargest(20, "Cook's D"), use_container_width=True, hide_index=True)

# ============================
# PREDICTION
# ============================
st.header("7️⃣ Prediksi Nilai Baru")
st.markdown("Masukkan nilai variabel independen untuk melakukan prediksi.")

pred_cols = st.columns(min(len(x_vars), 4))
input_values = {}
for i, var in enumerate(x_vars):
    col_idx = i % min(len(x_vars), 4)
    with pred_cols[col_idx]:
        input_values[var] = st.number_input(
            f"{var}",
            value=float(df[var].mean()),
            step=float(df[var].std() / 10),
            format="%.4f"
        )

if st.button("🔮 Prediksi", type="primary"):
    new_data = pd.DataFrame([input_values])
    new_data_const = sm.add_constant(new_data, has_constant='add')
    prediction = model.get_prediction(new_data_const)
    pred_summary = prediction.summary_frame(alpha=0.05)

    st.success(f"**Hasil Prediksi {y_var}: {pred_summary['mean'].values[0]:.4f}**")

    pred_result = pd.DataFrame({
        'Metrik': ['Predicted Value', 'Std. Error', 'CI Lower (95%)', 'CI Upper (95%)', 'PI Lower (95%)', 'PI Upper (95%)'],
        'Nilai': [
            pred_summary['mean'].values[0],
            pred_summary['mean_se'].values[0],
            pred_summary['mean_ci_lower'].values[0],
            pred_summary['mean_ci_upper'].values[0],
            pred_summary['obs_ci_lower'].values[0],
            pred_summary['obs_ci_upper'].values[0]
        ]
    }).round(4)
    st.dataframe(pred_result, use_container_width=True, hide_index=True)

# ============================
# EXPORT RESULTS
# ============================
st.header("8️⃣ Ekspor Hasil")
col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    summary_text = model.summary().as_text()
    st.download_button(
        label="📄 Download Summary (TXT)",
        data=summary_text,
        file_name="regression_summary.txt",
        mime="text/plain"
    )

with col_exp2:
    coef_csv = coef_df.to_csv(index=False)
    st.download_button(
        label="📊 Download Koefisien (CSV)",
        data=coef_csv,
        file_name="coefficients.csv",
        mime="text/csv"
    )

with col_exp3:
    result_data = df[all_vars].copy()
    result_data['Predicted'] = fitted
    result_data['Residual'] = residuals
    result_csv = result_data.to_csv(index=False)
    st.download_button(
        label="📈 Download Prediksi (CSV)",
        data=result_csv,
        file_name="predictions_residuals.csv",
        mime="text/csv"
    )

# ============================
# FOOTER
# ============================
st.markdown("---")
st.markdown(
    "💡 **Catatan:** Aplikasi ini menggunakan metode **Ordinary Least Squares (OLS)** "
    "dari library `statsmodels`. Pastikan data memenuhi asumsi regresi linier klasik "
    "untuk hasil yang valid."
)
st.markdown("🔧 Dibangun dengan **Streamlit** + **Statsmodels** + **Plotly** | Python 🐍")
