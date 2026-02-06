import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Uji Nonparametrik 1 Sampel", page_icon="🧪", layout="wide")
st.title("🧪 Uji Nonparametrik — Satu Sampel")
st.markdown("""
Aplikasi lengkap untuk **uji nonparametrik satu sampel**: Sign Test, Wilcoxon Signed-Rank,
Kolmogorov-Smirnov, Chi-Square Goodness of Fit, Runs Test, Shapiro-Wilk, Anderson-Darling,
Lilliefors, Jarque-Bera, D'Agostino-Pearson, dan Binomial Test.
""")

# ============================
# SAFE HELPERS
# ============================
def _arr(obj):
    return np.asarray(obj).flatten()

def effect_size_r(z, n):
    """Effect size r = |Z| / sqrt(n)."""
    return abs(z) / np.sqrt(n)

def interpret_effect(r):
    if r < 0.1: return "Sangat Kecil"
    elif r < 0.3: return "Kecil"
    elif r < 0.5: return "Sedang"
    else: return "Besar"

# ============================
# SIDEBAR: DATA INPUT
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
    n = 100
    data_demo = np.concatenate([
        np.random.exponential(5, 40),
        np.random.normal(12, 3, 35),
        np.random.uniform(2, 20, 25)
    ])
    df = pd.DataFrame({'Nilai': data_demo})
    st.sidebar.success("Data demo dimuat (100 obs, distribusi campuran)")
else:
    st.warning("Silakan upload data atau gunakan data demo.")
    st.stop()

# ============================
# 1. EKSPLORASI DATA
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
# 2. PILIH VARIABEL
# ============================
st.header("2. Pemilihan Variabel")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < 1:
    st.error("Minimal 1 kolom numerik.")
    st.stop()

selected_var = st.selectbox("Pilih variabel untuk diuji", numeric_cols, index=0)
data = df[selected_var].dropna().values.astype(float)
n_obs = len(data)

st.markdown(f"**Variabel terpilih:** `{selected_var}` | N = {n_obs}")

# ============================
# 3. STATISTIK DESKRIPTIF
# ============================
st.header("3. Statistik Deskriptif")
desc_col1, desc_col2, desc_col3, desc_col4 = st.columns(4)
desc_col1.metric("Mean", f"{np.mean(data):.4f}")
desc_col2.metric("Median", f"{np.median(data):.4f}")
desc_col3.metric("Std. Dev", f"{np.std(data, ddof=1):.4f}")
desc_col4.metric("IQR", f"{np.percentile(data, 75) - np.percentile(data, 25):.4f}")

desc2_c1, desc2_c2, desc2_c3, desc2_c4 = st.columns(4)
desc2_c1.metric("Min", f"{np.min(data):.4f}")
desc2_c2.metric("Max", f"{np.max(data):.4f}")
desc2_c3.metric("Skewness", f"{stats.skew(data):.4f}")
desc2_c4.metric("Kurtosis", f"{stats.kurtosis(data):.4f}")

full_desc = pd.DataFrame({
    'Statistik': ['Mean', 'Median', 'Modus (approx)', 'Std. Deviasi', 'Variansi',
                  'Skewness', 'Kurtosis', 'Min', 'Max', 'Range', 'Q1', 'Q3', 'IQR',
                  'CV (%)', 'Std. Error Mean'],
    'Nilai': [
        np.mean(data), np.median(data),
        float(stats.mode(data, keepdims=True).mode[0]) if len(data) > 0 else np.nan,
        np.std(data, ddof=1), np.var(data, ddof=1),
        stats.skew(data), stats.kurtosis(data),
        np.min(data), np.max(data), np.ptp(data),
        np.percentile(data, 25), np.percentile(data, 75),
        np.percentile(data, 75) - np.percentile(data, 25),
        (np.std(data, ddof=1) / np.mean(data)) * 100 if np.mean(data) != 0 else np.nan,
        np.std(data, ddof=1) / np.sqrt(n_obs)
    ]
}).round(6)
with st.expander("Tabel Deskriptif Lengkap"):
    st.dataframe(full_desc, use_container_width=True, hide_index=True)

# ============================
# 4. VISUALISASI DATA
# ============================
st.header("4. Visualisasi Data")
tab_v1, tab_v2, tab_v3, tab_v4, tab_v5 = st.tabs([
    "Histogram", "Box Plot", "Violin Plot", "ECDF", "Dot + Strip Plot"])

with tab_v1:
    fig_h = px.histogram(x=data, nbins=30, title=f"Histogram: {selected_var}",
                          labels={'x': selected_var}, marginal="rug")
    fig_h.add_vline(x=np.mean(data), line_dash="dash", line_color="red",
                     annotation_text=f"Mean={np.mean(data):.2f}")
    fig_h.add_vline(x=np.median(data), line_dash="dash", line_color="green",
                     annotation_text=f"Median={np.median(data):.2f}")
    st.plotly_chart(fig_h, use_container_width=True)

with tab_v2:
    fig_b = px.box(y=data, title=f"Box Plot: {selected_var}", labels={'y': selected_var},
                    points="all")
    st.plotly_chart(fig_b, use_container_width=True)

with tab_v3:
    fig_vio = px.violin(y=data, box=True, points="all",
                          title=f"Violin Plot: {selected_var}", labels={'y': selected_var})
    st.plotly_chart(fig_vio, use_container_width=True)

with tab_v4:
    sorted_data = np.sort(data)
    ecdf_y = np.arange(1, n_obs + 1) / n_obs
    fig_ecdf = go.Figure()
    fig_ecdf.add_trace(go.Scatter(x=sorted_data, y=ecdf_y, mode='lines+markers',
                                   name='ECDF (Empiris)', marker=dict(size=3)))
    x_norm = np.linspace(data.min(), data.max(), 200)
    cdf_norm = stats.norm.cdf(x_norm, loc=np.mean(data), scale=np.std(data, ddof=1))
    fig_ecdf.add_trace(go.Scatter(x=x_norm, y=cdf_norm, mode='lines',
                                   name='CDF Normal Teoritis', line=dict(color='red', dash='dash')))
    fig_ecdf.update_layout(title=f"ECDF vs CDF Normal: {selected_var}",
                            xaxis_title=selected_var, yaxis_title="Proporsi Kumulatif", height=450)
    st.plotly_chart(fig_ecdf, use_container_width=True)

with tab_v5:
    fig_strip = px.strip(y=data, title=f"Strip Plot: {selected_var}",
                          labels={'y': selected_var})
    st.plotly_chart(fig_strip, use_container_width=True)

# QQ Plot
st.subheader("QQ-Plot (Normal)")
qq = stats.probplot(data, dist="norm")
fig_qq = go.Figure()
fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Data',
                              marker=dict(size=5, opacity=0.6)))
fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0]*qq[0][0]+qq[1][1],
                              mode='lines', name='Garis Normal', line=dict(color='red')))
fig_qq.update_layout(title="QQ-Plot (Normal)", xaxis_title="Theoretical Quantiles",
                      yaxis_title="Sample Quantiles", height=400)
st.plotly_chart(fig_qq, use_container_width=True)

# ============================
# 5. PARAMETER UJI
# ============================
st.header("5. Parameter Pengujian")
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    alpha = st.selectbox("Tingkat Signifikansi (α)", [0.01, 0.05, 0.10], index=1)
with col_p2:
    m0 = st.number_input("Nilai Median/Proporsi Hipotesis (M₀)",
                          value=float(np.median(data)), format="%.4f")
with col_p3:
    alt_type = st.selectbox("Arah Uji", ["two-sided", "greater", "less"], index=0)

st.info(f"**H₀:** Median = {m0}  |  **H₁:** Median {'≠' if alt_type=='two-sided' else ('>' if alt_type=='greater' else '<')} {m0}  |  α = {alpha}")

# ============================
# 6. SIGN TEST
# ============================
st.header("6. Sign Test (Uji Tanda)")
st.markdown("Menguji apakah **median populasi** sama dengan nilai hipotesis M₀, berdasarkan jumlah tanda positif dan negatif.")

diffs = data - m0
n_pos = np.sum(diffs > 0)
n_neg = np.sum(diffs < 0)
n_zero = np.sum(diffs == 0)
n_eff = n_pos + n_neg  # exclude zeros

if n_eff > 0:
    if alt_type == "two-sided":
        sign_p = 2 * stats.binom.cdf(min(n_pos, n_neg), n_eff, 0.5)
        sign_p = min(sign_p, 1.0)
    elif alt_type == "greater":
        sign_p = stats.binom.cdf(n_neg, n_eff, 0.5)
    else:
        sign_p = stats.binom.cdf(n_pos, n_eff, 0.5)
    z_sign = (n_pos - n_eff / 2) / np.sqrt(n_eff / 4)
else:
    sign_p = 1.0
    z_sign = 0.0

sign_es = effect_size_r(z_sign, n_eff) if n_eff > 0 else 0
sign_result = pd.DataFrame({
    'Metrik': ['n positif (+)', 'n negatif (-)', 'n nol (=)', 'n efektif',
               'Z-approx', 'p-value', 'Effect Size (r)', 'Interpretasi Effect',
               f'Keputusan (α={alpha})'],
    'Nilai': [n_pos, n_neg, n_zero, n_eff,
              round(z_sign, 4), round(sign_p, 6),
              round(sign_es, 4), interpret_effect(sign_es),
              f'Tolak H₀' if sign_p < alpha else f'Gagal Tolak H₀']
})
st.dataframe(sign_result, use_container_width=True, hide_index=True)

# Visualisasi Sign Test
fig_sign = go.Figure()
fig_sign.add_trace(go.Bar(x=['Positif (+)', 'Negatif (-)', 'Nol (=)'],
                           y=[n_pos, n_neg, n_zero],
                           marker_color=['#2ca02c', '#d62728', '#7f7f7f']))
fig_sign.update_layout(title="Sign Test: Distribusi Tanda", yaxis_title="Frekuensi", height=350)
st.plotly_chart(fig_sign, use_container_width=True)

# ============================
# 7. WILCOXON SIGNED-RANK TEST
# ============================
st.header("7. Wilcoxon Signed-Rank Test")
st.markdown("Menguji median populasi terhadap M₀ dengan mempertimbangkan **besar (rank)** dan **arah** selisih setiap observasi.")

diffs_nz = diffs[diffs != 0]
if len(diffs_nz) >= 10:
    try:
        wil_stat, wil_p = stats.wilcoxon(diffs_nz, alternative=alt_type)
        n_wil = len(diffs_nz)
        mean_T = n_wil * (n_wil + 1) / 4
        std_T = np.sqrt(n_wil * (n_wil + 1) * (2 * n_wil + 1) / 24)
        z_wil = (wil_stat - mean_T) / std_T if std_T > 0 else 0
        es_wil = effect_size_r(z_wil, n_wil)

        # Rank detail
        abs_diffs = np.abs(diffs_nz)
        ranks = stats.rankdata(abs_diffs)
        T_plus = np.sum(ranks[diffs_nz > 0])
        T_minus = np.sum(ranks[diffs_nz < 0])

        wil_result = pd.DataFrame({
            'Metrik': ['T+ (rank positif)', 'T- (rank negatif)', 'W-statistic', 'Z-approx',
                       'p-value', 'n efektif', 'Effect Size (r)', 'Interpretasi Effect',
                       f'Keputusan (α={alpha})'],
            'Nilai': [round(T_plus, 2), round(T_minus, 2), round(float(wil_stat), 4),
                      round(z_wil, 4), round(float(wil_p), 6), n_wil,
                      round(es_wil, 4), interpret_effect(es_wil),
                      'Tolak H₀' if wil_p < alpha else 'Gagal Tolak H₀']
        })
        st.dataframe(wil_result, use_container_width=True, hide_index=True)

        # Visualisasi rank
        fig_wil = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Distribusi Selisih (data - M₀)", "Rank Positif vs Negatif"))
        fig_wil.add_trace(go.Histogram(x=diffs_nz, nbinsx=25, marker_color='steelblue',
                                        showlegend=False), row=1, col=1)
        fig_wil.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
        fig_wil.add_trace(go.Bar(x=['T+ (Positif)', 'T- (Negatif)'],
                                  y=[T_plus, T_minus],
                                  marker_color=['#2ca02c', '#d62728'],
                                  showlegend=False), row=1, col=2)
        fig_wil.update_layout(height=400)
        st.plotly_chart(fig_wil, use_container_width=True)
    except Exception as e:
        st.warning(f"Wilcoxon test gagal: {e}")
else:
    st.warning(f"Wilcoxon Signed-Rank membutuhkan minimal 10 observasi non-nol. Saat ini: {len(diffs_nz)}")

# ============================
# 8. KOLMOGOROV-SMIRNOV ONE-SAMPLE
# ============================
st.header("8. Kolmogorov-Smirnov One-Sample Test")
st.markdown("Menguji apakah data berasal dari suatu **distribusi teoritis** tertentu.")

dist_options = {
    "Normal": ("norm", {"loc": np.mean(data), "scale": np.std(data, ddof=1)}),
    "Exponential": ("expon", {"loc": 0, "scale": np.mean(data)}),
    "Uniform": ("uniform", {"loc": np.min(data), "scale": np.ptp(data)}),
    "Lognormal": ("lognorm", {"s": np.std(np.log(data[data > 0]), ddof=1),
                               "loc": 0,
                               "scale": np.exp(np.mean(np.log(data[data > 0])))} if np.all(data > 0) else None),
    "Gamma": ("gamma", None),
    "Weibull": ("weibull_min", None),
}

ks_selected = st.selectbox("Distribusi Referensi", list(dist_options.keys()), index=0)

ks_results_all = []
for dname, dinfo in dist_options.items():
    try:
        if dinfo[1] is not None:
            dist_obj = getattr(stats, dinfo[0])
            ks_s, ks_p = stats.kstest(data, dinfo[0], args=tuple(dinfo[1].values()))
        else:
            dist_obj = getattr(stats, dinfo[0])
            params = dist_obj.fit(data)
            ks_s, ks_p = stats.kstest(data, dinfo[0], args=params)
        ks_results_all.append({
            'Distribusi': dname, 'D-statistic': round(ks_s, 6),
            'p-value': round(ks_p, 6),
            f'Keputusan (α={alpha})': 'Sesuai distribusi' if ks_p > alpha else 'Tidak sesuai'
        })
    except:
        ks_results_all.append({
            'Distribusi': dname, 'D-statistic': np.nan,
            'p-value': np.nan, f'Keputusan (α={alpha})': 'N/A'
        })

ks_df = pd.DataFrame(ks_results_all)
st.dataframe(ks_df, use_container_width=True, hide_index=True)

# KS visualization for selected distribution
try:
    dinfo_sel = dist_options[ks_selected]
    if dinfo_sel[1] is not None:
        dist_obj = getattr(stats, dinfo_sel[0])
        ks_stat_sel, ks_p_sel = stats.kstest(data, dinfo_sel[0], args=tuple(dinfo_sel[1].values()))
        cdf_theo = dist_obj.cdf(x_norm, **dinfo_sel[1]) if ks_selected != "Lognormal" else \
                   dist_obj.cdf(np.linspace(max(0.01, data.min()), data.max(), 200), **dinfo_sel[1])
        x_cdf = x_norm if ks_selected != "Lognormal" else np.linspace(max(0.01, data.min()), data.max(), 200)
    else:
        dist_obj = getattr(stats, dinfo_sel[0])
        fit_params = dist_obj.fit(data)
        ks_stat_sel, ks_p_sel = stats.kstest(data, dinfo_sel[0], args=fit_params)
        x_cdf = np.linspace(data.min(), data.max(), 200)
        cdf_theo = dist_obj.cdf(x_cdf, *fit_params)

    fig_ks = go.Figure()
    fig_ks.add_trace(go.Scatter(x=sorted_data, y=ecdf_y, mode='lines', name='ECDF (Data)'))
    fig_ks.add_trace(go.Scatter(x=x_cdf, y=cdf_theo, mode='lines',
                                 name=f'CDF {ks_selected}', line=dict(color='red', dash='dash')))
    fig_ks.update_layout(title=f"KS Test: ECDF vs CDF {ks_selected} (D={ks_stat_sel:.4f}, p={ks_p_sel:.4f})",
                          xaxis_title=selected_var, yaxis_title="CDF", height=450)
    st.plotly_chart(fig_ks, use_container_width=True)
except Exception as e:
    st.warning(f"Visualisasi KS gagal: {e}")

# ============================
# 9. CHI-SQUARE GOODNESS OF FIT
# ============================
st.header("9. Chi-Square Goodness of Fit Test")
st.markdown("Menguji apakah **frekuensi observasi** data sesuai dengan distribusi yang diharapkan.")

n_bins_chi = st.slider("Jumlah kelas/bin", 3, 20, min(10, max(3, int(np.sqrt(n_obs)))))
observed_freq, bin_edges = np.histogram(data, bins=n_bins_chi)

chi_dist = st.selectbox("Distribusi Expected", ["Uniform (sama rata)", "Normal"], index=0)

if chi_dist == "Uniform (sama rata)":
    expected_freq = np.full(n_bins_chi, n_obs / n_bins_chi)
else:
    dist_n = stats.norm(loc=np.mean(data), scale=np.std(data, ddof=1))
    expected_freq = np.array([
        dist_n.cdf(bin_edges[i+1]) - dist_n.cdf(bin_edges[i]) for i in range(n_bins_chi)
    ]) * n_obs
    expected_freq = np.maximum(expected_freq, 0.001)

chi2_stat, chi2_p = stats.chisquare(observed_freq, f_exp=expected_freq)
df_chi = n_bins_chi - 1

chi_result = pd.DataFrame({
    'Metrik': ['Chi-Square Statistic', 'Degrees of Freedom', 'p-value', f'Keputusan (α={alpha})'],
    'Nilai': [round(float(chi2_stat), 4), df_chi, round(float(chi2_p), 6),
              'Sesuai distribusi' if chi2_p > alpha else 'Tidak sesuai']
})
st.dataframe(chi_result, use_container_width=True, hide_index=True)

# Chi-square visualization
bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(n_bins_chi)]
fig_chi = go.Figure()
fig_chi.add_trace(go.Bar(x=bin_labels, y=observed_freq, name='Observed', marker_color='steelblue'))
fig_chi.add_trace(go.Bar(x=bin_labels, y=expected_freq, name='Expected', marker_color='salmon'))
fig_chi.update_layout(title=f"Chi-Square GoF (χ²={chi2_stat:.2f}, p={chi2_p:.4f})",
                       xaxis_title="Kelas", yaxis_title="Frekuensi", barmode='group', height=400)
st.plotly_chart(fig_chi, use_container_width=True)

# Residual table
chi_detail = pd.DataFrame({
    'Kelas': bin_labels, 'Observed': observed_freq, 'Expected': expected_freq.round(2),
    'Residual': (observed_freq - expected_freq).round(2),
    'Std. Residual': ((observed_freq - expected_freq) / np.sqrt(expected_freq)).round(4)
})
with st.expander("Detail Frekuensi per Kelas"):
    st.dataframe(chi_detail, use_container_width=True, hide_index=True)

# ============================
# 10. RUNS TEST
# ============================
st.header("10. Runs Test (Uji Keacakan)")
st.markdown("Menguji apakah urutan data bersifat **acak** berdasarkan jumlah 'run' (kelompok nilai berturut-turut di atas/bawah median).")

median_val = np.median(data)
binary = (data >= median_val).astype(int)
runs = 1 + np.sum(np.diff(binary) != 0)
n1 = np.sum(binary)
n0 = len(binary) - n1
runs_mean = (2 * n1 * n0) / (n1 + n0) + 1
runs_var = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0)**2 * (n1 + n0 - 1)) if (n1+n0) > 1 else 0
runs_z = (runs - runs_mean) / np.sqrt(runs_var) if runs_var > 0 else 0
runs_p = 2 * (1 - stats.norm.cdf(abs(runs_z)))

runs_result = pd.DataFrame({
    'Metrik': ['n ≥ median', 'n < median', 'Jumlah Runs (observasi)', 'Expected Runs',
               'Z-statistic', 'p-value', f'Keputusan (α={alpha})'],
    'Nilai': [int(n1), int(n0), int(runs), round(runs_mean, 2),
              round(runs_z, 4), round(runs_p, 6),
              'Acak' if runs_p > alpha else 'Tidak Acak']
})
st.dataframe(runs_result, use_container_width=True, hide_index=True)

# Runs visualization
fig_runs = go.Figure()
colors_runs = ['#2ca02c' if b == 1 else '#d62728' for b in binary]
fig_runs.add_trace(go.Bar(x=list(range(len(data))), y=data, marker_color=colors_runs,
                           showlegend=False))
fig_runs.add_hline(y=median_val, line_dash="dash", line_color="black",
                    annotation_text=f"Median={median_val:.2f}")
fig_runs.update_layout(title=f"Runs Test: Pola Data terhadap Median (Runs={runs})",
                        xaxis_title="Observasi ke-", yaxis_title=selected_var, height=400)
st.plotly_chart(fig_runs, use_container_width=True)

# ============================
# 11. NORMALITY TESTS BATTERY
# ============================
st.header("11. Battery of Normality Tests")
st.markdown("Kumpulan lengkap uji normalitas untuk menentukan apakah data berasal dari distribusi normal.")

norm_tests = []

# Shapiro-Wilk
if n_obs <= 5000:
    sw_s, sw_p = stats.shapiro(data)
    norm_tests.append({'Uji': 'Shapiro-Wilk', 'Statistik': round(sw_s, 6), 'p-value': round(sw_p, 6),
                       f'Keputusan (α={alpha})': 'Normal' if sw_p > alpha else 'Tidak Normal'})
else:
    norm_tests.append({'Uji': 'Shapiro-Wilk', 'Statistik': np.nan, 'p-value': np.nan,
                       f'Keputusan (α={alpha})': 'N/A (n > 5000)'})

# Kolmogorov-Smirnov (Normal)
ks_s, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
norm_tests.append({'Uji': 'Kolmogorov-Smirnov', 'Statistik': round(ks_s, 6), 'p-value': round(ks_p, 6),
                   f'Keputusan (α={alpha})': 'Normal' if ks_p > alpha else 'Tidak Normal'})

# Lilliefors
try:
    from statsmodels.stats.diagnostic import lilliefors as lilliefors_test
    lil_s, lil_p = lilliefors_test(data, dist='norm')
    norm_tests.append({'Uji': 'Lilliefors', 'Statistik': round(lil_s, 6), 'p-value': round(lil_p, 6),
                       f'Keputusan (α={alpha})': 'Normal' if lil_p > alpha else 'Tidak Normal'})
except:
    norm_tests.append({'Uji': 'Lilliefors', 'Statistik': np.nan, 'p-value': np.nan,
                       f'Keputusan (α={alpha})': 'N/A'})

# Jarque-Bera
jb_s, jb_p = stats.jarque_bera(data)
norm_tests.append({'Uji': 'Jarque-Bera', 'Statistik': round(float(jb_s), 6), 'p-value': round(float(jb_p), 6),
                   f'Keputusan (α={alpha})': 'Normal' if jb_p > alpha else 'Tidak Normal'})

# Anderson-Darling
ad_res = stats.anderson(data, dist='norm')
ad_cv_idx = {0.15: 0, 0.10: 1, 0.05: 2, 0.025: 3, 0.01: 4}
ad_idx = ad_cv_idx.get(alpha, 2)
ad_decision = 'Normal' if ad_res.statistic < ad_res.critical_values[ad_idx] else 'Tidak Normal'
norm_tests.append({'Uji': 'Anderson-Darling', 'Statistik': round(ad_res.statistic, 6),
                   'p-value': f'CV({ad_res.significance_level[ad_idx]}%)={ad_res.critical_values[ad_idx]:.4f}',
                   f'Keputusan (α={alpha})': ad_decision})

# D'Agostino-Pearson
if n_obs >= 20:
    dag_s, dag_p = stats.normaltest(data)
    norm_tests.append({'Uji': "D'Agostino-Pearson K²", 'Statistik': round(float(dag_s), 6),
                       'p-value': round(float(dag_p), 6),
                       f'Keputusan (α={alpha})': 'Normal' if dag_p > alpha else 'Tidak Normal'})
else:
    norm_tests.append({'Uji': "D'Agostino-Pearson K²", 'Statistik': np.nan, 'p-value': np.nan,
                       f'Keputusan (α={alpha})': 'N/A (n < 20)'})

# Skewness Test
if n_obs >= 8:
    sk_s, sk_p = stats.skewtest(data)
    norm_tests.append({'Uji': 'Skewness Test (Z)', 'Statistik': round(float(sk_s), 6),
                       'p-value': round(float(sk_p), 6),
                       f'Keputusan (α={alpha})': 'Simetris' if sk_p > alpha else 'Tidak Simetris'})

# Kurtosis Test
if n_obs >= 20:
    ku_s, ku_p = stats.kurtosistest(data)
    norm_tests.append({'Uji': 'Kurtosis Test (Z)', 'Statistik': round(float(ku_s), 6),
                       'p-value': round(float(ku_p), 6),
                       f'Keputusan (α={alpha})': 'Mesokurtik' if ku_p > alpha else 'Tidak Mesokurtik'})

norm_df = pd.DataFrame(norm_tests)
st.dataframe(norm_df, use_container_width=True, hide_index=True)

# Count decisions
n_normal = sum(1 for t in norm_tests if 'Normal' == t.get(f'Keputusan (α={alpha})', '') or
               'Simetris' == t.get(f'Keputusan (α={alpha})', '') or
               'Mesokurtik' == t.get(f'Keputusan (α={alpha})', ''))
n_total_tests = len([t for t in norm_tests if 'N/A' not in str(t.get(f'Keputusan (α={alpha})', ''))])
if n_total_tests > 0:
    st.info(f"**Ringkasan:** {n_normal}/{n_total_tests} uji menunjukkan distribusi Normal.")

# ============================
# 12. BINOMIAL TEST
# ============================
st.header("12. Binomial Test")
st.markdown("Menguji apakah **proporsi** data yang memenuhi suatu kriteria sesuai dengan proporsi hipotesis.")

col_b1, col_b2 = st.columns(2)
with col_b1:
    binom_threshold = st.number_input("Threshold (nilai batas)", value=float(np.median(data)), format="%.4f")
with col_b2:
    p0 = st.number_input("Proporsi Hipotesis (p₀)", value=0.5, min_value=0.001, max_value=0.999, step=0.05)

n_success = np.sum(data > binom_threshold)
n_fail = n_obs - n_success
p_obs = n_success / n_obs

binom_res = stats.binomtest(n_success, n_obs, p0, alternative=alt_type)
binom_p = binom_res.pvalue
binom_ci = binom_res.proportion_ci(confidence_level=1-alpha)

binom_result = pd.DataFrame({
    'Metrik': [f'n > {binom_threshold:.2f}', f'n ≤ {binom_threshold:.2f}', 'Proporsi Observasi',
               'Proporsi Hipotesis', 'p-value',
               f'CI Lower ({(1-alpha)*100:.0f}%)', f'CI Upper ({(1-alpha)*100:.0f}%)',
               f'Keputusan (α={alpha})'],
    'Nilai': [n_success, n_fail, round(p_obs, 4), p0, round(float(binom_p), 6),
              round(binom_ci.low, 4), round(binom_ci.high, 4),
              'Tolak H₀' if binom_p < alpha else 'Gagal Tolak H₀']
})
st.dataframe(binom_result, use_container_width=True, hide_index=True)

fig_binom = go.Figure()
fig_binom.add_trace(go.Bar(x=[f'> {binom_threshold:.2f}', f'≤ {binom_threshold:.2f}'],
                            y=[n_success, n_fail],
                            marker_color=['#2ca02c', '#d62728']))
fig_binom.update_layout(title=f"Binomial Test: Proporsi ({p_obs:.3f} vs {p0})",
                         yaxis_title="Frekuensi", height=350)
st.plotly_chart(fig_binom, use_container_width=True)

# ============================
# 13. ONE-SAMPLE BOOTSTRAP TEST
# ============================
st.header("13. Bootstrap Test (Median)")
st.markdown("Uji non-parametrik berbasis **resampling**: estimasi distribusi sampling median melalui bootstrap.")

with st.expander("Jalankan Bootstrap Test", expanded=False):
    n_boot = st.number_input("Jumlah iterasi bootstrap", min_value=100, max_value=50000, value=5000, step=500)
    if st.button("Run Bootstrap", type="primary"):
        boot_medians = np.array([np.median(np.random.choice(data, n_obs, replace=True)) for _ in range(int(n_boot))])
        boot_ci_low = np.percentile(boot_medians, (alpha/2)*100)
        boot_ci_up = np.percentile(boot_medians, (1-alpha/2)*100)
        boot_p = 2 * min(np.mean(boot_medians <= m0), np.mean(boot_medians >= m0))
        boot_p = min(boot_p, 1.0)

        boot_result = pd.DataFrame({
            'Metrik': ['Median Observasi', 'Bootstrap Mean Median', 'Bootstrap Std. Error',
                       f'CI Lower ({(1-alpha)*100:.0f}%)', f'CI Upper ({(1-alpha)*100:.0f}%)',
                       'Bootstrap p-value (approx)', f'Keputusan (α={alpha})'],
            'Nilai': [round(np.median(data), 4), round(np.mean(boot_medians), 4),
                      round(np.std(boot_medians), 4),
                      round(boot_ci_low, 4), round(boot_ci_up, 4),
                      round(boot_p, 6),
                      'Tolak H₀' if boot_p < alpha else 'Gagal Tolak H₀']
        })
        st.dataframe(boot_result, use_container_width=True, hide_index=True)

        fig_boot = go.Figure()
        fig_boot.add_trace(go.Histogram(x=boot_medians, nbinsx=50, marker_color='steelblue', name='Bootstrap Medians'))
        fig_boot.add_vline(x=np.median(data), line_dash="solid", line_color="red",
                            annotation_text=f"Median Obs={np.median(data):.2f}")
        fig_boot.add_vline(x=m0, line_dash="dash", line_color="orange",
                            annotation_text=f"M₀={m0:.2f}")
        fig_boot.add_vline(x=boot_ci_low, line_dash="dot", line_color="green")
        fig_boot.add_vline(x=boot_ci_up, line_dash="dot", line_color="green")
        fig_boot.update_layout(title=f"Bootstrap Distribution of Median ({int(n_boot)} iterasi)",
                                xaxis_title="Median", yaxis_title="Frekuensi", height=400)
        st.plotly_chart(fig_boot, use_container_width=True)

# ============================
# 14. SUMMARY TABLE
# ============================
st.header("14. Ringkasan Seluruh Uji")

summary_rows = []
summary_rows.append({'Uji': 'Sign Test', 'Statistik': f'Z={z_sign:.4f}',
                     'p-value': round(sign_p, 6), 'Keputusan': 'Tolak H₀' if sign_p < alpha else 'Gagal Tolak H₀'})
try:
    summary_rows.append({'Uji': 'Wilcoxon Signed-Rank', 'Statistik': f'W={float(wil_stat):.4f}',
                         'p-value': round(float(wil_p), 6),
                         'Keputusan': 'Tolak H₀' if wil_p < alpha else 'Gagal Tolak H₀'})
except:
    summary_rows.append({'Uji': 'Wilcoxon Signed-Rank', 'Statistik': 'N/A', 'p-value': np.nan, 'Keputusan': 'N/A'})

for nt in norm_tests[:6]:
    p_str = nt['p-value']
    try: p_float = float(p_str)
    except: p_float = np.nan
    summary_rows.append({'Uji': f"Normalitas: {nt['Uji']}",
                         'Statistik': f"{nt['Statistik']}", 'p-value': p_float,
                         'Keputusan': nt[f'Keputusan (α={alpha})']})

summary_rows.append({'Uji': 'Chi-Square GoF', 'Statistik': f'χ²={float(chi2_stat):.4f}',
                     'p-value': round(float(chi2_p), 6),
                     'Keputusan': 'Sesuai' if chi2_p > alpha else 'Tidak Sesuai'})
summary_rows.append({'Uji': 'Runs Test', 'Statistik': f'Z={runs_z:.4f}',
                     'p-value': round(runs_p, 6), 'Keputusan': 'Acak' if runs_p > alpha else 'Tidak Acak'})
summary_rows.append({'Uji': 'Binomial Test', 'Statistik': f'p_obs={p_obs:.4f}',
                     'p-value': round(float(binom_p), 6),
                     'Keputusan': 'Tolak H₀' if binom_p < alpha else 'Gagal Tolak H₀'})

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ============================
# 15. EXPORT
# ============================
st.header("15. Ekspor Hasil")
col_e1, col_e2 = st.columns(2)

with col_e1:
    lines = [
        "=" * 65, "UJI NONPARAMETRIK - SATU SAMPEL", "=" * 65,
        f"Variabel           : {selected_var}",
        f"N Observasi        : {n_obs}",
        f"Mean               : {np.mean(data):.6f}",
        f"Median             : {np.median(data):.6f}",
        f"Std. Dev           : {np.std(data, ddof=1):.6f}",
        f"Skewness           : {stats.skew(data):.6f}",
        f"Kurtosis           : {stats.kurtosis(data):.6f}",
        f"Alpha              : {alpha}",
        f"Median Hipotesis   : {m0}",
        f"Arah Uji           : {alt_type}",
        "", "=" * 65, "HASIL UJI", "=" * 65]
    for _, row in summary_df.iterrows():
        p_str = f"{row['p-value']:.6f}" if not pd.isna(row['p-value']) else "N/A"
        lines.append(f"  {row['Uji']:35s} | Stat: {str(row['Statistik']):>15s} | p={p_str} | {row['Keputusan']}")
    st.download_button("Download Summary (TXT)", data="\n".join(lines),
                       file_name="nonparametric_onesample_summary.txt", mime="text/plain")

with col_e2:
    export_df = pd.DataFrame({selected_var: data})
    st.download_button("Download Data (CSV)", data=export_df.to_csv(index=False),
                       file_name="nonparametric_onesample_data.csv", mime="text/csv")

# ============================
# FOOTER
# ============================
st.markdown("---")
st.markdown("""
**Catatan Metodologis:**
- **Sign Test**: uji paling sederhana, hanya melihat arah (tanda) selisih terhadap M₀.
- **Wilcoxon Signed-Rank**: lebih powerful dari Sign Test karena mempertimbangkan magnitude selisih.
- **Kolmogorov-Smirnov**: membandingkan ECDF data dengan CDF teoritis; sensitif terhadap perbedaan lokasi, skala, dan bentuk.
- **Chi-Square GoF**: menguji distribusi frekuensi; cocok untuk data dengan kategori/kelas.
- **Runs Test**: menguji keacakan urutan data.
- **Battery Normalitas**: 8 uji sekaligus (Shapiro-Wilk, KS, Lilliefors, Jarque-Bera, Anderson-Darling, D'Agostino-Pearson, Skewness, Kurtosis).
- **Binomial Test**: menguji proporsi berdasarkan distribusi binomial eksak.
- **Bootstrap Test**: pendekatan resampling tanpa asumsi distribusi.
""")
st.markdown("Dibangun dengan **Streamlit** + **SciPy** + **Statsmodels** + **Plotly** | Python")
