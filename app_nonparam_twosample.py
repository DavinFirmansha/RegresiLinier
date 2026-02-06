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

st.set_page_config(page_title="Uji Nonparametrik 2 Sampel", page_icon="⚖️", layout="wide")
st.title("⚖️ Uji Nonparametrik — Dua Sampel")
st.markdown("""
Aplikasi lengkap untuk **uji nonparametrik dua sampel** — baik **independen** maupun **berpasangan**.
Dilengkapi 12+ metode uji, visualisasi interaktif, effect size, dan ekspor hasil.
""")

# ============================
# HELPERS
# ============================
def _arr(obj):
    return np.asarray(obj).flatten()

def effect_size_r(z, n):
    return abs(z) / np.sqrt(n) if n > 0 else 0

def rank_biserial(U, n1, n2):
    return 1 - (2 * U) / (n1 * n2) if (n1 * n2) > 0 else 0

def interpret_effect(r):
    r = abs(r)
    if r < 0.1: return "Sangat Kecil"
    elif r < 0.3: return "Kecil"
    elif r < 0.5: return "Sedang"
    else: return "Besar"

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.std(x,ddof=1)**2 + (ny-1)*np.std(y,ddof=1)**2) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0

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
    df = pd.DataFrame({
        'Kelompok_A': np.random.exponential(8, 60) + np.random.normal(0, 1, 60),
        'Kelompok_B': np.random.exponential(6, 60) + np.random.normal(3, 1.5, 60),
        'Pre_Test': np.random.normal(50, 10, 60),
    })
    df['Post_Test'] = df['Pre_Test'] + np.random.normal(5, 4, 60)
    st.sidebar.success("Data demo dimuat (60 obs, 4 variabel)")
else:
    st.warning("Silakan upload data atau gunakan data demo.")
    st.stop()

# ============================
# 1. EKSPLORASI DATA
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
    st.dataframe(df.describe().T.round(4), use_container_width=True)

# ============================
# 2. DESAIN PENELITIAN
# ============================
st.header("2. Desain Penelitian")
design = st.radio("Pilih jenis desain:", ["Independen (2 kelompok bebas)", "Berpasangan (paired/related)"],
                   horizontal=True)
is_independent = "Independen" in design

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Minimal 2 kolom numerik.")
    st.stop()

st.subheader("Pilih Variabel")
if is_independent:
    col_l, col_r = st.columns(2)
    with col_l:
        var1 = st.selectbox("Sampel 1 (Kelompok 1)", numeric_cols, index=0)
    with col_r:
        var2_opts = [c for c in numeric_cols if c != var1]
        var2 = st.selectbox("Sampel 2 (Kelompok 2)", var2_opts, index=0)
else:
    col_l, col_r = st.columns(2)
    with col_l:
        var1 = st.selectbox("Pengukuran 1 (Pre / Before)", numeric_cols, index=0)
    with col_r:
        var2_opts = [c for c in numeric_cols if c != var1]
        idx_post = 0
        for i, c in enumerate(var2_opts):
            if 'post' in c.lower() or 'after' in c.lower():
                idx_post = i; break
        var2 = st.selectbox("Pengukuran 2 (Post / After)", var2_opts, index=idx_post)

data1_raw = df[var1].dropna().values.astype(float)
data2_raw = df[var2].dropna().values.astype(float)

if not is_independent:
    min_len = min(len(data1_raw), len(data2_raw))
    data1 = data1_raw[:min_len]
    data2 = data2_raw[:min_len]
else:
    data1 = data1_raw
    data2 = data2_raw

n1, n2 = len(data1), len(data2)
N = n1 + n2

st.info(f"**Desain:** {'Independen' if is_independent else 'Berpasangan'} | "
        f"**{var1}** (n={n1}) vs **{var2}** (n={n2})")

# ============================
# 3. STATISTIK DESKRIPTIF
# ============================
st.header("3. Statistik Deskriptif Kedua Sampel")

desc_table = pd.DataFrame({
    'Statistik': ['N', 'Mean', 'Median', 'Std. Dev', 'Min', 'Max', 'Q1', 'Q3', 'IQR',
                  'Skewness', 'Kurtosis', 'CV (%)'],
    var1: [n1, np.mean(data1), np.median(data1), np.std(data1, ddof=1),
           np.min(data1), np.max(data1),
           np.percentile(data1, 25), np.percentile(data1, 75),
           np.percentile(data1, 75) - np.percentile(data1, 25),
           stats.skew(data1), stats.kurtosis(data1),
           (np.std(data1, ddof=1)/np.mean(data1))*100 if np.mean(data1) != 0 else np.nan],
    var2: [n2, np.mean(data2), np.median(data2), np.std(data2, ddof=1),
           np.min(data2), np.max(data2),
           np.percentile(data2, 25), np.percentile(data2, 75),
           np.percentile(data2, 75) - np.percentile(data2, 25),
           stats.skew(data2), stats.kurtosis(data2),
           (np.std(data2, ddof=1)/np.mean(data2))*100 if np.mean(data2) != 0 else np.nan],
}).round(4)
st.dataframe(desc_table, use_container_width=True, hide_index=True)

# ============================
# 4. VISUALISASI
# ============================
st.header("4. Visualisasi Data")

melted = pd.DataFrame({
    'Nilai': np.concatenate([data1, data2]),
    'Kelompok': [var1]*n1 + [var2]*n2
})

tab_v1, tab_v2, tab_v3, tab_v4, tab_v5 = st.tabs([
    "Histogram", "Box Plot", "Violin Plot", "ECDF", "Density"])

with tab_v1:
    fig_h = px.histogram(melted, x='Nilai', color='Kelompok', barmode='overlay',
                          nbins=30, opacity=0.6, title="Histogram Dua Sampel", marginal="rug")
    st.plotly_chart(fig_h, use_container_width=True)

with tab_v2:
    fig_b = px.box(melted, x='Kelompok', y='Nilai', color='Kelompok', points='all',
                    title="Box Plot Dua Sampel")
    st.plotly_chart(fig_b, use_container_width=True)

with tab_v3:
    fig_vio = px.violin(melted, x='Kelompok', y='Nilai', color='Kelompok', box=True,
                          points='all', title="Violin Plot Dua Sampel")
    st.plotly_chart(fig_vio, use_container_width=True)

with tab_v4:
    sorted1 = np.sort(data1)
    sorted2 = np.sort(data2)
    ecdf1 = np.arange(1, n1+1)/n1
    ecdf2 = np.arange(1, n2+1)/n2
    fig_ecdf = go.Figure()
    fig_ecdf.add_trace(go.Scatter(x=sorted1, y=ecdf1, mode='lines', name=var1))
    fig_ecdf.add_trace(go.Scatter(x=sorted2, y=ecdf2, mode='lines', name=var2))
    fig_ecdf.update_layout(title="ECDF Dua Sampel", xaxis_title="Nilai",
                            yaxis_title="Proporsi Kumulatif", height=450)
    st.plotly_chart(fig_ecdf, use_container_width=True)

with tab_v5:
    from scipy.stats import gaussian_kde
    x_range = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 300)
    kde1 = gaussian_kde(data1)(x_range)
    kde2 = gaussian_kde(data2)(x_range)
    fig_kde = go.Figure()
    fig_kde.add_trace(go.Scatter(x=x_range, y=kde1, mode='lines', name=var1, fill='tozeroy'))
    fig_kde.add_trace(go.Scatter(x=x_range, y=kde2, mode='lines', name=var2, fill='tozeroy'))
    fig_kde.update_layout(title="Density Plot (KDE)", xaxis_title="Nilai",
                           yaxis_title="Densitas", height=450)
    st.plotly_chart(fig_kde, use_container_width=True)

# Paired-specific visualization
if not is_independent:
    st.subheader("Visualisasi Paired Data")
    diffs = data1 - data2
    tab_p1, tab_p2, tab_p3 = st.tabs(["Selisih (Before-After)", "Paired Lines", "Bland-Altman"])
    with tab_p1:
        fig_diff = px.histogram(x=diffs, nbins=25, title=f"Histogram Selisih ({var1} - {var2})",
                                 labels={'x': 'Selisih'}, marginal="box")
        fig_diff.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_diff, use_container_width=True)
    with tab_p2:
        fig_pair = go.Figure()
        for i in range(min(n1, 50)):
            color = '#2ca02c' if data2[i] > data1[i] else '#d62728'
            fig_pair.add_trace(go.Scatter(x=[var1, var2], y=[data1[i], data2[i]],
                mode='lines+markers', line=dict(color=color, width=1),
                marker=dict(size=5), showlegend=False, opacity=0.5))
        fig_pair.update_layout(title="Paired Lines Plot (maks 50 pasang)",
                                yaxis_title="Nilai", height=450)
        st.plotly_chart(fig_pair, use_container_width=True)
    with tab_p3:
        means_ba = (data1 + data2) / 2
        diffs_ba = data1 - data2
        mean_diff = np.mean(diffs_ba)
        std_diff = np.std(diffs_ba, ddof=1)
        fig_ba = go.Figure()
        fig_ba.add_trace(go.Scatter(x=means_ba, y=diffs_ba, mode='markers',
                                     marker=dict(size=5, opacity=0.6), showlegend=False))
        fig_ba.add_hline(y=mean_diff, line_color="blue", annotation_text=f"Mean={mean_diff:.2f}")
        fig_ba.add_hline(y=mean_diff+1.96*std_diff, line_dash="dash", line_color="red",
                          annotation_text=f"+1.96SD={mean_diff+1.96*std_diff:.2f}")
        fig_ba.add_hline(y=mean_diff-1.96*std_diff, line_dash="dash", line_color="red",
                          annotation_text=f"-1.96SD={mean_diff-1.96*std_diff:.2f}")
        fig_ba.update_layout(title="Bland-Altman Plot", xaxis_title="Mean of Two Measurements",
                              yaxis_title="Difference", height=450)
        st.plotly_chart(fig_ba, use_container_width=True)

# ============================
# 5. PARAMETER UJI
# ============================
st.header("5. Parameter Pengujian")
col_a1, col_a2 = st.columns(2)
with col_a1:
    alpha = st.selectbox("Tingkat Signifikansi (α)", [0.01, 0.05, 0.10], index=1)
with col_a2:
    alt_type = st.selectbox("Arah Uji", ["two-sided", "greater", "less"], index=0)

alt_sym = '≠' if alt_type == 'two-sided' else ('>' if alt_type == 'greater' else '<')
st.info(f"**H₀:** Distribusi Sampel 1 = Sampel 2 | **H₁:** Distribusi Sampel 1 {alt_sym} Sampel 2 | α = {alpha}")

# ============================
# 6. UJI NORMALITAS KEDUA SAMPEL
# ============================
st.header("6. Uji Normalitas Kedua Sampel")
st.markdown("Uji normalitas menentukan apakah kita perlu menggunakan uji nonparametrik.")

def normality_battery(data, name):
    results = []
    if len(data) <= 5000:
        s, p = stats.shapiro(data)
        results.append({'Sampel': name, 'Uji': 'Shapiro-Wilk', 'Stat': round(s,6), 'p-value': round(p,6),
                        'Keputusan': 'Normal' if p > alpha else 'Tidak Normal'})
    s, p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
    results.append({'Sampel': name, 'Uji': 'Kolmogorov-Smirnov', 'Stat': round(s,6), 'p-value': round(p,6),
                    'Keputusan': 'Normal' if p > alpha else 'Tidak Normal'})
    s, p = stats.jarque_bera(data)
    results.append({'Sampel': name, 'Uji': 'Jarque-Bera', 'Stat': round(float(s),6), 'p-value': round(float(p),6),
                    'Keputusan': 'Normal' if p > alpha else 'Tidak Normal'})
    ad = stats.anderson(data, dist='norm')
    results.append({'Sampel': name, 'Uji': 'Anderson-Darling', 'Stat': round(ad.statistic,6),
                    'p-value': f'CV(5%)={ad.critical_values[2]:.4f}',
                    'Keputusan': 'Normal' if ad.statistic < ad.critical_values[2] else 'Tidak Normal'})
    if len(data) >= 20:
        s, p = stats.normaltest(data)
        results.append({'Sampel': name, 'Uji': "D'Agostino-Pearson", 'Stat': round(float(s),6),
                        'p-value': round(float(p),6),
                        'Keputusan': 'Normal' if p > alpha else 'Tidak Normal'})
    return results

norm_all = normality_battery(data1, var1) + normality_battery(data2, var2)
norm_df = pd.DataFrame(norm_all)
st.dataframe(norm_df, use_container_width=True, hide_index=True)

qq_c1, qq_c2 = st.columns(2)
for col, d, name in [(qq_c1, data1, var1), (qq_c2, data2, var2)]:
    with col:
        qq = stats.probplot(d, dist="norm")
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Data'))
        fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0]*qq[0][0]+qq[1][1],
                                     mode='lines', name='Normal', line=dict(color='red')))
        fig_qq.update_layout(title=f"QQ-Plot: {name}", height=350,
                              xaxis_title="Theoretical Q", yaxis_title="Sample Q")
        st.plotly_chart(fig_qq, use_container_width=True)

# ============================
# 7. UJI HOMOGENITAS VARIANSI
# ============================
st.header("7. Uji Homogenitas Variansi")

lev_s, lev_p = stats.levene(data1, data2, center='median')
flig_s, flig_p = stats.fligner(data1, data2)
bartlett_s, bartlett_p = stats.bartlett(data1, data2)
brown_s, brown_p = stats.levene(data1, data2, center='mean')

homo_df = pd.DataFrame({
    'Uji': ['Levene (median)', 'Brown-Forsythe (mean)', 'Fligner-Killeen', 'Bartlett'],
    'Statistik': [round(lev_s,4), round(brown_s,4), round(flig_s,4), round(bartlett_s,4)],
    'p-value': [round(lev_p,6), round(brown_p,6), round(flig_p,6), round(bartlett_p,6)],
    f'Keputusan (α={alpha})': [
        'Homogen' if lev_p > alpha else 'Tidak Homogen',
        'Homogen' if brown_p > alpha else 'Tidak Homogen',
        'Homogen' if flig_p > alpha else 'Tidak Homogen',
        'Homogen' if bartlett_p > alpha else 'Tidak Homogen']
})
st.dataframe(homo_df, use_container_width=True, hide_index=True)

# ============================
# INDEPENDENT SAMPLE TESTS
# ============================
summary_rows = []

if is_independent:
    # ============================
    # 8. MANN-WHITNEY U TEST
    # ============================
    st.header("8. Mann-Whitney U Test")
    st.markdown("Uji nonparametrik utama untuk **dua sampel independen** — membandingkan median melalui peringkat gabungan.")

    u_stat, u_p = stats.mannwhitneyu(data1, data2, alternative=alt_type)
    rb = rank_biserial(u_stat, n1, n2)
    z_mw = (u_stat - n1*n2/2) / np.sqrt(n1*n2*(n1+n2+1)/12)

    mw_result = pd.DataFrame({
        'Metrik': ['U-statistic', 'Z-approx', 'p-value', 'Rank-Biserial Correlation (r)',
                   'Effect Size Interpretation', 'Mean Rank Sampel 1', 'Mean Rank Sampel 2',
                   f'Keputusan (α={alpha})'],
        'Nilai': [round(float(u_stat),2), round(z_mw,4), round(float(u_p),6),
                  round(rb,4), interpret_effect(rb),
                  round(np.mean(stats.rankdata(np.concatenate([data1,data2]))[:n1]),2),
                  round(np.mean(stats.rankdata(np.concatenate([data1,data2]))[n1:]),2),
                  'Tolak H₀ (Berbeda signifikan)' if u_p < alpha else 'Gagal Tolak H₀']
    })
    st.dataframe(mw_result, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': 'Mann-Whitney U', 'Statistik': f'U={float(u_stat):.2f}',
                         'p-value': round(float(u_p),6),
                         'Effect Size': f'r={rb:.4f} ({interpret_effect(rb)})',
                         'Keputusan': 'Tolak H₀' if u_p < alpha else 'Gagal Tolak H₀'})

    # Rank visualization
    all_data = np.concatenate([data1, data2])
    all_ranks = stats.rankdata(all_data)
    ranks1, ranks2 = all_ranks[:n1], all_ranks[n1:]
    fig_rank = go.Figure()
    fig_rank.add_trace(go.Histogram(x=ranks1, name=f'{var1} Ranks', opacity=0.6, nbinsx=20))
    fig_rank.add_trace(go.Histogram(x=ranks2, name=f'{var2} Ranks', opacity=0.6, nbinsx=20))
    fig_rank.update_layout(title="Distribusi Peringkat Kedua Sampel", barmode='overlay',
                            xaxis_title="Rank", yaxis_title="Frekuensi", height=400)
    st.plotly_chart(fig_rank, use_container_width=True)

    # ============================
    # 9. KOLMOGOROV-SMIRNOV 2-SAMPLE
    # ============================
    st.header("9. Kolmogorov-Smirnov Two-Sample Test")
    st.markdown("Menguji apakah dua sampel berasal dari **distribusi yang sama** (sensitif terhadap lokasi, skala, dan bentuk).")

    ks_stat, ks_p = stats.ks_2samp(data1, data2, alternative=alt_type)

    ks_result = pd.DataFrame({
        'Metrik': ['D-statistic', 'p-value', f'Keputusan (α={alpha})'],
        'Nilai': [round(float(ks_stat),6), round(float(ks_p),6),
                  'Distribusi Berbeda' if ks_p < alpha else 'Distribusi Sama']
    })
    st.dataframe(ks_result, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': 'Kolmogorov-Smirnov 2-Sample', 'Statistik': f'D={float(ks_stat):.4f}',
                         'p-value': round(float(ks_p),6), 'Effect Size': '-',
                         'Keputusan': 'Berbeda' if ks_p < alpha else 'Sama'})

    fig_ks = go.Figure()
    fig_ks.add_trace(go.Scatter(x=sorted1, y=ecdf1, mode='lines', name=f'ECDF {var1}'))
    fig_ks.add_trace(go.Scatter(x=sorted2, y=ecdf2, mode='lines', name=f'ECDF {var2}'))
    fig_ks.update_layout(title=f"KS 2-Sample (D={ks_stat:.4f}, p={ks_p:.4f})",
                          xaxis_title="Nilai", yaxis_title="ECDF", height=400)
    st.plotly_chart(fig_ks, use_container_width=True)

    # ============================
    # 10. MEDIAN TEST
    # ============================
    st.header("10. Mood's Median Test")
    st.markdown("Menguji apakah dua sampel memiliki **median yang sama** menggunakan tabel kontingensi 2×2.")

    try:
        med_stat, med_p, med_med, med_table = stats.median_test(data1, data2)
        med_result = pd.DataFrame({
            'Metrik': ['Chi-Square Statistic', 'p-value', 'Grand Median',
                       f'Keputusan (α={alpha})'],
            'Nilai': [round(float(med_stat),4), round(float(med_p),6), round(float(med_med),4),
                      'Median Berbeda' if med_p < alpha else 'Median Sama']
        })
        st.dataframe(med_result, use_container_width=True, hide_index=True)
        summary_rows.append({'Uji': "Mood's Median Test", 'Statistik': f'χ²={float(med_stat):.4f}',
                             'p-value': round(float(med_p),6), 'Effect Size': '-',
                             'Keputusan': 'Berbeda' if med_p < alpha else 'Sama'})

        cont_df = pd.DataFrame(med_table, columns=[var1, var2],
                                index=['> Grand Median', '≤ Grand Median'])
        st.dataframe(cont_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Mood's Median Test gagal: {e}")

    # ============================
    # 11. BRUNNER-MUNZEL TEST
    # ============================
    st.header("11. Brunner-Munzel Test")
    st.markdown("Alternatif robust untuk Mann-Whitney U — **tidak** mengasumsikan bentuk distribusi yang sama.")

    try:
        bm_stat, bm_p = stats.brunnermunzel(data1, data2, alternative=alt_type)
        p_hat = float(u_stat) / (n1 * n2)  # probability estimate
        bm_result = pd.DataFrame({
            'Metrik': ['W-statistic', 'p-value', 'P(X₁ > X₂) est.',
                       f'Keputusan (α={alpha})'],
            'Nilai': [round(float(bm_stat),4), round(float(bm_p),6), round(p_hat,4),
                      'Berbeda Signifikan' if bm_p < alpha else 'Tidak Berbeda']
        })
        st.dataframe(bm_result, use_container_width=True, hide_index=True)
        summary_rows.append({'Uji': 'Brunner-Munzel', 'Statistik': f'W={float(bm_stat):.4f}',
                             'p-value': round(float(bm_p),6), 'Effect Size': f'P̂={p_hat:.4f}',
                             'Keputusan': 'Berbeda' if bm_p < alpha else 'Tidak Berbeda'})
    except Exception as e:
        st.warning(f"Brunner-Munzel gagal: {e}")

    # ============================
    # 12. PERMUTATION TEST
    # ============================
    st.header("12. Permutation Test (Exact)")
    st.markdown("Uji berbasis **resampling**: menghitung p-value dari distribusi selisih median yang dibangkitkan secara acak.")

    with st.expander("Jalankan Permutation Test", expanded=False):
        n_perm = st.number_input("Jumlah permutasi", min_value=100, max_value=50000, value=5000, step=500)
        if st.button("Run Permutation Test", type="primary"):
            obs_diff = np.median(data1) - np.median(data2)
            combined = np.concatenate([data1, data2])
            perm_diffs = np.zeros(int(n_perm))
            for i in range(int(n_perm)):
                perm = np.random.permutation(combined)
                perm_diffs[i] = np.median(perm[:n1]) - np.median(perm[n1:])

            if alt_type == "two-sided":
                perm_p = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
            elif alt_type == "greater":
                perm_p = np.mean(perm_diffs >= obs_diff)
            else:
                perm_p = np.mean(perm_diffs <= obs_diff)

            perm_result = pd.DataFrame({
                'Metrik': ['Observed Δ Median', 'Permutation p-value', f'Keputusan (α={alpha})'],
                'Nilai': [round(obs_diff, 4), round(perm_p, 6),
                          'Tolak H₀' if perm_p < alpha else 'Gagal Tolak H₀']
            })
            st.dataframe(perm_result, use_container_width=True, hide_index=True)

            fig_perm = go.Figure()
            fig_perm.add_trace(go.Histogram(x=perm_diffs, nbinsx=50, marker_color='steelblue',
                                             name='Permutation Diffs'))
            fig_perm.add_vline(x=obs_diff, line_color="red",
                                annotation_text=f"Obs Δ={obs_diff:.3f}")
            fig_perm.update_layout(title=f"Permutation Distribution (n={int(n_perm)})",
                                    xaxis_title="Δ Median", yaxis_title="Frekuensi", height=400)
            st.plotly_chart(fig_perm, use_container_width=True)
            summary_rows.append({'Uji': 'Permutation Test', 'Statistik': f'ΔMed={obs_diff:.4f}',
                                 'p-value': round(perm_p,6), 'Effect Size': '-',
                                 'Keputusan': 'Tolak H₀' if perm_p < alpha else 'Gagal Tolak H₀'})

    # ============================
    # 13. BOOTSTRAP 2-SAMPLE
    # ============================
    st.header("13. Bootstrap Two-Sample Test")
    with st.expander("Jalankan Bootstrap Test", expanded=False):
        n_boot = st.number_input("Jumlah iterasi bootstrap", min_value=100, max_value=50000, value=5000, step=500, key='boot_indep')
        if st.button("Run Bootstrap (Independent)", type="primary"):
            obs_diff_b = np.median(data1) - np.median(data2)
            boot_diffs = np.zeros(int(n_boot))
            for i in range(int(n_boot)):
                b1 = np.random.choice(data1, n1, replace=True)
                b2 = np.random.choice(data2, n2, replace=True)
                boot_diffs[i] = np.median(b1) - np.median(b2)
            ci_low = np.percentile(boot_diffs, (alpha/2)*100)
            ci_up = np.percentile(boot_diffs, (1-alpha/2)*100)
            boot_p = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))
            boot_p = min(boot_p, 1.0)

            boot_res = pd.DataFrame({
                'Metrik': ['Observed Δ Median', 'Bootstrap SE', f'CI Lower ({(1-alpha)*100:.0f}%)',
                           f'CI Upper ({(1-alpha)*100:.0f}%)', 'Bootstrap p-value',
                           f'Keputusan (α={alpha})'],
                'Nilai': [round(obs_diff_b,4), round(np.std(boot_diffs),4),
                          round(ci_low,4), round(ci_up,4), round(boot_p,6),
                          'Tolak H₀' if boot_p < alpha else 'Gagal Tolak H₀']
            })
            st.dataframe(boot_res, use_container_width=True, hide_index=True)

            fig_boot = go.Figure()
            fig_boot.add_trace(go.Histogram(x=boot_diffs, nbinsx=50, marker_color='steelblue'))
            fig_boot.add_vline(x=obs_diff_b, line_color="red", annotation_text=f"Obs={obs_diff_b:.3f}")
            fig_boot.add_vline(x=0, line_dash="dash", line_color="orange", annotation_text="H₀: Δ=0")
            fig_boot.update_layout(title=f"Bootstrap Distribution of Δ Median ({int(n_boot)} iter)",
                                    xaxis_title="Δ Median", yaxis_title="Frekuensi", height=400)
            st.plotly_chart(fig_boot, use_container_width=True)

# ============================
# PAIRED SAMPLE TESTS
# ============================
else:
    diffs = data1 - data2

    # ============================
    # 8. WILCOXON SIGNED-RANK
    # ============================
    st.header("8. Wilcoxon Signed-Rank Test")
    st.markdown("Uji nonparametrik utama untuk **dua sampel berpasangan** — mempertimbangkan arah dan besar selisih.")

    diffs_nz = diffs[diffs != 0]
    n_eff = len(diffs_nz)
    if n_eff >= 10:
        wil_stat, wil_p = stats.wilcoxon(diffs_nz, alternative=alt_type)
        mean_T = n_eff*(n_eff+1)/4
        std_T = np.sqrt(n_eff*(n_eff+1)*(2*n_eff+1)/24)
        z_wil = (float(wil_stat) - mean_T) / std_T if std_T > 0 else 0
        es_wil = effect_size_r(z_wil, n_eff)
        abs_diffs = np.abs(diffs_nz)
        ranks = stats.rankdata(abs_diffs)
        T_plus = np.sum(ranks[diffs_nz > 0])
        T_minus = np.sum(ranks[diffs_nz < 0])

        wil_result = pd.DataFrame({
            'Metrik': ['T+ (rank positif)', 'T- (rank negatif)', 'W-statistic', 'Z-approx',
                       'p-value', 'n efektif (non-zero)', 'Effect Size (r)',
                       'Interpretasi Effect', f'Keputusan (α={alpha})'],
            'Nilai': [round(T_plus,2), round(T_minus,2), round(float(wil_stat),4), round(z_wil,4),
                      round(float(wil_p),6), n_eff, round(es_wil,4), interpret_effect(es_wil),
                      'Tolak H₀ (Berbeda signifikan)' if wil_p < alpha else 'Gagal Tolak H₀']
        })
        st.dataframe(wil_result, use_container_width=True, hide_index=True)
        summary_rows.append({'Uji': 'Wilcoxon Signed-Rank', 'Statistik': f'W={float(wil_stat):.2f}',
                             'p-value': round(float(wil_p),6),
                             'Effect Size': f'r={es_wil:.4f} ({interpret_effect(es_wil)})',
                             'Keputusan': 'Tolak H₀' if wil_p < alpha else 'Gagal Tolak H₀'})

        fig_wil = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Distribusi Selisih", "T+ vs T-"))
        fig_wil.add_trace(go.Histogram(x=diffs_nz, nbinsx=25, marker_color='steelblue',
                                        showlegend=False), row=1, col=1)
        fig_wil.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
        fig_wil.add_trace(go.Bar(x=['T+ (Positif)', 'T- (Negatif)'],
                                  y=[T_plus, T_minus],
                                  marker_color=['#2ca02c', '#d62728'],
                                  showlegend=False), row=1, col=2)
        fig_wil.update_layout(height=400)
        st.plotly_chart(fig_wil, use_container_width=True)
    else:
        st.warning(f"Minimal 10 selisih non-nol. Saat ini: {n_eff}")

    # ============================
    # 9. SIGN TEST (PAIRED)
    # ============================
    st.header("9. Sign Test (Paired)")
    st.markdown("Uji paling sederhana: hanya mempertimbangkan **arah** selisih (positif vs negatif).")

    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    n_zero = np.sum(diffs == 0)
    n_sign_eff = n_pos + n_neg

    if n_sign_eff > 0:
        if alt_type == "two-sided":
            sign_p = 2 * stats.binom.cdf(min(n_pos, n_neg), n_sign_eff, 0.5)
            sign_p = min(sign_p, 1.0)
        elif alt_type == "greater":
            sign_p = stats.binom.cdf(n_neg, n_sign_eff, 0.5)
        else:
            sign_p = stats.binom.cdf(n_pos, n_sign_eff, 0.5)
        z_sign = (n_pos - n_sign_eff/2) / np.sqrt(n_sign_eff/4)
    else:
        sign_p, z_sign = 1.0, 0.0

    sign_es = effect_size_r(z_sign, n_sign_eff) if n_sign_eff > 0 else 0

    sign_result = pd.DataFrame({
        'Metrik': ['n positif (+)', 'n negatif (-)', 'n nol (ties)', 'n efektif',
                   'Z-approx', 'p-value', 'Effect Size (r)', 'Interpretasi',
                   f'Keputusan (α={alpha})'],
        'Nilai': [n_pos, n_neg, n_zero, n_sign_eff,
                  round(z_sign,4), round(sign_p,6), round(sign_es,4), interpret_effect(sign_es),
                  'Tolak H₀' if sign_p < alpha else 'Gagal Tolak H₀']
    })
    st.dataframe(sign_result, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': 'Sign Test (Paired)', 'Statistik': f'Z={z_sign:.4f}',
                         'p-value': round(sign_p,6),
                         'Effect Size': f'r={sign_es:.4f} ({interpret_effect(sign_es)})',
                         'Keputusan': 'Tolak H₀' if sign_p < alpha else 'Gagal Tolak H₀'})

    fig_sign = go.Figure()
    fig_sign.add_trace(go.Bar(x=['Positif (+)', 'Negatif (-)', 'Nol (=)'],
                               y=[n_pos, n_neg, n_zero],
                               marker_color=['#2ca02c', '#d62728', '#7f7f7f']))
    fig_sign.update_layout(title="Sign Test: Distribusi Tanda", yaxis_title="Frekuensi", height=350)
    st.plotly_chart(fig_sign, use_container_width=True)

    # ============================
    # 10. MCNEMAR TEST
    # ============================
    st.header("10. McNemar Test (Dikotomisasi)")
    st.markdown("Versi **kategorikal** dari uji berpasangan: menggunakan tabel 2×2 dari perubahan arah terhadap median.")

    med_combined = np.median(np.concatenate([data1, data2]))
    a = np.sum((data1 > med_combined) & (data2 > med_combined))
    b = np.sum((data1 > med_combined) & (data2 <= med_combined))
    c = np.sum((data1 <= med_combined) & (data2 > med_combined))
    d_mn = np.sum((data1 <= med_combined) & (data2 <= med_combined))

    if (b + c) > 0:
        mcnemar_chi2 = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0
        mcnemar_p = 1 - stats.chi2.cdf(mcnemar_chi2, 1)
    else:
        mcnemar_chi2, mcnemar_p = 0, 1.0

    cont_table = pd.DataFrame({'Post > Median': [a, c], 'Post ≤ Median': [b, d_mn]},
                               index=['Pre > Median', 'Pre ≤ Median'])
    st.dataframe(cont_table, use_container_width=True)

    mcn_result = pd.DataFrame({
        'Metrik': ['b (discordant +)', 'c (discordant -)', 'McNemar χ² (corrected)', 'p-value',
                   f'Keputusan (α={alpha})'],
        'Nilai': [int(b), int(c), round(mcnemar_chi2,4), round(mcnemar_p,6),
                  'Proporsi Berbeda' if mcnemar_p < alpha else 'Proporsi Sama']
    })
    st.dataframe(mcn_result, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': 'McNemar Test', 'Statistik': f'χ²={mcnemar_chi2:.4f}',
                         'p-value': round(mcnemar_p,6), 'Effect Size': '-',
                         'Keputusan': 'Berbeda' if mcnemar_p < alpha else 'Sama'})

    # ============================
    # 11. BOOTSTRAP PAIRED
    # ============================
    st.header("11. Bootstrap Paired Test")
    with st.expander("Jalankan Bootstrap Paired Test", expanded=False):
        n_boot_p = st.number_input("Jumlah iterasi bootstrap", min_value=100, max_value=50000, value=5000, step=500, key='boot_paired')
        if st.button("Run Bootstrap (Paired)", type="primary"):
            boot_med_diffs = np.zeros(int(n_boot_p))
            for i in range(int(n_boot_p)):
                idx = np.random.choice(len(diffs), len(diffs), replace=True)
                boot_med_diffs[i] = np.median(diffs[idx])
            ci_low_b = np.percentile(boot_med_diffs, (alpha/2)*100)
            ci_up_b = np.percentile(boot_med_diffs, (1-alpha/2)*100)
            boot_p_paired = 2 * min(np.mean(boot_med_diffs <= 0), np.mean(boot_med_diffs >= 0))
            boot_p_paired = min(boot_p_paired, 1.0)

            bp_res = pd.DataFrame({
                'Metrik': ['Median Selisih (obs)', 'Bootstrap SE',
                           f'CI Lower ({(1-alpha)*100:.0f}%)', f'CI Upper ({(1-alpha)*100:.0f}%)',
                           'Bootstrap p-value', f'Keputusan (α={alpha})'],
                'Nilai': [round(np.median(diffs),4), round(np.std(boot_med_diffs),4),
                          round(ci_low_b,4), round(ci_up_b,4), round(boot_p_paired,6),
                          'Tolak H₀' if boot_p_paired < alpha else 'Gagal Tolak H₀']
            })
            st.dataframe(bp_res, use_container_width=True, hide_index=True)

            fig_bp = go.Figure()
            fig_bp.add_trace(go.Histogram(x=boot_med_diffs, nbinsx=50, marker_color='steelblue'))
            fig_bp.add_vline(x=np.median(diffs), line_color="red",
                              annotation_text=f"Obs={np.median(diffs):.3f}")
            fig_bp.add_vline(x=0, line_dash="dash", line_color="orange", annotation_text="H₀: Δ=0")
            fig_bp.update_layout(title=f"Bootstrap Paired Median Difference ({int(n_boot_p)} iter)",
                                  xaxis_title="Δ Median", yaxis_title="Frekuensi", height=400)
            st.plotly_chart(fig_bp, use_container_width=True)

    # ============================
    # 12. PERMUTATION PAIRED
    # ============================
    st.header("12. Permutation Paired Test")
    with st.expander("Jalankan Permutation Paired Test", expanded=False):
        n_perm_p = st.number_input("Jumlah permutasi", min_value=100, max_value=50000, value=5000, step=500, key='perm_paired')
        if st.button("Run Permutation (Paired)", type="primary"):
            obs_med_diff = np.median(diffs)
            perm_med_diffs = np.zeros(int(n_perm_p))
            for i in range(int(n_perm_p)):
                signs = np.random.choice([-1, 1], len(diffs))
                perm_med_diffs[i] = np.median(diffs * signs)

            if alt_type == "two-sided":
                pp_p = np.mean(np.abs(perm_med_diffs) >= np.abs(obs_med_diff))
            elif alt_type == "greater":
                pp_p = np.mean(perm_med_diffs >= obs_med_diff)
            else:
                pp_p = np.mean(perm_med_diffs <= obs_med_diff)

            pp_res = pd.DataFrame({
                'Metrik': ['Observed Δ Median', 'Permutation p-value', f'Keputusan (α={alpha})'],
                'Nilai': [round(obs_med_diff,4), round(pp_p,6),
                          'Tolak H₀' if pp_p < alpha else 'Gagal Tolak H₀']
            })
            st.dataframe(pp_res, use_container_width=True, hide_index=True)

            fig_pp = go.Figure()
            fig_pp.add_trace(go.Histogram(x=perm_med_diffs, nbinsx=50, marker_color='steelblue'))
            fig_pp.add_vline(x=obs_med_diff, line_color="red",
                              annotation_text=f"Obs={obs_med_diff:.3f}")
            fig_pp.update_layout(title=f"Permutation Paired Distribution ({int(n_perm_p)} iter)",
                                  xaxis_title="Δ Median", yaxis_title="Frekuensi", height=400)
            st.plotly_chart(fig_pp, use_container_width=True)

# ============================
# COMMON: CLIFF'S DELTA
# ============================
st.header("13. Cliff's Delta (Effect Size)" if is_independent else "13. Matched-Pairs Rank-Biserial (Effect Size)")
st.markdown("Ukuran effect size nonparametrik yang **tidak tergantung** pada asumsi distribusi.")

if is_independent:
    count = 0
    for x in data1:
        for y in data2:
            if x > y: count += 1
            elif x < y: count -= 1
    cliff_d = count / (n1 * n2)
    es_name = "Cliff's Delta"
    es_val = cliff_d
else:
    n_conc = np.sum(diffs > 0)
    n_disc = np.sum(diffs < 0)
    n_t = n_conc + n_disc
    es_val = (n_conc - n_disc) / n_t if n_t > 0 else 0
    es_name = "Matched-Pairs Rank-Biserial"

es_result = pd.DataFrame({
    'Metrik': [es_name, 'Interpretasi'],
    'Nilai': [round(es_val, 4), interpret_effect(es_val)]
})
st.dataframe(es_result, use_container_width=True, hide_index=True)

# ============================
# 14. SUMMARY
# ============================
st.header("14. Ringkasan Seluruh Uji")
if len(summary_rows) > 0:
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
else:
    st.info("Jalankan uji-uji di atas terlebih dahulu.")

# ============================
# 15. EXPORT
# ============================
st.header("15. Ekspor Hasil")
col_e1, col_e2, col_e3 = st.columns(3)

with col_e1:
    lines = [
        "=" * 65, "UJI NONPARAMETRIK - DUA SAMPEL", "=" * 65,
        f"Desain             : {'Independen' if is_independent else 'Berpasangan'}",
        f"Sampel 1           : {var1} (n={n1})",
        f"Sampel 2           : {var2} (n={n2})",
        f"Alpha              : {alpha}",
        f"Arah Uji           : {alt_type}",
        f"Mean Sampel 1      : {np.mean(data1):.6f}",
        f"Mean Sampel 2      : {np.mean(data2):.6f}",
        f"Median Sampel 1    : {np.median(data1):.6f}",
        f"Median Sampel 2    : {np.median(data2):.6f}",
        "", "=" * 65, "HASIL UJI", "=" * 65]
    if len(summary_rows) > 0:
        for row in summary_rows:
            lines.append(f"  {row['Uji']:30s} | {str(row['Statistik']):>15s} | p={row['p-value']:<10} | {row['Keputusan']}")
    st.download_button("Download Summary (TXT)", data="\n".join(lines),
                       file_name="nonparam_twosample_summary.txt", mime="text/plain")

with col_e2:
    if is_independent:
        max_len = max(n1, n2)
        exp_df = pd.DataFrame({
            var1: np.pad(data1, (0, max_len-n1), constant_values=np.nan),
            var2: np.pad(data2, (0, max_len-n2), constant_values=np.nan)
        })
    else:
        exp_df = pd.DataFrame({var1: data1, var2: data2, 'Selisih': diffs})
    st.download_button("Download Data (CSV)", data=exp_df.to_csv(index=False),
                       file_name="nonparam_twosample_data.csv", mime="text/csv")

with col_e3:
    if len(summary_rows) > 0:
        st.download_button("Download Ringkasan Uji (CSV)",
                           data=pd.DataFrame(summary_rows).to_csv(index=False),
                           file_name="nonparam_twosample_tests.csv", mime="text/csv")

# ============================
# FOOTER
# ============================
st.markdown("---")
st.markdown("""
**Catatan Metodologis:**

**Sampel Independen:**
- **Mann-Whitney U**: uji peringkat gabungan, ekuivalen nonparametrik Independent t-Test.
- **Kolmogorov-Smirnov 2-Sample**: sensitif terhadap perbedaan lokasi, skala, DAN bentuk distribusi.
- **Mood's Median Test**: uji sederhana berbasis tabel kontingensi 2×2.
- **Brunner-Munzel**: robust, tidak mengasumsikan distribusi identik.
- **Permutation Test**: exact test berbasis resampling.

**Sampel Berpasangan:**
- **Wilcoxon Signed-Rank**: mempertimbangkan arah DAN besar selisih, ekuivalen nonparametrik Paired t-Test.
- **Sign Test**: hanya mempertimbangkan arah selisih, cocok untuk data ordinal.
- **McNemar Test**: uji perubahan proporsi pada data berpasangan yang dikotomisasi.
- **Bootstrap & Permutation**: pendekatan resampling modern tanpa asumsi distribusi.

**Effect Size:** Rank-Biserial Correlation (Mann-Whitney), r = |Z|/√n (Wilcoxon/Sign), Cliff's Delta.
""")
st.markdown("Dibangun dengan **Streamlit** + **SciPy** + **Plotly** | Python")
